# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Engine-side half of M-OPD: one function that is both the engine's logits hook and its loss.

The engine keeps one log-prob per position. M-OPD needs K per position, so every call sets
`distillation_use_topk`, which hands the full logits to the loss function before the loss
(`FSDPEngineWithLMHead.prepare_model_outputs`). The hook returns one per-position tensor per column, which
is the shape the engine accepts; `stack_columns` puts the K columns back together.

`data["mt_opd_mode"]` selects what a forward does:

- `topk`: the student's own top-K ids and their log-probs (rollout time, eval mode).
- `score`: log-probs on `data["student_top_k_ids"]` (the teachers).
- `train`: the same scoring with gradient, then the clipped loss.

"The reference" is Open-MOPD 4809a96, paths under its `training/verl/verl/`.
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.mt_opd.mt_opd import refresh_opd_advantage
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import no_padding_2_padding

# non-tensor settings of a call
MODE_KEY = "mt_opd_mode"
TOP_K_KEY = "mt_opd_top_k"
REFRESH_KEY = "mt_opd_refresh"
ON_POLICY_KEY = "mt_opd_on_policy"
# [B, R, K] tensors of a batch, and its [B] domain weight
IDS_KEY = "student_top_k_ids"
STUDENT_LOGP_KEY = "student_top_k_log_probs"
TEACHER_LOGP_KEY = "teacher_on_student_log_probs"
ADVANTAGES_KEY = "opd_advantages"
DOMAIN_WEIGHT_KEY = "domain_loss_weight"
# the hook's per-position outputs
LOGP_COLUMN = "mt_opd_logp_{}"
ID_COLUMN = "mt_opd_id_{}"


def _mode(data: TensorDict) -> str:
    mode = tu.get_non_tensor_data(data=data, key=MODE_KEY, default=None)
    if mode not in ("topk", "score", "train"):
        raise ValueError(f"mt_opd: data[{MODE_KEY!r}] must be topk, score or train, got {mode!r}")
    return mode


def _pack_response_rows(per_response: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """[B, Rmax, K] rows of the padded response -> [total_tokens, K] rows of the packed input.

    Response position r of a sequence is predicted at its input position prompt_len - 1 + r. Input positions
    that predict no response token (the prompt's all but last, the sequence's last) get zeros.
    """
    prompt_width = data["prompts"].shape[1]
    attention_mask = data["attention_mask"]
    prompt_lens = attention_mask[:, :prompt_width].sum(dim=1).tolist()
    response_lens = attention_mask[:, prompt_width:].sum(dim=1).tolist()
    rows = [
        F.pad(per_response[i, :response_len], (0, 0, prompt_len - 1, 1))
        for i, (prompt_len, response_len) in enumerate(zip(prompt_lens, response_lens, strict=True))
    ]
    return torch.cat(rows, dim=0)


def _logits_hook(student_logits: torch.Tensor, data: TensorDict) -> dict[str, torch.Tensor]:
    """Full logits (1, total_tokens, V) -> per-position columns, each (1, total_tokens)."""
    logits = student_logits.squeeze(0)
    # the reference's operation order (`workers/actor/dp_actor.py`, `_forward_micro_batch`): one log_softmax
    # over the vocabulary, then a gather
    log_probs_all = torch.log_softmax(logits, dim=-1)
    find_top_k = _mode(data) == "topk"
    if find_top_k:
        _, ids = torch.topk(logits, k=data[TOP_K_KEY], dim=-1)
    else:
        ids = _pack_response_rows(data[IDS_KEY], data)
    log_probs = log_probs_all.gather(dim=-1, index=ids)
    columns = {LOGP_COLUMN.format(k): log_probs[:, k].unsqueeze(0) for k in range(ids.shape[1])}
    if find_top_k:
        columns.update({ID_COLUMN.format(k): ids[:, k].unsqueeze(0) for k in range(ids.shape[1])})
    return columns


def stack_columns(get_output, data: TensorDict, column: str, top_k: int) -> torch.Tensor:
    """The hook's K per-position columns as one padded response tensor [B, Rmax, K]. `get_output(key)` reads
    one model output: a dict lookup on the worker, a TensorDict read on the driver."""
    return torch.stack([no_padding_2_padding(get_output(column.format(k)), data) for k in range(top_k)], dim=-1)


def clipped_topk_loss(log_prob, old_log_prob, advantages, config):
    """The reference's per-id PPO loss for [B, R, K] tensors (`trainer/ppo/core_algos.py`, the 3D branch of
    `compute_policy_loss_vanilla`).

    Returns the loss summed over K [B, R], the clamped `log_prob - old_log_prob` [B, R, K], and where the
    clipped term is the larger one [B, R, K].
    """
    clip_ratio = config.clip_ratio
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * config.clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    return pg_losses.sum(dim=-1), negative_approx_kl, pg_losses2 > pg_losses1


def _train_loss(config, model_output, data: TensorDict):
    top_k = data[IDS_KEY].shape[-1]
    log_prob = stack_columns(model_output.__getitem__, data, LOGP_COLUMN, top_k)  # [B, R, K], with gradient
    response_mask = data["response_mask"]
    # a single update on fresh data takes its own forward as "old", as the reference does
    # (`workers/actor/dp_actor.py`, `update_policy`, the `on_policy` branch)
    old_log_prob = log_prob.detach() if data[ON_POLICY_KEY] else data[STUDENT_LOGP_KEY]
    advantages = data[ADVANTAGES_KEY]

    metrics = {}
    if data[REFRESH_KEY]:
        # the reward of the current student against the cached teacher; the ratio's "old" stays the rollout's
        advantages = refresh_opd_advantage(
            student_top_k_log_probs=log_prob,
            teacher_on_student_log_probs=data[TEACHER_LOGP_KEY],
            response_mask=response_mask,
        )
        # the reference's metric name; "m4" is its label for the refresh
        metrics["mt_opd/m4_advantage_refreshed"] = 1.0
    advantages = advantages * data[DOMAIN_WEIGHT_KEY].to(advantages.dtype).view(-1, 1, 1)

    pg_losses, negative_approx_kl, clipped = clipped_topk_loss(log_prob, old_log_prob, advantages, config)

    # Each sequence's own token mean, then the mean over the update's sequences on all ranks. The reference's
    # `token-mean` gives this value when a micro-batch is one sequence, which is how its M-OPD runs are sized
    # (unverified beyond the one run this port was compared with); this form does not depend on the micro-batch.
    mask = response_mask.to(pg_losses.dtype)
    tokens = mask.sum(dim=-1)
    per_sequence = (pg_losses * mask).sum(dim=-1) / (tokens + 1e-8)
    # dp_size: the ranks' gradients are averaged, as in `verl.workers.utils.losses.sft_loss`
    loss = per_sequence.sum() * data["dp_size"] / data["global_batch_size"]

    with torch.no_grad():
        total = tokens.sum().clamp(min=1.0)
        metrics["ppo_kl"] = ((-negative_approx_kl.sum(dim=-1) * mask).sum() / total).item()
        # the reference's two reporting conventions: clipped ids per token (so the value can exceed 1), and
        # each micro-batch's share of the update's loss
        metrics["pg_clipfrac"] = ((clipped.to(mask.dtype) * mask.unsqueeze(-1)).sum() / total).item()
        metrics["pg_loss"] = loss.item()
    return loss, metrics


def mt_opd_loss(config, model_output=None, data: TensorDict = None, dp_group=None, student_logits=None):
    """The loss function of every M-OPD worker. The engine also calls it as its logits hook, with
    `student_logits` and `data` only."""
    if student_logits is not None:
        return _logits_hook(student_logits, data)
    if _mode(data) == "train":
        return _train_loss(config, model_output, data)
    # a forward-only call: the columns are the result
    return torch.zeros((), device=data["response_mask"].device), {}
