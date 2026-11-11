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
import math
from types import SimpleNamespace

import torch

from verl.trainer.mt_opd import losses
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding

VOCAB, TOP_K = 11, 3
CONFIG = SimpleNamespace(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=3.0)


def _batch():
    """Two sequences, left-padded prompts of 3 and 2 tokens, right-padded responses of 2 and 4 tokens."""
    prompts = torch.tensor([[0, 5, 6, 7], [0, 0, 8, 9]])
    responses = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1]])
    response_mask = attention_mask[:, 4:]
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    data = tu.get_tensordict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
    )
    return left_right_2_no_padding(data)


def test_hook_scores_each_response_position_at_the_input_position_that_predicts_it():
    data = _batch()
    torch.manual_seed(0)
    # packed input: 5 tokens of sequence 0, then 6 of sequence 1
    logits = torch.randn(1, 11, VOCAB)
    tu.assign_non_tensor(data, **{losses.MODE_KEY: "topk", losses.TOP_K_KEY: TOP_K})
    columns = losses._logits_hook(logits, data)
    log_probs = torch.log_softmax(logits[0], dim=-1)

    def padded(column):
        return losses.stack_columns(lambda key: columns[key].squeeze(0), data, column, TOP_K)

    ids, logp = padded(losses.ID_COLUMN), padded(losses.LOGP_COLUMN)
    # sequence 0: prompt 3, so response position 0 is predicted at packed position 2; sequence 1 starts at 5
    for row, first, length in ((0, 2, 2), (1, 5 + 1, 4)):
        expected = log_probs[first : first + length].topk(TOP_K, dim=-1)
        assert torch.equal(ids[row, :length], expected.indices)
        assert torch.allclose(logp[row, :length], expected.values)

    # scoring those ids again gives the same log-probs
    tu.assign_non_tensor(data, **{losses.MODE_KEY: "score"})
    data[losses.IDS_KEY] = ids
    rescored = losses._logits_hook(logits, data)
    again = losses.stack_columns(lambda key: rescored[key].squeeze(0), data, losses.LOGP_COLUMN, TOP_K)
    mask = data["response_mask"].bool()
    assert torch.allclose(again[mask], logp[mask])


def test_clipped_topk_loss_clips_each_id_by_the_sign_of_its_advantage():
    old = torch.zeros(1, 1, 4)
    log_prob = torch.tensor([[[math.log(2.0), math.log(2.0), math.log(0.5), math.log(8.0)]]])
    advantages = torch.tensor([[[1.0, -1.0, 1.0, -1.0]]])
    loss, negative_approx_kl, clipped = losses.clipped_topk_loss(log_prob, old, advantages, CONFIG)
    # A > 0, ratio 2: clipped at 1.28. A < 0, ratio 2: not clipped. A > 0, ratio 0.5: not clipped.
    # A < 0, ratio 8: capped at clip_ratio_c = 3.
    assert torch.allclose(loss, torch.tensor([[-1.28 + 2.0 - 0.5 + 3.0]]))
    assert torch.allclose(negative_approx_kl, log_prob)
    assert clipped.tolist() == [[[True, False, False, False]]]


def test_train_loss_is_the_mean_over_sequences_of_each_sequences_token_mean():
    data = _batch()
    torch.manual_seed(1)
    shape = (2, 4, TOP_K)
    old = torch.log_softmax(torch.randn(shape), dim=-1)
    log_prob = old + 0.3 * torch.randn(shape)
    teacher = old + torch.randn(shape)
    weight = torch.tensor([2.0, 0.5])
    mask = data["response_mask"]
    data[losses.IDS_KEY] = torch.zeros(shape, dtype=torch.long)
    data[losses.STUDENT_LOGP_KEY] = old
    data[losses.TEACHER_LOGP_KEY] = teacher
    data[losses.ADVANTAGES_KEY] = losses.refresh_opd_advantage(old, teacher, mask)
    data[losses.DOMAIN_WEIGHT_KEY] = weight
    # the model output the loss reads: one packed column per id, each response row at its input position
    packed = losses._pack_response_rows(log_prob, data)
    model_output = {losses.LOGP_COLUMN.format(k): packed[:, k] for k in range(TOP_K)}

    for refresh in (False, True):
        tu.assign_non_tensor(
            data,
            dp_size=1,
            global_batch_size=2,
            **{losses.MODE_KEY: "train", losses.ON_POLICY_KEY: False, losses.REFRESH_KEY: refresh},
        )
        loss, metrics = losses.mt_opd_loss(CONFIG, model_output=model_output, data=data)
        expected = 0.0
        for row, length in ((0, 2), (1, 4)):
            rows = slice(row, row + 1), slice(0, length)
            advantages = losses.refresh_opd_advantage(
                (log_prob if refresh else old)[rows], teacher[rows], torch.ones(1, length)
            )
            per_token, _, _ = losses.clipped_topk_loss(log_prob[rows], old[rows], advantages * weight[row], CONFIG)
            expected += per_token.mean().item() / 2
        assert math.isclose(loss.item(), expected, rel_tol=1e-5), (refresh, loss.item(), expected)
        assert ("mt_opd/m4_advantage_refreshed" in metrics) == refresh
