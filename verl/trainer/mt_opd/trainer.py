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
"""Driver-side half of M-OPD: the legacy trainer with the reference's reward and update.

One step, after the rollout:

1. A student forward in eval mode gives its top-K ids and log-probs S at every response position.
2. Each teacher scores those ids; the teacher of a sequence's domain gives T.
3. Reward r = softmax_K(S) * (T - S); a per-sequence domain weight balances the domains' gradient shares.
4. The update is the per-id clipped loss over [N, K]; with `refresh_advantage` every inner update rebuilds
   the reward from its own forward.

Settings live under `mt_opd.*`, all required (pass them with `+`):
  teacher_domains, teacher_paths   one teacher per domain, same order
  teacher_param_offload            True keeps a teacher's parameters on CPU between its forwards: less GPU
                                   memory during the update, a load per step
  top_k                            K, the ids scored per position: the reward's support against memory and
                                   teacher time
  refresh_advantage                True rebuilds the reward from each inner update's own forward (no extra
                                   forward); False keeps the rollout-time reward for all of them
  target_share_domains, target_share_values
                                   the share of the gradient each domain should get
  normalize_reward_scale           0 balances token shares only; up to 2, how strongly a domain's reward
                                   magnitude also moves its weight
  reward_scale_direction           divide | multiply: weight a domain down or up with its reward magnitude
(the last three are the arguments of `mt_opd.compute_domain_loss_weights`), and `data.domain_weights` for the
sampler. `algorithm.adv_estimator` must be `token_reward_direct`. Only the reference's `only_stu` candidate set
and `student_p` reward weight are ported. A LoRA student is not supported: the teachers take the student's
model settings.

"The reference" is Open-MOPD 4809a96, paths under its `training/verl/verl/`.
"""

import copy
from functools import partial

import numpy as np
import ray
from omegaconf import OmegaConf

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.trainer.mt_opd.domain_weighted_sampler import DomainWeightedSampler
from verl.trainer.mt_opd.losses import (
    ADVANTAGES_KEY,
    DOMAIN_WEIGHT_KEY,
    ID_COLUMN,
    IDS_KEY,
    LOGP_COLUMN,
    MODE_KEY,
    ON_POLICY_KEY,
    REFRESH_KEY,
    STUDENT_LOGP_KEY,
    TEACHER_LOGP_KEY,
    TOP_K_KEY,
    mt_opd_loss,
    stack_columns,
)
from verl.trainer.mt_opd.mt_opd import (
    build_domain_weights,
    compute_domain_loss_weights,
    compute_domain_share_metrics,
    compute_teacher_conflict_metrics,
    refresh_opd_advantage,
    select_routed_teacher_logprobs,
)
from verl.trainer.ppo.core_algos import register_adv_est
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.py_functional import rename_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.config import TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

ADV_ESTIMATOR = "token_reward_direct"


@register_adv_est(ADV_ESTIMATOR)
def compute_token_reward_direct_advantage(token_level_rewards, response_mask, **kwargs):
    """The token-level reward is the advantage. Here it is the reward summed over the K ids, for the
    trainer's metrics; the loss reads the per-id `opd_advantages`."""
    advantages = token_level_rewards * response_mask
    return advantages, advantages.clone()


def _required(config, key: str):
    value = OmegaConf.select(config, key, default=None)
    if value is None:
        raise ValueError(f"M-OPD needs `{key}`")
    return value


class MTOPDTrainer(RayPPOTrainer):
    def __init__(self, config, *args, **kwargs):
        self.teacher_domains = list(_required(config, "mt_opd.teacher_domains"))
        self.teacher_paths = list(_required(config, "mt_opd.teacher_paths"))
        if len(self.teacher_domains) != len(self.teacher_paths):
            raise ValueError(f"M-OPD: {len(self.teacher_domains)} domains, {len(self.teacher_paths)} teacher paths")
        self.teacher_param_offload = bool(_required(config, "mt_opd.teacher_param_offload"))
        self.top_k = int(_required(config, "mt_opd.top_k"))
        self.refresh_advantage = bool(_required(config, "mt_opd.refresh_advantage"))
        share_domains = list(_required(config, "mt_opd.target_share_domains"))
        share_values = [float(v) for v in _required(config, "mt_opd.target_share_values")]
        if len(share_domains) != len(share_values):
            raise ValueError("M-OPD: target_share_domains and target_share_values differ in length")
        self.target_shares = dict(zip(share_domains, share_values, strict=True))
        self.normalize_reward_scale = float(_required(config, "mt_opd.normalize_reward_scale"))
        self.reward_scale_direction = str(_required(config, "mt_opd.reward_scale_direction"))
        if config.actor_rollout_ref.model.lora_rank > 0:
            raise ValueError("M-OPD: a LoRA student is not supported")
        if config.algorithm.adv_estimator != ADV_ESTIMATOR:
            raise ValueError(f"M-OPD needs algorithm.adv_estimator={ADV_ESTIMATOR}")
        self._teacher_names = [f"mt_opd_teacher_{domain}" for domain in self.teacher_domains]
        self._old_log_prob = None
        super().__init__(config, *args, **kwargs)

    # ---- construction -------------------------------------------------------------------------------------

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        # the reference's sampler replaces the one passed in (so `data.shuffle` has no effect): an exact domain
        # mixture in every batch; it needs the built dataset
        if train_dataset is None:
            raise ValueError("M-OPD needs the training dataset at construction, for its sampler")
        sampler = DomainWeightedSampler(train_dataset, self.config.data)
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, sampler)

    def _extra_colocated_workers(self) -> dict:
        """One forward-only engine per teacher, in the actor's processes."""
        actor_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
        rollout = self.config.actor_rollout_ref.rollout
        workers = {}
        for name, path in zip(self._teacher_names, self.teacher_paths, strict=True):
            model = OmegaConf.to_container(self.config.actor_rollout_ref.model, resolve=True)
            model["path"] = path
            engine_config = copy.deepcopy(actor_config.engine)
            engine_config.forward_only = True
            engine_config.param_offload = self.teacher_param_offload
            engine_config.use_dynamic_bsz = rollout.log_prob_use_dynamic_bsz
            engine_config.infer_max_token_len_per_gpu = rollout.log_prob_max_token_len_per_gpu
            engine_config.infer_micro_batch_size_per_gpu = rollout.log_prob_micro_batch_size_per_gpu
            teacher_config = TrainingWorkerConfig(
                model_type="language_model",
                model_config=omega_conf_to_dataclass(OmegaConf.create(model)),
                engine_config=engine_config,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )
            workers[name] = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=teacher_config)
        return workers

    def init_workers(self):
        super().init_workers()
        actor_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
        self.actor_rollout_wg.set_loss_fn(partial(mt_opd_loss, config=actor_config))
        self.teacher_wgs = [self.all_wg[name] for name in self._teacher_names]
        for teacher_wg in self.teacher_wgs:
            teacher_wg.reset()
            teacher_wg.set_loss_fn(partial(mt_opd_loss, config=None))

    # ---- one step -----------------------------------------------------------------------------------------

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # the generation batch takes every key the reward does not use; the routing needs the domain back
        domains = batch.non_tensor_batch["domain"]
        gen_batch = super()._get_gen_batch(batch)
        batch.non_tensor_batch["domain"] = domains
        return gen_batch

    def _forward_only(self, worker_group_call, batch: DataProto, **non_tensors):
        batch_td = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(batch_td, compute_loss=True, distillation_use_topk=True, **non_tensors)
        return worker_group_call(batch_td), batch_td

    def _after_rollout(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        top_k = self.top_k
        response_mask = batch.batch["response_mask"]

        with marked_timer("mt_opd_student", timing_raw, color="blue"):
            # this forward is also the trainer's old-log-prob forward (`_compute_old_log_prob`)
            output, batch_td = self._forward_only(
                self.actor_rollout_wg.compute_log_prob,
                batch,
                calculate_entropy=True,
                **{MODE_KEY: "topk", TOP_K_KEY: top_k},
            )
            student_ids = stack_columns(partial(tu.get, output), batch_td, ID_COLUMN, top_k)
            student_logp = stack_columns(partial(tu.get, output), batch_td, LOGP_COLUMN, top_k).float()
            old_log_prob = DataProto.from_dict(
                tensors={
                    "old_log_probs": no_padding_2_padding(tu.get(output, "log_probs"), batch_td).float(),
                    "entropys": no_padding_2_padding(tu.get(output, "entropy"), batch_td).float(),
                }
            )
            self._old_log_prob = (old_log_prob, tu.get(output, "metrics")["mfu"])
        batch.batch[IDS_KEY] = student_ids
        batch.batch[STUDENT_LOGP_KEY] = student_logp

        teacher_logps = []
        for name, teacher_wg in zip(self._teacher_names, self.teacher_wgs, strict=True):
            with marked_timer(name, timing_raw, color="magenta"):
                output, batch_td = self._forward_only(
                    lambda td, wg=teacher_wg: wg.infer_batch(td).get(),
                    batch,
                    distillation_only=True,
                    **{MODE_KEY: "score"},
                )
                teacher_logps.append(stack_columns(partial(tu.get, output), batch_td, LOGP_COLUMN, top_k).float())

        domains = batch.non_tensor_batch["domain"]
        routing = build_domain_weights(domains, self.teacher_domains, device=student_logp.device)
        routed_logp = select_routed_teacher_logprobs(teacher_logps, routing)
        batch.batch[TEACHER_LOGP_KEY] = routed_logp

        # r = softmax_K(S) * (T - S): the reference's `only_stu` reward with the `student_p` weight
        # (`workers/actor/dp_actor.py`, `compute_distillation_reward`), which is the expression of its refresh
        reward = refresh_opd_advantage(student_logp, routed_logp, response_mask)
        batch.batch[ADVANTAGES_KEY] = reward
        # the trainer's reward and advantage keys hold the sum over the K ids; whatever the rollout's
        # scorer wrote there is not used
        batch.batch["rm_scores"] = reward.sum(dim=-1)

        metrics["mt_opd/rm_scores_mean"] = masked_mean(reward.sum(dim=-1), response_mask).item()
        metrics.update(compute_domain_share_metrics(domains=domains, response_mask=response_mask, rm_scores=reward))
        metrics.update(
            compute_teacher_conflict_metrics(
                teacher_logprobs=teacher_logps, response_mask=response_mask, domains=domains
            )
        )
        domain_weight = compute_domain_loss_weights(
            domains=domains,
            response_mask=response_mask,
            target_shares=self.target_shares,
            rm_scores=reward,
            normalize_reward_scale=self.normalize_reward_scale,
            reward_scale_direction=self.reward_scale_direction,
        )
        batch.batch[DOMAIN_WEIGHT_KEY] = domain_weight
        for domain in sorted(set(str(d) for d in domains)):
            first = next(i for i, d in enumerate(domains) if str(d) == domain)
            metrics[f"mt_opd/domain/{domain}/loss_weight"] = domain_weight[first].item()
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        if self._old_log_prob is None:
            raise RuntimeError("M-OPD: the student's rollout-time forward has not run for this step")
        result, self._old_log_prob = self._old_log_prob, None
        return result

    def _update_actor(self, batch: DataProto) -> DataProto:
        actor = self.config.actor_rollout_ref.actor
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        batch_td = left_right_2_no_padding(batch.to_tensordict())
        mini_batch_size = actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        if len(batch) % mini_batch_size != 0:
            raise ValueError(f"M-OPD: a batch of {len(batch)} is not a whole number of updates of {mini_batch_size}")
        # the reference's `on_policy` (`workers/actor/dp_actor.py`, `update_policy`): one update on fresh data
        on_policy = len(batch) == mini_batch_size and actor.ppo_epochs == 1
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            distillation_use_topk=True,
            distillation_only=True,
            global_batch_size=mini_batch_size,
            mini_batch_size=mini_batch_size,
            epochs=actor.ppo_epochs,
            seed=actor.data_loader_seed,
            dataloader_kwargs={"shuffle": actor.shuffle},
            compute_loss=True,
            **{MODE_KEY: "train", ON_POLICY_KEY: on_policy, REFRESH_KEY: self.refresh_advantage and not on_policy},
        )
        actor_output = tu.get(self.actor_rollout_wg.update_actor(batch_td), "metrics")
        # grad_norm has one value per inner update, a loss metric one per micro-batch and rank
        updates = len(actor_output["grad_norm"])
        per_update = {
            f"mt_opd/inner_update_{i + 1}/{key}": float(np.mean(chunk))
            for key in ("grad_norm", "ppo_kl", "pg_loss")
            for i, chunk in enumerate(np.array_split(np.asarray(actor_output[key]), updates))
        }
        actor_output = rename_dict(actor_output, "actor/")
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        # a metric the loss already names in full keeps its name
        for key in [key for key in actor_output if key.startswith("actor/mt_opd/")]:
            actor_output[key.removeprefix("actor/")] = actor_output.pop(key)
        actor_output.update(per_update)
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
