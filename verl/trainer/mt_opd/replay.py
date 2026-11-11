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
"""The M-OPD step on a recorded batch: a check of this stack against a recorded step of the reference.

`ReplayMTOPDTrainer` puts the batch saved at `mt_opd.replay_batch` in place of every rollout, runs the step, and
prints how its tensors differ from the recorded ones. The file is saved by torch:
`{"tensors": {...}, "domain": [...]}`, the tensors being those of the reference's `post_advantage` DataProto dump.
Run it with `trainer.balance_batch=False` to keep the recorded order, and a short `data.max_response_length`:
the rollout still runs and is thrown away.
"""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.mt_opd.losses import ADVANTAGES_KEY, DOMAIN_WEIGHT_KEY, IDS_KEY, STUDENT_LOGP_KEY, TEACHER_LOGP_KEY
from verl.trainer.mt_opd.trainer import MTOPDTrainer

REPLAY_TENSORS = ("prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask")


class ReplayMTOPDTrainer(MTOPDTrainer):
    def _after_rollout(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        recorded = torch.load(self.config.mt_opd.replay_batch, weights_only=True)
        tensors = recorded["tensors"]
        if len(recorded["domain"]) != len(batch):
            raise ValueError(f"M-OPD replay: {len(recorded['domain'])} recorded rows, a batch of {len(batch)}")
        # the recorded tensors and domains, this batch's other keys
        replayed = DataProto.from_dict(
            tensors={key: tensors[key] for key in REPLAY_TENSORS},
            non_tensors=dict(batch.non_tensor_batch),
            meta_info=dict(batch.meta_info),
        )
        replayed.non_tensor_batch["domain"] = np.array(recorded["domain"], dtype=object)
        replayed.meta_info["global_token_num"] = replayed.batch["attention_mask"].sum(dim=-1).tolist()
        replayed = super()._after_rollout(replayed, metrics, timing_raw)
        self._report(replayed, tensors)
        return replayed

    @staticmethod
    def _report(batch: DataProto, recorded: dict) -> None:
        """Print how this stack's tensors differ from the recorded step's, on the same tokens."""
        mask = batch.batch["response_mask"].bool()
        same_ids = (batch.batch[IDS_KEY] == recorded[IDS_KEY]).all(dim=-1) & mask
        same_set = (batch.batch[IDS_KEY].sort(dim=-1).values == recorded[IDS_KEY].sort(dim=-1).values).all(dim=-1)
        print(
            f"mt_opd/replay/positions: {int(mask.sum())}, same ids in the same order: "
            f"{same_ids.sum().item() / mask.sum().item():.6f}, same id set: "
            f"{(same_set & mask).sum().item() / mask.sum().item():.6f}"
        )
        for name, key, recorded_key in (
            ("student_logp", STUDENT_LOGP_KEY, STUDENT_LOGP_KEY),
            ("teacher_logp", TEACHER_LOGP_KEY, TEACHER_LOGP_KEY),
            ("reward", ADVANTAGES_KEY, "advantages"),
        ):
            diff = (batch.batch[key] - recorded[recorded_key])[same_ids].abs()
            print(f"mt_opd/replay/{name}: mean abs diff {diff.mean().item():.3e}, max {diff.max().item():.3e}")
        weight_diff = (batch.batch[DOMAIN_WEIGHT_KEY] - recorded[DOMAIN_WEIGHT_KEY]).abs().max().item()
        print(f"mt_opd/replay/domain_loss_weight: max abs diff {weight_diff:.3e}")
