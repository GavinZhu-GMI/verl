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
"""Multi-teacher on-policy distillation (Open-MOPD's M-OPD stage) on the engine workers.

A port of the M-OPD training step of https://github.com/BytedTsinghua-SIA/Open-MOPD (commit 4809a96,
`training/verl`, itself a patch of verl 0eb50ec4) onto the unified model engine:

- `mt_opd.py` and `domain_weighted_sampler.py` are that repository's files (the sampler's base class aside).
- `losses.py` is the engine-side half: the student's top-K, a model's log-probs on given ids, and the per-id
  clipped loss with the advantage refresh.
- `trainer.py` is the driver-side half, a `RayPPOTrainer` subclass for the legacy trainer.
- `replay.py` runs the step on a recorded batch of the reference, as a check.

Run with `trainer.use_v1=False +trainer.v0_trainer_class=verl.trainer.mt_opd.trainer.MTOPDTrainer`.
"""
