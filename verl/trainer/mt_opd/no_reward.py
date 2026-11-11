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
"""The rollout's scorer for an M-OPD run (`reward.custom_reward_function.path`).

The training reward is the teachers' log-probs, computed by the trainer after the rollout. The rollout still
calls a scorer per sample; this one scores nothing.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    return 0.0
