#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from lerobot.envs.factory import make_env, make_env_config
from tests.utils import DEVICE

AVAILABLE_ENV_NAMES = ["aloha", "pusht", "xarm", "maniskill", "isaaclab"]


class MockPolicyForEnv(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.ones(4, 4))
        self.action_dim = action_dim

    def reset(self) -> None:
        pass

    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        return torch.zeros((batch["observation.state"].shape[0], self.action_dim), dtype=torch.float32)


@pytest.mark.parametrize("env_name", AVAILABLE_ENV_NAMES)
def test_all_envs(env_name):
    cfg = make_env_config(env_name)
    env = make_env(cfg, n_envs=2)
    try:
        mock_policy = MockPolicyForEnv(action_dim=cfg.features["action"].shape[0]).to(DEVICE)
        rollout_data = env.rollout(policy=mock_policy, seeds=[0, 42])
        assert rollout_data is not None
    finally:
        env.close()
