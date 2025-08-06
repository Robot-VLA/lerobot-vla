"""
uv run pytest -s tests/envs/test_envs.py

- requires uv sync --all-extras
"""

from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from lerobot.envs.factory import make_env, make_env_config
from tests.utils import DEVICE

AVAILABLE_ENV_NAMES = ["aloha", "pusht", "xarm", "maniskill"]


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
    num_envs = 5
    seeds = np.random.randint(0, 1000, size=num_envs).tolist()
    cfg = make_env_config(env_name)
    env = make_env(cfg, n_envs=5)
    try:
        mock_policy = MockPolicyForEnv(action_dim=cfg.features["action"].shape[0]).to(DEVICE)
        rollout_data = env.rollout(policy=mock_policy, seeds=seeds)
        assert rollout_data is not None
    finally:
        env.close()
