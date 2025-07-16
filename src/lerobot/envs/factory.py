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


from lerobot.envs.base_env import LeRobotBaseEnv
from lerobot.envs.configs import AlohaEnv, EnvConfig, HILEnvConfig, ManiSkillEnvConfig, PushtEnv, XarmEnv
from lerobot.envs.gym_env import GymEnv
from lerobot.envs.maniskill_env import ManiSkillEnv


def get_env_class(name: str) -> LeRobotBaseEnv:
    """Get the environment class by name."""
    if name == "aloha" or name == "pusht" or name == "xarm" or name == "hil":
        return GymEnv
    elif name == "maniskill":
        return ManiSkillEnv
    else:
        raise ValueError(f"Environment type '{name}' is not available.")


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "hil":
        return HILEnvConfig(**kwargs)
    elif env_type == "maniskill":
        return ManiSkillEnvConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(cfg: EnvConfig, n_envs: int = 1, use_async_envs: bool = False) -> LeRobotBaseEnv:
    env_cls = get_env_class(cfg.type)
    env = env_cls.create_env(
        config=cfg,
        num_envs=n_envs,
        use_async_envs=use_async_envs,
    )
    return env
