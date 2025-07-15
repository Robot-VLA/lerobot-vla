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

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import draccus
import einops
import gymnasium as gym
import torch
from torch import nn
from tqdm import trange

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.policies.utils import get_device_from_parameters
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.utils import inside_slurm


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("maniskill")
@dataclass
class ManiskillEnv(EnvConfig):
    task: str = "StackCube-v1"
    task_description: str = "Stack the red cube on top of the green cube."
    fps: int = 30
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
            "observation.images.base_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(128, 128, 3)),
            "observation.images.hand_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(128, 128, 3)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(18,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.images.base_camera": OBS_IMAGE,
            "observation.images.hand_camera": OBS_IMAGE,
            "observation.state": OBS_STATE,
        }
    )

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_mode": "sensor_data",
            "control_mode": "pd_joint_pos",  # ['pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos', 'pd_ee_delta_pose', 'pd_ee_pose', 'pd_joint_target_delta_pos', 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 'pd_joint_vel', 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel']
        }

    def create_env(self, n_envs: int = 1, use_async_envs: bool = False) -> gym.Env:
        """
        Creates a ManiSkill environment, see https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html.
        """
        import mani_skill.envs  # noqa: F401

        if use_async_envs:
            raise NotImplementedError(
                "Async vector environments are not supported for ManiSkill environments. "
                "Please set `use_async_envs=False`."
            )

        env = gym.make(
            self.task,
            num_envs=n_envs,
            **self.gym_kwargs,
        )
        return env

    def rollout(
        self,
        env,
        policy,
        seeds: list[int] | None = None,
        return_observations: bool = False,
        render_callback: Callable[[gym.Env], None] | None = None,
    ):
        assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
        device = get_device_from_parameters(policy)

        # Reset the policy and environments.
        policy.reset()
        observation, info = env.reset(seed=seeds)
        if render_callback is not None:
            render_callback(env)

        all_actions = []
        all_rewards = []
        all_successes = []
        all_dones = []

        step = 0
        # Keep track of which environments are done.
        done = torch.tensor([False] * env.num_envs, dtype=torch.bool, device=device)
        max_steps = env._max_episode_steps
        progbar = trange(
            max_steps,
            desc=f"Running rollout with at most {max_steps} steps",
            disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
            leave=False,
        )

        def process_image(img):
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255
            return img

        while not torch.all(done):
            observation = {
                "observation.images.base_camera": process_image(
                    observation["sensor_data"]["base_camera"]["Color"][:, :, :, :3]
                ),
                "observation.images.hand_camera": process_image(
                    observation["sensor_data"]["hand_camera"]["Color"][:, :, :, :3]
                ),
                "observation.state": torch.concat(
                    [
                        observation["agent"]["qpos"],  # qpos
                        observation["agent"]["qvel"],  # qvel
                    ],
                    dim=-1,
                ),
            }
            observation = {
                key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
            }  # TODO(branyang02): check if this is needed.
            observation["task"] = [self.task_description for _ in range(env.num_envs)]

            with torch.inference_mode():
                action = policy.select_action(observation)

            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

            observation, reward, terminated, truncated, info = env.step(action)
            if render_callback is not None:
                render_callback(env)

            successes = info["success"]
            done = terminated | truncated | done

            all_actions.append(torch.from_numpy(action))
            all_rewards.append(reward)
            all_dones.append(done)
            all_successes.append(successes)

            step += 1
            running_success_rate = torch.cat(all_successes, dim=0).float().mean()
            progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
            progbar.update()

        # Track the final observation.
        if return_observations:
            raise NotImplementedError(
                "Returning observations is not implemented for ManiskillEnv. "
                "You can modify the `rollout` method to return observations if needed."
            )

        # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
        ret = {
            "action": torch.stack(all_actions, dim=1).cpu(),
            "reward": torch.stack(all_rewards, dim=1).cpu(),
            "success": torch.stack(all_successes, dim=1).cpu(),
            "done": torch.stack(all_dones, dim=1).cpu(),
        }

        return ret


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class VideoRecordConfig:
    """Configuration for video recording in ManiSkill environments."""

    enabled: bool = False
    record_dir: str = "videos"
    trajectory_name: str = "trajectory"


@dataclass
class EnvTransformConfig:
    """Configuration for environment wrappers."""

    # ee_action_space_params: EEActionSpaceConfig = field(default_factory=EEActionSpaceConfig)
    control_mode: str = "gamepad"
    display_cameras: bool = False
    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    add_ee_pose_to_observation: bool = False
    crop_params_dict: Optional[dict[str, tuple[int, int, int, int]]] = None
    resize_size: Optional[tuple[int, int]] = None
    control_time_s: float = 20.0
    fixed_reset_joint_positions: Optional[Any] = None
    reset_time_s: float = 5.0
    use_gripper: bool = True
    gripper_quantization_threshold: float | None = 0.8
    gripper_penalty: float = 0.0
    gripper_penalty_in_reward: bool = False


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: Optional[RobotConfig] = None
    teleop: Optional[TeleoperatorConfig] = None
    wrapper: Optional[EnvTransformConfig] = None
    fps: int = 10
    name: str = "real_robot"
    mode: str = None  # Either "record", "replay", None
    repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    task: str = ""
    num_episodes: int = 10  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = True
    pretrained_policy_name_or_path: Optional[str] = None
    reward_classifier_pretrained_path: Optional[str] = None
    # For the reward classifier, to record more positive examples after a success
    number_of_steps_after_success: int = 0

    def gym_kwargs(self) -> dict:
        return {}


@EnvConfig.register_subclass("hil")
@dataclass
class HILEnvConfig(EnvConfig):
    """Configuration for the HIL environment."""

    type: str = "hil"
    name: str = "PandaPickCube"
    task: str = "PandaPickCubeKeyboard-v0"
    use_viewer: bool = True
    gripper_penalty: float = 0.0
    use_gamepad: bool = True
    state_dim: int = 18
    action_dim: int = 4
    fps: int = 100
    episode_length: int = 100
    video_record: VideoRecordConfig = field(default_factory=VideoRecordConfig)
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(18,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.image": OBS_IMAGE,
            "observation.state": OBS_STATE,
        }
    )
    ################# args from hilserlrobotenv
    reward_classifier_pretrained_path: Optional[str] = None
    robot_config: Optional[RobotConfig] = None
    teleop_config: Optional[TeleoperatorConfig] = None
    wrapper: Optional[EnvTransformConfig] = None
    mode: str = None  # Either "record", "replay", None
    repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    num_episodes: int = 10  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = True
    pretrained_policy_name_or_path: Optional[str] = None
    # For the reward classifier, to record more positive examples after a success
    number_of_steps_after_success: int = 0
    ############################

    @property
    def gym_kwargs(self) -> dict:
        return {
            "use_viewer": self.use_viewer,
            "use_gamepad": self.use_gamepad,
            "gripper_penalty": self.gripper_penalty,
        }
