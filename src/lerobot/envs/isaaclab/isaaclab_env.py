"""
TODO(branyang02): WIP, creating IsaaclabEnv subclass for LeRobotEnv
"""

import argparse
from copy import deepcopy
from typing import Any

import einops
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import trange

from lerobot.envs.base_env import LeRobotBaseEnv, RolloutResult
from lerobot.envs.configs import IsaacLabEnvConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import inside_slurm


def print_nested_keys(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            print_nested_keys(v, new_prefix)
    elif isinstance(d, torch.Tensor):
        print(f"{prefix}: Tensor, shape={tuple(d.shape)}, dtype={d.dtype}, device={d.device}")
    else:
        print(f"{prefix}: {type(d).__name__}")


class IsaacLabEnv(LeRobotBaseEnv):
    def __init__(self, config: IsaacLabEnvConfig, num_envs: int, isaaclab_env, simulation_app: Any):
        super().__init__(config, num_envs)
        self.isaaclab_env = isaaclab_env
        self.simulation_app = simulation_app

        # HACK(branyang02): isaaclab_env.render() only returns a single image (x, y, 3)
        # https://isaac-sim.github.io/IsaacLab/v2.1.0/_modules/isaaclab/envs/manager_based_rl_env.html#ManagerBasedRLEnv.render
        # Therefore we save the observation images to self._render_images instead

        self._render_images = None

    @property
    def _max_steps(self) -> int:
        return self.isaaclab_env.env.max_episode_length

    def _preprocess_observation(self, raw_observation: dict, task_description: str) -> dict:
        # print_nested_keys(raw_observation)
        # policy.arm_joint_pos: Tensor, shape=(7,), dtype=torch.float32, device=cpu
        # policy.gripper_pos: Tensor, shape=(1,), dtype=torch.float32, device=cpu
        # policy.external_cam: Tensor, shape=(2, 720, 1280, 3), dtype=torch.uint8, device=cpu
        # policy.wrist_cam: Tensor, shape=(2, 720, 1280, 3), dtype=torch.uint8, device=cpu

        def _preprocess_image(img: torch.Tensor) -> torch.Tensor:
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255
            return img

        self._render_images = torch.concat(
            [
                raw_observation["policy"]["external_cam"],
                raw_observation["policy"]["wrist_cam"],
            ],
            dim=2,  # Concatenate horizontally (width)
        )

        processed = {
            "observation.images.external_cam": _preprocess_image(raw_observation["policy"]["external_cam"]),
            "observation.images.wrist_cam": _preprocess_image(raw_observation["policy"]["wrist_cam"]),
            "observation.state": torch.concat(
                [
                    raw_observation["policy"]["arm_joint_pos"],
                    raw_observation["policy"]["gripper_pos"],
                ],
                dim=-1,
            ),
            "task": [task_description] * self.num_envs,
        }

        return processed

    @classmethod
    def create_env(cls, config: IsaacLabEnvConfig, num_envs: int, use_async_envs: bool) -> "IsaacLabEnv":
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
        AppLauncher.add_app_launcher_args(parser)
        args_cli, _ = parser.parse_known_args()
        args_cli.enable_cameras = config.enable_cameras
        args_cli.headless = config.headless
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        # All IsaacLab dependent modules should be imported after the app is launched
        import lerobot.envs.isaaclab.environments  # noqa: F401, I001
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(
            config.task,
            device=args_cli.device,
            num_envs=num_envs,
            use_fabric=config.use_fabric,
        )

        match config.task_description:
            case "put the cube in the bowl":
                scene = 1
            case "put the can in the mug":
                scene = 2
            case "put banana in the bin":
                scene = 3
            case _:
                raise ValueError(f"Instruction '{config.task_description}' not recognized")

        env_cfg.set_scene(scene)

        env = gym.make(config.task, cfg=env_cfg, render_mode="rgb_array")
        return cls(config, num_envs=num_envs, isaaclab_env=env, simulation_app=simulation_app)

    def close(self) -> None:
        self.isaaclab_env.close()
        self.simulation_app.close()

    def _reset(self, seeds: list[int] | None = None) -> tuple[dict, dict]:
        reset_results = self.isaaclab_env.reset(seed=seeds[0])  # isaaclab only supports single seed reset
        observation, info = (self._move_to_device(res, self.env_device) for res in reset_results)
        observation = self._preprocess_observation(observation, task_description=self.config.task_description)
        return observation, info

    def _render(self) -> torch.Tensor:
        return self._render_images

    def _step(
        self, action: torch.Tensor, done: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.isaaclab_env.action_space.sample()
        action = self._move_to_device(action, self.isaaclab_env.env.device)
        step_results = self.isaaclab_env.step(action)
        observation, reward, terminated, truncated, info = (
            self._move_to_device(res, self.env_device) for res in step_results
        )
        observation = self._preprocess_observation(observation, task_description=self.config.task_description)

        print_nested_keys(info)
        done = terminated | done

        return observation, reward, done, done

    def rollout(
        self, policy: PreTrainedPolicy, seeds: list[int] | None = None, return_observations: bool = False
    ) -> RolloutResult:
        """
        Run a rollout with the given policy in the environment.
        Args:
            policy: The policy to use for action selection.
            seeds: Optional list of seeds for resetting the environment.
            return_observations: Whether to return the observations during the rollout.
        Returns:
            A RolloutResult object containing:
                - action: Actions taken during the rollout (shape: (batch, sequence, action_dim)).
                - reward: Rewards received during the rollout (shape: (batch, sequence)).
                - success: Successes during the rollout (shape: (batch, sequence)).
                - done: Done flags during the rollout (shape: (batch, sequence)).
                - frames: Rendered frames during the rollout (shape: (batch, sequence, height, width, channels)).
                - observation: Optional observations during the rollout (shape: (batch, sequence, observation_dim)).
        """

        assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
        policy_device = get_device_from_parameters(policy)

        all_actions, all_rewards, all_successes, all_dones = [], [], [], []
        all_observations, all_frames = [], []

        # Reset the policy and environments.
        policy.reset()
        observation, info = self._reset(seeds=seeds)
        all_frames.append(self._render().to(self.env_device))

        step = 0
        done = torch.tensor([False] * self.num_envs, dtype=torch.bool, device=self.env_device)
        progbar = trange(
            self._max_steps,
            desc=f"Running rollout with at most {self._max_steps} steps",
            disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
            leave=False,
        )

        while not torch.all(done) and step < self._max_steps:
            if return_observations:
                all_observations.append(deepcopy(observation))

            # Move observation to the same device as the policy.
            observation = {
                k: v.to(policy_device, non_blocking=policy_device.type == "cuda") if torch.is_tensor(v) else v
                for k, v in observation.items()
            }

            with torch.inference_mode():
                # action = policy.select_action(observation)
                action = torch.zeros(0)  # Placeholder for action selection, replace with actual policy call.

            observation, reward, done, successes = self._step(action, done)
            all_frames.append(self._render().to(self.env_device))
            if step + 1 >= self._max_steps:  # Force done if we reach max steps.
                done = torch.tensor([True] * self.num_envs, dtype=torch.bool, device=self.env_device)

            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            all_successes.append(successes)

            step += 1
            running_success_rate = torch.cat(all_successes, dim=0).float().mean()
            progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
            progbar.update()

        # Track the final observation.
        if return_observations:
            all_observations.append(deepcopy(observation))

        ret = RolloutResult(
            action=torch.stack(all_actions, dim=1),  # (batch, sequence, action_dim)
            reward=torch.stack(all_rewards, dim=1),  # (batch, sequence)
            success=torch.stack(all_successes, dim=1),  # (batch, sequence)
            done=torch.stack(all_dones, dim=1),  # (batch, sequence)
            frames=torch.stack(all_frames, dim=1),  # (batch, sequence, height, width, channels)
        )

        if return_observations:
            stacked_observations = {}
            for key in all_observations[0]:
                if isinstance(all_observations[0][key], torch.Tensor):
                    stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
            ret.observation = stacked_observations
        else:
            ret.observation = None

        return ret
