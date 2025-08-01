"""
TODO(branyang02): WIP, creating IsaaclabEnv subclass for LeRobotEnv
"""

import argparse
import logging
from typing import Any

import einops
import gymnasium as gym
import torch

from lerobot.envs.base_env import LeRobotBaseEnv
from lerobot.envs.configs import IsaacLabEnvConfig


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
        # policy.arm_joint_pos: Tensor, shape=(7,), dtype=torch.float32, device=cuda:0
        # policy.gripper_pos: Tensor, shape=(1,), dtype=torch.float32, device=cuda:0
        # policy.external_cam: Tensor, shape=(1, 720, 1280, 3), dtype=torch.uint8, device=cuda:0
        # policy.wrist_cam: Tensor, shape=(1, 720, 1280, 3), dtype=torch.uint8, device=cuda:0
        def _preprocess_image(img: torch.Tensor) -> torch.Tensor:
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            if img.max() > 1.0:
                img /= 255
            return img

        self._render_images = torch.concat(
            [
                raw_observation["policy"]["external_cam"],
                raw_observation["policy"]["wrist_cam"],
            ],
            dim=2,  # Concatenate horizontally for rendering
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
            ).unsqueeze(0),  # Add batch dimension
            "task": [task_description] * self.num_envs,
        }

        return processed

    @classmethod
    def create_env(cls, config: IsaacLabEnvConfig, num_envs: int, use_async_envs: bool) -> "IsaacLabEnv":
        if num_envs > 1:
            logging.warning("IsaacLabEnv does not support multiple environments. Using num_envs=1 instead.")
            num_envs = 1
        if use_async_envs:
            logging.warning(
                "IsaacLabEnv does not support async environments. Using synchronous mode instead."
            )

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
            num_envs=1,
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
        return cls(config, num_envs=1, isaaclab_env=env, simulation_app=simulation_app)

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
        step_results = self.isaaclab_env.step(action)
        observation, reward, terminated, truncated, info = (
            self._move_to_device(res, self.env_device) for res in step_results
        )
        observation = self._preprocess_observation(observation, task_description=self.config.task_description)

        # TODO(branyang02): Check handle success condition (copied from GymEnv)
        if "final_info" in info:
            successes = torch.tensor(
                [info["is_success"] if info is not None else False for info in info["final_info"]],
                device=self.env_device,
                dtype=torch.bool,
            )
        else:
            successes = torch.zeros(self.num_envs, device=self.env_device, dtype=torch.bool)

        done = terminated | truncated | done

        return observation, reward, done, successes
