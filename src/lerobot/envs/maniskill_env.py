import einops
import gymnasium as gym
import torch

from lerobot.envs.base_env import LeRobotBaseEnv
from lerobot.envs.configs import EnvConfig, ManiSkillEnvConfig


class ManiSkillEnv(LeRobotBaseEnv):
    def __init__(self, config: ManiSkillEnvConfig, num_envs: int, maniskill_env):
        super().__init__(config, num_envs)
        # https://maniskill.readthedocs.io/en/latest/index.html
        self.maniskill_env = maniskill_env

    @property
    def _max_steps(self) -> int:
        return self.config.max_episode_steps

    def _preprocess_observation(self, raw_observation: dict, task_description: str) -> dict:
        # Process images
        def _preprocess_image(img: torch.Tensor) -> torch.Tensor:
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255
            return img

        processed = {
            "observation.images.base_camera": _preprocess_image(
                raw_observation["sensor_data"]["base_camera"]["Color"][:, :, :, :3]
            ),
            "observation.images.hand_camera": _preprocess_image(
                raw_observation["sensor_data"]["hand_camera"]["Color"][:, :, :, :3]
            ),
            "observation.state": torch.concat(
                [raw_observation["agent"]["qpos"], raw_observation["agent"]["qvel"]], dim=-1
            ),
            "task": [task_description] * self.num_envs,
        }

        return processed

    @classmethod
    def create_env(cls, config: EnvConfig, num_envs: int, use_async_envs: bool) -> "ManiSkillEnv":
        import mani_skill.envs  # noqa: F401

        if use_async_envs:
            raise NotImplementedError(
                "Async vector environments are not supported for ManiSkill environments. "
                "Please set `use_async_envs=False`."
            )
        env = gym.make(config.task, num_envs=num_envs, **config.gym_kwargs)
        return cls(config, num_envs=num_envs, maniskill_env=env)

    def close(self) -> None:
        self.maniskill_env.close()

    def _reset(self, seeds: list[int] | None = None) -> tuple[dict, dict]:
        reset_results = self.maniskill_env.reset(seed=seeds)
        observation, info = (self._move_to_device(res, self.config.env_device) for res in reset_results)
        observation = self._preprocess_observation(
            raw_observation=observation, task_description=self.config.task_description
        )
        return observation, info

    def _render(self) -> torch.Tensor:
        return self.maniskill_env.render()

    def _step(
        self, action: torch.Tensor, done: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        # maniskill_env expects numpy arrays in step()
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        step_results = self.maniskill_env.step(action)
        observation, reward, terminated, truncated, info = (
            self._move_to_device(res, self.config.env_device) for res in step_results
        )

        # NOTE: truncated terminates in self.maniskill_env._max_episode_steps, which is often shorter than
        # the dataset demonstration length.
        # We only use terminated to indicate the end of an episode.
        observation = self._preprocess_observation(
            raw_observation=observation, task_description=self.config.task_description
        )
        successes = info["success"]
        done = terminated | done

        return observation, reward, done, successes
