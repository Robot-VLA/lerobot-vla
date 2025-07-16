import importlib

import einops
import gymnasium as gym
import numpy as np
import torch

from lerobot.envs.base_env import LeRobotBaseEnv
from lerobot.envs.configs import EnvConfig


class GymEnv(LeRobotBaseEnv):
    def __init__(
        self, config: EnvConfig, num_envs: int, gym_env: gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv
    ):
        super().__init__(config, num_envs)
        self.gym_env = gym_env

    @property
    def _max_steps(self) -> int:
        return self.gym_env.call("_max_episode_steps")[0]

    def _preprocess_observation(self, observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
        """Convert environment observation to LeRobot format observation.
        Args:
            observation: Dictionary of observation batches from a Gym vector environment.
        Returns:
            Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
        """
        # map to expected inputs for the policy
        return_observations = {}
        if "pixels" in observations:
            if isinstance(observations["pixels"], dict):
                imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
            else:
                imgs = {"observation.image": observations["pixels"]}

            for imgkey, img in imgs.items():
                # TODO(aliberts, rcadene): use transforms.ToTensor()?

                # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
                # This is the case for human-in-the-loop RL where there is only one environment.
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                # sanity check that images are channel last
                _, h, w, c = img.shape
                assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

                # sanity check that images are uint8
                assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

                # convert to channel first of type float32 in range [0,1]
                img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
                img = img.type(torch.float32)
                img /= 255

                return_observations[imgkey] = img

        if "environment_state" in observations:
            env_state = observations["environment_state"].float()
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)

            return_observations["observation.environment_state"] = env_state

        # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
        agent_pos = observations["agent_pos"].float()
        if agent_pos.dim() == 1:
            agent_pos = agent_pos.unsqueeze(0)
        return_observations["observation.state"] = agent_pos

        return return_observations

    @classmethod
    def create_env(cls, config: EnvConfig, num_envs: int, use_async_envs: bool) -> "GymEnv":
        """
        Makes a gym vector environment according to the config.
        Args:
            cfg (EnvConfig): the config of the environment to instantiate.
            n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
            use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
                False.
        """
        package_name = f"gym_{config.type}"
        try:
            importlib.import_module(package_name)
        except ModuleNotFoundError as e:
            print(
                f"{package_name} is not installed. Please install it with `pip install 'lerobot[{config.type}]'`"
            )
            raise e
        gym_handle = f"{package_name}/{config.task}"
        # batched version of the env that returns an observation of shape (b, c)
        env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
        env = env_cls(
            [
                lambda: gym.make(gym_handle, disable_env_checker=True, **config.gym_kwargs)
                for _ in range(num_envs)
            ]
        )
        return cls(config, num_envs, env)

    def close(self) -> None:
        self.gym_env.close()

    def _reset(self, seeds: list[int] | None = None) -> tuple[dict, dict]:
        reset_results = self.gym_env.reset(seed=seeds)
        observation, info = (self._move_to_device(res, self.env_device) for res in reset_results)
        observation = self._preprocess_observation(observation)
        return observation, info

    def _render(self) -> torch.Tensor:
        if isinstance(self.gym_env, gym.vector.SyncVectorEnv):
            return torch.from_numpy(np.stack([self.gym_env.envs[i].render() for i in range(self.num_envs)]))
        elif isinstance(self.gym_env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            return torch.from_numpy(np.stack(self.gym_env.call("render")[: self.num_envs]))
        else:
            raise TypeError(
                f"Unsupported gym environment type: {type(self.gym_env)}. "
                "Expected AsyncVectorEnv or SyncVectorEnv."
            )

    def _step(
        self, action: torch.Tensor, done: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        # gym_env expects numpy arrays in step()
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        step_results = self.gym_env.step(action)
        observation, reward, terminated, truncated, info = (
            self._move_to_device(res, self.env_device) for res in step_results
        )
        observation = self._preprocess_observation(observation)

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
