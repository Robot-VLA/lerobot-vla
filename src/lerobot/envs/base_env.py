import abc
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from lerobot.envs.configs import EnvConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import inside_slurm


@dataclass
class RolloutResult:
    """
    All data should be in LeRobotBaseEnv.config.env_device, which is "cpu" by default.
    """

    action: torch.Tensor  # (batch, sequence, action_dim)
    reward: torch.Tensor  # (batch, sequence)
    success: torch.Tensor  # (batch, sequence)
    done: torch.Tensor  # (batch, sequence)
    frames: torch.Tensor  # (batch, sequence, height, width, channels)

    observation: Optional[dict] = None


class LeRobotBaseEnv(abc.ABC):
    def __init__(self, config: EnvConfig, num_envs: int):
        self.config = config
        self.num_envs = num_envs

    @property
    @abc.abstractmethod
    def _max_steps(self) -> int: ...

    @classmethod
    @abc.abstractmethod
    def create_env(cls, config: EnvConfig, num_envs: int, use_async_envs: bool) -> "LeRobotBaseEnv":
        """
        Create an environment instance based on the provided configuration.
        Args:
            config: Environment configuration.
            num_envs: Number of parallel environments to create.
            use_async_envs: Whether to use asynchronous vectorized environments.
        Returns:
            An instance of the environment.
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Close the environment and release resources."""
        ...

    @abc.abstractmethod
    def _reset(self, seeds: list[int] | None = None) -> tuple[dict, dict]:
        """
        Reset the environment to its initial state.
        Args:
            seeds: Optional list of seeds for resetting the environment.
        Returns:
            # NOTE: Preprocessed observation and info dictionary with additional information.
            # observation will be passed into the policy for action selection.
        """
        ...

    @abc.abstractmethod
    def _render(self) -> torch.Tensor:
        """
        Render the current state of the environment.
        Returns:
            A tensor representing the rendered frame of the environment.
            The shape is (height, width, channels) for a single frame.
            For vectorized environments, the shape is (batch, height, width, channels).
            The channels are in RGB order.
            The tensor should be of type torch.uint8.
        """
        ...

    @abc.abstractmethod
    def _step(
        self, action: torch.Tensor, done: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a step in the environment with the given action.
        Args:
            action: A tensor representing the action to take.
            done: A tensor indicating which environments are done.
        Returns:
            A tuple containing:
                - observation: The next observation after taking the action.
                - reward: The reward received for taking the action.
                - done: A tensor indicating which environments are done after this step.
                - successes: A tensor indicating whether each environment succeeded in this step.
        """
        ...

        """
        The maximum number of steps per episode.
        This is used to limit the length of rollouts.
        Returns:
            An integer representing the maximum number of steps per episode.
        """

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
        all_frames.append(self._render().to("cpu"))

        step = 0
        done = torch.tensor([False] * self.num_envs, dtype=torch.bool, device=self.config.env_device)
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
                action = policy.select_action(observation)

            observation, reward, done, successes = self._step(action, done)
            all_frames.append(self._render().to("cpu"))
            if step + 1 >= self._max_steps:  # Force done if we reach max steps.
                done = torch.tensor([True] * self.num_envs, dtype=torch.bool, device=self.config.env_device)

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

    def _move_to_device(
        self, item: dict | torch.Tensor | np.ndarray, device: str | torch.device
    ) -> dict | torch.Tensor:
        """
        Move an item (tensor, numpy array, dict, or list) to the specified device.
        """
        device = torch.device(device) if isinstance(device, str) else device

        if isinstance(item, torch.Tensor):
            return item.to(device, non_blocking=device.type == "cuda")
        elif isinstance(item, np.ndarray):
            if item.dtype == np.object_:
                return item
            else:
                return torch.from_numpy(item).to(device, non_blocking=device.type == "cuda")
        elif isinstance(item, dict):
            return {k: self._move_to_device(v, device) for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            return type(item)(self._move_to_device(v, device) for v in item)
        else:
            return item
