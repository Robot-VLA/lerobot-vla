import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from .droid_environment import EnvCfg as DroidEnvCfg

gym.register(
    id="DROID",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": DroidEnvCfg,
    },
    disable_env_checker=True,
)
