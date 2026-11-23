"""SO-101 lift-cube task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="ArmHand-Lift-Cube-SO101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SO101LiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101LiftPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# Same task with a short training budget, used by the local reward agent (agents/reward_agent.py).
gym.register(
    id="ArmHand-Lift-Cube-SO101-Quick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SO101LiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101LiftQuickPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
