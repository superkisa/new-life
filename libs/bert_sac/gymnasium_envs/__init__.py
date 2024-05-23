from gymnasium import register
from libs.bert_sac.gymnasium_envs.mujoco_ant_legs import AntLegsEnv

register(
    id="mujoco/AntLegs",
    entry_point="libs.bert_sac.gymnasium_envs:AntLegsEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

__all__ = ["AntLegsEnv"]
