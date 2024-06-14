from gymnasium import register

from libs.bert_sac.gymnasium_envs.ants.mujoco_ant_legs import AntLegsEnv

register(
    id="gymnasium_env/ants_4_legs",
    entry_point="libs.bert_sac.gymnasium_envs.ants:AntLegsEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs={"num_obs": 27},
)

__all__ = ["AntLegsEnv"]
