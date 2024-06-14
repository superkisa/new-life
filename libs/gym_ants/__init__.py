import gymnasium

gymnasium.register(
    id="manylegs/ants_4_legs",
    entry_point="gym_ants.ants:AntLegsEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs={"num_obs": 27},
)
