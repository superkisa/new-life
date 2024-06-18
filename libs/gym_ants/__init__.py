from pathlib import Path

import gymnasium

ASSETS_PATH = Path(__file__).resolve().parent / "assets"

gymnasium.register(
    id="manylegs/ants_3_legs",
    entry_point="gym_ants.ants:AntLegsEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs={
        "xml_file": str(ASSETS_PATH / "ant-3.xml"),
        "num_obs": 23,
    },
)

gymnasium.register(
    id="manylegs/ants_4_legs",
    entry_point="gym_ants.ants:AntLegsEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs={
        "xml_file": str(ASSETS_PATH / "ant-4.xml"),
        "num_obs": 27,
    },
)

gymnasium.register(
    id="manylegs/ants_5_legs",
    entry_point="gym_ants.ants:AntLegsEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs={
        "xml_file": str(ASSETS_PATH / "ant-5.xml"),
        "num_obs": 31,
    },
)

gymnasium.register(
    id="manylegs/ants_6_legs",
    entry_point="gym_ants.ants:AntLegsEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs={
        "xml_file": str(ASSETS_PATH / "ant-6.xml"),
        "num_obs": 35,
    },
)
