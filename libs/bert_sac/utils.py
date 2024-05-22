import gymnasium as gym
from gymnasium.envs.registration import EnvSpec


def make_env(env_id: str | EnvSpec, seed: int | None = None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk
