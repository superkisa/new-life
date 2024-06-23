import gymnasium as gym
import torch
from gymnasium.envs.registration import EnvSpec
from tqdm import tqdm

from bert_sac.sac_trainer import AntSAC


def make_env(env_id: str | EnvSpec, seed: int | None = None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def record_inference(
    model: AntSAC,
    env_name: str,
    num_steps: int = 5_000,
    episode_trigger=lambda t: t % 1 == 0,
    save_folder="artifacts/recordings",
) -> gym.Env:
    obs_env = gym.make(env_name, render_mode="rgb_array")
    num_actions = obs_env.action_space.shape[0]  # type: ignore
    obs_env = gym.wrappers.RecordVideo(
        obs_env, video_folder=save_folder, episode_trigger=episode_trigger
    )

    obs, _ = obs_env.reset()
    obs = torch.tensor(obs, device=model.device, dtype=torch.float32).unsqueeze(0)
    for _ in tqdm(range(num_steps)):
        actions, _, _ = model.actor.get_action0(obs)
        obs, _, terminated, truncated, _ = obs_env.step(
            actions.view(num_actions).detach().cpu().numpy()
        )
        if terminated or truncated:
            obs, _ = obs_env.reset()
    return obs_env
