import argparse
import pandas as pd
import gymnasium as gym
import torch

import config
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='CartPole-v1')
parser.add_argument('--path', type=str, help='Path to stored DQN model.')
parser.add_argument('--n_eval_episodes', type=int, default=100, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
parser.set_defaults(render=False)
parser.set_defaults(save_video=False)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'ALE/Pong-v5': config.Pong
}


def evaluate_policy(dqn, env, env_config, args, n_episodes, render=False, verbose=False):
    """Runs {n_episodes} episodes to evaluate current policy."""
    total_return = 0
    episode_returns = []
    for i in range(n_episodes):
        obs, info = env.reset()
        obs = preprocess(obs, env=args.env)
        if dqn.pong:
            obs_stack = torch.cat(env_config['observation_stack_size'] * [obs]).unsqueeze(0).to(device)
        terminated = False
        truncated = False
        episode_return = 0

        while not terminated and not truncated:
            if render:
                env.render()
            if dqn.pong:
                action = dqn.act(obs_stack, exploit=True).item()
                obs, reward, terminated, truncated, info = env.step(dqn.convert_action[action])
            else:
                action = dqn.act(obs, exploit=True).item()
                obs, reward, terminated, truncated, info = env.step(action)
            obs = preprocess(obs, env=args.env)

            if dqn.pong:
                obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)

            episode_return += reward
        
        total_return += episode_return
        
        if verbose:
            print(f'Finished episode {i+1} with a total return of {episode_return}')
            episode_returns.append(episode_return)

    df = pd.DataFrame(episode_returns) 
    df.to_csv('target10.csv')
    return total_return / n_episodes

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    if args.save_video:
        env = gym.make(args.env, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, './video/', episode_trigger=lambda episode_id: True)

    # Load model from provided path.
    dqn = torch.load(args.path, map_location=torch.device('cpu'))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, args, args.n_eval_episodes, render=args.render and not args.save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')

    env.close()