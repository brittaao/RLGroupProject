import argparse
import csv

import gymnasium as gym
import torch
import copy

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
from gymnasium.wrappers import AtariPreprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--nmodel', type=int, help='Name of the model.', nargs='?')
parser.add_argument('--save_video', type=bool, default=True)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'ALE/Pong-v5': config.Pong
}
# Set to True if Pong else False.
Pong = True

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    if Pong:
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env_config = ENV_CONFIGS[args.env]
    nmodel = args.nmodel

    # Initialize deep Q-networks.
    if Pong:
        dqn = DQN(env_config=env_config, Pong=True).to(device)
    else:
        dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = copy.deepcopy(dqn).to(device)
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    L = []
    for episode in range(env_config['n_episodes']):
        terminated = False
        truncated = False
        obs, info = env.reset()

        counter = 0 # Counter for frequencies

        obs = preprocess(obs, env=args.env)
        if Pong:
                obs_stack = torch.cat(env_config['observation_stack_size'] * [obs]).unsqueeze(0).to(device)
        while not terminated and not truncated:
            
            # TODO: Get action from DQN.
            if Pong:
                action = dqn.act(obs_stack)            
                next_obs, reward, terminated, truncated, info = env.step(dqn.convert_action[action.item()])
            else:
                action = dqn.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            # Preprocess incoming observation.
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env)
                if Pong:
                    next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
            else:
                next_obs = None
                if Pong:
                    next_obs_stack = None
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!

            reward = torch.tensor([reward], device=device) # Convert reward to tensor
            if Pong:
                memory.push(obs_stack,action,next_obs_stack,reward)
            else:
                memory.push(obs, action, next_obs, reward)   # Push all tensors to memory
            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if counter%env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)
                
            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if counter%env_config["target_update_frequency"] == 0:
                target_dqn = copy.deepcopy(dqn)
            
            if Pong:
                obs_stack = next_obs_stack
            else:
                obs = next_obs
            counter += 1
            
            
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            L.append(mean_return)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best_model{nmodel}{episode}.pt')
        
    # Close environment after training is completed.
    with open(f'./train_results/model{nmodel}.csv', 'w') as f: 
        write = csv.writer(f)
        write.writerow(L)
    env.close()
