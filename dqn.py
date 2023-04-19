import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    
      
    
    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        

        # anneal epsilon until eps_end
        self.eps_start -= 1/self.anneal_length

        if self.eps_start < self.eps_end:
            self.eps_start = self.eps_end
        
        if random.random() > self.eps_start:
            return_tensor = self.forward(observation) # tensor([1,2])
            return torch.argmax(return_tensor, dim=1)   # Returns tensor size (1 x 1)
         
                                           
        else:
            return torch.IntTensor([[random.randint(0, self.n_actions-1)]]) # Returns IntTensor size (1 x 1)
                                                
def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    samples = memory.sample(dqn.batch_size)                      # Returns tuple batchsize x ((s),(a),(s'),(r))
    obs_tensor = torch.cat(samples[0], dim= 0).to(device)        # Each row contains 4 values hence we have tensor size (32 x 4)
    action_tensor = torch.cat(samples[1], dim= 0).to(device)     # Tensor size (32 x 1)
    next_obs_tensor = torch.cat(samples[2], dim=0).to(device)    # Tensor size (32 x 4)
    reward_tensor = torch.cat(samples[3], dim=0).to(device)      # Tensor size (32 x 1) 


    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()








