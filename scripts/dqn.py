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
    def __init__(self, env_config, Pong=False):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.pong = Pong

        # If Pong -> use convolutional NN
        if self.pong == True:
            self.convert_action = {0:2,1:3} # Converts actions to fit Pong
            self.layer_stack = nn.Sequential(
                nn.Conv2d(env_config['observation_stack_size'], 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_actions)
            )
        else:
            self.layer_stack = nn.Sequential(
                nn.Linear(4,256),
                nn.ReLU(),
                nn.Linear(256, self.n_actions)
            )
    
      
    
    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        return self.layer_stack(x)
    
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

        if random.random() > self.eps_start or exploit==True:
            return_tensor = self.forward(observation) # tensor([1,2])
            return torch.argmax(return_tensor, dim=1).unsqueeze(0)   # Returns tensor size (1 x 1)
                                        
        else:
            return torch.tensor([[random.randint(0, self.n_actions-1)]], device=device) # Returns tensor size (1 x 1)
                                            
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
    transitions = Transition(*samples)
    mask = torch.tensor(tuple(map(lambda state: state is not None, # Create mask, if next state == None --> False else True
                                          transitions.next_state)), device=device, dtype=torch.bool) 
    non_term_next_states = torch.cat([state for state in transitions.next_state
                                                if state is not None])
    
    state_tensor = torch.cat(transitions.state).to(device)        # Each row contains 4 values hence we have tensor size (32 x 4)
    action_tensor = torch.cat(transitions.action).to(device)     # Tensor size (32 x 1)
    reward_tensor = torch.cat(transitions.reward).to(device)      # Tensor size (32 x 1) 

    # Compute q values with DQN NN
    q_values = dqn(state_tensor).gather(1, action_tensor)
    
    # Compute q values for target-DQN NN
    q_value_targets = torch.zeros(target_dqn.batch_size, device=device)
    with torch.no_grad():
        q_value_targets[mask] = target_dqn(non_term_next_states).max(1)[0]
    q_value_targets = reward_tensor+(target_dqn.gamma*q_value_targets)
    
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