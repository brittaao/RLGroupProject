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
        
        #print(observation.type())
        #observation = torch.from_numpy(observation) # Convert np.array --> tensor
        if random.random() > self.eps_start:
            #print("exploit")
            return_tensor = self.forward(observation) # tensor([1,2])
            #print(return_tensor)
            return torch.argmax(return_tensor, dim=1).unsqueeze(0)   # tensor[1] -> tensor([1,1])
         
                                           
        else:
            #print("Explore")
            #print(torch.IntTensor([[random.randint(0, self.n_actions-1)]]).unsqueeze(0))
            return torch.IntTensor([[random.randint(0, self.n_actions-1)]]).unsqueeze(0) # tensor([0]) scalar --> 
                                                
def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return
    else:
        target_samples = memory.samples()

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!

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






    """
    batch = Transition(*zip(*transition))
     # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = torch.Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = torch.Variable(torch.cat(batch.state))
    action_batch = torch.Variable(torch.cat(batch.action))
    reward_batch = torch.Variable(torch.cat(batch.reward))
    """
    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    #        state_action_values = dqn(state_batch).gather(1, action_batch)

    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    # Compute V(s_{t+1}) for all next states.
    #       next_state_values = torch.Variable(torch.zeros(dqn.batch_size).type(torch.FloatTensor))
    #       next_state_values[non_final_mask] = dqn(non_final_next_states).max(1)[0]

    """
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * dqn.gamma) + reward_batch
    """



