import random
from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    """A simple FIFO experience replay buffer for DQN agents."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        # Convert to numpy arrays first for efficiency, then to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)) # Action indices are integers
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)) # Use float for calculations (0.0 or 1.0)
        
        return states, actions, rewards, next_states, dones


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)