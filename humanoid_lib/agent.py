import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    """MLP for the DQN Agent."""
    # [cite: 67]
    def __init__(self, state_dim, action_dim):
        # [cite: 69]
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), # [cite: 70, 71]
            nn.ReLU(), # [cite: 71]
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim) # [cite: 72]
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN Agent with Experience Replay."""
    # [cite: 65, 89]
    def __init__(self, state_dim, action_space_def):
        # 1. Define main and target Q-networks [cite: 67]
        # 2. Define optimizer
        # 3. Initialize Replay Buffer [cite: 89]
        # 4. Store action space definition for discretization [cite: 74]
        pass
    
    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        # 1. Get Q-values from network
        # 2. Reshape Q-values to [N_joints, K_bins] [cite: 76]
        # 3. Take argmax for each joint to get discrete action [cite: 76]
        pass
        
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        pass

    def learn(self):
        """Sample a batch from replay and update the network."""
        # [cite: 89]
        pass