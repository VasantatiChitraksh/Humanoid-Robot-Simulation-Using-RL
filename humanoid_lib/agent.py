import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from humanoid_lib.utils import ReplayBuffer


class QNetwork(nn.Module):
    """MLP for the DQN Agent."""

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        """Defines the forward pass of the Q-network."""
        return self.network(x)


class DQNAgent:
    """DQN Agent implementing Q-learning with function approximation, using experience replay and a target network."""

    def __init__(self, state_dim, action_space, seed,
                 lr=5e-4, gamma=0.99, tau=1e-3,
                 buffer_size=int(1e5), batch_size=64, update_every=4):
        """Initializes an Agent object."""

        self.state_dim = state_dim
        self.action_space = action_space
        self.num_joints = self.action_space.shape[0]
        self.num_bins = self.action_space.nvec[0]
        self.action_dim = self.num_joints * self.num_bins

        self.seed = random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.gamma = gamma          # Discount factor
        self.tau = tau              # For soft update of target parameters
        self.lr = lr                # Learning rate
        self.batch_size = batch_size      # Minibatch size
        self.update_every = update_every  # How often to update the network

        self.qnetwork_local = QNetwork(state_dim, self.action_dim)
        self.qnetwork_target = QNetwork(state_dim, self.action_dim)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.lr)  # Adam optimizer

        self.memory = ReplayBuffer(buffer_size)
        self.t_step = 0

        self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def select_action(self, state, eps=0.):
        """Selects an action for the given state using an epsilon-greedy policy."""
        state = torch.from_numpy(state).float().unsqueeze(
            0)  # Convert state to tensor
        self.qnetwork_local.eval()  # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(
                state)  # Get Q-values from the network
        self.qnetwork_local.train()  # Set network back to train mode

        if random.random() > eps:
            action_values_reshaped = action_values.view(
                self.num_joints, self.num_bins)
            action = np.argmax(
                action_values_reshaped.cpu().data.numpy(), axis=1)
        else:
            action = self.action_space.sample()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Adds an experience tuple to the replay buffer."""
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def learn(self, experiences):
        """Updates the Q-network weights using a batch of experiences."""
        #
        states, actions, rewards, next_states, dones = experiences

        # --- Calculate Target Q-values ---
        # Get max predicted Q-values (for next states) from target model
        Q_targets_next_flat = self.qnetwork_target(
            next_states).detach()  # Shape: [batch_size, action_dim]
        # Reshape to easily find max per joint
        Q_targets_next_reshaped = Q_targets_next_flat.view(
            self.batch_size, self.num_joints, self.num_bins)
        # Find the max Q-value *for each joint* in the next state  implicitly suggests per-joint max
        max_Q_targets_next_per_joint, _ = torch.max(
            Q_targets_next_reshaped, dim=2)  # Shape: [batch_size, num_joints]

        # We need a single Q-value target for the TD update.
        # Standard DQN uses the max over ALL possible actions in the next state.
        # Let's use the average max Q-value across joints as the target basis.
        # More sophisticated approaches exist (e.g., averaging loss per joint),
        # but average Q fits standard DQN update rule better.
        avg_max_Q_targets_next = torch.mean(
            max_Q_targets_next_per_joint, dim=1)  # Shape: [batch_size]

        # Compute Q targets for current states: R + gamma * max_Q(s', a')
        # Target = reward if done, else reward + gamma * avg_max_Q_next
        # Shape: [batch_size]
        Q_targets = rewards + \
            (self.gamma * avg_max_Q_targets_next * (1 - dones))

        # --- Calculate Expected Q-values ---
        # Get expected Q values from local model for the *actions actually taken*
        Q_expected_flat = self.qnetwork_local(
            states)  # Shape: [batch_size, action_dim]
        # Reshape to match action structure
        # Shape: [batch_size, num_joints, num_bins]
        Q_expected_reshaped = Q_expected_flat.view(
            self.batch_size, self.num_joints, self.num_bins)

        # Gather the Q-values corresponding to the specific bin chosen for each joint
        # actions has shape [batch_size, num_joints]. Need to add dim for gather.
        # Shape: [batch_size, num_joints, 1]
        actions_unsqueezed = actions.unsqueeze(-1)
        Q_expected_for_actions = Q_expected_reshaped.gather(
            # Shape: [batch_size, num_joints]
            2, actions_unsqueezed).squeeze(-1)

        # Average the Q-values across joints for the action taken
        avg_Q_expected = torch.mean(
            Q_expected_for_actions, dim=1)  # Shape: [batch_size]

        # --- Compute Loss ---
        # Calculate the Mean Squared Error loss between target and expected Q-values
        loss = F.mse_loss(avg_Q_expected, Q_targets)

        # --- Optimize the model ---
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()            # Calculate gradients
        self.optimizer.step()       # Update network weights

        # --- Update target network ---
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Softly updates target network weights: θ_target = τ*θ_local + (1 - τ)*θ_target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """Copies weights from local network to target network: θ_target = θ_local."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
