from humanoid_lib.agent import DQNAgent
from humanoid_lib.environment import HumanoidWalkEnv
import os
import sys
import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


POSE_LIBRARY_PATH = "assets/output/poses.npy"
MODEL_SAVE_PATH = "assets/output/dqn_humanoid_model.pth"
SEED = 0

# --- Hyperparameters ---
N_EPISODES = 10       # Total number of training episodes
MAX_T = 1000            # Max number of timesteps per episode
EPS_START = 1.0         # Starting value of epsilon, for exploration
EPS_END = 0.01          # Minimum value of epsilon
# Multiplicative factor (per episode) for decreasing epsilon
EPS_DECAY = 0.995
GAMMA = 0.99            # Discount factor
LR = 5e-4               # Learning rate
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size
UPDATE_EVERY = 4        # How often to update the network
TAU = 1e-3              # For soft update of target parameters

# Action Mapping Configuration
NUM_BINS = 5            # Must match env.action_space.nvec[0]
# Max absolute torque value (adjust based on URDF limits if known)
TORQUE_LIMIT = 1.0
# Create the torque values for each bin
TORQUE_VALUES = np.linspace(-TORQUE_LIMIT, TORQUE_LIMIT, NUM_BINS)


def train_dqn(env, agent, pose_library, n_episodes=N_EPISODES, max_t=MAX_T,
              eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY):
    """Deep Q-Learning training loop."""
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    start_time = time.time()

    print("\n--- Starting Training ---")
    print(f"Using {len(pose_library)} initial poses.")

    for i_episode in range(1, n_episodes+1):
        # --- Reset environment with a random pose ---
        initial_pose = random.choice(pose_library)
        try:
            state, _ = env.reset(initial_pose=initial_pose)
            # Verify state shape
            if state.shape[0] != agent.state_dim:
                print(f"\n!!! ERROR: State dimension mismatch !!!")
                print(
                    f"Env state shape: {state.shape}, Agent expected: ({agent.state_dim},)")
                # Attempt to reshape or pad if possible, otherwise raise error
                # This can happen if _get_observation calculation is wrong
                raise ValueError("State dimension mismatch")
        except ValueError as e:
            print(
                f"Error resetting env for episode {i_episode} with pose {initial_pose[:4]}...: {e}")
            print("Skipping episode...")
            continue  # Skip to next episode if reset fails

        score = 0
        for t in range(max_t):
            # --- Agent selects action ---
            action_indices = agent.select_action(
                state, eps)  # Get bin indices [num_joints]

            # --- Map discrete action indices to continuous torques ---
            applied_torques = TORQUE_VALUES[action_indices]

            # --- Environment steps ---
            next_state, reward, terminated, truncated, _ = env.step(
                applied_torques)
            done = terminated or truncated  # Gymnasium uses terminated and truncated

            # --- Agent stores experience and learns ---
            agent.store_transition(state, action_indices,
                                   reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon

        # --- Print Progress ---
        if i_episode % 10 == 0:
            avg_score = np.mean(scores_window)
            elapsed_time = time.time() - start_time
            print(
                f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {eps:.4f} \tTime: {elapsed_time:.1f}s')

        if i_episode % 100 == 0:
            avg_score = np.mean(scores_window)
            print(
                f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {eps:.4f} \tTime: {elapsed_time:.1f}s')
            # Save checkpoint
            # torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_{i_episode}.pth')

        # --- Check for solving (optional criteria) ---
        # if np.mean(scores_window)>=200.0:
        #     print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
        #     break

    print("\n--- Training Complete ---")
    return scores


def main():
    """Main function to load poses, initialize env/agent, and train."""

    # --- Load Preprocessed Poses ---
    print(f"Loading pose library from: {POSE_LIBRARY_PATH}")
    try:
        pose_library = np.load(POSE_LIBRARY_PATH)
        if pose_library.ndim != 2 or pose_library.shape[0] == 0:
            raise ValueError(
                "Pose library is empty or has incorrect dimensions.")
        print(f"Loaded {pose_library.shape[0]} poses.")
    except Exception as e:
        print(f"Error loading pose library: {e}")
        print("Please run scripts/preprocess_poses.py first.")
        return

    # --- Initialize Environment ---
    print("Initializing environment in DIRECT mode (no GUI)...")
    try:
        # Use DIRECT mode for faster training
        env = HumanoidWalkEnv(render_mode=None)
        state_dim = env.observation_space.shape[0]
        action_space = env.action_space  # Pass the gym space object
        print(f"State dimension: {state_dim}")
        print(f"Action space: {action_space}")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        if 'env' in locals():
            env.close()
        return

    # --- Initialize Agent ---
    print("Initializing DQN agent...")
    agent = DQNAgent(state_dim=state_dim, action_space=action_space, seed=SEED,
                     lr=LR, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE,
                     batch_size=BATCH_SIZE, update_every=UPDATE_EVERY)

    # --- Train ---
    scores = train_dqn(env, agent, pose_library)

    # --- Save Model ---
    print(f"\nSaving trained model weights to: {MODEL_SAVE_PATH}")
    try:
        torch.save(agent.qnetwork_local.state_dict(), MODEL_SAVE_PATH)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    # --- Plot Scores ---
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.title('DQN Training Scores')
        plt.show()
    except Exception as e:
        print(f"Could not plot scores: {e}")

    # --- Clean up ---
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
