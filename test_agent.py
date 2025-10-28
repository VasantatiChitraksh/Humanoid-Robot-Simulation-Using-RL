# Make sure DQNAgent class is importable
from humanoid_lib.agent import DQNAgent
from humanoid_lib.environment import HumanoidWalkEnv
import os
import sys
import torch
import numpy as np
import random
import time
import pybullet as p

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


# --- Configuration ---
POSE_LIBRARY_PATH = "assets/output/poses.npy"
MODEL_LOAD_PATH = "assets/output/dqn_humanoid_model.pth"  # Path to your saved model
SEED = 42  # Use a fixed seed for testing if desired
NUM_TEST_EPISODES = 10  # How many different starting poses to test
MAX_T_TEST = 1000      # Max steps per test episode

# Action Mapping Configuration (MUST match training script)
NUM_BINS = 5
TORQUE_LIMIT = 1.0
TORQUE_VALUES = np.linspace(-TORQUE_LIMIT, TORQUE_LIMIT, NUM_BINS)
# --------------------


def test_agent(env, agent, pose_library, n_episodes=NUM_TEST_EPISODES, max_t=MAX_T_TEST):
    """Loads the agent's weights and runs it in the environment."""

    print(f"\n--- Starting Agent Test ---")
    print(f"Loading model weights from: {MODEL_LOAD_PATH}")
    try:
        # Load the saved state dictionary
        agent.qnetwork_local.load_state_dict(torch.load(MODEL_LOAD_PATH))
        # Ensure the network is in evaluation mode (disables dropout, etc.)
        agent.qnetwork_local.eval()
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- Run Test Episodes ---
    for i_episode in range(1, n_episodes + 1):
        print(f"\n--- Episode {i_episode}/{n_episodes} ---")
        # --- Reset environment with a random pose ---
        initial_pose = random.choice(pose_library)
        try:
            state, _ = env.reset(initial_pose=initial_pose)
            if state.shape[0] != agent.state_dim:
                raise ValueError("State dimension mismatch on reset")
        except ValueError as e:
            print(f"Error resetting env: {e}. Skipping episode.")
            continue

        score = 0
        print("Running simulation...")
        for t in range(max_t):
            # --- Agent selects BEST action (no exploration) ---
            # Set eps=0.0 to ensure the agent uses the learned policy
            action_indices = agent.select_action(state, eps=0.0)

            # --- Map discrete action indices to continuous torques ---
            applied_torques = TORQUE_VALUES[action_indices]

            # --- Environment steps ---
            try:
                next_state, reward, terminated, truncated, _ = env.step(
                    applied_torques)
                done = terminated or truncated
            except p.error as e:  # Catch PyBullet errors if simulation becomes unstable
                print(f"\nPyBullet Error during step {t}: {e}")
                print("Ending episode early.")
                break

            state = next_state
            score += reward

            # --- Slow down rendering for visualization ---
            # Need a small sleep to see the movement in GUI mode
            # Match PyBullet's default timestep for smooth viewing
            time.sleep(1./240.)

            if done:
                print(f"Episode finished after {t+1} timesteps.")
                break
        else:  # This else block runs if the loop completes without a 'break'
            print(f"Episode reached max timesteps ({max_t}).")

        print(f"Episode Score: {score:.2f}")

        # Optional: Pause between episodes
        # input("Press Enter to start next episode...")

    print("\n--- Agent Test Complete ---")


def main():
    """Loads poses, initializes env/agent, and runs the test."""

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
        return

    # --- Initialize Environment ---
    print("Initializing environment in GUI mode ('human')...")
    env = None
    try:
        # MUST use render_mode='human' to see the visualization
        env = HumanoidWalkEnv(render_mode='human')
        state_dim = env.observation_space.shape[0]
        action_space = env.action_space
        print(f"State dimension: {state_dim}")
        print(f"Action space: {action_space}")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        if env:
            env.close()
        return

    # --- Initialize Agent ---
    # Use dummy hyperparameters for testing, only state/action dims matter
    print("Initializing DQN agent structure...")
    agent = DQNAgent(state_dim=state_dim, action_space=action_space, seed=SEED,
                     lr=0, gamma=0, tau=0, buffer_size=1, batch_size=1, update_every=1)

    # --- Run Test ---
    try:
        test_agent(env, agent, pose_library)
    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        # --- Clean up ---
        if env:
            env.close()
            print("Environment closed.")


if __name__ == "__main__":
    main()
