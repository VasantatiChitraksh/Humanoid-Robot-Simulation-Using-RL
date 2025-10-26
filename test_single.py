import os
import time
import argparse
import pybullet as p
import numpy as np  # Import numpy
from humanoid_lib.perception import get_pose_from_image
from humanoid_lib.environment import HumanoidWalkEnv


def main(image_path, use_zero_pose):
    """
    Runs the M1 -> M2 pipeline, takes one step with zero action,
    prints the results, and visualizes the pose.
    """
    print(f"--- Visualizing Pose for: {image_path} ---")

    # --- 1. Check if image exists ---
    if not use_zero_pose and not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # --- 2. Initialize Module 2 (Environment) ---
    print("Initializing Environment (Module 2) in 'human' mode...")
    env = None
    try:
        env = HumanoidWalkEnv(render_mode='human')
        URDF_JOINT_LIST = env.actuated_joint_names
        print(f"Robot has {len(URDF_JOINT_LIST)} actuated joints.")
        # Determine state and action dimensions for later
        state_dim = env.observation_space.shape[0]
        # Should match env.num_actuated_joints
        num_joints = len(URDF_JOINT_LIST)

    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        if env:
            env.close()
        return

    # --- 3. Run Pipeline & Step ---
    try:
        theta_init_vector = None

        # --- Get Initial Pose ---
        if use_zero_pose:
            print("!!! DEBUG: Testing with ZERO POSE vector...")
            theta_init_vector = np.zeros(len(URDF_JOINT_LIST))
            print(
                f"M1: Skipped. Using zero vector shape: {theta_init_vector.shape}")
        else:
            print("M1: Running Perception pipeline...")
            theta_init_vector = get_pose_from_image(
                image_path, URDF_JOINT_LIST)

            print("\n!!! DEBUG: Raw theta_init vector:")
            print(theta_init_vector)
            if theta_init_vector.size == 0:
                raise ValueError("Module 1 returned empty vector.")
            if np.isnan(theta_init_vector).any() or np.isinf(theta_init_vector).any():
                raise ValueError(
                    "Invalid values (NaN/Inf) in theta_init vector!")

            print(f"M1: Success. Shape: {theta_init_vector.shape}")

        # --- Reset Environment ---
        print("M2: Calling env.reset() with the pose vector...")
        observation, info = env.reset(initial_pose=theta_init_vector)
        print(
            f"M2: Reset complete. Initial observation shape: {observation.shape}")

        # --- Take One Step with Zero Action ---
        print("\n--- Taking one step with ZERO action ---")
        # Create a zero torque action vector matching the number of joints
        zero_action_torques = np.zeros(num_joints)

        # Call the step function
        next_observation, reward, terminated, truncated, step_info = env.step(
            zero_action_torques)

        # --- Print Step Results ---
        print("\n--- Step Results ---")
        print(f"Next Observation Shape: {next_observation.shape}")
        # print(f"Next Observation (first 10 vals): {next_observation[:10]}") # Optionally print part of the state
        print(f"Reward: {reward:.4f}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {step_info}")
        # ------------------------

        print("\n--- VISUAL CHECK ---")
        print("PyBullet window shows state AFTER one step.")
        print("Close the PyBullet window or press CTRL+C in the terminal to exit.")

        # --- Keep simulation running for viewing ---
        while True:
            # Continue stepping simulation ONLY FOR VISUALIZATION
            # Do NOT call env.step() again here
            p.stepSimulation()
            time.sleep(1./240.)

    except KeyboardInterrupt:
        print("\nUser requested exit.")
    except Exception as e:
        print(f"\n--- TEST FAILED for {image_path} ---")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 4. Clean up ---
        if env:
            env.close()
            print("M2: Environment closed.")


if __name__ == "__main__":
    # --- Set up argument parser ---
    parser = argparse.ArgumentParser(
        description="Visualize humanoid pose from an image.")
    parser.add_argument("image_path", type=str,
                        help="Path to the input image file.")
    # --- DEBUG: Add --zero_pose flag ---
    parser.add_argument("--zero_pose", action="store_true",
                        help="Skip perception and use a zero pose vector.")

    # --- Parse arguments ---
    args = parser.parse_args()

    # --- Run the main function ---
    main(args.image_path, args.zero_pose)

    print("\n--- Visualization Complete ---")
