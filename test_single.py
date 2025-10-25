import os
import time
import argparse
import pybullet as p
import numpy as np  # Import numpy
from humanoid_lib.perception import get_pose_from_image
from humanoid_lib.environment import HumanoidWalkEnv


def main(image_path, use_zero_pose):
    """
    Runs the M1 -> M2 pipeline for a single image and visualizes the pose,
    with added debugging options.
    """
    print(f"--- Visualizing Pose for: {image_path} ---")

    # --- 1. Check if image exists (only if not using zero pose) ---
    if not use_zero_pose and not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # --- 2. Initialize Module 2 (Environment) ---
    print("Initializing Environment (Module 2) in 'human' mode...")
    env = None  # Initialize env to None for the finally block
    try:
        env = HumanoidWalkEnv(render_mode='human')
        URDF_JOINT_LIST = env.actuated_joint_names
        print(f"Robot has {len(URDF_JOINT_LIST)} actuated joints.")
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        if env:
            env.close()
        return

    # --- 3. Run Pipeline ---
    try:
        theta_init_vector = None  # Initialize

        # --- DEBUG: Option to use Zero Pose ---
        if use_zero_pose:
            print("!!! DEBUG: Testing with ZERO POSE vector...")
            theta_init_vector = np.zeros(len(URDF_JOINT_LIST))
            print(
                f"M1: Skipped. Using zero vector shape: {theta_init_vector.shape}")
        else:
            # --- 3a. Run Module 1 ---
            print("M1: Running Perception pipeline...")
            theta_init_vector = get_pose_from_image(
                image_path, URDF_JOINT_LIST)

            # --- DEBUG: Print and Check theta_init ---
            print("\n!!! DEBUG: Raw theta_init vector:")
            print(theta_init_vector)
            if theta_init_vector.size == 0:
                raise ValueError(
                    "Module 1 returned an empty vector (check file/pose).")
            if np.isnan(theta_init_vector).any() or np.isinf(theta_init_vector).any():
                print("!!! ERROR: Invalid values (NaN/Inf) found in theta_init!")
                raise ValueError(
                    "Invalid values (NaN/Inf) in theta_init vector!")
            # ----------------------------------------

            print(
                f"M1: Success. Generated pose vector with shape {theta_init_vector.shape}")

        # --- 3b. Run Module 2 ---
        print("M2: Calling env.reset() with the pose vector...")
        observation, info = env.reset(initial_pose=theta_init_vector)

        print("\n--- VISUAL CHECK ---")
        print("PyBullet window is open with the robot pose.")
        print("Close the PyBullet window or press CTRL+C in the terminal to exit.")

        # --- Keep simulation running for viewing ---
        while True:
            p.stepSimulation()
            time.sleep(1./240.)

    except KeyboardInterrupt:
        print("\nUser requested exit.")
    except Exception as e:
        print(f"\n--- TEST FAILED for {image_path} ---")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
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
