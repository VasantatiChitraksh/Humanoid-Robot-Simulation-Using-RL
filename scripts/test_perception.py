import os
import time
import pybullet as p
import numpy as np  # Import numpy
from humanoid_lib.perception import get_pose_from_image
from humanoid_lib.environment import HumanoidWalkEnv

print("--- STARTING INTEGRATION TEST (M1 -> M2 + Single Step) ---")

# --- 1. Load Pose Library ---
IMAGE_DIR = "assets/pose_images/"
try:
    pose_images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(
        IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not pose_images:
        print(f"No images found in {IMAGE_DIR}. Please add test images.")
        exit()
    print(f"Found {len(pose_images)} images to test.")
except FileNotFoundError:
    print(f"Directory not found: {IMAGE_DIR}")
    exit()

# --- 2. Initialize Module 2 (Environment) ---
print("Initializing Environment (Module 2) in 'human' mode...")
env = None
try:
    env = HumanoidWalkEnv(render_mode='human')
    URDF_JOINT_LIST = env.actuated_joint_names
    num_joints = len(URDF_JOINT_LIST)  # Get number of joints
    print(f"Robot has {num_joints} actuated joints.")
except Exception as e:
    print(f"Failed to initialize environment: {e}")
    if env:
        env.close()  # Ensure cleanup if init fails partially
    exit()

# --- 3. Run Stress Test Loop ---
try:
    for i, image_path in enumerate(pose_images):
        print(f"\n--- TEST {i+1}/{len(pose_images)}: {image_path} ---")

        try:
            # --- 3a. Run Module 1 ---
            print("M1: Running Perception pipeline...")
            theta_init_vector = get_pose_from_image(
                image_path, URDF_JOINT_LIST)

            if theta_init_vector.size == 0:
                raise ValueError(
                    "Module 1 returned an empty vector (check file/pose).")

            print(
                f"M1: Success. Generated pose vector with shape {theta_init_vector.shape}")

            # --- 3b. Run Module 2 Reset ---
            print("M2: Calling env.reset() with the pose vector...")
            observation, info = env.reset(initial_pose=theta_init_vector)
            print(
                f"M2: Reset complete. Initial observation shape: {observation.shape}")

            # --- NEW: Take One Step with Zero Action ---
            print("\n--- Taking one step with ZERO action ---")
            zero_action_torques = np.zeros(num_joints)
            next_observation, reward, terminated, truncated, step_info = env.step(
                zero_action_torques)

            # --- Print Step Results ---
            print("\n--- Step Results ---")
            print(f"Next Observation Shape: {next_observation.shape}")
            print(f"Reward: {reward:.4f}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {step_info}")
            # ------------------------

            print("\n--- VISUAL CHECK ---")
            print("PyBullet window shows state AFTER one step.")
            print("Waiting 5 seconds... (CTRL+C in terminal for next image)")

            # Hold the pose for 5 seconds (showing the state *after* the step)
            start_time = time.time()
            while time.time() - start_time < 5:
                # Need p.stepSimulation ONLY for visualization updates if env.step isn't called
                # In this case, env.step was already called, so just sleep might be okay,
                # but stepping ensures physics visualization keeps running.
                p.stepSimulation()
                time.sleep(1./240.)  # PyBullet's standard tick rate

        except KeyboardInterrupt:
            print("\nUser skipped to next image.")
            continue  # Go to the next loop iteration
        except Exception as e:
            print(f"--- TEST FAILED for {image_path} ---")
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging

except KeyboardInterrupt:
    print("\n\nUser stopped the integration test.")
finally:
    # --- 4. Clean up ---
    if env:
        env.close()
        print("M2: Environment closed.")

print("\n--- INTEGRATION TEST COMPLETE ---")
