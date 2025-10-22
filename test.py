import os
import time
import pybullet as p
from humanoid_lib.perception import get_pose_from_image
from humanoid_lib.environment import HumanoidWalkEnv

print("--- STARTING INTEGRATION TEST (M1 -> M2) ---")

# --- 1. Load Pose Library ---
IMAGE_DIR = "assets/pose_images"
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
# *** THIS IS THE FIX ***
# We now create the environment ONCE, outside the loop.
print("Initializing Environment (Module 2) in 'human' mode...")
try:
    env = HumanoidWalkEnv(render_mode='human')
    URDF_JOINT_LIST = env.actuated_joint_names
    print(f"Robot has {len(URDF_JOINT_LIST)} actuated joints.")
except Exception as e:
    print(f"Failed to initialize environment: {e}")
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

            # --- 3b. Run Module 2 ---
            # *** THIS IS THE FIX ***
            # We now just call reset() on the *existing* environment.
            print("M2: Calling env.reset() with the pose vector...")
            observation, info = env.reset(initial_pose=theta_init_vector)

            print("\n--- VISUAL CHECK ---")
            print("Does the robot's pose match the image?")
            print("Waiting 5 seconds... (CTRL+C in terminal for next image)")

            # Hold the pose for 5 seconds
            start_time = time.time()
            while time.time() - start_time < 5:
                # We need to call stepSimulation to make the GUI update
                # This is a good place to just let gravity run.
                p.stepSimulation()
                time.sleep(1./240.)  # PyBullet's standard tick rate

        except KeyboardInterrupt:
            print("\nUser skipped to next image.")
            continue  # Go to the next loop iteration
        except Exception as e:
            print(f"--- TEST FAILED for {image_path} ---")
            print(f"ERROR: {e}")

except KeyboardInterrupt:
    print("\n\nUser stopped the integration test.")
finally:
    # --- 4. Clean up ---
    # *** THIS IS THE FIX ***
    # We close the environment ONCE, after the loop is finished.
    if 'env' in locals():
        env.close()
        print("M2: Environment closed.")

print("\n--- INTEGRATION TEST COMPLETE ---")
