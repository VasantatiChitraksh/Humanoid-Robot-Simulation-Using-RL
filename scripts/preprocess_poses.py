from humanoid_lib.environment import HumanoidWalkEnv
from humanoid_lib.perception import get_pose_from_image
import os
import numpy as np
import sys

# Add project root to path to allow importing humanoid_lib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


IMAGE_DIR = "assets/pose_images"
OUTPUT_FILE = "assets/output/poses.npy"


def main():
    """
    Processes all images in IMAGE_DIR using Module 1 and saves
    the resulting pose vectors to OUTPUT_FILE.
    """
    print("--- Starting Pose Preprocessing ---")

    print("Initializing dummy environment to get joint list...")
    try:
        temp_env = HumanoidWalkEnv(render_mode=None)
        urdf_joints_list = temp_env.actuated_joint_names
        temp_env.close()
        print(
            f"Using {len(urdf_joints_list)} joints from URDF: {urdf_joints_list}")
    except Exception as e:
        print(f"Error initializing environment to get joint list: {e}")
        return

    try:
        image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print(f"Error: No images found in {IMAGE_DIR}")
            return
        print(f"Found {len(image_files)} images to process.")
    except FileNotFoundError:
        print(f"Error: Image directory not found at {IMAGE_DIR}")
        return

    all_poses = []
    valid_image_count = 0
    for i, image_path in enumerate(image_files):
        print(
            f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        try:
            pose_vector = get_pose_from_image(image_path, urdf_joints_list)

            if pose_vector is not None and pose_vector.size == len(urdf_joints_list):
                all_poses.append(pose_vector)
                valid_image_count += 1
                print("-> Success: Pose vector generated.")
            else:
                print("-> Failed: Could not generate valid pose vector.")

        except Exception as e:
            print(f"-> Error processing image: {e}")

    if not all_poses:
        print("\nError: No valid poses were generated from any image.")
        return

    pose_library = np.array(all_poses)
    print(f"\nSuccessfully processed {valid_image_count} images.")
    print(
        f"Saving pose library with shape {pose_library.shape} to {OUTPUT_FILE}")
    try:
        np.save(OUTPUT_FILE, pose_library)
        print("--- Pose Preprocessing Complete ---")
    except Exception as e:
        print(f"Error saving pose file: {e}")


if __name__ == "__main__":
    main()
