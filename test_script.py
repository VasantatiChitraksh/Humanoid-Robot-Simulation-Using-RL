import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from humanoid_lib.perception import get_pose_from_image, load_image, extract_skeletons, select_target_skeleton

POSE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 13),  # Left shoulder to elbow
    (13, 15),  # Left elbow to wrist
    (12, 14),  # Right shoulder to elbow
    (14, 16),  # Right elbow to wrist
    (11, 23),  # Left shoulder to hip
    (12, 24),  # Right shoulder to hip
    (23, 24),  # Hips
    (23, 25),  # Left hip to knee
    (25, 27),  # Left knee to ankle
    (27, 29),  # Left ankle to heel
    (27, 31),  # Left ankle to foot index
    (29, 31),  # Left heel to foot index
    (24, 26),  # Right hip to knee
    (26, 28),  # Right knee to ankle
    (28, 30),  # Right ankle to heel
    (28, 32),  # Right ankle to foot index
    (30, 32)   # Right heel to foot index
]


def visualize_pose(image_path: str, pose_result: mp.tasks.vision.PoseLandmarkerResult):
    """
    Draws the detected skeleton on the image using CV2 and displays it.
    """
    print("\nDisplaying pose estimation visual...")
    try:
        # Load the image for drawing
        image_cv = cv2.imread(image_path)
        # We'll draw on the BGR image and convert to RGB later
        image_for_drawing = np.copy(image_cv)

        # Select the main skeleton
        target_skeleton = select_target_skeleton(pose_result)

        # Convert normalized landmarks to pixel coordinates
        h, w, _ = image_for_drawing.shape

        # Store pixel coordinates in a simple list
        pixel_landmarks = []
        for lm in target_skeleton:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pixel_landmarks.append((px, py))

        # --- Draw with CV2 ---
        print("Len of pixel_landmarks:", len(pixel_landmarks))

        # 1. Draw the connections (lines)
        for connection in POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            # Get the (x, y) coordinates
            p1 = pixel_landmarks[start_idx]
            p2 = pixel_landmarks[end_idx]

            cv2.line(image_for_drawing, p1, p2, (0, 255, 0), 2)  # Green lines

        # 2. Draw the landmarks (circles)
        for px, py in pixel_landmarks:
            cv2.circle(image_for_drawing, (px, py), 3,
                       (255, 0, 0), -1)  # Red circles

        # ---------------------

        # Convert from BGR (OpenCV) to RGB (Matplotlib)
        image_rgb = cv2.cvtColor(image_for_drawing, cv2.COLOR_BGR2RGB)

        # Display with Matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title(f"Pose Estimation for {image_path}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Could not visualize pose: {e}")


# -----------------------------------------------------------------
# --- BATCH TEST SCRIPT FOR MULTIPLE IMAGES ---
# -----------------------------------------------------------------
if __name__ == "__main__":

    # --- Configuration ---
    URDF_JOINT_LIST = ['chest', 'neck', 'right_shoulder', 'right_elbow', 'left_shoulder',
                       'left_elbow', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']

    IMAGE_FOLDER = "assets/test_images"  # Folder containing test images

    if not os.path.isdir(IMAGE_FOLDER):
        print(f"❌ Folder not found: {IMAGE_FOLDER}")
        exit(1)

    # Get all supported image files in the folder
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"⚠️ No image files found in {IMAGE_FOLDER}")
        exit(1)

    print(f"\n✅ Found {len(image_files)} image(s) in {IMAGE_FOLDER}")
    print("Starting batch processing...\n")

    for idx, image_name in enumerate(sorted(image_files), start=1):
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(image_files)}] Processing: {image_name}")
        print(f"{'='*60}")

        try:
            # --- 1. Run the full pipeline to get the vector output ---
            theta_init_vector = get_pose_from_image(
                image_path, URDF_JOINT_LIST)

            # --- 2. Print the init vector ---
            if theta_init_vector.size > 0:
                print("\n--- TEST SUCCESS! ---")
                print(f"Final Output Shape: {theta_init_vector.shape}")
                print(f"Expected Shape: ({len(URDF_JOINT_LIST)},)")
                print("\nFinal Initial Pose Vector (theta_init):")
                print(theta_init_vector)

                # --- 3. Run visualization ---
                try:
                    mp_image = load_image(image_path)
                    pose_result = extract_skeletons(mp_image)
                    visualize_pose(image_path, pose_result)
                except Exception as e:
                    print(f"Visualization error: {e}")
            else:
                print("--- TEST FAILED ---")
                print("Could not generate pose vector.")

        except KeyboardInterrupt:
            print("\n🛑 Process interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Error while processing {image_name}: {e}")

        # Wait for user input to continue to next image
        try:
            input("\nPress Enter to continue to the next image (Ctrl+C to exit)...")
        except KeyboardInterrupt:
            print("\n🛑 Process stopped by user.")
            break

    print("\n✅ Batch processing complete.")
