import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os

# --- Global MediaPipe Initialization ---
# This is more efficient than creating it in every function call
try:
    # **TODO**: Update this path to where you saved the model
    MODEL_PATH = 'assets/pose_landmarker_heavy.task'
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=5  # Detect up to 5 people
    )
    POSE_LANDMARKER = vision.PoseLandmarker.create_from_options(options)
    print("MediaPipe PoseLandmarker initialized successfully.")
except Exception as e:
    print(f"Error initializing MediaPipe PoseLandmarker: {e}")
    print("Please download 'pose_landmarker_heavy.task' from Google's MediaPipe page")
    print("and place it in your project's root directory (or update MODEL_PATH).")
    POSE_LANDMARKER = None

# -----------------------------------------------------------------
# Task 1.1: Image Acquisition and Preprocessing
# -----------------------------------------------------------------


def load_image(image_path: str) -> mp.Image:
    """
    Loads an image from a file path and converts it into 
    MediaPipe's Image format.
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert from BGR (OpenCV default) to RGB (MediaPipe requirement)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe's Image format
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# -----------------------------------------------------------------
# Task 1.2: Multi-Person Keypoint Extraction
# -----------------------------------------------------------------


def extract_skeletons(image: mp.Image) -> vision.PoseLandmarkerResult:
    """
    Processes the image using MediaPipe to detect multiple people
    and output their skeletons.
    """
    if POSE_LANDMARKER is None:
        raise ImportError("MediaPipe PoseLandmarker is not initialized.")

    # Run pose detection
    pose_landmarker_result = POSE_LANDMARKER.detect(image)
    return pose_landmarker_result

# -----------------------------------------------------------------
# Task 1.3: Target Selection and Kinematic Conversion
# -----------------------------------------------------------------


def select_target_skeleton(pose_result: vision.PoseLandmarkerResult) -> landmark_pb2.NormalizedLandmarkList:
    """
    Selects one skeleton using the 'largest bounding box area'
    heuristic.
    """
    if not pose_result.pose_landmarks:
        raise ValueError("No valid skeletons found in the image.")

    target_skeleton = None
    max_area = -1

    # pose_result.pose_landmarks is a list of detected skeletons
    for skeleton_landmarks in pose_result.pose_landmarks:
        # Get all x, y coordinates for the current skeleton
        x_coords = [lm.x for lm in skeleton_landmarks]
        y_coords = [lm.y for lm in skeleton_landmarks]

        # Find the min/max to define the bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Calculate area
        area = (x_max - x_min) * (y_max - y_min)

        if area > max_area:
            max_area = area
            target_skeleton = skeleton_landmarks

    if target_skeleton is None:
        # This case should be rare if pose_landmarks is not empty
        raise ValueError("No suitable skeleton found.")

    return target_skeleton


def convert_to_joint_angles(skeleton_landmarks: landmark_pb2.NormalizedLandmarkList,
                            urdf_joints_list: list) -> np.ndarray:
    """
    Converts the selected skeleton's Cartesian coordinates into
    initial joint angles (theta_init) using trigonometric
    functions like atan2.
    """

    # --- Helper Functions ---
    def get_coords(lm_index):
        """Get 2D (x, y) coordinates from the landmark list."""
        # We use 2D (x, y) for this project
        lm = skeleton_landmarks[lm_index]
        return np.array([lm.x, lm.y])

    def _calc_angle_2d_from_points(p1, p2, p3):
        """
        Calculates the 2D angle at p2 formed by p1-p2-p3.
        This implements the "vectors between adjacent joints"
        and "atan2" logic.
        """
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using atan2
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

        # Normalize angle to a useful range, e.g., 0 to 2pi
        if angle < 0:
            angle += 2 * np.pi

        return angle

    def calc_angle_2d(p1_idx, p2_idx, p3_idx):
        """Wrapper to calculate angle from landmark indices."""
        return _calc_angle_2d_from_points(
            get_coords(p1_idx),
            get_coords(p2_idx),
            get_coords(p3_idx)
        )

    # --- MediaPipe Pose Landmark Indices ---
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16

    # --- Angle Calculation ---
    # We calculate all possible angles and store them in a dictionary
    theta_init_dict = {}

    try:
        # --- NEW: Calculate midpoints for torso and neck ---
        p_mid_hip = (get_coords(L_HIP) + get_coords(R_HIP)) / 2
        p_mid_shoulder = (get_coords(L_SHOULDER) + get_coords(R_SHOULDER)) / 2
        p_nose = get_coords(NOSE)

        # --- NEW: Calculate 'chest' and 'neck' angles ---
        # 'chest': Angle of the torso (mid_hip -> mid_shoulder) relative to vertical
        # This approximates the torso's "lean"
        v_torso = p_mid_shoulder - p_mid_hip
        theta_init_dict['chest'] = np.arctan2(v_torso[0], v_torso[1])

        # 'neck': Angle at the shoulder line, formed by the hips and nose
        # This approximates the neck's "bend" relative to the torso
        theta_init_dict['neck'] = _calc_angle_2d_from_points(
            p_mid_hip, p_mid_shoulder, p_nose)

        # --- Leg Joints (FIXED: removed '_joint' from names) ---
        theta_init_dict['left_knee'] = calc_angle_2d(
            L_HIP, L_KNEE, L_ANKLE)
        theta_init_dict['right_knee'] = calc_angle_2d(
            R_HIP, R_KNEE, R_ANKLE)

        theta_init_dict['left_hip'] = calc_angle_2d(
            L_SHOULDER, L_HIP, L_KNEE)
        theta_init_dict['right_hip'] = calc_angle_2d(
            R_SHOULDER, R_HIP, R_KNEE)

        theta_init_dict['left_ankle'] = calc_angle_2d(
            L_KNEE, L_ANKLE, 29)  # 29 is L_HEEL
        theta_init_dict['right_ankle'] = calc_angle_2d(
            R_KNEE, R_ANKLE, 30)  # 30 is R_HEEL

        # --- Arm Joints (FIXED: removed '_joint' from names) ---
        theta_init_dict['left_elbow'] = calc_angle_2d(
            L_SHOULDER, L_ELBOW, L_WRIST)
        theta_init_dict['right_elbow'] = calc_angle_2d(
            R_SHOULDER, R_ELBOW, R_WRIST)

        theta_init_dict['left_shoulder'] = calc_angle_2d(
            L_HIP, L_SHOULDER, L_ELBOW)
        theta_init_dict['right_shoulder'] = calc_angle_2d(
            R_HIP, R_SHOULDER, R_ELBOW)

    except Exception as e:
        print(f"Error calculating angles (check landmark visibility?): {e}")
        return np.array([])  # Return empty array on failure

    # --- Final Step: Order the angles ---
    # Create the final vector in the *exact* order required by the URDF
    final_pose_vector = []
    for joint_name in urdf_joints_list:
        if joint_name not in theta_init_dict:
            print(
                f"Warning: Joint '{joint_name}' from URDF list was not calculated.")
            final_pose_vector.append(0.0)  # Default to 0 angle
        else:
            final_pose_vector.append(theta_init_dict[joint_name])

    # This is the final output of Module 1
    return np.array(final_pose_vector)

# -----------------------------------------------------------------
# --- Main Pipeline Function ---
# -----------------------------------------------------------------


def get_pose_from_image(image_path: str, urdf_joints_list: list) -> np.ndarray:
    """
    Runs the full Module 1 pipeline.

    1. Loads an image (Task 1.1)
    2. Extracts skeletons (Task 1.2)
    3. Selects a target and converts to joint angles (Task 1.3)

    Returns:
        np.ndarray: The Initial Pose Vector (theta_init).
    """
    if POSE_LANDMARKER is None:
        print("Module 1 cannot run. PoseLandmarker not initialized.")
        return

    try:
        # Task 1.1
        print(f"Loading image: {image_path}...")
        image = load_image(image_path)

        # Task 1.2
        print("Extracting skeletons...")
        pose_result = extract_skeletons(image)

        # Task 1.3 (Selection)
        print("Selecting target skeleton...")
        target_skeleton = select_target_skeleton(pose_result)

        # Task 1.3 (Conversion)
        print("Converting to joint angles...")
        theta_init = convert_to_joint_angles(target_skeleton, urdf_joints_list)

        print("\n--- Module 1 Pipeline Complete ---")
        return theta_init

    except Exception as e:
        print(f"\n--- Module 1 Pipeline FAILED ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


# -----------------------------------------------------------------
# --- TEST BLOCK ---
# -----------------------------------------------------------------
if __name__ == "__main__":
    """
    This block lets you run `python humanoid_lib/perception.py`
    to test the module independently.
    """

    # ---------------------------------------------------------------
    # This list is now correct and matches your URDF.
    # ---------------------------------------------------------------
    YOUR_URDF_JOINT_LIST = ['chest', 'neck', 'right_shoulder', 'right_elbow', 'left_shoulder',
                            'left_elbow', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']

    # **TODO**: Update this to an image on your computer
    # (You can create an 'assets/pose_images/' folder)
    TEST_IMAGE_PATH = 'assets/pose_images/test_pose.jpg'

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image not found at {TEST_IMAGE_PATH}")
        print("Please set TEST_IMAGE_PATH to a valid image file.")
    else:
        # Run the full pipeline
        theta_init_vector = get_pose_from_image(
            TEST_IMAGE_PATH, YOUR_URDF_JOINT_LIST)

        if theta_init_vector.size > 0:
            print("\n--- TEST SUCCESS! ---")
            print(f"Final Output Shape: {theta_init_vector.shape}")
            print(f"Expected Shape: ({len(YOUR_URDF_JOINT_LIST)},)")
            print("\nFinal Initial Pose Vector (theta_init):")
            print(theta_init_vector)
