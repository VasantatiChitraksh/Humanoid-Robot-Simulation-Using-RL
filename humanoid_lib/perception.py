import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os

try:
    MODEL_PATH = 'assets/pose_landmarker_heavy.task'
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=5
    )
    POSE_LANDMARKER = vision.PoseLandmarker.create_from_options(options)
    print("MediaPipe PoseLandmarker initialized successfully.")
except Exception as e:
    print(f"Error initializing MediaPipe PoseLandmarker: {e}")
    print("Please download 'pose_landmarker_heavy.task' from Google's MediaPipe page")
    print("and place it in your project's root directory (or update MODEL_PATH).")
    POSE_LANDMARKER = None


def load_image(image_path: str) -> mp.Image:
    """Loads an image from a file path and converts it into MediaPipe's Image format."""
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)


def extract_skeletons(image: mp.Image) -> vision.PoseLandmarkerResult:
    """Processes the image using MediaPipe to detect multiple people and output their skeletons."""
    if POSE_LANDMARKER is None:
        raise ImportError("MediaPipe PoseLandmarker is not initialized.")

    pose_landmarker_result = POSE_LANDMARKER.detect(image)
    return pose_landmarker_result


def select_target_skeleton(pose_result: vision.PoseLandmarkerResult) -> landmark_pb2.NormalizedLandmarkList:
    """Selects one skeleton using the 'largest bounding box area' heuristic."""
    if not pose_result.pose_landmarks:
        raise ValueError("No valid skeletons found in the image.")

    target_skeleton = None
    max_area = -1

    for skeleton_landmarks in pose_result.pose_landmarks:
        x_coords = [lm.x for lm in skeleton_landmarks]
        y_coords = [lm.y for lm in skeleton_landmarks]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        area = (x_max - x_min) * (y_max - y_min)

        if area > max_area:
            max_area = area
            target_skeleton = skeleton_landmarks

    if target_skeleton is None:
        raise ValueError("No suitable skeleton found.")

    return target_skeleton


def convert_to_joint_angles(skeleton_landmarks: landmark_pb2.NormalizedLandmarkList,
                            urdf_joints_list: list) -> np.ndarray:
    """Converts the selected skeleton's Cartesian coordinates into initial joint angles (theta_init), adjusted for URDF zero pose."""

    def get_coords(lm_index):
        lm = skeleton_landmarks[lm_index]
        return np.array([lm.x, lm.y])

    def _calc_angle_2d_from_points(p1, p2, p3):
        """Calculates angle at p2 in range (-pi, pi]."""
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        while angle <= -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def calc_angle_2d(p1_idx, p2_idx, p3_idx):
        return _calc_angle_2d_from_points(
            get_coords(p1_idx),
            get_coords(p2_idx),
            get_coords(p3_idx)
        )

    NOSE = 0
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16

    theta_init_dict = {}

    try:
        raw_left_knee = calc_angle_2d(L_HIP, L_KNEE, L_ANKLE)
        raw_right_knee = calc_angle_2d(R_HIP, R_KNEE, R_ANKLE)
        raw_left_elbow = calc_angle_2d(L_SHOULDER, L_ELBOW, L_WRIST)
        raw_right_elbow = calc_angle_2d(R_SHOULDER, R_ELBOW, R_WRIST)

        raw_left_hip = calc_angle_2d(L_SHOULDER, L_HIP, L_KNEE)
        raw_right_hip = calc_angle_2d(R_SHOULDER, R_HIP, R_KNEE)
        raw_left_shoulder = calc_angle_2d(L_HIP, L_SHOULDER, L_ELBOW)
        raw_right_shoulder = calc_angle_2d(R_HIP, R_SHOULDER, R_ELBOW)

        raw_left_ankle = calc_angle_2d(L_KNEE, L_ANKLE, 29)
        raw_right_ankle = calc_angle_2d(R_KNEE, R_ANKLE, 30)

        p_mid_hip = (get_coords(L_HIP) + get_coords(R_HIP)) / 2
        p_mid_shoulder = (get_coords(L_SHOULDER) + get_coords(R_SHOULDER)) / 2
        p_nose = get_coords(NOSE)
        v_torso = p_mid_shoulder - p_mid_hip
        raw_chest = np.arctan2(v_torso[0], v_torso[1])
        raw_neck = _calc_angle_2d_from_points(
            p_mid_hip, p_mid_shoulder, p_nose)

        theta_init_dict['left_knee'] = raw_left_knee - \
            np.pi if raw_left_knee > 0 else raw_left_knee + np.pi
        theta_init_dict['right_knee'] = raw_right_knee - \
            np.pi if raw_right_knee > 0 else raw_right_knee + np.pi
        theta_init_dict['left_elbow'] = raw_left_elbow - \
            np.pi if raw_left_elbow > 0 else raw_left_elbow + np.pi
        theta_init_dict['right_elbow'] = raw_right_elbow - \
            np.pi if raw_right_elbow > 0 else raw_right_elbow + np.pi

        theta_init_dict['left_hip'] = raw_left_hip - np.pi/2
        theta_init_dict['right_hip'] = raw_right_hip - np.pi/2
        theta_init_dict['left_shoulder'] = raw_left_shoulder - np.pi/2
        theta_init_dict['right_shoulder'] = raw_right_shoulder - np.pi/2
        theta_init_dict['left_ankle'] = raw_left_ankle
        theta_init_dict['right_ankle'] = raw_right_ankle
        theta_init_dict['chest'] = raw_chest
        theta_init_dict['neck'] = raw_neck

    except Exception as e:
        print(f"Error calculating angles (check landmark visibility?): {e}")
        return np.array([])

    final_pose_vector = []
    for joint_name in urdf_joints_list:
        if joint_name not in theta_init_dict:
            print(
                f"Warning: Joint '{joint_name}' from URDF list was not calculated.")
            final_pose_vector.append(0.0)
        else:
            final_pose_vector.append(theta_init_dict[joint_name])

    return np.array(final_pose_vector)


def get_pose_from_image(image_path: str, urdf_joints_list: list) -> np.ndarray:
    """Runs the full pipeline from image path to angle vector."""
    if POSE_LANDMARKER is None:
        print("Module 1 cannot run. PoseLandmarker not initialized.")
        return

    try:
        print(f"Loading image: {image_path}...")
        image = load_image(image_path)

        print("Extracting skeletons...")
        pose_result = extract_skeletons(image)

        print("Selecting target skeleton...")
        target_skeleton = select_target_skeleton(pose_result)

        print("Converting to joint angles...")
        theta_init = convert_to_joint_angles(target_skeleton, urdf_joints_list)

        print("\n--- Pipeline Complete ---")
        return theta_init

    except Exception as e:
        print(f"\n--- Pipeline FAILED ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])
