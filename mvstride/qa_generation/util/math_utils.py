# -*- coding: utf-8 -*-
import random
import numpy as np
import math
from typing import Dict, List, Tuple, Set, Union, Any
from scipy.spatial.transform import Rotation as R
from icecream import ic
from pathlib import Path
import json

def calculate_signed_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the signed rotation angle (in degrees) from vector v1 to vector v2.
    Assumes a standard Cartesian coordinate system with clockwise as positive
    (to match the North->East->South list order).
    Note: atan2 is normally counter-clockwise positive; here we adapt to the list order.
    """
    # Compute the angle of each vector (atan2 returns in [-pi, pi])
    # np.arctan2(y, x)
    ang1 = np.arctan2(v1[1], v1[0])
    ang2 = np.arctan2(v2[1], v2[0])

    # Compute the difference (ang2 - ang1)
    # In the image coordinate system (Y down), rotating from +X to +Y is clockwise; atan2 still applies
    # But we must ensure the direction matches the direction_name list order (usually clockwise: north->east->south)

    # Here we compute how many degrees v1 rotates to reach v2
    diff_rad = ang2 - ang1
    diff_deg = np.degrees(diff_rad)

    # Normalize to [0, 360)
    diff_deg = (diff_deg + 360) % 360
    return diff_deg

def get_camera_rotation_matrix(cam_data: Dict[str, Any]) -> np.ndarray:
    """
    Extract the camera-to-world matrix from cam_extrinsics.
    Supports both string and list formats.
    """
    extrinsics = cam_data.get("cam_extrinsics")

    # 1. Parse the extrinsics matrix M (M_world_to_camera)
    try:
        if isinstance(extrinsics, str):
            # If it is a string, strip the brackets and parse
            clean_str = extrinsics.replace('[', '').replace(']', '').strip()
            M = np.fromstring(clean_str, sep=' ', dtype=np.float32).reshape(4, 4)
        elif isinstance(extrinsics, (list, np.ndarray)):
            # If it is a list or numpy array, convert and reshape directly
            M = np.array(extrinsics, dtype=np.float32).reshape(4, 4)
        else:
            raise TypeError(f"不支持的外参类型: {type(extrinsics)}")

    except Exception as e:
        print(f"错误: 无法解析 cam_extrinsics。类型: {type(extrinsics)}, 错误: {e}")
        return np.eye(3)

    # 2. Extract the rotation part R_world_to_camera (top-left 3x3 of M)
    R_world_to_camera = M[:3, :3]

    # 3. Compute R_camera_to_world
    # Since a rotation matrix is orthogonal, its inverse equals its transpose
    R_camera_to_world = R_world_to_camera.T

    return R_camera_to_world

def get_world_to_camera_matrix(cam_data: Dict[str, Any]) -> np.ndarray:
    """
    Extract the camera-to-world matrix from cam_extrinsics.
    Supports both string and list formats.
    """
    extrinsics = cam_data.get("cam_extrinsics")

    # 1. Parse the extrinsics matrix M (M_world_to_camera)
    try:
        if isinstance(extrinsics, str):
            # If it is a string, strip the brackets and parse
            clean_str = extrinsics.replace('[', '').replace(']', '').strip()
            M = np.fromstring(clean_str, sep=' ', dtype=np.float32).reshape(4, 4)
        elif isinstance(extrinsics, (list, np.ndarray)):
            # If it is a list or numpy array, convert and reshape directly
            M = np.array(extrinsics, dtype=np.float32).reshape(4, 4)
        else:
            raise TypeError(f"不支持的外参类型: {type(extrinsics)}")

    except Exception as e:
        print(f"错误: 无法解析 cam_extrinsics。类型: {type(extrinsics)}, 错误: {e}")
        return np.eye(3)

    # 2. Extract the rotation part R_world_to_camera (top-left 3x3 of M)
    P_world_to_camera = M

    return P_world_to_camera

def calculate_angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the angle between two 3D vectors (in radians).
    """
    # Normalize the vectors
    norm_vec1 = vec1 / np.linalg.norm(vec1)
    norm_vec2 = vec2 / np.linalg.norm(vec2)

    # Compute the dot product (i.e., the cosine)
    dot_product = np.dot(norm_vec1, norm_vec2)

    # Clip the dot product to [-1, 1] to avoid floating-point error
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Return the angle (radians)
    return np.arccos(dot_product)


def calculate_rotation(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the relative rotation angle from vec1 to vec2 (range: -180 to 180 degrees).
    Convention: clockwise is positive, counter-clockwise is negative (based on +x=right, +y=front).

    Args:
        vec1: vector from object to cam1 (2D numpy array, [x, y])
        vec2: vector from object to cam2 (2D numpy array, [x, y])

    Returns:
        angle: relative rotation angle (-180.0 to 180.0, rounded to 1 decimal)
    """
    # Normalize the vectors (keep only direction)
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    # Step 1: compute the angle between the two vectors (0~180 degrees, via dot product)
    dot_product = np.dot(vec1_norm, vec2_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # handle numerical precision issues
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)  # base angle in 0~180 degrees

    # Step 2: use the cross product to determine direction (for the +x=right, +y=front frame)
    # 2D cross product formula: vec1.x * vec2.y - vec1.y * vec2.x
    cross_product = vec1_norm[0] * vec2_norm[1] - vec1_norm[1] * vec2_norm[0]

    # Step 3: set the sign from the direction (positive clockwise, negative counter-clockwise)
    if cross_product < 0:
        # cross<0 -> vec2 is to the right of vec1 -> clockwise (positive)
        return round(angle_deg, 1)
    elif cross_product > 0:
        # cross>0 -> vec2 is to the left of vec1 -> counter-clockwise (negative)
        return round(-angle_deg, 1)
    else:
        # cross=0 -> collinear (same direction 0 degrees, opposite 180 degrees)
        return 180.0 if angle_deg > 90 else 0.0


def vector_angle(v1, v2) -> float:
    """Compute the angle between two vectors (in degrees)."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot = np.dot(v1_norm, v2_norm)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def get_relative_direction(cam_loc, obj_loc, cam_for, thresh=0.01):
    # Define the threshold for mixed movement (consistent with rotation, but may need tuning for translation)
    TWO_DIR_RATIO = 0.1
    # 1. Negate the camera forward vector
    cam_for = - cam_for

    # 2. Compute the displacement vector
    T_vec = obj_loc - cam_loc

    # 3. Build camera 1's local coordinate frame
    # world Z axis is defined as the up vector
    Wz_vec = np.array([0, 0, 1])
    # Compute the right vector (R_vec)
    # R = F x Wz (right-hand rule, perpendicular to F and Wz)
    R_vec = np.cross(cam_for, Wz_vec)
    # Normalize the right axis (unless F and Wz are parallel, R_vec is non-zero)
    # If F_vec and Wz_vec are exactly parallel (camera looking straight up/down), R_vec has zero norm.
    # We assume the camera does not shoot exactly perpendicular to the ground.
    R_norm = np.linalg.norm(R_vec)
    if R_norm < 1e-6:
        # Fallback: if the camera points straight up/down, use the world X axis as the right axis
        R_vec = np.array([1, 0, 0])
    else:
        R_vec = R_vec / R_norm

    # 4. Project the displacement vector onto the local axes
    # Use the dot product to obtain the components
    D_forward = np.dot(T_vec, cam_for)
    D_right = np.dot(T_vec, R_vec)

    # 5. Convert into a textual description

    # Determine the forward/backward direction
    if D_forward > thresh:
        forward_dir = "forward"
    elif D_forward < -thresh:
        forward_dir = "backward"
    else:
        forward_dir = ""  # no significant movement, or purely left/right translation

    # Determine the left/right direction
    if D_right > thresh:
        right_dir = "right"
    elif D_right < -thresh:
        right_dir = "left"
    else:
        right_dir = ""  # no significant movement, or purely forward/backward translation

    if forward_dir and right_dir:
        # both directions exceed the absolute threshold (thresh)
        # Get the absolute values
        abs_D_forward = abs(D_forward)
        abs_D_right = abs(D_right)

        # Determine the dominant and secondary movement
        if abs_D_forward > abs_D_right:
            main_move = abs_D_forward
            sub_move = abs_D_right
            sub_dir_name = 'right'  # here right is the secondary direction
        else:
            main_move = abs_D_right
            sub_move = abs_D_forward
            sub_dir_name = 'forward'  # here forward is the secondary direction

        # Check whether the weaker direction's contribution is too small
        if sub_move < main_move * TWO_DIR_RATIO:
            # the weaker direction's contribution is too small; keep only the dominant one
            if sub_dir_name == 'right':
                right_dir = ""  # only forward is significant
            else:  # sub_dir_name == 'forward'
                forward_dir = ""  # only right is significant
        # otherwise, keep both directions and return a mixed movement

    return forward_dir, right_dir

def get_absolute_direction(cam_loc, obj_loc, cam_for, thresh=0.01):
    direction_name = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    # 1. Negate the camera forward vector
    cam_for = - cam_for

    # 2. Compute the displacement vector
    T_vec = obj_loc - cam_loc

    # Compute the angle
    angle = calculate_rotation(cam_for, T_vec)

    # Randomly choose the reference object's direction toward cam1
    main_direction = random.choice(direction_name)

    # Derive vec2's base direction (based on the angle offset)
    main_idx = direction_name.index(main_direction)  # use the raw direction to find the index (avoid errors after replacement)
    rel_idx = int(round(angle / 45)) % 8  # relative offset index (0~7)
    rel_direction = direction_name[(main_idx + rel_idx) % 8]

    return main_direction, rel_direction

def get_relative_orientation(ref_obj_loc, obj1_loc, obj2_loc):
    direction_name = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    direction_name_2 = ['front', 'right', 'back', 'left']
    # Compute the position vector from object to camera (core: relative position)
    vec_obj_to_cam1 = obj1_loc - ref_obj_loc  # vector from object to cam1
    vec_obj_to_cam2 = obj2_loc - ref_obj_loc  # vector from object to cam2

    # Compute the angle
    angle = calculate_rotation(vec_obj_to_cam1, vec_obj_to_cam2)

    # Randomly choose the reference object's direction toward cam1
    main_direction = random.choice(direction_name)

    # Derive vec2's base direction (based on the angle offset)
    main_idx = direction_name.index(main_direction)  # use the raw direction to find the index (avoid errors after replacement)
    rel_idx = int(round(angle / 45)) % 8  # relative offset index (0~7)
    rel_direction = direction_name[(main_idx + rel_idx) % 8]
    direct_8 = True
    # Randomly replace with front/back/left/right
    if rel_idx % 2 == 0 and random.random() < 0.5:
        main_direction = random.choice(direction_name_2)
        main_idx = direction_name_2.index(main_direction)  # use the raw direction to find the index (avoid errors after replacement)
        rel_idx = rel_idx // 2  # relative offset index (0~7)
        rel_direction = direction_name_2[(main_idx + rel_idx) % 4]
        direct_8 = False

    return main_direction, rel_direction, direct_8

def get_relative_orientation_vec(ref_vec, ref_dir, query_vec):
    direction_name_geo = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    direction_name_ego = ['front', 'front right', 'right', 'back right', 'back', 'back left', 'left', 'front left']
    # Compute the position vector from object to camera (core: relative position)

    # Compute the angle
    angle = calculate_rotation(ref_vec, query_vec)

    # Randomly choose the reference object's direction toward cam1
    if ref_dir in direction_name_ego:
        direction_name = direction_name_ego
        direct_geo = False
    else:
        direction_name = direction_name_geo
        direct_geo = True

    # Derive vec2's base direction (based on the angle offset)
    main_idx = direction_name.index(ref_dir)  # use the raw direction to find the index (avoid errors after replacement)
    rel_idx = int(round(angle / 45)) % 8  # relative offset index (0~7)
    query_direction = direction_name[(main_idx + rel_idx) % 8]

    return query_direction, direct_geo

def get_relative_rotation(R1: np.ndarray, R2: np.ndarray, degree_threshold: int, is_scannetpp = False) -> Tuple[str, str]:
    """
    Compute the relative rotation from R1 to R2 and determine the Yaw/Pitch
    directions (left/right, up/down).

    :param R1: camera 1 R_camera_to_world matrix (3x3).
    :param R2: camera 2 R_camera_to_world matrix (3x3).
    :return: (yaw_dir, pitch_dir) string tuple
    """

    # Define thresholds (adjust to your data)
    ROT_RAD_THRESHOLD = np.radians(degree_threshold)  # rotations above 5 degrees are considered significant
    TWO_DIR_RATIO = 0.2  # mixed-rotation threshold: the weaker direction's magnitude must exceed 50% of the dominant one

    # 1. Compute the relative rotation matrix R_rel = R2 * R1_T
    # R_rel is the rotation needed to go from cam1's pose to cam2's pose
    R_rel = R2 @ R1.T

    # 2. Convert to Euler angles (Yaw, Pitch, Roll)
    # Use 'zyx' order (equivalent to the reverse of Roll, Pitch, Yaw) to extract
    # Euler angles are usually in radians.
    r = R.from_matrix(R_rel)
    # [Yaw, Pitch, Roll] - note SciPy extracts YPR by default in ZYX order (Roll-Pitch-Yaw common)
    # We care about the Z-axis rotation (Yaw) and the Y-axis rotation (Pitch)

    # Use the 'ZYX' convention (Yaw-Pitch-Roll)
    # angles[0] is Yaw (around Z), angles[1] is Pitch (around Y), angles[2] is Roll (around X)
    # Note: this ZYX convention is in the world frame, but for relative rotations it approximates local rotation.

    # A better way is to use the 'XYZ' convention (Roll-Pitch-Yaw)
    # We directly use the 'YPR' convention:
    ypr_angles = r.as_euler('yxz', degrees=False)  # Yaw (Y), Pitch (X), Roll (Z)

    # Assuming your coordinate frame:
    # Yaw (Y axis) = ypr_angles[0]
    # Pitch (X axis) = ypr_angles[1]
    # Roll (Z axis) = ypr_angles[2]
    delta_yaw = ypr_angles[0]
    delta_pitch = ypr_angles[1]

    # 3. Determine the directions
    # Convention:
    # positive Yaw (counter-clockwise, viewed from above): view turns right
    # negative Yaw (clockwise, viewed from above): view turns left
    # positive Pitch (looking up): view tilts up
    # negative Pitch (looking down): view tilts down

    yaw_dir = ""
    pitch_dir = ""

    if delta_yaw > ROT_RAD_THRESHOLD:
        yaw_dir = "right" if is_scannetpp else "left"
    elif delta_yaw < -ROT_RAD_THRESHOLD:
        yaw_dir = "left" if is_scannetpp else "right"

    if delta_pitch > ROT_RAD_THRESHOLD:
        pitch_dir = "down"
    elif delta_pitch < -ROT_RAD_THRESHOLD:
        pitch_dir = "up"

    if yaw_dir and pitch_dir:
        # If both directions exceed ROT_RAD_THRESHOLD, we have a mixed rotation (case B)
        # TWO_DIR_RATIO prevents keeping a direction that barely exceeds the threshold while being far smaller than the other
        # e.g., Yaw=6 degrees, Pitch=1 degree (not mixed; only Yaw is significant)
        # Get the absolute values for comparison
        abs_yaw = abs(delta_yaw)
        abs_pitch = abs(delta_pitch)

        # Determine the dominant and secondary rotations
        if abs_yaw > abs_pitch:
            main_rot = abs_yaw
            sub_rot = abs_pitch
        else:
            main_rot = abs_pitch
            sub_rot = abs_yaw

        # Check whether the weaker direction is significant enough to be considered mixed
        if sub_rot < main_rot * TWO_DIR_RATIO:
            # the weaker direction's contribution is too small; keep only the dominant one
            if abs_yaw > abs_pitch:
                pitch_dir = ""  # only Yaw is significant
            else:
                yaw_dir = ""  # only Pitch is significant
        # otherwise, keep both directions and return a mixed rotation

    return yaw_dir, pitch_dir
