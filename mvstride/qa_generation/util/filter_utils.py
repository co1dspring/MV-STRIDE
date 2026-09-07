# -*- coding: utf-8 -*-
import random
import numpy as np
import math
from typing import Dict, List, Tuple, Set, Union, Any
from scipy.spatial.transform import Rotation as R
from icecream import ic
from pathlib import Path
import json
from .math_utils import calculate_angle_between_vectors, calculate_rotation

def is_object_center_in_room(
        object_id: str,
        room_id: str,
        objects: Dict[str, Dict[str, Any]],
        rooms: Dict[str, Dict[str, Any]]
) -> bool:
    """
    Check whether an object's center point (3d_center) falls within a room's AABB.

    Args:
        object_id: ID key of the object to check.
        room_id: ID key of the room to check.
        objects: dict containing all object metadata.
        rooms: dict containing all room metadata.

    Returns:
        True if the object's center is inside the room AABB, otherwise False.
    """

    # 1. Extract the center point and bounding box data
    try:
        # Extract the object's 3d center point [x, y, z]
        center_loc = objects[object_id]['3d_center']
        room_bbox = rooms[room_id]['bbox_3d_aabb']
    except KeyError as e:
        print(f"错误: 无法找到物体 '{object_id}' 的中心点或房间 '{room_id}' 的数据。缺少字段: {e}")
        return False
    except TypeError:
        print("错误: 物体或房间的元数据结构不正确。")
        return False

    # 2. Extract the room's AABB bounds
    room_min_x = room_bbox['min']['x']
    room_max_x = room_bbox['max']['x']
    room_min_y = room_bbox['min']['y']
    room_max_y = room_bbox['max']['y']
    room_min_z = room_bbox['min']['z']
    room_max_z = room_bbox['max']['z']

    # 3. Extract the center point coordinates
    center_x = center_loc[0]
    center_y = center_loc[1]
    center_z = center_loc[2]

    # 4. Perform the point-in-AABB test

    # Check X axis: whether the center x is within the room's min/max x
    is_in_x = (room_min_x <= center_x) and (center_x <= room_max_x)

    # Check Y axis
    is_in_y = (room_min_y <= center_y) and (center_y <= room_max_y)

    # Check Z axis
    is_in_z = (room_min_z <= center_z) and (center_z <= room_max_z)

    # 5. Final result
    return is_in_x and is_in_y and is_in_z

def is_in_direction(
        ref_obj_A_loc: Union[np.ndarray, List[float]],
        ref_obj_B_loc: Union[np.ndarray, List[float]],
        query_obj_loc: Union[np.ndarray, List[float]],
        dir_B_to_A: str,
        dir_query: str
) -> bool:
    """
    Based on the calculate_rotation logic:
    assuming ref_obj_B is in the dir_B_to_A direction from ref_obj_A,
    decide whether query_obj is in the dir_query direction from ref_obj_A.
    """

    # 1. Define the direction list (must match the reference code)
    # Reference logic: main_idx + rel_idx, and calculate_rotation is positive clockwise
    # This means the list must be in clockwise order
    direction_names = [
        'north', 'northeast', 'east', 'southeast',
        'south', 'southwest', 'west', 'northwest'
    ]

    # Validate the directions
    if dir_B_to_A not in direction_names or dir_query not in direction_names:
        return False

    # 2. Convert to numpy arrays
    loc_A = np.array(ref_obj_A_loc[:2])
    loc_B = np.array(ref_obj_B_loc[:2])
    loc_Q = np.array(query_obj_loc[:2])

    # 3. Compute the vectors
    # vec1: base vector (A -> B), corresponding to vec_obj_to_cam1 in the reference code
    vec_A_to_B = loc_B - loc_A
    # vec2: query vector (A -> Q), corresponding to vec_obj_to_cam2 in the reference code
    vec_A_to_Q = loc_Q - loc_A

    # Ignore the case where the distance is too small
    if np.linalg.norm(vec_A_to_B) < 1e-3 or np.linalg.norm(vec_A_to_Q) < 1e-3:
        return False

    # 4. Compute the rotation angle (fully reusing the reference logic)
    # angle is the rotation from vec_A_to_B to vec_A_to_Q
    # positive means clockwise, negative means counter-clockwise
    angle = calculate_rotation(vec_A_to_B, vec_A_to_Q)

    # 5. Compute the index offset
    # Reference: rel_idx = int(round(angle / 45)) % 8
    rel_idx = int(round(angle / 45.0))

    # 6. Derive the predicted direction
    # Get the base index of B
    base_idx = direction_names.index(dir_B_to_A)

    # Compute Q's expected index (note the modulo for looping)
    # Because the list is clockwise and calculate_rotation is positive clockwise, add directly
    target_idx = (base_idx + rel_idx) % 8

    predicted_dir = direction_names[target_idx]

    # 7. Compare
    return predicted_dir == dir_query

def is_object_too_small(bbox_2d: dict, min_area: float, min_side: float) -> bool:
    """
    Check whether a single object's 2D bounding box is too small.

    Args:
        bbox_2d: 2D bounding box dict of the object,
                 containing 'max_x', 'min_x', 'max_y', 'min_y'.
        min_area: minimum allowed area threshold.
        min_side: minimum allowed side-length threshold.

    Returns:
        True if the object is too small, otherwise False.
    """

    # Note: it is assumed the caller already ensured bbox_2d is not empty.
    # If the caller cannot guarantee it, check before calling or inside the function.
    if not bbox_2d:
        # If there is no bbox data, treat as invalid (too small or invalid), return True
        return True

    # 1. Compute the area
    width = bbox_2d['max_x'] - bbox_2d['min_x']
    height = bbox_2d['max_y'] - bbox_2d['min_y']
    area = width * height

    # 2. Check whether the area and side length are too small
    if area < min_area or min(width, height) < min_side:
        # object too small
        return True

    # object passes the check
    return False

def should_filter_camera_pair(cam1_loc: np.ndarray, cam1_for: np.ndarray,
                              cam2_loc: np.ndarray, cam2_for: np.ndarray) -> bool:
    # --- Define thresholds (adjust as needed) ---
    # 1. Distance threshold: if camera separation is below this, positions are considered highly overlapping
    DISTANCE_THRESHOLD = 0.2  # e.g., 0.5 meters

    # 2. Angle threshold: if the camera orientation angle is below this, orientations are considered highly overlapping
    # in radians, e.g., 5 degrees
    ANGLE_THRESHOLD_DEG = 5.0
    ANGLE_THRESHOLD_RAD = np.radians(ANGLE_THRESHOLD_DEG)

    """
    Decide whether a pair of cameras should be filtered because both position
    and orientation overlap are too high.

    Args:
        cam1_loc, cam1_for: camera 1 position and forward vector.
        cam2_loc, cam2_for: camera 2 position and forward vector.

    Returns:
        bool: True (should filter) if both position and orientation overlap are high.
    """

    # --- 1. Position overlap check (distance) ---

    # Compute the Euclidean distance between the two camera positions
    distance = np.linalg.norm(cam1_loc - cam2_loc)

    # If the distance is very small, the position overlap is high
    position_overlap_high = (distance < DISTANCE_THRESHOLD)

    # --- 2. Orientation overlap check (angle) ---

    # Compute the angle (radians) between the two forward vectors
    angle_rad = calculate_angle_between_vectors(cam1_for, cam2_for)

    # If the angle is very small, the orientation overlap is high
    orientation_overlap_high = (angle_rad < ANGLE_THRESHOLD_RAD)

    # --- 3. Combined judgment ---

    # Only filter when BOTH position overlap and orientation overlap are high
    return position_overlap_high and orientation_overlap_high

def should_filter_camera_pair_strong(cam1_loc: np.ndarray, cam1_for: np.ndarray,
                                     cam2_loc: np.ndarray, cam2_for: np.ndarray) -> bool:
    # --- Threshold settings ---
    MIN_DISTANCE = 0.2  # distance too small (no sense of movement)
    MAX_DISTANCE = 5.0  # distance too large (may not be in the same room or loses association)

    MIN_ANGLE_DEG = 5.0  # angle too small (almost coincident)
    MAX_ANGLE_DEG = 150.0  # angle too large (fields of view may not overlap at all)

    # --- 1. Position filtering ---
    distance = np.linalg.norm(cam1_loc - cam2_loc)
    # Filter if the distance is too small or too large
    if distance < MIN_DISTANCE or distance > MAX_DISTANCE:
        return True

    # --- 2. Orientation filtering ---
    # Compute the angle (radians)
    angle_rad = calculate_angle_between_vectors(cam1_for, cam2_for)
    angle_deg = np.degrees(angle_rad)

    # Filter if the angle is too small (parallel) or too large (excessive turn)
    if angle_deg < MIN_ANGLE_DEG or angle_deg > MAX_ANGLE_DEG:
        return True

    # --- 3. Combined judgment ---
    # Only camera pairs within the defined [MIN, MAX] range return False (keep)
    return False
