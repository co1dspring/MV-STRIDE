# -*- coding: gbk -*-
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
    筛选物体的中心点（3d_center）是否落在指定房间的 AABB 范围内。

    参数:
        object_id: 要检查的物体的 ID 键。
        room_id: 要检查的房间的 ID 键。
        objects: 包含所有物体元数据的字典。
        rooms: 包含所有房间元数据的字典。

    返回:
        如果物体的中心点在房间 AABB 内，返回 True；否则返回 False。
    """

    # 1. 提取中心点和边界框数据
    try:
        # 提取物体的 3d 中心点 [x, y, z]
        center_loc = objects[object_id]['3d_center']
        room_bbox = rooms[room_id]['bbox_3d_aabb']
    except KeyError as e:
        print(f"错误: 无法找到物体 '{object_id}' 的中心点或房间 '{room_id}' 的数据。缺少字段: {e}")
        return False
    except TypeError:
        print("错误: 物体或房间的元数据结构不正确。")
        return False

    # 2. 提取房间的 AABB 边界
    room_min_x = room_bbox['min']['x']
    room_max_x = room_bbox['max']['x']
    room_min_y = room_bbox['min']['y']
    room_max_y = room_bbox['max']['y']
    room_min_z = room_bbox['min']['z']
    room_max_z = room_bbox['max']['z']

    # 3. 提取中心点坐标
    center_x = center_loc[0]
    center_y = center_loc[1]
    center_z = center_loc[2]

    # 4. 执行点在包围盒内检测 (Point-in-AABB Test)

    # 检查 X 轴：中心点是否大于等于房间最小 X 且小于等于房间最大 X
    is_in_x = (room_min_x <= center_x) and (center_x <= room_max_x)

    # 检查 Y 轴
    is_in_y = (room_min_y <= center_y) and (center_y <= room_max_y)

    # 检查 Z 轴
    is_in_z = (room_min_z <= center_z) and (center_z <= room_max_z)

    # 5. 最终结果
    return is_in_x and is_in_y and is_in_z

def is_in_direction(
        ref_obj_A_loc: Union[np.ndarray, List[float]],
        ref_obj_B_loc: Union[np.ndarray, List[float]],
        query_obj_loc: Union[np.ndarray, List[float]],
        dir_B_to_A: str,
        dir_query: str
) -> bool:
    """
    基于 calculate_rotation 的逻辑判断：
    假设 ref_obj_B 在 ref_obj_A 的 dir_B_to_A 方向上，
    判断 query_obj 是否在 ref_obj_A 的 dir_query 方向上。
    """

    # 1. 定义方向列表 (必须与参考代码一致)
    # 参考代码逻辑：main_idx + rel_idx，且 calculate_rotation 顺时针为正
    # 这意味着列表必须是【顺时针排列】的
    direction_names = [
        'north', 'northeast', 'east', 'southeast',
        'south', 'southwest', 'west', 'northwest'
    ]

    # 校验方向合法性
    if dir_B_to_A not in direction_names or dir_query not in direction_names:
        return False

    # 2. 转换为 numpy 数组
    loc_A = np.array(ref_obj_A_loc[:2])
    loc_B = np.array(ref_obj_B_loc[:2])
    loc_Q = np.array(query_obj_loc[:2])

    # 3. 计算向量
    # vec1: 基准向量 (A -> B)，对应参考代码中的 vec_obj_to_cam1
    vec_A_to_B = loc_B - loc_A
    # vec2: 待测向量 (A -> Q)，对应参考代码中的 vec_obj_to_cam2
    vec_A_to_Q = loc_Q - loc_A

    # 忽略距离过近的情况
    if np.linalg.norm(vec_A_to_B) < 1e-3 or np.linalg.norm(vec_A_to_Q) < 1e-3:
        return False

    # 4. 计算旋转角度 (完全复用参考代码逻辑)
    # angle 是从 vec_A_to_B 到 vec_A_to_Q 的旋转角
    # 正值代表顺时针，负值代表逆时针
    angle = calculate_rotation(vec_A_to_B, vec_A_to_Q)

    # 5. 计算索引偏移
    # 参考代码：rel_idx = int(round(angle / 45)) % 8
    rel_idx = int(round(angle / 45.0))

    # 6. 推导预期方向
    # 获取 B 的基准索引
    base_idx = direction_names.index(dir_B_to_A)

    # 计算 Q 的预期索引 (注意取模处理循环)
    # 因为列表是顺时针排列，且 calculate_rotation 顺时针为正，直接相加即可
    target_idx = (base_idx + rel_idx) % 8

    predicted_dir = direction_names[target_idx]

    # 7. 比对
    return predicted_dir == dir_query

def is_object_too_small(bbox_2d: dict, min_area: float, min_side: float) -> bool:
    """
    检查单个物体的2D包围框是否过小。

    Args:
        bbox_2d: 单个物体的 2D 包围框字典，
                 包含 'max_x', 'min_x', 'max_y', 'min_y'。
        min_area: 最小允许面积阈值。
        min_side: 最小允许边长阈值。

    Returns:
        如果物体过小，返回 True；否则返回 False。
    """

    # 注意：这里假设调用者已确保 bbox_2d 字典非空。
    # 如果调用者不能保证，则需要在调用前或函数内部检查。
    if not bbox_2d:
        # 如果没有Bbox数据，视为不合格（过小或无效），返回 True
        return True

    # 1. 计算面积
    width = bbox_2d['max_x'] - bbox_2d['min_x']
    height = bbox_2d['max_y'] - bbox_2d['min_y']
    area = width * height

    # 2. 检查面积和边长是否过小
    if area < min_area or min(width, height) < min_side:
        # 物体过小
        return True

    # 物体通过检查
    return False

def should_filter_camera_pair(cam1_loc: np.ndarray, cam1_for: np.ndarray,
                              cam2_loc: np.ndarray, cam2_for: np.ndarray) -> bool:
    # --- 定义阈值 (可根据需求调整) ---
    # 1. 距离阈值：如果相机距离小于此值，则认为位置重合度“高”
    DISTANCE_THRESHOLD = 0.2  # 例如，0.5 米

    # 2. 角度阈值：如果相机朝向夹角小于此值，则认为朝向重合度“高”
    # 使用弧度，例如 5 度
    ANGLE_THRESHOLD_DEG = 5.0
    ANGLE_THRESHOLD_RAD = np.radians(ANGLE_THRESHOLD_DEG)

    """
    判断一对相机是否因位置和朝向重合度都过高而应该被过滤。

    Args:
        cam1_loc, cam1_for: 相机 1 的位置和前向向量。
        cam2_loc, cam2_for: 相机 2 的位置和前向向量。

    Returns:
        bool: 如果位置和朝向都高度重合，返回 True (应过滤)。
    """

    # --- 1. 位置重合度检查 (距离检查) ---

    # 计算两个相机位置之间的欧几里得距离
    distance = np.linalg.norm(cam1_loc - cam2_loc)

    # 如果距离非常小，则位置重合度高
    position_overlap_high = (distance < DISTANCE_THRESHOLD)

    # --- 2. 朝向重合度检查 (角度检查) ---

    # 计算两个前向向量之间的夹角（弧度）
    angle_rad = calculate_angle_between_vectors(cam1_for, cam2_for)

    # 如果夹角非常小，则朝向重合度高
    orientation_overlap_high = (angle_rad < ANGLE_THRESHOLD_RAD)

    # --- 3. 综合判断 ---

    # 只有当“位置重合度高” AND “朝向重合度高”时，才进行过滤
    return position_overlap_high and orientation_overlap_high

def should_filter_camera_pair_strong(cam1_loc: np.ndarray, cam1_for: np.ndarray,
                                     cam2_loc: np.ndarray, cam2_for: np.ndarray) -> bool:
    # --- 阈值设置 ---
    MIN_DISTANCE = 0.2  # 距离太近（没有位移感）
    MAX_DISTANCE = 5.0  # 距离太远（可能不在同一个房间或失去关联）

    MIN_ANGLE_DEG = 5.0  # 角度太小（几乎重合）
    MAX_ANGLE_DEG = 150.0  # 角度太大（视野可能完全不交叠）

    # --- 1. 位置筛选 ---
    distance = np.linalg.norm(cam1_loc - cam2_loc)
    # 如果距离太近 或 距离太远，过滤
    if distance < MIN_DISTANCE or distance > MAX_DISTANCE:
        return True

    # --- 2. 朝向筛选 ---
    # 计算夹角（弧度）
    angle_rad = calculate_angle_between_vectors(cam1_for, cam2_for)
    angle_deg = np.degrees(angle_rad)

    # 如果角度太小（平行） 或 角度太大（转向过大），过滤
    if angle_deg < MIN_ANGLE_DEG or angle_deg > MAX_ANGLE_DEG:
        return True

    # --- 3. 综合判断 ---
    # 只有在定义的 [MIN, MAX] 区间内的相机对才会返回 False (保留)
    return False
