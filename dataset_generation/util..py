# -*- coding: gbk -*-
import random
import numpy as np
import math
from typing import Dict, List, Tuple, Set, Union, Any
from scipy.spatial.transform import Rotation as R
from icecream import ic
from pathlib import Path
import json


def load_config(config_path: Path) -> Dict:
    """加载外部配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            print(f"成功加载配置: {config_path}")
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL ERROR: Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"FATAL ERROR: Configuration file {config_path} has invalid JSON format.")

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

def calculate_signed_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算从向量 v1 到向量 v2 的旋转角度（角度制）。
    假设坐标系为标准笛卡尔坐标系，顺时针为正方向（为了匹配 North->East->South 的列表顺序）。
    注意：通常 atan2 是逆时针为正，这里我们需要根据列表顺序做适配。
    """
    # 计算两个向量的角度 (atan2 返回范围是 -pi 到 pi)
    # np.arctan2(y, x)
    ang1 = np.arctan2(v1[1], v1[0])
    ang2 = np.arctan2(v2[1], v2[0])

    # 计算差值 (ang2 - ang1)
    # 在图像坐标系(Y向下)中，X正向转到Y正向是顺时针，atan2依然适用
    # 但我们需要确保方向与 direction_name 列表的顺序一致 (通常是顺时针：北->东->南)

    # 这里我们计算 v1 转到 v2 需要多少度
    diff_rad = ang2 - ang1
    diff_deg = np.degrees(diff_rad)

    # 归一化到 [0, 360)
    diff_deg = (diff_deg + 360) % 360
    return diff_deg


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

def int_to_simple_ordinal_word(n: Union[int, str]) -> str:
    """
    将 1 到 10 的整数转换为其英文序数词（first, second, ..., tenth）。

    Args:
        n: 待转换的整数或字符串形式的数字。

    Returns:
        对应的英文序数词，如果输入超出范围 [1, 10] 或无效则返回空字符串。
    """

    ORDINAL_MAP = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth"
    }

    try:
        n_int = int(n)
    except (ValueError, TypeError):
        return ""  # 非数字输入

    return ORDINAL_MAP.get(n_int, "")

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

def fill_template_placeholders(template: str, is_case_1: bool) -> str:
    """
    根据角色分配（Case 1 或 Case 2）填充模板中的语言指示符。

    :param is_case_1: True 表示 Case 1 (C2为参考, C1为目标)
                      False 表示 Case 2 (C1为参考, C2为目标)
    """
    second_or_last = "second" if random.random() < 0.8 else "last"
    if is_case_1:
        # Case 1: C1 (first) 是参考，C2 (second) 是目标
        ref_word, tar_word = "first", second_or_last
        ref_num, tar_num = "1", "2"
    else:
        # Case 2: C2 (second) 是参考，C1 (first) 是目标
        ref_word, tar_word = second_or_last, "first"
        ref_num, tar_num = "2", "1"


    replacements = {
        "$REF_WORD$": ref_word, "$TAR_WORD$": tar_word,
        "$REF_NUM$": ref_num, "$TAR_NUM$": tar_num,
    }

    # 执行替换
    filled_template = template
    for placeholder, replacement in replacements.items():
        filled_template = filled_template.replace(placeholder, replacement)

    return filled_template


def get_camera_rotation_matrix(cam_data: Dict[str, Any]) -> np.ndarray:
    """
    从 cam_extrinsics 中提取 R_camera_to_world (3x3 旋转矩阵)。
    兼容 字符串(string) 和 列表(list) 格式。
    """
    extrinsics = cam_data.get("cam_extrinsics")

    # 1. 解析外参矩阵 M (M_world_to_camera)
    try:
        if isinstance(extrinsics, str):
            # 如果是字符串，清理括号并解析
            clean_str = extrinsics.replace('[', '').replace(']', '').strip()
            M = np.fromstring(clean_str, sep=' ', dtype=np.float32).reshape(4, 4)
        elif isinstance(extrinsics, (list, np.ndarray)):
            # 如果是列表或 numpy 数组，直接转换并重塑
            M = np.array(extrinsics, dtype=np.float32).reshape(4, 4)
        else:
            raise TypeError(f"不支持的外参类型: {type(extrinsics)}")

    except Exception as e:
        print(f"错误: 无法解析 cam_extrinsics。类型: {type(extrinsics)}, 错误: {e}")
        return np.eye(3)

    # 2. 提取旋转部分 R_world_to_camera (M 的左上角 3x3)
    R_world_to_camera = M[:3, :3]

    # 3. 计算 R_camera_to_world
    # 由于旋转矩阵是正交阵，其逆矩阵等于其转置矩阵
    R_camera_to_world = R_world_to_camera.T

    return R_camera_to_world

def calculate_angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个 3D 向量之间的夹角（弧度）。
    """
    # 归一化向量
    norm_vec1 = vec1 / np.linalg.norm(vec1)
    norm_vec2 = vec2 / np.linalg.norm(vec2)

    # 计算点积 (即余弦值)
    dot_product = np.dot(norm_vec1, norm_vec2)

    # 限制点积范围在 [-1, 1] 以避免浮点误差
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 返回夹角 (弧度)
    return np.arccos(dot_product)


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


import numpy as np


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

def calculate_rotation(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算从vec1到vec2的相对转动角度（范围：-180°~180°）
    规则：顺时针转动为正值，逆时针转动为负值（基于+x=右，+y=前坐标系）
    
    参数：
        vec1: 物体指向cam1的向量（2D numpy数组，[x, y]）
        vec2: 物体指向cam2的向量（2D numpy数组，[x, y]）
    
    返回：
        angle: 相对转动角度（-180.0~180.0，保留1位小数）
    """
    # 归一化向量（仅保留方向信息）
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # 步骤1：计算两向量的夹角（0~180°，点积法）
    dot_product = np.dot(vec1_norm, vec2_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # 处理数值精度问题
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)  # 0~180°基础夹角
    
    # 步骤2：用叉积判断方向（适配+x=右，+y=前坐标系）
    # 2D叉积公式：vec1.x * vec2.y - vec1.y * vec2.x
    cross_product = vec1_norm[0] * vec2_norm[1] - vec1_norm[1] * vec2_norm[0]
    
    # 步骤3：根据方向确定符号（顺时针为正，逆时针为负）
    if cross_product < 0:
        # 叉积<0 → vec2在vec1右侧 → 顺时针转动（取正值）
        return round(angle_deg, 1)
    elif cross_product > 0:
        # 叉积>0 → vec2在vec1左侧 → 逆时针转动（取负值）
        return round(-angle_deg, 1)
    else:
        # 叉积=0 → 共线（同向为0°，反向为180°）
        return 180.0 if angle_deg > 90 else 0.0

def vector_angle(v1, v2) -> float:
    """计算两向量的夹角（度）"""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot = np.dot(v1_norm, v2_norm)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def generate_random_int_choices(answer_int):
    # --------------------------
    # 生成4个选项（1正确+3干扰）
    # --------------------------
    options = set()
    # 1. 添加正确选项
    options.add(answer_int)
    
    # 2. 生成干扰选项：围绕正确答案浮动±1~3，避免0或负数（物体数量≥1）
    while len(options) < 4:
        # 随机浮动值（±1、±2、±3，避免干扰项与正确答案差距过大）
        offset = random.choice([-3, -2, -1, 1, 2, 3])
        distractor = answer_int + offset
        # 确保干扰项是正整数（物体数量不可能为0或负数）
        if distractor > 0:
            options.add(distractor)
    
    # 3. 选项排序+格式化（A/B/C/D选项）
    correct_description = str(answer_int)
    sorted_options = list(options)
    distractors = [str(option) for option in sorted_options if option != answer_int]
    # random.shuffle(sorted_options)
    # option_str = ", ".join([
    #     f"A: {sorted_options[0]}",
    #     f"B: {sorted_options[1]}",
    #     f"C: {sorted_options[2]}",
    #     f"D: {sorted_options[3]}"
    # ])
    #
    # # 4. 找到正确选项的字母（A/B/C/D）
    # correct_letter = chr(65 + sorted_options.index(answer_int))  # 65是'A'的ASCII码
    # answer = correct_letter # f"{correct_letter}. {answer_int}"
    # return option_str, answer
    return generate_shuffled_choices_text([correct_description], distractors)

def generate_orientation_choices(rel_direction, direct_8=True, direct_geo=True):
    # --------------------------
    # 生成4个方向选项（1正确+3干扰）
    # --------------------------
    options = set()
    
    # 确定方向词库
    if direct_8:
        if direct_geo:
            direction_pool = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
        else:
            direction_pool = ['front', 'front right', 'right', 'back right', 'back', 'back left', 'left', 'front left']
    else:
        direction_pool = ['front', 'right', 'back', 'left']
    
    # 1. 添加正确选项
    options.add(rel_direction)
    
    # 2. 生成干扰选项：从方向词库中随机选择，避免重复
    while len(options) < 4:
        # 从方向词库中随机选择干扰项
        distractor = random.choice(direction_pool)
        # 确保干扰项与正确选项不同
        if distractor != rel_direction:
            options.add(distractor)
    
    # 3. 选项排序（按字母顺序或固定顺序）
    # if direct_8:
    #     # 8方向时按字母顺序排序
    #     sorted_options = sorted(options)
    # else:
    #     # 4方向时按固定逻辑顺序：front, right, back, left
    #     direction_order = ['front', 'right', 'back', 'left']
    #     # 确保选项按逻辑顺序排列，缺失的用其他补全
    #     sorted_options = []
    #     for direction in direction_order:
    #         if direction in options:
    #             sorted_options.append(direction)
    #     # 添加剩余选项
    #     for direction in options:
    #         if direction not in sorted_options:
    #             sorted_options.append(direction)

    correct_description = rel_direction
    sorted_options = list(options)
    distractors = [option for option in sorted_options if option != correct_description]
    # random.shuffle(sorted_options)
    #
    # # 4. 格式化选项（A/B/C/D）
    # option_str = ", ".join([
    #     f"A: {sorted_options[0]}",
    #     f"B: {sorted_options[1]}",
    #     f"C: {sorted_options[2]}",
    #     f"D: {sorted_options[3]}"
    # ])
    #
    # # 5. 找到正确选项的字母（A/B/C/D）
    # correct_letter = chr(65 + sorted_options.index(rel_direction))  # 65是'A'的ASCII码
    # answer = correct_letter # f"{correct_letter}. {rel_direction}"
    #
    # return option_str, answer
    return generate_shuffled_choices_text([correct_description], distractors)

def get_description(ref_obj_id: str, cam1_data: Dict, objects: Dict) -> str:
    """
    生成参考物体在相机1视角中的描述
    
    参数：
        ref_obj_id: 参考物体的ID
        cam1_data: 相机1的数据（包含物体标注，格式如{"objects": {物体ID: {"bbox_2d": ...}, ...}}）
        objects: 所有物体的元数据（包含类别，格式如{物体ID: {"category": ...}, ...}）
    
    返回：
        str: 参考物体的描述（如"画"、"左侧的椅子"、"中间的花瓶"）
    """
    # 1. 提取参考物体的类别
    ref_category = objects[ref_obj_id]['category']
    
    # 2. 提取相机1中所有物体及同类别物体
    cam1_objects = cam1_data.get('objects', {})  # 相机1中的物体标注
    
    # 收集相机1中与参考物体同类别的所有物体（ID列表）
    same_category_ids = [
        obj_id for obj_id in cam1_objects.keys()
        if objects[obj_id]['category'] == ref_category
    ]
    count = len(same_category_ids)
    
    # 3. 处理1个同类别物体的情况
    if count == 1:
        return True, ref_category, ref_category
    
    # 4. 处理超过3个同类别物体的情况（丢弃，返回基础类别）
    if count > 3:
        return False, ref_category, ref_category
    
    # 5. 提取同类别物体的2D bbox中心坐标（用于位置比较）
    def get_bbox_center(bbox: Dict) -> Tuple[float, float]:
        """计算bbox中心的(x, y)坐标（x：左右，y：上下）"""
        center_x = (bbox['min_x'] + bbox['max_x']) / 2
        center_y = (bbox['min_y'] + bbox['max_y']) / 2
        return (center_x, center_y)
    
    # 生成 (物体ID, 中心x, 中心y) 列表
    obj_centers: List[Tuple[str, float, float]] = []
    for obj_id in same_category_ids:
        bbox = cam1_objects[obj_id]['bbox_2d']
        cx, cy = get_bbox_center(bbox)
        obj_centers.append((obj_id, cx, cy))
    
    # 6. 判断用左右还是上下区分（基于坐标离散程度）
    x_coords = [cx for _, cx, _ in obj_centers]
    y_coords = [cy for _, _, cy in obj_centers]
    
    # 计算x（左右）和y（上下）方向的离散度（方差越大，分布越分散）
    x_variance = np.var(x_coords)
    y_variance = np.var(y_coords)
    use_horizontal = x_variance > y_variance  # 左右更分散则用水平方向
    
    # 7. 按方向排序物体
    if use_horizontal:
        # 左右方向：按x坐标升序（左→右）
        sorted_objs = sorted(obj_centers, key=lambda x: x[1])
    else:
        # 上下方向：按y坐标升序（上→下，y越小越靠上）
        sorted_objs = sorted(obj_centers, key=lambda x: x[2])
    
    # 8. 确定参考物体的位置描述
    # 位置词汇（2个或3个物体）
    if use_horizontal:
        # 左右方向
        pos_2 = ["leftmost", "rightmost"]  # 最左、最右
        pos_3 = ["leftmost", "middle", "rightmost"]  # 最左、中间、最右
    else:
        # 上下方向
        pos_2 = ["topmost", "bottommost"]  # 最上、最下
        pos_3 = ["topmost", "middle", "bottommost"]  # 最上、中间、最下
    positions = pos_2 if count == 2 else pos_3
    
    # 找到参考物体在排序后的索引
    ref_index = next(i for i, (obj_id, _, _) in enumerate(sorted_objs) if obj_id == ref_obj_id)
    
    # 9. 生成最终描述（位置+类别）
    return True, ref_category, f"{positions[ref_index]} {ref_category}"


def generate_size_comparison_choices(obj1_size, obj2_size, obj1_des, obj2_des,
                                     question_des, proximity_factor, ask_for_greater):
    """
    比较两个对象的大小，并生成选择题选项和答案。

    Args:
        obj1_size (float/int): 第一个对象的大小数值。
        obj2_size (float/int): 第二个对象的大小数值。
        obj1_des (str): 第一个对象的描述 (如 'obj1's length')。
        obj2_des (str): 第二个对象的描述 (如 'obj2's length')。
        question_des (str): 问题的描述部分，用于'相同'选项 (如 'length is the same')。
        proximity_factor (float): 邻近因子。如果二者比值不超过此值，则认为大小“相同”。

    Returns:
        dict: 包含 'options' (格式化的选项字符串) 和 'answer' (正确答案字符串) 的字典。
    """

    # 1. 确定比较结果的描述文本
    # 注意：这里的选项内容是比较结果的“关系方向”描述。
    SAME_TEXT = f"The same {question_des}"
    OBJ1_LARGER_TEXT = f"The {obj1_des} in Figure 1"  # 例如 'The length of object 1'
    OBJ2_LARGER_TEXT = f"The {obj2_des} in Figure 2"  # 例如 'The length of object 2'
    UNCERTAIN_TEXT = "Sometimes the former, sometimes the latter"

    # 2. 计算比值和确定正确的关系方向 (Relational Direction)
    # 确保 obj1_size 和 obj2_size 都是正数，避免除以零或复杂情况
    if obj1_size <= 0 or obj2_size <= 0:
        raise ValueError("obj1_size and obj2_size must be positive.")

    # 计算比值，确保分子是较大的数值
    ratio = max(obj1_size, obj2_size) / min(obj1_size, obj2_size)

    # 判断正确选项
    if ratio <= proximity_factor:
        # 情况 A: 接近相等，选择 '相同'
        correct_text = SAME_TEXT
    elif obj1_size > obj2_size:
        # 情况 B: obj1 更大
        correct_text = OBJ1_LARGER_TEXT if ask_for_greater else OBJ2_LARGER_TEXT
    else:  # obj2_size > obj1_size
        # 情况 C: obj2 更大
        correct_text = OBJ2_LARGER_TEXT if ask_for_greater else OBJ1_LARGER_TEXT

    # 3. 定义选项内容
    # 为了简化和符合您的要求，我们固定 A/B/C/D 选项的内容和顺序
    # 注意：根据您的要求，选项内容是：相同 / obj1 / obj2 / 不确定

    # 选项内容列表（固定顺序 A, B, C, D）
    correct_description = correct_text
    fixed_options = [
        SAME_TEXT,
        OBJ1_LARGER_TEXT,
        OBJ2_LARGER_TEXT,
        UNCERTAIN_TEXT  # 假设 D 永远是不确定项
    ]
    distractors = [option for option in fixed_options if option != correct_description]
    # random.shuffle(fixed_options)
    #
    # # 4. 格式化选项（A/B/C/D）
    # option_str_list = []
    # for i, option_text in enumerate(fixed_options):
    #     letter = chr(65 + i)  # A=65, B=66, ...
    #     option_str_list.append(f"{letter}: {option_text}")
    #
    # option_str = ", ".join(option_str_list)
    #
    # # 5. 找到正确选项的字母（A/B/C/D）
    # # 在固定选项列表中找到正确结果的索引
    # try:
    #     correct_index = fixed_options.index(correct_text)
    #     correct_letter = chr(65 + correct_index)
    # except ValueError:
    #     # 理论上不会发生，除非 rel_direction 不在 fixed_options 中
    #     correct_letter = "?"
    #
    # answer = correct_letter # f"{correct_letter}. {rel_direction}"
    #
    # return option_str, answer
    return generate_shuffled_choices_text([correct_description], distractors)


# --- 词汇映射常量 ---
FORWARD_VOCAB_T3 = ["front", "forward"]
BACKWARD_VOCAB_T3 = ["back", "rear", "backward"]
LEFT_VOCAB = ["left"]
RIGHT_VOCAB = ["right"]

# 模板库定义
DESCRIPTION_TEMPLATES = [
    # 模板 1: 动作 + 侧面方向 (保留 forward/backward)
    "{forward_text} to the {right_text}",
    # 模板 2: 侧面方向 + 动作 (保留 forward/backward)
    "to the {right_text} while moving {forward_text}",
    # 模板 3: 简洁的方向组合 (使用替换词汇，如 'back right')
    "{forward_text} {right_text}",
]

# 纯粹的前后/左右描述模板 (统一使用 moving XXX)
PURE_TEMPLATES = {
    "forward": "moving forward",
    "backward": "moving backward",
    "left": "moving left",
    "right": "moving right"
}

PURE_WITHOUT_MOVING_TEMPLATES = {
    "forward": ["In front"],
    "backward": ["Behind"],
    "left": ["On the left", "Left", "Directly to the left"],
    "right": ["One the right", "Right", "Directly to the right"]
}

# 所有可能的混合方向组合（核心）
ALL_MIXED_CORES: List[Tuple[str, str]] = [
    ("forward", "right"), ("forward", "left"),
    ("backward", "right"), ("backward", "left"),
]


def _get_random_vocab_for_template_3(direction: str) -> str:
    """仅为模板 3 获取随机词汇（front/back/rear）。"""
    if direction == "forward":
        return random.choice(FORWARD_VOCAB_T3)
    elif direction == "backward":
        return random.choice(BACKWARD_VOCAB_T3)
    elif direction == "left":
        return random.choice(LEFT_VOCAB)
    elif direction == "right":
        return random.choice(RIGHT_VOCAB)
    return ""


def _generate_description(f_core: str, r_core: str, template_idx: int, vocab_map: Dict[str, str]) -> str:
    """根据核心方向和模板索引生成描述文本，并使用固定的词汇映射。"""

    selected_template = DESCRIPTION_TEMPLATES[template_idx]

    # 根据 f_core 和 r_core 从固定的 vocab_map 中获取词汇
    f_text = vocab_map[f_core]
    r_text = vocab_map[r_core]

    # 模板 1 & 2 描述风格：如果词汇是 'front'/'back'/'left'/'right'，我们可能需要特殊处理
    # 为了简化，我们假设模板 1 & 2 使用 core 词汇 ('forward'/'backward')
    # 或者，我们让 vocab_map 决定一切

    if template_idx != 2:
        # 模板 1 & 2 可能是描述性的，需要使用 'moving forward' 的形式
        # 这里需要您定义：模板 1/2 究竟是 'forward to the left' 还是 'moving front to the left'？
        # 假设模板 1/2 始终使用核心词 'forward'/'backward'，除非模板 3 强制替换
        if f_core in ['forward', 'backward']:
            f_text = f_core
        if r_core in ['left', 'right']:
            r_text = r_core

    # 如果是模板 3 (简洁风格)，则使用 vocab_map 中预选的词汇
    if template_idx == 2:
        f_text = vocab_map[f_core]
        r_text = vocab_map[r_core]

    return selected_template.format(forward_text=f_text, right_text=r_text)


def _generate_pure_description(direction: str, vocab_map: Dict[str, str]) -> str:
    """生成纯粹的前后/左右描述。"""

    # 我们的 PURE_TEMPLATES 已经包含了 'moving XXX'，这里不需要替换
    return PURE_TEMPLATES.get(direction, "Unknown Direction")

def _get_random_pure_vocab(direction: str) -> str:
    """从 PURE_VOCAB_MAPS 中随机选择一个表述。"""
    return random.choice(PURE_WITHOUT_MOVING_TEMPLATES.get(direction, [f"moving {direction}"]))


def generate_direction_choices(forward_dir: str, right_dir: str, moving=True) -> Tuple[str, str]:
    """
    根据给定的前后/左右方向描述，生成关于移动方向的选择题选项和答案。
    """

    # --- 1. 核心步骤：生成并锁定词汇映射表 (VOCAB_MAP) ---
    vocab_map: Dict[str, str] = {}

    # 1.1 锁定前后词汇
    vocab_map['forward'] = random.choice(FORWARD_VOCAB_T3)
    vocab_map['backward'] = random.choice(BACKWARD_VOCAB_T3)

    # 1.2 锁定左右词汇
    vocab_map['left'] = random.choice(LEFT_VOCAB)  # 总是 'left'
    vocab_map['right'] = random.choice(RIGHT_VOCAB)  # 总是 'right'

    # 1. 确定核心方向 (确保输入兼容性)
    # f_core = 'forward' if forward_dir.lower() in FORWARD_VOCAB_T3 else ('backward' if forward_dir.lower() in BACKWARD_VOCAB_T3 else '')
    # r_core = 'right' if right_dir.lower() in RIGHT_VOCAB else ('left' if right_dir.lower() in LEFT_VOCAB else '')
    f_core = forward_dir
    r_core = right_dir

    correct_description = ""
    distractors: List[str] = []

    # --- 2. 确定正确答案的文本、模板和干扰项 ---

    if f_core and r_core:
        # 情况 A: 混合方向移动
        template_idx = random.randrange(len(DESCRIPTION_TEMPLATES))
        if not moving:
            template_idx = 2

        # 2.1 生成正确答案
        correct_description = _generate_description(f_core, r_core, template_idx, vocab_map)

        # 2.2 生成干扰项 (3个明确错误的混合方向)
        distractor_directions = [
            ('backward' if f_core == 'forward' else 'forward', r_core),
            (f_core, 'left' if r_core == 'right' else 'right'),
            ('backward' if f_core == 'forward' else 'forward', 'left' if r_core == 'right' else 'right')
        ]

        for d_f_core, d_r_core in distractor_directions:
            distractors.append(_generate_description(d_f_core, d_r_core, template_idx, vocab_map))
        if moving:
            distractors.append('Not moving')

    elif f_core or r_core:
        move_dir = f_core if f_core else r_core
        if moving:
            # 情况 B: 纯粹方向移动
            correct_description = _generate_pure_description(move_dir, vocab_map)

            all_distractors_set: Set[str] = set()

            # 2.1 引入纯粹方向的干扰项 (2个)
            all_pure_moves = ["forward", "backward", "left", "right"]
            pure_distractors_cores = [d for d in all_pure_moves if d != move_dir]

            # 随机选择 2 个纯粹方向作为干扰项
            selected_pure_distractors = random.sample(pure_distractors_cores, 2)
            for d in selected_pure_distractors:
                all_distractors_set.add(_generate_pure_description(d, vocab_map))

            # 2.2 引入混合方向的干扰项 (1个)

            # 找到与正确答案不冲突的混合方向
            mixed_distractor_cores = [
                (f, r) for f, r in ALL_MIXED_CORES
                if f != f_core and r != r_core
            ]

            # 随机选择一个混合模板和一组核心方向
            if mixed_distractor_cores:
                d_f_core, d_r_core = random.choice(mixed_distractor_cores)
                template_idx = random.randrange(len(DESCRIPTION_TEMPLATES))
                all_distractors_set.add(_generate_description(d_f_core, d_r_core, template_idx, vocab_map))

            # 确保干扰项至少有 3 个
            distractors = list(all_distractors_set)
            while len(distractors) < 3:
                distractors.append(f"Some other movement ({len(distractors)})")
            distractors = random.sample(distractors, 3)  # 最终选择 3 个
            distractors.append('Not moving')
        else:
            # --- 新模式：强制 4 个纯粹方向，随机表述 ---

            # 1. 确定选项核心
            pure_directions = ["forward", "backward", "left", "right"]

            # 2. 生成选项列表，每个方向随机选择一个表述
            for d in pure_directions:
                description = _get_random_pure_vocab(d)  # <--- 使用随机表述函数
                # 确定正确答案
                if d == move_dir:
                    correct_description = description
                else:
                    distractors.append(description)

    else:
        # 情况 C: 没有明显移动
        correct_description = "Not moving"
        all_possible_descriptions: Set[str] = set()

        # C.1 生成所有混合方向的干扰项
        for d_f_core, d_r_core in ALL_MIXED_CORES:
            # 随机选择一个模板 (0, 1, 或 2)
            template_idx = random.randrange(len(DESCRIPTION_TEMPLATES))
            description = _generate_description(d_f_core, d_r_core, template_idx, vocab_map)
            all_possible_descriptions.add(description)

        # C.2 生成所有纯粹方向的干扰项
        pure_directions = ["forward", "backward", "left", "right"]
        for direction in pure_directions:
            # 使用 'moving XXX' 模板
            pure_desc = _generate_pure_description(direction, vocab_map)
            all_possible_descriptions.add(pure_desc)

            # 使用 PURE_WITHOUT_MOVING_TEMPLATES 模板 (如果需要更多样化)
            # pure_desc_rand = _get_random_pure_vocab(direction)
            # all_possible_descriptions.add(pure_desc_rand)

        # C.3 从所有可能性中随机抽取 3 个干扰项
        # 确保选项数量足够，如果少于 3 个（不应发生，但以防万一）
        max_distractors = 3
        if len(all_possible_descriptions) < max_distractors:
            distractors = list(all_possible_descriptions)
            while len(distractors) < max_distractors:
                distractors.append(f"Generic Movement {len(distractors)}")
        else:
            distractors = random.sample(list(all_possible_descriptions), max_distractors)

    # --- 3. 格式化和输出 ---

    # 将正确答案和干扰项组合
    # 按现在的代码逻辑distractors不能包含correct_description，否则随机选择会导致如果选到了correct_description，那总体选项数量就少于4
    distractors = random.sample(list(distractors), 3)
    # options_list = list(set([correct_description] + distractors))  # 使用 set 去重
    #
    # # 如果选项不足 4 个，则填充
    # # ic(f_core, r_core, correct_description, distractors, options_list, moving)
    # assert len(options_list) == 4, f"选项列表生成失败，选项数量不足4个，实际数量: {len(options_list)}"
    #
    # random.shuffle(options_list)
    #
    # option_str_list = []
    # correct_letter = ""
    #
    # for i, option_text in enumerate(options_list[:4]):  # 只取前四个选项
    #     letter = chr(65 + i)
    #     option_str_list.append(f"{letter}: {option_text}")
    #
    #     if option_text == correct_description:
    #         correct_letter = letter
    #
    # options_str = ", ".join(option_str_list)
    # answer_str = correct_letter # f"{correct_letter}. {correct_description}"
    #
    # return options_str, answer_str
    return generate_shuffled_choices_text([correct_description], distractors)

def get_relative_direction(cam_loc, obj_loc, cam_for, thresh=0.01):
    # 定义混合移动的判定阈值 (与旋转场景保持一致，但可能需要根据平移数据的特性调整)
    TWO_DIR_RATIO = 0.1
    # 1. 相机方向向量取反
    cam_for = - cam_for

    # 2. 计算位移向量
    T_vec = obj_loc - cam_loc

    # 3. 建立cam1的局部坐标系
    # 世界Z轴定义为上轴 (Up Vector)
    Wz_vec = np.array([0, 0, 1])
    # 计算右轴 (Right Vector, R_vec)
    # R = F x Wz (遵循右手定则，且垂直于 F 和 Wz)
    R_vec = np.cross(cam_for, Wz_vec)
    # 归一化右轴（除非F和Wz平行，否则R_vec不会是零向量）
    # 如果 F_vec 与 Wz_vec 完全平行（相机垂直朝上或朝下），则 R_vec 模长为 0。
    # 我们假设相机不会精确地垂直于地面拍摄。
    R_norm = np.linalg.norm(R_vec)
    if R_norm < 1e-6:
        # 紧急处理：如果相机垂直朝向，定义右轴为世界X轴
        R_vec = np.array([1, 0, 0])
    else:
        R_vec = R_vec / R_norm

    # 4. 投影位移向量到局部轴上
    # 使用点积 (Dot Product) 获取分量
    D_forward = np.dot(T_vec, cam_for)
    D_right = np.dot(T_vec, R_vec)

    # 5. 转化为文本描述

    # 确定前后方向
    if D_forward > thresh:
        forward_dir = "forward"
    elif D_forward < -thresh:
        forward_dir = "backward"
    else:
        forward_dir = ""  # 移动不明显，或纯粹的左右平移

    # 确定左右方向
    if D_right > thresh:
        right_dir = "right"
    elif D_right < -thresh:
        right_dir = "left"
    else:
        right_dir = ""  # 移动不明显，或纯粹的前后平移

    if forward_dir and right_dir:
        # 两个方向都超过了绝对阈值 (thresh)
        # 获取绝对值
        abs_D_forward = abs(D_forward)
        abs_D_right = abs(D_right)

        # 确定主导和次要移动量
        if abs_D_forward > abs_D_right:
            main_move = abs_D_forward
            sub_move = abs_D_right
            sub_dir_name = 'right'  # 此时 right 是次要方向
        else:
            main_move = abs_D_right
            sub_move = abs_D_forward
            sub_dir_name = 'forward'  # 此时 forward 是次要方向

        # 检查较弱方向的贡献是否太小
        if sub_move < main_move * TWO_DIR_RATIO:
            # 较弱方向的贡献太小，只保留主导方向
            if sub_dir_name == 'right':
                right_dir = ""  # 仅 forward 显著
            else:  # sub_dir_name == 'forward'
                forward_dir = ""  # 仅 right 显著
        # 否则，两个方向都保留，返回混合移动

    return forward_dir, right_dir

def get_absolute_direction(cam_loc, obj_loc, cam_for, thresh=0.01):
    direction_name = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    # 1. 相机方向向量取反
    cam_for = - cam_for

    # 2. 计算位移向量
    T_vec = obj_loc - cam_loc

    # 计算夹角
    angle = calculate_rotation(cam_for, T_vec)

    # 随机选择参考物体朝向cam1的方向
    main_direction = random.choice(direction_name)

    # 推导vec2的基础方向（基于角度偏移）
    main_idx = direction_name.index(main_direction)  # 用原始方向找索引（避免替换后出错）
    rel_idx = int(round(angle / 45)) % 8  # 相对偏移索引（0~7）
    rel_direction = direction_name[(main_idx + rel_idx) % 8]

    return main_direction, rel_direction

def get_relative_orientation(ref_obj_loc, obj1_loc, obj2_loc):
    direction_name = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    direction_name_2 = ['front', 'right', 'back', 'left']
    # 计算“物体到相机的位置向量”（核心：相对位置）
    vec_obj_to_cam1 = obj1_loc - ref_obj_loc  # 物体指向cam1的向量
    vec_obj_to_cam2 = obj2_loc - ref_obj_loc  # 物体指向cam2的向量

    # 计算夹角
    angle = calculate_rotation(vec_obj_to_cam1, vec_obj_to_cam2)

    # 随机选择参考物体朝向cam1的方向
    main_direction = random.choice(direction_name)

    # 推导vec2的基础方向（基于角度偏移）
    main_idx = direction_name.index(main_direction)  # 用原始方向找索引（避免替换后出错）
    rel_idx = int(round(angle / 45)) % 8  # 相对偏移索引（0~7）
    rel_direction = direction_name[(main_idx + rel_idx) % 8]
    direct_8 = True
    # 随机替换为前后左右
    if rel_idx % 2 == 0 and random.random() < 0.5:
        main_direction = random.choice(direction_name_2)
        main_idx = direction_name_2.index(main_direction)  # 用原始方向找索引（避免替换后出错）
        rel_idx = rel_idx // 2  # 相对偏移索引（0~7）
        rel_direction = direction_name_2[(main_idx + rel_idx) % 4]
        direct_8 = False

    return main_direction, rel_direction, direct_8

def get_relative_orientation_vec(ref_vec, ref_dir, query_vec):
    direction_name_geo = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    direction_name_ego = ['front', 'front right', 'right', 'back right', 'back', 'back left', 'left', 'front left']
    # 计算“物体到相机的位置向量”（核心：相对位置）

    # 计算夹角
    angle = calculate_rotation(ref_vec, query_vec)

    # 随机选择参考物体朝向cam1的方向
    if ref_dir in direction_name_ego:
        direction_name = direction_name_ego
        direct_geo = False
    else:
        direction_name = direction_name_geo
        direct_geo = True

    # 推导vec2的基础方向（基于角度偏移）
    main_idx = direction_name.index(ref_dir)  # 用原始方向找索引（避免替换后出错）
    rel_idx = int(round(angle / 45)) % 8  # 相对偏移索引（0~7）
    query_direction = direction_name[(main_idx + rel_idx) % 8]

    return query_direction, direct_geo

def get_relative_rotation(R1: np.ndarray, R2: np.ndarray, degree_threshold: int, is_scannetpp = False) -> Tuple[str, str]:
    """
    计算从 R1 到 R2 的相对旋转，并确定 Yaw 和 Pitch 的方向（left/right, up/down）。

    :param R1: 相机 1 的 R_camera_to_world 矩阵 (3x3)。
    :param R2: 相机 2 的 R_camera_to_world 矩阵 (3x3)。
    :return: (yaw_dir, pitch_dir) 字符串元组
    """

    # 定义阈值 (需要根据您的数据调整)
    ROT_RAD_THRESHOLD = np.radians(degree_threshold)  # 5度以上视为显著旋转
    TWO_DIR_RATIO = 0.2  # 混合旋转的判定阈值：较弱方向的绝对值必须大于主导方向的 50%

    # 1. 计算相对旋转矩阵 R_rel = R2 * R1_T
    # R_rel 表示从 Cam1 姿态转到 Cam2 姿态所需的旋转
    R_rel = R2 @ R1.T

    # 2. 转换为欧拉角 (Yaw, Pitch, Roll)
    # 使用 'zyx' 顺序（等价于 Roll, Pitch, Yaw 顺序的倒序）来提取
    # 欧拉角通常以弧度表示。
    r = R.from_matrix(R_rel)
    # [Yaw, Pitch, Roll] - 注意 SciPy 默认 ZYX 顺序提取 YPR（Roll-Pitch-Yaw 常用）
    # 我们关注 Z-axis rotation (Yaw) 和 Y-axis rotation (Pitch)

    # 使用 'ZYX' 约定 (Yaw-Pitch-Roll)
    # angles[0] is Yaw (绕Z轴), angles[1] is Pitch (绕Y轴), angles[2] is Roll (绕X轴)
    # 注意：这里的 ZYX 约定是世界坐标系中的，但对于相对旋转，它近似于局部旋转。

    # 更好的方法是使用 'XYZ' 约定 (Roll-Pitch-Yaw)
    # 我们直接使用 'YPR' 约定：
    ypr_angles = r.as_euler('yxz', degrees=False)  # Yaw (Y), Pitch (X), Roll (Z)

    # 假设您的坐标系：
    # Yaw (Y轴) = ypr_angles[0]
    # Pitch (X轴) = ypr_angles[1]
    # Roll (Z轴) = ypr_angles[2]
    delta_yaw = ypr_angles[0]
    delta_pitch = ypr_angles[1]

    # 3. 确定方向
    # 约定：
    # Yaw 正值 (逆时针, 从上往下看): 视野向右转 (Rotates Right)
    # Yaw 负值 (顺时针, 从上往下看): 视野向左转 (Rotates Left)
    # Pitch 正值 (向上看): 视野向上抬 (Rotates Up)
    # Pitch 负值 (向下看): 视野向下俯 (Rotates Down)

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
        # 如果两个方向都超过了 ROT_RAD_THRESHOLD，我们已经得到了混合旋转 (情况 B)
        # 此时 TWO_DIR_RATIO 的作用是防止两个方向只是勉强超过阈值，但其中一个远小于另一个
        # 比如 Yaw=6度，Pitch=1度 (不满足混合条件，只算 Yaw 显著)
        # 获取绝对值，用于比较
        abs_yaw = abs(delta_yaw)
        abs_pitch = abs(delta_pitch)

        # 确定主导和次要旋转
        if abs_yaw > abs_pitch:
            main_rot = abs_yaw
            sub_rot = abs_pitch
        else:
            main_rot = abs_pitch
            sub_rot = abs_yaw

        # 检查较弱的方向是否显著到足以被视为混合
        if sub_rot < main_rot * TWO_DIR_RATIO:
            # 较弱方向的贡献太小，只保留主导方向
            if abs_yaw > abs_pitch:
                pitch_dir = ""  # 仅 Yaw 显著
            else:
                yaw_dir = ""  # 仅 Pitch 显著
        # 否则，两个方向都保留，返回混合旋转

    return yaw_dir, pitch_dir


# 定义选项常量
SINGLE_DIRS = ["up", "down", "left", "right"]
MIXED_DIRS = ["upper right", "upper left", "lower left", "lower right"]
NULL_DIR = "Unable to determine"


def generate_rotation_choices(yaw_dir: str, pitch_dir: str) -> Tuple[str, str]:
    """
    根据给定的 Yaw 和 Pitch 方向生成选项和答案。

    :param yaw_dir: 'left', 'right', or ''
    :param pitch_dir: 'up', 'down', or ''
    :return: (options_str, answer_str)
    """

    # 1. 确定核心答案
    is_yaw_sig = bool(yaw_dir)
    is_pitch_sig = bool(pitch_dir)

    if not is_yaw_sig and not is_pitch_sig:
        # 情况 A: 无明显旋转
        correct_description = NULL_DIR

        # 选项：包含 NULL_DIR + 3个随机单向或混合方向干扰项
        all_possible_moves = SINGLE_DIRS + MIXED_DIRS
        distractors = random.sample(all_possible_moves, 3)
        options_list = [correct_description] + distractors

    elif (is_yaw_sig and is_pitch_sig):
        # 情况 B: 混合旋转 (Both significant)

        # 确定正确答案文本
        p_text = pitch_dir
        y_text = yaw_dir

        # 构造混合答案，例如 "upper right"
        correct_description = f"{p_text} {y_text}".replace("up ", "upper ").replace("down ", "lower ")

        # 2. 生成干扰项

        # 干扰项 1: 仅改变 Yaw
        d1_yaw = 'left' if y_text == 'right' else 'right'
        d1_desc = f"{p_text} {d1_yaw}".replace("up ", "upper ").replace("down ", "lower ")

        # 干扰项 2: 仅改变 Pitch
        d2_pitch = 'up' if p_text == 'down' else 'down'
        d2_desc = f"{d2_pitch} {y_text}".replace("up ", "upper ").replace("down ", "lower ")

        # 干扰项 3: 随机选择一个纯粹方向作为干扰项
        all_pure_moves = SINGLE_DIRS
        d3_desc = random.choice([d for d in all_pure_moves if d != p_text and d != y_text])

        options_list = [correct_description, d1_desc, d2_desc, d3_desc]

    else:
        # 情况 C: 单一显著旋转 (Only Yaw or Only Pitch)

        # 确定正确答案文本
        correct_description = yaw_dir if is_yaw_sig else pitch_dir

        # # 2. 生成干扰项 (3个，包含其他单向和混合方向)
        #
        # # 干扰项 1: 反方向 (单向)
        # d1 = ""
        # if is_yaw_sig: d1 = 'left' if yaw_dir == 'right' else 'right'
        # if is_pitch_sig: d1 = 'up' if pitch_dir == 'down' else 'down'
        #
        # # 干扰项 2 & 3: 随机选择两个不冲突的方向 (可能包含混合)
        #
        # # 排除正确答案和反方向
        # all_distractor_candidates = [d for d in SINGLE_DIRS if d != correct_description and d != d1]
        # all_distractor_candidates += [
        #     d for d in MIXED_DIRS if correct_description not in d and d1 not in d
        # ]
        #
        # # 确保至少有 3 个干扰项
        # if len(all_distractor_candidates) < 3:
        #     # Fallback: 补充一些明显的错误方向
        #     all_distractor_candidates.extend(SINGLE_DIRS)
        #
        # distractors = random.sample(list(set(all_distractor_candidates)), min(3, len(all_distractor_candidates)))
        #
        # options_list = [correct_description] + distractors
        options_list = ['up', 'down', 'left', 'right']

    # --- 3. 最终格式化 ---
    distractors = [option for option in options_list if option != correct_description]
    # # 去重并打乱
    # options_list = list(set(options_list))
    # while len(options_list) < 4:
    #     # 避免选项不足
    #     options_list.append(random.choice(SINGLE_DIRS + MIXED_DIRS))
    #
    # options_list = random.sample(options_list, 4)  # 最终选取 4 个
    #
    # option_str_list = []
    # correct_letter = ""
    #
    # for i, option_text in enumerate(options_list):
    #     letter = chr(65 + i)
    #     option_str_list.append(f"{letter}: {option_text}")
    #
    #     if option_text == correct_description:
    #         correct_letter = letter
    #
    # options_str = ", ".join(option_str_list)
    # answer_str = correct_letter # f"{correct_letter}. {correct_description}"
    #
    # return options_str, answer_str
    return generate_shuffled_choices_text([correct_description], distractors)

def generate_coordinate_direction_choices(forward_dir: str, right_dir: str, setting: str) -> Tuple[str, str]:
    """
    根据给定的前后/左右方向描述，生成固定坐标系正方向设定下移动方向的选择题选项和答案。
    """

    correct_description = ""
    distractors: List[str] = []
    choice_templates = [
        "moving in the {right_dir} and {forward_dir}",
        # "{right_dir}, {forward_dir}"
    ]
    choice_templates_1dir = [
        "moving in {dir}"
        # "{right_dir}, {forward_dir}"
    ]

    if setting == "+Y up, -Z forward":
        direction_map = {
            'left': 'negative X direction',
            'right': 'positive X direction',
            'forward': 'negative Z direction',
            'backward': 'positive Z direction'
        }
    elif setting == "+Z up, +X forward":
        direction_map = {
            'left': 'positive Y direction',
            'right': 'negative Y direction',
            'forward': 'positive X direction',
            'backward': 'negative X direction'
        }

    if forward_dir and right_dir:
        selected_template = random.choice(choice_templates)
        correct_description = selected_template.format(right_dir=direction_map[right_dir], forward_dir=direction_map[forward_dir])
        distractor_directions = [
            ('backward' if forward_dir == 'forward' else 'forward', right_dir),
            (forward_dir, 'left' if right_dir == 'right' else 'right'),
            ('backward' if forward_dir == 'forward' else 'forward', 'left' if right_dir == 'right' else 'right')
        ]
        for d_f_core, d_r_core in distractor_directions:
            distractors.append(selected_template.format(right_dir=direction_map[d_r_core], forward_dir=direction_map[d_f_core]))
    elif forward_dir or right_dir:
        selected_template = random.choice(choice_templates_1dir)
        move_dir = forward_dir if forward_dir else right_dir
        correct_description = selected_template.format(dir=direction_map[move_dir])
        all_pure_moves = ["forward", "backward", "left", "right"]
        distractors = [selected_template.format(dir=direction_map[d]) for d in all_pure_moves if d != move_dir]

    # --- 3. 格式化和输出 ---

    # 将正确答案和干扰项组合
    # options_list = list(set([correct_description] + distractors))  # 使用 set 去重
    #
    # # 如果选项不足 4 个，则填充
    # while len(options_list) < 4:
    #     options_list.append(f"Undefined Movement {len(options_list)}")
    #
    # random.shuffle(options_list)
    #
    # option_str_list = []
    # correct_letter = ""
    #
    # for i, option_text in enumerate(options_list[:4]):  # 只取前四个选项
    #     letter = chr(65 + i)
    #     option_str_list.append(f"{letter}: {option_text}")
    #
    #     if option_text == correct_description:
    #         correct_letter = letter
    #
    # options_str = ", ".join(option_str_list)
    # answer_str = correct_letter #f"{correct_letter}: {correct_description}"
    #
    # return options_str, answer_str
    return generate_shuffled_choices_text([correct_description], distractors)

def generate_coordinate_rotation_choices(yaw_dir: str, pitch_dir: str, setting: str) -> Tuple[str, str]:
    """
    根据给定的 Yaw 和 Pitch 方向生成选项和答案。

    :param yaw_dir: 'left', 'right', or ''
    :param pitch_dir: 'up', 'down', or ''
    :return: (options_str, answer_str)
    """

    # 1. 确定核心答案

    distractors: List[str] = []
    choice_templates_2dir = [
        "rotate by {yaw_dir}, then by {pitch_dir}",
        # "{right_dir}, {forward_dir}"
    ]
    choice_templates_1dir = [
        "rotate {dir}",
        "rotate by {dir}",
        # "{right_dir}, {forward_dir}"
    ]

    if setting == "+Y up, -Z forward":
        direction_map = {
            'left': 'a negative angle around the Y-axis',
            'right': 'a positive angle around the Y-axis',
            'up': 'a negative angle around the X-axis',
            'down': 'a positive angle around the X-axis'
        }
    elif setting == "+Z up, +X forward":
        direction_map = {
            'left': 'a negative angle around the Z-axis',
            'right': 'a positive angle around the Z-axis',
            'up': 'a positive angle around the Y-axis',
            'down': 'a negative angle around the Y-axis'
        }

    if yaw_dir and pitch_dir:
        selected_template = random.choice(choice_templates_2dir)
        correct_description = selected_template.format(yaw_dir=direction_map[yaw_dir], pitch_dir=direction_map[pitch_dir])
        distractor_directions = [
            ('down' if pitch_dir == 'up' else 'up', yaw_dir),
            (pitch_dir, 'left' if yaw_dir == 'right' else 'right'),
            ('down' if pitch_dir == 'up' else 'up', 'left' if yaw_dir == 'right' else 'right')
        ]
        for pitch, yaw in distractor_directions:
            distractors.append(selected_template.format(yaw_dir=direction_map[yaw], pitch_dir=direction_map[pitch]))
    elif yaw_dir or pitch_dir:
        selected_template = random.choice(choice_templates_1dir)
        move_dir = pitch_dir if pitch_dir else yaw_dir
        correct_description = selected_template.format(dir=direction_map[move_dir])
        all_pure_moves = ["up", "down", "left", "right"]
        distractors = [
            selected_template.format(dir=direction_map[d])
            for d in all_pure_moves if d != move_dir
        ]
    else:
        raise ValueError("这两个值不可能都为空")

    # --- 3. 最终格式化 ---

    # 去重并打乱
    # options_list = list(set([correct_description] + distractors))
    # while len(options_list) < 4:
    #     # 避免选项不足
    #     options_list.append(random.choice(SINGLE_DIRS + MIXED_DIRS))
    #
    # options_list = random.sample(options_list, 4)  # 最终选取 4 个
    #
    # option_str_list = []
    # correct_letter = ""
    #
    # for i, option_text in enumerate(options_list):
    #     letter = chr(65 + i)
    #     option_str_list.append(f"{letter}: {option_text}")
    #
    #     if option_text == correct_description:
    #         correct_letter = letter
    #
    # options_str = ", ".join(option_str_list)
    # answer_str = correct_letter # f"{correct_letter}: {correct_description}"
    #
    # return options_str, answer_str
    return generate_shuffled_choices_text([correct_description], distractors)

def generate_shuffled_choices_text(correct_option: List[str], wrong_option: List[str]) -> Tuple[str, str]:
    # 输入长度为3的误导项列表和长度为1的正确选项列表，输出连接在一起的选项字符串和正确选项字母
    assert len(correct_option) == 1, "The number of correct option must be 1"
    assert len(wrong_option) == 3, "The number of wrong option must be 3"

    options_list = correct_option + wrong_option
    random.shuffle(options_list)
    option_str_list = []
    correct_letter = ""
    correct_description = correct_option[0]

    for i, option_text in enumerate(options_list):
        letter = chr(65 + i)
        option_str_list.append(f"{letter}: {option_text}")

        if option_text == correct_description:
            correct_letter = letter

    options_str = ", ".join(option_str_list)
    answer_str = f"{correct_letter}: {correct_description}"  # correct_letter

    return options_str, answer_str
