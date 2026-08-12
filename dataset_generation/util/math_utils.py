# -*- coding: gbk -*-
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

def get_world_to_camera_matrix(cam_data: Dict[str, Any]) -> np.ndarray:
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
    P_world_to_camera = M

    return P_world_to_camera

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
