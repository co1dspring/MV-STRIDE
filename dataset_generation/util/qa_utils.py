# -*- coding: gbk -*-
import random
import numpy as np
import math
from typing import Dict, List, Tuple, Set, Union, Any
from scipy.spatial.transform import Rotation as R
from icecream import ic
from pathlib import Path
import json
import re

def get_image_size(cam_data):
    if 'width' in cam_data:
        W = cam_data['width']
        H = cam_data['height']
    else:
        HW_str = cam_data['image_size_HW']

        # 匹配括号内的数字
        # \d+ 表示匹配一个或多个数字
        matches = re.findall(r'\d+', HW_str)

        if len(matches) >= 2:
            # 假设第一个是高度，第二个是宽度
            H = int(matches[0])
            W = int(matches[1])

        else:
            print("未能正确解析字符串")
    return H, W

def format_bbox_dict_to_str(bbox_dict, H, W):
    """
    将字典形式的绝对坐标 bbox 归一化并转换为字符串 "[min_x, min_y, max_x, max_y]"
    范围：0-1000
    """
    # 归一化计算并限制在 0-1000 范围内
    # x 轴使用 W (宽度)，y 轴使用 H (高度)
    min_x = min(1000, max(0, round(bbox_dict['min_x'] / W * 1000)))
    min_y = min(1000, max(0, round(bbox_dict['min_y'] / H * 1000)))
    max_x = min(1000, max(0, round(bbox_dict['max_x'] / W * 1000)))
    max_y = min(1000, max(0, round(bbox_dict['max_y'] / H * 1000)))

    # 按照 [x1, y1, x2, y2] 顺序排列
    res = [min_x, min_y, max_x, max_y]

    # 返回字符串形式，例如 "[123, 456, 789, 900]"
    return str(res)

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

    correct_description = rel_direction
    sorted_options = list(options)
    distractors = [option for option in sorted_options if option != correct_description]

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

    return generate_shuffled_choices_text([correct_description], distractors)

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

        options_list = ['up', 'down', 'left', 'right']

    # --- 3. 最终格式化 ---
    distractors = [option for option in options_list if option != correct_description]

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
