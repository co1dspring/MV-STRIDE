# -*- coding: gbk -*-
import os, sys
import random
import json
import logging
import numpy as np

from collections import Counter
from typing import List, Dict, Tuple, Any, NamedTuple, Callable
from pathlib import Path
from icecream import ic
from tqdm import tqdm
from itertools import combinations, permutations

from util.qa_utils import (generate_random_int_choices, generate_direction_choices, generate_size_comparison_choices, get_description, generate_rotation_choices, generate_orientation_choices, fill_template_placeholders,
                            generate_coordinate_direction_choices, generate_coordinate_rotation_choices, int_to_simple_ordinal_word, generate_shuffled_choices_text)
from util.filter_utils import should_filter_camera_pair, is_object_too_small, is_in_direction, is_object_center_in_room, should_filter_camera_pair_strong
from util.math_utils import vector_angle, calculate_rotation, get_relative_direction, get_relative_orientation, get_camera_rotation_matrix, get_relative_rotation, get_absolute_direction, get_relative_orientation_vec
from util.common_utils import load_config, setup_logging, save_json_data
from multilevel_qa import Multilevel_QA_Generator

class QATypeConfig(NamedTuple):
    generator: Callable
    needs_swap: bool
    max_per_pair: int
    sampling_rate: float


class MSR_QATypeConfig(NamedTuple):
    generator: Callable
    num_of_views_range: Tuple[int, int]
    max_num: int
    sampling_rate: float

class SceneQAGenerator:
    def __init__(self, config_path: str = "./qa_config.json"):
        """
        初始化场景QA生成器
        :param base_dir: 场景父目录
        :param output_root: 问题类型JSON文件的保存根目录
        :param metadata_filename: 元数据文件名
        :param camera_count: 每个场景的相机数量
        """
        # 读取配置文件
        self.config = load_config(Path(config_path))

        # 加载路径
        self.source_data_dir = Path(self.config.get('source_data_dir'))
        self.version_name = self.config.get('version_name')
        self.output_dir = Path(self.config.get('output_dir').format(VERSION_NAME=self.version_name))  # 用于按问题类型保存文件
        self.training_environment_base_dir = self.config.get('training_environment_base_dir')
        self.metadata_filename = self.config.get('metadata_filename')
        self.qa_templates_path = Path(self.config.get('qa_templates_path'))
        self.data_source = self.config.get('data_source')
        self.qa_dependency_tree_path = self.config.get('qa_dependency_tree_path')

        # 加载数值参数
        self.camera_count = self.config.get('camera_count')
        self.if_flipping_enhencement = self.config.get('if_flipping_enhencement')
        self.if_MCA = self.config.get('if_MCA')
        self.MIN_AREA_THRESHOLD = self.config.get('MIN_AREA_THRESHOLD')  # 最小边界框面积（像素）
        self.MIN_SIDE_THRESHOLD = self.config.get('MIN_SIDE_THRESHOLD')   # 最小边界框的短边长度（像素）
        self.common_num_threshold = self.config.get('common_num_threshold')
        self.global_seed = self.config.get('global_seed')
        self.qa_types_per_camera_pair = self.config.get('qa_types_per_camera_pair')
        self.stage_1_proportion = self.config.get('stage_1_proportion')
        self.stage_2_proportion = self.config.get('stage_2_proportion')
        self.stage_3_proportion = self.config.get('stage_3_proportion')
        self.level12_sampling_rate = self.config.get('level12_sampling_rate')

        # 加载文本参数
        self.direction_name = self.config.get('direction_name')
        self.direction_name_2 = self.config.get('direction_name_2')
        self.direction_map = self.config.get('direction_map')
        self.obj_area_map = self.config.get('obj_area_map')
        self.unwanted_cats = self.config.get('unwanted_cats')
        self.room_type = self.config.get('room_type')
        self.obj_cat_with_orientation = self.config.get('obj_cat_with_orientation')
        self.multilevel_qa_mode = self.config.get('multilevel_qa_mode')

        # 初始化函数
        self.scene_folders = self._get_all_scene_folders()
        self.scene_names = [scene.name for scene in self.scene_folders]
        self.multilevel_qa_generator = Multilevel_QA_Generator(parent=self)
        self.qa_types = self._define_qa_types()  # 问题类型：{类型名: 生成函数}
        self.msr_qa_types = self._define_msr_qa_types()
        self.level2_qa_functions = self._define_level12_qa_types("level2_qa_types")
        self.level1_qa_functions = self._define_level12_qa_types("level1_qa_types")
        self.level3_qa_types = self.qa_types | self.msr_qa_types
        self.all_qa_types = self.level3_qa_types | self.level2_qa_functions | self.level1_qa_functions
        self._init_output_dirs()  # 创建问题类型对应的输出目录
        self.set_global_seed(self.global_seed) # 设置全局随机种子
        self.templates = load_config(self.qa_templates_path)
        self.qa_dependency_tree = load_config(self.qa_dependency_tree_path)
        self.log_path = setup_logging(self.output_dir)

    def set_global_seed(self, seed):
        """统一设置多个库的随机种子"""
        # 1. Python 内建的 random 模块
        random.seed(seed)

        # 2. NumPy 库（常用于科学计算和数据处理）
        np.random.seed(seed)

        # 3. 如果您使用了 PyTorch 或 TensorFlow，还需要设置它们
        # import torch
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        # 4. 设置哈希种子（通常用于防止不同 Python 实例间的哈希冲突，间接影响随机性）
        os.environ['PYTHONHASHSEED'] = str(seed)

        print(f"全局随机种子已设置为: {seed}")

    def _get_all_scene_folders(self) -> List[Path]:
        """获取所有场景文件夹"""
        if self.data_source == "scannetpp":
            folders = [p for p in self.source_data_dir.iterdir() if p.is_dir() and str(p).endswith("iphone")]
        else:
            folders = [p for p in self.source_data_dir.iterdir() if p.is_dir()]
        folders.sort()
        print(f"发现 {len(folders)} 个场景文件夹")
        return folders

    def _init_output_dirs(self):
        """创建输出目录"""
        (self.output_dir).mkdir(parents=True, exist_ok=True)

    def _define_qa_types(self) -> Dict[str, callable]:
        """定义问题类型生成函数（键为问题类型名，值为生成函数）"""
        # True代表会交换相机顺序再生成一次QA
        # 数字代表每个相机组最多保留多少条qa
        """从配置加载问题类型生成函数"""
        qa_config_data = self.config.get("qa_types", {})
        qa_types_map: Dict[str, Any] = {}

        # 获取默认值 (这里假设 max_qa_per_cams_default 也是 1)
        default_max_qa = self.config.get("max_qa_per_cams_default", 1)
        overall_sampling_rate = self.config.get("overall_sampling_rate", 1)

        for type_name, params in qa_config_data.items():
            # 1. 使用 getattr 动态查找对应的生成函数
            # function_name 必须是类中已定义的成员方法名称字符串
            generator_func = getattr(self, params['function_name'], None)

            if not generator_func:
                print(f"警告: 未找到对应的生成函数: {params['function_name']}，跳过该类型 {type_name}")
                continue

            # 2. 从配置中读取参数，如果缺失则使用默认值
            flipping = params.get('flipping_enhancement', self.if_flipping_enhencement)
            max_qa = params.get('max_qa_per_cams', default_max_qa)
            sampling_rate = params.get('sampling_rate', 1)

            # 3. 构造 QATypeConfig (假设它的构造函数与您代码中使用的参数一致)
            # 注意: QATypeConfig 构造函数中的最后一个参数 (1) 仍是硬编码的，可以考虑将其也放入配置。
            qa_types_map[type_name] = QATypeConfig(generator_func, flipping, max_qa, sampling_rate * overall_sampling_rate)

        return qa_types_map

    def _define_msr_qa_types(self) -> Dict[str, callable]:
        """从配置加载 MSR 问题类型生成函数"""
        msr_config_data = self.config.get("msr_qa_types", {})
        msr_types_map: Dict[str, Any] = {}

        # 假设 MSQ_QATypeConfig 的 max_num 默认为 10
        default_max_num = self.config.get("max_msr_num_default", 10)
        overall_sampling_rate = self.config.get("overall_sampling_rate", 1)

        for type_name, params in msr_config_data.items():
            generator_func = getattr(self, params['function_name'], None)

            if not generator_func:
                print(f"警告: 未找到对应的 MSR 生成函数: {params['function_name']}，跳过该类型 {type_name}")
                continue

            # MSR_QATypeConfig 需要一个 range (例如 [3, 3])
            range_tuple = tuple(params['range'])
            max_num = params.get('max_num', default_max_num)
            sampling_rate = params.get('sampling_rate', 1)

            msr_types_map[type_name] = MSR_QATypeConfig(generator_func, range_tuple, max_num, sampling_rate * overall_sampling_rate)

        return msr_types_map

    def _define_level12_qa_types(self, typename) -> Dict[str, callable]:
        qa_config_data = self.config.get(typename, {})
        qa_types_map: Dict[str, Any] = {}

        for type_name, params in qa_config_data.items():
            # 1. 使用 getattr 动态查找对应的生成函数
            # function_name 必须是类中已定义的成员方法名称字符串
            qa_types_map[type_name] = {}
            generator_func = getattr(self.multilevel_qa_generator, params['function_name'], None)

            if not generator_func:
                print(f"警告: 未找到对应的生成函数: {params['function_name']}，跳过该类型 {type_name}")
                continue

            qa_types_map[type_name]['func'] = generator_func
            qa_types_map[type_name]['sampling_rate'] = params['sampling_rate']

        return qa_types_map

    def _get_camera_pairs(self, cameras: Dict[str, Any]) -> List[Tuple[str, str, Dict, Dict]]:
        # 计算基准组合数 (N=20 时的组合数 190)
        base_n = 50
        max_pairs_limit = (base_n * (base_n - 1)) // 2  # 190
        camera_names = sorted(list(cameras.keys()))
        random.shuffle(camera_names)
        if len(camera_names) != self.camera_count:
            print(f"警告：相机数量不符（预期{self.camera_count}，实际{len(camera_names)}）")
        all_cameras_pairs = combinations(camera_names, 2)
        if len(cameras.keys()) <= base_n:
            """生成相机两两无重复组合"""
            # 使用permutations生成有序对，第二个参数2表示每个组合包含2个元素
            return [
                (c1, c2, cameras[c1], cameras[c2])
                for c1, c2 in all_cameras_pairs  # 这里替换为permutations
            ]
        else:
            # 相机数量过多
            # 修复：数量过多时，使用集合采样避免生成巨大列表
            sampled_name_pairs = set()

            # 持续采样直到达到 190 对
            # 使用 set 自动处理重复抽样的情况
            while len(sampled_name_pairs) < max_pairs_limit:
                # 随机抽取两个不重复的索引
                c1_name, c2_name = random.sample(camera_names, 2)

                # 排序以保证 (A, B) 和 (B, A) 视为同一种无序组合，存入 set 去重
                pair = tuple(sorted((c1_name, c2_name)))
                sampled_name_pairs.add(pair)

            sorted_sampled_pairs = sorted(list(sampled_name_pairs))
            return [
                (c1, c2, cameras[c1], cameras[c2])
                for c1, c2 in sorted_sampled_pairs
            ]

    # ------------------------------
    # 问题生成函数（与之前一致，仅返回QA列表）
    # ------------------------------
    def _attribute_appr_counting_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        drop_prop = 0.5
        level3_qa_type = "Attribute(Appr.)_Counting"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]

        # 1. 提取两相机中出现的物体ID及类别
        # cam1中物体：{物体ID: 类别}
        cam1_objs = {obj_id: objects[obj_id]['category'] for obj_id in cam1_data.get('objects', {}).keys()}
        # cam2中物体：{物体ID: 类别}
        cam2_objs = {obj_id: objects[obj_id]['category'] for obj_id in cam2_data.get('objects', {}).keys()}

        # 获取两相机共有的类别（先取类别交集）
        cam1_cats = set(cam1_objs.values())
        cam2_cats = set(cam2_objs.values())
        common_cats = sorted(list(cam1_cats & cam2_cats))

        for cat in common_cats:
            # 获取两相机中该类别的物体ID 一定不是空集
            cam1_ids = set([obj_id for obj_id, c in cam1_objs.items() if c == cat])
            cam2_ids = set([obj_id for obj_id, c in cam2_objs.items() if c == cat])

            # 2. 多步筛选
            # 筛选大小，根据物体2D包围框进行筛选。只要有一个物体被判断过小，就跳过。
            should_skip_category = False
            # 遍历 Cam1 中该类别的所有物体
            for obj_id in cam1_ids:
                # 提取 bbox_2d 数据
                bbox_2d = cam1_data.get('objects', {}).get(obj_id, {}).get('bbox_2d')

                # 使用新函数进行检查
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    should_skip_category = True
                    break

            if should_skip_category:
                continue  # 跳过该类别

            # 遍历 Cam2 中该类别的所有物体
            for obj_id in cam2_ids:
                # 提取 bbox_2d 数据
                bbox_2d = cam2_data.get('objects', {}).get(obj_id, {}).get('bbox_2d')

                # 使用新函数进行检查
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    should_skip_category = True
                    break

            if should_skip_category:
                continue  # 跳过该类别

            # 检查有效性：是否存在不同ID的物体。如果两个相机中出现的物体完全相同，则以0.5的概率丢弃。
            if cam1_ids == cam2_ids and random.random() < drop_prop:
                continue
            # 计算去重总数
            all_ids = cam1_ids | cam2_ids
            unique_count = len(set(all_ids))
            # 一定概率丢弃答案为1的QA
            if unique_count == 1 and random.random() < 0.75:
                continue

            # 3. 完成全部筛选后，生成Level 3 QA
            qa_group = []
            options, answer = generate_random_int_choices(unique_count)

            selected_template = random.choice(QUESTION_TEMPLATES)
            if self.if_MCA:
                selected_template += '\nOptions: {options}'
            else:
                answer = answer[3:]
            question = selected_template.format(cat=cat.lower(), options=options)
            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. 生成与该level3问题能力依赖的level1、2问题
            # 4.1 准备生成level1、2问题所需的元数据

            # 4.2 首先生成level2问题
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                for obj_id in sorted(list(cam1_ids)):
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_id, cat, "1,2"))
                for obj_id in sorted(list(cam2_ids)):
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_id, cat, "2,1"))

            # 4.1 其次生成level1问题
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, cam1_ids, cat, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, cam2_ids, cat, "2"))

            qa.append(qa_group)

        return qa

    def _attribute_appr_orientation_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        drop_prop = 0.8
        level3_qa_type = "Attribute(Appr.)_Orientation"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]

        # 1. 找到在两个相机中都出现的物体作为参考物体（ID交集）
        # 提取两相机中物体的ID集合
        cam1_ids = set(cam1_data['objects'].keys())
        cam2_ids = set(cam2_data['objects'].keys())

        # 求交集：同时出现在两个相机中的物体ID
        common_ids = sorted(list(cam1_ids & cam2_ids))  # 集合的 & 运算符表示交集
        if not common_ids:
            return qa  # 无共同物体时不生成问题

        # 2. 相机位置与朝向
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y']])  # , cam1_data['location_3d']['z']
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y']])  # , cam2_data['location_3d']['z']

        for ref_obj_id in common_ids:
            # 3. 一系列筛选
            # 生成对物体的描述：如果cam1中只存在一个该类物体，直接用类别描述，否则使用该物体在这个类别的物体中的相对方位来描述
            success, ref_obj_cat, ref_obj_des = get_description(ref_obj_id, cam1_data, objects)
            if not success:
                continue
            # 过滤特殊类别
            if ref_obj_cat in ['rug']:
                continue
            # 过滤物体大小
            bbox_2d = cam2_data.get('objects', {}).get(ref_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            bbox_2d = cam1_data.get('objects', {}).get(ref_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue

            # 4. 生成level3问题
            qa_group = []
            # 获取参考物体的坐标和三轴朝向
            ref_obj_loc = np.array(objects[ref_obj_id]['3d_center'][:2])

            main_direction, rel_direction, direct_8 = get_relative_orientation(ref_obj_loc, cam1_loc, cam2_loc)
            # 一定概率丢弃方向相同的问题
            if main_direction == rel_direction and random.random() < drop_prop:
                continue
            # 生成选项
            options, answer = generate_orientation_choices(rel_direction, direct_8)
            # 生成QA
            selected_template = random.choice(QUESTION_TEMPLATES)
            if self.if_MCA:
                selected_template += '\nOptions: {options}'
            else:
                answer = answer[3:]
            question = selected_template.format(main_direction=main_direction, ref_obj_des=ref_obj_des, ref_obj_cat=ref_obj_cat, options=options)
            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 5. 生成与该level3问题能力依赖的level1、2问题
            # 5.1 准备生成level1、2问题所需的元数据

            # 5.2 首先生成level2问题
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des, "2,1"))

            # 5.3 其次生成level1问题
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, ref_obj_id, ref_obj_des, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, ref_obj_id, ref_obj_des, "2"))

            qa.append(qa_group)

        return qa

    def _attribute_meas_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "Attribute(Meas.)"
        min_height = 0.8
        min_volume = 0.125
        disparity_factor = 1.7
        proximity_factor = 1.1
        k_threshold = 4
        eliminate_the_same = 0.85
        prop_ask_for_greater = 0.3
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        TEMPLATES_POSITION_IN_HEIGHT = QUESTION_TEMPLATES["POSITION_IN_HEIGHT"]
        TEMPLATES_HEIGHT = QUESTION_TEMPLATES["HEIGHT"]
        TEMPLATES_LENGTH = QUESTION_TEMPLATES["LENGTH"]

        # 1. 提取三个物体集合
        cam1_objs = set(cam1_data.get('objects', {}).keys())  # 相机1中的物体ID
        cam2_objs = set(cam2_data.get('objects', {}).keys())  # 相机2中的物体ID

        # 只在cam1出现的物体
        only_cam1 = sorted([obj_id for obj_id in cam1_objs if obj_id not in cam2_objs])
        # 只在cam2出现的物体
        only_cam2 = sorted([obj_id for obj_id in cam2_objs if obj_id not in cam1_objs])
        # 同时出现在两个相机的物体（用于过滤，非空才继续）
        both_cams = sorted(list(cam1_objs & cam2_objs))
        # 2. 如果同时出现的物体集合为空，则不生成问题
        if not both_cams:
            return qa

        # 3. 对只在cam1和只在cam2的物体进行两两组合，生成尺寸比较问题
        # 对于每组物体只生成一个问题，但是有很多物体组，最终在外层函数随机选择
        for obj1_id in only_cam1:
            bbox_2d = cam1_data.get('objects', {}).get(obj1_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, obj1_cat, obj1_des = get_description(obj1_id, cam1_data, objects)
            if not success:
                continue
            for obj2_id in only_cam2:
                qa_group = []
                # 决定问法：正确答案是数值更大还是更小的一方
                ask_for_greater = random.random() > prop_ask_for_greater
                bbox_2d = cam2_data.get('objects', {}).get(obj2_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                success, obj2_cat, obj2_des = get_description(obj2_id, cam2_data, objects)
                if not success:
                    continue
                # 获取物体尺寸
                obj1_3d_bbox = objects[obj1_id]['bbox_3d_aabb']
                obj2_3d_bbox = objects[obj2_id]['bbox_3d_aabb']
                obj1_volume = obj1_3d_bbox['dimensions']['x'] * obj1_3d_bbox['dimensions']['y'] * obj1_3d_bbox['dimensions']['z']
                obj2_volume = obj2_3d_bbox['dimensions']['x'] * obj2_3d_bbox['dimensions']['y'] * obj2_3d_bbox['dimensions']['z']
                obj1_pos_z = (obj1_3d_bbox['min']['z'] + obj1_3d_bbox['max']['z']) / 2
                obj2_pos_z = (obj2_3d_bbox['min']['z'] + obj2_3d_bbox['max']['z']) / 2
                obj1_height = obj1_3d_bbox['dimensions']['z']
                obj2_height = obj2_3d_bbox['dimensions']['z']
                obj1_length = max(obj1_3d_bbox['dimensions']['x'], obj1_3d_bbox['dimensions']['y'])
                obj2_length = max(obj2_3d_bbox['dimensions']['x'], obj2_3d_bbox['dimensions']['y'])

                # 决定问题类型
                # 首先判断是否为高位小物体，如果是的话，比较二者位置高度
                if obj1_pos_z > min_height and obj2_pos_z > min_height and obj1_volume < min_volume and obj2_volume < min_volume:
                    # 比较位置高度不涉及倍数
                    dimension_to_compare = 'altitude'
                    # 筛掉过于悬殊的比较
                    if obj1_pos_z / obj2_pos_z > disparity_factor or obj2_pos_z / obj1_pos_z > disparity_factor:
                        continue
                    # 一定概率丢弃the same
                    ratio = max(obj1_pos_z, obj2_pos_z) / min(obj1_pos_z, obj2_pos_z)
                    if ratio <= proximity_factor and random.random() < eliminate_the_same:
                        continue
                    # 选择不同问法的模板
                    templates = TEMPLATES_POSITION_IN_HEIGHT['greater'] if ask_for_greater else TEMPLATES_POSITION_IN_HEIGHT['smaller']
                    options, answer = generate_size_comparison_choices(obj1_3d_bbox['max']['z'], obj2_3d_bbox['max']['z'], obj1_des, obj2_des, 'height', proximity_factor, ask_for_greater)
                    selected_template = random.choice(templates) + '\nOptions: {options}'
                    question = selected_template.format(obj1_des=obj1_des, obj2_des=obj2_des, options=options)
                    qa_group.append((question, answer, level3_qa_type, "1,2"))
                # 否则比较二者尺寸的高度或者宽度（长度与宽度在描述上有歧义，改描述为比较物体的长边或短边）
                else:
                    # 随机选择是比较长度(长边/短边)还是高度
                    # 0.5 的随机选择模式被抽象到这一步：
                    dimension_to_compare = random.choice(['length', 'height'])

                    # 1. 确定本次比较的维度、数值和模板
                    if dimension_to_compare == 'length':
                        # 长度比较相关的变量
                        obj1_val = obj1_length
                        obj2_val = obj2_length
                        # 随机选择 'length' 或 'width' 作为问题描述
                        question_des = random.choice(['length', 'width'])
                        # 对应的模板集合（需要从 TEMPLATES_LENGTH 中获取）
                        template_group = TEMPLATES_LENGTH[question_des]
                    else:  # dimension_to_compare == 'height'
                        # 高度比较相关的变量
                        obj1_val = obj1_height
                        obj2_val = obj2_height
                        question_des = 'height'
                        # 对应的模板集合（需要从 TEMPLATES_HEIGHT 中获取）
                        template_group = TEMPLATES_HEIGHT

                    # 2. 倍数比较（使用抽象的 obj1_val 和 obj2_val）
                    if obj1_val / obj2_val > disparity_factor:
                        k = round(obj1_val / obj2_val)
                        if k > k_threshold:
                            continue
                        # 直接修改 obj2_val 和 obj2_des
                        obj2_val *= k
                        obj2_des_new = f"{k} times the {question_des} of the {obj2_des}"
                        obj1_des_new = f"{question_des} of the {obj1_des}"
                    elif obj2_val / obj1_val > disparity_factor:
                        k = round(obj2_val / obj1_val)
                        if k > k_threshold:
                            continue
                        # 直接修改 obj1_val 和 obj1_des
                        obj1_val *= k
                        obj1_des_new = f"{k} times the {question_des} of the {obj1_des}"
                        obj2_des_new = f"{question_des} of the {obj2_des}"
                    else:
                        obj1_des_new = f"{question_des} of the {obj1_des}"
                        obj2_des_new = f"{question_des} of the {obj2_des}"

                    # 3. 相似度丢弃检查
                    ratio = max(obj1_val, obj2_val) / min(obj1_val, obj2_val)
                    if ratio <= proximity_factor and random.random() < eliminate_the_same:
                        continue

                    # 4. 问法选择与 QA 生成
                    # 选择不同问法的模板
                    templates = template_group['greater'] if ask_for_greater else template_group['smaller']
                    options, answer = generate_size_comparison_choices(obj1_val, obj2_val, obj1_des_new, obj2_des_new, dimension_to_compare, proximity_factor, ask_for_greater)
                    selected_template = random.choice(templates)
                    if self.if_MCA:
                        selected_template += '\nOptions: {options}'
                    else:
                        answer = answer[3:]
                    question = selected_template.format(obj1_des=obj1_des_new, obj2_des=obj2_des_new, options=options)
                    qa_group.append((question, answer, level3_qa_type, "1,2"))

                # 5. 生成与该level3问题能力依赖的level1、2问题
                # 5.1 准备生成level1、2问题所需的元数据
                # 随机选择一个common物体作为参考
                objs_in_both_cams = sorted(list(both_cams))
                random.shuffle(objs_in_both_cams)
                success = False
                for ref_obj_id in objs_in_both_cams:
                    success1, ref_obj_cat, ref_obj_des_1 = get_description(ref_obj_id, cam1_data, objects)
                    success2, ref_obj_cat, ref_obj_des_2 = get_description(ref_obj_id, cam2_data, objects)
                    if success1 and success2:
                        success = True
                        break
                if success == True:
                    # 5.2 首先生成level2问题
                    level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
                    for level2_qa_type in level2_qa_types:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

                    # 5.3 其次生成level1问题
                    level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
                    for level1_qa_type in level1_qa_types:
                        if level1_qa_type == "Measurement_comparison":
                            qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, [ref_obj_id, obj1_id], [ref_obj_des_1, obj1_des], "1", dimension_to_compare))
                            qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, [ref_obj_id, obj2_id], [ref_obj_des_2, obj2_des], "2", dimension_to_compare))
                        else:
                            qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, ref_obj_id, ref_obj_des_1, "1"))
                            qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, ref_obj_id, ref_obj_des_2, "2"))
                            qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, obj1_id, obj1_des, "1"))
                            qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, obj2_id, obj2_des, "2"))

                qa.append(qa_group)

        return qa

    def _motion_cam_translation_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "Motion(Cam.)_Translation"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        translation_threshold = 0.3
        rotation_threshold = 110
        z_threshold = 0.9

        # 1. 相机位置与朝向
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])  #
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])  #
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])
        # 2. 筛选
        # 筛除z轴变化过大的相机组合
        if abs(cam1_loc[-1] - cam2_loc[-1]) > z_threshold:
            return qa
        # 筛掉转动角度过大的相机组合
        rel_angel = calculate_rotation(cam1_for[:2], cam2_for[:2])
        if abs(rel_angel) > rotation_threshold:
            return qa

        # 保证两个相机包含公共物体
        cam1_objs = set(cam1_data['objects'].keys())
        cam2_objs = set(cam2_data['objects'].keys())
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = cam1_objs & cam2_objs  # 集合的 & 运算符表示交集
        if len(common_ids) < self.common_num_threshold:
            return qa  # 无共同物体时不生成问题

        # 3. 生成level 3问题
        qa_group = []

        forward_dir, right_dir = get_relative_direction(cam1_loc, cam2_loc, cam1_for, thresh=translation_threshold)
        # if forward_dir == "" and right_dir == "":
        #     return qa
        # 加入Not Moving选项
        options, answer = generate_direction_choices(forward_dir, right_dir)
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(options=options)
        qa_group.append((question, answer, level3_qa_type, "1,2"))

        # 4. 生成与该level3问题能力依赖的level1、2问题
        # 4.1 准备生成level1、2问题所需的元数据
        # 随机选择一个common物体作为参考
        objs_in_both_cams = sorted(list(common_ids))
        random.shuffle(objs_in_both_cams)
        success = False
        for ref_obj_id in objs_in_both_cams:
            success1, ref_obj_cat, ref_obj_des_1 = get_description(ref_obj_id, cam1_data, objects)
            success2, ref_obj_cat, ref_obj_des_2 = get_description(ref_obj_id, cam2_data, objects)
            if success1 and success2:
                success = True
                break
        if success == True:

            # 4.2 首先生成level2问题
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

            # 4.3 其次生成level1问题
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, ref_obj_id, ref_obj_des_1, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, ref_obj_id, ref_obj_des_2, "2"))

        qa.append(qa_group)

        return qa

    def _motion_cam_rotation_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "Motion(Cam.)_Rotation"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        translation_threshold = 1.5
        rotation_max_threshold = 150
        rotation_min_threshold = 5

        # 1. 相机位置与朝向
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])  #
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])  #
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])

        # 2. 筛选
        # 筛掉位移过大的组合
        translation = np.linalg.norm(cam1_loc - cam2_loc)
        if translation > translation_threshold:
            return qa
        # 筛掉转动角度过大的相机组合
        rel_angel = calculate_rotation(cam1_for[:2], cam2_for[:2])
        if abs(rel_angel) > rotation_max_threshold:
            return qa

        # 保证两个相机包含公共物体
        cam1_objs = set(cam1_data['objects'].keys())
        cam2_objs = set(cam2_data['objects'].keys())
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = cam1_objs & cam2_objs  # 集合的 & 运算符表示交集
        if len(common_ids) < self.common_num_threshold:
            return qa  # 无共同物体时不生成问题

        # 3. 生成level 3问题
        qa_group = []
        # 获取两个相机的旋转矩阵
        R1 = get_camera_rotation_matrix(cam1_data)
        R2 = get_camera_rotation_matrix(cam2_data)
        # 计算从cam1到cam2的相对旋转
        yaw_dir, pitch_dir = get_relative_rotation(R1, R2, rotation_min_threshold, is_scannetpp=(self.data_source=="scannetpp"))

        options, answer = generate_rotation_choices(yaw_dir, pitch_dir)
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(options=options)
        qa_group.append((question, answer, level3_qa_type, "1,2"))

        # 4. 生成与该level3问题能力依赖的level1、2问题
        # 4.1 准备生成level1、2问题所需的元数据
        # 随机选择一个common物体作为参考
        objs_in_both_cams = sorted(list(common_ids))
        random.shuffle(objs_in_both_cams)
        success = False
        for ref_obj_id in objs_in_both_cams:
            success1, ref_obj_cat, ref_obj_des_1 = get_description(ref_obj_id, cam1_data, objects)
            success2, ref_obj_cat, ref_obj_des_2 = get_description(ref_obj_id, cam2_data, objects)
            if success1 and success2:
                success = True
                break
        if success == True:

            # 4.2 首先生成level2问题
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

            # 4.3 其次生成level1问题
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, ref_obj_id, ref_obj_des_1, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, ref_obj_id, ref_obj_des_2, "2"))

        qa.append(qa_group)

        return qa

    def _positional_relationship_cam_cam_translation_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "Positional Relationship(Cam.-Cam.)_Translation"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        modes = ['MODE_1', 'MODE_2']
        coor_settings = ['+Y up, -Z forward', '+Z up, +X forward']
        translation_threshold = 0.3
        rotation_threshold = 110
        z_threshold = 0.9

        # 1. 相机位置与朝向
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])  #
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])  #
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])

        # 2. 视角筛选
        # 筛除z轴变化过大的相机组合
        if abs(cam1_loc[-1] - cam2_loc[-1]) > z_threshold:
            return qa
        # 筛掉转动角度过大的相机组合
        rel_angel = calculate_rotation(cam1_for[:2], cam2_for[:2])
        if abs(rel_angel) > rotation_threshold:
            return qa
        # 保证两个相机包含公共物体
        cam1_objs = set(cam1_data['objects'].keys())
        cam2_objs = set(cam2_data['objects'].keys())
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = cam1_objs & cam2_objs  # 集合的 & 运算符表示交集
        if len(common_ids) < self.common_num_threshold:
            return qa  # 无共同物体时不生成问题

        # 3. 问题生成
        # 定义两种问法情景 (Scenario)
        # Scenario 1: C2 为参考系 (RefCam), C1 为目标 (TargetObj) -> 对应 T1 问法
        scenario_1 = {
            'ref_cam_loc': cam1_loc,
            'ref_cam_for': cam1_for,
            'tar_cam_loc': cam2_loc,
            'is_case_1': True  # 用于模板填充
        }
        # Scenario 2: C1 为参考系 (RefCam), C2 为目标 (TargetObj) -> 对应 T2 问法
        scenario_2 = {
            'ref_cam_loc': cam2_loc,
            'ref_cam_for': cam2_for,
            'tar_cam_loc': cam1_loc,
            'is_case_1': False  # 用于模板填充
        }

        for scenario in [scenario_1, scenario_2]:
            qa_group = []
            # 4. level 3 问题生成
            forward_dir, right_dir = get_relative_direction(scenario['ref_cam_loc'], scenario['tar_cam_loc'], scenario['ref_cam_for'], thresh=translation_threshold)
            if forward_dir == "" and right_dir == "":
                continue

            # for mode in modes:
            mode = random.choice(modes)
            MODE_TEMPLATES = QUESTION_TEMPLATES[mode]
            # MODE_1 代表设定坐标系的问法
            if mode == 'MODE_1':
                # for setting in coor_settings:
                setting = random.choice(coor_settings)
                filled_templates = [
                    fill_template_placeholders(t, scenario['is_case_1'])
                    for t in MODE_TEMPLATES
                ]
                selected_template = random.choice(filled_templates)
                selected_template = selected_template.replace("$DIR_COOR$", setting)
                options, answer = generate_coordinate_direction_choices(forward_dir, right_dir, setting)
                if self.if_MCA:
                    selected_template += '\nOptions: {options}'
                else:
                    answer = answer[3:]
                question = selected_template.format(options=options)
                qa_group.append((question, answer, level3_qa_type, "1,2"))
            # MODE_2 代表传统方向表示的问法
            else:
                options, answer = generate_direction_choices(forward_dir, right_dir, moving=False)
                filled_templates = [
                    fill_template_placeholders(t, scenario['is_case_1'])
                    for t in MODE_TEMPLATES
                ]
                selected_template = random.choice(filled_templates)
                if self.if_MCA:
                    selected_template += '\nOptions: {options}'
                else:
                    answer = answer[3:]
                question = selected_template.format(options=options)
                qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. 生成与该level3问题能力依赖的level1、2问题
            # 4.1 准备生成level1、2问题所需的元数据
            # 随机选择一个common物体作为参考
            objs_in_both_cams = sorted(list(common_ids))
            random.shuffle(objs_in_both_cams)
            success = False
            for ref_obj_id in objs_in_both_cams:
                success1, ref_obj_cat, ref_obj_des_1 = get_description(ref_obj_id, cam1_data, objects)
                success2, ref_obj_cat, ref_obj_des_2 = get_description(ref_obj_id, cam2_data, objects)
                if success1 and success2:
                    success = True
                    break
            if success == True:

                # 4.2 首先生成level2问题
                level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
                for level2_qa_type in level2_qa_types:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                    if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

                # 4.3 其次生成level1问题
                level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
                for level1_qa_type in level1_qa_types:
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, ref_obj_id, ref_obj_des_1, "1"))
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, ref_obj_id, ref_obj_des_2, "2"))
            qa.append(qa_group)

        return qa

    def _positional_relationship_cam_cam_rotation_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "Positional Relationship(Cam.-Cam.)_Rotation"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        modes = ['MODE_1', 'MODE_2']
        coor_settings = ['+Y up, -Z forward', '+Z up, +X forward']
        translation_threshold = 1.5
        rotation_max_threshold = 150
        rotation_min_threshold = 5

        # 1. 相机位置与朝向
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])  #
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])  #
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])

        # 筛掉位移过大的组合
        translation = np.linalg.norm(cam1_loc - cam2_loc)
        if translation > translation_threshold:
            return qa
        # 筛掉转动角度过大的相机组合
        rel_angel = calculate_rotation(cam1_for[:2], cam2_for[:2])
        if abs(rel_angel) > rotation_max_threshold:
            return qa
        # 保证两个相机包含公共物体
        cam1_objs = set(cam1_data['objects'].keys())
        cam2_objs = set(cam2_data['objects'].keys())
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = cam1_objs & cam2_objs  # 集合的 & 运算符表示交集
        if len(common_ids) < self.common_num_threshold:
            return qa  # 无共同物体时不生成问题

        # 获取两个相机的旋转矩阵
        R1 = get_camera_rotation_matrix(cam1_data)
        R2 = get_camera_rotation_matrix(cam2_data)

        # 3. 问题生成
        # 定义两种问法情景 (Scenario)
        # Scenario 1: C2 为参考系 (RefCam), C1 为目标 (TargetObj) -> 对应 T1 问法
        scenario_1 = {
            'ref_cam_R': R1,
            'tar_cam_R': R2,
            'is_case_1': True  # 用于模板填充
        }
        # Scenario 2: C1 为参考系 (RefCam), C2 为目标 (TargetObj) -> 对应 T2 问法
        scenario_2 = {
            'ref_cam_R': R2,
            'tar_cam_R': R1,
            'is_case_1': False  # 用于模板填充
        }
        for scenario in [scenario_1, scenario_2]:
            qa_group = []
            # 4. level 3 问题生成
            # 计算从cam1到cam2的相对旋转
            yaw_dir, pitch_dir = get_relative_rotation(scenario['ref_cam_R'], scenario['tar_cam_R'], rotation_min_threshold, is_scannetpp=(self.data_source=="scannetpp"))
            if yaw_dir == "" and pitch_dir == "":
                continue
            # for mode in modes:
            mode = random.choice(modes)
            MODE_TEMPLATES = QUESTION_TEMPLATES[mode]
            # MODE_1 代表设定坐标系的问法
            if mode == 'MODE_1':
                # for setting in coor_settings:
                setting = random.choice(coor_settings)
                filled_templates = [
                    fill_template_placeholders(t, scenario['is_case_1'])
                    for t in MODE_TEMPLATES
                ]
                selected_template = random.choice(filled_templates)
                selected_template = selected_template.replace("$DIR_COOR$", setting)
                options, answer = generate_coordinate_rotation_choices(yaw_dir, pitch_dir, setting)
                if self.if_MCA:
                    selected_template += '\nOptions: {options}'
                else:
                    answer = answer[3:]
                question = selected_template.format(options=options)
                qa_group.append((question, answer, level3_qa_type, "1,2"))
            # MODE_2 代表传统方向表示的问法
            else:
                options, answer = generate_rotation_choices(yaw_dir, pitch_dir)
                filled_templates = [
                    fill_template_placeholders(t, scenario['is_case_1'])
                    for t in MODE_TEMPLATES
                ]
                selected_template = random.choice(filled_templates)
                if self.if_MCA:
                    selected_template += '\nOptions: {options}'
                else:
                    answer = answer[3:]
                question = selected_template.format(options=options)
                qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. 生成与该level3问题能力依赖的level1、2问题
            # 4.1 准备生成level1、2问题所需的元数据
            # 随机选择一个common物体作为参考
            objs_in_both_cams = sorted(list(common_ids))
            random.shuffle(objs_in_both_cams)
            success = False
            for ref_obj_id in objs_in_both_cams:
                success1, ref_obj_cat, ref_obj_des_1 = get_description(ref_obj_id, cam1_data, objects)
                success2, ref_obj_cat, ref_obj_des_2 = get_description(ref_obj_id, cam2_data, objects)
                if success1 and success2:
                    success = True
                    break
            if success == True:

                # 4.2 首先生成level2问题
                level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
                for level2_qa_type in level2_qa_types:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                    if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

                # 4.3 其次生成level1问题
                level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
                for level1_qa_type in level1_qa_types:
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, ref_obj_id, ref_obj_des_1, "1"))
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, ref_obj_id, ref_obj_des_2, "2"))
            qa.append(qa_group)

        return qa

    def _positional_relationship_cam_obj_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "Positional Relationship(Cam.-Obj.)"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        translation_threshold = 0.3
        # 1. 找到在两个相机中都出现的物体作为参考物体（ID交集）
        # 提取两相机中物体的ID集合
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        only_cam1 = cam1_objs - cam2_objs
        only_cam2 = cam2_objs - cam1_objs
      
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = cam1_objs & cam2_objs  # 集合的 & 运算符表示交集
        if len(common_ids) < self.common_num_threshold:
            return qa  # 无共同物体时不生成问题

        # 定义两种问法情景 (Scenario)
        # Scenario 1: C2 为参考系 (RefCam), C1 为目标 (TargetObj) -> 对应 T1 问法
        scenario_1 = {
            'target_ids': sorted(list(only_cam2)),
            'target_cam_data': cam2_data,  # C1 描述物体
            'ref_cam_data': cam1_data,  # C2 计算方向
            'is_case_1': True  # 用于模板填充
        }
        # Scenario 2: C1 为参考系 (RefCam), C2 为目标 (TargetObj) -> 对应 T2 问法
        scenario_2 = {
            'target_ids': sorted(list(only_cam1)),
            'target_cam_data': cam1_data,  # C2 描述物体
            'ref_cam_data': cam2_data,  # C1 计算方向
            'is_case_1': False  # 用于模板填充
        }

        for scenario in [scenario_1, scenario_2]:
            ref_cam_loc = np.array([scenario['ref_cam_data']['location_3d']['x'], scenario['ref_cam_data']['location_3d']['y'], scenario['ref_cam_data']['location_3d']['z']])
            ref_cam_for = np.array([scenario['ref_cam_data']['forward_direction']['x'], scenario['ref_cam_data']['forward_direction']['y'], scenario['ref_cam_data']['forward_direction']['z']])

            # 1. 预填充模板中的语言指示符
            filled_templates = [
                fill_template_placeholders(t, scenario['is_case_1'])
                for t in QUESTION_TEMPLATES
            ]

            for obj_id in scenario['target_ids']:
                qa_group = []
                bbox_2d = scenario['target_cam_data'].get('objects', {}).get(obj_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue

                # ... (获取 obj_des 和计算方向的逻辑保持不变) ...
                success, _, obj_des = get_description(obj_id, scenario['target_cam_data'], objects)
                if not success: continue

                obj_loc = np.array(objects[obj_id]['3d_center'])
                forward_dir, right_dir = get_relative_direction(ref_cam_loc, obj_loc, ref_cam_for, thresh=translation_threshold)
                if forward_dir == "" and right_dir == "":
                    continue

                # 2. 格式化最终问题
                options, answer = generate_direction_choices(forward_dir, right_dir, moving=False)
                selected_template = random.choice(filled_templates)
                if self.if_MCA:
                    selected_template += '\nOptions: {options}'
                else:
                    answer = answer[3:]
                question = selected_template.format(obj_des=obj_des, options=options)
                qa_group.append((question, answer, level3_qa_type, "1,2"))

                # 4. 生成与该level3问题能力依赖的level1、2问题
                # 4.1 准备生成level1、2问题所需的元数据

                # 4.2 首先生成level2问题
                level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
                for level2_qa_type in level2_qa_types:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_id, obj_des, "1,2"))
                    if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_id, obj_des, "2,1"))

                # 4.3 其次生成level1问题
                level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
                for level1_qa_type in level1_qa_types:
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, scenario['target_cam_data'], obj_id, obj_des, "2" if scenario['is_case_1'] else "1"))

                qa.append(qa_group)

        return qa

    def _positional_relationship_obj_obj_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        max_num = 5
        max_try = 1000
        translation_threshold = 0.3
        level3_qa_type = "Positional Relationship(Obj.-Obj.)"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        # 1. 找到在两个相机中都出现的物体作为参考物体（ID交集）
        # 提取两相机中物体的ID集合
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        # cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])
        # cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = sorted(list(cam1_objs & cam2_objs))  # 集合的 & 运算符表示交集
        if not common_ids:
            return qa  # 无共同物体时不生成问题

        # 2. 找到只在cam1、cam2中出现的物体（不滤除类别）
        only_cam1 = sorted([obj_id for obj_id in cam1_objs if obj_id not in cam2_objs])
        only_cam2 = sorted([obj_id for obj_id in cam2_objs if obj_id not in cam1_objs])
        if len(only_cam1) == 0 or len(only_cam2) == 0:
            return qa

        i = 0
        while len(qa) < max_num and i < max_try:
            i += 1

            only_cam1_obj_id = random.choice(only_cam1)
            bbox_2d = cam1_data.get('objects', {}).get(only_cam1_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, only_cam1_obj_cat, only_cam1_obj_des = get_description(only_cam1_obj_id, cam1_data, objects)
            if not success:
                continue

            only_cam2_obj_id = random.choice(only_cam2)
            bbox_2d = cam2_data.get('objects', {}).get(only_cam2_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, only_cam2_obj_cat, only_cam2_obj_des = get_description(only_cam2_obj_id, cam2_data, objects)
            if not success:
                continue

            common_obj_id = random.choice(common_ids)
            bbox_2d = cam2_data.get('objects', {}).get(common_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, common_obj_cat, common_obj_des = get_description(common_obj_id, cam2_data, objects)
            if not success:
                continue
            objs = [
                {'id': only_cam1_obj_id, 'des': f"{only_cam1_obj_des} in figure 1"},
                {'id': only_cam2_obj_id, 'des': f"{only_cam2_obj_des} in figure 2"},
                {'id': common_obj_id, 'des': f"{common_obj_des} in figure 2"},
            ]
            random.shuffle(objs)
            # 提问：假设objs[0]在objs[1]的什么方向，求objs[2]在objs[1]的什么方向
            obj0_loc = np.array(objects[objs[0]['id']]['3d_center'][:2])
            obj1_loc = np.array(objects[objs[1]['id']]['3d_center'][:2])
            obj2_loc = np.array(objects[objs[2]['id']]['3d_center'][:2])
            # 筛掉物体之间距离过小的组合
            translation = np.linalg.norm(obj1_loc - obj0_loc)
            if translation < translation_threshold:
                continue
            translation = np.linalg.norm(obj2_loc - obj0_loc)
            if translation < translation_threshold:
                continue

            qa_group = []
            main_direction, rel_direction, direct_8 = get_relative_orientation(obj0_loc, obj1_loc, obj2_loc)
            # 生成选项
            options, answer = generate_orientation_choices(rel_direction, direct_8)
            # 生成QA
            selected_template = random.choice(QUESTION_TEMPLATES)
            if self.if_MCA:
                selected_template += '\nOptions: {options}'
            else:
                answer = answer[3:]
            question = selected_template.format(obj1_des=objs[1]['des'], obj0_des=objs[0]['des'], obj2_des=objs[2]['des'], main_direction=main_direction, options=options)
            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. 生成与该level3问题能力依赖的level1、2问题
            # 4.1 准备生成level1、2问题所需的元数据

            # 4.2 首先生成level2问题
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, common_obj_id, common_obj_des, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, common_obj_id, common_obj_des, "2,1"))

            # 4.3 其次生成level1问题
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, only_cam1_obj_id, only_cam1_obj_des, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, only_cam2_obj_id, only_cam2_obj_des, "2"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, common_obj_id, common_obj_des, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, common_obj_id, common_obj_des, "2"))

            qa.append(qa_group)

        return qa

    def _positional_relationship_obj_obj_orientation_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        # 初始化超参
        qa = []
        max_num = 5
        max_try = 1000
        translation_threshold = 0.3
        level3_qa_type = "Positional Relationship(Obj.-Obj.)_Orientation"
        TEMPLATES = self.templates[level3_qa_type]

        # 1. 对场景中物体，划分不同的集合
        # 找到场景中是否包含带有语义朝向的物体
        objs_with_orientation = {obj: objects[obj] for obj in objects if objects[obj]['category'] in list(self.obj_cat_with_orientation.keys())}
        if not objs_with_orientation:
            return qa
        # 提取两相机中物体的ID集合
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = sorted(list(cam1_objs & cam2_objs))  # 集合的 & 运算符表示交集
        if not common_ids:
            return qa  # 无共同物体时不生成问题
        # 找到只在cam1、cam2中出现的物体（不滤除类别）
        only_cam1 = sorted([obj_id for obj_id in cam1_objs if obj_id not in cam2_objs])
        only_cam2 = sorted([obj_id for obj_id in cam2_objs if obj_id not in cam1_objs])
        if len(only_cam1) == 0 or len(only_cam2) == 0:
            return qa
        # 找到所有出现的物体的{类别：id}集合
        all_obj_cat = [(obj, objects[obj]['category']) for obj in objects]
        category_counts = Counter(category for obj_id, category in all_obj_cat)

        # 2. 生成QA
        i = 0
        while len(qa) < max_num and i < max_try:
            i += 1
            # 2.1 随机选择一个带有语义朝向的物体A
            obj_with_orientation = random.choice(list(objs_with_orientation.keys()))
            obj_with_orientation_cat = objects[obj_with_orientation]['category']
            if obj_with_orientation_cat == 'window' and random.random() < 0.7:
                continue
            obj_with_orientation_loc = np.array(objects[obj_with_orientation]['3d_center'][:2])
            if self.obj_cat_with_orientation[obj_with_orientation_cat] == '+Y':
                obj_with_orientation_for = np.array(objects[obj_with_orientation]['axis_directions']['local_y'][:2])
            elif self.obj_cat_with_orientation[obj_with_orientation_cat] == '+X':
                obj_with_orientation_for = np.array(objects[obj_with_orientation]['axis_directions']['local_x'][:2])
            elif self.obj_cat_with_orientation[obj_with_orientation_cat] == '-X':
                obj_with_orientation_for = -np.array(objects[obj_with_orientation]['axis_directions']['local_x'][:2])
            # 获取物体A的描述：如果在场景中类别唯一，则使用类别名称进行描述，否则按顺序在cam1与cam2中找到描述
            # 以下部分可以整理为函数，输入较多，但输出为场景中任意一个物体的描述，描述方法见上
            if category_counts[obj_with_orientation_cat] == 1:
                obj_with_orientation_des = obj_with_orientation_cat
                current_des = obj_with_orientation_cat
                cam_idx = 1 if obj_with_orientation in cam1_objs else 2
            else:
                # 确定物体出现在哪些相机中
                is_in_cam1 = obj_with_orientation in cam1_objs
                is_in_cam2 = obj_with_orientation in cam2_objs
                # 确定要处理的相机列表，并随机打乱优先级
                cameras_to_try = []
                if is_in_cam1:
                    cameras_to_try.append(1)
                if is_in_cam2:
                    cameras_to_try.append(2)
                if not cameras_to_try:
                    # 如果物体在两个筛选后的集合中都不存在，理论上不应发生，但作为保护
                    continue
                # 随机打乱尝试顺序
                random.shuffle(cameras_to_try)
                success = False
                # 遍历随机后的相机顺序
                for cam_idx in cameras_to_try:
                    if cam_idx == 1:
                        cam_data = cam1_data
                        fig_suffix = "in figure 1"
                    else:  # cam_idx == 2
                        cam_data = cam2_data
                        fig_suffix = "in figure 2"
                    # 尝试从当前相机生成描述
                    bbox_2d = cam_data.get('objects', {}).get(obj_with_orientation, {}).get('bbox_2d')
                    # 假设 get_description 函数返回 (success, _, description)
                    current_success, _, current_des = get_description(obj_with_orientation, cam_data, objects)
                    # 检查是否成功且不至于太小
                    is_too_small = is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD)
                    if current_success and not is_too_small:
                        # 成功生成描述且物体尺寸合适，确定使用这个描述
                        obj_with_orientation_des = current_des + " " + fig_suffix
                        success = True
                        break  # 找到第一个满足条件的就跳出循环
                # 如果随机尝试两个相机都失败了，则跳过本次循环
                if not success:
                    continue
            # 随机选择物体A正面朝向的文本描述：大概率选择front，否则在八方向中随机选择
            orientation_des = 'front' if random.random() < 0.4 else random.choice(self.direction_name)
            # 生成前提的文本描述
            if obj_with_orientation_cat in TEMPLATES['premise']:
                PREMISE_TEMPLATE = TEMPLATES['premise'][obj_with_orientation_cat]
            else:
                PREMISE_TEMPLATE = TEMPLATES['premise']['others']
            selected_template = random.choice(PREMISE_TEMPLATE)
            premise_text = selected_template.format(obj_des=obj_with_orientation_des, orientation_des=orientation_des)

            # 2.2 随机寻找问题中的其他物体（这里为了命中率应该也使用循环的）
            # 两种模式
            if random.random() < 0.5:
                # 第一种是问物体B在物体A的什么方向
                # 选取方式为如果物体A在两个相机中都出现了，那么就在only_cam1和only_cam2中随机选择，否则就在物体A没出现的那个相机中选择
                if obj_with_orientation in common_ids:
                    obj_B_cam = random.choice(['1', '2'])
                elif obj_with_orientation in cam1_objs:
                    obj_B_cam = '2'
                else:
                    obj_B_cam = '1'
                if obj_B_cam == '1':
                    obj_B = random.choice(only_cam1)
                    if obj_B == obj_with_orientation: continue
                    bbox_2d = cam1_data.get('objects', {}).get(obj_B, {}).get('bbox_2d')
                    if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                        continue
                    success, _, obj_B_des = get_description(obj_B, cam1_data, objects)
                    if not success:
                        continue
                    obj_B_des += ' in figure 1'
                else:
                    obj_B = random.choice(only_cam2)
                    if obj_B == obj_with_orientation: continue
                    bbox_2d = cam2_data.get('objects', {}).get(obj_B, {}).get('bbox_2d')
                    if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                        continue
                    success, _, obj_B_des = get_description(obj_B, cam2_data, objects)
                    if not success:
                        continue
                    obj_B_des += ' in figure 2'
                # 查询向量为obj_B_loc - obj_A_loc 水平xy平面二维向量
                obj_tar_des = obj_B_des
                obj_ref_des = obj_with_orientation_des
                obj_B_loc = np.array(objects[obj_B]['3d_center'][:2])
                query_vec = obj_B_loc - obj_with_orientation_loc
                mode = '1'
            else:
                # 第二种是问物体B在物体C的什么方向
                # 选取方式为在only_cam1和only_cam2中各选一个然后随机分配顺序
                obj_in_cam1 = random.choice(only_cam1)
                if obj_in_cam1 == obj_with_orientation: continue
                bbox_2d = cam1_data.get('objects', {}).get(obj_in_cam1, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                success, _, obj_in_cam1_des = get_description(obj_in_cam1, cam1_data, objects)
                if not success:
                    continue
                obj_in_cam2 = random.choice(only_cam2)
                if obj_in_cam1 == obj_with_orientation: continue
                bbox_2d = cam2_data.get('objects', {}).get(obj_in_cam2, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                success, _, obj_in_cam2_des = get_description(obj_in_cam2, cam2_data, objects)
                if not success:
                    continue
                objs_candidate = [
                    {'id': obj_in_cam1, 'des': obj_in_cam1_des + ' in figure 1', 'loc': np.array(objects[obj_in_cam1]['3d_center'][:2])},
                    {'id': obj_in_cam2, 'des': obj_in_cam2_des + ' in figure 2', 'loc': np.array(objects[obj_in_cam2]['3d_center'][:2])}
                ]
                random.shuffle(objs_candidate)
                obj_tar_des = objs_candidate[0]['des']
                obj_ref_des = objs_candidate[1]['des']
                query_vec = objs_candidate[0]['loc'] - objs_candidate[1]['loc']
                mode = '2'

            # 重构答案生成逻辑：函数功能为根据一个已经绑定了方向文本描述的向量方向，求另一个向量方向的文本描述
            qa_group = []
            answer_direction, direct_geo = get_relative_orientation_vec(obj_with_orientation_for, orientation_des, query_vec)
            options, answer = generate_orientation_choices(answer_direction, direct_8=True, direct_geo=direct_geo)

            QUESTION_TEMPLATE = TEMPLATES['question']
            selected_template = random.choice(QUESTION_TEMPLATE)
            question = selected_template.format(obj_tar_des=obj_tar_des, obj_ref_des=obj_ref_des)
            full_question_text = f"{premise_text}, {question}"
            if self.if_MCA:
                full_question_text += f'\nOptions: {options}'
            else:
                answer = answer[3:]

            qa_group.append((full_question_text, answer, level3_qa_type, "1,2"))

            # 4. 生成与该level3问题能力依赖的level1、2问题
            # 4.1 准备生成level1、2问题所需的元数据

            # 4.2 首先生成level2问题
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                if not (level2_qa_type == "Object_correspondence" and obj_with_orientation not in common_ids):
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_with_orientation, current_des, "1,2"))
                    if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_with_orientation, current_des, "2,1"))

            # 4.3 其次生成level1问题
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                if cam_idx == 1:
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, obj_with_orientation, current_des, "1"))
                else:
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, obj_with_orientation, current_des, "2"))
                if mode == "1":
                    if obj_B_cam == '1':
                        qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, obj_B, obj_B_des, "1"))
                    else:
                        qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, obj_B, obj_B_des, "2"))
                else:
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, obj_in_cam1, obj_in_cam1_des, "1"))
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, obj_in_cam2, obj_in_cam2_des, "2"))

            qa.append(qa_group)

        return qa

    def _positional_relationship_obj_obj_obj_in_options_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        max_num = 5
        max_try = 1000
        translation_threshold = 0.3
        level3_qa_type = "Positional Relationship(Obj.-Obj.)_Obj_in_options"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        # 1. 找到在两个相机中都出现的物体作为参考物体（ID交集）
        # 提取两相机中物体的ID集合
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = sorted(list(cam1_objs & cam2_objs))  # 集合的 & 运算符表示交集
        if not common_ids:
            return qa  # 无共同物体时不生成问题

        # 2. 找到只在cam1、cam2中出现的物体（不滤除类别）
        only_cam1 = sorted([obj_id for obj_id in cam1_objs if obj_id not in cam2_objs])
        only_cam2 = sorted([obj_id for obj_id in cam2_objs if obj_id not in cam1_objs])
        if len(only_cam1) == 0 or len(only_cam2) == 0:
            return qa

        i = 0
        all_objs = list(objects.keys())
        while len(qa) < max_num and i < max_try:
            i += 1

            # 选择参考物体A
            ref_obj_A_id = random.choice(common_ids)
            bbox_2d = cam2_data.get('objects', {}).get(ref_obj_A_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, common_obj_cat, common_obj_des = get_description(ref_obj_A_id, cam2_data, objects)
            if not success:
                continue
            ref_obj_A_loc = np.array(objects[ref_obj_A_id]['3d_center'][:2])
            ref_obj_A_des = f"{common_obj_des} in figure 2"

            # 选择参考物体B
            only_cam1_obj_id = random.choice(only_cam1)
            bbox_2d = cam1_data.get('objects', {}).get(only_cam1_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, only_cam1_obj_cat, only_cam1_obj_des = get_description(only_cam1_obj_id, cam1_data, objects)
            if not success:
                continue

            only_cam2_obj_id = random.choice(only_cam2)
            bbox_2d = cam2_data.get('objects', {}).get(only_cam2_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, only_cam2_obj_cat, only_cam2_obj_des = get_description(only_cam2_obj_id, cam2_data, objects)
            if not success:
                continue

            objs = [
                {'id': only_cam1_obj_id, 'des': f"{only_cam1_obj_des} in figure 1"},
                {'id': only_cam2_obj_id, 'des': f"{only_cam2_obj_des} in figure 2"},
            ]
            ref_obj_B = random.choice(objs)
            ref_obj_B_id = ref_obj_B['id']
            ref_obj_B_des = ref_obj_B['des']
            ref_obj_B_loc = np.array(objects[ref_obj_B_id]['3d_center'][:2])
            # ref_obj_B不能离ref_obj_A太近
            translation = np.linalg.norm(ref_obj_A_loc - ref_obj_B_loc)
            if translation < translation_threshold:
                continue

            # 选择参考物体B相对于参考物体A的方位
            dir_B_to_A = random.choice(self.direction_name)
            # 选择提问方向
            dir_query = random.choice([d for d in self.direction_name if d != dir_B_to_A])

            # 遍历整个场景中的物体 找到一个正确选项和三个错误选项
            random.shuffle(all_objs)
            correct_option = []
            wrong_option = []
            options_obj = []
            for query_obj in all_objs:
                if query_obj == ref_obj_B['id'] or query_obj == ref_obj_A_id:
                    continue
                # query_obj不能离ref_obj_A太近
                query_obj_loc = np.array(objects[query_obj]['3d_center'][:2])
                translation = np.linalg.norm(ref_obj_A_loc - query_obj_loc)
                if translation < translation_threshold:
                    continue
                # 获取query_obj的描述
                if query_obj in cam1_objs:
                    bbox_2d = cam1_data.get('objects', {}).get(query_obj, {}).get('bbox_2d')
                    if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                        continue
                    success, query_obj_cat, query_obj_des = get_description(query_obj, cam1_data, objects)
                    if not success:
                        continue
                    query_obj_des = f"{query_obj_des} in figure 1"
                    query_obj_info = (query_obj, query_obj_des, "1")
                else:
                    bbox_2d = cam2_data.get('objects', {}).get(query_obj, {}).get('bbox_2d')
                    if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                        continue
                    success, query_obj_cat, query_obj_des = get_description(query_obj, cam2_data, objects)
                    if not success:
                        continue
                    query_obj_des = f"{query_obj_des} in figure 2"
                    query_obj_info = (query_obj, query_obj_des, "2")
                # 判断该物体是否在参考物体A的提问方向上
                if is_in_direction(ref_obj_A_loc, ref_obj_B_loc, query_obj_loc, dir_B_to_A, dir_query):
                    correct_option.append(query_obj_info)
                else:
                    wrong_option.append(query_obj_info)
                if len(correct_option) > 0 and len(wrong_option) > 2:
                    break
            if len(correct_option) < 1 or len(wrong_option) < 3:
                continue

            qa_group= []
            correct_option = random.sample(correct_option, 1)
            wrong_option = random.sample(wrong_option, 3)
            options_obj = correct_option + wrong_option
            # 生成选项
            options, answer = generate_shuffled_choices_text([info[1] for info in correct_option], [info[1] for info in wrong_option])
            # 生成QA
            selected_template = random.choice(QUESTION_TEMPLATES) + '\nOptions: {options}'
            question = selected_template.format(ref_obj_A_des=ref_obj_A_des, ref_obj_B_des=ref_obj_B_des, main_direction=dir_B_to_A, query_direction=dir_query, options=options)

            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. 生成与该level3问题能力依赖的level1、2问题
            # 4.1 准备生成level1、2问题所需的元数据

            # 4.2 首先生成level2问题
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_A_id, common_obj_des, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_A_id, common_obj_des, "2,1"))

            # 4.3 其次生成level1问题
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                if ref_obj_B_id in only_cam1:
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, ref_obj_B_id, only_cam1_obj_des, "1"))
                else:
                    qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, ref_obj_B_id, only_cam2_obj_des, "2"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, ref_obj_A_id, common_obj_des, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, ref_obj_A_id, common_obj_des, "2"))
                for query_obj in options_obj:
                    if query_obj[2] == "1":
                        qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, query_obj[0], query_obj[1], "1"))
                    else:
                        qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, query_obj[0], query_obj[1], "2"))

            qa.append(qa_group)

        return qa

    def _positional_relationship_obj_reg_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "Positional Relationship(Obj.-Reg.)"
        max_num = 5
        max_try = 1000
        translation_threshold = 0.5

        # 筛选物体级别Reg：搜索场景中是否有能代表Reg的物体，如果有，有几个？1个就直接转化为Reg描述，否则看是不是挨在一起，是的话也可以转化为Reg描述，并平均坐标；如果分开就跳过
        areas = {}
        area_cats = list(self.obj_area_map.keys())
        for obj in objects:
            obj_cat = objects[obj]['category']
            if obj_cat in area_cats:
                if obj_cat not in areas:
                    areas[obj_cat] = [objects[obj]]
                else:
                    areas[obj_cat].append(objects[obj])
        for cat in list(areas.keys()):
            if len(areas[cat]) == 1:
                areas[cat] = areas[cat][0]
            else:
                # 检测列表中的物体是不是挨着的，如果能够聚成一族，那么可以算作一个area，并综合它们的坐标和包围框，否则丢弃
                # 简化检测方法，直接丢弃
                del areas[cat]
        # 筛选房间级别Reg：直接使用输入的rooms

        # 1. 找到在两个相机中都出现的物体作为参考物体（ID交集）
        # 提取两相机中物体的ID集合
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        # 求交集：同时出现在两个相机中的物体ID
        common_ids = list(cam1_objs & cam2_objs)  # 集合的 & 运算符表示交集
        if not common_ids:
            return qa  # 无共同物体时不生成问题

        # 2. 找到只在cam1、cam2中出现的物体（不滤除类别）
        only_cam1 = sorted([obj_id for obj_id in cam1_objs if obj_id not in cam2_objs])
        only_cam2 = sorted([obj_id for obj_id in cam2_objs if obj_id not in cam1_objs])
        if len(only_cam1) == 0 or len(only_cam2) == 0:
            return qa

        i = 0
        while len(qa) < max_num and i < max_try:
            i += 1

            only_cam1_obj_id = random.choice(only_cam1)
            bbox_2d = cam1_data.get('objects', {}).get(only_cam1_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, only_cam1_obj_cat, only_cam1_obj_des = get_description(only_cam1_obj_id, cam1_data, objects)
            if not success:
                continue

            only_cam2_obj_id = random.choice(only_cam2)
            bbox_2d = cam2_data.get('objects', {}).get(only_cam2_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, only_cam2_obj_cat, only_cam2_obj_des = get_description(only_cam2_obj_id, cam2_data, objects)
            if not success:
                continue

            # common_obj_id = random.choice(common_ids)
            # bbox_2d = cam2_data.get('objects', {}).get(common_obj_id, {}).get('bbox_2d')
            # if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
            #     continue
            # success, common_obj_cat, common_obj_des = get_description(common_obj_id, cam2_data, objects)
            # if not success:
            #     continue

            # 随机选择对room提问和对area提问两种方式
            if random.random() < 0.35 and areas:
                # 对area提问：与Pos-Obj-Obj类似，把area当成common_obj，其他不变
                QUESTION_TEMPLATES = self.templates[level3_qa_type]['area']
                area_cat = random.choice(sorted(list(areas.keys())))
                objs = [
                    {'id': only_cam1_obj_id, 'des': f"{only_cam1_obj_des} in figure 1", 'loc': np.array(objects[only_cam1_obj_id]['3d_center'][:2])},
                    {'id': only_cam2_obj_id, 'des': f"{only_cam2_obj_des} in figure 2", 'loc': np.array(objects[only_cam2_obj_id]['3d_center'][:2])},
                    {'id': "1", 'des': random.choice(self.obj_area_map[area_cat]), 'loc': np.array(areas[area_cat]['3d_center'][:2])},
                ]
                random.shuffle(objs)
                # 提问：假设objs[1]在objs[0]的什么方向，求objs[2]在objs[0]的什么方向
                obj0_loc = objs[0]['loc']  # np.array(objects[objs[0]['id']]['3d_center'][:2])
                obj1_loc = objs[1]['loc']  # np.array(objects[objs[1]['id']]['3d_center'][:2])
                obj2_loc = objs[2]['loc']  # np.array(objects[objs[2]['id']]['3d_center'][:2])
                # 筛掉物体之间距离过小的组合
                translation = np.linalg.norm(obj1_loc - obj0_loc)
                if translation < translation_threshold:
                    continue
                translation = np.linalg.norm(obj2_loc - obj0_loc)
                if translation < translation_threshold:
                    continue
            else:
                # 对room提问
                if not rooms:
                    continue
                QUESTION_TEMPLATES = self.templates[level3_qa_type]['room']
                room_id = random.choice(sorted(list(rooms.keys())))
                objs = [
                    {'id': room_id, 'des': rooms[room_id]['category'], 'loc': np.array(rooms[room_id]['3d_center'][:2])},
                    {'id': only_cam1_obj_id, 'des': f"{only_cam1_obj_des} in figure 1", 'loc': np.array(objects[only_cam1_obj_id]['3d_center'][:2])},
                    {'id': only_cam2_obj_id, 'des': f"{only_cam2_obj_des} in figure 2", 'loc': np.array(objects[only_cam2_obj_id]['3d_center'][:2])},
                ]
                # 提问：假设objs[1]在objs[0]的什么方向，求objs[2]在objs[0]的什么方向
                obj0_loc = objs[0]['loc']  # np.array(objects[objs[0]['id']]['3d_center'][:2])
                obj1_loc = objs[1]['loc']  # np.array(objects[objs[1]['id']]['3d_center'][:2])
                obj2_loc = objs[2]['loc']  # np.array(objects[objs[2]['id']]['3d_center'][:2])
                # 筛掉物体之间距离过小的组合
                translation = np.linalg.norm(obj1_loc - obj0_loc)
                if translation < translation_threshold + 0.6:
                    continue
                translation = np.linalg.norm(obj2_loc - obj0_loc)
                if translation < translation_threshold + 0.6:
                    continue
                # 判断物体是否在房间内部
                if not is_object_center_in_room(only_cam1_obj_id, room_id, objects, rooms) or not is_object_center_in_room(only_cam2_obj_id, room_id, objects, rooms):
                    continue

            qa_group = []
            main_direction, rel_direction, direct_8 = get_relative_orientation(obj0_loc, obj1_loc, obj2_loc)
            # 生成选项
            options, answer = generate_orientation_choices(rel_direction, direct_8)
            # 生成QA
            selected_template = random.choice(QUESTION_TEMPLATES)
            if self.if_MCA:
                selected_template += '\nOptions: {options}'
            else:
                answer = answer[3:]
            question = selected_template.format(obj1_des=objs[1]['des'], obj0_des=objs[0]['des'], obj2_des=objs[2]['des'], main_direction=main_direction, options=options)
            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. 生成与该level3问题能力依赖的level1、2问题
            # 4.1 准备生成level1、2问题所需的元数据

            # 4.2 首先生成level2问题
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, only_cam1_obj_id, only_cam1_obj_des, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, only_cam1_obj_id, only_cam1_obj_des, "2,1"))

            # 4.3 其次生成level1问题
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, only_cam1_obj_id, only_cam1_obj_des, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, only_cam2_obj_id, only_cam2_obj_des, "2"))

            qa.append(qa_group)
        return qa

    def _msr_cam_qa(self, cameras: List, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "MSR_Cam"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        translation_threshold = 1  # 位移阈值，用于确保相机之间有足够的位移
        rotation_max_threshold = 120

        # 确保至少有三个相机数据用于 T(arget), C(urrent), A(nchor)
        if len(cameras) < 3:
            return qa

        chosen_indices = random.sample(range(len(cameras)), 3)

        # 2. 随机分配角色：T (Target), C (Current), A (Anchor)
        #    我们使用随机选择的索引，并映射回原始 cameras 列表
        idx_target = chosen_indices[0]  # 作为 Photo T 的索引
        idx_current = chosen_indices[1]  # 作为 Photo C 的索引
        idx_anchor = chosen_indices[2]  # 作为 Photo A 的索引

        # 3. 提取对应相机数据 (使用原始列表的索引)
        cam_target_data = cameras[idx_target]
        cam_current_data = cameras[idx_current]
        cam_anchor_data = cameras[idx_anchor]

        # 注意：我们应该使用用户可见的相机编号 (通常是 1-based index)
        cam_target_num = idx_target + 1
        cam_current_num = idx_current + 1
        cam_anchor_num = idx_anchor + 1

        # 4. 提取位置和朝向
        cam_target_loc = np.array([cam_target_data['location_3d']['x'], cam_target_data['location_3d']['y']])
        cam_current_loc = np.array([cam_current_data['location_3d']['x'], cam_current_data['location_3d']['y']])
        cam_target_for = np.array([cam_target_data['forward_direction']['x'], cam_target_data['forward_direction']['y']])
        cam_current_for = np.array([cam_current_data['forward_direction']['x'], cam_current_data['forward_direction']['y']])
        cam_anchor_for = np.array([cam_anchor_data['forward_direction']['x'], cam_anchor_data['forward_direction']['y']])

        # 筛掉位移过小的组合
        translation = np.linalg.norm(cam_target_loc - cam_current_loc)
        if translation < translation_threshold:
            return qa
        # 筛掉转动角度过大的相机组合
        rel_angel = calculate_rotation(cam_target_for, cam_current_for)
        if abs(rel_angel) > rotation_max_threshold:
            return qa

        # 5. 计算相对位移向量（固定cam_anchor_for为随机绝对方向，计算cam_target_loc相对于cam_current_loc的绝对方向）
        main_direction, rel_direction = get_absolute_direction(cam_current_loc, cam_target_loc, cam_anchor_for, thresh=translation_threshold)

        options, answer = generate_orientation_choices(rel_direction)

        # 6. 生成问题
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(
            # 使用随机的、1-based 的相机编号
            cam_anchor_idx=cam_anchor_num,  # Photo A
            anchor_dir=main_direction,  # 例如: East
            cam_current_idx=cam_current_num,  # Photo C
            cam_target_idx=cam_target_num,  # Photo T
            options=options
        )

        qa_group = []
        qa_group.append((question, answer, level3_qa_type, "1,2"))
        qa.append(qa_group)

        return qa_group

    def _msr_cam_obj_qa(self, cameras: List, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        """
        随机选择一个相机作为参考视角，并随机寻找一个存在与其他视角但不存在于参考视角的物体。
        提问：物体相对于参考视角的方位。
        与Pos-Cam-Obj类似
        """
        qa = []
        level3_qa_type = "MSR_Cam_Obj"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        translation_threshold = 0.15
        distance_threshold = 0.5
        # 获取参考视角
        ref_cam_index = random.randrange(len(cameras))
        ref_cam_num = ref_cam_index + 1
        ref_cam_data = cameras[ref_cam_index]
        ref_cam_loc = np.array([ref_cam_data['location_3d']['x'], ref_cam_data['location_3d']['y'], ref_cam_data['location_3d']['z']])
        ref_cam_for = np.array([ref_cam_data['forward_direction']['x'], ref_cam_data['forward_direction']['y'], ref_cam_data['forward_direction']['z']])
        ref_objs_set = set(ref_cam_data['objects'].keys())
        ref_objs_cat = {objects[obj]['category'] for obj in ref_objs_set}
        # 获取不在参考视角中的物体id集合
        other_cams = [cam for cam in cameras if cam != ref_cam_data]
        seen_ids = set()
        other_objs_id = []
        for other_cam in other_cams:
            for obj_id in other_cam['objects']:
                # 1. 确保 ID 尚未处理 (避免重复处理，提高效率)
                if obj_id in seen_ids:
                    continue
                # 2. 排除参考视角中的物体
                if obj_id in ref_objs_set:
                    continue
                # 3. 排除不需要的类别
                category = objects[obj_id]['category']
                # 4. 筛选尺寸
                bbox_2d = other_cam.get('objects', {}).get(obj_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                # 5. 通过所有检查，记录 ID 和类别，并标记为已处理
                other_objs_id.append((obj_id, category))
                seen_ids.add(obj_id)

        # 筛选出类别只出现一次的物体: 这里应该算上参考视角一起判断，不然会出现一个物体的类别在参考视角外出现一次，但在参考视角内出现很多次
        # 所以应该加一步：所选择的物体的类别不能出现在参考视角中物体类别里
        category_counts = Counter(category for obj_id, category in other_objs_id if category not in ref_objs_cat)
        unique_categories = {category for category, count in category_counts.items() if count == 1}
        single_occurrence_objects: List[Tuple[str, str]] = [
            (obj_id, category)
            for obj_id, category in other_objs_id
            if category in unique_categories
        ]
        if not single_occurrence_objects:
            return qa

        # for other_obj_id, category in single_occurrence_objects:
        other_obj_id, category = random.choice(single_occurrence_objects)
        obj_des = category
        obj_loc = np.array(objects[other_obj_id]['3d_center'])
        # ic(ref_objs_set, other_objs_id, unique_categories, single_occurrence_objects, other_obj_id, category)

        # 筛掉位移过小的组合
        translation = np.linalg.norm(ref_cam_loc - obj_loc)
        if translation < distance_threshold:
            return qa

        forward_dir, right_dir = get_relative_direction(ref_cam_loc, obj_loc, ref_cam_for, thresh=translation_threshold)
        if forward_dir == "" and right_dir == "":
            return qa

        # 2. 格式化最终问题
        options, answer = generate_direction_choices(forward_dir, right_dir, moving=False)
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(ref_cam_num=int_to_simple_ordinal_word(ref_cam_num), obj_des=obj_des, options=options)
        qa_group = []
        qa_group.append((question, answer, level3_qa_type, "1,2"))
        qa.append(qa_group)
        return qa_group

    def _msr_counting_qa(self, cameras: List, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        level3_qa_type = "MSR_Counting"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]

        # 1. 统计所有出现在这些视角中的物体id集合
        obj_category = []
        for cam in cameras:
            obj_category += [(obj_id, objects[obj_id]['category']) for obj_id in cam.get('objects', {}).keys()]
        unique_objects = list(set(obj_category))
        unique_categories = [category for obj_id, category in unique_objects]
        category_counts = Counter(unique_categories)
        category = random.choice(sorted(list(category_counts.keys())))
        unique_count = int(category_counts[category])
        # 随机丢弃答案为1的问题
        if random.random() < 0.65 and unique_count == 1:
            return qa

        options, answer = generate_random_int_choices(unique_count)

        # 生成QA
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(cat=category.lower(), options=options)
        qa_group = []
        qa_group.append((question, answer, level3_qa_type, "1,2"))
        qa.append(qa_group)

        return qa_group

    def _msr_obj_obj_qa(self, cameras: List, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        max_num = 5
        max_try = 1000
        translation_threshold = 0.3
        level3_qa_type = "MSR_Obj_Obj"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]

        # 1. 找到在所有相机出现过物体的集合中，同类别唯一的物体集合。并且在每个相机的物体集合中去掉这些物体。
        seen_ids = set()
        all_objs_id = []
        for cam in cameras:
            for obj_id in cam['objects']:
                # 1. 确保 ID 尚未处理 (避免重复处理，提高效率)
                if obj_id in seen_ids:
                    continue
                # 2. 排除不需要的类别
                category = objects[obj_id]['category']
                # 3. 筛选尺寸
                bbox_2d = cam.get('objects', {}).get(obj_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                # 4. 通过所有检查，记录 ID 和类别，并标记为已处理
                all_objs_id.append((obj_id, category))
                seen_ids.add(obj_id)

        # 筛选出类别只出现一次的物体: 这里应该算上参考视角一起判断，不然会出现一个物体的类别在参考视角外出现一次，但在参考视角内出现很多次
        # 所以应该加一步：所选择的物体的类别不能出现在参考视角中物体类别里
        category_counts = Counter(category for obj_id, category in all_objs_id)
        unique_categories = {category for category, count in category_counts.items() if count == 1}
        single_occurrence_objects: List[Tuple[str, str]] = [
            (obj_id, category)
            for obj_id, category in all_objs_id
            if category in unique_categories
        ]
        # if not single_occurrence_objects:
        #     return qa
        excluded_ids = {obj_id for obj_id, category in single_occurrence_objects}

        # 2. 随机采样三个物体。在采样每个物体时都随机使用2种方式：一种是从同类别唯一物体集合中采样，物体描述直接使用类别；另一种是先随机采样一个相机，然后再从相机中出现的物体进行采样，生成描述。
        objs = []
        selected_ids = set()
        i = 0
        while len(objs) < 3:
            i += 1
            if i >= max_try:
                return qa
            temp_obj_id = None
            temp_description = None
            # 从类别唯一的物体集合中采样
            if random.random() < 0.5:
                if single_occurrence_objects:
                    temp_obj_id, category = random.choice(single_occurrence_objects)
                    temp_description = category
            # 从某个相机的物体集合中采样
            else:
                random_index = random.randrange(len(cameras))
                random_cam = cameras[random_index]
                if not random_cam['objects']:
                    continue
                temp_obj_id = random.choice(sorted(list(random_cam['objects'].keys())))
                if temp_obj_id in excluded_ids:
                    continue
                bbox_2d = random_cam.get('objects', {}).get(temp_obj_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                temp_category = objects[temp_obj_id]['category']
                success, only_cam1_obj_cat, temp_obj_des = get_description(temp_obj_id, random_cam, objects)
                if not success:
                    continue
                temp_description = f"{temp_obj_des} in figure {random_index + 1}"
            # 确保成功采到了 ID 和描述
            if temp_obj_id and temp_description:
                # 检查 ID 是否已经被选中
                if temp_obj_id not in selected_ids:
                    # 1. 添加到最终结果列表
                    objs.append({'id': temp_obj_id, 'des': temp_description})

                    # 2. 将 ID 添加到集合中，防止下次重复
                    selected_ids.add(temp_obj_id)

        # 3. 采样三个物体后，仍采用之前的方式生成问题。
        # 提问：假设objs[0]在objs[1]的什么方向，求objs[2]在objs[1]的什么方向
        obj0_loc = np.array(objects[objs[0]['id']]['3d_center'][:2])
        obj1_loc = np.array(objects[objs[1]['id']]['3d_center'][:2])
        obj2_loc = np.array(objects[objs[2]['id']]['3d_center'][:2])
        # 筛掉物体之间距离过小的组合 防止出现物体是上下关系但却询问水平方位的情况
        translation = np.linalg.norm(obj1_loc - obj0_loc)
        if translation < translation_threshold:
            return qa
        translation = np.linalg.norm(obj2_loc - obj0_loc)
        if translation < translation_threshold:
            return qa
        main_direction, rel_direction, direct_8 = get_relative_orientation(obj0_loc, obj1_loc, obj2_loc)
        # 生成选项
        options, answer = generate_orientation_choices(rel_direction, direct_8)
        # 生成QA
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(obj1_des=objs[1]['des'], obj0_des=objs[0]['des'], obj2_des=objs[2]['des'], main_direction=main_direction, options=options)
        qa_group = []
        qa_group.append((question, answer, level3_qa_type, "1,2"))
        qa.append(qa_group)
        return qa_group

    def get_objects_in_either_cam(self, cam1_objects: Dict, cam2_objects: Dict, all_objects: Dict) -> Dict:
        """
        筛选在cam1中出现 **或** 在cam2中出现的所有物体（返回完整信息）
        :param cam1_objects: cam1视野内的物体（key与all_objects匹配）
        :param cam2_objects: cam2视野内的物体（key与all_objects匹配）
        :param all_objects: 场景中所有物体的完整信息
        :return: 所有在cam1或cam2中出现的物体（key为物体标识，value为完整信息）
        """
