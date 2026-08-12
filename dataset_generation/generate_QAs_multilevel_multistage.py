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
