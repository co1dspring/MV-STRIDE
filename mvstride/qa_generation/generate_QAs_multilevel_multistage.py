# -*- coding: utf-8 -*-
import os, sys
import argparse
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
    def __init__(self, config_path: str = "./configs/qa/qa_config_infinigen.json"):
        """
        Initialize the scene QA generator.
        :param base_dir: scene parent directory
        :param output_root: root directory for saving per-question-type JSON files
        :param metadata_filename: metadata filename
        :param camera_count: number of cameras per scene
        """
        # Read the config file
        self.config = load_config(Path(config_path))

        # Config files live in configs/qa/ (shipped layout <repo-root>/configs/qa/).
        # Resolve every relative path against the repo root (not the CWD), so
        # values like "data/infinigen/saved_scenes" work from any working directory.
        self.config_dir = Path(config_path).resolve().parent
        path_base = self.config_dir
        if self.config_dir.name == "qa" and self.config_dir.parent.name == "configs":
            path_base = self.config_dir.parent.parent

        def _cfg_path(key: str, *, fmt: bool = False, as_str: bool = False):
            v = self.config.get(key, "")
            if fmt:
                v = v.format(VERSION_NAME=self.version_name or "")
            p = Path(v)
            if not p.is_absolute():
                p = path_base / p
            return str(p) if as_str else p

        # Load paths
        self.source_data_dir = _cfg_path('source_data_dir')
        self.version_name = self.config.get('version_name')
        self.output_dir = _cfg_path('output_dir', fmt=True)  # for saving per-question-type files
        self.training_environment_base_dir = _cfg_path('training_environment_base_dir', as_str=True)
        self.metadata_filename = self.config.get('metadata_filename')
        self.qa_templates_path = _cfg_path('qa_templates_path')
        self.data_source = self.config.get('data_source')
        self.qa_dependency_tree_path = _cfg_path('qa_dependency_tree_path', as_str=True)

        # Load numeric parameters
        self.camera_count = self.config.get('camera_count')
        self.if_flipping_enhencement = self.config.get('if_flipping_enhencement')
        self.if_MCA = self.config.get('if_MCA')
        self.MIN_AREA_THRESHOLD = self.config.get('MIN_AREA_THRESHOLD')  # minimum bounding-box area (pixels)
        self.MIN_SIDE_THRESHOLD = self.config.get('MIN_SIDE_THRESHOLD')   # minimum bounding-box short-side length (pixels)
        self.common_num_threshold = self.config.get('common_num_threshold')
        self.global_seed = self.config.get('global_seed')
        self.qa_types_per_camera_pair = self.config.get('qa_types_per_camera_pair')
        self.stage_1_proportion = self.config.get('stage_1_proportion')
        self.stage_2_proportion = self.config.get('stage_2_proportion')
        self.stage_3_proportion = self.config.get('stage_3_proportion')
        self.level12_sampling_rate = self.config.get('level12_sampling_rate')

        # Load text parameters
        self.direction_name = self.config.get('direction_name')
        self.direction_name_2 = self.config.get('direction_name_2')
        self.direction_map = self.config.get('direction_map')
        self.obj_area_map = self.config.get('obj_area_map')
        self.unwanted_cats = self.config.get('unwanted_cats')
        self.room_type = self.config.get('room_type')
        self.obj_cat_with_orientation = self.config.get('obj_cat_with_orientation')
        self.multilevel_qa_mode = self.config.get('multilevel_qa_mode')

        # Initialize functions
        self.scene_folders = self._get_all_scene_folders()
        self.scene_names = [scene.name for scene in self.scene_folders]
        self.multilevel_qa_generator = Multilevel_QA_Generator(parent=self)
        self.qa_types = self._define_qa_types()  # question types: {type name: generator function}
        self.msr_qa_types = self._define_msr_qa_types()
        self.level2_qa_functions = self._define_level12_qa_types("level2_qa_types")
        self.level1_qa_functions = self._define_level12_qa_types("level1_qa_types")
        self.level3_qa_types = self.qa_types | self.msr_qa_types
        self.all_qa_types = self.level3_qa_types | self.level2_qa_functions | self.level1_qa_functions
        self._init_output_dirs()  # create the output directories for question types
        self.set_global_seed(self.global_seed) # set the global random seed
        self.templates = load_config(self.qa_templates_path)
        self.qa_dependency_tree = load_config(self.qa_dependency_tree_path)
        self.log_path = setup_logging(self.output_dir)

    def set_global_seed(self, seed):
        """Set the random seeds across multiple libraries."""
        # 1. Python built-in random module
        random.seed(seed)

        # 2. NumPy library (commonly used for scientific computing and data processing)
        np.random.seed(seed)

        # 3. If you use PyTorch or TensorFlow, you also need to set those
        # import torch
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        # 4. Set the hash seed (usually to avoid hash collisions across Python instances, which indirectly affects randomness)
        os.environ['PYTHONHASHSEED'] = str(seed)

        print(f"全局随机种子已设置为: {seed}")

    def _get_all_scene_folders(self) -> List[Path]:
        """Get all scene folders."""
        if self.data_source == "scannetpp":
            folders = [p for p in self.source_data_dir.iterdir() if p.is_dir() and str(p).endswith("iphone")]
        else:
            folders = [p for p in self.source_data_dir.iterdir() if p.is_dir()]
        folders.sort()
        print(f"发现 {len(folders)} 个场景文件夹")
        return folders

    def _init_output_dirs(self):
        """Create the output directory."""
        (self.output_dir).mkdir(parents=True, exist_ok=True)

    def _define_qa_types(self) -> Dict[str, callable]:
        """Define the question-type generator functions (key is the type name, value is the generator)."""
        # True means the camera order is swapped to generate the QA again
        # The number represents how many QA items to keep at most per camera group
        """Load the question-type generator functions from the config."""
        qa_config_data = self.config.get("qa_types", {})
        qa_types_map: Dict[str, Any] = {}

        # Get the default value (here assuming max_qa_per_cams_default is also 1)
        default_max_qa = self.config.get("max_qa_per_cams_default", 1)
        overall_sampling_rate = self.config.get("overall_sampling_rate", 1)

        for type_name, params in qa_config_data.items():
            # 1. Use getattr to dynamically look up the matching generator function
            # function_name must be a member-method-name string defined in the class
            generator_func = getattr(self, params['function_name'], None)

            if not generator_func:
                print(f"警告: 未找到对应的生成函数: {params['function_name']}，跳过该类型 {type_name}")
                continue

            # 2. Read parameters from the config, using defaults if missing
            flipping = params.get('flipping_enhancement', self.if_flipping_enhencement)
            max_qa = params.get('max_qa_per_cams', default_max_qa)
            sampling_rate = params.get('sampling_rate', 1)

            # 3. Construct QATypeConfig (assuming its constructor matches the parameters used in your code)
            # Note: the last parameter (1) in the QATypeConfig constructor is still hard-coded; consider moving it into the config.
            qa_types_map[type_name] = QATypeConfig(generator_func, flipping, max_qa, sampling_rate * overall_sampling_rate)

        return qa_types_map

    def _define_msr_qa_types(self) -> Dict[str, callable]:
        """Load the MSR question-type generator functions from the config."""
        msr_config_data = self.config.get("msr_qa_types", {})
        msr_types_map: Dict[str, Any] = {}

        # Assume MSQ_QATypeConfig max_num defaults to 10
        default_max_num = self.config.get("max_msr_num_default", 10)
        overall_sampling_rate = self.config.get("overall_sampling_rate", 1)

        for type_name, params in msr_config_data.items():
            generator_func = getattr(self, params['function_name'], None)

            if not generator_func:
                print(f"警告: 未找到对应的 MSR 生成函数: {params['function_name']}，跳过该类型 {type_name}")
                continue

            # MSR_QATypeConfig needs a range (e.g. [3, 3])
            range_tuple = tuple(params['range'])
            max_num = params.get('max_num', default_max_num)
            sampling_rate = params.get('sampling_rate', 1)

            msr_types_map[type_name] = MSR_QATypeConfig(generator_func, range_tuple, max_num, sampling_rate * overall_sampling_rate)

        return msr_types_map

    def _define_level12_qa_types(self, typename) -> Dict[str, callable]:
        qa_config_data = self.config.get(typename, {})
        qa_types_map: Dict[str, Any] = {}

        for type_name, params in qa_config_data.items():
            # 1. Use getattr to dynamically look up the matching generator function
            # function_name must be a member-method-name string defined in the class
            qa_types_map[type_name] = {}
            generator_func = getattr(self.multilevel_qa_generator, params['function_name'], None)

            if not generator_func:
                print(f"警告: 未找到对应的生成函数: {params['function_name']}，跳过该类型 {type_name}")
                continue

            qa_types_map[type_name]['func'] = generator_func
            qa_types_map[type_name]['sampling_rate'] = params['sampling_rate']

        return qa_types_map

    def _get_camera_pairs(self, cameras: Dict[str, Any]) -> List[Tuple[str, str, Dict, Dict]]:
        # Compute the base number of combinations (190 combinations when N=20)
        base_n = 50
        max_pairs_limit = (base_n * (base_n - 1)) // 2  # 190
        camera_names = sorted(list(cameras.keys()))
        random.shuffle(camera_names)
        if len(camera_names) != self.camera_count:
            print(f"警告：相机数量不符（预期{self.camera_count}，实际{len(camera_names)}）")
        all_cameras_pairs = combinations(camera_names, 2)
        if len(cameras.keys()) <= base_n:
            """Generate pairwise non-repeating camera combinations."""
            # Use permutations to generate ordered pairs; the second argument 2 means each pair contains 2 elements
            return [
                (c1, c2, cameras[c1], cameras[c2])
                for c1, c2 in all_cameras_pairs  # replaced here with permutations
            ]
        else:
            # Too many cameras
            # Fix: when there are too many, sample with a set to avoid generating a huge list
            sampled_name_pairs = set()

            # Keep sampling until reaching 190 pairs
            # Use a set to automatically handle duplicate sampling
            while len(sampled_name_pairs) < max_pairs_limit:
                # Randomly draw two non-repeating indices
                c1_name, c2_name = random.sample(camera_names, 2)

                # Sort so (A, B) and (B, A) count as the same unordered pair, stored in a set to dedupe
                pair = tuple(sorted((c1_name, c2_name)))
                sampled_name_pairs.add(pair)

            sorted_sampled_pairs = sorted(list(sampled_name_pairs))
            return [
                (c1, c2, cameras[c1], cameras[c2])
                for c1, c2 in sorted_sampled_pairs
            ]

    # ------------------------------
    # QA generator functions (same as before; only return the QA list)
    # ------------------------------
    def _attribute_appr_counting_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        qa = []
        drop_prop = 0.5
        level3_qa_type = "Attribute(Appr.)_Counting"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]

        # 1. Extract the object IDs and categories appearing in both cameras
        # Objects in cam1: {object ID: category}
        cam1_objs = {obj_id: objects[obj_id]['category'] for obj_id in cam1_data.get('objects', {}).keys()}
        # Objects in cam2: {object ID: category}
        cam2_objs = {obj_id: objects[obj_id]['category'] for obj_id in cam2_data.get('objects', {}).keys()}

        # Get the categories common to both cameras (category intersection first)
        cam1_cats = set(cam1_objs.values())
        cam2_cats = set(cam2_objs.values())
        common_cats = sorted(list(cam1_cats & cam2_cats))

        for cat in common_cats:
            # Get the object IDs of this category in both cameras; must not be empty
            cam1_ids = set([obj_id for obj_id, c in cam1_objs.items() if c == cat])
            cam2_ids = set([obj_id for obj_id, c in cam2_objs.items() if c == cat])

            # 2. Multi-step filtering
            # Filter by size using the 2D bounding box. Skip if any object is judged too small.
            should_skip_category = False
            # Iterate over all objects of this category in Cam1
            for obj_id in cam1_ids:
                # Extract the bbox_2d data
                bbox_2d = cam1_data.get('objects', {}).get(obj_id, {}).get('bbox_2d')

                # Check using the new function
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    should_skip_category = True
                    break

            if should_skip_category:
                continue  # skip this category

            # Iterate over all objects of this category in Cam2
            for obj_id in cam2_ids:
                # Extract the bbox_2d data
                bbox_2d = cam2_data.get('objects', {}).get(obj_id, {}).get('bbox_2d')

                # Check using the new function
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    should_skip_category = True
                    break

            if should_skip_category:
                continue  # skip this category

            # Check validity: whether objects with different IDs exist. If the two cameras show exactly the same objects, drop with probability 0.5.
            if cam1_ids == cam2_ids and random.random() < drop_prop:
                continue
            # Compute the de-duplicated total
            all_ids = cam1_ids | cam2_ids
            unique_count = len(set(all_ids))
            # Drop QAs with answer 1 with some probability
            if unique_count == 1 and random.random() < 0.75:
                continue

            # 3. After all filtering, generate the Level 3 QA
            qa_group = []
            options, answer = generate_random_int_choices(unique_count)

            selected_template = random.choice(QUESTION_TEMPLATES)
            if self.if_MCA:
                selected_template += '\nOptions: {options}'
            else:
                answer = answer[3:]
            question = selected_template.format(cat=cat.lower(), options=options)
            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
            # 4.1 Prepare the metadata needed to generate level 1 and 2 questions

            # 4.2 First generate the level 2 questions
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                for obj_id in sorted(list(cam1_ids)):
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_id, cat, "1,2"))
                for obj_id in sorted(list(cam2_ids)):
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_id, cat, "2,1"))

            # 4.1 Then generate the level 1 questions
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

        # 1. Find objects appearing in both cameras as reference objects (ID intersection)
        # Extract the set of object IDs in both cameras
        cam1_ids = set(cam1_data['objects'].keys())
        cam2_ids = set(cam2_data['objects'].keys())

        # Intersection: object IDs appearing in both cameras
        common_ids = sorted(list(cam1_ids & cam2_ids))  # the & operator on sets means intersection
        if not common_ids:
            return qa  # generate no questions when there is no common object

        # 2. Camera position and orientation
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y']])  # , cam1_data['location_3d']['z']
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y']])  # , cam2_data['location_3d']['z']

        for ref_obj_id in common_ids:
            # 3. A series of filters
            # Generate a description of the object: if cam1 contains only one object of this category, describe it directly by category; otherwise describe it by its relative position among objects of this category
            success, ref_obj_cat, ref_obj_des = get_description(ref_obj_id, cam1_data, objects)
            if not success:
                continue
            # Filter out special categories
            if ref_obj_cat in ['rug']:
                continue
            # Filter by object size
            bbox_2d = cam2_data.get('objects', {}).get(ref_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            bbox_2d = cam1_data.get('objects', {}).get(ref_obj_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue

            # 4. Generate the level 3 question
            qa_group = []
            # Get the reference object coordinates and the three axis orientations
            ref_obj_loc = np.array(objects[ref_obj_id]['3d_center'][:2])

            main_direction, rel_direction, direct_8 = get_relative_orientation(ref_obj_loc, cam1_loc, cam2_loc)
            # Drop same-direction questions with some probability
            if main_direction == rel_direction and random.random() < drop_prop:
                continue
            # Generate the options
            options, answer = generate_orientation_choices(rel_direction, direct_8)
            # Generate the QA
            selected_template = random.choice(QUESTION_TEMPLATES)
            if self.if_MCA:
                selected_template += '\nOptions: {options}'
            else:
                answer = answer[3:]
            question = selected_template.format(main_direction=main_direction, ref_obj_des=ref_obj_des, ref_obj_cat=ref_obj_cat, options=options)
            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 5. Generate level 1 and 2 questions that this level 3 question capability depends on
            # 5.1 Prepare the metadata needed to generate level 1 and 2 questions

            # 5.2 First generate the level 2 questions
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des, "2,1"))

            # 5.3 Then generate the level 1 questions
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

        # 1. Extract the three object sets
        cam1_objs = set(cam1_data.get('objects', {}).keys())  # object IDs in camera 1
        cam2_objs = set(cam2_data.get('objects', {}).keys())  # object IDs in camera 2

        # Objects appearing only in cam1
        only_cam1 = sorted([obj_id for obj_id in cam1_objs if obj_id not in cam2_objs])
        # Objects appearing only in cam2
        only_cam2 = sorted([obj_id for obj_id in cam2_objs if obj_id not in cam1_objs])
        # Objects appearing in both cameras (used for filtering; continue only if non-empty)
        both_cams = sorted(list(cam1_objs & cam2_objs))
        # 2. If the set of objects in both cameras is empty, generate no questions
        if not both_cams:
            return qa

        # 3. Pair objects in only-cam1 with only-cam2 to generate size-comparison questions
        # Generate only one question per object group, but there are many groups; the outer function finally selects randomly
        for obj1_id in only_cam1:
            bbox_2d = cam1_data.get('objects', {}).get(obj1_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, obj1_cat, obj1_des = get_description(obj1_id, cam1_data, objects)
            if not success:
                continue
            for obj2_id in only_cam2:
                qa_group = []
                # Decide the phrasing: whether the correct answer is the numerically larger or smaller side
                ask_for_greater = random.random() > prop_ask_for_greater
                bbox_2d = cam2_data.get('objects', {}).get(obj2_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                success, obj2_cat, obj2_des = get_description(obj2_id, cam2_data, objects)
                if not success:
                    continue
                # Get the object dimensions
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

                # Decide the question type
                # First check whether both are small high objects; if so, compare their position heights
                if obj1_pos_z > min_height and obj2_pos_z > min_height and obj1_volume < min_volume and obj2_volume < min_volume:
                    # Comparing position heights does not involve multiples
                    dimension_to_compare = 'altitude'
                    # Filter out overly disparate comparisons
                    if obj1_pos_z / obj2_pos_z > disparity_factor or obj2_pos_z / obj1_pos_z > disparity_factor:
                        continue
                    # Drop "the same" with some probability
                    ratio = max(obj1_pos_z, obj2_pos_z) / min(obj1_pos_z, obj2_pos_z)
                    if ratio <= proximity_factor and random.random() < eliminate_the_same:
                        continue
                    # Choose the template for a different phrasing
                    templates = TEMPLATES_POSITION_IN_HEIGHT['greater'] if ask_for_greater else TEMPLATES_POSITION_IN_HEIGHT['smaller']
                    options, answer = generate_size_comparison_choices(obj1_3d_bbox['max']['z'], obj2_3d_bbox['max']['z'], obj1_des, obj2_des, 'height', proximity_factor, ask_for_greater)
                    selected_template = random.choice(templates) + '\nOptions: {options}'
                    question = selected_template.format(obj1_des=obj1_des, obj2_des=obj2_des, options=options)
                    qa_group.append((question, answer, level3_qa_type, "1,2"))
                # Otherwise compare height or width of the two sizes (length and width are ambiguous in description; the description compares the long or short side instead)
                else:
                    # Randomly choose whether to compare length (long/short side) or height
                    # The 0.5 random-choice pattern is abstracted to this step:
                    dimension_to_compare = random.choice(['length', 'height'])

                    # 1. Determine the dimension, values, and template for this comparison
                    if dimension_to_compare == 'length':
                        # Variables related to the length comparison
                        obj1_val = obj1_length
                        obj2_val = obj2_length
                        # Randomly choose 'length' or 'width' as the question description
                        question_des = random.choice(['length', 'width'])
                        # The matching template set (obtained from TEMPLATES_LENGTH)
                        template_group = TEMPLATES_LENGTH[question_des]
                    else:  # dimension_to_compare == 'height'
                        # Variables related to the height comparison
                        obj1_val = obj1_height
                        obj2_val = obj2_height
                        question_des = 'height'
                        # The matching template set (obtained from TEMPLATES_HEIGHT)
                        template_group = TEMPLATES_HEIGHT

                    # 2. Multiple comparison (using the abstract obj1_val and obj2_val)
                    if obj1_val / obj2_val > disparity_factor:
                        k = round(obj1_val / obj2_val)
                        if k > k_threshold:
                            continue
                        # Directly modify obj2_val and obj2_des
                        obj2_val *= k
                        obj2_des_new = f"{k} times the {question_des} of the {obj2_des}"
                        obj1_des_new = f"{question_des} of the {obj1_des}"
                    elif obj2_val / obj1_val > disparity_factor:
                        k = round(obj2_val / obj1_val)
                        if k > k_threshold:
                            continue
                        # Directly modify obj1_val and obj1_des
                        obj1_val *= k
                        obj1_des_new = f"{k} times the {question_des} of the {obj1_des}"
                        obj2_des_new = f"{question_des} of the {obj2_des}"
                    else:
                        obj1_des_new = f"{question_des} of the {obj1_des}"
                        obj2_des_new = f"{question_des} of the {obj2_des}"

                    # 3. Similarity drop check
                    ratio = max(obj1_val, obj2_val) / min(obj1_val, obj2_val)
                    if ratio <= proximity_factor and random.random() < eliminate_the_same:
                        continue

                    # 4. Phrasing selection and QA generation
                    # Choose the template for a different phrasing
                    templates = template_group['greater'] if ask_for_greater else template_group['smaller']
                    options, answer = generate_size_comparison_choices(obj1_val, obj2_val, obj1_des_new, obj2_des_new, dimension_to_compare, proximity_factor, ask_for_greater)
                    selected_template = random.choice(templates)
                    if self.if_MCA:
                        selected_template += '\nOptions: {options}'
                    else:
                        answer = answer[3:]
                    question = selected_template.format(obj1_des=obj1_des_new, obj2_des=obj2_des_new, options=options)
                    qa_group.append((question, answer, level3_qa_type, "1,2"))

                # 5. Generate level 1 and 2 questions that this level 3 question capability depends on
                # 5.1 Prepare the metadata needed to generate level 1 and 2 questions
                # Randomly choose a common object as reference
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
                    # 5.2 First generate the level 2 questions
                    level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
                    for level2_qa_type in level2_qa_types:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

                    # 5.3 Then generate the level 1 questions
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

        # 1. Camera position and orientation
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])  #
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])  #
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])
        # 2. Filtering
        # Filter out camera pairs with excessive z-axis change
        if abs(cam1_loc[-1] - cam2_loc[-1]) > z_threshold:
            return qa
        # Filter out camera pairs with excessive rotation angle
        rel_angel = calculate_rotation(cam1_for[:2], cam2_for[:2])
        if abs(rel_angel) > rotation_threshold:
            return qa

        # Ensure the two cameras share a common object
        cam1_objs = set(cam1_data['objects'].keys())
        cam2_objs = set(cam2_data['objects'].keys())
        # Intersection: object IDs appearing in both cameras
        common_ids = cam1_objs & cam2_objs  # the & operator on sets means intersection
        if len(common_ids) < self.common_num_threshold:
            return qa  # generate no questions when there is no common object

        # 3. Generate the level 3 question
        qa_group = []

        forward_dir, right_dir = get_relative_direction(cam1_loc, cam2_loc, cam1_for, thresh=translation_threshold)
        # if forward_dir == "" and right_dir == "":
        #     return qa
        # Add the "Not Moving" option
        options, answer = generate_direction_choices(forward_dir, right_dir)
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(options=options)
        qa_group.append((question, answer, level3_qa_type, "1,2"))

        # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
        # 4.1 Prepare the metadata needed to generate level 1 and 2 questions
        # Randomly choose a common object as reference
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

            # 4.2 First generate the level 2 questions
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

            # 4.3 Then generate the level 1 questions
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

        # 1. Camera position and orientation
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])  #
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])  #
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])

        # 2. Filtering
        # Filter out pairs with excessive translation
        translation = np.linalg.norm(cam1_loc - cam2_loc)
        if translation > translation_threshold:
            return qa
        # Filter out camera pairs with excessive rotation angle
        rel_angel = calculate_rotation(cam1_for[:2], cam2_for[:2])
        if abs(rel_angel) > rotation_max_threshold:
            return qa

        # Ensure the two cameras share a common object
        cam1_objs = set(cam1_data['objects'].keys())
        cam2_objs = set(cam2_data['objects'].keys())
        # Intersection: object IDs appearing in both cameras
        common_ids = cam1_objs & cam2_objs  # the & operator on sets means intersection
        if len(common_ids) < self.common_num_threshold:
            return qa  # generate no questions when there is no common object

        # 3. Generate the level 3 question
        qa_group = []
        # Get the rotation matrices of the two cameras
        R1 = get_camera_rotation_matrix(cam1_data)
        R2 = get_camera_rotation_matrix(cam2_data)
        # Compute the relative rotation from cam1 to cam2
        yaw_dir, pitch_dir = get_relative_rotation(R1, R2, rotation_min_threshold, is_scannetpp=(self.data_source=="scannetpp"))

        options, answer = generate_rotation_choices(yaw_dir, pitch_dir)
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(options=options)
        qa_group.append((question, answer, level3_qa_type, "1,2"))

        # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
        # 4.1 Prepare the metadata needed to generate level 1 and 2 questions
        # Randomly choose a common object as reference
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

            # 4.2 First generate the level 2 questions
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

            # 4.3 Then generate the level 1 questions
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

        # 1. Camera position and orientation
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])  #
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])  #
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])

        # 2. View filtering
        # Filter out camera pairs with excessive z-axis change
        if abs(cam1_loc[-1] - cam2_loc[-1]) > z_threshold:
            return qa
        # Filter out camera pairs with excessive rotation angle
        rel_angel = calculate_rotation(cam1_for[:2], cam2_for[:2])
        if abs(rel_angel) > rotation_threshold:
            return qa
        # Ensure the two cameras share a common object
        cam1_objs = set(cam1_data['objects'].keys())
        cam2_objs = set(cam2_data['objects'].keys())
        # Intersection: object IDs appearing in both cameras
        common_ids = cam1_objs & cam2_objs  # the & operator on sets means intersection
        if len(common_ids) < self.common_num_threshold:
            return qa  # generate no questions when there is no common object

        # 3. Question generation
        # Define two phrasings (Scenarios)
        # Scenario 1: C2 is the reference frame (RefCam), C1 is the target (TargetObj) -> corresponds to the T1 phrasing
        scenario_1 = {
            'ref_cam_loc': cam1_loc,
            'ref_cam_for': cam1_for,
            'tar_cam_loc': cam2_loc,
            'is_case_1': True  # used for template filling
        }
        # Scenario 2: C1 is the reference frame (RefCam), C2 is the target (TargetObj) -> corresponds to the T2 phrasing
        scenario_2 = {
            'ref_cam_loc': cam2_loc,
            'ref_cam_for': cam2_for,
            'tar_cam_loc': cam1_loc,
            'is_case_1': False  # used for template filling
        }

        for scenario in [scenario_1, scenario_2]:
            qa_group = []
            # 4. Level 3 question generation
            forward_dir, right_dir = get_relative_direction(scenario['ref_cam_loc'], scenario['tar_cam_loc'], scenario['ref_cam_for'], thresh=translation_threshold)
            if forward_dir == "" and right_dir == "":
                continue

            # for mode in modes:
            mode = random.choice(modes)
            MODE_TEMPLATES = QUESTION_TEMPLATES[mode]
            # MODE_1 is the phrasing that sets an explicit coordinate system
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
            # MODE_2 is the phrasing using conventional direction descriptions
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

            # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
            # 4.1 Prepare the metadata needed to generate level 1 and 2 questions
            # Randomly choose a common object as reference
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

                # 4.2 First generate the level 2 questions
                level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
                for level2_qa_type in level2_qa_types:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                    if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

                # 4.3 Then generate the level 1 questions
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

        # 1. Camera position and orientation
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])  #
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])  #
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])

        # Filter out pairs with excessive translation
        translation = np.linalg.norm(cam1_loc - cam2_loc)
        if translation > translation_threshold:
            return qa
        # Filter out camera pairs with excessive rotation angle
        rel_angel = calculate_rotation(cam1_for[:2], cam2_for[:2])
        if abs(rel_angel) > rotation_max_threshold:
            return qa
        # Ensure the two cameras share a common object
        cam1_objs = set(cam1_data['objects'].keys())
        cam2_objs = set(cam2_data['objects'].keys())
        # Intersection: object IDs appearing in both cameras
        common_ids = cam1_objs & cam2_objs  # the & operator on sets means intersection
        if len(common_ids) < self.common_num_threshold:
            return qa  # generate no questions when there is no common object

        # Get the rotation matrices of the two cameras
        R1 = get_camera_rotation_matrix(cam1_data)
        R2 = get_camera_rotation_matrix(cam2_data)

        # 3. Question generation
        # Define two phrasings (Scenarios)
        # Scenario 1: C2 is the reference frame (RefCam), C1 is the target (TargetObj) -> corresponds to the T1 phrasing
        scenario_1 = {
            'ref_cam_R': R1,
            'tar_cam_R': R2,
            'is_case_1': True  # used for template filling
        }
        # Scenario 2: C1 is the reference frame (RefCam), C2 is the target (TargetObj) -> corresponds to the T2 phrasing
        scenario_2 = {
            'ref_cam_R': R2,
            'tar_cam_R': R1,
            'is_case_1': False  # used for template filling
        }
        for scenario in [scenario_1, scenario_2]:
            qa_group = []
            # 4. Level 3 question generation
            # Compute the relative rotation from cam1 to cam2
            yaw_dir, pitch_dir = get_relative_rotation(scenario['ref_cam_R'], scenario['tar_cam_R'], rotation_min_threshold, is_scannetpp=(self.data_source=="scannetpp"))
            if yaw_dir == "" and pitch_dir == "":
                continue
            # for mode in modes:
            mode = random.choice(modes)
            MODE_TEMPLATES = QUESTION_TEMPLATES[mode]
            # MODE_1 is the phrasing that sets an explicit coordinate system
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
            # MODE_2 is the phrasing using conventional direction descriptions
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

            # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
            # 4.1 Prepare the metadata needed to generate level 1 and 2 questions
            # Randomly choose a common object as reference
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

                # 4.2 First generate the level 2 questions
                level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
                for level2_qa_type in level2_qa_types:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_1, "1,2"))
                    if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_id, ref_obj_des_2, "2,1"))

                # 4.3 Then generate the level 1 questions
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
        # 1. Find objects appearing in both cameras as reference objects (ID intersection)
        # Extract the set of object IDs in both cameras
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        only_cam1 = cam1_objs - cam2_objs
        only_cam2 = cam2_objs - cam1_objs
      
        # Intersection: object IDs appearing in both cameras
        common_ids = cam1_objs & cam2_objs  # the & operator on sets means intersection
        if len(common_ids) < self.common_num_threshold:
            return qa  # generate no questions when there is no common object

        # Define two phrasings (Scenarios)
        # Scenario 1: C2 is the reference frame (RefCam), C1 is the target (TargetObj) -> corresponds to the T1 phrasing
        scenario_1 = {
            'target_ids': sorted(list(only_cam2)),
            'target_cam_data': cam2_data,  # C1 describes the object
            'ref_cam_data': cam1_data,  # C2 computes the direction
            'is_case_1': True  # used for template filling
        }
        # Scenario 2: C1 is the reference frame (RefCam), C2 is the target (TargetObj) -> corresponds to the T2 phrasing
        scenario_2 = {
            'target_ids': sorted(list(only_cam1)),
            'target_cam_data': cam1_data,  # C2 describes the object
            'ref_cam_data': cam2_data,  # C1 computes the direction
            'is_case_1': False  # used for template filling
        }

        for scenario in [scenario_1, scenario_2]:
            ref_cam_loc = np.array([scenario['ref_cam_data']['location_3d']['x'], scenario['ref_cam_data']['location_3d']['y'], scenario['ref_cam_data']['location_3d']['z']])
            ref_cam_for = np.array([scenario['ref_cam_data']['forward_direction']['x'], scenario['ref_cam_data']['forward_direction']['y'], scenario['ref_cam_data']['forward_direction']['z']])

            # 1. Pre-fill the language indicators in the templates
            filled_templates = [
                fill_template_placeholders(t, scenario['is_case_1'])
                for t in QUESTION_TEMPLATES
            ]

            for obj_id in scenario['target_ids']:
                qa_group = []
                bbox_2d = scenario['target_cam_data'].get('objects', {}).get(obj_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue

                # ... (logic for getting obj_des and computing the direction remains unchanged) ...
                success, _, obj_des = get_description(obj_id, scenario['target_cam_data'], objects)
                if not success: continue

                obj_loc = np.array(objects[obj_id]['3d_center'])
                forward_dir, right_dir = get_relative_direction(ref_cam_loc, obj_loc, ref_cam_for, thresh=translation_threshold)
                if forward_dir == "" and right_dir == "":
                    continue

                # 2. Format the final question
                options, answer = generate_direction_choices(forward_dir, right_dir, moving=False)
                selected_template = random.choice(filled_templates)
                if self.if_MCA:
                    selected_template += '\nOptions: {options}'
                else:
                    answer = answer[3:]
                question = selected_template.format(obj_des=obj_des, options=options)
                qa_group.append((question, answer, level3_qa_type, "1,2"))

                # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
                # 4.1 Prepare the metadata needed to generate level 1 and 2 questions

                # 4.2 First generate the level 2 questions
                level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
                for level2_qa_type in level2_qa_types:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_id, obj_des, "1,2"))
                    if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_id, obj_des, "2,1"))

                # 4.3 Then generate the level 1 questions
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
        # 1. Find objects appearing in both cameras as reference objects (ID intersection)
        # Extract the set of object IDs in both cameras
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        # cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])
        # cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])
        # Intersection: object IDs appearing in both cameras
        common_ids = sorted(list(cam1_objs & cam2_objs))  # the & operator on sets means intersection
        if not common_ids:
            return qa  # generate no questions when there is no common object

        # 2. Find objects appearing in only cam1 or only cam2 (without filtering by category)
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
            # Ask: given the direction of objs[0] from objs[1], find the direction of objs[2] from objs[1]
            obj0_loc = np.array(objects[objs[0]['id']]['3d_center'][:2])
            obj1_loc = np.array(objects[objs[1]['id']]['3d_center'][:2])
            obj2_loc = np.array(objects[objs[2]['id']]['3d_center'][:2])
            # Filter out combinations whose objects are too close together
            translation = np.linalg.norm(obj1_loc - obj0_loc)
            if translation < translation_threshold:
                continue
            translation = np.linalg.norm(obj2_loc - obj0_loc)
            if translation < translation_threshold:
                continue

            qa_group = []
            main_direction, rel_direction, direct_8 = get_relative_orientation(obj0_loc, obj1_loc, obj2_loc)
            # Generate the options
            options, answer = generate_orientation_choices(rel_direction, direct_8)
            # Generate the QA
            selected_template = random.choice(QUESTION_TEMPLATES)
            if self.if_MCA:
                selected_template += '\nOptions: {options}'
            else:
                answer = answer[3:]
            question = selected_template.format(obj1_des=objs[1]['des'], obj0_des=objs[0]['des'], obj2_des=objs[2]['des'], main_direction=main_direction, options=options)
            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
            # 4.1 Prepare the metadata needed to generate level 1 and 2 questions

            # 4.2 First generate the level 2 questions
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, common_obj_id, common_obj_des, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, common_obj_id, common_obj_des, "2,1"))

            # 4.3 Then generate the level 1 questions
            level1_qa_types = self.qa_dependency_tree[level3_qa_type]['level1']
            for level1_qa_type in level1_qa_types:
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, only_cam1_obj_id, only_cam1_obj_des, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, only_cam2_obj_id, only_cam2_obj_des, "2"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam1_data, common_obj_id, common_obj_des, "1"))
                qa_group.append(self.level1_qa_functions[level1_qa_type]['func'](objects, cam2_data, common_obj_id, common_obj_des, "2"))

            qa.append(qa_group)

        return qa

    def _positional_relationship_obj_obj_orientation_qa(self, cam1_data: Dict, cam2_data: Dict, objects: Dict, rooms: Dict) -> List[Tuple[str, str]]:
        # Initialize hyperparameters
        qa = []
        max_num = 5
        max_try = 1000
        translation_threshold = 0.3
        level3_qa_type = "Positional Relationship(Obj.-Obj.)_Orientation"
        TEMPLATES = self.templates[level3_qa_type]

        # 1. Partition the scene objects into different sets
        # Find whether the scene contains objects with a semantic orientation
        objs_with_orientation = {obj: objects[obj] for obj in objects if objects[obj]['category'] in list(self.obj_cat_with_orientation.keys())}
        if not objs_with_orientation:
            return qa
        # Extract the set of object IDs in both cameras
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        # Intersection: object IDs appearing in both cameras
        common_ids = sorted(list(cam1_objs & cam2_objs))  # the & operator on sets means intersection
        if not common_ids:
            return qa  # generate no questions when there is no common object
        # Find objects appearing in only cam1 or only cam2 (without filtering by category)
        only_cam1 = sorted([obj_id for obj_id in cam1_objs if obj_id not in cam2_objs])
        only_cam2 = sorted([obj_id for obj_id in cam2_objs if obj_id not in cam1_objs])
        if len(only_cam1) == 0 or len(only_cam2) == 0:
            return qa
        # Find the {category: id} set of all appearing objects
        all_obj_cat = [(obj, objects[obj]['category']) for obj in objects]
        category_counts = Counter(category for obj_id, category in all_obj_cat)

        # 2. Generate the QA
        i = 0
        while len(qa) < max_num and i < max_try:
            i += 1
            # 2.1 Randomly choose an object A with a semantic orientation
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
            # Get the description of object A: if its category is unique in the scene, describe by category; otherwise find a description in cam1 and cam2 in order
            # The following could be refactored into a function; given many inputs, it outputs the description of any object in the scene as described above
            if category_counts[obj_with_orientation_cat] == 1:
                obj_with_orientation_des = obj_with_orientation_cat
                current_des = obj_with_orientation_cat
                cam_idx = 1 if obj_with_orientation in cam1_objs else 2
            else:
                # Determine which cameras the object appears in
                is_in_cam1 = obj_with_orientation in cam1_objs
                is_in_cam2 = obj_with_orientation in cam2_objs
                # Determine the list of cameras to try, randomly shuffled by priority
                cameras_to_try = []
                if is_in_cam1:
                    cameras_to_try.append(1)
                if is_in_cam2:
                    cameras_to_try.append(2)
                if not cameras_to_try:
                    # If the object is in neither filtered set, this should not happen in theory, but keep as a safeguard
                    continue
                # Randomly shuffle the attempt order
                random.shuffle(cameras_to_try)
                success = False
                # Iterate over the shuffled camera order
                for cam_idx in cameras_to_try:
                    if cam_idx == 1:
                        cam_data = cam1_data
                        fig_suffix = "in figure 1"
                    else:  # cam_idx == 2
                        cam_data = cam2_data
                        fig_suffix = "in figure 2"
                    # Try to generate the description from the current camera
                    bbox_2d = cam_data.get('objects', {}).get(obj_with_orientation, {}).get('bbox_2d')
                    # Assume get_description returns (success, _, description)
                    current_success, _, current_des = get_description(obj_with_orientation, cam_data, objects)
                    # Check whether it succeeded and is not too small
                    is_too_small = is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD)
                    if current_success and not is_too_small:
                        # The description was generated successfully and the object size is fine; use this description
                        obj_with_orientation_des = current_des + " " + fig_suffix
                        success = True
                        break  # break out of the loop once the first qualifying one is found
                # If both camera attempts fail, skip this loop iteration
                if not success:
                    continue
            # Randomly choose the text description of object A front-facing: mostly "front", otherwise random among the eight directions
            orientation_des = 'front' if random.random() < 0.4 else random.choice(self.direction_name)
            # Generate the premise text description
            if obj_with_orientation_cat in TEMPLATES['premise']:
                PREMISE_TEMPLATE = TEMPLATES['premise'][obj_with_orientation_cat]
            else:
                PREMISE_TEMPLATE = TEMPLATES['premise']['others']
            selected_template = random.choice(PREMISE_TEMPLATE)
            premise_text = selected_template.format(obj_des=obj_with_orientation_des, orientation_des=orientation_des)

            # 2.2 Randomly find the other objects in the question (a loop would improve the hit rate here)
            # Two modes
            if random.random() < 0.5:
                # The first asks in what direction object B is from object A
                # Selection: if object A appears in both cameras, choose randomly from only_cam1 and only_cam2; otherwise choose from the camera where A does not appear
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
                # The query vector is obj_B_loc - obj_A_loc, a 2D vector in the horizontal xy plane
                obj_tar_des = obj_B_des
                obj_ref_des = obj_with_orientation_des
                obj_B_loc = np.array(objects[obj_B]['3d_center'][:2])
                query_vec = obj_B_loc - obj_with_orientation_loc
                mode = '1'
            else:
                # The second asks in what direction object B is from object C
                # Selection: pick one from each of only_cam1 and only_cam2, then randomly assign their order
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

            # Refactor the answer generation: the function, given a vector direction already bound to a direction text description, finds the text description of another vector direction
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

            # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
            # 4.1 Prepare the metadata needed to generate level 1 and 2 questions

            # 4.2 First generate the level 2 questions
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                if not (level2_qa_type == "Object_correspondence" and obj_with_orientation not in common_ids):
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_with_orientation, current_des, "1,2"))
                    if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                        qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, obj_with_orientation, current_des, "2,1"))

            # 4.3 Then generate the level 1 questions
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
        # 1. Find objects appearing in both cameras as reference objects (ID intersection)
        # Extract the set of object IDs in both cameras
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        # Intersection: object IDs appearing in both cameras
        common_ids = sorted(list(cam1_objs & cam2_objs))  # the & operator on sets means intersection
        if not common_ids:
            return qa  # generate no questions when there is no common object

        # 2. Find objects appearing in only cam1 or only cam2 (without filtering by category)
        only_cam1 = sorted([obj_id for obj_id in cam1_objs if obj_id not in cam2_objs])
        only_cam2 = sorted([obj_id for obj_id in cam2_objs if obj_id not in cam1_objs])
        if len(only_cam1) == 0 or len(only_cam2) == 0:
            return qa

        i = 0
        all_objs = list(objects.keys())
        while len(qa) < max_num and i < max_try:
            i += 1

            # Choose reference object A
            ref_obj_A_id = random.choice(common_ids)
            bbox_2d = cam2_data.get('objects', {}).get(ref_obj_A_id, {}).get('bbox_2d')
            if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                continue
            success, common_obj_cat, common_obj_des = get_description(ref_obj_A_id, cam2_data, objects)
            if not success:
                continue
            ref_obj_A_loc = np.array(objects[ref_obj_A_id]['3d_center'][:2])
            ref_obj_A_des = f"{common_obj_des} in figure 2"

            # Choose reference object B
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
            # ref_obj_B must not be too close to ref_obj_A
            translation = np.linalg.norm(ref_obj_A_loc - ref_obj_B_loc)
            if translation < translation_threshold:
                continue

            # Choose the direction of reference object B relative to reference object A
            dir_B_to_A = random.choice(self.direction_name)
            # Choose the query direction
            dir_query = random.choice([d for d in self.direction_name if d != dir_B_to_A])

            # Iterate over all objects in the scene to find one correct option and three wrong options
            random.shuffle(all_objs)
            correct_option = []
            wrong_option = []
            options_obj = []
            for query_obj in all_objs:
                if query_obj == ref_obj_B['id'] or query_obj == ref_obj_A_id:
                    continue
                # query_obj must not be too close to ref_obj_A
                query_obj_loc = np.array(objects[query_obj]['3d_center'][:2])
                translation = np.linalg.norm(ref_obj_A_loc - query_obj_loc)
                if translation < translation_threshold:
                    continue
                # Get the description of query_obj
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
                # Check whether the object is in the query direction from reference object A
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
            # Generate the options
            options, answer = generate_shuffled_choices_text([info[1] for info in correct_option], [info[1] for info in wrong_option])
            # Generate the QA
            selected_template = random.choice(QUESTION_TEMPLATES) + '\nOptions: {options}'
            question = selected_template.format(ref_obj_A_des=ref_obj_A_des, ref_obj_B_des=ref_obj_B_des, main_direction=dir_B_to_A, query_direction=dir_query, options=options)

            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
            # 4.1 Prepare the metadata needed to generate level 1 and 2 questions

            # 4.2 First generate the level 2 questions
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_A_id, common_obj_des, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, ref_obj_A_id, common_obj_des, "2,1"))

            # 4.3 Then generate the level 1 questions
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

        # Filter object-level Reg: search the scene for objects that can represent a Reg. If 1, convert directly to a Reg description; otherwise check whether they are adjacent, in which case also convert to a Reg description and average coordinates; skip if separated
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
                # Check whether the objects in the list are adjacent; if they can form a cluster, count them as one area and combine their coordinates and bounding boxes; otherwise discard
                # Simplified detection: discard directly
                del areas[cat]
        # Filter room-level Reg: use the input rooms directly

        # 1. Find objects appearing in both cameras as reference objects (ID intersection)
        # Extract the set of object IDs in both cameras
        cam1_objs = {obj_id for obj_id in set(cam1_data['objects'].keys())}
        cam2_objs = {obj_id for obj_id in set(cam2_data['objects'].keys())}
        # Intersection: object IDs appearing in both cameras
        common_ids = list(cam1_objs & cam2_objs)  # the & operator on sets means intersection
        if not common_ids:
            return qa  # generate no questions when there is no common object

        # 2. Find objects appearing in only cam1 or only cam2 (without filtering by category)
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

            # Randomly choose between asking about a room and asking about an area
            if random.random() < 0.35 and areas:
                # Ask about an area: similar to Pos-Obj-Obj, treat the area as common_obj, everything else unchanged
                QUESTION_TEMPLATES = self.templates[level3_qa_type]['area']
                area_cat = random.choice(sorted(list(areas.keys())))
                objs = [
                    {'id': only_cam1_obj_id, 'des': f"{only_cam1_obj_des} in figure 1", 'loc': np.array(objects[only_cam1_obj_id]['3d_center'][:2])},
                    {'id': only_cam2_obj_id, 'des': f"{only_cam2_obj_des} in figure 2", 'loc': np.array(objects[only_cam2_obj_id]['3d_center'][:2])},
                    {'id': "1", 'des': random.choice(self.obj_area_map[area_cat]), 'loc': np.array(areas[area_cat]['3d_center'][:2])},
                ]
                random.shuffle(objs)
                # Ask: given the direction of objs[1] from objs[0], find the direction of objs[2] from objs[0]
                obj0_loc = objs[0]['loc']  # np.array(objects[objs[0]['id']]['3d_center'][:2])
                obj1_loc = objs[1]['loc']  # np.array(objects[objs[1]['id']]['3d_center'][:2])
                obj2_loc = objs[2]['loc']  # np.array(objects[objs[2]['id']]['3d_center'][:2])
                # Filter out combinations whose objects are too close together
                translation = np.linalg.norm(obj1_loc - obj0_loc)
                if translation < translation_threshold:
                    continue
                translation = np.linalg.norm(obj2_loc - obj0_loc)
                if translation < translation_threshold:
                    continue
            else:
                # Ask about a room
                if not rooms:
                    continue
                QUESTION_TEMPLATES = self.templates[level3_qa_type]['room']
                room_id = random.choice(sorted(list(rooms.keys())))
                objs = [
                    {'id': room_id, 'des': rooms[room_id]['category'], 'loc': np.array(rooms[room_id]['3d_center'][:2])},
                    {'id': only_cam1_obj_id, 'des': f"{only_cam1_obj_des} in figure 1", 'loc': np.array(objects[only_cam1_obj_id]['3d_center'][:2])},
                    {'id': only_cam2_obj_id, 'des': f"{only_cam2_obj_des} in figure 2", 'loc': np.array(objects[only_cam2_obj_id]['3d_center'][:2])},
                ]
                # Ask: given the direction of objs[1] from objs[0], find the direction of objs[2] from objs[0]
                obj0_loc = objs[0]['loc']  # np.array(objects[objs[0]['id']]['3d_center'][:2])
                obj1_loc = objs[1]['loc']  # np.array(objects[objs[1]['id']]['3d_center'][:2])
                obj2_loc = objs[2]['loc']  # np.array(objects[objs[2]['id']]['3d_center'][:2])
                # Filter out combinations whose objects are too close together
                translation = np.linalg.norm(obj1_loc - obj0_loc)
                if translation < translation_threshold + 0.6:
                    continue
                translation = np.linalg.norm(obj2_loc - obj0_loc)
                if translation < translation_threshold + 0.6:
                    continue
                # Check whether the object is inside the room
                if not is_object_center_in_room(only_cam1_obj_id, room_id, objects, rooms) or not is_object_center_in_room(only_cam2_obj_id, room_id, objects, rooms):
                    continue

            qa_group = []
            main_direction, rel_direction, direct_8 = get_relative_orientation(obj0_loc, obj1_loc, obj2_loc)
            # Generate the options
            options, answer = generate_orientation_choices(rel_direction, direct_8)
            # Generate the QA
            selected_template = random.choice(QUESTION_TEMPLATES)
            if self.if_MCA:
                selected_template += '\nOptions: {options}'
            else:
                answer = answer[3:]
            question = selected_template.format(obj1_des=objs[1]['des'], obj0_des=objs[0]['des'], obj2_des=objs[2]['des'], main_direction=main_direction, options=options)
            qa_group.append((question, answer, level3_qa_type, "1,2"))

            # 4. Generate level 1 and 2 questions that this level 3 question capability depends on
            # 4.1 Prepare the metadata needed to generate level 1 and 2 questions

            # 4.2 First generate the level 2 questions
            level2_qa_types = self.qa_dependency_tree[level3_qa_type]['level2']
            for level2_qa_type in level2_qa_types:
                qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, only_cam1_obj_id, only_cam1_obj_des, "1,2"))
                if level2_qa_type not in ["Cam_trans_forward", "Cam_trans_right", "Cam_rot_yaw", "Cam_rot_pitch"]:
                    qa_group.append(self.level2_qa_functions[level2_qa_type]['func'](objects, cam1_data, cam2_data, only_cam1_obj_id, only_cam1_obj_des, "2,1"))

            # 4.3 Then generate the level 1 questions
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
        translation_threshold = 1  # translation threshold to ensure enough displacement between cameras
        rotation_max_threshold = 120

        # Ensure there are at least three camera data entries for T(arget), C(urrent), A(nchor)
        if len(cameras) < 3:
            return qa

        chosen_indices = random.sample(range(len(cameras)), 3)

        # 2. Randomly assign roles: T (Target), C (Current), A (Anchor)
        #    We use randomly chosen indices and map them back to the original cameras list
        idx_target = chosen_indices[0]  # index used as Photo T
        idx_current = chosen_indices[1]  # index used as Photo C
        idx_anchor = chosen_indices[2]  # index used as Photo A

        # 3. Extract the corresponding camera data (using the original list indices)
        cam_target_data = cameras[idx_target]
        cam_current_data = cameras[idx_current]
        cam_anchor_data = cameras[idx_anchor]

        # Note: we should use the user-visible camera number (usually 1-based)
        cam_target_num = idx_target + 1
        cam_current_num = idx_current + 1
        cam_anchor_num = idx_anchor + 1

        # 4. Extract the positions and orientations
        cam_target_loc = np.array([cam_target_data['location_3d']['x'], cam_target_data['location_3d']['y']])
        cam_current_loc = np.array([cam_current_data['location_3d']['x'], cam_current_data['location_3d']['y']])
        cam_target_for = np.array([cam_target_data['forward_direction']['x'], cam_target_data['forward_direction']['y']])
        cam_current_for = np.array([cam_current_data['forward_direction']['x'], cam_current_data['forward_direction']['y']])
        cam_anchor_for = np.array([cam_anchor_data['forward_direction']['x'], cam_anchor_data['forward_direction']['y']])

        # Filter out pairs with too little translation
        translation = np.linalg.norm(cam_target_loc - cam_current_loc)
        if translation < translation_threshold:
            return qa
        # Filter out camera pairs with excessive rotation angle
        rel_angel = calculate_rotation(cam_target_for, cam_current_for)
        if abs(rel_angel) > rotation_max_threshold:
            return qa

        # 5. Compute the relative displacement vector (fix cam_anchor_for as a random absolute direction; compute the absolute direction of cam_target_loc relative to cam_current_loc)
        main_direction, rel_direction = get_absolute_direction(cam_current_loc, cam_target_loc, cam_anchor_for, thresh=translation_threshold)

        options, answer = generate_orientation_choices(rel_direction)

        # 6. Generate the question
        selected_template = random.choice(QUESTION_TEMPLATES)
        if self.if_MCA:
            selected_template += '\nOptions: {options}'
        else:
            answer = answer[3:]
        question = selected_template.format(
            # Use a random, 1-based camera number
            cam_anchor_idx=cam_anchor_num,  # Photo A
            anchor_dir=main_direction,  # e.g. East
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
        Randomly choose a camera as the reference view, and randomly find an object that exists in another view but not in the reference view.
        Ask about the object's direction relative to the reference view.
        Similar to Pos-Cam-Obj
        """
        qa = []
        level3_qa_type = "MSR_Cam_Obj"
        QUESTION_TEMPLATES = self.templates[level3_qa_type]
        translation_threshold = 0.15
        distance_threshold = 0.5
        # Get the reference view
        ref_cam_index = random.randrange(len(cameras))
        ref_cam_num = ref_cam_index + 1
        ref_cam_data = cameras[ref_cam_index]
        ref_cam_loc = np.array([ref_cam_data['location_3d']['x'], ref_cam_data['location_3d']['y'], ref_cam_data['location_3d']['z']])
        ref_cam_for = np.array([ref_cam_data['forward_direction']['x'], ref_cam_data['forward_direction']['y'], ref_cam_data['forward_direction']['z']])
        ref_objs_set = set(ref_cam_data['objects'].keys())
        ref_objs_cat = {objects[obj]['category'] for obj in ref_objs_set}
        # Get the set of object IDs not in the reference view
        other_cams = [cam for cam in cameras if cam != ref_cam_data]
        seen_ids = set()
        other_objs_id = []
        for other_cam in other_cams:
            for obj_id in other_cam['objects']:
                # 1. Ensure the ID has not been processed yet (avoid duplicate processing for efficiency)
                if obj_id in seen_ids:
                    continue
                # 2. Exclude objects in the reference view
                if obj_id in ref_objs_set:
                    continue
                # 3. Exclude unwanted categories
                category = objects[obj_id]['category']
                # 4. Filter by size
                bbox_2d = other_cam.get('objects', {}).get(obj_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                # 5. Passed all checks; record the ID and category and mark as processed
                other_objs_id.append((obj_id, category))
                seen_ids.add(obj_id)

        # Filter objects whose category appears only once: this should be judged together with the reference view, otherwise a category could occur once outside the reference view but many times inside it
        # So add a step: the chosen object category must not appear in the reference-view object categories
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

        # Filter out pairs with too little translation
        translation = np.linalg.norm(ref_cam_loc - obj_loc)
        if translation < distance_threshold:
            return qa

        forward_dir, right_dir = get_relative_direction(ref_cam_loc, obj_loc, ref_cam_for, thresh=translation_threshold)
        if forward_dir == "" and right_dir == "":
            return qa

        # 2. Format the final question
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

        # 1. Count the set of object IDs appearing in these views
        obj_category = []
        for cam in cameras:
            obj_category += [(obj_id, objects[obj_id]['category']) for obj_id in cam.get('objects', {}).keys()]
        unique_objects = list(set(obj_category))
        unique_categories = [category for obj_id, category in unique_objects]
        category_counts = Counter(unique_categories)
        category = random.choice(sorted(list(category_counts.keys())))
        unique_count = int(category_counts[category])
        # Randomly drop questions whose answer is 1
        if random.random() < 0.65 and unique_count == 1:
            return qa

        options, answer = generate_random_int_choices(unique_count)

        # Generate the QA
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

        # 1. Find the set of objects whose category is unique among all camera-appearing objects, and remove these from each camera object set.
        seen_ids = set()
        all_objs_id = []
        for cam in cameras:
            for obj_id in cam['objects']:
                # 1. Ensure the ID has not been processed yet (avoid duplicate processing for efficiency)
                if obj_id in seen_ids:
                    continue
                # 2. Exclude unwanted categories
                category = objects[obj_id]['category']
                # 3. Filter by size
                bbox_2d = cam.get('objects', {}).get(obj_id, {}).get('bbox_2d')
                if is_object_too_small(bbox_2d, self.MIN_AREA_THRESHOLD, self.MIN_SIDE_THRESHOLD):
                    continue
                # 4. Passed all checks; record the ID and category and mark as processed
                all_objs_id.append((obj_id, category))
                seen_ids.add(obj_id)

        # Filter objects whose category appears only once: this should be judged together with the reference view, otherwise a category could occur once outside the reference view but many times inside it
        # So add a step: the chosen object category must not appear in the reference-view object categories
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

        # 2. Randomly sample three objects. For each object, randomly use one of two ways: sample from the category-unique object set (description is the category), or first sample a camera and then sample an object among those appearing in it to generate a description.
        objs = []
        selected_ids = set()
        i = 0
        while len(objs) < 3:
            i += 1
            if i >= max_try:
                return qa
            temp_obj_id = None
            temp_description = None
            # Sample from the category-unique object set
            if random.random() < 0.5:
                if single_occurrence_objects:
                    temp_obj_id, category = random.choice(single_occurrence_objects)
                    temp_description = category
            # Sample from a given camera object set
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
            # Ensure both the ID and description were sampled successfully
            if temp_obj_id and temp_description:
                # Check whether the ID was already selected
                if temp_obj_id not in selected_ids:
                    # 1. Add to the final result list
                    objs.append({'id': temp_obj_id, 'des': temp_description})

                    # 2. Add the ID to the set to prevent repeats next time
                    selected_ids.add(temp_obj_id)

        # 3. After sampling three objects, generate the question the same way as before.
        # Ask: given the direction of objs[0] from objs[1], find the direction of objs[2] from objs[1]
        obj0_loc = np.array(objects[objs[0]['id']]['3d_center'][:2])
        obj1_loc = np.array(objects[objs[1]['id']]['3d_center'][:2])
        obj2_loc = np.array(objects[objs[2]['id']]['3d_center'][:2])
        # Filter out combinations whose objects are too close together, to avoid objects stacked vertically yet asked about horizontal direction
        translation = np.linalg.norm(obj1_loc - obj0_loc)
        if translation < translation_threshold:
            return qa
        translation = np.linalg.norm(obj2_loc - obj0_loc)
        if translation < translation_threshold:
            return qa
        main_direction, rel_direction, direct_8 = get_relative_orientation(obj0_loc, obj1_loc, obj2_loc)
        # Generate the options
        options, answer = generate_orientation_choices(rel_direction, direct_8)
        # Generate the QA
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
        Filter all objects appearing in cam1 **or** cam2 (return full info).
        :param cam1_objects: objects in the cam1 view (keys match all_objects)
        :param cam2_objects: objects in the cam2 view (keys match all_objects)
        :param all_objects: full info of all objects in the scene
        :return: all objects appearing in cam1 or cam2 (key is the object identifier, value is the full info)
        """
        # Extract the keys (unique identifiers) of the objects in both cameras
        cam1_keys = set(cam1_objects.keys())
        cam2_keys = set(cam2_objects.keys())

        # Union: object keys appearing in cam1 or cam2
        either_keys = cam1_keys | cam2_keys  # | means union (not the & intersection)

        # Extract the full info for these keys from the global objects
        objects_in_either = {
            obj_key: all_objects[obj_key]
            for obj_key in sorted(list(either_keys))
            if obj_key in all_objects  # validate the key
        }

        return objects_in_either

    # ------------------------------
    # Core processing logic (save per question type)
    # ------------------------------
    def process_single_scene(self, scene_dir: Path) -> Dict[str, Any]:
        """Process a single scene, returning the QA grouped by question type."""
        scene_name = scene_dir.name
        # print(f"\n===== Processing scene: {scene_name} =====")

        # Read the metadata
        metadata = load_config(scene_dir / self.metadata_filename)
        if not metadata:
            return {scene_name: {qa_type: [] for qa_type in self.qa_types}}

        cameras = metadata.get("cameras", {})
        objects = metadata.get("objects", {})
        if not cameras:
            print("场景无相机数据，跳过")
            return {scene_name: {qa_type: [] for qa_type in self.qa_types}}
        cameras = {
            cam_name: cam_data
            for cam_name, cam_data in cameras.items()
            if cam_data is not None
        }

        # Category filtering
        objects = {obj: objects[obj] for obj in list(objects.keys()) if not any(unwanted in objects[obj]['category'] for unwanted in self.unwanted_cats)}

        # Read all room data
        rooms = {}
        for obj in list(objects.keys()):
            if objects[obj]['category'] in self.room_type:
                rooms[obj] = objects[obj]
                del objects[obj]

        # Generate the camera pairs
        camera_pairs = self._get_camera_pairs(cameras)
        # print(f"Generated {len(camera_pairs)} camera pairs")

        # Collect QAs by question type
        scene_qa_by_type = {qa_type: [] for qa_type in list(self.level3_qa_types.keys())}
        for cam1_name, cam2_name, cam1_data, cam2_data in camera_pairs:
            camera_qa_by_type = {qa_type: [] for qa_type in list(self.level3_qa_types.keys())}
            # Added: check whether the camera data is None (i.e. null)
            if cam1_data is None or cam2_data is None:
                print("警告：存在相机元数据为 null，跳过当前场景处理")
                continue  # skip the rest and return directly
            # Category filtering
            cam1_data['objects'] = {obj: cam1_data['objects'][obj] for obj in list(cam1_data['objects'].keys()) if obj in objects}
            cam2_data['objects'] = {obj: cam2_data['objects'][obj] for obj in list(cam2_data['objects'].keys()) if obj in objects}
            # Filter out camera groups whose position and orientation nearly coincide
            cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])
            cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
            cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])
            cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])
            common_ids = cam1_data['objects'].keys() & cam2_data['objects'].keys()  # the & operator on sets means intersection
            if self.data_source == "scannetpp":
                if len(common_ids) < self.common_num_threshold or should_filter_camera_pair_strong(cam1_loc, cam1_for, cam2_loc, cam2_for):
                    continue
            else:
                if should_filter_camera_pair(cam1_loc, cam1_for, cam2_loc, cam2_for):
                    continue
            # Filter the objects that appear in both cam1 and cam2
            objects_in_cams = self.get_objects_in_either_cam(cam1_data["objects"], cam2_data["objects"], objects)
            # Iterate over all question types
            for qa_type, config in self.qa_types.items():
                qa_func = config.generator
                need_order = config.needs_swap
                max_qa_num = config.max_per_pair
                sampling_rate = config.sampling_rate
                # try:
                # Generate the QA for this type
                qas = qa_func(cam1_data, cam2_data, objects_in_cams, rooms)
                if len(qas) > max_qa_num:
                    qas = random.sample(qas, max_qa_num)
                if self.data_source == "scannetpp":
                    image1_path = f"{self.training_environment_base_dir}/{cam1_data['image_path']}"
                    image2_path = f"{self.training_environment_base_dir}/{cam2_data['image_path']}"
                else:
                    image1_path = f"{self.training_environment_base_dir}/{scene_name}/{os.path.basename(cam1_data['image_path'])}"
                    image2_path = f"{self.training_environment_base_dir}/{scene_name}/{os.path.basename(cam2_data['image_path'])}"
                # Record the QA, including scene and camera-pair info
                for qa_groups in qas:
                    multilevel_qa_groups = []
                    conversation = {
                        "messages": [],
                        "images": [image1_path, image2_path],
                        "category": qa_type,
                        "scene_name": scene_name,
                        "data_source": self.data_source,
                    }
                    messages = []
                    qa_groups = list(reversed(qa_groups))
                    for index, (q, a, type, camera_order) in enumerate(qa_groups):
                        if self.multilevel_qa_mode == "atomic":
                            if camera_order == "1":
                                q_content = f"<image>{q}\n"
                                image_list = [image1_path]
                            elif camera_order == "2":
                                q_content = f"<image>{q}\n"
                                image_list = [image2_path]
                            elif camera_order == "1,2":
                                q_content = f"<image><image>{q}\n"
                                image_list = [image1_path, image2_path]
                            multilevel_qa_groups.append({
                                "messages": [
                                    {"role": "user", "content": q_content},  # Answer with the option's letter from the given options directly.
                                    {"role": "assistant", "content": a}
                                ],
                                "images": image_list,
                                "category": type,
                                "scene_name": scene_name,
                                "data_source": self.data_source,
                            })
                        else:
                            # === Conversation-format training data ===
                            if type in self.level3_qa_types.keys():
                                level_num = 3
                            elif type in self.level2_qa_functions.keys():
                                level_num = 2
                            else:
                                level_num = 1
                            if index == 0:
                                # Insert a background prompt before the first User message
                                header = (
                                    "I have provided two images captured by a camera in a 3D environment. "
                                    "I will ask you a series of questions to test your spatial perception and reasoning abilities. "
                                    "Please analyze the visual content carefully.\n\n"
                                )
                                q_content = f"{header}<image><image>Question {index+1}(Level {level_num}:{type}): {q}"
                            else:
                                # Subsequent turns no longer carry the <image> token or header
                                q_content = f"Question {index+1}(Level {level_num}:{type}): {q}"
                            messages.extend([
                                    {"role": "user", "content": q_content},  # Answer with the option's letter from the given options directly.
                                    {"role": "assistant", "content": f"Answer {index+1}: {a}"}
                                ])
                    conversation['messages'].extend(messages)
                    if self.multilevel_qa_mode == "atomic":
                        camera_qa_by_type[qa_type].append(multilevel_qa_groups)
                    else:
                        camera_qa_by_type[qa_type].append(conversation)
                if need_order:
                    # Swap the camera order and generate the QA for this type
                    qas = qa_func(cam2_data, cam1_data, objects_in_cams, rooms)
                    if len(qas) > max_qa_num:
                        qas = random.sample(qas, max_qa_num)
                    # Record the QA, including scene and camera-pair info
                    for qa_groups in qas:
                        multilevel_qa_groups = []
                        conversation = {
                            "messages": [],
                            "images": [image2_path, image1_path],
                            "category": qa_type,
                            "scene_name": scene_name,
                            "data_source": self.data_source,
                        }
                        messages = []
                        qa_groups = list(reversed(qa_groups))
                        for index, (q, a, type, camera_order) in enumerate(qa_groups):
                            if self.multilevel_qa_mode == "atomic":
                                if camera_order == "1":
                                    q_content = f"<image>{q}\n"
                                    image_list = [image2_path]
                                elif camera_order == "2":
                                    q_content = f"<image>{q}\n"
                                    image_list = [image1_path]
                                elif camera_order == "1,2":
                                    q_content = f"<image><image>{q}\n"
                                    image_list = [image2_path, image1_path]
                                multilevel_qa_groups.append({
                                    "messages": [
                                        {"role": "user", "content": q_content},  # Answer with the option's letter from the given options directly.
                                        {"role": "assistant", "content": a}
                                    ],
                                    "images": image_list,
                                    "category": type,
                                    "scene_name": scene_name,
                                    "data_source": self.data_source,
                                })
                            else:
                                # === Conversation-format training data ===
                                if type in self.level3_qa_types.keys():
                                    level_num = 3
                                elif type in self.level2_qa_functions.keys():
                                    level_num = 2
                                else:
                                    level_num = 1
                                if index == 0:
                                    # Insert a background prompt before the first User message
                                    header = (
                                        "I have provided two images captured by a camera in a 3D environment. "
                                        "I will ask you a series of questions to test your spatial perception and reasoning abilities. "
                                        "Please analyze the visual content carefully.\n\n"
                                    )
                                    q_content = f"{header}<image><image>Question {index+1}(Level {level_num}:{type}): {q}"
                                else:
                                    # Subsequent turns no longer carry the <image> token or header
                                    q_content = f"Question {index+1}(Level {level_num}:{type}): {q}"
                                messages.extend([
                                    {"role": "user", "content": q_content},  # Answer with the option's letter from the given options directly.
                                    {"role": "assistant", "content": f"Answer {index+1}: {a}"}
                                ])
                        conversation['messages'].extend(messages)
                        if self.multilevel_qa_mode == "atomic":
                            camera_qa_by_type[qa_type].append(multilevel_qa_groups)
                        else:
                            camera_qa_by_type[qa_type].append(conversation)
            # Sample the question categories for a group of camera pairs
            if self.qa_types_per_camera_pair == 0:
                for qa_type in list(self.level3_qa_types.keys()):
                    scene_qa_by_type[qa_type].extend(camera_qa_by_type[qa_type])
            else:
                selected_qa_types = sorted(list(self.level3_qa_types.keys()))
                random.shuffle(selected_qa_types)
                selected_num = 0
                for qa_type in selected_qa_types:
                    if camera_qa_by_type[qa_type]:
                        scene_qa_by_type[qa_type].extend(camera_qa_by_type[qa_type])
                        selected_num += 1
                    if selected_num == self.qa_types_per_camera_pair:
                        break

        # Sample for each question category
        # if sampling_rate != 1:
        #     scene_qa_by_type[qa_type] = random.sample(scene_qa_by_type[qa_type], k=round(len(scene_qa_by_type[qa_type]) * sampling_rate))
        # except Exception as e:
        #     print(f"  Failed to generate {qa_type} for pair {cam1_name}&{cam2_name}: {e}")

        # MSR complex reasoning question types
        max_available = len(cameras)
        for qa_type, config in self.msr_qa_types.items():
            qa_func = config.generator
            min_views, max_views = config.num_of_views_range
            max_num = config.max_num
            sampling_rate = config.sampling_rate
            while len(scene_qa_by_type[qa_type]) < max_num:
                # Randomly pick num_of_views camera views
                num_of_views = random.randint(min_views, min(max_views, max_available))
                all_cameras_name = list(cameras.keys())
                random_cameras_name = random.sample(all_cameras_name, num_of_views)
                random_cameras = [cameras[cam] for cam in random_cameras_name]
                # Category filtering
                for camera in random_cameras:
                    camera['objects'] = {obj: camera['objects'][obj] for obj in list(camera['objects'].keys()) if obj in objects}
                qas = qa_func(random_cameras, objects, rooms) # returns a list
                if not qas:
                    continue
                if self.data_source == "scannetpp":
                    images_list = [f"{self.training_environment_base_dir}/{cameras[cam]['image_path']}" for cam in random_cameras_name]
                else:
                    images_list = [f"{self.training_environment_base_dir}/{scene_name}/{os.path.basename(cameras[cam]['image_path'])}" for cam in random_cameras_name]
                multilevel_qa_groups = []
                conversation = {
                    "messages": [],
                    "images": images_list,
                    "category": qa_type,
                    "scene_name": scene_name,
                    "data_source": self.data_source,
                }
                messages = []
                qas = list(reversed(qas))
                for index, (q, a, type, camera_order) in enumerate(qas):
                    if self.multilevel_qa_mode == "atomic":
                        multilevel_qa_groups.append({
                            "messages": [
                                {"role": "user", "content": f"{'<image>' * num_of_views}{q}\n"},  # Answer with the option's letter from the given options directly.
                                {"role": "assistant", "content": a}
                            ],
                            "images": images_list,
                            "category": qa_type,
                            "scene_name": scene_name,
                            "data_source": self.data_source,
                        })
                    else:
                        if index == 0:
                            # Insert a background prompt before the first User message
                            header = (
                                "I have provided two images captured by a camera in a 3D environment. "
                                "I will ask you a series of questions to test your spatial perception and reasoning abilities. "
                                "Please analyze the visual content carefully.\n\n"
                            )
                            q_content = f"{header}<image><image>Question: {q}"
                        else:
                            # Subsequent turns no longer carry the <image> token or header
                            q_content = q
                        messages.extend([
                            {"role": "user", "content": q_content},  # Answer with the option's letter from the given options directly.
                            {"role": "assistant", "content": a}
                        ])
                conversation['messages'].extend(messages)
                if self.multilevel_qa_mode == "atomic":
                    scene_qa_by_type[qa_type].append(multilevel_qa_groups)
                else:
                    scene_qa_by_type[qa_type].append(conversation)
            # Sample for each question category
            # if sampling_rate != 1:
            #     scene_qa_by_type[qa_type] = random.sample(scene_qa_by_type[qa_type], k=round(len(scene_qa_by_type[qa_type]) * sampling_rate))

        return {scene_name: scene_qa_by_type}

    def process_all_scenes(self):
        """Process all scenes, saving each question type to a separate JSON file."""
        # Globally collect the QAs of all scenes by question type
        all_qa_keys = list(self.all_qa_types.keys())
        level3_qa_keys = list(self.level3_qa_types.keys())
        global_qa_by_type = {qa_type: [] for qa_type in all_qa_keys}
        qa_groups_by_type = {qa_type: [] for qa_type in level3_qa_keys}

        # Process each scene and aggregate
        pbar = tqdm(self.scene_folders, desc="总进度", unit="场景", leave=True)
        for scene_dir in pbar:
            # Get which stage the scene belongs to
            scene_result = self.process_single_scene(scene_dir)
            scene_name, scene_qa = next(iter(scene_result.items()))  # get the scene name and QAs; returns the question groups led by level 3 question types
            # tqdm.write(f"\n===== Start processing scene: {scene_name} =====")
            pbar.set_description(f"处理中: {scene_name}")
            # Store the question groups by level 3 category
            for qa_type, qas in scene_qa.items():
                qa_groups_by_type[qa_type].extend(qas)

        # Sample the question groups
        for qa_type, qa_groups in qa_groups_by_type.items():
            sampling_rate = self.level3_qa_types[qa_type].sampling_rate
            if sampling_rate != 1:
                qa_groups_by_type[qa_type] = random.sample(qa_groups, k=round(len(qa_groups) * sampling_rate))

        if self.multilevel_qa_mode == "atomic":
            # Split the question groups across all levels and categories
            for qa_type, qa_groups in qa_groups_by_type.items():
                for qa_group in qa_groups:
                    for qa in qa_group:
                        global_qa_by_type[qa['category']].append(qa)
            # Sample the level 1 and 2 questions
            for qa_type in list(self.level1_qa_functions.keys()):
                global_qa_by_type[qa_type] = random.sample(global_qa_by_type[qa_type], k=round(len(global_qa_by_type[qa_type]) * self.level12_sampling_rate))
            for qa_type in list(self.level2_qa_functions.keys()):
                global_qa_by_type[qa_type] = random.sample(global_qa_by_type[qa_type], k=round(len(global_qa_by_type[qa_type]) * self.level12_sampling_rate))  # self.level1_qa_functions[qa_type].sampling_rate
        else:
            global_qa_by_type = qa_groups_by_type



        # Save to different JSON files by question type and by level
        all_qas_combined = []
        level3_qas_combined = []
        level2_qas_combined = []
        level1_qas_combined = []
        for qa_type, all_qas in global_qa_by_type.items():
            all_qas_combined.extend(all_qas)
            if qa_type in list(self.level3_qa_types.keys()):
                subdir = 'level_3'
                level3_qas_combined.extend(all_qas)
            elif qa_type in list(self.level1_qa_functions.keys()):
                subdir = 'level_1'
                level1_qas_combined.extend(all_qas)
            elif qa_type in list(self.level2_qa_functions.keys()):
                subdir = 'level_2'
                level2_qas_combined.extend(all_qas)
            output_path = self.output_dir / self.multilevel_qa_mode / subdir / f"{qa_type}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_json_data(all_qas, output_path, f"{qa_type} 类型数据")

        # if not self.if_flipping_enhencement:
        #     all_qas_combined *= 2

        # Save the combined training data
        all_combined_output_path = self.output_dir / self.multilevel_qa_mode / f"{self.version_name}_{self.multilevel_qa_mode}.json"
        level3_combined_output_path = self.output_dir / self.multilevel_qa_mode / "level_3" / f"{self.version_name}_{self.multilevel_qa_mode}_level3.json"
        level2_combined_output_path = self.output_dir / self.multilevel_qa_mode / "level_2" / f"{self.version_name}_{self.multilevel_qa_mode}_level2.json"
        level1_combined_output_path = self.output_dir / self.multilevel_qa_mode / "level_1" / f"{self.version_name}_{self.multilevel_qa_mode}_level1.json"
        save_json_data(all_qas_combined, all_combined_output_path, f"所有类型合并")
        save_json_data(level3_qas_combined, level3_combined_output_path, f"level 3合并")
        save_json_data(level2_qas_combined, level2_combined_output_path, f"level 2合并")
        save_json_data(level1_qas_combined, level1_combined_output_path, f"level 1合并")

        # Save the data ratios
        config_output_path = self.output_dir / f"qa_config.json"
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

        # Compute the option distribution
        # if self.if_MCA:
        #     self.calculate_options_proportions(all_qas_combined)

        return global_qa_by_type

    def repartition_data_by_stage(self):
        # --- Added: initialize the statistics dict ---
        stage1_stats = {
            "Level 1": {"total": 0, "categories": {}},
            "Level 2": {"total": 0, "categories": {}},
            "Level 3": {"total": 0, "categories": {}}
        }
        # Lookup table: map categories to Levels
        cat_to_level = {}
        for c in self.level1_qa_functions.keys(): cat_to_level[c] = "Level 1"
        for c in self.level2_qa_functions.keys(): cat_to_level[c] = "Level 2"
        # Note: if Level 3 contains categories not in the first two levels, assign them to Level 3
        for c in self.level3_qa_types.keys():
            if c not in cat_to_level: cat_to_level[c] = "Level 3"

        # Given that the atomic and conversation data are already generated from the same batch, read the generated data, split by training stage, and re-save
        stage1_output_path = self.output_dir / f"{self.version_name}_stage1.json"
        stage2_output_path = self.output_dir / f"{self.version_name}_stage2.json"
        stage3_output_path = self.output_dir / f"{self.version_name}_stage3.json"
        s1_l1_path = self.output_dir / f"{self.version_name}_stage1_level1.json"
        s1_l2_path = self.output_dir / f"{self.version_name}_stage1_level2.json"
        s1_l3_path = self.output_dir / f"{self.version_name}_stage1_level3.json"
        stage1_dataset = []
        stage1_l1_dataset = []
        stage1_l2_dataset = []
        stage1_l3_dataset = []
        stage2_dataset = []
        stage3_dataset = []

        total_scenes = len(self.scene_names)

        # Compute the split points from the ratios
        idx_sft_end = int(total_scenes * self.stage_1_proportion)
        idx_cold_start_end = idx_sft_end + int(total_scenes * self.stage_2_proportion)

        # Physically isolate the scene pools
        scene_pools = {
            "SFT": self.scene_names[:idx_sft_end],
            "ColdStart": self.scene_names[idx_sft_end:idx_cold_start_end],
            "RL": self.scene_names[idx_cold_start_end:]
        }

        # Read the atomic and conversation data of each category in turn, and save per stage by scene.
        # Read the atomic full-set json
        atomic_output_path = self.output_dir / "atomic" / f"{self.version_name}_atomic.json"
        atomic_data = load_config(atomic_output_path)
        for data in atomic_data:
            cat = data['category']
            if data['scene_name'] in scene_pools["SFT"] and cat in (list(self.qa_types.keys())+list(self.level1_qa_functions.keys())+list(self.level2_qa_functions.keys())):
                stage1_dataset.append(data)
                # --- Added: Stage 1 statistics logic ---
                target_level = cat_to_level.get(cat, "Level 1")  # default category in case any is missed
                stage1_stats[target_level]["total"] += 1
                stage1_stats[target_level]["categories"][cat] = stage1_stats[target_level]["categories"].get(cat, 0) + 1
                if target_level == "Level 1":
                    stage1_l1_dataset.append(data)
                elif target_level == "Level 2":
                    stage1_l2_dataset.append(data)
                elif target_level == "Level 3":
                    stage1_l3_dataset.append(data)
            elif data['scene_name'] in scene_pools["RL"] and data["category"] in self.level3_qa_types.keys():
                stage3_dataset.append(data)

        # Read the conversation full-set json
        # conversation_output_path = self.output_dir / "conversation" / f"{self.version_name}_conversation.json"
        # conversation_data = load_config(conversation_output_path)
        # for data in conversation_data:
        #     if data['scene_name'] in scene_pools["ColdStart"] and data["category"] in self.qa_types.keys():
        #         stage2_dataset.append(data)

        # --- Added: print the statistics so you can paste the values into the plotting code ---
        print("\n" + "=" * 30 + " Stage 1 Data Distribution " + "=" * 30)
        for lvl, info in stage1_stats.items():
            print(f"[{lvl}] Total: {info['total']}")
            # Print sorted by category count in descending order
            sorted_cats = sorted(info['categories'].items(), key=lambda x: x[1], reverse=True)
            for c_name, c_count in sorted_cats:
                print(f"  - {c_name}: {c_count}")
        print("=" * 70 + "\n")

        save_json_data(stage1_dataset, stage1_output_path, f"stage 1 数据保存")
        save_json_data(stage2_dataset, stage2_output_path, f"stage 2 数据保存")
        save_json_data(stage3_dataset, stage3_output_path, f"stage 3 数据保存")
        save_json_data(stage1_l1_dataset, s1_l1_path, "stage 1 Level 1 数据保存")
        save_json_data(stage1_l2_dataset, s1_l2_path, "stage 1 Level 2 数据保存")
        save_json_data(stage1_l3_dataset, s1_l3_path, "stage 1 Level 3 数据保存")

    def save_all_categories(self):
        categories = {}

        # 1. Instantiate tqdm as pbar (progress bar)
        pbar = tqdm(self.scene_folders, desc="总进度", unit="场景", leave=True)

        for scene_dir in pbar:
            scene_name = scene_dir.name

            # 2. Correctly use the instance method to update the description
            pbar.set_description(f"处理中: {scene_name}")

            # Read the metadata
            metadata = load_config(scene_dir / self.metadata_filename)

            # 3. Fix a logic bug: this should be continue, not return
            # return would abort the whole statistics task
            if not metadata:
                print(f"警告：场景 {scene_name} 无法读取元数据，跳过")
                continue

            objects = metadata.get("objects", {})
            for obj_id in objects.keys():
                # Simplify the dict counting logic
                cat_name = objects[obj_id]['category']
                categories[cat_name] = categories.get(cat_name, 0) + 1

        output_path = self.output_dir / "categories_statistics.json"
        try:
            # Ensure the output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(categories, f, indent=4, ensure_ascii=False)
            print(f"\n统计完成！结果已保存至: {output_path}")
        except Exception as e:
            print(f"保存失败：{e}")

    def calculate_options_proportions(self, qa_list):
        answers = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for qa in qa_list:
            answers[qa['messages'][1]['content'][0]] += 1
        for option in answers:
            count = answers[option]
            percentage = (count / len(qa_list)) * 100
            logging.info(f"  选项 {option}: {count} 条 ({percentage:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the multilevel/multistage QA generator for one config under configs/qa/."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/qa/qa_config_infinigen.json",
        help="Path to the QA config JSON (e.g. configs/qa/qa_config_scannetpp.json). "
             "Relative paths are resolved against the current working directory; run from the repo root. "
             "Default: %(default)s",
    )
    args = parser.parse_args()

    # Run the generator
    generator = SceneQAGenerator(
        config_path=args.config
    )
    # generator.save_all_categories()
    generator.process_all_scenes()
    generator.repartition_data_by_stage()
