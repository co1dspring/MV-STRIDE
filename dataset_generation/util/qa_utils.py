# -*- coding: utf-8 -*-
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

        # Match the numbers inside the brackets
        # \d+ matches one or more digits
        matches = re.findall(r'\d+', HW_str)

        if len(matches) >= 2:
            # Assume the first is height, the second is width
            H = int(matches[0])
            W = int(matches[1])

        else:
            print("未能正确解析字符串")
    return H, W

def format_bbox_dict_to_str(bbox_dict, H, W):
    """
    Normalize an absolute-coordinate bbox dict and convert it to the string "[min_x, min_y, max_x, max_y]".
    Range: 0-1000
    """
    # Normalize and clamp to the 0-1000 range
    # use W (width) for the x axis and H (height) for the y axis
    min_x = min(1000, max(0, round(bbox_dict['min_x'] / W * 1000)))
    min_y = min(1000, max(0, round(bbox_dict['min_y'] / H * 1000)))
    max_x = min(1000, max(0, round(bbox_dict['max_x'] / W * 1000)))
    max_y = min(1000, max(0, round(bbox_dict['max_y'] / H * 1000)))

    # Arrange in the order [x1, y1, x2, y2]
    res = [min_x, min_y, max_x, max_y]

    # Return the string form, e.g. "[123, 456, 789, 900]"
    return str(res)

def int_to_simple_ordinal_word(n: Union[int, str]) -> str:
    """
    Convert an integer from 1 to 10 into its English ordinal word (first, second, ..., tenth).

    Args:
        n: the integer, or a string form of the number, to convert.

    Returns:
        The corresponding English ordinal, or an empty string if out of range [1, 10] or invalid.
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
        return ""  # non-numeric input

    return ORDINAL_MAP.get(n_int, "")

def fill_template_placeholders(template: str, is_case_1: bool) -> str:
    """
    Fill the language indicators in the template based on the role assignment (Case 1 or 2).

    :param is_case_1: True for Case 1 (C2 is reference, C1 is target)
                      False for Case 2 (C1 is reference, C2 is target)
    """
    second_or_last = "second" if random.random() < 0.8 else "last"
    if is_case_1:
        # Case 1: C1 (first) is the reference, C2 (second) is the target
        ref_word, tar_word = "first", second_or_last
        ref_num, tar_num = "1", "2"
    else:
        # Case 2: C2 (second) is the reference, C1 (first) is the target
        ref_word, tar_word = second_or_last, "first"
        ref_num, tar_num = "2", "1"


    replacements = {
        "$REF_WORD$": ref_word, "$TAR_WORD$": tar_word,
        "$REF_NUM$": ref_num, "$TAR_NUM$": tar_num,
    }

    # Perform the replacement
    filled_template = template
    for placeholder, replacement in replacements.items():
        filled_template = filled_template.replace(placeholder, replacement)

    return filled_template


def generate_random_int_choices(answer_int):
    # --------------------------
    # Generate 4 options (1 correct + 3 distractors)
    # --------------------------
    options = set()
    # 1. Add the correct option
    options.add(answer_int)

    # 2. Generate distractors: float around the answer by ±1~3, avoid 0/negatives (object count >= 1)
    while len(options) < 4:
        # random float value (±1, ±2, ±3, avoiding too-large gaps from the answer)
        offset = random.choice([-3, -2, -1, 1, 2, 3])
        distractor = answer_int + offset
        # ensure the distractor is a positive integer (object count can never be 0 or negative)
        if distractor > 0:
            options.add(distractor)

    # 3. Sort and format the options (A/B/C/D)
    correct_description = str(answer_int)
    sorted_options = list(options)
    distractors = [str(option) for option in sorted_options if option != answer_int]

    return generate_shuffled_choices_text([correct_description], distractors)


def generate_orientation_choices(rel_direction, direct_8=True, direct_geo=True):
    # --------------------------
    # Generate 4 direction options (1 correct + 3 distractors)
    # --------------------------
    options = set()

    # Determine the direction vocabulary
    if direct_8:
        if direct_geo:
            direction_pool = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
        else:
            direction_pool = ['front', 'front right', 'right', 'back right', 'back', 'back left', 'left', 'front left']
    else:
        direction_pool = ['front', 'right', 'back', 'left']

    # 1. Add the correct option
    options.add(rel_direction)

    # 2. Generate distractors: randomly pick from the direction pool, avoiding duplicates
    while len(options) < 4:
        # randomly pick a distractor from the direction pool
        distractor = random.choice(direction_pool)
        # ensure the distractor differs from the correct option
        if distractor != rel_direction:
            options.add(distractor)

    correct_description = rel_direction
    sorted_options = list(options)
    distractors = [option for option in sorted_options if option != correct_description]

    return generate_shuffled_choices_text([correct_description], distractors)


def get_description(ref_obj_id: str, cam1_data: Dict, objects: Dict) -> str:
    """
    Generate a description of the reference object as seen from camera 1.

    Args:
        ref_obj_id: ID of the reference object
        cam1_data: camera 1 data (contains object annotations, e.g. {"objects": {obj_id: {"bbox_2d": ...}, ...}})
        objects: metadata of all objects (contains categories, e.g. {obj_id: {"category": ...}, ...})

    Returns:
        str: description of the reference object (e.g. "painting", "the chair on the left", "the vase in the middle")
    """
    # 1. Extract the reference object's category
    ref_category = objects[ref_obj_id]['category']

    # 2. Extract all objects in cam1 and those of the same category
    cam1_objects = cam1_data.get('objects', {})  # object annotations in cam1

    # Collect all objects in cam1 whose category matches the reference (list of IDs)
    same_category_ids = [
        obj_id for obj_id in cam1_objects.keys()
        if objects[obj_id]['category'] == ref_category
    ]
    count = len(same_category_ids)

    # 3. Handle the case of one same-category object
    if count == 1:
        return True, ref_category, ref_category

    # 4. Handle more than 3 same-category objects (discard, return the base category)
    if count > 3:
        return False, ref_category, ref_category

    # 5. Extract the 2D bbox center of same-category objects (for position comparison)
    def get_bbox_center(bbox: Dict) -> Tuple[float, float]:
        """Compute the (x, y) center of a bbox (x is left/right, y is up/down)."""
        center_x = (bbox['min_x'] + bbox['max_x']) / 2
        center_y = (bbox['min_y'] + bbox['max_y']) / 2
        return (center_x, center_y)

    # Build the (object ID, center x, center y) list
    obj_centers: List[Tuple[str, float, float]] = []
    for obj_id in same_category_ids:
        bbox = cam1_objects[obj_id]['bbox_2d']
        cx, cy = get_bbox_center(bbox)
        obj_centers.append((obj_id, cx, cy))

    # 6. Decide between left/right vs up/down separation (based on coordinate spread)
    x_coords = [cx for _, cx, _ in obj_centers]
    y_coords = [cy for _, _, cy in obj_centers]

    # Compute the spread of x (left/right) and y (up/down); larger variance means more spread
    x_variance = np.var(x_coords)
    y_variance = np.var(y_coords)
    use_horizontal = x_variance > y_variance  # if left/right is more spread, use the horizontal direction

    # 7. Sort the objects by direction
    if use_horizontal:
        # left/right: sort by x ascending (left to right)
        sorted_objs = sorted(obj_centers, key=lambda x: x[1])
    else:
        # up/down: sort by y ascending (top to bottom, smaller y is higher)
        sorted_objs = sorted(obj_centers, key=lambda x: x[2])

    # 8. Determine the reference object's position description
    # position words (for 2 or 3 objects)
    if use_horizontal:
        # left/right direction
        pos_2 = ["leftmost", "rightmost"]  # leftmost, rightmost
        pos_3 = ["leftmost", "middle", "rightmost"]  # leftmost, middle, rightmost
    else:
        # up/down direction
        pos_2 = ["topmost", "bottommost"]  # topmost, bottommost
        pos_3 = ["topmost", "middle", "bottommost"]  # topmost, middle, bottommost
    positions = pos_2 if count == 2 else pos_3

    # Find the reference object's index in the sorted list
    ref_index = next(i for i, (obj_id, _, _) in enumerate(sorted_objs) if obj_id == ref_obj_id)

    # 9. Generate the final description (position + category)
    return True, ref_category, f"{positions[ref_index]} {ref_category}"

def generate_size_comparison_choices(obj1_size, obj2_size, obj1_des, obj2_des,
                                     question_des, proximity_factor, ask_for_greater):
    """
    Compare two objects by size and generate multiple-choice options and the answer.

    Args:
        obj1_size (float/int): size value of the first object.
        obj2_size (float/int): size value of the second object.
        obj1_des (str): description of the first object (e.g. 'obj1's length').
        obj2_des (str): description of the second object (e.g. 'obj2's length').
        question_des (str): the question' description part, used for the 'same' option (e.g. 'length is the same').
        proximity_factor (float): proximity factor. If the ratio of the two sizes is within this value, they are considered 'same'.

    Returns:
        dict: a dict containing 'options' (formatted option string) and 'answer' (correct answer string).
    """

    # 1. Determine the description text of the comparison result
    # Note: the options describe the "relational direction" of the comparison result.
    SAME_TEXT = f"The same {question_des}"
    OBJ1_LARGER_TEXT = f"The {obj1_des} in Figure 1"  # e.g. 'The length of object 1'
    OBJ2_LARGER_TEXT = f"The {obj2_des} in Figure 2"  # e.g. 'The length of object 2'
    UNCERTAIN_TEXT = "Sometimes the former, sometimes the latter"

    # 2. Compute the ratio and determine the correct relational direction
    # ensure both sizes are positive to avoid division-by-zero or edge cases
    if obj1_size <= 0 or obj2_size <= 0:
        raise ValueError("obj1_size and obj2_size must be positive.")

    # compute the ratio, ensuring the numerator is the larger value
    ratio = max(obj1_size, obj2_size) / min(obj1_size, obj2_size)

    # Determine the correct option
    if ratio <= proximity_factor:
        # Case A: nearly equal, select 'same'
        correct_text = SAME_TEXT
    elif obj1_size > obj2_size:
        # Case B: obj1 is larger
        correct_text = OBJ1_LARGER_TEXT if ask_for_greater else OBJ2_LARGER_TEXT
    else:  # obj2_size > obj1_size
        # Case C: obj2 is larger
        correct_text = OBJ2_LARGER_TEXT if ask_for_greater else OBJ1_LARGER_TEXT

    # 3. Define the option contents
    # For simplicity and per your requirement, we fix the A/B/C/D option contents and order
    # Note: per your requirement the options are: same / obj1 / obj2 / uncertain

    # option content list (fixed order A, B, C, D)
    correct_description = correct_text
    fixed_options = [
        SAME_TEXT,
        OBJ1_LARGER_TEXT,
        OBJ2_LARGER_TEXT,
        UNCERTAIN_TEXT  # assume D is always the uncertain item
    ]
    distractors = [option for option in fixed_options if option != correct_description]

    return generate_shuffled_choices_text([correct_description], distractors)

# --- Vocabulary mapping constants ---
FORWARD_VOCAB_T3 = ["front", "forward"]
BACKWARD_VOCAB_T3 = ["back", "rear", "backward"]
LEFT_VOCAB = ["left"]
RIGHT_VOCAB = ["right"]

# Template library definitions
DESCRIPTION_TEMPLATES = [
    # Template 1: action + sideways direction (keep forward/backward)
    "{forward_text} to the {right_text}",
    # Template 2: sideways direction + action (keep forward/backward)
    "to the {right_text} while moving {forward_text}",
    # Template 3: concise direction combination (uses replaced vocab, e.g. 'back right')
    "{forward_text} {right_text}",
]

# Pure forward/backward and left/right description templates (all use 'moving XXX')
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

# All possible mixed-direction core pairs
ALL_MIXED_CORES: List[Tuple[str, str]] = [
    ("forward", "right"), ("forward", "left"),
    ("backward", "right"), ("backward", "left"),
]


def _get_random_vocab_for_template_3(direction: str) -> str:
    """Get random vocabulary only for template 3 (front/back/rear)."""
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
    """Generate the description from the core directions and template index, using the fixed vocab map."""

    selected_template = DESCRIPTION_TEMPLATES[template_idx]

    # Get the words for f_core and r_core from the fixed vocab_map
    f_text = vocab_map[f_core]
    r_text = vocab_map[r_core]

    # Templates 1 & 2 style: if the word is front/back/left/right, special handling may be needed
    # For simplicity, assume templates 1 & 2 use the core words (forward/backward)
    # Alternatively, let vocab_map decide everything

    if template_idx != 2:
        # Templates 1 & 2 may be descriptive and need the 'moving forward' form
        # Define here: do templates 1/2 use 'forward to the left' or 'moving front to the left'?
        # Assume templates 1/2 always use the core words, unless template 3 forces replacement
        if f_core in ['forward', 'backward']:
            f_text = f_core
        if r_core in ['left', 'right']:
            r_text = r_core

    # If template 3 (concise), use the pre-selected words from vocab_map
    if template_idx == 2:
        f_text = vocab_map[f_core]
        r_text = vocab_map[r_core]

    return selected_template.format(forward_text=f_text, right_text=r_text)


def _generate_pure_description(direction: str, vocab_map: Dict[str, str]) -> str:
    """Generate a pure forward/backward / left/right description."""

    # PURE_TEMPLATES already include 'moving XXX'; no replacement needed here
    return PURE_TEMPLATES.get(direction, "Unknown Direction")

def _get_random_pure_vocab(direction: str) -> str:
    """Randomly select one phrasing from PURE_WITHOUT_MOVING_TEMPLATES."""
    return random.choice(PURE_WITHOUT_MOVING_TEMPLATES.get(direction, [f"moving {direction}"]))


def generate_direction_choices(forward_dir: str, right_dir: str, moving=True) -> Tuple[str, str]:
    """
    Given descriptions of forward/backward and left/right directions, generate multiple-choice options and the answer about movement direction.
    """

    # --- 1. Core step: build and lock the vocabulary map (VOCAB_MAP) ---
    vocab_map: Dict[str, str] = {}

    # 1.1 lock the forward/backward vocabulary
    vocab_map['forward'] = random.choice(FORWARD_VOCAB_T3)
    vocab_map['backward'] = random.choice(BACKWARD_VOCAB_T3)

    # 1.2 lock the left/right vocabulary
    vocab_map['left'] = random.choice(LEFT_VOCAB)  # always 'left'
    vocab_map['right'] = random.choice(RIGHT_VOCAB)  # always 'right'

    # 1. Determine the core directions (ensure input compatibility)
    # f_core = 'forward' if forward_dir.lower() in FORWARD_VOCAB_T3 else ('backward' if forward_dir.lower() in BACKWARD_VOCAB_T3 else '')
    # r_core = 'right' if right_dir.lower() in RIGHT_VOCAB else ('left' if right_dir.lower() in LEFT_VOCAB else '')
    f_core = forward_dir
    r_core = right_dir

    correct_description = ""
    distractors: List[str] = []

    # --- 2. Determine the correct answer text, template, and distractors ---

    if f_core and r_core:
        # Case A: mixed-direction movement
        template_idx = random.randrange(len(DESCRIPTION_TEMPLATES))
        if not moving:
            template_idx = 2

        # 2.1 generate the correct answer
        correct_description = _generate_description(f_core, r_core, template_idx, vocab_map)

        # 2.2 generate distractors (3 clearly-wrong mixed directions)
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
            # Case B: pure directional movement
            correct_description = _generate_pure_description(move_dir, vocab_map)

            all_distractors_set: Set[str] = set()

            # 2.1 introduce pure-direction distractors (2)
            all_pure_moves = ["forward", "backward", "left", "right"]
            pure_distractors_cores = [d for d in all_pure_moves if d != move_dir]

            # randomly choose 2 pure directions as distractors
            selected_pure_distractors = random.sample(pure_distractors_cores, 2)
            for d in selected_pure_distractors:
                all_distractors_set.add(_generate_pure_description(d, vocab_map))

            # 2.2 introduce a mixed-direction distractor (1)

            # find mixed directions that do not conflict with the correct answer
            mixed_distractor_cores = [
                (f, r) for f, r in ALL_MIXED_CORES
                if f != f_core and r != r_core
            ]

            # randomly choose a mixed template and a set of core directions
            if mixed_distractor_cores:
                d_f_core, d_r_core = random.choice(mixed_distractor_cores)
                template_idx = random.randrange(len(DESCRIPTION_TEMPLATES))
                all_distractors_set.add(_generate_description(d_f_core, d_r_core, template_idx, vocab_map))

            # ensure at least 3 distractors
            distractors = list(all_distractors_set)
            while len(distractors) < 3:
                distractors.append(f"Some other movement ({len(distractors)})")
            distractors = random.sample(distractors, 3)  # finally select 3
            distractors.append('Not moving')
        else:
            # --- New mode: force 4 pure directions with random phrasings ---

            # 1. determine the option cores
            pure_directions = ["forward", "backward", "left", "right"]

            # 2. build the option list, choosing a random phrasing per direction
            for d in pure_directions:
                description = _get_random_pure_vocab(d)  # <--- use the random phrasing function
                # determine the correct answer
                if d == move_dir:
                    correct_description = description
                else:
                    distractors.append(description)

    else:
        # Case C: no significant movement
        correct_description = "Not moving"
        all_possible_descriptions: Set[str] = set()

        # C.1 generate all mixed-direction distractors
        for d_f_core, d_r_core in ALL_MIXED_CORES:
            # randomly choose a template (0, 1, or 2)
            template_idx = random.randrange(len(DESCRIPTION_TEMPLATES))
            description = _generate_description(d_f_core, d_r_core, template_idx, vocab_map)
            all_possible_descriptions.add(description)

        # C.2 generate all pure-direction distractors
        pure_directions = ["forward", "backward", "left", "right"]
        for direction in pure_directions:
            # use the 'moving XXX' template
            pure_desc = _generate_pure_description(direction, vocab_map)
            all_possible_descriptions.add(pure_desc)

            # use PURE_WITHOUT_MOVING_TEMPLATES if more variety is needed
            # pure_desc_rand = _get_random_pure_vocab(direction)
            # all_possible_descriptions.add(pure_desc_rand)

        # C.3 randomly draw 3 distractors from all possibilities
        # ensure enough options; if fewer than 3 (should not happen, but to be safe)
        max_distractors = 3
        if len(all_possible_descriptions) < max_distractors:
            distractors = list(all_possible_descriptions)
            while len(distractors) < max_distractors:
                distractors.append(f"Generic Movement {len(distractors)}")
        else:
            distractors = random.sample(list(all_possible_descriptions), max_distractors)

    # --- 3. Format and output ---

    # Combine the correct answer with the distractors
    # Per the current logic, distractors must not include correct_description, otherwise a random sample could select it and leave fewer than 4 options
    distractors = random.sample(list(distractors), 3)

    return generate_shuffled_choices_text([correct_description], distractors)

# Define option constants
SINGLE_DIRS = ["up", "down", "left", "right"]
MIXED_DIRS = ["upper right", "upper left", "lower left", "lower right"]
NULL_DIR = "Unable to determine"


def generate_rotation_choices(yaw_dir: str, pitch_dir: str) -> Tuple[str, str]:
    """
    Given the Yaw and Pitch directions, generate the options and answer.

    :param yaw_dir: 'left', 'right', or ''
    :param pitch_dir: 'up', 'down', or ''
    :return: (options_str, answer_str)
    """

    # 1. Determine the core answer
    is_yaw_sig = bool(yaw_dir)
    is_pitch_sig = bool(pitch_dir)

    if not is_yaw_sig and not is_pitch_sig:
        # Case A: no significant rotation
        correct_description = NULL_DIR

        # options: NULL_DIR + 3 random single- or mixed-direction distractors
        all_possible_moves = SINGLE_DIRS + MIXED_DIRS
        distractors = random.sample(all_possible_moves, 3)
        options_list = [correct_description] + distractors

    elif (is_yaw_sig and is_pitch_sig):
        # Case B: mixed rotation (both significant)

        # determine the correct answer text
        p_text = pitch_dir
        y_text = yaw_dir

        # build the mixed answer, e.g. "upper right"
        correct_description = f"{p_text} {y_text}".replace("up ", "upper ").replace("down ", "lower ")

        # 2. generate the distractors

        # distractor 1: only change Yaw
        d1_yaw = 'left' if y_text == 'right' else 'right'
        d1_desc = f"{p_text} {d1_yaw}".replace("up ", "upper ").replace("down ", "lower ")

        # distractor 2: only change Pitch
        d2_pitch = 'up' if p_text == 'down' else 'down'
        d2_desc = f"{d2_pitch} {y_text}".replace("up ", "upper ").replace("down ", "lower ")

        # distractor 3: randomly pick a pure direction as a distractor
        all_pure_moves = SINGLE_DIRS
        d3_desc = random.choice([d for d in all_pure_moves if d != p_text and d != y_text])

        options_list = [correct_description, d1_desc, d2_desc, d3_desc]

    else:
        # Case C: single significant rotation (only Yaw or only Pitch)

        # determine the correct answer text
        correct_description = yaw_dir if is_yaw_sig else pitch_dir

        options_list = ['up', 'down', 'left', 'right']

    # --- 3. Final formatting ---
    distractors = [option for option in options_list if option != correct_description]

    return generate_shuffled_choices_text([correct_description], distractors)

def generate_coordinate_direction_choices(forward_dir: str, right_dir: str, setting: str) -> Tuple[str, str]:
    """
    Given forward/backward and left/right direction descriptions, generate multiple-choice options and the answer about movement direction under a fixed coordinate-frame convention.
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
    Given the Yaw and Pitch directions, generate the options and answer.

    :param yaw_dir: 'left', 'right', or ''
    :param pitch_dir: 'up', 'down', or ''
    :return: (options_str, answer_str)
    """

    # 1. Determine the core answer

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
    # Given a list of 3 wrong options and a list of 1 correct option, return the joined option string and the correct option letter
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
