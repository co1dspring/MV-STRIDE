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
