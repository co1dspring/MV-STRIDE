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
