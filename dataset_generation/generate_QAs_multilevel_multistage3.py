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
        # print(f"\n===== 处理场景：{scene_name} =====")

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
        # print(f"生成 {len(camera_pairs)} 个相机组合")

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
        #     print(f"  组合 {cam1_name}&{cam2_name} 生成 {qa_type} 失败：{e}")

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
    # CONFIG_PATH = './qa_config_infinigen.json'
    # CONFIG_PATH = './qa_config_scannetpp.json'
    # CONFIG_PATH = './qa_config_infinigen_sparse.json'
    # CONFIG_PATH = './qa_config_scannetpp_sparse.json'
    # CONFIG_PATH = './qa_config_infinigen_ablation.json'
    CONFIG_PATH = './qa_config_scannetpp_ablation.json'

    # Run the generator
    generator = SceneQAGenerator(
        config_path=CONFIG_PATH
    )
    # generator.save_all_categories()
    generator.process_all_scenes()
    generator.repartition_data_by_stage()
