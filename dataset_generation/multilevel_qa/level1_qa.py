import numpy as np
from util.qa_utils import format_bbox_dict_to_str, get_image_size
from util.math_utils import get_world_to_camera_matrix

class Level1Mixin:
    def _2d_location_perception_category(self, objects, cam_data, obj_ids, cat, camera_name):
        obj_ids = sorted(list(obj_ids))
        qa_type = "2D_location_perception_category"
        template = self.parent.templates[qa_type]
        H, W = get_image_size(cam_data)
        multilevel_qa_mode = self.parent.multilevel_qa_mode
        bbox_list = []
        for obj_id in obj_ids:
            bbox_2d = cam_data.get('objects', {}).get(obj_id, {}).get('bbox_2d')
            bbox_list.append(bbox_2d)
        bbox_strings = [format_bbox_dict_to_str(bbox, H, W) for bbox in bbox_list]
        bbox_list_str = ", ".join(bbox_strings)

        fig_num = camera_name if multilevel_qa_mode=="conversation" else ""
        question = template["question"].format(cat=cat.lower(), fig_num=fig_num)
        answer = template["answer_exist"].format(count=str(len(obj_ids)), cat=cat.lower(), bbox_list=bbox_list_str, fig_num=fig_num)
        return (question, answer, qa_type, camera_name)

    def _2d_location_perception_object(self, objects, cam_data, obj_ids, cat, camera_name):
        qa_type = "2D_location_perception_object"
        template = self.parent.templates[qa_type]
        H, W = get_image_size(cam_data)
        multilevel_qa_mode = self.parent.multilevel_qa_mode
        bbox_2d = cam_data.get('objects', {}).get(obj_ids, {}).get('bbox_2d')

        fig_num = camera_name if multilevel_qa_mode=="conversation" else ""
        question = template["question"].format(obj_des=cat, fig_num=fig_num)
        answer = template["answer_exist"].format(obj_des=cat, bbox=format_bbox_dict_to_str(bbox_2d, H, W), fig_num=fig_num)
        return (question, answer, qa_type, camera_name)

    def _depth_perception(self, objects, cam_data, obj_ids, cat, camera_name):
        qa_type = "Depth_perception"
        template = self.parent.templates[qa_type]
        multilevel_qa_mode = self.parent.multilevel_qa_mode
        ref_obj_loc = np.array(objects[obj_ids]['3d_center'])
        cam_loc = np.array([cam_data['location_3d']['x'], cam_data['location_3d']['y'], cam_data['location_3d']['z']])
        offset_vec = ref_obj_loc - cam_loc
        depth = np.linalg.norm(offset_vec)
        depth_final = round(float(depth), 2)

        fig_num = camera_name if multilevel_qa_mode=="conversation" else ""
        question = template["question"].format(obj_des=cat, fig_num=fig_num)
        answer = template["answer"].format(obj_des=cat, depth_val=depth_final, fig_num=fig_num)
        return (question, answer, qa_type, camera_name)

    def _cam_obj_yaw(self, objects, cam_data, obj_ids, cat, camera_name):
        qa_type = "Cam_obj_yaw"
        template = self.parent.templates[qa_type]
        multilevel_qa_mode = self.parent.multilevel_qa_mode

        ref_obj_loc = np.array(objects[obj_ids]['3d_center'])
        cam_loc = np.array([cam_data['location_3d']['x'], cam_data['location_3d']['y'], cam_data['location_3d']['z']])
        cam_for = -np.array([cam_data['forward_direction']['x'], cam_data['forward_direction']['y'], cam_data['forward_direction']['z']])
        T_vec = ref_obj_loc - cam_loc
        # 2. 建立相机局部坐标系的轴
        # 假设世界 Z 轴 [0,0,1] 为 Up
        Wz_vec = np.array([0, 0, 1])

        # 计算右轴 (Right Vector)
        # R = Forward x Up
        R_vec = np.cross(cam_for, Wz_vec)
        R_norm = np.linalg.norm(R_vec)
        if R_norm < 1e-6:
            R_vec = np.array([1, 0, 0])
        else:
            R_vec = R_vec / R_norm

        # 3. 投影位移向量到局部轴上
        # D_f: 物体在相机前方多少距离
        # D_r: 物体在相机右方多少距离（正为右，负为左）
        D_f = np.dot(T_vec, cam_for)
        D_r = np.dot(T_vec, R_vec)

        # 4. 计算水平面上的夹角 (Yaw)
        # 我们只关心水平偏移，所以忽略垂直高度差
        # 使用 arctan2(y, x) 即 arctan2(右向分量, 前向分量)
        angle_rad = np.arctan2(D_r, D_f)
        angle_deg = np.degrees(angle_rad)

        # 5. 整理输出
        side = "left" if angle_deg < 0 else "right"
        abs_angle = round(abs(angle_deg) ,2)

        if abs_angle < 1.0:  # 1度以内视为中心
            side = "center"

        fig_num = camera_name if multilevel_qa_mode=="conversation" else ""
        question = template["question"].format(obj_des=cat, fig_num=fig_num)
        answer = template["answer"].format(obj_des=cat, angle=str(abs_angle), side=side, fig_num=fig_num)
        return (question, answer, qa_type, camera_name)

    def _cam_space_pitch(self, objects, cam_data, obj_ids, cat, camera_name):
        qa_type = "Cam_space_pitch"
        template = self.parent.templates[qa_type]
        multilevel_qa_mode = self.parent.multilevel_qa_mode

        cam_for = -np.array([cam_data['forward_direction']['x'], cam_data['forward_direction']['y'], cam_data['forward_direction']['z']])
        f_norm = cam_for / np.linalg.norm(cam_for)
        pitch_rad = np.arcsin(f_norm[2])
        pitch_deg = np.degrees(pitch_rad)

        # 判定方向
        if abs(pitch_deg) < 1.0:
            direction = "staying almost level"
            angle_str = "0"
        elif pitch_deg > 0:
            direction = "upwards"
            angle_str = f"{abs(pitch_deg):.2f}"
        else:
            direction = "downwards"
            angle_str = f"{abs(pitch_deg):.2f}"

        fig_num = camera_name if multilevel_qa_mode=="conversation" else ""
        question = template["question"].format(fig_num=fig_num)
        answer = template["answer"].format(angle=angle_str, direction=direction, fig_num=fig_num)
        return (question, answer, qa_type, camera_name)

    def _measurement_comparison(self, objects, cam_data, obj_ids, cats, camera_name, dimension_to_compare):
        qa_type = "Measurement_comparison"
        template = self.parent.templates[qa_type][dimension_to_compare]
        multilevel_qa_mode = self.parent.multilevel_qa_mode

        ref_obj_id = obj_ids[0]
        tar_obj_id = obj_ids[1]
        ref_obj_des = cats[0]
        tar_obj_des = cats[1]
        ref_obj_3d_bbox = objects[ref_obj_id]['bbox_3d_aabb']
        tar_obj_3d_bbox = objects[tar_obj_id]['bbox_3d_aabb']

        if dimension_to_compare == 'altitude':
            ref_val = (ref_obj_3d_bbox['min']['z'] + ref_obj_3d_bbox['max']['z']) / 2
            tar_val = (tar_obj_3d_bbox['min']['z'] + tar_obj_3d_bbox['max']['z']) / 2
        elif dimension_to_compare == 'height':
            ref_val = ref_obj_3d_bbox['dimensions']['z']
            tar_val = tar_obj_3d_bbox['dimensions']['z']
        elif dimension_to_compare == 'length':
            ref_val = max(ref_obj_3d_bbox['dimensions']['x'], ref_obj_3d_bbox['dimensions']['y'])
            tar_val = max(tar_obj_3d_bbox['dimensions']['x'], tar_obj_3d_bbox['dimensions']['y'])
        ratio = ref_val / tar_val

        fig_num = camera_name if multilevel_qa_mode == "conversation" else ""
        question = template["question"].format(fig_num=fig_num, tar_obj=tar_obj_des, ref_obj=ref_obj_des)
        answer = template["answer"].format(fig_num=fig_num, tar_obj=tar_obj_des, ref_obj=ref_obj_des, tar_val=round(tar_val, 2), ref_val=round(ref_val, 2), ratio=round(ratio, 2))
        return (question, answer, qa_type, camera_name)

    def _3D_location_cam_obj(self, objects, cam_data, obj_ids, cat, camera_name):
        qa_type = "3D_location_cam_obj"
        template = self.parent.templates[qa_type]
        multilevel_qa_mode = self.parent.multilevel_qa_mode
        ref_obj_loc = np.array(objects[obj_ids]['3d_center'])
        P_w2c = get_world_to_camera_matrix(cam_data)
        world_pos = np.append(ref_obj_loc, 1.0)  # 转为齐次坐标 [x, y, z, 1]
        cam_pos_homo = np.dot(P_w2c, world_pos)  # 矩阵乘法

        x_c, y_c, z_c = cam_pos_homo[:3]

        fig_num = camera_name if multilevel_qa_mode=="conversation" else ""
        question = template["question"].format(obj_des=cat, fig_num=fig_num)
        answer = template["answer"].format(obj_des=cat, x_c=round(x_c, 2), y_c=round(y_c, 2), z_c=round(z_c, 2), fig_num=fig_num)
        return (question, answer, qa_type, camera_name)
