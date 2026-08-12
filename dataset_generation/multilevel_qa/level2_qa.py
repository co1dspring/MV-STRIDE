from util.math_utils import get_camera_rotation_matrix
from util.qa_utils import format_bbox_dict_to_str, get_image_size
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

class Level2Mixin:
    def _object_correspondence(self, objects, cam1_data, cam2_data, obj_ids, cat, camera_name):
        # 确定物体在其中一张图像中，找该物体在另一张图像中的对应
        qa_type = "Object_correspondence"
        template = self.parent.templates[qa_type]
        H, W = get_image_size(cam1_data)

        if camera_name == "1,2":
            # 确定物体在cam1中
            bbox_2d_1 = cam1_data.get('objects', {}).get(obj_ids, {}).get('bbox_2d')
            question = template["question"].format(fig_num_1="1", cat=cat.lower(), bbox1=format_bbox_dict_to_str(bbox_2d_1, H, W), fig_num_2="2")
            if obj_ids in cam2_data.get('objects', {}).keys():
                bbox_2d_2 = cam2_data.get('objects', {}).get(obj_ids, {}).get('bbox_2d')
                answer = template["answer_exist"].format(cat=cat.lower(), fig_num_2="2", bbox2=format_bbox_dict_to_str(bbox_2d_2, H, W))
            else:
                answer = template["answer_not_exist"].format(cat=cat.lower(), fig_num_2="2")
        elif camera_name == "2,1":
            # 确定物体在cam1中
            bbox_2d_1 = cam2_data.get('objects', {}).get(obj_ids, {}).get('bbox_2d')
            question = template["question"].format(fig_num_1="2", cat=cat.lower(), bbox1=format_bbox_dict_to_str(bbox_2d_1, H, W), fig_num_2="1")
            if obj_ids in cam1_data.get('objects', {}).keys():
                bbox_2d_2 = cam1_data.get('objects', {}).get(obj_ids, {}).get('bbox_2d')
                answer = template["answer_exist"].format(cat=cat.lower(), fig_num_2="1", bbox2=format_bbox_dict_to_str(bbox_2d_2, H, W))
            else:
                answer = template["answer_not_exist"].format(cat=cat.lower(), fig_num_2="1")
        return (question, answer, qa_type, "1,2")

    def _cam_rot_yaw(self, objects, cam1_data, cam2_data, obj_ids, cat, camera_name):
        qa_type = "Cam_rot_yaw"
        template = self.parent.templates[qa_type]

        R1 = get_camera_rotation_matrix(cam1_data)
        R2 = get_camera_rotation_matrix(cam2_data)
        is_scannetpp = self.parent.data_source == "scannetpp"

        if camera_name == "1,2":
            # 提问从cam1到cam2的yaw转动角度
            R_rel = R2 @ R1.T
            question = template["question"].format(fig_num_1="1", fig_num_2="2")
        elif camera_name == "2,1":
            # 提问从cam2到cam1的yaw转动角度
            R_rel = R1 @ R2.T
            question = template["question"].format(fig_num_1="2", fig_num_2="1")

        r = R.from_matrix(R_rel)
        ypr_angles = r.as_euler('yxz', degrees=False)
        delta_yaw = ypr_angles[0]
        angle = abs(round(math.degrees(delta_yaw), 2))
        if delta_yaw > 0:
            yaw_dir = "right" if is_scannetpp else "left"
        elif delta_yaw < 0:
            yaw_dir = "left" if is_scannetpp else "right"

        answer = template["answer"].format(side=yaw_dir, angle=angle)
        return (question, answer, qa_type, "1,2")

    def _cam_rot_pitch(self, objects, cam1_data, cam2_data, obj_ids, cat, camera_name):
        qa_type = "Cam_rot_pitch"
        template = self.parent.templates[qa_type]

        R1 = get_camera_rotation_matrix(cam1_data)
        R2 = get_camera_rotation_matrix(cam2_data)
        is_scannetpp = self.parent.data_source == "scannetpp"

        if camera_name == "1,2":
            # 提问从cam1到cam2的pitch转动角度
            R_rel = R2 @ R1.T
            question = template["question"].format(fig_num_1="1", fig_num_2="2")
        elif camera_name == "2,1":
            # 提问从cam2到cam1的pitch转动角度
            R_rel = R1 @ R2.T
            question = template["question"].format(fig_num_1="2", fig_num_2="1")

        r = R.from_matrix(R_rel)
        ypr_angles = r.as_euler('yxz', degrees=False)
        delta_pitch = ypr_angles[1]
        angle = abs(round(math.degrees(delta_pitch), 2))
        if delta_pitch > 0:
            yaw_dir = "down"
        elif delta_pitch < 0:
            yaw_dir = "up"

        answer = template["answer"].format(side=yaw_dir, angle=angle)
        return (question, answer, qa_type, "1,2")

    def _cam_trans_forward(self, objects, cam1_data, cam2_data, obj_ids, cat, camera_name):
        qa_type = "Cam_trans_forward"
        template = self.parent.templates[qa_type]
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])

        if camera_name == "1,2":
            # 提问从cam1到cam2的前后向移动
            cam_for = - cam1_for
            # 计算位移向量
            T_vec = cam2_loc - cam1_loc

            question = template["question"].format(fig_num_1="1", fig_num_2="2")
        elif camera_name == "2,1":
            # 提问从cam2到cam1的前后向移动
            cam_for = - cam2_for
            # 计算位移向量
            T_vec = cam1_loc - cam2_loc

            question = template["question"].format(fig_num_1="2", fig_num_2="1")

        # 投影位移向量到局部轴上
        # 使用点积 (Dot Product) 获取分量
        D_forward = np.dot(T_vec, cam_for)

        direction = "forward" if D_forward > 0 else "backward"
        distance = round(abs(D_forward), 4)

        answer = template["answer"].format(side=direction, distance=distance)
        return (question, answer, qa_type, "1,2")

    def _cam_trans_right(self, objects, cam1_data, cam2_data, obj_ids, cat, camera_name):
        qa_type = "Cam_trans_right"
        template = self.parent.templates[qa_type]
        cam1_loc = np.array([cam1_data['location_3d']['x'], cam1_data['location_3d']['y'], cam1_data['location_3d']['z']])
        cam1_for = np.array([cam1_data['forward_direction']['x'], cam1_data['forward_direction']['y'], cam1_data['forward_direction']['z']])
        cam2_loc = np.array([cam2_data['location_3d']['x'], cam2_data['location_3d']['y'], cam2_data['location_3d']['z']])
        cam2_for = np.array([cam2_data['forward_direction']['x'], cam2_data['forward_direction']['y'], cam2_data['forward_direction']['z']])

        if camera_name == "1,2":
            # 提问从cam1到cam2的前后向移动
            cam_for = - cam1_for
            # 计算位移向量
            T_vec = cam2_loc - cam1_loc

            question = template["question"].format(fig_num_1="1", fig_num_2="2")
        elif camera_name == "2,1":
            # 提问从cam2到cam1的前后向移动
            cam_for = - cam2_for
            # 计算位移向量
            T_vec = cam1_loc - cam2_loc

            question = template["question"].format(fig_num_1="2", fig_num_2="1")

        # 建立cam1的局部坐标系
        # 世界Z轴定义为上轴 (Up Vector)
        Wz_vec = np.array([0, 0, 1])
        # 计算右轴 (Right Vector, R_vec)
        R_vec = np.cross(cam_for, Wz_vec)
        # 归一化右轴（除非F和Wz平行，否则R_vec不会是零向量）
        R_norm = np.linalg.norm(R_vec)
        if R_norm < 1e-6:
            # 紧急处理：如果相机垂直朝向，定义右轴为世界X轴
            R_vec = np.array([1, 0, 0])
        else:
            R_vec = R_vec / R_norm

        # 投影位移向量到局部轴上
        # 使用点积 (Dot Product) 获取分量
        D_right = np.dot(T_vec, R_vec)

        direction = "right" if D_right > 0 else "left"
        distance = round(abs(D_right), 4)

        answer = template["answer"].format(side=direction, distance=distance)
        return (question, answer, qa_type, "1,2")
