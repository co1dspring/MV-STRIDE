# -*- coding: gbk -*-
import os
import re
import numpy as np
import json
from PIL import Image  # 新增：用于读取PNG图像
from typing import Dict, Any
from pathlib import Path
import bpy
import json
import argparse  # 改用argparse解析参数
import sys
from mathutils import Vector
from tqdm import tqdm
import shutil

def extract_category(obj_name: str) -> str:
    """根据物体名称提取类别"""
    # 1. 处理带Factory的名称（如"BookStackFactory(xxx).spawn_asset(xxx)"）
    factory_match = re.match(r"^(\w+)Factory\(\d+\)\.spawn_asset\(\d+\)(\.\d+)?$", obj_name)
    if factory_match:
        base = factory_match.group(1)
        # 驼峰命名转小写+空格连接（如BookStack → book stack）
        base = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', base).lower()
        return base
    
    # 2. 处理场景结构类名称（如"dining-room_0/0.floor"、"skirtingboard_ceiling"）
    # 提取房屋名称前缀+结构类型（如"dining-room ceiling"、"skirtingboard ceiling"）
    # 先处理带"/"的格式（如"dining-room_0/0.floor"）
    if '/' in obj_name:
        # 分割房屋名称部分和结构部分（如["dining-room_0", "0.floor"]）
        house_part, struct_part = obj_name.split('/', 1)
        # 提取房屋名称核心（去除数字和下划线，如"dining-room_0" → "dining-room"）
        house_core = re.sub(r'_\d+$', '', house_part)
        # 提取结构类型（如"0.floor" → "floor"）
        struct_type = re.sub(r'^\d+\.', '', struct_part).lower()
        return f"{house_core} {struct_type}"
    
    # 处理带"_"的格式（如"skirtingboard_ceiling"）
    if '_' in obj_name:
        # 分割为前缀和结构类型（保留房屋相关前缀）
        parts = obj_name.split('_', 1)
        # 确保两部分都有效（非空且非纯数字）
        if all(parts) and not parts[0].isdigit() and not parts[1].isdigit():
            return f"{parts[0].lower()} {parts[1].lower()}"
    
    # 3. 无特殊格式时返回原名称小写
    return obj_name.lower()

def save_dict_to_json(data: Dict[Any, Any], file_path: str or Path, indent: int = 4, ensure_ascii: bool = False) -> None:
    """
    将字典数据保存为JSON文件
    
    参数:
        data: 要保存的字典数据
        file_path: 保存的文件路径（支持字符串或Path对象）
        indent: JSON格式化缩进空格数，默认4（0表示不格式化）
        ensure_ascii: 是否确保ASCII编码（False可正常显示中文），默认False
    """
    # 转换为Path对象，方便处理路径
    file_path = Path(file_path)
    
    # 创建父目录（如果不存在）
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 写入JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                default=str  # 对无法序列化的对象自动转为字符串
            )
        print(f"字典已成功保存到: {file_path.resolve()}")
    except Exception as e:
        print(f"保存JSON文件失败: {e}")

def get_object_2d_bboxes(object_segmentation: np.ndarray) -> Dict[int, Dict[str, int]]:
    """
    从单通道分割图中提取每个物体ID的2D边界框（bbox）
    
    参数:
        object_segmentation: 单通道分割图的numpy数组（形状为HxW，像素值为物体ID）
    
    返回:
        字典，key为物体ID，value为包含边界框坐标的字典{"min_x", "min_y", "max_x", "max_y"}
    """
    # 获取所有唯一的物体ID（排除背景时可添加条件，如id != 0）
    unique_ids = np.unique(object_segmentation)
    
    object_bboxes = {}
    for obj_id in unique_ids:
        # 找到当前物体ID对应的所有像素坐标
        # np.where返回两个数组：第一个是y坐标（行），第二个是x坐标（列）
        y_coords, x_coords = np.where(object_segmentation == obj_id)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            continue  # 理论上不会出现，跳过空结果
        
        # 计算边界框：min/max x和y
        min_x = int(np.min(x_coords))
        max_x = int(np.max(x_coords))
        min_y = int(np.min(y_coords))
        max_y = int(np.max(y_coords))
        
        # 存储边界框信息
        object_bboxes[str(obj_id)] = {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y
        }
    
    return object_bboxes

def read_single_scene_gt(scene_dir: str, camera_id: int, frame_prefix: str = "0_0048_0") -> Dict[str, Any]:
    """
    读取单个场景的GT数据（相机视角、RGB图像、实例分割、物体元信息、物体分割）
    
    参数:
        scene_dir: 场景根目录路径（对应batch_generation/场景名）
        camera_id: 相机编号（0~9）
        frame_prefix: 帧标识前缀（默认"0_0048_0"，对应文件名中的固定部分）
    
    返回:
        包含各类GT数据的字典，键包括：
            - camview: 相机视角数据
            - image: RGB图像数据（numpy数组，形状为HxWx3）
            - instance_segmentation: 实例分割结果
            - objects_meta: 物体元信息（JSON解析结果）
            - object_segmentation: 物体分割结果
    """
    

    # 构建各数据路径模板
    camview_path = os.path.join(
        scene_dir, "frames", "camview", f"camera_0",
        f"camview_{camera_id}_{frame_prefix}.npz"
    )
    # 修改：图像路径后缀改为.png
    image_path = os.path.join(
        scene_dir, "frames", "Image", f"camera_0",
        f"Image_{camera_id}_{frame_prefix}.png"
    )
    instance_seg_path = os.path.join(
        scene_dir, "frames", "InstanceSegmentation", f"camera_0",
        f"InstanceSegmentation_{camera_id}_{frame_prefix}.npy"
    )
    objects_meta_path = os.path.join(
        scene_dir, "frames", "Objects", f"camera_0",
        f"Objects_{camera_id}_{frame_prefix}.json"
    )
    object_seg_path = os.path.join(
        scene_dir, "frames", "ObjectSegmentation", f"camera_0",
        f"ObjectSegmentation_{camera_id}_{frame_prefix}.npy"
    )
    
    # 读取数据
    camera_info = {}
    
    # 读取相机参数（关键修正：区分内参和外参）
    if os.path.exists(camview_path):
        with np.load(camview_path) as data:
            # 显式读取内参（3x3）和外参（4x4），键名参考官方文档或实际文件
            # 常见键名："intrinsics"（内参）、"extrinsics"（外参）
            extrinsics = data["T"] 
            # 读取内参（3x3）
            camera_info["cam_intrinsics"] = data["K"]  # K = 内参矩阵
            # 读取外参（4x4）
            camera_info["cam_extrinsics"] = extrinsics  # T = 外参矩阵（相机→世界）
            # 读取图像尺寸 [H, W]
            camera_info["image_size_HW"] = data["HW"]     # HW = [高度, 宽度]）

            # 3. 提取相机在世界坐标系中的位置（Tx, Ty, Tz）
            # 外参矩阵的前3行第4列是相机中心的世界坐标
            cam_world_loc = extrinsics[:3, 3]
            camera_info["location_3d"] = {
                "x": float(cam_world_loc[0]),
                "y": float(cam_world_loc[1]),
                "z": float(cam_world_loc[2])
            }
            
            # 4. 提取相机朝向（拍摄方向向量，世界坐标系下）
            # 相机坐标系中，拍摄方向为-Z轴（向量[0, 0, -1]）
            # 通过旋转矩阵R将其转换到世界坐标系
            R = extrinsics[:3, :3]  # 旋转矩阵（3x3）
            cam_local_forward = np.array([0, 0, -1])  # 相机本地拍摄方向（-Z轴）
            cam_world_forward = R @ cam_local_forward  # 转换到世界坐标系
            cam_world_forward = cam_world_forward / np.linalg.norm(cam_world_forward)  # 归一化
            camera_info["forward_direction"] = {
                "x": float(cam_world_forward[0]),
                "y": float(cam_world_forward[1]),
                "z": float(cam_world_forward[2])
            }
    else:
        raise FileNotFoundError(f"相机参数文件不存在: {camview_path}")
    
    # 读取RGB图像数据（.png）- 修改部分
    if os.path.exists(image_path):
        # 使用PIL读取PNG，转换为RGB模式（避免alpha通道），再转为numpy数组
        # with Image.open(image_path) as img:
        #     rgb_img = img.convert("RGB")  # 确保是3通道RGB
        #     gt_data["image"] = np.array(rgb_img)  # 形状为 (H, W, 3)
        camera_info["image_path"] = image_path  # 仅保存图像路径，避免大数组存储
    else:
        raise FileNotFoundError(f"RGB图像文件不存在: {image_path}")
    
    # 读取实例分割结果（.npy）
    # if os.path.exists(instance_seg_path):
    #     gt_data["instance_segmentation"] = np.load(instance_seg_path)
    # else:
    #     raise FileNotFoundError(f"实例分割文件不存在: {instance_seg_path}")
    
    # 读取物体元信息（.json）
    if os.path.exists(objects_meta_path):
        with open(objects_meta_path, "r", encoding="utf-8") as f:
            scene_metadata = json.load(f)
    else:
        raise FileNotFoundError(f"物体元信息文件不存在: {objects_meta_path}")
    
    # 读取物体分割结果（.npy）
    if os.path.exists(object_seg_path):
        object_segmentation = np.load(object_seg_path)
        objects_in_camera = get_object_2d_bboxes(object_segmentation)
    else:
        raise FileNotFoundError(f"物体分割文件不存在: {object_seg_path}")
    
    # 把分割结果和物体元信息整合
    camera_info['objects'] = {}
    for obj_id in objects_in_camera:
        for obj_meta in scene_metadata:
            if scene_metadata[obj_meta]['object_index'] == int(obj_id):
                camera_info['objects'][obj_meta] = {
                    "object_index": int(obj_id),
                    "bbox_2d": objects_in_camera[obj_id]
                }
                break
    
    return camera_info

def collect_scene_metadata(scene_dir: str, camera_num: int, stage: str, frame_prefix: str = "0_0048_0") -> Dict[str, Any]:
    blend_file_path = os.path.join(scene_dir, stage, "scene.blend")
    # 清除默认场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # 加载.blend文件
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    
    annotations = {"cameras": {}, "objects": {}}
    depsgraph = bpy.context.evaluated_depsgraph_get()  # 获取依赖图（用于计算物体变换后的数据）
    
    # 遍历场景中所有物体，除了相机
    for obj in bpy.data.objects:
        # # 1. 处理相机（单独提取相机特有的参数）
        # if obj.type == "CAMERA":
        #     cam_data = obj.data  # 相机内参数据
            
        #     # 相机位置（世界坐标系下的原点）
        #     cam_location = obj.location
        #     cam_world_location = obj.matrix_world.translation  # 世界位置
            
        #     # 相机朝向（通过旋转欧拉角表示，或转换为前向向量）
        #     rotation_euler = obj.rotation_euler
        #     # 计算相机前向向量（Blender相机默认朝-Z方向，需转换为世界坐标系）
        #     forward = -obj.matrix_world.to_quaternion() @ Vector((0, 0, 1))  # 前向向量（指向拍摄方向）

        #     # 2. 相机世界旋转（绝对旋转）
        #     # 方法1：欧拉角（世界坐标系）
        #     cam_world_rot_euler = obj.matrix_world.to_euler()
        #     # 方法2：前向向量（世界坐标系下的拍摄方向）
        #     cam_forward = -obj.matrix_world.to_quaternion() @ Vector((0, 0, 1))  # 相机默认朝-Z
            
        #     # 相机内参（焦距、主点等）
        #     intrinsics = {
        #         "focal_length_mm": cam_data.lens,  # 焦距（毫米）
        #         "sensor_width_mm": cam_data.sensor_width,  # 传感器宽度（毫米）
        #         "sensor_height_mm": cam_data.sensor_height,  # 传感器高度（毫米）
        #         "principal_point": {  # 主点偏移（相对于传感器中心，单位：毫米）
        #             "x": cam_data.shift_x * cam_data.sensor_width,
        #             "y": cam_data.shift_y * cam_data.sensor_height
        #         }
        #     }
            
        #     # 整合相机信息
        #     camera_info = {
        #         "name": obj.name,
        #         "object_index": obj.pass_index,  # 若相机有object_index
        #         "type": "CAMERA",
        #         "location_3d": {  # 世界坐标系下的位置
        #             "x": float(cam_world_location.x),
        #             "y": float(cam_world_location.y),
        #             "z": float(cam_world_location.z)
        #         },
        #         "rotation_world": {  # 世界坐标系绝对旋转
        #             "euler_rad": [
        #                 float(cam_world_rot_euler.x),
        #                 float(cam_world_rot_euler.y),
        #                 float(cam_world_rot_euler.z)
        #             ],
        #             "euler_deg": [
        #                 float(cam_world_rot_euler.x * (180 / 3.1415926535)),
        #                 float(cam_world_rot_euler.y * (180 / 3.1415926535)),
        #                 float(cam_world_rot_euler.z * (180 / 3.1415926535))
        #             ],
        #             "forward_direction": {  # 世界坐标系下的拍摄方向
        #                 "x": float(cam_forward.x),
        #                 "y": float(cam_forward.y),
        #                 "z": float(cam_forward.z)
        #             }
        #         },
        #         "intrinsics": intrinsics  # 内参参数
        #     }
        #     annotations.append(camera_info)
        #     continue  # 相机处理完毕，跳过后续通用物体逻辑


        # 过滤非实体物体
        if obj.type not in ["MESH", "CURVE", "SURFACE", "META"]:
            continue
        
        if obj.name not in annotations["objects"]:
            annotations["objects"][obj.name] = {
                # "name": obj.name,
                # "object_index": obj.pass_index,
                "type": obj.type,
                "is_visible": obj.visible_get(),
                "category": extract_category(obj.name),
            }
        else:
            # 报错有重复物体，退出
            raise ValueError(f"发现重复物体名称: {obj.name}")
        
        # 1. 物体世界位置（绝对坐标）
        obj_world_loc = obj.matrix_world.translation  # 不受父物体影响
        
        # 2. 物体世界旋转（绝对旋转）
        # obj_world_rot = obj.matrix_world.to_euler()   # 从世界矩阵提取旋转
        world_rot_matrix = obj.matrix_world # 获取物体的世界旋转矩阵（从世界变换矩阵中提取）

        # 旋转矩阵的列向量分别对应物体本地X、Y、Z轴在世界坐标系中的方向
        # 归一化确保向量长度为1（单位向量）
        local_x_in_world = world_rot_matrix.col[0].normalized()  # 本地X轴（右方向）
        local_y_in_world = world_rot_matrix.col[1].normalized()  # 本地Y轴（前方向）
        local_z_in_world = world_rot_matrix.col[2].normalized()  # 本地Z轴（上方向）

        # 中心坐标（物体原点）
        # center_world = obj.location
        annotations["objects"][obj.name]["3d_center"] = [float(obj_world_loc.x), float(obj_world_loc.y), float(obj_world_loc.z)]
        
        # 朝向（欧拉角）
        # rotation_euler = obj.rotation_euler
        # annotations["objects"][obj.name]["rotation"] = {
        #     "euler_rad": [float(obj_world_rot.x), float(obj_world_rot.y), float(obj_world_rot.z)],
        #     "euler_deg": [
        #         float(obj_world_rot.x * (180 / 3.1415926535)),
        #         float(obj_world_rot.y * (180 / 3.1415926535)),
        #         float(obj_world_rot.z * (180 / 3.1415926535))
        #     ]
        # }
        # 轴方向（物体坐标系的X、Y、Z轴在世界坐标系中的方向向量）
        annotations["objects"][obj.name]["axis_directions"] = {
            "local_x": [float(local_x_in_world.x), float(local_x_in_world.y), float(local_x_in_world.z)],
            "local_y": [float(local_y_in_world.x), float(local_y_in_world.y), float(local_y_in_world.z)],
            "local_z": [float(local_z_in_world.x), float(local_z_in_world.y), float(local_z_in_world.z)]
        }
        
        # 三维包围框（关键修复：从物体获取bound_box，而非网格）
        try:
            # 获取评估后的物体（应用所有变换）
            evaluated_obj = obj.evaluated_get(depsgraph)
            
            # 方法1：直接使用物体的bound_box（推荐，已包含所有变换）
            local_bbox = [Vector(v) for v in evaluated_obj.bound_box]  # 从物体获取边界框
            
            # 转换到世界坐标系
            world_bbox = [evaluated_obj.matrix_world @ v for v in local_bbox]
            
            # 计算AABB
            min_x = min(v.x for v in world_bbox)
            max_x = max(v.x for v in world_bbox)
            min_y = min(v.y for v in world_bbox)
            max_y = max(v.y for v in world_bbox)
            min_z = min(v.z for v in world_bbox)
            max_z = max(v.z for v in world_bbox)
            
            annotations["objects"][obj.name]["bbox_3d_aabb"] = {
                "min": {"x": min_x, "y": min_y, "z": min_z},
                "max": {"x": max_x, "y": max_y, "z": max_z},
                "dimensions": {
                    "x": max_x - min_x,
                    "y": max_y - min_y,
                    "z": max_z - min_z
                }
            }
            
        except Exception as e:
            print(f"物体 {obj.name} 计算包围框失败: {str(e)}")
            annotations["objects"][obj.name]["bbox_3d_aabb"] = None  # 标记为失败
        

    # 读取各相机的GT数据（包含相机视角和物体分割结果）
    for cam_id in range(camera_num):
        try:
            gt = read_single_scene_gt(scene_dir, cam_id, frame_prefix)
            annotations["cameras"][f"camera_{cam_id}_0"] = gt
        except Exception as e:
            print(f"读取相机 {cam_id} 的GT数据失败: {str(e)}")
            annotations["cameras"][f"camera_{cam_id}_0"] = None  # 标记为失败

    # 新增步骤：过滤未在任何相机中出现的物体
    # 1. 收集所有相机中出现过的物体名称
    camera_object_names = set()
    for cam_name, cam_data in annotations["cameras"].items():
        if cam_data is None:
            continue  # 跳过读取失败的相机
        # 收集当前相机中所有物体的名称
        camera_object_names.update(cam_data.get("objects", {}).keys())

    # 2. 过滤场景物体：只保留在相机中出现过的物体
    filtered_objects = {}
    for obj_name, obj_data in annotations["objects"].items():
        if obj_name in camera_object_names:
            filtered_objects[obj_name] = obj_data
        # else:
            # print(f"移除未在任何相机中出现的物体: {obj_name}")

    # 3. 更新annotations中的物体数据
    annotations["objects"] = filtered_objects
    
    return annotations

# # 使用示例
# if __name__ == "__main__":
#     # 配置路径和参数
#     scene_name = "a71bd01"  # 替换为实际场景名
#     batch_name = "batch_generation_5"
#     base_dir = "/data/xujin/infinigen-main/outputs"   # 替换为实际的batch_generation根目录
#     scene_dir = os.path.join(base_dir, scene_name)
#     camera_num = 10  # 相机数量
#     frame_prefix = "0_0048_0"  # 帧标识（根据实际文件名调整）
#     stage = "fine"

#     scene_metadata = collect_scene_metadata(scene_dir, camera_num, stage, frame_prefix)

#     save_dict_to_json(scene_metadata, os.path.join(scene_dir, stage, "scene_metadata.json"), indent=4, ensure_ascii=False)
def copy_png_files(source_dir, target_dir):
    """
    复制源目录下所有.png文件到目标目录
    
    参数:
        source_dir: 源文件夹路径（要复制的.png文件所在目录）
        target_dir: 目标文件夹路径（复制到的目录）
    """
    # 检查源目录是否存在
    if not os.path.isdir(source_dir):
        print(f"错误：源目录不存在 - {source_dir}")
        return
    
    # 确保目标目录存在（不存在则创建）
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源目录下的所有文件
    for filename in os.listdir(source_dir):
        # 检查是否为.png文件
        if filename.lower().endswith(".png"):
            # 构建源文件和目标文件的完整路径
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            # 确保是文件（排除目录）
            if os.path.isfile(source_path):
                try:
                    # 复制文件（copy2保留元数据，如创建时间）
                    shutil.copy2(source_path, target_path)
                    # print(f"已复制：{filename}")
                except Exception as e:
                    print(f"复制失败 {filename}：{str(e)}")

def process_scene(scene_dir, camera_num, frame_prefix, save_dir):
    """处理单个场景：检查frames文件夹、选择stage并生成元数据"""
    # 检查frames文件夹是否存在
    frames_dir_candidate = os.path.join(scene_dir, 'frames')
    if not os.path.exists(frames_dir_candidate):
        return False, "frames folder not found"
    png_count = 0
    for item in os.listdir(os.path.join(frames_dir_candidate, 'Image', 'camera_0')):
        if item.lower().endswith('.png'):
            png_count += 1
    if png_count < camera_num:
        return False, "not enough frames"

    # 创建保存文件夹
    scene_id = os.path.basename(scene_dir)
    scene_save_dir = os.path.join(save_dir, scene_id)
    os.makedirs(scene_save_dir, exist_ok=True)

    # 自动选择stage
    if os.path.exists(os.path.join(scene_dir, 'fine')):
        stage = 'fine'
    else:
        stage = 'coarse'

    # 收集并保存元数据
    try:
        scene_metadata = collect_scene_metadata(scene_dir, camera_num, stage, frame_prefix)
        save_path = os.path.join(scene_save_dir, "scene_metadata.json")
        save_dict_to_json(scene_metadata, save_path)
        # 保存图像
        copy_png_files(os.path.join(frames_dir_candidate, 'Image', 'camera_0'), scene_save_dir)
        return True, f"successfully processed (stage: {stage})"
    except Exception as e:
        return False, f"processing failed: {str(e)}"
    
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Batch process scenes to generate metadata')
    parser.add_argument('--batch-names', nargs='+', required=True, 
                      help='List of batch names to process (space-separated)')
    parser.add_argument('--base-dir', required=True, 
                      help='Root directory containing batch folders')
    parser.add_argument('--save-dir', required=True, 
                      help='Root directory containing batch folders')
    parser.add_argument('--camera-num', type=int, default=10, 
                      help='Number of cameras (default: 10)')
    parser.add_argument('--frame-prefix', default="0_0048_0", 
                      help='Frame identifier prefix (default: "0_0048_0")')
    args = parser.parse_args()

    # 收集所有需要处理的场景路径
    all_scenes = []
    for batch_name in args.batch_names:
        batch_dir = os.path.join(args.base_dir, batch_name)
        if not os.path.isdir(batch_dir):
            print(f"Warning: Batch directory not found - {batch_dir}, skipping")
            continue
        
        # 获取批量下的所有场景文件夹
        for item in os.listdir(batch_dir):
            scene_path = os.path.join(batch_dir, item)
            if os.path.isdir(scene_path):
                all_scenes.append(scene_path)

    if not all_scenes:
        print("No valid scenes found for processing")
        return

    # 批量处理场景并显示进度条
    print(f"Found {len(all_scenes)} scenes to process...")
    success_count = 0
    for scene_dir in tqdm(all_scenes, desc="Processing scenes"):
        scene_name = os.path.basename(scene_dir)
        success, msg = process_scene(scene_dir, args.camera_num, args.frame_prefix, args.save_dir)
        # 统计成功数量
        if success:
            success_count += 1
        tqdm.write(f"Scene {scene_name}: {msg}")

    # 计算并显示统计结果
    total = len(all_scenes)
    if total == 0:
        print("No scenes processed")
    else:
        success_rate = (success_count / total) * 100
        print(f"\nProcessing complete!")
        print(f"Total scenes: {total}")
        print(f"Successful scenes: {success_count}")
        print(f"Success rate: {success_rate:.2f}%")  # 保留两位小数

if __name__ == "__main__":
    main()
