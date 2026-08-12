# -*- coding: gbk -*-
import os
import numpy as np
import json
from PIL import Image  # 新增：用于读取PNG图像
from typing import Dict, Any


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
        scene_dir, "frames", "camview", f"camera_{camera_id}",
        f"camview_{camera_id}_{frame_prefix}.npz"
    )
    # 修改：图像路径后缀改为.png
    image_path = os.path.join(
        scene_dir, "frames", "Image", f"camera_{camera_id}",
        f"Image_{camera_id}_{frame_prefix}.png"
    )
    instance_seg_path = os.path.join(
        scene_dir, "frames", "InstanceSegmentation", f"camera_{camera_id}",
        f"InstanceSegmentation_{camera_id}_{frame_prefix}.npy"
    )
    objects_meta_path = os.path.join(
        scene_dir, "frames", "Objects", f"camera_{camera_id}",
        f"Objects_{camera_id}_{frame_prefix}.json"
    )
    object_seg_path = os.path.join(
        scene_dir, "frames", "ObjectSegmentation", f"camera_{camera_id}",
        f"ObjectSegmentation_{camera_id}_{frame_prefix}.npy"
    )
    
    # 读取数据
    gt_data = {}
    
    # 读取相机参数（关键修正：区分内参和外参）
    if os.path.exists(camview_path):
        with np.load(camview_path) as data:
            # 显式读取内参（3x3）和外参（4x4），键名参考官方文档或实际文件
            # 常见键名："intrinsics"（内参）、"extrinsics"（外参）
            # 读取内参（3x3）
            gt_data["cam_intrinsics"] = data["K"]  # K = 内参矩阵
            # 读取外参（4x4）
            gt_data["cam_extrinsics"] = data["T"]  # T = 外参矩阵（相机→世界）
            # 读取图像尺寸 [H, W]
            gt_data["image_size"] = data["HW"]     # HW = [高度, 宽度]）
    else:
        raise FileNotFoundError(f"相机参数文件不存在: {camview_path}")
    
    # 读取RGB图像数据（.png）- 修改部分
    if os.path.exists(image_path):
        # 使用PIL读取PNG，转换为RGB模式（避免alpha通道），再转为numpy数组
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")  # 确保是3通道RGB
            gt_data["image"] = np.array(rgb_img)  # 形状为 (H, W, 3)
    else:
        raise FileNotFoundError(f"RGB图像文件不存在: {image_path}")
    
    # 读取实例分割结果（.npy）
    if os.path.exists(instance_seg_path):
        gt_data["instance_segmentation"] = np.load(instance_seg_path)
    else:
        raise FileNotFoundError(f"实例分割文件不存在: {instance_seg_path}")
    
    # 读取物体元信息（.json）
    if os.path.exists(objects_meta_path):
        with open(objects_meta_path, "r", encoding="utf-8") as f:
            gt_data["objects_meta"] = json.load(f)
    else:
        raise FileNotFoundError(f"物体元信息文件不存在: {objects_meta_path}")
    
    # 读取物体分割结果（.npy）
    if os.path.exists(object_seg_path):
        gt_data["object_segmentation"] = np.load(object_seg_path)
    else:
        raise FileNotFoundError(f"物体分割文件不存在: {object_seg_path}")
    
    return gt_data


# 使用示例
if __name__ == "__main__":
    # 配置路径和参数
    scene_name = "a71bd01"  # 替换为实际场景名
    base_dir = "/data/xujin/infinigen-main/outputs/batch_generation_5"   # 替换为实际的batch_generation根目录
    scene_dir = os.path.join(base_dir, scene_name)
    camera_id = 0  # 相机编号（0~9）
    frame_prefix = "0_0048_0"  # 帧标识（根据实际文件名调整）
    
    try:
        # 读取GT数据
        gt = read_single_scene_gt(scene_dir, camera_id, frame_prefix)
        
        # 打印数据信息（验证读取结果）
        print(f"相机视角数据形状: {gt['cam_intrinsics'].shape}, {gt['cam_extrinsics'].shape}")
        print(f"RGB图像数据形状: {gt['image'].shape} (HxWx3)")
        print(f"实例分割数据形状: {gt['instance_segmentation'].shape}")
        print(f"物体元信息包含物体数量: {len(gt['objects_meta'])}")
        print(f"物体分割数据形状: {gt['object_segmentation'].shape}")
        
        # 打印数据信息（验证读取结果）
        print(f"相机视角数据: {gt['cam_intrinsics']}, {gt['cam_extrinsics']}")
        print(f"实例分割数据: {gt['instance_segmentation']}")
        print(f"物体元信息: {gt['objects_meta']}")
        print(f"物体分割数据: {gt['object_segmentation']}")

    except Exception as e:
        print(f"读取失败: {str(e)}")
