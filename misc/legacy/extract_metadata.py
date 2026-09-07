# -*- coding: utf-8 -*-
import os
import numpy as np
import json
from PIL import Image  # Added: for reading PNG images
from typing import Dict, Any


def read_single_scene_gt(scene_dir: str, camera_id: int, frame_prefix: str = "0_0048_0") -> Dict[str, Any]:
    """
    Read GT data for a single scene (camera views, RGB images, instance segmentation, object metadata, object segmentation).
    
    Args:
        scene_dir: scene root directory (corresponds to batch_generation/<scene name>)
        camera_id: camera id (0~9)
        frame_prefix: frame name prefix (default "0_0048_0", the fixed part of file names)
    
    Returns:
        A dict containing the various GT data, with keys:
            - camview: camera view data
            - image: RGB image data (numpy array, shape HxWx3)
            - instance_segmentation: instance segmentation result
            - objects_meta: object metadata (parsed JSON)
            - object_segmentation: object segmentation result
    """
    # Build path templates for each data source
    camview_path = os.path.join(
        scene_dir, "frames", "camview", f"camera_{camera_id}",
        f"camview_{camera_id}_{frame_prefix}.npz"
    )
    # Note: image path suffix changed to .png
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
    
    # Read the data
    gt_data = {}
    
    # Read camera parameters (key fix: distinguish intrinsics from extrinsics)
    if os.path.exists(camview_path):
        with np.load(camview_path) as data:
            # Explicitly read intrinsics (3x3) and extrinsics (4x4); key names follow the official docs or actual files
            # Common key names: "intrinsics", "extrinsics"
            # Read intrinsics (3x3)
            gt_data["cam_intrinsics"] = data["K"]  # K = intrinsics matrix
            # Read extrinsics (4x4)
            gt_data["cam_extrinsics"] = data["T"]  # T = extrinsics matrix (camera to world)
            # Read image size [H, W]
            gt_data["image_size"] = data["HW"]     # HW = [height, width]
    else:
        raise FileNotFoundError(f"相机参数文件不存在: {camview_path}")
    
    # Read RGB image data (.png) - modified part
    if os.path.exists(image_path):
        # Use PIL to read PNG, convert to RGB mode (avoid alpha channel), then to a numpy array
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")  # ensure it is 3-channel RGB
            gt_data["image"] = np.array(rgb_img)  # shape is (H, W, 3)
    else:
        raise FileNotFoundError(f"RGB图像文件不存在: {image_path}")
    
    # Read instance segmentation result (.npy)
    if os.path.exists(instance_seg_path):
        gt_data["instance_segmentation"] = np.load(instance_seg_path)
    else:
        raise FileNotFoundError(f"实例分割文件不存在: {instance_seg_path}")
    
    # Read object metadata (.json)
    if os.path.exists(objects_meta_path):
        with open(objects_meta_path, "r", encoding="utf-8") as f:
            gt_data["objects_meta"] = json.load(f)
    else:
        raise FileNotFoundError(f"物体元信息文件不存在: {objects_meta_path}")
    
    # Read object segmentation result (.npy)
    if os.path.exists(object_seg_path):
        gt_data["object_segmentation"] = np.load(object_seg_path)
    else:
        raise FileNotFoundError(f"物体分割文件不存在: {object_seg_path}")
    
    return gt_data


# Usage example
if __name__ == "__main__":
    # Configure the paths and parameters
    scene_name = "a71bd01"  # replace with the actual scene name
    base_dir = "/path/to/infinigen_main/outputs/batch_generation_5"   # replace with the actual batch_generation root directory
    scene_dir = os.path.join(base_dir, scene_name)
    camera_id = 0  # camera id (0~9)
    frame_prefix = "0_0048_0"  # frame prefix (adjust according to the actual file names)
    
    try:
        # Read GT data
        gt = read_single_scene_gt(scene_dir, camera_id, frame_prefix)
        
        # Print data info (verify read results)
        print(f"相机视角数据形状: {gt['cam_intrinsics'].shape}, {gt['cam_extrinsics'].shape}")
        print(f"RGB图像数据形状: {gt['image'].shape} (HxWx3)")
        print(f"实例分割数据形状: {gt['instance_segmentation'].shape}")
        print(f"物体元信息包含物体数量: {len(gt['objects_meta'])}")
        print(f"物体分割数据形状: {gt['object_segmentation'].shape}")
        
        # Print data info (verify read results)
        print(f"相机视角数据: {gt['cam_intrinsics']}, {gt['cam_extrinsics']}")
        print(f"实例分割数据: {gt['instance_segmentation']}")
        print(f"物体元信息: {gt['objects_meta']}")
        print(f"物体分割数据: {gt['object_segmentation']}")

    except Exception as e:
        print(f"读取失败: {str(e)}")
