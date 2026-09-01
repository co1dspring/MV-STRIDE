# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import json
from PIL import Image  # (added) for reading PNG images
from typing import Dict, Any
from pathlib import Path
import bpy
import json
import argparse  # use argparse for argument parsing
import sys
from mathutils import Vector
from tqdm import tqdm
import shutil

def extract_category(obj_name: str) -> str:
    """Extract the category from an object name."""
    # 1. Handle names with Factory (e.g. "BookStackFactory(xxx).spawn_asset(xxx)")
    factory_match = re.match(r"^(\w+)Factory\(\d+\)\.spawn_asset\(\d+\)(\.\d+)?$", obj_name)
    if factory_match:
        base = factory_match.group(1)
        # CamelCase to lowercase with spaces (e.g. BookStack -> book stack)
        base = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', base).lower()
        return base
    
    # 2. Handle scene-structure class names (e.g. "dining-room_0/0.floor", "skirtingboard_ceiling")
    # Extract the house-name prefix + structure type (e.g. "dining-room ceiling", "skirtingboard ceiling")
    # First handle the "/" format (e.g. "dining-room_0/0.floor")
    if '/' in obj_name:
        # Split into house-name part and structure part (e.g. ["dining-room_0", "0.floor"])
        house_part, struct_part = obj_name.split('/', 1)
        # Extract the house-name core (drop digits and underscores, e.g. "dining-room_0" -> "dining-room")
        house_core = re.sub(r'_\d+$', '', house_part)
        # Extract the structure type (e.g. "0.floor" -> "floor")
        struct_type = re.sub(r'^\d+\.', '', struct_part).lower()
        return f"{house_core} {struct_type}"
    
    # Handle the "_" format (e.g. "skirtingboard_ceiling")
    if '_' in obj_name:
        # Split into prefix and structure type (keep house-related prefix)
        parts = obj_name.split('_', 1)
        # Ensure both parts are valid (non-empty and not purely numeric)
        if all(parts) and not parts[0].isdigit() and not parts[1].isdigit():
            return f"{parts[0].lower()} {parts[1].lower()}"
    
    # 3. No special format: return the original name lowercased
    return obj_name.lower()

def save_dict_to_json(data: Dict[Any, Any], file_path: str or Path, indent: int = 4, ensure_ascii: bool = False) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data: The dictionary data to save.
        file_path: The file path to save to (str or Path object).
        indent: Number of spaces for JSON formatting indentation, default 4 (0 means no formatting).
        ensure_ascii: Whether to ensure ASCII encoding (False allows Chinese to display), default False.
    """
    # Convert to a Path object for easier path handling
    file_path = Path(file_path)
    
    # Create the parent directory (if it does not exist)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Write the JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                default=str  # Auto-convert non-serializable objects to strings
            )
        print(f"字典已成功保存到: {file_path.resolve()}")
    except Exception as e:
        print(f"保存JSON文件失败: {e}")

def get_object_2d_bboxes(object_segmentation: np.ndarray) -> Dict[int, Dict[str, int]]:
    """
    Extract a 2D bounding box for each object ID from a single-channel segmentation map.

    Args:
        object_segmentation: single-channel segmentation map as a numpy array (shape HxW, pixel values are object IDs).

    Returns:
        A dict keyed by object ID, with values being bbox dicts {"min_x", "min_y", "max_x", "max_y"}.
    """
    # Get all unique object IDs (a condition such as id != 0 can be added to exclude the background)
    unique_ids = np.unique(object_segmentation)
    
    object_bboxes = {}
    for obj_id in unique_ids:
        # Find all pixel coordinates for the current object ID
        # np.where returns two arrays: the first is the y coordinate (rows), the second is the x coordinate (columns)
        y_coords, x_coords = np.where(object_segmentation == obj_id)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            continue  # Should not happen in theory; skip empty results
        
        # Compute the bounding box: min/max x and y
        min_x = int(np.min(x_coords))
        max_x = int(np.max(x_coords))
        min_y = int(np.min(y_coords))
        max_y = int(np.max(y_coords))
        
        # Store the bounding box info
        object_bboxes[str(obj_id)] = {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y
        }
    
    return object_bboxes

def read_single_scene_gt(scene_dir: str, camera_id: int, frame_prefix: str = "0_0048_0") -> Dict[str, Any]:
    """
    Read GT data for a single scene (camera view, RGB image, instance segmentation, object metadata, object segmentation).

    Args:
        scene_dir: The scene root directory path (corresponds to a batch_generation/scene_name).
        camera_id: The camera ID (0~9).
        frame_prefix: The frame identifier prefix (default "0_0048_0", matching the fixed part of filenames).

    Returns:
        A dict containing various GT data, keyed as follows:
            - camview: camera view data
            - image: RGB image data (numpy array, shape HxWx3)
            - instance_segmentation: instance segmentation result
            - objects_meta: object metadata (JSON parse result)
            - object_segmentation: object segmentation result
    """
    

    # Build the data path templates
    camview_path = os.path.join(
        scene_dir, "frames", "camview", f"camera_0",
        f"camview_{camera_id}_{frame_prefix}.npz"
    )
    # Change: image path suffix changed to .png
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
    
    # Read the data
    camera_info = {}
    
    # Read the camera parameters (key fix: distinguish intrinsics from extrinsics)
    if os.path.exists(camview_path):
        with np.load(camview_path) as data:
            # Explicitly read intrinsics (3x3) and extrinsics (4x4); key names follow official docs or the actual file
            # Common key names: "intrinsics", "extrinsics"
            extrinsics = data["T"] 
            # Read the intrinsics (3x3)
            camera_info["cam_intrinsics"] = data["K"]  # K = intrinsic matrix
            # Read the extrinsics (4x4)
            camera_info["cam_extrinsics"] = extrinsics  # T = extrinsic matrix (camera -> world)
            # Read the image size [H, W]
            camera_info["image_size_HW"] = data["HW"]     # HW = [height, width])

            # 3. Extract the camera position in the world coordinate system (Tx, Ty, Tz)
            # In the extrinsic matrix, column 4 of the first 3 rows is the world coordinate of the camera center
            cam_world_loc = extrinsics[:3, 3]
            camera_info["location_3d"] = {
                "x": float(cam_world_loc[0]),
                "y": float(cam_world_loc[1]),
                "z": float(cam_world_loc[2])
            }
            
            # 4. Extract the camera orientation (view direction vector, in world coordinates)
            # In the camera coordinate system, the view direction is the -Z axis (vector [0, 0, -1])
            # Transform it to the world coordinate system via the rotation matrix R
            R = extrinsics[:3, :3]  # Rotation matrix (3x3)
            cam_local_forward = np.array([0, 0, -1])  # camera-local view direction (-Z axis)
            cam_world_forward = R @ cam_local_forward  # Convert to the world coordinate system
            cam_world_forward = cam_world_forward / np.linalg.norm(cam_world_forward)  # Normalize
            camera_info["forward_direction"] = {
                "x": float(cam_world_forward[0]),
                "y": float(cam_world_forward[1]),
                "z": float(cam_world_forward[2])
            }
    else:
        raise FileNotFoundError(f"相机参数文件不存在: {camview_path}")
    
    # Read the RGB image data (.png) - modified part
    if os.path.exists(image_path):
        # Read the PNG with PIL, convert to RGB mode (to avoid the alpha channel), then convert to a numpy array
        # with Image.open(image_path) as img:
        #     rgb_img = img.convert("RGB")  # Make sure it is 3-channel RGB
        #     gt_data["image"] = np.array(rgb_img)  # Shape is (H, W, 3)
        camera_info["image_path"] = image_path  # Only save the image path to avoid storing a large array
    else:
        raise FileNotFoundError(f"RGB图像文件不存在: {image_path}")
    
    # Read the instance segmentation result (.npy)
    # if os.path.exists(instance_seg_path):
    #     gt_data["instance_segmentation"] = np.load(instance_seg_path)
    # else:
    #     raise FileNotFoundError(f"Instance segmentation file not found: {instance_seg_path}")
    
    # Read the object metadata (.json)
    if os.path.exists(objects_meta_path):
        with open(objects_meta_path, "r", encoding="utf-8") as f:
            scene_metadata = json.load(f)
    else:
        raise FileNotFoundError(f"物体元信息文件不存在: {objects_meta_path}")
    
    # Read the object segmentation result (.npy)
    if os.path.exists(object_seg_path):
        object_segmentation = np.load(object_seg_path)
        objects_in_camera = get_object_2d_bboxes(object_segmentation)
    else:
        raise FileNotFoundError(f"物体分割文件不存在: {object_seg_path}")
    
    # Merge the segmentation result with the object metadata
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
    # Clear the default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Load the .blend file
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    
    annotations = {"cameras": {}, "objects": {}}
    depsgraph = bpy.context.evaluated_depsgraph_get()  # Get the dependency graph (used to compute data after object transforms)
    
    # Iterate over all objects in the scene except cameras
    for obj in bpy.data.objects:
        # # 1. Handle cameras (extract camera-specific parameters separately)
        # if obj.type == "CAMERA":
        #     cam_data = obj.data  # camera intrinsic data
            
        #     # Camera position (origin in world coordinates)
        #     cam_location = obj.location
        #     cam_world_location = obj.matrix_world.translation  # world position
            
        #     # Camera orientation (as rotation Euler angles, or converted to a forward vector)
        #     rotation_euler = obj.rotation_euler
        #     # Compute the camera forward vector (Blender cameras face -Z by default; needs conversion to world coordinates)
        #     forward = -obj.matrix_world.to_quaternion() @ Vector((0, 0, 1))  # forward vector (pointing in the view direction)

        #     # 2. Camera world rotation (absolute rotation)
        #     # Method 1: Euler angles (world coordinates)
        #     cam_world_rot_euler = obj.matrix_world.to_euler()
        #     # Method 2: forward vector (view direction in world coordinates)
        #     cam_forward = -obj.matrix_world.to_quaternion() @ Vector((0, 0, 1))  # cameras face -Z by default
            
        #     # Camera intrinsics (focal length, principal point, etc.)
        #     intrinsics = {
        #         "focal_length_mm": cam_data.lens,  # focal length (mm)
        #         "sensor_width_mm": cam_data.sensor_width,  # sensor width (mm)
        #         "sensor_height_mm": cam_data.sensor_height,  # sensor height (mm)
        #         "principal_point": {  # principal point offset (relative to sensor center, in mm)
        #             "x": cam_data.shift_x * cam_data.sensor_width,
        #             "y": cam_data.shift_y * cam_data.sensor_height
        #         }
        #     }
            
        #     # Assemble the camera info
        #     camera_info = {
        #         "name": obj.name,
        #         "object_index": obj.pass_index,  # if the camera has an object_index
        #         "type": "CAMERA",
        #         "location_3d": {  # position in world coordinates
        #             "x": float(cam_world_location.x),
        #             "y": float(cam_world_location.y),
        #             "z": float(cam_world_location.z)
        #         },
        #         "rotation_world": {  # absolute rotation in world coordinates
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
        #             "forward_direction": {  # view direction in world coordinates
        #                 "x": float(cam_forward.x),
        #                 "y": float(cam_forward.y),
        #                 "z": float(cam_forward.z)
        #             }
        #         },
        #         "intrinsics": intrinsics  # intrinsic parameters
        #     }
        #     annotations.append(camera_info)
        #     continue  # camera handled; skip the following generic object logic


        # Filter out non-entity objects
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
            # Raise an error on duplicate objects and exit
            raise ValueError(f"发现重复物体名称: {obj.name}")
        
        # 1. Object world position (absolute coordinates)
        obj_world_loc = obj.matrix_world.translation  # Not affected by the parent object
        
        # 2. Object world rotation (absolute rotation)
        # obj_world_rot = obj.matrix_world.to_euler()   # extract rotation from the world matrix
        world_rot_matrix = obj.matrix_world # get the object world rotation matrix (extracted from the world transform matrix)

        # The rotation matrix column vectors correspond to the object local X, Y, Z axes in world coordinates
        # Normalize to ensure unit vector length
        local_x_in_world = world_rot_matrix.col[0].normalized()  # local X axis (right direction)
        local_y_in_world = world_rot_matrix.col[1].normalized()  # local Y axis (forward direction)
        local_z_in_world = world_rot_matrix.col[2].normalized()  # local Z axis (up direction)

        # Center coordinate (object origin)
        # center_world = obj.location
        annotations["objects"][obj.name]["3d_center"] = [float(obj_world_loc.x), float(obj_world_loc.y), float(obj_world_loc.z)]
        
        # Orientation (Euler angles)
        # rotation_euler = obj.rotation_euler
        # annotations["objects"][obj.name]["rotation"] = {
        #     "euler_rad": [float(obj_world_rot.x), float(obj_world_rot.y), float(obj_world_rot.z)],
        #     "euler_deg": [
        #         float(obj_world_rot.x * (180 / 3.1415926535)),
        #         float(obj_world_rot.y * (180 / 3.1415926535)),
        #         float(obj_world_rot.z * (180 / 3.1415926535))
        #     ]
        # }
        # Axis directions (direction vectors of the object X, Y, Z axes in world coordinates)
        annotations["objects"][obj.name]["axis_directions"] = {
            "local_x": [float(local_x_in_world.x), float(local_x_in_world.y), float(local_x_in_world.z)],
            "local_y": [float(local_y_in_world.x), float(local_y_in_world.y), float(local_y_in_world.z)],
            "local_z": [float(local_z_in_world.x), float(local_z_in_world.y), float(local_z_in_world.z)]
        }
        
        # 3D bounding box (key fix: get bound_box from the object, not the mesh)
        try:
            # Get the evaluated object (with all transforms applied)
            evaluated_obj = obj.evaluated_get(depsgraph)
            
            # Method 1: use the object bound_box directly (recommended, already includes all transforms)
            local_bbox = [Vector(v) for v in evaluated_obj.bound_box]  # Get the bounding box from the object
            
            # Convert to the world coordinate system
            world_bbox = [evaluated_obj.matrix_world @ v for v in local_bbox]
            
            # Compute the AABB
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
            annotations["objects"][obj.name]["bbox_3d_aabb"] = None  # Mark as failed
        

    # Read GT data for each camera (camera view and object segmentation result)
    for cam_id in range(camera_num):
        try:
            gt = read_single_scene_gt(scene_dir, cam_id, frame_prefix)
            annotations["cameras"][f"camera_{cam_id}_0"] = gt
        except Exception as e:
            print(f"读取相机 {cam_id} 的GT数据失败: {str(e)}")
            annotations["cameras"][f"camera_{cam_id}_0"] = None  # Mark as failed

    # New step: filter out objects not appearing in any camera
    # 1. Collect the names of objects appearing in all cameras
    camera_object_names = set()
    for cam_name, cam_data in annotations["cameras"].items():
        if cam_data is None:
            continue  # Skip cameras that failed to read
        # Collect the names of all objects in the current camera
        camera_object_names.update(cam_data.get("objects", {}).keys())

    # 2. Filter scene objects: keep only objects appearing in the cameras
    filtered_objects = {}
    for obj_name, obj_data in annotations["objects"].items():
        if obj_name in camera_object_names:
            filtered_objects[obj_name] = obj_data
        # else:
            # print(f"Removed an object that does not appear in any camera: {obj_name}")

    # 3. Update the object data in annotations
    annotations["objects"] = filtered_objects
    
    return annotations

# # Usage example
# if __name__ == "__main__":
#     # Configure the paths and parameters
#     scene_name = "a71bd01"  # replace with an actual scene name
#     batch_name = "batch_generation_5"
#     base_dir = "/path/to/infinigen_main/outputs"   # replace with the actual batch_generation root directory
#     scene_dir = os.path.join(base_dir, scene_name)
#     camera_num = 10  # number of cameras
#     frame_prefix = "0_0048_0"  # frame identifier (adjust per actual filenames)
#     stage = "fine"

#     scene_metadata = collect_scene_metadata(scene_dir, camera_num, stage, frame_prefix)

#     save_dict_to_json(scene_metadata, os.path.join(scene_dir, stage, "scene_metadata.json"), indent=4, ensure_ascii=False)
def copy_png_files(source_dir, target_dir):
    """
    Copy all .png files from the source directory to the target directory.

    Args:
        source_dir: the source folder path (directory containing the .png files to copy).
        target_dir: the target folder path (directory to copy into).
    """
    # Check whether the source directory exists
    if not os.path.isdir(source_dir):
        print(f"错误：源目录不存在 - {source_dir}")
        return
    
    # Ensure the target directory exists (create if missing)
    os.makedirs(target_dir, exist_ok=True)
    
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        # Check whether it is a .png file
        if filename.lower().endswith(".png"):
            # Build the full source and target paths
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            # Make sure it is a file (exclude directories)
            if os.path.isfile(source_path):
                try:
                    # Copy the file (copy2 preserves metadata such as the creation time)
                    shutil.copy2(source_path, target_path)
                    # print(f"Copied: {filename}")
                except Exception as e:
                    print(f"复制失败 {filename}：{str(e)}")

def process_scene(scene_dir, camera_num, frame_prefix, save_dir):
    """Process a single scene: check the frames folder, choose the stage, and generate metadata."""
    # Check whether the frames folder exists
    frames_dir_candidate = os.path.join(scene_dir, 'frames')
    if not os.path.exists(frames_dir_candidate):
        return False, "frames folder not found"
    png_count = 0
    for item in os.listdir(os.path.join(frames_dir_candidate, 'Image', 'camera_0')):
        if item.lower().endswith('.png'):
            png_count += 1
    if png_count < camera_num:
        return False, "not enough frames"

    # Create the save folder
    scene_id = os.path.basename(scene_dir)
    scene_save_dir = os.path.join(save_dir, scene_id)
    os.makedirs(scene_save_dir, exist_ok=True)

    # Auto-select the stage
    if os.path.exists(os.path.join(scene_dir, 'fine')):
        stage = 'fine'
    else:
        stage = 'coarse'

    # Collect and save the metadata
    try:
        scene_metadata = collect_scene_metadata(scene_dir, camera_num, stage, frame_prefix)
        save_path = os.path.join(scene_save_dir, "scene_metadata.json")
        save_dict_to_json(scene_metadata, save_path)
        # Save the images
        copy_png_files(os.path.join(frames_dir_candidate, 'Image', 'camera_0'), scene_save_dir)
        return True, f"successfully processed (stage: {stage})"
    except Exception as e:
        return False, f"processing failed: {str(e)}"
    
def main():
    # Parse the command-line arguments
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

    # Collect all scene paths that need processing
    all_scenes = []
    for batch_name in args.batch_names:
        batch_dir = os.path.join(args.base_dir, batch_name)
        if not os.path.isdir(batch_dir):
            print(f"Warning: Batch directory not found - {batch_dir}, skipping")
            continue
        
        # Get all scene folders under the batch
        for item in os.listdir(batch_dir):
            scene_path = os.path.join(batch_dir, item)
            if os.path.isdir(scene_path):
                all_scenes.append(scene_path)

    if not all_scenes:
        print("No valid scenes found for processing")
        return

    # Process scenes in a batch and show a progress bar
    print(f"Found {len(all_scenes)} scenes to process...")
    success_count = 0
    for scene_dir in tqdm(all_scenes, desc="Processing scenes"):
        scene_name = os.path.basename(scene_dir)
        success, msg = process_scene(scene_dir, args.camera_num, args.frame_prefix, args.save_dir)
        # Count the successes
        if success:
            success_count += 1
        tqdm.write(f"Scene {scene_name}: {msg}")

    # Compute and show the statistics
    total = len(all_scenes)
    if total == 0:
        print("No scenes processed")
    else:
        success_rate = (success_count / total) * 100
        print(f"\nProcessing complete!")
        print(f"Total scenes: {total}")
        print(f"Successful scenes: {success_count}")
        print(f"Success rate: {success_rate:.2f}%")  # Keep two decimal places

if __name__ == "__main__":
    main()
