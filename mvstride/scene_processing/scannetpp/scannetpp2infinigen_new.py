# -*- coding: utf-8 -*-
from __future__ import annotations  # Python 3.8 兼容：避免注解在 import 时求值
import argparse
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, List, Union, Dict
import numpy as np
from PIL import Image, UnidentifiedImageError

# ===================== Global config =====================
# Raw data root directory (used to read images for width/height)
RAW_ROOT_DIR = "./scannetpp_sampled"
# New JSON annotation root directory (JSON is extracted from here)
NEW_ANNO_ROOT = "./scannetpp_sampled_new"
# Output data root directory (results are written here)
OUTPUT_ROOT_DIR = "./scannetpp_sampled_modified"

# Annotation filename config
ANNO_CONFIG = {
    "iphone": "scene_metadata.json",
    "dslr": "obj_annotation_dslr.json"
}
# ===================== Config end =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_image_size_pillow(img_path: Optional[Union[str, Path]] = None) -> tuple[int, int] or None:
    if img_path is None:
        logger.warning("原始图像不存在，使用默认尺寸 (1920, 1080)")
        return (1920, 1080)  # 默认值防止崩溃
    img_path = str(img_path)
    try:
        with Image.open(img_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"读取图片尺寸失败：{img_path} | 错误：{e}")
        return (1920, 1080)  # 默认值防止崩溃


class ScanNetPPDataProcessor:
    def __init__(self, raw_root: str = RAW_ROOT_DIR, new_anno_root: str = NEW_ANNO_ROOT, output_root: str = OUTPUT_ROOT_DIR,
                 copy_images: bool = True):
        self.raw_root = Path(raw_root)
        self.new_anno_root = Path(new_anno_root)
        self.output_root = Path(output_root)
        self.copy_images = copy_images
        self.output_root.mkdir(parents=True, exist_ok=True)
        # Use the folders under the new annotation directory as scene sources
        self.scene_dirs = [d for d in self.new_anno_root.iterdir() if d.is_dir()]
        logger.info(f"初始化完成，共发现 {len(self.scene_dirs)} 个待处理场景")

    def _load_json(self, file_path: Path) -> Optional[Union[dict, list]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取JSON失败：{file_path} | {e}")
            return None

    def _save_json(self, data: Union[dict, list], file_path: Path) -> bool:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            logger.error(f"保存JSON失败：{file_path} | {e}")
            return False

    def _convert_obb_to_aabb_format(self, centroid, axes_lengths, normalized_axes, coord_precision=6):
        try:
            centroid_np = np.array(centroid, dtype=np.float64)
            axes_lengths_np = np.array(axes_lengths, dtype=np.float64)
            rotation_matrix = np.array(normalized_axes, dtype=np.float64).reshape(3, 3)

            half_lengths = axes_lengths_np / 2.0
            signs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
            vertices = [centroid_np + rotation_matrix @ (sign * half_lengths) for sign in signs]
            vertices_np = np.array(vertices)

            min_coords = vertices_np.min(axis=0)
            max_coords = vertices_np.max(axis=0)
            dimensions = max_coords - min_coords

            return {
                "3d_center": [round(x, coord_precision) for x in centroid_np.tolist()],
                "axis_directions": {
                    "local_x": [round(x, coord_precision) for x in rotation_matrix[0].tolist()],
                    "local_y": [round(x, coord_precision) for x in rotation_matrix[1].tolist()],
                    "local_z": [round(x, coord_precision) for x in rotation_matrix[2].tolist()]
                },
                "bbox_3d_aabb": {
                    "min": {"x": round(min_coords[0], coord_precision), "y": round(min_coords[1], coord_precision), "z": round(min_coords[2], coord_precision)},
                    "max": {"x": round(max_coords[0], coord_precision), "y": round(max_coords[1], coord_precision), "z": round(max_coords[2], coord_precision)},
                    "dimensions": {"x": round(dimensions[0], coord_precision), "y": round(dimensions[1], coord_precision), "z": round(dimensions[2], coord_precision)}
                }
            }
        except:
            return None

    def _resolve_raw_image(self, scene_id: str, data_type: str, img_filename: str) -> Optional[Path]:
        # 原始图像可能位于两处：raw_root（历史抽帧目录为 *.jpg.jpg 双后缀）、
        # anno_root（preprocess_iphone.py 带 --no-images 之外会把帧图拷到 metadata 旁）
        candidates = [
            self.raw_root / scene_id / data_type / img_filename,
            self.raw_root / scene_id / data_type / (img_filename + ".jpg"),
            self.new_anno_root / scene_id / data_type / img_filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        logger.warning(f"找不到原始图像：{scene_id}/{data_type}/{img_filename}")
        return None

    def _process_annotation(self, raw_data, scene_id, data_type, dst_images_dir: Optional[Path] = None) -> dict:
        scene_objs = {}
        scene_objs_loc_list = {}
        scene_objs_id = 0
        modified_data = {
            "scene_id": scene_id,
            "data_type": data_type,
            "cameras": {},
            "objects": scene_objs
        }

        for img in raw_data:
            # 解析出与 scene_metadata.json 同名的原始帧图
            img_filename = Path(img.get("image_path", "")).name
            raw_img_path = self._resolve_raw_image(scene_id, data_type, img_filename)

            size = get_image_size_pillow(raw_img_path)
            W, H = size if size else (1920, 1080)

            camera_name = Path(img.get("image_path", "")).stem
            new_image_path = f"{scene_id}_{data_type}/images/{Path(img.get('image_path', '')).name}"
            if dst_images_dir is not None and raw_img_path is not None:
                dst_img = dst_images_dir / img_filename
                if not dst_img.exists():
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(raw_img_path, dst_img)

            c2w_colmap = np.linalg.inv(np.array(img.get("extrinsic", [])))
            R_conversion = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) if data_type == "iphone" else np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

            R_old = c2w_colmap[:3, :3]
            t = c2w_colmap[:3, 3]
            R_new = R_old @ R_conversion

            camera_extrinsics = np.eye(4)
            camera_extrinsics[:3, :3] = R_new
            camera_extrinsics[:3, 3] = t

            modified_data["cameras"][camera_name] = {
                "cam_intrinsics": img.get("intrinsic", []),
                "cam_extrinsics": camera_extrinsics.tolist(),
                "width": W, "height": H,
                "image_path": new_image_path,
                "location_3d": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
                "forward_direction": {"x": float(camera_extrinsics[0, 2]), "y": float(camera_extrinsics[1, 2]), "z": float(camera_extrinsics[2, 2])},
                "objects": {}
            }

            camera_objs = {}
            for obj in img.get("objects", []):
                obj_category = obj.get("category", "unknown")
                obj_3d_center = obj.get("3D_location", [])

                key = (obj_category, tuple(obj_3d_center))
                if key in scene_objs_loc_list:
                    current_obj_id = scene_objs_loc_list[key]
                else:
                    current_obj_id = scene_objs_id
                    scene_objs_loc_list[key] = current_obj_id
                    res = self._convert_obb_to_aabb_format(obj_3d_center, obj['3D_size'], obj['3D_rotation'])
                    if res:
                        modified_data["objects"][str(current_obj_id)] = {
                            "category": obj_category,
                            "3d_center": obj_3d_center,
                            "axis_directions": res["axis_directions"],
                            "bbox_3d_aabb": res["bbox_3d_aabb"]
                        }
                        scene_objs_id += 1

                camera_objs[str(current_obj_id)] = {
                    "object_index": current_obj_id,
                    "bbox_2d": {
                        "min_x": obj.get("2D_bbox", [])[1], "min_y": obj.get("2D_bbox", [])[0],
                        "max_x": obj.get("2D_bbox", [])[3], "max_y": obj.get("2D_bbox", [])[2]
                    }
                }
            modified_data["cameras"][camera_name]["objects"] = camera_objs

        return modified_data

    def process_single_scene(self, scene_anno_dir: Path) -> None:
        scene_id = scene_anno_dir.name
        # Per the requirement, only process iphone
        data_type = "iphone"

        # 1. Build the paths
        # Read the JSON from the new path
        src_anno_path = scene_anno_dir / ANNO_CONFIG[data_type]
        # Save to the iphone folder under the original path structure
        dst_scene_dir = self.output_root / f"{scene_id}_{data_type}"
        dst_anno_path = dst_scene_dir / "scene_metadata_new.json"

        # 2. Process
        raw_anno = self._load_json(src_anno_path)
        if raw_anno:
            dst_images_dir = dst_scene_dir / "images" if self.copy_images else None
            modified_anno = self._process_annotation(raw_anno, scene_id, data_type, dst_images_dir)
            if self._save_json(modified_anno, dst_anno_path):
                logger.info(f"场景 {scene_id} 处理完成并保存至 {dst_anno_path}")
        else:
            logger.warning(f"场景 {scene_id} 未找到新标注文件，跳过")

    def run(self) -> None:
        for scene_dir in tqdm(self.scene_dirs, desc="处理进度"):
            self.process_single_scene(scene_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="把 scene_metadata.json 转为 Infinigen 统一 schema 的 scene_metadata_new.json，并拷贝帧图到输出目录")
    parser.add_argument("--raw-root", default=RAW_ROOT_DIR,
                        help="原始帧图目录（含 <scene>/iphone/*.jpg；兼容历史 *.jpg.jpg 双后缀）")
    parser.add_argument("--anno-root", default=NEW_ANNO_ROOT,
                        help="scene_metadata.json 所在目录（preprocess_iphone.py 的输出）")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DIR,
                        help="输出目录，按 <scene>_iphone/ 组织，与 QA 配置对齐")
    parser.add_argument("--no-images", action="store_true", default=False,
                        help="不把帧图拷贝到 <scene>_iphone/images/（默认会拷贝）")
    args = parser.parse_args()
    processor = ScanNetPPDataProcessor(
        raw_root=args.raw_root,
        new_anno_root=args.anno_root,
        output_root=args.output_root,
        copy_images=not args.no_images,
    )
    processor.run()
