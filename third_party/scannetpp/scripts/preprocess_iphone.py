#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project per-object 3D annotations of a ScanNet++ scene onto sampled iPhone
frames, and emit a compact per-scene `scene_metadata.json`.

The heavy lifting (scene asset paths, COLMAP pose loading, mesh rasterization,
2D bbox extraction) is delegated to the official ScanNet++ codebase, which must
be installed and importable as the `scannetpp` package (see the parent README):

  * scannetpp/common/scene_release.py    -- official per-scene directory layout
  * scannetpp/common/utils/colmap.py     -- get_camera_images_poses: COLMAP -> poses
  * scannetpp/common/utils/rasterize.py  -- pytorch3d mesh rasterization helpers
  * scannetpp/common/utils/anno.py       -- load_anno_wrapper / get_bboxes_2d / get_vtx_prop_on_2d
  * scannetpp/common/utils/image.py      -- load_image

This script is adapted from the official
`scannetpp/semantic/prep/semantics_2d.py`; differences:

  * rasterization is done on the fly per frame (the official script consumes
    precomputed pix_to_face/zbuf `.pth` caches);
  * only the iPhone branch is kept (the DSLR/undistortion branch is dropped);
  * no semantic GT, image crops or visualization outputs are produced;
  * frames / objects are filtered by visibility (see the constants below) and
    an unexpected-class blacklist, because background/structure objects such as
    walls or floors are not meaningful QA targets;
  * every selected frame is written to a JSON record consumed by
    `scannetpp2infinigen_new.py` (the Infinigen-schema converter);
  * with images (default; disable with --no-images), the selected frames are
    copied next to the metadata so that the output directory is self-contained.

Frame sampling is deterministic: the COLMAP-registered frames listed in
`data/<scene>/iphone/colmap/images.txt` are sorted by id and every
`--sample-rate`-th frame is kept (official get_camera_images_poses). The
official iPhone registration is done on roughly every 10th video frame, so
`--sample-rate 5` yields ~1 frame per 50 video frames (~120-130 frames/scene).
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from pytorch3d.structures import Meshes
from tqdm import tqdm

from scannetpp.common.scene_release import ScannetppScene_Release
from scannetpp.common.utils.anno import (
    get_bboxes_2d,
    get_vtx_prop_on_2d,
    load_anno_wrapper,
)
from scannetpp.common.utils.colmap import camera_to_intrinsic, get_camera_images_poses
from scannetpp.common.utils.image import load_image
from scannetpp.common.utils.rasterize import (
    get_opencv_cameras_batch,
    prep_pt3d_inputs,
    rasterize_mesh,
)

logger = logging.getLogger("scannetpp_preprocess_iphone")


def save_json(data, file_path, indent=4, ensure_ascii=False):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def read_txt_to_list(file_path, skip_empty=True):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if skip_empty:
        lines = [line for line in lines if line.strip()]
    return lines


class scannetpp_dataset:
    def __init__(self, data_dir, output_dir, sample_rate=5, device="cuda:0",
                 scene_list_file=None, with_images=True):
        # data_dir directly contains the per-scene directories of the official
        # release layout: data_dir/<scene_id>/{iphone,scans,...}
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.with_images = with_images
        if scene_list_file:
            self.scene_list = read_txt_to_list(scene_list_file)
        else:
            # process every scene that ships the required iPhone + scan assets
            self.scene_list = sorted(
                p.name for p in self.data_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )

        self.image_type = "iphone"  # only the iPhone branch is supported
        # blacklist classes that are background / structure rather than objects
        self.unexpected_classes = [
            "wall", "floor", "ceiling", "carpet", "doorframe",
            "split", "SPLIT", "remove", "REMOVE", "object",
        ]
        # an object must cover at least 0.01% of the image to be considered
        self.obj_pixel_thresh = 0.0001
        # at least 0.1% of the image pixels must belong to the object
        self.min_visible_pixel_ratio = 0.001
        # at least 5% of the object mesh vertices must be visible
        self.min_visible_vertex_ratio = 0.05

    def _required_assets(self, scene):
        return [
            scene.iphone_colmap_dir / "cameras.txt",
            scene.iphone_colmap_dir / "images.txt",
            scene.scan_mesh_path,
            scene.scan_mesh_segs_path,
            scene.scan_anno_json_path,
        ]

    def _has_required_assets(self, scene):
        missing = [str(p) for p in self._required_assets(scene) if not p.exists()]
        if missing:
            logger.warning("[%s] missing required assets, skipping: %s",
                           scene.scene_id, missing)
            return False
        return True

    def preprocess(self):
        logger.info("Found %d scene(s) in %s", len(self.scene_list), self.data_dir)

        for scene_id in tqdm(self.scene_list, desc="scene"):
            images_annotation = []
            scene = ScannetppScene_Release(scene_id, data_root=self.data_dir)
            if not self._has_required_assets(scene):
                continue
            logger.info("Processing scene: %s", scene_id)

            anno = load_anno_wrapper(scene)
            vtx_obj_ids = anno["vertex_obj_ids"]

            # move the mesh to the GPU once per scene
            mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))
            verts, faces, _ = prep_pt3d_inputs(mesh)
            mesh_torch = Meshes(
                verts=torch.Tensor(np.array([verts])),
                faces=torch.Tensor(np.array([faces])),
            ).to(self.device)

            obj_ids = np.unique(vtx_obj_ids)
            obj_ids = sorted(obj_ids[obj_ids != 0])  # 0 means "no object"
            obj_id_locations = {obj_id: anno["objects"][obj_id]["obb"]["centroid"] for obj_id in obj_ids}
            obj_id_size = {obj_id: anno["objects"][obj_id]["obb"]["axesLengths"] for obj_id in obj_ids}
            obj_id_rotation = {obj_id: anno["objects"][obj_id]["obb"]["normalizedAxes"] for obj_id in obj_ids}

            # sample the COLMAP-registered frames, sorted by id, stride sample_rate
            colmap_camera, image_list, poses, _ = get_camera_images_poses(
                scene, self.sample_rate, self.image_type
            )
            intrinsic = camera_to_intrinsic(colmap_camera)
            img_height, img_width = colmap_camera.height, colmap_camera.width
            logger.info("[%s] %d sampled frame(s) at %dx%d", scene_id,
                        len(image_list), img_width, img_height)

            for i, image_name in enumerate(tqdm(image_list, desc="image", leave=False)):
                img_path = scene.iphone_rgb_dir / image_name
                if not img_path.exists():
                    logger.warning("Image not found: %s, skipping", img_path)
                    continue
                try:
                    img = load_image(str(img_path))
                except Exception as e:
                    logger.warning("Error loading image: %s (%s), skipping", img_path, e)
                    continue

                # rasterize the mesh for this frame on the fly; poses are colmap
                # world-to-camera 4x4 matrices
                pose = torch.Tensor(np.array(poses[i:i + 1]))
                camera = get_opencv_cameras_batch(pose, img_height, img_width, intrinsic)
                raster_out_dict = rasterize_mesh(mesh_torch, img_height, img_width, camera)

                pix_to_face = raster_out_dict["pix_to_face"].squeeze().cpu().numpy()

                try:
                    pix_obj_ids = get_vtx_prop_on_2d(pix_to_face, vtx_obj_ids, mesh)
                except IndexError:  # something wrong with the rasterization
                    logger.warning("Rasterization error in %s/%s, skipping",
                                   scene_id, image_name)
                    continue

                bboxes_2d = get_bboxes_2d(pix_obj_ids)

                # vertices that are visible in this frame
                visible_faces = pix_to_face[pix_to_face != -1]
                visible_faces = np.unique(visible_faces)
                visible_verts = np.unique(faces[visible_faces].reshape(-1))

                objs_info = []
                for obj_id, obj_bbox in bboxes_2d.items():
                    if obj_id == 0:
                        continue

                    obj_label = anno["objects"][obj_id]["label"]
                    if obj_label in self.unexpected_classes:
                        continue

                    # only keep objects of a minimum 2D size
                    x, y, w, h = obj_bbox
                    if (w * h) / (img_height * img_width) < self.obj_pixel_thresh:
                        continue

                    # ---------- visible pixel ratio ----------
                    obj_pixel_mask = pix_obj_ids == obj_id
                    visible_pixel_ratio = np.sum(obj_pixel_mask) / (img_height * img_width)
                    if visible_pixel_ratio < self.min_visible_pixel_ratio:
                        continue

                    # ---------- visible vertex ratio ----------
                    obj_verts = np.where(vtx_obj_ids == obj_id)[0]
                    if len(obj_verts) == 0:
                        continue
                    visible_obj_verts = np.intersect1d(obj_verts, visible_verts)
                    visible_vertex_ratio = len(visible_obj_verts) / len(obj_verts)
                    if visible_vertex_ratio < self.min_visible_vertex_ratio:
                        continue

                    obj_location_3d = np.round(obj_id_locations[obj_id], 2).tolist()
                    obj_2Dbbox = np.round([x, y, x + w, y + h]).tolist()
                    objs_info.append({
                        "category": obj_label,
                        "3D_location": obj_location_3d,
                        "3D_size": obj_id_size[obj_id],
                        "3D_rotation": obj_id_rotation[obj_id],
                        "2D_bbox": obj_2Dbbox,
                    })

                images_annotation.append({
                    "scene_id": scene_id,
                    "image_name": image_name,
                    "image_path": os.path.join(scene_id, self.image_type, image_name),
                    "extrinsic": np.asarray(poses[i]).tolist(),
                    "intrinsic": intrinsic.tolist(),
                    "objects": objs_info,
                })

                if self.with_images:
                    # keep the sampled frames next to the metadata so the output
                    # directory is self-contained and matches image_path
                    dst = self.output_dir / scene_id / self.image_type / image_name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_path, dst)

            scene_out_dir = self.output_dir / scene_id
            scene_out_dir.mkdir(parents=True, exist_ok=True)
            save_json(images_annotation, scene_out_dir / "scene_metadata.json")
            logger.info("[%s] saved %d frame record(s) to %s",
                        scene_id, len(images_annotation),
                        scene_out_dir / "scene_metadata.json")


def main():
    parser = argparse.ArgumentParser(
        description="Project ScanNet++ 3D object annotations onto sampled iPhone frames")
    parser.add_argument("--data-root", default="./data",
                        help="directory that contains the per-scene release folders "
                             "(data/<scene_id>/{iphone,scans,...}), as used by ScannetppScene_Release")
    parser.add_argument("--output-root", default="./scannetpp_sampled_new",
                        help="output directory; per scene it receives scene_metadata.json "
                             "(and the sampled frames with --with-images)")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="keep every k-th COLMAP-registered frame")
    parser.add_argument("--device", default="cuda:0",
                        help="torch device for the mesh; the official rasterize helpers "
                             "are hardcoded to cuda:0, so keep the default")
    parser.add_argument("--scene-list", default=None,
                        help="optional file with one scene id per line; default: all scenes")
    parser.add_argument("--no-images", action="store_true", default=False,
                        help="do not copy the sampled frames next to the metadata")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    dataset = scannetpp_dataset(
        data_dir=args.data_root,
        output_dir=args.output_root,
        sample_rate=args.sample_rate,
        device=args.device,
        scene_list_file=args.scene_list,
        with_images=not args.no_images,
    )
    dataset.preprocess()


if __name__ == "__main__":
    main()
