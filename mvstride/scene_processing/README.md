# scene_processing

从生成（Infinigen）或拍摄/重建（ScanNet++）的场景与图像中，**提取有效标注**并**打包成 QA pipeline 输入格式**的模块。

## 职责

- **Infinigen**：在 Blender 中从场景提取相机 / 物体 / 房间 / 2D-3D 标注 → 产出 `scene_metadata.json`。
- **ScanNet++**：把官方标注转换为统一 schema（`scene_metadata_new.json`），并做 bbox 修复 / 路径重映射。

## 目录内容

| 模块 | 文件 | 说明 |
|---|---|---|
| `infinigen/` | `save_scene_annotation.py` | 在 Blender（`bpy`）中从 Infinigen 场景提取标注 → 汇总为 `scene_metadata.json`；早期版本见 `misc/legacy/extract_metadata.py` |
| `scannetpp/` | `scannetpp2infinigen_new.py`、`repair_bbox.py` | 把 ScanNet++ 投影标注（`scene_metadata.json`）转为统一 schema 并拷贝帧图；bbox 像素 → 0–1000 相对坐标 + 路径重映射 |

> ScanNet++ 上游的标注投影脚本（`scene_metadata.json` 的源头）位于
> `third_party/scannetpp/scripts/preprocess_iphone.py`（见 [`third_party/scannetpp/README.md`](../../third_party/scannetpp/README.md)）。
