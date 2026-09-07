# scene_processing

Module that **extracts valid annotations** from generated (Infinigen) or captured/reconstructed
(ScanNet++) scenes and images, and **packs them into the QA pipeline input format**.

## Responsibilities

- **Infinigen**: extract camera / object / room / 2D-3D annotations from scenes in Blender →
  produce `scene_metadata.json`.
- **ScanNet++**: convert the official annotations to the unified schema
  (`scene_metadata_new.json`), with bbox repair / path remapping.

## Contents

| Module | Files | Description |
|---|---|---|
| `infinigen/` | `save_scene_annotation.py` | Extract annotations from Infinigen scenes in Blender (`bpy`) → aggregate into `scene_metadata.json`; early version archived at `misc/legacy/extract_metadata.py` |
| `scannetpp/` | `scannetpp2infinigen_new.py`, `repair_bbox.py` | Convert projected ScanNet++ annotations (`scene_metadata.json`) to the unified schema and copy frame images; bbox pixels → 0–1000 relative coordinates + path remapping |

> The upstream ScanNet++ annotation-projection script (source of `scene_metadata.json`) lives at
> `third_party/scannetpp/scripts/preprocess_iphone.py` (see
> [`third_party/scannetpp/README.md`](../../third_party/scannetpp/README.md)).
