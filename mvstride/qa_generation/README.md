# qa_generation

**Core data-construction module**: generates hierarchical multi-view spatial-reasoning QA groups from
per-scene geometric annotation metadata.

## The Three Capability Levels

- **Level I — single-view spatial perception**: `2D_location_perception_category/_object`, `Cam_obj_yaw`, `Cam_space_pitch`, `Depth_perception`, `Measurement_comparison`, `3D_location_cam_obj`
- **Level II — cross-view scene understanding**: `Object_correspondence`, `Cam_rot_yaw/_pitch`, `Cam_trans_forward/_right`
- **Level III — multi-view contextual reasoning**: various `Positional Relationship`, `Motion(Cam.)_*`, `Attribute(Appr.)_*`, etc. + 4 MSR types (`MSR_Cam`, `MSR_Cam_Obj`, `MSR_Counting`, `MSR_Obj_Obj`)

## Design Principles

1. **Hierarchical capability modeling**: each Level III question is bundled with its prerequisite
   Level I/II sub-questions (see `configs/qa/qa_dependency_tree.json`).
2. **Multiple-choice answers (MCA)**: A/B/C/D options with verifiable geometric ground truth
   (`if_MCA` config).
3. **Cross-view dependency filtering**: camera pairing spreads Level III evidence across views,
   reducing single-image shortcuts.
4. Two output modes: `atomic` (default; groups are flattened into single-turn QAs, stored per category
   under `atomic/level_{1,2,3}/`) and `conversation` (the whole group as a multi-turn dialogue). All 6
   configs shipped with the repo are `atomic`; the `conversation` branch and the scene-pool stage1/2/3
   split live in `repartition_data_by_stage()` (`stage2` depends on conversation groups and is empty
   under the current configs).

## Contents

| Role | Files |
|---|---|
| Main generator | `generate_QAs_multilevel_multistage.py` (`*2.py` / `*3.py` are non-runnable refactor fragments, archived in `misc/legacy/`) |
| Level I/II generators | `multilevel_qa/` (`level1_qa.py`, `level2_qa.py`) |
| Geometry/QA utilities | `util/` (`math_utils.py`, `qa_utils.py`, `filter_utils.py`, `common_utils.py`) |
| Stage sampling | `sample_stage2.py`, `sample_stage3.py` |
| Generic json utilities | `merge_json.py`, `duplicate_json.py` (the per-category `tools/split_json.py` is the parameterized new version; the old hard-coded one is in `misc/legacy/split_json.py`) |

Configs live in `configs/qa/` (relative paths inside configs resolve from the repo root; the entry
point takes the config via `--config`; see [`configs/qa/README.md`](../../configs/qa/README.md)).
