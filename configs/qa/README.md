# configs/qa

QA generation **configs & templates** (kept separate from source code). This directory is also the
**hub** of data reproduction: the two full configs (Infinigen / ScanNet++), each wiring the complete
naming chain from "scene metadata → hierarchical QA → stage-split training sets".

## Files

| File | Role |
|---|---|
| `qa_config_infinigen.json` / `qa_config_scannetpp.json` | Full per-data-source QA generation configs (**main configs**; the data released with the repo is produced by these) |
| `qa_config_{infinigen,scannetpp}_sparse.json` | Early small-scale configs |
| `qa_config_{infinigen,scannetpp}_ablation.json` | Cross-view dependency ablation configs (single-view vs. multi-view experiments) |
| `qa_templates.json` | Question/answer text templates for each category |
| `qa_dependency_tree.json` | Level III question → its prerequisite Level I/II sub-questions mapping |
| `README.md` | This document |

## Path Resolution Rules

After reading a config, the generator (`mvstride/qa_generation/generate_QAs_multilevel_multistage.py`)
resolves its **relative paths against the repository root** (independent of the current working
directory): when the config lives under `<repo_root>/configs/qa/`, relative paths are based on
`<repo_root>`; configs placed elsewhere are resolved relative to their own directory.

The configs committed to the repo are therefore written as (all relative to the repo root):

| Config field | Default value | Actual location |
|---|---|---|
| `source_data_dir` / `training_environment_base_dir` (Infinigen configs) | `data/infinigen/saved_scenes` | `<repo_root>/data/infinigen/saved_scenes/` |
| `source_data_dir` / `training_environment_base_dir` (ScanNet++ configs) | `data/scannetpp/scannetpp_sampled_modified` | `<repo_root>/data/scannetpp/scannetpp_sampled_modified/` |
| `output_dir` | `data/QA_jsons_{VERSION_NAME}` | `<repo_root>/data/QA_jsons_<version_name>/` |
| `qa_templates_path` / `qa_dependency_tree_path` | `configs/qa/qa_templates.json` etc. | this directory |

> Once the data is in place, **no code or config changes are needed** to run (see below). The
> `{VERSION_NAME}` placeholder in `output_dir` is replaced with the `version_name` field of the same
> config — it is the single source of the output directory name; change the version to change the
> output directory.

## Running (Stage 1 QA Generation)

Main entry point: `mvstride/qa_generation/generate_QAs_multilevel_multistage.py`, selecting the config
via `--config` (one data source per run; default `qa_config_infinigen.json`). Run from the repo root:

```bash
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py \
  --config ./configs/qa/qa_config_infinigen.json

python mvstride/qa_generation/generate_QAs_multilevel_multistage.py \
  --config ./configs/qa/qa_config_scannetpp.json
```

A single run executes: full-scene QA generation (`process_all_scenes()`) → stage split by scene pools
(`repartition_data_by_stage()`); i.e., all files in the "output naming chain" below are produced within
the same run.

### Key Config Fields at a Glance

| Field | Infinigen main config | ScanNet++ main config |
|---|---|---|
| `version_name` | `Infinigen_MultilevelCategories_20260403_sampled_MCA_Multistage` | `ScannetppIphone_MultilevelCategories_20260313_sampled_MCA_Multistage` |
| `metadata_filename` | `scene_metadata.json` | `scene_metadata_new.json` |
| `data_source` | `infinigen` | `scannetpp` |
| `multilevel_qa_mode` | `atomic` | `atomic` |
| `stage_1/2/3_proportion` | 0.6 / 0.0 / 0.4 | 0.6 / 0.2 / 0.2 |
| `global_seed` | 1840 | 1840 |
| `MIN_AREA_THRESHOLD` / `MIN_SIDE_THRESHOLD` | 500 / 20 (px) | 20000 / 100 (0–1000 relative coords) |

The remaining fields (per-QA-type `sampling_rate`, `obj_area_map`, `unwanted_cats`, etc.) are read
straight from the json. The generator only consumes metadata (**it does not read images**); the
`images` field in output json = `training_environment_base_dir` resolved to an **absolute path** +
relative path from the metadata (ScanNet++) or scene name + image basename (Infinigen).

## Output Naming Chain (Core Contract for Data Reproduction)

Let `V = version_name`, `OUT = data/QA_jsons_{V}/`. One `--config` run produces:

```
data/QA_jsons_<V>/
├── qa_config.json                  # copy of the config used in this run (self-documenting)
├── process_log.txt                 # run log
├── atomic/                         # full single-turn QAs when multilevel_qa_mode=atomic
│   ├── <V>_atomic.json             # ★ all levels merged (single-turn QAs from Level I+II+III)
│   ├── level_1/
│   │   ├── <QA_type>.json          # per-type Level I QA (e.g. 2D_location_perception_category.json)
│   │   └── <V>_atomic_level1.json  # Level I merged
│   ├── level_2/ ...                # same as above: per-type + <V>_atomic_level2.json
│   └── level_3/ ...                # same as above: per-type + <V>_atomic_level3.json (incl. MSR_*)
├── <V>_stage1.json                 # ★ SFT: all-level QAs from the stage_1 scene pool
├── <V>_stage1_level1.json          # Level I subset of stage1 (likewise _level2/_level3)
├── <V>_stage2.json                 # reserved: ColdStart pool + conversation groups (empty in current atomic mode)
└── <V>_stage3.json                 # ★ RL: Level III QAs from the stage_3 scene pool
```

Naming rule in one sentence: **directories = level/stage, file names = `{V}` + stage suffix or
`{QA_type}`**; `atomic/` is the full set, and the `<V>_stageN.json` files are the training-stage
splits re-partitioned by scene pool from `atomic/<V>_atomic.json`.

### Stage Split (`repartition_data_by_stage`)

Scenes are cut into three mutually exclusive pools in `sorted(scene_names)` order, with proportions
rounded to integers:

- **SFT pool**: first `N × stage_1_proportion` scenes → `<V>_stage1.json` (all levels);
- **ColdStart pool**: next `N × stage_2_proportion` scenes → reserved `<V>_stage2.json`
  (the conversation-assembly branch in code is commented out, so under `atomic` mode this file is
  empty and not written);
- **RL pool**: the remaining scenes → `<V>_stage3.json` (Level III only, for downstream RL /
  CoT cold-start processing).

> When the scene count is small, rounding can leave a pool empty (e.g. running a single scene with
> stage_1=0.6 → SFT pool has 0 scenes, everything falls into the RL pool). This is expected, not a bug.

### Input-Side Naming Chain (How the Two Data Sources Reach `source_data_dir`)

| Data source | Artifact on the chain | Description / producer |
|---|---|---|
| Infinigen | `<scene_id>/{scene_metadata.json, Image_<i>_0_0048_0.png}` | `mvstride/scene_processing/infinigen/save_scene_annotation.py`; 10 camera images per scene (`data/infinigen/saved_scenes/`, 439 scenes) |
| ScanNet++ | `<scene_id>_iphone/{scene_metadata_new.json, images/frame_*.jpg}` | official projection outputs unified to one schema by `mvstride/scene_processing/scannetpp/scannetpp2infinigen_new.py` (`data/scannetpp/scannetpp_sampled_modified/`, 137 scenes) |

For concrete data acquisition/conversion commands see [`data/README.md`](../../data/README.md); the
full three-step flows from raw scenes to metadata on both the Infinigen and ScanNet++ sides are in the
per-data-source READMEs under [`third_party/`](../../third_party/README.md).

### A Single Sample (atomic mode)

```json
{
  "messages": [
    {"role": "user", "content": "<image><image>Question ...\nOptions: A: ..., B: ..., C: ..., D: ..."},
    {"role": "assistant", "content": "B: ..."}
  ],
  "images": ["<absolute path>/data/infinigen/saved_scenes/4501660c/Image_0_0_0048_0.png", "..."],
  "category": "Positional Relationship(Obj.-Obj.)",
  "scene_name": "4501660c",
  "data_source": "infinigen"
}
```

`category` ∈ {Level I 7 types, Level II 5 types, Level III 12 types + 4 MSR types} (see the
`level1_qa_types` / `level2_qa_types` / `qa_types` + `msr_qa_types` keys of each config).

## Downstream (After Stage 2/3)

`<V>_stage1.json` goes straight into SFT; `<V>_stage3.json` is used for cold-start CoT and RL after
processing by `mvstride/llm/` (CoT generation/cleaning). LLM-side scripts default to writing
`./output/` and take explicit `--input-file/--output-dir` arguments; see
[`mvstride/llm/README.md`](../../mvstride/llm/README.md).

## Verified Runs on a Local Machine (2026-09-07, reproduction reference)

Both **main configs** were run end-to-end over the whole "ScanNet++ unified-schema conversion → QA
generation" chain; artifact sizes below (Python 3.10, single process, output dir =
`data/QA_jsons_{VERSION_NAME}/`):

| Item | Infinigen main config | ScanNet++ main config |
|---|---|---|
| Input scenes | 439 | 137 (`scannetpp_sampled_new/` also contains 1 `debug_viz/` debug directory, auto-skipped by the conversion script) |
| QA generation wall time | ~7 min | ~15 min |
| `atomic/<V>_atomic.json` (all levels, single-turn) | 253,400 items | 284,325 items |
| of which level_1 / level_2 / level_3 | 96,440 / 65,707 / 91,253 | 107,317 / 80,552 / 96,456 |
| `<V>_stage1.json` (SFT pool) | 147,382 items / 263 scenes | 167,889 items / 82 scenes |
| `<V>_stage2.json` | not produced (pool proportion 0) | not produced (conversation-assembly branch commented out; 27 scenes in pool) |
| `<V>_stage3.json` (RL pool) | 36,631 items / 176 scenes | 21,608 items / 28 scenes |

Validation results: stage1/stage3 scene pools do not overlap and match the proportional split of
`sorted(scene_names)`; a 40-item spot check of the files pointed to by `images` all exist; the
`atomic.json` count equals the sum of level_1 + level_2 + level_3 merges. Input-side conversion time:
ScanNet++ 137 scenes ≈ 2.2 min (incl. copying ~4.4 GB of frame images).
