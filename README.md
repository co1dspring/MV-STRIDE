# MV-STRIDE

**MV-STRIDE: Enabling MLLMs to Master Multi-View Spatial Reasoning via Hierarchical Capability Modeling** — ECCV 2026

Official code for the data construction and training-data production pipeline of MV-STRIDE. This repository builds **hierarchical, multi-view spatial reasoning training data** for multimodal large language models (MLLMs) from **Infinigen** (synthetic 3D scenes) and **ScanNet++** (real-world reconstructed scenes), and converts them into formats consumable by MLLM training frameworks.

> **Note.** This codebase was developed on a private training cluster (Windows workstations + a ModelArts NPU cluster). Original environment-specific paths have been neutralized to placeholders (e.g. `/path/to/data/...`); adapt them to your own setup before running — see [Known Issues](#known-issues).

---

## Overview

MV-STRIDE organizes multi-view spatial reasoning into **three progressive capability levels** with explicit cross-level dependencies, instead of a flat collection of tasks:

| Level | Capability | Question types (categories) |
|---|---|---|
| **Level I** — Single-View Spatial Perception | Egocentric perception of objects and cameras in a single view | `2D_location_perception_category`, `2D_location_perception_object`, `Cam_obj_yaw`, `Cam_space_pitch`, `Depth_perception`, `Measurement_comparison`, `3D_location_cam_obj` |
| **Level II** — Cross-View Scene Understanding | Correspondence and relative geometry across two views | `Object_correspondence`, `Cam_rot_yaw`, `Cam_rot_pitch`, `Cam_trans_forward`, `Cam_trans_right` |
| **Level III** — Multi-View Contextual Reasoning | High-level reasoning over multiple views | `Attribute(Appr.)_Counting`, `Attribute(Appr.)_Orientation`, `Attribute(Meas.)`, `Motion(Cam.)_Translation`, `Motion(Cam.)_Rotation`, `Positional Relationship(Cam.-Cam.)_Translation`, `Positional Relationship(Cam.-Cam.)_Rotation`, `Positional Relationship(Cam.-Obj.)`, `Positional Relationship(Obj.-Obj.)`, `Positional Relationship(Obj.-Obj.)_Orientation`, `Positional Relationship(Obj.-Obj.)_Obj_in_options`, `Positional Relationship(Obj.-Reg.)` + 4 MSR multi-view types (`MSR_Cam`, `MSR_Cam_Obj`, `MSR_Counting`, `MSR_Obj_Obj`) |

Key design principles implemented in this codebase:

1. **Hierarchical capability modeling** — each Level III question is generated together with its *prerequisite* Level I/II sub-questions (see `qa_dependency_tree.json`), forming a **multi-level QA group** (a "spatial cognition process question group").
2. **Multi-choice answers (MCA)** — answers are generated as A/B/C/D options with verifiable geometric ground truth (`if_MCA` config flag).
3. **Cross-view dependency filtering** — camera pairs are filtered so that evidence for Level III questions is distributed across views (minimal overlap / sufficient baseline separation), reducing single-image shortcuts.
4. **Cognition-guided CoT** — the multi-level QA groups are fed to an LLM to synthesize long chain-of-thought supervision for Level III questions, then verified against ground-truth sub-answers.

### Data pipeline at a glance

```
Infinigen / ScanNet++ scenes
        │  (scene generation / capture: third_party/infinigen, third_party/scannetpp)
        ▼
┌───────────────────────────────────────────────────────────────┐
│  mvstride/scene_processing/   (annotation extraction)         │
│  ├─ infinigen/save_scene_annotation.py                        │
│  │    Infinigen GT → scene_metadata.json                      │
│  └─ scannetpp/scannetpp2infinigen_new.py                      │
│       ScanNet++ 3D annotations → unified schema               │
│       (frame projection: third_party/scannetpp/scripts/       │
│        preprocess_iphone.py, adapted from official)           │
└───────────────────────────────────────────────────────────────┘
        │  scene_metadata.json (cameras, objects, rooms, 2D/3D annotations)
        ▼
┌───────────────────────────────────────────────────────────────┐
│  mvstride/qa_generation/   (Stage 1: QA generation)           │
│  └─ generate_QAs_multilevel_multistage.py                     │
│     per camera pair → Level III QA + dependent L1/L2 sub-QAs  │
│     per scene → MSR multi-view QA (≥3 views)                  │
│     output: atomic/ or conversation/ data by category         │
│  └─ repartition_data_by_stage() → stage1/2/3 json by scene    │
└───────────────────────────────────────────────────────────────┘
        │  stage2.json (Level III QA groups, MCA)
        ▼
┌───────────────────────────────────────────────────────────────┐
│  mvstride/llm/   (Stage 2: CoT generation & verification)     │
│  ├─ choice_generation.py      QA → MCQ (optional)             │
│  ├─ paraphrasing.py           paraphrase questions            │
│  ├─ mvstride_cot_generation.py    LLM CoT generation          │
│  ├─ cot_jsonl2swift.py            <think>/<answer> repair     │
│  ├─ remove_system.py              strip system prompts        │
│  ├─ match_input_output.py         align CoT with original GT  │
│  ├─ mvstride_cot_verification.py  LLM CoT quality check       │
│  ├─ analyse_cot_verification_result.py  report                │
│  ├─ rebalance_options.py       option-letter balancing (MCA)  │
│  └─ mca_jsonl2swift.py         strip metadata → json          │
└───────────────────────────────────────────────────────────────┘
        │  cleaned CoT / MCA json
        ▼
┌───────────────────────────────────────────────────────────────┐
│  tools/ + misc/   (format conversion & utilities)             │
│  ├─ tools/split_json.py / expand_json.py / remove_string.py   │
│  │  (category split, ×10 replication, <image>-tag stripping)  │
│  ├─ tools/acc_sample.py / count_total_qa.py / read_completions│
│  └─ misc/visualization/  viewers + GRPO training plots        │
└───────────────────────────────────────────────────────────────┘
        │  training data (jsonl) → MLLM SFT / Cold-start CoT / RL
        ▼
┌───────────────────────────────────────────────────────────────┐
│  mvstride/evaluation/   (evaluation & QC)                     │
│  ├─ cross_view_dependency_test_sampling.py  test set sampling │
│  ├─ human_eval_sampling.py / human_eval_cot_sampling.py       │
│  ├─ calculate_metrics_local.py / _server.py   accuracy stats  │
│  └─ collect_qc_samples.py (QC)                                │
│  (interactive viewers → misc/visualization/)                  │
└───────────────────────────────────────────────────────────────┘
```

### Progressive training strategy

| Stage | Data | Scripts involved |
|---|---|---|
| **Stage 1 · SFT** | Level I–III QA pairs, direct-answer supervision | `mvstride/qa_generation/` → `repartition_data_by_stage()` |
| **Stage 2 · Cold start** | CoT annotations synthesized from cross-level QA groups | `mvstride/llm/mvstride_cot_generation.py` → `cot_jsonl2swift.py` → `remove_system.py` |
| **Stage 3 · RL** | Difficulty-filtered multi-view reasoning problems (GRPO) | `tools/acc_sample.py` (10-shot pass-rate based filtering) |

---

## Repository structure

```
MV-STRIDE/
├── mvstride/                    # ★ Main source package (data construction)
│   ├── scene_processing/        # Extract annotations from scenes into unified
│   │                            #   metadata (infinigen/ + scannetpp/ subdirs)
│   ├── qa_generation/           # ★ Core: 3D-grounded hierarchical QA generation
│   │                            #   (generator + multilevel_qa/ + util/)
│   ├── llm/                     # LLM annotation generation: CoT, verification, MCA
│   ├── evaluation/              # Test-set sampling, metrics, QC
│   └── __init__.py
├── configs/                     # Run configuration, separated from code
│   └── qa/                      # qa_config_{infinigen,scannetpp}[_sparse|_ablation].json
│                                #   + qa_templates.json + qa_dependency_tree.json
├── third_party/                 # Part A: upstream scene generation / real scenes
│   ├── infinigen/               # Infinigen (synthetic scenes; our gin/scripts + guide)
│   └── scannetpp/               # ScanNet++ (real scenes; adapted projection script + guide)
├── tools/                       # Conversion / sampling / statistics helper scripts
├── misc/                        # Misc / auxiliary code archive
│   ├── visualization/           # PyQt5 / Streamlit browsers & plotting
│   ├── legacy/                  # Outdated / early-version code (reference only)
│   └── ...
├── assets/                      # Project assets
├── index.html                   # Project page (ECCV 2026) draft
└── README.md
```

> The repository was restructured from a flat layout (`dataset_generation/`, `api/`, `scripts/`,
> `data_test_sampling/`, `data_qc_sampling/`) into the package layout above. Pre-migration file
> versions and obsolete early scripts are archived under `misc/legacy/` for reference; the git
> history still contains the original flat layout.


---

## Module details

### 1. `mvstride/qa_generation/` — 3D-grounded hierarchical QA generation (Stage 1)

**Input:** per-scene metadata JSON (`.json` per scene folder; Infinigen: `scene_metadata.json`; ScanNet++: `scene_metadata_new.json`), containing:
- `cameras`: per-camera intrinsics / extrinsics (`cam_extrinsics` W2C 4×4 matrix), `location_3d`, `forward_direction`, `image_path`, `image_size_HW`, and per-camera `objects` with `bbox_2d`
- `objects`: global object metadata — `category`, `3d_center`, `bbox_3d_aabb` (min/max/dimensions), `3D_rotation`, `3D_size`, optional `axis_directions`
- `rooms`: object instances whose category is in `room_type` (kitchen/bedroom/bathroom/dining-room/living-room)

**Flow** (`SceneQAGenerator` in `mvstride/qa_generation/generate_QAs_multilevel_multistage.py`):
1. Load config (`qa_config_{source}.json`) → register QA-type generators (Level I/II from `multilevel_qa/`, Level III and MSR from the main class) and the dependency tree.
2. For each scene: filter unwanted categories (`floor`, `wall`, `ceiling`, ...), split rooms, enumerate camera pairs (capped at 190 for scenes with >50 cameras), filter pairs that are too overlapping (position < 0.2 m **and** orientation < 5°, ScanNet++ additionally requires ≥ `common_num_threshold` shared objects and uses stricter distance/angle bounds).
3. For each valid pair, run each Level III generator, which internally emits the Level III question **plus** its prerequisite Level I/II sub-questions via `qa_dependency_tree.json`. Questions with tiny objects (area/side thresholds), ambiguous descriptions, or degenerate geometry are dropped by `filter_utils`.
4. MSR (multi-view spatial reasoning) types run on 3–10 randomly sampled views, up to `max_num` per scene.
5. Outputs are saved **by category** under `output_dir/atomic/level_{1,2,3}/*.json` (atomic mode: each QA as an independent `<image>`-prefixed sample) or `conversation/` (multi-turn dialogue with a header prompt).
6. `repartition_data_by_stage()` splits the data **by scene** into `stage1` (SFT: all levels), `stage2` (cold start: Level III only, optional), `stage3` (RL: Level III only) according to `stage_*_proportion`.

**Key geometry utilities** (`util/math_utils.py`): camera rotation matrix extraction, relative yaw/pitch/translation between two cameras, 8-direction relative orientation, signed angle computation — all with verifiable GT answers.

**Config files:** `qa_config_infinigen.json` / `qa_config_scannetpp.json` (full), `*_sparse.json` (early small-scale), `*_ablation.json` (cross-view-dependency ablation, used for the 31.5% → 41.2% single-view vs multi-view experiment).

### 2. `mvstride/llm/` — LLM-based CoT generation & verification (Stage 2)

All scripts share a common skeleton (`mvstride/llm/api_interface.py` + per-script worker/thread-pool/checkpoint code) and produce `./output/<stem>_<model>_CoT.jsonl` style outputs.

- **`mvstride_cot_generation.py`** — reads Stage-2 QA groups (Level I→III questions in a conversation), sends the full multi-round context to an LLM (vision), and asks it to produce a step-by-step CoT for the Level III main question in `<think> ... <answer> ... </answer>` format. Output entries carry `messages`, `old_messages`, `images`, `category`, `scene_name`, `data_source`, `response_lst`.
- **`cot_jsonl2swift.py`** — repairs malformed `<think>/<answer>` tags, drops too-short CoTs and the `Positional Relationship(Obj.-Obj.)_Orientation` category, appends a system prompt, and writes a cleaned JSON.
- **`remove_system.py`** — removes the system prompt line for an ablation ("no system") variant.
- **`match_input_output.py`** — joins the cleaned CoT data back to the original multi-round QA by `(scene_name, normalized question)`; injects `original_intermediate_qa` and `original_ground_truth_answer` for verification. (*renamed from the misnamed `match_input_output,py`*)
- **`mvstride_cot_verification.py`** — a "lenient" reviewer LLM checks each CoT against the GT intermediate QAs and final answer, classifying into Correct / Factual inconsistency / Reasoning unfaithfulness / Final-answer inconsistency / Hallucination.
- **`analyse_cot_verification_result.py`** — aggregates verification labels into a console report (source of the reported 97.5% CoT pass rate).
- **`choice_generation.py` / `paraphrasing.py`** — convert open QA into A/B/C/D MCQs and paraphrase questions for diversity.
- **`rebalance_options.py` / `mca_jsonl2swift.py`** — shuffle option letters for uniform A/B/C/D distribution and strip redundant metadata.
- **`mvstride_cvd_test.py`** — pure multiple-choice inference on the cross-view-dependency test set (for evaluation, not training).

### 3. `mvstride/scene_processing/` — annotation conversion into unified metadata

- **Infinigen branch** (`infinigen/save_scene_annotation.py`): reads the Infinigen-rendered GT (`K` / `T` / `HW`, segmentation, object metadata) and packs per-scene `scene_metadata.json` for the QA generator; batch driver examples live in `third_party/infinigen/scripts/`.
- **ScanNet++ branch** (`scannetpp/`):
  - `scannetpp2infinigen_new.py` converts raw ScanNet++ annotations (COLMAP W2C extrinsics, OBB) into the unified Infinigen-style schema (`scene_metadata_new.json`) with corrected camera coordinate conventions (Blender-style -Z forward, Y up). It is CLI-parameterized (`--raw-root/--anno-root/--output-root`; image lookup tolerates the legacy `.jpg.jpg` frames) and copies the sampled frames into each `<scene>_iphone/images/`, so its output directory is directly consumable by the QA generator.
  - `repair_bbox.py` — pixel bbox → 0–1000 relative coordinates + path remapping.
  - The upstream projection step that produces its input (`scene_metadata.json`) lives in `third_party/scannetpp/scripts/preprocess_iphone.py` (author-adapted from the official `semantics_2d.py`; see `third_party/scannetpp/README.md`).

### 4. `tools/` + `misc/visualization/` — cleaning, analysis & visualization

- **Cleaning (`tools/`):** `remove_string.py` (strip `<image>` tags from the training text), `split_json.py` / `expand_json.py` (category split / ×10 replication for 10-shot inference).
- **Analysis (`tools/`):** `acc_sample.py` (difficulty-based sampling from 10-shot pass rates), `count_total_qa.py`, `read_completions.py` (RL looping-sample diagnosis).
- **Visualization (`misc/visualization/`):** `visualization.py` (PyQt5 QA browser), `train_logging_visualization.py` (GRPO training-log charts), plus the Streamlit annotation/QC apps referenced in Section 5.

### 5. `mvstride/evaluation/` — evaluation data, metrics & QC

- `cross_view_dependency_test_sampling.py` — samples 1000 multi-view test samples across Infinigen/ScanNet++ Level III, then derives a single-view variant (one random image, one `<image>` token) for the cross-view dependency ablation.
- `human_eval_sampling.py` / `human_eval_cot_sampling.py` — 200-sample human-eval sets (multi-view / CoT); annotated with the Streamlit tools `human_eval_visualization.py` / `human_eval_cot_visualization.py` in `misc/visualization/`.
- `calculate_metrics_local.py` / `calculate_metrics_server.py` — per-category accuracy vs. Random(0.25)/Majority baselines; server version is CLI-parameterized and supports CoT-mode answer extraction (`<answer>` / `\boxed{X}` / letter regex).
- `collect_qc_samples.py` — stratified 100-sample QC set (MSR categories fixed at 2 each, others proportionally), with images physically copied to `qc_task_v1/images/`; labeled with `qc_visualizer_app.py` (Streamlit, `misc/visualization/`), exporting `qc_final_results.json`.

---

## Data format

**Atomic mode** (default, `multilevel_qa_mode: atomic`):
```json
{
  "messages": [
    {"role": "user", "content": "<image><image>Question ...\nOptions: A: ..., B: ..., C: ..., D: ..."},
    {"role": "assistant", "content": "B: ..."}
  ],
  "images": ["path/to/image1.png", "path/to/image2.png"],
  "category": "Positional Relationship(Obj.-Obj.)",
  "scene_name": "scene_0001",
  "data_source": "infinigen"
}
```

**Conversation mode** — `messages` is a multi-turn QA group (`Question 1 (Level 1: ...) ... Answer 1: ...`), images are attached to the first user message.

**Stage split** — `*_stage1.json` (all levels, SFT), `*_stage2.json` (Level III, cold start), `*_stage3.json` (Level III, RL). Stage 2 additionally gets CoT: `messages[0]` contains the question + system prompt, `messages[1]` contains `<think>...</think><answer>...</answer>`.

---

## Environment

The code was developed and run in the following environments. Paths are environment-specific and have been neutralized to placeholders (`/path/to/data/...`) — adapt them to your own setup (see Known Issues):

| Component | Environment |
|---|---|
| Dataset generation | Python 3.8+ (`numpy`, `scipy`, `Pillow`, `tqdm`, `icecream`) |
| LLM API calls | OpenAI-compatible endpoints (API keys / proxy credentials must be supplied via environment variables) |
| Training | ModelArts NPU cluster, Qwen3-VL-8B / MiMo-VL, ms-swift |
| Evaluation | Server with vLLM inference (`calculate_metrics_server.py`) |

To run the pipeline you will need: an **Infinigen** installation producing synthetic scenes, a **ScanNet++** raw-data directory, and a working directory laid out as the various `/path/to/data/...` placeholders expect (scene metadata, `infinigen_metadata_ver2`, `scannetpp_sampled_modified`, `QA_jsons_*`, etc.).

---

## Known issues (pre-release checklist)

> These are the items to be aware of before running or releasing.

1. **Hard-coded paths** — environment-specific paths have been neutralized to placeholders (e.g. `/path/to/data/...`, `/path/to/infinigen_main/...`), and relative paths (e.g. `./output/`) remain mixed with them. Ideally these should all become CLI args / config entries (configs in `configs/qa/` are already resolved relative to that directory).
2. **Credentials** — API keys and proxy credentials were removed from the published code. Supply them via environment variables; see `mvstride/llm/api_interface.py`.
3. **Consolidated / damaged files**:
   - ~~`mvstride/llm/match_input_output,py`~~ — comma instead of dot in filename (cannot be imported/run); **fixed** → renamed to `match_input_output.py`
   - During restructuring the old flat layout (`dataset_generation/`, `api/`, `scripts/`, `data_test_sampling/`, `data_qc_sampling/`) was consolidated; obsolete early-version files (e.g. the half-refactored `generate_QAs_multilevel_multistage2/3.py`, the single-file `utils.py`, a hard-coded `split_json.py`) are archived under `misc/legacy/` — some are non-runnable historical fragments kept for reference only.
4. **Broken data links** — `misc/visualization/human_eval_cot_visualization.py` expects `./human_eval/cot_sampled_200.json`, but `mvstride/evaluation/human_eval_cot_sampling.py` produces `cot_with_original_aligned_sampled_200.json`; `mvstride/evaluation/calculate_metrics_local.py` overwrites its `RESULT_PATH` constant (multiview path shadowed by singleview).
5. **Runtime risks** — `mvstride/scene_processing/scannetpp/scannetpp2infinigen_new.py` can crash on missing `2D_bbox`/`extrinsic` fields; `mvstride/llm/mvstride_cot_generation.py` line ~367 uses an f-string with nested same-style quotes (SyntaxError on Python < 3.12); `mvstride/llm/choice_generation.py` / `paraphrasing.py` dereference `None` after failed API parsing.
6. **Undefined code paths** — the four large `mvstride/llm/` scripts carry ~150 lines of dead code (unused `process_video`, `get_output`, `main`) referencing undefined variables.
7. **Heavy code duplication** — the worker/thread-pool/checkpoint/stats scaffolding is copy-pasted across `mvstride/llm/` scripts; should be refactored into a shared module.
8. **Unused dependencies** — `icecream` imports throughout; `PyQt5` required by `misc/visualization/visualization.py` (not a default dependency).

---

## Roadmap to open source

- [x] Remove credentials from published code (keys/proxy that were exposed are now stripped)
- [x] Neutralize private, environment-specific paths to placeholders
- [x] Fix misnamed files (`match_input_output,py` → `match_input_output.py`)
- [ ] Parameterize all remaining hard-coded paths (config/CLI)
- [ ] Fix broken data links
- [ ] Refactor shared API scaffolding into a common module
- [ ] Add dependency requirements (`requirements.txt`) and setup docs
- [ ] Release sample data + generated QA examples
- [ ] Reproduce pipeline end-to-end (Infinigen → metadata → QA → CoT → format)
- [ ] Integrate training configs for Stage 1/2/3 (ms-swift)
- [ ] Benchmark evaluation scripts for MMSI-Bench / CV-Bench / ViewSpatial-Bench / 3DSRBench

---

## Citation

```bibtex
@inproceedings{xu2026mvstride,
  title     = {MV-STRIDE: Enabling MLLMs to Master Multi-View Spatial Reasoning via Hierarchical Capability Modeling},
  author    = {Xu, Jin and Huang, Xiaojian and Luo, Zhuodong and Zhang, Zhihong and Liu, Xin and Wei, Jiansheng and Wang, Xinzhi and Zhao, Jie and Chen, Xuejin},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```

---

## Acknowledgements

- [Infinigen](https://github.com/mit-han-lab/infinigen) — procedural 3D scene generation
- [ScanNet++](https://github.com/scannetpp/scannetpp) — real-world scene reconstruction
- Qwen3-VL / MiMo-VL base models, ms-swift
