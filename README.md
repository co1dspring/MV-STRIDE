# MV-STRIDE

**MV-STRIDE: Enabling MLLMs to Master Multi-View Spatial Reasoning via Hierarchical Capability Modeling** — ECCV 2026

Official code for the data construction and training-data production pipeline of MV-STRIDE. This repository builds **hierarchical, multi-view spatial reasoning training data** for multimodal large language models (MLLMs) from **Infinigen** (synthetic 3D scenes) and **ScanNet++** (real-world reconstructed scenes), and converts them into formats consumable by MLLM training frameworks.

> ⚠️ **Repository status: under migration.** This codebase was developed on a private training cluster (Windows workstations + a ModelArts NPU cluster) and is being migrated to a portable, open-source form. Original environment-specific paths have been neutralized to placeholders (e.g. `/path/to/data/...`); you will need to adapt them to your own setup before running — see [Known Issues](#known-issues).

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
        │  scene_metadata.json (cameras, objects, rooms, 2D/3D annotations)
        ▼
┌───────────────────────────────────────────────────────────────┐
│  dataset_generation/   (Stage 1: QA generation)               │
│  └─ generate_QAs_multilevel_multistage.py                     │
│     per camera pair → Level III QA + dependent L1/L2 sub-QAs  │
│     per scene → MSR multi-view QA (≥3 views)                  │
│     output: atomic/ or conversation/ data by category         │
│  └─ repartition_data_by_stage() → stage1/2/3 json by scene    │
└───────────────────────────────────────────────────────────────┘
        │  stage2.json (Level III QA groups, MCA)
        ▼
┌───────────────────────────────────────────────────────────────┐
│  api/   (Stage 2: CoT generation & verification)              │
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
│  scripts/   (format conversion & utilities)                   │
│  ├─ scannetpp2infinigen_new.py   ScanNet++ anno → unified     │
│  ├─ repair_bbox.py / remove_string.py / split_json.py /       │
│  │  expand_json.py / acc_sample.py / count_total_qa.py ...    │
│  └─ visualization.py / train_logging_visualization.py         │
└───────────────────────────────────────────────────────────────┘
        │  training data (jsonl) → MLLM SFT / Cold-start CoT / RL
        ▼
┌───────────────────────────────────────────────────────────────┐
│  data_test_sampling/ + data_qc_sampling/ (evaluation & QC)    │
│  ├─ cross_view_dependency_test_sampling.py  test set sampling │
│  ├─ human_eval_sampling.py / human_eval_cot_sampling.py       │
│  ├─ calculate_metrics_local.py / _server.py   accuracy stats  │
│  ├─ human_eval_visualization.py (Streamlit annotation tool)   │
│  └─ collect_qc_samples.py / qc_visualizer_app.py (QC)         │
└───────────────────────────────────────────────────────────────┘
```

### Progressive training strategy

| Stage | Data | Scripts involved |
|---|---|---|
| **Stage 1 · SFT** | Level I–III QA pairs, direct-answer supervision | `dataset_generation/` → `repartition_data_by_stage()` |
| **Stage 2 · Cold start** | CoT annotations synthesized from cross-level QA groups | `api/mvstride_cot_generation.py` → `cot_jsonl2swift.py` → `remove_system.py` |
| **Stage 3 · RL** | Difficulty-filtered multi-view reasoning problems (GRPO) | `scripts/acc_sample.py` (10-shot pass-rate based filtering) |

---

## Repository structure

```
MV-STRIDE/
├── dataset_generation/        # ★ Core: 3D-grounded hierarchical QA generation
│   ├── generate_QAs_multilevel_multistage.py      # Main generator (v1, full)
│   ├── generate_QAs_multilevel_multistage2.py     # Iteration (partial refactor of v1)
│   ├── generate_QAs_multilevel_multistage3.py     # Refactored version (current)
│   ├── qa_config_{infinigen,scannetpp}[_sparse|_ablation].json  # Per-datasource configs
│   ├── qa_templates.json       # Question/answer text templates (30 categories)
│   ├── qa_dependency_tree.json # Level III → prerequisite Level I/II mapping
│   ├── multilevel_qa/          # Level1Mixin / Level2Mixin QA generators
│   ├── util/                   # math_utils / qa_utils / filter_utils / common_utils
│   ├── save_scene_annotation.py   # Infinigen scene → scene_metadata.json
│   ├── extract_metadata.py        # (early version of the above)
│   ├── sample_stage2.py / sample_stage3.py   # Ratio-based nested sampling
│   └── merge_json.py / split_json.py / duplicate_json.py  # Generic json utils
├── api/                        # ★ LLM API layer: CoT generation, verification, MCA
│   ├── api_interface.py        # OpenAI-compatible API wrapper (vision + text)
│   ├── mvstride_cot_generation.py    # CoT generation (Level III main question)
│   ├── mvstride_cot_verification.py  # CoT quality verification vs. GT sub-answers
│   ├── mvstride_cvd_test.py          # Cross-view-dependency test inference
│   ├── choice_generation.py          # Open QA → MCQ (A/B/C/D)
│   ├── paraphrasing.py               # Question paraphrasing
│   ├── cot_jsonl2swift.py            # <think>/<answer> format repair & filtering
│   ├── remove_system.py              # Strip system instruction from user prompts
│   ├── match_input_output.py         # Align CoT ↔ original multi-round QA (GT injection)
│   ├── analyse_cot_verification_result.py  # Verification statistics report
│   ├── rebalance_options.py          # Option-letter balancing (avoid letter bias)
│   └── mca_jsonl2swift.py            # MCA metadata stripping → json
├── scripts/                    # ★ Conversion & utility scripts
│   ├── scannetpp2infinigen_new.py    # ScanNet++ annotation → unified schema
│   ├── acc_sample.py                 # 10-shot pass-rate based difficulty sampling
│   ├── expand_json.py / split_json.py / remove_string.py / repair_bbox.py
│   ├── count_total_qa.py / read_completions.py
│   └── visualization.py / train_logging_visualization.py
├── data_test_sampling/         # ★ Test set sampling & metric evaluation
│   ├── cross_view_dependency_test_sampling.py  # multi/single-view test sampling
│   ├── human_eval_sampling.py / human_eval_cot_sampling.py
│   ├── human_eval_visualization.py / human_eval_cot_visualization.py  # Streamlit
│   └── calculate_metrics_local.py / calculate_metrics_server.py
├── data_qc_sampling/           # ★ Quality control
│   ├── collect_qc_samples.py   # stratified QC sampling (100 samples)
│   └── qc_visualizer_app.py    # Streamlit QC annotation tool
└── index.html                  # Project page (ECCV 2026) draft
```

---

## Module details

### 1. `dataset_generation/` — 3D-grounded hierarchical QA generation (Stage 1)

**Input:** per-scene metadata JSON (`.json` per scene folder; Infinigen: `scene_metadata.json`; ScanNet++: `scene_metadata_new.json`), containing:
- `cameras`: per-camera intrinsics / extrinsics (`cam_extrinsics` W2C 4×4 matrix), `location_3d`, `forward_direction`, `image_path`, `image_size_HW`, and per-camera `objects` with `bbox_2d`
- `objects`: global object metadata — `category`, `3d_center`, `bbox_3d_aabb` (min/max/dimensions), `3D_rotation`, `3D_size`, optional `axis_directions`
- `rooms`: object instances whose category is in `room_type` (kitchen/bedroom/bathroom/dining-room/living-room)

**Flow** (`SceneQAGenerator` in `generate_QAs_multilevel_multistage.py`):
1. Load config (`qa_config_{source}.json`) → register QA-type generators (Level I/II from `multilevel_qa/`, Level III and MSR from the main class) and the dependency tree.
2. For each scene: filter unwanted categories (`floor`, `wall`, `ceiling`, ...), split rooms, enumerate camera pairs (capped at 190 for scenes with >50 cameras), filter pairs that are too overlapping (position < 0.2 m **and** orientation < 5°, ScanNet++ additionally requires ≥ `common_num_threshold` shared objects and uses stricter distance/angle bounds).
3. For each valid pair, run each Level III generator, which internally emits the Level III question **plus** its prerequisite Level I/II sub-questions via `qa_dependency_tree.json`. Questions with tiny objects (area/side thresholds), ambiguous descriptions, or degenerate geometry are dropped by `filter_utils`.
4. MSR (multi-view spatial reasoning) types run on 3–10 randomly sampled views, up to `max_num` per scene.
5. Outputs are saved **by category** under `output_dir/atomic/level_{1,2,3}/*.json` (atomic mode: each QA as an independent `<image>`-prefixed sample) or `conversation/` (multi-turn dialogue with a header prompt).
6. `repartition_data_by_stage()` splits the data **by scene** into `stage1` (SFT: all levels), `stage2` (cold start: Level III only, optional), `stage3` (RL: Level III only) according to `stage_*_proportion`.

**Key geometry utilities** (`util/math_utils.py`): camera rotation matrix extraction, relative yaw/pitch/translation between two cameras, 8-direction relative orientation, signed angle computation — all with verifiable GT answers.

**Config files:** `qa_config_infinigen.json` / `qa_config_scannetpp.json` (full), `*_sparse.json` (early small-scale), `*_ablation.json` (cross-view-dependency ablation, used for the 31.5% → 41.2% single-view vs multi-view experiment).

### 2. `api/` — LLM-based CoT generation & verification (Stage 2)

All scripts share a common skeleton (`api_interface.py` + per-script worker/thread-pool/checkpoint code) and produce `./output/<stem>_<model>_CoT.jsonl` style outputs.

- **`mvstride_cot_generation.py`** — reads Stage-2 QA groups (Level I→III questions in a conversation), sends the full multi-round context to an LLM (vision), and asks it to produce a step-by-step CoT for the Level III main question in `<think> ... <answer> ... </answer>` format. Output entries carry `messages`, `old_messages`, `images`, `category`, `scene_name`, `data_source`, `response_lst`.
- **`cot_jsonl2swift.py`** — repairs malformed `<think>/<answer>` tags, drops too-short CoTs and the `Positional Relationship(Obj.-Obj.)_Orientation` category, appends a system prompt, and writes a cleaned JSON.
- **`remove_system.py`** — removes the system prompt line for an ablation ("no system") variant.
- **`match_input_output.py`** — joins the cleaned CoT data back to the original multi-round QA by `(scene_name, normalized question)`; injects `original_intermediate_qa` and `original_ground_truth_answer` for verification. (*renamed from the misnamed `match_input_output,py`*)
- **`mvstride_cot_verification.py`** — a "lenient" reviewer LLM checks each CoT against the GT intermediate QAs and final answer, classifying into Correct / Factual inconsistency / Reasoning unfaithfulness / Final-answer inconsistency / Hallucination.
- **`analyse_cot_verification_result.py`** — aggregates verification labels into a console report (source of the reported 97.5% CoT pass rate).
- **`choice_generation.py` / `paraphrasing.py`** — convert open QA into A/B/C/D MCQs and paraphrase questions for diversity.
- **`rebalance_options.py` / `mca_jsonl2swift.py`** — shuffle option letters for uniform A/B/C/D distribution and strip redundant metadata.
- **`mvstride_cvd_test.py`** — pure multiple-choice inference on the cross-view-dependency test set (for evaluation, not training).

### 3. `scripts/` — conversion, cleaning & visualization

- **Annotation conversion:** `scannetpp2infinigen_new.py` converts raw ScanNet++ annotations (COLMAP W2C extrinsics, OBB) into the unified Infinigen-style schema (`scene_metadata_new.json`) with corrected camera coordinate conventions (Blender-style -Z forward, Y up).
- **Cleaning:** `repair_bbox.py` (pixel bbox → 0–1000 relative coordinates + path remapping), `remove_string.py` (strip `<image>` tags from the training text), `split_json.py` / `expand_json.py` (category split / ×10 replication for 10-shot inference).
- **Analysis:** `acc_sample.py` (difficulty-based sampling from 10-shot pass rates), `count_total_qa.py`, `read_completions.py` (RL looping-sample diagnosis).
- **Visualization:** `visualization.py` (PyQt5 QA browser), `train_logging_visualization.py` (GRPO training-log charts).

### 4. `data_test_sampling/` — evaluation data & metrics

- `cross_view_dependency_test_sampling.py` — samples 1000 multi-view test samples across Infinigen/ScanNet++ Level III, then derives a single-view variant (one random image, one `<image>` token) for the cross-view dependency ablation.
- `human_eval_sampling.py` / `human_eval_cot_sampling.py` — 200-sample human-eval sets (multi-view / CoT).
- `human_eval_visualization.py` / `human_eval_cot_visualization.py` — Streamlit annotation tools (5-class / 4-class error labeling, saved incrementally).
- `calculate_metrics_local.py` / `calculate_metrics_server.py` — per-category accuracy vs. Random(0.25)/Majority baselines; server version is CLI-parameterized and supports CoT-mode answer extraction (`<answer>` / `\boxed{X}` / letter regex).

### 5. `data_qc_sampling/` — quality control

- `collect_qc_samples.py` — stratified 100-sample QC set (MSR categories fixed at 2 each, others proportionally), with images physically copied to `qc_task_v1/images/`.
- `qc_visualizer_app.py` — Streamlit QC tool: view samples side-by-side with multi-view images, mark correct/incorrect, export `qc_final_results.json`.

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

1. **Hard-coded paths** — environment-specific paths have been neutralized to placeholders (e.g. `/path/to/data/...`, `/path/to/infinigen_main/...`), and relative paths (`./output/`, `./api/output/`) remain mixed with them. Ideally these should all become CLI args / config entries.
2. **Credentials** — API keys and proxy credentials were removed from the published code. Supply them via environment variables; see `api/api_interface.py`.
3. **Misnamed / damaged files**:
   - ~~`api/match_input_output,py`~~ — comma instead of dot in filename (cannot be imported/run); **fixed** → renamed to `match_input_output.py`
   - `dataset_generation/utils.py` — legacy single-file merged util module (renamed from `util..py`); unused by any code — all imports use the `util/` package. Removed from the published repository.
4. **Broken data links** — `data_test_sampling/human_eval_cot_visualization.py` expects `./human_eval/cot_sampled_200.json`, but `human_eval_cot_sampling.py` produces `cot_with_original_aligned_sampled_200.json`; `calculate_metrics_local.py` overwrites its `RESULT_PATH` constant (multiview path shadowed by singleview).
5. **Runtime risks** — `scannetpp2infinigen_new.py` can crash on missing `2D_bbox`/`extrinsic` fields; `mvstride_cot_generation.py` line ~367 uses an f-string with nested same-style quotes (SyntaxError on Python < 3.12); `choice_generation.py` / `paraphrasing.py` dereference `None` after failed API parsing.
6. **Undefined code paths** — the four large `api/` scripts carry ~150 lines of dead code (unused `process_video`, `get_output`, `main`) referencing undefined variables.
7. **Heavy code duplication** — the worker/thread-pool/checkpoint/stats scaffolding is copy-pasted across `api/` scripts; should be refactored into a shared module.
8. **Unused dependencies** — `icecream` imports throughout; `PyQt5` required by `scripts/visualization.py` (not a default dependency).

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
