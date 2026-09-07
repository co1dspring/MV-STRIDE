# MV-STRIDE

[English](README.md) | [简体中文](README.zh-CN.md)

Data construction and training-data production pipeline: starting from **Infinigen** (synthetic 3D scenes) and
**ScanNet++** (real scanned scenes), it builds **multi-view spatial reasoning** question-answering (QA)
training data (hierarchical Level I/II/III + cross-view multi-step reasoning groups) for MLLM training and
evaluation. Paper: see [Citation](#citation).

> This repository contains **code and configs only**; the QA datasets (json text + images) are released on
> HuggingFace: **[data link (TBD)]**.

## Pipeline Overview

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

## Modules & Documentation

Each module has its own document (responsibilities, directory contents, how to run); the repository
structure and data flow are shown above:

| Module | Role | Doc |
|---|---|---|
| `third_party/infinigen/` | Data source A: Infinigen scene generation (gin/driver scripts + reproduction guide; official code is cloned by yourself) | [README](third_party/infinigen/README.md) |
| `third_party/scannetpp/` | Data source B: ScanNet++ download / frame sampling / annotation projection (self-adapted scripts + guide) | [README](third_party/scannetpp/README.md) |
| `mvstride/scene_processing/` | Scene annotations → unified metadata (Infinigen / ScanNet++ branches) | [README](mvstride/scene_processing/README.md) |
| `mvstride/qa_generation/` | **Core**: hierarchical multi-view QA generation (Stage 1) | [README](mvstride/qa_generation/README.md) |
| `configs/qa/` | QA generation configs & templates + output naming chain (relative paths resolve from the repo root) | [README](configs/qa/README.md) |
| `mvstride/llm/` | LLM CoT generation / verification / MCA variants (Stage 2) | [README](mvstride/llm/README.md) |
| `mvstride/evaluation/` | Test-set sampling, metric computation, QC | [README](mvstride/evaluation/README.md) |
| `tools/` | Utility scripts: cleaning / splitting / statistics | [README](tools/README.md) |
| `misc/` | Visualization tools + legacy code archive | [README](misc/README.md) |
| `data/` | Local reproduction data (QA input & output; not committed to GitHub, released on HuggingFace) | [README](data/README.md) |

## Quick Start (Reproduce QA Data)

```bash
# 1) Dependencies (Python 3.9+ for the QA generator, which uses the dict-merge operator;
#    scene_processing works with 3.8+)
pip install -r requirements.txt

# 2) Data: put each data source's QA-ready input under data/ (layout: data/README.md)
#    —— Infinigen:  data/infinigen/saved_scenes/ (output of save_scene_annotation.py)
#    —— ScanNet++:  data/scannetpp/scannetpp_sampled_modified/ (needs the unified-schema
#                   conversion first; command in data/README.md)

# 3) Generate QA: run from the repository root; --config selects the data-source config
#    (one data source per run):
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py \
  --config ./configs/qa/qa_config_infinigen.json
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py \
  --config ./configs/qa/qa_config_scannetpp.json
#    Output: data/QA_jsons_<version_name>/ with the full atomic/ set + *_stage1/2/3.json
#    stage splits (output naming chain & field docs: configs/qa/README.md)
```

Optional Stage 2 (LLM CoT annotation) and the evaluation workflow: see `mvstride/llm/README.md`,
`mvstride/evaluation/README.md`.

## Data Formats

**Atomic mode** (default, `multilevel_qa_mode: atomic`) — a single sample:

```json
{
  "messages": [
    {"role": "user", "content": "<image><image>Question ...\nOptions: A: ..., B: ..., C: ..., D: ..."},
    {"role": "assistant", "content": "B: ..."}
  ],
  "images": ["/path/to/repo/data/infinigen/saved_scenes/4501660c/Image_0_0_0048_0.png", "..."],
  "category": "Positional Relationship(Obj.-Obj.)",
  "scene_name": "4501660c",
  "data_source": "infinigen"
}
```

- `images` are real images under the **absolute path** resolved from `training_environment_base_dir`
  (Infinigen joins `<scene>/<image file name>`; ScanNet++ joins `<scene>_iphone/images/...` from the
  metadata). When released on HuggingFace these will be rewritten to repo-relative paths.
- **Conversation mode**: `messages` is a multi-turn QA group (Level I→III, question by question), with
  images attached to the first user message.
- **Stage splits**: `*_stage1.json` (all levels, SFT) / `*_stage2.json` (reserved: Level III
  conversation groups for CoT cold-start; not produced under the shipped atomic configs) /
  `*_stage3.json` (Level III, RL). The Stage-2 CoT outputs use the
  `<think>...</think><answer>...</answer>` format.

## Path Conventions

The code contains **no private absolute paths** (originally developed on a private cluster; now
neutralized). Runtime paths resolve against two bases:

1. Relative paths inside the QA configs resolve **from the repository root** (auto-derived when the
   config lives in `configs/qa/`, independent of the working directory; see
   [`configs/qa/README.md`](configs/qa/README.md));
2. All data lives under `data/` at the repository root (see [`data/README.md`](data/README.md)).

## Environment

| Stage | Dependencies |
|---|---|
| Data construction (scene processing / QA generation / tools) | Python 3.9+ (dict-merge operator in QA generation; 3.8+ suffices for scene processing): `numpy` `scipy` `Pillow` `tqdm` `opencv-python` (see `requirements.txt`) |
| ScanNet++ annotation projection (third_party) | additionally `torch` `pytorch3d` `open3d`, plus the official scannetpp repo (see its README) |
| LLM annotation (Stage 2) | OpenAI-compatible API; API key / proxy passed via **environment variables** (see `mvstride/llm/api_interface.py`) |
| Evaluation | vLLM server (`mvstride/evaluation/calculate_metrics_server.py`) |

<details>
<summary><b>Known issues & TODOs (expand)</b></summary>

Notes before running / releasing:

1. **Placeholder paths**: most script entry points are parameterized (`--input-file` / `--output-dir`,
   see `--help`), but defaults are still neutralized placeholders or `./output/` relative paths
   (resolved from the repo root) — pass arguments explicitly before running. LLM scripts rely on the
   `API_USERNAME` / `API_PASSWORD` / `PROXY_URL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` environment
   variables (see `mvstride/llm/api_interface.py`); the model names in examples
   (`gemini-3-flash-preview` / `gpt-5.5`) are only historical run records — replace them with models
   that are actually available to you.
2. **Optional dependencies**: third-party dependencies of `mvstride/llm/` and the visualization
   scripts (`openai`, `jsonlines`, `cv2`, `streamlit`, etc.) are not listed in `requirements.txt`
   (see its comments) — install as needed.
3. **Legacy archived**: the old flat directories (`dataset_generation/`, `api/`, `scripts/`, etc.)
   were refactored into the `mvstride/` package structure; non-runnable refactor fragments
   (`generate_QAs_multilevel_multistage2/3.py`, etc.) are kept under `misc/legacy/` for reference only.
4. **Fixed (records from earlier versions)**: cross-script data-chain inconsistencies
   (`human_eval_cot_visualization.py` input file names aligned with sampler output; old
   `../api/output/` paths migrated to `./output/`; the duplicated `RESULT_PATH` in
   `calculate_metrics_local.py` replaced with a CLI argument); defensive handling for missing fields
   that crashed the ScanNet++ conversion; null checks for `None` dereferences on parse failures in
   `choice_generation.py` / `paraphrasing.py`; `mvstride_cot_generation.py` used to call
   `aggregate_all_stats()` at entry (produced no data) and now calls `process_data()`; the QA
   generator entry switched from a hard-coded `CONFIG_PATH` to a `--config` argument, and relative
   paths in configs uniformly resolve from the repo root.
5. **Maintenance risk**: the four large `mvstride/llm/` scripts still contain ~150 lines of duplicated
   scaffolding and dead code (video branches, etc.) awaiting refactor into a shared module (does not
   affect running); `icecream` / `PyQt5` are low-value dependencies.

TODOs: HuggingFace data release & packaging scripts, training configs (ms-swift), and benchmark
evaluation scripts. The data-construction chain (ScanNet++ unified-schema conversion → QA generation
with both main configs) has been fully run and verified on a local machine; timings and artifact sizes
are in [`configs/qa/README.md`](configs/qa/README.md).
</details>

## Citation

```bibtex
@inproceedings{xu2026mvstride,
  title     = {MV-STRIDE: Enabling MLLMs to Master Multi-View Spatial Reasoning via Hierarchical Capability Modeling},
  author    = {Xu, Jin and Huang, Xiaojian and Luo, Zhuodong and Zhang, Zhihong and Liu, Xin and Wei, Jiansheng and Wang, Xinzhi and Zhao, Jie and Chen, Xuejin},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```

## Acknowledgements

- [Infinigen](https://github.com/mit-han-lab/infinigen) — procedural 3D scene generation (synthetic data source)
- [ScanNet++](https://github.com/scannetpp/scannetpp) (data: [scannetpp.cs.uni-bonn.de](https://scannetpp.cs.uni-bonn.de)) — real-scene reconstruction (real data source)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — base vision-language model
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — base vision-language model
- [ms-swift](https://github.com/modelscope/ms-swift) — training / evaluation framework
