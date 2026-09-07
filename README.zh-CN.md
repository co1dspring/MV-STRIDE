# MV-STRIDE

[English](README.md) | [简体中文](README.zh-CN.md)

数据构建与训练数据生产管线：从 **Infinigen**（合成 3D 场景）与 **ScanNet++**（真实重建场景）出发，
构建**多视角空间推理**的问答（QA）训练数据（分层 Level I/II/III + 跨视角多步推理组），供 MLLM
训练与评测使用。论文见文末 [Citation](#citation)。

> 本仓库只含**代码与配置**；QA 数据集（json 文本 + 图像）发布在 HuggingFace：**[数据链接（待发布）]**。

## 流水线一览

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

## 模块与文档

各模块独立成文（含职责、目录内容、运行方式），仓库结构与数据流见上：

| 模块 | 作用 | 文档 |
|---|---|---|
| `third_party/infinigen/` | 数据源 A：Infinigen 场景生成（gin/驱动脚本 + 复现指引，官方代码自行 clone） | [README](third_party/infinigen/README.md) |
| `third_party/scannetpp/` | 数据源 B：ScanNet++ 下载 / 抽帧 / 标注投影（自研改编脚本 + 指引） | [README](third_party/scannetpp/README.md) |
| `mvstride/scene_processing/` | 场景标注 → 统一元数据（Infinigen / ScanNet++ 两分支） | [README](mvstride/scene_processing/README.md) |
| `mvstride/qa_generation/` | **核心**：分层多视角 QA 生成（Stage 1） | [README](mvstride/qa_generation/README.md) |
| `configs/qa/` | QA 生成配置与模板 + 输出命名链（相对路径以仓库根为基准解析） | [README](configs/qa/README.md) |
| `mvstride/llm/` | LLM CoT 生成 / 校验 / MCA 变体（Stage 2） | [README](mvstride/llm/README.md) |
| `mvstride/evaluation/` | 测试集采样、指标计算、QC | [README](mvstride/evaluation/README.md) |
| `tools/` | 清洗 / 切分 / 统计等脚本工具 | [README](tools/README.md) |
| `misc/` | 可视化工具 + 旧版本代码归档 | [README](misc/README.md) |
| `data/` | 本地复现数据（QA 输入与输出；不入 GitHub，发布到 HuggingFace） | [README](data/README.md) |

## 快速开始（复现 QA 数据）

```bash
# 1) 依赖（Python 3.9+，QA 生成器用到 dict 合并运算符；scene_processing 3.8+ 即可）
pip install -r requirements.txt

# 2) 数据：把两个数据源的 QA-ready 输入放到 data/（布局见 data/README.md）
#    —— Infinigen：data/infinigen/saved_scenes/（save_scene_annotation.py 的产物）
#    —— ScanNet++：data/scannetpp/scannetpp_sampled_modified/（需先跑统一 schema 转换，命令见 data/README.md）

# 3) 生成 QA：从仓库根目录运行，--config 指定数据源配置（一次跑一个数据源）：
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py \
  --config ./configs/qa/qa_config_infinigen.json
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py \
  --config ./configs/qa/qa_config_scannetpp.json
#    输出：data/QA_jsons_<version_name>/ 下 atomic/ 全量 + *_stage1/2/3.json 阶段切分
#    （输出文件命名链与字段说明见 configs/qa/README.md）
```

可选的 Stage 2（LLM CoT 标注）与评测流程见 `mvstride/llm/README.md`、`mvstride/evaluation/README.md`。

## 数据格式

**Atomic 模式**（默认，`multilevel_qa_mode: atomic`）单条样本：

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

- `images` 为 `training_environment_base_dir` 解析后的**绝对路径**下的真实图像（Infinigen 按
  `<scene>/<Image 文件名>`、ScanNet++ 按元数据内 `<scene>_iphone/images/...` 拼接；发布到 HuggingFace
  时会重写为仓库内相对路径）。
- **Conversation 模式**：`messages` 为多轮 QA 组（Level I→III 逐问逐答），图像挂在首条 user 消息。
- **Stage 划分**：`*_stage1.json`（全层级，SFT）/ `*_stage2.json`（预留：Level III
  conversation 组，供 CoT cold-start；当前随仓库的 atomic 配置下不产出）/ `*_stage3.json`
  （Level III，RL）；Stage 2 的 CoT 产物为 `<think>...</think><answer>...</answer>` 格式。

## 路径约定

代码**不含私有绝对路径**（原开发于私有集群，已中性化）。运行路径统一按两个基准解析：

1. QA 配置中的相对路径**以仓库根为基准**解析（配置位于 `configs/qa/` 时自动推导，与运行目录无关；
   详见 [`configs/qa/README.md`](configs/qa/README.md)）；
2. 数据一律放仓库根目录 `data/`（详见 [`data/README.md`](data/README.md)）。

## 环境

| 环节 | 依赖 |
|---|---|
| 数据构建（场景处理 / QA 生成 / 工具） | Python 3.9+（QA 生成用 dict 合并运算符；场景处理 3.8+）：`numpy` `scipy` `Pillow` `tqdm` `opencv-python`（见 `requirements.txt`） |
| ScanNet++ 标注投影（third_party） | 另需 `torch` `pytorch3d` `open3d`，及官方 scannetpp 仓库（见其 README） |
| LLM 标注（Stage 2） | OpenAI 兼容接口；API key / 代理经**环境变量**传入（见 `mvstride/llm/api_interface.py`） |
| 评测 | vLLM 服务（`mvstride/evaluation/calculate_metrics_server.py`） |

<details>
<summary><b>已知问题与待办（展开）</b></summary>

运行/发布前注意：

1. **示例路径**：多数脚本入口已参数化（`--input-file` / `--output-dir` 等，`--help` 可查），但默认值仍是中性化占位符或 `./output/` 相对路径（以仓库根为基准）；运行前请显式传参。LLM 类脚本依赖 `API_USERNAME` / `API_PASSWORD` / `PROXY_URL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` 环境变量（见 `mvstride/llm/api_interface.py`），示例中的模型名（`gemini-3-flash-preview` / `gpt-5.5`）仅为历史运行记录，需替换为实际可用模型。
2. **可选依赖**：`mvstride/llm/` 与可视化脚本的第三方依赖（`openai`、`jsonlines`、`cv2`、`streamlit` 等）未列入 `requirements.txt`（见其注释说明），按需自装。
3. **历史遗留已归档**：旧扁平目录（`dataset_generation/`、`api/`、`scripts/` 等）已重构为 `mvstride/` 等包结构；不可运行的重构片段（`generate_QAs_multilevel_multistage2/3.py` 等）在 `misc/legacy/` 仅供查阅。
4. **已修复（此前版本记录）**：跨脚本数据链不一致（`human_eval_cot_visualization.py` 输入文件名与采样输出对齐、`../api/output/` 旧路径已全部迁移为 `./output/`、`calculate_metrics_local.py` 重复 `RESULT_PATH` 改为 CLI 参数）；ScanNet++ 转换缺字段崩溃已加防御；`choice_generation.py`/`paraphrasing.py` 解析失败 `None` 解引用已判空；`mvstride_cot_generation.py` 入口此前误调 `aggregate_all_stats()` 不生成数据，已改为 `process_data()`；QA 生成器入口已从硬编码 `CONFIG_PATH` 改为 `--config` 参数，配置内相对路径统一以仓库根为基准解析。
5. **运行风险**：`mvstride/llm/` 四个大脚本仍含约 150 行重复脚手架与死代码（视频分支等），待重构为公共模块（不影响运行）；`icecream`/`PyQt5` 为低价值依赖。

待办：HuggingFace 数据发布与打包脚本、训练配置（ms-swift）与基准评测脚本整理。数据构建链路（ScanNet++ 统一 schema 转换 → 两个主配置的 QA 生成）已在本机完整实跑验证，耗时与产物规模见 [`configs/qa/README.md`](configs/qa/README.md)。
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

- [Infinigen](https://github.com/mit-han-lab/infinigen) — 程序化 3D 场景生成（合成数据源）
- [ScanNet++](https://github.com/scannetpp/scannetpp)（数据：[scannetpp.cs.uni-bonn.de](https://scannetpp.cs.uni-bonn.de)）— 真实场景重建（真实数据源）
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — 基座视觉语言模型
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — 基座视觉语言模型
- [ms-swift](https://github.com/modelscope/ms-swift) — 训练 / 评测框架
