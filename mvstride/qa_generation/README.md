# qa_generation

**核心数据构建模块**：从每场景的几何标注元数据，生成分层的多视角空间推理 QA 组。

## 三个能力层级

- **Level I — 单视角空间感知**：`2D_location_perception_category/_object`、`Cam_obj_yaw`、`Cam_space_pitch`、`Depth_perception`、`Measurement_comparison`、`3D_location_cam_obj`
- **Level II — 跨视角场景理解**：`Object_correspondence`、`Cam_rot_yaw/_pitch`、`Cam_trans_forward/_right`
- **Level III — 多视角上下文推理**：各种 `Positional Relationship`、`Motion(Cam.)_*`、`Attribute(Appr.)_*` 等 + 4 种 MSR 类型（`MSR_Cam`、`MSR_Cam_Obj`、`MSR_Counting`、`MSR_Obj_Obj`）

## 设计原则

1. **分层能力建模**：每条 Level III 问题与其前置的 Level I/II 子问题捆绑（见 `configs/qa/qa_dependency_tree.json`）。
2. **多选答案（MCA）**：A/B/C/D 选项与可校验的几何真值（`if_MCA` 配置）。
3. **跨视角依赖过滤**：相机配对使 Level III 的证据跨视角分布，减少单图捷径。
4. 输出分为 `atomic/` 与 `conversation/` 两种模式，再按场景划分 stage1/2/3。

## 目录内容

| 逻辑 | 文件 |
|---|---|
| 主生成器 | `generate_QAs_multilevel_multistage.py`（`*2.py` / `*3.py` 为不可运行的重构片段，已归档 `misc/legacy/`） |
| Level I/II 生成器 | `multilevel_qa/`（`level1_qa.py`、`level2_qa.py`） |
| 几何/QA 工具 | `util/`（`math_utils.py`、`qa_utils.py`、`filter_utils.py`、`common_utils.py`） |
| 阶段采样 | `sample_stage2.py`、`sample_stage3.py` |
| 通用 json 工具 | `merge_json.py`、`duplicate_json.py`（按类别切分的 `tools/split_json.py` 为参数化新版，旧硬编码版见 `misc/legacy/split_json.py`） |

配置位于 `configs/qa/`（相对路径以 `configs/qa/` 为基准解析，见 [`configs/qa/README.md`](../../configs/qa/README.md)）。
