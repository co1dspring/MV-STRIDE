# third_party/infinigen

本目录存放 MV-STRIDE 数据构建**第一步（Part A）** 的原始数据构造代码：使用
[Infinigen](https://github.com/mit-han-lab/infinigen) 生成合成 3D 室内场景并渲染出带标注的图像。

> **产出**：Infinigen 场景数据（`.blend` 场景文件 + 渲染图像 + 相机姿态/实例分割/物体元信息等 GT），
> 供 [`mvstride/scene_processing/`](../../mvstride/scene_processing/README.md)
> 提取标注、打包成后续 QA 生成 pipeline 的输入。

Infinigen 是 MIT 的开源授权（BSD-3-Clause）库。**上游官方仓库本体并不直接包含在本仓库内**——
它通常太大、且随版本频繁变动。本目录只保留我们**围绕官方代码自行编写/修改**的部分，并说明如何把它们应用回
你自己 clone 的官方仓库。请按本文档的 **「怎么应用」** 一节把官方代码拉下来、把修改放进去。

---

## 一、整体数据流水线（三步）定位

MV-STRIDE 的完整数据构建分三步：

| 步骤 | 内容 | 代码位置 |
|---|---|---|
| **Step 1** | 生成场景：用 Infinigen 生成合成室内场景并渲染出 RGB + GT（相机姿态、分割、物体元信息） | 本目录 `third_party/infinigen/` |
| **Step 2** | 提取标注：从场景/渲染图像提取有效标注、打包成元数据（`scene_metadata.json`） | `mvstride/scene_processing/`（Part B，作者自写的两个脚本之一） |
| **Step 3** | 合成 QA：用已有的 QA 生成器基于元数据合成问答对 | `mvstride/qa_generation/` |

本 README 只讲 **Step 1**。Step 2 的 `save_scene_annotation.py` 见
[`mvstride/scene_processing/`](../../mvstride/scene_processing/README.md)。

---

## 二、两条可复现路径（重要）

Infinigen 是活跃演进的库，作者当年基于特定 commit 做了一些小修改。为了让使用者既能**开箱即用**、又能
**精确复现作者的产出**，本项目提供两条路径：

### Path A —— 直接使用官方代码（推荐，无需任何修改）

MV-STRIDE 的 Step 2（`read_single_scene_gt`）只读取 Infinigen **标准输出**（`K` 内参、`T` 外参、`HW`，
见 [`camview/*.npz`](#四输出-gt-格式)）。也就是说：**Step 2 / Step 3 并不依赖作者对 Infinigen 的任何魔改**。
因此绝大多数用户直接 clone 官方仓库、套用我们这里的 3 个自定义 gin 与驱动脚本即可，无需打补丁。

```bash
git clone https://github.com/mit-han-lab/infinigen.git
```

把本目录的 `configs/*.gin` 与 `scripts/*.sh` 复制进官方仓库相应位置（见下文），照第五节命令运行。

### Path B —— 精确复现作者当年产出（可选）

如果希望与作者当年生成的场景**逐字节级**一致，需要：
1. 使用作者当时的官方 commit（见下方「修改说明」中列的核心文件基线版本）；
2. 套用作者对 4 个核心文件所做的**轻量修改**（主要是调试打印与并行/设备参数）。

这些修改**不改变输出语义**，仅当你要严格复现时才需要。作者当年的完整运行命令已整理在
[第五节](#五模板化运行命令-来自作者开发日志)。

> **建议**：默认走 **Path A**。Path B 仅在需要「与论文数据完全对齐」时使用。

---

## 三、目录内容

```
third_party/infinigen/
├── README.md              # 本文档
├── configs/               # 作者新增/改编的 gin 配置（覆盖官方默认参数）
│   ├── custom_indoor.gin  # ★ 核心：室内批量生成参数（相机数量/高度/视角多样性/分辨率）
│   ├── overhead.gin       # 俯视（top-down/overhead）渲染配置
│   └── topview.gin        # 顶端平视渲染配置（720×720）
└── scripts/               # 作者自写的驱动脚本（shell）
    ├── indoor.sh                     # 单房间逐场景生成（coarse）
    ├── rebuttal.sh                   # 论文 rebuttal 用：多房间逐场景生成（coarse）
    ├── rebuttal_retry_render.sh      # 上述场景的补渲染（render）
    └── save_all_scenes_annotation.sh # ★ 批量提取标注（调用 save_scene_annotation.py，即 Step 2）
```

> 上游官方代码本体（`infinigen/`、`infinigen_examples/` 等）**不在本仓库**，请自行 clone。
> `save_scene_annotation.py`（Step 2 脚本）不属于 Infinigen，它位于 `mvstride/scene_processing/infinigen/`。

---

## 四、作者对 Infinigen 做了哪些修改

作者当年是在 `pip install -e ".[terrain,vis]"` 安装的官方仓库（本地 clone 目录，如
`/path/to/infinigen-main`）基础上，做了**两类**修改：

### 1）新增/改编配置文件（已含在本目录 `configs/`，随仓库提交）

| 文件 | 作用 | 关键参数 |
|---|---|---|
| `custom_indoor.gin` | 室内批量生成的核心配置 | `camera.spawn_camera_rigs.n_camera_rigs = 10`（相机组数）；`camera.camera_pose_proposal.altitude = ("clip_gaussian", 1.5, 0.5, 0.3, 2.0)`（限制相机高度 0.3–2.0m，避免贴天花板）；`compute_base_views.min_candidates_ratio = 20`、`configure_cameras.mvs_radius = ("uniform", 0.5, 5)`（增强视角多样性）；`execute_tasks.generate_resolution = (640, 480)`、`get_sensor_coords.H/W = 480/640`；`full/render_image.passes_to_save = []`（只保留需要的渲染通道）；`BlueprintSolidifier.enable_open=False`、`restrict_solving.solve_max_rooms=2`（房间数≥2） |
| `overhead.gin` | 俯视渲染 | 关闭天花板/隐藏其它房间、关闭地形与自然背景、关闭灯光随机，`overhead_cam_enabled=True` |
| `topview.gin` | 顶端平视渲染 | 720×720 分辨率，`compose_indoors.topview=True`，关闭地形与大解算 |

> 这些 gin 依赖官方仓库 `infinigen_examples/configs_indoor/` 与 `configs_nature/` 里的**既有配置**
> （如 `camera`、`compute_base_views`、`execute_tasks` 等 binding），因此不能脱离官方仓库裸用。

### 2）对 4 个核心文件做了轻量源码修改（未随仓库提交，Path B 可选）

作者在官方源码中做了**仅用于调试/复现的轻量改动**，语义不变。它们是：

| 文件（官方相对路径） | 改动性质 |
|---|---|
| `infinigen/core/placement/camera.py` | 在 `spawn_camera_rigs` 中增加打印 `n_camera_rigs:` 等调试输出；支持 `mvs_setting/mvs_radius` 视角采样 |
| `infinigen/core/execute_tasks.py` | 在读取 `camera_id` 处增加 `print("camera_id value:", ...)` 调试输出 |
| `infinigen/datagen/manage_jobs/monitor_tasks.py` | 批处理调度器监控相关的小参数调整（与并行/warmup 调参相关） |
| `infinigen/datagen/manage_jobs.py` | `local_submit_cmd` 等本机并行提交参数调整 |

**这些修改不影响 Step 2/Step 3 的输入**（`read_single_scene_gt` 读取的是标准 GT 输出），因此：
- 走 **Path A**（默认）**不需要**打任何补丁；
- 走 **Path B**（精确复现）时，才需要对上述文件应用 `print` 调试 + 并行参数的对应改动——由于改动量极小且
  均为调试性质，建议直接按官方 API 自行适配，本项目不维护 patch 文件（避免随官方版本漂移失效）。

---

## 五、模板化运行命令（来自作者开发日志）

> 下方命令摘自作者当年开发日志（`development_log/`，未随开源仓库提交，仅作内部参考）。
> 命令中的**相对路径**以「官方仓库根目录」为当前目录为准；`<INFINIGEN_ROOT>` 为官方仓库绝对路径。

### 0）环境安装

```bash
# 需满足 Infinigen 依赖（BLENDER_PATH 等），官方文档为准
pip install -e ".[terrain,vis]"
```

### 1）单场景调试生成（coarse，只生成 .blend）

```bash
# 单个随机房间，第一视角（CPU，约 10~13 min）
python -m infinigen_examples.generate_indoors --seed 0 --task coarse \
  --output_folder outputs/indoors/coarse \
  -g fast_solve.gin singleroom.gin \
  -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=["Kitchen"]
```

常用房间类型把 `["Kitchen"]` 换成 `["Bathroom"]` / `["Bedroom"]` / `["DiningRoom"]` / `["LivingRoom"]`。

### 2）单场景渲染（render，输出 RGB + GT）

```bash
# 渲染 RGB（需要先有 coarse 输出的 .blend）
python -m infinigen_examples.generate_indoors --seed 0 --task render \
  --input_folder outputs/indoors/coarse --output_folder outputs/indoors/frames
```

### 3）批量生成 + 渲染（核心命令，作者实际用于刷批次）

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m infinigen.datagen.manage_jobs \
  --output_folder outputs/batch_generation --num_scenes 38 \
  --pipeline_configs cuda_terrain.gin local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin \
  --configs custom_indoor.gin multiview_stereo.gin fast_solve.gin \
  --pipeline_overrides iterate_scene_tasks.n_camera_rigs=10 LocalScheduleHandler.use_gpu=True get_cmd.driver_script='infinigen_examples.generate_indoors' manage_datagen_jobs.num_concurrent=4 --warmup_sec 2000 \
  --overrides compose_indoors.terrain_enabled=False Terrain.device='cuda' compose_indoors.restrict_single_supported_roomtype=True
```

参数速查：

| 参数 | 含义 |
|---|---|
| `--num_scenes` | 生成场景总数（作者按硬件调过 30/38/40） |
| `--configs custom_indoor.gin ...` | 合并作者自定义 + 官方加速配置 |
| `iterate_scene_tasks.n_camera_rigs=10` | 每个场景实际渲染 10 个相机组 |
| `manage_datagen_jobs.num_concurrent=4` | 本机并行任务数 |
| `--warmup_sec 2000` | 任务启动预热/错峰（作者从 1200–2000 都试过） |
| `compose_indoors.restrict_single_supported_roomtype=True` | 限定单房间类型，加速且更稳 |

> 提速/排错经验（来自日志）：`rm -rf /tmp/blender_*` 清缓存；偶发「卡在 Searching for camera viewpoints」
> 是随机种子不适合，换个 `--specific_seed` 即可；`queue_render.gpus`/`ground_truth/queue_render.gpus` 改 1
> 可让 GPU 渲染（8 视角约 25 min）。

### 4）批量提取标注 → Step 2 元数据

> 这一步调用 **[`save_scene_annotation.py`](../../mvstride/scene_processing/infinigen/save_scene_annotation.py)**
> （MV-STRIDE 自己的脚本，随本仓库发布），把一批场景文件夹的 GT 汇总成 `scene_metadata.json`。
> 作者当年是把该脚本拷进官方仓库 `scripts/` 再运行，因此也可直接
> `python <INFINIGEN_ROOT>/scripts/save_scene_annotation.py`：

```bash
python <INFINIGEN_ROOT>/scripts/save_scene_annotation.py \
  --batch-names batch_generation batch_generation_coarse batch_generation_3 ... \
  --base-dir ${INFINIGEN_ROOT}/outputs \
  --save-dir ${INFINIGEN_ROOT}/saved_scenes \
  --camera-num 10 \
  --frame-prefix "0_0048_0"
```

驱动脚本见 [`scripts/save_all_scenes_annotation.sh`](scripts/save_all_scenes_annotation.sh)（已把所有私有绝对路径
中性化为 `${INFINIGEN_ROOT}` 环境变量）。

---

## 六、批量输出的 GT 格式

批量生成后，每个场景在其 `frames/` 下按相机组织（`camera_0` … `camera_9`）：

| 路径 | 内容 |
|---|---|
| `camview/camera_0/camview_<i>_0_0048_0.npz` | 相机位姿：`K`（3×3 内参）、`T`（4×4 外参）、`HW` |
| `Image/camera_0/Image_<i>_0_0048_0.png` | 渲染 RGB 图 |
| `InstanceSegmentation/camera_0/..._<i>_0_0048_0.npy` | 实例分割（H×W×3，每个实例可区分） |
| `ObjectSegmentation/camera_0/..._<i>_0_0048_0.npy` | 物体分割（值 = `Objects_*.json` 的 `object_index`） |
| `Objects/camera_0/Objects_<i>_0_0048_0.json` | 物体元信息（类别、房间从属 `room_0/0.floor` 等） |
| `*.blend` | 场景文件（含相机、物体 3D 坐标/包围框，可后续提取 3D 标注） |

> 坐标惯例（含作者验证）：`+X → 右，+Y → 前，+Z → 上`；物体本地 `Y 轴 = 语义前向`。
> `frame` 编号由 `iterate_scene_tasks.frame_range/render_frame_range` 决定（示例里固定渲染第 48 帧）。

---

## 七、License 与上游

- 上游：https://github.com/mit-han-lab/infinigen （BSD-3-Clause）
- 本目录仅含作者对官方仓库的**增量配置与驱动脚本**；官方代码版权归其作者所有，请遵守其许可证。
- ScanNet++（另一上游数据源）见 [`third_party/scannetpp/`](../scannetpp/README.md)。
