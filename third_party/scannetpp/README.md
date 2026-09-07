# third_party/scannetpp

本目录存放 MV-STRIDE 数据构建中 **ScanNet++（真实场景）数据源**的处理代码：把官方 ScanNet++
**iPhone 拍摄 + 重建**的场景数据，转成与 Infinigen 支路相同的统一元数据格式，供同一套 QA 生成器使用。

> **产出**：每场景一份 `scene_metadata.json`（自研投影脚本产出）+ `scene_metadata_new.json`
> （统一 schema，`mvstride/scene_processing/scannetpp/scannetpp2infinigen_new.py` 产出），即
> [`configs/qa/qa_config_scannetpp.json`](../../configs/qa/qa_config_scannetpp.json) 中
> `source_data_dir` 指向的目录内容。

ScanNet++ 官方仓库**本体不包含在本仓库内**。本目录只收录作者围绕官方数据/代码**自行编写（或深度改编）**
的部分，并说明如何取得官方代码与数据。请自行 clone 官方仓库，把它的根目录加入 `PYTHONPATH`（见下）。

---

## 一、整体数据流水线（三步）定位

MV-STRIDE 的数据构建分三步；对 ScanNet++ 而言这三步的代码分布是：

| 步骤 | 内容 | 代码位置 |
|---|---|---|
| **Step 1** | 获得场景数据：官方下载（原始数据 + 相机位姿 + 网格 + 标注），并抽帧出 iPhone RGB | 官方工具（下载器 / `prepare_iphone_data.py`），本目录只给指引 |
| **Step 2** | 提取标注：把 3D 标注投影/采样到 iPhone 帧上 → `scene_metadata.json`；再统一 schema → `scene_metadata_new.json` | `third_party/scannetpp/scripts/preprocess_iphone.py`（本目录，**自研改编**）+ `mvstride/scene_processing/scannetpp/scannetpp2infinigen_new.py`（见 [scene_processing](../../mvstride/scene_processing/README.md)） |
| **Step 3** | 合成 QA：与 Infinigen 共用 QA 生成器 | `mvstride/qa_generation/generate_QAs_multilevel_multistage.py` + [`configs/qa/`](../../configs/qa/README.md) |

> 本仓库只打通 **iPhone 支路**（`data_type = "iphone"`）。iPhone-only 的定位在 Step 2 转换脚本
> （`data_type = "iphone"` 硬编码）与 QA 阶段（按目录名 `*_iphone` 过滤）两层确定；官方 DSLR 数据
> 不在本链路内（早期曾产出 `obj_annotation_dslr.json` 变体，非 QA 链路，未收录）。

---

## 二、代码归属：官方 vs 自研

ScanNet++ 场景的标注投影本身是官方流程（`semantic/prep/semantics_2d.py` + `common/` 工具包），
作者**没有重写几何逻辑**，而是深度改编官方脚本 + 自研了过滤策略与输出格式。本目录只提交自研部分：

| 文件（本目录） | 性质 | 说明 |
|---|---|---|
| `scripts/preprocess_iphone.py` | **自研改编**（基于官方 `semantic/prep/semantics_2d.py`） | 逐行派生于官方语义投影脚本，但把「预计算 raster .pth 缓存 + 官方全套输出」改为「**逐帧即时 rasterization** + 紧凑 JSON 输出」；只保留 iPhone 分支；新增可见性/类别过滤与统一输出 |

改动/新增的要点：

1. **即时光栅化**：官方脚本消费预先算好的 `pix_to_face/zbuf .pth` 缓存；作者版本直接用
   `common/utils/rasterize.py` 对每帧光栅化（省去中间缓存，代价是需 GPU）。
2. **物体过滤**（官方没有）：
   - 黑名单类别：`wall / floor / ceiling / carpet / doorframe / split / remove / object` 等结构/背景类；
   - 2D 面积占比 ≥ `0.01%`（`obj_pixel_thresh`）；
   - 可见像素占比 ≥ `0.1%`（`min_visible_pixel_ratio`）；
   - 可见网格顶点占比 ≥ `5%`（`min_visible_vertex_ratio`）。
3. **输出格式**：每帧一条记录（见[第六节](#六输出格式)），写入 `<output>/<scene_id>/scene_metadata.json`；
   默认还会把选中的帧图拷到 `<output>/<scene_id>/iphone/`，使输出目录自洽。

作者**直接引用、不改动**的官方代码（需自行 clone 后提供）：

| 官方模块 | 用途 |
|---|---|
| `scannetpp/common/scene_release.py` | 官方场景目录布局（`data_root/<scene_id>/iphone|scans/...`） |
| `scannetpp/common/utils/colmap.py` | 读 `cameras.txt/images.txt` → 相机内参 + 采样帧位姿（`get_camera_images_poses`） |
| `scannetpp/common/utils/rasterize.py` | pytorch3d 网格光栅化（内部硬编码 `cuda:0`） |
| `scannetpp/common/utils/anno.py` | `load_anno_wrapper`（读取 `segments_anno.json`/`segments.json`）、2D bbox 与像素级物体 id 提取 |
| `scannetpp/common/utils/image.py` | 图片读取 |

---

## 三、目录内容

```
third_party/scannetpp/
├── README.md                    # 本文档
└── scripts/
    └── preprocess_iphone.py     # ★ Step 2a：官方 3D 标注 → 投影到采样帧 → scene_metadata.json
```

> 官方代码本体不在本仓库。`scannetpp2infinigen_new.py`（Step 2b，纯自研，无官方依赖）位于
> [`mvstride/scene_processing/scannetpp/`](../../mvstride/scene_processing/scannetpp/scannetpp2infinigen_new.py)；
> 早期「抽帧」脚本与官方 `prepare_iphone_data.py` 几乎逐字重复，故不收录，直接用官方脚本（见[第五节](#五运行步骤)）。

---

## 四、官方资产清单（每个场景需要什么）

以官方数据根目录 `DATA_ROOT`（内含 `<scene_id>/`）为准，跑通本链路每个场景**最少**需要：

| 官方资产 | 路径（相对 `DATA_ROOT/<scene_id>/`） | 用途 |
|---|---|---|
| iPhone 视频 | `iphone/rgb.mkv` | 抽帧来源（60fps；抽帧后 `iphone/rgb/frame_%06d.jpg`） |
| iPhone COLMAP 位姿 | `iphone/colmap/{cameras.txt, images.txt}` | 内参 + 注册帧位姿（帧采样依据） |
| 对齐网格 | `scans/mesh_aligned_0.05.ply` | 光栅化/投影的几何 |
| 实例标注 | `scans/segments_anno.json` + `scans/segments.json` | 物体类别、OBB（centroid / axesLengths / normalizedAxes）、顶点→物体 id |
| 姿态元数据 | `iphone/pose_intrinsic_imu.json`、`exif.json` 等（官方场景完整目录自带） | `ScannetppScene_Release` 解析所需 |

不需要的（本链路不读）：DSLR 图/位姿、iPhone depth、mask、语义网格等——按需下载可显著省流量。

> 下载方式请遵循官方指引：在 [scannetpp.cs.uni-bonn.de](https://scannetpp.cs.uni-bonn.de) 申请
> token，用官方下载器 <https://github.com/scannetpp/scannetpp_downloader> 按资产下载；
> 官方代码仓库 <https://github.com/scannetpp/scannetpp>。

---

## 五、运行步骤

依赖：Python 3.8+，`torch / pytorch3d / open3d / numpy / opencv / tqdm`（GPU，官方 rasterize 硬编码
`cuda:0`），并把**官方 scannetpp 仓库根目录**加入 `PYTHONPATH`（`import scannetpp.common...`）。

### 0) 环境

```bash
git clone https://github.com/scannetpp/scannetpp.git <SCANNETPP_ROOT>
# 安装官方仓库依赖（pytorch3d / open3d / ...），并让 Python 能找到它：
export PYTHONPATH=<SCANNETPP_ROOT>:$PYTHONPATH
```

### 1) 下载数据（官方工具）

下载后数据布局应满足 `DATA_ROOT/<scene_id>/iphone/...`、`DATA_ROOT/<scene_id>/scans/...`。

### 2) iPhone 抽帧（官方脚本，本仓库不重复收录）

```bash
# 官方 prepare_iphone_data.py 的 extract_rgb：逐场景执行等价于
ffmpeg -i <scene>/iphone/rgb.mkv -start_number 0 -q:v 1 <scene>/iphone/rgb/frame_%06d.jpg
```

### 3) 标注投影 + 帧采样 → scene_metadata.json（本目录脚本）

```bash
python third_party/scannetpp/scripts/preprocess_iphone.py \
  --data-root $DATA_ROOT \
  --output-root ./scannetpp_sampled_new \
  --sample-rate 5 \
  [--scene-list scenes.txt]        # 可选：只处理部分场景
  [--no-images]                    # 默认会拷帧到输出目录
```

### 4) 统一 schema → scene_metadata_new.json + images/（mvstride/scene_processing/scannetpp/ 下）

```bash
python mvstride/scene_processing/scannetpp/scannetpp2infinigen_new.py \
  --raw-root ./scannetpp_sampled_new \    # 帧图目录（默认 ./scannetpp_sampled，兼容历史数据）
  --anno-root ./scannetpp_sampled_new \   # scene_metadata.json 目录
  --output-root ./scannetpp_sampled_modified
```

产出 `<output>/<scene_id>_iphone/{scene_metadata_new.json, images/frame_*.jpg}`，
即 `qa_config_scannetpp.json` 的 `source_data_dir` 输入。

### 5) QA 生成（与 Infinigen 共用）

```bash
# 从仓库根目录运行；把 mvstride/qa_generation/generate_QAs_multilevel_multistage.py 末尾的
# CONFIG_PATH 设为 './configs/qa/qa_config_scannetpp.json' 后执行：
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py
```

> 生成器以 `configs/qa/` 为基准解析其中的相对路径；`qa_config_scannetpp.json` 的 `source_data_dir` /
> `training_environment_base_dir` 指向本流程第 4 步的输出目录（见 [`configs/qa/README.md`](../../configs/qa/README.md)）。

---

## 六、输出格式

### `scene_metadata.json`（每帧一条）

```jsonc
{
  "scene_id": "036bce3393",
  "image_name": "frame_000000.jpg",
  "image_path": "036bce3393/iphone/frame_000000.jpg",
  "extrinsic": [[...]],   // 4x4 世界→相机（COLMAP 约定，官方 Image.to_transform_mat）
  "intrinsic": [[...]],   // 3x3 K
  "objects": [{
    "category": "chair",
    "3D_location": [6.56, 2.13, 0.54],     // OBB centroid，保留 2 位小数
    "3D_size": [0.52, 0.83, 0.50],         // OBB axesLengths
    "3D_rotation": [[...]],                // OBB normalizedAxes
    "2D_bbox": [785, 753, 1296, 1113]        // 像素坐标 [y1, x1, y2, x2]（y=图像行轴）
  }]
}
```

> 2D_bbox 的分量顺序与官方 `get_bboxes_2d` 的返回**逐位一致**：该函数按 `np.nonzero` 的索引顺序返回
> `[轴0最小, 轴1最小, 轴0最大, 轴1最大]`，轴0 即图像行（y）、轴1 即图像列（x）。转换脚本据此把
> `bbox[1]/bbox[3]` 解析为 `min_x/max_x`、`bbox[0]/bbox[2]` 解析为 `min_y/max_y`，QA 侧以此为准。

### `scene_metadata_new.json`（统一 schema，与 Infinigen 支路对齐）

顶层：`{scene_id, data_type: "iphone", cameras: {...}, objects: {...}}`

- `cameras.<frame_stem>`：`cam_intrinsics`(3x3)、`cam_extrinsics`(4x4，相机→世界；对 COLMAP 位姿取逆后
  左乘坐标轴翻转 `diag(1,-1,-1)`，对齐 Infinigen 的相机坐标约定)、`width/height`、`image_path`
  （`<scene>_iphone/images/<frame>.jpg`）、`location_3d`、`forward_direction`、`objects`（该帧可见的
  物体索引 → `bbox_2d{min_x,min_y,max_x,max_y}`）。
- `objects.<index>`：跨帧去重后的全局物体——`category`、`3d_center`、`axis_directions{local_x/y/z}`、
  `bbox_3d_aabb{min/max/dimensions}`（由 OBB 8 顶点转 AABB）。

> 同一帧若经 `repair_bbox.py`（`mvstride/scene_processing/scannetpp/`）后处理，bbox 会从像素坐标转为 0–1000 相对坐标。

---

## 七、帧采样算法（作者当时的做法）

帧采样**没有任何随机性**，逻辑在官方 `common/utils/colmap.py::get_camera_images_poses`：

1. 读 `iphone/colmap/images.txt` 中所有 COLMAP **注册帧**（该场景约 600+ 帧）；
2. 按 image id 排序；
3. 每隔 `sample_rate` 帧取一帧：`all_extrinsics[::sample_rate]`，脚本默认 `--sample-rate 5`。

官方 iPhone 注册大约每 10 个视频帧注册一次，因此实际输出约为**每 50 个视频帧取 1 帧**，
每场景约 **120–130 帧**（本仓库 138 场景采样数据即按此产出，`global_seed` 无关、可逐位复现）。
采样后的帧照常经过[第二节](#二代码归属官方-vs-自研)的可见性过滤，因此不是所有帧都会保留物体记录。

---

## 八、作者环境中的目录对照（复现历史产物用）

作者机（原始数据处理机）上的实际产物与仓库脚本默认参数一一对应：

| 目录 | 内容 | 对应仓库脚本 / 说明 |
|---|---|---|
| `data/` | 官方下载的原始数据（每场景 `iphone/ + scans/ + dslr/`） | 下载器输出；`preprocess_iphone.py --data-root` 指向其父目录 |
| `scannetpp_sampled/` | 历史抽帧树（`<scene>/iphone/frame_*.jpg`，曾带 `.jpg.jpg` 双后缀）+ 早期 `obj_annotation*.json` 变体 | 早期变体**非 QA 链路**；帧图树是 `scannetpp2infinigen_new.py --raw-root` 的默认值（双后缀兼容已内建：找不到时自动补 `.jpg`） |
| `scannetpp_sampled_new/` | 每场景 `scene_metadata.json`（138 场景） | `preprocess_iphone.py` 输出；`--anno-root` 默认值 |
| `scannetpp_sampled_modified/` | 每场景 `<scene>_iphone/{scene_metadata_new.json, images/}` | `scannetpp2infinigen_new.py` 输出；**QA 生成器的输入**（`qa_config_scannetpp.json` 的 `source_data_dir`） |

> 从零复现时推荐：让 `preprocess_iphone.py` 直接把帧图拷进输出目录（默认行为），再令
> `--raw-root` 与 `--anno-root` 指向同一输出目录，即可得到与历史 `scannetpp_sampled_modified`
> 结构一致的产物。

---

## 九、License 与上游

- 上游代码：<https://github.com/scannetpp/scannetpp>（官方仓库未附带明确 LICENSE 文件）
- 数据下载：<https://scannetpp.cs.uni-bonn.de> + <https://github.com/scannetpp/scannetpp_downloader>
- 本目录仅含作者**自研改编**的脚本（`scripts/preprocess_iphone.py`）；几何逻辑与资产路径大量依赖官方
  `common/` 工具包，请自行 clone 官方仓库并遵守其条款。Infinigen（另一上游）见
  [`third_party/infinigen/`](../infinigen/README.md)。
