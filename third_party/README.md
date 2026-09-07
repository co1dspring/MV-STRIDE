# third_party

**Part A：原始数据构造** —— 上游场景生成 + 重建/拍摄代码。

本仓库假设已获得生成 / 重建的场景数据。这里的两个子目录存放**数据源相关的增量代码**：官方代码本体
（通常很大、随版本变动）不在本仓库，这里只保留作者围绕官方代码**自行编写/改编**的部分与使用指引。

| 子目录 | 内容 | 状态 |
|---|---|---|
| `infinigen/` | Infinigen 场景生成 + 渲染。保留围绕官方代码的**增量配置与驱动脚本**（`configs/` + `scripts/`）及[完整说明](infinigen/README.md)（两条可复现路径 + 模板命令） | 配置/脚本已就位，官方代码请自行 clone |
| `scannetpp/` | ScanNet++ 真实场景数据源。保留作者**自研改编**的标注投影脚本 `scripts/preprocess_iphone.py`（基于官方 `semantics_2d.py`），及下载/抽帧/转换/复现指引 | 脚本已就位，官方代码与数据请自行获取 |

## 说明

- 这些代码产出**原始场景数据**（Infinigen 生成+渲染的图像、ScanNet++ 拍摄/重建的数据）。
- Infinigen 的**上游官方仓库**建议直接 `git clone https://github.com/mit-han-lab/infinigen`（默认 Path A，无需修改）；
  如要精确复现作者产出可参考 `infinigen/README.md` 的 Path B。
- ScanNet++ 的官方代码/下载器/数据申请方式见 `scannetpp/README.md`；官方仓库本体请自行
  `git clone https://github.com/scannetpp/scannetpp` 并加入 `PYTHONPATH`。
- 从这些场景中**提取标注并打包成 QA pipeline 输入格式**的代码（Infinigen 侧 `save_scene_annotation.py`、
  ScanNet++ 侧 `scannetpp2infinigen_new.py`/`repair_bbox.py`），属于自研的 [scene_processing](../mvstride/scene_processing/)
  模块（Part B），不放在这里。
