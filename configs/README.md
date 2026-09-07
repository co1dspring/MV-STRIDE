# configs

运行配置，**与源码分离**（`mvstride/` 只含代码，不含数据/环境相关配置）。

## 目录

- `qa/` — QA 生成的配置与模板（按数据源划分，见 [`qa/README.md`](qa/README.md)）：
  - `qa_config_{infinigen,scannetpp}.json`（完整配置）
  - `qa_config_{infinigen,scannetpp}_sparse.json`（早期小规模）
  - `qa_config_{infinigen,scannetpp}_ablation.json`（跨视角依赖消融）
  - `qa_templates.json`（30 类问题/答案文本模板）
  - `qa_dependency_tree.json`（Level III → 前置 Level I/II 映射）

> **路径约定**：`qa/` 内配置中的**相对路径以 `qa/` 目录为基准**解析（与运行时的当前目录无关），
> 生成器（`mvstride/qa_generation/generate_QAs_multilevel_multistage.py`）按该约定读取，详见
> [`qa/README.md`](qa/README.md)。
