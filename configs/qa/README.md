# configs/qa

QA 生成的**配置与模板**文件（与源码分离）。生成器从任意工作目录运行，这些配置中的**相对路径均以本目录**
（`configs/qa/`）为基准解析，与运行时「当前目录」无关。

## 文件

| 文件 | 作用 |
|---|---|
| `qa_config_infinigen.json` / `qa_config_scannetpp.json` | 按数据源的完整 QA 生成配置 |
| `qa_config_{infinigen,scannetpp}_sparse.json` | 早期小规模配置 |
| `qa_config_{infinigen,scannetpp}_ablation.json` | 跨视角依赖消融配置（单视角 vs 多视角实验，31.5% → 41.2%） |
| `qa_templates.json` | 各类别的问题/答案文本模板 |
| `qa_dependency_tree.json` | Level III 问题 → 其前置 Level I/II 子问题的映射 |

## 路径解析规则

生成器（`mvstride/qa_generation/generate_QAs_multilevel_multistage.py`）读取配置后，会把其中的**相对路径统一解析为**`configs/qa/`**之下的路径**，
而不是相对于运行命令所在目录。因此：

- `qa_templates_path = ./qa_templates.json` → 实际读取 `configs/qa/qa_templates.json`；
- `qa_dependency_tree_path = ./qa_dependency_tree.json` → 同理；
- `source_data_dir` / `training_environment_base_dir` / `output_dir` 若为相对路径，同样以本目录为基准。

> 因此，指向场景数据/输出目录的路径，也应写成**相对本目录**的相对路径（或绝对路径）。

## 运行

主入口是 `mvstride/qa_generation/generate_QAs_multilevel_multistage.py`。从**仓库根目录**运行即可，
`CONFIG_PATH` 默认指向 `./configs/qa/qa_config_scannetpp_ablation.json`；
改文件末尾的 `CONFIG_PATH` 即可切换数据源/规模：

```bash
# 在仓库根目录
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py
```
