# tools

转换 / 清洗 / 采样 / 统计等**脚本工具**（数据流水线中会用到的辅助脚本，不入 `mvstride` 包）。

| 文件 | 用途 |
|---|---|
| `split_json.py` | 按 `category` 把 QA JSON 切成多个文件（参数化；旧硬编码版见 `misc/legacy/split_json.py`） |
| `expand_json.py` | 列表元素交错复制 ×10（10-shot 推理数据扩充） |
| `remove_string.py` | 从训练文本中剥离 `<image>` 等标记 |
| `acc_sample.py` | 按 10-shot pass rate 的难度采样（Stage 3 RL 数据筛选） |
| `count_total_qa.py` | QA 数量统计 |
| `read_completions.py` | RL 循环采样结果读取 / 诊断 |

> 说明：QA/CoT 相关的转换与清洗脚本大多随 `mvstride/` 重构归入各子包
> （`mvstride/llm/`、`mvstride/scene_processing/`），纯统计与辅助工具在此归档。
