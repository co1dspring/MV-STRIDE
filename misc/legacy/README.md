# misc/legacy

过期 / 早期版本代码（保留作参考，不推荐使用；当前主实现见对应 `mvstride/*` 模块）。

| 文件 | 说明 |
|---|---|
| `extract_metadata.py` | `mvstride/scene_processing/infinigen/save_scene_annotation.py` 的早期版本 |
| `generate_QAs_multilevel_multistage2.py` | 主生成器的部分重构迭代版本（与 `*3.py` 为同一类的两半，**不可运行**） |
| `generate_QAs_multilevel_multistage3.py` | 同上，不可运行的扩展片段（主入口为 `mvstride/qa_generation/generate_QAs_multilevel_multistage.py`） |
| `split_json.py` | 按类别切分 QA 的旧硬编码版（无参数）；新版参数化实现见 `tools/split_json.py` |
