# evaluation

评估与质量控制模块：测试集采样、指标计算、QC / 人工标注工具。

## 职责

- **跨视角依赖测试采样** `cross_view_dependency_test_sampling.py`：采样 1000 条多视角测试样本，并派生单视角变体用于跨视角依赖消融。
- **人工评估采样** `human_eval_sampling.py` / `human_eval_cot_sampling.py`：200 条 multi-view / CoT 人工评估集。
- **指标计算** `calculate_metrics_local.py` / `calculate_metrics_server.py`：按类别计算相对 Random(0.25)/Majority 基线的准确率；server 版支持 CoT 答案提取。
- **质量控制** `collect_qc_samples.py`：分层 100 样本 QC 集（配套的 Streamlit 标注工具 `qc_visualizer_app.py` 见 `misc/visualization/`）。

> 来源：原 `data_test_sampling/`、`data_qc_sampling/` 目录（交互式标注/可视化工具已归档 `misc/visualization/`）。
