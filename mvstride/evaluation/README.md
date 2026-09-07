# evaluation

Evaluation and quality-control module: test-set sampling, metric computation, QC / human-labeling
tools.

## Responsibilities

- **Cross-view dependency test sampling** `cross_view_dependency_test_sampling.py`: samples 1000
  multi-view test items and derives single-view variants for the cross-view dependency ablation.
- **Human evaluation sampling** `human_eval_sampling.py` / `human_eval_cot_sampling.py`: 200-item
  multi-view / CoT human-evaluation sets.
- **Metric computation** `calculate_metrics_local.py` / `calculate_metrics_server.py`: per-category
  accuracy relative to the Random (0.25) / Majority baselines; the server version supports CoT answer
  extraction.
- **Quality control** `collect_qc_samples.py`: stratified 100-sample QC set (companion Streamlit
  labeling app `qc_visualizer_app.py` in `misc/visualization/`).

> Origin: the former `data_test_sampling/` and `data_qc_sampling/` directories (interactive
> labeling/visualization tools archived under `misc/visualization/`).
