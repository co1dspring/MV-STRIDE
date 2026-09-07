"""Test-set sampling, metric computation and human/automatic QC.

Evaluation helpers for the dataset: build test sets (incl. cross-view
dependency and human-eval sampling), compute per-category accuracy against
baselines, and run quality-control sampling/annotation.

Planned layout (pending user guidance on file placement):

- ``test_sampling.py`` : multi/single-view + human-eval sampling (moved from
                         ``data_test_sampling/cross_view_dependency_test_sampling.py``,
                         ``human_eval_sampling.py``, ``human_eval_cot_sampling.py``).
- ``metrics.py``       : accuracy vs Random/Majority baselines (moved from
                         ``data_test_sampling/calculate_metrics_local.py`` /
                         ``calculate_metrics_server.py``).
- ``qc.py``            : stratified QC sampling + Streamlit QC / human-eval tools
                         (moved from ``data_qc_sampling/`` and the
                         ``*_visualization.py`` annotation apps).

The interactive annotation apps (Streamlit) may alternatively be treated as
dev tools in ``misc/``; placement is pending user guidance.
"""
