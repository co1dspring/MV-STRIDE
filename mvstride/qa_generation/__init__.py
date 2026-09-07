"""Hierarchical multi-view spatial reasoning QA generation.

Core data-construction module. Builds multi-view QA groups organized into
three progressive capability levels (with explicit cross-level dependencies)
plus MSR multi-view types, from per-scene geometry metadata.

Planned layout (pending user guidance on file placement):

- ``generators/`` : the main generator and Level I/II mixin generators
                    (moved from ``generate_QAs_multilevel_multistage*.py`` and
                    ``multilevel_qa/``).
- ``geometry/``   : math / geometry utilities (moved from ``util/math_utils.py``).
- ``utils/``      : shared QA / filter / common utilities
                    (moved from ``util/*.py``).
- ``sampling/``   : ratio-based stage sampling & generic json utils
                    (moved from ``sample_stage*.py``, ``merge_json.py`` ...).

Run configuration lives in ``configs/qa/`` (``qa_config_*.json``,
``qa_templates.json``, ``qa_dependency_tree.json``).
"""
