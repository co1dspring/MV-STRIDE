"""MV-STRIDE: data-construction + training-data production package.

This package builds hierarchical, multi-view spatial reasoning training data
for multimodal large language models (MLLMs) from Infinigen (synthetic 3D
scenes) and ScanNet++ (real-world reconstructed scenes).

Pipeline sub-packages
---------------------
- ``mvstride.scene_processing`` : extract annotations from generated/captured
  scenes and package them into the QA pipeline input format.
- ``mvstride.qa_generation``    : generate hierarchical (Level I/II/III + MSR)
  multi-view spatial reasoning QA groups.
- ``mvstride.llm``              : LLM-based CoT generation, verification, MCA
  and post-processing.
- ``mvstride.evaluation``       : test-set sampling, metric computation and QC.

For the upstream scene/generation code (Infinigen, ScanNet++) see
``third_party/``; for run configuration see ``configs/``.
"""

__version__ = "0.1.0"

__all__ = [
    "scene_processing",
    "qa_generation",
    "llm",
    "evaluation",
]
