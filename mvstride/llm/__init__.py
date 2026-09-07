"""LLM-based annotation generation: CoT generation, verification and MCA.

This is the Stage-2 of the data pipeline: it turns Level III QA groups into
long chain-of-thought (CoT) supervision, verifies the CoT against the
ground-truth sub-answers, and produces multiple-choice (MCA) / paraphrased
variants, then strips metadata into a clean training format.

Planned layout (pending user guidance on file placement):

- ``clients.py``      : OpenAI-compatible API wrapper (vision + text)
                        (moved from ``api/api_interface.py``).
- ``cot_generation.py`` : LLM CoT generation (moved from
                        ``api/mvstride_cot_generation.py``).
- ``cot_verification.py`` : CoT quality verification vs GT (moved from
                        ``api/mvstride_cot_verification.py``).
- ``cvd_test.py``     : multiple-choice inference on the cross-view-dependency
                        test set (moved from ``api/mvstride_cvd_test.py``).
- ``variants.py``     : QA -> MCQ / paraphrase (moved from
                        ``api/choice_generation.py``, ``api/paraphrasing.py``).
- ``postprocess.py``  : format repair & metadata stripping (moved from
                        ``api/cot_jsonl2swift.py``, ``remove_system.py``,
                        ``match_input_output.py``, ``rebalance_options.py``,
                        ``mca_jsonl2swift.py``).
- ``analysis.py``     : verification statistics report (moved from
                        ``api/analyse_cot_verification_result.py``).
"""
