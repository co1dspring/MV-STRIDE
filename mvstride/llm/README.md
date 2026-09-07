# llm

LLM-based **data annotation generation** module (data pipeline Stage 2): turns Level III QA groups into
long chain-of-thought (CoT) supervision, verifies them against ground truth, produces MCA / paraphrase
variants, and finally cleans everything into training formats.

> Note: this layer is part of **data construction** (producing CoT/MCA annotations), so it lives inside
> the `mvstride` package.

## Flow

1. **CoT generation** `mvstride_cot_generation`: sends the multi-turn QA context to an LLM (vision),
   asking it to produce a step-by-step CoT in `thinking ... <answer> ... </answer>` format for the
   Level III main question.
2. **Post-processing**: `cot_jsonl2swift` (format repair), `match_input_output` (align with the
   original QA and inject ground truth), `remove_system` (ablation variant that strips the system
   prompt).
3. **Verification** `mvstride_cot_verification`: a lenient reviewer LLM checks the CoT against the GT
   intermediate QAs and the final answer, labeling it correct / factual inconsistency / unfaithful
   reasoning / final-answer mismatch / hallucination.
4. **Variants**: `choice_generation` (open QA → A/B/C/D), `paraphrasing` (question paraphrase),
   `rebalance_options` (option-letter balancing), `mca_jsonl2swift` (strip metadata).
5. **Report** `analyse_cot_verification_result`: summarizes verification labels.

## Contents

All Stage-2 LLM annotation scripts (migrated from the old `api/` directory):

`api_interface.py` (LLM inference entry; credentials & proxy via environment variables),
`mvstride_cot_generation.py`, `mvstride_cot_verification.py`, `mvstride_cvd_test.py`,
`choice_generation.py`, `paraphrasing.py`, `cot_jsonl2swift.py`, `remove_system.py`,
`match_input_output.py`, `rebalance_options.py`, `mca_jsonl2swift.py`,
`analyse_cot_verification_result.py`.

> Scripts run as `python mvstride/llm/<script>.py` (sibling scripts import each other via
> `api_interface`). Every entry point supports `--help` for input/output arguments; defaults are
> neutralized placeholders or `./output/` relative paths (resolved from the repo root) — pass
> arguments explicitly for real runs.
