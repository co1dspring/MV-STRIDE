# configs

Runtime configs, **kept separate from source code** (`mvstride/` contains code only, no
data/environment-related config).

## Contents

- `qa/` — QA generation configs & templates (per data source; see [`qa/README.md`](qa/README.md)):
  - `qa_config_{infinigen,scannetpp}.json` (full configs)
  - `qa_config_{infinigen,scannetpp}_sparse.json` (early small-scale)
  - `qa_config_{infinigen,scannetpp}_ablation.json` (cross-view dependency ablation)
  - `qa_templates.json` (question/answer text templates for ~30 categories)
  - `qa_dependency_tree.json` (Level III → prerequisite Level I/II mapping)

> **Path convention**: relative paths inside the `qa/` configs resolve **from the repository root**
> (independent of the current working directory) — the generator
> (`mvstride/qa_generation/generate_QAs_multilevel_multistage.py`) auto-uses `<repo_root>` as the base
> when it detects the config under `<repo_root>/configs/qa/`; configs placed elsewhere are resolved
> relative to their own directory. See [`qa/README.md`](qa/README.md).
