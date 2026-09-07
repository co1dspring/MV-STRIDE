# data

**Local reproduction data root** (input & output of the QA generation flow). Contents here are
**not committed to GitHub** — the QA datasets (json + images below) are eventually released to
HuggingFace.

> Path convention: relative paths inside `configs/qa/qa_config_*.json` resolve **from the repository
> root**, so paths are written uniformly as `data/...` here (detailed rules in
> `configs/qa/README.md`). Once the data is in place, QA generation runs without any code/config
> changes.

## Directory Layout

| Path | Contents | Status |
|---|---|---|
| `infinigen/saved_scenes/` | **Infinigen QA input**: per scene `<scene>/{scene_metadata.json, Image_*.png}` (annotations + 10 rendered images) | in place (a symlink to an existing data dir on this machine) |
| `scannetpp/scannetpp_sampled_modified/` | **ScanNet++ QA input**: per scene `<scene>_iphone/{scene_metadata_new.json, images/frame_*.jpg}` (unified schema) | produced by the conversion command below |
| `QA_jsons_{VERSION_NAME}/` | **QA generation output** (naming chain: [`configs/qa/README.md`](../configs/qa/README.md)): `atomic/level_{1,2,3}/`, `*_stage1/2/3.json`, `qa_config.json`, etc. | created by the generator |

## How Each Data Source Gets Its QA Input

### Infinigen

On the Infinigen side, annotations are already extracted by `save_scene_annotation.py`
(`mvstride/scene_processing/infinigen/`) after scene generation: just place its output directory at
`data/infinigen/saved_scenes/` (currently a symlink on this machine, no copying needed).

### ScanNet++ (needs one format-conversion step)

ScanNet++'s official projection outputs `scene_metadata.json` (one record per frame; the raw ScanNet++
data lives outside this repo, e.g. under `<scannetpp_root>/scannetpp_sampled_new/` for annotations and
`<scannetpp_root>/scannetpp_sampled/` for frame images) are **not** the direct input to QA generation;
first run the unified-schema conversion (`scannetpp2infinigen_new.py`), pointing its output straight at
this directory:

```bash
# Run from the repo root; fill in --raw-root/--anno-root with your actual ScanNet++ data locations
python mvstride/scene_processing/scannetpp/scannetpp2infinigen_new.py \
  --raw-root /path/to/scannetpp_sampled \
  --anno-root /path/to/scannetpp_sampled_new \
  --output-root data/scannetpp/scannetpp_sampled_modified
```

Conversion details and the `scene_metadata_new.json` unified-schema description:
[`third_party/scannetpp/README.md`](../third_party/scannetpp/README.md).

## Running QA Generation

```bash
# Run from the repo root; --config selects the data-source config
# (idempotent: the output dir is determined by version_name, runs never overwrite each other)
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py \
  --config ./configs/qa/qa_config_infinigen.json
python mvstride/qa_generation/generate_QAs_multilevel_multistage.py \
  --config ./configs/qa/qa_config_scannetpp.json
```

- The generator resolves relative paths in configs against the repository root (derived from
  `configs/qa/`), so inputs point into this directory and outputs are written to
  `data/QA_jsons_{VERSION_NAME}/` (re-running the same config overwrites the same output directory).
- The `images` field in the output json points to real images under the absolute path resolved from
  `training_environment_base_dir`.

## HuggingFace Release (planned)

- **Images**: `infinigen/saved_scenes/` (Infinigen renders) and `scannetpp/scannetpp_sampled_modified/`
  (iPhone sampled frames).
- **Annotations**: the json files under `QA_jsons_*/` (question/answer text + image path references).
- At release time the content will be reorganized into an HF dataset layout (e.g.
  `images/<scene>/...` + `annotations/*.json`) and image path prefixes in the json will be rewritten
  (currently machine-dependent absolute paths); packaging scripts will be added then.
