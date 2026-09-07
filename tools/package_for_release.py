#!/usr/bin/env python3
"""Package QA outputs + images into a self-contained dataset dir for release.

No image bytes are copied: files are hard-linked (same filesystem) into the
release tree, and every `images` entry in the QA json files is rewritten from
machine-absolute to dataset-relative ("images/<source>/...").

Usage (run from the repo root; one or more --config, repeatable):

    python tools/package_for_release.py \
        --config ./configs/qa/qa_config_infinigen.json \
        --config ./configs/qa/qa_config_scannetpp.json \
        --out ./data/release/MV-STRIDE
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def resolve_base(config_path: Path) -> Path:
    """Mirror the QA generator: relative config paths resolve from the repo root
    when the config lives under <root>/configs/qa/, else from its own directory."""
    cfg_dir = config_path.resolve().parent
    if cfg_dir.name == "qa" and cfg_dir.parent.name == "configs":
        return cfg_dir.parent.parent
    return cfg_dir


def load_config(config_path: Path):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    base = resolve_base(config_path)
    version = cfg["version_name"]
    out_rel = cfg.get("output_dir", "").format(VERSION_NAME=version)
    out_path = Path(out_rel)
    out_dir = out_path if out_path.is_absolute() else (base / out_path).resolve()
    return cfg, out_dir


def longest_common_prefix_paths(paths):
    if not paths:
        return ""
    common = os.path.commonpath(paths)
    return common


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True, help="qa_config_*.json (repeatable)")
    ap.add_argument("--out", default="./data/release/MV-STRIDE")
    ap.add_argument("--no-images", action="store_true", help="only rewrite json, skip image mirror")
    args = ap.parse_args()

    release = Path(args.out)
    annotation_root = release / "annotations"
    image_root = release / "images"
    config_root = release / "configs"
    (release / "configs").mkdir(parents=True, exist_ok=True)

    summary = []
    all_stats = []

    for config_path in map(Path, args.config):
        cfg, out_dir = load_config(config_path)
        version = cfg["version_name"]
        source = cfg["data_source"]
        print(f"\n=== {source}: {version}")

        json_files = [
            out_dir / "atomic" / f"{version}_atomic.json",
            out_dir / f"{version}_stage1.json",
            out_dir / f"{version}_stage3.json",
        ]
        qa_config_src = out_dir / "qa_config.json"
        if qa_config_src.exists():
            shutil.copy2(qa_config_src, config_root / f"qa_config_{source}.json")

        # Compute the common absolute prefix over all image paths of this source,
        # so rewriting does not depend on the config text.
        all_images = []
        for jf in json_files:
            if not jf.exists():
                print(f"  [skip] missing {jf.name}")
                continue
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            for s in data:
                all_images.extend(s.get("images", []))
        if not all_images:
            print("  [warn] no image entries found")
            continue
        prefix = longest_common_prefix_paths(all_images)
        prefix_dir = Path(prefix)
        if not prefix_dir.is_dir():
            prefix_dir = prefix_dir.parent
        prefix = str(prefix_dir)
        print(f"  image root: {prefix}")
        suffix = lambda p: p[len(prefix):].lstrip("/")

        # Rewrite image paths per file and save under annotations/<source>/.
        out_source = annotation_root / source
        out_source.mkdir(parents=True, exist_ok=True)
        per_file = []
        for jf in json_files:
            if not jf.exists():
                continue
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            for s in data:
                s["images"] = [f"images/{source}/{suffix(p)}" for p in s.get("images", [])]
            dst = out_source / jf.name
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            per_file.append((jf.name, len(data)))
            print(f"  wrote {dst.relative_to(release)}  ({len(data)} samples)")

        # Mirror the whole image/metadata tree with hard links (no copy).
        n_img = 0
        if not args.no_images:
            src_root = Path(prefix)
            dst_root = image_root / source
            dst_root.mkdir(parents=True, exist_ok=True)
            for dirpath, _dirnames, filenames in os.walk(src_root):
                rel = os.path.relpath(dirpath, src_root)
                target_dir = dst_root / rel if rel != "." else dst_root
                target_dir.mkdir(parents=True, exist_ok=True)
                for fn in filenames:
                    s = Path(dirpath) / fn
                    d = target_dir / fn
                    try:
                        os.link(s, d)
                    except FileExistsError:
                        pass
                    n_img += 1
            print(f"  linked {n_img} files under {image_root / source}")

        # Validate: every rewritten image entry must exist in the release tree.
        missing = 0
        total_refs = 0
        for fname, _n in per_file:
            with open(out_source / fname, encoding="utf-8") as f:
                data = json.load(f)
            for s in data:
                for p in s.get("images", []):
                    total_refs += 1
                    if not (release / p).exists():
                        missing += 1
        print(f"  validation: {total_refs} image refs, {missing} missing")
        all_stats.append((source, per_file, total_refs, missing))

    # Dataset card (built from real run numbers in all_stats).
    role = {"atomic": "full set, all levels, single-turn", "stage1": "SFT split (stage-1 scene pool)",
            "stage3": "Level III only, RL split (stage-3 scene pool)"}
    total_samples = sum(n for _, per_file, _, _ in all_stats
                        for _, n in per_file if _.endswith("_atomic.json"))
    lines = [
        "# MV-STRIDE — Multi-View Spatial Reasoning QA Dataset",
        "",
        "Hierarchical (Level I/II/III) multi-view spatial-reasoning QA data generated by the "
        "[MV-STRIDE](https://github.com/co1dspring/MV-STRIDE) pipeline from two scene sources: "
        "**Infinigen** (synthetic, procedurally generated) and **ScanNet++** (real scanned scenes). "
        "Every sample asks a multiple-choice spatial question about 2–3 images of the same scene "
        "taken from different camera viewpoints.",
        "",
        "## Layout",
        "",
        "- `annotations/<source>/<version>_{atomic,stage1,stage3}.json` — QA samples; each sample:",
        "  `messages` (user: `<image><image>Question + Options`, assistant: answer), `images` "
        "(**relative to this dataset repo**, `images/<source>/...`), `category`, `scene_name`, `data_source`.",
        "- `images/<source>/...` — the images referenced by the annotations (per-scene dirs mirror the "
        "source layout; Infinigen scenes also carry `scene_metadata.json`, ScanNet++ scenes "
        "`scene_metadata_new.json`).",
        "- `configs/` — the exact `qa_config_*.json` used to generate each split.",
        "",
        "## Files",
        "",
        "| source | scenes | file | samples | role |",
        "|---|---|---|---|---|",
    ]
    for source, per_file, _refs, _miss in all_stats:
        img_src_dir = image_root / source
        n_scenes = (len([d for d in img_src_dir.iterdir() if d.is_dir()])
                    if img_src_dir.is_dir() else "-")
        for fname, n in per_file:
            ftype = fname.rsplit("_", 1)[-1].replace(".json", "")
            lines.append(f"| {source} | {n_scenes} | `{fname}` | {n:,} | {role.get(ftype, ftype)} |")
    lines += [
        "",
        f"Total: **{total_samples:,}** single-turn QA samples "
        "(Level I perception + Level II cross-view + Level III reasoning incl. MSR).",
        "",
        "Load with the `datasets` library:",
        "",
        "```python",
        "from datasets import load_dataset",
        "ds = load_dataset(\"json\", data_files=\"annotations/*/*_stage1.json\")",
        "# images are paths relative to this repo root; join before reading",
        "```",
        "",
        "## License & provenance",
        "",
        "- QA texts/annotations and Infinigen-derived content: generated by the MV-STRIDE pipeline "
        "(cite below); Infinigen is MIT-licensed.",
        "- ScanNet++ images/frames are derived from the [ScanNet++ dataset]"
        "(https://scannetpp.cs.uni-bonn.de); check and respect its license terms before "
        "commercial use / redistribution.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@inproceedings{xu2026mvstride,",
        "  title     = {MV-STRIDE: Enabling MLLMs to Master Multi-View Spatial Reasoning via Hierarchical Capability Modeling},",
        "  author    = {Xu, Jin and Huang, Xiaojian and Luo, Zhuodong and Zhang, Zhihong and Liu, Xin and Wei, Jiansheng and Wang, Xinzhi and Zhao, Jie and Chen, Xuejin},",
        "  booktitle = {European Conference on Computer Vision},",
        "  year      = {2026}",
        "}",
        "```",
        "",
    ]
    (release / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nrelease dir: {release}")
    for source, per_file, refs, missing in all_stats:
        print(f"  {source}: {per_file} | refs={refs} missing={missing}")
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
