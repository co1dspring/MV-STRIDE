import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Iterable

# Path configuration.
ORIGINAL_DATA_PATH = "/path/to/data/infinigen_20251031/QA_jsons_ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_rel.json"
COT_DATA_PATH = "./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00_no_system.json"
OUTPUT_MERGED_PATH = "./output/cot_with_original_aligned.json"


# JSON / JSONL unified loading helper.
def load_data(file_path: Union[str, Path]) -> List[Dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
    except json.JSONDecodeError:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse line {line_num} of {path}: {e}")
        return data


# Text cleaning and feature extraction.
def clean_question_text(text: str) -> str:
    """
    Strip the following noise from both raw and CoT data:
    1. Leading <image> markers
    2. Leading "Question <n>(Level <x>:<y>):" prefixes
    3. The trailing "Output your step-by-step thinking process..." instruction
    """
    text = text.strip()

    # 1. Strip leading <image> markers.
    text = re.sub(r"^(<image>)+", "", text).strip()

    # 2. Strip "Question X (Level Y: Z):" prefixes (and variants like "Question 1:").
    text = re.sub(r"^Question\s*\d+\s*\(.*?\):\s*", "", text, flags=re.IGNORECASE).strip()

    # 3. Strip the trailing CoT instruction template.
    template_pattern = r"\s*Output\s+your\s+step-by-step\s+thinking\s+process\s+in\s+ thinking.*$"
    text = re.sub(template_pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    return text


def normalize_text(text: str) -> str:
    """
    Normalize text: lowercase, drop all punctuation, collapse whitespace.
    Handles newlines, spaces, and full-width punctuation that otherwise
    prevent otherwise-identical strings from matching.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9一-龥\s]", " ", text)  # keep letters, digits, Chinese chars, spaces
    return "".join(text.split())


# Core precise data alignment.
def merge_datasets(original_data_path=ORIGINAL_DATA_PATH, cot_data_path=COT_DATA_PATH, output_merged_path=OUTPUT_MERGED_PATH):
    print("Loading datasets...")
    orig_data = load_data(original_data_path)
    cot_data = load_data(cot_data_path)
    print(f"Loaded: {len(orig_data)} original items, {len(cot_data)} CoT items.")

    # 1. Build a precise text-mapping index over the original data.
    orig_registry = {}

    for idx, orig_item in enumerate(orig_data):
        scene_name = orig_item.get("scene_name", "")

        # Use the last user question of the multi-turn data (the main question).
        messages = orig_item.get("messages", [])
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            continue

        raw_last_question = user_messages[-1]["content"]

        # Extract and normalize the core question text.
        cleaned_orig_q = clean_question_text(raw_last_question)
        norm_orig_q = normalize_text(cleaned_orig_q)

        # Composite key: scene name + normalized core question text.
        # Since the text and options differ per question within a scene, this
        # yields a unique (100%) deterministic key.
        key = f"{scene_name}_{norm_orig_q}"
        orig_registry[key] = orig_item

    print(f"Built high-precision text index: {len(orig_registry)} unique keys.")

    # 2. Match each CoT item 1:1.
    merged_dataset = []
    matched_count = 0
    unmatched_count = 0

    print("\nStarting strict text-hash alignment and field merging...")
    for cot_item in cot_data:
        scene_name = cot_item.get("scene_name", "")

        # Use the first user question of the CoT item.
        cot_messages = cot_item.get("messages", [])
        if not cot_messages:
            continue

        raw_cot_q = cot_messages[0]["content"]
        cleaned_cot_q = clean_question_text(raw_cot_q)
        norm_cot_q = normalize_text(cleaned_cot_q)

        lookup_key = f"{scene_name}_{norm_cot_q}"

        # Strong matching: only allow exact alignment on the core text.
        if lookup_key in orig_registry:
            matched_orig = orig_registry[lookup_key]
            matched_count += 1

            # Deep-copy to avoid reference pollution.
            merged_item = dict(cot_item)

            # Extract the intermediate multi-turn QA from the original data.
            orig_msgs = matched_orig.get("messages", [])
            intermediate_qa = []

            # Iterate over the multi-turn conversation in Q-A pairs.
            # e.g. with 6 turns (12 messages), the first 5 turns are intermediate facts.
            for i in range(0, len(orig_msgs), 2):
                q = orig_msgs[i]["content"]
                a = orig_msgs[i + 1]["content"]
                intermediate_qa.append({
                    "question": q,
                    "answer": a
                })

            # Merge the relevant fields.
            merged_item["original_intermediate_qa"] = intermediate_qa
            # Keep the original ground-truth answer for quick verification.
            raw_gt_answer = orig_msgs[-1]["content"]
            # Strip a leading "Answer X:" prefix (case-insensitive).
            clean_gt_answer = re.sub(r"^Answer\s*\d+:\s*", "", raw_gt_answer, flags=re.IGNORECASE).strip()
            merged_item["original_ground_truth_answer"] = clean_gt_answer
            merged_item["matched_original_scene_name"] = matched_orig.get("scene_name")
            merged_item["matched_original_category"] = matched_orig.get("category")

            merged_dataset.append(merged_item)
        else:
            unmatched_count += 1

    # 3. Write the result file.
    print("-" * 50)
    print("Alignment finished!")
    print(f"   - Exactly matched and merged: {matched_count} items")
    print(f"   - Failed to match: {unmatched_count} items")

    with open(output_merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged_dataset, f, indent=4, ensure_ascii=False)

    print(f"Data saved to: {output_merged_path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="把清洗后的 CoT 数据按 (scene_name, 归一化问题) 对齐回原始多轮 QA，并注入真值字段")
    parser.add_argument("--original-data", default=ORIGINAL_DATA_PATH, help="原始 stage2 QA 文件")
    parser.add_argument("--cot-data", default=COT_DATA_PATH, help="remove_system.py 的输出")
    parser.add_argument("--output-merged", default=OUTPUT_MERGED_PATH, help="对齐合并后的输出文件")
    args = parser.parse_args()

    merge_datasets(args.original_data, args.cot_data, args.output_merged)
