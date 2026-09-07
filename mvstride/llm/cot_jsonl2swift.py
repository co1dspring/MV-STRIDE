# -*- coding = utf-8 -*-
import json
import base64
import jsonlines
import cv2
import concurrent.futures
from api_interface import gpt4o_image_text_inference, gpt4o_text_inference
from datasets import Dataset
import os
import copy
from pathlib import Path
from tqdm import tqdm
import time
import threading
import random
import re

def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_jsonl(jsonl_file):
    annotations = []
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader:
            annotations.append(obj)
    return annotations

def save_to_json(output_json_file, data):
    with open(output_json_file, 'w', encoding='utf-8') as f:
        # indent=4 keeps the JSON readable.
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved to: {output_json_file}")

# Write a new jsonl file with the image filename, caption, and generated question.
def save_to_jsonl(output_jsonl_file, results):
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def try_repair_format(content):
    """
    Attempt to repair formatting issues in the CoT response.
    """
    content = content.strip()

    # Case A: reject if seriously truncated (no answer-like marker present).
    if not re.search(r'[A-E]:\s*\w+', content) and "answer" not in content.lower():
        return None

    # Case B: extract the answer.
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # No tags: fall back to the last option pattern (e.g. "D: ...").
        answer_parts = re.findall(r'([A-E]:\s*.*)', content)
        if answer_parts:
            answer_text = answer_parts[-1].strip()
        else:
            return None

    # Case C: extract the thinking part.
    think_match = re.search(r' thinking(.*?) response', content, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
    else:
        # Tags missing: treat everything before the answer as the thinking part.
        clean_content = re.sub(r'</?(think|answer)>', '', content).strip()
        think_text = re.split(r'[A-E]:', clean_content)[0].strip()

    # Case D: if the thinking part still contains an answer tag, strip it.
    think_text = re.sub(r'<answer>.*?</answer>', '', think_text, flags=re.DOTALL).strip()

    # Final validation: nothing usable left.
    if not think_text or not answer_text:
        return None

    # Return the normalized format.
    return f" thinking\n{think_text}\n response\n<answer>{answer_text}</answer>"

def process_and_filter_data(input_path, output_path, min_think_length=100):
    data = read_jsonl(input_path)
    output_data = []
    system_prompt = '\nOutput your step-by-step thinking process in  thinking  response tags and the final choice (e.g., A: option) in <answer> </answer> tags.'

    stats = {
        "total": len(data),
        "perfect": 0,
        "repaired": 0,
        "discarded_format": 0,
        "discarded_short": 0,
        "kept": 0
    }

    pattern = re.compile(r'^ thinking(.*?) response\s*<answer>(.*?)</answer>$', re.DOTALL)

    for d in tqdm(data, desc="Auditing and repairing data quality"):
        try:
            content = d['messages'][1]['content'].strip()
        except (KeyError, IndexError):
            stats["discarded_format"] += 1
            continue

        # 1. Check whether the content already matches the expected format.
        match = pattern.search(content)
        final_content = None

        if match:
            stats["perfect"] += 1
            final_content = content
        else:
            # 2. Try to repair it.
            repaired = try_repair_format(content)
            if repaired:
                stats["repaired"] += 1
                final_content = repaired
                print(repaired)
            else:
                stats["discarded_format"] += 1
                continue

        # 3. Length check on the thinking part.
        think_part = re.search(r' thinking(.*?) response', final_content, re.DOTALL).group(1).strip()
        if len(think_part) < min_think_length:
            stats["discarded_short"] += 1
            continue

        # 4. Drop the undesirable category.
        if d['category'] == 'Positional Relationship(Obj.-Obj.)_Orientation':
            continue

        # 5. Keep the qualified item.
        output_d = d.copy()
        output_d['messages'][1]['content'] = final_content
        output_d['messages'][0]['content'] = d['messages'][0]['content'] + system_prompt
        output_d.pop('old_messages', None)
        output_d.pop('response_lst', None)

        output_data.append(output_d)
        stats["kept"] += 1

    # Print the audit report.
    print("\n" + "=" * 40)
    print("Data quality audit and repair report")
    print("-" * 20)
    print(f"Total processed: {stats['total']}")
    print(f"Perfect match:   {stats['perfect']}")
    print(f"Repaired:        {stats['repaired']}")
    print(f"Discarded (bad format): {stats['discarded_format']}")
    print(f"Discarded (short think): {stats['discarded_short']}")
    print(f"Kept:            {stats['kept']} ({(stats['kept'] / stats['total'] * 100):.2f}%)")
    print("=" * 40)

    if output_data:
        save_to_json(output_path, output_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="修复 <think>/<answer> 标签、过滤过短 CoT 并输出清洗 JSON")
    parser.add_argument("--input-file",
                        default='./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT.jsonl',
                        help="mvstride_cot_generation.py 的 CoT 输出")
    parser.add_argument("--output-file",
                        default='./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned.json',
                        help="清洗后的输出文件")
    args = parser.parse_args()

    # Configure the minimum acceptable thinking length (e.g. 100 chars).
    process_and_filter_data(args.input_file, args.output_file, min_think_length=100)
