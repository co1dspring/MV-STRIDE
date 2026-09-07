# -*- coding = utf-8 -*-
import json
import jsonlines
from pathlib import Path
from tqdm import tqdm


def read_jsonl(jsonl_file):
    annotations = []
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader:
            annotations.append(obj)
    return annotations


def save_to_json(output_json_file, data):
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Converted and saved to: {output_json_file}")


def convert_jsonl_to_json(input_path, output_path):
    data = read_jsonl(input_path)
    output_data = []

    # The instruction to inject.
    system_prompt = '\nOutput your step-by-step thinking process in  thinking  response tags and the final choice (e.g., A: option) in <answer> </answer> tags.'

    for d in tqdm(data, desc="Cleaning and converting data"):
        # Deep-copy the item so the original object is not mutated.
        output_d = d.copy()

        # Remove redundant metadata, keeping only the core training fields.
        keys_to_remove = ['old_messages', 'response_lst', 'stats']
        for key in keys_to_remove:
            output_d.pop(key, None)

        output_data.append(output_d)

    # Save as JSON.
    if output_data:
        save_to_json(output_path, output_data)
    else:
        print("Warning: input data is empty, no file generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="去掉 MCA 冗余元数据 → 训练用 json")
    parser.add_argument("--input-file",
                        default='./output/SAT_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.jsonl',
                        help="rebalance_options.py 的输出")
    parser.add_argument("--output-file",
                        default='./output/SAT_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.json',
                        help="清洗后的输出文件")
    args = parser.parse_args()

    # Run the conversion.
    convert_jsonl_to_json(args.input_file, args.output_file)
