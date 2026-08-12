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
    print(f"成功转换并保存到: {output_json_file}")


def convert_jsonl_to_json(input_path, output_path):
    data = read_jsonl(input_path)
    output_data = []

    # 定义需要注入的引导语
    system_prompt = '\nOutput your step-by-step thinking process in <think> </think> tags and the final choice (e.g., A: option) in <answer> </answer> tags.'

    for d in tqdm(data, desc="正在清理并转换数据"):
        # 深度复制原始数据，避免影响原对象
        output_d = d.copy()

        # 1. 注入引导语（如果原本没有）
        # if 'messages' in output_d and len(output_d['messages']) > 0:
        #     if system_prompt not in output_d['messages'][0]['content']:
        #         output_d['messages'][0]['content'] += system_prompt

        # 2. 移除冗余元数据，仅保留训练核心字段
        keys_to_remove = ['old_messages', 'response_lst', 'stats']
        for key in keys_to_remove:
            output_d.pop(key, None)

        output_data.append(output_d)

    # 保存为 JSON 格式
    if output_data:
        save_to_json(output_path, output_data)
    else:
        print("警告：输入数据为空，未生成文件。")


if __name__ == "__main__":
    # 输入输出路径
    # input_file = './output/Infinigen_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.jsonl'
    # output_file = './output/Infinigen_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.json'
    # input_file = './output/ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.jsonl'
    # output_file = './output/ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.json'
    input_file = './output/SAT_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.jsonl'
    output_file = './output/SAT_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.json'
    # input_file = './output/SPAR_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.jsonl'
    # output_file = './output/SPAR_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.json'

    # 执行转换
    convert_jsonl_to_json(input_file, output_file)
