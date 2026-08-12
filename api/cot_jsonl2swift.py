# -*- coding = utf-8 -*-
import json
import base64
import jsonlines
import cv2
import concurrent.futures
from api_interface import gpt4o_image_text_inference, gpt4o_text_inference
from datasets import Dataset
import os
import json
import copy
import concurrent.futures
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

# --- 补全 save_to_json 函数 ---
def save_to_json(output_json_file, data):
    with open(output_json_file, 'w', encoding='utf-8') as f:
        # indent=4 可以让生成的 JSON 文件带缩进，方便阅读
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"成功保存到: {output_json_file}")

# 创建新的jsonl文件，包含图像文件名、caption和生成的问题
def save_to_jsonl(output_jsonl_file, results):
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def try_repair_format(content):
    """
    尝试修复格式问题
    """
    content = content.strip()

    # --- 情况 A: 检查是否严重截断 ---
    # 如果内容不包含任何类似答案的标识(比如 'A:' 或 'Answer:')，通常是截断了
    if not re.search(r'[A-E]:\s*\w+', content) and "answer" not in content.lower():
        return None

        # --- 情况 B: 提取 Answer ---
    # 匹配 <answer>标签内内容 或 结尾处类似 A: xxx 的内容
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # 如果没有标签，尝试找最后出现的选项模式 (如 D: back)
        answer_parts = re.findall(r'([A-E]:\s*.*)', content)
        if answer_parts:
            answer_text = answer_parts[-1].strip()
        else:
            return None  # 连答案都找不到，没法修

    # --- 情况 C: 提取 Think ---
    # 移除已找到的 answer 部分，剩下的主要是 think 部分
    # 或者是提取 <think> 标签内容
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
    else:
        # 如果标签丢失，把 content 中 answer 之前的部分全部作为 think
        # 先把所有的 <think>, </think>, <answer>, </answer> 删掉
        clean_content = re.sub(r'</?(think|answer)>', '', content).strip()
        # 截取到答案模式之前
        think_text = re.split(r'[A-E]:', clean_content)[0].strip()

    # --- 情况 D: 嵌套处理 ---
    # 如果 think 依然包含了 answer 标签，进一步清洗
    think_text = re.sub(r'<answer>.*?</answer>', '', think_text, flags=re.DOTALL).strip()

    # 最终验证：如果 think 太空或者修复后依然没东西
    if not think_text or not answer_text:
        return None

    # 返回标准化的格式
    return f"<think>\n{think_text}\n</think>\n<answer>{answer_text}</answer>"

def process_and_filter_data(input_path, output_path, min_think_length=100):
    data = read_jsonl(input_path)
    output_data = []
    system_prompt = '\nOutput your step-by-step thinking process in <think> </think> tags and the final choice (e.g., A: option) in <answer> </answer> tags.'

    stats = {
        "total": len(data),
        "perfect": 0,    # 原本就完美的
        "repaired": 0,   # 修正成功的
        "discarded_format": 0,
        "discarded_short": 0,
        "kept": 0
    }

    pattern = re.compile(r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$', re.DOTALL)

    for d in tqdm(data, desc="审计并修复数据质量"):
        try:
            content = d['messages'][1]['content'].strip()
        except (KeyError, IndexError):
            stats["discarded_format"] += 1
            continue

        # 1. 检查是否完美符合格式
        match = pattern.search(content)
        final_content = None

        if match:
            stats["perfect"] += 1
            final_content = content
        else:
            # 2. 尝试修复
            repaired = try_repair_format(content)
            if repaired:
                stats["repaired"] += 1
                final_content = repaired
                print(repaired)
            else:
                stats["discarded_format"] += 1
                # print(content)
                continue

        # 3. 长度检查 (针对修复后的内容再次提取 think)
        think_part = re.search(r'<think>(.*?)</think>', final_content, re.DOTALL).group(1).strip()
        if len(think_part) < min_think_length:
            stats["discarded_short"] += 1
            continue

        # 4. 类别检查：去除不想要的类别
        if d['category'] == 'Positional Relationship(Obj.-Obj.)_Orientation':
            continue

        # 4. 保存合格数据
        output_d = d.copy()
        output_d['messages'][1]['content'] = final_content # 替换为修复后的内容
        output_d['messages'][0]['content'] = d['messages'][0]['content'] + system_prompt
        output_d.pop('old_messages', None)
        output_d.pop('response_lst', None)

        output_data.append(output_d)
        stats["kept"] += 1

    # 打印审计报告
    print("\n" + "=" * 40)
    print("数据质量审计与修复报告")
    print("-" * 20)
    print(f"总处理条数: {stats['total']}")
    print(f"完美命中:   {stats['perfect']}")
    print(f"修复成功:   {stats['repaired']}")
    print(f"丢弃 (格式不可救): {stats['discarded_format']}")
    print(f"丢弃 (Think过短):  {stats['discarded_short']}")
    print(f"最终保留:   {stats['kept']} ({(stats['kept'] / stats['total'] * 100):.2f}%)")
    print("=" * 40)

    if output_data:
        save_to_json(output_path, output_data)

# input_path = './output/ScannetppIphone_MultilevelCategories_20260121_sampled_MCA_Multistage_stage2_CoT.jsonl'
# output_path = './output/ScannetppIphone_MultilevelCategories_20260121_sampled_MCA_Multistage_stage2_CoT.json'
# input_path = './output/ScannetppIphone_MultilevelCategories_20260121_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT.jsonl'
# output_path = './output/ScannetppIphone_MultilevelCategories_20260121_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT.json'
# data = read_jsonl(input_path)
# output_data = []
# for d in data:
#     output_d = d.copy()
#     output_d.pop('old_messages', None)  # 如果不存在这个 key，也不会报错
#     output_d.pop('response_lst', None)
#     output_data.append(output_d)
# save_to_json(output_path, output_data)
# --- 执行脚本 ---
if __name__ == "__main__":
    # input_file = './output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT.jsonl'
    # output_file = './output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned.json'
    input_file = './output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT.jsonl'
    output_file = './output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned.json'

    # 设置你认为“太短”的标准，比如少于 100 个字符
    process_and_filter_data(input_file, output_file, min_think_length=100)
