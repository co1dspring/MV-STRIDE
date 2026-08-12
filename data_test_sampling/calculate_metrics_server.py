import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Union, Iterable
from collections import Counter
import pandas as pd


# =========================================================
# JSON / JSONL IO
# =========================================================

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")


def write_json(data, file_path, indent=4):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_num}: {e}")


# =========================================================
# Answer Extraction
# =========================================================

def extract_answer(text: str) -> str:
    # 步骤 1: 尝试标准标签
    answer_match = re.search(
        r'<answer>(.*?)</answer>',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip()

    # 步骤 2: 尝试 LaTeX 格式
    boxed_match = re.search(
        r'\\boxed\{([A-G])\}',
        text,
        re.IGNORECASE
    )
    if boxed_match:
        return boxed_match.group(1).upper()

    # 步骤 3: 彻底移除 <think>...</think> 块
    clean_text = re.sub(
        r'<think>.*?</think>',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # 步骤 4: 找最后出现的独立字母
    pattern = r'\b([A-G])\b'
    choices = re.findall(pattern, clean_text.upper())
    if choices:
        return choices[-1]

    # 步骤 5: 宽松正则匹配
    loose_choices = re.findall(
        r'([A-G])(?![A-Za-z0-9])',
        clean_text.upper()
    )
    if loose_choices:
        return loose_choices[-1]

    return ""


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate Metrics with Label-Cleaning")

    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="model prediction jsonl path"
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="validation set json path"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "cot"],
        default="direct"
    )

    args = parser.parse_args()

    result_path = Path(args.result_path)
    metadata_path = Path(args.metadata_path)

    save_path = result_path.parent / (
            result_path.stem + "_metrics.json"
    )

    # =====================================================
    # Load Data
    # =====================================================
    results = list(read_jsonl(result_path))
    metadata = read_json(metadata_path)

    # 容错：防止推理被意外中断导致两边长度不一致
    if len(results) != len(metadata):
        print(f"⚠️ Warning: Size mismatch! results({len(results)}) vs metadata({len(metadata)}).")
        print("Using the minimum intersection length for evaluation.")
        min_len = min(len(results), len(metadata))
        results = results[:min_len]
        metadata = metadata[:min_len]

    # =====================================================
    # Auto Collect Categories
    # =====================================================
    categories = sorted(
        list(set(item.get("category", "Unknown") for item in metadata))
    )

    stats = {}
    for cat in categories + ["All"]:
        stats[cat] = {
            "sum": 0,
            "true": 0,
            "false": 0,
            "acc": 0.0
        }

    # =====================================================
    # Label Distribution
    # =====================================================
    label_counter = Counter()
    for meta in metadata:
        raw_gt = str(meta.get("answer", meta.get("labels", ""))).strip().upper()
        # 清洗标签：将 "B: 1" 统一规范化为首字母 "B" 用于分布统计
        gt = raw_gt[:1]
        if gt:
            label_counter[gt] += 1

    total_labels = sum(label_counter.values())
    majority_ratio = (max(label_counter.values()) / total_labels) if total_labels > 0 else 0.0
    random_ratio = 0.25

    # =====================================================
    # Metric Calculation
    # =====================================================
    for res, meta in zip(results, metadata):
        category = meta.get("category", "Unknown")

        stats[category]["sum"] += 1
        stats["All"]["sum"] += 1

        response_text = str(res.get("response", ""))

        if args.mode == "cot":
            processed_response = extract_answer(response_text)
        else:
            processed_response = response_text.strip()

        # 1. 提取预测标签（取首字母并大写，如 "B"）
        pred_label = processed_response[:1].upper()

        # 2. 提取并清洗真实标签（兼容 labels/answer 字段，并统一截取首字母，如 "B: 1" -> "B"）
        raw_true_label = str(meta.get("answer", res.get("labels", ""))).strip().upper()
        true_label = raw_true_label[:1]

        # 3. 进行精准比对
        if pred_label == true_label and pred_label != "":
            stats[category]["true"] += 1
            stats["All"]["true"] += 1
        else:
            stats[category]["false"] += 1
            stats["All"]["false"] += 1

    # =====================================================
    # Accuracy Calculation
    # =====================================================
    for cat in stats:
        if stats[cat]["sum"] > 0:
            stats[cat]["acc"] = round(stats[cat]["true"] / stats[cat]["sum"], 4)
        else:
            stats[cat]["acc"] = 0.0

    # =====================================================
    # Extra Baselines
    # =====================================================
    stats["Random_Baseline"] = {
        "sum": "-", "true": "-", "false": "-", "acc": round(random_ratio, 4)
    }
    stats["Majority_Baseline"] = {
        "sum": "-", "true": "-", "false": "-", "acc": round(majority_ratio, 4)
    }

    # =====================================================
    # Save & Print
    # =====================================================
    write_json(stats, save_path)

    print("\n" + "=" * 60)
    print("Metric Calculation Finished")
    print(f"\nResult Path:   {result_path}")
    print(f"Metadata Path: {metadata_path}")
    print(f"Save Path:     {save_path}")
    print("-" * 60)

    # 使用 Pandas 漂亮地展示结果并对齐字段顺序
    df = pd.DataFrame.from_dict(stats, orient='index')
    df = df[['sum', 'true', 'false', 'acc']]
    print(df)
    print("=" * 60 + "\n")
