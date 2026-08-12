# -*- coding = utf-8 -*-
import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd

# =========================================================
# ⚙️ 路径配置 (仅需配置你的结果文件路径)
# =========================================================
RESULT_PATH = "../api/output/cross_view_dependency_sampled_multiview_1000_gemini-3-flash-preview_CoT.jsonl"
RESULT_PATH = "../api/output/cross_view_dependency_sampled_singleview_1000_gemini-3-flash-preview_CoT.jsonl"


# =========================================================
# JSON / JSONL IO
# =========================================================

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


def write_json(data, file_path, indent=4):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':

    result_path = Path(RESULT_PATH)
    save_path = result_path.parent / (result_path.stem + "_metrics.json")

    # =====================================================
    # 1. 加载并统计数据
    # =====================================================
    results = list(read_jsonl(result_path))

    # 自动收集所有类别
    categories = sorted(list(set(item.get("category", "Unknown") for item in results)))

    stats = {}
    for cat in categories + ["All"]:
        stats[cat] = {
            "sum": 0,
            "true": 0,
            "false": 0,
            "acc": 0.0
        }

    # 用于基线计算的标签分布统计
    label_counter = Counter()

    # =====================================================
    # 2. 指标计算 (完全自闭环)
    # =====================================================
    for idx, item in enumerate(results, 1):
        category = item.get("category", "Unknown")

        # 🚀 a. 提取并清洗预测值 (model_answer)
        raw_pred = str(item.get("model_answer", "")).strip().upper()
        pred_label = raw_pred[:1]  # 提取首字母，例如 "D"

        # 🚀 b. 从 messages 历史中提取真实答案 (GT)
        # 结构：messages[1]['content'] 对应标准答案 (如 "D: 2")
        messages = item.get("messages", [])
        if len(messages) >= 2 and messages[1].get("role") == "assistant":
            raw_gt = str(messages[1].get("content", "")).strip().upper()
        else:
            # 兼容性兜底：万一格式不对，尝试读取 labels 字段
            raw_gt = str(item.get("labels", "")).strip().upper()

        true_label = raw_gt[:1]  # 提取首字母，例如 "D"

        if not true_label:
            print(f"⚠️ Warning: Line {idx} has empty GT answer. Skipped.")
            continue

        # 记录真实标签分布
        label_counter[true_label] += 1

        stats[category]["sum"] += 1
        stats["All"]["sum"] += 1

        # 🚀 c. 判定对错
        if pred_label == true_label:
            stats[category]["true"] += 1
            stats["All"]["true"] += 1
        else:
            stats[category]["false"] += 1
            stats["All"]["false"] += 1

    # =====================================================
    # 3. 计算 Accuracy
    # =====================================================
    for cat in stats:
        if stats[cat]["sum"] > 0:
            stats[cat]["acc"] = round(stats[cat]["true"] / stats[cat]["sum"], 4)
        else:
            stats[cat]["acc"] = 0.0

    # =====================================================
    # 4. 计算 Baselines
    # =====================================================
    total_labels = sum(label_counter.values())
    majority_ratio = (max(label_counter.values()) / total_labels) if total_labels > 0 else 0.0
    random_ratio = 0.25

    stats["Random_Baseline"] = {
        "sum": "-", "true": "-", "false": "-", "acc": round(random_ratio, 4)
    }
    stats["Majority_Baseline"] = {
        "sum": "-", "true": "-", "false": "-", "acc": round(majority_ratio, 4)
    }

    # =====================================================
    # 5. 保存并打印结果
    # =====================================================
    write_json(stats, save_path)

    print("\n" + "=" * 65)
    print("🎯 Metric Calculation Finished (Self-Contained Direct Mode)")
    print(f"\nResult Path: {result_path}")
    print(f"Save Path:   {save_path}")
    print("-" * 65)

    # 漂亮的控制台输出
    df = pd.DataFrame.from_dict(stats, orient='index')
    df = df[['sum', 'true', 'false', 'acc']]
    print(df)
    print("=" * 65 + "\n")
