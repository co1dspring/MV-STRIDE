# -*- coding = utf-8 -*-
import json
import re
from pathlib import Path
from collections import Counter


def analyze_verification_results(jsonl_path: str):
    path = Path(jsonl_path)
    if not path.exists():
        print(f"❌ 未找到结果文件: {path}")
        return

    total_count = 0
    correct_count = 0
    error_counter = Counter()
    unclassified_count = 0

    # 定义我们期待的标准标签集合
    standard_labels = {
        "Correct",
        "Factual inconsistency",
        "Reasoning unfaithfulness",
        "Final-answer inconsistency",
        "Hallucination"
    }

    # 用于提取 <answer> 标签内容的正则表达式
    answer_pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

    print(f"📊 开始解析文件: {path.name} ...")

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)

                # 1. 优先读取保存的 verification_result，若不存在则读取原始 response 文本作为备用
                raw_result = item.get("verification_result", "")
                if not raw_result and "response_lst" in item:
                    # 如果存盘的是原始 response，这里做个兼容支持
                    raw_result = item["response_lst"]

                raw_result = str(raw_result).strip()

                # 2. 从文本中提取 <answer> 内的内容
                ans_match = answer_pattern.search(raw_result)
                if ans_match:
                    result = ans_match.group(1).strip()
                else:
                    # 容错：如果找不到 <answer> 标签，尝试退回到整文本匹配
                    result = raw_result

                # 3. 基础文本清洗（剥离多余标点、前缀等干扰）
                result = result.replace(".", "").replace('"', '').replace("'", "").strip()
                result = re.sub(r"^Verdict:\s*", "", result, flags=re.IGNORECASE).strip()

                if not result:
                    unclassified_count += 1
                    continue

                total_count += 1

                # 4. 对齐到标准标签
                if result.lower() == "correct":
                    correct_count += 1
                elif result in standard_labels:
                    error_counter[result] += 1
                else:
                    # 处理可能存在微小格式或大小写不一致的问题
                    matched = False
                    for standard_label in standard_labels:
                        if result.lower() == standard_label.lower():
                            if standard_label == "Correct":
                                correct_count += 1
                            else:
                                error_counter[standard_label] += 1
                            matched = True
                            break

                    if not matched:
                        # 无法对齐时，归类为未知分类（并显示前30个字符便于排查）
                        error_counter[f"Unknown ({result[:30]})"] += 1

            except json.JSONDecodeError as e:
                print(f"⚠️ 警告: 第 {line_num} 行 JSON 解析失败，已跳过。错误: {e}")

    if total_count == 0:
        print("❌ 没有找到有效的统计数据！")
        return

    accuracy = (correct_count / total_count) * 100
    total_errors = total_count - correct_count

    # ==================== 📊 打印统计报告 ====================
    print("\n" + "=" * 50)
    print("       🎉 多视角空间推理 CoT 质量校验报告 🎉")
    print("=" * 50)
    print(f"📈 数据规模统计:")
    print(f"  - 已校验样本总量: {total_count} 条")
    print(f"  - 完美通过 (Correct): {correct_count} 条")
    print(f"  - 存在错误 (Failed) : {total_errors} 条")
    if unclassified_count > 0:
        print(f"  - 结果为空的样本    : {unclassified_count} 条 (未计入总量)")

    print("-" * 50)
    print(f"🎯 核心指标:")
    print(f"  - 💡 CoT 整体正确率 (Accuracy): {accuracy:.2f}%")
    print("-" * 50)

    print("🚨 错误维度分布 (Error Breakdown):")
    if total_errors == 0:
        print("  - 完美！未检测到任何推理错误。")
    else:
        # 按错误频次降序排列
        sorted_errors = error_counter.most_common()

        # 打印表格表头
        print(f"  {'错误维度 (Error Dimension)':<30} | {'频次':<6} | {'在总样本中占比':<14} | {'在错误中占比'}")
        print("  " + "-" * 75)

        for err_name, count in sorted_errors:
            pct_of_total = (count / total_count) * 100
            pct_of_errors = (count / total_errors) * 100
            print(f"  {err_name:<30} | {count:<6} | {pct_of_total:>12.2f}% | {pct_of_errors:>10.2f}%")

        print("\n📊 错误分布直方图 (Visual Distribution):")
        print("  " + "-" * 50)
        max_bar_length = 30  # 最大字符进度条长度
        for err_name, count in sorted_errors:
            # 计算进度条比例
            bar_length = int((count / total_errors) * max_bar_length)
            bar = "█" * bar_length + "░" * (max_bar_length - bar_length)
            pct_of_errors = (count / total_errors) * 100
            print(f"  {err_name:<26} {bar} {pct_of_errors:.1f}%")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    # 🟢 在这里替换为你实际保存的 jsonl 结果路径
    RESULT_FILE_PATH = "./output/cot_with_original_aligned_sampled_200_gpt-5.5_CoT.jsonl"

    analyze_verification_results(RESULT_FILE_PATH)
