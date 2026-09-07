# -*- coding = utf-8 -*-
import json
import random
import re
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import copy


def extract_option(text):
    """
    Extract the letter A from strings like "(A) bus" or "A: option".
    """
    if not text: return ""
    match = re.search(r'([A-D])', text.upper())
    return match.group(1) if match else ""


def plot_distribution(accuracies, save_path):
    """
    Plot a histogram of the accuracy distribution.
    """
    # Count the frequency of each point 0.0, 0.1 ... 1.0
    counts = Counter(accuracies)
    x = [i / 10.0 for i in range(11)]  # generate [0.0, 0.1, ..., 1.0]
    y = [counts.get(rate, 0) for rate in x]

    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(rate) for rate in x], y, color='skyblue', edgecolor='black')

    # Add the actual counts above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom')

    plt.title('Pass Rate Distribution (10-shot Inference)')
    plt.xlabel('Accuracy (Pass Rate)')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path)
    print(f"分布图已保存至: {save_path}")
    plt.close()


def analyze_and_sample(json_path, jsonl_path, output_path):
    PROMPT = '\nOutput your step-by-step thinking process in <think> </think> tags and the final choice (e.g., A: option) in <answer> </answer> tags.'
    with open(json_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)

    inference_results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                inference_results.append(json.loads(line))

    num_train = len(training_data)
    num_inf = len(inference_results)

    if num_inf != num_train * 10:
        print(f"警告：数量不匹配！JSON={num_train}条, JSONL={num_inf}条")

    groups = {
        "zero": [], "hard": [], "medium": [], "easy_but": [], "perfect": []
    }
    all_accuracies = []

    print("正在分析正确率...")
    for i in range(num_train):
        batch = inference_results[i * 10: (i + 1) * 10]
        correct_count = 0
        for res in batch:
            pred = extract_option(res.get("response", ""))
            label = extract_option(res.get("labels", ""))
            if pred == label and pred != "":
                correct_count += 1

        accuracy = round(correct_count / 10.0, 1)  # force one decimal place
        all_accuracies.append(accuracy)

        item = training_data[i]

        # --- Added logic: extract the solution and modify messages ---
        # Back up
        original_item = copy.deepcopy(item)
        item['original'] = original_item
        # Add the thinking prompt
        # item["messages"][0]["content"] += PROMPT
        # Assume messages[1] in the original data is the model answer
        if len(item.get("messages", [])) > 1:
            # Extract content as the solution
            item["solution"] = item["messages"][1].get("content", "")
            # Keep only the first message (the User question)
            item["messages"] = [item["messages"][0]]
        # ----------------------------------------------

        item["pass_rate"] = accuracy

        if accuracy == 0:
            groups["zero"].append(item)
        elif 0 < accuracy <= 0.3:
            groups["hard"].append(item)
        elif 0.3 < accuracy < 0.7:
            groups["medium"].append(item)
        elif 0.7 <= accuracy < 1.0:
            groups["easy_but"].append(item)
        elif accuracy == 1.0:
            groups["perfect"].append(item)

    # Plot the distribution
    plot_distribution(all_accuracies, str(Path(output_path).with_suffix('.png')))

    print("\n" + "=" * 40)
    print("难度分布统计表")
    print("-" * 20)
    for k, v in groups.items():
        print(f"{k:<10}: {len(v):>6} 条")
    print("=" * 40)

    # Sampling logic
    sampled_data = []
    sampled_raw_data = []
    sampled_data.extend(groups["hard"])
    sampled_data.extend(groups["medium"])
    sampled_data.extend(groups["easy_but"])

    # if groups["medium"]:
    #     medium_sampled = random.sample(groups["medium"], int(len(groups["medium"]) * 0.5))
    #     sampled_data.extend(medium_sampled)
    #
    # if groups["easy_but"]:
    #     easy_sampled = random.sample(groups["easy_but"], int(len(groups["easy_but"]) * 0.3))
    #     sampled_data.extend(easy_sampled)

    random.shuffle(sampled_data)
    for item in sampled_data:
        sampled_raw_data.append(item['original'])
        del item['original']
    raw_output_path = str(Path(output_path).with_name(Path(output_path).stem + "_raw.json"))
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_raw_data, f, ensure_ascii=False, indent=4)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    print(f"\n采样完成！保留样本总数: {len(sampled_data)}")


if __name__ == "__main__":
    random.seed(42)

    # Replace with your actual filenames
    # input_json = "./api/output/Infinigen_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.json"
    input_json = "./api/output/ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.json"
    # input_jsonl = "./api/output/infinigen_stage3_x10.jsonl"
    input_jsonl = "./api/output/scannetpp_stage3_x10.jsonl"
    # output_json = "./api/output/Infinigen_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA_accsampled_nosystem.json"
    output_json = "./api/output/ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA_accsampled_nosystem.json"

    if Path(input_json).exists() and Path(input_jsonl).exists():
        analyze_and_sample(input_json, input_jsonl, output_json)
    else:
        print("错误：找不到输入文件，请确认路径。")
