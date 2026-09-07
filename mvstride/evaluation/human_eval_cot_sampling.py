import json
import random
from collections import defaultdict


def stratified_sample_json(input_file_path, output_file_path, target_total=200):
    # 1. Read the JSON file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件根节点必须是一个列表（List of Dicts）")

    total_input_count = len(data)
    if total_input_count < target_total:
        raise ValueError(f"原始数据只有 {total_input_count} 条，无法采样出 {target_total} 条数据！")

    # 2. Bucket the data by the category value
    category_buckets = defaultdict(list)
    for item in data:
        # If some data has no category field, put it in "unknown"
        cat = item.get('category', 'unknown')
        category_buckets[cat].append(item)

    # 3. Compute each category original ratio and the theoretical sample size
    sampled_data = []
    category_sample_sizes = {}

    for cat, items in category_buckets.items():
        ratio = len(items) / total_input_count
        # Round to compute how many to sample for this category
        sample_size = round(ratio * target_total)
        # Make sure the sample size does not exceed the category actual count
        sample_size = min(sample_size, len(items))
        category_sample_sizes[cat] = sample_size

    # 4. Fix the "total not exactly equal to target_total" problem caused by rounding
    current_total = sum(category_sample_sizes.values())
    difference = target_total - current_total

    if difference != 0:
        # Sort categories by data size descending, adjusting larger ones first
        sorted_categories = sorted(category_buckets.keys(), key=lambda k: len(category_buckets[k]), reverse=True)

        if difference > 0:
            # Need to fill in the missing count
            for _ in range(difference):
                for cat in sorted_categories:
                    # As long as the category total is not exhausted, sample one more
                    if category_sample_sizes[cat] < len(category_buckets[cat]):
                        category_sample_sizes[cat] += 1
                        difference -= 1
                        break
                if difference == 0:
                    break
        elif difference < 0:
            # Need to remove the extra count
            for _ in range(abs(difference)):
                for cat in sorted_categories:
                    if category_sample_sizes[cat] > 0:
                        category_sample_sizes[cat] -= 1
                        difference += 1
                        break
                if difference == 0:
                    break

    # 5. Perform the random sampling
    print("各类别采样分布情况：")
    for cat, size in category_sample_sizes.items():
        original_count = len(category_buckets[cat])
        print(f" - 类别 [{cat}]: 原始数量 {original_count} -> 采样数量 {size} (占比: {size / target_total:.2%})")

        # Randomly draw the specified number from this category bucket
        sampled_items = random.sample(category_buckets[cat], size)
        sampled_data.extend(sampled_items)

    # Shuffle the final list again so same-category data are not stacked
    random.shuffle(sampled_data)

    # 6. Save the result to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 成功！已从 {total_input_count} 条数据中等比例采样出 {len(sampled_data)} 条数据，并保存至: {output_file_path}")


# ==================== Usage example ====================
if __name__ == "__main__":
    # Replace with your actual file paths
    # 输入 = mvstride/llm/match_input_output.py 的默认输出（./output/cot_with_original_aligned.json）
    input_json = "./output/cot_with_original_aligned.json"
    output_json = "./human_eval/cot_with_original_aligned_sampled_200.json"

    # Run the sampling
    stratified_sample_json(input_json, output_json, target_total=200)
