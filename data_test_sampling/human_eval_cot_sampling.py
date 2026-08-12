import json
import random
from collections import defaultdict


def stratified_sample_json(input_file_path, output_file_path, target_total=200):
    # 1. 读取 JSON 文件
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件根节点必须是一个列表（List of Dicts）")

    total_input_count = len(data)
    if total_input_count < target_total:
        raise ValueError(f"原始数据只有 {total_input_count} 条，无法采样出 {target_total} 条数据！")

    # 2. 根据 category 的值进行分类归堆
    category_buckets = defaultdict(list)
    for item in data:
        # 如果某些数据没有 category 字段，归类为 "unknown"
        cat = item.get('category', 'unknown')
        category_buckets[cat].append(item)

    # 3. 计算每一类原本的比例，并计算理论采样数
    sampled_data = []
    category_sample_sizes = {}

    for cat, items in category_buckets.items():
        ratio = len(items) / total_input_count
        # 四舍五入计算该分类应采样的数量
        sample_size = round(ratio * target_total)
        # 确保采样数不超过该类别的实际拥有数量
        sample_size = min(sample_size, len(items))
        category_sample_sizes[cat] = sample_size

    # 4. 修正因四舍五入导致的“总数不刚好等于 target_total”的问题
    current_total = sum(category_sample_sizes.values())
    difference = target_total - current_total

    if difference != 0:
        # 按照类别拥有的数据量从大到小排序，优先调整大类
        sorted_categories = sorted(category_buckets.keys(), key=lambda k: len(category_buckets[k]), reverse=True)

        if difference > 0:
            # 需要补齐缺失的条数
            for _ in range(difference):
                for cat in sorted_categories:
                    # 只要该类别的总数还没被采空，就可以多采一条
                    if category_sample_sizes[cat] < len(category_buckets[cat]):
                        category_sample_sizes[cat] += 1
                        difference -= 1
                        break
                if difference == 0:
                    break
        elif difference < 0:
            # 需要扣除多余的条数
            for _ in range(abs(difference)):
                for cat in sorted_categories:
                    if category_sample_sizes[cat] > 0:
                        category_sample_sizes[cat] -= 1
                        difference += 1
                        break
                if difference == 0:
                    break

    # 5. 执行随机采样
    print("各类别采样分布情况：")
    for cat, size in category_sample_sizes.items():
        original_count = len(category_buckets[cat])
        print(f" - 类别 [{cat}]: 原始数量 {original_count} -> 采样数量 {size} (占比: {size / target_total:.2%})")

        # 从该类别的桶里随机抽取指定数量的数据
        sampled_items = random.sample(category_buckets[cat], size)
        sampled_data.extend(sampled_items)

    # 再次打乱最终列表，避免相同类别的数据堆叠在一起
    random.shuffle(sampled_data)

    # 6. 保存结果到新的 JSON 文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 成功！已从 {total_input_count} 条数据中等比例采样出 {len(sampled_data)} 条数据，并保存至: {output_file_path}")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 替换为你的实际文件路径
    input_json = "../api/output/cot_with_original_aligned.json"
    output_json = "./human_eval/cot_with_original_aligned_sampled_200.json"

    # 执行采样
    stratified_sample_json(input_json, output_json, target_total=200)
