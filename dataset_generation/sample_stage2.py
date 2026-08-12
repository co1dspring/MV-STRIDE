import json
import random
from collections import defaultdict
from pathlib import Path


def clean_item(item):
    """
    仅保留核心字段，剔除导致 Schema 冲突的冗余元数据
    """
    # 核心字段：对话内容和图像路径
    new_item = {
        "messages": item.get("messages", []),
        "images": item.get("images", [])
    }

    # 可选：保留简单的描述性字段（这些通常不会引起 Arrow 报错）
    safe_keys = ["category", "data_source", "scene_name"]
    for key in safe_keys:
        if key in item:
            new_item[key] = item[key]

    return new_item


def sample_nested_raw(input_path):
    # 定义采样率
    RATES = [0.25, 0.5, 0.75, 1.0]
    input_p = Path(input_path)

    # 1. 读取数据 (兼容 json 和 jsonl)
    raw_data = []
    if input_p.suffix == '.jsonl':
        with open(input_p, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
    else:
        with open(input_p, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

    print(f"读取完成: {input_p.name} | 总数: {len(raw_data)}")

    # 2. 按类别归类
    category_map = defaultdict(list)
    for item in raw_data:
        cat = item.get('category', 'Unknown')
        category_map[cat].append(item)

    # 3. 核心步骤：每个类别内部仅进行一次性打乱，保证后续切片的包含关系
    for cat in category_map:
        random.shuffle(category_map[cat])

    # 4. 针对每个比例生成数据集
    for rate in RATES:
        final_list = []
        print(f"\n正在处理采样率: {rate:.2f}")

        for cat, items in category_map.items():
            # 计算切片位置
            target_num = int(len(items) * rate)
            if rate > 0 and len(items) > 0 and target_num == 0:
                target_num = 1

            # 顺序切片：保证 0.5 包含 0.25 的所有内容
            sampled_items = items[:target_num]

            # 清理数据结构 (移除 response_lst 等)
            cleaned_items = [clean_item(it) for it in sampled_items]
            final_list.extend(cleaned_items)

        # 5. 全局乱序 (打破类别聚集，但不会破坏嵌套包含关系)
        random.shuffle(final_list)

        # 6. 保存
        output_filename = f"{input_p.stem}_ratio_{rate:.2f}.json"
        output_p = input_p.parent / output_filename
        with open(output_p, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False, indent=4)

        print(f"已保存: {output_p.name} | 条数: {len(final_list)}")


if __name__ == "__main__":
    # 固定随机种子，保证可复现性
    random.seed(42)

    # 输入文件列表
    datasets = [
        '../api/output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel.json'
    ]

    for ds in datasets:
        if Path(ds).exists():
            sample_nested_raw(ds)
        else:
            print(f"错误: 找不到文件 {ds}")
