import json
import random
from collections import defaultdict
from pathlib import Path


def clean_item(item):
    """
    Keep only the core fields and drop redundant metadata that causes schema conflicts.
    """
    # core fields: conversation content and image paths
    new_item = {
        "messages": item.get("messages", []),
        "images": item.get("images", [])
    }

    # optional: keep simple descriptive fields (these usually do not cause Arrow errors)
    safe_keys = ["category", "data_source", "scene_name"]
    for key in safe_keys:
        if key in item:
            new_item[key] = item[key]

    return new_item


def sample_nested_raw(input_path):
    # define the sampling rates
    RATES = [0.25, 0.5, 0.75, 1.0]
    input_p = Path(input_path)

    # 1. Read the data (supports both json and jsonl)
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

    # 2. Group by category
    category_map = defaultdict(list)
    for item in raw_data:
        cat = item.get('category', 'Unknown')
        category_map[cat].append(item)

    # 3. Core step: shuffle each category once so later slices remain nested/inclusive
    for cat in category_map:
        random.shuffle(category_map[cat])

    # 4. Generate a dataset for each ratio
    for rate in RATES:
        final_list = []
        print(f"\n正在处理采样率: {rate:.2f}")

        for cat, items in category_map.items():
            # compute the slice position
            target_num = int(len(items) * rate)
            if rate > 0 and len(items) > 0 and target_num == 0:
                target_num = 1

            # sequential slicing: ensures 0.5 contains all of 0.25's content
            sampled_items = items[:target_num]

            # clean the data structure (remove response_lst, etc.)
            cleaned_items = [clean_item(it) for it in sampled_items]
            final_list.extend(cleaned_items)

        # 5. Global shuffle (breaks category clustering without breaking the nested inclusion)
        random.shuffle(final_list)

        # 6. Save
        output_filename = f"{input_p.stem}_ratio_{rate:.2f}.json"
        output_p = input_p.parent / output_filename
        with open(output_p, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False, indent=4)

        print(f"已保存: {output_p.name} | 条数: {len(final_list)}")


if __name__ == "__main__":
    # fix the random seed for reproducibility
    random.seed(42)

    # input file list
    datasets = [
        './output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel.json'
    ]

    for ds in datasets:
        if Path(ds).exists():
            sample_nested_raw(ds)
        else:
            print(f"错误: 找不到文件 {ds}")
