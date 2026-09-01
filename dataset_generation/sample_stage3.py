import json
import random
from collections import defaultdict
from pathlib import Path

def reformat_item(item, prompt_suffix):
    """
    Convert the old format into a new format containing a solution and an eliciting prompt.
    """
    # 1. Extract the original question and answer
    user_content = item['messages'][0]['content']
    assistant_content = item['messages'][1]['content']

    # 2. Build the new user content (append the eliciting prompt)
    # make the append logic clean; add a newline if the original question does not end with one
    if not user_content.endswith('\n'):
        user_content += '\n'
    new_user_content = f"{user_content}{prompt_suffix}"
    # new_user_content = user_content

    # 3. Build the new data object
    new_item = {
        "images": item.get("images", []),
        "messages": [
            {
                "role": "user",
                "content": new_user_content
            }
        ],
        # put the original answer into the solution field, wrapped in the standard tags
        "solution": f"<answer> {assistant_content.strip()} </answer>"
    }

    # 4. Keep the metadata (optional, but recommended for later tracing)
    new_item["category"] = item.get("category")
    new_item["scene_name"] = item.get("scene_name")
    new_item["data_source"] = item.get("data_source")

    return new_item

def sample_single_dataset(input_path, config_path):
    PROMPT_SUFFIX = '\nOutput your step-by-step thinking process in <think> </think> tags and the final choice (e.g., A: option) in <answer> </answer> tags.'
    # 1. Load the config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # get the global sampling rate, default 1.0 (no scaling)
    overall_rate = config.get('overall_sampling_rate', 1.0)

    # 2. Read the input file
    input_p = Path(input_path)
    with open(input_p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"正在处理: {input_p.name}")
    print(f"原始数据总数: {len(data)}")

    # 3. Group by sub-category
    # structure: category_name -> [items]
    category_map = defaultdict(list)
    for item in data:
        cat = item.get('category', 'Unknown')
        category_map[cat].append(item)

    # 4. Perform multi-level sampling
    final_data = []
    stats = {}

    print(f"全局二次采样率 (overall_sampling_rate): {overall_rate}")
    print("\n--- 分类别采样细节 ---")

    for cat, items in category_map.items():
        # get this category's config
        cat_config = config.get(cat)

        if not cat_config:
            print(f"警告: 类别 [{cat}] 未在配置文件中定义，将跳过。")
            continue

        # compute this category's sampling count
        # logic: items * category sampling rate * global sampling rate
        cat_rate = cat_config.get('sampling_rate', 1.0)
        target_num = int(len(items) * cat_rate * overall_rate)

        # ensure at least 1 sample (if the raw data is non-empty and the result is 0)
        if len(items) > 0 and target_num == 0 and (cat_rate * overall_rate > 0):
            target_num = 1

        sampled_items = random.sample(items, min(len(items), target_num))
        final_data.extend(sampled_items)

        stats[cat] = {
            "original": len(items),
            "sampled": len(sampled_items),
            "rate": cat_rate
        }
        print(f"  [{cat:<45}] 原有:{len(items):>5} | 采样率:{cat_rate:>4.2f} | 最终:{len(sampled_items):>5}")

    # 5. Shuffle and save
    random.shuffle(final_data)

    final_reformatted_data = []

    for item in final_data:
        new_item = reformat_item(item, PROMPT_SUFFIX)
        final_reformatted_data.append(new_item)

    # build the output path: {stem}_sampled.json
    output_p = input_p.parent / f"{input_p.stem}_sampled.json"
    with open(output_p, 'w', encoding='utf-8') as f:
        json.dump(final_reformatted_data, f, ensure_ascii=False, indent=4)

    # 6. Print a summary
    print("\n" + "=" * 60)
    print(f"采样完成报告")
    print("-" * 60)
    print(f"总原始数据: {len(data)} 条")
    print(f"总输出数据: {len(final_reformatted_data)} 条 (约总量的 {len(final_reformatted_data) / len(data) * 100:.2f}%)")
    print(f"结果已保存: {output_p.absolute()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # config file path
    config_file = 'stage3_sampling_config.json'
    random.seed(42)

    # you can run it twice manually, or put two paths here
    datasets = [
        # '../api/output/Infinigen_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.json'
        '../api/output/ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.json'
        # '../QA_jsons_Infinigen_MultilevelCategories_20260125_sampled_MCA_Multistage/Infinigen_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3.json',
        # '../QA_jsons_ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage/ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3.json'
    ]

    for ds in datasets:
        if Path(ds).exists():
            sample_single_dataset(ds, config_file)
        else:
            print(f"错误: 找不到文件 {ds}")
