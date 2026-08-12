import json
import random
from collections import defaultdict
from pathlib import Path

def reformat_item(item, prompt_suffix):
    """
    将旧格式转化为包含 solution 和诱导 prompt 的新格式
    """
    # 1. 提取原始问题和回答
    user_content = item['messages'][0]['content']
    assistant_content = item['messages'][1]['content']

    # 2. 构造新的 user content (追加诱导 Prompt)
    # 确保追加前逻辑清晰，如果原问题末尾没换行，加一个换行
    if not user_content.endswith('\n'):
        user_content += '\n'
    new_user_content = f"{user_content}{prompt_suffix}"
    # new_user_content = user_content

    # 3. 构造新的数据对象
    new_item = {
        "images": item.get("images", []),
        "messages": [
            {
                "role": "user",
                "content": new_user_content
            }
        ],
        # 将原回答放入 solution 字段，并包裹标准标签
        "solution": f"<answer> {assistant_content.strip()} </answer>"
    }

    # 4. 保留元数据（可选，建议保留方便后续追溯）
    new_item["category"] = item.get("category")
    new_item["scene_name"] = item.get("scene_name")
    new_item["data_source"] = item.get("data_source")

    return new_item

def sample_single_dataset(input_path, config_path):
    PROMPT_SUFFIX = '\nOutput your step-by-step thinking process in <think> </think> tags and the final choice (e.g., A: option) in <answer> </answer> tags.'
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 提取全局采样率，默认为 1.0 (不缩放)
    overall_rate = config.get('overall_sampling_rate', 1.0)

    # 2. 读取输入文件
    input_p = Path(input_path)
    with open(input_p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"正在处理: {input_p.name}")
    print(f"原始数据总数: {len(data)}")

    # 3. 按子类别进行归类
    # 结构: category_name -> [items]
    category_map = defaultdict(list)
    for item in data:
        cat = item.get('category', 'Unknown')
        category_map[cat].append(item)

    # 4. 执行多级采样
    final_data = []
    stats = {}

    print(f"全局二次采样率 (overall_sampling_rate): {overall_rate}")
    print("\n--- 分类别采样细节 ---")

    for cat, items in category_map.items():
        # 获取该类别的配置
        cat_config = config.get(cat)

        if not cat_config:
            print(f"警告: 类别 [{cat}] 未在配置文件中定义，将跳过。")
            continue

        # 计算该类别的采样数量
        # 逻辑：该类样本数 * 类别采样率 * 全局采样率
        cat_rate = cat_config.get('sampling_rate', 1.0)
        target_num = int(len(items) * cat_rate * overall_rate)

        # 确保至少采 1 条（如果原始数据不为空且计算结果为0时）
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

    # 5. 打乱顺序并保存
    random.shuffle(final_data)

    final_reformatted_data = []

    for item in final_data:
        new_item = reformat_item(item, PROMPT_SUFFIX)
        final_reformatted_data.append(new_item)

    # 构造输出路径: {stem}_sampled.json
    output_p = input_p.parent / f"{input_p.stem}_sampled.json"
    with open(output_p, 'w', encoding='utf-8') as f:
        json.dump(final_reformatted_data, f, ensure_ascii=False, indent=4)

    # 6. 打印总结
    print("\n" + "=" * 60)
    print(f"采样完成报告")
    print("-" * 60)
    print(f"总原始数据: {len(data)} 条")
    print(f"总输出数据: {len(final_reformatted_data)} 条 (约总量的 {len(final_reformatted_data) / len(data) * 100:.2f}%)")
    print(f"结果已保存: {output_p.absolute()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 配置文件路径
    config_file = 'stage3_sampling_config.json'
    random.seed(42)

    # 你可以手动运行两次，或者在这里写两个路径
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
