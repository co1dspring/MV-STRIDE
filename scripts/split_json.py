import json
import os
import re


def split_json_by_category(file_path):
    # 1. 获取原文件名并创建目标文件夹
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    output_dir = base_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建文件夹: {output_dir}")

    # 2. 加载原数据
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("错误：无法解析 JSON 文件，请检查格式。")
            return

    # 3. 根据 category 拆分数据
    category_map = {}
    for item in data:
        cat = item.get("category", "Uncategorized")
        if cat not in category_map:
            category_map[cat] = []
        category_map[cat].append(item)

    # 4. 保存到各自的 json 文件
    for cat, items in category_map.items():
        # 清理文件名：将非法字符（如 / \ : * ? " < > |）替换为下划线
        safe_cat_name = re.sub(r'[\\/*?:"<>|]', '_', cat)
        save_path = os.path.join(output_dir, f"{safe_cat_name}.json")

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=4, ensure_ascii=False)

        print(f"已保存: {save_path} (条目数: {len(items)})")

    print("\n拆分完成！")


# 使用示例
if __name__ == "__main__":
    # 请将 'your_data.json' 替换为你实际的文件名
    split_json_by_category('./api/output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00_no_system.json')
