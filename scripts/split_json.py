import json
import os
import re


def split_json_by_category(file_path):
    # 1. Get the original filename and create the target folder
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    output_dir = base_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建文件夹: {output_dir}")

    # 2. Load the original data
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("错误：无法解析 JSON 文件，请检查格式。")
            return

    # 3. Split the data by category
    category_map = {}
    for item in data:
        cat = item.get("category", "Uncategorized")
        if cat not in category_map:
            category_map[cat] = []
        category_map[cat].append(item)

    # 4. Save to the respective json files
    for cat, items in category_map.items():
        # Clean the filename: replace illegal characters (e.g. / \ : * ? " < > |) with underscores
        safe_cat_name = re.sub(r'[\\/*?:"<>|]', '_', cat)
        save_path = os.path.join(output_dir, f"{safe_cat_name}.json")

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=4, ensure_ascii=False)

        print(f"已保存: {save_path} (条目数: {len(items)})")

    print("\n拆分完成！")


# Usage example
if __name__ == "__main__":
    # Replace 'your_data.json' with your actual filename
    split_json_by_category('./api/output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00_no_system.json')
