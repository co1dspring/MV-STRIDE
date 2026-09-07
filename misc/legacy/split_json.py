import os
import json
from collections import defaultdict
from typing import List, Dict, Any


def split_json_by_category():
    """
    Read a JSON file (a list of dicts), group the dicts by their 'category' field,
    and save each group to a separate JSON file named after the category value.
    """
    # =========================================================
    # Set your file path and output directory here
    # =========================================================
    INPUT_FILE = "../QA_jsons_all_categories_20251128_sampled/infinigen_mmsibench_all_categories_20251128_sampled.json"  # replace with your original JSON file path
    OUTPUT_DIRECTORY = "../QA_jsons_all_categories_20251128_sampled/categories"  # replace with the directory where the split files should be saved
    # =========================================================

    # 1. Check the input file and output directory
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件不存在: {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出目录: {OUTPUT_DIRECTORY}")

    # dict used to store the data grouped by category
    # key is the category (str), value is the list of dicts for that category (List[Dict])
    grouped_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 2. Read the original JSON file
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if not isinstance(data, list):
                print("错误：JSON 文件的根元素必须是一个列表 (List)。")
                return

    except json.JSONDecodeError:
        print(f"错误：文件 {INPUT_FILE} 不是有效的 JSON 格式。")
        return
    except Exception as e:
        print(f"错误：读取文件时发生未知错误: {e}")
        return

    print(f"\n成功读取 {len(data)} 个数据项。开始分组...")

    # 3. Group by the 'category' field
    for item in data:
        if isinstance(item, dict) and "category" in item:
            category_key = item["category"]
            # ensure the category key is a string to avoid problems
            if isinstance(category_key, str):
                grouped_data[category_key].append(item)
            else:
                print(f"警告：跳过数据项，因为 'category' 键值不是字符串: {category_key}")
        else:
            print("警告：跳过数据项，因为它不是字典或缺少 'category' 键。")

    # 4. Save each group to its own JSON file
    print("\n开始保存拆分文件...")
    for category, items in grouped_data.items():
        # sanitize the category string for use as a file name (replace characters not allowed in the file system, e.g. "/")
        safe_filename = category.replace(os.path.sep, "_").replace(":", "_")
        output_filepath = os.path.join(OUTPUT_DIRECTORY, f"{safe_filename}.json")

        try:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                # use indent=4 to format the output for readability
                json.dump(items, outfile, ensure_ascii=False, indent=4)
            print(f"  - 保存 {category} 成功，共 {len(items)} 项，文件: {output_filepath}")
        except Exception as e:
            print(f"  - 错误：保存文件 {output_filepath} 时发生错误: {e}")

    print("\n所有文件拆分及保存完成。")


if __name__ == "__main__":
    split_json_by_category()
