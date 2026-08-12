import os
import json
from collections import defaultdict
from typing import List, Dict, Any


def split_json_by_category():
    """
    读取一个 JSON 文件（包含字典列表），根据每个字典的 'category' 字段进行分组，
    并将每个分组保存为单独的 JSON 文件，文件名即为 category 的值。
    """
    # =========================================================
    # 请在这里设置您的文件路径和输出目录
    # =========================================================
    INPUT_FILE = "../QA_jsons_all_categories_20251128_sampled/infinigen_mmsibench_all_categories_20251128_sampled.json"  # 替换为你的原始 JSON 文件路径
    OUTPUT_DIRECTORY = "../QA_jsons_all_categories_20251128_sampled/categories"  # 替换为你希望保存拆分文件的目录
    # =========================================================

    # 1. 检查输入文件和输出目录
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件不存在: {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出目录: {OUTPUT_DIRECTORY}")

    # 用于按 category 分组存储数据的字典
    # 键是 category (str)，值是该 category 下所有字典的列表 (List[Dict])
    grouped_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 2. 读取原始 JSON 文件
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

    # 3. 按 'category' 字段进行分组
    for item in data:
        if isinstance(item, dict) and "category" in item:
            category_key = item["category"]
            # 确保 category 键值是字符串，防止出现问题
            if isinstance(category_key, str):
                grouped_data[category_key].append(item)
            else:
                print(f"警告：跳过数据项，因为 'category' 键值不是字符串: {category_key}")
        else:
            print("警告：跳过数据项，因为它不是字典或缺少 'category' 键。")

    # 4. 将每个分组保存为单独的 JSON 文件
    print("\n开始保存拆分文件...")
    for category, items in grouped_data.items():
        # 清理 category 字符串以作为文件名（替换掉文件系统中不允许的字符，如 "/"）
        safe_filename = category.replace(os.path.sep, "_").replace(":", "_")
        output_filepath = os.path.join(OUTPUT_DIRECTORY, f"{safe_filename}.json")

        try:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                # 使用 indent=4 格式化输出，便于阅读
                json.dump(items, outfile, ensure_ascii=False, indent=4)
            print(f"  - 保存 {category} 成功，共 {len(items)} 项，文件: {output_filepath}")
        except Exception as e:
            print(f"  - 错误：保存文件 {output_filepath} 时发生错误: {e}")

    print("\n所有文件拆分及保存完成。")


if __name__ == "__main__":
    split_json_by_category()
