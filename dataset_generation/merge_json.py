import os
import json
from typing import List, Any


def merge_json_files():
    """
    遍历预设目录下的所有 JSON 文件，并将它们的内容合并到一个列表中，
    然后将结果保存到指定的输出文件。
    """
    # =========================================================
    # 请在这里设置您的输入目录和输出文件名
    # =========================================================
    INPUT_DIRECTORY = "../QA_jsons_all_categories_20251128_sampled_5/categories"  # 替换为你的 JSON 文件所在的目录路径
    OUTPUT_FILENAME = "../QA_jsons_all_categories_20251128_sampled_5/infinigen_mmsibench_all_categories_20251128_sampled_5_merged.json"  # 替换为你希望的输出文件名
    # =========================================================

    all_data: List[Any] = []

    # 检查输入目录是否存在
    if not os.path.isdir(INPUT_DIRECTORY):
        print(f"错误：输入目录不存在: {INPUT_DIRECTORY}")
        return

    print(f"开始扫描目录: {INPUT_DIRECTORY}")

    # 遍历目录中的所有文件
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".json"):
            filepath = os.path.join(INPUT_DIRECTORY, filename)

            try:
                # 确保只处理文件，跳过目录
                if not os.path.isfile(filepath):
                    continue

                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 整合逻辑：将读取到的内容添加到总列表中
                    if isinstance(data, list):
                        # 如果文件内容本身是列表，则扩展到总列表
                        all_data.extend(data)
                        print(f"  - 成功读取并扩展列表: {filename} ({len(data)} 项)")
                    else:
                        # 否则，将其作为一个元素（无论是字典、字符串还是其他对象）添加到总列表
                        all_data.append(data)
                        print(f"  - 成功读取并添加对象: {filename}")

            except json.JSONDecodeError:
                print(f"  - 错误: 文件 {filename} 不是有效的 JSON 格式，已跳过。")
            except Exception as e:
                print(f"  - 错误: 读取文件 {filename} 时发生未知错误: {e}")

    print(f"\n扫描完成。共收集到 {len(all_data)} 个数据项。")

    # 将整合后的数据写入新的 JSON 文件
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            # 使用 indent=4 使输出文件格式化，便于阅读
            json.dump(all_data, outfile, ensure_ascii=False, indent=4)
        print(f"成功将所有数据整合并保存到: {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"错误：保存文件 {OUTPUT_FILENAME} 时发生错误: {e}")


if __name__ == "__main__":
    merge_json_files()
