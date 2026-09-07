import os
import json
from typing import List, Any


def merge_json_files():
    """
    Iterate over all JSON files in a preset directory, merge their contents into one list,
    and save the result to the specified output file.
    """
    # =========================================================
    # Set your input directory and output file name here
    # =========================================================
    INPUT_DIRECTORY = "../QA_jsons_all_categories_20251128_sampled_5/categories"  # replace with the directory containing your JSON files
    OUTPUT_FILENAME = "../QA_jsons_all_categories_20251128_sampled_5/infinigen_mmsibench_all_categories_20251128_sampled_5_merged.json"  # replace with your desired output file name
    # =========================================================

    all_data: List[Any] = []

    # check whether the input directory exists
    if not os.path.isdir(INPUT_DIRECTORY):
        print(f"错误：输入目录不存在: {INPUT_DIRECTORY}")
        return

    print(f"开始扫描目录: {INPUT_DIRECTORY}")

    # iterate over all files in the directory
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".json"):
            filepath = os.path.join(INPUT_DIRECTORY, filename)

            try:
                # process only files, skip directories
                if not os.path.isfile(filepath):
                    continue

                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # merge logic: append the read content to the total list
                    if isinstance(data, list):
                        # if the file content is a list, extend the total list
                        all_data.extend(data)
                        print(f"  - 成功读取并扩展列表: {filename} ({len(data)} 项)")
                    else:
                        # otherwise, append it as a single element (dict, string, or other) to the total list
                        all_data.append(data)
                        print(f"  - 成功读取并添加对象: {filename}")

            except json.JSONDecodeError:
                print(f"  - 错误: 文件 {filename} 不是有效的 JSON 格式，已跳过。")
            except Exception as e:
                print(f"  - 错误: 读取文件 {filename} 时发生未知错误: {e}")

    print(f"\n扫描完成。共收集到 {len(all_data)} 个数据项。")

    # write the merged data to the new JSON file
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            # use indent=4 to format the output for readability
            json.dump(all_data, outfile, ensure_ascii=False, indent=4)
        print(f"成功将所有数据整合并保存到: {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"错误：保存文件 {OUTPUT_FILENAME} 时发生错误: {e}")


if __name__ == "__main__":
    merge_json_files()
