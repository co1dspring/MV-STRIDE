import json
import os
from typing import Any, Dict, List, Union


def duplicate_and_save_json(input_filepath: str, output_filepath: str) -> None:
    """
    Read a JSON file, double its content (repeat elements for a list, keep a dict as-is),
    and save the result to a new JSON file.

    Args:
        input_filepath (str): path of the input JSON file.
        output_filepath (str): path of the new output JSON file.
    """
    print(f"--- 开始处理文件 ---")
    print(f"输入文件: {input_filepath}")

    # 1. Check whether the input file exists
    if not os.path.exists(input_filepath):
        print(f"❌ 错误: 输入文件 '{input_filepath}' 不存在。")
        return

    # 2. Read the JSON data
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data: Union[Dict, List] = json.load(f)
        print("✅ JSON文件读取成功。")
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 '{input_filepath}' 不是有效的JSON格式。")
        return
    except Exception as e:
        print(f"❌ 错误: 读取文件时发生未知错误: {e}")
        return

    # 3. Duplicate the data content
    duplicated_data: Any

    if isinstance(data, list):
        # if it is a list, duplicate all elements with the * 2 operator
        duplicated_data = data * 2
        print(f"👉 数据类型为列表 (List)，已将 {len(data)} 个元素复制为 {len(duplicated_data)} 个元素。")
    elif isinstance(data, dict):
        # if it is a dict, duplicate it as-is. Because dicts have no notion of order/duplication,
        # the duplication is usually keeping it as-is, or wrapping it in a two-element list.
        # the goal is to double the content; we return the two dicts in a single list.
        duplicated_data = [data, data]
        print(f"👉 数据类型为字典 (Dict)，已将其作为两个元素放入新的列表中。")
    else:
        # for other primitive types (strings, numbers, etc.), wrap them in a list too
        duplicated_data = [data, data]
        print(f"👉 数据类型为 {type(data).__name__}，已将其复制两份放入新的列表中。")

    # 4. Save to the new JSON file
    try:
        # ensure the output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            # indent the output for readability
            json.dump(duplicated_data, f, indent=4, ensure_ascii=False)

        print(f"🎉 成功保存复制后的内容到: {output_filepath}")
        print(f"--- 处理完成 ---")

    except Exception as e:
        print(f"❌ 错误: 写入文件 '{output_filepath}' 时失败: {e}")


# --- Example usage ---
if __name__ == '__main__':

    # assumed input file path
    INPUT_FILE = 'input_data.json'
    OUTPUT_FILE = 'output_data_doubled.json'

    # --- 1. Create an example input file (if it does not exist) ---
    example_list_data = [
        {"id": 1, "text": "第一条记录"},
        {"id": 2, "text": "第二条记录"}
    ]
    if not os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(example_list_data, f, indent=4, ensure_ascii=False)
        print(f"已创建示例输入文件: {INPUT_FILE}")

    # --- 2. Call the processing function ---
    duplicate_and_save_json(INPUT_FILE, OUTPUT_FILE)

    # ----------------------------------------------------
    # Another dict-data example (if testing is needed)
    # DICT_INPUT = 'input_dict.json'
    # DICT_OUTPUT = 'output_dict_doubled.json'

    # example_dict_data = {"config": "model_v1", "batch_size": 32}
    # if not os.path.exists(DICT_INPUT):
    #     with open(DICT_INPUT, 'w', encoding='utf-8') as f:
    #         json.dump(example_dict_data, f, indent=4, ensure_ascii=False)

    # duplicate_and_save_json(DICT_INPUT, DICT_OUTPUT)
