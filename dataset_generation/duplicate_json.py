import json
import os
from typing import Any, Dict, List, Union


def duplicate_and_save_json(input_filepath: str, output_filepath: str) -> None:
    """
    读取JSON文件，将其内容复制两倍（如果是列表则重复元素，如果是字典则保持不变），
    并将结果保存到新的JSON文件中。

    Args:
        input_filepath (str): 输入的JSON文件路径。
        output_filepath (str): 输出的新JSON文件路径。
    """
    print(f"--- 开始处理文件 ---")
    print(f"输入文件: {input_filepath}")

    # 1. 检查输入文件是否存在
    if not os.path.exists(input_filepath):
        print(f"❌ 错误: 输入文件 '{input_filepath}' 不存在。")
        return

    # 2. 读取JSON数据
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

    # 3. 复制数据内容
    duplicated_data: Any

    if isinstance(data, list):
        # 如果是列表，直接使用 * 2 运算符复制所有元素
        duplicated_data = data * 2
        print(f"👉 数据类型为列表 (List)，已将 {len(data)} 个元素复制为 {len(duplicated_data)} 个元素。")
    elif isinstance(data, dict):
        # 如果是字典，复制其本身。因为字典没有“顺序”和“重复”的概念，
        # 通常复制操作就是保持原样，或将其放入一个包含两个元素的列表中。
        # 这里的目标是“复制内容2倍”，我们选择将两个字典放入一个列表中返回。
        duplicated_data = [data, data]
        print(f"👉 数据类型为字典 (Dict)，已将其作为两个元素放入新的列表中。")
    else:
        # 针对其他基本类型（如字符串、数字等），同样放入列表中
        duplicated_data = [data, data]
        print(f"👉 数据类型为 {type(data).__name__}，已将其复制两份放入新的列表中。")

    # 4. 保存到新的JSON文件
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            # 使用缩进使输出文件更易读
            json.dump(duplicated_data, f, indent=4, ensure_ascii=False)

        print(f"🎉 成功保存复制后的内容到: {output_filepath}")
        print(f"--- 处理完成 ---")

    except Exception as e:
        print(f"❌ 错误: 写入文件 '{output_filepath}' 时失败: {e}")


# --- 示例使用 ---
if __name__ == '__main__':

    # 假设的输入文件路径
    INPUT_FILE = 'input_data.json'
    OUTPUT_FILE = 'output_data_doubled.json'

    # --- 1. 创建一个示例输入文件 (如果不存在) ---
    example_list_data = [
        {"id": 1, "text": "第一条记录"},
        {"id": 2, "text": "第二条记录"}
    ]
    if not os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(example_list_data, f, indent=4, ensure_ascii=False)
        print(f"已创建示例输入文件: {INPUT_FILE}")

    # --- 2. 调用处理函数 ---
    duplicate_and_save_json(INPUT_FILE, OUTPUT_FILE)

    # ----------------------------------------------------
    # 另一个字典数据的示例（如果需要测试）
    # DICT_INPUT = 'input_dict.json'
    # DICT_OUTPUT = 'output_dict_doubled.json'

    # example_dict_data = {"config": "model_v1", "batch_size": 32}
    # if not os.path.exists(DICT_INPUT):
    #     with open(DICT_INPUT, 'w', encoding='utf-8') as f:
    #         json.dump(example_dict_data, f, indent=4, ensure_ascii=False)

    # duplicate_and_save_json(DICT_INPUT, DICT_OUTPUT)
