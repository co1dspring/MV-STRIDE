import os
import json
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def remove_image_tags_from_jsonl(target_dir):
    """
    遍历指定目录下所有 .jsonl 文件，去除 JSON 对象中 text 字段的 <image> 标记，并保存。

    :param target_dir: 包含 .jsonl 文件的目标目录路径。
    """
    if not os.path.isdir(target_dir):
        logging.error(f"目标目录不存在: {target_dir}")
        return

    logging.info(f"开始处理目录: {target_dir}")

    # 遍历目标目录下的所有文件
    for filename in os.listdir(target_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(target_dir, filename)
            logging.info(f"正在处理文件: {filename}")

            modified_lines = []

            try:
                # 1. 读取文件内容
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                modified_count = 0

                for line in lines:
                    try:
                        data = json.loads(line)

                        # 2. 定位到需要修改的文本字段
                        # 字段位于 data[0]['content'] 数组的最后一个元素中

                        if 'data' in data and len(data['data']) > 0:
                            # 'data' 字段是一个列表，通常我们只关心第一个元素 (即用户提问)
                            user_content = data['data'][0].get('content')

                            if isinstance(user_content, list):
                                # 寻找 content 列表中 type 为 'text' 的元素 (通常是最后一个)

                                for item in user_content:
                                    if item.get('type') == 'text' and 'text' in item:
                                        text_data = item['text']

                                        if 'string' in text_data:
                                            original_string = text_data['string']

                                            # 3. 核心操作：替换 <image> 标记
                                            modified_string = original_string.replace("<image>", "")

                                            # 如果字符串确实发生了变化，则更新数据并计数
                                            if modified_string != original_string:
                                                text_data['string'] = modified_string
                                                modified_count += 1
                                                # logging.debug(f"  - 替换成功在 line: {lines.index(line)}") # 可选：用于调试

                        # 4. 将修改后的 JSON 对象转换回字符串
                        modified_lines.append(json.dumps(data, ensure_ascii=False) + '\n')

                    except json.JSONDecodeError as e:
                        logging.warning(f"文件 {filename} 中存在非法的 JSON 行，跳过。错误: {e}")
                        modified_lines.append(line)  # 保留原始行以防丢失数据

                # 5. 将修改后的内容写回文件 (覆盖原文件)
                if modified_count > 0:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(modified_lines)
                    logging.info(f"  -> 处理完成。共修改了 {modified_count} 条记录的文本。")
                else:
                    logging.info("  -> 未发现需要替换的 <image> 标记，文件未修改。")

            except Exception as e:
                logging.error(f"处理文件 {filename} 时发生错误: {e}")


# =================================================================
# 配置区
# 请将此路径修改为你的 jsonl 文件所在的目录
# 例如: TARGET_DIR = "/path/to/your/jsonl/files"
TARGET_DIR = "./jsonl"
# =================================================================

if __name__ == "__main__":
    remove_image_tags_from_jsonl(TARGET_DIR)
    logging.info("所有文件处理完毕。")
