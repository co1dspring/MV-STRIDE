import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def remove_image_tags_from_jsonl(target_dir):
    """
    Iterate over all .jsonl files under the given directory, remove the <image> tags from the text field of each JSON object, and save.

    :param target_dir: the target directory path containing the .jsonl files.
    """
    if not os.path.isdir(target_dir):
        logging.error(f"目标目录不存在: {target_dir}")
        return

    logging.info(f"开始处理目录: {target_dir}")

    # Iterate over all files in the target directory
    for filename in os.listdir(target_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(target_dir, filename)
            logging.info(f"正在处理文件: {filename}")

            modified_lines = []

            try:
                # 1. Read the file content
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                modified_count = 0

                for line in lines:
                    try:
                        data = json.loads(line)

                        # 2. Locate the text field to modify
                        # The field is in the last element of the data[0]['content'] array

                        if 'data' in data and len(data['data']) > 0:
                            # The 'data' field is a list; we usually only care about the first element (the user question)
                            user_content = data['data'][0].get('content')

                            if isinstance(user_content, list):
                                # Find the element with type 'text' in the content list (usually the last)

                                for item in user_content:
                                    if item.get('type') == 'text' and 'text' in item:
                                        text_data = item['text']

                                        if 'string' in text_data:
                                            original_string = text_data['string']

                                            # 3. Core operation: replace the <image> tag
                                            modified_string = original_string.replace("<image>", "")

                                            # If the string actually changed, update the data and count
                                            if modified_string != original_string:
                                                text_data['string'] = modified_string
                                                modified_count += 1
                                                # logging.debug(f"  - replaced successfully on line: {lines.index(line)}") # optional: for debugging

                        # 4. Convert the modified JSON object back to a string
                        modified_lines.append(json.dumps(data, ensure_ascii=False) + '\n')

                    except json.JSONDecodeError as e:
                        logging.warning(f"文件 {filename} 中存在非法的 JSON 行，跳过。错误: {e}")
                        modified_lines.append(line)  # 保留原始行以防丢失数据

                # 5. Write the modified content back to the file (overwrite)
                if modified_count > 0:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(modified_lines)
                    logging.info(f"  -> 处理完成。共修改了 {modified_count} 条记录的文本。")
                else:
                    logging.info("  -> 未发现需要替换的 <image> 标记，文件未修改。")

            except Exception as e:
                logging.error(f"处理文件 {filename} 时发生错误: {e}")


# =================================================================
# Configuration
# Change this path to the directory containing your jsonl files
# For example: TARGET_DIR = "/path/to/your/jsonl/files"
TARGET_DIR = "./jsonl"
# =================================================================

if __name__ == "__main__":
    remove_image_tags_from_jsonl(TARGET_DIR)
    logging.info("所有文件处理完毕。")
