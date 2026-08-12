import json
import random
import os
import shutil
import zipfile
from math import floor
from PIL import Image
import tarfile


def process_dataset(input_json_files, dataset_name, output_dir, image_base_mapping):
    """
    处理多个JSON数据集：合并、打乱、分割、根据不同规则映射图像路径并打包

    参数:
    input_json_files: 输入的JSON文件路径列表
    dataset_name: 数据集名称（如 'pilottest_msr'）
    output_dir: 输出目录
    image_base_mapping: 字典，包含各个数据集的本地根目录
    """
    induction_prompt = "\nPlease select the correct option and output it in the format: <Option Letter>. <Option Text>"

    # 创建主输出目录结构
    main_dir = os.path.join(output_dir, dataset_name)
    jsonl_dir = os.path.join(main_dir, "jsonl")
    images_dir = os.path.join(main_dir, "images")

    # 创建目录
    for dir_path in [main_dir, jsonl_dir, images_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 1. 读取并合并所有JSON文件
    data = []
    for json_file in input_json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                file_data = json.load(file)
                # 为每条数据标记来源
                for item in file_data:
                    item['_source_file'] = json_file
                data.extend(file_data)
            print(f"成功读取JSON文件: {os.path.basename(json_file)}，共{len(file_data)}条数据")
        except Exception as e:
            print(f"读取JSON文件失败 {json_file}: {e}")
            return

    print(f"合并完成，总计数据量: {len(data)} 条")

    # 2. 随机打乱数据顺序
    random.shuffle(data)
    print("数据顺序已随机打乱")

    # 3. 自适应计算分割份数n
    total_length = len(data)
    max_items_per_chunk = 30000

    min_chunks = (total_length + max_items_per_chunk - 1) // max_items_per_chunk
    base = 16
    n = ((min_chunks + base - 1) // base) * base

    print(f"数据总量: {total_length} 条")
    print(f"计算得到的分割份数 (n): {n} (16的倍数)")

    # 4. 将数据均匀分为n份
    chunk_size = total_length // n
    remainder = total_length % n

    chunks = []
    start_index = 0

    for i in range(n):
        end_index = start_index + chunk_size + (1 if i < remainder else 0)
        chunk = data[start_index:end_index]
        chunks.append(chunk)
        start_index = end_index

    print(f"数据已均匀分割为{len(chunks)}份，每份大小约{len(chunks[0])}条数据")

    # 5. 处理每个数据分块
    for i, chunk in enumerate(chunks):
        chunk_id = f"{i:06d}"
        jsonl_filename = f"data_{chunk_id}.jsonl"
        jsonl_filepath = os.path.join(jsonl_dir, jsonl_filename)

        temp_image_dir = os.path.join(images_dir, f"temp_{chunk_id}")
        os.makedirs(temp_image_dir, exist_ok=True)

        processed_chunk = []

        for j, item in enumerate(chunk):
            image_content = []

            source_file = item.get('_source_file', '').lower()
            json_filename = os.path.basename(item.get('_source_file', '')).lower()

            for image_name in item['images']:
                # 统一将路径分隔符替换为 '/' 方便做切片
                image_name_fixed = image_name.replace('\\', '/')
                path_parts = image_name_fixed.split('/')

                # 根据数据集类型，分别处理相对路径和重命名
                if 'infinigen' in json_filename:
                    current_image_base = image_base_mapping['infinigen']
                    # 截取：saved_scenes/ 后面两层 (e.g., 5253c5d8/Image_9_0_0048_0.png)
                    image_sub_path = os.path.join(*path_parts[-2:])

                elif 'scannetpp' in json_filename:
                    current_image_base = image_base_mapping['scannetpp']
                    # 截取：scannetpp_sampled_modified/ 后面三层 (e.g., 281ba69af1_iphone/images/frame_000690.jpg)
                    image_sub_path = os.path.join(*path_parts[-3:])

                else:
                    print(f"警告: 无法识别的数据来源 {json_filename}，跳过该图片")
                    continue

                # 拼接成真实的本地图像路径
                image_path = os.path.join(current_image_base, image_sub_path)

                # 统一转成新图名：将相对路径里的斜杠换成下划线 (实现 场景名+图像名)
                image_new_name = image_sub_path.replace("\\", "_").replace("/", "_")
                image_new_path = os.path.join(temp_image_dir, image_new_name)

                # 读取图像以获取真实尺寸并复制
                try:
                    shutil.copy2(image_path, image_new_path)
                    with Image.open(image_path) as img:
                        real_width, real_height = img.size
                        image_format = img.format.lower() if img.format else 'png'

                except FileNotFoundError:
                    print(f"警告: 图像文件未找到: {image_path}，使用默认尺寸。")
                    real_width, real_height = 640, 480
                    image_format = 'png'
                except Exception as e:
                    print(f"警告: 无法读取图像尺寸: {image_path}。错误: {e}，使用默认尺寸。")
                    real_width, real_height = 640, 480
                    image_format = 'png'

                image_dict = {
                    "type": "image",
                    "image": {
                        "type": "relative_path",
                        "format": f"image/{image_format}",
                        "relative_path": image_new_name,
                        "width": real_width,
                        "height": real_height
                    }
                }
                image_content.append(image_dict)

            # 重新整理每条数据结构并储存
            new_item = {
                "meta_prompt": [""],
                "data": [
                    {
                        "role": "user",
                        "content": image_content + [
                            {"type": "text", "text": {"type": "string", "format": "utf-8", "string": item['messages'][0]['content'].replace("<image>", "") + induction_prompt}}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": {"type": "string", "format": "utf-8", "string": item['messages'][1]['content']}}
                        ]
                    }
                ],
                "text_dup": 1
            }
            processed_chunk.append(new_item)

        # 保存处理后的JSONL文件
        try:
            with open(jsonl_filepath, 'w', encoding='utf-8') as f:
                for item in processed_chunk:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"已保存JSONL文件: {jsonl_filename}")
        except Exception as e:
            print(f"保存JSONL文件失败: {e}")

        # 创建tar压缩包
        tar_filename = f"data_{chunk_id}.tar"
        tar_filepath = os.path.join(images_dir, tar_filename)

        try:
            with tarfile.open(tar_filepath, 'w') as tarf:
                for root, dirs, files in os.walk(temp_image_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_image_dir)
                        tarf.add(file_path, arcname=arcname)

            print(f"已创建压缩包: {tar_filename}")
            shutil.rmtree(temp_image_dir)  # 删除临时图像目录
        except Exception as e:
            print(f"创建压缩包失败: {e}")

    print(f"\n处理完成！")
    print(f"输出目录: {main_dir}")
    print(f"JSONL文件: {len(os.listdir(jsonl_dir))}个")
    print(f"图像压缩包: {len([f for f in os.listdir(images_dir) if f.endswith('.tar')])}个")


# 使用示例
if __name__ == "__main__":
    # 1. 输入的两个特定阶段的 JSON 文件路径
    input_files = [
        r"D:\Data\infinigen_20251031\QA_jsons_Infinigen_MultilevelCategories_20260313_sampled_MCA_Multistage\Infinigen_MultilevelCategories_20260313_sampled_MCA_Multistage_stage1.json",
        r"D:\Data\infinigen_20251031\QA_jsons_ScannetppIphone_MultilevelCategories_20260313_sampled_MCA_Multistage\ScannetppIphone_MultilevelCategories_20260313_sampled_MCA_Multistage_stage1.json"
    ]

    # 2. 新的数据集文件夹命名
    dataset_name = "pilottest_msr"

    # 3. 输出目录保持不变
    output_directory = r"D:\Data\infinigen_20251031\refresh"

    # 4. 根据你提供的正确本地映射根目录
    image_base_mapping = {
        "infinigen": r"D:\Data\infinigen_20251031\infinigen_metadata_ver2\saved_scenes",
        "scannetpp": r"D:\Data\scannetpp\scannetpp_sampled_modified"
    }

    # 执行处理
    process_dataset(input_files, dataset_name, output_directory, image_base_mapping)
