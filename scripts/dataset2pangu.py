import json
import random
import os
import shutil
import zipfile
from math import floor
from PIL import Image
import tarfile

def process_dataset(input_json_file, dataset_name, output_dir, image_base_dir):
    """
    处理JSON数据集：打乱、分割、复制图像并打包

    参数:
    input_json_file: 输入的JSON文件路径
    dataset_name: 数据集名称（用于创建文件夹）
    output_dir: 输出目录（默认为当前目录）
    """
    induction_prompt = "\nPlease select the correct option and output it in the format: <Option Letter>. <Option Text>"

    # 创建主输出目录结构
    main_dir = os.path.join(output_dir, dataset_name)
    jsonl_dir = os.path.join(main_dir, "jsonl")
    images_dir = os.path.join(main_dir, "images")

    # 创建目录
    for dir_path in [main_dir, jsonl_dir, images_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 1. 读取JSON文件
    try:
        with open(input_json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"成功读取JSON文件，共{len(data)}条数据")
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return

    # 2. 随机打乱数据顺序 [1](@ref)
    random.shuffle(data)
    print("数据顺序已随机打乱")

    # 3. 自适应计算分割份数n，确保n是16的倍数，且每份数据不超过30000条
    total_length = len(data)
    max_items_per_chunk = 30000

    # 计算满足最大条目限制所需的最小份数
    min_chunks = (total_length + max_items_per_chunk - 1) // max_items_per_chunk  # 向上取整

    # 将最小份数向上取整到最接近的16的倍数
    base = 16
    n = ((min_chunks + base - 1) // base) * base

    print(f"数据总量: {total_length} 条")
    print(f"目标单文件最大条目数: {max_items_per_chunk} 条")
    print(f"计算得到的分割份数 (n): {n} (16的倍数)")

    # 4. 将数据均匀分为n份 (使用您原有的均匀分配算法处理余数)
    chunk_size = total_length // n
    remainder = total_length % n

    chunks = []
    start_index = 0

    for i in range(n):
        # 计算当前分块的大小（处理不能整除的情况）
        end_index = start_index + chunk_size + (1 if i < remainder else 0)
        chunk = data[start_index:end_index]
        chunks.append(chunk)
        start_index = end_index

    print(f"数据已均匀分割为{len(chunks)}份，每份大小约{len(chunks[0])}条数据")

    # 4. 处理每个数据分块
    for i, chunk in enumerate(chunks):
        # 生成分块编号（6位数字，前导零）
        chunk_id = f"{i:06d}"
        jsonl_filename = f"data_{chunk_id}.jsonl"
        jsonl_filepath = os.path.join(jsonl_dir, jsonl_filename)

        # 创建临时目录用于存储当前分块的图像
        temp_image_dir = os.path.join(images_dir, f"temp_{chunk_id}")
        os.makedirs(temp_image_dir, exist_ok=True)

        # 处理分块中的每条数据
        processed_chunk = []

        for j, item in enumerate(chunk):
            image_content = []
            # 将图像改名并转存至新地址
            for image_name in item['images']:
                image_name = os.path.join(*image_name.split('/')[-2:])
                image_path = os.path.join(image_base_dir, image_name).replace('/', '\\')
                image_new_name = image_name.replace("\\", "_")
                image_new_path = os.path.join(temp_image_dir, image_new_name).replace('/', '\\')
                shutil.copy2(image_path, image_new_path)
                # 读取图像以获取真实尺寸
                try:
                    with Image.open(image_path) as img:
                        real_width, real_height = img.size
                        image_format = img.format.lower() if img.format else 'png'  # 尝试获取实际格式，否则默认为 png

                except FileNotFoundError:
                    # 如果文件不存在，记录错误并跳过
                    print(f"警告: 图像文件未找到: {image_path}，使用默认尺寸。")
                    real_width, real_height = 640, 480
                    image_format = 'png'
                except Exception as e:
                    # 其他读取错误（如文件损坏），记录错误并跳过
                    print(f"警告: 无法读取图像尺寸: {image_path}。错误: {e}，使用默认尺寸。")
                    real_width, real_height = 640, 480
                    image_format = 'png'
                image_dict = {
                    "type": "image",
                    "image": {
                        "type": "relative_path",
                        "format": f"image/{image_format}",  # 使用读取到的格式
                        "relative_path": image_new_name,
                        "width": real_width,  # <-- 真实宽度
                        "height": real_height  # <-- 真实高度
                    }
                }
                image_content.append(image_dict)
            # 重新整理每条数据结构并储存
            new_item = {
                "meta_prompt": [""],
                "data": [
                    {
                        "role": "user",
                        "content": image_content+[
                            {"type": "text", "text": {"type": "string", "format": "utf-8", "string": item['messages'][0]['content'].replace("<image>", "")+induction_prompt}}
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

        # 5. 保存处理后的JSONL文件 [9,10](@ref)
        try:
            with open(jsonl_filepath, 'w', encoding='utf-8') as f:
                for item in processed_chunk:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"已保存JSONL文件: {jsonl_filename}")
        except Exception as e:
            print(f"保存JSONL文件失败: {e}")

        # 6. 创建压缩包 [12](@ref)
        tar_filename = f"data_{chunk_id}.tar"
        tar_filepath = os.path.join(images_dir, tar_filename)

        try:
            # 使用 tarfile 创建 .tar 压缩包
            # 'w' 模式表示写入
            with tarfile.open(tar_filepath, 'w') as tarf:
                # os.walk 遍历 temp_image_dir 内部的所有文件和子目录
                for root, dirs, files in os.walk(temp_image_dir):
                    for file in files:
                        file_path = os.path.join(root, file)

                        # 计算文件在压缩包中的相对路径 (arcname)
                        # 这样压缩包内就不会包含完整的绝对路径
                        arcname = os.path.relpath(file_path, temp_image_dir)

                        # 添加文件到 tar 包中
                        tarf.add(file_path, arcname=arcname)

            print(f"已创建压缩包: {tar_filename}")

            # 删除临时图像目录
            shutil.rmtree(temp_image_dir)

        except Exception as e:
            print(f"创建压缩包失败: {e}")

    # 7. 创建说明文件
    readme_content = f"""
# 数据集处理说明

## 基本信息
- 原始文件: {os.path.basename(input_json_file)}
- 处理时间: {os.path.getctime(input_json_file)}
- 总数据量: {total_length}条
- 分割份数: {n}份

## 文件结构
{dataset_name}/
├── jsonl/           # JSONL文件目录
│   ├── data_000000.jsonl
│   ├── data_000001.jsonl
│   └── ...
└── images/          # 图像文件目录
    ├── data_000000.tar
    ├── data_000001.tar
    └── ...

## 使用说明
1. JSONL文件包含处理后的数据记录
2. 每个.tar文件包含对应JSONL文件中引用的图像
3. 图像已按规则重命名：分块ID_序列号.扩展名

## 处理统计
- 总数据条数: {total_length}
- 每份数据约: {len(chunks[0]) if chunks else 0}条
- 成功处理: {sum(len(chunk) for chunk in chunks)}条
    """

    readme_path = os.path.join(main_dir, "README.txt")
    # with open(readme_path, 'w', encoding='utf-8') as f:
    #     f.write(readme_content)

    print(f"\n处理完成！")
    print(f"输出目录: {main_dir}")
    print(f"JSONL文件: {len(os.listdir(jsonl_dir))}个")
    print(f"图像压缩包: {len([f for f in os.listdir(images_dir) if f.endswith('.tar')])}个")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_file = "infinigen_mmsibench_20251031/infinigen_mmsibench.json"  # 替换为你的JSON文件路径
    dataset_name = "Infinigen_MMSIBench"  # 数据集名称
    output_directory = "D:\Data\infinigen_20251031\\refresh"  # 输出目录（当前目录）
    image_base_dir = 'D:\Data\infinigen_20251031\infinigen_mmsibench_20251031/images'

    # 执行处理
    process_dataset(input_file, dataset_name, output_directory, image_base_dir)
