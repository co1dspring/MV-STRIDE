import json
import re
import os
from pathlib import Path
from PIL import Image

# ================= 配置路径映射 =================
# JSON 中原始的路径前缀
REMOTE_PREFIX = "/cache/xj/data/scannetpp_sampled_modified"
# 你本地实际存放数据的路径前缀
LOCAL_PREFIX = "D:/Data/scannetpp/scannetpp_sampled_modified"
# ===============================================

def get_local_path(remote_path):
    """
    将远程路径转换为本地路径
    """
    return remote_path.replace(REMOTE_PREFIX, LOCAL_PREFIX)

def convert_to_1000_scale(text, width, height):
    """
    使用固定的宽高将文本中的所有 [x1, y1, x2, y2] 转换为 0-1000 坐标
    """
    # 匹配 [x1, y1, x2, y2] 格式，支持空格
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'

    def replace_func(match):
        # 提取原始像素坐标
        coords = [int(c) for c in match.groups()]

        # 计算归一化坐标 (0-1000)
        # 公式: (pixel / original_size) * 1000
        x1 = min(1000, round(coords[0] / width * 1000))
        y1 = min(1000, round(coords[1] / height * 1000))
        x2 = min(1000, round(coords[2] / width * 1000))
        y2 = min(1000, round(coords[3] / height * 1000))

        return f"[{x1}, {y1}, {x2}, {y2}]"

    return re.sub(pattern, replace_func, text)


def main(input_path, output_path):
    # 读取原始 JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"开始处理，总计 {len(data)} 条数据...")

    processed_count = 0
    for item in data:
        # 1. 获取参考图像的分辨率（取第一张图）
        image_paths = item.get("images", [])
        if not image_paths:
            continue

        first_img_path = image_paths[0]
        first_img_path = get_local_path(first_img_path)

        if not os.path.exists(first_img_path):
            print(f"跳过：找不到图像 {first_img_path}")
            continue

        try:
            with Image.open(first_img_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"读取图像出错 {first_img_path}: {e}")
            continue

        # 2. 处理该条数据下所有的 messages
        for msg in item.get("messages", []):
            if "content" in msg:
                msg["content"] = convert_to_1000_scale(msg["content"], w, h)

        processed_count += 1

    # 3. 保存到新 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"处理完成！成功处理 {processed_count} 条数据。")
    print(f"文件已保存至: {output_path}")

# 绝对坐标转0-1000相对坐标
if __name__ == "__main__":
    # 配置你的路径
    # INPUT_JSON = './api/output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned.json'
    # OUTPUT_JSON = './api/output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel.json'
    INPUT_JSON = 'D:/Data/infinigen_20251031/QA_jsons_ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2.json'
    OUTPUT_JSON = 'D:/Data/infinigen_20251031/QA_jsons_ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_rel.json'

    main(INPUT_JSON, OUTPUT_JSON)
