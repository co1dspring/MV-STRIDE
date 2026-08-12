import os
import json
import random
import math
from collections import defaultdict

# =========================
# 配置部分
# =========================

# 两个数据文件夹
DATA_DIRS = [
    "../QA_jsons_Infinigen_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_3",
    "../QA_jsons_Infinigen_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_2",
    "../QA_jsons_Infinigen_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_1",
    "../QA_jsons_ScannetppIphone_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_3",
    "../QA_jsons_ScannetppIphone_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_2",
    "../QA_jsons_ScannetppIphone_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_1"
]

# 输出文件
OUTPUT_MULTI_VIEW = "./human_eval/human_eval_sampled_multiview_200.json"
# OUTPUT_SINGLE_VIEW = "./cross_view_dependency/cross_view_dependency_sampled_singleview_1000.json"
os.makedirs(os.path.dirname(OUTPUT_MULTI_VIEW), exist_ok=True)
# os.makedirs(os.path.dirname(OUTPUT_SINGLE_VIEW), exist_ok=True)

# 总采样数
TOTAL_SAMPLES = 200

# 随机种子（保证可复现）
SEED = 42
random.seed(SEED)

# 需要过滤的前缀
EXCLUDE_PREFIXES = ("Infinigen", "Scannetpp")


# =========================
# Step1: 收集所有合法 json 文件
# =========================

json_files = []

for data_dir in DATA_DIRS:
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".json"):
                continue

            # 过滤前缀
            if file.startswith(EXCLUDE_PREFIXES):
                continue

            json_files.append(os.path.join(root, file))

print(f"Found {len(json_files)} valid json files.")


# =========================
# Step2: 读取每个文件的数据
# =========================

file_data = {}
file_lengths = {}

total_entries = 0

for path in json_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Skip non-list json: {path}")
            continue

        file_data[path] = data
        file_lengths[path] = len(data)
        total_entries += len(data)

    except Exception as e:
        print(f"Error reading {path}: {e}")

print(f"Total entries across all files: {total_entries}")


# =========================
# Step3: 按比例采样
# =========================

sampled_multiview = []

for path, data in file_data.items():

    ratio = len(data) / total_entries
    sample_num = max(1, round(ratio * TOTAL_SAMPLES))

    # 防止超过原数据量
    sample_num = min(sample_num, len(data))

    sampled = random.sample(data, sample_num)

    sampled_multiview.extend(sampled)

print(f"Initial sampled size: {len(sampled_multiview)}")


# =========================
# Step4: 修正到严格1000条
# =========================

if len(sampled_multiview) > TOTAL_SAMPLES:
    sampled_multiview = random.sample(sampled_multiview, TOTAL_SAMPLES)

elif len(sampled_multiview) < TOTAL_SAMPLES:

    remaining = TOTAL_SAMPLES - len(sampled_multiview)

    all_remaining = []

    sampled_ids = set(id(x) for x in sampled_multiview)

    for data in file_data.values():
        for item in data:
            if id(item) not in sampled_ids:
                all_remaining.append(item)

    extra = random.sample(
        all_remaining,
        min(remaining, len(all_remaining))
    )

    sampled_multiview.extend(extra)

print(f"Final multiview sample size: {len(sampled_multiview)}")


# =========================
# Step5: 保存 multi-view 数据
# =========================

with open(OUTPUT_MULTI_VIEW, "w", encoding="utf-8") as f:
    json.dump(sampled_multiview, f, indent=2, ensure_ascii=False)

print(f"Saved multiview dataset to {OUTPUT_MULTI_VIEW}")


# =========================
# Step6: 构造 single-view 数据
# =========================

# sampled_singleview = []
#
# for item in sampled_multiview:
#
#     # 深拷贝
#     new_item = json.loads(json.dumps(item))
#
#     images = new_item.get("images", [])
#
#     if len(images) > 0:
#
#         # 随机保留一个图像
#         selected_image = random.choice(images)
#
#         new_item["images"] = [selected_image]
#
#     # 修改 messages[0]["content"]
#     try:
#         content = new_item["messages"][0]["content"]
#
#         # 统计原始 <image> 数量
#         image_count = content.count("<image>")
#
#         if image_count > 1:
#             # 全部移除后只保留一个
#             text_part = content.replace("<image>", "")
#             new_content = "<image>" + text_part
#
#             new_item["messages"][0]["content"] = new_content
#
#     except Exception as e:
#         print(f"Error processing content: {e}")
#
#     sampled_singleview.append(new_item)
#
# print(f"Single-view dataset size: {len(sampled_singleview)}")
#
#
# # =========================
# # Step7: 保存 single-view 数据
# # =========================
#
# with open(OUTPUT_SINGLE_VIEW, "w", encoding="utf-8") as f:
#     json.dump(sampled_singleview, f, indent=2, ensure_ascii=False)
#
# print(f"Saved single-view dataset to {OUTPUT_SINGLE_VIEW}")
