import os
import json
import random
import math
from collections import defaultdict

# =========================
# Configuration
# =========================

# The two data folders
DATA_DIRS = [
    "../QA_jsons_Infinigen_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_3",
    "../QA_jsons_Infinigen_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_2",
    "../QA_jsons_Infinigen_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_1",
    "../QA_jsons_ScannetppIphone_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_3",
    "../QA_jsons_ScannetppIphone_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_2",
    "../QA_jsons_ScannetppIphone_MultilevelCategories_20260507_sampled_MCA_Multistage_crossviewdependency/atomic/level_1"
]

# Output files
OUTPUT_MULTI_VIEW = "./human_eval/human_eval_sampled_multiview_200.json"
# OUTPUT_SINGLE_VIEW = "./cross_view_dependency/cross_view_dependency_sampled_singleview_1000.json"
os.makedirs(os.path.dirname(OUTPUT_MULTI_VIEW), exist_ok=True)
# os.makedirs(os.path.dirname(OUTPUT_SINGLE_VIEW), exist_ok=True)

# Total number of samples
TOTAL_SAMPLES = 200

# Random seed (for reproducibility)
SEED = 42
random.seed(SEED)

# Prefixes to filter out
EXCLUDE_PREFIXES = ("Infinigen", "Scannetpp")


# =========================
# Step1: collect all valid json files
# =========================

json_files = []

for data_dir in DATA_DIRS:
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".json"):
                continue

            # Filter by prefix
            if file.startswith(EXCLUDE_PREFIXES):
                continue

            json_files.append(os.path.join(root, file))

print(f"Found {len(json_files)} valid json files.")


# =========================
# Step2: read the data from each file
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
# Step3: sample proportionally
# =========================

sampled_multiview = []

for path, data in file_data.items():

    ratio = len(data) / total_entries
    sample_num = max(1, round(ratio * TOTAL_SAMPLES))

    # Avoid exceeding the original data size
    sample_num = min(sample_num, len(data))

    sampled = random.sample(data, sample_num)

    sampled_multiview.extend(sampled)

print(f"Initial sampled size: {len(sampled_multiview)}")


# =========================
# Step4: correct to exactly 1000 items
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
# Step5: save the multi-view data
# =========================

with open(OUTPUT_MULTI_VIEW, "w", encoding="utf-8") as f:
    json.dump(sampled_multiview, f, indent=2, ensure_ascii=False)

print(f"Saved multiview dataset to {OUTPUT_MULTI_VIEW}")


# =========================
# Step6: construct the single-view data
# =========================

# sampled_singleview = []
#
# for item in sampled_multiview:
#
#     # Deep copy
#     new_item = json.loads(json.dumps(item))
#
#     images = new_item.get("images", [])
#
#     if len(images) > 0:
#
#         # Keep one random image
#         selected_image = random.choice(images)
#
#         new_item["images"] = [selected_image]
#
#     # Modify messages[0]["content"]
#     try:
#         content = new_item["messages"][0]["content"]
#
#         # Count the original <image> occurrences
#         image_count = content.count("<image>")
#
#         if image_count > 1:
#             # Remove all but keep one
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
# # Step7: save the single-view data
# # =========================
#
# with open(OUTPUT_SINGLE_VIEW, "w", encoding="utf-8") as f:
#     json.dump(sampled_singleview, f, indent=2, ensure_ascii=False)
#
# print(f"Saved single-view dataset to {OUTPUT_SINGLE_VIEW}")
