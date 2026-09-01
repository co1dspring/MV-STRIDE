import os
import json
import random
import shutil
from pathlib import Path


def collect_qc_data(base_path, local_img_base, output_dir):
    # Initialize config.
    base_path = Path(base_path)
    local_img_base = Path(local_img_base)
    output_dir = Path(output_dir)
    img_output_dir = output_dir / "images"

    os.makedirs(img_output_dir, exist_ok=True)

    all_data = {"level_1": {}, "level_2": {}, "level_3": {}}

    # 1. Scan and load all data.
    for level in ["level_1", "level_2", "level_3"]:
        level_dir = base_path / level
        if not level_dir.exists():
            continue

        for json_file in level_dir.glob("*.json"):
            # Skip files starting with Infinigen.
            if json_file.name.startswith("Infinigen"):
                continue

            category = json_file.stem
            with open(json_file, 'r', encoding='utf-8') as f:
                data_list = json.load(f)

            # Attach a level tag to each item.
            for item in data_list:
                item["level"] = level

            if category not in all_data[level]:
                all_data[level][category] = []
            all_data[level][category].extend(data_list)

    sampled_data = []

    level3_data = all_data.get("level_3", {})

    # Split into MSR and other categories.
    msr_categories = {cat: items for cat, items in level3_data.items() if cat.startswith("MSR")}
    other_categories = {cat: items for cat, items in level3_data.items() if not cat.startswith("MSR")}

    # A. MSR category: fixed 2 samples per class.
    for cat, items in msr_categories.items():
        count = min(len(items), 2)
        sampled_data.extend(random.sample(items, count))

    # B. Compute the remaining quota.
    current_count = len(sampled_data)
    target_total = 100
    remaining_needed = target_total - current_count

    if remaining_needed > 0 and other_categories:
        # How many samples each remaining class should take.
        other_cat_names = list(other_categories.keys())
        num_other_cats = len(other_cat_names)

        # Base equal share.
        base_per_cat = remaining_needed // num_other_cats
        # Remainder (assigned to the first few classes to reach 100).
        extra_seats = remaining_needed % num_other_cats

        for i, cat in enumerate(other_cat_names):
            items = other_categories[cat]
            # Sample count = base share + (1 extra if among the first N).
            take_count = base_per_cat + (1 if i < extra_seats else 0)

            # Safety check: the class must have enough samples.
            actual_take = min(len(items), take_count)
            sampled_data.extend(random.sample(items, actual_take))

    # C. Fallback: top up from all remaining L3 data if still under 100.
    if len(sampled_data) < target_total:
        current_ids = {id(item) for item in sampled_data}
        l3_pool = [item for cat_items in level3_data.values() for item in cat_items
                   if id(item) not in current_ids]

        needed = target_total - len(sampled_data)
        if needed > 0 and l3_pool:
            sampled_data.extend(random.sample(l3_pool, min(len(l3_pool), needed)))

    random.shuffle(sampled_data)  # shuffle the order

    # 4. Image processing and path updates.
    final_json_data = []
    for item in sampled_data:
        new_item = item.copy()
        new_images = []

        for old_path_str in item.get("images", []):
            # Path conversion logic.
            # Original path: /path/to/cache/data/.../Image_5_0_0048_0.png
            # Extract the scene name and filename parts.
            path_parts = Path(old_path_str).parts
            scene_name = path_parts[-2]  # 9db2f3d
            img_filename = path_parts[-1]  # Image_5_0_0048_0.png

            # Build the local source path.
            # ../infinigen_metadata_ver2/saved_scenes/9db2f3d/Image_5_0_0048_0.png
            src_img_path = local_img_base / "saved_scenes" / scene_name / img_filename

            # Target filename and path.
            new_img_name = f"{scene_name}_{img_filename}"
            dest_img_path = img_output_dir / new_img_name

            # Physically copy and rename.
            if src_img_path.exists():
                shutil.copy2(src_img_path, dest_img_path)
                # Update to a relative path.
                new_images.append(f"images/{new_img_name}")
            else:
                print(f"Warning: Image not found at {src_img_path}")
                new_images.append(None)

        new_item["images"] = new_images
        final_json_data.append(new_item)

    # 5. Save the aggregated results.
    output_json = output_dir / "qc_samples_100.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_json_data, f, indent=4, ensure_ascii=False)

    print(f"Done! Sampled {len(final_json_data)} items.")
    print(f"Summary saved to: {output_json}")


# --- Usage example ---
if __name__ == "__main__":
    config = {
        "base_path": "../QA_jsons_Infinigen_MultilevelCategories_sampled_MCA_Multistage/atomic",  # path containing level_1~3
        "local_img_base": "../infinigen_metadata_ver2",  # real local image base directory
        "output_dir": "./qc_task_v1"  # directory for sampling results
    }

    collect_qc_data(**config)
