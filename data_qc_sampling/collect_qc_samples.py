import os
import json
import random
import shutil
from pathlib import Path


def collect_qc_data(base_path, local_img_base, output_dir):
    # 初始化配置
    base_path = Path(base_path)
    local_img_base = Path(local_img_base)
    output_dir = Path(output_dir)
    img_output_dir = output_dir / "images"

    os.makedirs(img_output_dir, exist_ok=True)

    all_data = {"level_1": {}, "level_2": {}, "level_3": {}}

    # 1. 扫描并加载所有数据
    for level in ["level_1", "level_2", "level_3"]:
        level_dir = base_path / level
        if not level_dir.exists():
            continue

        for json_file in level_dir.glob("*.json"):
            # 跳过 Infinigen 打头的文件
            if json_file.name.startswith("Infinigen"):
                continue

            category = json_file.stem
            with open(json_file, 'r', encoding='utf-8') as f:
                data_list = json.load(f)

            # 为每条数据注入 level 标识
            for item in data_list:
                item["level"] = level

            if category not in all_data[level]:
                all_data[level][category] = []
            all_data[level][category].extend(data_list)

    sampled_data = []

    level3_data = all_data.get("level_3", {})

    # 分类：MSR类别 和 其他类别
    msr_categories = {cat: items for cat, items in level3_data.items() if cat.startswith("MSR")}
    other_categories = {cat: items for cat, items in level3_data.items() if not cat.startswith("MSR")}

    # A. MSR 类别：每类固定采样 2 条
    for cat, items in msr_categories.items():
        count = min(len(items), 2)
        sampled_data.extend(random.sample(items, count))

    # B. 计算剩余配额
    current_count = len(sampled_data)
    target_total = 100
    remaining_needed = target_total - current_count

    if remaining_needed > 0 and other_categories:
        # 计算其他类别每个类应该分担多少条
        other_cat_names = list(other_categories.keys())
        num_other_cats = len(other_cat_names)

        # 基础平分量
        base_per_cat = remaining_needed // num_other_cats
        # 余数（用于分配给前几个类别，确保凑够100）
        extra_seats = remaining_needed % num_other_cats

        for i, cat in enumerate(other_cat_names):
            items = other_categories[cat]
            # 该类应采数量 = 基础平分量 + (如果是前 N 个则多采 1 条)
            take_count = base_per_cat + (1 if i < extra_seats else 0)

            # 安全检查：防止该类总数不足
            actual_take = min(len(items), take_count)
            sampled_data.extend(random.sample(items, actual_take))

    # C. 最终兜底 (如果因为某些类数据太少没凑够100，从所有 L3 剩余数据中补齐)
    if len(sampled_data) < target_total:
        current_ids = {id(item) for item in sampled_data}
        l3_pool = [item for cat_items in level3_data.values() for item in cat_items
                   if id(item) not in current_ids]

        needed = target_total - len(sampled_data)
        if needed > 0 and l3_pool:
            sampled_data.extend(random.sample(l3_pool, min(len(l3_pool), needed)))

    random.shuffle(sampled_data)  # 打乱顺序

    # 4. 图像处理与路径更新
    final_json_data = []
    for item in sampled_data:
        new_item = item.copy()
        new_images = []

        for old_path_str in item.get("images", []):
            # 路径转换逻辑
            # 原路径: /cache/xj/.../saved_scenes/9db2f3d/Image_5_0_0048_0.png
            # 提取场景名和文件名部分
            path_parts = Path(old_path_str).parts
            scene_name = path_parts[-2]  # 9db2f3d
            img_filename = path_parts[-1]  # Image_5_0_0048_0.png

            # 拼接本地真实源路径
            # ../infinigen_metadata_ver2/saved_scenes/9db2f3d/Image_5_0_0048_0.png
            src_img_path = local_img_base / "saved_scenes" / scene_name / img_filename

            # 目标文件名与路径
            new_img_name = f"{scene_name}_{img_filename}"
            dest_img_path = img_output_dir / new_img_name

            # 物理拷贝并重命名
            if src_img_path.exists():
                shutil.copy2(src_img_path, dest_img_path)
                # 更新为相对路径
                new_images.append(f"images/{new_img_name}")
            else:
                print(f"Warning: Image not found at {src_img_path}")
                new_images.append(None)

        new_item["images"] = new_images
        final_json_data.append(new_item)

    # 5. 保存汇总结果
    output_json = output_dir / "qc_samples_100.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_json_data, f, indent=4, ensure_ascii=False)

    print(f"完成！采样数据量: {len(final_json_data)}")
    print(f"汇总文件已保存至: {output_json}")


# --- 使用示例 ---
if __name__ == "__main__":
    config = {
        "base_path": "../QA_jsons_Infinigen_MultilevelCategories_20260313_sampled_MCA_Multistage/atomic",  # 包含 level_1~3 的路径
        "local_img_base": "../infinigen_metadata_ver2",  # 真实的本地图像基准目录
        "output_dir": "./qc_task_v1"  # 采样结果存放目录
    }

    collect_qc_data(**config)
