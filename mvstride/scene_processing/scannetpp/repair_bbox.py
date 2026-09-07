import json
import re
import os
from pathlib import Path
from PIL import Image

# ================= Path mapping config =================
# Remote path prefix stored in the JSON
REMOTE_PREFIX = "/path/to/cache/scannetpp_sampled_modified"
# Local path prefix where the data actually lives
LOCAL_PREFIX = "/path/to/data/scannetpp/scannetpp_sampled_modified"
# ===============================================

def get_local_path(remote_path):
    """
    Convert a remote path to the local path.
    """
    return remote_path.replace(REMOTE_PREFIX, LOCAL_PREFIX)

def convert_to_1000_scale(text, width, height):
    """
    Convert every [x1, y1, x2, y2] in the text to 0-1000 coordinates
    using a fixed width and height.
    """
    # Match [x1, y1, x2, y2], allowing spaces.
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'

    def replace_func(match):
        # Extract the raw pixel coordinates.
        coords = [int(c) for c in match.groups()]

        # Compute normalized coordinates (0-1000).
        # Formula: (pixel / original_size) * 1000
        x1 = min(1000, round(coords[0] / width * 1000))
        y1 = min(1000, round(coords[1] / height * 1000))
        x2 = min(1000, round(coords[2] / width * 1000))
        y2 = min(1000, round(coords[3] / height * 1000))

        return f"[{x1}, {y1}, {x2}, {y2}]"

    return re.sub(pattern, replace_func, text)


def main(input_path, output_path):
    # Read the raw JSON input.
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} items in total...")

    processed_count = 0
    for item in data:
        # 1. Get the reference image resolution (first image).
        image_paths = item.get("images", [])
        if not image_paths:
            continue

        first_img_path = image_paths[0]
        first_img_path = get_local_path(first_img_path)

        if not os.path.exists(first_img_path):
            print(f"Skipping: image not found {first_img_path}")
            continue

        try:
            with Image.open(first_img_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"Error reading image {first_img_path}: {e}")
            continue

        # 2. Process all messages for this item.
        for msg in item.get("messages", []):
            if "content" in msg:
                msg["content"] = convert_to_1000_scale(msg["content"], w, h)

        processed_count += 1

    # 3. Save to the new JSON.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Done! Successfully processed {processed_count} items.")
    print(f"File saved to: {output_path}")

# Convert absolute coordinates to 0-1000 relative coordinates.
if __name__ == "__main__":
    # Configure your paths here.
    # INPUT_JSON = './api/output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned.json'
    # OUTPUT_JSON = './api/output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel.json'
    INPUT_JSON = '/path/to/data/QA_jsons_MultilevelCategories_sampled_MCA_Multistage_stage2.json'
    OUTPUT_JSON = '/path/to/data/QA_jsons_MultilevelCategories_sampled_MCA_Multistage_stage2_rel.json'

    main(INPUT_JSON, OUTPUT_JSON)
