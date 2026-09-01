import json
from pathlib import Path

def expand_json_interleaved(file_path, factor=10):
    """
    Repeatedly copy each item in the JSON list factor times.
    E.g. [A, B] -> [A, A, A... (x10), B, B, B... (x10)]
    """
    input_path = Path(file_path)
    if not input_path.exists():
        print(f"Error: file not found {file_path}")
        return

    # 1. Read the data.
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("错误: 该脚本仅支持处理 JSON 列表格式 ([...])")
        return

    # 2. Expand the data (interleaved/consecutive copy logic).
    expanded_data = []
    for item in data:
        # For each original item, append factor copies consecutively.
        for _ in range(factor):
            # Using .copy() prevents later edits to one copy from affecting the others.
            # For deeply nested dicts, prefer copy.deepcopy(item).
            expanded_data.append(item.copy() if isinstance(item, dict) else item)

    # 3. Build the new filename.
    output_path = input_path.parent / f"{input_path.stem}_x10{input_path.suffix}"

    # 4. Write to file.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_data, f, indent=4, ensure_ascii=False)

    print(f"Done!")
    print(f"Original count: {len(data)}")
    print(f"Expanded count: {len(expanded_data)}")
    print(f"Layout: each item repeated {factor} times consecutively")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    target_file = "/path/to/data/api/output/Infinigen_MultilevelCategories_sampled_MCA_Multistage_stage3_MCA.json"
    expand_json_interleaved(target_file, factor=10)
