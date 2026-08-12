import json
from pathlib import Path

def expand_json_interleaved(file_path, factor=10):
    """
    将 JSON 列表中的每一条数据连续复制 factor 次。
    例如: [A, B] -> [A, A, A... (10个), B, B, B... (10个)]
    """
    input_path = Path(file_path)
    if not input_path.exists():
        print(f"错误: 找不到文件 {file_path}")
        return

    # 1. 读取数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("错误: 该脚本仅支持处理 JSON 列表格式 ([...])")
        return

    # 2. 扩增数据 (交错/连续复制逻辑)
    expanded_data = []
    for item in data:
        # 每处理一条原数据，就连续往新列表中添加 10 次该对象的副本
        for _ in range(factor):
            # 使用 .copy() 是为了防止后续修改其中一个副本时影响到其他副本
            # 如果是深度嵌套字典，建议用 copy.deepcopy(item)
            expanded_data.append(item.copy() if isinstance(item, dict) else item)

    # 3. 生成新文件名
    output_path = input_path.parent / f"{input_path.stem}_x10{input_path.suffix}"

    # 4. 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_data, f, indent=4, ensure_ascii=False)

    print(f"处理完成！")
    print(f"原始条数: {len(data)}")
    print(f"扩增后条数: {len(expanded_data)}")
    print(f"排列方式: 每条原始数据连续重复 {factor} 次")
    print(f"保存路径: {output_path}")

if __name__ == "__main__":
    # target_file = r"D:\Data\MMSIBench\MMSIBench.json"
    target_file = r"D:\Data\infinigen_20251031\api\output\Infinigen_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.json"
    # target_file = r"D:\Data\infinigen_20251031\api\output\ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage_stage3_gemini-3-flash-preview_MCA.json"
    expand_json_interleaved(target_file, factor=10)
