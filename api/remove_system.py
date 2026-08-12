import json
import os


def clean_instruction_from_json(input_file, output_file):
    # 你要删除的精确指令字符串
    target_instruction = "Output your step-by-step thinking process in <think> </think> tags and the final choice (e.g., A: option) in <answer> </answer> tags."

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_count = 0

    for item in data:
        if "messages" in item:
            for message in item["messages"]:
                if message["role"] == "user":
                    content = message["content"]
                    if target_instruction in content:
                        # 替换指令为空，并去除可能残留在末尾的换行符或空格
                        new_content = content.replace(target_instruction, "").strip()
                        message["content"] = new_content
                        cleaned_count += 1

    # 保存清洗后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"清洗完成！共处理了 {cleaned_count} 条 user 消息。")
    print(f"结果已保存至: {output_file}")


# --- 使用示例 ---
input_path = "./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00.json"  # 你的原始文件名
output_path = "./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00_no_system.json"  # 清洗后的文件名

if os.path.exists(input_path):
    clean_instruction_from_json(input_path, output_path)
else:
    print(f"找不到文件: {input_path}")
