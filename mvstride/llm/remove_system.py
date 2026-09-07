import json
import os


def clean_instruction_from_json(input_file, output_file):
    # The exact instruction string to remove from user messages.
    target_instruction = "Output your step-by-step thinking process in  thinking  response tags and the final choice (e.g., A: option) in <answer> </answer> tags."

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_count = 0

    for item in data:
        if "messages" in item:
            for message in item["messages"]:
                if message["role"] == "user":
                    content = message["content"]
                    if target_instruction in content:
                        # Remove the instruction and strip any trailing whitespace/newlines.
                        new_content = content.replace(target_instruction, "").strip()
                        message["content"] = new_content
                        cleaned_count += 1

    # Save the cleaned file.
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Done! Processed {cleaned_count} user messages.")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="从 CoT 数据的 user 消息中删除 step-by-step 系统提示（'no system' 消融变体）")
    parser.add_argument("--input-file",
                        default="./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00.json",
                        help="cot_jsonl2swift 清洗后的输入文件")
    parser.add_argument("--output-file",
                        default="./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00_no_system.json",
                        help="去除 system 提示后的输出文件")
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        clean_instruction_from_json(args.input_file, args.output_file)
    else:
        print(f"File not found: {args.input_file}")
