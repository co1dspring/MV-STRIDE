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


# --- Usage example ---
input_path = "./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00.json"
output_path = "./output/ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00_no_system.json"

if os.path.exists(input_path):
    clean_instruction_from_json(input_path, output_path)
else:
    print(f"File not found: {input_path}")
