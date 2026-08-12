import json
import re
import random
from collections import Counter


def rebalance_jsonl_to_json(input_file, output_file):
    processed_data = []
    stats_by_count = {}
    exceptions = []

    # 预编译正则表达式以提高性能
    # 匹配 Options 块的正则
    OPTIONS_BLOCK_RE = re.compile(r"(.*)\nOptions:\s*(.*)", re.DOTALL | re.IGNORECASE)
    # 终极选项解析正则：匹配 [字母][冒号]，并利用正向预查确保不会被内容里的字母误导
    OPT_PATTERN = re.compile(r"([A-D])[:：]\s*(.*?)(?=\s*[A-D][:：]|$)")
    # 提取回答中首个字母的正则
    ANSWER_LETTER_RE = re.compile(r"^\s*([A-D])\s*[:：]?")

    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue

            try:
                item = json.loads(line)
                user_content = item['messages'][0]['content']
                raw_answer_text = item['messages'][1]['content'].strip()
                images = item.get('images', [])
                if 'SPAR' in input_file:
                    for i in range(len(images)):
                        images[i] = '/cache/hxj/data/'+images[i]
                    images = [img.replace("/images", "") if 'structured3d' in img else img for img in images]

                # 1. 分离问题主体和 Options 字符串
                match = OPTIONS_BLOCK_RE.search(user_content)
                if not match:
                    exceptions.append({"idx": idx, "reason": "No 'Options:' tag found", "data": item})
                    continue

                question_body = match.group(1).strip()
                options_str = match.group(2).strip()

                # 2. 提取原始回答中的正确字母 (A/B/C/D)
                letter_match = ANSWER_LETTER_RE.match(raw_answer_text)
                if not letter_match:
                    exceptions.append({"idx": idx, "reason": f"Cannot find answer letter in: {raw_answer_text[:30]}", "data": item})
                    continue
                orig_correct_letter = letter_match.group(1).upper()

                # 3. 提取所有选项内容 (核心：正向预查防止误切内容里的字母)
                # findall 返回的是 [(字母, 内容), ...] 列表
                found_opts = OPT_PATTERN.findall(options_str)

                # 将选项存入列表，同时记录字母到内容的映射，以防乱序
                # 即使 Options 字符串是 "A:.. C:.. B:.." 这种乱序也能处理
                orig_options_map = {letter.upper(): content.strip().rstrip(',; ') for letter, content in found_opts}

                if orig_correct_letter not in orig_options_map:
                    exceptions.append({"idx": idx, "reason": f"Letter {orig_correct_letter} not in parsed options", "data": item})
                    continue

                # 锁定正确答案的内容字符串
                target_content = orig_options_map[orig_correct_letter]
                # 获取所有选项的内容供洗牌
                all_option_contents = list(orig_options_map.values())
                num_options = len(all_option_contents)

                if num_options < 2:
                    exceptions.append({"idx": idx, "reason": f"Parsed too few options ({num_options})", "data": item})
                    continue

                # 4. 重新洗牌并分配新字母
                random.shuffle(all_option_contents)

                new_options_list = []
                new_correct_letter = ""
                letters_pool = ["A", "B", "C", "D"]

                for i, content in enumerate(all_option_contents):
                    current_letter = letters_pool[i]
                    new_options_list.append(f"{current_letter}: {content}")
                    # 通过内容一致性找回新字母
                    if content == target_content:
                        new_correct_letter = current_letter

                # 5. 统计分布
                if num_options not in stats_by_count:
                    stats_by_count[num_options] = Counter()
                stats_by_count[num_options][new_correct_letter] += 1

                # 6. 构造新条目
                new_options_str = ", ".join(new_options_list)
                new_item = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{question_body}\nOptions: {new_options_str}\n"
                        },
                        # {
                        #     "role": "assistant",
                        #     # 保持 "字母: 内容" 的标准格式
                        #     "content": f"{new_correct_letter}: {target_content}"
                        # }
                    ],
                    "images": images,
                    "solution": f"{new_correct_letter}: {target_content}"
                }
                processed_data.append(new_item)

            except Exception as e:
                exceptions.append({"idx": idx, "reason": f"Runtime error: {str(e)}", "data": item})

    # 7. 写入最终 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 8. 打印质量报告
    print(f"\n{'=' * 50}")
    print(f"处理完成！成功: {len(processed_data)} 条 | 异常: {len(exceptions)} 条")
    print(f"{'=' * 50}")

    for count in sorted(stats_by_count.keys()):
        print(f"\n[{count} 选项题型] 分布统计:")
        total_type = sum(stats_by_count[count].values())
        for letter in ["A", "B", "C", "D"][:count]:
            c_val = stats_by_count[count][letter]
            print(f"  {letter}: {c_val} ({(c_val / total_type) * 100:.1f}%)")

    if exceptions:
        print(f"\n{'!' * 20} 异常详情 (前5条) {'!' * 20}")
        for ex in exceptions[:5]:
            print(f"索引 {ex['idx']}: {ex['reason']}")


if __name__ == "__main__":
    # input_path = "./output/SAT_stage3_grpo_sampled_gemini-3-flash-preview_MCA.jsonl"
    input_path = "./output/SPAR_stage3_grpo_sampled_gemini-3-flash-preview_MCA.jsonl"
    # output_path = "./output/SAT_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.json"
    output_path = "./output/SPAR_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.json"
    rebalance_jsonl_to_json(input_path, output_path)
