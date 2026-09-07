import json
import re
import random
from collections import Counter


def rebalance_jsonl_to_json(input_file, output_file):
    processed_data = []
    stats_by_count = {}
    exceptions = []

    # Pre-compiled regexes for performance.
    OPTIONS_BLOCK_RE = re.compile(r"(.*)\nOptions:\s*(.*)", re.DOTALL | re.IGNORECASE)
    # Matches "[letter]: content", using a lookahead so letters inside content are not split.
    OPT_PATTERN = re.compile(r"([A-D])[:：]\s*(.*?)(?=\s*[A-D][:：]|$)")
    # Extracts the leading answer letter from the raw answer text.
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
                        images[i] = '/path/to/cache/data/' + images[i]
                    images = [img.replace("/images", "") if 'structured3d' in img else img for img in images]

                # 1. Split the question body from the Options string.
                match = OPTIONS_BLOCK_RE.search(user_content)
                if not match:
                    exceptions.append({"idx": idx, "reason": "No 'Options:' tag found", "data": item})
                    continue

                question_body = match.group(1).strip()
                options_str = match.group(2).strip()

                # 2. Extract the correct answer letter (A/B/C/D) from the raw text.
                letter_match = ANSWER_LETTER_RE.match(raw_answer_text)
                if not letter_match:
                    exceptions.append({"idx": idx, "reason": f"Cannot find answer letter in: {raw_answer_text[:30]}", "data": item})
                    continue
                orig_correct_letter = letter_match.group(1).upper()

                # 3. Extract all options; keep a letter -> content map in case options are out of order.
                found_opts = OPT_PATTERN.findall(options_str)
                orig_options_map = {letter.upper(): content.strip().rstrip(',; ') for letter, content in found_opts}

                if orig_correct_letter not in orig_options_map:
                    exceptions.append({"idx": idx, "reason": f"Letter {orig_correct_letter} not in parsed options", "data": item})
                    continue

                # Lock the content of the correct answer.
                target_content = orig_options_map[orig_correct_letter]
                all_option_contents = list(orig_options_map.values())
                num_options = len(all_option_contents)

                if num_options < 2:
                    exceptions.append({"idx": idx, "reason": f"Parsed too few options ({num_options})", "data": item})
                    continue

                # 4. Shuffle the options and assign new letters.
                random.shuffle(all_option_contents)

                new_options_list = []
                new_correct_letter = ""
                letters_pool = ["A", "B", "C", "D"]

                for i, content in enumerate(all_option_contents):
                    current_letter = letters_pool[i]
                    new_options_list.append(f"{current_letter}: {content}")
                    # Recover the new letter by matching content.
                    if content == target_content:
                        new_correct_letter = current_letter

                # 5. Track letter distribution per option count.
                if num_options not in stats_by_count:
                    stats_by_count[num_options] = Counter()
                stats_by_count[num_options][new_correct_letter] += 1

                # 6. Build the new entry.
                new_options_str = ", ".join(new_options_list)
                new_item = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{question_body}\nOptions: {new_options_str}\n"
                        },
                    ],
                    "images": images,
                    "solution": f"{new_correct_letter}: {target_content}"
                }
                processed_data.append(new_item)

            except Exception as e:
                exceptions.append({"idx": idx, "reason": f"Runtime error: {str(e)}", "data": item})

    # 7. Write the final JSON file.
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 8. Print a quality report.
    print(f"\n{'=' * 50}")
    print(f"Done! Success: {len(processed_data)} | Exceptions: {len(exceptions)}")
    print(f"{'=' * 50}")

    for count in sorted(stats_by_count.keys()):
        print(f"\n[{count}-option type] Distribution:")
        total_type = sum(stats_by_count[count].values())
        for letter in ["A", "B", "C", "D"][:count]:
            c_val = stats_by_count[count][letter]
            print(f"  {letter}: {c_val} ({(c_val / total_type) * 100:.1f}%)")

    if exceptions:
        print(f"\n{'!' * 20} Exception details (first 5) {'!' * 20}")
        for ex in exceptions[:5]:
            print(f"Index {ex['idx']}: {ex['reason']}")


if __name__ == "__main__":
    input_path = "./output/SPAR_stage3_grpo_sampled_gemini-3-flash-preview_MCA.jsonl"
    output_path = "./output/SPAR_stage3_grpo_sampled_gemini-3-flash-preview_MCA_balanced.json"
    rebalance_jsonl_to_json(input_path, output_path)
