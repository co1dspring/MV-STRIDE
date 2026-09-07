# -*- coding = utf-8 -*-
import json
import re
from pathlib import Path
from collections import Counter


def analyze_verification_results(jsonl_path: str):
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Result file not found: {path}")
        return

    total_count = 0
    correct_count = 0
    error_counter = Counter()
    unclassified_count = 0

    # The expected set of standard labels.
    standard_labels = {
        "Correct",
        "Factual inconsistency",
        "Reasoning unfaithfulness",
        "Final-answer inconsistency",
        "Hallucination"
    }

    # Regex to extract the content inside <answer> tags.
    answer_pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

    print(f"Parsing file: {path.name} ...")

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)

                # 1. Prefer the saved verification_result; fall back to the raw response.
                raw_result = item.get("verification_result", "")
                if not raw_result and "response_lst" in item:
                    raw_result = item["response_lst"]

                raw_result = str(raw_result).strip()

                # 2. Extract the content inside <answer>.
                ans_match = answer_pattern.search(raw_result)
                if ans_match:
                    result = ans_match.group(1).strip()
                else:
                    # Fallback: match the whole text if no <answer> tag is found.
                    result = raw_result

                # 3. Basic cleaning (strip stray punctuation and prefixes).
                result = result.replace(".", "").replace('"', '').replace("'", "").strip()
                result = re.sub(r"^Verdict:\s*", "", result, flags=re.IGNORECASE).strip()

                if not result:
                    unclassified_count += 1
                    continue

                total_count += 1

                # 4. Map to a standard label.
                if result.lower() == "correct":
                    correct_count += 1
                elif result in standard_labels:
                    error_counter[result] += 1
                else:
                    # Handle minor case/format mismatches.
                    matched = False
                    for standard_label in standard_labels:
                        if result.lower() == standard_label.lower():
                            if standard_label == "Correct":
                                correct_count += 1
                            else:
                                error_counter[standard_label] += 1
                            matched = True
                            break

                    if not matched:
                        # Unmappable: classify as unknown (show first 30 chars for inspection).
                        error_counter[f"Unknown ({result[:30]})"] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse line {line_num}, skipped. Error: {e}")

    if total_count == 0:
        print("No valid statistics found!")
        return

    accuracy = (correct_count / total_count) * 100
    total_errors = total_count - correct_count

    # Print the statistical report.
    print("\n" + "=" * 50)
    print("       Multi-view CoT quality verification report")
    print("=" * 50)
    print("Data scale:")
    print(f"  - Verified samples: {total_count}")
    print(f"  - Correct: {correct_count}")
    print(f"  - With errors: {total_errors}")
    if unclassified_count > 0:
        print(f"  - Empty results: {unclassified_count} (not counted)")

    print("-" * 50)
    print("Core metric:")
    print(f"  - CoT accuracy: {accuracy:.2f}%")
    print("-" * 50)

    print("Error breakdown:")
    if total_errors == 0:
        print("  - No reasoning errors detected.")
    else:
        sorted_errors = error_counter.most_common()

        print(f"  {'Error dimension':<30} | {'Count':<6} | {'% of total':<14} | {'% of errors'}")
        print("  " + "-" * 75)

        for err_name, count in sorted_errors:
            pct_of_total = (count / total_count) * 100
            pct_of_errors = (count / total_errors) * 100
            print(f"  {err_name:<30} | {count:<6} | {pct_of_total:>12.2f}% | {pct_of_errors:>10.2f}%")

        print("\nError distribution histogram:")
        print("  " + "-" * 50)
        max_bar_length = 30
        for err_name, count in sorted_errors:
            bar_length = int((count / total_errors) * max_bar_length)
            bar = "#" * bar_length + "." * (max_bar_length - bar_length)
            pct_of_errors = (count / total_errors) * 100
            print(f"  {err_name:<26} {bar} {pct_of_errors:.1f}%")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Replace with the actual saved jsonl result path.
    RESULT_FILE_PATH = "./output/cot_with_original_aligned_sampled_200_gpt-5.5_CoT.jsonl"

    analyze_verification_results(RESULT_FILE_PATH)
