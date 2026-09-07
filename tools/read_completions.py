import json
import os
import re

def full_diagnostic_jsonl(file_path):
    try:
        parent_dir = os.path.dirname(file_path)
        output_report = os.path.join(parent_dir, "rl_looping_samples_report.txt")

        with open(file_path, 'r', encoding='utf-8') as f, open(output_report, 'w', encoding='utf-8') as out:
            print(f"🚀 开始精准提取回环样本... \n📂 报告路径: {output_report}")

            row_count = sum(1 for _ in f)
            f.seek(0)

            # Regex to detect loops: a 5-50 character fragment repeated 3+ times
            # Loosen a bit to catch patterns like "D: option"
            loop_pattern = re.compile(r"(.{5,50})\1{2,}")

            out.write("📋 【所有回环样本详细记录】\n")
            out.write("="*60 + "\n\n")

            total_loops_found = 0

            for line_idx, line in enumerate(f):
                data = json.loads(line)
                completions = data.get('completion', [])
                accuracies = data.get('MultiModalAccuracyORM', [])
                prompts = data.get('prompt', []) # include if a prompt field exists

                current_step_loops = []

                for i, c in enumerate(completions):
                    match = loop_pattern.search(c)
                    if match:
                        total_loops_found += 1
                        current_step_loops.append((i, c, match.group(1)))

                if current_step_loops:
                    out.write(f"🚩 Step {line_idx} 发现 {len(current_step_loops)} 个回环样本：\n")
                    for idx, content, pattern in current_step_loops:
                        score = accuracies[idx] if idx < len(accuracies) else "N/A"
                        out.write(f"  - 样本索引: {idx} | 准确率得分: {score} | 长度: {len(content)}\n")
                        out.write(f"  - 检测到的重复模式: \"{pattern}\"\n")
                        out.write(f"  - 末尾 300 字符展示:\n")
                        out.write(f"    [...] {content[-300:]}\n")
                        out.write(f"  {'-'*40}\n")
                    out.write("\n")

                if line_idx % 20 == 0:
                    print(f"进度: {line_idx}/{row_count} | 已累计发现回环样本: {total_loops_found}")

            out.write(f"\n📊 诊断结束。全量数据中总计发现回环样本: {total_loops_found} 条。\n")
            print(f"✅ 诊断报告生成成功！总计发现 {total_loops_found} 条回环。")

    except Exception as e:
        print(f"❌ 运行出错: {e}")


# Run
file_path = r"/path/to/data/completions.jsonl"
full_diagnostic_jsonl(file_path)
