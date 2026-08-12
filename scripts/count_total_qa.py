import os


def count_total_qa(jsonl_dir):
    """
    统计指定目录下所有 jsonl 文件的 QA 总数
    """
    if not os.path.exists(jsonl_dir):
        print(f"错误: 目录不存在 -> {jsonl_dir}")
        return

    total_qa_count = 0
    file_count = 0

    print("开始统计 JSONL 文件...")
    print("-" * 40)

    # 遍历目录下所有文件
    for filename in os.listdir(jsonl_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(jsonl_dir, filename)
            file_count += 1

            # 计算单个文件的行数
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # lines = f.readlines()
                    # count = len(lines)
                    # 使用生成器表达式更省内存
                    line_count = sum(1 for _ in f)

                print(f"文件: {filename} -> {line_count} 条 QA")
                total_qa_count += line_count
            except Exception as e:
                print(f"读取文件失败 {filename}: {e}")

    print("-" * 40)
    print(f"统计完成！")
    print(f"共扫描了 {file_count} 个 .jsonl 文件")
    print(f"所有文件加在一起的 QA 总数量为: {total_qa_count} 条")


if __name__ == "__main__":
    # 替换为你刚才生成的 pilottest_msr 里面的 jsonl 文件夹路径
    target_dir = r"D:\Data\infinigen_20251031\refresh\pilottest_msr\jsonl"

    count_total_qa(target_dir)
