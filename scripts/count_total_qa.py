import os


def count_total_qa(jsonl_dir):
    """
    Count the total number of QA entries across all jsonl files in a directory.
    """
    if not os.path.exists(jsonl_dir):
        print(f"Error: directory does not exist -> {jsonl_dir}")
        return

    total_qa_count = 0
    file_count = 0

    print("开始统计 JSONL 文件...")
    print("-" * 40)

    # Iterate over all files in the directory.
    for filename in os.listdir(jsonl_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(jsonl_dir, filename)
            file_count += 1

            # Count lines in a single file.
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # lines = f.readlines()
                    # count = len(lines)
                    # Generator expression saves memory.
                    line_count = sum(1 for _ in f)

                print(f"文件: {filename} -> {line_count}  QA entries")
                total_qa_count += line_count
            except Exception as e:
                print(f"Failed to read file {filename}: {e}")

    print("-" * 40)
    print(f"Done!")
    print(f"Scanned {file_count} .jsonl files")
    print(f"Total QA entries across all files: {total_qa_count}")


if __name__ == "__main__":
    # Set this to the directory containing the jsonl files to count.
    target_dir = "/path/to/data/refresh/pilottest_msr/jsonl"

    count_total_qa(target_dir)
