import os
from pathlib import Path


def consolidate_worker_logs(log_dir: str, output_filename: str = "final_missing_texts_clean.txt"):
    """
    遍历指定目录下的所有 worker 日志文件，整合内容并去重。

    Args:
        log_dir (str): 包含 missing_rankX_workerY.txt 文件的目录路径。
        output_filename (str): 最终干净的输出文件名。
    """

    log_path = Path(log_dir)
    if not log_path.is_dir():
        print(f"Error: Directory not found at {log_dir}")
        return

    # 全局集合，用于存储所有不重复的文本
    master_missing_set = set()
    total_lines_read = 0

    # 遍历目录中所有匹配 worker 日志模式的文件
    log_files = sorted(list(log_path.glob("missing_rank*worker*.txt")))

    if not log_files:
        print(f"No worker log files found in {log_dir}. Exiting.")
        return

    print(f"Found {len(log_files)} log files. Consolidating...")

    for file_path in log_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                total_lines_read += len(lines)
                master_missing_set.update(lines)  # 使用 update 集合操作实现去重
        except Exception as e:
            print(f"Warning: Could not read file {file_path}. Error: {e}")

    # 将最终的集合排序后写入最终文件
    final_clean_list = sorted(list(master_missing_set))
    output_path = log_path / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        for word in final_clean_list:
            f.write(word + '\n')

    print("--- Consolidation Summary ---")
    print(f"Total lines read from all workers: {total_lines_read}")
    print(f"Total unique missing words found: {len(final_clean_list)}")
    print(f"Clean list saved to: {output_path}")

# --- 运行示例 ---
# 请将 'missing_txt_logs' 替换为你实际的日志文件夹路径
consolidate_worker_logs('./missing_txt_logs', 'final_missing_texts_clean.txt')