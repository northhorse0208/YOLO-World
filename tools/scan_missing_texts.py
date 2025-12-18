import argparse
import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Scan dataset to find missing text embeddings')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir', help='the dir to save logs and models')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载配置文件
    cfg = Config.fromfile(args.config)

    # 2. 修改配置以加速扫描 (关键步骤)
    # -------------------------------------------------------
    # 关闭 Mosaic 等耗时的图像混合增强，只保留 RandomLoadText 和必要的格式化
    # 注意：如果你担心去掉 Mosaic 会改变 RandomLoadText 的行为（通常不会），
    # 你可以保留它们，但速度会变慢。
    # 建议：为了最快速度，只要 pipeline 里有 RandomLoadText 就行，
    # 但是为了保险起见（避免索引对齐问题），最好不要动 pipeline，
    # 而是通过多进程来加速。

    # 强制把 batch_size 调大（因为不需要进 GPU 计算，只吃 CPU 内存）
    # 根据你的内存大小调整，比如 64 或 128
    # cfg.train_dataloader.batch_size = 64

    # 拉满 CPU 核心数，极大加速数据加载
    # cfg.train_dataloader.num_workers = 8 # 或者更多，取决于你的 CPU 核心数

    # 如果你的 RandomLoadText 也是在这个 config 里配置的，它会被自动加载
    # -------------------------------------------------------

    # 3. 初始化 Runner
    # 我们利用 Runner 来帮我们构建正确的 DataLoader，省去手动构建 Dataset 的麻烦
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs/scan_text'

    # 构建 Runner，但我们要把模型部分置空或用个假的，因为我们不跑模型
    # 不过 MMDetection 的 Runner 构建需要 model，我们直接初始化它，不调用 train() 即可
    # 这会加载模型到显存，如果不想占显存，可以把 cfg.model 改成 None (可能报错)
    # 最简单的方法是：让它加载，但我们不用它。
    runner = Runner.from_cfg(cfg)

    # 4. 获取训练集的 DataLoader
    dataloader = runner.train_dataloader

    print(f"Starting to scan dataset with {cfg.train_dataloader.num_workers} workers...")
    print("This will trigger RandomLoadText but skip model forward/backward pass.")

    # 5. 核心循环：只遍历，不训练
    # tqdm 会显示进度条
    total_batches = len(dataloader)

    # 这里的 batch 就是经过了 pipeline 处理后的数据
    # 在这一步取出 batch 的瞬间，RandomLoadText 已经被执行了！
    for i, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Scanning"):
        # 什么都不用做，直接进入下一次循环
        # 你的 RandomLoadText 里的逻辑会自动记录缺失文本到 txt
        pass

    print("\nScanning finished!")
    print("Check your missing texts log file (e.g., tools/missing_texts_log.txt).")


if __name__ == '__main__':
    main()