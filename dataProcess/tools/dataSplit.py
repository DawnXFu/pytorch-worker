import argparse
import concurrent.futures
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm  # 添加进度条支持


def split_data(
    source_dir,
    target_base_dir,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    copy_files=True,
    seed=42,
    num_workers=8,
):
    """
    将源文件夹中的数据按照指定比例划分到训练集、验证集和测试集

    参数:
        source_dir (str): 源数据文件夹路径
        target_base_dir (str): 目标文件夹的基础路径
        train_ratio (float): 训练集比例，默认0.7
        valid_ratio (float): 验证集比例，默认0.15
        test_ratio (float): 测试集比例，默认0.15
        copy_files (bool): 是否复制文件而不是移动，默认True
        seed (int): 随机种子，确保结果可复现，默认42
        num_workers (int): 并行工作线程数，默认8
    """
    # 验证比例之和是否为1
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("比例之和必须等于1")

    # 设置随机种子
    random.seed(seed)

    # 创建目标文件夹
    train_dir = os.path.join(target_base_dir, "Train")
    valid_dir = os.path.join(target_base_dir, "Valid")
    test_dir = os.path.join(target_base_dir, "Test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取源文件夹中的所有文件
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    # 随机打乱文件顺序
    random.shuffle(all_files)

    # 计算每个集合应该包含的文件数量
    total_files = len(all_files)
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)

    # 划分文件到三个集合
    train_files = all_files[:train_count]
    valid_files = all_files[train_count : train_count + valid_count]
    test_files = all_files[train_count + valid_count :]

    # 处理单个文件的函数
    def process_file(file_data):
        file_path, target_dir = file_data
        # 保持源文件夹的相对目录结构
        rel_path = os.path.relpath(file_path, source_dir)
        target_path = os.path.join(target_dir, rel_path)

        # 确保目标目录存在
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # 复制或移动文件
        try:
            if copy_files:
                shutil.copy2(file_path, target_path)
            else:
                shutil.move(file_path, target_path)
            return True
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return False

    # 并行复制或移动文件的函数
    def transfer_files(file_list, target_dir):
        # 创建任务列表
        tasks = [(file_path, target_dir) for file_path in file_list]
        successful = 0

        # 使用线程池并行处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务并显示进度条
            results = list(
                tqdm(
                    executor.map(process_file, tasks),
                    total=len(tasks),
                    desc=f"处理 {os.path.basename(target_dir)} 数据集",
                )
            )

            # 计算成功处理的文件数
            successful = sum(results)

        return successful

    # 复制或移动文件到目标文件夹
    print("开始处理数据...")
    train_success = transfer_files(train_files, train_dir)
    valid_success = transfer_files(valid_files, valid_dir)
    test_success = transfer_files(test_files, test_dir)

    # 打印统计信息
    print(f"\n数据集划分完成:")
    print(f"总文件数: {total_files}")
    print(f"训练集: {train_success}/{len(train_files)} 文件成功处理 ({train_success/total_files:.2%})")
    print(f"验证集: {valid_success}/{len(valid_files)} 文件成功处理 ({valid_success/total_files:.2%})")
    print(f"测试集: {test_success}/{len(test_files)} 文件成功处理 ({test_success/total_files:.2%})")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="将数据集分割为训练集、验证集和测试集")
    parser.add_argument("--source", type=str, default="/mnt/h/DataSet/Merged_padded/", help="源数据文件夹路径")
    parser.add_argument("--target", type=str, default="/mnt/h/Data/", help="目标基础路径")
    parser.add_argument("--train", type=float, default=0.7, help="训练集比例 (默认: 0.7)")
    parser.add_argument("--valid", type=float, default=0.15, help="验证集比例 (默认: 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="测试集比例 (默认: 0.15)")
    parser.add_argument("--move", action="store_true", help="移动文件而不是复制")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--workers", type=int, default=8, help="并行工作线程数 (默认: 8)")

    args = parser.parse_args()

    split_data(
        source_dir=args.source,
        target_base_dir=args.target,
        train_ratio=args.train,
        valid_ratio=args.valid,
        test_ratio=args.test,
        copy_files=not args.move,
        seed=args.seed,
        num_workers=args.workers,
    )
