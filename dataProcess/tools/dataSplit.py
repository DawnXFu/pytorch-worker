import os
import random
import shutil


def split_nc_files(input_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """
    随机将input_dir中的nc文件划分为train、valid、test三个子集，并分别复制到output_dir下对应文件夹。

    参数:
    input_dir: 原始nc文件夹路径
    output_dir: 输出根目录
    train_ratio, valid_ratio, test_ratio: 各子集比例，和应为1
    seed: 随机种子，保证可复现
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    os.makedirs(output_dir, exist_ok=True)
    for sub in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    nc_files = [f for f in os.listdir(input_dir) if f.endswith(".nc")]
    random.seed(seed)
    random.shuffle(nc_files)

    n_total = len(nc_files)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = n_total - n_train - n_valid

    train_files = nc_files[:n_train]
    valid_files = nc_files[n_train : n_train + n_valid]
    test_files = nc_files[n_train + n_valid :]

    for fname in train_files:
        shutil.move(os.path.join(input_dir, fname), os.path.join(output_dir, "train", fname))
    for fname in valid_files:
        shutil.move(os.path.join(input_dir, fname), os.path.join(output_dir, "valid", fname))
    for fname in test_files:
        shutil.move(os.path.join(input_dir, fname), os.path.join(output_dir, "test", fname))

    print(f"总文件数: {n_total}, 训练集: {len(train_files)}, 验证集: {len(valid_files)}, 测试集: {len(test_files)}")


# 用法示例
if __name__ == "__main__":
    input_dir = "/mnt/d/Data/temp"  # 修改为你的nc文件夹路径
    output_dir = "/mnt/d/Data/"  # 输出根目录
    split_nc_files(input_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42)
