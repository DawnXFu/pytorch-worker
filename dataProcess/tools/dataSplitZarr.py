import os
import random
import shutil

import numpy as np
import xarray as xr


def split_nc_files(
    input_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42, convert_to_zarr=False
):
    """
    随机将input_dir中的nc文件划分为train、valid、test三个子集，并分别复制到output_dir下对应文件夹。
    可选择将分割后的数据集转换为zarr格式。

    参数:
    input_dir: 原始nc文件夹路径
    output_dir: 输出根目录
    train_ratio, valid_ratio, test_ratio: 各子集比例，和应为1
    seed: 随机种子，保证可复现
    convert_to_zarr: 是否将分割后的数据集转换为zarr格式
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

    # 复制文件到对应子集文件夹
    for fname in train_files:
        shutil.move(os.path.join(input_dir, fname), os.path.join(output_dir, "train", fname))
    for fname in valid_files:
        shutil.move(os.path.join(input_dir, fname), os.path.join(output_dir, "valid", fname))
    for fname in test_files:
        shutil.move(os.path.join(input_dir, fname), os.path.join(output_dir, "test", fname))

    print(f"总文件数: {n_total}, 训练集: {len(train_files)}, 验证集: {len(valid_files)}, 测试集: {len(test_files)}")

    # 如果需要，将子集转换为zarr格式
    if convert_to_zarr:
        convert_subsets_to_zarr(output_dir)


def convert_subsets_to_zarr(base_dir):
    """
    将base_dir下的train、valid、test子集中的nc文件分别合并并转换为zarr格式。

    参数:
    base_dir: 包含train、valid、test子文件夹的基础目录
    """
    try:
        import zarr
    except ImportError:
        print("请先安装zarr库: pip install zarr")
        return

    for subset in ["train", "valid", "test"]:
        subset_dir = os.path.join(base_dir, subset)
        zarr_path = os.path.join(base_dir, f"{subset}.zarr")

        # 获取子集中所有nc文件
        nc_files = [os.path.join(subset_dir, f) for f in os.listdir(subset_dir) if f.endswith(".nc")]

        if not nc_files:
            print(f"{subset}子集中没有nc文件，跳过转换")
            continue

        print(f"正在转换{subset}子集 ({len(nc_files)}个文件) 为zarr格式...")

        try:
            # 使用xarray打开所有文件并合并
            # 注意：根据数据结构和大小，可能需要调整合并策略
            datasets = []
            for nc_file in nc_files:
                ds = xr.open_dataset(nc_file)
                datasets.append(ds)

            # 合并所有数据集
            # 注意：根据数据的具体结构，可能需要修改合并维度
            merged_ds = xr.concat(datasets, dim="time")

            # 保存为zarr格式
            print(f"正在保存{subset}.zarr...")
            merged_ds.to_zarr(zarr_path, mode="w")
            print(f"{subset}.zarr 保存完成")

            # 关闭所有数据集
            for ds in datasets:
                ds.close()

        except Exception as e:
            print(f"转换{subset}子集时出错: {str(e)}")


def convert_nc_to_zarr_chunked(base_dir, chunk_size={"time": 10}):
    """
    使用dask进行分块处理，将大型nc文件集合转换为zarr格式。
    适用于内存受限的情况。

    参数:
    base_dir: 包含train、valid、test子文件夹的基础目录
    chunk_size: 分块大小字典
    """
    try:
        import dask
        import zarr
    except ImportError:
        print("请先安装必要的库: pip install zarr dask")
        return

    for subset in ["train", "valid", "test"]:
        subset_dir = os.path.join(base_dir, subset)
        zarr_path = os.path.join(base_dir, f"{subset}.zarr")

        # 获取子集中所有nc文件
        nc_files = [os.path.join(subset_dir, f) for f in os.listdir(subset_dir) if f.endswith(".nc")]

        if not nc_files:
            print(f"{subset}子集中没有nc文件，跳过转换")
            continue

        print(f"正在转换{subset}子集 ({len(nc_files)}个文件) 为zarr格式 (使用分块处理)...")

        try:
            # 使用dask和xarray以分块方式打开文件
            ds = xr.open_mfdataset(nc_files, chunks=chunk_size, combine="by_coords")

            # 保存为zarr格式
            print(f"正在保存{subset}.zarr (这可能需要一些时间)...")
            ds.to_zarr(zarr_path, mode="w")
            print(f"{subset}.zarr 保存完成")

            # 关闭数据集
            ds.close()

        except Exception as e:
            print(f"转换{subset}子集时出错: {str(e)}")


# 用法示例
if __name__ == "__main__":
    input_dir = "/mnt/d/Data/temp"  # 修改为你的nc文件夹路径
    output_dir = "/mnt/d/Data/"  # 输出根目录

    # 方法1: 分离文件并转换为zarr (适用于小到中等大小的数据集)
    split_nc_files(
        input_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42, convert_to_zarr=True
    )

    # 方法2: 如果数据集很大，先分离文件，然后使用分块处理转换为zarr
    # split_nc_files(input_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42)
    # convert_nc_to_zarr_chunked(output_dir, chunk_size={"time": 10})
