#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
气象数据预处理脚本
用于处理NC文件中的异常值，进行数据归一化，并生成全局统计信息
"""

import multiprocessing as mp
import os
import sys
from functools import partial

import numpy as np
import xarray as xr
from tqdm import tqdm

# ====================== 配置参数 ======================
# 输入和输出目录
INPUT_DIR = "/mnt/d/Data/train/"  # 输入NC文件目录
OUTPUT_DIR = "/mnt/d/Data/temp/"  # 输出处理后NC文件目录

# 预处理选项
COMPUTE_STATS = True  # 是否计算并保存全局统计信息
LOG_TRANSFORM = True  # 是否对降水数据进行对数变换
VARIABLES = ["TAIR", "UWIN", "VWIN", "PRE", "corrected_precip"]  # 要处理的变量列表
WORKERS = 4  # 并行处理的工作进程数

# 变量的合理值范围配置
VAR_LIMITS = {
    "TAIR": (0,),  # 气温范围 (开尔文)
    "UWIN": (-100.0, 100.0),  # 东西风速范围 (m/s)
    "VWIN": (-100.0, 100.0),  # 南北风速范围 (m/s)
    "PRE": (0.0, 100.0),  # 降水范围 (mm)
    "corrected_precip": (0.0, 100.0),  # 校正后的降水范围 (mm)
}

# 缺失值标记
MISSING_VALUES = [9999.0, -9999.0, 999999, -999999]


def preprocess_variable(data, var_name, missing_values=MISSING_VALUES, var_limits=VAR_LIMITS, log_transform=False):
    """
    对单个变量进行预处理

    参数:
        data: 变量数据数组
        var_name: 变量名称
        missing_values: 缺失值标记列表
        var_limits: 变量合理值范围字典
        log_transform: 是否对降水数据进行对数变换

    返回:
        处理后的数据
    """
    # 创建数据拷贝以避免修改原始数据
    data = data.copy()

    # 1. 处理缺失值
    for missing_val in missing_values:
        mask = np.abs(data - missing_val) < 1e-3
        if np.any(mask):
            fill_value = 0.0 if var_name in ["PRE", "corrected_precip"] else np.nanmean(data)
            data = np.where(mask, fill_value, data)

    # 2. 处理超出合理范围的值 - 设置为0而不是裁剪
    if var_name in var_limits:
        min_val, max_val = var_limits[var_name]
        # 创建超出范围的掩码
        out_of_range_mask = (data < min_val) | (data > max_val)
        # 将超出范围的值设置为0
        if np.any(out_of_range_mask):
            data = np.where(out_of_range_mask, 0.0, data)
            # 可以添加一条打印语句来记录被替换的值的数量
            print(f"变量 {var_name} 中有 {np.sum(out_of_range_mask)} 个值超出范围 [{min_val}, {max_val}]，已设置为0")

    # 3. 对于降水数据的对数变换 (如果启用)
    if log_transform and var_name in ["PRE", "corrected_precip"]:
        # 保留零值的对数变换: log(1+x)
        data = np.log1p(data)

    # 4. 最终检查确保没有NaN或无穷值
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # 这里也将无穷值设置为0

    return data


def compute_dataset_stats(file_list, variables, input_dir, log_transform=False):
    """
    计算数据集的全局统计量

    参数:
        file_list: NC文件列表
        variables: 要处理的变量列表
        input_dir: 输入文件目录
        log_transform: 是否在计算前应用对数变换

    返回:
        变量统计量字典 {变量名: {'mean': 均值, 'std': 标准差}}
    """
    print("计算数据集统计量...")

    # 初始化累加器
    sums = {var: 0.0 for var in variables}
    sum_squares = {var: 0.0 for var in variables}
    counts = {var: 0 for var in variables}

    # 限制用于计算统计量的文件数，避免处理过多数据
    max_files = min(50, len(file_list))
    sample_files = file_list[:max_files]

    for filename in tqdm(sample_files, desc="计算统计量"):
        file_path = os.path.join(input_dir, filename)
        try:
            with xr.open_dataset(file_path) as ds:
                for var in variables:
                    if var in ds:
                        # 获取并预处理数据
                        data = ds[var].values
                        data = preprocess_variable(data, var, log_transform=log_transform)

                        # 更新统计量
                        non_zero_mask = (
                            ~np.isclose(data, 0.0)
                            if var in ["PRE", "corrected_precip"]
                            else np.ones_like(data, dtype=bool)
                        )
                        sums[var] += np.sum(data[non_zero_mask])
                        sum_squares[var] += np.sum(data[non_zero_mask] ** 2)
                        counts[var] += np.sum(non_zero_mask)
        except Exception as e:
            print(f"警告: 计算统计量时无法处理文件 {filename}: {e}")

    # 计算每个变量的均值和标准差
    stats = {}
    for var in variables:
        if counts[var] > 0:
            mean = sums[var] / counts[var]
            var_value = (sum_squares[var] / counts[var]) - (mean**2)
            std = np.sqrt(max(var_value, 1e-6))
            stats[var] = {"mean": mean, "std": std}
            print(f"变量 {var} 统计量: 均值={mean:.6f}, 标准差={std:.6f}, 样本数={counts[var]}")
        else:
            # 如果没有有效数据，设置默认值
            stats[var] = {"mean": 0.0, "std": 1.0}
            print(f"警告: 变量 {var} 没有有效数据，使用默认统计量")

    return stats


def process_file(filename, input_dir, output_dir, variables, stats=None, log_transform=False, normalize=False):
    """
    处理单个NC文件

    参数:
        filename: 文件名
        input_dir: 输入目录
        output_dir: 输出目录
        variables: 要处理的变量列表
        stats: 用于标准化的统计量
        log_transform: 是否应用对数变换
        normalize: 是否应用标准化
    """
    try:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 如果输出文件已存在则跳过
        if os.path.exists(output_path):
            return

        # 打开数据集
        ds = xr.open_dataset(input_path)

        # 预处理每个变量
        for var in variables:
            if var in ds:
                # 获取原始数据
                data = ds[var].values

                # 预处理数据
                processed_data = preprocess_variable(data, var, log_transform=log_transform)

                # 如果提供了统计量，应用标准化
                if normalize and stats and var in stats:
                    mean, std = stats[var]["mean"], stats[var]["std"]
                    processed_data = (processed_data - mean) / std

                # 更新数据集中的变量
                ds[var].values = processed_data

        # 保存处理后的数据集
        ds.to_netcdf(output_path)
        ds.close()

    except Exception as e:
        print(f"错误: 处理文件 {filename} 时失败: {e}")


def main():
    """主函数"""
    # 确认输入目录存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 {INPUT_DIR} 不存在")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有NC文件
    file_list = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".nc")])
    if not file_list:
        print(f"错误: 在 {INPUT_DIR} 中未找到NC文件")
        sys.exit(1)

    print(f"找到 {len(file_list)} 个NC文件待处理")

    # 计算数据集统计量
    stats = None
    if COMPUTE_STATS:
        stats = compute_dataset_stats(file_list, VARIABLES, INPUT_DIR, LOG_TRANSFORM)

        # 保存统计量
        stats_file = os.path.join(OUTPUT_DIR, "data_stats.npz")
        save_dict = {}
        for var, stat in stats.items():
            save_dict[f"{var}_mean"] = stat["mean"]
            save_dict[f"{var}_std"] = stat["std"]
        np.savez(stats_file, **save_dict)
        print(f"已保存全局统计量到 {stats_file}")

    # 多进程处理文件
    process_func = partial(
        process_file,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        variables=VARIABLES,
        stats=stats,
        log_transform=LOG_TRANSFORM,
        normalize=COMPUTE_STATS,
    )

    print(f"使用 {WORKERS} 个进程开始处理文件...")
    with mp.Pool(processes=WORKERS) as pool:
        list(tqdm(pool.imap(process_func, file_list), total=len(file_list), desc="处理文件"))

    print("数据预处理完成!")


if __name__ == "__main__":
    main()
