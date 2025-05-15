#!/usr/bin/env python
# filepath: /home/dawnfu/PrecipitionCorrecrion/pytorch-worker/batch_pad_nc_files.py

import argparse
import logging
import math
import multiprocessing
import os

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_target_size(original_height, original_width, divisible_by=32, target_size=None):
    """计算目标尺寸，确保能被divisible_by整除"""
    if target_size is not None:
        return target_size

    # 计算调整后的高度和宽度
    target_height = math.ceil(original_height / divisible_by) * divisible_by
    target_width = math.ceil(original_width / divisible_by) * divisible_by

    return (target_height, target_width)


def pad_variable(data_var, target_height, target_width, pad_mode="constant", bottom_pad_ratio=1.0):
    """
    对数据变量进行填充，优先向下填充

    参数:
    data_var: xarray DataArray
    target_height, target_width: 目标高度和宽度
    pad_mode: 填充模式，'constant' 或 'reflect'
    bottom_pad_ratio: 优先下方填充比例，1.0表示全部填充在下方，0.5表示均匀填充

    返回:
    填充后的数据变量
    """
    # 获取原始维度
    dims = data_var.dims
    shape = data_var.shape

    # 检查数据是否有lat和lon维度
    if "lat" not in dims or "lon" not in dims:
        return data_var  # 如果没有lat或lon维度，直接返回

    # 找出lat和lon的位置
    lat_idx = dims.index("lat")
    lon_idx = dims.index("lon")

    # 获取原始高度和宽度
    orig_height = shape[lat_idx]
    orig_width = shape[lon_idx]

    # 如果已经是目标尺寸，直接返回
    if orig_height == target_height and orig_width == target_width:
        return data_var

    # 计算需要填充的尺寸
    pad_height = max(0, target_height - orig_height)
    pad_width = max(0, target_width - orig_width)

    # 计算填充量 (左、右、上、下)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # 优先向下填充 - 根据bottom_pad_ratio决定上下填充比例
    pad_top = int(pad_height * (1 - bottom_pad_ratio))
    pad_bottom = pad_height - pad_top

    # 准备填充配置
    padding = [(0, 0)] * len(dims)
    padding[lat_idx] = (pad_top, pad_bottom)
    padding[lon_idx] = (pad_left, pad_right)

    # 执行填充
    if pad_mode == "reflect":
        padded_data = np.pad(data_var.values, padding, mode="reflect")
    else:
        padded_data = np.pad(data_var.values, padding, mode="constant", constant_values=0)

    # 更新坐标
    coords = {}

    for dim in dims:
        if dim == "lat":
            # 扩展lat坐标
            lat_vals = data_var.lat.values
            if len(lat_vals) > 0:
                # 检查坐标是否是降序排列
                is_decreasing = (lat_vals[0] > lat_vals[-1]) if len(lat_vals) > 1 else False

                if is_decreasing:
                    # 从大到小排列（北到南）
                    lat_step = (lat_vals[0] - lat_vals[-1]) / max(1, len(lat_vals) - 1)

                    # 创建顶部填充（值大于原始数组的最大值）
                    top_pad = (
                        np.linspace(lat_vals[0] + lat_step, lat_vals[0] + pad_top * lat_step, pad_top)
                        if pad_top > 0
                        else np.array([])
                    )

                    # 创建底部填充（值小于原始数组的最小值）
                    bottom_pad = (
                        np.linspace(lat_vals[-1] - pad_bottom * lat_step, lat_vals[-1] - lat_step, pad_bottom)[::-1]
                        if pad_bottom > 0
                        else np.array([])
                    )

                    # 合并数组
                    new_lats = np.concatenate([top_pad, lat_vals, bottom_pad])
                else:
                    # 从小到大排列（南到北）
                    lat_step = (lat_vals[-1] - lat_vals[0]) / max(1, len(lat_vals) - 1)

                    # 创建顶部填充（值小于原始数组的最小值）
                    top_pad = (
                        np.linspace(lat_vals[0] - pad_top * lat_step, lat_vals[0] - lat_step, pad_top)
                        if pad_top > 0
                        else np.array([])
                    )

                    # 创建底部填充（值大于原始数组的最大值）
                    bottom_pad = (
                        np.linspace(lat_vals[-1] + lat_step, lat_vals[-1] + pad_bottom * lat_step, pad_bottom)
                        if pad_bottom > 0
                        else np.array([])
                    )

                    # 合并数组
                    new_lats = np.concatenate([top_pad, lat_vals, bottom_pad])

                coords[dim] = new_lats

        elif dim == "lon":
            # 扩展lon坐标
            lon_vals = data_var.lon.values
            if len(lon_vals) > 0:
                # 检查坐标是否是降序排列
                is_decreasing = (lon_vals[0] > lon_vals[-1]) if len(lon_vals) > 1 else False

                if is_decreasing:
                    # 从大到小排列（东到西）
                    lon_step = (lon_vals[0] - lon_vals[-1]) / max(1, len(lon_vals) - 1)

                    # 创建左侧填充
                    left_pad = (
                        np.linspace(lon_vals[0] + lon_step, lon_vals[0] + pad_left * lon_step, pad_left)
                        if pad_left > 0
                        else np.array([])
                    )

                    # 创建右侧填充
                    right_pad = (
                        np.linspace(lon_vals[-1] - pad_right * lon_step, lon_vals[-1] - lon_step, pad_right)[::-1]
                        if pad_right > 0
                        else np.array([])
                    )

                    # 合并数组
                    new_lons = np.concatenate([left_pad, lon_vals, right_pad])
                else:
                    # 从小到大排列（西到东）
                    lon_step = (lon_vals[-1] - lon_vals[0]) / max(1, len(lon_vals) - 1)

                    # 创建左侧填充
                    left_pad = (
                        np.linspace(lon_vals[0] - pad_left * lon_step, lon_vals[0] - lon_step, pad_left)
                        if pad_left > 0
                        else np.array([])
                    )

                    # 创建右侧填充
                    right_pad = (
                        np.linspace(lon_vals[-1] + lon_step, lon_vals[-1] + pad_right * lon_step, pad_right)
                        if pad_right > 0
                        else np.array([])
                    )

                    # 合并数组
                    new_lons = np.concatenate([left_pad, lon_vals, right_pad])

                coords[dim] = new_lons
        else:
            coords[dim] = data_var[dim]

    # 创建新的DataArray
    padded_var = xr.DataArray(data=padded_data, dims=dims, coords=coords, attrs=data_var.attrs)

    # 验证坐标长度是否匹配
    for dim in dims:
        if len(padded_var[dim]) != padded_data.shape[dims.index(dim)]:
            print(
                f"警告: 维度 {dim} 坐标长度 ({len(padded_var[dim])}) 与数据形状 ({padded_data.shape[dims.index(dim)]}) 不匹配"
            )

    return padded_var


def process_nc_file(args):
    """处理单个NC文件"""
    file_path, output_dir, target_size, divisible_by, pad_mode, bottom_pad_ratio = args

    try:
        # 构建输出路径
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        # 跳过已经处理过的文件
        if os.path.exists(output_path):
            logger.info(f"跳过已存在的文件: {output_path}")
            return True

        # 打开NC文件
        ds = xr.open_dataset(file_path)

        # 确定需要处理的变量
        data_vars = ds.data_vars

        # 查找一个有lat和lon的变量来确定原始尺寸
        original_height, original_width = None, None
        for var_name in data_vars:
            var = ds[var_name]
            if "lat" in var.dims and "lon" in var.dims:
                lat_idx = var.dims.index("lat")
                lon_idx = var.dims.index("lon")
                original_height = var.shape[lat_idx]
                original_width = var.shape[lon_idx]
                break

        # 如果没有找到合适的变量，跳过此文件
        if original_height is None or original_width is None:
            logger.warning(f"文件 {filename} 中没有包含lat和lon维度的变量，跳过")
            return False

        # 计算目标尺寸
        actual_target_size = get_target_size(
            original_height, original_width, divisible_by=divisible_by, target_size=target_size
        )

        # 创建一个空的数据集
        new_ds = xr.Dataset()

        # 收集所有处理后的变量
        data_arrays = []

        # 处理每个变量
        for var_name in data_vars:
            var = ds[var_name]
            if "lat" in var.dims and "lon" in var.dims:
                # 填充变量
                padded_var = pad_variable(
                    var,
                    actual_target_size[0],
                    actual_target_size[1],
                    pad_mode=pad_mode,
                    bottom_pad_ratio=bottom_pad_ratio,
                )
                # 确保数据类型为float32并添加到列表中
                data_arrays.append(padded_var.astype(np.float32).rename(var_name))
            else:
                # 对于不需要填充的变量，保持原样
                if np.issubdtype(var.dtype, np.floating):
                    data_arrays.append(var.astype(np.float32).rename(var_name))
                else:
                    data_arrays.append(var.rename(var_name))

        # 使用xr.merge合并所有变量，这样可以确保坐标被正确处理
        new_ds = xr.merge(data_arrays)

        # 复制原始数据集的全局属性
        new_ds.attrs.update(ds.attrs)

        # 添加填充信息到全局属性
        new_ds.attrs["padding_info"] = (
            f"Padded from {original_height}x{original_width} to {actual_target_size[0]}x{actual_target_size[1]} using {pad_mode} mode"
        )

        # 创建编码设置，确保所有变量都以float32保存
        encoding = {}
        for var_name in new_ds.data_vars:
            # 仅对浮点类型变量设置float32编码
            if np.issubdtype(new_ds[var_name].dtype, np.floating):
                encoding[var_name] = {
                    "dtype": "float32",
                    "zlib": True,  # 启用压缩
                    "complevel": 5,  # 压缩级别 (1-9)
                }

        # 保存处理后的文件，使用编码设置
        new_ds.to_netcdf(output_path, encoding=encoding)
        new_ds.close()
        ds.close()

        return True

    except Exception as e:
        logger.error(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量填充NC文件到适合U-Net的尺寸")
    parser.add_argument("--input_dir", type=str, default="/mnt/h/DataSet/Merge/temp", help="输入NC文件目录")
    parser.add_argument("--output_dir", type=str, default="/mnt/d/Data/temp", help="输出NC文件目录")
    parser.add_argument("--target_height", type=int, help="目标高度")
    parser.add_argument("--target_width", type=int, help="目标宽度")
    parser.add_argument("--divisible_by", type=int, default=32, help="确保尺寸能被此数整除(默认32)")
    parser.add_argument(
        "--pad_mode", type=str, choices=["constant", "reflect"], default="constant", help="填充模式(默认constant)"
    )
    parser.add_argument("--workers", type=int, default=4, help="并行处理的工作线程数")
    parser.add_argument(
        "--bottom_pad_ratio", type=float, default=1.0, help="优先向下填充比例，1.0表示全部在下方，0.5表示均匀分布"
    )

    args = parser.parse_args()

    # 设置目标尺寸
    target_size = None
    if args.target_height is not None and args.target_width is not None:
        target_size = (args.target_height, args.target_width)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有NC文件
    nc_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".nc")]

    if not nc_files:
        logger.warning(f"在 {args.input_dir} 目录下未找到NC文件")
        return

    logger.info(f"找到 {len(nc_files)} 个NC文件待处理")

    # 准备任务参数，添加bottom_pad_ratio
    task_args = [
        (file_path, args.output_dir, target_size, args.divisible_by, args.pad_mode, args.bottom_pad_ratio)
        for file_path in nc_files
    ]
    # 使用多进程处理文件
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(process_nc_file, task_args), total=len(nc_files), desc="处理文件"))

    # 统计处理结果
    success_count = results.count(True)
    logger.info(f"处理完成: 成功 {success_count}/{len(nc_files)} 个文件")


if __name__ == "__main__":
    main()
