import logging
import os
from datetime import datetime

import numpy as np
import xarray as xr

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Checkpoint_Restructure")


def get_sorted_checkpoint_files(input_dir):
    """获取排序后的检查点文件列表"""
    checkpoint_files = []
    for f in os.listdir(input_dir):
        if f.startswith("prism_checkpoint_") and f.endswith(".nc"):
            try:
                chunk_num = int(f.replace("prism_checkpoint_", "").replace(".nc", ""))
                checkpoint_files.append((chunk_num, os.path.join(input_dir, f)))
            except ValueError:
                logger.warning(f"跳过无效的检查点文件: {f}")

    checkpoint_files.sort()  # 按chunk编号排序
    return checkpoint_files


def find_closest_time_boundary(times, target_hour=8):
    """
    在时间序列中找到最近的目标小时(如8:00)边界
    返回找到的时间点索引和时间值
    """
    # 转换为datetime对象列表，便于操作
    dt_times = [datetime.utcfromtimestamp(t.astype("int64") / 1e9) for t in times]

    # 找到第一个等于或大于目标小时的时间点
    for i, dt in enumerate(dt_times):
        if dt.hour == target_hour and dt.minute == 0:
            return i, times[i]

    # 如果没有找到精确匹配，找到下一个8:00时间点
    closest_time = np.datetime64(dt_times[0].replace(hour=target_hour, minute=0, second=0, microsecond=0))
    if closest_time < times[0]:
        closest_time = closest_time + np.timedelta64(1, "D")

    return None, closest_time


def process_batch(batch_files, output_dir, start_chunk_idx=0, last_processed_time=None):
    """
    处理一批检查点文件

    参数:
    batch_files: 要处理的文件列表(编号, 路径)
    output_dir: 输出目录
    start_chunk_idx: 输出文件的起始编号
    last_processed_time: 上一批次处理到的最后时间点

    返回:
    最后处理的时间点，供下一批次使用
    """
    if not batch_files:
        return last_processed_time

    logger.info(f"处理批次，包含 {len(batch_files)} 个文件")

    # 读取这批文件
    datasets = []
    for chunk_num, file_path in batch_files:
        try:
            logger.info(f"读取文件: {os.path.basename(file_path)}")
            ds = xr.open_dataset(file_path)
            datasets.append(ds)
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {str(e)}")

    if not datasets:
        return last_processed_time

    # 合并数据集
    logger.info("合并批次数据...")
    merged_ds = xr.concat(datasets, dim="time")

    # 按照排序后的时间确保唯一性
    merged_ds = merged_ds.sortby("time")

    # 确定起始时间
    all_times = merged_ds.time.values

    # 如果有上一批次的时间点，从其后开始
    if last_processed_time is not None:
        # 找到第一个大于上次处理时间点的索引
        start_idx = np.searchsorted(all_times, last_processed_time, side="right")
        if start_idx >= len(all_times):
            logger.info("该批次中没有新的时间点需要处理")
            # 关闭数据集
            for ds in datasets:
                ds.close()
            return last_processed_time

        current_times = all_times[start_idx:]
        merged_ds = merged_ds.isel(time=slice(start_idx, None))
    else:
        current_times = all_times

    if len(current_times) == 0:
        logger.info("没有新的时间点需要处理")
        # 关闭数据集
        for ds in datasets:
            ds.close()
        return last_processed_time

    # 找到第一个8点时间边界
    if last_processed_time is None:
        _, start_time = find_closest_time_boundary(current_times, 8)
    else:
        # 如果已经有上一个处理时间，找到下一个8点
        last_dt = datetime.utcfromtimestamp(last_processed_time.astype("int64") / 1e9)
        next_boundary = last_dt.replace(hour=8, minute=0, second=0, microsecond=0)
        if next_boundary <= last_dt:
            next_boundary = next_boundary.replace(day=next_boundary.day + 1)
        start_time = np.datetime64(next_boundary)

    logger.info(f"新批次起始时间: {str(start_time)}")

    # 开始处理数据块
    chunk_idx = start_chunk_idx
    last_time = last_processed_time

    while True:
        end_time = start_time + np.timedelta64(1, "D")

        # 如果数据不足24小时，跳出循环
        if end_time > all_times[-1]:
            break

        # 选择这个时间范围内的数据
        try:
            chunk_data = merged_ds.sel(time=slice(start_time, end_time - np.timedelta64(1, "m")))
        except Exception as e:
            logger.error(f"选择时间范围时出错: {str(e)}")
            break

        if len(chunk_data.time) == 0:
            break

        logger.info(
            f"创建新检查点 {chunk_idx}: {str(start_time)} 到 {str(end_time - np.timedelta64(1, 'm'))}, 包含 {len(chunk_data.time)} 个时间点"
        )

        # 保存新的检查点文件
        # 只用年月日
        start_time_str = np.datetime_as_string(start_time, unit="D").replace("-", "")
        output_file = os.path.join(output_dir, f"prism_checkpoint_{start_time_str}.nc")

        # 添加属性
        chunk_data.attrs.update(
            {
                "description": "重构的PRISM插值降水数据",
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_index": chunk_idx,
                "time_range": f"{str(start_time)} to {str(end_time - np.timedelta64(1, 'm'))}",
            }
        )

        try:
            chunk_data.to_netcdf(output_file)
            logger.info(f"保存检查点文件: {output_file}")
        except Exception as e:
            logger.error(f"保存文件时出错: {str(e)}")

        # 更新下一个时间范围
        start_time = end_time
        chunk_idx += 1
        last_time = chunk_data.time.values[-1]

        # 检查是否超出了当前批次的范围
        if start_time > all_times[-1]:
            break

    # 关闭数据集
    for ds in datasets:
        ds.close()

    return last_time


def restructure_in_batches(input_dir, output_dir, batch_size=5):
    """
    分批处理检查点文件，确保每个重构的检查点包含完整的24小时数据
    """
    # 获取所有文件
    checkpoint_files = get_sorted_checkpoint_files(input_dir)

    if not checkpoint_files:
        logger.error("没有找到检查点文件!")
        return

    logger.info(f"找到 {len(checkpoint_files)} 个检查点文件")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理起始设置
    last_processed_time = None
    chunk_idx = 0

    # 使用重叠批次方法
    i = 0
    total_files = len(checkpoint_files)

    while i < total_files:
        # 计算当前批次结束位置
        end_pos = min(i + batch_size, total_files)

        # 如果不是最后一批，添加额外的文件以确保时间连续性
        if end_pos < total_files:
            current_batch = checkpoint_files[i : end_pos + 1]  # 添加一个额外文件
        else:
            current_batch = checkpoint_files[i:end_pos]

        # 批次编号计算
        batch_num = i // batch_size + 1
        total_batches = (total_files + batch_size - 1) // batch_size

        logger.info(f"处理批次 {batch_num}/{total_batches}，包含 {len(current_batch)} 个文件")

        # 处理当前批次
        last_processed_time = process_batch(
            current_batch, output_dir, start_chunk_idx=chunk_idx, last_processed_time=last_processed_time
        )

        # 更新下一批次的输出文件索引
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("prism_checkpoint_") and f.endswith(".nc")]
        if existing_files:
            try:
                max_idx = max([int(f.replace("prism_checkpoint_", "").replace(".nc", "")) for f in existing_files]) + 1
                chunk_idx = max_idx
            except ValueError:
                logger.warning("解析文件名索引时出错，使用增量方式递增索引")
                chunk_idx += len(existing_files)

        # 重要：向前移动到下一批，但减去1个重叠文件（最后一批除外）
        if end_pos < total_files:
            i = end_pos  # 注意不再减1，因为我们想要下一批次从当前批次的最后一个文件开始
        else:
            i = end_pos

    logger.info("所有批次处理完成!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="重构PRISM检查点文件的时间范围")
    parser.add_argument("--input", default="/mnt/h/DataSet/PrecipGrid/", help="原始检查点文件目录")
    parser.add_argument("--output", default="/mnt/h/DataSet/PrecipGrid_Restructured/", help="新检查点文件输出目录")
    parser.add_argument("--batch-size", type=int, default=5, help="每批次处理的文件数")

    args = parser.parse_args()

    logger.info(f"开始处理，输入目录: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"批次大小: {args.batch_size}")

    restructure_in_batches(args.input, args.output, args.batch_size)
