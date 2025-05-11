import gc
import glob
import logging
import os
import shutil
from datetime import datetime

import dask
import numpy as np
import xarray as xr

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# 设置dask参数，处理大型数据集时避免内存溢出
dask.config.set({"array.slicing.split_large_chunks": True})
# 增加文件缓存大小
xr.set_options(file_cache_maxsize=32)


def merge_precip_and_meteo(
    precip_zarr_path, meteo_folder_path, output_zarr_path, time_chunk_size=24, batch_size=72, resume=True
):
    """
    合并降水站点插值数据和气象数据到一个zarr数据集，使用分批处理减少内存使用，支持断点续传

    参数:
    precip_zarr_path: 降水站点插值数据的zarr文件路径
    meteo_folder_path: 气象数据文件夹路径，包含NC文件
    output_zarr_path: 输出zarr文件路径
    time_chunk_size: 处理数据时的时间维度分块大小
    batch_size: 每批处理的时间点数量
    resume: 是否启用断点续传功能
    """
    logger.info("1. 读取降水站点插值数据...")
    # 使用更小的chunk读取降水站点插值数据
    precip_ds = xr.open_zarr(precip_zarr_path, chunks={"time": time_chunk_size})

    # 获取完整时间范围
    start_date = datetime(2022, 4, 2)
    end_date = datetime(2022, 12, 31, 23)

    # 选择指定的时间范围
    precip_ds = precip_ds.sel(time=slice(start_date, end_date))
    all_times = precip_ds.time.values
    total_times = len(all_times)

    logger.info(f"数据集总时间点数: {total_times}")

    # 获取空间维度信息
    lats = precip_ds.lat.values
    lons = precip_ds.lon.values
    lat_len = len(lats)
    lon_len = len(lons)

    # 检查是否存在已处理的数据并确定从哪个时间点继续
    first_batch = True
    start_batch_idx = 0

    if resume and os.path.exists(output_zarr_path) and os.path.isdir(output_zarr_path):
        try:
            # 尝试读取现有的zarr数据集
            logger.info(f"检测到已有输出文件：{output_zarr_path}，尝试读取以继续处理...")
            existing_ds = xr.open_zarr(output_zarr_path)

            if len(existing_ds.time) > 0:
                # 找到最后一个已处理的时间点
                last_processed_time = existing_ds.time.values[-1]

                # 在原始数据中找到对应索引
                for i, t in enumerate(all_times):
                    if np.datetime64(t) >= np.datetime64(last_processed_time):
                        start_batch_idx = (i // batch_size) * batch_size
                        break

                # 如果找到了有效的索引，则表示有数据需要继续处理
                if start_batch_idx < total_times:
                    logger.info(f"找到上次处理的时间点：{last_processed_time}，将从索引 {start_batch_idx} 继续处理")
                    first_batch = False
                else:
                    logger.info("已处理的数据覆盖了请求的全部时间范围，无需继续处理")
                    return
            else:
                logger.warning("现有的zarr数据集不包含任何时间点，将重新开始处理")
                # 如果数据集存在但没有时间数据，可能是空的或损坏的，删除它重新开始
                shutil.rmtree(output_zarr_path)

        except Exception as e:
            logger.warning(f"无法读取现有的zarr数据集，将重新开始处理: {e}")
            # 如果读取失败，删除可能损坏的数据集并重新开始
            if os.path.exists(output_zarr_path):
                shutil.rmtree(output_zarr_path)

    # 批量处理时间范围
    for i in range(start_batch_idx, total_times, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, total_times)
        batch_times = all_times[batch_start:batch_end]

        logger.info(
            f"处理批次 {i//batch_size + 1}/{(total_times-1)//batch_size + 1}: " f"时间点 {batch_start} 到 {batch_end-1}"
        )

        # 为当前批次选择降水数据
        curr_precip_ds = precip_ds.sel(time=batch_times)

        # 释放一些内存
        gc.collect()

        logger.info("2. 读取对应的气象数据...")
        # 获取所有气象数据文件路径 - 仅处理包含当前批次时间的文件
        batch_start_date = pd.to_datetime(batch_times[0])
        batch_end_date = pd.to_datetime(batch_times[-1])

        # 选择与此批次时间范围相关的气象文件
        meteo_files = []
        all_nc_files = sorted(glob.glob(os.path.join(meteo_folder_path, "*.nc")))

        for file in all_nc_files:
            # 从文件名提取日期信息 - 假设文件名包含YYYYMMDD格式
            try:
                file_basename = os.path.basename(file)
                # 根据实际文件命名规则调整此提取逻辑
                date_part = file_basename.split("_")[0]  # 调整为实际格式
                file_date = datetime.strptime(date_part, "%Y%m%d")

                # 只处理与当前批次相关的文件
                if file_date.date() >= batch_start_date.date() and file_date.date() <= batch_end_date.date():
                    meteo_files.append(file)
            except:
                # 如果无法从文件名解析日期，包含该文件以确保不丢失数据
                meteo_files.append(file)

        if not meteo_files:
            logger.warning(f"在{meteo_folder_path}中没有找到适用于此批次的nc文件，尝试包含所有文件")
            meteo_files = all_nc_files

        logger.info(f"为当前批次找到{len(meteo_files)}个气象数据文件")

        # 以更稳定的方式读取气象数据，使用串行模式并控制块大小
        try:
            # 尝试禁用并行，以更稳定的方式读取
            meteo_ds = xr.open_mfdataset(
                meteo_files, parallel=False, combine="by_coords", chunks={"time": time_chunk_size}
            )

            # 选择当前批次的时间范围
            meteo_ds = meteo_ds.sel(time=slice(batch_start_date, batch_end_date))

            # 找出两个数据集的公共时间点
            common_times = sorted(list(set(curr_precip_ds.time.values) & set(meteo_ds.time.values)))
            if not common_times:
                logger.warning("当前批次没有共同的时间点，跳过")
                continue

            curr_precip_ds = curr_precip_ds.sel(time=common_times)
            meteo_ds = meteo_ds.sel(time=common_times)

            logger.info("3. 处理空间网格匹配...")
            # 将气象数据插值到降水数据的网格上 - 分块处理以减少内存使用
            meteo_ds_interp = meteo_ds.interp(lat=lats, lon=lons, method="linear")

            # 释放原始气象数据的内存
            del meteo_ds
            gc.collect()

            logger.info("4. 合并当前批次数据集...")
            # 从气象数据中去掉PRE变量（如果存在）
            if "PRE" in meteo_ds_interp:
                meteo_ds_interp = meteo_ds_interp.drop_vars("PRE")

            # 合并数据集
            merged_ds = xr.merge([curr_precip_ds, meteo_ds_interp])

            # 释放中间数据集占用的内存
            del curr_precip_ds, meteo_ds_interp
            gc.collect()

            # 添加属性说明
            merged_ds.attrs["description"] = "合并的降水站点插值数据与气象数据"
            merged_ds.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            merged_ds.attrs["batch"] = f"{batch_start}-{batch_end-1}"

            # 设置更小的chunk大小
            chunks = {"time": min(50, len(merged_ds.time)), "lat": min(30, lat_len), "lon": min(50, lon_len)}
            merged_ds = merged_ds.chunk(chunks)

            # 保存当前批次数据
            logger.info(f"5. 保存批次结果到{output_zarr_path}...")

            if first_batch:
                # 第一批次使用覆盖模式，并提供编码信息
                encoding = {}
                for var_name in merged_ds.data_vars:
                    encoding[var_name] = {"chunks": None}

                merged_ds.to_zarr(output_zarr_path, mode="w", encoding=encoding, consolidated=True)
                first_batch = False
            else:
                # 后续批次使用追加模式，不提供编码信息
                merged_ds.to_zarr(output_zarr_path, mode="a", append_dim="time", consolidated=True)

            # 释放合并数据集的内存
            del merged_ds
            gc.collect()

            logger.info(f"批次 {i//batch_size + 1} 处理完成")

        except Exception as e:
            logger.error(f"处理批次 {i//batch_size + 1} 时出错: {e}")
            # 继续下一个批次

    logger.info("所有批次处理完成!")


if __name__ == "__main__":
    # 导入pandas - 仅在主程序中需要
    import pandas as pd

    # 请替换为实际路径
    precip_zarr_path = "/mnt/h/DataSet/PreGrids_IDW/temp_output.zarr"
    meteo_folder_path = "/mnt/h/DataSet/Grids/"
    output_zarr_path = "/mnt/h/Merge/merged_data.zarr"

    # 运行合并函数，使用较小的批次大小以减少内存使用
    merge_precip_and_meteo(
        precip_zarr_path=precip_zarr_path,
        meteo_folder_path=meteo_folder_path,
        output_zarr_path=output_zarr_path,
        time_chunk_size=24,  # 一天的数据量
        batch_size=72,  # 每批处理3天的数据
        resume=True,  # 启用断点续传功能
    )
