import gc
import glob
import logging
import os
import shutil
import time
from datetime import datetime

import dask
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# 设置dask参数，处理大型数据集时避免内存溢出
dask.config.set({"array.slicing.split_large_chunks": True})
# 增加文件缓存大小
xr.set_options(file_cache_maxsize=64)


def merge_precip_and_meteo_nc(
    precip_zarr_path, meteo_folder_path, output_nc_path, time_chunk_size=24, batch_size=168, resume=True, temp_dir=None
):
    """
    合并降水站点插值数据和气象数据到一个NetCDF数据集，使用分批处理减少内存使用，支持断点续传

    参数:
    precip_zarr_path: 降水站点插值数据的zarr文件路径
    meteo_folder_path: 气象数据文件夹路径，包含NC文件
    output_nc_path: 输出NC文件路径
    time_chunk_size: 处理数据时的时间维度分块大小
    batch_size: 每批处理的时间点数量
    resume: 是否启用断点续传功能
    temp_dir: 临时文件目录，如果为None则使用output_nc_path所在目录
    """
    start_time = time.time()

    # 设置临时目录
    if temp_dir is None:
        temp_dir = os.path.dirname(output_nc_path)
    os.makedirs(temp_dir, exist_ok=True)

    # 临时NC文件的基本名称
    base_filename = os.path.basename(output_nc_path).split(".")[0]

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
    start_batch_idx = 0
    processed_batches = []

    if resume:
        # 查找已处理的临时文件
        pattern = os.path.join(temp_dir, f"{base_filename}_batch_*.nc")
        existing_batch_files = sorted(glob.glob(pattern))

        if existing_batch_files:
            logger.info(f"找到{len(existing_batch_files)}个已处理的批次文件")

            # 读取最后一个批次文件，获取其时间范围
            try:
                last_batch_ds = xr.open_dataset(existing_batch_files[-1])
                last_processed_time = last_batch_ds.time.values[-1]
                last_batch_ds.close()

                # 寻找对应的索引位置
                for i, t in enumerate(all_times):
                    if np.datetime64(t) > np.datetime64(last_processed_time):
                        start_batch_idx = (i // batch_size) * batch_size
                        break

                logger.info(f"将从索引 {start_batch_idx} (批次 {start_batch_idx // batch_size + 1}) 继续处理")
                processed_batches = existing_batch_files
            except Exception as e:
                logger.warning(f"读取已处理批次文件时出错: {e}")
                logger.warning("将重新开始处理")
                # 清除所有临时文件
                for file in existing_batch_files:
                    try:
                        os.remove(file)
                    except:
                        pass

    # 获取nc文件列表
    all_nc_files = sorted(glob.glob(os.path.join(meteo_folder_path, "*.nc")))
    if not all_nc_files:
        raise ValueError(f"在{meteo_folder_path}中没有找到nc文件")

    # 预先生成所有批次的元数据
    total_batches = (total_times - 1) // batch_size + 1
    logger.info(f"总批次数: {total_batches}")

    # 批量处理时间范围
    for i in range(start_batch_idx, total_times, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, total_times)
        batch_times = all_times[batch_start:batch_end]

        batch_number = i // batch_size + 1
        batch_file = os.path.join(temp_dir, f"{base_filename}_batch_{batch_number:04d}.nc")

        # 检查此批次是否已处理
        if batch_file in processed_batches:
            logger.info(f"批次 {batch_number}/{total_batches} 已处理，跳过")
            continue

        logger.info(f"处理批次 {batch_number}/{total_batches}: 时间点 {batch_start} 到 {batch_end-1}")

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

        # 优化文件筛选 - 使用集合提高查找效率
        date_range = set()
        current_date = batch_start_date.date()
        while current_date <= batch_end_date.date():
            date_range.add(current_date)
            current_date = current_date + pd.Timedelta(days=1)

        for file in all_nc_files:
            # 从文件名提取日期信息 - 假设文件名包含YYYYMMDD格式
            try:
                file_basename = os.path.basename(file)
                # 根据实际文件命名规则调整此提取逻辑
                date_part = file_basename.split("_")[0]  # 调整为实际格式
                file_date = datetime.strptime(date_part, "%Y%m%d")

                # 只处理与当前批次相关的文件
                if file_date.date() in date_range:
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
            # 启用并行计算模式进行插值以加速
            with dask.config.set(scheduler="threads", num_workers=4):
                # 将气象数据插值到降水数据的网格上
                meteo_ds_interp = meteo_ds.interp(lat=lats, lon=lons, method="linear")

            # 释放原始气象数据的内存
            del meteo_ds
            gc.collect()

            logger.info("4. 合并当前批次数据集...")

            # 合并数据集
            merged_ds = xr.merge([curr_precip_ds, meteo_ds_interp])

            # 释放中间数据集占用的内存
            del curr_precip_ds, meteo_ds_interp
            gc.collect()

            # 添加属性说明
            merged_ds.attrs["description"] = "合并的降水站点插值数据与气象数据"
            merged_ds.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            merged_ds.attrs["batch"] = f"{batch_start}-{batch_end-1}"

            # 优化chunk大小 - 增加时间维度的chunk大小以减少写入次数
            chunks = {"time": min(100, len(merged_ds.time)), "lat": min(30, lat_len), "lon": min(50, lon_len)}
            merged_ds = merged_ds.chunk(chunks)

            # 保存当前批次数据到临时文件
            logger.info(f"5. 保存批次结果到临时文件: {batch_file}")
            save_start_time = time.time()

            # 使用压缩设置提高NC文件写入效率，并使用float32减小文件大小
            encoding = {}
            for var_name in merged_ds.data_vars:
                # 检查变量类型，只对浮点类型变量进行转换
                if np.issubdtype(merged_ds[var_name].dtype, np.floating):
                    encoding[var_name] = {
                        "zlib": True,  # 启用压缩
                        "complevel": 5,  # 压缩级别
                        "dtype": "float32",  # 转换为float32
                        "chunksizes": [min(len(merged_ds.time), 100), min(lat_len, 30), min(lon_len, 50)],  # 块大小
                    }
                else:
                    encoding[var_name] = {
                        "zlib": True,  # 启用压缩
                        "complevel": 5,  # 压缩级别
                        "chunksizes": [min(len(merged_ds.time), 100), min(lat_len, 30), min(lon_len, 50)],  # 块大小
                    }
            # 写入NC文件，使用优化的编码设置
            merged_ds.to_netcdf(batch_file, encoding=encoding)

            save_duration = time.time() - save_start_time
            logger.info(f"批次保存完成，耗时：{save_duration:.2f}秒")

            # 更新已处理批次列表
            processed_batches.append(batch_file)

            # 释放合并数据集的内存
            del merged_ds
            gc.collect()

            logger.info(f"批次 {batch_number} 处理完成")

        except Exception as e:
            logger.error(f"处理批次 {batch_number} 时出错: {e}")
            # 继续下一个批次

    # 合并所有临时批次文件
    if processed_batches:
        logger.info(f"开始合并{len(processed_batches)}个批次文件到最终NC文件: {output_nc_path}")
        merge_start_time = time.time()

        try:
            # 尝试使用xr.open_mfdataset高效合并文件
            combined_ds = xr.open_mfdataset(
                processed_batches,
                combine="by_coords",
                parallel=True,  # 可以启用并行以加速
                chunks={"time": 100, "lat": 30, "lon": 50},
            )

            # 确保时间维度已排序
            combined_ds = combined_ds.sortby("time")

            # 设置最终输出的压缩编码
            encoding = {}
            for var_name in combined_ds.data_vars:
                # 检查变量类型，只对浮点类型变量进行转换
                if np.issubdtype(combined_ds[var_name].dtype, np.floating):
                    encoding[var_name] = {
                        "zlib": True,  # 启用压缩
                        "complevel": 5,  # 压缩级别
                        "dtype": "float32",  # 转换为float32
                    }
                else:
                    encoding[var_name] = {"zlib": True, "complevel": 5}  # 启用压缩  # 压缩级别
            # 保存最终合并文件
            logger.info(f"写入最终NC文件: {output_nc_path}")
            combined_ds.to_netcdf(output_nc_path, encoding=encoding)
            combined_ds.close()

            # 清理临时文件
            logger.info("清理临时批次文件")
            for temp_file in processed_batches:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"无法删除临时文件 {temp_file}: {e}")

        except Exception as e:
            logger.error(f"合并批次文件时出错: {e}")
            logger.info("尝试使用替代方法合并...")

            # 替代方法：逐个读取并合并到一个新数据集
            try:
                # 读取第一个文件作为基础
                combined_ds = xr.open_dataset(processed_batches[0])

                # 依次合并其他文件
                for i, file in enumerate(processed_batches[1:], 1):
                    logger.info(f"合并批次文件 {i}/{len(processed_batches)-1}")
                    temp_ds = xr.open_dataset(file)
                    combined_ds = xr.concat([combined_ds, temp_ds], dim="time")
                    temp_ds.close()

                # 确保时间维度已排序
                combined_ds = combined_ds.sortby("time")

                # 保存最终合并文件
                logger.info(f"写入最终NC文件: {output_nc_path}")
                # 创建包含dtype='float32'的编码字典
                encoding = {}
                for var in combined_ds.data_vars:
                    if np.issubdtype(combined_ds[var].dtype, np.floating):
                        encoding[var] = {"zlib": True, "complevel": 5, "dtype": "float32"}
                    else:
                        encoding[var] = {"zlib": True, "complevel": 5}

                combined_ds.to_netcdf(output_nc_path, encoding=encoding)
                combined_ds.close()

                # 清理临时文件
                logger.info("清理临时批次文件")
                for temp_file in processed_batches:
                    try:
                        os.remove(temp_file)
                    except:
                        pass

            except Exception as e2:
                logger.error(f"替代合并方法也失败: {e2}")
                logger.info("保留临时批次文件，请手动合并")

        merge_duration = time.time() - merge_start_time
        logger.info(f"合并操作完成，耗时：{merge_duration:.2f}秒")
    else:
        logger.warning("没有找到任何已处理的批次文件，无法创建最终NC文件")

    # 计算总耗时
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"所有处理完成! 总耗时: {int(hours)}小时 {int(minutes)}分 {seconds:.2f}秒")


if __name__ == "__main__":
    # 请替换为实际路径
    precip_zarr_path = "/mnt/h/DataSet/PreGrids_IDW/temp_output.zarr"
    meteo_folder_path = "/mnt/h/DataSet/Grids/"
    output_nc_path = "/mnt/h/DataSet/Merge/merged_data.nc"
    temp_dir = "/mnt/h/DataSet/Merge/temp"

    # 运行合并函数，使用更大的批次大小以减少写入次数
    merge_precip_and_meteo_nc(
        precip_zarr_path=precip_zarr_path,
        meteo_folder_path=meteo_folder_path,
        output_nc_path=output_nc_path,
        time_chunk_size=24,  # 一天的数据量
        batch_size=24,  # 每批处理7天的数据，大幅减少写入次数
        resume=True,  # 启用断点续传功能
        temp_dir=temp_dir,  # 临时文件存储目录
    )
