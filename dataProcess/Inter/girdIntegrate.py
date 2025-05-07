import argparse
import gc
import glob
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import cfgrib  # 替换pygrib为cfgrib
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


# 配置日志记录
def setup_logging(log_file="process.log"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 如果已经有处理程序，不要添加新的
    if not logger.handlers:
        # 文件处理程序
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # 控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

    return logger


# 从文件名中获取时间戳和小时信息
def parse_filename(filename):
    basename = os.path.basename(filename)
    # 提取时间戳 (BABJ_20220402220521)
    timestamp_match = re.search(r"BABJ_(\d{14})_P", basename)
    timestamp = timestamp_match.group(1) if timestamp_match else None

    # 提取小时信息 (2022040222)
    hour_match = re.search(r"HOR-[A-Z]+-(\d{10})\.GRB2", basename)
    hour_id = hour_match.group(1) if hour_match else None
    hour = hour_id[-2:] if hour_id else None

    # 提取数据类型 (TAIR, UWIN, VWIN, PRE)
    data_type_match = re.search(r"HOR-([A-Z]+)-\d{10}\.GRB2", basename)
    data_type = data_type_match.group(1) if data_type_match else None

    return timestamp, hour, data_type


# 获取DEM的边界框
def get_dem_bounds(dem_file, logger):
    try:
        ds = xr.open_dataset(dem_file)

        # 获取经纬度范围
        if "lat" in ds.coords:
            lat_min = float(ds.lat.min().values)
            lat_max = float(ds.lat.max().values)
        elif "latitude" in ds.coords:
            lat_min = float(ds.latitude.min().values)
            lat_max = float(ds.latitude.max().values)
        else:
            logger.error("在DEM文件中未找到纬度坐标")
            return None

        if "lon" in ds.coords:
            lon_min = float(ds.lon.min().values)
            lon_max = float(ds.lon.max().values)
        elif "longitude" in ds.coords:
            lon_min = float(ds.longitude.min().values)
            lon_max = float(ds.longitude.max().values)
        else:
            logger.error("在DEM文件中未找到经度坐标")
            return None

        logger.info(f"DEM边界: 纬度 [{lat_min}, {lat_max}], 经度 [{lon_min}, {lon_max}]")
        return lat_min, lat_max, lon_min, lon_max

    except Exception as e:
        logger.error(f"读取DEM文件出错 {dem_file}: {e}")
        return None


# 裁切GRB2数据 - 使用cfgrib替代pygrib
def crop_grb_data(grb_file, dem_bounds, logger):
    try:
        result = {}
        # 使用with语句自动关闭数据集，防止内存泄漏
        with xr.open_dataset(grb_file, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
            # 获取边界框参数
            lat_min, lat_max, lon_min, lon_max = dem_bounds

            # 确定经纬度坐标名称
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"

            # 裁剪数据集到边界框
            try:
                ds_cropped = ds.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
            except Exception as e:
                logger.warning(f"使用slice裁剪失败，尝试使用where方法: {e}")
                ds_cropped = ds.where(
                    (ds[lat_name] >= lat_min)
                    & (ds[lat_name] <= lat_max)
                    & (ds[lon_name] >= lon_min)
                    & (ds[lon_name] <= lon_max),
                    drop=True,
                )

            # 遍历数据集中的变量
            for var_name in ds_cropped.data_vars:
                var = ds_cropped[var_name]
                lats = ds_cropped[lat_name].values
                lons = ds_cropped[lon_name].values
                data = var.values

                if data.ndim >= 2:
                    if data.ndim > 2:
                        data = data[0]
                    short_name = var.attrs.get("GRIB_shortName", var_name)
                    flat_data = data.flatten()
                    if lats.ndim == 1 and lons.ndim == 1:
                        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
                        flat_lats = lat_mesh.flatten()
                        flat_lons = lon_mesh.flatten()
                    else:
                        flat_lats = lats.flatten()
                        flat_lons = lons.flatten()
                    valid_mask = ~np.isnan(flat_data)
                    if np.any(valid_mask):
                        cropped_data = flat_data[valid_mask]
                        cropped_lats = flat_lats[valid_mask]
                        cropped_lons = flat_lons[valid_mask]
                        if short_name not in result:
                            result[short_name] = (cropped_data, cropped_lats, cropped_lons)

            # 显式关闭裁剪后的数据集（如果不是原始ds的视图）
            if hasattr(ds_cropped, "close"):
                ds_cropped.close()
            del ds_cropped
        # 强制垃圾回收
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"处理文件出错 {grb_file}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


# 为每小时选择最接近整点的文件，对于PRE(降水)类型保留所有文件
def select_hourly_files(files_by_hour, data_type, logger):
    selected_files = {}

    # 如果是降水数据(PRE)，保留该小时的所有文件
    if data_type == "PRE":
        return files_by_hour  # 直接返回所有PRE文件，按小时分组

    # 对其他数据类型，仍然只选择最接近整点的文件
    for hour, files in files_by_hour.items():
        if len(files) == 1:
            selected_files[hour] = files[0]
            continue

        # 计算每个文件的时间戳离整点的距离
        best_file = None
        min_diff = float("inf")

        for file in files:
            timestamp, _, _ = parse_filename(file)
            if timestamp:
                minutes = int(timestamp[10:12])
                seconds = int(timestamp[12:14])
                time_diff = minutes * 60 + seconds

                if time_diff < min_diff:
                    min_diff = time_diff
                    best_file = file

        if best_file:
            selected_files[hour] = best_file
            logger.debug(f"已为{hour}时选择文件: {os.path.basename(best_file)}")
        else:
            logger.warning(f"无法为{hour}时选择文件")

    return selected_files


def process_single_date(date_dir, dem_bounds, output_dir, existing_nc_files, logger):
    date_name = os.path.basename(date_dir)
    nc_basename = f"{date_name}_ALL"
    if nc_basename in existing_nc_files:
        logger.info(f"  [跳过] {nc_basename}.nc 已存在。")
        return

    logger.info(f"  >>> 处理日期: {date_name}")

    # 收集所有小时、所有变量的数据
    daily_data = {}
    all_hours = set()
    unique_lats, unique_lons = None, None  # 只计算一次

    for data_type in ["TAIR", "UWIN", "VWIN", "PRE"]:
        data_files = glob.glob(os.path.join(date_dir, f"*HOR-{data_type}-*.GRB2"))
        if not data_files:
            logger.warning(f"    [警告] {date_name} 未找到 {data_type} 类型文件。")
            continue

        hourly_files = {}
        for file in data_files:
            _, hour, _ = parse_filename(file)
            if hour:
                if hour not in hourly_files:
                    hourly_files[hour] = []
                hourly_files[hour].append(file)

        selected_files = select_hourly_files(hourly_files, data_type, logger)

        # 处理非PRE类型数据（每小时一个文件）
        if data_type != "PRE":
            for hour, file in selected_files.items():
                logger.info(f"    处理 {data_type} {hour} 时文件: {os.path.basename(file)}")
                result = crop_grb_data(file, dem_bounds, logger)
                if result and hour not in daily_data:
                    daily_data[hour] = {}
                if result:
                    for var in result:
                        daily_data[hour][data_type] = result[var]
                        # 只在第一次裁剪时获取经纬度
                        if unique_lats is None or unique_lons is None:
                            unique_lats = np.unique(result[var][1])
                            unique_lons = np.unique(result[var][2])
                    all_hours.add(hour)
        # 处理PRE类型数据（累加同一小时的所有文件）
        else:
            for hour, files in selected_files.items():
                if hour not in daily_data:
                    daily_data[hour] = {}

                # 初始化累加数组和计数
                accumulated_data = None
                precipitation_lats = None
                precipitation_lons = None
                file_count = 0  # 新增计数器

                logger.info(f"    处理 PRE {hour} 时的 {len(files)} 个文件")
                for file in files:
                    logger.debug(f"      累加文件: {os.path.basename(file)}")
                    result = crop_grb_data(file, dem_bounds, logger)
                    if result:
                        for var in result:
                            data, lats, lons = result[var]

                            # 如果是第一个文件，初始化累加变量
                            if accumulated_data is None:
                                accumulated_data = data.copy()
                                precipitation_lats = lats.copy()
                                precipitation_lons = lons.copy()

                                # 只在第一次裁剪时获取经纬度（如果还未设置）
                                if unique_lats is None or unique_lons is None:
                                    unique_lats = np.unique(lats)
                                    unique_lons = np.unique(lons)
                            else:
                                # 累加降水数据
                                accumulated_data += data
                            file_count += 1  # 每处理一个文件计数加一

                # 保存平均后的降水数据
                if accumulated_data is not None and file_count > 0:
                    mean_data = accumulated_data / file_count  # 计算平均
                    daily_data[hour]["PRE"] = (mean_data, precipitation_lats, precipitation_lons)
                    all_hours.add(hour)
    if not daily_data:
        logger.warning(f"  [跳过] {date_name} 无有效数据。")
        return

    # 确保经纬度有值
    if unique_lats is None or unique_lons is None:
        logger.error(f"  [失败] {date_name} 未能获取经纬度信息。")
        return

    unique_lats.sort()
    unique_lons.sort()

    # 统一小时顺序：从8点到次日8点
    def hour_sort_key(h):
        h_int = int(h)
        return h_int if h_int >= 8 else h_int + 24

    sorted_hours = sorted(all_hours, key=hour_sort_key)

    output_file = os.path.join(output_dir, f"{date_name}_ALL.nc")
    logger.info(f"  正在保存 {date_name} 合并NC文件: {output_file}")
    if save_daily_to_netcdf(
        daily_data, sorted_hours, unique_lats, unique_lons, output_file, logger, date_name=date_name
    ):
        logger.info(f"  [完成] {date_name} 数据已保存。")
    else:
        logger.error(f"  [失败] {date_name} 数据保存失败。")


def save_daily_to_netcdf(daily_data, hours, unique_lats, unique_lons, output_file, logger, date_name=None):
    try:
        ds = xr.Dataset()
        # 生成时间轴
        if date_name is not None:
            base_date = pd.to_datetime(date_name, format="%Y%m%d")
            times = []
            for h in hours:
                hour_int = int(h)
                if hour_int < 8:
                    dt = base_date + pd.Timedelta(days=1)
                else:
                    dt = base_date
                times.append(pd.to_datetime(dt.strftime("%Y%m%d") + h, format="%Y%m%d%H"))
        else:
            times = [pd.to_datetime(h, format="%H") for h in hours]
        ds = ds.assign_coords(
            time=("time", times),
            lat=("lat", unique_lats),
            lon=("lon", unique_lons),
        )

        # 优化：提前构建经纬度到索引的映射
        lat_to_idx = {v: i for i, v in enumerate(unique_lats)}
        lon_to_idx = {v: i for i, v in enumerate(unique_lons)}

        for var_name in ["TAIR", "UWIN", "VWIN", "PRE"]:
            data_arr = np.full((len(hours), len(unique_lats), len(unique_lons)), np.nan)
            for t_idx, hour in enumerate(hours):
                if hour in daily_data and var_name in daily_data[hour]:
                    data, lats, lons = daily_data[hour][var_name]
                    # 向量化索引查找，减少循环
                    lat_indices = np.array([lat_to_idx.get(lat, -1) for lat in lats])
                    lon_indices = np.array([lon_to_idx.get(lon, -1) for lon in lons])
                    valid_mask = (lat_indices >= 0) & (lon_indices >= 0)
                    data_arr[t_idx, lat_indices[valid_mask], lon_indices[valid_mask]] = data[valid_mask]
            ds[var_name] = xr.DataArray(
                data_arr,
                dims=["time", "lat", "lon"],
                coords={"time": times, "lat": unique_lats, "lon": unique_lons},
                attrs={"units": "mm", "long_name": var_name},
            )

        ds.attrs["title"] = "每日合并气象数据"
        ds.attrs["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.attrs["description"] = "每日TAIR、UWIN、VWIN、PRE合并，按time/lat/lon组织"
        ds.to_netcdf(output_file)
        logger.info(f"成功保存数据至 {output_file}")
        return True
    except Exception as e:
        logger.error(f"保存NC文件出错 {output_file}: {e}")
        return False


# 主处理函数
def process_data(root_dir, dem_file, output_dir, n_jobs=4):
    logger = setup_logging()
    logger.info("========== 开始数据处理 ==========")

    dem_bounds = get_dem_bounds(dem_file, logger)
    if dem_bounds is None:
        logger.error("获取DEM边界失败，程序终止。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 直接读取目标文件夹，判断已存在的nc文件
    existing_nc_files = set(os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith("_ALL.nc"))

    month_dirs = sorted(glob.glob(os.path.join(root_dir, "2022*")))
    logger.info(f"共检测到 {len(month_dirs)} 个待处理月份目录。")
    for month_dir in month_dirs:
        month_name = os.path.basename(month_dir)
        logger.info(f"------ 开始处理月份: {month_name} ------")

        date_dirs = sorted(glob.glob(os.path.join(month_dir, month_name + "*")))
        logger.info(f"  {month_name} 包含 {len(date_dirs)} 个日期目录。")

        # 使用joblib并行处理每一天
        Parallel(n_jobs=n_jobs)(
            delayed(process_single_date)(date_dir, dem_bounds, output_dir, existing_nc_files, logger)
            for date_dir in date_dirs
        )

    logger.info("========== 数据处理全部完成 ==========")


# 脚本入口
if __name__ == "__main__":

    root_dir = "/mnt/h/DataSet/2022/"
    dem_file = "/mnt/h/DataSet/3-DEM/DEM_clip.nc"
    out_dir = "/mnt/h/DataSet/Grids/"

    process_data(root_dir, dem_file, out_dir, n_jobs=1)
