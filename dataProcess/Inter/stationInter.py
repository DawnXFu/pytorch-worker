import glob
import json
import os
import time

import numba
import numpy as np
import pandas as pd
import xarray as xr
from filelock import FileLock
from joblib import Parallel, delayed, parallel_backend
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm


@numba.njit(parallel=True)
def calculate_distances_and_weights(col_lats, col_lons, valid_lats, valid_lons):
    """计算距离矩阵和零距离掩码的Numba加速函数"""
    n_points = len(col_lats)
    n_stations = len(valid_lats)

    # 初始化距离矩阵
    distances = np.zeros((n_points, n_stations), dtype=np.float32)
    zero_dist_mask = np.zeros((n_points, n_stations), dtype=np.bool_)

    # 并行计算所有距离
    for i in numba.prange(n_points):
        for j in range(n_stations):
            lat_diff = col_lats[i] - valid_lats[j]
            lon_diff = col_lons[i] - valid_lons[j]
            dist = np.sqrt(lat_diff**2 + lon_diff**2)
            distances[i, j] = dist
            zero_dist_mask[i, j] = dist < 1e-10

    # 计算零距离和非零距离掩码
    has_zero_dist = np.zeros(n_points, dtype=np.bool_)
    for i in range(n_points):
        for j in range(n_stations):
            if zero_dist_mask[i, j]:
                has_zero_dist[i] = True
                break

    return distances, zero_dist_mask, has_zero_dist


@numba.njit
def handle_zero_distances(idx, zero_dist_mask, valid_values, n_stations):
    """处理零距离点的Numba加速函数"""
    station_count = 0
    value_sum = 0.0

    for j in range(n_stations):
        if zero_dist_mask[idx, j]:
            value_sum += valid_values[j]
            station_count += 1

    if station_count > 0:
        return value_sum / station_count
    else:
        return 0.0


@numba.njit
def calculate_idw_weights(distances, power=2.0):
    """计算IDW权重的Numba加速函数"""
    weights = 1.0 / (distances**power + 1e-10)
    # 不使用keepdims参数，手动reshape
    weights_sum = np.sum(weights, axis=1)
    # 手动调整形状以匹配原始行为
    weights_sum = weights_sum.reshape(weights_sum.shape[0], 1)
    return weights / weights_sum


@numba.njit
def predict_linear_model(elevs, coef, intercept):
    """线性模型预测的Numba加速函数"""
    return coef * elevs + intercept


@numba.njit(parallel=True)
def process_grid_column(
    col_lats,
    col_lons,
    col_elevs,
    col_regions,
    valid_lats,
    valid_lons,
    valid_values,
    region_coefs,
    region_intercepts,
    station_residuals,
    n_regions,
    has_model,
):
    """处理单列网格点的Numba加速函数"""
    n_points = len(col_lats)
    n_stations = len(valid_lats)
    result = np.zeros(n_points, dtype=np.float32)

    # 计算距离矩阵
    distances, zero_dist_mask, has_zero_dist = calculate_distances_and_weights(
        col_lats, col_lons, valid_lats, valid_lons
    )

    # 处理零距离点
    for i in range(n_points):
        if has_zero_dist[i]:
            result[i] = handle_zero_distances(i, zero_dist_mask, valid_values, n_stations)

    # 计算非零距离点的IDW权重
    non_zero_mask = ~has_zero_dist
    if np.any(non_zero_mask):
        # 计算IDW权重
        idw_weights = calculate_idw_weights(distances[non_zero_mask])

        # 处理每个区域
        for r in range(n_regions):
            # 找出当前区域的网格点
            region_points_mask = np.zeros(n_points, dtype=np.bool_)
            for i in range(n_points):
                if non_zero_mask[i] and col_regions[i] == r:
                    region_points_mask[i] = True

            if not np.any(region_points_mask):
                continue

            # 检查该区域是否有模型
            if has_model[r]:
                # 使用模型预测值 + IDW残差
                for i in numba.prange(n_points):
                    if region_points_mask[i]:
                        # 获取该点在non_zero_mask中的索引
                        non_zero_idx = np.sum(non_zero_mask[:i])

                        # 模型预测
                        pred_value = predict_linear_model(col_elevs[i], region_coefs[r], region_intercepts[r])

                        # 残差IDW插值
                        residual_sum = 0.0
                        for j in range(n_stations):
                            residual_sum += idw_weights[non_zero_idx, j] * station_residuals[r, j]

                        # 最终结果 = 模型预测 + 残差插值
                        result[i] = pred_value + residual_sum
            else:
                # 无模型，使用简单IDW
                for i in numba.prange(n_points):
                    if region_points_mask[i]:
                        # 获取该点在non_zero_mask中的索引
                        non_zero_idx = np.sum(non_zero_mask[:i])

                        # 直接计算IDW
                        value_sum = 0.0
                        for j in range(n_stations):
                            value_sum += idw_weights[non_zero_idx, j] * valid_values[j]

                        result[i] = value_sum

    return result


@numba.njit(parallel=True)
def idw_numba_kernel(grid_points, points, values, p=2, epsilon=1e-10, search_radius=0.0, min_points=1, max_points=0):
    """增强版Numba加速的IDW核心计算函数

    参数:
    grid_points: 网格点坐标数组 [n_grid, 2]
    points: 站点坐标数组 [n_points, 2]
    values: 站点观测值数组 [n_points]
    p: IDW幂指数，默认为2
    epsilon: 防止除零的小量，默认为1e-10
    search_radius: 搜索半径，0表示使用所有站点
    min_points: 最少有效站点数，少于此数使用全部站点
    max_points: 每个网格点最多使用的站点数，0表示使用所有站点
    """
    n_grid = len(grid_points)
    n_points = len(points)
    result = np.zeros(n_grid, dtype=np.float32)
    use_radius = search_radius > 0.0
    use_max_points = max_points > 0 and max_points < n_points

    for i in numba.prange(n_grid):
        # 初始化距离和值数组
        dists = np.zeros(n_points, dtype=np.float32)
        valid_count = 0

        # 计算所有距离
        for j in range(n_points):
            dist = np.sqrt(((grid_points[i, 0] - points[j, 0]) ** 2) + ((grid_points[i, 1] - points[j, 1]) ** 2))
            dists[j] = dist
            # 检查是否在搜索半径内
            if not use_radius or dist <= search_radius:
                valid_count += 1

        # 如果有效点不足，忽略半径限制
        if valid_count < min_points and use_radius:
            valid_count = n_points

        # 处理零距离点
        zero_dist_indices = np.zeros(n_points, dtype=np.int32)
        zero_count = 0

        for j in range(n_points):
            if dists[j] < epsilon:
                zero_dist_indices[zero_count] = j
                zero_count += 1

        # 如果有零距离点，直接取平均
        if zero_count > 0:
            value_sum = 0.0
            for k in range(zero_count):
                j = zero_dist_indices[k]
                value_sum += values[j]
            result[i] = value_sum / zero_count
            continue

        # 限制使用点数量 (仅保留最近的max_points个点)
        if use_max_points and valid_count > max_points:
            # 创建索引数组来排序距离
            indices = np.argsort(dists)
            valid_count = max_points
        else:
            # 使用所有点，或搜索半径内的点
            indices = np.arange(n_points)

        # 计算权重和加权和
        weight_sum = 0.0
        value_sum = 0.0

        # 使用有效点计算IDW
        actual_points = 0
        for k in range(min(valid_count, n_points)):
            j = indices[k]

            # 如果使用半径限制，检查距离
            if use_radius and dists[j] > search_radius:
                continue

            weight = 1.0 / (dists[j] ** p + epsilon)
            weight_sum += weight
            value_sum += weight * values[j]
            actual_points += 1

        # 如果没有有效点，使用全部点
        if actual_points == 0:
            for j in range(n_points):
                weight = 1.0 / (dists[j] ** p + epsilon)
                weight_sum += weight
                value_sum += weight * values[j]

        # 计算最终结果
        result[i] = value_sum / weight_sum

    return result


class PrecipitationInterpolation:
    def __init__(self, station_file, era5_folder, dem_file, output_folder):
        """
        初始化插值类

        参数:
        station_file: 站点降水数据文件路径
        era5_folder: ERA5降水数据文件夹路径
        dem_file: DEM数据文件路径
        output_folder: 输出结果文件夹路径
        """
        self.station_file = station_file
        self.era5_folder = era5_folder
        self.dem_file = dem_file
        self.output_folder = output_folder

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载站点数据、ERA5数据和DEM数据"""
        print("加载数据...")

        # 1. 加载站点数据
        self.station_ds = xr.open_dataset(self.station_file)
        print(f"站点数据时间范围: {self.station_ds.time.values[0]} 到 {self.station_ds.time.values[-1]}")
        print(f"站点数量: {len(self.station_ds.station)}")

        # 2. 惰性加载ERA5数据，使用chunks减少内存使用
        era5_files = sorted(glob.glob(os.path.join(self.era5_folder, "*.nc")))
        print(f"找到 {len(era5_files)} 个ERA5文件")
        self.era5_ds = xr.open_mfdataset(
            era5_files, combine="by_coords", chunks={"time": 10}, engine="netcdf4"  # 按时间分块处理
        )

        # 重命名ERA5数据的维度以便后续处理
        self.era5_ds = self.era5_ds.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
        # 转换数据类型为float32，减少内存占用
        if self.era5_ds["tp"].dtype != np.float32:
            self.era5_ds["tp"] = self.era5_ds["tp"].astype(np.float32)
        print(f"ERA5数据时间范围: {self.era5_ds.time.values[0]} 到 {self.era5_ds.time.values[-1]}")

        # 3. 延迟加载DEM数据
        # 仅存储路径，在需要时再加载
        self.dem_data = self.preprocess_dem()
        print("DEM数据已加载")

        # 确保所有数据在相同的时间范围内
        common_times = np.intersect1d(self.station_ds.time, self.era5_ds.time)
        self.station_ds = self.station_ds.sel(time=common_times)
        self.era5_ds = self.era5_ds.sel(time=common_times)
        print(f"共同时间点数量: {len(common_times)}")

        # 提前提取站点位置信息，避免重复计算
        self.station_lats = self.station_ds["lat"].values
        self.station_lons = self.station_ds["lon"].values

        # 预计算每个站点对应的ERA5网格点索引，避免重复计算
        print("预计算站点对应的ERA5网格点索引...")
        era5_lats = self.era5_ds.lat.values
        era5_lons = self.era5_ds.lon.values

        self.station_era5_lat_indices = []
        self.station_era5_lon_indices = []

        for i in range(len(self.station_lats)):
            lat_idx = np.abs(era5_lats - self.station_lats[i]).argmin()
            lon_idx = np.abs(era5_lons - self.station_lons[i]).argmin()
            self.station_era5_lat_indices.append(lat_idx)
            self.station_era5_lon_indices.append(lon_idx)

        # 转换为numpy数组以便更高效地索引
        self.station_era5_lat_indices = np.array(self.station_era5_lat_indices)
        self.station_era5_lon_indices = np.array(self.station_era5_lon_indices)

        # 预计算站点高程，避免重复计算
        print("预计算站点高程...")
        self.station_dem = np.zeros(len(self.station_lats), dtype=np.float32)
        for i in range(len(self.station_lats)):
            lat = self.station_lats[i]
            lon = self.station_lons[i]
            lat_idx = np.abs(self.dem_data.lat.values - lat).argmin()
            lon_idx = np.abs(self.dem_data.lon.values - lon).argmin()
            self.station_dem[i] = self.dem_data.values[lat_idx, lon_idx]

        # 处理站点高程中可能的NaN值
        nan_mask = np.isnan(self.station_dem)
        if nan_mask.any():
            print(f"⚠️ 站点高程数据中存在 {nan_mask.sum()} 个NaN值，将进行填充")
            self.station_dem[nan_mask] = np.nanmean(self.station_dem[~nan_mask])

        print(f"站点高程范围: {np.nanmin(self.station_dem):.1f}m - {np.nanmax(self.station_dem):.1f}m")

    def preprocess_dem(self):
        """
        按需加载和预处理DEM数据，确保与ERA5格点一致

        参数:
        thread_id: 线程/进程ID，用于在并行环境中区分输出

        返回:
        处理后的DEM数据
        """

        print(f"📊 加载DEM数据...")
        try:
            dem_ds = xr.open_dataset(self.dem_file)
        except Exception as e:
            print(f"❌ 加载DEM数据失败: {e}")
            return None

        print(f"📊 对DEM数据进行重采样到ERA5网格...")
        try:
            # 将DEM数据插值到ERA5格点，并转换为float32
            dem_regrid = (
                dem_ds["Band1"].interp(lat=self.era5_ds.lat, lon=self.era5_ds.lon, method="linear").astype(np.float32)
            )

            # 检查是否存在NaN值
            nan_count = np.isnan(dem_regrid.values).sum()
            if nan_count > 0:
                print(f"⚠️ DEM数据中存在 {nan_count} 个NaN值，将进行填充")
                # 使用插值填充NaN值

                dem_array = dem_regrid.values
                mask = np.isnan(dem_array)
                dem_array[mask] = ndimage.gaussian_filter(np.where(~mask, dem_array, 0), sigma=3)[mask]
                dem_regrid.values = dem_array

            # 检查是否存在Inf值
            inf_count = np.isinf(dem_regrid.values).sum()
            if inf_count > 0:
                print(f"⚠️ DEM数据中存在 {inf_count} 个Inf值，将替换为有效值")
                # 替换Inf值
                dem_regrid.values[np.isinf(dem_regrid.values)] = np.nanmean(
                    dem_regrid.values[~np.isinf(dem_regrid.values)]
                )

            print(
                f"✅ DEM数据处理完成，高程范围: {np.nanmin(dem_regrid.values):.1f}m - {np.nanmax(dem_regrid.values):.1f}m"
            )

            dem_ds.close()

        except Exception as e:
            print(f"❌ 重采样DEM数据失败: {e}")
            return None

        return dem_regrid

    def idw_interpolation(
        self,
        points,
        values,
        xi,
        yi,
        p=2,
        epsilon=1e-10,
        min_points=3,
        search_radius=None,
        max_points=None,
        batch_size=5000,
    ):
        # 确保输入为numpy数组
        points = np.asarray(points, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        # 过滤无效值
        mask = ~np.isnan(values)
        if not np.all(mask):
            points = points[mask]
            values = values[mask]

        # Numba参数预处理 - 处理None值
        numba_search_radius = 0.0 if search_radius is None else float(search_radius)
        numba_max_points = 0 if max_points is None else int(max_points)

        # 检查有效站点数量
        if len(points) < min_points:
            # 站点数不足，返回均值填充结果
            print(f"警告: 有效站点数量({len(points)})小于最小要求({min_points})，使用均值填充")
            return np.ones_like(xi) * np.mean(values)

        # 创建网格点
        grid_points = np.column_stack([yi.flatten(), xi.flatten()]).astype(np.float32)
        zi = np.zeros(grid_points.shape[0], dtype=np.float32)

        # 分批处理网格点
        for i in range(0, grid_points.shape[0], batch_size):
            end_idx = min(i + batch_size, grid_points.shape[0])
            batch_points = grid_points[i:end_idx]

            # 调用Numba加速的IDW内核
            result_batch = idw_numba_kernel(
                batch_points, points, values, p, epsilon, numba_search_radius, min_points, numba_max_points
            )

            # 保存结果
            zi[i:end_idx] = result_batch

        return zi.reshape(xi.shape)

    def mprism_interpolation_aligned_grid(
        self, points, values, xi, yi, n_regions=5, min_points_per_region=3, batch_size=100, verbose=False
    ):
        """
        MPRISM残差插值 - 使用Numba加速的优化版本
        针对dem_data与目标网格完全一致的情况，使用Numba加速计算

        参数:
        points: 站点坐标数组 [lat, lon]
        values: 站点残差值数组
        xi, yi: 目标网格坐标
        n_regions: 高程分区数量
        min_points_per_region: 每个分区最少站点数
        batch_size: 每批处理的行数，控制内存使用
        verbose: 是否输出详细信息
        """
        # 基础数据准备
        points = np.asarray(points, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        mask = ~np.isnan(values)

        # 有效点检查
        if mask.sum() < min_points_per_region * 2:
            if verbose:
                print("有效点太少，切换到IDW插值...")
            return self.idw_interpolation(points[mask], values[mask], xi, yi)

        valid_points = points[mask]
        valid_values = values[mask]
        station_dem = self.station_dem[mask]

        # 分区和模型训练部分
        dem_min, dem_max = station_dem.min(), station_dem.max()
        dem_range = dem_max - dem_min
        if dem_range < 1:
            if verbose:
                print("高程差过小，切换到IDW插值")
            return self.idw_interpolation(valid_points, valid_values, xi, yi)

        region_boundaries = np.linspace(dem_min, dem_max, n_regions + 1)

        # 为Numba函数准备数据结构
        region_coefs = np.zeros(n_regions, dtype=np.float32)
        region_intercepts = np.zeros(n_regions, dtype=np.float32)
        has_model = np.zeros(n_regions, dtype=np.bool_)

        # 为每个分区建模 - 提取模型系数为Numba准备
        for r in range(n_regions):
            if r == n_regions - 1:
                region_mask = (station_dem >= region_boundaries[r]) & (station_dem <= region_boundaries[r + 1])
            else:
                region_mask = (station_dem >= region_boundaries[r]) & (station_dem < region_boundaries[r + 1])

            if np.sum(region_mask) < min_points_per_region:
                continue

            elev = station_dem[region_mask]
            vals = valid_values[region_mask]
            if len(elev) <= 1:
                continue

            model = LinearRegression()
            model.fit(elev.reshape(-1, 1), vals)

            # 存储系数为Numba可用格式
            region_coefs[r] = model.coef_[0]
            region_intercepts[r] = model.intercept_
            has_model[r] = True

        # 初始化结果数组
        result = np.zeros_like(xi, dtype=np.float32)

        # 直接获取DEM数据
        grid_elevs = self.dem_data.values

        # 预计算每个分区模型对所有站点的残差
        n_stations = len(valid_values)
        station_residuals = np.zeros((n_regions, n_stations), dtype=np.float32)

        for r in range(n_regions):
            if has_model[r]:
                preds = region_coefs[r] * station_dem + region_intercepts[r]
                station_residuals[r] = valid_values - preds

        # 提取站点坐标
        valid_lats = valid_points[:, 0]  # 第一列是纬度
        valid_lons = valid_points[:, 1]  # 第二列是经度

        # 按行分批处理网格点
        height, width = xi.shape
        for i in range(0, height, batch_size):
            i_end = min(i + batch_size, height)

            # 批量获取当前行块的高程和区域
            block_elevs = grid_elevs[i:i_end, :]
            block_regions = np.searchsorted(region_boundaries[1:], block_elevs)
            block_regions = np.minimum(block_regions, n_regions - 1)

            # 获取当前批次的网格点坐标
            block_lats = yi[i:i_end, :]
            block_lons = xi[i:i_end, :]

            # 预分配当前批次的结果数组
            block_result = np.zeros_like(block_elevs, dtype=np.float32)

            # 逐列处理，使用Numba加速函数
            for j in range(width):
                # 提取当前列
                col_lats = block_lats[:, j]
                col_lons = block_lons[:, j]
                col_elevs = block_elevs[:, j]
                col_regions = block_regions[:, j]

                # 使用Numba加速的函数处理整列
                col_result = process_grid_column(
                    col_lats,
                    col_lons,
                    col_elevs,
                    col_regions,
                    valid_lats,
                    valid_lons,
                    valid_values,
                    region_coefs,
                    region_intercepts,
                    station_residuals,
                    n_regions,
                    has_model,
                )

                block_result[:, j] = col_result

            # 将当前批次结果写入最终结果数组
            result[i:i_end, :] = block_result

        # 确保非负值
        result[result < 0] = 0
        return result

    def process_one_time_compute(self, t_idx, method="mprism", use_dem=True, idw_params=None):
        """单个时间点的插值操作，只计算不写入，返回计算结果"""
        # try:
        # 默认IDW参数
        if idw_params is None:
            idw_params = {
                "p": 2,  # 幂指数
                "epsilon": 1e-10,  # 防除零
                "min_points": 3,  # 最少站点数
                "search_radius": None,  # 搜索半径
                "batch_size": 5000,  # 批处理大小
            }

        current_time = self.station_ds.time.values[t_idx]
        era5_precip = self.era5_ds["tp"].sel(time=current_time).values.astype(np.float32)
        station_precip = self.station_ds["rain1h_qc"].sel(time=current_time).values

        # 筛选降水值大于0的站点
        rain_mask = station_precip > 0.1

        # 检查是否有足够多的降水站点
        if np.sum(rain_mask) < idw_params["min_points"]:
            print(f"警告: 时间点 {t_idx} 的降水站点数量不足 ({np.sum(rain_mask)}/{len(station_precip)})")
            # 如果降水站点太少，可选择以下两种处理方式：

            # 选项1: 仍使用所有站点，包括零降水站点
            # rain_mask = np.ones_like(rain_mask, dtype=bool)

            # 选项2: 直接使用ERA5数据，不进行站点订正
            return {
                "t_idx": t_idx,
                "time_str": pd.Timestamp(current_time).strftime("%Y-%m-%d %H:%M"),
                "precip": era5_precip,
                "status": "success",
                "note": "使用原始ERA5数据（降水站点不足）",
            }

        # 获取有降水的站点的位置和ERA5值
        rainy_station_lats = self.station_lats[rain_mask]
        rainy_station_lons = self.station_lons[rain_mask]
        rainy_station_precip = station_precip[rain_mask]
        rainy_station_points = np.column_stack([rainy_station_lats, rainy_station_lons])

        # 获取对应的ERA5值
        rainy_era5_at_stations = era5_precip[
            self.station_era5_lat_indices[rain_mask], self.station_era5_lon_indices[rain_mask]
        ]

        # 计算残差
        residuals = rainy_station_precip - rainy_era5_at_stations

        # 创建网格点
        grid_lon, grid_lat = np.meshgrid(self.era5_ds.lon.values, self.era5_ds.lat.values)

        # 如果使用DEM，还需要筛选对应的站点高程数据
        if method == "mprism" and use_dem:
            # 筛选降水站点对应的DEM数据
            rainy_station_dem = self.station_dem[rain_mask]

            # 保存原始站点DEM
            original_station_dem = self.station_dem

            # 临时替换为只包含降水站点的DEM值
            self.station_dem = rainy_station_dem

            # 执行插值
            residual_grid = self.mprism_interpolation_aligned_grid(rainy_station_points, residuals, grid_lon, grid_lat)

            # 恢复原始站点DEM
            self.station_dem = original_station_dem
        else:
            # 使用IDW插值，只基于有降水的站点
            residual_grid = self.idw_interpolation(
                rainy_station_points,
                residuals,
                grid_lon,
                grid_lat,
                p=idw_params["p"],
                epsilon=idw_params["epsilon"],
                min_points=idw_params["min_points"],
                search_radius=idw_params["search_radius"],
                max_points=idw_params["max_points"],
                batch_size=idw_params["batch_size"],
            )

        final_precip = era5_precip + residual_grid
        final_precip[final_precip < 0] = 0

        # 返回计算结果
        time_str = pd.Timestamp(current_time).strftime("%Y-%m-%d %H:%M")
        return {
            "t_idx": t_idx,
            "time_str": time_str,
            "precip": final_precip,
            "status": "success",
            "used_stations": np.sum(rain_mask),
        }
        # except Exception as e:
        #     return {"t_idx": t_idx, "status": "failed", "error": str(e)}

    def process_all_times_parallel(self, method="mprism", use_dem=True, resume=True, zarr_name="temp_output.zarr"):
        """
        使用 Joblib 并行处理所有时间点，支持断点续传

        参数:
        method: 插值方法，'mprism'或'idw'
        use_dem: 是否使用DEM数据
        resume: 是否断点续传
        zarr_name: 输出zarr文件名
        idw_params: IDW插值参数字典
        """
        # 自动计算适合的搜索半径 (基于站点空间分布)
        station_points = np.column_stack([self.station_lats, self.station_lons])
        radius = self.calculate_adaptive_radius(station_points, min_points=5)
        print(f"计算的自适应搜索半径: {radius:.4f}")

        idw_params = {
            "p": 2,
            "epsilon": 1e-10,
            "min_points": 3,
            "search_radius": radius,
            "max_points": None,
            "batch_size": 5000,
        }

        # 创建 zarr 存储路径
        zarr_path = os.path.join(self.output_folder, zarr_name)
        # 创建跟踪文件路径
        tracking_file = os.path.join(self.output_folder, f"{zarr_name.split('.')[0]}_processed_times.json")

        if not os.path.exists(zarr_path):
            # 构建初始空 DataArray
            ds_template = xr.Dataset(
                data_vars={
                    "corrected_precip": (
                        ("time", "lat", "lon"),
                        np.zeros(
                            (
                                len(self.station_ds.time),
                                len(self.era5_ds.lat),
                                len(self.era5_ds.lon),
                            ),
                            dtype=np.float32,
                        ),
                    )
                },
                coords={
                    "time": self.station_ds.time.values,
                    "lat": self.era5_ds.lat.values,
                    "lon": self.era5_ds.lon.values,
                },
            )
            ds_template.to_zarr(zarr_path, mode="w")

            # 创建新的跟踪文件
            with open(tracking_file, "w") as f:
                json.dump({"processed_indices": []}, f)

        # 断点续传：从跟踪文件读取已处理的时间点
        processed_indices = []
        if resume and os.path.exists(tracking_file):
            try:
                with open(tracking_file, "r") as f:
                    tracking_data = json.load(f)
                    processed_indices = tracking_data.get("processed_indices", [])
                print(f"从跟踪文件加载已处理时间点: {len(processed_indices)}")
            except Exception as e:
                print(f"读取跟踪文件失败: {e}，将重新创建")
                processed_indices = []

        all_indices = set(range(len(self.station_ds.time)))
        remain_indices = list(all_indices - set(processed_indices))
        total_tasks = len(remain_indices)
        print(f"已处理时间点数量: {len(processed_indices)}, 剩余: {total_tasks}")

        if len(remain_indices) == 0:
            print("所有时间点均已处理，无需重复运行。")
            return xr.open_zarr(zarr_path)

        # 使用并行处理，但进度条留在主进程
        print("开始并行处理...")
        with parallel_backend("loky", n_jobs=4):  # 减少并行度，避免过多进程争用资源
            # 初始化进度条但不传递给子进程
            with tqdm(total=total_tasks, desc="处理进度", unit="时间点") as progress:
                # 将任务分批处理，避免一次性创建过多进程
                batch_size = min(24, total_tasks)  # 减小批次大小，减轻内存压力
                for i in range(0, total_tasks, batch_size):
                    end_idx = min(i + batch_size, total_tasks)
                    batch_indices = remain_indices[i:end_idx]

                    batch_results = Parallel()(
                        delayed(self.process_one_time_compute)(t_idx, method, use_dem, idw_params)
                        for t_idx in batch_indices
                    )

                    # 获取成功计算的结果
                    successful_results = [r for r in batch_results if r["status"] == "success"]
                    failed_results = [r for r in batch_results if r["status"] == "failed"]

                    # 在主进程中集中写入结果到Zarr - 避免并发写入
                    if successful_results:
                        # 读取已有的 Zarr 数据集
                        ds_zarr = xr.open_zarr(zarr_path)

                        # 更新数据
                        for result in successful_results:
                            t_idx = result["t_idx"]
                            # 更新内存中的数据数组
                            ds_zarr["corrected_precip"][t_idx] = result["precip"]

                        # 重新写入 Zarr，使用 mode='a' 追加模式
                        # 使用 region 参数指定只写入已经修改的区域
                        # 这里我们使用了 consolidated=True 确保元数据被正确处理
                        for result in successful_results:
                            t_idx = result["t_idx"]
                            # 提取单个时间点的切片
                            ds_slice = ds_zarr.isel(time=[t_idx])
                            # 写入这个时间点，使用 append_dim=None 表示覆盖而非追加
                            ds_slice.to_zarr(
                                zarr_path,
                                region={"time": slice(t_idx, t_idx + 1), "lat": slice(None), "lon": slice(None)},
                                consolidated=True,
                            )

                        # 确保资源释放
                        ds_zarr = None

                        # 批量更新跟踪文件
                        processed_batch_indices = [r["t_idx"] for r in successful_results]
                        lock_file = tracking_file + ".lock"
                        with FileLock(lock_file):
                            with open(tracking_file, "r") as f:
                                tracking_data = json.load(f)

                            # 更新已处理索引
                            tracking_data["processed_indices"].extend(processed_batch_indices)

                            # 记录处理时间（可选）
                            if "processing_times" not in tracking_data:
                                tracking_data["processing_times"] = {}

                            for result in successful_results:
                                tracking_data["processing_times"][str(result["t_idx"])] = result.get("process_time", 0)

                            with open(tracking_file, "w") as f:
                                json.dump(tracking_data, f)

                    # 更新进度条
                    for result in batch_results:
                        progress.update(1)
                        if result["status"] == "success":
                            progress.set_postfix({"当前处理": result["time_str"]})
                        else:
                            progress.write(f"时间点 {result['t_idx']} 处理失败: {result.get('error', '未知错误')}")

                    # 输出批次处理统计
                    if failed_results:
                        print(
                            f"批次 {i//batch_size + 1}: 成功 {len(successful_results)}/{len(batch_results)}，"
                            f"失败 {len(failed_results)}"
                        )

                    # 执行垃圾回收，减轻内存压力
                    import gc

                    gc.collect()

        # 统计处理结果
        processed_indices = []
        with open(tracking_file, "r") as f:
            tracking_data = json.load(f)
            processed_indices = tracking_data.get("processed_indices", [])

        print(f"处理完成：成功 {len(processed_indices)}/{len(all_indices)} 个时间点")
        print("并行插值处理完成，Zarr 数据写入完成。")
        print(f"Zarr 文件路径: {zarr_path}")
        print(f"处理进度跟踪文件: {tracking_file}")

        return xr.open_zarr(zarr_path)

    def validate_results(self, corrected_ds, method="idw", use_dem=True, min_points=3, min_precip=0.1):
        """
        验证插值结果，采用留一法交叉验证 - 使用预计算的ERA5网格点索引，支持多种插值方法

        参数:
        corrected_ds: 已订正的数据集（实际未使用，仅为接口一致性）
        method: 插值方法，支持'idw'和'mprism'
        use_dem: 是否使用DEM数据
        min_points: 最少需要的站点数
        min_precip: 最小有效降水值

        返回:
        验证结果DataFrame
        """
        print(f"开始验证结果 (方法: {method}, 使用DEM: {use_dem})...")

        # 选择多个时间点进行验证，增加验证的稳定性
        total_times = len(self.station_ds.time)
        time_indices = [total_times // 4, total_times // 2, total_times * 3 // 4]  # 选择三个时间点

        # 初始化所有验证结果
        all_validation_results = []

        for time_idx in time_indices:
            current_time = self.station_ds.time.values[time_idx]
            time_str = pd.Timestamp(current_time).strftime("%Y-%m-%d %H:%M")
            print(f"验证时间点: {time_str}")

            # 提取站点位置和降水
            station_lats = self.station_lats
            station_lons = self.station_lons
            station_precip = self.station_ds["rain1h_qc"].sel(time=current_time).values

            # 获取当前时间的ERA5数据
            era5_time = self.era5_ds["tp"].sel(time=current_time).values

            # 筛选有效降水站点（与process_one_time_compute保持一致）
            rain_mask = station_precip > min_precip
            if np.sum(rain_mask) < min_points + 1:  # +1是因为留一法会减少一个站点
                print(f"警告: 时间点 {time_str} 的有效降水站点数量不足 ({np.sum(rain_mask)}/{len(station_precip)})")
                continue  # 跳过此时间点

            # 初始化此时间点的验证结果
            validation_results = []

            # 创建网格点（用于MPRISM方法）
            grid_lon, grid_lat = np.meshgrid(self.era5_ds.lon.values, self.era5_ds.lat.values)

            # 对每个有效降水站点进行留一法交叉验证
            for i in range(len(station_lats)):
                if not rain_mask[i]:
                    continue  # 跳过无降水站点

                # 排除当前站点
                leave_one_mask = rain_mask.copy()
                leave_one_mask[i] = False

                # 检查剩余站点数量
                if np.sum(leave_one_mask) < min_points:
                    continue  # 站点数不足，跳过该站点

                # 提取留一后的站点数据
                leave_one_lats = station_lats[leave_one_mask]
                leave_one_lons = station_lons[leave_one_mask]
                leave_one_precip = station_precip[leave_one_mask]
                leave_one_points = np.column_stack([leave_one_lats, leave_one_lons])

                # 使用预计算的索引获取ERA5值
                era5_at_stations = era5_time[
                    self.station_era5_lat_indices[leave_one_mask], self.station_era5_lon_indices[leave_one_mask]
                ]

                # 获取测试站点信息
                test_lat = station_lats[i]
                test_lon = station_lons[i]
                test_precip = station_precip[i]
                test_point = np.array([[test_lat, test_lon]], dtype=np.float32)

                # 获取测试站点的ERA5值
                test_era5 = era5_time[self.station_era5_lat_indices[i], self.station_era5_lon_indices[i]]

                # 计算残差
                residuals = leave_one_precip - era5_at_stations

                # 对测试站点进行残差插值
                if method == "mprism" and use_dem:
                    try:
                        # 保存原始站点DEM数据
                        original_dem = self.station_dem

                        # 筛选留一后的站点高程
                        leave_one_dem = self.station_dem[leave_one_mask]
                        # 临时替换站点DEM为留一后的值
                        self.station_dem = leave_one_dem

                        # 使用MPRISM方法插值（需要修改为单点插值）
                        # 创建包含测试点的小网格
                        test_grid_lat = np.array([[test_lat]])
                        test_grid_lon = np.array([[test_lon]])

                        # 执行MPRISM插值
                        interpolated_residual = self.mprism_interpolation_aligned_grid(
                            leave_one_points, residuals, test_grid_lon, test_grid_lat, verbose=False
                        ).flatten()[0]

                        # 恢复原始站点DEM
                        self.station_dem = original_dem
                    except Exception as e:
                        print(f"站点 {i} MPRISM插值失败，回退到IDW: {e}")
                        # 使用IDW插值作为备选
                        distances = np.sqrt((leave_one_lats - test_lat) ** 2 + (leave_one_lons - test_lon) ** 2)
                        if np.all(distances > 1e-10):  # 避免除零错误
                            weights = 1.0 / (distances**2 + 1e-10)
                            weights_sum = np.sum(weights)
                            interpolated_residual = np.sum(weights * residuals) / weights_sum
                        else:
                            # 如果有完全重合的点，直接使用该点的值
                            idx = np.argmin(distances)
                            interpolated_residual = residuals[idx]
                else:
                    # 使用IDW插值
                    distances = np.sqrt((leave_one_lats - test_lat) ** 2 + (leave_one_lons - test_lon) ** 2)
                    if np.all(distances > 1e-10):  # 避免除零错误
                        weights = 1.0 / (distances**2 + 1e-10)
                        weights_sum = np.sum(weights)
                        interpolated_residual = np.sum(weights * residuals) / weights_sum
                    else:
                        # 如果有完全重合的点，直接使用该点的值
                        idx = np.argmin(distances)
                        interpolated_residual = residuals[idx]

                # 计算插值预测降水
                predicted_precip = test_era5 + interpolated_residual
                if predicted_precip < 0:
                    predicted_precip = 0

                # 存储结果
                validation_results.append(
                    {
                        "time": time_str,
                        "station_id": self.station_ds.station.values[i],
                        "lat": test_lat,
                        "lon": test_lon,
                        "elev": self.station_dem[i],
                        "obs_precip": test_precip,
                        "era5_precip": test_era5,
                        "corrected_precip": predicted_precip,
                        "abs_error": abs(predicted_precip - test_precip),
                        "rel_error": abs(predicted_precip - test_precip) / (test_precip + 1e-5),
                        "era5_error": abs(test_era5 - test_precip),
                    }
                )

            # 将此时间点的结果添加到总结果中
            all_validation_results.extend(validation_results)

        # 如果没有有效结果
        if not all_validation_results:
            print("警告: 未能进行有效验证，可能是由于可用站点太少")
            return None

        # 转换为DataFrame
        validation_df = pd.DataFrame(all_validation_results)

        # 计算详细验证指标
        # 1. RMSE
        rmse_era5 = np.sqrt(mean_squared_error(validation_df["obs_precip"], validation_df["era5_precip"]))
        rmse_corrected = np.sqrt(mean_squared_error(validation_df["obs_precip"], validation_df["corrected_precip"]))

        # 2. MAE (平均绝对误差)
        mae_era5 = np.mean(np.abs(validation_df["obs_precip"] - validation_df["era5_precip"]))
        mae_corrected = np.mean(np.abs(validation_df["obs_precip"] - validation_df["corrected_precip"]))

        # 3. 相关系数
        corr_era5 = np.corrcoef(validation_df["obs_precip"], validation_df["era5_precip"])[0, 1]
        corr_corrected = np.corrcoef(validation_df["obs_precip"], validation_df["corrected_precip"])[0, 1]

        # 4. 降水量分组指标
        # 小雨 (0.1-10mm)、中雨 (10-25mm)、大雨 (>25mm)
        light_mask = (validation_df["obs_precip"] > 0.1) & (validation_df["obs_precip"] <= 10)
        medium_mask = (validation_df["obs_precip"] > 10) & (validation_df["obs_precip"] <= 25)
        heavy_mask = validation_df["obs_precip"] > 25

        # 打印分组统计
        print("\n===== 验证结果总览 =====")
        print(f"总样本数: {len(validation_df)}")
        print(f"小雨样本数: {np.sum(light_mask)}")
        print(f"中雨样本数: {np.sum(medium_mask)}")
        print(f"大雨样本数: {np.sum(heavy_mask)}")

        # 整体统计
        print("\n===== 整体统计 =====")
        print(
            f"RMSE: ERA5 = {rmse_era5:.4f}, 订正后 = {rmse_corrected:.4f}, 改进率 = {(rmse_era5-rmse_corrected)/rmse_era5*100:.2f}%"
        )
        print(
            f"MAE:  ERA5 = {mae_era5:.4f}, 订正后 = {mae_corrected:.4f}, 改进率 = {(mae_era5-mae_corrected)/mae_era5*100:.2f}%"
        )
        print(f"相关系数: ERA5 = {corr_era5:.4f}, 订正后 = {corr_corrected:.4f}")

        # 计算分组RMSE
        if np.sum(light_mask) > 0:
            rmse_light_era5 = np.sqrt(
                mean_squared_error(
                    validation_df.loc[light_mask, "obs_precip"], validation_df.loc[light_mask, "era5_precip"]
                )
            )
            rmse_light_corrected = np.sqrt(
                mean_squared_error(
                    validation_df.loc[light_mask, "obs_precip"], validation_df.loc[light_mask, "corrected_precip"]
                )
            )
            print(f"\n小雨 (0.1-10mm):")
            print(
                f"  RMSE: ERA5 = {rmse_light_era5:.4f}, 订正后 = {rmse_light_corrected:.4f}, 改进率 = {(rmse_light_era5-rmse_light_corrected)/rmse_light_era5*100:.2f}%"
            )

        if np.sum(medium_mask) > 0:
            rmse_medium_era5 = np.sqrt(
                mean_squared_error(
                    validation_df.loc[medium_mask, "obs_precip"], validation_df.loc[medium_mask, "era5_precip"]
                )
            )
            rmse_medium_corrected = np.sqrt(
                mean_squared_error(
                    validation_df.loc[medium_mask, "obs_precip"], validation_df.loc[medium_mask, "corrected_precip"]
                )
            )
            print(f"\n中雨 (10-25mm):")
            print(
                f"  RMSE: ERA5 = {rmse_medium_era5:.4f}, 订正后 = {rmse_medium_corrected:.4f}, 改进率 = {(rmse_medium_era5-rmse_medium_corrected)/rmse_medium_era5*100:.2f}%"
            )

        if np.sum(heavy_mask) > 0:
            rmse_heavy_era5 = np.sqrt(
                mean_squared_error(
                    validation_df.loc[heavy_mask, "obs_precip"], validation_df.loc[heavy_mask, "era5_precip"]
                )
            )
            rmse_heavy_corrected = np.sqrt(
                mean_squared_error(
                    validation_df.loc[heavy_mask, "obs_precip"], validation_df.loc[heavy_mask, "corrected_precip"]
                )
            )
            print(f"\n大雨 (>25mm):")
            print(
                f"  RMSE: ERA5 = {rmse_heavy_era5:.4f}, 订正后 = {rmse_heavy_corrected:.4f}, 改进率 = {(rmse_heavy_era5-rmse_heavy_corrected)/rmse_heavy_era5*100:.2f}%"
            )

        # 保存验证结果
        validation_file = os.path.join(
            self.output_folder, f'validation_{method}_{"with_dem" if use_dem else "no_dem"}.csv'
        )
        validation_df.to_csv(validation_file, index=False)
        print(f"\n验证结果已保存至: {validation_file}")

        return validation_df

    def calculate_adaptive_radius(self, points, min_points=5):
        """
        计算自适应搜索半径，确保每个网格点至少有min_points个站点

        参数:
        points: 站点坐标数组
        min_points: 最少站点数量

        返回:
        推荐的搜索半径
        """
        from scipy.spatial import cKDTree

        # 构建KD树
        tree = cKDTree(points)

        # 计算每个站点到第min_points个最近邻站点的距离
        all_dists = []
        for point in points:
            # 查询min_points+1个点(包括自身)
            dists, _ = tree.query(point, k=min(min_points + 1, len(points)))
            # 取最远的那个距离
            if len(dists) > 1:
                all_dists.append(dists[-1])

        if not all_dists:
            # 如果站点太少，使用默认值
            return 1.0

        # 使用距离的95%分位数作为推荐半径
        radius = np.percentile(all_dists, 95)

        # 增加一些余量
        return radius * 1.2


def main():
    """主函数"""
    # 基本配置
    station_file = "/mnt/h/DataSet/station_precipitation_data_filled.nc"
    era5_folder = "/mnt/h/DataSet/Pre_DEM"
    dem_file = "/mnt/h/DataSet/3-DEM/DEM_clip.nc"
    output_folder = "/mnt/h/DataSet/PreGrids_MPRISM"

    # 创建插值器实例
    interpolator = PrecipitationInterpolation(
        station_file=station_file, era5_folder=era5_folder, dem_file=dem_file, output_folder=output_folder
    )

    # 执行插值计算
    corrected_ds = interpolator.process_all_times_parallel(method="mprism", use_dem=True, resume=True)

    # 验证结果
    # interpolator.validate_results(corrected_ds, method="mprism", use_dem=True)
    print("处理完成！")


if __name__ == "__main__":
    # 对于多进程程序，保护入口点很重要
    main()
