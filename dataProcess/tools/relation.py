import glob
import logging
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PrecipitationCorrelationAnalyzer:
    def __init__(self, grids_folder, era5_folder, output_folder, start_date=None, end_date=None):
        """
        初始化降水相关性分析器

        参数:
        grids_folder: 包含日降水数据的Grids文件夹路径
        era5_folder: 包含ERA5月降水数据的Pre_DEM文件夹路径
        output_folder: 分析结果输出路径
        start_date: 分析开始日期，格式为'YYYY-MM-DD'
        end_date: 分析结束日期，格式为'YYYY-MM-DD'
        """
        self.grids_folder = grids_folder
        self.era5_folder = era5_folder
        self.output_folder = output_folder
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)

        # 初始化结果存储
        self.results = {"daily": [], "monthly": [], "overall": None}

    def find_grids_files(self):
        """查找Grids文件夹中的所有文件"""
        all_files = glob.glob(os.path.join(self.grids_folder, "*.nc"))

        # 如果指定了日期范围，则过滤文件
        if self.start_date and self.end_date:
            filtered_files = []
            for file_path in all_files:
                try:
                    # 从文件名中提取日期（假设文件名包含日期信息）
                    # 这里需要根据实际文件命名规则调整
                    file_name = os.path.basename(file_path)
                    # 示例：如果文件名为 Grids_20220401.nc
                    if "_" in file_name:
                        date_str = file_name.split("_")[1].split(".")[0]
                        file_date = datetime.strptime(date_str, "%Y%m%d")
                        if self.start_date <= file_date <= self.end_date:
                            filtered_files.append(file_path)
                except:
                    # 如果无法从文件名提取日期，则包含该文件
                    filtered_files.append(file_path)

            logger.info(f"找到 {len(filtered_files)}/{len(all_files)} 个符合日期范围的Grids文件")
            return filtered_files

        logger.info(f"找到 {len(all_files)} 个Grids文件")
        return all_files

    def find_era5_files(self):
        """查找ERA5文件夹中的所有月度文件"""
        # 如果指定了日期范围，则只查找该范围内的月份
        if self.start_date and self.end_date:
            current_date = self.start_date.replace(day=1)  # 将日期设为当月第一天
            end_month = self.end_date.replace(day=1)  # 将结束日期设为当月第一天

            era5_files = []
            while current_date <= end_month:
                file_name = f"{current_date.year}-{current_date.month:02d}.nc"
                file_path = os.path.join(self.era5_folder, file_name)
                if os.path.exists(file_path):
                    era5_files.append(file_path)
                else:
                    logger.warning(f"未找到ERA5文件: {file_name}")

                # 移至下个月
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)

            logger.info(f"找到 {len(era5_files)} 个符合日期范围的ERA5月度文件")
            return era5_files

        # 如果未指定日期范围，则获取所有文件
        era5_files = glob.glob(os.path.join(self.era5_folder, "*.nc"))
        logger.info(f"找到 {len(era5_files)} 个ERA5月度文件")
        return era5_files

    def calculate_daily_correlation(self, grids_files, era5_files):
        """计算每日的PRE和tp之间的相关系数"""
        daily_results = []

        # 首先加载所有ERA5数据，避免重复读取
        era5_datasets = {}
        for file_path in tqdm(era5_files, desc="加载ERA5数据"):
            try:
                file_name = os.path.basename(file_path)
                year_month = file_name.split(".")[0]  # 例如：2022-04
                era5_datasets[year_month] = xr.open_dataset(file_path)
                # 确保tp变量存在
                if "tp" not in era5_datasets[year_month].data_vars:
                    logger.warning(f"ERA5文件 {file_name} 中未找到tp变量")
                    del era5_datasets[year_month]
            except Exception as e:
                logger.error(f"加载ERA5文件 {file_path} 时出错: {str(e)}")

        # 处理每个Grids文件
        for grid_file in tqdm(grids_files, desc="计算每日相关系数"):
            try:
                # 加载Grids数据
                grid_ds = xr.open_dataset(grid_file)

                # 确保PRE变量存在
                if "PRE" not in grid_ds.data_vars:
                    logger.warning(f"Grids文件 {os.path.basename(grid_file)} 中未找到PRE变量")
                    grid_ds.close()
                    continue

                # 获取文件日期
                grid_times = grid_ds.time.values
                if len(grid_times) == 0:
                    logger.warning(f"Grids文件 {os.path.basename(grid_file)} 没有时间数据")
                    grid_ds.close()
                    continue

                # 假设这是一个日文件，我们使用第一个时间点的日期
                grid_date = pd.to_datetime(grid_times[0])
                year_month = f"{grid_date.year}-{grid_date.month:02d}"

                # 检查对应的ERA5数据是否存在
                if year_month not in era5_datasets:
                    logger.warning(f"未找到对应的ERA5数据: {year_month}")
                    grid_ds.close()
                    continue

                # 获取对应的ERA5数据
                era5_ds = era5_datasets[year_month]

                # 选择当天的ERA5数据
                day_str = grid_date.strftime("%Y-%m-%d")
                try:
                    era5_day = era5_ds.sel(valid_time=day_str, method="nearest")
                except:
                    # 尝试直接使用日期索引
                    era5_times = pd.to_datetime(era5_ds.time.values)
                    closest_time_idx = np.abs(era5_times - grid_date).argmin()
                    era5_day = era5_ds.isel(valid_time=closest_time_idx)

                # 计算两个数据集的空间范围交集
                # 假设两个数据集都有lat和lon维度
                grid_lats = grid_ds.lat.values
                grid_lons = grid_ds.lon.values
                era5_lats = era5_day.latitude.values
                era5_lons = era5_day.longitude.values

                # 计算交集范围
                min_lat = max(grid_lats.min(), era5_lats.min())
                max_lat = min(grid_lats.max(), era5_lats.max())
                min_lon = max(grid_lons.min(), era5_lons.min())
                max_lon = min(grid_lons.max(), era5_lons.max())

                # 选择交集区域的数据
                grid_subset = grid_ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
                era5_subset = era5_day.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))

                # 如果空间分辨率不同，需要对其中一个进行插值
                if (
                    grid_subset.lat.size != era5_subset.latitude.size
                    or grid_subset.lon.size != era5_subset.longitude.size
                ):
                    # 插值到Grids数据的分辨率
                    era5_subset = era5_subset.interp(latitude=grid_subset.lat, longitude=grid_subset.lon)

                # 提取PRE和tp数据
                pre_data = grid_subset.PRE.values.flatten()
                tp_data = era5_subset.tp.values.flatten()

                # 移除NaN值
                valid_idx = ~(np.isnan(pre_data) | np.isnan(tp_data))
                pre_valid = pre_data[valid_idx]
                tp_valid = tp_data[valid_idx]

                # 计算相关系数
                if len(pre_valid) > 0:
                    correlation, p_value = stats.pearsonr(pre_valid, tp_valid)

                    # 计算其他统计量
                    rmse = np.sqrt(np.mean((pre_valid - tp_valid) ** 2))
                    bias = np.mean(tp_valid - pre_valid)
                    mean_pre = np.mean(pre_valid)
                    mean_tp = np.mean(tp_valid)

                    daily_result = {
                        "date": day_str,
                        "correlation": correlation,
                        "p_value": p_value,
                        "rmse": rmse,
                        "bias": bias,
                        "mean_pre": mean_pre,
                        "mean_tp": mean_tp,
                        "valid_points": len(pre_valid),
                    }
                    daily_results.append(daily_result)

                # 关闭数据集
                grid_ds.close()

            except Exception as e:
                logger.error(f"处理文件 {os.path.basename(grid_file)} 时出错: {str(e)}")

        # 关闭所有ERA5数据集
        for ds in era5_datasets.values():
            ds.close()

        return daily_results

    def calculate_monthly_correlation(self, grids_files, era5_files):
        """计算每月的PRE和tp之间的相关系数"""
        # 将Grids文件按月分组
        monthly_grids = {}
        for file_path in grids_files:
            try:
                # 从文件名或文件内容中提取日期
                ds = xr.open_dataset(file_path)
                if "time" in ds.dims:
                    date = pd.to_datetime(ds.time.values[0])
                    month_key = f"{date.year}-{date.month:02d}"

                    if month_key not in monthly_grids:
                        monthly_grids[month_key] = []

                    monthly_grids[month_key].append(file_path)
                ds.close()
            except Exception as e:
                logger.error(f"处理Grids文件 {os.path.basename(file_path)} 时出错: {str(e)}")

        monthly_results = []

        # 处理每个月
        for month_key, grid_files_month in tqdm(monthly_grids.items(), desc="计算月度相关系数"):
            try:
                # 检查是否有对应的ERA5文件
                era5_file = os.path.join(self.era5_folder, f"{month_key}.nc")
                if not os.path.exists(era5_file):
                    logger.warning(f"未找到对应的ERA5月度文件: {month_key}.nc")
                    continue

                # 加载ERA5月度数据
                era5_ds = xr.open_dataset(era5_file)

                # 合并该月的所有Grids数据
                grid_datasets = []
                for grid_file in grid_files_month:
                    try:
                        ds = xr.open_dataset(grid_file)
                        if "PRE" in ds.data_vars:
                            grid_datasets.append(ds)
                        else:
                            ds.close()
                    except:
                        pass

                if not grid_datasets:
                    logger.warning(f"月份 {month_key} 没有有效的Grids数据")
                    era5_ds.close()
                    continue

                try:
                    # 合并数据集（假设它们可以沿time维度合并）
                    grid_monthly = xr.concat(grid_datasets, dim="time")

                    # 计算月平均
                    grid_monthly_mean = grid_monthly.mean(dim="time")
                    era5_monthly_mean = era5_ds.mean(dim="time")

                    # 统一空间网格（如果需要）
                    if (
                        grid_monthly_mean.lat.size != era5_monthly_mean.lat.size
                        or grid_monthly_mean.lon.size != era5_monthly_mean.lon.size
                    ):
                        era5_monthly_mean = era5_monthly_mean.interp(
                            lat=grid_monthly_mean.lat, lon=grid_monthly_mean.lon
                        )

                    # 提取数据
                    pre_monthly = grid_monthly_mean.PRE.values.flatten()
                    tp_monthly = era5_monthly_mean.tp.values.flatten()

                    # 移除NaN
                    valid_idx = ~(np.isnan(pre_monthly) | np.isnan(tp_monthly))
                    pre_valid = pre_monthly[valid_idx]
                    tp_valid = tp_monthly[valid_idx]

                    # 计算相关系数
                    if len(pre_valid) > 0:
                        correlation, p_value = stats.pearsonr(pre_valid, tp_valid)

                        rmse = np.sqrt(np.mean((pre_valid - tp_valid) ** 2))
                        bias = np.mean(tp_valid - pre_valid)

                        monthly_result = {
                            "month": month_key,
                            "correlation": correlation,
                            "p_value": p_value,
                            "rmse": rmse,
                            "bias": bias,
                            "mean_pre": np.mean(pre_valid),
                            "mean_tp": np.mean(tp_valid),
                            "valid_points": len(pre_valid),
                        }
                        monthly_results.append(monthly_result)

                    # 关闭数据集
                    grid_monthly.close()

                except Exception as e:
                    logger.error(f"处理月份 {month_key} 时出错: {str(e)}")
                    for ds in grid_datasets:
                        ds.close()

                era5_ds.close()

            except Exception as e:
                logger.error(f"处理月份 {month_key} 时出错: {str(e)}")

        return monthly_results

    def visualize_results(self):
        """可视化相关系数结果"""
        # 创建日度相关系数时间序列图
        if self.results["daily"]:
            daily_df = pd.DataFrame(self.results["daily"])

            plt.figure(figsize=(12, 6))
            plt.plot(daily_df["date"], daily_df["correlation"], "o-", markersize=4)
            plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
            plt.title("PRE与tp的日度相关系数", fontsize=14)
            plt.xlabel("日期")
            plt.ylabel("相关系数")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, "daily_correlation.png"), dpi=300)
            plt.close()

            # 创建RMSE和偏差图
            plt.figure(figsize=(12, 10))

            plt.subplot(2, 1, 1)
            plt.plot(daily_df["date"], daily_df["rmse"], "o-", color="orange", markersize=4)
            plt.title("日度RMSE", fontsize=14)
            plt.xlabel("日期")
            plt.ylabel("RMSE")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            plt.subplot(2, 1, 2)
            plt.plot(daily_df["date"], daily_df["bias"], "o-", color="green", markersize=4)
            plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
            plt.title("日度偏差 (tp - PRE)", fontsize=14)
            plt.xlabel("日期")
            plt.ylabel("偏差")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, "daily_metrics.png"), dpi=300)
            plt.close()

        # 创建月度相关系数柱状图
        if self.results["monthly"]:
            monthly_df = pd.DataFrame(self.results["monthly"])

            plt.figure(figsize=(12, 6))
            bars = plt.bar(monthly_df["month"], monthly_df["correlation"], color="skyblue")
            plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)

            # 在柱状图上添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom")

            plt.title("PRE与tp的月度相关系数", fontsize=14)
            plt.xlabel("月份")
            plt.ylabel("相关系数")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, "monthly_correlation.png"), dpi=300)
            plt.close()

            # 创建月度均值比较图
            plt.figure(figsize=(12, 6))
            x = np.arange(len(monthly_df))
            width = 0.35

            plt.bar(x - width / 2, monthly_df["mean_pre"], width, label="PRE 均值", color="skyblue")
            plt.bar(x + width / 2, monthly_df["mean_tp"], width, label="tp 均值", color="salmon")

            plt.title("PRE与tp的月均值比较", fontsize=14)
            plt.xlabel("月份")
            plt.ylabel("均值")
            plt.xticks(x, monthly_df["month"], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, "monthly_means.png"), dpi=300)
            plt.close()

    def calculate_overall_correlation(self):
        """计算整体相关系数统计"""
        # 从日度结果中计算总体统计
        if self.results["daily"]:
            daily_df = pd.DataFrame(self.results["daily"])

            # 总体相关性指标
            overall_stats = {
                "total_days": len(daily_df),
                "mean_correlation": daily_df["correlation"].mean(),
                "median_correlation": daily_df["correlation"].median(),
                "min_correlation": daily_df["correlation"].min(),
                "max_correlation": daily_df["correlation"].max(),
                "mean_rmse": daily_df["rmse"].mean(),
                "mean_bias": daily_df["bias"].mean(),
                "positive_corr_days": (daily_df["correlation"] > 0).sum(),
                "negative_corr_days": (daily_df["correlation"] <= 0).sum(),
                "significant_days": (daily_df["p_value"] < 0.05).sum(),
            }

            self.results["overall"] = overall_stats

            # 创建结果摘要文件
            with open(os.path.join(self.output_folder, "correlation_summary.txt"), "w") as f:
                f.write("PRE与tp相关系数分析摘要\n")
                f.write("=" * 50 + "\n\n")

                f.write("总体统计:\n")
                f.write(f"总计分析天数: {overall_stats['total_days']}\n")
                f.write(f"平均相关系数: {overall_stats['mean_correlation']:.4f}\n")
                f.write(f"中位数相关系数: {overall_stats['median_correlation']:.4f}\n")
                f.write(f"最小相关系数: {overall_stats['min_correlation']:.4f}\n")
                f.write(f"最大相关系数: {overall_stats['max_correlation']:.4f}\n")
                f.write(f"平均RMSE: {overall_stats['mean_rmse']:.4f}\n")
                f.write(f"平均偏差: {overall_stats['mean_bias']:.4f}\n")
                f.write(
                    f"正相关天数: {overall_stats['positive_corr_days']} ({overall_stats['positive_corr_days']/overall_stats['total_days']*100:.1f}%)\n"
                )
                f.write(
                    f"负相关天数: {overall_stats['negative_corr_days']} ({overall_stats['negative_corr_days']/overall_stats['total_days']*100:.1f}%)\n"
                )
                f.write(
                    f"显著相关天数 (p<0.05): {overall_stats['significant_days']} ({overall_stats['significant_days']/overall_stats['total_days']*100:.1f}%)\n"
                )

                f.write("\n月度统计:\n")
                if self.results["monthly"]:
                    monthly_df = pd.DataFrame(self.results["monthly"])
                    for _, row in monthly_df.iterrows():
                        f.write(
                            f"月份: {row['month']}, 相关系数: {row['correlation']:.4f}, RMSE: {row['rmse']:.4f}, 偏差: {row['bias']:.4f}\n"
                        )
                else:
                    f.write("无月度统计数据\n")

    def save_results_to_csv(self):
        """将结果保存为CSV文件"""
        if self.results["daily"]:
            daily_df = pd.DataFrame(self.results["daily"])
            daily_df.to_csv(os.path.join(self.output_folder, "daily_correlation.csv"), index=False)
            logger.info(f"已保存日度相关系数结果到: {os.path.join(self.output_folder, 'daily_correlation.csv')}")

        if self.results["monthly"]:
            monthly_df = pd.DataFrame(self.results["monthly"])
            monthly_df.to_csv(os.path.join(self.output_folder, "monthly_correlation.csv"), index=False)
            logger.info(f"已保存月度相关系数结果到: {os.path.join(self.output_folder, 'monthly_correlation.csv')}")

    def run_analysis(self):
        """运行完整的分析流程"""
        logger.info("开始降水数据相关性分析")

        # 查找文件
        grids_files = self.find_grids_files()
        era5_files = self.find_era5_files()

        if not grids_files or not era5_files:
            logger.error("未找到足够的数据文件，分析终止")
            return False

        # 计算日度相关系数
        logger.info("计算日度相关系数...")
        self.results["daily"] = self.calculate_daily_correlation(grids_files, era5_files)
        logger.info(f"完成日度相关系数计算，共 {len(self.results['daily'])} 天")

        # 计算月度相关系数
        logger.info("计算月度相关系数...")
        self.results["monthly"] = self.calculate_monthly_correlation(grids_files, era5_files)
        logger.info(f"完成月度相关系数计算，共 {len(self.results['monthly'])} 个月")

        # 计算总体统计
        self.calculate_overall_correlation()

        # 保存结果
        self.save_results_to_csv()

        # 可视化结果
        self.visualize_results()

        logger.info(f"分析完成！结果已保存到: {self.output_folder}")
        return True


if __name__ == "__main__":
    # 配置参数
    GRIDS_FOLDER = "/mnt/h/DataSet/Grids"  # Grids文件夹路径
    ERA5_FOLDER = "/mnt/h/DataSet/Pre_DEM"  # ERA5文件夹路径
    OUTPUT_FOLDER = "/mnt/h/DataSet/Correlation"  # 输出文件夹路径

    # 设置分析时间范围（可选）
    START_DATE = "2022-04-01"  # 分析开始日期
    END_DATE = "2022-12-30"  # 分析结束日期

    # 创建分析器并运行分析
    analyzer = PrecipitationCorrelationAnalyzer(
        grids_folder=GRIDS_FOLDER,
        era5_folder=ERA5_FOLDER,
        output_folder=OUTPUT_FOLDER,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    analyzer.run_analysis()
