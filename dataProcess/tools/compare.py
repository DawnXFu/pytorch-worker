import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

rc("font", **{"sans-serif": ["DejaVu Sans"], "family": "sans-serif"})
rc("axes", unicode_minus=False)


class PrecipitationInterpolationEvaluator:
    """降水插值方法评价类"""

    def __init__(
        self, station_file, idw_result_file, mprism_result_file, dem_file=None, output_folder="evaluation_results"
    ):
        """
        初始化评价类

        参数:
        station_file: 站点观测数据文件路径
        idw_result_file: IDW插值结果文件路径
        mprism_result_file: MPRISM插值结果文件路径
        dem_file: DEM数据文件路径 (可选)
        output_folder: 评价结果输出文件夹
        """
        self.station_file = station_file
        self.idw_result_file = idw_result_file
        self.mprism_result_file = mprism_result_file
        self.dem_file = dem_file
        self.output_folder = output_folder

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载站点数据和插值结果数据"""
        print("加载数据...")

        # 1. 加载站点数据
        self.station_ds = xr.open_dataset(self.station_file)
        print(f"站点数据时间范围: {self.station_ds.time.values[0]} 到 {self.station_ds.time.values[-1]}")
        print(f"站点数量: {len(self.station_ds.station)}")

        # 2. 加载IDW插值结果
        self.idw_ds = xr.open_zarr(self.idw_result_file)
        print(f"IDW数据时间范围: {self.idw_ds.time.values[0]} 到 {self.idw_ds.time.values[-1]}")

        # 3. 加载MPRISM插值结果
        self.mprism_ds = xr.open_zarr(self.mprism_result_file)
        print(f"MPRISM数据时间范围: {self.mprism_ds.time.values[0]} 到 {self.mprism_ds.time.values[-1]}")

        # 4. 可选: 加载DEM数据
        if self.dem_file:
            self.dem_data = xr.open_dataset(self.dem_file)
            print("DEM数据已加载")
        else:
            self.dem_data = None

        # 确保时间范围一致
        common_times = np.intersect1d(np.intersect1d(self.station_ds.time, self.idw_ds.time), self.mprism_ds.time)

        self.station_ds = self.station_ds.sel(time=common_times)
        self.idw_ds = self.idw_ds.sel(time=common_times)
        self.mprism_ds = self.mprism_ds.sel(time=common_times)

        print(f"共同时间点数量: {len(common_times)}")

        # 提取站点位置信息
        self.station_lats = self.station_ds["lat"].values
        self.station_lons = self.station_ds["lon"].values

        # 提取站点高程信息(如果有DEM)
        if self.dem_data:
            self.station_dem = np.zeros(len(self.station_lats), dtype=np.float32)
            for i in range(len(self.station_lats)):
                lat = self.station_lats[i]
                lon = self.station_lons[i]
                lat_idx = np.abs(self.dem_data.lat.values - lat).argmin()
                lon_idx = np.abs(self.dem_data.lon.values - lon).argmin()
                self.station_dem[i] = self.dem_data["Band1"].values[lat_idx, lon_idx]

    def evaluate_at_stations(self, n_samples=None, min_precip=0.1):
        """
        在站点位置评估两种插值方法的表现

        参数:
        n_samples: 用于评估的时间样本数量，None表示使用所有时间点
        min_precip: 最小有效降水量阈值

        返回:
        包含评估结果的DataFrame
        """
        print("开始站点位置评估...")

        # 选择时间样本
        if n_samples is None or n_samples >= len(self.station_ds.time):
            time_indices = np.arange(len(self.station_ds.time))
        else:
            # 随机选择时间样本
            time_indices = np.random.choice(np.arange(len(self.station_ds.time)), size=n_samples, replace=False)

        results = []

        # 对每个选定的时间点进行评估
        for t_idx in tqdm(time_indices, desc="评估时间点"):
            current_time = self.station_ds.time.values[t_idx]
            time_str = pd.Timestamp(current_time).strftime("%Y-%m-%d %H:%M")

            # 获取站点观测值
            station_precip = self.station_ds["rain1h_qc"].sel(time=current_time).values

            # 获取插值结果
            idw_grid = self.idw_ds["corrected_precip"].sel(time=current_time).values
            mprism_grid = self.mprism_ds["corrected_precip"].sel(time=current_time).values

            # 提取站点位置处的插值值
            for i in range(len(self.station_lats)):
                obs_value = station_precip[i]

                # 只评估有效降水
                if obs_value < min_precip:
                    continue

                # 找到最接近站点的网格点
                lat_idx = np.abs(self.idw_ds.lat.values - self.station_lats[i]).argmin()
                lon_idx = np.abs(self.idw_ds.lon.values - self.station_lons[i]).argmin()

                # 提取插值值
                idw_value = idw_grid[lat_idx, lon_idx]
                mprism_value = mprism_grid[lat_idx, lon_idx]

                # 存储结果
                result = {
                    "time": time_str,
                    "station_id": self.station_ds.station.values[i],
                    "lat": self.station_lats[i],
                    "lon": self.station_lons[i],
                    "obs_precip": obs_value,
                    "idw_precip": idw_value,
                    "mprism_precip": mprism_value,
                    "idw_error": idw_value - obs_value,
                    "mprism_error": mprism_value - obs_value,
                    "idw_abs_error": abs(idw_value - obs_value),
                    "mprism_abs_error": abs(mprism_value - obs_value),
                    "idw_rel_error": abs(idw_value - obs_value) / (obs_value + 1e-5),
                    "mprism_rel_error": abs(mprism_value - obs_value) / (obs_value + 1e-5),
                }

                # 添加高程信息(如果有)
                if hasattr(self, "station_dem"):
                    result["elevation"] = self.station_dem[i]

                results.append(result)

        # 转换为DataFrame
        eval_df = pd.DataFrame(results)

        # 保存评估结果
        eval_file = os.path.join(self.output_folder, "station_evaluation.csv")
        eval_df.to_csv(eval_file, index=False)
        print(f"站点评估结果已保存至: {eval_file}")

        return eval_df

    def compute_metrics(self, eval_df, group_by=None):
        """
        计算统计指标

        参数:
        eval_df: 站点评估DataFrame
        group_by: 分组评估的列名

        返回:
        包含指标的DataFrame
        """

        # 定义计算单个组的指标函数
        def calculate_group_metrics(group_df):
            # 检查组是否为空
            if len(group_df) == 0:
                return None

            # 计算各种评价指标
            idw_metrics = {
                "方法": "IDW",
                "样本数": len(group_df),
                "均方根误差(RMSE)": np.sqrt(mean_squared_error(group_df["obs_precip"], group_df["idw_precip"])),
                "平均绝对误差(MAE)": mean_absolute_error(group_df["obs_precip"], group_df["idw_precip"]),
                "偏差(Bias)": np.mean(group_df["idw_precip"] - group_df["obs_precip"]),
                "决定系数(R²)": r2_score(group_df["obs_precip"], group_df["idw_precip"]),
                "相关系数": np.corrcoef(group_df["obs_precip"], group_df["idw_precip"])[0, 1],
                "平均相对误差": np.mean(group_df["idw_rel_error"]),
                "中位数相对误差": np.median(group_df["idw_rel_error"]),
            }

            mprism_metrics = {
                "方法": "MPRISM",
                "样本数": len(group_df),
                "均方根误差(RMSE)": np.sqrt(mean_squared_error(group_df["obs_precip"], group_df["mprism_precip"])),
                "平均绝对误差(MAE)": mean_absolute_error(group_df["obs_precip"], group_df["mprism_precip"]),
                "偏差(Bias)": np.mean(group_df["mprism_precip"] - group_df["obs_precip"]),
                "决定系数(R²)": r2_score(group_df["obs_precip"], group_df["mprism_precip"]),
                "相关系数": np.corrcoef(group_df["obs_precip"], group_df["mprism_precip"])[0, 1],
                "平均相对误差": np.mean(group_df["mprism_rel_error"]),
                "中位数相对误差": np.median(group_df["mprism_rel_error"]),
            }

            # 计算NSE (Nash-Sutcliffe Efficiency)
            obs_mean = np.mean(group_df["obs_precip"])
            nse_denom = np.sum((group_df["obs_precip"] - obs_mean) ** 2)

            if nse_denom > 0:
                idw_nse = 1 - np.sum((group_df["idw_precip"] - group_df["obs_precip"]) ** 2) / nse_denom
                mprism_nse = 1 - np.sum((group_df["mprism_precip"] - group_df["obs_precip"]) ** 2) / nse_denom

                idw_metrics["NSE系数"] = idw_nse
                mprism_metrics["NSE系数"] = mprism_nse

            return pd.DataFrame([idw_metrics, mprism_metrics])

        # 分组计算指标
        if group_by is not None:
            metrics_list = []
            # 显式指定 observed=False 以消除警告
            for name, group in eval_df.groupby(group_by, observed=False):
                # 检查组是否为空
                if len(group) == 0:
                    print(f"警告: 组 '{name}' 没有数据，已跳过")
                    continue

                group_metrics = calculate_group_metrics(group)
                if group_metrics is not None:
                    group_metrics[group_by] = name
                    metrics_list.append(group_metrics)

            if not metrics_list:
                print(f"警告: 所有 '{group_by}' 分组都没有数据")
                return pd.DataFrame()

            metrics_df = pd.concat(metrics_list, ignore_index=True)
        else:
            # 检查整体数据是否为空
            if len(eval_df) == 0:
                print("警告: 评估数据为空")
                return pd.DataFrame()

            metrics_df = calculate_group_metrics(eval_df)

        return metrics_df

    def categorize_precipitation(self, eval_df):
        """
        根据降水量级别对评估数据进行分类

        参数:
        eval_df: 站点评估DataFrame

        返回:
        添加降水等级的DataFrame
        """
        # 复制DataFrame避免修改原始数据
        df = eval_df.copy()

        # 增加降水等级列
        conditions = [
            (df["obs_precip"] > 0.1) & (df["obs_precip"] <= 10),
            (df["obs_precip"] > 10) & (df["obs_precip"] <= 25),
            (df["obs_precip"] > 25) & (df["obs_precip"] <= 50),
            (df["obs_precip"] > 50),
        ]

        categories = ["小雨(0.1-10mm)", "中雨(10-25mm)", "大雨(25-50mm)", "暴雨(>50mm)"]
        df["降水等级"] = np.select(conditions, categories, default="无降水")

        return df

    def categorize_elevation(self, eval_df, bins=4):
        """
        根据海拔高度对评估数据进行分类

        参数:
        eval_df: 站点评估DataFrame
        bins: 高程分区数量

        返回:
        添加高程等级的DataFrame
        """
        if "elevation" not in eval_df.columns:
            print("警告: 无高程数据，无法进行高程分类")
            return eval_df

        # 复制DataFrame避免修改原始数据
        df = eval_df.copy()

        # 计算高程分位数
        elev_bins = np.linspace(df["elevation"].min(), df["elevation"].max(), bins + 1)

        # 创建高程分类
        df["高程等级"] = pd.cut(
            df["elevation"], bins=elev_bins, labels=[f"{int(elev_bins[i])}-{int(elev_bins[i+1])}m" for i in range(bins)]
        )

        return df

    def perform_comprehensive_evaluation(self, n_samples=None, min_precip=0.1):
        """
        执行全面评估

        参数:
        n_samples: 用于评估的时间样本数量
        min_precip: 最小有效降水量阈值

        返回:
        包含评估结果的字典
        """
        print("开始全面评估...")

        # 1. 在站点位置评估
        eval_df = self.evaluate_at_stations(n_samples, min_precip)

        # 2. 计算整体统计指标
        overall_metrics = self.compute_metrics(eval_df)
        print("\n===== 整体评估指标 =====")
        print(overall_metrics.to_string(index=False))

        # 3. 按降水等级分类评估
        category_df = self.categorize_precipitation(eval_df)
        precip_metrics = self.compute_metrics(category_df, group_by="降水等级")
        print("\n===== 按降水等级评估 =====")
        print(precip_metrics.to_string(index=False))

        # 4. 按高程分类评估 (如果有高程数据)
        if "elevation" in eval_df.columns:
            elev_df = self.categorize_elevation(eval_df)
            elev_metrics = self.compute_metrics(elev_df, group_by="高程等级")
            print("\n===== 按高程等级评估 =====")
            print(elev_metrics.to_string(index=False))
        else:
            elev_metrics = None

        # 5. 按季节分类评估
        eval_df["month"] = pd.to_datetime(eval_df["time"]).dt.month
        eval_df["季节"] = pd.cut(
            eval_df["month"],
            bins=[0, 2, 5, 8, 12],
            labels=["冬季(12-2月)", "春季(3-5月)", "夏季(6-8月)", "秋季(9-11月)"],
            include_lowest=True,
            right=True,
        )
        season_metrics = self.compute_metrics(eval_df, group_by="季节")
        print("\n===== 按季节评估 =====")
        print(season_metrics.to_string(index=False))

        # 6. 保存所有评价指标
        overall_metrics.to_csv(os.path.join(self.output_folder, "overall_metrics.csv"), index=False)
        precip_metrics.to_csv(os.path.join(self.output_folder, "precipitation_category_metrics.csv"), index=False)
        if elev_metrics is not None:
            elev_metrics.to_csv(os.path.join(self.output_folder, "elevation_category_metrics.csv"), index=False)
        season_metrics.to_csv(os.path.join(self.output_folder, "seasonal_metrics.csv"), index=False)

        # 7. 生成可视化
        self.generate_evaluation_plots(eval_df, category_df)

        return {
            "eval_df": eval_df,
            "overall_metrics": overall_metrics,
            "precip_metrics": precip_metrics,
            "elev_metrics": elev_metrics,
            "season_metrics": season_metrics,
        }

    def generate_evaluation_plots(self, eval_df, category_df):
        """生成评估可视化图表"""
        print("生成评估图表...")

        import matplotlib.font_manager as fm

        # 直接指定SimHei字体文件路径（请根据实际路径修改）
        simhei_paths = [
            "/usr/share/fonts/truetype/simhei.ttf",
            "/usr/share/fonts/truetype/simhei/SimHei.ttf",
            "/usr/share/fonts/simhei.ttf",
            "/usr/share/fonts/opentype/simhei.ttf",
            "/usr/share/fonts/truetype/wqy/simhei.ttf",
            "/usr/local/share/fonts/simhei.ttf",
            # 你可以添加更多可能的路径
        ]
        simhei_font = None
        for path in simhei_paths:
            if os.path.exists(path):
                simhei_font = path
                break

        if simhei_font:
            print(f"使用SimHei字体文件: {simhei_font}")
            my_font = fm.FontProperties(fname=simhei_font)
            plt.rcParams["font.sans-serif"] = ["SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
        else:
            print("警告: 未找到SimHei字体文件，尝试使用系统字体")
            plt.rcParams["font.sans-serif"] = [
                "SimHei",
                "WenQuanYi Micro Hei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
                "Arial",
            ]
            plt.rcParams["axes.unicode_minus"] = False

        # 设置绘图样式
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

        # 1. 散点图：观测值vs插值值
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(eval_df["obs_precip"], eval_df["idw_precip"], alpha=0.5, label="数据点")
        max_val = max(eval_df["obs_precip"].max(), eval_df["idw_precip"].max()) * 1.1
        plt.plot([0, max_val], [0, max_val], "r--", label="1:1线")
        plt.xlabel("站点观测值 (mm)")
        plt.ylabel("IDW插值值 (mm)")
        plt.title("IDW插值vs观测值")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(eval_df["obs_precip"], eval_df["mprism_precip"], alpha=0.5, label="数据点")
        plt.plot([0, max_val], [0, max_val], "r--", label="1:1线")
        plt.xlabel("站点观测值 (mm)")
        plt.ylabel("MPRISM插值值 (mm)")
        plt.title("MPRISM插值vs观测值")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "scatter_comparison.png"), dpi=300)

        # 2. 箱线图：不同降水等级的误差分布
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.boxplot(x="降水等级", y="idw_rel_error", data=category_df)
        plt.title("IDW相对误差分布 (按降水等级)")
        plt.ylabel("相对误差")
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        sns.boxplot(x="降水等级", y="mprism_rel_error", data=category_df)
        plt.title("MPRISM相对误差分布 (按降水等级)")
        plt.ylabel("相对误差")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "error_boxplot_by_category.png"), dpi=300)

        # 3. 误差的空间分布图
        if len(eval_df) > 0:
            # 计算每个站点的平均误差
            station_avg = (
                eval_df.groupby("station_id")
                .agg(
                    {
                        "lat": "first",
                        "lon": "first",
                        "idw_error": "mean",
                        "mprism_error": "mean",
                        "idw_abs_error": "mean",
                        "mprism_abs_error": "mean",
                    }
                )
                .reset_index()
            )

            # 绘制误差空间分布图
            plt.figure(figsize=(14, 12))

            # 使用普通子图替代Cartopy地图
            ax1 = plt.subplot(2, 1, 1)

            # 使用普通散点图，用颜色表示误差
            scatter = ax1.scatter(
                station_avg["lon"],
                station_avg["lat"],
                c=station_avg["idw_abs_error"],
                cmap="RdYlBu_r",
                s=50,
                alpha=0.7,
            )
            plt.colorbar(scatter, ax=ax1, label="平均绝对误差 (mm)")
            ax1.set_title("IDW平均绝对误差空间分布")
            ax1.set_xlabel("经度")
            ax1.set_ylabel("纬度")
            ax1.grid(True, linestyle="--", alpha=0.7)

            ax2 = plt.subplot(2, 1, 2)

            scatter = ax2.scatter(
                station_avg["lon"],
                station_avg["lat"],
                c=station_avg["mprism_abs_error"],
                cmap="RdYlBu_r",
                s=50,
                alpha=0.7,
            )
            plt.colorbar(scatter, ax=ax2, label="平均绝对误差 (mm)")
            ax2.set_title("MPRISM平均绝对误差空间分布")
            ax2.set_xlabel("经度")
            ax2.set_ylabel("纬度")
            ax2.grid(True, linestyle="--", alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, "error_spatial_distribution.png"), dpi=300)

        # 4. 降水强度评估图
        plt.figure(figsize=(10, 6))
        # 计算两种方法的误差统计
        error_stats = []

        for intensity in sorted(category_df["降水等级"].unique()):
            subset = category_df[category_df["降水等级"] == intensity]
            if len(subset) == 0:
                continue

            idw_rmse = np.sqrt(mean_squared_error(subset["obs_precip"], subset["idw_precip"]))
            mprism_rmse = np.sqrt(mean_squared_error(subset["obs_precip"], subset["mprism_precip"]))

            error_stats.append(
                {"等级": intensity, "IDW_RMSE": idw_rmse, "MPRISM_RMSE": mprism_rmse, "样本数": len(subset)}
            )

        error_df = pd.DataFrame(error_stats)

        # 绘制条形图
        x = np.arange(len(error_df))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(12, 7))

        # 绘制RMSE条形图
        bar1 = ax1.bar(x - width / 2, error_df["IDW_RMSE"], width, label="IDW RMSE", color="skyblue")
        bar2 = ax1.bar(x + width / 2, error_df["MPRISM_RMSE"], width, label="MPRISM RMSE", color="salmon")

        # 添加样本数量的线图
        ax2 = ax1.twinx()
        ax2.plot(x, error_df["样本数"], "o-", color="green", label="样本数量")

        # 设置图表
        ax1.set_xlabel("降水等级")
        ax1.set_ylabel("RMSE (mm)")
        ax2.set_ylabel("样本数量")
        ax1.set_title("不同降水等级下的插值方法RMSE对比")
        ax1.set_xticks(x)
        ax1.set_xticklabels(error_df["等级"], rotation=45)
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "intensity_evaluation.png"), dpi=300)

        # 5. 降水季节性评估
        if "季节" in eval_df.columns:
            seasonal_stats = []

            for season in sorted(eval_df["季节"].unique()):
                subset = eval_df[eval_df["季节"] == season]
                if len(subset) == 0:
                    continue

                idw_rmse = np.sqrt(mean_squared_error(subset["obs_precip"], subset["idw_precip"]))
                mprism_rmse = np.sqrt(mean_squared_error(subset["obs_precip"], subset["mprism_precip"]))

                seasonal_stats.append(
                    {"季节": season, "IDW_RMSE": idw_rmse, "MPRISM_RMSE": mprism_rmse, "样本数": len(subset)}
                )

            season_df = pd.DataFrame(seasonal_stats)

            # 绘制季节性评估图
            plt.figure(figsize=(12, 7))

            x = np.arange(len(season_df))
            width = 0.35

            fig, ax1 = plt.subplots(figsize=(12, 7))

            bar1 = ax1.bar(x - width / 2, season_df["IDW_RMSE"], width, label="IDW RMSE", color="cornflowerblue")
            bar2 = ax1.bar(x + width / 2, season_df["MPRISM_RMSE"], width, label="MPRISM RMSE", color="lightcoral")

            ax2 = ax1.twinx()
            ax2.plot(x, season_df["样本数"], "o-", color="darkgreen", label="样本数量")

            ax1.set_xlabel("季节")
            ax1.set_ylabel("RMSE (mm)")
            ax2.set_ylabel("样本数量")
            ax1.set_title("不同季节的插值方法RMSE对比")
            ax1.set_xticks(x)
            ax1.set_xticklabels(season_df["季节"])
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, "seasonal_evaluation.png"), dpi=300)

        print(f"所有评估图表已保存至文件夹: {self.output_folder}")

    def compare_spatial_patterns(self, time_index=None):
        """
        比较特定时间点的空间分布模式
        """
        if time_index is None:
            # 随机选择一个有降水的时间点
            time_indices = []
            for i in range(len(self.station_ds.time)):
                if np.mean(self.station_ds["rain1h_qc"].isel(time=i).values) > 1.0:
                    time_indices.append(i)

            if not time_indices:
                time_index = np.random.randint(0, len(self.station_ds.time))
            else:
                time_index = np.random.choice(time_indices)

        current_time = self.station_ds.time.values[time_index]
        time_str = pd.Timestamp(current_time).strftime("%Y-%m-%d %H:%M")

        # 获取站点观测值
        station_precip = self.station_ds["rain1h_qc"].sel(time=current_time).values

        # 获取插值网格
        idw_grid = self.idw_ds["corrected_precip"].sel(time=current_time).values
        mprism_grid = self.mprism_ds["corrected_precip"].sel(time=current_time).values

        # 计算差异场
        diff_grid = mprism_grid - idw_grid

        # 创建降水色标
        colors = plt.cm.jet(np.linspace(0, 1, 256))
        colors[:1, :] = np.array([1, 1, 1, 1])  # 第一个颜色设为白色（对应0值）
        precip_cmap = LinearSegmentedColormap.from_list("precip_cmap", colors)

        # 创建差异色标
        diff_cmap = plt.cm.RdBu_r

        # 绘制空间分布对比图
        plt.figure(figsize=(18, 12))

        # IDW结果 - 使用普通子图
        ax1 = plt.subplot(2, 2, 1)

        max_val = max(np.max(idw_grid), np.max(mprism_grid))
        levels = np.linspace(0, max_val, 15)

        # 绘制等值线
        cs1 = ax1.contourf(self.idw_ds.lon, self.idw_ds.lat, idw_grid, levels=levels, cmap=precip_cmap)

        # 添加站点
        scatter = ax1.scatter(
            self.station_lons,
            self.station_lats,
            c=station_precip,
            cmap=precip_cmap,
            s=50,
            edgecolor="black",
        )

        plt.colorbar(cs1, ax=ax1, label="降水量 (mm)")
        ax1.set_title(f"IDW插值 ({time_str})")
        ax1.set_xlabel("经度")
        ax1.set_ylabel("纬度")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # MPRISM结果 - 使用普通子图
        ax2 = plt.subplot(2, 2, 2)

        cs2 = ax2.contourf(
            self.mprism_ds.lon,
            self.mprism_ds.lat,
            mprism_grid,
            levels=levels,
            cmap=precip_cmap,
        )

        # 添加站点
        ax2.scatter(
            self.station_lons,
            self.station_lats,
            c=station_precip,
            cmap=precip_cmap,
            s=50,
            edgecolor="black",
        )

        plt.colorbar(cs2, ax=ax2, label="降水量 (mm)")
        ax2.set_title(f"MPRISM插值 ({time_str})")
        ax2.set_xlabel("经度")
        ax2.set_ylabel("纬度")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 差异场 - 使用普通子图
        ax3 = plt.subplot(2, 2, 3)

        max_diff = np.max([np.abs(np.min(diff_grid)), np.abs(np.max(diff_grid))])
        diff_levels = np.linspace(-max_diff, max_diff, 15)

        cs3 = ax3.contourf(
            self.idw_ds.lon,
            self.idw_ds.lat,
            diff_grid,
            levels=diff_levels,
            cmap=diff_cmap,
        )

        plt.colorbar(cs3, ax=ax3, label="差异 (MPRISM-IDW) (mm)")
        ax3.set_title(f"差异场 (MPRISM-IDW)")
        ax3.set_xlabel("经度")
        ax3.set_ylabel("纬度")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # 直方图 - 不变
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(diff_grid.flatten(), bins=50, alpha=0.7, color="steelblue")
        ax4.set_xlabel("差异值 (mm)")
        ax4.set_ylabel("频数")
        ax4.set_title("差异值分布")

        # 添加均值和标准差信息
        mean_diff = np.mean(diff_grid)
        std_diff = np.std(diff_grid)
        ax4.axvline(mean_diff, color="r", linestyle="--", label=f"均值: {mean_diff:.3f}mm")
        ax4.text(
            0.7,
            0.9,
            f"均值: {mean_diff:.3f}mm\n标准差: {std_diff:.3f}mm",
            transform=ax4.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f"spatial_comparison_{time_str.replace(':', '')}.png"), dpi=300)
        print(f"空间分布对比图已保存 (时间点: {time_str})")

        return {
            "time": time_str,
            "idw_grid": idw_grid,
            "mprism_grid": mprism_grid,
            "diff_grid": diff_grid,
            "station_values": station_precip,
        }


# 使用示例
def main():
    """主函数，运行降水插值评估"""
    # 基本配置
    station_file = "/mnt/h/DataSet/station_precipitation_data_filled.nc"
    idw_result_file = "/mnt/h/DataSet/PreGrids_IDW/temp_output.zarr"
    mprism_result_file = "/mnt/h/DataSet/PreGrids_MPRISM/temp_output.zarr"
    dem_file = "/mnt/h/DataSet/3-DEM/DEM_clip.nc"
    output_folder = "/mnt/h/DataSet/Interpolation_Evaluation"

    # 创建评估器实例
    evaluator = PrecipitationInterpolationEvaluator(
        station_file=station_file,
        idw_result_file=idw_result_file,
        mprism_result_file=mprism_result_file,
        dem_file=dem_file,
        output_folder=output_folder,
    )

    # 执行全面评估
    results = evaluator.perform_comprehensive_evaluation(n_samples=10)  # 使用所有可用时间点

    # 比较几个时间点的空间分布
    for _ in range(3):
        evaluator.compare_spatial_patterns()

    print("评估完成！")


if __name__ == "__main__":
    main()
