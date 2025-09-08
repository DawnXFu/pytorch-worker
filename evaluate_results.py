import glob
import json
import logging
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_lat_lon(nc_path):
    ds = nc.Dataset(nc_path)
    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]
    ds.close()
    return lat, lon


def calculate_grid_metrics(pred, label):
    """
    pred, label: [time, height, width]
    返回: dict，每个指标都是 [height, width] 的数组
    """
    h, w = pred.shape[1:]
    cc_map = np.zeros((h, w))
    rmse_map = np.zeros((h, w))
    mae_map = np.zeros((h, w))
    rb_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            p = pred[:, i, j]
            l = label[:, i, j]
            mask = ~(np.isnan(p) | np.isnan(l))
            p = p[mask]
            l = l[mask]
            if len(p) == 0:
                cc_map[i, j] = np.nan
                rmse_map[i, j] = np.nan
                mae_map[i, j] = np.nan
                rb_map[i, j] = np.nan
                continue
            # CC
            if np.all(p == p[0]) or np.all(l == l[0]):
                cc = 0.0
            else:
                cc = np.corrcoef(p, l)[0, 1]
            cc_map[i, j] = cc
            # RMSE
            rmse_map[i, j] = np.sqrt(np.mean((p - l) ** 2))
            # MAE
            mae_map[i, j] = np.mean(np.abs(p - l))
            # RB
            rb_map[i, j] = np.mean((p - l) / (l + 1e-6))
    return {"CC": cc_map, "RMSE": rmse_map, "MAE": mae_map, "RB": rb_map}


def save_prediction_to_nc(prediction, label, timestamps, lat, lon, save_path):
    """
    将单个模型的预测结果和真实值保存为NC文件

    Args:
        prediction: [time, height, width] 的预测值数组
        label: [time, height, width] 的真实值数组
        timestamps: 时间戳列表
        lat: 纬度数组
        lon: 经度数组
        save_path: 保存路径
    """
    with nc.Dataset(save_path, "w") as ds:
        # 创建维度
        ds.createDimension("time", len(timestamps))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))

        # 创建坐标变量
        time_var = ds.createVariable("time", "f8", ("time",))
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))

        # 写入坐标数据
        lat_var[:] = lat
        lon_var[:] = lon

        # 写入时间数据
        if isinstance(timestamps[0], str):
            timestamps = [datetime.strptime(str(t), "%Y%m%d%H") for t in timestamps]
        elif isinstance(timestamps[0], np.datetime64):
            timestamps = pd.to_datetime(timestamps)

        # 将时间转换为相对于参考时间的小时数
        reference_date = datetime(1900, 1, 1)
        time_hours = [(t - reference_date).total_seconds() / 3600 for t in timestamps]
        time_var[:] = time_hours

        # 设置时间变量的属性
        time_var.units = f'hours since {reference_date.strftime("%Y-%m-%d %H:%M:%S")}'
        time_var.calendar = "gregorian"

        # 创建预测结果变量
        pred_var = ds.createVariable("prediction", "f4", ("time", "lat", "lon"), zlib=True, complevel=5)  # 启用压缩
        pred_var[:] = prediction
        pred_var.long_name = "Precipitation Prediction"
        pred_var.units = "mm"

        # 创建真实值变量
        label_var = ds.createVariable("label", "f4", ("time", "lat", "lon"), zlib=True, complevel=5)  # 启用压缩
        label_var[:] = label
        label_var.long_name = "Precipitation Label"
        label_var.units = "mm"

        # 添加全局属性
        ds.description = "Model prediction and label for precipitation"
        ds.history = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


def filter_time_range(pred, label, timestamps, start=None, end=None):
    """
    根据时间范围筛选数据
    start, end: 字符串，格式为 'YYYYMMDDhh'，如 '2023110100'
    timestamps: np.ndarray of np.datetime64 或 datetime.datetime 或 str
    """
    if start is None and end is None:
        return pred, label, timestamps

    # 将timestamps转换为numpy数组
    timestamps = np.array(timestamps)

    # 转换为 datetime 对象
    if isinstance(timestamps[0], np.datetime64):
        ts_dt = np.array([pd.to_datetime(str(t)) for t in timestamps])
    elif isinstance(timestamps[0], datetime):
        ts_dt = np.array(timestamps)
    else:
        ts_dt = np.array([datetime.strptime(str(t), "%Y%m%d%H") for t in timestamps])

    start_dt = datetime.strptime(start, "%Y%m%d%H") if start else ts_dt.min()
    end_dt = datetime.strptime(end, "%Y%m%d%H") if end else ts_dt.max()

    # 创建掩码数组
    mask = np.array([(t >= start_dt) and (t <= end_dt) for t in ts_dt])

    # 应用掩码
    filtered_pred = pred[mask]
    filtered_label = label[mask]
    filtered_timestamps = timestamps[mask]

    return filtered_pred, filtered_label, filtered_timestamps


def load_results_from_dir(results_dir):
    # 查找最新的测试结果文件
    pkl_files = glob.glob(os.path.join(results_dir, "test_results_*", "full_results.pkl"))
    if not pkl_files:
        logger.error(f"在 {results_dir} 中未找到测试结果文件")
        return

    latest_pkl = max(pkl_files, key=os.path.getctime)
    logger.info(f"正在处理文件: {latest_pkl}")

    # 加载测试结果
    with open(latest_pkl, "rb") as f:
        results = pickle.load(f)
    return results


def save_hourly_metrics_to_csv(predictions_dict, labels_dict, timestamps, save_dir):
    """
    将每个时刻的评估指标保存为CSV文件

    Args:
        predictions_dict: 包含各模型预测结果的字典
        labels_dict: 包含各模型真实值的字典
        timestamps: 时间戳列表
        save_dir: 保存目录路径
    """
    # 创建一个存储所有模型指标的字典
    all_metrics = {
        "Datetime": [],
        "Model": [],
        "CC": [],
        "RMSE": [],
        "MAE": [],
        "Mean_Pred": [],
        "Mean_Label": [],
        "Max_Pred": [],
        "Max_Label": [],
        "Min_Pred": [],
        "Min_Label": [],
    }

    for model_name in predictions_dict.keys():
        predictions = predictions_dict[model_name]
        labels = labels_dict[model_name]

        # 确保时间戳格式正确
        if isinstance(timestamps[0], str):
            formatted_times = [datetime.strptime(str(t), "%Y%m%d%H") for t in timestamps]
        elif isinstance(timestamps[0], np.datetime64):
            formatted_times = pd.to_datetime(timestamps)
        else:
            formatted_times = timestamps

        for t in range(len(formatted_times)):
            # 获取当前时刻的数据
            current_pred = predictions[t]
            current_label = labels[t]

            # 计算当前时刻的指标
            valid_mask = ~(np.isnan(current_pred) | np.isnan(current_label))
            pred_valid = current_pred[valid_mask]
            label_valid = current_label[valid_mask]

            if len(pred_valid) > 0:
                # 计算相关系数
                cc = np.corrcoef(pred_valid.flatten(), label_valid.flatten())[0, 1]
                # 计算RMSE
                rmse = np.sqrt(np.mean((pred_valid - label_valid) ** 2))
                # 计算MAE
                mae = np.mean(np.abs(pred_valid - label_valid))

                # 计算统计值
                mean_pred = np.nanmean(current_pred)
                mean_label = np.nanmean(current_label)
                max_pred = np.nanmax(current_pred)
                max_label = np.nanmax(current_label)
                min_pred = np.nanmin(current_pred)
                min_label = np.nanmin(current_label)

                # 添加到字典中
                all_metrics["Datetime"].append(formatted_times[t])
                all_metrics["Model"].append(model_name)
                all_metrics["CC"].append(cc)
                all_metrics["RMSE"].append(rmse)
                all_metrics["MAE"].append(mae)
                all_metrics["Mean_Pred"].append(mean_pred)
                all_metrics["Mean_Label"].append(mean_label)
                all_metrics["Max_Pred"].append(max_pred)
                all_metrics["Max_Label"].append(max_label)
                all_metrics["Min_Pred"].append(min_pred)
                all_metrics["Min_Label"].append(min_label)

    # 转换为DataFrame并保存
    df = pd.DataFrame(all_metrics)
    # 将时间戳转换为字符串格式
    df["Datetime"] = df["Datetime"].dt.strftime("%Y%m%d%H")
    save_path = os.path.join(save_dir, "hourly_metrics.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"已保存每小时指标到: {save_path}")


def plot_hourly_metrics_from_csv(save_dir, model_names):
    """
    从CSV文件读取并绘制每小时的评估指标散点图和降水统计数据对比图
    真实值只绘制一次作为参考
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    metrics_names = ["CC", "RMSE", "MAE"]
    stats_pairs = [("Mean", "平均"), ("Max", "最大")]

    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 15))
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 10))

    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    # 用于存储真实标签数据
    label_data = None

    # 读取并绘制每个模型的数据
    for model_name, color in zip(model_names, colors):
        # 读取CSV文件
        csv_path = os.path.join(save_dir, f"{model_name}/hourly_metrics_fixed.csv")
        df = pd.read_csv(csv_path)
        # 将时间戳转换为datetime格式
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d%H")

        # 绘制评估指标散点图
        for idx, metric_name in enumerate(metrics_names):
            axes1[idx].scatter(
                df["Datetime"],
                df[metric_name],
                label=model_name,
                color=color,
                s=20,  # 设置点的大小
                alpha=0.6,  # 设置透明度
            )
            axes1[idx].set_title(f"{metric_name}")
            axes1[idx].set_xlabel("时间")
            axes1[idx].set_ylabel(metric_name)
            axes1[idx].grid(True)
            axes1[idx].legend()
            axes1[idx].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d %H"))
            plt.setp(axes1[idx].xaxis.get_majorticklabels(), rotation=45)

            # 设置y轴范围
            if idx == 1:  # RMSE
                axes1[idx].set_ylim(0, 10)
            elif idx == 2:  # MAE
                axes1[idx].set_ylim(0, 10)

        # 绘制降水统计预测值散点图
        for idx, (col_prefix, title) in enumerate(stats_pairs):
            pred_col = f"{col_prefix}_Pred"
            label_col = f"{col_prefix}_Label"

            # 绘制预测值
            axes2[idx].scatter(
                df["Datetime"],
                df[pred_col],
                label=f"{model_name}",
                color=color,
                s=20,  # 设置点的大小
                alpha=0.6,  # 设置透明度
            )

            # 记录标签数据(只需记录一次)
            if label_data is None:
                label_data = {
                    "Datetime": df["Datetime"],
                    "Mean": df["Mean_Label"],
                    "Max": df["Max_Label"],
                    "Min": df["Min_Label"],
                }

    # 在所有模型绘制完成后，绘制真实值
    for idx, (col_prefix, title) in enumerate(stats_pairs):
        axes2[idx].scatter(
            label_data["Datetime"],
            label_data[col_prefix],
            label="Gound Truth",
            color="black",
            alpha=0.3,
            s=15,  # 真实值点稍小一些
        )

        axes2[idx].set_title(f"{title}降水量")
        axes2[idx].set_xlabel("时间")
        axes2[idx].set_ylabel("降水量 (mm)")
        axes2[idx].grid(True)
        axes2[idx].legend()
        axes2[idx].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d %H"))
        plt.setp(axes2[idx].xaxis.get_majorticklabels(), rotation=45)

        # 设置y轴范围
        if idx == 0:  # 平均值
            axes2[idx].set_ylim(0, 10)
        elif idx == 1:  # 最大值
            axes2[idx].set_ylim(0, 50)

    # 调整布局并保存图片
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(os.path.join(save_dir, "hourly_metrics_comparison.png"), dpi=300)
    fig2.savefig(os.path.join(save_dir, "hourly_precipitation_stats_comparison.png"), dpi=300)
    plt.close(fig1)
    plt.close(fig2)

    logger.info(f"已保存小时尺度对比散点图到: {save_dir}")


def calc_metrics_by_label_bins_timewise(pred, label, bins):
    """
    pred, label: [time, h, w]，bins: list
    返回: dict，每个区间的各指标均值
    """
    n_time = pred.shape[0]
    n_bins = len(bins) - 1
    # 初始化累加器
    sum_rmse = np.zeros(n_bins)
    sum_mae = np.zeros(n_bins)
    sum_cc = np.zeros(n_bins)
    sum_pod = np.zeros(n_bins)
    sum_far = np.zeros(n_bins)
    count = np.zeros(n_bins)
    valid_time = np.zeros(n_bins)  # 统计每个区间有多少个时间点有有效样本

    for t in range(n_time):
        p = pred[t]
        l = label[t]
        mask_valid = (~np.isnan(p)) & (~np.isnan(l))
        if not np.any(mask_valid):
            continue
        p = p[mask_valid]
        l = l[mask_valid]
        bin_ids = np.digitize(l, bins, right=False) - 1
        for i in range(n_bins):
            mask = bin_ids == i
            if np.sum(mask) == 0:
                continue
            pp = p[mask]
            ll = l[mask]
            n = pp.size
            count[i] += n
            valid_time[i] += 1
            # RMSE
            sum_rmse[i] += np.sqrt(np.mean((pp - ll) ** 2))
            # MAE
            sum_mae[i] += np.mean(np.abs(pp - ll))
            # CC
            if n > 1 and np.std(pp) > 0 and np.std(ll) > 0:
                sum_cc[i] += np.corrcoef(pp, ll)[0, 1]
            else:
                sum_cc[i] += 0
            # POD/FAR（二分类示例，阈值可自定义）
            threshold = bins[i]
            hits = np.sum((pp >= threshold) & (ll >= threshold))
            misses = np.sum((pp < threshold) & (ll >= threshold))
            false_alarms = np.sum((pp >= threshold) & (ll < threshold))
            pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
            far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
            sum_pod[i] += pod if not np.isnan(pod) else 0
            sum_far[i] += far if not np.isnan(far) else 0

    # 取平均
    avg_rmse = sum_rmse / valid_time
    avg_mae = sum_mae / valid_time
    avg_cc = sum_cc / valid_time
    avg_pod = sum_pod / valid_time
    avg_far = sum_far / valid_time

    results = []
    for i in range(n_bins):
        results.append(
            {
                "range": (bins[i], bins[i + 1]),
                "RMSE": avg_rmse[i],
                "CC": avg_cc[i],
                "MAE": avg_mae[i],
                "POD": avg_pod[i],
                "FAR": avg_far[i],
                "count": count[i],
            }
        )
    return results


def calculate_categorical_metrics(pred, label, threshold):
    """
    计算二分类评估指标(POD, FAR, TS)

    Args:
        pred: 预测值数组
        label: 真实值数组
        threshold: 降水阈值

    Returns:
        dict: 包含POD、FAR、TS的字典
    """
    # 将数据二值化
    pred_binary = pred >= threshold
    label_binary = label >= threshold

    # 计算混淆矩阵元素
    hits = np.sum((pred_binary) & (label_binary))
    misses = np.sum((~pred_binary) & (label_binary))
    false_alarms = np.sum((pred_binary) & (~label_binary))

    # 计算指标
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan  # Probability of Detection
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan  # False Alarm Ratio
    ts = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan  # Threat Score

    return {"POD": pod, "FAR": far, "TS": ts, "Hits": hits, "Misses": misses, "FalseAlarms": false_alarms}


def save_categorical_metrics_to_csv(predictions_dict, labels_dict, timestamps, thresholds, save_dir):
    """
    计算并保存不同阈值下的分类评估指标

    Args:
        predictions_dict: 包含各模型预测结果的字典
        labels_dict: 包含各模型真实值的字典
        timestamps: 时间戳列表
        thresholds: 阈值列表，如 [0.1, 1, 5, 10]
        save_dir: 保存目录路径
    """
    # 创建存储所有指标的字典
    all_metrics = {
        "Datetime": [],
        "Model": [],
        "Threshold": [],
        "POD": [],
        "FAR": [],
        "TS": [],
        "Hits": [],
        "Misses": [],
        "FalseAlarms": [],
    }

    # 确保时间戳格式正确
    if isinstance(timestamps[0], str):
        formatted_times = [datetime.strptime(str(t), "%Y%m%d%H") for t in timestamps]
    elif isinstance(timestamps[0], np.datetime64):
        formatted_times = pd.to_datetime(timestamps)
    else:
        formatted_times = timestamps

    for model_name in predictions_dict.keys():
        predictions = predictions_dict[model_name]
        labels = labels_dict[model_name]

        for t in range(len(formatted_times)):
            current_pred = predictions[t]
            current_label = labels[t]

            # 处理缺失值
            valid_mask = ~(np.isnan(current_pred) | np.isnan(current_label))
            pred_valid = current_pred[valid_mask]
            label_valid = current_label[valid_mask]

            if len(pred_valid) > 0:
                for threshold in thresholds:
                    metrics = calculate_categorical_metrics(pred_valid, label_valid, threshold)

                    # 添加到字典中
                    all_metrics["Datetime"].append(formatted_times[t])
                    all_metrics["Model"].append(model_name)
                    all_metrics["Threshold"].append(threshold)
                    all_metrics["POD"].append(metrics["POD"])
                    all_metrics["FAR"].append(metrics["FAR"])
                    all_metrics["TS"].append(metrics["TS"])
                    all_metrics["Hits"].append(metrics["Hits"])
                    all_metrics["Misses"].append(metrics["Misses"])
                    all_metrics["FalseAlarms"].append(metrics["FalseAlarms"])

    # 转换为DataFrame并保存
    df = pd.DataFrame(all_metrics)
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.strftime("%Y%m%d%H")
    save_path = os.path.join(save_dir, "categorical_metrics.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"已保存分类评估指标到: {save_path}")

    # 计算并保存各阈值下的平均指标
    mean_metrics = df.groupby(["Model", "Threshold"])[["POD", "FAR", "TS"]].mean().reset_index()
    mean_metrics_path = os.path.join(save_dir, "categorical_metrics_mean.csv")
    mean_metrics.to_csv(mean_metrics_path, index=False, float_format="%.4f")
    logger.info(f"已保存平均分类评估指标到: {mean_metrics_path}")


def evaluate_model_results(results_dir):
    """
    评估某个模型的测试结果，计算累积降水的空间分布和评估指标

    Args:
        results_dir: 包含测试结果的目录路径
        time_window: 字典，包含 'start' 和 'end' 的时间范围（可选）
    """
    model_dict = {
        # "UNET": "UNET",
        # "ConvLSTM": "ConvLSTM",
        "UGALSTM": "UGALSTM",
    }

    lat, lon = get_lat_lon("/mnt/d/Data/train/train_batch_0001.nc")

    # 创建结果目录
    for model_name in model_dict.values():
        model_dict[model_name] = os.path.join(results_dir, model_name)

    predictions_dict = {}
    labels_dict = {}

    for model_name, model_dir in model_dict.items():
        logger.info(f"正在处理模型: {model_name}，目录: {model_dir}")

        results = load_results_from_dir(model_dir)
        if not results:
            continue

        predictions = results["predictions"]
        labels = results["labels"]
        timestamps = results["timestamps"]

        predictions, labels, timestamps = filter_time_range(
            predictions, labels, timestamps, start="2022120100", end="2022120700"
        )

        save_prediction_to_nc(predictions, labels, timestamps, lat, lon, os.path.join(model_dir, "predictions.nc"))

        # # 转换数据类型以节省内存
        # predictions_dict[model_name] = predictions.astype(np.float32)
        # labels_dict[model_name] = labels.astype(np.float32)

        # # 添加以下代码来计算分类评估指标
        # thresholds = [5.19, 5.82, 6.20, 6.93, 7.01, 7.15]  # 可以根据需要调整阈值
        # save_categorical_metrics_to_csv(predictions_dict, labels_dict, timestamps, thresholds, results_dir)

        # results_dir = os.path.join(results_dir, model_name)

        # save_hourly_metrics_to_csv(predictions_dict, labels_dict, timestamps, results_dir)

    # # 计算并显示分位数统计
    # calculate_precipitation_percentiles(labels_dict)


def calculate_precipitation_percentiles(labels_dict):
    """
    计算降水量的各个分位数

    Args:
        labels_dict: 包含各模型真实值的字典
    """
    print("\n降水量分位数统计：")
    print("-" * 50)

    for model_name, labels in labels_dict.items():
        # 移除NaN值
        valid_data = labels[~np.isnan(labels)]

        # 计算分位数
        percentiles = [5, 10, 15, 25, 50, 75, 90]
        values = np.percentile(valid_data, percentiles)

        print(f"\n模型 {model_name} 的标签统计：")
        print(f"数据总量: {len(valid_data)}")
        for p, v in zip(percentiles, values):
            print(f"{p}% 分位数: {v:.2f} mm")

        # 额外显示一些基本统计量
        print(f"平均值: {np.mean(valid_data):.2f} mm")
        print(f"最大值: {np.max(valid_data):.2f} mm")
        print(f"非零降水占比: {(valid_data > 0).sum() / len(valid_data) * 100:.2f}%")


def plot_mean_categorical_metrics(save_dir, model_names):
    """
    绘制不同阈值下的平均分类评估指标

    Args:
        save_dir: 结果保存目录
        model_names: 模型名称列表
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 读取平均指标数据
    mean_metrics_path = os.path.join(save_dir, "categorical_metrics_mean.csv")
    if not os.path.exists(mean_metrics_path):
        logger.error(f"未找到平均指标文件: {mean_metrics_path}")
        return

    df = pd.read_csv(mean_metrics_path)

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["POD", "FAR", "TS"]
    titles = ["命中率(POD)", "虚警率(FAR)", "威胁评分(TS)"]

    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    # 绘制每个指标
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        for model_name, color in zip(model_names, colors):
            model_data = df[df["Model"] == model_name]
            axes[idx].plot(model_data["Threshold"], model_data[metric], marker="o", label=model_name, color=color)

        axes[idx].set_title(title)
        axes[idx].set_xlabel("降水阈值 (mm)")
        axes[idx].set_ylabel("指标值")
        axes[idx].grid(True)
        axes[idx].legend()

        # 设置y轴范围
        axes[idx].set_ylim(0, 1)

        # 对x轴使用对数刻度
        axes[idx].set_xscale("log")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "categorical_metrics_mean.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"已保存分类评估指标均值图到: {save_path}")

    # 打印均值表格
    print("\n各阈值下的平均指标值：")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))


def load_test_data():
    ds = xr.open_mfdataset("/mnt/d/Data/test/test_batch_*.nc", combine="by_coords")
    origin = ds["PRE"].values
    label = ds["corrected_precip"].values

    pridction_dict = {"origin": origin}
    label_dict = {"origin": label}
    timestaps = ds["time"].values

    results_dir = "/mnt/d/Data/result/"

    thresholds = [5.19, 5.82, 6.20, 6.93, 7.01, 7.15]  # 可以根据需要调整阈值
    save_categorical_metrics_to_csv(pridction_dict, label_dict, timestaps, thresholds, results_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="/mnt/d/Data/result", help="结果目录路径")
    parser.add_argument("--models", nargs="+", default=["UNET", "ConvLSTM", "UGALSTM"], help="要对比的模型列表")
    args = parser.parse_args()

    plot_hourly_metrics_from_csv(args.result_dir, args.models)
    # evaluate_model_results(args.result_dir)

    # load_test_data()
