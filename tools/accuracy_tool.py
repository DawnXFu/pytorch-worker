import logging
import re

import numpy as np
import torch

# 修改为使用正确的logger，而不是Logger类
logger = logging.getLogger(__name__)  # 修正logger初始化


# ...existing code...
def PrecipitionCorrection_accuracy_function(outputs, label, config, result, mode="train"):
    """
    计算并将每步指标追加存储在result中，每个step只记录当前batch的指标
    """
    precip_thresholds = {
        "light_rain": 0.1,
        "moderate_rain": 10.0,
        "heavy_rain": 25.0,
    }

    # 将单值替换为列表
    if result is None:
        result = {"CC": [], "RMSE": [], "MAE": [], "KGE": [], "NULL": True}
        for rain_type in precip_thresholds:
            result[f"POD_{rain_type}"] = []
            result[f"FAR_{rain_type}"] = []
            result[f"BIAS_{rain_type}"] = []
            result[f"TS_{rain_type}"] = []
        logger.info(f"初始化评估指标 (模式={mode})")

    outputs = outputs.detach()
    label = label.detach()
    batch_size = outputs.size(0)

    metric_sums = {
        "CC": 0.0,
        "RMSE": 0.0,
        "MAE": 0.0,
        "KGE": 0.0,
    }
    for rain_type in precip_thresholds:
        metric_sums[f"POD_{rain_type}"] = 0.0
        metric_sums[f"FAR_{rain_type}"] = 0.0
        metric_sums[f"BIAS_{rain_type}"] = 0.0
        metric_sums[f"TS_{rain_type}"] = 0.0

    for i in range(batch_size):
        pred = outputs[i].reshape(-1)
        true = label[i].reshape(-1)
        metric_sums["CC"] += Correlation_score(pred, true)
        metric_sums["RMSE"] += MSE_score(pred, true) ** 0.5
        metric_sums["MAE"] += MAE_score(pred, true)
        metric_sums["KGE"] += KGE_score(pred, true)

        for rain_type, threshold in precip_thresholds.items():
            metric_sums[f"POD_{rain_type}"] += POD_score(pred, true, threshold)
            metric_sums[f"FAR_{rain_type}"] += FAR_score(pred, true, threshold)
            metric_sums[f"BIAS_{rain_type}"] += Bias_score(pred, true, threshold)
            metric_sums[f"TS_{rain_type}"] += TS_score(pred, true, threshold)

    # 将当前batch的平均指标追加到列表
    for m in metric_sums:
        ave_val = metric_sums[m] / batch_size
        result[m].append(ave_val)

    if "NULL" in result:
        result["NULL"] = False

    return result


def gen_precipition_result(result):
    """
    计算降水预测评估指标的平均值
    """
    # 如果result字典的列表有空值，返回空字典
    if result["NULL"]:
        return {
            "CC": 0.0,
            "RMSE": 0.0,
            "MAE": 0.0,
            "KGE": 0.0,
            "POD_light_rain": 0.0,
            "FAR_light_rain": 0.0,
            "BIAS_light_rain": 0.0,
            "TS_light_rain": 0.0,
            "POD_moderate_rain": 0.0,
            "FAR_moderate_rain": 0.0,
            "BIAS_moderate_rain": 0.0,
            "TS_moderate_rain": 0.0,
            "POD_heavy_rain": 0.0,
            "FAR_heavy_rain": 0.0,
            "BIAS_heavy_rain": 0.0,
            "TS_heavy_rain": 0.0,
        }

    # 定义降水等级
    rain_types = [
        "light_rain",
        "moderate_rain",
        "heavy_rain",
    ]

    # 创建基本结果字典，处理可能的NaN值
    final_result = {}
    for metric in ["CC", "RMSE", "KGE", "MAE"]:
        # 现在 result[metric] 是一个列表，需 sum(...) 后再 / count
        vals = result[metric]
        avg_val = sum(vals) / len(vals) if vals else 0.0
        final_result[metric] = round(0.0 if np.isnan(avg_val) else avg_val, 3)

    # 添加各降水等级的命中率和空报率
    for rain_type in rain_types:
        for mt in [f"POD_{rain_type}", f"FAR_{rain_type}", f"BIAS_{rain_type}", f"TS_{rain_type}"]:
            if mt in result:
                vals = result[mt]
                avg_val = sum(vals) / len(vals) if vals else 0.0
                final_result[mt] = round(0.0 if np.isnan(avg_val) else avg_val, 3)

    return final_result


def MSE_score(pred_x, y_label):
    """
    计算均方误差（MSE）

    参数:
    - pred_x: 预测值 (1D 或 ND torch.Tensor)
    - y_label: 真实标签 (与pred_x同shape的torch.Tensor)

    返回:
    - 均方误差MSE评分
    """
    return ((pred_x - y_label) ** 2).mean().item()


def MAE_score(pred_x, y_label):
    """
    计算平均绝对误差（MAE）

    参数:
    - pred_x: 预测值 (1D 或 ND torch.Tensor)
    - y_label: 真实标签 (与pred_x同shape的torch.Tensor)

    返回:
    - 平均绝对误差MAE评分
    """
    return (pred_x - y_label).abs().mean().item()


def Correlation_score(pred_x, y_label):
    """计算相关系数，处理边缘情况以避免NaN"""
    pred_x = pred_x.reshape(-1)
    y_label = y_label.reshape(-1)

    pred_mean = pred_x.mean().item()
    label_mean = y_label.mean().item()
    pred_std = pred_x.std().item()
    label_std = y_label.std().item()
    covariance = ((pred_x - pred_mean) * (y_label - label_mean)).mean().item()
    # 计算相关系数
    if pred_std > 0 and label_std > 0:
        correlation = covariance / (pred_std * label_std)
        return max(-1.0, min(1.0, correlation))
    else:
        # 如果标准差为0，返回0.0
        logger.warning("Standard deviation is zero, returning 0.0 for correlation.")
        return 0.0


def KGE_score(pred_x, y_label):
    """计算Kling-Gupta效率系数，增强数值稳定性"""
    pred_mean = pred_x.mean().item()
    true_mean = y_label.mean().item()
    pred_std = pred_x.std().item()
    true_std = y_label.std().item()

    # 获取相关系数，已处理可能的NaN情况
    r = Correlation_score(pred_x, y_label)

    # 处理零或极小值
    if abs(true_std) < 1e-8:
        alpha = 1.0 if abs(pred_std) < 1e-8 else 0.0
    else:
        alpha = pred_std / true_std

    if abs(true_mean) < 1e-8:
        beta = 1.0 if abs(pred_mean) < 1e-8 else 0.0
    else:
        beta = pred_mean / true_mean

    # 计算KGE，确保结果在有效范围内
    try:
        kge = 1 - ((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
        return max(-1.0, min(1.0, kge))  # 限制KGE在-1到1之间
    except:
        return 0.0  # 计算失败返回0


def TS_score(pred_x, y_label, threshold=0.1):
    """
    计算二分类降水威胁评分（TS）

    参数:
    - pred_x: 预测值 (1D 或 ND torch.Tensor)
    - y_label: 真实标签 (与pred_x同shape的torch.Tensor)
    - threshold: 判定降水的阈值（如0.1mm）

    返回:
    - 风险评分TS评分
    """
    hits, misses, falsealarms, correctnegatives = calc_confusion_matrix(pred_x, y_label, threshold)
    return hits / (hits + misses + falsealarms) if (hits + misses + falsealarms) > 0 else 1.0


def EST_score(pred_x, y_label, threshold=0.1):
    """
    计算二分类降水估计评分（EST）

    参数:
    - pred_x: 预测值 (1D 或 ND torch.Tensor)
    - y_label: 真实标签 (与pred_x同shape的torch.Tensor)
    - threshold: 判定降水的阈值（如0.1mm）

    返回:
    - 公平技巧评分EST评分
    """
    hits, misses, falsealarms, correctnegatives = calc_confusion_matrix(pred_x, y_label, threshold)
    return hits / (hits + misses + falsealarms) if (hits + misses + falsealarms) > 0 else 1.0


def FAR_score(pred_x, y_label, threshold=0.1):
    """
    计算二分类降水虚警率（FAR）

    参数:
    - pred_x: 预测值 (1D 或 ND torch.Tensor)
    - y_label: 真实标签 (与pred_x同shape的torch.Tensor)
    - threshold: 判定降水的阈值（如0.1mm）

    返回:
    - 空报率FAR评分
    """
    hits, misses, falsealarms, correctnegatives = calc_confusion_matrix(pred_x, y_label, threshold)
    return falsealarms / (hits + falsealarms) if (hits + falsealarms) > 0 else 1.0


def MAR_score(pred_x, y_label, threshold=0.1):
    """
    计算二分类降水漏报率（MAR）

    参数:
    - pred_x: 预测值 (1D 或 ND torch.Tensor)
    - y_label: 真实标签 (与pred_x同shape的torch.Tensor)
    - threshold: 判定降水的阈值（如0.1mm）

    返回:
    - 漏报率MAR评分
    """
    hits, misses, falsealarms, correctnegatives = calc_confusion_matrix(pred_x, y_label, threshold)
    return misses / (hits + misses) if (hits + misses) > 0 else 1.0


def POD_score(pred_x, y_label, threshold=0.1):
    """
    计算二分类降水命中率（POD）

    参数:
    - pred_x: 预测值 (1D 或 ND torch.Tensor)
    - y_label: 真实标签 (与pred_x同shape的torch.Tensor)
    - threshold: 判定降水的阈值（如0.1mm）

    返回:
    - 命中率POD评分
    """
    hits, misses, falsealarms, correctnegatives = calc_confusion_matrix(pred_x, y_label, threshold)
    return hits / (hits + misses) if (hits + misses) > 0 else 1.0


def Bias_score(pred_x, y_label, threshold=0.1):
    """计算二分类降水偏差率（Bias），确保结果更稳定"""
    hits, misses, falsealarms, correctnegatives = calc_confusion_matrix(pred_x, y_label, threshold)

    # 当分母为0时返回1.0而不是0.0
    if (hits + misses) == 0:
        # 检查是否有预测的降水事件
        if (hits + falsealarms) > 0:
            return float("inf")  # 有预测但无实际事件，设为无穷大
        else:
            return 1.0  # 既无预测也无实际事件，默认为1.0

    return (hits + falsealarms) / (hits + misses)


def calc_confusion_matrix(pred_x, y_label, threshold=0.1):
    """
    计算二分类降水混淆矩阵

    返回: hits, misses, falsealarms, correctnegatives
    """
    # 转为一维
    pred_x = pred_x.reshape(-1)
    y_label = y_label.reshape(-1)

    # 二值化
    pred_bin = (pred_x >= threshold).int()
    label_bin = (y_label >= threshold).int()

    hits = int(((pred_bin == 1) & (label_bin == 1)).sum().item())  # TP
    misses = int(((pred_bin == 0) & (label_bin == 1)).sum().item())  # FN
    falsealarms = int(((pred_bin == 1) & (label_bin == 0)).sum().item())  # FP
    correctnegatives = int(((pred_bin == 0) & (label_bin == 0)).sum().item())  # TN

    return hits, misses, falsealarms, correctnegatives


def null_accuracy_function(outputs, label, config, result=None):
    return None
