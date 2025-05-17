import logging
import os

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
    """计算相关系数，处理边缘情况以避免NaN，并记录异常情况"""
    pred_x = pred_x.reshape(-1)
    y_label = y_label.reshape(-1)

    # 检查输入张量是否包含NaN或Inf值
    if torch.isnan(pred_x).any() or torch.isinf(pred_x).any():
        nan_count = torch.isnan(pred_x).sum().item()
        inf_count = torch.isinf(pred_x).sum().item()
        logger.error(f"预测值包含异常值: NaN={nan_count}, Inf={inf_count}, 总数={pred_x.numel()}")

        # 保存错误样本用于分析
        save_error_sample(pred_x, y_label, "nan_inf_values")

        # 替换NaN和Inf为0以允许计算继续
        pred_x = torch.nan_to_num(pred_x, nan=0.0, posinf=0.0, neginf=0.0)

    pred_mean = pred_x.mean().item()
    label_mean = y_label.mean().item()

    try:
        pred_std = pred_x.std().item()
        label_std = y_label.std().item()

        # 检查计算结果是否为NaN
        if np.isnan(pred_std) or np.isnan(label_std):
            logger.error(f"标准差计算结果为NaN: pred_values={pred_x[:5].tolist()}, pred_mean={pred_mean}")
            save_error_sample(pred_x, y_label, "nan_std")
            pred_std = 0.0 if np.isnan(pred_std) else pred_std
            label_std = 0.0 if np.isnan(label_std) else label_std
    except Exception as e:
        logger.error(f"计算标准差时发生错误: {str(e)}")
        pred_std = 0.0
        label_std = 0.0
        save_error_sample(pred_x, y_label, "std_error")

    covariance = ((pred_x - pred_mean) * (y_label - label_mean)).mean().item()

    # 计算相关系数
    if pred_std > 0 and label_std > 0:
        correlation = covariance / (pred_std * label_std)
        return max(-1.0, min(1.0, correlation))
    else:
        # 如果标准差为0或NaN，返回0.0
        logger.warning(f"标准差为零或NaN: pred_std={pred_std}, target_std={label_std}，返回相关系数0.0")
        return 0.0


def save_error_sample(pred_x, y_label, error_type):
    """保存错误样本到文件，用于离线分析"""
    try:
        # 创建调试目录
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug_data")
        os.makedirs(debug_dir, exist_ok=True)

        # 生成唯一文件名
        import time
        import uuid

        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        debug_file = os.path.join(debug_dir, f"{error_type}_{timestamp}_{unique_id}.npz")

        # 转换为numpy并保存
        pred_np = pred_x.detach().cpu().numpy() if isinstance(pred_x, torch.Tensor) else pred_x
        label_np = y_label.detach().cpu().numpy() if isinstance(y_label, torch.Tensor) else y_label

        # 保存数据和一些统计信息
        np.savez(
            debug_file,
            pred_values=pred_np,
            label_values=label_np,
            pred_contains_nan=np.isnan(pred_np).any(),
            label_contains_nan=np.isnan(label_np).any(),
            pred_contains_inf=np.isinf(pred_np).any(),
            label_contains_inf=np.isinf(label_np).any(),
            pred_min=float(np.nanmin(pred_np)) if not np.all(np.isnan(pred_np)) else "all_nan",
            pred_max=float(np.nanmax(pred_np)) if not np.all(np.isnan(pred_np)) else "all_nan",
            label_min=float(np.nanmin(label_np)) if not np.all(np.isnan(label_np)) else "all_nan",
            label_max=float(np.nanmax(label_np)) if not np.all(np.isnan(label_np)) else "all_nan",
        )

        logger.info(f"已保存错误样本到 {debug_file}")
    except Exception as e:
        logger.error(f"保存错误样本时发生错误: {str(e)}")


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
