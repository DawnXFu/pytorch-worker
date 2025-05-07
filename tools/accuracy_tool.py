import logging

import torch

logger = logging.Logger(__name__)


def PrecipitionCorrection_accuracy_function(outputs, label, config, result):
    """
    计算降水预测评估指标

    参数:
    - outputs: 模型预测结果，形状为 [B,C,H,W]
    - label: 真实标签，形状为 [B,C,H,W]
    - config: 配置文件
    - result: 历史累计评估结果

    返回:
    - 更新后的评估结果，包含CC、RMSE、KGE、PBIAS和TS指标
    """
    if result is None:
        result = {"CC": 0.0, "RMSE": 0.0, "KGE": 0.0, "PBIAS": 0.0, "TS": 0.0, "count": 0}

    # 确保张量在同一设备上并分离梯度
    outputs = outputs.detach()
    label = label.detach()

    # 计算各项指标
    batch_size = outputs.size(0)

    # 初始化指标累加器
    cc_sum = 0.0
    rmse_sum = 0.0
    kge_sum = 0.0
    pbias_sum = 0.0
    ts_sum = 0.0  # 添加TS评分累加器

    # 获取降水阈值（默认0.1mm，可在配置中自定义）
    precip_threshold = config.getfloat("metrics", "precip_threshold", fallback=0.1)

    # 对每个样本单独计算指标
    for i in range(batch_size):
        # 提取单个样本并展平
        output = outputs[i].reshape(-1)  # 展平为一维
        target = label[i].reshape(-1)  # 展平为一维

        # 1. 计算相关系数 (CC)
        # CC = cov(x,y) / (std(x) * std(y))
        output_mean = output.mean()
        target_mean = target.mean()

        output_var = ((output - output_mean) ** 2).mean()
        target_var = ((target - target_mean) ** 2).mean()

        # 计算协方差
        cov = ((output - output_mean) * (target - target_mean)).mean()

        # 计算相关系数
        if output_var.sqrt() == 0 or target_var.sqrt() == 0:
            cc = torch.tensor(0.0, device=outputs.device)
        else:
            cc = cov / (output_var.sqrt() * target_var.sqrt())

        # 2. 计算均方根误差 (RMSE)
        mse = ((output - target) ** 2).mean()
        rmse = mse.sqrt()

        # 3. 计算百分比偏差 (PBIAS)
        if target.sum() == 0:
            pbias = torch.tensor(0.0, device=outputs.device)
        else:
            pbias = 100.0 * (output.sum() - target.sum()) / target.sum()

        # 4. 计算Kling-Gupta系数 (KGE)
        # 计算变异系数比 (alpha)
        output_std = output_var.sqrt()
        target_std = target_var.sqrt()

        if target_std == 0:
            alpha = torch.tensor(1.0, device=outputs.device)
        else:
            alpha = output_std / target_std

        # 计算均值比 (beta)
        if target_mean == 0:
            beta = torch.tensor(1.0, device=outputs.device)
        else:
            beta = output_mean / target_mean

        # 计算KGE
        kge = 1 - torch.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # 5. 计算威胁评分 (TS)
        # 首先构建二分类结果
        pred_binary = (output >= precip_threshold).float()
        target_binary = (target >= precip_threshold).float()

        # 计算混淆矩阵元素
        tp = torch.sum((pred_binary == 1) & (target_binary == 1)).float()
        fp = torch.sum((pred_binary == 1) & (target_binary == 0)).float()
        fn = torch.sum((pred_binary == 0) & (target_binary == 1)).float()

        # 计算TS (TS = TP / (TP + FP + FN))
        if tp + fp + fn == 0:
            ts = torch.tensor(1.0, device=outputs.device)  # 完美预测无降水情况
        else:
            ts = tp / (tp + fp + fn)

        # 累加指标
        cc_sum += cc.item()
        rmse_sum += rmse.item()
        kge_sum += kge.item()
        pbias_sum += pbias.item()
        ts_sum += ts.item()

    # 计算平均值
    cc_avg = cc_sum / batch_size
    rmse_avg = rmse_sum / batch_size
    kge_avg = kge_sum / batch_size
    pbias_avg = pbias_sum / batch_size
    ts_avg = ts_sum / batch_size  # 计算平均TS评分

    # 更新结果字典
    result["CC"] += cc_avg
    result["RMSE"] += rmse_avg
    result["KGE"] += kge_avg
    result["PBIAS"] += pbias_avg
    result["TS"] += ts_avg  # 添加TS评分
    result["count"] += 1

    return result


def gen_precipition_result(result):
    """
    计算降水预测评估指标的平均值

    参数:
    - result: 包含CC、RMSE、KGE、PBIAS和TS指标的字典

    返回:
    - 平均评估结果，包含CC、RMSE、KGE、PBIAS和TS指标
    """
    if result["count"] == 0:
        return {"CC": 0.0, "RMSE": 0.0, "KGE": 0.0, "PBIAS": 0.0, "TS": 0.0}

    return {
        "CC": round(result["CC"] / result["count"], 3),
        "RMSE": round(result["RMSE"] / result["count"], 3),
        "KGE": round(result["KGE"] / result["count"], 3),
        "PBIAS": round(result["PBIAS"] / result["count"], 3),
        "TS": round(result["TS"] / result["count"], 3),  # 添加TS评分的输出
    }


def get_prf(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_micro_macro_result(res):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_prf(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3),
    }


def null_accuracy_function(outputs, label, config, result=None):
    return None


def single_label_top1_accuracy(outputs, label, config, result=None):
    if result is None:
        result = []
    id1 = torch.max(outputs, dim=1)[1]
    # id2 = torch.max(label, dim=1)[1]
    id2 = label
    nr_classes = outputs.size(1)
    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    for a in range(0, len(id1)):
        # if len(result) < a:
        #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        it_is = int(id1[a])
        should_be = int(id2[a])
        if it_is == should_be:
            result[it_is]["TP"] += 1
        else:
            result[it_is]["FP"] += 1
            result[should_be]["FN"] += 1

    return result


def multi_label_accuracy(outputs, label, config, result=None):
    if len(label[0]) != len(outputs[0]):
        raise ValueError("Input dimensions of labels and outputs must match.")

    outputs = outputs.data
    labels = label.data

    if result is None:
        result = []

    total = 0
    nr_classes = outputs.size(1)

    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

    for i in range(nr_classes):
        outputs1 = (outputs[:, i] >= 0.5).long()
        labels1 = (labels[:, i].float() >= 0.5).long()
        total += int((labels1 * outputs1).sum())
        total += int(((1 - labels1) * (1 - outputs1)).sum())

        if result is None:
            continue

        # if len(result) < i:
        #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        result[i]["TP"] += int((labels1 * outputs1).sum())
        result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

    return result
