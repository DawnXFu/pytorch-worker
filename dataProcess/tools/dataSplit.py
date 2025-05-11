import glob
import logging
import os
import time

import numpy as np
import xarray as xr
from tqdm import tqdm  # 添加进度条

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_batch_files_precision(
    input_dir, output_dir=None, target_dtype="float32", overwrite=False, file_pattern="*.nc"
):
    """
    将批次文件中的变量从float64转换为float32

    参数:
    input_dir: 包含批次文件的目录
    output_dir: 输出转换后文件的目录，如果为None且overwrite=True则覆盖原文件
    target_dtype: 目标数据类型，默认为'float32'
    overwrite: 是否覆盖原文件
    file_pattern: 文件匹配模式
    """
    start_time = time.time()

    # 确保输出目录存在
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    elif not overwrite:
        raise ValueError("必须指定output_dir或设置overwrite=True")

    # 获取所有批次文件
    batch_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    if not batch_files:
        raise ValueError(f"在 {input_dir} 中未找到符合 {file_pattern} 的文件")

    total_files = len(batch_files)
    logger.info(f"找到 {total_files} 个批次文件")
    logger.info(f"开始将浮点变量转换为 {target_dtype}...")

    # 统计信息
    converted_files = 0
    converted_vars = 0
    total_vars = 0
    total_size_before = 0
    total_size_after = 0

    # 遍历所有文件
    for file_path in tqdm(batch_files, desc="处理文件"):
        file_basename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_basename) if output_dir else file_path

        try:
            # 读取数据集
            ds = xr.open_dataset(file_path)
            file_size_before = os.path.getsize(file_path)
            total_size_before += file_size_before

            # 检查并转换变量
            encoding = {}
            has_conversion = False
            file_vars = 0
            file_converted = 0

            for var_name, da in ds.data_vars.items():
                total_vars += 1
                file_vars += 1

                # 检查是否为浮点类型且非目标类型
                if np.issubdtype(da.dtype, np.floating) and da.dtype != np.dtype(target_dtype):
                    logger.debug(f"转换变量 {var_name}: {da.dtype} -> {target_dtype}")
                    ds[var_name] = da.astype(target_dtype)
                    has_conversion = True
                    converted_vars += 1
                    file_converted += 1

                # 设置压缩编码
                encoding[var_name] = {
                    "zlib": True,
                    "complevel": 5,
                    "_FillValue": None if np.issubdtype(da.dtype, np.floating) else getattr(da, "_FillValue", None),
                }

            # 如果有变量被转换或强制覆盖，则保存文件
            if has_conversion or output_dir is not None:
                # 如果需要备份原文件
                if overwrite and output_dir is None and os.path.exists(file_path):
                    temp_backup = file_path + ".backup"
                    os.rename(file_path, temp_backup)

                # 保存转换后的文件
                ds.to_netcdf(output_path, encoding=encoding)
                converted_files += 1

                # 删除备份文件
                if overwrite and output_dir is None and os.path.exists(temp_backup):
                    os.remove(temp_backup)

                # 计算文件大小差异
                if os.path.exists(output_path):
                    file_size_after = os.path.getsize(output_path)
                    total_size_after += file_size_after
                    size_change = (file_size_after - file_size_before) / file_size_before * 100
                    logger.debug(
                        f"文件 {file_basename}: {file_converted}/{file_vars} 个变量已转换, 大小变化: {size_change:.1f}%"
                    )

            # 关闭数据集
            ds.close()

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")

    # 计算空间节省
    if total_size_before > 0:
        space_saving = (total_size_before - total_size_after) / total_size_before * 100
        space_saving_mb = (total_size_before - total_size_after) / (1024 * 1024)
    else:
        space_saving = 0
        space_saving_mb = 0

    # 计算总耗时
    total_duration = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"转换完成! 总耗时: {total_duration:.2f}秒")
    logger.info(f"处理文件: {total_files} 个")
    logger.info(f"转换文件: {converted_files} 个")
    logger.info(f"转换变量: {converted_vars}/{total_vars} 个 ({converted_vars/total_vars*100:.1f}%)")
    logger.info(f"空间节省: {space_saving:.1f}% ({space_saving_mb:.1f} MB)")
    logger.info("=" * 60)

    return {
        "total_files": total_files,
        "converted_files": converted_files,
        "total_vars": total_vars,
        "converted_vars": converted_vars,
        "space_saving_percent": space_saving,
        "space_saving_mb": space_saving_mb,
        "duration_seconds": total_duration,
    }


if __name__ == "__main__":
    # 批次文件目录
    input_dir = "/mnt/h/Merge/temp"

    # 输出目录，设为None表示原地覆盖
    output_dir = None  # 或者指定一个新目录如 "/mnt/h/Merge/temp_float32"

    # 转换批次文件精度
    convert_batch_files_precision(
        input_dir=input_dir,
        output_dir=output_dir,
        target_dtype="float32",  # 转换为float32
        overwrite=True,  # 覆盖原文件
        file_pattern="*.nc",  # 文件匹配模式
    )
