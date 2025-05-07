import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from collections import defaultdict

def extract_datetime_from_filename(filename):
    """从文件名中提取日期时间信息"""
    # 从类似SURF2022010108.dat的文件名提取时间
    basename = os.path.basename(filename)
    if basename.startswith('SURF') and basename.endswith('.dat'):
        date_str = basename[4:14]  # 提取2022010108部分
        try:
            return datetime.strptime(date_str, '%Y%m%d%H')
        except ValueError:
            print(f"无法从文件名 {basename} 解析时间")
    return None

def read_station_data(file_path):
    """读取单个站点数据文件中的数据"""
    try:
        with open(file_path, 'r') as f:
            header = f.readline().strip()
            if not header:
                return None
                
            header_fields = header.split(',')
            
            # 尝试确定字段索引
            field_indices = {}
            required_fields = ['stn', 'lat', 'lon', 'rain1h_qc']
            
            for field in required_fields:
                try:
                    field_indices[field] = header_fields.index(field)
                except ValueError:
                    # 如果找不到必要字段，返回None
                    print(f"文件 {file_path} 中缺少字段: {field}")
                    return None
            
            # 读取数据
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                values = line.split(',')
                if len(values) <= max(field_indices.values()):
                    continue  # 忽略无效行
                
                try:
                    record = {
                        'stn': str(values[field_indices['stn']]),  # 确保站点编号作为字符串标识符
                        'lat': float(values[field_indices['lat']]),
                        'lon': float(values[field_indices['lon']]),
                        'rain1h_qc': int(float(values[field_indices['rain1h_qc']]))
                    }
                    data.append(record)
                except (ValueError, IndexError) as e:
                    pass  # 忽略无法解析的行
            
            return data
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def merge_stations_to_netcdf(base_dir, output_file, station_types=None, start_time=None, end_time=None):
    """
    将多个站点数据文件夹下的数据合并到一个NetCDF文件
    
    参数:
    base_dir: str - 基础数据目录
    output_file: str - 输出NetCDF文件路径
    station_types: list - 站点类型列表，默认为['gjz', 'qyz', 'swz']
    start_time: datetime - 开始时间，默认为None（不限制）
    end_time: datetime - 结束时间，默认为None（不限制）
    """
    if station_types is None:
        station_types = ['gjz', 'qyz', 'swz']
    
    # 用于存储所有站点信息
    all_stations = {}  # 使用字典以站点ID为键存储唯一站点信息
    all_time_data = []  # 存储所有时间点的数据
    
    print("正在扫描文件...")
    
    # 先扫描所有文件以构建时间索引
    time_files_map = defaultdict(list)  # {datetime: [(station_type, file_path), ...]}
    
    for station_type in station_types:
        station_dir = os.path.join(base_dir, 'STATION', station_type)
        if not os.path.exists(station_dir):
            print(f"警告: 站点目录 {station_dir} 不存在!")
            continue
        
        # 递归扫描所有.dat文件
        for data_file in glob.glob(os.path.join(station_dir, '**', 'SURF*.dat'), recursive=True):
            dt = extract_datetime_from_filename(data_file)
            if dt:
                # 根据时间范围过滤
                if start_time and dt < start_time:
                    continue
                if end_time and dt > end_time:
                    continue
                time_files_map[dt].append((station_type, data_file))
    
    print(f"找到 {len(time_files_map)} 个唯一时间点")
    
    # 按时间顺序处理数据
    sorted_times = sorted(time_files_map.keys())
    
    print("开始处理数据...")
    processed_count = 0
    
    for dt in sorted_times:
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"正在处理时间点 {dt} ({processed_count}/{len(sorted_times)})")
        
        # 处理此时间点的所有站点数据
        time_data = {
            'time': dt,
            'stations': [],
            'latitudes': [],
            'longitudes': [],
            'rain1h_qc': []
        }
        
        for station_type, file_path in time_files_map[dt]:
            station_data = read_station_data(file_path)
            if not station_data:
                continue
                
            for record in station_data:
                # 将站点信息添加到全局字典
                station_id = record['stn']
                if station_id not in all_stations:
                    all_stations[station_id] = {
                        'id': station_id,
                        'lat': record['lat'],
                        'lon': record['lon']
                    }
                
                # 将此时间点的观测添加到列表
                time_data['stations'].append(station_id)
                time_data['latitudes'].append(record['lat'])
                time_data['longitudes'].append(record['lon'])
                time_data['rain1h_qc'].append(record['rain1h_qc'])
        
        all_time_data.append(time_data)
    
    print("所有数据读取完成，正在构建NetCDF数据集...")
    
    # 构建站点列表
    station_ids = sorted(all_stations.keys())
    station_lats = [all_stations[sid]['lat'] for sid in station_ids]
    station_lons = [all_stations[sid]['lon'] for sid in station_ids]
    
    # 构建时间列表
    times = [data['time'] for data in all_time_data]
    
    # 创建一个空的rain1h_qc数组，形状为 (time, station)
    n_times = len(times)
    n_stations = len(station_ids)
    rain1h_qc = np.full((n_times, n_stations), np.nan)  # 初始化为NaN
    
    # 填充rain1h_qc数组
    station_id_to_index = {sid: i for i, sid in enumerate(station_ids)}
    
    for t_idx, time_data in enumerate(all_time_data):
        for s_idx, station_id in enumerate(time_data['stations']):
            if station_id in station_id_to_index:
                station_index = station_id_to_index[station_id]
                rain1h_qc[t_idx, station_index] = time_data['rain1h_qc'][s_idx]
    
    # 创建xarray数据集
    ds = xr.Dataset(
        data_vars={
            'rain1h_qc': (['time', 'station'], rain1h_qc),
        },
        coords={
            'time': times,
            'station': station_ids,
            'lat': ('station', station_lats),
            'lon': ('station', station_lons),
        },
        attrs={
            'description': '合并的气象站点降水质量控制数据',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': base_dir
        }
    )
    
    # 设置变量属性
    ds.rain1h_qc.attrs = {
        'long_name': '一小时降水量质量控制标志',
        'units': '-',
        '_FillValue': np.nan
    }
    
    # 设置坐标属性
    ds.lat.attrs = {'units': 'degrees_north', 'long_name': '纬度'}
    ds.lon.attrs = {'units': 'degrees_east', 'long_name': '经度'}
    ds.station.attrs = {'long_name': '站点ID'}
    
    # 添加时间范围信息到属性中
    if start_time:
        ds.attrs['start_time'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
    if end_time:
        ds.attrs['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存为NetCDF文件
    print(f"正在保存NetCDF文件到 {output_file}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 保存数据集
    ds.to_netcdf(output_file)
    
    print(f"数据已成功保存到 {output_file}")
    print(f"数据集包含 {n_times} 个时间点和 {n_stations} 个站点")
    
    return ds

# 使用示例
if __name__ == "__main__":
    base_dir = "/mnt/h/DataSet/1-obsvation"  # 基础数据目录
    output_file = "/mnt/h/DataSet/station_precipitation_data.nc"  # NetCDF输出文件
    
    # 指定时间范围，例如只导出2022年1月的数据
    from datetime import datetime
    start_time = datetime(2022, 4, 2, 22)
    end_time = datetime(2023, 1, 1, 8)
    
    merge_stations_to_netcdf(base_dir, output_file, 
                            start_time=start_time, 
                            end_time=end_time)