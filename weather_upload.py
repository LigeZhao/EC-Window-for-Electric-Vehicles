import pandas as pd
import os
from datetime import datetime

# EPW 数据行各字段的列索引（0-based）
# 完整格式参见 EnergyPlus 文档 "EnergyPlus Auxiliary Programs" Appendix A
_COL_YEAR  = 0
_COL_MONTH = 1
_COL_DAY   = 2
_COL_HOUR  = 3
_COL_TEMP  = 6   # Dry Bulb Temperature [°C]
_COL_DNI   = 13  # Direct Normal Radiation [Wh/m²]
_COL_DHI   = 14  # Diffuse Horizontal Radiation [Wh/m²]

# 按顺序尝试的文件编码（巴西 INMET 等文件使用 Latin-1）
_ENCODINGS = ['utf-8', 'latin-1', 'cp1252']


def _parse_epw(full_path):
    """
    直接解析 EPW 文本文件，只提取仿真所需字段，不依赖 pyepw。
    - 避免 pyepw 对风速、云量等无关字段的范围校验报错
    - 支持 UTF-8 / Latin-1 等多种编码
    返回 (location_dict, DataFrame)，失败时抛出异常。
    """
    lines = None
    for enc in _ENCODINGS:
        try:
            with open(full_path, 'r', encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        raise ValueError(f"无法解码文件（已尝试编码：{_ENCODINGS}）")

    # ── 第 1 行：地理信息 ──
    # 格式：LOCATION,City,State,Country,DataSource,WMO,Latitude,Longitude,Timezone,Elevation
    loc_parts = lines[0].strip().split(',')
    city_name = loc_parts[1].strip() if len(loc_parts) > 1 else ''
    try:
        latitude  = float(loc_parts[6])
        longitude = float(loc_parts[7])
    except (IndexError, ValueError):
        raise ValueError("无法解析地理坐标（LOCATION 行格式异常）")

    location = {
        'latitude':  latitude,
        'longitude': longitude,
        'city_name': city_name,
    }

    # ── 第 9 行起：逐小时数据（前 8 行为文件头）──
    records = []
    for line in lines[8:]:
        parts = line.strip().split(',')
        if len(parts) < 15:
            continue
        try:
            records.append({
                'Year':  2020,                        # 统一为 2020
                'Month': int(parts[_COL_MONTH]),
                'Day':   int(parts[_COL_DAY]),
                'Hour':  int(parts[_COL_HOUR]),
                'DryBulbTemperature':        float(parts[_COL_TEMP]),
                'DirectNormalRadiation':     float(parts[_COL_DNI]),
                'DiffuseHorizontalRadiation':float(parts[_COL_DHI]),
            })
        except (ValueError, IndexError):
            continue  # 跳过格式异常的单行，不影响整体

    if not records:
        raise ValueError("文件中未找到有效数据行")

    df = pd.DataFrame(records)
    # 删除闰年 2 月 29 日
    df = df[~((df['Month'] == 2) & (df['Day'] == 29))].reset_index(drop=True)

    return location, df


def read_epw(folder_path):
    weather_dfs   = {}
    location_info = {}

    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return {}, {}

    count   = 0
    skipped = 0

    # os.walk 递归扫描，支持按大洲/国家分级存储的子目录结构
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if not file_name.endswith('.epw'):
                continue

            full_path = os.path.join(root, file_name)
            city_key  = os.path.splitext(file_name)[0]

            try:
                location, df        = _parse_epw(full_path)
                location_info[city_key] = location
                weather_dfs[city_key]   = df
                count += 1
            except Exception as e:
                skipped += 1
                print(f"[跳过] {file_name}: {e}")

    print(f"共加载 {count} 个 EPW 文件（跳过 {skipped} 个）")
    return weather_dfs, location_info

def get_weather(city_keyword, month, day, hour, all_weather_dfs, all_locations):
    # --- A. 模糊匹配城市名称 ---
    target_city_key = None
    for city_name in all_weather_dfs.keys():
        if city_keyword in city_name:
            target_city_key = city_name
            break
    
    if target_city_key is None:
        print(f"未找到包含 '{city_keyword}' 的城市。")
        return

    # --- B. 获取地理信息 (静态) ---
    loc_info = all_locations[target_city_key]
    latitude = loc_info['latitude']
    longitude = loc_info['longitude']

    # --- C. 获取气象数据 (动态) ---
    df = all_weather_dfs[target_city_key]
    
    row = df[(df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour)]
    
    if row.empty:
        print("未找到指定时间的数据（请检查日期是否有效）。")
        return

    dry_bulb_temp = row['DryBulbTemperature'].values[0]
    dni = row['DirectNormalRadiation'].values[0]
    dhi = row['DiffuseHorizontalRadiation'].values[0]

    # --- D. 计算年积日 (Day of Year) ---
    date_obj = datetime(2020, month, day)
    day_of_year = date_obj.timetuple().tm_yday
    time_of_day = hour
    
    return latitude, longitude, time_of_day, day_of_year, dry_bulb_temp, dni, dhi

def get_weather_monthly_avg(city_keyword, month, hour, all_weather_dfs, all_locations, ref_day, dni_threshold):
    """
    获取指定月份、指定小时的气象统计代表值。
    
    参数：
        city_keyword  : 城市名关键字
        month         : 月份（1-12）
        hour          : 小时（1-24）
        ref_day       : 太阳角度参考日（默认15日）
        dni_threshold : 筛选晴天的DNI下限（W/m²），用于计算晴天均值
    
    返回：
        dict，包含 mean（全样本均值）和 clear_sky（晴天均值）两组数据
    """
    # 匹配城市
    target_city_key = None
    for city_name in all_weather_dfs.keys():
        if city_keyword in city_name:
            target_city_key = city_name
            break
    if target_city_key is None:
        print(f"未找到城市：{city_keyword}")
        return None

    loc_info   = all_locations[target_city_key]
    latitude   = loc_info['latitude']
    longitude  = loc_info['longitude']

    df = all_weather_dfs[target_city_key]

    # 筛选目标月份和小时
    mask = (df['Month'] == month) & (df['Hour'] == hour)
    subset = df[mask].copy()

    if subset.empty:
        print(f"无数据：month={month}, hour={hour}")
        return None

    # --- 全样本均值 ---
    mean_temp = subset['DryBulbTemperature'].mean()
    mean_dni  = subset['DirectNormalRadiation'].mean()
    mean_dhi  = subset['DiffuseHorizontalRadiation'].mean()

    # --- 晴天筛选均值（DNI > threshold） ---
    clear = subset[subset['DirectNormalRadiation'] > dni_threshold]
    if len(clear) > 0:
        clear_temp = clear['DryBulbTemperature'].mean()
        clear_dni  = clear['DirectNormalRadiation'].mean()
        clear_dhi  = clear['DiffuseHorizontalRadiation'].mean()
        clear_ratio = len(clear) / len(subset)
    else:
        clear_temp = mean_temp
        clear_dni  = mean_dni
        clear_dhi  = mean_dhi
        clear_ratio = 0.0

    # --- P75分位数（偏高辐射典型工况）---
    p75_dni  = subset['DirectNormalRadiation'].quantile(0.75)
    p75_dhi  = subset['DiffuseHorizontalRadiation'].quantile(0.75)
    p75_temp = subset['DryBulbTemperature'].quantile(0.75)

    # 太阳角度用参考日
    from datetime import datetime
    date_obj   = datetime(2020, month, ref_day)
    day_of_year = date_obj.timetuple().tm_yday
    time_of_day = hour

    return {
        'latitude':     latitude,
        'longitude':    longitude,
        'time_of_day':  time_of_day,
        'day_of_year':  day_of_year,
        'n_samples':    len(subset),
        'clear_ratio':  clear_ratio,
        # 三种统计口径
        'mean':      {'temp': mean_temp, 'dni': mean_dni,  'dhi': mean_dhi},
        'clear_sky': {'temp': clear_temp,'dni': clear_dni, 'dhi': clear_dhi},
        'p75':       {'temp': p75_temp,  'dni': p75_dni,   'dhi': p75_dhi},
    }