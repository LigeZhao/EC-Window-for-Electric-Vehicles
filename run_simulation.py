"""
车辆热管理 × 动力系统耦合仿真 — 服务器批量运行脚本
======================================================
用法示例：
  python run_simulation.py                              # 全量运行（读取 weather data 下所有 EPW）
  python run_simulation.py --epw-dir /data/epw          # 指定 EPW 目录
  python run_simulation.py --jobs 16                    # 指定并行进程数（默认 -1 = 全部核心）
  python run_simulation.py --output results/out         # 指定输出目录（Parquet 分区目录）
  python run_simulation.py --months 6 7 8               # 只跑指定月份
  python run_simulation.py --headings 0 180             # 只跑指定朝向
  python run_simulation.py --glass normal_glass tinted_glass  # 只跑指定玻璃方案
  python run_simulation.py --hour 12 --stat mean        # 气象统计口径（mean/clear_sky/p75）
  python run_simulation.py --resume                     # 跳过已存在的 Parquet 文件（断点续跑）

输出目录结构（Hive 分区格式）：
  {output}/
    ts/
      glass=normal_glass/
        city=Beijing/          ← EPW LOCATION header 中的城市名（与 notebook 一致）
          month=06/
            heading=0/
              data.parquet
    summary.parquet            ← 含 city（短名）和 city_key（EPW 文件名）两列

city 命名规则：
  - city     = EPW LOCATION 第一行的城市名（如 "Beijing"），用于 Parquet 路径和后处理
  - city_key = EPW 完整文件名（如 "CHN_Beijing.Beijing.545110_IWEC2"），保存在 summary 供溯源
  - 与 notebook 的 CITIES 关键字保持一致，确保本地小批次和服务器全量后处理逻辑完全匹配
"""

import os
import argparse
import time
import warnings
import multiprocessing as mp
import pandas as pd

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# Worker 级别缓存（每个子进程只初始化一次，后续任务直接复用）
# ──────────────────────────────────────────────
_worker_cache       = {}
_parquet_output_dir = None   # 由 _pool_initializer 注入
_resume_mode        = False  # 由 _pool_initializer 注入


def _pool_initializer(output_dir, resume):
    """子进程启动时由 Pool 调用，注入输出目录和续跑模式。"""
    global _parquet_output_dir, _resume_mode
    _parquet_output_dir = output_dir
    _resume_mode        = resume


def _ensure_worker_cache():
    """在 worker 进程首次执行任务时初始化缓存，之后直接复用。"""
    if _worker_cache:
        return
    import co_simulation as _co
    import thermal_functions as _TMS
    _worker_cache['cyc']       = _co.get_standard_cycle('cltc')
    _worker_cache['base_veh']  = _co.get_veh(23)
    _worker_cache['glass_lib'] = _TMS.load_glass_library()


# ──────────────────────────────────────────────
# 单次模拟函数（在子进程中执行）
# 每个任务写自己独立的 Parquet 文件，无需任何锁
# task_args: (city_key, city, city_name, glass_preset, month, heading,
#              lat, lon, temp, dni, dhi, time_of_day, day_of_year)
#   city_key  = EPW 文件名（内部溯源用）
#   city      = EPW LOCATION 城市短名（用于 Parquet 路径，与 notebook 一致）
# ──────────────────────────────────────────────
def run_single(task_args):
    (city_key, city, city_name, glass_preset, month, heading,
     lat, lon, temp, dni, dhi, time_of_day, day_of_year) = task_args
    try:
        # 构造输出路径：用城市短名，与 notebook 保持一致
        parquet_path = os.path.join(
            _parquet_output_dir, 'ts',
            f'glass={glass_preset}',
            f'city={city}',
            f'month={month:02d}',
            f'heading={heading}',
            'data.parquet'
        )

        # 断点续跑：文件已存在则跳过计算，直接返回 summary
        if _resume_mode and os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                driving_mi = df['distance_mi'].iloc[-1] * 100 / (df['soc'].iloc[0] - df['soc'].iloc[-1])
                driving_km = driving_mi * 1.6093
                return {
                    'glass':        glass_preset,
                    'city':         city,
                    'city_name':    city_name,
                    'city_key':     city_key,
                    'lat':          lat,
                    'lon':          lon,
                    'month':        month,
                    'heading':      heading,
                    'driving_km':   round(driving_km, 2),
                    'mean_sumPTMS': round(df['sumPTMS'].mean(), 1),
                    'mean_mrt':     round(df['mrt_val'].mean(), 2),
                    'mean_cabTair': round(df['cabTair'].mean(), 2),
                    'mean_solar':   round(df['solar_in'].mean(), 1),
                    'skipped':      True,
                }
            except Exception:
                pass  # 文件损坏则重新计算

        _ensure_worker_cache()
        import co_simulation

        glass_lib = _worker_cache['glass_lib']
        cyc       = _worker_cache['cyc']
        veh       = dict(_worker_cache['base_veh'])  # 浅拷贝，避免任务间互相污染

        veh['car']          = 'model3'
        veh['TMS']          = 'on'
        veh['n_passenger']  = 0
        veh['Tamb']         = 273.15 + temp
        veh['dni']          = dni
        veh['dhi']          = dhi
        veh['time_of_day']  = time_of_day
        veh['day_of_year']  = day_of_year
        veh['lat']          = lat
        veh['lon']          = lon
        veh['veh_heading']  = heading
        veh['glass_preset'] = glass_preset
        veh['glass_lib']    = glass_lib

        output = co_simulation.sim_drive(cyc, veh)

        keys_to_include = [
            'soc', 'distance_mi', 'operation_mode',
            'cabTair', 'mrt_val',
            'cabTwindshield', 'cabTrear', 'cabTside_left', 'cabTside_right',
            'cabTroof_ext', 'cabTroof_int',
            'cabTdoor_left_ext', 'cabTdoor_left_int',
            'cabTdoor_right_ext', 'cabTdoor_right_int',
            'cabTdashboard', 'cabTseats',
            'essT', 'mcT', 'T_airsupply',
            'solar_in', 'q_cab', 'q_bat', 'batHeat', 'q_motor',
            'm_dot', 'rfgNcomp',
            'cabmdotair', 'condmdotair', 'essmdotcool', 'mcmdotcool',
            'mcradmdotair', 'essradmdotair',
            'sumPTMS', 'rfgPcomp', 'cabPfanair', 'condPfanair',
            'esscoolPpump', 'mccoolPpump', 'mcradPfanair', 'essradPfanair', 'EER',
            'mcpin', 'mcpout', 'power',
            'e_cabin', 'i_term_cabin', 'e_battery', 'i_term_battery',
            'e_motor', 'i_term_motor',
        ]

        data = {key: output[key] for key in keys_to_include if key in output}
        df   = pd.DataFrame(data)
        df['soc'] = df['soc'] * 100 if df['soc'].max() <= 1.0 else df['soc']

        driving_mi = df['distance_mi'].iloc[-1] * 100 / (df['soc'].iloc[0] - df['soc'].iloc[-1])
        driving_km = driving_mi * 1.6093

        summary = {
            'glass':        glass_preset,
            'city':         city,       # 短名，与 notebook 一致，用于后处理
            'city_name':    city_name,  # EPW LOCATION 完整城市名
            'city_key':     city_key,   # EPW 文件名，供溯源
            'lat':          lat,
            'lon':          lon,
            'month':        month,
            'heading':      heading,
            'driving_km':   round(driving_km, 2),
            'mean_sumPTMS': round(df['sumPTMS'].mean(), 1),
            'mean_mrt':     round(df['mrt_val'].mean(), 2),
            'mean_cabTair': round(df['cabTair'].mean(), 2),
            'mean_solar':   round(df['solar_in'].mean(), 1),
            'skipped':      False,
        }

        # 每个任务写自己独立的 Parquet 文件，无锁，224 核全并行
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.to_parquet(parquet_path, index=False, compression='snappy')

        return summary

    except Exception as e:
        import traceback
        print(f'ERROR: {city} ({city_key}) {glass_preset} m{month} h{heading} — {e}')
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='EV热管理 × 动力系统耦合仿真批量运行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--epw-dir',  default='weather data all',
                        help='EPW 文件目录（支持子目录，默认：weather data all）')
    parser.add_argument('--output',   default='simulation result 0503/simulation_results',
                        help='输出目录（Parquet 分区目录）')
    parser.add_argument('--jobs',     type=int, default=-1,
                        help='并行进程数，-1 表示全部核心（默认）')
    parser.add_argument('--months',   type=int, nargs='+', default=list(range(1, 13)),
                        help='模拟月份列表（默认 1-12）')
    parser.add_argument('--headings', type=int, nargs='+', default=[0, 90, 180, 270],
                        help='车辆朝向列表，单位度（默认 0 90 180 270）')
    parser.add_argument('--glass',    nargs='+',
                        default=[
                            'normal_glass',
                            'tinted_glass',
                            'high_trans_glass',
                            'normal_glass_with_ec_trans',
                            'normal_glass_with_ec_colored',
                            # 反射式 EC（新增）
                            'normal_glass_with_ec_ref_trans',
                            'normal_glass_with_ec_ref_colored',
                        ],
                        help='玻璃方案列表（默认7种，含吸收式和反射式EC）')
    parser.add_argument('--hour',     type=int,   default=14,
                        help='代表时刻（默认 14 时）')
    parser.add_argument('--stat',     default='clear_sky',
                        choices=['mean', 'clear_sky', 'p75'],
                        help='气象统计口径（默认 clear_sky）')
    parser.add_argument('--ref-day',  type=int,   default=15,
                        help='月内参考日（默认 15 日）')
    parser.add_argument('--resume',   action='store_true',
                        help='断点续跑：跳过已存在的 Parquet 文件')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 加载气象数据（主进程，只做一次）──
    import weather_upload
    t0 = time.time()
    print(f'正在读取 EPW 文件：{args.epw_dir}')
    all_weather_dfs, all_locations = weather_upload.read_epw(args.epw_dir)
    city_keys = sorted(all_weather_dfs.keys())   # EPW 完整文件名列表
    print(f'EPW 站点数：{len(city_keys)}，玻璃方案：{len(args.glass)}，'
          f'月份：{len(args.months)}，朝向：{len(args.headings)}')

    # ── 2. 构建 city_key → city（短名）映射 ──
    # city_key = EPW 文件名（唯一），city = EPW LOCATION 第一行城市名（供后处理）
    # 例：CHN_Beijing.Beijing.545110_IWEC2  →  Beijing
    city_short = {k: all_locations[k]['city_name'] for k in city_keys}

    # ── 3. 主进程预计算天气标量（避免把大 DataFrame 传给每个 worker）──
    print('预计算气象代表值...')
    weather_cache = {}   # (city_key, month) -> (lat, lon, city, city_name, temp, dni, dhi, tod, doy)
    skipped_wx = 0
    for city_key in city_keys:
        for month in args.months:
            wx = weather_upload.get_weather_monthly_avg(
                city_key, month, args.hour,
                all_weather_dfs, all_locations,
                ref_day=args.ref_day, dni_threshold=50)
            if wx is None:
                skipped_wx += 1
                continue
            w = wx[args.stat]
            weather_cache[(city_key, month)] = (
                wx['latitude'], wx['longitude'],
                city_short[city_key],                  # city（短名，用于路径和后处理）
                all_locations[city_key]['city_name'],  # city_name（EPW 完整城市名）
                w['temp'], w['dni'], w['dhi'],
                wx['time_of_day'], wx['day_of_year'],
            )
    print(f'气象预计算完成（有效：{len(weather_cache)} 条，跳过：{skipped_wx} 条）')

    # ── 4. 生成任务列表（只含标量）──
    tasks = []
    for glass in args.glass:
        for month in args.months:
            for heading in args.headings:
                for city_key in city_keys:
                    if (city_key, month) not in weather_cache:
                        continue
                    lat, lon, city, city_name, temp, dni, dhi, tod, doy = weather_cache[(city_key, month)]
                    tasks.append((
                        city_key, city, city_name, glass, month, heading,
                        lat, lon, temp, dni, dhi, tod, doy
                    ))

    total = len(tasks)
    n_jobs = mp.cpu_count() if args.jobs == -1 else args.jobs
    print(f'总任务数: {total}')
    if args.resume:
        print('断点续跑模式：已存在的 Parquet 文件将被跳过')
    print(f'开始流式处理（n_jobs={n_jobs}）...\n')

    # ── 5. 多进程流式处理：每个 worker 独立写 Parquet，无锁全并行 ──
    summary_records = []
    write_errors    = 0
    skipped_count   = 0
    done_count      = 0

    with mp.Pool(processes=n_jobs,
                 initializer=_pool_initializer,
                 initargs=(output_dir, args.resume)) as pool:
        for summary in pool.imap_unordered(run_single, tasks):
            done_count += 1
            if summary is None:
                write_errors += 1
            else:
                if summary.pop('skipped', False):
                    skipped_count += 1
                summary_records.append(summary)

            if done_count % 100 == 0 or done_count == total:
                elapsed = time.time() - t0
                print(f'  进度: {done_count}/{total}  '
                      f'成功: {len(summary_records)}  跳过: {skipped_count}  '
                      f'失败: {write_errors}  用时: {elapsed:.1f}s')

    # ── 6. 汇总表写为单独的 Parquet 文件 ──
    summary_path = os.path.join(output_dir, 'summary.parquet')
    pd.DataFrame(summary_records).to_parquet(summary_path, index=False, compression='snappy')

    elapsed = time.time() - t0
    print(f'\n全部完成！')
    print(f'成功: {len(summary_records)} 条，失败: {write_errors} 条')
    print(f'总耗时: {elapsed:.1f}s  ({elapsed/3600:.2f} h)')
    print(f'数据已保存至: {output_dir}')
    print(f'汇总表: {summary_path}')


if __name__ == '__main__':
    main()
