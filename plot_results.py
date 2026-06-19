"""
plot_results.py — 智能窗玻璃 EV 热管理仿真结果可视化（服务器版）
=================================================================
用法：
  python plot_results.py                          # 读取默认目录
  python plot_results.py --output-dir /path/to/results

功能：
  读取服务器全量仿真 Parquet 结果，生成全套论文级别图片。

兼容性：
  - 自动检测旧格式（city 列 = EPW 完整文件名）并转换为城市短名
  - city_coords 自动从 df_summary 提取，无需手动维护坐标字典
  - 陆地掩膜自动缓存，避免重复计算
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # 服务器无 GUI，强制使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# 0. 命令行参数
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='EV 热管理仿真结果可视化')
parser.add_argument('--output-dir', default='simulation result 0503/simulation_results',
                    help='Parquet 结果目录（默认：simulation result 0503/simulation_results）')
parser.add_argument('--epw-dir', default='weather data all',
                    help='EPW 气象文件目录（用于补充 Tamb，默认：weather data）')
parser.add_argument('--fig-dir', default='simulation result 0503',
                    help='图片输出目录（默认：simulation result 0503）')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
EPW_DIR    = args.epw_dir
FIG_DIR    = args.fig_dir
os.makedirs(FIG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. 全局字体设置（论文级别）
# ──────────────────────────────────────────────────────────────────────
# 优先尝试 Arial/Helvetica，服务器没有时自动回退到 DejaVu Sans（无报错）
import matplotlib.font_manager as _fm
import math
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
_preferred = ['Arial', 'Helvetica', 'Liberation Sans', 'FreeSans', 'DejaVu Sans']
_font_family = next((f for f in _preferred if f in _available_fonts), 'sans-serif')

plt.rcParams.update({
    'font.family':        _font_family,
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     8.5,
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    7.5,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

# ──────────────────────────────────────────────────────────────────────
# 2. 读取数据 + 兼容性修复
# ──────────────────────────────────────────────────────────────────────
print('读取汇总表...')
summary_path = os.path.join(OUTPUT_DIR, 'summary.parquet')
df_summary   = pd.read_parquet(summary_path)

# ── 旧格式兼容：city 列若为完整 EPW 文件名（含下划线），转换为城市短名 ──
# 旧格式示例：CHN_Beijing.Beijing.545110_IWEC2
# 新格式示例：Beijing
if df_summary['city'].str.contains('_').mean() > 0.5:
    if 'city_name' in df_summary.columns:
        print('[兼容] 检测到旧格式 city 列（EPW 完整文件名），已替换为 city_name')
        df_summary['city'] = df_summary['city_name']
    else:
        # 没有 city_name 列时，从 EPW 文件名中提取首段城市名
        # CHN_Beijing.Beijing.545110_IWEC2  →  Beijing
        def _extract_city(full_name):
            try:
                return full_name.split('.')[1]
            except Exception:
                return full_name
        print('[兼容] 检测到旧格式 city 列，从文件名提取城市名')
        df_summary['city'] = df_summary['city'].apply(_extract_city)

print(f'汇总表：{len(df_summary)} 条  |  城市：{df_summary["city"].nunique()}  |  '
      f'玻璃：{sorted(df_summary["glass"].unique())}')

# ── city → city_key 映射：ts/ 目录用原始 EPW 文件名（旧格式服务器结果）──
# 旧格式：ts/city=CHN_Beijing.Beijing.545110_IWEC2/...
# 兼容性修复后 df_summary['city'] = 'BEIJING'，但磁盘目录仍用原始 city_key
if 'city_key' in df_summary.columns:
    _city_to_key = (df_summary.drop_duplicates('city')
                    .set_index('city')['city_key'].to_dict())
    print(f'city→key 映射已建立（示例：{list(_city_to_key.items())[:2]}）')
else:
    _city_to_key = {}
    print('[提示] summary 中无 city_key 列，read_ts 直接使用 city 名')

# ── 读取时序 Parquet 的工具函数 ──────────────────────────────────────
def read_ts(glass, city, month, heading, output_dir=OUTPUT_DIR):
    """读取指定条件的时序 DataFrame。
    优先用 city_key（旧格式EPW文件名），找不到时自动回退到归一化城市名或大写。
    解决吸收式EC（旧格式目录）与反射式EC（新格式目录）命名不一致的问题。
    """
    city_path = _city_to_key.get(city, city)
    path = os.path.join(output_dir, 'ts',
                        f'glass={glass}', f'city={city_path}',
                        f'month={month:02d}', f'heading={heading}',
                        'data.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    # 回退1：直接用归一化城市名
    if city_path != city:
        path2 = os.path.join(output_dir, 'ts',
                             f'glass={glass}', f'city={city}',
                             f'month={month:02d}', f'heading={heading}',
                             'data.parquet')
        if os.path.exists(path2):
            return pd.read_parquet(path2)
    # 回退2：大写城市名
    city_upper = city.upper()
    if city_upper not in (city_path, city):
        path3 = os.path.join(output_dir, 'ts',
                             f'glass={glass}', f'city={city_upper}',
                             f'month={month:02d}', f'heading={heading}',
                             'data.parquet')
        if os.path.exists(path3):
            return pd.read_parquet(path3)
    return pd.read_parquet(path)   # 抛出原始路径错误

# ── 四朝向平均月度摘要 ───────────────────────────────────────────────
df_avg = df_summary.groupby(['glass', 'city', 'month'])[
    ['driving_km', 'mean_sumPTMS', 'mean_mrt', 'mean_cabTair', 'mean_solar']
].mean().reset_index()

# ── EC 最优方案（按月、按城市选续航最优的 EC 状态）──────────────────
EC_PAIR_ABS = ['normal_glass_with_ec_trans',     'normal_glass_with_ec_colored']
EC_PAIR_REF = ['normal_glass_with_ec_ref_trans', 'normal_glass_with_ec_ref_colored']

def build_ec_optimal(df_avg, ec_pair_list):
    """
    对给定的 EC 方案列表，按城市×月份选续航最优状态，返回 df_ec。
    ec_pair_list: 两个 glass key（透明态在前，着色态在后）
    """
    trans_key, colored_key = ec_pair_list
    records = []
    for city in df_avg['city'].unique():
        for month in sorted(df_avg['month'].unique()):
            rows = {}
            for g in ec_pair_list:
                sub = df_avg[(df_avg['glass']==g) & (df_avg['city']==city) & (df_avg['month']==month)]
                if not sub.empty:
                    rows[g] = sub.iloc[0]
            if not rows:
                continue
            best = max(rows, key=lambda g: rows[g]['driving_km'])
            r = rows[best]
            records.append({
                'city':         city,
                'month':        month,
                'ec_choice':    'EC_trans' if best == trans_key else 'EC_colored',
                'driving_km':   r['driving_km'],
                'mean_sumPTMS': r['mean_sumPTMS'],
                'mean_mrt':     r['mean_mrt'],
                'mean_cabTair': r['mean_cabTair'],
                'mean_solar':   r['mean_solar'],
            })
    return pd.DataFrame(records)

df_ec_abs = build_ec_optimal(df_avg, EC_PAIR_ABS)
df_ec_ref = build_ec_optimal(df_avg, EC_PAIR_REF)
df_ec     = df_ec_abs   # 向后兼容别名（吸收式 EC）
print(f'df_avg: {len(df_avg)} 条  |  df_ec_abs: {len(df_ec_abs)} 条  |  df_ec_ref: {len(df_ec_ref)} 条')

# ── 补充 Tamb ────────────────────────────────────────────────────────
print('补充 Tamb...')
try:
    import weather_upload as _wu
    _all_wx, _all_loc = _wu.read_epw(EPW_DIR)

    def _get_monthly_tamb(city_name, month):
        # 大小写不敏感匹配：BEIJING 能匹配 EPW key 中的 Beijing
        city_lower = city_name.lower()
        for k in _all_wx:
            if city_lower in k.lower():
                df_w = _all_wx[k]
                sub  = df_w[df_w['Month'] == month]
                return float(sub['DryBulbTemperature'].mean()) if not sub.empty else np.nan
        return np.nan

    _cities   = df_summary['city'].unique()
    _months   = sorted(df_summary['month'].unique())
    _tamb_map = {}
    for _c in _cities:
        for _m in _months:
            _tamb_map[(_c, _m)] = _get_monthly_tamb(_c, _m)

    df_summary['tamb'] = df_summary.apply(
        lambda r: _tamb_map.get((r['city'], r['month']), np.nan), axis=1)
    df_avg['tamb'] = df_avg.apply(
        lambda r: _tamb_map.get((r['city'], r['month']), np.nan), axis=1)
    df_ec_abs['tamb'] = df_ec_abs.apply(
        lambda r: _tamb_map.get((r['city'], r['month']), np.nan), axis=1)
    df_ec_ref['tamb'] = df_ec_ref.apply(
        lambda r: _tamb_map.get((r['city'], r['month']), np.nan), axis=1)
    df_ec = df_ec_abs   # 同步别名
    print(f'Tamb 补充完成（NaN数: {df_summary["tamb"].isna().sum()}）')
except Exception as _e:
    print(f'[警告] Tamb 补充失败（{_e}），图6将跳过室外温度轴')
    df_summary['tamb'] = np.nan
    df_avg['tamb']     = np.nan
    df_ec_abs['tamb']  = np.nan
    df_ec_ref['tamb']  = np.nan
    df_ec = df_ec_abs

# ── 从 df_summary 自动构建 city_coords（全量城市经纬度）────────────
# 服务器全量数据直接从 summary 提取，无需维护手工坐标字典
_coord_df = df_summary.groupby('city')[['lat', 'lon']].mean()
city_coords = {city: (row['lon'], row['lat'])
               for city, row in _coord_df.iterrows()}
print(f'城市坐标字典：{len(city_coords)} 个城市')

# ── 城市名模糊匹配工具（大小写不敏感 + 部分匹配）────────────────────
# 城市名可能是全大写（BEIJING）或首字母大写（Beijing），用此函数统一查找
def _find_city(name, available):
    """
    在 available 中查找匹配 name 的城市，忽略大小写和斜杠/空格差异。
    优先精确匹配，其次部分匹配（name 是城市名前缀）。
    返回匹配到的实际城市名，未找到返回 None。
    """
    name_up = name.upper().replace(' ', '_').replace('-', '_')
    for c in available:
        c_up = c.upper().replace(' ', '_').replace('-', '_')
        if c_up == name_up:
            return c
    # 部分匹配：name 完整包含在城市名中（如 BEIJING 在 BEIJING/X）
    for c in available:
        c_up = c.upper().replace(' ', '_').replace('-', '_')
        if name_up in c_up or c_up.startswith(name_up):
            return c
    return None


def _resolve_cities_sel(cfg, available):
    """将 CITIES_SEL_CFG 中的城市名解析为实际数据中的城市名。"""
    result = {}
    for label, name in cfg.items():
        resolved = _find_city(name, available)
        if resolved:
            result[label] = resolved
        else:
            print(f'[警告] 城市 "{name}" 未找到，跳过')
    return result


# ──────────────────────────────────────────────────────────────────────
# 3. 全局变量 / 常量
# ──────────────────────────────────────────────────────────────────────
glass_base = ['normal_glass', 'tinted_glass', 'high_trans_glass']

GLASS_STYLE = {
    'normal_glass':     {'color': '#2166AC', 'marker': 'o', 'label': 'Normal glass'},
    'tinted_glass':     {'color': '#D6604D', 'marker': 's', 'label': 'Tinted glass'},
    'high_trans_glass': {'color': '#4DAC26', 'marker': '^', 'label': 'High-trans glass'},
    'ec_abs_optimal':   {'color': '#762A83', 'marker': 'D', 'label': 'EC absorptive'},
    'ec_ref_optimal':   {'color': '#E08214', 'marker': 'P', 'label': 'EC reflective'},
}
glass_keys = list(GLASS_STYLE.keys())

# 代表城市（图1/2/5/6/7 使用，仅展示典型气候带）
CITIES_SEL_CFG = {
    'Tropical\n(Singapore)': 'Singapore',
    'Arid\n(Cairo)':         'Cairo',
    'Temperate\n(Beijing)':  'Beijing',
    'Cold\n(Helsinki)':      'Helsinki',
}

# ──────────────────────────────────────────────────────────────────────
# 4. 世界地图插值：陆地掩膜（全局缓存）
# ──────────────────────────────────────────────────────────────────────
GRID_RES = 0.5
lon_grid = np.arange(-180, 180 + GRID_RES, GRID_RES)
lat_grid = np.arange(-90,  90  + GRID_RES, GRID_RES)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

_land_mask = None   # 延迟初始化，首次使用地图函数时生成

def _ensure_land_mask():
    global _land_mask
    if _land_mask is not None:
        return _land_mask
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.ops import unary_union
    print('正在生成陆地掩膜（首次约20秒，后续直接复用）...')
    shpfilename = shpreader.natural_earth(
        resolution='110m', category='physical', name='land')
    reader    = shpreader.Reader(shpfilename)
    land_geom = unary_union([g.geometry for g in reader.records()])
    _land_mask = np.array([
        land_geom.contains(Point(lo, la))
        for lo, la in zip(lon_mesh.ravel(), lat_mesh.ravel())
    ]).reshape(lon_mesh.shape)
    print('陆地掩膜生成完成。')
    return _land_mask


# ──────────────────────────────────────────────────────────────────────
# 5. 图1a：月度分面图 — TMS 功耗柱状图
# 5b. 图1b：月度分面图 — MRT 折线图
# 两张图分别保存，与 notebook 保持一致
# ──────────────────────────────────────────────────────────────────────
def plot_fig1():
    available  = df_avg['city'].unique()
    CITIES_SEL = _resolve_cities_sel(CITIES_SEL_CFG, available)
    if not CITIES_SEL:
        print(f'[跳过] 图1：代表城市均不在数据中，可用: {list(available[:10])}')
        return

    n_cities     = len(CITIES_SEL)
    N_COLS       = 4
    n_cols       = min(N_COLS, n_cities)
    n_rows_main  = int(np.ceil(n_cities / n_cols))

    months       = sorted(df_avg['month'].unique().tolist())
    n_months     = len(months)
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'][:n_months]
    x            = np.arange(n_months)

    N_G     = len(glass_keys)
    BAR_W   = 0.15
    offsets = (np.arange(N_G) - (N_G - 1) / 2) * BAR_W
    MRT_COMFORT = 25.0

    def get_monthly(city, metric):
        result = {}
        for g in glass_base:
            vals = []
            for m in months:
                sub = df_avg[(df_avg['glass']==g) & (df_avg['city']==city) & (df_avg['month']==m)]
                vals.append(float(sub[metric].iloc[0]) if not sub.empty else np.nan)
            result[g] = vals
        for ec_key, df_ec_src in [('ec_abs_optimal', df_ec_abs), ('ec_ref_optimal', df_ec_ref)]:
            ec_vals = []
            for m in months:
                row = df_ec_src[(df_ec_src['city']==city) & (df_ec_src['month']==m)]
                ec_vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            result[ec_key] = ec_vals
        return result

    def _make_facet_fig(n_rows, title_per_ax=True):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.6 * n_cols, 3.0 * n_rows + 1.2),
            gridspec_kw={'hspace': 0.42, 'wspace': 0.28,
                         'left': 0.08, 'right': 0.97, 'top': 0.92, 'bottom': 0.14}
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        return fig, axes

    def style_ax(ax, show_xlabels=False, hide_left=False):
        ax.set_xlim(-0.55, n_months - 0.45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(axis='both', length=3, width=0.6, direction='out')
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(month_labels if show_xlabels else [], fontsize=7.5)
        if hide_left:
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)

    handles_bar = [mpatches.Patch(facecolor=GLASS_STYLE[gk]['color'], edgecolor='white',
                                   linewidth=0.5, label=GLASS_STYLE[gk]['label'])
                   for gk in glass_keys]
    handles_mrt = [Line2D([0],[0], color=GLASS_STYLE[gk]['color'], linewidth=1.5,
                           marker=GLASS_STYLE[gk]['marker'], markersize=4,
                           markerfacecolor='white', markeredgewidth=0.8,
                           label=GLASS_STYLE[gk]['label'])
                   for gk in glass_keys]
    handles_mrt.append(Line2D([0],[0], color='#CC0000', linewidth=0.8,
                               linestyle='--', label=f'Comfort limit ({MRT_COMFORT}°C)'))

    # ── 图1a：TMS 功耗柱状图 ──
    fig_tms, axes_tms = _make_facet_fig(n_rows_main)
    for city_i, (climate_key, city) in enumerate(CITIES_SEL.items()):
        row_i = city_i // n_cols
        col_i = city_i %  n_cols
        hide  = (col_i > 0)
        ax    = axes_tms[row_i, col_i]
        ax.set_title(climate_key, fontsize=9, fontweight='bold', pad=6, linespacing=1.4)
        if col_i == 0:
            ax.set_ylabel('TMS power (W)', fontsize=8.5, labelpad=4)
        ptms = get_monthly(city, 'mean_sumPTMS')
        for gi, gk in enumerate(glass_keys):
            ax.bar(x + offsets[gi], ptms[gk], BAR_W,
                   color=GLASS_STYLE[gk]['color'], label=GLASS_STYLE[gk]['label'],
                   edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3)
        style_ax(ax, show_xlabels=True, hide_left=hide)
        all_v = [v for gk in glass_keys for v in ptms[gk] if not np.isnan(v)]
        if all_v:
            ax.set_ylim(0, max(all_v) * 1.13)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))

    for empty_i in range(n_cities, n_rows_main * n_cols):
        axes_tms[empty_i // n_cols, empty_i % n_cols].set_visible(False)

    fig_tms.legend(handles=handles_bar, loc='lower center', bbox_to_anchor=(0.5, 0.00),
                   ncol=min(len(handles_bar), 5), fontsize=7.5, frameon=True, framealpha=0.9,
                   edgecolor='#cccccc', borderpad=0.5, handlelength=1.4, columnspacing=1.0)

    path_tms = os.path.join(FIG_DIR, 'fig_facet_tms.png')
    fig_tms.savefig(path_tms, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_tms)
    print(f'Saved → {path_tms}')

    # ── 图1b：MRT 折线图 ──
    fig_mrt, axes_mrt = _make_facet_fig(n_rows_main)
    for city_i, (climate_key, city) in enumerate(CITIES_SEL.items()):
        row_i = city_i // n_cols
        col_i = city_i %  n_cols
        hide  = (col_i > 0)
        ax    = axes_mrt[row_i, col_i]
        ax.set_title(climate_key, fontsize=9, fontweight='bold', pad=6, linespacing=1.4)
        if col_i == 0:
            ax.set_ylabel('MRT (°C)', fontsize=8.5, labelpad=4)
        mrt = get_monthly(city, 'mean_mrt')
        for gk in glass_keys:
            st = GLASS_STYLE[gk]
            ax.plot(x, mrt[gk], color=st['color'], linewidth=1.5,
                    marker=st['marker'], markersize=3.8,
                    markerfacecolor='white', markeredgewidth=0.8, zorder=4)
        ax.axhline(MRT_COMFORT, color='#CC0000', linewidth=0.8, linestyle='--', alpha=0.75, zorder=3)
        if col_i == 0:
            ax.text(-0.5, MRT_COMFORT + 0.4, f'{MRT_COMFORT}°C comfort limit',
                    fontsize=6, color='#CC0000', va='bottom')
        style_ax(ax, show_xlabels=True, hide_left=hide)
        all_v = [v for gk in glass_keys for v in mrt[gk] if not np.isnan(v)]
        if all_v:
            ax.set_ylim(0, max(all_v) * 1.12)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))

    for empty_i in range(n_cities, n_rows_main * n_cols):
        axes_mrt[empty_i // n_cols, empty_i % n_cols].set_visible(False)

    fig_mrt.legend(handles=handles_mrt, loc='lower center', bbox_to_anchor=(0.5, 0.00),
                   ncol=min(len(handles_mrt), 6), fontsize=7.5, frameon=True, framealpha=0.9,
                   edgecolor='#cccccc', borderpad=0.5, handlelength=1.4, columnspacing=1.0)

    path_mrt = os.path.join(FIG_DIR, 'fig_facet_mrt.png')
    fig_mrt.savefig(path_mrt, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_mrt)
    print(f'Saved → {path_mrt}')


# ──────────────────────────────────────────────────────────────────────
# 6. 图2：年均综合对比
#    图2a/b：TMS 功耗绝对值 + 节能率（fig_annual_comparison.png）
#    图2c/d：续航里程绝对值 + 提升率（fig_annual_range_comparison.png）
# ──────────────────────────────────────────────────────────────────────
def plot_fig2():
    available  = df_avg['city'].unique()
    CITIES_SEL = _resolve_cities_sel(CITIES_SEL_CFG, available)
    if not CITIES_SEL:
        print(f'[跳过] 图2：代表城市均不在数据中')
        return

    city_list    = list(CITIES_SEL.values())
    climate_list = [k.replace('\n', ' ') for k in CITIES_SEL.keys()]
    n_c          = len(city_list)
    x_c          = np.arange(n_c)
    N_G          = len(glass_keys)
    BAR_W2       = 0.14
    off_all      = (np.arange(N_G) - (N_G - 1) / 2) * BAR_W2
    compare_keys = ['tinted_glass', 'high_trans_glass', 'ec_abs_optimal', 'ec_ref_optimal']
    off_cmp      = (np.arange(len(compare_keys)) - (len(compare_keys) - 1) / 2) * BAR_W2

    annual_city     = df_avg.groupby(['glass', 'city'])[['driving_km', 'mean_sumPTMS']].mean()
    ec_abs_ann_city = df_ec_abs.groupby('city')[['driving_km', 'mean_sumPTMS']].mean()
    ec_ref_ann_city = df_ec_ref.groupby('city')[['driving_km', 'mean_sumPTMS']].mean()

    def get_annual_mean(city, metric):
        result = {}
        for g in glass_base:
            try:    result[g] = annual_city.loc[(g, city), metric]
            except: result[g] = np.nan
        result['ec_abs_optimal']     = ec_abs_ann_city.loc[city, metric] if city in ec_abs_ann_city.index else np.nan
        result['ec_ref_optimal'] = ec_ref_ann_city.loc[city, metric] if city in ec_ref_ann_city.index else np.nan
        return result

    def _style_bar_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(x_c)
        ax.set_xticklabels(climate_list, fontsize=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.tick_params(axis='both', length=3, width=0.6)

    handles_all = [mpatches.Patch(facecolor=GLASS_STYLE[gk]['color'], edgecolor='white',
                                   linewidth=0.5, label=GLASS_STYLE[gk]['label'])
                   for gk in glass_keys]

    # ── 图2：2×2 布局（上：绝对值，下：提升率）─────────────────────────
    fig2, axes2 = plt.subplots(
        2, 2, figsize=(11, 8.0),
        gridspec_kw={'hspace': 0.42, 'wspace': 0.32,
                     'left': 0.08, 'right': 0.97,
                     'top': 0.92, 'bottom': 0.12})

    def _s2(ax, xl):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6); ax.spines['bottom'].set_linewidth(0.6)
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', zorder=0)
        ax.set_axisbelow(True); ax.tick_params(axis='both', length=3, width=0.6)
        ax.set_xticks(x_c); ax.set_xticklabels(xl, fontsize=8)

    # 上左：TMS功耗绝对值
    ax = axes2[0, 0]
    for gi, gk in enumerate(glass_keys):
        vals = [get_annual_mean(city, 'mean_sumPTMS')[gk] for city in city_list]
        ax.bar(x_c + off_all[gi], vals, BAR_W2, color=GLASS_STYLE[gk]['color'],
               edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3)
    ax.set_title('Annual mean TMS power by representative cities',
                 fontsize=9, fontweight='bold', pad=5)
    ax.set_ylabel('Mean TMS power (W)', fontsize=8.5)
    ax.set_ylim(0); ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _s2(ax, climate_list)

    # 上右：续航里程绝对值
    ax = axes2[0, 1]
    for gi, gk in enumerate(glass_keys):
        vals = [get_annual_mean(city, 'driving_km')[gk] for city in city_list]
        ax.bar(x_c + off_all[gi], vals, BAR_W2, color=GLASS_STYLE[gk]['color'],
               edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3)
    ax.set_title('Annual mean driving range by representative cities',
                 fontsize=9, fontweight='bold', pad=5)
    ax.set_ylabel('Driving range (km)', fontsize=8.5)
    ax.set_ylim(0); ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _s2(ax, climate_list)

    # 下左：TMS节能率
    ax = axes2[1, 0]
    for gi, gk in enumerate(compare_keys):
        savings = []
        for city in city_list:
            ann = get_annual_mean(city, 'mean_sumPTMS'); base = ann['normal_glass']
            savings.append((base - ann[gk]) / base * 100
                           if (base and not np.isnan(base) and not np.isnan(ann[gk])) else np.nan)
        ax.bar(x_c + off_cmp[gi], savings, BAR_W2, color=GLASS_STYLE[gk]['color'],
               edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3)
    ax.axhline(0, color='#333333', linewidth=0.8, zorder=4)
    ax.set_title('TMS power reduction vs. normal glass',
                 fontsize=9, fontweight='bold', pad=5)
    ax.set_ylabel('Energy saving (%)', fontsize=8.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _s2(ax, climate_list)

    # 下右：续航提升率
    ax = axes2[1, 1]
    for gi, gk in enumerate(compare_keys):
        gains = []
        for city in city_list:
            ann = get_annual_mean(city, 'driving_km'); base = ann['normal_glass']
            gains.append((ann[gk] - base) / base * 100
                         if (base and not np.isnan(base) and not np.isnan(ann[gk])) else np.nan)
        ax.bar(x_c + off_cmp[gi], gains, BAR_W2, color=GLASS_STYLE[gk]['color'],
               edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3)
    ax.axhline(0, color='#333333', linewidth=0.8, zorder=4)
    ax.set_title('Driving range improvement vs. normal glass',
                 fontsize=9, fontweight='bold', pad=5)
    ax.set_ylabel('Range gain (%)', fontsize=8.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _s2(ax, climate_list)

    fig2.legend(handles=handles_all, loc='lower center', bbox_to_anchor=(0.5, 0.00),
                ncol=min(len(handles_all), 5), fontsize=7.5, frameon=True, framealpha=0.9,
                edgecolor='#cccccc', borderpad=0.5, handlelength=1.4, columnspacing=1.0)

    path2 = os.path.join(FIG_DIR, 'fig_annual_comparison_2x2.png')
    fig2.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)

    ## ── CSV export: data_annual_comparison.csv ──────────────────────────
    _rows2 = []
    for gk in glass_keys:
        for city, clbl in zip(city_list, climate_list):
            ann_t = get_annual_mean(city, 'mean_sumPTMS')
            ann_r = get_annual_mean(city, 'driving_km')
            _rows2.append({'city': city, 'climate_label': clbl, 'glass': gk,
                           'annual_tms_W':    ann_t.get(gk, np.nan),
                           'annual_range_km': ann_r.get(gk, np.nan)})
    pd.DataFrame(_rows2).round(3).to_csv(
        os.path.join(FIG_DIR, 'data_annual_comparison.csv'), index=False)
    print('CSV → data_annual_comparison.csv')
    print(f'Saved → {path2}')


# ──────────────────────────────────────────────────────────────────────
# 7. 图3：Nature 风格连续热力世界地图（全量城市）
#    分别生成吸收式EC和反射式EC的地图面板
# ──────────────────────────────────────────────────────────────────────
def plot_fig3():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    import numpy.ma as ma

    # 只用有坐标的城市（全量数据 city_coords 来自 df_summary，覆盖全部城市）
    cities_map = list(city_coords.keys())
    cities_map = [c for c in cities_map if c in df_avg['city'].unique()]
    print(f'[图3] 使用 {len(cities_map)} 个城市进行插值')

    annual = df_avg.groupby(['glass', 'city'])[['driving_km', 'mean_sumPTMS']].mean()

    def _build_map_df(ec_ann):
        """根据给定的 EC 年均数据构建空间插值用 DataFrame。"""
        records = []
        for city in cities_map:
            def get(glass, metric, _city=city):
                try:    return annual.loc[(glass, _city), metric]
                except: return np.nan

            base_range = get('normal_glass', 'driving_km')
            base_ptms  = get('normal_glass', 'mean_sumPTMS')
            ec_range   = ec_ann.loc[city, 'driving_km']   if city in ec_ann.index else np.nan
            ec_ptms    = ec_ann.loc[city, 'mean_sumPTMS'] if city in ec_ann.index else np.nan
            tint_range = get('tinted_glass',    'driving_km')
            hite_range = get('high_trans_glass','driving_km')
            tint_ptms  = get('tinted_glass',    'mean_sumPTMS')
            hite_ptms  = get('high_trans_glass','mean_sumPTMS')

            best_static_range = max(tint_range, hite_range)
            best_static_ptms  = min(tint_ptms,  hite_ptms)

            dETR_range     = ec_range - base_range
            dETR_ptms      = base_ptms - ec_ptms
            dETR_range_pct = dETR_range / base_range * 100       if base_range else np.nan
            dETR_ptms_pct  = dETR_ptms  / base_ptms  * 100       if base_ptms  else np.nan
            dEn_range      = ec_range - best_static_range
            dEn_ptms       = best_static_ptms - ec_ptms
            dEn_range_pct  = dEn_range / best_static_range * 100 if best_static_range else np.nan
            dEn_ptms_pct   = dEn_ptms  / best_static_ptms  * 100 if best_static_ptms  else np.nan
            TRRI_raw       = max(0, dETR_ptms) * max(0, dEn_ptms)

            records.append({
                'city': city,
                'lon': city_coords[city][0], 'lat': city_coords[city][1],
                'dETR_range':     dETR_range,      'dETR_range_pct': dETR_range_pct,
                'dETR_ptms':      dETR_ptms,       'dETR_ptms_pct':  dETR_ptms_pct,
                'dEn_range':      dEn_range,       'dEn_range_pct':  dEn_range_pct,
                'dEn_ptms':       dEn_ptms,        'dEn_ptms_pct':   dEn_ptms_pct,
                'TRRI_raw':       TRRI_raw,
            })

        df = pd.DataFrame(records)
        trri_max = df['TRRI_raw'].max()
        df['TRRI'] = df['TRRI_raw'] / trri_max if trri_max > 0 else df['TRRI_raw']
        return df

    land_mask = _ensure_land_mask()
    MAX_INTERP_DIST = 8.0
    INTERP_METHOD   = 'linear'

    def _interp_and_mask(df_plot, col):
        """对指定列做插值+陆地+距离掩膜，返回 masked array。"""
        points = df_plot[['lon', 'lat']].values
        values = df_plot[col].values
        valid  = ~np.isnan(values)
        grid_v = griddata(points[valid], values[valid], (lon_mesh, lat_mesh), method=INTERP_METHOD)
        grid_v = np.where(land_mask, grid_v, np.nan)
        _tr = cKDTree(points[valid])
        _d, _ = _tr.query(np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), k=1)
        grid_v = np.where(_d.reshape(lon_mesh.shape) <= MAX_INTERP_DIST, grid_v, np.nan)
        return ma.masked_invalid(grid_v)

    def _draw_panel(df_plot, panel_configs, suptitle, save_path):
        """
        绘制 2×2 子图面板。
        panel_configs: list of (col, title, unit, cmap_name, diverging, label)
        """
        proj  = ccrs.Robinson()
        # top=0.89 为顶部总标题留出足够空间，避免与子图标签重叠
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 8),
            subplot_kw={'projection': proj},
            gridspec_kw={'hspace': 0.48, 'wspace': 0.06,
                         'top': 0.89, 'bottom': 0.06,
                         'left': 0.03, 'right': 0.97}
        )
        axes_flat = axes.ravel()

        for ax, (col, title, unit, cmap_name, diverging, label) in zip(axes_flat, panel_configs):
            grid_plot = _interp_and_mask(df_plot, col)

            vmin = float(np.nanmin(grid_plot))
            vmax = float(np.nanmax(grid_plot))
            cmap_ = plt.get_cmap(cmap_name, 256)
            cmap_.set_bad(color='none')

            if diverging:
                abs_max = max(abs(vmin), abs(vmax))
                norm_ = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            else:
                norm_ = mcolors.Normalize(vmin=max(0, vmin), vmax=vmax)

            ax.set_global()
            ax.add_feature(cfeature.OCEAN,     facecolor='#e8eef4', zorder=0)
            ax.add_feature(cfeature.LAND,      facecolor='#f2f0eb', zorder=1)
            pcm = ax.pcolormesh(lon_mesh, lat_mesh, grid_plot,
                                cmap=cmap_, norm=norm_,
                                transform=ccrs.PlateCarree(),
                                shading='auto', zorder=2, rasterized=True)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#555555', zorder=5)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor='#888888', zorder=5)
            ax.gridlines(draw_labels=False, linewidth=0.25, color='#aaaaaa',
                         alpha=0.4, linestyle='--', zorder=3)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#333333')
                spine.set_linewidth(0.5)

            # 面板标签（左上角），y=1.03 避免与总标题重叠
            ax.text(-0.02, 1.03, label, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='bottom', ha='left')
            # 子图标题
            ax.set_title(title, fontsize=8.5, fontweight='bold', pad=5)

            # Colorbar（水平，贴在子图底部）；自动刻度避免小值时全零
            cb = plt.colorbar(pcm, ax=ax, orientation='horizontal',
                              pad=0.04, fraction=0.046, aspect=38,
                              extend='both' if diverging else 'max')
            cb.set_label(unit, fontsize=7.5, labelpad=2)
            cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=3))
            cb.ax.tick_params(labelsize=6.5, length=2, width=0.4)

        if suptitle:
            # fig.text 在 top=0.89 上方放置总标题，与子图标签不重叠
            fig.text(0.5, 0.95, suptitle,
                     ha='center', va='top', fontsize=11, fontweight='bold')

        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'  Panel saved → {save_path}')

    def _draw_single_map(df_plot, col, title, unit, cmap_name, diverging, save_path, panel_label=''):
        """绘制单张 Robinson 世界地图（用于独立展示单个指标）。"""
        proj = ccrs.Robinson()
        fig, ax = plt.subplots(
            1, 1, figsize=(10, 5.5),
            subplot_kw={'projection': proj},
            gridspec_kw={'top': 0.92, 'bottom': 0.12,
                         'left': 0.03, 'right': 0.97}
        )
        grid_plot = _interp_and_mask(df_plot, col)
        vmin = float(np.nanmin(grid_plot))
        vmax = float(np.nanmax(grid_plot))
        cmap_ = plt.get_cmap(cmap_name, 256)
        cmap_.set_bad(color='none')
        if diverging:
            abs_max = max(abs(vmin), abs(vmax))
            norm_ = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        else:
            norm_ = mcolors.Normalize(vmin=max(0, vmin), vmax=vmax)
        ax.set_global()
        ax.add_feature(cfeature.OCEAN,     facecolor='#e8eef4', zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor='#f2f0eb', zorder=1)
        pcm = ax.pcolormesh(lon_mesh, lat_mesh, grid_plot,
                            cmap=cmap_, norm=norm_,
                            transform=ccrs.PlateCarree(),
                            shading='auto', zorder=2, rasterized=True)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#555555', zorder=5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor='#888888', zorder=5)
        ax.gridlines(draw_labels=False, linewidth=0.25, color='#aaaaaa',
                     alpha=0.4, linestyle='--', zorder=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(0.5)
        if panel_label:
            ax.text(-0.02, 1.03, panel_label, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='bottom', ha='left')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        cb = plt.colorbar(pcm, ax=ax, orientation='horizontal',
                          pad=0.04, fraction=0.046, aspect=45,
                          extend='both' if diverging else 'max')
        cb.set_label(unit, fontsize=8, labelpad=2)
        cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=3))
        cb.ax.tick_params(labelsize=7, length=2, width=0.4)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'  Panel saved → {save_path}')

    # ── 构建吸收式 EC 和反射式 EC 的空间数据 ────────────────────────
    ec_abs_ann = df_ec_abs.groupby('city')[['driving_km', 'mean_sumPTMS']].mean()
    ec_ref_ann = df_ec_ref.groupby('city')[['driving_km', 'mean_sumPTMS']].mean()

    df_plot_abs = _build_map_df(ec_abs_ann)
    df_plot_ref = _build_map_df(ec_ref_ann)

    ETRDETR_PANELS = [
        ('dETR_range',
         r'Range change: EC vs. Normal Glass',
         'Δ Range (km)', 'RdYlGn', True, 'a'),
        ('dETR_range_pct',
         r'Range change (%): EC vs. Normal Glass',
         'Δ Range (%)',  'RdYlGn', True, 'b'),
        ('dETR_ptms',
         r'TMS power change: EC vs. Normal Glass',
         'Δ TMS Power (W)', 'RdYlGn', True, 'c'),
        ('dETR_ptms_pct',
         r'TMS power change (%): EC vs. Normal Glass',
         'Δ TMS Power (%)', 'RdYlGn', True, 'd'),
    ]
    DEN_PANELS = [
        ('dEn_range',
         r'Range change: EC vs. Best Static Glass',
         'Δ Range (km)', 'RdYlGn', True, 'a'),
        ('dEn_range_pct',
         r'Range change (%): EC vs. Best Static Glass',
         'Δ Range (%)',  'RdYlGn', True, 'b'),
        ('dEn_ptms',
         r'TMS power change: EC vs. Best Static Glass',
         'Δ TMS Power (W)', 'RdYlGn', True, 'c'),
        ('dEn_ptms_pct',
         r'TMS power change (%): EC vs. Best Static Glass',
         'Δ TMS Power (%)', 'RdYlGn', True, 'd'),
    ]

    # ── 吸收式 EC 面板 ─────────────────────────────────────────────
    print('\n[图3a] 吸收式EC — ΔE_TR 面板...')
    _draw_panel(df_plot_abs, ETRDETR_PANELS,
                suptitle=r'Absorptive EC vs. Normal Glass',
                save_path=os.path.join(FIG_DIR, 'fig_worldmap_dETR_panel.png'))

    print('\n[图3b] 吸收式EC — ΔE_n 面板...')
    _draw_panel(df_plot_abs, DEN_PANELS,
                suptitle=r'Absorptive EC vs. Best Static Glass',
                save_path=os.path.join(FIG_DIR, 'fig_worldmap_dEn_panel.png'))
    # [已删除] fig_worldmap_TRRI.png

    # ── 反射式 EC 面板（新增）─────────────────────────────────────
    print('\n[图3d] 反射式EC — ΔE_TR 面板...')
    _draw_panel(df_plot_ref, ETRDETR_PANELS,
                suptitle=r'Reflective EC vs. Normal Glass',
                save_path=os.path.join(FIG_DIR, 'fig_worldmap_ref_dETR_panel.png'))

    print('\n[图3e] 反射式EC — ΔE_n 面板...')
    _draw_panel(df_plot_ref, DEN_PANELS,
                suptitle=r'Reflective EC vs. Best Static Glass',
                save_path=os.path.join(FIG_DIR, 'fig_worldmap_ref_dEn_panel.png'))
    # [已删除] fig_worldmap_ref_TRRI.png


# ──────────────────────────────────────────────────────────────────────
# 8. 图4：EC 切换地图（全量城市插值）
# ──────────────────────────────────────────────────────────────────────
    ## ── 新增：Ref vs Abs 差异图（续航%差 / TMS%差）──────────────────────
    from matplotlib.colors import LinearSegmentedColormap as _LSC3
    import cartopy.feature as _cf3
    from scipy.interpolate import griddata as _gd3
    from scipy.spatial import cKDTree as _cKD3
    import numpy.ma as _ma3

    cmap_diff3 = _LSC3.from_list(
        'diff_ref_abs',
        ['#D94801','#FDAE6B','#FFF5EB','#FFFFFF','#C7E9C0','#41AB5D','#00441B'],
        N=256)
    cmap_diff3.set_bad(color='none')

    def _draw_diff_map3(col, title, unit, save_path):
        # col: same column name in both df_plot_ref and df_plot_abs (no prefix)
        df_d = df_plot_abs[['city','lon','lat']].copy()
        df_d['diff'] = df_plot_ref[col].values - df_plot_abs[col].values
        pts    = df_d[['lon','lat']].values
        vals   = df_d['diff'].values
        valid  = ~np.isnan(vals)
        grid_v = _gd3(pts[valid], vals[valid], (lon_mesh, lat_mesh), method='linear')
        grid_v = np.where(land_mask, grid_v, np.nan)
        _tree  = _cKD3(pts[valid])
        _dist, _ = _tree.query(
            np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), k=1)
        grid_v = np.where(_dist.reshape(lon_mesh.shape) <= 8.0, grid_v, np.nan)
        abs_max = float(np.nanmax(np.abs(grid_v)))
        norm_d  = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        fig = plt.figure(figsize=(13, 5.5))
        ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(_cf3.OCEAN,     facecolor='#e8eef4', zorder=0)
        ax.add_feature(_cf3.LAND,      facecolor='#f2f0eb', zorder=1)
        pcm = ax.pcolormesh(lon_mesh, lat_mesh, _ma3.masked_invalid(grid_v),
                            cmap=cmap_diff3, norm=norm_d,
                            transform=ccrs.PlateCarree(),
                            shading='auto', zorder=2, rasterized=True)
        ax.add_feature(_cf3.COASTLINE, linewidth=0.35, edgecolor='#555555', zorder=5)
        ax.add_feature(_cf3.BORDERS,   linewidth=0.20, edgecolor='#888888', zorder=5)
        ax.gridlines(draw_labels=False, linewidth=0.25,
                     color='#aaaaaa', alpha=0.4, linestyle='--', zorder=3)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('#333333'); spine.set_linewidth(0.5)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        cb = plt.colorbar(pcm, ax=ax, orientation='horizontal',
                          pad=0.04, fraction=0.038, aspect=50, extend='both')
        cb.set_label(unit + '  (green = reflective better  |  orange = absorptive better)',
                     fontsize=8, labelpad=3)
        cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, min_n_ticks=3))
        cb.ax.tick_params(labelsize=7.5, length=2.5, width=0.5)
        plt.tight_layout(pad=0.5)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'Saved → {save_path}')

    _draw_diff_map3(
        'dETR_range_pct',
        'Driving range improvement: Reflective EC − Absorptive EC\n'
        'Percentage point difference vs. normal glass  (green = reflective better)',
        'Δ Range improvement (pp)',
        os.path.join(FIG_DIR, 'fig_worldmap_diff_range_pct.png'))

    _draw_diff_map3(
        'dETR_ptms_pct',
        'TMS power reduction: Reflective EC − Absorptive EC\n'
        'Percentage point difference vs. normal glass  (green = reflective better)',
        'Δ TMS reduction (pp)',
        os.path.join(FIG_DIR, 'fig_worldmap_diff_tms_pct.png'))


    ## ── CSV export: data_worldmap.csv ───────────────────────────────────
    _wm = df_plot_abs[['city','lon','lat',
                        'dETR_range','dETR_range_pct','dETR_ptms','dETR_ptms_pct',
                        'dEn_range','dEn_range_pct','dEn_ptms','dEn_ptms_pct',
                        'TRRI']].copy()
    _wm.columns = ['city','lon','lat',
                   'abs_dETR_range_km','abs_dETR_range_pct','abs_dETR_tms_W','abs_dETR_tms_pct',
                   'abs_dEn_range_km','abs_dEn_range_pct','abs_dEn_tms_W','abs_dEn_tms_pct',
                   'abs_TRRI']
    _wm_r = df_plot_ref[['city',
                          'dETR_range','dETR_range_pct','dETR_ptms','dETR_ptms_pct',
                          'dEn_range','dEn_range_pct','dEn_ptms','dEn_ptms_pct',
                          'TRRI']].copy()
    _wm_r.columns = ['city',
                     'ref_dETR_range_km','ref_dETR_range_pct','ref_dETR_tms_W','ref_dETR_tms_pct',
                     'ref_dEn_range_km','ref_dEn_range_pct','ref_dEn_tms_W','ref_dEn_tms_pct',
                     'ref_TRRI']
    _wm = _wm.merge(_wm_r, on='city', how='left')
    _wm['diff_range_pct'] = _wm['ref_dETR_range_pct'] - _wm['abs_dETR_range_pct']
    _wm['diff_tms_pct']   = _wm['ref_dETR_tms_pct']   - _wm['abs_dETR_tms_pct']
    _wm.round(4).to_csv(os.path.join(FIG_DIR, 'data_worldmap.csv'), index=False)
    print('CSV → data_worldmap.csv')
    print('[图3-diff] 差异图完成。')

def plot_fig4():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    import numpy.ma as ma
    from matplotlib.colors import LinearSegmentedColormap

    # 代表城市月份条标注（用模糊匹配找到实际城市名）
    _annotate_candidates = [
        'Singapore', 'Cairo', 'Beijing', 'Helsinki',
        'Moscow', 'Los.Angeles', 'Sydney', 'Mumbai',
    ]
    CITIES_ANNOTATE = [
        _find_city(c, df_avg['city'].unique())
        for c in _annotate_candidates
    ]
    CITIES_ANNOTATE = [c for c in CITIES_ANNOTATE if c is not None]
    COLOR_COLORED = '#6B4226'   # 深棕
    COLOR_TRANS   = '#D6EAF8'   # 浅蓝
    COLOR_NODATA  = '#E8E8E8'

    cities_map = [c for c in df_avg['city'].unique() if c in city_coords]

    records = []
    for city in cities_map:
        lon, lat = city_coords[city]
        monthly = {}
        for m in range(1, 13):
            row = df_ec[(df_ec['city'] == city) & (df_ec['month'] == m)]
            monthly[m] = None if row.empty else row.iloc[0]['ec_choice']
        n_colored = sum(1 for v in monthly.values() if v == 'EC_colored')
        records.append({'city': city, 'lon': lon, 'lat': lat,
                        'n_colored': n_colored, 'monthly': monthly})

    df_ec_map = pd.DataFrame(records)
    print(f'[图4] 城市数（有坐标）: {len(df_ec_map)}')

    land_mask = _ensure_land_mask()
    MAX_INTERP_DIST = 8.0

    pts   = df_ec_map[['lon', 'lat']].values
    vals  = df_ec_map['n_colored'].values.astype(float)
    valid = ~np.isnan(vals)

    grid_vals = griddata(pts[valid], vals[valid], (lon_mesh, lat_mesh), method='linear')
    grid_vals = np.where(land_mask, grid_vals, np.nan)
    _tree = cKDTree(pts[valid])
    _dist, _ = _tree.query(np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), k=1)
    grid_vals = np.where(_dist.reshape(lon_mesh.shape) <= MAX_INTERP_DIST, grid_vals, np.nan)
    grid_vals = np.clip(grid_vals, 0, 12)

    fig = plt.figure(figsize=(10, 5.2))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    ax.add_feature(cfeature.OCEAN, facecolor='#e8eef4', zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor='#f2f0eb', zorder=1)

    cmap_ec = LinearSegmentedColormap.from_list(
        'ec_colored_months',
        ['#FFFFFF', '#F5ECD7', '#C8A97E', '#8B6340', '#6B4226', '#3E1F0A'], N=256)
    cmap_ec.set_bad(color='none')
    cmap_ec.set_under(color='none')
    norm_ec = mcolors.Normalize(vmin=0, vmax=12)

    grid_plot = ma.masked_invalid(grid_vals)
    pcm = ax.pcolormesh(lon_mesh, lat_mesh, grid_plot, cmap=cmap_ec, norm=norm_ec,
                        transform=ccrs.PlateCarree(), shading='auto', zorder=2, rasterized=True)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor='#555555', zorder=5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.2,  edgecolor='#888888', zorder=5)

    MONTH_ABBR = ['J','F','M','A','M','J','J','A','S','O','N','D']
    # OFFSET key 用城市实际名（已通过 _find_city 解析为数据中的大写名）
    _offset_src = {
        'Singapore':   ( 0.08,  0.03, 'right'),
        'Cairo':       (-0.06,  0.03, 'left'),
        'Beijing':     ( 0.06,  0.03, 'right'),
        'Helsinki':    (-0.06,  0.03, 'left'),
        'Moscow':      (-0.07,  0.04, 'left'),
        'Los.Angeles': (-0.09, -0.04, 'left'),
        'Sydney':      ( 0.06, -0.04, 'right'),
        'Mumbai':      ( 0.06,  0.00, 'right'),
    }
    # 将 OFFSET key 也解析成实际城市名
    _all_av = df_avg['city'].unique()
    OFFSET = {_find_city(k, _all_av) or k: v for k, v in _offset_src.items()}

    cities_to_draw = [c for c in CITIES_ANNOTATE if c in df_ec_map['city'].values]

    for city in cities_to_draw:
        row   = df_ec_map[df_ec_map['city'] == city].iloc[0]
        lon_c, lat_c = row['lon'], row['lat']
        try:
            rob = ccrs.Robinson()
            xr, yr = rob.transform_point(lon_c, lat_c, ccrs.PlateCarree())
        except Exception:
            continue

        ROB_XMAX = 1.70e7
        ROB_YMAX = 0.86e7
        BAR_W_  = 0.085
        BAR_H_  = 0.018
        dx_frac, dy_frac, lside = OFFSET.get(city, (0, 0, 'right'))

        cx_frac = (xr / ROB_XMAX + 1) / 2
        cy_frac = (yr / ROB_YMAX + 1) / 2
        ix = cx_frac + dx_frac - (0 if lside == 'right' else BAR_W_)
        iy = cy_frac + dy_frac
        ix = np.clip(ix, 0.01, 1 - BAR_W_ - 0.01)
        iy = np.clip(iy, 0.01, 1 - BAR_H_ - 0.01)

        fig_bbox = ax.get_position()
        abs_x = fig_bbox.x0 + ix * fig_bbox.width
        abs_y = fig_bbox.y0 + iy * fig_bbox.height
        abs_w = BAR_W_ * fig_bbox.width
        abs_h = BAR_H_ * fig_bbox.height

        ax_bar = fig.add_axes([abs_x, abs_y, abs_w, abs_h])
        monthly = row['monthly']
        for mi, month_num in enumerate(range(1, 13)):
            choice = monthly.get(month_num, None)
            fc = COLOR_COLORED if choice == 'EC_colored' else (
                 COLOR_TRANS   if choice == 'EC_trans'   else COLOR_NODATA)
            rect = mpatches.FancyBboxPatch((mi + 0.05, 0.05), 0.88, 0.90,
                                           boxstyle='square,pad=0',
                                           facecolor=fc, edgecolor='white', linewidth=0.3)
            ax_bar.add_patch(rect)
        ax_bar.set_xlim(0, 12)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_aspect('auto')
        ax_bar.axis('off')

        dot_x  = fig_bbox.x0 + cx_frac * fig_bbox.width
        dot_y  = fig_bbox.y0 + cy_frac * fig_bbox.height
        bar_cx = abs_x + abs_w * (0 if lside == 'right' else 1)
        bar_cy = abs_y + abs_h / 2
        fig.add_artist(plt.Line2D([dot_x, bar_cx], [dot_y, bar_cy],
                                   transform=fig.transFigure,
                                   color='#555555', linewidth=0.5, linestyle='--', zorder=10))

        name_display = city.replace('.', ' ')
        tx = (abs_x + abs_w + 0.003) if lside == 'right' else (abs_x - 0.003)
        ha = 'left' if lside == 'right' else 'right'
        fig.text(tx, abs_y + abs_h / 2, name_display, transform=fig.transFigure,
                 fontsize=6.5, va='center', ha=ha, color='#222222', fontweight='bold', zorder=11)

        ax.plot(lon_c, lat_c, 'o', markersize=3, color='white',
                markeredgecolor='#333333', markeredgewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=8)

    cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.04,
                        fraction=0.038, aspect=45, extend='neither')
    cbar.set_label('Months per year in EC colored state', fontsize=8.5, labelpad=3)
    cbar.set_ticks([0, 3, 6, 9, 12])
    cbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.5)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_COLORED, edgecolor='#555', linewidth=0.5,
                       label='EC colored'),
        mpatches.Patch(facecolor=COLOR_TRANS,   edgecolor='#555', linewidth=0.5,
                       label='EC clear'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
              framealpha=0.85, edgecolor='#cccccc', borderpad=0.6,
              handlelength=1.2, handleheight=0.9)

    ax.set_title('EC Switching Map: Months in Dark State per Year',
                 fontsize=10, fontweight='bold', pad=6)
    ax.text(-0.02, 1.02, 'a', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    ax.gridlines(draw_labels=False, linewidth=0.3, color='#aaaaaa', alpha=0.5,
                 linestyle='--', zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(0.6)

    plt.tight_layout(pad=0.5)
    save_path = os.path.join(FIG_DIR, 'fig_ec_switch_map.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    ## ── CSV export: data_ec_switch.csv ──────────────────────────────────
    pd.DataFrame(records)[['city','lon','lat','n_colored']].rename(
        columns={'n_colored': 'abs_colored_months'}
    ).round(3).to_csv(
        os.path.join(FIG_DIR, 'data_ec_switch.csv'), index=False)
    print('CSV → data_ec_switch.csv')
    print(f'Saved → {save_path}')


# ──────────────────────────────────────────────────────────────────────
# 9. 图5：朝向玫瑰图（折线版）
# ──────────────────────────────────────────────────────────────────────
def plot_fig5():
    available  = df_summary['city'].unique()
    _cfg5 = {
        'Tropical\n(Singapore)': ('Singapore',  1.35),
        'Arid\n(Cairo)':         ('Cairo',      30.06),
        'Temperate\n(Beijing)':  ('Beijing',    39.90),
        'Cold\n(Helsinki)':      ('Helsinki',   60.17),
    }
    CITIES_SEL = {}
    for label, (name, lat) in _cfg5.items():
        resolved = _find_city(name, available)
        if resolved:
            CITIES_SEL[label] = (resolved, lat)
    if not CITIES_SEL:
        print('[跳过] 图5：代表城市不在数据中')
        return

    GLASS_STYLE_R = {
        'normal_glass':     {'color': '#4C72B0', 'lw': 2.0, 'ls': '-', 'alpha_fill': 0.10,
                             'zorder': 4, 'label': 'Normal glass (baseline)'},
        'tinted_glass':     {'color': '#DD8452', 'lw': 1.4, 'ls': '-', 'alpha_fill': 0.12,
                             'zorder': 3, 'label': 'Tinted glass'},
        'high_trans_glass': {'color': '#55A868', 'lw': 1.4, 'ls': '-', 'alpha_fill': 0.12,
                             'zorder': 3, 'label': 'High-trans glass'},
        'ec_abs_optimal':   {'color': '#762A83', 'lw': 1.8, 'ls': '-', 'alpha_fill': 0.15,
                             'zorder': 5, 'label': 'EC glass (optimal)'},
    }
    glass_keys_r = list(GLASS_STYLE_R.keys())
    EC_PAIR_R    = ['normal_glass_with_ec_trans', 'normal_glass_with_ec_colored']
    HEADINGS     = [0, 90, 180, 270]

    def get_direction_labels(lat):
        if lat >= 10:    return {0: 'PL\n(N)', 90: 'E', 180: 'EQ\n(S)', 270: 'W'}
        elif lat <= -10: return {0: 'EQ\n(N)', 90: 'E', 180: 'PL\n(S)', 270: 'W'}
        else:            return {0: 'N', 90: 'E', 180: 'S', 270: 'W'}

    def heading_to_theta(h):
        return np.deg2rad(90 - h)

    THETAS        = np.array([heading_to_theta(h) for h in HEADINGS])
    THETAS_CLOSED = np.append(THETAS, THETAS[0])

    def get_heading_range(city, glass_key):
        vals = []
        for h in HEADINGS:
            if glass_key == 'ec_abs_optimal':
                month_vals = []
                for m in df_summary['month'].unique():
                    cands = df_summary[
                        (df_summary['city']  == city) &
                        (df_summary['glass'].isin(EC_PAIR_R)) &
                        (df_summary['month'] == m) &
                        (df_summary['heading'] == h)
                    ]
                    if not cands.empty:
                        month_vals.append(cands['driving_km'].max())
                vals.append(np.nanmean(month_vals) if month_vals else np.nan)
            else:
                sub = df_summary[
                    (df_summary['city']  == city) &
                    (df_summary['glass'] == glass_key) &
                    (df_summary['heading'] == h)
                ]
                vals.append(sub['driving_km'].mean() if not sub.empty else np.nan)
        return np.array(vals)

    n_cities = len(CITIES_SEL)
    N_COLS   = 4
    n_cols   = min(N_COLS, n_cities)
    n_rows   = int(np.ceil(n_cities / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows + 0.8),
                              subplot_kw={'projection': 'polar'},
                              gridspec_kw={'hspace': 0.55, 'wspace': 0.45,
                                           'left': 0.04, 'right': 0.96,
                                           'top': 0.90, 'bottom': 0.14})

    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(axes.ravel())

    for ax in axes_flat[n_cities:]:
        ax.set_visible(False)

    for ax_i, (climate_label, (city, lat)) in enumerate(CITIES_SEL.items()):
        ax   = axes_flat[ax_i]
        data = {gk: get_heading_range(city, gk) for gk in glass_keys_r}

        all_vals = np.concatenate(list(data.values()))
        all_vals = all_vals[~np.isnan(all_vals)]
        r_min = max(np.nanmin(all_vals) * 0.93, 0)
        r_max = np.nanmax(all_vals) * 1.07

        for gk in glass_keys_r:
            st  = GLASS_STYLE_R[gk]
            r   = data[gk]
            r_c = np.append(r, r[0])
            ax.fill(THETAS_CLOSED, r_c, color=st['color'], alpha=st['alpha_fill'],
                    zorder=st['zorder'] - 1)
            ax.plot(THETAS_CLOSED, r_c, color=st['color'], linewidth=st['lw'],
                    linestyle=st['ls'], zorder=st['zorder'],
                    solid_capstyle='round', solid_joinstyle='round')
            ax.scatter(THETAS, r, color=st['color'], s=16, zorder=st['zorder'] + 1,
                       edgecolors='white', linewidths=0.5)

        ax.set_ylim(r_min, r_max)
        r_ticks = np.linspace(r_min, r_max, 3)
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f'{v:.0f}' for v in r_ticks], fontsize=5.5, color='#555555')
        ax.set_rlabel_position(40)

        dir_labels = get_direction_labels(lat)
        theta_deg_map = {h: np.rad2deg(heading_to_theta(h)) % 360 for h in HEADINGS}
        ax.set_thetagrids([theta_deg_map[h] for h in HEADINGS],
                          labels=[dir_labels[h] for h in HEADINGS],
                          fontsize=7.5, fontweight='bold')

        ax.grid(True, linewidth=0.3, color='#bbbbbb', linestyle='--', alpha=0.8)
        ax.spines['polar'].set_linewidth(0.5)
        ax.spines['polar'].set_color('#999999')
        ax.set_title(climate_label, fontsize=8.5, fontweight='bold', pad=14, linespacing=1.4)
        ax.text(np.deg2rad(40), r_max * 1.01, 'km', fontsize=5.5, color='#666666',
                ha='left', va='bottom')

    handles = [Line2D([0], [0], color=GLASS_STYLE_R[gk]['color'],
                      linewidth=GLASS_STYLE_R[gk]['lw'], linestyle=GLASS_STYLE_R[gk]['ls'],
                      marker='o', markersize=4, markerfacecolor='white', markeredgewidth=0.6,
                      label=GLASS_STYLE_R[gk]['label'])
               for gk in glass_keys_r]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(handles), fontsize=7.5, frameon=True, framealpha=0.9,
               edgecolor='#cccccc', borderpad=0.5, handlelength=1.8, columnspacing=1.2)
    fig.text(0.5, 0.055,
             'EQ = equator-facing  |  PL = pole-facing  |  '
             'Values = annual mean driving range (km)  |  Normal glass shown as baseline',
             ha='center', va='top', fontsize=6.5, color='#555555', style='italic')

    save_path = os.path.join(FIG_DIR, 'fig_heading_rose.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved → {save_path}')


# ──────────────────────────────────────────────────────────────────────
# 10. 图6：TMS 功耗分解堆叠柱状图
# ──────────────────────────────────────────────────────────────────────
def plot_fig6():
    available  = df_summary['city'].unique()
    CITIES_SEL = _resolve_cities_sel(CITIES_SEL_CFG, available)
    if not CITIES_SEL:
        print('[跳过] 图6：代表城市不在数据中')
        return

    GLASS_SCHEMES = [
        ('normal_glass',     'Normal',   '#2166AC'),
        ('tinted_glass',     'Tinted',   '#D6604D'),
        ('high_trans_glass', 'Hi-trans', '#4DAC26'),
        ('ec_abs_optimal',   'EC-Abs',   '#762A83'),
        ('ec_ref_optimal',   'EC-Ref',   '#E08214'),
    ]
    EC_PAIR_ABS_B = ['normal_glass_with_ec_trans', 'normal_glass_with_ec_colored']
    EC_PAIR_REF_B = ['normal_glass_with_ec_ref_trans', 'normal_glass_with_ec_ref_colored']
    COMPONENTS = [
        ('rfgPcomp',      'Compressor',          '#7F2704'),
        ('cabPfanair',    'Cabin fan',            '#D94801'),
        ('condPfanair',   'Condenser fan',        '#F16913'),
        ('esscoolPpump',  'Battery cool. pump',   '#FD8D3C'),
        ('mccoolPpump',   'Motor cool. pump',     '#FDAE6B'),
        ('mcradPfanair',  'Motor rad. fan',       '#FDD0A2'),
        ('essradPfanair', 'Battery rad. fan',     '#FEE6CE'),
    ]
    HEADINGS_B = [0, 90, 180, 270]
    MONTHS_B   = sorted(df_summary['month'].unique().tolist())

    def load_annual_components(city, glass_key):
        totals = {ck: [] for ck, _, _ in COMPONENTS}
        _first_err_logged = [False]
        for month in MONTHS_B:
            for h in HEADINGS_B:
                _pair_b = EC_PAIR_ABS_B if glass_key == 'ec_abs_optimal' else EC_PAIR_REF_B
                if glass_key in ('ec_abs_optimal', 'ec_ref_optimal'):
                    best_vals = None
                    best_ptms = np.inf
                    for g in _pair_b:
                        try:
                            df_ts = read_ts(g, city, month, h)
                            ptms  = df_ts['sumPTMS'].mean()
                            if ptms < best_ptms:
                                best_ptms = ptms
                                best_vals = {ck: df_ts[ck].mean()
                                             for ck, _, _ in COMPONENTS if ck in df_ts.columns}
                        except Exception as _e:
                            if not _first_err_logged[0]:
                                # 打印首个失败路径供诊断
                                _city_path = _city_to_key.get(city, city)
                                _sample = os.path.join(
                                    OUTPUT_DIR, 'ts',
                                    f'glass={g}', f'city={_city_path}',
                                    f'month={month:02d}', f'heading={h}',
                                    'data.parquet')
                                print(f'  [!] read_ts 失败: {_e}')
                                print(f'      路径: {_sample}')
                                print(f'      存在: {os.path.exists(_sample)}')
                                _first_err_logged[0] = True
                            continue
                    if best_vals:
                        for ck, _, _ in COMPONENTS:
                            totals[ck].append(best_vals.get(ck, 0.0))
                else:
                    try:
                        df_ts = read_ts(glass_key, city, month, h)
                        for ck, _, _ in COMPONENTS:
                            if ck in df_ts.columns:
                                totals[ck].append(df_ts[ck].mean())
                    except Exception as _e:
                        if not _first_err_logged[0]:
                            _city_path = _city_to_key.get(city, city)
                            _sample = os.path.join(
                                OUTPUT_DIR, 'ts',
                                f'glass={glass_key}', f'city={_city_path}',
                                f'month={month:02d}', f'heading={h}',
                                'data.parquet')
                            print(f'  [!] read_ts 失败: {_e}')
                            print(f'      路径: {_sample}')
                            print(f'      存在: {os.path.exists(_sample)}')
                            _first_err_logged[0] = True
                        continue
        return {ck: (np.mean(v) if v else 0.0) for ck, v in totals.items()}

    city_vals  = list(CITIES_SEL.values())
    n_cities   = len(CITIES_SEL)
    N_COLS     = 4
    n_cols     = min(N_COLS, n_cities)
    n_rows     = int(np.ceil(n_cities / n_cols))

    # ── 路径诊断：打印样本路径确认 ts/ 目录结构 ──────────────────────
    _diag_city  = city_vals[0] if city_vals else None
    _diag_glass = GLASS_SCHEMES[0][0]
    if _diag_city:
        _diag_city_path = _city_to_key.get(_diag_city, _diag_city)
        _diag_path = os.path.join(
            OUTPUT_DIR, 'ts',
            f'glass={_diag_glass}', f'city={_diag_city_path}',
            'month=01', 'heading=0', 'data.parquet')
        print(f'[图6] 样本路径: {_diag_path}')
        print(f'[图6] 样本文件存在: {os.path.exists(_diag_path)}')
        # 列出 ts/ 下的前几个目录供参考
        _ts_dir = os.path.join(OUTPUT_DIR, 'ts')
        if os.path.isdir(_ts_dir):
            _glass_dirs = sorted(os.listdir(_ts_dir))[:3]
            print(f'[图6] ts/ 下目录（前3）: {_glass_dirs}')
            for _gd in _glass_dirs[:1]:
                _city_dirs = sorted(os.listdir(os.path.join(_ts_dir, _gd)))[:3]
                print(f'[图6]   {_gd}/ 城市目录（前3）: {_city_dirs}')
        # 列出一个 ts 文件的列名
        import glob as _glob
        _sample_files = _glob.glob(os.path.join(OUTPUT_DIR, 'ts', '**', 'data.parquet'),
                                   recursive=True)
        if _sample_files:
            try:
                _sample_df = pd.read_parquet(_sample_files[0])
                print(f'[图6] 样本 ts 列名: {list(_sample_df.columns)}')
                _comp_cols = [ck for ck, _, _ in COMPONENTS]
                _missing   = [c for c in _comp_cols if c not in _sample_df.columns]
                if _missing:
                    print(f'[图6] ⚠ COMPONENTS 中缺失列: {_missing}')
                else:
                    print(f'[图6] ✓ 所有 COMPONENTS 列均存在')
            except Exception as _e:
                print(f'[图6] 读取样本 ts 失败: {_e}')

    print('读取时序数据（图6）...')
    cache = {}
    for city in city_vals:
        for gk, _, _ in GLASS_SCHEMES:
            print(f'  {city:12s} / {gk}')
            cache[(city, gk)] = load_annual_components(city, gk)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.6 * n_rows + 1.0),
                             gridspec_kw={'hspace': 0.42, 'wspace': 0.30,
                                          'left': 0.08, 'right': 0.97,
                                          'top': 0.93, 'bottom': 0.13})
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    x       = np.arange(len(GLASS_SCHEMES))
    bar_w   = 0.55
    x_labels = [label for _, label, _ in GLASS_SCHEMES]

    for city_i, (climate_key, city) in enumerate(CITIES_SEL.items()):
        row_i, col_i = city_i // n_cols, city_i % n_cols
        ax = axes[row_i, col_i]
        bottom = np.zeros(len(GLASS_SCHEMES))
        for ck, clabel, color in COMPONENTS:
            vals = np.array([cache[(city, gk)][ck] for gk, _, _ in GLASS_SCHEMES])
            ax.bar(x, vals, bar_w, bottom=bottom, color=color,
                   edgecolor='white', linewidth=0.3, zorder=3)
            bottom += vals
        for xi, total in enumerate(bottom):
            ax.text(xi, total + bottom.max() * 0.015, f'{total:.0f}',
                    ha='center', va='bottom', fontsize=6.5, color='#333333')
        ax.set_title(climate_key, fontsize=9, fontweight='bold', pad=5, linespacing=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylim(0, bottom.max() * 1.16)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(axis='both', length=3, width=0.6, direction='out')
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        if col_i == 0:
            ax.set_ylabel('Annual mean TMS power (W)', fontsize=7.5, labelpad=3)
        else:
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)

    for empty_i in range(n_cities, n_rows * n_cols):
        axes[empty_i // n_cols, empty_i % n_cols].set_visible(False)

    comp_handles = [mpatches.Patch(facecolor=color, edgecolor='none', label=clabel)
                    for _, clabel, color in COMPONENTS]
    fig.legend(handles=comp_handles, loc='lower center', bbox_to_anchor=(0.5, 0.00),
               ncol=len(comp_handles), fontsize=7.2, frameon=True, framealpha=0.9,
               edgecolor='#cccccc', borderpad=0.5, handlelength=1.2, columnspacing=0.9)

    save_path = os.path.join(FIG_DIR, 'fig_tms_breakdown.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    ## ── CSV export: data_tms_breakdown.csv ──────────────────────────────
    _rows6 = []
    for clabel6, city6 in CITIES_SEL.items():
        for gk6, glabel6, _ in GLASS_SCHEMES:
            comp6 = cache.get((city6, gk6), {})
            row6 = {'city': city6,
                    'climate_label': clabel6.replace('\n', ' '),
                    'glass': glabel6}
            for ck6, clbl6, _ in COMPONENTS:
                col6 = clbl6.lower().replace(' ','_').replace('.','').replace('/','_')
                row6[col6] = round(comp6.get(ck6, 0.0), 3)
            _rows6.append(row6)
    pd.DataFrame(_rows6).to_csv(
        os.path.join(FIG_DIR, 'data_tms_breakdown.csv'), index=False)
    print('CSV → data_tms_breakdown.csv')
    print(f'Saved → {save_path}')


# ──────────────────────────────────────────────────────────────────────
# 11. 图7：热舒适散点图（MRT vs Tamb）
# ──────────────────────────────────────────────────────────────────────
def plot_fig7():
    import matplotlib.lines as mlines

    if df_summary['tamb'].isna().all():
        print('[跳过] 图7：无 Tamb 数据')
        return

    available = df_summary['city'].unique()
    HIGHLIGHT_CITIES = {}
    for label, (name, color) in {
        'Tropical (Singapore)': ('Singapore',  '#E74C3C'),
        'Arid (Cairo)':         ('Cairo',      '#E67E22'),
        'Temperate (Beijing)':  ('Beijing',    '#2980B9'),
        'Cold (Helsinki)':      ('Helsinki',   '#27AE60'),
    }.items():
        resolved = _find_city(name, available)
        if resolved:
            HIGHLIGHT_CITIES[label] = (resolved, color)

    GLASS_SCHEMES_S = [
        ('normal_glass',     'Normal glass'),
        ('tinted_glass',     'Tinted glass'),
        ('high_trans_glass', 'High-trans glass'),
        ('ec_abs_optimal',   'EC absorptive'),
        ('ec_ref_optimal',   'EC reflective'),
    ]
    EC_PAIR_ABS_S = ['normal_glass_with_ec_trans', 'normal_glass_with_ec_colored']
    EC_PAIR_REF_S = ['normal_glass_with_ec_ref_trans', 'normal_glass_with_ec_ref_colored']
    MRT_COMFORT  = 25.0
    TAMB_COOLING = 25.0

    def get_ec_mrt_grp(df_sum, ec_pair, ec_key):
        rows = []
        for (city, month, heading), grp in df_sum[
                df_sum['glass'].isin(ec_pair)].groupby(['city', 'month', 'heading']):
            best = grp.loc[grp['mean_sumPTMS'].idxmin()]
            rows.append({'city': city, 'month': month, 'heading': heading,
                         'tamb': best['tamb'], 'mean_mrt': best['mean_mrt'],
                         'glass': ec_key})
        return rows

    plot_records = []
    base_glasses = [g for g, _ in GLASS_SCHEMES_S[:3]]
    for _, row in df_summary.iterrows():
        if row['glass'] not in base_glasses:
            continue
        plot_records.append({'city': row['city'], 'month': row['month'],
                             'heading': row['heading'], 'tamb': row['tamb'],
                             'mean_mrt': row['mean_mrt'], 'glass': row['glass']})
    plot_records.extend(get_ec_mrt_grp(df_summary, EC_PAIR_ABS_S, 'ec_abs_optimal'))
    plot_records.extend(get_ec_mrt_grp(df_summary, EC_PAIR_REF_S, 'ec_ref_optimal'))
    df_plot_s = pd.DataFrame(plot_records)

    n_schemes = len(GLASS_SCHEMES_S)
    fig, axes = plt.subplots(1, n_schemes, figsize=(3.0 * n_schemes, 4.2), sharey=True,
                             gridspec_kw={'wspace': 0.06, 'left': 0.09, 'right': 0.97,
                                          'top': 0.88, 'bottom': 0.18})

    tamb_all = df_plot_s['tamb'].dropna()
    mrt_all  = df_plot_s['mean_mrt'].dropna()
    x_min = np.floor(tamb_all.min() / 5) * 5 - 2
    x_max = np.ceil (tamb_all.max() / 5) * 5 + 2
    y_min = np.floor(mrt_all.min()  / 5) * 5 - 2
    y_max = np.ceil (mrt_all.max()  / 5) * 5 + 2
    x_min = min(x_min, TAMB_COOLING - 5)
    x_max = max(x_max, TAMB_COOLING + 5)
    y_min = min(y_min, MRT_COMFORT  - 5)
    y_max = max(y_max, MRT_COMFORT  + 5)

    for col_i, (gk, scheme_label) in enumerate(GLASS_SCHEMES_S):
        ax  = axes[col_i]
        df_g = df_plot_s[df_plot_s['glass'] == gk]

        ax.scatter(df_g['tamb'], df_g['mean_mrt'], s=8, color='#CCCCCC', alpha=0.45,
                   linewidths=0, zorder=2, rasterized=True)

        for climate_label, (city, color) in HIGHLIGHT_CITIES.items():
            sub = df_g[df_g['city'] == city]
            if sub.empty:
                continue
            monthly = sub.groupby('month')[['tamb', 'mean_mrt']].mean()
            ax.scatter(monthly['tamb'], monthly['mean_mrt'], s=45, color=color, alpha=0.90,
                       linewidths=0.5, edgecolors='white', zorder=5)
            monthly_sorted = monthly.sort_index()
            ax.plot(monthly_sorted['tamb'], monthly_sorted['mean_mrt'],
                    color=color, linewidth=0.6, alpha=0.5, zorder=4, linestyle='-')

        ax.axhline(MRT_COMFORT,  color='#CC0000', linewidth=0.7, linestyle=':', zorder=3, alpha=0.7)
        ax.axvline(TAMB_COOLING, color='#2980B9', linewidth=0.7, linestyle=':', zorder=3, alpha=0.7)

        if col_i == n_schemes - 1:
            ax.text(x_max - 0.5, MRT_COMFORT + 0.6, f'{MRT_COMFORT}°C', fontsize=6.5,
                    color='#CC0000', ha='right', va='bottom')
            ax.text(TAMB_COOLING + 0.4, y_min + 1, f'{TAMB_COOLING}°C', fontsize=6.5,
                    color='#2980B9', ha='left', va='bottom', rotation=90)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(axis='both', length=3, width=0.6, direction='out', labelsize=7.5)
        ax.yaxis.grid(True, linewidth=0.3, color='#dddddd', linestyle='--', zorder=0)
        ax.xaxis.grid(True, linewidth=0.3, color='#dddddd', linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(scheme_label, fontsize=9, fontweight='bold', pad=5)
        ax.set_xlabel('Outdoor temperature (°C)', fontsize=8)
        if col_i == 0:
            ax.set_ylabel('MRT (°C)', fontsize=8, labelpad=3)

    city_handles = [
        mlines.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markersize=6,
                      markeredgecolor='white', markeredgewidth=0.5,
                      label=climate_label.replace('\n', ' '))
        for climate_label, (city, color) in HIGHLIGHT_CITIES.items()
    ]
    bg_handle = mlines.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='#CCCCCC', markersize=5,
                               label='All cities (background)')
    ref_handles = [
        mlines.Line2D([0], [0], color='#CC0000', linewidth=0.9, linestyle=':',
                      label=f'Baseline ({MRT_COMFORT}°C MRT)'),
        mlines.Line2D([0], [0], color='#2980B9', linewidth=0.9, linestyle=':',
                      label=f'Baseline ({TAMB_COOLING}°C Tamb)'),
    ]
    fig.legend(handles=city_handles + [bg_handle] + ref_handles,
               loc='lower center', bbox_to_anchor=(0.5, 0.00),
               ncol=len(city_handles) + 3, fontsize=7.2, frameon=True, framealpha=0.9,
               edgecolor='#cccccc', borderpad=0.5, handlelength=1.4, columnspacing=0.9)

    save_path = os.path.join(FIG_DIR, 'fig_thermal_comfort_scatter.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    ## ── CSV export: data_thermal_scatter.csv ────────────────────────────
    _sc = df_plot_s.groupby(['city','glass'])[['tamb','mean_mrt']].mean().reset_index()
    _sc.round(3).to_csv(
        os.path.join(FIG_DIR, 'data_thermal_scatter.csv'), index=False)
    print('CSV → data_thermal_scatter.csv')
    print(f'Saved → {save_path}')


# ──────────────────────────────────────────────────────────────────────
# 12. 图8：全球城市 EC 节能气泡图（全量城市）
#     分别生成吸收式EC（fig_ec_bubble_abs.png）和反射式EC（fig_ec_bubble_ref.png）
# ──────────────────────────────────────────────────────────────────────
def plot_fig8():
    if df_summary['tamb'].isna().all():
        print('[跳过] 图8：无 Tamb 数据（需先补充）')
        return

    LABEL_THRESHOLD = 30    # 城市数超过此阈值则关闭标注（全量城市自动关闭）
    BUBBLE_SCALE    = 55

    # 确保 df_avg['tamb'] 已填充（从 df_summary 补充）
    if df_avg['tamb'].isna().mean() > 0.5:   # 超过50%缺失则尝试从df_summary补
        _tamb_map_b = df_summary.groupby(['city','month'])['tamb'].mean()
        df_avg['tamb'] = df_avg.apply(
            lambda r: _tamb_map_b.get((r['city'], r['month']), r['tamb']), axis=1)
    annual_g = df_avg.groupby(['glass', 'city'])[['driving_km', 'mean_solar', 'tamb']].mean()
    # tamb 来自 df_summary 的 tamb 列（仿真写入的月均气温）；若仍为NaN则跳过该城市

    def _build_bubble_df(df_ec_src):
        """根据给定的 EC optimal df 构建气泡图数据。"""
        ec_annual = df_ec_src.groupby('city')[['driving_km']].mean()
        ec_annual.columns = ['ec_driving_km']
        records = []
        for city in df_avg['city'].unique():
            try:
                base_km = annual_g.loc[('normal_glass', city), 'driving_km']
                solar   = annual_g.loc[('normal_glass', city), 'mean_solar']
                tamb    = annual_g.loc[('normal_glass', city), 'tamb']
                ec_km   = ec_annual.loc[city, 'ec_driving_km']
            except KeyError:
                continue
            if any(np.isnan([base_km, solar, tamb, ec_km])):
                continue
            delta_km   = ec_km - base_km
            saving_pct = delta_km / base_km * 100 if base_km else np.nan
            records.append({'city': city, 'solar': solar, 'tamb': tamb,
                            'delta_km': delta_km, 'saving_pct': saving_pct,
                            'base_km': base_km, 'ec_km': ec_km})
        return pd.DataFrame(records)

    def _draw_bubble(df_bubble, title, save_path, panel_label='f'):
        n_cities  = len(df_bubble)
        print(f'[图8] 气泡图城市数：{n_cities}  ({title})')

        s_min = df_bubble['saving_pct'].min()
        s_max = df_bubble['saving_pct'].max()

        saving_vals = df_bubble['saving_pct'].values
        FIXED_SIZE  = 40   # 固定气泡大小
        bubble_area = np.full(len(df_bubble), FIXED_SIZE)


        # 双极色标：绿=节能正优化，红=负优化
        abs_max_b   = max(abs(s_min) if s_min < 0 else 0, abs(s_max))
        norm_rdylgn = mcolors.TwoSlopeNorm(vmin=-abs_max_b, vcenter=0, vmax=abs_max_b)
        cmap_rdylgn = plt.get_cmap('RdYlGn')
        colors_all  = [cmap_rdylgn(norm_rdylgn(v)) for v in saving_vals]
        norm_pos = norm_rdylgn; cmap_pos = cmap_rdylgn

        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.yaxis.grid(True, linewidth=0.3, color='#dddddd', linestyle='--', zorder=0)
        ax.xaxis.grid(True, linewidth=0.3, color='#dddddd', linestyle='--', zorder=0)
        ax.set_axisbelow(True)

        solar_median = df_bubble['solar'].median()
        tamb_median  = df_bubble['tamb'].median()
        ax.axvline(solar_median, color='#bbbbbb', linewidth=0.6, linestyle=':', zorder=1)
        ax.axhline(tamb_median,  color='#bbbbbb', linewidth=0.6, linestyle=':', zorder=1)

        ax.scatter(df_bubble['solar'], df_bubble['tamb'],
                   s=bubble_area, c=colors_all,
                   alpha=0.82, linewidths=0, edgecolors='none', zorder=4)

        show_labels = (n_cities <= LABEL_THRESHOLD)
        if show_labels:
            for _, row in df_bubble.iterrows():
                ax.annotate(row['city'].replace('.', ' '),
                            xy=(row['solar'], row['tamb']),
                            xytext=(5, 4), textcoords='offset points',
                            fontsize=7, color='#333333', zorder=6)

        x_range = df_bubble['solar'].max() - df_bubble['solar'].min()
        y_range = df_bubble['tamb'].max()  - df_bubble['tamb'].min()
        pad_x   = x_range * 0.03
        pad_y   = y_range * 0.03
        quadrant_labels = [
            (df_bubble['solar'].max() - pad_x, df_bubble['tamb'].max() - pad_y,
             'Hot & Sunny\n(High EC benefit)',  'right', 'top',    '#8B1A1A'),
            (df_bubble['solar'].min() + pad_x, df_bubble['tamb'].min() + pad_y,
             'Cold & Low irradiance\n(Low EC benefit)', 'left', 'bottom', '#1A3A8B'),
        ]
        for qx, qy, qlabel, ha, va, qcolor in quadrant_labels:
            ax.text(qx, qy, qlabel, fontsize=7, color=qcolor, alpha=0.65,
                    ha=ha, va=va, style='italic', zorder=5)

        sm_b = cm.ScalarMappable(cmap=cmap_rdylgn, norm=norm_rdylgn)
        sm_b.set_array([])
        cbar_b = plt.colorbar(sm_b, ax=ax, orientation='vertical',
                              pad=0.02, fraction=0.030, aspect=28, extend='both')
        cbar_b.set_label('Range improvement vs. normal glass (%)', fontsize=8, labelpad=4)
        cbar_b.ax.tick_params(labelsize=7, length=2, width=0.4)
        # (气泡大小 legend 已移除)

        ax.set_xlabel('Annual mean solar irradiance at 14:00 (W/m²)', fontsize=9, labelpad=5)
        ax.set_ylabel('Annual mean outdoor temperature at 14:00 (°C)', fontsize=9, labelpad=5)
        ax.set_title(f'{title}\nBubble size = |Δ range (km)|  |  Green = gain  |  Red = loss',
                     fontsize=10, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(axis='both', length=3, width=0.6, direction='out', labelsize=8)
        ax.text(-0.09, 1.02, panel_label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom', ha='left')

        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'Saved → {save_path}')

        ## ── 新增：合并双面板气泡图 ────────────────────────────────────────
        x_lo_c = min(df_bubble['solar'].min(), df_bub_ref['solar'].min())
        x_hi_c = max(df_bubble['solar'].max(), df_bub_ref['solar'].max())
        y_lo_c = min(df_bubble['tamb'].min(),  df_bub_ref['tamb'].min())
        y_hi_c = max(df_bubble['tamb'].max(),  df_bub_ref['tamb'].max())
        pad_xc = (x_hi_c - x_lo_c) * 0.04
        pad_yc = (y_hi_c - y_lo_c) * 0.04
        all_pct   = np.concatenate([df_bubble['saving_pct'].values,
                                    df_bub_ref['saving_pct'].values])
        abs_max_c = max(abs(np.nanmin(all_pct)) if np.nanmin(all_pct) < 0 else 0,
                        abs(np.nanmax(all_pct)))
        norm_c = mcolors.TwoSlopeNorm(vmin=-abs_max_c, vcenter=0, vmax=abs_max_c)
        cmap_c = plt.get_cmap('RdYlGn')
        fig_c, axes_c = plt.subplots(
            1, 2, figsize=(14, 5.5), sharey=True,
            gridspec_kw={'wspace': 0.08, 'left': 0.07, 'right': 0.88,
                         'top': 0.88, 'bottom': 0.14})
        for ax_c, df_c, title_c, is_left in [
            (axes_c[0], df_bubble,  'Absorptive EC vs. Normal Glass', True),
            (axes_c[1], df_bub_ref, 'Reflective EC vs. Normal Glass', False),
        ]:
            bub_c = np.full(len(df_c), FIXED_SIZE)
            col_c = [cmap_c(norm_c(v)) for v in df_c['saving_pct'].values]
            ax_c.scatter(df_c['solar'], df_c['tamb'], s=bub_c, c=col_c,
                         alpha=0.82, linewidths=0, edgecolors='none', zorder=4)
            ax_c.axvline(df_c['solar'].median(), color='#bbbbbb',
                         linewidth=0.6, linestyle=':', zorder=2)
            ax_c.axhline(df_c['tamb'].median(), color='#bbbbbb',
                         linewidth=0.6, linestyle=':', zorder=2)
            ax_c.set_xlim(x_lo_c - pad_xc, x_hi_c + pad_xc)
            ax_c.set_ylim(y_lo_c - pad_yc, y_hi_c + pad_yc)
            ax_c.set_xlabel('Annual mean solar irradiance at 14:00 (W/m²)',
                            fontsize=9, labelpad=4)
            if is_left:
                ax_c.set_ylabel('Annual mean outdoor temperature at 14:00 (°C)',
                                fontsize=9, labelpad=4)
            ax_c.set_title(title_c, fontsize=10, fontweight='bold', pad=6)
            ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
            ax_c.spines['left'].set_linewidth(0.6); ax_c.spines['bottom'].set_linewidth(0.6)
            ax_c.tick_params(axis='both', length=3, width=0.6, labelsize=8)
            xr_c = x_hi_c - x_lo_c; yr_c = y_hi_c - y_lo_c
        # (合并气泡大小 legend 已移除)
        sm_c = cm.ScalarMappable(cmap=cmap_c, norm=norm_c); sm_c.set_array([])
        cbar_c = fig_c.colorbar(sm_c, ax=axes_c, orientation='vertical',
                                 pad=0.02, fraction=0.022, aspect=32, extend='both')
        cbar_c.set_label('Range improvement vs. normal glass (%)', fontsize=8.5, labelpad=4)
        cbar_c.ax.tick_params(labelsize=7.5, length=2.5, width=0.5)
        fig_c.suptitle('EC Glazing: Absorptive vs. Reflective\n'
                       'Colour = range improvement vs. normal glass (%)  |  Green = gain  |  Red = loss',
                       fontsize=10, fontweight='bold', y=0.97)
        path_c = os.path.join(FIG_DIR, 'fig_ec_bubble_combined.png')
        fig_c.savefig(path_c, dpi=300, bbox_inches='tight',
                      facecolor='white', edgecolor='none')
        plt.close(fig_c)

        ## ── CSV export: data_bubble.csv ─────────────────────────────────
        _ba = df_bubble[['city','solar','tamb','base_km','ec_km','delta_km','saving_pct']].copy()
        _ba.insert(0, 'ec_type', 'absorptive')
        _br = df_bub_ref[['city','solar','tamb','base_km','ec_km','delta_km','saving_pct']].copy()
        _br.insert(0, 'ec_type', 'reflective')
        def _add_coords_b(df):
            df = df.copy()
            df['lon'] = df['city'].map(lambda c: city_coords.get(c,(np.nan,np.nan))[0])
            df['lat'] = df['city'].map(lambda c: city_coords.get(c,(np.nan,np.nan))[1])
            return df
        pd.concat([_add_coords_b(_ba), _add_coords_b(_br)], ignore_index=True
            ).round(4).to_csv(os.path.join(FIG_DIR, 'data_bubble.csv'), index=False)
        print('CSV → data_bubble.csv')
        print(f'Saved → {path_c}')

    # ── 分别生成吸收式EC和反射式EC气泡图 ─────────────────────────────
    df_bub_abs = _build_bubble_df(df_ec_abs)
    df_bub_ref = _build_bubble_df(df_ec_ref)

    _draw_bubble(df_bub_abs,
                 title='EC Absorptive Glass Performance Across Global Cities',
                 save_path=os.path.join(FIG_DIR, 'fig_ec_bubble_abs.png'),
                 panel_label='f')
    _draw_bubble(df_bub_ref,
                 title='EC Reflective Glass Performance Across Global Cities',
                 save_path=os.path.join(FIG_DIR, 'fig_ec_bubble_ref.png'),
                 panel_label='g')




# ──────────────────────────────────────────────────────────────────────
# 13. 图9：电池经济性分析（EC 节能 → 节约成本）
#     图9a/b：吸收式/反射式EC全球热力地图
#     图9c：4典型城市柱状图
# ──────────────────────────────────────────────────────────────────────
def plot_fig9():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    import numpy.ma as ma

    BATTERY_KWH  = 60.0    # Tesla Model 3 电池容量 (kWh)
    BATTERY_COST = 100.0   # 动力电池成本 (USD/kWh)

    annual_e   = df_avg.groupby(['glass', 'city'])[['driving_km']].mean()
    coord_df_e = df_summary.groupby('city')[['lat', 'lon']].mean()

    def build_econ_df(df_ec_src, label):
        ec_annual = df_ec_src.groupby('city')[['driving_km']].mean()
        records = []
        for city in df_avg['city'].unique():
            try:
                r_norm = float(annual_e.loc[('normal_glass', city), 'driving_km'])
                r_ec   = float(ec_annual.loc[city, 'driving_km'])
                lon_c  = float(coord_df_e.loc[city, 'lon'])
                lat_c  = float(coord_df_e.loc[city, 'lat'])
            except KeyError:
                continue
            if r_norm <= 0 or np.isnan(r_norm) or np.isnan(r_ec):
                continue
            eps         = BATTERY_KWH / r_norm
            delta_r     = r_ec - r_norm
            delta_e     = delta_r * eps
            delta_c     = delta_e * BATTERY_COST
            records.append({'city': city, 'lon': lon_c, 'lat': lat_c,
                            'r_norm': r_norm, 'delta_r': delta_r,
                            'delta_e': delta_e, 'delta_c': delta_c})
        df = pd.DataFrame(records)
        print(f'[图9] {label}：{len(df)} 城市，'
              f'ΔC [{df["delta_c"].min():.1f}, {df["delta_c"].max():.1f}] USD')
        return df

    df_econ_abs = build_econ_df(df_ec_abs, '吸收式EC')
    df_econ_ref = build_econ_df(df_ec_ref, '反射式EC')

    land_mask = _ensure_land_mask()

    def draw_econ_map(df_e, col, title, unit, save_path):
        import numpy.ma as ma2
        points = df_e[['lon', 'lat']].values
        values = df_e[col].values
        valid  = ~np.isnan(values)
        grid_vals = griddata(points[valid], values[valid],
                             (lon_mesh, lat_mesh), method='linear')
        grid_vals = np.where(land_mask, grid_vals, np.nan)
        _tree = cKDTree(points[valid])
        _dist, _ = _tree.query(np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), k=1)
        grid_vals = np.where(_dist.reshape(lon_mesh.shape) <= 8.0, grid_vals, np.nan)

        vmin = float(np.nanmin(grid_vals))
        vmax = float(np.nanmax(grid_vals))
        abs_max = max(abs(vmin), abs(vmax))
        norm  = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        from matplotlib.colors import LinearSegmentedColormap as _LSC9
        # 橙(负/亏损) → 白(零) → 紫(正/节省)，补色对比
        cmap = _LSC9.from_list('econ_orange_purple',
            ['#7F3B08', '#E08214', '#FEE8C8', '#FFFFFF', '#EDE0F3', '#9E9AC8', '#54278F'], N=256)
        cmap.set_bad(color='none')

        fig = plt.figure(figsize=(7.2, 3.8))
        ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.OCEAN,     facecolor='#e8eef4', zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor='#f2f0eb', zorder=1)
        pcm = ax.pcolormesh(lon_mesh, lat_mesh, ma2.masked_invalid(grid_vals),
                            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                            shading='auto', zorder=2, rasterized=True)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor='#555555', zorder=5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.20, edgecolor='#888888', zorder=5)
        # 全量城市点太密，不显示（2768城市）
        ax.gridlines(draw_labels=False, linewidth=0.25, color='#aaaaaa',
                     alpha=0.4, linestyle='--', zorder=3)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('#333333'); spine.set_linewidth(0.5)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        cb = plt.colorbar(pcm, ax=ax, orientation='horizontal',
                          pad=0.04, fraction=0.038, aspect=45, extend='both')
        cb.set_label(unit, fontsize=8.5, labelpad=3)
        cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=3))
        cb.ax.tick_params(labelsize=7.5, length=2.5, width=0.5)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'Saved → {save_path}')

    print('\n[图9a] 吸收式EC 经济性地图...')
    draw_econ_map(df_econ_abs, 'delta_c',
                  f'Battery Cost Saving — Absorptive EC vs. Normal Glass',
                  f'Cost saving (USD)',
                  os.path.join(FIG_DIR, 'fig_econ_abs_usd.png'))


    ## ── CSV export: data_econ.csv ───────────────────────────────────────
    _ea = df_econ_abs[['city','lon','lat','delta_r','delta_e','delta_c']].copy()
    _ea.insert(0, 'ec_type', 'absorptive')
    _er = df_econ_ref[['city','lon','lat','delta_r','delta_e','delta_c']].copy()
    _er.insert(0, 'ec_type', 'reflective')
    _ea.columns = ['ec_type','city','lon','lat',
                   'delta_range_km','delta_energy_kwh','cost_saving_usd']
    _er.columns = ['ec_type','city','lon','lat',
                   'delta_range_km','delta_energy_kwh','cost_saving_usd']
    pd.concat([_ea, _er], ignore_index=True).round(4).to_csv(
        os.path.join(FIG_DIR, 'data_econ.csv'), index=False)
    print('CSV → data_econ.csv')
    print('\n[图9b] 反射式EC 经济性地图...')
    draw_econ_map(df_econ_ref, 'delta_c',
                  f'Battery Cost Saving — Reflective EC vs. Normal Glass',
                  f'Cost saving (USD)',
                  os.path.join(FIG_DIR, 'fig_econ_ref_usd.png'))

    # ── 图9c：4典型城市柱状图 ──────────────────────────────────────
    available_e = df_avg['city'].unique()
    CITIES_ECON = {}
    for lbl, name in {'Tropical (Singapore)': 'Singapore', 'Arid (Cairo)': 'Cairo',
                       'Temperate (Beijing)': 'Beijing', 'Cold (Helsinki)': 'Helsinki'}.items():
        resolved = _find_city(name, available_e)
        if resolved:
            CITIES_ECON[lbl] = resolved

    if CITIES_ECON:
        city_labels = list(CITIES_ECON.keys())
        city_vals   = list(CITIES_ECON.values())
        n_c  = len(city_vals)
        x_c  = np.arange(n_c)
        BAR_W_E = 0.35
        off_e   = np.array([-0.5, 0.5]) * BAR_W_E

        def _get_delta_c(df_e, city):
            row = df_e[df_e['city'] == city]
            return float(row['delta_c'].values[0]) if not row.empty else np.nan

        abs_vals = [_get_delta_c(df_econ_abs, c) for c in city_vals]
        ref_vals = [_get_delta_c(df_econ_ref, c) for c in city_vals]

        fig_b, ax_b = plt.subplots(figsize=(8, 4.5),
                                   gridspec_kw={'left': 0.10, 'right': 0.95,
                                                'top': 0.88, 'bottom': 0.18})
        bars_abs = ax_b.bar(x_c + off_e[0], abs_vals, BAR_W_E,
                            color='#7B68B0', label='EC absorptive (optimal)',
                            edgecolor='white', linewidth=0.4, alpha=0.88, zorder=3)
        bars_ref = ax_b.bar(x_c + off_e[1], ref_vals, BAR_W_E,
                            color='#E08214', label='EC reflective',
                            edgecolor='white', linewidth=0.4, alpha=0.88, zorder=3)
        for bar in list(bars_abs) + list(bars_ref):
            h = bar.get_height()
            if np.isnan(h): continue
            ax_b.text(bar.get_x() + bar.get_width() / 2, h,
                      f'{h:.1f}', ha='center',
                      va='bottom' if h >= 0 else 'top',
                      fontsize=7, color='#333333')
        ax_b.axhline(0, color='#333333', linewidth=0.8, zorder=4)
        ax_b.set_title(
            f'Annual battery cost saving by EC glazing vs. normal glass\n'
            f'(Assumes {BATTERY_KWH:.0f} kWh battery @ ${BATTERY_COST:.0f}/kWh; '
            f'ε = {BATTERY_KWH:.0f} kWh / Range_normal)',
            fontsize=8.5, fontweight='bold', pad=6)
        ax_b.set_ylabel(f'Cost saving (USD / {BATTERY_KWH:.0f} kWh battery)', fontsize=8.5)
        ax_b.set_xticks(x_c)
        ax_b.set_xticklabels(city_labels, fontsize=8.5)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', zorder=0)
        ax_b.set_axisbelow(True)
        ax_b.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax_b.tick_params(axis='both', length=3, width=0.6)
        ax_b.legend(loc='upper right', fontsize=8, framealpha=0.9, edgecolor='#cccccc')

        path_b = os.path.join(FIG_DIR, 'fig_econ_city_bar.png')
        fig_b.savefig(path_b, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig_b)
        print(f'Saved → {path_b}')
    else:
        print('[跳过] 图9c：典型城市未找到')


# ──────────────────────────────────────────────────────────────────────
# 14. 图10：PMV/PPD 热舒适深入分析
#     图10a：月度 PMV 折线分面图（4城市 × 5玻璃方案）
#     图10b：月度 PPD 折线分面图
#     图10c：全城市 PMV 箱型图（5方案）
#     图10d：全城市 PPD 箱型图
# ──────────────────────────────────────────────────────────────────────
def plot_fig10():
    def _pmv_ppd(T_a, T_mrt, v_r=0.1, RH=50.0, M=1.2, I_cl=0.5):
        """PMV/PPD 计算（Fanger，ISO 7730）。"""
        M_W  = M * 58.15
        W    = 0.0
        f_cl = 1.05 + 0.1 * I_cl if I_cl > 0.5 else 1.0 + 0.2 * I_cl
        t_cl = T_a + (35.5 - T_a) / (3.5 * (6.45 * I_cl + 0.1))
        for _ in range(100):
            h_c = max(2.38 * abs(t_cl - T_a) ** 0.25, 12.1 * v_r ** 0.5)
            t_cl_new = 35.7 - 0.028 * (M_W - W) - I_cl * 0.155 * (
                3.96e-8 * f_cl * ((t_cl + 273)**4 - (T_mrt + 273)**4) +
                f_cl * h_c * (t_cl - T_a))
            if abs(t_cl_new - t_cl) < 0.01:
                break
            t_cl = t_cl_new
        h_c = max(2.38 * abs(t_cl - T_a) ** 0.25, 12.1 * v_r ** 0.5)
        p_a = RH / 100 * np.exp(16.6536 - 4030.183 / (T_a + 235))
        L = (M_W - W
             - 3.05e-3 * (5733 - 6.99 * (M_W - W) - p_a)
             - 0.42 * ((M_W - W) - 58.15)
             - 1.7e-5 * M_W * (5867 - p_a)
             - 0.0014 * M_W * (34 - T_a)
             - 3.96e-8 * f_cl * ((t_cl + 273)**4 - (T_mrt + 273)**4)
             - f_cl * h_c * (t_cl - T_a))
        pmv = float(np.clip((0.303 * np.exp(-0.036 * M_W) + 0.028) * L, -3, 3))
        ppd = float(100 - 95 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2))
        return pmv, ppd

    def calc_pmv_series(T_a_arr, T_mrt_arr, month):
        I_cl = 1.0 if month in (11, 12, 1, 2, 3) else 0.5
        pmv_vals, ppd_vals = [], []
        for ta, tmrt in zip(T_a_arr, T_mrt_arr):
            try:
                p, d = _pmv_ppd(float(ta), float(tmrt), I_cl=I_cl)
                pmv_vals.append(p)
                ppd_vals.append(d)
            except Exception:
                pass
        return (np.nanmean(pmv_vals) if pmv_vals else np.nan,
                np.nanmean(ppd_vals) if ppd_vals else np.nan)

    available_pmv = df_summary['city'].unique()
    CITIES_PMV_CFG = {
        'Tropical\n(Singapore)': 'Singapore',
        'Arid\n(Cairo)':         'Cairo',
        'Temperate\n(Beijing)':  'Beijing',
        'Cold\n(Helsinki)':      'Helsinki',
    }
    CITIES_PMV = {lbl: _find_city(name, available_pmv)
                  for lbl, name in CITIES_PMV_CFG.items()}
    CITIES_PMV = {lbl: c for lbl, c in CITIES_PMV.items() if c}
    if not CITIES_PMV:
        print('[跳过] 图10：代表城市不在数据中')
        return
    print(f'[图10] 典型城市：{CITIES_PMV}')

    GLASS_STYLE_PMV = {
        'normal_glass':     {'color': '#4C72B0', 'lw': 1.8, 'ls': '-',  'label': 'Normal glass'},
        'tinted_glass':     {'color': '#DD8452', 'lw': 1.3, 'ls': '-',  'label': 'Tinted glass'},
        'high_trans_glass': {'color': '#55A868', 'lw': 1.3, 'ls': '-',  'label': 'High-trans glass'},
        'ec_abs_optimal':   {'color': '#8172B2', 'lw': 1.6, 'ls': '-',  'label': 'EC absorptive'},
        'ec_ref_optimal':   {'color': '#C44E52', 'lw': 1.6, 'ls': '--', 'label': 'EC reflective'},
    }
    glass_keys_pmv = list(GLASS_STYLE_PMV.keys())
    MONTHS_PMV   = list(range(1, 13))
    HEADINGS_PMV = [0, 90, 180, 270]

    print('计算月均 PMV/PPD（读取时序 Parquet，请稍候）...')
    pmv_results = {}
    for city_label, city in CITIES_PMV.items():
        for month in MONTHS_PMV:
            for glass_key in glass_keys_pmv:
                if glass_key in ('normal_glass', 'tinted_glass', 'high_trans_glass'):
                    pair = [glass_key]
                elif glass_key == 'ec_abs_optimal':
                    pair = list(EC_PAIR_ABS)
                else:
                    pair = list(EC_PAIR_REF)

                best_ptms = np.inf
                best_pmv = best_ppd = np.nan
                for g in pair:
                    for h in HEADINGS_PMV:
                        try:
                            df_ts = read_ts(g, city, month, h)
                            p, d  = calc_pmv_series(df_ts['cabTair'].values,
                                                     df_ts['mrt_val'].values, month)
                            ptms  = df_ts['sumPTMS'].mean()
                            if glass_key in ('ec_abs_optimal', 'ec_ref_optimal'):
                                if ptms < best_ptms:
                                    best_ptms = ptms
                                    best_pmv, best_ppd = p, d
                            else:
                                best_pmv = np.nanmean([best_pmv, p]) if not np.isnan(best_pmv) else p
                                best_ppd = np.nanmean([best_ppd, d]) if not np.isnan(best_ppd) else d
                        except Exception:
                            pass
                pmv_results[(city, glass_key, month)] = (best_pmv, best_ppd)

    ## ── CSV export: data_pmv_ppd.csv ────────────────────────────────────
    _pmv_rows = []
    for gk_s, glabel_s in _GLASS_LABELS_S.items():
        for metric_s, m_label_s in [('pmv','PMV'), ('ppd','PPD')]:
            sub_s = df_box.loc[df_box['glass'] == gk_s, metric_s].dropna()
            if sub_s.empty: continue
            q_s = sub_s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            _pmv_rows.append({
                'glass': glabel_s, 'metric': m_label_s,
                'mean':   round(sub_s.mean(), 4),
                'std':    round(sub_s.std(),  4),
                'p10':    round(q_s[0.10], 4),
                'q25':    round(q_s[0.25], 4),
                'median': round(q_s[0.50], 4),
                'q75':    round(q_s[0.75], 4),
                'p90':    round(q_s[0.90], 4),
            })
    pd.DataFrame(_pmv_rows).to_csv(
        os.path.join(FIG_DIR, 'data_pmv_ppd.csv'), index=False)
    print('CSV → data_pmv_ppd.csv')
    print(f'PMV/PPD 计算完成，共 {len(pmv_results)} 条记录')

    months_pmv   = MONTHS_PMV
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    x_m       = np.arange(12)
    n_c_pmv   = len(CITIES_PMV)
    N_COLS_P  = min(4, n_c_pmv)

    def _plot_pmv_grid(metric_idx, ylabel, ref_lines, title_suffix, save_path):
        fig, axes = plt.subplots(
            1, N_COLS_P, figsize=(3.6 * N_COLS_P, 3.8),
            gridspec_kw={'wspace': 0.28, 'left': 0.08, 'right': 0.97,
                         'top': 0.88, 'bottom': 0.18})
        if N_COLS_P == 1:
            axes = [axes]
        for col_i, (climate_key, city) in enumerate(CITIES_PMV.items()):
            ax = axes[col_i]
            for gk in glass_keys_pmv:
                st   = GLASS_STYLE_PMV[gk]
                vals = [pmv_results.get((city, gk, m), (np.nan, np.nan))[metric_idx]
                        for m in months_pmv]
                ax.plot(x_m, vals, color=st['color'], linewidth=st['lw'],
                        linestyle=st['ls'], marker='o', markersize=3.5,
                        markerfacecolor='white', markeredgewidth=0.7, zorder=4)
            for val, color, ls, lbl in ref_lines:
                if lbl is not None:
                    ax.axhline(val, color=color, linewidth=0.8, linestyle=ls, alpha=0.7, zorder=3)
                else:
                    ax.axhline(val, color=color, linewidth=0.8, linestyle=ls, alpha=0.7, zorder=3)
            ax.set_title(climate_key, fontsize=9, fontweight='bold', pad=5, linespacing=1.4)
            ax.set_xlim(-0.5, 11.5)
            ax.set_xticks(x_m)
            ax.set_xticklabels(month_labels, fontsize=7.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis='both', length=3, width=0.6)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, labelpad=4)
            else:
                ax.set_yticklabels([])
                ax.spines['left'].set_visible(False)
                ax.tick_params(left=False)

        handles = [Line2D([0],[0], color=GLASS_STYLE_PMV[gk]['color'],
                          linewidth=GLASS_STYLE_PMV[gk]['lw'],
                          linestyle=GLASS_STYLE_PMV[gk]['ls'],
                          label=GLASS_STYLE_PMV[gk]['label'])
                   for gk in glass_keys_pmv]
        for val, color, ls, lbl in ref_lines:
            if lbl:
                handles.append(Line2D([0],[0], color=color, linewidth=0.8,
                                       linestyle=ls, label=lbl))
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.00),
                   ncol=min(len(handles), 6), fontsize=7.5, frameon=True, framealpha=0.9,
                   edgecolor='#cccccc', borderpad=0.5, handlelength=1.6)
        fig.suptitle(title_suffix, fontsize=9.5, fontweight='bold', y=0.97)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'Saved → {save_path}')

    # [已删除] fig_pmv_monthly.png
    # [已删除] fig_ppd_monthly.png

    # ── 箱型图：所有城市 × 12月 × 4朝向 ──────────────────────────
    box_records = []
    for (city, gk, month), (pmv_v, ppd_v) in pmv_results.items():
        box_records.append({'glass': gk, 'city': city, 'month': month,
                            'pmv': pmv_v, 'ppd': ppd_v})
    df_box = pd.DataFrame(box_records)

    ## ── 诊断：若 df_box 为空说明 ts 文件读取全部失败 ────────────────
    if df_box.empty or df_box['pmv'].isna().all():
        print('[图10] ⚠ pmv_results 为空，ts 文件可能缺失或列名变更')
        print('[图10] 改用 df_avg 的 mean_mrt + mean_cabTair 计算年均 PMV/PPD')
        # Fallback: compute from df_avg monthly means
        def _pmv_from_means(T_a, T_mrt, I_cl=0.5, v_r=0.1, RH=50.0, M=1.2):
            M_W = M * 58.15
            f_cl = 1.05 + 0.1*I_cl if I_cl > 0.5 else 1.0 + 0.2*I_cl
            t_cl = T_a + (35.5 - T_a) / (3.5*(6.45*I_cl + 0.1))
            for _ in range(100):
                h_c = max(2.38*abs(t_cl-T_a)**0.25, 12.1*v_r**0.5)
                t_cl_n = 35.7 - 0.028*M_W - I_cl*0.155*(
                    3.96e-8*f_cl*((t_cl+273)**4-(T_mrt+273)**4)+f_cl*h_c*(t_cl-T_a))
                if abs(t_cl_n-t_cl) < 0.01: break
                t_cl = t_cl_n
            h_c = max(2.38*abs(t_cl-T_a)**0.25, 12.1*v_r**0.5)
            p_a = RH/100*np.exp(16.6536-4030.183/(T_a+235))
            L = (M_W - 3.05e-3*(5733-6.99*M_W-p_a) - 0.42*(M_W-58.15)
                 - 1.7e-5*M_W*(5867-p_a) - 0.0014*M_W*(34-T_a)
                 - 3.96e-8*f_cl*((t_cl+273)**4-(T_mrt+273)**4) - f_cl*h_c*(t_cl-T_a))
            pmv = float(np.clip((0.303*np.exp(-0.036*M_W)+0.028)*L, -3, 3))
            ppd = float(100 - 95*np.exp(-0.03353*pmv**4 - 0.2179*pmv**2))
            return pmv, ppd
        box_fb = []
        for gk in glass_keys_pmv:
            if gk in ('normal_glass','tinted_glass','high_trans_glass'):
                df_g = df_avg[df_avg['glass']==gk]
            elif gk == 'ec_abs_optimal':
                df_g = df_ec_abs
            else:
                df_g = df_ec_ref
            for _, row_g in df_g.iterrows():
                I_cl = 1.0 if row_g['month'] in (11,12,1,2,3) else 0.5
                try:
                    pmv_v, ppd_v = _pmv_from_means(
                        float(row_g['mean_cabTair']), float(row_g['mean_mrt']), I_cl)
                    box_fb.append({'glass': gk, 'city': row_g['city'],
                                   'month': row_g['month'],
                                   'pmv': pmv_v, 'ppd': ppd_v})
                except: pass
        df_box = pd.DataFrame(box_fb)
        print(f'[图10] Fallback df_box: {len(df_box)} rows')

    def _boxplot_metric(df_b, metric, ylabel, ref_lines, title, save_path):
        fig_bx, ax_bx = plt.subplots(figsize=(7, 4.5),
                                     gridspec_kw={'left': 0.12, 'right': 0.97,
                                                  'top': 0.88, 'bottom': 0.22})
        data_by_glass = [df_b[df_b['glass'] == gk][metric].dropna().values
                         for gk in glass_keys_pmv]
        colors_bx = [GLASS_STYLE_PMV[gk]['color'] for gk in glass_keys_pmv]
        labels_bx = [GLASS_STYLE_PMV[gk]['label'] for gk in glass_keys_pmv]

        bp = ax_bx.boxplot(
            data_by_glass, patch_artist=True, widths=0.5,
            medianprops=dict(color='white', linewidth=2.0),
            whiskerprops=dict(linewidth=0.8, color='#555555'),
            capprops=dict(linewidth=0.8, color='#555555'),
            flierprops=dict(marker='o', markersize=2.5, markerfacecolor='#aaaaaa',
                            markeredgewidth=0, alpha=0.5),
            boxprops=dict(linewidth=0.5), zorder=3
        )
        for patch, color in zip(bp['boxes'], colors_bx):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)

        for val, color, ls, lbl in ref_lines:
            ax_bx.axhline(val, color=color, linewidth=0.9, linestyle=ls, alpha=0.8, zorder=2)

        ax_bx.set_xticks(range(1, len(glass_keys_pmv) + 1))
        ax_bx.set_xticklabels(labels_bx, fontsize=8.5, rotation=15, ha='right')
        ax_bx.set_ylabel(ylabel, fontsize=9, labelpad=4)
        ax_bx.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax_bx.spines['top'].set_visible(False)
        ax_bx.spines['right'].set_visible(False)
        ax_bx.yaxis.grid(True, linewidth=0.3, color='#dddddd', linestyle='--', zorder=0)
        ax_bx.set_axisbelow(True)
        ax_bx.tick_params(axis='both', length=3, width=0.6)

        ref_handles = [Line2D([0],[0], color=r[1], linewidth=0.9,
                              linestyle=r[2], label=r[3])
                       for r in ref_lines if r[3] is not None]
        if ref_handles:
            ax_bx.legend(handles=ref_handles, loc='upper right',
                         fontsize=8, framealpha=0.9, edgecolor='#cccccc')

        fig_bx.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        plt.close(fig_bx)
        print(f'Saved → {save_path}')

    ## ── 新增：PMV / PPD 统计汇总表打印 ─────────────────────────────────
    _GLASS_LABELS_S = {
        'normal_glass':     'Normal glass',
        'tinted_glass':     'Tinted glass',
        'high_trans_glass': 'High-trans glass',
        'ec_abs_optimal':   'EC absorptive',
        'ec_ref_optimal':   'EC reflective',
    }
    _quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    for metric_s, m_label_s in [('pmv', 'PMV'), ('ppd', 'PPD (%)')]:
        rows_s = []
        for gk_s, glabel_s in _GLASS_LABELS_S.items():
            sub_s = df_box.loc[df_box['glass'] == gk_s, metric_s].dropna()
            if sub_s.empty: continue
            q_s = sub_s.quantile(_quantiles)
            rows_s.append({
                'Glazing':  glabel_s,
                'Mean':     round(sub_s.mean(), 3),
                'Std':      round(sub_s.std(),  3),
                'P10':      round(q_s[0.10], 3),
                'Q1 (25%)': round(q_s[0.25], 3),
                'Median':   round(q_s[0.50], 3),
                'Q3 (75%)': round(q_s[0.75], 3),
                'P90':      round(q_s[0.90], 3),
            })
        df_stats_s = pd.DataFrame(rows_s).set_index('Glazing')
        print(f'\n{"="*70}')
        print(f'  {m_label_s} — Statistics (all cities × months × orientations)')
        print(f'{"="*70}')
        print(df_stats_s.to_string())
        print()

    _boxplot_metric(
        df_box, 'pmv', 'PMV',
        ref_lines=[(0,    '#2980B9', '-',  'PMV = 0 (neutral)'),
                   (0.5,  '#CC0000', '--', 'PMV = +/-0.5 (comfort limit)'),
                   (-0.5, '#CC0000', '--', None)],
        title='PMV distribution by glazing scheme',
        save_path=os.path.join(FIG_DIR, 'fig_pmv_boxplot.png')
    )
    _boxplot_metric(
        df_box, 'ppd', 'PPD (%)',
        ref_lines=[(10, '#CC0000', '--', 'PPD = 10% (ISO 7730 comfort)')],
        title='PPD distribution by glazing scheme',
        save_path=os.path.join(FIG_DIR, 'fig_ppd_boxplot.png')
    )


# ──────────────────────────────────────────────────────────────────────
# 15. 图11：玻璃材料光学性能对比（堆叠柱状图）
#     图11a：遮阳组（着色/tinted 状态）
#     图11b：透光组（透明/clear 状态）
# ──────────────────────────────────────────────────────────────────────
def plot_fig11():
    """Recommendation Index (energy efficiency) — 两张蓝色世界地图"""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from matplotlib.colors import LinearSegmentedColormap
    import numpy.ma as ma

    cities_ri = [c for c in city_coords if c in df_avg['city'].unique()]
    annual_ri  = df_avg.groupby(['glass', 'city'])[['driving_km']].mean()
    ec_abs_ann = df_ec_abs.groupby('city')[['driving_km']].mean()
    ec_ref_ann = df_ec_ref.groupby('city')[['driving_km']].mean()

    def _build_ri_energy(ec_annual, label):
        records = []
        for city in cities_ri:
            try:
                tint_r = float(annual_ri.loc[('tinted_glass',     city), 'driving_km'])
                hite_r = float(annual_ri.loc[('high_trans_glass', city), 'driving_km'])
                ec_r   = float(ec_annual.loc[city, 'driving_km'])
                lon_c, lat_c = city_coords[city]
            except KeyError:
                continue
            delta_r = ec_r - max(tint_r, hite_r)
            records.append({'city': city, 'lon': lon_c, 'lat': lat_c, 'delta_r': delta_r})
        df = pd.DataFrame(records)
        mx = df['delta_r'].max()
        df['RI_E'] = df['delta_r'] / mx if mx > 0 else df['delta_r']
        print(f'[RI_E] {label}: {len(df)} 城市, [{df["RI_E"].min():.3f}, {df["RI_E"].max():.3f}]')
        return df

    df_ri_abs = _build_ri_energy(ec_abs_ann, 'Absorptive EC')
    df_ri_ref = _build_ri_energy(ec_ref_ann, 'Reflective EC')

    land_mask = _ensure_land_mask()

    cmap_blue = LinearSegmentedColormap.from_list(
        'ri_blue', ['#FFFFFF', '#C6DBEF', '#6BAED6', '#2171B5', '#084594'], N=256)
    cmap_blue.set_bad(color='none')

    def _draw_ri(df_plot, col, title, save_path):
        points = df_plot[['lon', 'lat']].values
        values = df_plot[col].values
        valid  = ~np.isnan(values)
        grid_v = griddata(points[valid], values[valid],
                          (lon_mesh, lat_mesh), method='linear')
        grid_v = np.where(land_mask, grid_v, np.nan)
        _tree  = cKDTree(points[valid])
        _dist, _ = _tree.query(
            np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), k=1)
        grid_v = np.where(_dist.reshape(lon_mesh.shape) <= 8.0, grid_v, np.nan)
        vmin = float(np.nanmin(grid_v))
        vmax = float(np.nanmax(grid_v))
        norm = mcolors.Normalize(vmin=max(0, vmin), vmax=vmax)
        fig = plt.figure(figsize=(7.2, 3.8))
        ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.OCEAN,     facecolor='#e8eef4', zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor='#f2f0eb', zorder=1)
        pcm = ax.pcolormesh(lon_mesh, lat_mesh, ma.masked_invalid(grid_v),
                            cmap=cmap_blue, norm=norm,
                            transform=ccrs.PlateCarree(),
                            shading='auto', zorder=2, rasterized=True)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor='#555555', zorder=5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.20, edgecolor='#888888', zorder=5)
        ax.gridlines(draw_labels=False, linewidth=0.25, color='#aaaaaa',
                     alpha=0.4, linestyle='--', zorder=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(0.5)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        cb = plt.colorbar(pcm, ax=ax, orientation='horizontal',
                          pad=0.04, fraction=0.038, aspect=45, extend='neither')
        cb.set_label('RI — energy efficiency (normalised, blue = better)',
                     fontsize=8.5, labelpad=3)
        cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=3))
        cb.ax.tick_params(labelsize=7.5, length=2.5, width=0.5)
        plt.tight_layout(pad=0.5)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'Saved → {save_path}')

    _draw_ri(df_ri_abs, 'RI_E',
             'Recommendation Index (energy efficiency) — Absorptive EC\n'
             'Normalised range improvement vs. best static glass',
             os.path.join(FIG_DIR, 'fig_RI_energy_abs.png'))
    _draw_ri(df_ri_ref, 'RI_E',
             'Recommendation Index (energy efficiency) — Reflective EC\n'
             'Normalised range improvement vs. best static glass',
             os.path.join(FIG_DIR, 'fig_RI_energy_ref.png'))
    print('RI(energy efficiency) 完成。')

    # ── RI(thermal comfort)：直接从 df_avg / df_ec_abs / df_ec_ref 计算 ──
    # 使用月均 mean_cabTair + mean_mrt 计算年均 PPD，无需外部 CSV

    def _pmv_ppd_ri11(T_a, T_mrt, I_cl=0.5, v_r=0.1, RH=50.0, M=1.2):
        M_W = M * 58.15; W = 0.0
        f_cl = 1.05 + 0.1*I_cl if I_cl > 0.5 else 1.0 + 0.2*I_cl
        t_cl = T_a + (35.5 - T_a) / (3.5*(6.45*I_cl + 0.1))
        for _ in range(100):
            h_c = max(2.38*abs(t_cl-T_a)**0.25, 12.1*v_r**0.5)
            t_cl_new = 35.7 - 0.028*(M_W-W) - I_cl*0.155*(
                3.96e-8*f_cl*((t_cl+273)**4-(T_mrt+273)**4) + f_cl*h_c*(t_cl-T_a))
            if abs(t_cl_new - t_cl) < 0.01: break
            t_cl = t_cl_new
        h_c = max(2.38*abs(t_cl-T_a)**0.25, 12.1*v_r**0.5)
        p_a = RH/100 * np.exp(16.6536 - 4030.183/(T_a+235))
        L = (M_W-W - 3.05e-3*(5733-6.99*(M_W-W)-p_a) - 0.42*((M_W-W)-58.15)
             - 1.7e-5*M_W*(5867-p_a) - 0.0014*M_W*(34-T_a)
             - 3.96e-8*f_cl*((t_cl+273)**4-(T_mrt+273)**4) - f_cl*h_c*(t_cl-T_a))
        pmv = float(np.clip((0.303*np.exp(-0.036*M_W)+0.028)*L, -3, 3))
        return float(100 - 95*np.exp(-0.03353*pmv**4 - 0.2179*pmv**2))

    def _city_annual_ppd11(city, df_src):
        ppd_vals = []
        for month in range(1, 13):
            I_cl = 1.0 if month in (11, 12, 1, 2, 3) else 0.5
            sub = df_src[(df_src['city'] == city) & (df_src['month'] == month)]
            if sub.empty: continue
            try:
                ppd_vals.append(_pmv_ppd_ri11(
                    float(sub['mean_cabTair'].mean()),
                    float(sub['mean_mrt'].mean()), I_cl=I_cl))
            except: pass
        return float(np.nanmean(ppd_vals)) if ppd_vals else float('nan')

    print('[RI_C] 计算全量城市年均PPD...')
    _glass_ppd11 = {}
    for gk in ['tinted_glass', 'high_trans_glass']:
        df_g = df_avg[df_avg['glass'] == gk]
        for city in cities_ri:
            _glass_ppd11[(city, gk)] = _city_annual_ppd11(city, df_g)
    for gk, df_s in [('ec_abs_optimal', df_ec_abs), ('ec_ref_optimal', df_ec_ref)]:
        for city in cities_ri:
            _glass_ppd11[(city, gk)] = _city_annual_ppd11(city, df_s)

    def _build_ri_comfort11(ec_key, label):
        records = []
        for city in cities_ri:
            pt = _glass_ppd11.get((city,'tinted_glass'),    float('nan'))
            ph = _glass_ppd11.get((city,'high_trans_glass'), float('nan'))
            pe = _glass_ppd11.get((city, ec_key),            float('nan'))
            if any(np.isnan([pt, ph, pe])): continue
            lon_c, lat_c = city_coords[city]
            delta_ppd = min(pt, ph) - pe
            records.append({'city': city, 'lon': lon_c, 'lat': lat_c,
                            'delta_ppd': delta_ppd})
        df = pd.DataFrame(records)
        mx = df['delta_ppd'].max()
        df['RI_C'] = df['delta_ppd'] / mx if mx > 0 else df['delta_ppd']
        print(f'[RI_C] {label}: {len(df)} 城市, [{df["RI_C"].min():.3f}, {df["RI_C"].max():.3f}]')
        return df

    df_ric_abs = _build_ri_comfort11('ec_abs_optimal', 'Absorptive EC')
    df_ric_ref = _build_ri_comfort11('ec_ref_optimal', 'Reflective EC')

    _draw_ri(df_ric_abs, 'RI_C',
             'Recommendation Index (thermal comfort) — Absorptive EC\n'
             'Normalised PPD improvement vs. best static glass',
             os.path.join(FIG_DIR, 'fig_RI_comfort_abs.png'))
    _draw_ri(df_ric_ref, 'RI_C',
             'Recommendation Index (thermal comfort) — Reflective EC\n'
             'Normalised PPD improvement vs. best static glass',
             os.path.join(FIG_DIR, 'fig_RI_comfort_ref.png'))
    print('RI(thermal comfort) 完成。')

    ## ── 新增：RI 差异图（Reflective − Absorptive）────────────────────────
    from matplotlib.colors import LinearSegmentedColormap as _LSC_diff
    cmap_diff = _LSC_diff.from_list(
        'ri_diff',
        ['#B2182B','#EF8A62','#FDDBC7','#FFFFFF','#D1E5F0','#4393C3','#2166AC'],
        N=256)
    cmap_diff.set_bad(color='none')

    def _draw_ri_diff(df_a, df_b, col, title, save_path):
        df_m = df_a[['city','lon','lat', col]].merge(
            df_b[['city', col]].rename(columns={col: col+'_b'}),
            on='city', how='inner')
        df_m['diff'] = df_m[col+'_b'] - df_m[col]
        points = df_m[['lon','lat']].values
        values = df_m['diff'].values
        valid  = ~np.isnan(values)
        grid_v = griddata(points[valid], values[valid],
                          (lon_mesh, lat_mesh), method='linear')
        grid_v = np.where(land_mask, grid_v, np.nan)
        _tree  = cKDTree(points[valid])
        _dist, _ = _tree.query(
            np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), k=1)
        grid_v = np.where(_dist.reshape(lon_mesh.shape) <= 20.0, grid_v, np.nan)
        abs_max_d = float(np.nanmax(np.abs(grid_v)))
        norm_d = mcolors.TwoSlopeNorm(vmin=-abs_max_d, vcenter=0, vmax=abs_max_d)
        fig = plt.figure(figsize=(7.2, 3.8))
        ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.OCEAN,     facecolor='#e8eef4', zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor='#f2f0eb', zorder=1)
        pcm = ax.pcolormesh(lon_mesh, lat_mesh, ma.masked_invalid(grid_v),
                            cmap=cmap_diff, norm=norm_d,
                            transform=ccrs.PlateCarree(),
                            shading='auto', zorder=2, rasterized=True)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor='#555555', zorder=5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.20, edgecolor='#888888', zorder=5)
        ax.gridlines(draw_labels=False, linewidth=0.25, color='#aaaaaa',
                     alpha=0.4, linestyle='--', zorder=3)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('#333333'); spine.set_linewidth(0.5)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        cb = plt.colorbar(pcm, ax=ax, orientation='horizontal',
                          pad=0.04, fraction=0.038, aspect=45, extend='both')
        cb.set_label('RI difference: Reflective − Absorptive  (blue = reflective better)',
                     fontsize=8, labelpad=3)
        cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=3))
        cb.ax.tick_params(labelsize=7.5, length=2.5, width=0.5)
        plt.tight_layout(pad=0.5)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'Saved → {save_path}')

    _draw_ri_diff(df_ri_abs, df_ri_ref, 'RI_E',
        'RI (energy efficiency): Reflective EC − Absorptive EC\n'
        'Blue = reflective better  |  Red = absorptive better',
        os.path.join(FIG_DIR, 'fig_RI_energy_diff.png'))

    _draw_ri_diff(df_ric_abs, df_ric_ref, 'RI_C',
        'RI (thermal comfort): Reflective EC − Absorptive EC\n'
        'Blue = reflective better  |  Red = absorptive better',
        os.path.join(FIG_DIR, 'fig_RI_comfort_diff.png'))


    ## ── CSV export: data_ri.csv ─────────────────────────────────────────
    _ri = df_ri_abs[['city','lon','lat','RI_E']].rename(columns={'RI_E':'RI_E_abs'})
    _ri = _ri.merge(
        df_ri_ref[['city','RI_E']].rename(columns={'RI_E':'RI_E_ref'}),
        on='city', how='outer')
    _ri = _ri.merge(
        df_ric_abs[['city','RI_C']].rename(columns={'RI_C':'RI_C_abs'}),
        on='city', how='outer')
    _ri = _ri.merge(
        df_ric_ref[['city','RI_C']].rename(columns={'RI_C':'RI_C_ref'}),
        on='city', how='outer')
    _ri['RI_E_diff'] = _ri['RI_E_ref'] - _ri['RI_E_abs']
    _ri['RI_C_diff'] = _ri['RI_C_ref'] - _ri['RI_C_abs']
    # fill lon/lat for cities only in ref
    _coord_fill = {c: (city_coords[c][0], city_coords[c][1])
                   for c in _ri['city'] if c in city_coords}
    _ri['lon'] = _ri.apply(
        lambda r: r['lon'] if not pd.isna(r['lon'])
        else _coord_fill.get(r['city'], (np.nan, np.nan))[0], axis=1)
    _ri['lat'] = _ri.apply(
        lambda r: r['lat'] if not pd.isna(r['lat'])
        else _coord_fill.get(r['city'], (np.nan, np.nan))[1], axis=1)
    _ri.round(4).to_csv(os.path.join(FIG_DIR, 'data_ri.csv'), index=False)
    print('CSV → data_ri.csv')
    print('RI 差异图完成。')




# ──────────────────────────────────────────────────────────────────────
# 附：全球城市仿真覆盖分布图
# ──────────────────────────────────────────────────────────────────────
def plot_city_coverage():
    """从 df_summary 提取所有城市经纬度，画全球覆盖分布图。"""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.lines import Line2D

    # 从city_coords全局字典获取经纬度（覆盖全量EPW城市，格式：{city:(lon,lat)}）
    _cities_cov = [c for c in city_coords if c in df_summary['city'].unique()]
    lons = np.array([city_coords[c][0] for c in _cities_cov])
    lats = np.array([city_coords[c][1] for c in _cities_cov])
    n_total = len(lons)
    print(f'[覆盖图] 总城市数：{n_total}')

    def lat_color(lat):
        a = abs(lat)
        if a >= 66.5: return '#4C72B0'
        if a >= 45:   return '#55A868'
        if a >= 23.5: return '#DD8452'
        return '#C44E52'

    colors = [lat_color(la) for la in lats]

    fig = plt.figure(figsize=(13, 6.5))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN,     facecolor='#e8eef4', zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor='#f2f0eb', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor='#666666', zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.20, edgecolor='#999999', zorder=3)

    ax.scatter(lons, lats,
               c=colors, s=5,
               alpha=0.70,
               linewidths=0.0,
               transform=ccrs.PlateCarree(),
               zorder=5)

    ax.gridlines(draw_labels=False, linewidth=0.3,
                 color='#aaaaaa', alpha=0.5, linestyle='--', zorder=2)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#333333'); spine.set_linewidth(0.6)

    legend_items = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#C44E52',
               markersize=6, label='Tropical  (|lat| < 23.5°)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#DD8452',
               markersize=6, label='Subtropical / Temperate  (23.5–45°)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#55A868',
               markersize=6, label='Cold / Boreal  (45–66.5°)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#4C72B0',
               markersize=6, label='Polar  (|lat| ≥ 66.5°)'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=7.5,
              framealpha=0.88, edgecolor='#cccccc',
              borderpad=0.6, handletextpad=0.5)

    ax.set_title(f'Global simulation coverage  —  {n_total} cities',
                 fontsize=10, fontweight='bold', pad=7)

    bands = {
        'Tropical  (|lat|<23.5°)':          np.sum(np.abs(lats) <  23.5),
        'Subtropical/Temperate (23.5–45°)': np.sum((np.abs(lats)>=23.5) & (np.abs(lats)<45)),
        'Cold/Boreal (45–66.5°)':           np.sum((np.abs(lats)>=45)   & (np.abs(lats)<66.5)),
        'Polar (|lat|≥66.5°)':              np.sum(np.abs(lats) >= 66.5),
    }
    print('\n── 各纬度带城市数 ──')
    for k, v in bands.items():
        print(f'  {k:<40s} {v:>5d}  ({v/n_total*100:.1f}%)')

    save_path = os.path.join(FIG_DIR, 'fig_city_coverage.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved → {save_path}')

if __name__ == '__main__':
    import time
    t0 = time.time()

    steps = [
        ('图1：月度分面图',          plot_fig1),
        ('图2：年均对比',            plot_fig2),
        ('图3：Nature 热力世界地图', plot_fig3),
        ('图4：EC 切换地图',         plot_fig4),
        ('图6：TMS 功耗分解',        plot_fig6),
        ('图7：热舒适散点图',        plot_fig7),
        ('图8：EC 节能气泡图',       plot_fig8),
        ('图9：经济性分析',          plot_fig9),
        ('图10：PMV/PPD 热舒适分析', plot_fig10),
        ('图11：RI 推荐指数地图',    plot_fig11),
        ('附：全球城市覆盖分布图',   plot_city_coverage),
    ]

    for name, func in steps:
        print(f'\n{"="*60}')
        print(f'▶ {name}')
        print('='*60)
        try:
            func()
        except Exception as e:
            import traceback
            print(f'[ERROR] {name} 失败：{e}')
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f'\n全部完成！耗时 {elapsed:.1f}s  图片目录：{FIG_DIR}')
