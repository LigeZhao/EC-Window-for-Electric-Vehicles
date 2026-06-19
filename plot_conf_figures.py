"""
plot_conf_figures.py — Nexus Forum 2026 会议论文配图脚本
==========================================================

本脚本是从 plot_results.py 精简而来的会议论文专用版本，与期刊版本严格隔离：
  - 只画 4 个代表城市（Singapore, Cairo, Beijing, Helsinki）
  - **只使用吸收式 EC（absorptive EC），不读取、不绘制反射式 EC 数据**
  - 不绘制任何全球地图、经济性分析、RI 推荐指数等期刊核心图
  - 图片输出到独立目录 conf_figures/，绝不污染期刊投稿用的 figure 目录

用法：
  python plot_conf_figures.py
  python plot_conf_figures.py --output-dir /path/to/results --fig-dir conf_figures

会议论文 docx 中对应的图：
  Fig. 2(a) — 4 城市年均 TMS 功耗柱状图              → fig_conf_tms_annual.png
  Fig. 2(b) — TMS 功耗相对 normal glass 的百分比降低 → fig_conf_tms_reduction.png
  Fig. 3    — 4 城市月度 MRT 折线图                   → fig_conf_mrt_monthly.png

  （会议论文 Fig. 1 是框架方框图，与本仿真无关，不在此脚本内生成）
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# 0. 命令行参数
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Nexus Forum 2026 会议论文配图（独立脚本）')
parser.add_argument('--output-dir', default='simulation result 0503/simulation_results',
                    help='Parquet 结果目录')
parser.add_argument('--fig-dir', default='conf_figures',
                    help='图片输出目录（独立于期刊版本，默认 conf_figures）')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
FIG_DIR    = args.fig_dir
os.makedirs(FIG_DIR, exist_ok=True)

print(f'输出目录：{FIG_DIR}（与期刊投稿配图目录隔离）')


# ──────────────────────────────────────────────────────────────────────
# 1. 全局字体设置
# ──────────────────────────────────────────────────────────────────────
import matplotlib.font_manager as _fm
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
# 2. 读取数据（兼容旧/新格式 city 列）
# ──────────────────────────────────────────────────────────────────────
print('读取汇总表...')
summary_path = os.path.join(OUTPUT_DIR, 'summary.parquet')
df_summary   = pd.read_parquet(summary_path)

# 兼容性：city 列若为 EPW 完整文件名，转换为城市短名
if df_summary['city'].str.contains('_').mean() > 0.5:
    if 'city_name' in df_summary.columns:
        print('[兼容] city 列为 EPW 完整文件名，替换为 city_name')
        df_summary['city'] = df_summary['city_name']
    else:
        def _extract_city(full_name):
            try:
                return full_name.split('.')[1]
            except Exception:
                return full_name
        print('[兼容] city 列为 EPW 完整文件名，从中提取城市名')
        df_summary['city'] = df_summary['city'].apply(_extract_city)

print(f'汇总表：{len(df_summary)} 条  |  城市：{df_summary["city"].nunique()}  |  '
      f'玻璃：{sorted(df_summary["glass"].unique())}')

# 四朝向平均月度摘要
df_avg = df_summary.groupby(['glass', 'city', 'month'])[
    ['driving_km', 'mean_sumPTMS', 'mean_mrt', 'mean_cabTair', 'mean_solar']
].mean().reset_index()


# ──────────────────────────────────────────────────────────────────────
# 3. EC 最优状态选择（仅吸收式 EC）
#
#    会议论文严格只展示 absorptive EC 的结果，
#    完全不读取 reflective EC 相关数据（也不构建对应的 DataFrame）。
# ──────────────────────────────────────────────────────────────────────
EC_PAIR_ABS = ['normal_glass_with_ec_trans', 'normal_glass_with_ec_colored']

def build_ec_optimal(df_avg, ec_pair_list):
    """按城市×月份选续航最优的 EC 状态"""
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
print(f'df_avg: {len(df_avg)} 条  |  df_ec_abs (吸收式EC月度最优): {len(df_ec_abs)} 条')


# ──────────────────────────────────────────────────────────────────────
# 4. 全局常量
# ──────────────────────────────────────────────────────────────────────
# 4 种静态玻璃 + 吸收式 EC 月度最优。共 4 个序列。
# 注意：完全不包含反射式 EC（'ec_ref_optimal' 序列被显式排除）。
glass_base = ['normal_glass', 'tinted_glass', 'high_trans_glass']

GLASS_STYLE = {
    'normal_glass':     {'color': '#2166AC', 'marker': 'o', 'label': 'Normal glass'},
    'tinted_glass':     {'color': '#D6604D', 'marker': 's', 'label': 'Tinted glass'},
    'high_trans_glass': {'color': '#4DAC26', 'marker': '^', 'label': 'High-trans glass'},
    'ec_abs_optimal':   {'color': '#762A83', 'marker': 'D', 'label': 'EC glazing'},
    # ↑ 会议论文里使用泛指 "EC glazing" 而非 "EC absorptive"，
    #   以避免在会议公开场合暗示 absorptive vs reflective 的对比框架。
}
glass_keys = list(GLASS_STYLE.keys())   # 4 项：3 静态 + 1 EC

CITIES_SEL_CFG = {
    'Tropical\n(Singapore)': 'Singapore',
    'Arid\n(Cairo)':         'Cairo',
    'Temperate\n(Beijing)':  'Beijing',
    'Cold\n(Helsinki)':      'Helsinki',
}


def _find_city(name, available):
    """大小写不敏感匹配城市名"""
    name_up = name.upper().replace(' ', '_').replace('-', '_')
    for c in available:
        c_up = c.upper().replace(' ', '_').replace('-', '_')
        if c_up == name_up:
            return c
    for c in available:
        c_up = c.upper().replace(' ', '_').replace('-', '_')
        if name_up in c_up or c_up.startswith(name_up):
            return c
    return None


def _resolve_cities_sel(cfg, available):
    result = {}
    for label, name in cfg.items():
        resolved = _find_city(name, available)
        if resolved:
            result[label] = resolved
        else:
            print(f'[警告] 城市 "{name}" 未在数据中找到，跳过')
    return result


# ──────────────────────────────────────────────────────────────────────
# 5. 会议论文 Fig. 3：4 城市月度 MRT 折线图
# ──────────────────────────────────────────────────────────────────────
def plot_conf_mrt_monthly():
    """对应会议论文 Fig. 3：4 个代表城市的月度 MRT 折线图，仅含吸收式 EC"""
    available  = df_avg['city'].unique()
    CITIES_SEL = _resolve_cities_sel(CITIES_SEL_CFG, available)
    if not CITIES_SEL:
        print('[跳过] MRT 月度图：代表城市均不在数据中')
        return

    n_cities = len(CITIES_SEL)
    n_cols   = min(4, n_cities)
    n_rows   = int(np.ceil(n_cities / n_cols))

    months       = sorted(df_avg['month'].unique().tolist())
    n_months     = len(months)
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'][:n_months]
    x = np.arange(n_months)
    MRT_COMFORT = 25.0

    def get_monthly_mrt(city):
        result = {}
        # 3 种静态玻璃
        for g in glass_base:
            vals = []
            for m in months:
                sub = df_avg[(df_avg['glass']==g) & (df_avg['city']==city) & (df_avg['month']==m)]
                vals.append(float(sub['mean_mrt'].iloc[0]) if not sub.empty else np.nan)
            result[g] = vals
        # 吸收式 EC 月度最优
        ec_vals = []
        for m in months:
            row = df_ec_abs[(df_ec_abs['city']==city) & (df_ec_abs['month']==m)]
            ec_vals.append(float(row['mean_mrt'].iloc[0]) if not row.empty else np.nan)
        result['ec_abs_optimal'] = ec_vals
        return result

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.6 * n_cols, 3.0 * n_rows + 1.2),
        gridspec_kw={'hspace': 0.42, 'wspace': 0.28,
                     'left': 0.08, 'right': 0.97,
                     'top': 0.92, 'bottom': 0.18}
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    def style_ax(ax, hide_left=False):
        ax.set_xlim(-0.55, n_months - 0.45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(axis='both', length=3, width=0.6, direction='out')
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(month_labels, fontsize=7.5)
        if hide_left:
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)

    for city_i, (climate_key, city) in enumerate(CITIES_SEL.items()):
        row_i = city_i // n_cols
        col_i = city_i %  n_cols
        hide  = (col_i > 0)
        ax    = axes[row_i, col_i]
        ax.set_title(climate_key, fontsize=9, fontweight='bold', pad=6, linespacing=1.4)
        if col_i == 0:
            ax.set_ylabel('MRT (°C)', fontsize=8.5, labelpad=4)
        mrt = get_monthly_mrt(city)
        for gk in glass_keys:
            st = GLASS_STYLE[gk]
            ax.plot(x, mrt[gk], color=st['color'], linewidth=1.5,
                    marker=st['marker'], markersize=3.8,
                    markerfacecolor='white', markeredgewidth=0.8, zorder=4)
        ax.axhline(MRT_COMFORT, color='#CC0000', linewidth=0.8, linestyle='--', alpha=0.75, zorder=3)
        if col_i == 0:
            ax.text(-0.5, MRT_COMFORT + 0.4, f'{MRT_COMFORT}°C comfort limit',
                    fontsize=6, color='#CC0000', va='bottom')
        style_ax(ax, hide_left=hide)
        all_v = [v for gk in glass_keys for v in mrt[gk] if not np.isnan(v)]
        if all_v:
            ax.set_ylim(0, max(all_v) * 1.12)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))

    for empty_i in range(n_cities, n_rows * n_cols):
        axes[empty_i // n_cols, empty_i % n_cols].set_visible(False)

    handles = [Line2D([0],[0], color=GLASS_STYLE[gk]['color'], linewidth=1.5,
                       marker=GLASS_STYLE[gk]['marker'], markersize=4,
                       markerfacecolor='white', markeredgewidth=0.8,
                       label=GLASS_STYLE[gk]['label'])
               for gk in glass_keys]
    handles.append(Line2D([0],[0], color='#CC0000', linewidth=0.8,
                           linestyle='--', label=f'Comfort limit ({MRT_COMFORT}°C)'))

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=min(len(handles), 5), fontsize=7.5, frameon=True, framealpha=0.9,
               edgecolor='#cccccc', borderpad=0.5, handlelength=1.4, columnspacing=1.0)

    out_path = os.path.join(FIG_DIR, 'fig_conf_mrt_monthly.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out_path}')


# ──────────────────────────────────────────────────────────────────────
# 6. 会议论文 Fig. 2(a) + 2(b)：年均 TMS 功耗 + 节能率
#
#    与期刊版本的 fig2 不同：
#    - 会议版本只有 2 个子图（TMS 绝对值 + TMS 节能率），上下排列
#    - 期刊版本是 2×2（额外含续航绝对值 + 续航提升率）
#    - 完全不出现反射式 EC
# ──────────────────────────────────────────────────────────────────────
def plot_conf_tms_panels():
    """对应会议论文 Fig. 2：年均 TMS 功耗（a）+ 相对 normal glass 的节能率（b）"""
    available  = df_avg['city'].unique()
    CITIES_SEL = _resolve_cities_sel(CITIES_SEL_CFG, available)
    if not CITIES_SEL:
        print('[跳过] 年均 TMS 图：代表城市均不在数据中')
        return

    city_list    = list(CITIES_SEL.values())
    climate_list = [k.replace('\n', ' ') for k in CITIES_SEL.keys()]
    n_c = len(city_list)
    x_c = np.arange(n_c)
    N_G = len(glass_keys)            # 4 个序列
    BAR_W = 0.18
    off_all = (np.arange(N_G) - (N_G - 1) / 2) * BAR_W

    # 节能率子图只对比 3 个非基线方案：tinted, high-trans, EC
    compare_keys = ['tinted_glass', 'high_trans_glass', 'ec_abs_optimal']
    off_cmp = (np.arange(len(compare_keys)) - (len(compare_keys) - 1) / 2) * BAR_W

    annual_city     = df_avg.groupby(['glass', 'city'])[['driving_km', 'mean_sumPTMS']].mean()
    ec_abs_ann_city = df_ec_abs.groupby('city')[['driving_km', 'mean_sumPTMS']].mean()

    def get_annual_mean(city, metric):
        result = {}
        for g in glass_base:
            try:    result[g] = annual_city.loc[(g, city), metric]
            except: result[g] = np.nan
        result['ec_abs_optimal'] = (ec_abs_ann_city.loc[city, metric]
                                     if city in ec_abs_ann_city.index else np.nan)
        return result

    def _style_ax(ax, xl):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.yaxis.grid(True, linewidth=0.3, color='#cccccc', linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=3, width=0.6)
        ax.set_xticks(x_c)
        ax.set_xticklabels(xl, fontsize=8)

    # ── 1×2 横向布局：左 = 绝对值，右 = 节能率 ─────────────────────────
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2),
        gridspec_kw={'wspace': 0.28,
                     'left': 0.07, 'right': 0.98,
                     'top': 0.90, 'bottom': 0.22})

    # (a) 年均 TMS 功耗绝对值
    ax = axes[0]
    for gi, gk in enumerate(glass_keys):
        vals = [get_annual_mean(city, 'mean_sumPTMS')[gk] for city in city_list]
        ax.bar(x_c + off_all[gi], vals, BAR_W, color=GLASS_STYLE[gk]['color'],
               edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3,
               label=GLASS_STYLE[gk]['label'])
    ax.set_title('(a) Annual mean TMS power', fontsize=9.5, fontweight='bold', pad=5)
    ax.set_ylabel('TMS power (W)', fontsize=8.5)
    ax.set_ylim(0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _style_ax(ax, climate_list)

    # (b) TMS 功耗相对 normal glass 的节能率
    ax = axes[1]
    for gi, gk in enumerate(compare_keys):
        savings = []
        for city in city_list:
            ann = get_annual_mean(city, 'mean_sumPTMS')
            base = ann['normal_glass']
            savings.append((base - ann[gk]) / base * 100
                           if (base and not np.isnan(base) and not np.isnan(ann[gk])) else np.nan)
        ax.bar(x_c + off_cmp[gi], savings, BAR_W, color=GLASS_STYLE[gk]['color'],
               edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3,
               label=GLASS_STYLE[gk]['label'])
    ax.axhline(0, color='#333333', linewidth=0.8, zorder=4)
    ax.set_title('(b) TMS power reduction vs. normal glass',
                 fontsize=9.5, fontweight='bold', pad=5)
    ax.set_ylabel('Reduction (%)', fontsize=8.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _style_ax(ax, climate_list)

    # 共享底部图例（4 项；节能率图里少 normal glass，但放完整 4 项以保持一致）
    handles = [mpatches.Patch(facecolor=GLASS_STYLE[gk]['color'], edgecolor='white',
                               linewidth=0.5, label=GLASS_STYLE[gk]['label'])
               for gk in glass_keys]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.00),
               ncol=4, fontsize=7.8, frameon=True, framealpha=0.9,
               edgecolor='#cccccc', borderpad=0.5, handlelength=1.4, columnspacing=1.2)

    out_path = os.path.join(FIG_DIR, 'fig_conf_tms_panels.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out_path}')

    # 也单独输出两张子图，方便会议论文按需替换
    # 子图 (a) 单独
    fig_a, ax_a = plt.subplots(1, 1, figsize=(5.6, 4.0))
    for gi, gk in enumerate(glass_keys):
        vals = [get_annual_mean(city, 'mean_sumPTMS')[gk] for city in city_list]
        ax_a.bar(x_c + off_all[gi], vals, BAR_W, color=GLASS_STYLE[gk]['color'],
                 edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3,
                 label=GLASS_STYLE[gk]['label'])
    ax_a.set_title('Annual mean TMS power', fontsize=10, fontweight='bold', pad=5)
    ax_a.set_ylabel('TMS power (W)', fontsize=9)
    ax_a.set_ylim(0)
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _style_ax(ax_a, climate_list)
    ax_a.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor='#cccccc',
                loc='upper right')
    out_a = os.path.join(FIG_DIR, 'fig_conf_tms_annual.png')
    fig_a.savefig(out_a, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_a)
    print(f'Saved → {out_a}')

    # 子图 (b) 单独
    fig_b, ax_b = plt.subplots(1, 1, figsize=(5.6, 4.0))
    for gi, gk in enumerate(compare_keys):
        savings = []
        for city in city_list:
            ann = get_annual_mean(city, 'mean_sumPTMS')
            base = ann['normal_glass']
            savings.append((base - ann[gk]) / base * 100
                           if (base and not np.isnan(base) and not np.isnan(ann[gk])) else np.nan)
        ax_b.bar(x_c + off_cmp[gi], savings, BAR_W, color=GLASS_STYLE[gk]['color'],
                 edgecolor='white', linewidth=0.3, alpha=0.88, zorder=3,
                 label=GLASS_STYLE[gk]['label'])
    ax_b.axhline(0, color='#333333', linewidth=0.8, zorder=4)
    ax_b.set_title('TMS power reduction vs. normal glass',
                   fontsize=10, fontweight='bold', pad=5)
    ax_b.set_ylabel('Reduction (%)', fontsize=9)
    ax_b.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _style_ax(ax_b, climate_list)
    ax_b.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor='#cccccc',
                loc='upper right')
    out_b = os.path.join(FIG_DIR, 'fig_conf_tms_reduction.png')
    fig_b.savefig(out_b, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_b)
    print(f'Saved → {out_b}')

    # CSV 数据导出（供论文 Table 引用核对）
    rows = []
    for gk in glass_keys:
        for city, clbl in zip(city_list, climate_list):
            ann_t = get_annual_mean(city, 'mean_sumPTMS')
            ann_r = get_annual_mean(city, 'driving_km')
            rows.append({
                'city': city, 'climate_label': clbl, 'glass': gk,
                'annual_tms_W':    ann_t.get(gk, np.nan),
                'annual_range_km': ann_r.get(gk, np.nan),
            })
    pd.DataFrame(rows).round(3).to_csv(
        os.path.join(FIG_DIR, 'data_conf_annual_4cities.csv'), index=False)
    print(f'CSV → {os.path.join(FIG_DIR, "data_conf_annual_4cities.csv")}')


# ──────────────────────────────────────────────────────────────────────
# 7. 入口
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    t0 = time.time()

    steps = [
        ('会议 Fig. 2(a)(b)：4 城市年均 TMS 功耗 + 节能率', plot_conf_tms_panels),
        ('会议 Fig. 3   ：4 城市月度 MRT 折线图',           plot_conf_mrt_monthly),
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
    print(f'\n全部完成！耗时 {elapsed:.1f}s')
    print(f'图片目录：{FIG_DIR}  （与期刊投稿配图严格隔离）')
