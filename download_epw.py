#!/usr/bin/env python3
"""
EnergyPlus EPW 气象数据批量下载脚本（全站点版）
数据来源：https://energyplus.net/weather
原理：从 NREL/EnergyPlus GitHub 仓库的 master.geojson 获取所有站点的下载链接

与原版 download_epw.py 的区别：
  - 原版：两轮去重（同台站留最优 + 同城市名只留一个）→ 约 1958 个城市
  - 本版：仅一轮去重（同台站不同数据源留最优），保留所有不同物理台站 → 约 2400+ 站点
  - 默认输出目录改为 weather data all（与原版区别，可在模拟时自由选择）

用法：
  python download_epw_all_stations.py                          # 下载全部站点
  python download_epw_all_stations.py --region China           # 只下载某个地区
  python download_epw_all_stations.py --list-regions           # 列出所有可用地区
  python download_epw_all_stations.py --output ./my_epw_data   # 指定输出目录
  python download_epw_all_stations.py --workers 10             # 并发下载线程数（默认 5）
  python download_epw_all_stations.py --clean                  # 清理目录中同台站重复文件
  python download_epw_all_stations.py --clean --dry-run        # 预览待删除文件（不实际删除）
"""

import json
import re
import os
import sys
import glob
import time
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

GEOJSON_URL     = "https://github.com/NREL/EnergyPlus/raw/develop/weather/master.geojson"
DEFAULT_OUTPUT  = "weather data all"   # ← 与原版 "weather data" 区分
DEFAULT_WORKERS = 5

# 数据源质量优先级（越靠前越优先，索引越小越好）
DATA_SOURCE_PREF = ['IWEC2', 'IWEC', 'TMY3', 'TMY2', 'CSWD', 'ITMY', 'SWERA', 'ISHRAE', 'TMY']


# ──────────────────────────────────────────────
# 数据获取与解析
# ──────────────────────────────────────────────

def fetch_geojson():
    print(f"正在获取站点列表：{GEOJSON_URL}")
    req = urllib.request.Request(GEOJSON_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(f"共找到 {len(data['features'])} 个条目")
    return data


def extract_url(html_snippet):
    if not html_snippet:
        return None
    match = re.search(r'href=[\'"]?([^\'" >]+)', html_snippet)
    return match.group(1) if match else None


def get_location_parts(feature):
    props   = feature.get("properties", {})
    title   = props.get("title", "Unknown")
    region  = props.get("region", "") or title.split("/")[0].strip() or "Unknown_Region"
    country = props.get("country", "") or "Unknown_Country"
    return region, country


def get_source_priority(source_or_url):
    text = source_or_url.upper()
    for i, src in enumerate(DATA_SOURCE_PREF):
        if src in text:
            return i
    return len(DATA_SOURCE_PREF)


# ──────────────────────────────────────────────
# 站点 key（仅用于第一轮台站去重）
# ──────────────────────────────────────────────

def _station_key_from_url(url):
    """
    从 URL 提取台站标识（去掉数据源后缀的文件名基名）。
    例：CHN_Beijing.Beijing.545110_CSWD → CHN_Beijing.Beijing.545110
    同一物理台站的 TMY/IWEC2/CSWD 等版本共享同一个 station_key。
    """
    if not url:
        return None
    filename = os.path.splitext(url.split('/')[-1])[0]
    for src in DATA_SOURCE_PREF:
        if filename.upper().endswith('_' + src):
            return filename[:-(len(src) + 1)]
    return filename


def _preferred(existing_feat, new_feat):
    """若 new_feat 的数据源优先级更高则返回 new_feat，否则返回 existing_feat"""
    existing_url = extract_url(existing_feat.get("properties", {}).get("epw", "") or "") or ""
    new_url      = extract_url(new_feat.get("properties",    {}).get("epw", "") or "") or ""
    if get_source_priority(new_url) < get_source_priority(existing_url):
        return new_feat
    return existing_feat


# ──────────────────────────────────────────────
# 去重逻辑（仅保留第一轮：台站级别去重）
# ──────────────────────────────────────────────

def dedupe_features_station_only(features):
    """
    仅做台站级别去重：同一物理台站（相同基名/WMO号）的多个数据源版本只保留最优。
    不做城市名级别去重，保留所有不同坐标的台站，最大化覆盖面。

    原版两轮去重结果约 1958 个，本版预计约 2400+ 个。
    """
    station_seen = {}   # station_key -> feature
    no_url       = []

    for feat in features:
        url = extract_url(feat.get("properties", {}).get("epw", "") or "")
        if not url:
            no_url.append(feat)
            continue
        sk = _station_key_from_url(url)
        if not sk:
            no_url.append(feat)
            continue
        if sk not in station_seen:
            station_seen[sk] = feat
        else:
            station_seen[sk] = _preferred(station_seen[sk], feat)

    result = list(station_seen.values()) + no_url
    print(f"台站去重完成：{len(features)} 个条目 → {len(result)} 个台站")
    print("（已取消城市名去重，同城市内多台站全部保留）")

    # 数据源分布统计
    source_count = {}
    for feat in result:
        url = extract_url(feat.get("properties", {}).get("epw", "") or "") or ""
        src = "其他"
        for s in DATA_SOURCE_PREF:
            if s in url.upper():
                src = s
                break
        source_count[src] = source_count.get(src, 0) + 1
    print("数据源分布：")
    for src, cnt in sorted(source_count.items(), key=lambda x: -x[1]):
        print(f"  {src:10s}: {cnt} 个台站")

    return result


# ──────────────────────────────────────────────
# --clean 模式（仅清理同台站重复，不清理同城市）
# ──────────────────────────────────────────────

def _parse_epw_file_info(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    source   = "其他"
    base     = filename
    for src in DATA_SOURCE_PREF:
        if filename.upper().endswith('_' + src):
            source = src
            base   = filename[:-(len(src) + 1)]
            break
    priority = get_source_priority(source)
    return base, source, priority


def clean_existing_duplicates(output_dir, dry_run=False):
    """
    仅清理同一台站（相同基名）的重复数据源文件，保留最优版本。
    不清理同城市名的不同台站——保留所有不同物理位置的台站。
    """
    epw_files = glob.glob(os.path.join(output_dir, '**', '*.epw'), recursive=True)
    if not epw_files:
        print(f"在 {output_dir} 下未找到任何 EPW 文件。")
        return

    print(f"共扫描到 {len(epw_files)} 个 EPW 文件")

    # 同台站（同目录 + 同基名）去重
    station_groups = {}
    for fp in epw_files:
        base, source, priority = _parse_epw_file_info(fp)
        group_key = (os.path.dirname(fp), base)
        station_groups.setdefault(group_key, []).append((priority, source, fp))

    to_delete = set()
    for group_key, candidates in station_groups.items():
        if len(candidates) <= 1:
            continue
        candidates.sort(key=lambda x: x[0])
        best_fp = candidates[0][2]
        for _, src, fp in candidates[1:]:
            to_delete.add(fp)
            print(f"  [台站重复] 删除 {os.path.basename(fp)}"
                  f"  (保留 {os.path.basename(best_fp)})")

    print(f"合计将删除 {len(to_delete)} 个重复文件，保留 {len(epw_files) - len(to_delete)} 个")

    if not to_delete:
        print("没有发现重复文件。")
        return

    if dry_run:
        print("\n[dry-run] 以上文件不会被实际删除，去掉 --dry-run 后重新运行以执行删除。")
        return

    deleted = 0
    for fp in to_delete:
        try:
            os.remove(fp)
            deleted += 1
        except Exception as e:
            print(f"  删除失败 {fp}: {e}")
    print(f"\n清理完成，实际删除 {deleted} 个文件。")


# ──────────────────────────────────────────────
# 下载任务构建与执行
# ──────────────────────────────────────────────

def list_regions(data):
    regions = set()
    for feature in data["features"]:
        region, _ = get_location_parts(feature)
        regions.add(region)
    for r in sorted(regions):
        print(r)


def download_file(url, dest_path, retries=3):
    if os.path.exists(dest_path):
        return dest_path, "skipped"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(dest_path, "wb") as f:
                    f.write(resp.read())
            return dest_path, "ok"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return dest_path, f"error: {e}"


def build_download_tasks(features, output_dir, region_filter=None):
    tasks = []
    for feature in features:
        props           = feature.get("properties", {})
        region, country = get_location_parts(feature)

        if region_filter and region_filter.lower() not in region.lower():
            continue

        url = extract_url(props.get("epw", ""))
        if url:
            safe_region  = re.sub(r'[<>:"/\\|?*]', "_", region)
            safe_country = re.sub(r'[<>:"/\\|?*]', "_", country)
            dest_dir     = os.path.join(output_dir, safe_region, safe_country)
            os.makedirs(dest_dir, exist_ok=True)
            filename = url.split("/")[-1]
            dest     = os.path.join(dest_dir, filename)
            tasks.append((url, dest, filename))

    return tasks


def run_downloads(tasks, workers):
    total = len(tasks)
    done = ok = skipped = 0
    errors = []

    print(f"\n开始下载，共 {total} 个 EPW 文件，并发线程数：{workers}\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_file, url, dest): name
                   for url, dest, name in tasks}
        for future in as_completed(futures):
            name = futures[future]
            dest_path, status = future.result()
            done += 1
            if status == "ok":
                ok += 1
                print(f"[{done}/{total}] ✓ {name}")
            elif status == "skipped":
                skipped += 1
                print(f"[{done}/{total}] — 跳过（已存在）: {name}")
            else:
                errors.append((name, status))
                print(f"[{done}/{total}] ✗ 失败: {name} | {status}")

    print(f"\n{'='*60}")
    print(f"下载完成：成功 {ok}，跳过 {skipped}，失败 {len(errors)}，共 {total} 个文件")
    if errors:
        print("\n失败文件列表：")
        for name, err in errors:
            print(f"  - {name}: {err}")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EnergyPlus EPW 气象数据批量下载工具（全站点版，保留所有不同台站）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--output",  "-o", default=DEFAULT_OUTPUT,
                        help=f"文件保存目录（默认：{DEFAULT_OUTPUT}）")
    parser.add_argument("--region",  "-r", default=None,
                        help="只下载指定大洲/地区（模糊匹配，如 China、Asia、Europe）")
    parser.add_argument("--list-regions", action="store_true",
                        help="列出所有可用大洲/地区后退出")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"并发下载线程数（默认：{DEFAULT_WORKERS}）")
    parser.add_argument("--clean",   action="store_true",
                        help="清理目录中同台站的重复数据源文件（保留最优版本）")
    parser.add_argument("--dry-run", action="store_true",
                        help="与 --clean 配合使用，只预览待删除文件，不实际删除")
    args = parser.parse_args()

    if args.clean:
        print(f"{'[dry-run] ' if args.dry_run else ''}清理目录：{os.path.abspath(args.output)}")
        clean_existing_duplicates(args.output, dry_run=args.dry_run)
        return

    try:
        data = fetch_geojson()
    except Exception as e:
        print(f"错误：无法获取站点列表 - {e}")
        sys.exit(1)

    if args.list_regions:
        print("\n可用大洲/地区列表：")
        list_regions(data)
        return

    # 仅做台站级去重，保留所有不同物理位置台站
    features = dedupe_features_station_only(data["features"])

    os.makedirs(args.output, exist_ok=True)

    tasks = build_download_tasks(
        features,
        output_dir=args.output,
        region_filter=args.region,
    )

    if not tasks:
        print("没有找到符合条件的文件，请检查 --region 参数。")
        return

    run_downloads(tasks, workers=args.workers)
    print(f"\n所有文件已保存到：{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
