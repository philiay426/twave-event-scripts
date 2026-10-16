#!/usr/bin/env python3
"""
震源 → 台站 高程剖面图（通用版）
用法:
  python3 plot_profile.py <sac_file>                     # 从 SAC 头段读取台站坐标
  python3 plot_profile.py --stla 43.46 --stlo -74.50 --sta J59A  # 手动指定
"""
import sys
import os
import argparse

sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from obspy.geodetics.base import degrees2kilometers
from obspy import read

from t_phase_analysis.traveltime_profile import get_track, tt_t

# ===== 默认参数 =====
DEFAULT_EVLO = -66.87
DEFAULT_EVLA = 18.99
DEFAULT_SOURCE_DEPTH = 10.0
DEFAULT_DIST_INT = 0.02
DEFAULT_REF_EL = -1000
DEFAULT_OUTDIR = "/mnt/f/sharedate/case"


def parse_args():
    p = argparse.ArgumentParser(
        description='绘制震源→台站 高程剖面 + T波走时')
    p.add_argument('sac_file', nargs='?', default=None,
                   help='SAC 文件路径 (从中读取台站坐标)')
    p.add_argument('--sta', default=None, help='台站名')
    p.add_argument('--stla', type=float, default=None, help='台站纬度')
    p.add_argument('--stlo', type=float, default=None, help='台站经度')
    p.add_argument('--evlo', type=float, default=DEFAULT_EVLO, help='震源经度')
    p.add_argument('--evla', type=float, default=DEFAULT_EVLA, help='震源纬度')
    p.add_argument('--depth', type=float, default=DEFAULT_SOURCE_DEPTH,
                   help='震源深度 (km)')
    p.add_argument('--ref-el', type=float, default=DEFAULT_REF_EL,
                   help='海岸参考高程 (m), 低于此=海洋')
    p.add_argument('--dist-int', type=float, default=DEFAULT_DIST_INT,
                   help='采样间隔 (度)')
    p.add_argument('--outdir', default=DEFAULT_OUTDIR, help='输出目录')
    p.add_argument('--prefix', default='profile', help='输出文件名前缀')
    return p.parse_args()


def get_station_info(args):
    """从 SAC 文件或命令行获取台站信息"""
    if args.sac_file:
        if not os.path.exists(args.sac_file):
            sys.exit(f"❌ SAC 文件不存在: {args.sac_file}")
        st = read(args.sac_file, headonly=True)
        sac = st[0].stats.sac
        sta = args.sta or sac.kstnm.strip()
        stla = args.stla or sac.stla
        stlo = args.stlo or sac.stlo
        print(f"从 SAC 头段读取: {sta} ({stla:.4f}°N, {stlo:.4f}°E)")
    else:
        if args.stla is None or args.stlo is None:
            sys.exit("❌ 请指定 --stla --stlo (或提供 SAC 文件路径)")
        sta = args.sta or 'UNKNOWN'
        stla = args.stla
        stlo = args.stlo

    return sta, stla, stlo


# ===== 主流程 =====
def main():
    args = parse_args()
    sta, stla, stlo = get_station_info(args)

    os.makedirs(args.outdir, exist_ok=True)

    # ── 提取高程剖面 ──
    print(f"提取大圆路径高程 ({args.evla}°N,{args.evlo}°E → {stla}°N,{stlo}°E)...")
    track = get_track(args.evlo, args.evla, stlo, stla, args.dist_int,
                      ref_el=args.ref_el, plot_map=False, plot_elevation=False)

    # ── 计算T相走时 ──
    print("计算T相走时...")
    result = tt_t(track, ref_el=args.ref_el, source_depth_km=args.depth,
                  ignore_small_islands=True)

    dist_km = degrees2kilometers(track.p.values)
    elev_m = track.el.values
    total_dist_km = dist_km[-1]

    # ── 识别海岸线 ──
    ocean_mask = elev_m <= args.ref_el
    land_mask = elev_m > args.ref_el

    transitions = []
    for i in range(1, len(dist_km)):
        if ocean_mask[i - 1] != ocean_mask[i]:
            transitions.append(dist_km[i])

    # ── 制图 ──
    fig, ax = plt.subplots(figsize=(14, 5))

    # 填充海洋
    ax.fill_between(dist_km, elev_m, -11000,
                    where=ocean_mask, color='lightblue', alpha=0.6, lw=0)
    # 填充陆地
    ax.fill_between(dist_km, elev_m, -11000,
                    where=land_mask, color='sandybrown', alpha=0.5, lw=0)

    # 高程曲线
    ax.plot(dist_km, elev_m, 'k-', lw=0.8)

    # 海平面 + 参考面
    ax.axhline(y=0, color='dodgerblue', lw=1, ls='--', alpha=0.7, label='Sea level')
    ax.axhline(y=args.ref_el, color='gray', lw=0.8, ls=':', alpha=0.5,
               label=f'Ref. level ({args.ref_el} m)')

    # 海岸转换点
    for d in transitions:
        ax.axvline(x=d, color='red', lw=0.8, ls='--', alpha=0.5)
        ax.text(d, 7000, f'{d:.0f} km', ha='center', fontsize=7, color='red',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # 震源 & 台站
    ax.plot(0, 0, 'r*', markersize=14, zorder=5, label='Source')
    ax.plot(total_dist_km, 0, 'r^', markersize=10, zorder=5, label=f'Station {sta}')
    ax.text(0, 8000, f'Source\n{args.evla:.2f}°N, {abs(args.evlo):.2f}°W',
            ha='center', fontsize=8, fontweight='bold')
    ax.text(total_dist_km, 8000, f'{sta}\n{stla:.2f}°N, {abs(stlo):.2f}°W',
            ha='center', fontsize=8, fontweight='bold')

    # 坐标轴
    ax.set_xlim(0, total_dist_km * 1.02)
    ax.set_ylim(-11000, 10000)
    ax.set_xlabel('Distance along great circle (km)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)

    # 走时信息
    tt_total_min = result['tt_t'] / 60
    tt_ocean_min = result['tt_ocean'] / 60
    tt_land_min = result['tt_land'] / 60
    dist_ocean_km = degrees2kilometers(result['dist_ocean'])
    dist_land_km = degrees2kilometers(result['dist_land'])

    title = (
        f'Source → {sta}  Elevation Profile  |  '
        f'Total dist: {total_dist_km:.0f} km  |  '
        f'T-wave TT: {tt_total_min:.1f} min  '
        f'(ocean: {dist_ocean_km:.0f} km / {tt_ocean_min:.1f} min, '
        f'land: {dist_land_km:.0f} km / {tt_land_min:.1f} min)'
    )
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.15, linestyle=':')
    ax.tick_params(labelsize=9)

    # 度数轴
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    deg_ticks_km = np.linspace(0, total_dist_km, 6)
    ax2.set_xticks(deg_ticks_km)
    ax2.set_xticklabels([f'{d/111.19:.1f}°' for d in deg_ticks_km], fontsize=8)
    ax2.set_xlabel('Distance (degrees)', fontsize=10, labelpad=8)

    plt.tight_layout()

    # ── 输出 ──
    out_png = os.path.join(args.outdir, f"{args.prefix}_{sta}.png")
    out_pdf = os.path.join(args.outdir, f"{args.prefix}_{sta}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ PNG: {out_png}")
    print(f"✅ PDF: {out_pdf}")

    # ── 走时详情 ──
    print(f"\n{'='*10} 走时计算 {'='*10}")
    print(f"  台站:         {sta} ({stla:.4f}°N, {stlo:.4f}°E)")
    print(f"  总距离:       {total_dist_km:.1f} km ({track.p.iloc[-1]:.2f}°)")
    print(f"  海洋段距离:   {dist_ocean_km:.1f} km ({result['dist_ocean']:.2f}°)")
    print(f"  陆地段距离:   {dist_land_km:.1f} km ({result['dist_land']:.2f}°)")
    print(f"  海洋段走时:   {tt_ocean_min:.1f} min ({result['tt_ocean']:.0f} s)")
    print(f"  陆地P波走时:  {tt_land_min:.1f} min ({result['tt_land']:.0f} s)")
    print(f"  陆地S波走时:  {result['tt_land_s']/60:.1f} min ({result['tt_land_s']:.0f} s)")
    print(f"  T相总走时:    {tt_total_min:.1f} min ({result['tt_t']:.0f} s)")
    print(f"  海岸转换点:   {[f'{d:.0f} km' for d in transitions]}")


if __name__ == "__main__":
    main()
