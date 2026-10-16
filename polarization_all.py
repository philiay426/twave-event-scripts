#!/usr/bin/env python3
"""
批量三分量极化分析 — 11 台站
输出: 每台 polarization + rose 图 + 汇总地图
用法: python3 polarization_all.py
"""
import os, sys
sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, UTCDateTime
from obspy.geodetics.base import locations2degrees, gps2dist_azimuth, degrees2kilometers
from obspy.taup import TauPyModel
from t_phase_analysis.traveltime_profile import get_track, tt_t

PROC_DIR = "/home/philiay/shared/case/sacprocess"
OUT_DIR  = "/home/philiay/shared/case/sacprocess/polar_analysis"
EVLO, EVLA = -66.87, 18.99
SOURCE_DEPTH = 10.0

os.makedirs(OUT_DIR, exist_ok=True)

# ── 台站列表 ──
STATIONS = ['Y59A','W57A','U55A','S53A','Q51A','N48A',
            'N63A','K60A','J59A','H57A','F55A']


def load_3c(sta):
    traces = {}
    for chn in ['BHZ', 'BHN', 'BHE']:
        f = os.path.join(PROC_DIR, f"TA.{sta}..{chn}.M.2014.013.035104_proc.sac")
        if not os.path.exists(f): return None
        tr = read(f)[0]; traces[chn] = tr
    t0 = max(tr.stats.starttime for tr in traces.values())
    t1 = min(tr.stats.endtime for tr in traces.values())
    for chn in traces: traces[chn].trim(t0, t1)
    return traces


def polarization_flinn(data_z, data_n, data_e, fs, window_s=3.0):
    """Flinn (1965) — 方位角折到 0-180°"""
    ws = int(window_s * fs); step = max(ws // 4, 1)
    n_wins = (len(data_z) - ws) // step + 1
    t_mid = np.zeros(n_wins); rect = np.zeros(n_wins)
    azi = np.zeros(n_wins); inc = np.zeros(n_wins)

    for i in range(n_wins):
        i0 = i * step; i1 = i0 + ws
        t_mid[i] = (i0 + i1) / 2 / fs
        X = np.column_stack([data_z[i0:i1], data_n[i0:i1], data_e[i0:i1]]).T
        X = X - X.mean(axis=1, keepdims=True)
        C = X @ X.T / (ws - 1)
        e, v = np.linalg.eigh(C); e = e[::-1]; v = v[:, ::-1]
        l1, l2, l3 = e
        rect[i] = 1.0 - (l2 + l3) / (2 * l1) if l1 > 0 else 0
        v1 = v[:, 0]
        az = np.degrees(np.arctan2(v1[2], v1[1]))
        if az < 0: az += 360
        if az >= 180: az -= 180
        azi[i] = az
        inc[i] = np.degrees(np.arccos(np.clip(np.abs(v1[0]), 0, 1)))
    return t_mid, rect, azi, inc


def analyze_station(sta):
    """单台极化分析，返回汇总结果"""
    traces = load_3c(sta)
    if traces is None:
        print(f"  {sta:6s}  ❌ 缺三分量数据")
        return None

    tr_z = traces['BHZ']; fs = tr_z.stats.sampling_rate; sac = tr_z.stats.sac
    origin = UTCDateTime(
        year=sac.nzyear, julday=sac.nzjday, hour=sac.nzhour, minute=sac.nzmin,
        second=sac.nzsec, microsecond=(getattr(sac, 'nzmsec', 0) or 0) * 1000
    ) + (getattr(sac, 'o', 0) or 0)
    stla, stlo = sac.stla, sac.stlo
    dist_deg = locations2degrees(EVLA, EVLO, stla, stlo)
    dist_km = degrees2kilometers(dist_deg)
    _, baz, _ = gps2dist_azimuth(stla, stlo, EVLA, EVLO)

    # 理论到时
    taup = TauPyModel(model='iasp91')
    arrs = taup.get_travel_times(source_depth_in_km=SOURCE_DEPTH,
                                  distance_in_degree=dist_deg, phase_list=['P', 'S'])
    tt_p = tt_s = None
    for a in arrs:
        if a.name == 'P' and tt_p is None: tt_p = a.time
        elif a.name == 'S' and tt_s is None: tt_s = a.time

    track = get_track(EVLO, EVLA, stlo, stla, 0.02, ref_el=-1000,
                      plot_map=False, plot_elevation=False)
    tr_tt = tt_t(track, ref_el=-1000, source_depth_km=SOURCE_DEPTH,
                 ignore_small_islands=True)
    tt_tp = tr_tt['tt_t'] if tr_tt else None

    # 极化
    data_z = tr_z.data.astype(np.float64)
    data_n = traces['BHN'].data.astype(np.float64)
    data_e = traces['BHE'].data.astype(np.float64)
    t_mid, rect, azi_180, inc = polarization_flinn(data_z, data_n, data_e, fs)

    # 窗口到时的偏振方向
    def get_peak_azimuth(t_center, half_win):
        mask = (t_mid >= t_center - half_win) & (t_mid <= t_center + half_win)
        mask = mask & (rect > 0.4)
        if mask.sum() == 0: return None
        # 取中位数方位角
        return np.median(azi_180[mask])

    p_az = get_peak_azimuth(tt_p, 5.0) if tt_p else None
    s_az = get_peak_azimuth(tt_s, 3.0) if tt_s else None
    t_az = get_peak_azimuth(tt_tp, 3.0) if tt_tp else None

    # ── 画图 ──
    b = tr_z.stats.starttime - origin
    t_rel = t_mid + b
    t_vec = np.arange(len(data_z)) / fs + b
    colors_pol = np.where(rect > 0.4, 'red', 'gray')

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    ax = axes[0]
    for chn, d, c, yoff in [('BHZ', data_z, 'black', 2),
                              ('BHN', data_n, 'blue',  0),
                              ('BHE', data_e, 'red',  -2)]:
        ax.plot(t_vec, d / max(np.abs(d).max(), 1e-20) + yoff,
                color=c, lw=0.3, alpha=0.8, label=chn)
    for t_arr, lbl, clr in [(tt_p,'P','green'),(tt_s,'S','orange'),
                              (tt_tp,'T-P','blue')]:
        if t_arr: ax.axvline(t_arr, color=clr, lw=1, ls='--', alpha=0.6)
    ax.set_ylabel('Norm.'); ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.15)
    ax.set_title(f'{sta}  {dist_km:.0f} km  P={tt_p:.0f}s  T-P={tt_tp:.0f}s' if tt_tp else f'{sta}  {dist_km:.0f} km',
                 fontsize=10, fontweight='bold')

    for ax_i, (data, ylabel, ylim) in enumerate([
        (rect,   'Rectilinearity', (0, 1.05)),
        (azi_180,'Azimuth 0-180°', (0, 180)),
        (inc,    'Incidence (°)',  (0, 90)),
    ]):
        ax = axes[ax_i + 1]; ax.scatter(t_rel, data, c=colors_pol, s=2, alpha=0.5, edgecolors='none')
        ax.set_ylabel(ylabel, fontsize=8); ax.set_ylim(*ylim)
        if ax_i == 1:
            ax.axhline(y=baz % 180, color='green', lw=0.8, ls='--', alpha=0.5)
        for t_arr in [tt_p, tt_s, tt_tp]:
            if t_arr: ax.axvline(t_arr, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax.grid(True, alpha=0.15)
    axes[3].set_xlabel('Time (s)'); axes[0].set_xlim(-500, 3500)
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, f'polarization_{sta}.png')
    fig.savefig(out1, dpi=150, bbox_inches='tight'); plt.close(fig)

    # ── 玫瑰图 ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), subplot_kw={'projection': 'polar'})
    for ax, (lbl, tc, hw, clr) in zip(axes2, [
        ('P', tt_p, 5.0, 'green'), ('S', tt_s, 3.0, 'orange'), ('T', tt_tp, 3.0, 'blue')]):
        if tc is None: continue
        m = (t_mid >= tc - hw) & (t_mid <= tc + hw) & (rect > 0.4)
        if m.sum() == 0: continue
        az = azi_180[m]; bins = np.arange(0, 181, 10)
        hist, _ = np.histogram(az, bins=bins)
        ax.bar(np.radians(bins[:-1] + 5), hist, width=np.radians(10),
               color=clr, alpha=0.7, edgecolor='white', lw=0.3)
        ax.axvline(np.radians(baz % 180), color='red', lw=1.2, ls='--', alpha=0.7)
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_title(f'{lbl} (±{hw}s) n={m.sum()}', fontsize=9, fontweight='bold')
    fig2.suptitle(f'{sta}  Rose (0-180°), BAZ folded={baz%180:.0f}°', fontsize=10, fontweight='bold')
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, f'rose_{sta}.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight'); plt.close(fig2)

    return {'sta': sta, 'stla': stla, 'stlo': stlo, 'dist_km': dist_km,
            'baz': baz, 'p_az': p_az, 's_az': s_az, 't_az': t_az,
            'tt_p': tt_p, 'tt_tp': tt_tp}


def plot_summary_map(results):
    """汇总地图: 台站位置 + 偏振方向箭头 (纯 matplotlib)"""
    fig, ax = plt.subplots(figsize=(12, 14))

    # 粗略海岸线背景
    coast_lon = np.arange(-82, -63, 0.1)
    # 简化: 画一个矩形表示陆地区域
    ax.set_xlim(-82, -64)
    ax.set_ylim(15, 48)
    ax.set_aspect(1.0 / np.cos(np.radians(32)))  # 中纬度投影修正

    ax.set_xlabel('Longitude (°W)'); ax.set_ylabel('Latitude (°N)')
    ax.grid(True, alpha=0.3, linestyle=':')

    # 震源
    ax.plot(EVLO, EVLA, 'r*', markersize=16, zorder=5)
    ax.text(EVLO + 0.3, EVLA - 0.3, 'Mw6.3\nPuerto Rico',
            fontsize=9, fontweight='bold', color='red')

    arrow_len = 1.5  # 箭头长度 (度)

    for r in results:
        if r is None: continue
        ax.plot(r['stlo'], r['stla'], 'k^', markersize=8, zorder=4)
        ax.text(r['stlo'] + 0.2, r['stla'] + 0.2, r['sta'], fontsize=6.5)

        for az, color, label in [(r['p_az'], '#2ca02c', 'P'),
                                   (r['t_az'], '#1f77b4', 'T')]:
            if az is None: continue
            rad = np.radians(az)
            dx = arrow_len * np.sin(rad)
            dy = arrow_len * np.cos(rad)
            ax.arrow(r['stlo'], r['stla'], dx, dy,
                     head_width=0.4, head_length=0.5, fc=color, ec=color,
                     alpha=0.7, lw=1.0, length_includes_head=True, zorder=5)

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=2, label='P-wave polarization'),
        Line2D([0], [0], color='#1f77b4', lw=2, label='T-wave polarization'),
        Line2D([0], [0], marker='*', color='red', lw=0, markersize=12, label='Epicenter'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.85)
    ax.set_title('P & T-wave Polarization Directions — 2014-01-13 Mw6.3 Puerto Rico',
                 fontsize=12, fontweight='bold')

    # 震源-台站大圆路径 (选画几条)
    for r in results[:5]:
        if r is None: continue
        ax.plot([EVLO, r['stlo']], [EVLA, r['stla']], 'gray', lw=0.3, alpha=0.3, zorder=1)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'summary_map.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"\n✅ {out}")


def main():
    results = []
    for sta in STATIONS:
        print(f"── {sta} ──", end=" ", flush=True)
        try:
            r = analyze_station(sta)
            if r:
                print(f"P_az={r['p_az']:.0f}°" if r['p_az'] else "P_az=None",
                      f"T_az={r['t_az']:.0f}°" if r['t_az'] else "T_az=None")
            else:
                print("SKIP")
            results.append(r)
        except Exception as e:
            print(f"❌ {str(e)[:60]}")
            results.append(None)

    valid = [r for r in results if r is not None]
    print(f"\n完成: {len(valid)}/{len(STATIONS)} 台站")

    # 汇总表
    print(f"\n{'Station':<8} {'Dist':>6} {'BAZ':>5} {'P_az':>6} {'T_az':>6}")
    print("-" * 40)
    for r in valid:
        p_str = f"{r['p_az']:.0f}°" if r['p_az'] else "  -"
        t_str = f"{r['t_az']:.0f}°" if r['t_az'] else "  -"
        print(f"{r['sta']:<8} {r['dist_km']:>6.0f} {r['baz']:>5.0f} {p_str:>6} {t_str:>6}")

    # 汇总地图
    if len(valid) >= 2:
        plot_summary_map(results)


if __name__ == "__main__":
    main()
