#!/usr/bin/env python3
"""
单台三分量极化分析 (通用版)
Flinn (1965) 协方差矩阵法
输出: 极化属性离散点 + P/S/T 偏振玫瑰图 (0-180°)
用法: python3 polarization.py --sta J59A
"""
import os, sys, argparse
sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, UTCDateTime
from obspy.geodetics.base import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
from t_phase_analysis.traveltime_profile import get_track, tt_t

PROC_DIR = "/home/philiay/shared/case/sacprocess"
OUT_DIR  = "/home/philiay/shared/case/sacprocess/polar_analysis"
EVLO, EVLA = -66.87, 18.99
SOURCE_DEPTH = 10.0


def parse_args():
    p = argparse.ArgumentParser(description='三分量极化分析')
    p.add_argument('--sta', default='J59A')
    p.add_argument('--win', type=float, default=3.0, help='滑动窗长 (s)')
    p.add_argument('--tmin', type=float, default=-500)
    p.add_argument('--tmax', type=float, default=3500)
    p.add_argument('--p-win', type=float, default=5.0, help='P波玫瑰图窗半宽 (s)')
    p.add_argument('--s-win', type=float, default=3.0, help='S波玫瑰图窗半宽 (s)')
    p.add_argument('--t-win', type=float, default=3.0, help='T波玫瑰图窗半宽 (s)')
    p.add_argument('--r-min', type=float, default=0.4, help='线性度最低阈值')
    return p.parse_args()


def load_3c(sta):
    traces = {}
    for chn in ['BHZ', 'BHN', 'BHE']:
        f = os.path.join(PROC_DIR, f"TA.{sta}..{chn}.M.2014.013.035104_proc.sac")
        if not os.path.exists(f): sys.exit(f"❌ 缺少 {chn}: {f}")
        tr = read(f)[0]; traces[chn] = tr
    t0 = max(tr.stats.starttime for tr in traces.values())
    t1 = min(tr.stats.endtime   for tr in traces.values())
    for chn in traces: traces[chn].trim(t0, t1)
    return traces


def polarization_flinn(data_z, data_n, data_e, fs, window_s=3.0):
    """Flinn (1965) — 方位角折到 0-180° (偏振是轴，非矢量)"""
    win_samp = int(window_s * fs)
    step = max(win_samp // 4, 1)
    n_wins = (len(data_z) - win_samp) // step + 1

    t_mid = np.zeros(n_wins); rect = np.zeros(n_wins)
    plan = np.zeros(n_wins); azi = np.zeros(n_wins); inc = np.zeros(n_wins)

    for i in range(n_wins):
        i0 = i * step; i1 = i0 + win_samp
        t_mid[i] = (i0 + i1) / 2 / fs
        X = np.column_stack([data_z[i0:i1], data_n[i0:i1], data_e[i0:i1]]).T
        X = X - X.mean(axis=1, keepdims=True)
        C = X @ X.T / (win_samp - 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
        l1, l2, l3 = eigvals
        rect[i] = 1.0 - (l2 + l3) / (2 * l1) if l1 > 0 else 0
        plan[i] = 1.0 - 2 * l3 / (l1 + l2) if (l1 + l2) > 0 else 0
        v1 = eigvecs[:, 0]  # (Z, N, E)
        azi[i] = np.degrees(np.arctan2(v1[2], v1[1]))
        # 折到 0-180° (偏振方向 ±v 等价)
        if azi[i] < 0: azi[i] += 360
        if azi[i] >= 180: azi[i] -= 180
        inc[i] = np.degrees(np.arccos(np.clip(np.abs(v1[0]), 0, 1)))
    return t_mid, rect, plan, azi, inc


def plot_rose(ax, t_mid, azi, rect, t_center, win_half, r_min, sta, phase_label, color, baz):
    """画单个偏振玫瑰图 (0-180°)"""
    mask = (t_mid >= t_center - win_half) & (t_mid <= t_center + win_half)
    mask = mask & (rect > r_min)
    az_high = azi[mask]

    if len(az_high) == 0:
        ax.set_title(f'{phase_label} (±{win_half}s, R>{r_min}) — 0 pts', fontsize=9)
        return

    bins = np.arange(0, 181, 10)
    hist, _ = np.histogram(az_high, bins=bins)
    theta = np.radians(bins[:-1] + 5)
    width = np.radians(10)
    ax.bar(theta, hist, width=width, color=color, alpha=0.7, edgecolor='white', lw=0.3)

    # back-azimuth (±180° 等价)
    baz_folded = baz % 180
    ax.axvline(x=np.radians(baz_folded), color='red', lw=1.5, ls='--', alpha=0.7,
               label=f'BAZ {baz:.0f}°')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f'{sta}  {phase_label} (±{win_half}s, R>{r_min})  n={len(az_high)}',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')


def main():
    args = parse_args()
    sta = args.sta; os.makedirs(OUT_DIR, exist_ok=True)

    print(f"加载 {sta} 三分量...")
    traces = load_3c(sta)
    tr_z = traces['BHZ']; fs = tr_z.stats.sampling_rate
    sac = tr_z.stats.sac

    origin = UTCDateTime(
        year=sac.nzyear, julday=sac.nzjday, hour=sac.nzhour, minute=sac.nzmin,
        second=sac.nzsec, microsecond=(getattr(sac, 'nzmsec', 0) or 0) * 1000
    ) + (getattr(sac, 'o', 0) or 0)

    stla, stlo = sac.stla, sac.stlo
    dist_deg = locations2degrees(EVLA, EVLO, stla, stlo)
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
    tt_ts = tr_tt['tt_ocean'] + tr_tt['tt_land_s'] if tr_tt else None
    print(f"  P={tt_p:.0f}s  S={tt_s:.0f}s  T-P={tt_tp:.0f}s  T-S={tt_ts:.0f}s  BAZ={baz:.0f}°")

    # 极化分析
    print(f"极化分析 (窗长 {args.win}s)...")
    data_z = tr_z.data.astype(np.float64)
    data_n = traces['BHN'].data.astype(np.float64)
    data_e = traces['BHE'].data.astype(np.float64)

    t_mid, rect, plan, azi_180, inc = polarization_flinn(
        data_z, data_n, data_e, fs, args.win)

    b = tr_z.stats.starttime - origin
    t_rel = t_mid + b
    t_vec = np.arange(len(data_z)) / fs + b

    # ========== 图1: 极化属性 (离散点) ==========
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    colors_pol = np.where(rect > args.r_min, 'red', 'gray')
    ms = 2

    ax = axes[0]
    for chn, d, c, yoff in [('BHZ', data_z, 'black', 2),
                              ('BHN', data_n, 'blue',  0),
                              ('BHE', data_e, 'red',  -2)]:
        ax.plot(t_vec, d / max(np.abs(d).max(), 1e-20) + yoff,
                color=c, lw=0.3, alpha=0.8, label=chn)
    for t_arr, lbl, clr in [(tt_p,'P','green'),(tt_s,'S','orange'),
                              (tt_tp,'T-P','#1f77b4'),(tt_ts,'T-S','#d62728')]:
        if t_arr: ax.axvline(t_arr, color=clr, lw=1, ls='--', alpha=0.6)
    ax.set_ylabel('Norm.'); ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.set_title(f'{sta}  3-C  |  P={tt_p:.0f}s  S={tt_s:.0f}s  T-P={tt_tp:.0f}s',
                 fontsize=10, fontweight='bold'); ax.grid(True, alpha=0.15)

    for ax_i, (data, ylabel, ylim) in enumerate([
        (rect,   'Rectilinearity', (0, 1.05)),
        (azi_180, 'Azimuth 0-180°', (0, 180)),
        (inc,    'Incidence (°)',  (0, 90)),
    ]):
        ax = axes[ax_i + 1]
        ax.scatter(t_rel, data, c=colors_pol, s=ms**2, alpha=0.6, edgecolors='none')
        ax.set_ylabel(ylabel, fontsize=9); ax.set_ylim(*ylim)
        if ax_i == 1:  # azimuth: 标 back-azimuth
            baz_folded = baz % 180
            ax.axhline(y=baz_folded, color='green', lw=0.8, ls='--', alpha=0.5,
                       label=f'BAZ {baz:.0f}° (folded {baz_folded:.0f}°)')
            ax.legend(fontsize=7)
        for t_arr in [tt_p, tt_s, tt_tp, tt_ts]:
            if t_arr: ax.axvline(t_arr, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax.grid(True, alpha=0.15)

    axes[3].set_xlabel('Time from origin (s)', fontsize=12)
    axes[0].set_xlim(args.tmin, args.tmax)
    fig.suptitle(f'{sta}  Polarization  |  {dist_deg:.2f}°  BAZ={baz:.0f}°  |  Flinn win={args.win}s',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, f'polarization_{sta}.png')
    fig.savefig(out1, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"✅ {out1}")

    # ========== 图2: P / S / T 偏振玫瑰图 (0-180°) ==========
    phases = [
        ('P', tt_p, args.p_win, 'green'),
        ('S', tt_s, args.s_win, 'orange'),
        ('T', tt_tp, args.t_win, 'blue'),
    ]
    valid = [(l, tc, w, c) for l, tc, w, c in phases if tc is not None]
    n_cols = len(valid)

    fig2, axes2 = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5),
                                subplot_kw={'projection': 'polar'})
    if n_cols == 1: axes2 = [axes2]

    for ax, (label, t_center, win_half, color) in zip(axes2, valid):
        plot_rose(ax, t_mid, azi_180, rect, t_center, win_half,
                  args.r_min, sta, label, color, baz)

    fig2.suptitle(f'{sta}  Polarization Rose (0-180°)  |  BAZ={baz:.0f}° (folded {baz%180:.0f}°)',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, f'rose_{sta}.png')
    fig2.savefig(out2, dpi=200, bbox_inches='tight'); plt.close(fig2)
    print(f"✅ {out2}")


if __name__ == "__main__":
    main()
