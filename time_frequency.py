#!/usr/bin/env python3
"""
时频图 (Spectrogram) — 单个台站
默认 H57A，可用 --sta 切换其他台站
"""
import sys
import os
import glob
import argparse

sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, UTCDateTime
from obspy.geodetics.base import locations2degrees, degrees2kilometers
from obspy.taup import TauPyModel
from t_phase_analysis.traveltime_profile import get_track, tt_t
from t_phase_analysis.waveform import plot_spectrogram_panel, compute_spectrogram

DATA_DIR = "/home/philiay/shared/case/sacprocess"
RAW_DIR  = "/home/philiay/shared/case/raw"
OUT_DIR  = "/home/philiay/shared/case/waveform_contrast"
EVLO, EVLA = -66.87, 18.99
SOURCE_DEPTH = 10.0
REF_EL = -1000


def parse_args():
    p = argparse.ArgumentParser(description='时频图 (Spectrogram)')
    p.add_argument('--sta', default='H57A', help='台站名 (默认 H57A)')
    p.add_argument('--channel', default='BHZ', help='通道 (BHZ/BHN/BHE)')
    p.add_argument('--tmin', type=float, default=-500, help='起始时间 (s)')
    p.add_argument('--tmax', type=float, default=3500, help='结束时间 (s)')
    p.add_argument('--fmax', type=float, default=10, help='频率上限 (Hz)')
    p.add_argument('--out', default=None)
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


def find_file(sta, channel, directory, suffix):
    """在目录中找台站+通道文件"""
    for pat in [f"*.{sta}..{channel}.*{suffix}", f"*{sta}*{channel}*{suffix}"]:
        matches = glob.glob(os.path.join(directory, pat))
        if matches: return matches[0]
    return None


def main():
    args = parse_args()
    sta = args.sta

    proc_file = find_file(sta, args.channel, DATA_DIR, '_proc.sac')
    raw_file  = find_file(sta, args.channel, RAW_DIR, '.SAC')
    if not proc_file: sys.exit(f"❌ 找不到 {sta} 的 proc 文件")
    if not raw_file:  sys.exit(f"❌ 找不到 {sta} 的原始 SAC")

    # 读取
    tr_raw  = read(raw_file)[0];  sac = tr_raw.stats.sac
    tr_proc = read(proc_file)[0]
    stla, stlo = sac.stla, sac.stlo

    origin = UTCDateTime(year=sac.nzyear, julday=sac.nzjday,
                         hour=sac.nzhour, minute=sac.nzmin,
                         second=sac.nzsec, microsecond=sac.nzmsec*1000) + sac.o
    dist_deg = locations2degrees(EVLA, EVLO, stla, stlo)
    dist_km  = degrees2kilometers(dist_deg)
    b = tr_proc.stats.starttime - origin

    # TauP P/S
    taup = TauPyModel(model='iasp91')
    try:
        arrs = taup.get_travel_times(source_depth_in_km=SOURCE_DEPTH,
                                      distance_in_degree=dist_deg,
                                      phase_list=['P', 'S'])
        tt_p_taup = tt_s_taup = None
        for a in arrs:
            if a.name == 'P' and tt_p_taup is None: tt_p_taup = a.time
            elif a.name == 'S' and tt_s_taup is None: tt_s_taup = a.time
        if tt_p_taup is None: tt_p_taup = dist_km / 8.0
        if tt_s_taup is None: tt_s_taup = tt_p_taup * 1.82
    except:
        tt_p_taup = dist_km / 8.0
        tt_s_taup = tt_p_taup * 1.82

    # tt_t T-P
    has_twave = True; tt_t_val = None
    try:
        track = get_track(EVLO, EVLA, stlo, stla, 0.02,
                          ref_el=REF_EL, plot_map=False, plot_elevation=False)
        tr = tt_t(track, ref_el=REF_EL, source_depth_km=SOURCE_DEPTH,
                   ignore_small_islands=True)
        if tr: tt_t_val = tr['tt_t']
    except:
        has_twave = False

    print(f"{sta}: {dist_km:.0f} km ({dist_deg:.2f}°)")
    print(f"  P={tt_p_taup:.0f}s  S={tt_s_taup:.0f}s  T-P={'N/A' if tt_t_val is None else f'{tt_t_val:.0f}s'}")

    # 时频谱
    data_proc = tr_proc.data.astype(np.float64)
    f_spec, t_spec, Sxx_db = compute_spectrogram(data_proc, tr_proc.stats.sampling_rate)
    t_spec_rel = t_spec + b

    # 绘图
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_spectrogram_panel(ax, t_spec_rel, f_spec, Sxx_db,
                           tt_p=tt_p_taup, tt_s=tt_s_taup, tt_t=tt_t_val,
                           has_twave=has_twave, freq_max=args.fmax,
                           cmap='inferno')

    ax.set_xlim(args.tmin, args.tmax)
    ax.set_xlabel('Time from origin (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)

    title = (f'{sta} Spectrogram  |  {dist_km:.0f} km ({dist_deg:.2f}°)  |  '
             f'2014-01-13 Mw6.3 Puerto Rico  |  deconvolved to velocity')
    ax.set_title(title, fontsize=11, fontweight='bold')

    plt.tight_layout()

    out = args.out or os.path.join(OUT_DIR, f'spectrogram_{sta}_{args.channel}.png')
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
