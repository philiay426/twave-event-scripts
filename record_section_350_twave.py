#!/usr/bin/env python3
"""
350° 剖面 T 波窗口放大图
- 时间窗: 1550-1850s
- 每条 trace 在 T 窗口内独立归一化
- 黑色波形 + 红色包络
- T-P、T-S、T-Love 到时用小点标注
"""
import sys, os, glob

sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass, envelope
from obspy.geodetics.base import locations2degrees, degrees2kilometers
from obspy.taup import TauPyModel
from t_phase_analysis.traveltime_profile import get_track, tt_t

# ── 常量 ──
EVLO, EVLA = -66.87, 18.99
PROC_DIR = "/home/philiay/shared/case/sacprocess"
RAW_DIR  = "/home/philiay/shared/case/raw"
OUT_DIR  = "/home/philiay/shared/case/waveform_contrast"
SOURCE_DEPTH = 10.0
REF_EL = -1000
AZIMUTH = 350
AZ_WIDTH = 2.0
SNR_MIN = 2.5
TMIN, TMAX = 1550, 1850
FMIN, FMAX = 2.0, 8.0
V_LOVE = 3.6  # km/s, T-Love 陆地速度


def find_raw(sta, rawdir):
    for ext in ['.SAC', '.sac']:
        pattern = os.path.join(rawdir, f'*.{sta}.*.BHZ*{ext}')
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def compute_twave_snr(env, t_arr, t_tp):
    noise_start = 800.0
    noise_end = t_tp - 100.0
    signal_start = t_tp
    signal_end = min(t_tp + 180.0, t_arr[-1])
    if noise_end <= noise_start or signal_end <= signal_start:
        return 0.0
    idx_noise = (t_arr >= noise_start) & (t_arr <= noise_end)
    idx_signal = (t_arr >= signal_start) & (t_arr <= signal_end)
    noise_rms = np.sqrt(np.mean(env[idx_noise]**2)) if np.any(idx_noise) else 1e-30
    signal_rms = np.sqrt(np.mean(env[idx_signal]**2)) if np.any(idx_signal) else 0
    return signal_rms / noise_rms if noise_rms > 0 else 0


def main():
    taup = TauPyModel(model='iasp91')
    
    # ── 加载台站 ──
    proc_files = sorted(glob.glob(os.path.join(PROC_DIR, '*.BHZ*_proc.sac')))
    seen = set()
    stations = []
    for pf in proc_files:
        parts = os.path.basename(pf).split('.')
        sta = parts[1]
        if sta in seen:
            continue
        seen.add(sta)
        raw_file = find_raw(sta, RAW_DIR)
        if raw_file is None:
            continue
        try:
            tr_raw = read(raw_file, headonly=True)[0]
            sac = tr_raw.stats.sac
            az = sac.az or 0
            diff = min(abs(az - AZIMUTH), abs(az - AZIMUTH + 360), abs(az - AZIMUTH - 360))
            if diff > AZ_WIDTH:
                continue
            stations.append({
                'sta': sta, 'az': az, 'dist_km': sac.dist or 0,
                'stla': sac.stla or 0, 'stlo': sac.stlo or 0,
                'proc_file': pf, 'raw_file': raw_file,
            })
        except:
            continue
    
    print(f"方位角 {AZIMUTH}° ±{AZ_WIDTH}° : {len(stations)} 台站")
    
    # ── 计算 ──
    for s in stations:
        # 陆地距离
        try:
            track = get_track(EVLO, EVLA, s['stlo'], s['stla'], 0.02,
                              ref_el=REF_EL, plot_map=False, plot_elevation=False)
            tr = tt_t(track, ref_el=REF_EL, source_depth_km=SOURCE_DEPTH,
                       ignore_small_islands=True)
            if tr:
                s['land_km'] = degrees2kilometers(tr.get('dist_land', 0))
                s['tt_ocean'] = tr['tt_ocean']
                s['T-P'] = tr['tt_t']
                s['T-S'] = tr['tt_ocean'] + tr['tt_land_s']
                # T-Love: ocean time + land distance / v_love
                s['T-Love'] = tr['tt_ocean'] + s['land_km'] / V_LOVE
            else:
                s['land_km'] = 0; s['tt_ocean'] = 0
                s['T-P'] = 0; s['T-S'] = 0; s['T-Love'] = 0
        except:
            s['land_km'] = 0; s['tt_ocean'] = 0
            s['T-P'] = 0; s['T-S'] = 0; s['T-Love'] = 0
        
        # 加载数据
        tr_proc = read(s['proc_file'])[0]
        origin = UTCDateTime(
            year=tr_proc.stats.sac.nzyear, julday=tr_proc.stats.sac.nzjday,
            hour=tr_proc.stats.sac.nzhour, minute=tr_proc.stats.sac.nzmin,
            second=tr_proc.stats.sac.nzsec,
            microsecond=tr_proc.stats.sac.nzmsec * 1000
        ) + tr_proc.stats.sac.o
        
        b = tr_proc.stats.starttime - origin
        delta = tr_proc.stats.delta
        t = np.arange(tr_proc.stats.npts) * delta + b
        
        data_bp = bandpass(tr_proc.data.astype(np.float64), FMIN, FMAX,
                           1/delta, corners=4, zerophase=True)
        env = envelope(data_bp)
        
        s['t'] = t
        s['data_bp'] = data_bp
        s['env'] = env
        
        # SNR
        if s['T-P']:
            s['snr'] = compute_twave_snr(env, t, s['T-P'])
        else:
            s['snr'] = 0
    
    # ── SNR 筛选 ──
    n_before = len(stations)
    stations = [s for s in stations if s['snr'] >= SNR_MIN]
    print(f"SNR ≥ {SNR_MIN}: {len(stations)}/{n_before} 台站保留")
    
    # 按陆地距离排序
    stations.sort(key=lambda s: s['land_km'])
    
    for s in stations:
        print(f"  {s['sta']:6s} AZ={s['az']:.1f}°  land={s['land_km']:.0f}km  SNR={s['snr']:.1f}  "
              f"T-P={s['T-P']:.0f}s  T-S={s['T-S']:.0f}s  T-Love={s['T-Love']:.0f}s")
    
    # ── 绘图 ──
    n = len(stations)
    land_vals = [s['land_km'] for s in stations]
    land_min, land_max = min(land_vals), max(land_vals)
    land_range = max(land_max - land_min, 50)
    spacing = land_range / max(n - 1, 1) if n > 1 else land_range
    amp_scale = spacing * 0.28  # 28% spacing → waveform height = 56% of total per trace (no overlap)
    
    fig_h = max(8, n * 0.8)  # taller canvas
    fig, ax = plt.subplots(figsize=(14, fig_h))
    
    # ── 全局归一化因子 (保留真实相对振幅) ──
    global_data_max = 1e-30
    for s in stations:
        mask = (s['t'] >= TMIN) & (s['t'] <= TMAX)
        mx = np.max(np.abs(s['data_bp'][mask])) if np.any(mask) else 0
        if mx > global_data_max:
            global_data_max = mx
    
    for i, s in enumerate(stations):
        y0 = s['land_km']
        mask = (s['t'] >= TMIN) & (s['t'] <= TMAX)
        t_plot = s['t'][mask]
        data_plot = s['data_bp'][mask]
        env_plot = s['env'][mask]
        
        # ── 全局归一化 (保留相对振幅) ──
        data_norm = data_plot / global_data_max
        env_norm = env_plot / global_data_max
        
        # 波形 (黑色)
        ax.plot(t_plot, y0 + data_norm * amp_scale, 'k-', lw=0.35, alpha=0.7, zorder=4)
        
        # 包络填充 + 线 (红色)
        ax.fill_between(t_plot, y0, y0 + env_norm * amp_scale,
                        color='#e74c3c', alpha=0.18, lw=0, zorder=2)
        ax.plot(t_plot, y0 + env_norm * amp_scale,
                color='#e74c3c', lw=0.8, alpha=0.8, zorder=3)
        
        # 水平参考线
        ax.axhline(y=y0, color='gray', lw=0.15, ls='-', alpha=0.15)
        
        # ── 到时标注 (小圆点，统一大小) ──
        for ph_key, color in [
            ('T-P', '#1f77b4'),
            ('T-S', '#d62728'),
            ('T-Love', '#9467bd'),
        ]:
            ta = s.get(ph_key, 0)
            if TMIN <= ta <= TMAX and ta > 0:
                idx = np.argmin(np.abs(t_plot - ta))
                if idx < len(data_norm):
                    y_dot = y0 + data_norm[idx] * amp_scale
                    ax.plot(ta, y_dot, 'o', color=color,
                            markersize=4, zorder=12, markeredgewidth=0)
        
        # 台站标签 (左侧)
        ax.text(TMIN + 8, y0, f"{s['sta']}",
                fontsize=7, fontfamily='monospace', fontweight='bold',
                va='center', ha='left', color='#333333', zorder=10)
    
    # ── 轴标签 ──
    ax.set_xlim(TMIN, TMAX)
    y_pad = land_range * 0.06
    ax.set_ylim(land_min - y_pad - amp_scale, land_max + y_pad + amp_scale * 1.2)
    
    ax.set_xlabel('Time from origin (s)', fontsize=12)
    ax.set_ylabel('Land distance (km)', fontsize=12)
    
    # Y 轴 = 陆地距离刻度
    ax.set_yticks(land_vals)
    ax.set_yticklabels([f"{v:.0f}" for v in land_vals], fontsize=7.5)
    
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(True, axis='x', alpha=0.08, linestyle=':')
    ax.tick_params(axis='y', length=0)
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markersize=5, label='T-P (ocean + land P)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markersize=5, label='T-S (ocean + land S)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd',
               markersize=5, label=f'T-Love (v={V_LOVE} km/s)'),
        Line2D([0], [0], color='black', lw=0.5, label='2-8 Hz waveform'),
        Line2D([0], [0], color='#e74c3c', lw=1.5, label='Envelope'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
              framealpha=0.85, ncol=1, borderpad=0.5)
    
    ax.set_title(
        f'350° Profile — T-wave Window  |  {n} stations (SNR≥{SNR_MIN})  |  '
        f'2014-01-13 Mw6.3 Puerto Rico  |  '
        f'{FMIN:.0f}-{FMAX:.0f} Hz, per-trace norm in window',
        fontsize=9, fontweight='bold'
    )
    
    plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.08)
    
    out_path = os.path.join(OUT_DIR, 'record_section_350_twave_zoom.pdf')
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    # also save PNG for quick preview
    png_path = out_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ {out_path}")


if __name__ == '__main__':
    main()
