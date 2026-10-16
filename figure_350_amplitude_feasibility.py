#!/usr/bin/env python3
"""
Figure: azimuthal T-wave amplitude feasibility

(a) T-wave zoom record section, global max norm
(b) Same, global P98 scaling
(c) ln envelope peak vs land distance (T-Love window)
(d) ln spectral amplitude vs land distance (5 bands)

Usage:
  python3 figure_amplitude_feasibility.py --azimuth 350 --snr-min 2.5
  python3 figure_amplitude_feasibility.py --azimuth 327 --snr-min 1.0
"""
import sys, os, glob, argparse
sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from matplotlib.lines import Line2D
import numpy as np
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass, envelope
from obspy.geodetics.base import locations2degrees, degrees2kilometers
from obspy.taup import TauPyModel
from scipy.stats import linregress
from t_phase_analysis.traveltime_profile import get_track, tt_t

# ── 常量 ──
EVLO, EVLA = -66.87, 18.99
PROC_DIR = "/home/philiay/shared/case/sacprocess"
RAW_DIR  = "/home/philiay/shared/case/raw"
OUT_DIR  = "/home/philiay/shared/case/waveform_contrast"
SOURCE_DEPTH = 10.0
REF_EL = -1000
AZIMUTH = 350  # overridden by command-line arg
AZ_WIDTH = 2.0
SNR_MIN = 2.5  # overridden by command-line arg
TMIN, TMAX = 1550, 1850
FMIN, FMAX = 2.0, 8.0
V_LOVE = 3.6

# 分频带
BANDS = {
    '2–3 Hz': (2.0, 3.0),
    '3–4 Hz': (3.0, 4.0),
    '4–5 Hz': (4.0, 5.0),
    '5–6 Hz': (5.0, 6.0),
    '6–8 Hz': (6.0, 8.0),
}
BAND_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# T-Love 窗口半宽
TLOVE_HALF = 45  # seconds


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
    p = argparse.ArgumentParser(description='T-wave amplitude feasibility figure')
    p.add_argument('--azimuth', type=float, default=350, help='Azimuth center (default: 350)')
    p.add_argument('--az-width', type=float, default=2.0, help='Azimuth half-width')
    p.add_argument('--snr-min', type=float, default=2.5, help='SNR threshold')
    p.add_argument('--tmin', type=float, default=None, help='Time window start')
    p.add_argument('--tmax', type=float, default=None, help='Time window end')
    args = p.parse_args()
    
    azimuth = args.azimuth
    az_width = args.az_width
    snr_min = args.snr_min
    
    if args.tmin is None:
        tmin, tmax = (1100, 1950) if azimuth < 340 else (1550, 1850)
    else:
        tmin, tmax = args.tmin, args.tmax
    
    print(f"方位角 {azimuth}° ±{az_width}°, SNR ≥ {snr_min}, 时窗 {tmin}-{tmax}s")
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
                s['T-Love'] = tr['tt_ocean'] + s['land_km'] / V_LOVE
        except:
            s['land_km'] = 0; s['tt_ocean'] = 0
            s['T-P'] = 0; s['T-S'] = 0; s['T-Love'] = 0
        
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
        s['delta'] = delta
        
        if s['T-P']:
            s['snr'] = compute_twave_snr(env, t, s['T-P'])
        else:
            s['snr'] = 0
        
        # 分频带滤波
        s['band_data'] = {}
        for bname, (bmin, bmax) in BANDS.items():
            s['band_data'][bname] = bandpass(
                tr_proc.data.astype(np.float64), bmin, bmax,
                1/delta, corners=4, zerophase=True
            )
    
    # ── SNR 筛选 ──
    stations = [s for s in stations if s['snr'] >= SNR_MIN]
    stations.sort(key=lambda s: s['land_km'])
    print(f"SNR ≥ {SNR_MIN}: {len(stations)} 台站")
    n = len(stations)
    
    # ── 提取 (c)(d) 用到的 T-Love 窗口包络峰值 ──
    for s in stations:
        t_love = s.get('T-Love', 0)
        if t_love == 0:
            continue
        mask = (s['t'] >= t_love - TLOVE_HALF) & (s['t'] <= t_love + TLOVE_HALF)
        if np.any(mask):
            s['env_peak'] = np.max(s['env'][mask])
            s['ln_env_peak'] = np.log(s['env_peak'])
            # 分频 RMS
            s['band_ln_rms'] = {}
            for bname in BANDS:
                bdata = s['band_data'][bname]
                bmask = mask[:len(bdata)] if len(bdata) > len(mask) else mask
                # adjust if lengths differ
                if len(bdata) != len(mask):
                    bmask = np.zeros(len(bdata), dtype=bool)
                    min_len = min(len(bdata), len(mask))
                    bmask[:min_len] = mask[:min_len]
                if np.any(bmask):
                    rms = np.sqrt(np.mean(bdata[bmask]**2))
                    s['band_ln_rms'][bname] = np.log(rms) if rms > 0 else -30
                else:
                    s['band_ln_rms'][bname] = -30
    
    # ── 打印 ──
    for s in stations:
        print(f"  {s['sta']:6s} land={s['land_km']:.0f}km  SNR={s['snr']:.1f}  "
              f"ln(env_peak)={s.get('ln_env_peak', -99):.2f}")
    
    # ── 全局归一化因子 ──
    global_max = 1e-30
    global_p98 = 1e-30
    all_data = []
    for s in stations:
        mask = (s['t'] >= TMIN) & (s['t'] <= TMAX)
        d = s['data_bp'][mask]
        all_data.extend(d)
        mx = np.max(np.abs(d)) if len(d) > 0 else 0
        if mx > global_max:
            global_max = mx
    if len(all_data) > 0:
        global_p98 = np.percentile(np.abs(all_data), 98)
        if global_p98 == 0:
            global_p98 = global_max
    print(f"Global max={global_max:.2e}, P98={global_p98:.2e}")
    
    # ══════════════════════════════════════════
    #  绘图 (2×2)
    # ══════════════════════════════════════════
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.30, wspace=0.32,
                           height_ratios=[1, 1],
                           left=0.06, right=0.97, top=0.95, bottom=0.06)
    
    land_vals = [s['land_km'] for s in stations]
    land_min, land_max = min(land_vals), max(land_vals)
    land_range = max(land_max - land_min, 50)
    spacing = land_range / max(n - 1, 1) if n > 1 else land_range
    amp_scale_a = spacing * 0.28
    # (b) P98 scaling: larger amp scale to see weak signals
    # scale factor adjusts P98-normalized trace to ~0.25 of spacing
    p98_factor = land_range * 0.22 / max(n, 1)
    
    # ── (a) Global max norm ──
    ax_a = fig.add_subplot(gs[0, 0])
    for s in stations:
        y0 = s['land_km']
        mask = (s['t'] >= TMIN) & (s['t'] <= TMAX)
        t_plot = s['t'][mask]
        dn = s['data_bp'][mask] / global_max
        en = s['env'][mask] / global_max
        ax_a.plot(t_plot, y0 + dn * amp_scale_a, 'k-', lw=0.35, alpha=0.7, zorder=4)
        ax_a.fill_between(t_plot, y0, y0 + en * amp_scale_a,
                          color='#e74c3c', alpha=0.18, lw=0, zorder=2)
        ax_a.plot(t_plot, y0 + en * amp_scale_a,
                  color='#e74c3c', lw=0.8, alpha=0.8, zorder=3)
        ax_a.axhline(y=y0, color='gray', lw=0.15, ls='-', alpha=0.15)
        # dots
        for ph_key, color in [('T-P','#1f77b4'),('T-S','#d62728'),('T-Love','#9467bd')]:
            ta = s.get(ph_key, 0)
            if TMIN <= ta <= TMAX and ta > 0:
                idx = np.argmin(np.abs(t_plot - ta))
                if idx < len(dn):
                    ax_a.plot(ta, y0 + dn[idx] * amp_scale_a, 'o',
                              color=color, markersize=3, zorder=12, markeredgewidth=0)
        ax_a.text(TMIN + 10, y0, s['sta'], fontsize=6, fontfamily='monospace',
                  fontweight='bold', va='center', ha='left', color='#333', zorder=10)
    ax_a.set_xlim(TMIN, TMAX)
    y_pad = land_range * 0.06
    ax_a.set_ylim(land_min - y_pad - amp_scale_a, land_max + y_pad + amp_scale_a * 1.2)
    ax_a.set_xlabel('Time from origin (s)', fontsize=10)
    ax_a.set_ylabel('Land distance (km)', fontsize=10)
    ax_a.set_yticks(land_vals)
    ax_a.set_yticklabels([f"{v:.0f}" for v in land_vals], fontsize=6.5)
    ax_a.xaxis.set_major_locator(MultipleLocator(100))
    ax_a.xaxis.set_minor_locator(MultipleLocator(20))
    ax_a.grid(True, axis='x', alpha=0.08, linestyle=':')
    ax_a.tick_params(axis='y', length=0)
    ax_a.set_title('(a) Record section — global max norm', fontsize=11, fontweight='bold')
    
    # ── (b) Global P98 norm ──
    ax_b = fig.add_subplot(gs[0, 1])
    for s in stations:
        y0 = s['land_km']
        mask = (s['t'] >= TMIN) & (s['t'] <= TMAX)
        t_plot = s['t'][mask]
        dn = s['data_bp'][mask] / global_p98
        en = s['env'][mask] / global_p98
        ax_b.plot(t_plot, y0 + dn * p98_factor, 'k-', lw=0.35, alpha=0.7, zorder=4)
        ax_b.fill_between(t_plot, y0, y0 + en * p98_factor,
                          color='#e74c3c', alpha=0.18, lw=0, zorder=2)
        ax_b.plot(t_plot, y0 + en * p98_factor,
                  color='#e74c3c', lw=0.8, alpha=0.8, zorder=3)
        ax_b.axhline(y=y0, color='gray', lw=0.15, ls='-', alpha=0.15)
        for ph_key, color in [('T-P','#1f77b4'),('T-S','#d62728'),('T-Love','#9467bd')]:
            ta = s.get(ph_key, 0)
            if TMIN <= ta <= TMAX and ta > 0:
                idx = np.argmin(np.abs(t_plot - ta))
                if idx < len(dn):
                    ax_b.plot(ta, y0 + dn[idx] * p98_factor, 'o',
                              color=color, markersize=3, zorder=12, markeredgewidth=0)
        ax_b.text(TMIN + 10, y0, s['sta'], fontsize=6, fontfamily='monospace',
                  fontweight='bold', va='center', ha='left', color='#333', zorder=10)
    ax_b.set_xlim(TMIN, TMAX)
    ax_b.set_ylim(land_min - y_pad - p98_factor, land_max + y_pad + p98_factor * 1.2)
    ax_b.set_xlabel('Time from origin (s)', fontsize=10)
    ax_b.set_ylabel('Land distance (km)', fontsize=10)
    ax_b.set_yticks(land_vals)
    ax_b.set_yticklabels([f"{v:.0f}" for v in land_vals], fontsize=6.5)
    ax_b.xaxis.set_major_locator(MultipleLocator(100))
    ax_b.xaxis.set_minor_locator(MultipleLocator(20))
    ax_b.grid(True, axis='x', alpha=0.08, linestyle=':')
    ax_b.tick_params(axis='y', length=0)
    ax_b.set_title('(b) Record section — global P98 norm', fontsize=11, fontweight='bold')
    
    # ── (c) ln envelope peak vs land distance ──
    ax_c = fig.add_subplot(gs[1, 0])
    land_arr = np.array([s['land_km'] for s in stations])
    ln_arr = np.array([s.get('ln_env_peak', -30) for s in stations])
    snr_arr = np.array([s['snr'] for s in stations])
    
    sc = ax_c.scatter(land_arr, ln_arr, c=snr_arr, cmap='plasma',
                      s=60, edgecolors='k', linewidth=0.5, zorder=5)
    # station labels
    for s in stations:
        ax_c.annotate(s['sta'], (s['land_km'], s.get('ln_env_peak', -30)),
                      fontsize=6.5, fontfamily='monospace', fontweight='bold',
                      xytext=(4, 4), textcoords='offset points',
                      color='#333', alpha=0.8)
    # linear fit
    if len(land_arr) >= 3:
        slope, intercept, r_val, p_val, std_err = linregress(land_arr, ln_arr)
        fit_x = np.linspace(land_min - 20, land_max + 20, 100)
        fit_y = slope * fit_x + intercept
        ax_c.plot(fit_x, fit_y, 'k--', lw=1.0, alpha=0.6, zorder=4)
        r2 = r_val ** 2
        ax_c.text(0.95, 0.05, f'ln(A) = {slope:.4f}·R + {intercept:.1f}\n'
                  f'R² = {r2:.3f}  |  b = {slope:.4f} km⁻¹',
                  transform=ax_c.transAxes, fontsize=8, fontfamily='monospace',
                  va='bottom', ha='right', bbox=dict(boxstyle='round,pad=0.3',
                                                      facecolor='wheat', alpha=0.6))
    cbar = plt.colorbar(sc, ax=ax_c, label='SNR')
    cbar.ax.tick_params(labelsize=7)
    ax_c.set_xlabel('Land distance (km)', fontsize=10)
    ax_c.set_ylabel('ln(Envelope peak)', fontsize=10)
    ax_c.set_title('(c) T-wave amplitude decay (T-Love window ±45s)', fontsize=11, fontweight='bold')
    ax_c.grid(True, alpha=0.15, linestyle=':')
    
    # ── (d) ln spectral amplitude vs land distance (5 bands) ──
    ax_d = fig.add_subplot(gs[1, 1])
    for ib, (bname, color) in enumerate(zip(BANDS.keys(), BAND_COLORS)):
        vals = []
        lands = []
        for s in stations:
            lv = s['band_ln_rms'].get(bname)
            if lv is not None and lv > -20:
                vals.append(lv)
                lands.append(s['land_km'])
        if len(lands) >= 3:
            slope, intercept, r_val, p_val, std_err = linregress(lands, vals)
            fit_x = np.linspace(land_min - 20, land_max + 20, 100)
            fit_y = slope * fit_x + intercept
            ax_d.plot(fit_x, fit_y, '-', color=color, lw=1.2, alpha=0.7, zorder=3)
            r2 = r_val ** 2
            ax_d.plot(lands, vals, 'o', color=color, markersize=6,
                      markerfacecolor=color, markeredgecolor='k',
                      markeredgewidth=0.3, zorder=4, alpha=0.8)
            # label with slope
            ax_d.text(0.97, 0.97 - ib * 0.07, f'{bname}: b={slope:.4f}  R²={r2:.3f}',
                      transform=ax_d.transAxes, fontsize=7.5, fontfamily='monospace',
                      va='top', ha='right', color=color, fontweight='bold')
    
    ax_d.set_xlabel('Land distance (km)', fontsize=10)
    ax_d.set_ylabel('ln(RMS spectral amplitude)', fontsize=10)
    ax_d.set_title('(d) Frequency-dependent amplitude decay', fontsize=11, fontweight='bold')
    ax_d.grid(True, alpha=0.15, linestyle=':')
    
    # ── 图注 ──
    fig.suptitle(
        f'350° Profile — T-wave Amplitude Feasibility  |  {n} stations, SNR≥{SNR_MIN}  |  '
        f'2014-01-13 Mw6.3 Puerto Rico',
        fontsize=12, fontweight='bold', y=0.98
    )
    
    # ── 保存 ──
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, 'figure_350_amplitude_feasibility.pdf')
    png_path = pdf_path.replace('.pdf', '.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ {pdf_path}")
    print(f"✅ {png_path}")


if __name__ == '__main__':
    main()
