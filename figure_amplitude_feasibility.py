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

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import numpy as np
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass, envelope
from obspy.geodetics.base import locations2degrees, degrees2kilometers
from obspy.taup import TauPyModel
from scipy.stats import linregress
from t_phase_analysis.traveltime_profile import get_track, tt_t

EVLO, EVLA = -66.87, 18.99
PROC_DIR = "/home/philiay/shared/case/sacprocess"
RAW_DIR  = "/home/philiay/shared/case/raw"
OUT_DIR  = "/home/philiay/shared/case/waveform_contrast/one"
SOURCE_DEPTH = 10.0
REF_EL = -1000
FMIN, FMAX = 2.0, 8.0
V_LOVE = 3.6
TLOVE_HALF = 45

BANDS = {
    '2-3 Hz': (2.0, 3.0), '3-4 Hz': (3.0, 4.0),
    '4-5 Hz': (4.0, 5.0), '5-6 Hz': (5.0, 6.0), '6-8 Hz': (6.0, 8.0),
}
BAND_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def find_raw(sta, rawdir):
    for ext in ['.SAC', '.sac']:
        for f in glob.glob(os.path.join(rawdir, f'*.{sta}.*.BHZ*{ext}')):
            return f
    return None


def compute_twave_snr(env, t_arr, t_tp):
    noise_start, noise_end = 800.0, t_tp - 100.0
    signal_start, signal_end = t_tp, min(t_tp + 180.0, t_arr[-1])
    if noise_end <= noise_start or signal_end <= signal_start:
        return 0.0
    n_idx = (t_arr >= noise_start) & (t_arr <= noise_end)
    s_idx = (t_arr >= signal_start) & (t_arr <= signal_end)
    nr = np.sqrt(np.mean(env[n_idx]**2)) if np.any(n_idx) else 1e-30
    sr = np.sqrt(np.mean(env[s_idx]**2)) if np.any(s_idx) else 0
    return sr / nr if nr > 0 else 0


def main():
    p = argparse.ArgumentParser(description='T-wave amplitude feasibility figure')
    p.add_argument('--azimuth', type=float, default=350)
    p.add_argument('--az-width', type=float, default=2.0)
    p.add_argument('--snr-min', type=float, default=2.5)
    p.add_argument('--exclude', type=str, default='', help='Comma-separated stations to exclude')
    p.add_argument('--tmin', type=float, default=None)
    p.add_argument('--tmax', type=float, default=None)
    args = p.parse_args()
    
    az = args.azimuth; azw = args.az_width; snr = args.snr_min
    tmin_in = args.tmin if args.tmin is not None else (1100 if az < 340 else 1550)
    tmax_in = args.tmax if args.tmax is not None else (1950 if az < 340 else 1850)
    print(f"方位角 {az}° ±{azw}°, SNR ≥ {snr}, 时窗 {tmin_in}-{tmax_in}s")
    
    taup = TauPyModel(model='iasp91')
    
    # ── Load stations ──
    proc_files = sorted(glob.glob(os.path.join(PROC_DIR, '*.BHZ*_proc.sac')))
    seen = set(); stations = []
    for pf in proc_files:
        sta = os.path.basename(pf).split('.')[1]
        if sta in seen: continue
        seen.add(sta)
        raw_file = find_raw(sta, RAW_DIR)
        if raw_file is None: continue
        try:
            tr = read(raw_file, headonly=True)[0]; sac = tr.stats.sac
            diff = min(abs((sac.az or 0) - az), abs((sac.az or 0) - az + 360), abs((sac.az or 0) - az - 360))
            if diff > azw: continue
            stations.append({'sta':sta, 'az':sac.az or 0, 'dist_km':sac.dist or 0,
                            'stla':sac.stla or 0, 'stlo':sac.stlo or 0, 'proc_file':pf, 'raw_file':raw_file})
        except: continue
    
    print(f"  总台站: {len(stations)}")
    
    # ── Compute ──
    for s in stations:
        try:
            track = get_track(EVLO, EVLA, s['stlo'], s['stla'], 0.02, ref_el=REF_EL, plot_map=False, plot_elevation=False)
            tr = tt_t(track, ref_el=REF_EL, source_depth_km=SOURCE_DEPTH, ignore_small_islands=True)
            if tr:
                s['land_km'] = degrees2kilometers(tr.get('dist_land', 0))
                s['tt_ocean'] = tr['tt_ocean']; s['T-P'] = tr['tt_t']
                s['T-S'] = tr['tt_ocean'] + tr['tt_land_s']
                s['T-Love'] = tr['tt_ocean'] + s['land_km'] / V_LOVE
        except:
            s['land_km'] = 0; s['T-P'] = 0; s['T-S'] = 0; s['T-Love'] = 0
        
        tr_proc = read(s['proc_file'])[0]
        origin = UTCDateTime(year=tr_proc.stats.sac.nzyear, julday=tr_proc.stats.sac.nzjday,
            hour=tr_proc.stats.sac.nzhour, minute=tr_proc.stats.sac.nzmin,
            second=tr_proc.stats.sac.nzsec, microsecond=(tr_proc.stats.sac.nzmsec or 0)*1000
        ) + tr_proc.stats.sac.o
        b = tr_proc.stats.starttime - origin; delta = tr_proc.stats.delta
        t = np.arange(tr_proc.stats.npts) * delta + b
        data_bp = bandpass(tr_proc.data.astype(np.float64), FMIN, FMAX, 1/delta, corners=4, zerophase=True)
        env = envelope(data_bp)
        s['t'] = t; s['data_bp'] = data_bp; s['env'] = env; s['delta'] = delta
        s['snr'] = compute_twave_snr(env, t, s['T-P']) if s['T-P'] else 0
        
        s['band_data'] = {}
        for bn, (bf, bt) in BANDS.items():
            s['band_data'][bn] = bandpass(tr_proc.data.astype(np.float64), bf, bt, 1/delta, corners=4, zerophase=True)
    
    # ── SNR filter & exclude & sort ──
    stations = [s for s in stations if s['snr'] >= snr]
    exclude_list = [x.strip() for x in args.exclude.split(',') if x.strip()]
    if exclude_list:
        before = len(stations)
        stations = [s for s in stations if s['sta'] not in exclude_list]
        print(f"  排除 {exclude_list}: {before} → {len(stations)} 台站")
    stations.sort(key=lambda s: s['land_km'])
    n = len(stations)
    print(f"  SNR≥{snr}: {n} 台站")
    
    # ── Extract T-Love window values for (c)(d) ──
    for s in stations:
        tl = s.get('T-Love', 0)
        if tl == 0: continue
        m = (s['t'] >= tl - TLOVE_HALF) & (s['t'] <= tl + TLOVE_HALF)
        if np.any(m):
            s['env_peak'] = np.max(s['env'][m])
            s['ln_env_peak'] = np.log(s['env_peak'])
            s['band_ln_rms'] = {}
            for bn in BANDS:
                bd = s['band_data'][bn]; bm = m[:len(bd)] if len(bd) > len(m) else m
                if len(bd) != len(m):
                    bm = np.zeros(len(bd), dtype=bool); bm[:min(len(bd),len(m))] = m[:min(len(bd),len(m))]
                if np.any(bm):
                    rms = np.sqrt(np.mean(bd[bm]**2))
                    s['band_ln_rms'][bn] = np.log(rms) if rms > 0 else -30
                else:
                    s['band_ln_rms'][bn] = -30
    
    for s in stations:
        print(f"  {s['sta']:6s} land={s['land_km']:.0f}km SNR={s['snr']:.1f} ln_env={s.get('ln_env_peak',-99):.2f}")
    
    # ── Global normalization factors ──
    global_max = 1e-30; all_abs = []
    for s in stations:
        m = (s['t'] >= tmin_in) & (s['t'] <= tmax_in); d = s['data_bp'][m]; all_abs.extend(np.abs(d))
        mx = np.max(np.abs(d)) if len(d) > 0 else 0
        if mx > global_max: global_max = mx
    global_p98 = np.percentile(all_abs, 98) if all_abs else global_max
    if global_p98 == 0: global_p98 = global_max
    print(f"  Global max={global_max:.2e}, P98={global_p98:.2e}")
    
    land_vals = [s['land_km'] for s in stations]
    land_min, land_max = min(land_vals), max(land_vals)
    land_range = max(land_max - land_min, 50)
    spacing = land_range / max(n - 1, 1) if n > 1 else land_range
    
    # ══════ 2×2 Figure ══════
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.32,
                           height_ratios=[1, 1], left=0.06, right=0.97, top=0.95, bottom=0.06)
    
    amp_scale_a = spacing * 0.28
    p98_factor = land_range * 0.22 / max(n, 1)
    
    # (a) Global max norm
    ax_a = fig.add_subplot(gs[0, 0])
    for s in stations:
        y0 = s['land_km']; m = (s['t'] >= tmin_in) & (s['t'] <= tmax_in)
        tp = s['t'][m]; dn = s['data_bp'][m] / global_max; en = s['env'][m] / global_max
        ax_a.plot(tp, y0 + dn * amp_scale_a, 'k-', lw=0.35, alpha=0.7)
        ax_a.fill_between(tp, y0, y0 + en * amp_scale_a, color='#e74c3c', alpha=0.18, lw=0)
        ax_a.plot(tp, y0 + en * amp_scale_a, color='#e74c3c', lw=0.8, alpha=0.8)
        ax_a.axhline(y=y0, color='gray', lw=0.15, alpha=0.15)
        for pk, pc in [('T-P','#1f77b4'),('T-S','#d62728'),('T-Love','#9467bd')]:
            ta = s.get(pk, 0)
            if tmin_in <= ta <= tmax_in and ta > 0:
                idx = np.argmin(np.abs(tp - ta))
                if idx < len(dn): ax_a.plot(ta, y0 + dn[idx] * amp_scale_a, 'o', color=pc, markersize=3, markeredgewidth=0)
        ax_a.text(tmin_in + 10, y0, s['sta'], fontsize=6, fontfamily='monospace', fontweight='bold', va='center', ha='left')
    ax_a.set_xlim(tmin_in, tmax_in); ax_a.set_ylabel('Land distance (km)', fontsize=10)
    ax_a.set_yticks(land_vals); ax_a.set_yticklabels([f'{v:.0f}' for v in land_vals], fontsize=6.5)
    ax_a.xaxis.set_major_locator(MultipleLocator(100)); ax_a.grid(True, axis='x', alpha=0.08, linestyle=':')
    ax_a.tick_params(axis='y', length=0)
    y_pad = land_range * 0.06
    ax_a.set_ylim(land_min - y_pad - amp_scale_a, land_max + y_pad + amp_scale_a * 1.2)
    ax_a.set_title(f'(a) Record section — global max norm', fontsize=11, fontweight='bold')
    
    # (b) P98 norm
    ax_b = fig.add_subplot(gs[0, 1])
    for s in stations:
        y0 = s['land_km']; m = (s['t'] >= tmin_in) & (s['t'] <= tmax_in)
        tp = s['t'][m]; dn = s['data_bp'][m] / global_p98; en = s['env'][m] / global_p98
        ax_b.plot(tp, y0 + dn * p98_factor, 'k-', lw=0.35, alpha=0.7)
        ax_b.fill_between(tp, y0, y0 + en * p98_factor, color='#e74c3c', alpha=0.18, lw=0)
        ax_b.plot(tp, y0 + en * p98_factor, color='#e74c3c', lw=0.8, alpha=0.8)
        ax_b.axhline(y=y0, color='gray', lw=0.15, alpha=0.15)
        for pk, pc in [('T-P','#1f77b4'),('T-S','#d62728'),('T-Love','#9467bd')]:
            ta = s.get(pk, 0)
            if tmin_in <= ta <= tmax_in and ta > 0:
                idx = np.argmin(np.abs(tp - ta))
                if idx < len(dn): ax_b.plot(ta, y0 + dn[idx] * p98_factor, 'o', color=pc, markersize=3, markeredgewidth=0)
        ax_b.text(tmin_in + 10, y0, s['sta'], fontsize=6, fontfamily='monospace', fontweight='bold', va='center', ha='left')
    ax_b.set_xlim(tmin_in, tmax_in); ax_b.set_ylabel('Land distance (km)', fontsize=10)
    ax_b.set_yticks(land_vals); ax_b.set_yticklabels([f'{v:.0f}' for v in land_vals], fontsize=6.5)
    ax_b.xaxis.set_major_locator(MultipleLocator(100)); ax_b.grid(True, axis='x', alpha=0.08, linestyle=':')
    ax_b.tick_params(axis='y', length=0)
    ax_b.set_ylim(land_min - y_pad - p98_factor, land_max + y_pad + p98_factor * 1.2)
    ax_b.set_title(f'(b) Record section — global P98 norm', fontsize=11, fontweight='bold')
    
    # (c) ln envelope peak
    ax_c = fig.add_subplot(gs[1, 0])
    land_a = np.array([s['land_km'] for s in stations])
    ln_a = np.array([s.get('ln_env_peak', -30) for s in stations])
    snr_a = np.array([s['snr'] for s in stations])
    sc = ax_c.scatter(land_a, ln_a, c=snr_a, cmap='plasma', s=60, edgecolors='k', linewidth=0.5, zorder=5)
    for s in stations:
        ax_c.annotate(s['sta'], (s['land_km'], s.get('ln_env_peak', -30)),
                     fontsize=6.5, fontfamily='monospace', fontweight='bold', xytext=(4,4), textcoords='offset points')
    if len(land_a) >= 3:
        slope, intercept, r_val, _, _ = linregress(land_a, ln_a)
        fx = np.linspace(land_min-20, land_max+20, 100); fy = slope*fx + intercept
        ax_c.plot(fx, fy, 'k--', lw=1.0, alpha=0.6)
        ax_c.text(0.95, 0.05, f'ln(A) = {slope:.4f}·R + {intercept:.1f}\nR² = {r_val**2:.3f}',
                 transform=ax_c.transAxes, fontsize=8, fontfamily='monospace', va='bottom', ha='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6))
    plt.colorbar(sc, ax=ax_c, label='SNR').ax.tick_params(labelsize=7)
    ax_c.set_xlabel('Land distance (km)', fontsize=10); ax_c.set_ylabel('ln(Envelope peak)', fontsize=10)
    ax_c.set_title(f'(c) T-wave amplitude decay (T-Love ±{TLOVE_HALF}s)', fontsize=11, fontweight='bold')
    ax_c.grid(True, alpha=0.15, linestyle=':')
    
    # (d) Spectral bands
    ax_d = fig.add_subplot(gs[1, 1])
    for ib, (bn, clr) in enumerate(zip(BANDS.keys(), BAND_COLORS)):
        vals, lands = [], []
        for s in stations:
            lv = s['band_ln_rms'].get(bn)
            if lv is not None and lv > -20: vals.append(lv); lands.append(s['land_km'])
        if len(lands) >= 3:
            slope, intercept, r_val, _, _ = linregress(lands, vals)
            fx = np.linspace(land_min-20, land_max+20, 100); fy = slope*fx + intercept
            ax_d.plot(fx, fy, '-', color=clr, lw=1.2, alpha=0.7)
            ax_d.plot(lands, vals, 'o', color=clr, markersize=6, markeredgecolor='k', markeredgewidth=0.3, alpha=0.8)
            ax_d.text(0.97, 0.97-ib*0.07, f'{bn}: b={slope:.4f}  R²={r_val**2:.3f}',
                     transform=ax_d.transAxes, fontsize=7.5, fontfamily='monospace', va='top', ha='right', color=clr, fontweight='bold')
    ax_d.set_xlabel('Land distance (km)', fontsize=10); ax_d.set_ylabel('ln(RMS spectral amplitude)', fontsize=10)
    ax_d.set_title(f'(d) Frequency-dependent amplitude decay', fontsize=11, fontweight='bold')
    ax_d.grid(True, alpha=0.15, linestyle=':')
    
    fig.suptitle(f'{az:.0f}° Profile — T-wave Amplitude Feasibility  |  {n} stations, SNR≥{snr}  |  2014-01-13 Mw6.3',
                 fontsize=12, fontweight='bold', y=0.98)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'figure_{az:.0f}_amplitude_feasibility')
    fig.savefig(out+'.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ {out}.pdf /.png")

if __name__ == '__main__':
    main()
