#!/usr/bin/env python3
"""
纯波形记录剖面图 — 仅 T 波窗口，2-8 Hz 波形+包络
用法:
  python3 record_section_waveform.py --azimuth 327 --snr-min 1.0
  python3 record_section_waveform.py --azimuth 350 --snr-min 2.5 --exclude M47A
"""
import sys, os, glob, argparse
sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass, envelope
from obspy.geodetics.base import degrees2kilometers
from obspy.taup import TauPyModel
from t_phase_analysis.traveltime_profile import get_track, tt_t

EVLO, EVLA = -66.87, 18.99
PROC_DIR = "/home/philiay/shared/case/sacprocess"
RAW_DIR  = "/home/philiay/shared/case/raw"
OUT_DIR  = "/home/philiay/shared/case/waveform_contrast/one"
SOURCE_DEPTH = 10.0; REF_EL = -1000
FMIN, FMAX = 2.0, 8.0; V_LOVE = 3.6


def find_raw(sta, rd):
    for ext in ['.SAC', '.sac']:
        for f in glob.glob(os.path.join(rd, f'*.{sta}.*.BHZ*{ext}')):
            return f
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--azimuth', type=float, default=350)
    p.add_argument('--az-width', type=float, default=2.0)
    p.add_argument('--snr-min', type=float, default=2.5)
    p.add_argument('--exclude', type=str, default='')
    p.add_argument('--tmin', type=float, default=None)
    p.add_argument('--tmax', type=float, default=None)
    p.add_argument('--norm', choices=['max', 'p98'], default='p98')
    args = p.parse_args()
    
    az, azw, snr = args.azimuth, args.az_width, args.snr_min
    tmin_in = args.tmin if args.tmin is not None else (1100 if az < 340 else 1550)
    tmax_in = args.tmax if args.tmax is not None else (1950 if az < 340 else 1850)
    print(f"方位角 {az}° ±{azw}°, SNR≥{snr}, 时窗 {tmin_in}-{tmax_in}s")
    
    taup = TauPyModel(model='iasp91')
    
    # ── Load ──
    proc_files = sorted(glob.glob(os.path.join(PROC_DIR, '*.BHZ*_proc.sac')))
    seen = {}  # sta → data
    for pf in proc_files:
        sta = os.path.basename(pf).split('.')[1]
        if sta in seen: continue
        rf = find_raw(sta, RAW_DIR)
        if rf is None: continue
        try:
            tr = read(rf, headonly=True)[0]; sac = tr.stats.sac
            diff = min(abs((sac.az or 0) - az), abs((sac.az or 0) - az + 360), abs((sac.az or 0) - az - 360))
            if diff > azw: continue
            seen[sta] = {'sta':sta, 'az':sac.az or 0, 'dist_km':sac.dist or 0,
                        'stla':sac.stla or 0, 'stlo':sac.stlo or 0, 'pf':pf}
        except: continue
    
    exc = [x.strip() for x in args.exclude.split(',') if x.strip()]
    for s in exc:
        seen.pop(s, None)
    
    stations = list(seen.values())
    
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
        
        tr_p = read(s['pf'])[0]
        origin = UTCDateTime(year=tr_p.stats.sac.nzyear, julday=tr_p.stats.sac.nzjday,
            hour=tr_p.stats.sac.nzhour, minute=tr_p.stats.sac.nzmin,
            second=tr_p.stats.sac.nzsec, microsecond=(tr_p.stats.sac.nzmsec or 0)*1000
        ) + tr_p.stats.sac.o
        b = tr_p.stats.starttime - origin; delta = tr_p.stats.delta
        t = np.arange(tr_p.stats.npts) * delta + b
        db = bandpass(tr_p.data.astype(np.float64), FMIN, FMAX, 1/delta, corners=4, zerophase=True)
        env = envelope(db)
        s['t'] = t; s['db'] = db; s['env'] = env; s['delta'] = delta
        
        # SNR
        if s['T-P']:
            n_idx = (t >= 800) & (t <= s['T-P'] - 100)
            s_idx = (t >= s['T-P']) & (t <= min(s['T-P'] + 180, t[-1]))
            nr = np.sqrt(np.mean(env[n_idx]**2)) if np.any(n_idx) else 1e-30
            sr = np.sqrt(np.mean(env[s_idx]**2)) if np.any(s_idx) else 0
            s['snr'] = sr / nr if nr > 0 else 0
        else:
            s['snr'] = 0
    
    stations = [s for s in stations if s['snr'] >= snr]
    stations.sort(key=lambda s: s['land_km'])
    n = len(stations)
    print(f"  {n} 台站")
    
    for s in stations:
        print(f"  {s['sta']:6s} land={s['land_km']:.0f}km SNR={s['snr']:.1f}")
    
    if n == 0: print("无台站"); return
    
    # ── Global scaling ──
    all_abs = []
    for s in stations:
        m = (s['t'] >= tmin_in) & (s['t'] <= tmax_in); all_abs.extend(np.abs(s['db'][m]))
    if args.norm == 'p98':
        scale_val = np.percentile(all_abs, 98) if all_abs else 1e-30
        label = 'P98 norm'
    else:
        scale_val = np.max(np.abs(all_abs)) if all_abs else 1e-30
        label = 'max norm'
    if scale_val == 0: scale_val = 1e-30
    
    land_vals = [s['land_km'] for s in stations]
    lmin, lmax = min(land_vals), max(land_vals)
    lr = max(lmax - lmin, 50)
    spacing = lr / max(n - 1, 1) if n > 1 else lr
    amp_s = spacing * 0.28
    
    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.5)))
    
    for s in stations:
        y0 = s['land_km']
        m = (s['t'] >= tmin_in) & (s['t'] <= tmax_in)
        tp = s['t'][m]; dn = s['db'][m] / scale_val; en = s['env'][m] / scale_val
        
        ax.fill_between(tp, y0, y0 + en * amp_s, color='#e74c3c', alpha=0.18, lw=0)
        ax.plot(tp, y0 + en * amp_s, color='#e74c3c', lw=0.8, alpha=0.8)
        ax.plot(tp, y0 + dn * amp_s, 'k-', lw=0.35, alpha=0.7)
        ax.axhline(y=y0, color='gray', lw=0.15, alpha=0.15)
        
        for pk, pc, mk in [('T-P','#1f77b4','o'),('T-S','#d62728','D'),('T-Love','#9467bd','s')]:
            ta = s.get(pk, 0)
            if tmin_in <= ta <= tmax_in and ta > 0:
                idx = np.argmin(np.abs(tp - ta))
                if idx < len(dn):
                    vs = y0 + dn[idx] * amp_s
                    ax.plot(ta, vs, marker=mk, color=pc, markersize=5, markeredgewidth=0, zorder=10)
        
        ax.text(tmin_in + 10, y0, f"{s['sta']} ({s['land_km']:.0f}km)", fontsize=6.5, fontfamily='monospace', fontweight='bold', va='center', ha='left')
    
    ax.set_xlim(tmin_in, tmax_in)
    yp = lr * 0.06
    ax.set_ylim(lmin - yp - amp_s, lmax + yp + amp_s * 1.3)
    ax.set_xlabel('Time from origin (s)', fontsize=11)
    ax.set_ylabel('Land distance (km)', fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(labelsize=8)
    ax.grid(True, axis='x', alpha=0.08, linestyle=':')
    
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0],[0], color='black', lw=0.5, label='2-8 Hz waveform'),
        Line2D([0],[0], color='#e74c3c', lw=1.2, label='Envelope'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=5, label='T-P'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#d62728', markersize=5, label='T-S'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#9467bd', markersize=5, label=f'T-Love'),
    ]
    ax.legend(handles=leg, loc='upper right', fontsize=7, framealpha=0.85, ncol=3)
    
    ax.set_title(f'{az:.0f}° Profile — Record section ({label})  |  {n} stations, SNR≥{snr}  |  {FMIN:.0f}-{FMAX:.0f} Hz',
                 fontsize=10, fontweight='bold')
    
    plt.subplots_adjust(left=0.10, right=0.96, top=0.94, bottom=0.08)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'record_section_{az:.0f}_waveform')
    fig.savefig(out+'.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ {out}.pdf /.png")

if __name__ == '__main__':
    main()
