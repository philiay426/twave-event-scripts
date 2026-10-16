#!/usr/bin/env python3
"""
Figure 2a — azimuth-split record section (341° dividing line)
Sampled for visibility, explicit x-axis in seconds, T-P dots
"""
import sys, os, glob
sys.path.insert(0, '/home/philiay/shared/codex/t_phase_analysis')
sys.path.insert(0, '/home/philiay/shared/codex')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib.lines import Line2D
import numpy as np
from obspy import read
from obspy.signal.filter import envelope as obs_env
from geographiclib.geodesic import Geodesic
from t_phase_analysis.traveltime_profile import get_track, tt_t

EVLA, EVLO = 18.99, -66.87; EVDP = 10.0
PROC = "/mnt/f/sharedate/case/2014-01-13-mw63-process/sacprocess"
OUT  = "/home/philiay/shared/case/2014-01-13-mw63-process/waveform_contrast/oi/all_polar_analysis"
os.makedirs(OUT, exist_ok=True)

AZ_DIV = 341
geod = Geodesic.WGS84

# ── Load all stations ──
raw = []
for fn in sorted(glob.glob(f"{PROC}/*.BHZ*_proc.sac")):
    try:
        tr = read(fn, headonly=True)[0]; sac = tr.stats.sac
        stla = getattr(sac,'stla',0); stlo = getattr(sac,'stlo',0)
        if stla == 0: continue
        g = geod.Inverse(EVLA, EVLO, stla, stlo)
        az = g['azi1']; dist = g['s12']/1000
        if az < 0: az += 360
        raw.append(dict(sta=fn.split('.')[1], stla=stla, stlo=stlo, az=az, dist=dist, fn=fn))
    except: continue
print(f"Total: {len(raw)}")

# Compute T-P
for s in raw:
    try:
        track = get_track(EVLO, EVLA, s['stlo'], s['stla'], 0.02, ref_el=-1000,
                         plot_map=False, plot_elevation=False)
        tr = tt_t(track, ref_el=-1000, source_depth_km=EVDP, ignore_small_islands=True)
        s['tp'] = tr['tt_t'] if tr else 0
    except: s['tp'] = 0

# Split, sample every 4th
low = sorted([s for s in raw if s['az'] < AZ_DIV], key=lambda x: x['dist'])[::4]
high = sorted([s for s in raw if s['az'] >= AZ_DIV], key=lambda x: x['dist'])[::4]
print(f"AZ<{AZ_DIV}°: {len(low)}  |  AZ≥{AZ_DIV}°: {len(high)}")

T0, T1 = 1200, 2160  # 20-36 min
TICKS = [1200, 1400, 1600, 1800, 2000, 2160]
MINOR_TICKS = list(range(1200, 2200, 100))

# ── Draw ──
def draw_panel(ax, group, tmin, tmax):
    n = len(group)
    if n == 0: return
    group.sort(key=lambda x: x['dist'])
    ds = [s['dist'] for s in group]; dmin, dmax = min(ds), max(ds)
    dr = max(dmax-dmin, 50); amp = dr / max(n-1,1) * 0.30

    for s in group:
        y0 = s['dist']
        tr = read(s['fn'])[0]
        d = tr.data.astype(np.float64); e = obs_env(d)
        t = np.arange(len(d))*tr.stats.delta
        m = (t >= tmin) & (t <= tmax)
        tv, wv, ev = t[m], d[m], e[m]
        mx = max(np.max(np.abs(wv)), 1e-30)
        wn = wv/mx; en = ev/mx

        ax.fill_between(tv, y0, y0+en*amp, color='#e74c3c', alpha=0.18, lw=0)
        ax.plot(tv, y0+en*amp, color='#e74c3c', lw=0.6, alpha=0.8)
        ax.plot(tv, y0+wn*amp, 'k-', lw=0.2, alpha=0.6)
        ax.axhline(y0, color='gray', lw=0.04, alpha=0.08)

        tp = s.get('tp', 0)
        if tmin <= tp <= tmax:
            idx = np.argmin(np.abs(tv - tp))
            if idx < len(wn):
                yd = y0 + wn[idx]*amp
                ax.plot(tp, yd, 'o', color='#1f77b4', markersize=5,
                        markeredgecolor='white', markeredgewidth=0.5, zorder=10)

    yp = dr*0.05
    ax.set_ylim(dmin-yp-amp, dmax+yp+amp*1.3)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

draw_panel(a1, low, T0, T1)
draw_panel(a2, high, T0, T1)

for ax, ttl in [(a1, f'AZ < {AZ_DIV}°  ({len(low)})'), (a2, f'AZ ≥ {AZ_DIV}°  ({len(high)})')]:
    ax.set_xlim(T0, T1)
    ax.set_xticks(TICKS)
    ax.set_xticks(MINOR_TICKS, minor=True)
    ax.set_xlabel('Time from origin (s)', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis='x', alpha=0.08, linestyle=':')
    ax.set_title(ttl, fontsize=11, fontweight='bold')

a1.set_ylabel('Epicentral distance (km)', fontsize=11)

leg = [Line2D([0],[0],color='black',lw=0.4,label='2-8 Hz waveform'),
       Line2D([0],[0],color='#e74c3c',lw=1.0,label='Envelope'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor='#1f77b4',markersize=5,label='T-P')]
fig.legend(handles=leg, loc='upper center', ncol=3, fontsize=10, framealpha=0.85)
fig.suptitle('2014-01-13 Mw 6.3 Puerto Rico — Record Section by Azimuth', fontsize=13, fontweight='bold')
plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06, wspace=0.08)

fig.savefig(f"{OUT}/figure2a_record_section_v3.png", dpi=200, bbox_inches='tight')
fig.savefig(f"{OUT}/figure2a_record_section_v3.pdf", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\n✅ PNG/PDF saved")
