#!/usr/bin/env python3
"""原始风格 — 全部350台站，单面板记录剖面"""
import sys, os, glob
sys.path.insert(0, '/home/philiay/shared/codex/t_phase_analysis')
sys.path.insert(0, '/home/philiay/shared/codex')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import numpy as np
from obspy import read
from obspy.signal.filter import envelope as obs_env
from obspy.geodetics.base import degrees2kilometers
from geographiclib.geodesic import Geodesic
from t_phase_analysis.traveltime_profile import get_track, tt_t

EVLA, EVLO = 18.99, -66.87; EVDP = 10.0
PROC = "/mnt/f/sharedate/case/2014-01-13-mw63-process/sacprocess"
OUT  = "/home/philiay/shared/case/2014-01-13-mw63-process/waveform_contrast/oi/all_polar_analysis"
os.makedirs(OUT, exist_ok=True)
geod = Geodesic.WGS84

stations = []
for fn in sorted(glob.glob(f"{PROC}/*.BHZ*_proc.sac")):
    try:
        tr = read(fn, headonly=True)[0]; sac = tr.stats.sac
        stla = getattr(sac,'stla',0); stlo = getattr(sac,'stlo',0)
        if stla == 0: continue
        g = geod.Inverse(EVLA, EVLO, stla, stlo)
        az = g['azi1']; dist = g['s12']/1000
        if az < 0: az += 360
        stations.append(dict(sta=fn.split('.')[1], stla=stla, stlo=stlo, az=az, dist=dist, fn=fn))
    except: continue
print(f"{len(stations)} stations")

for s in stations:
    try:
        track = get_track(EVLO, EVLA, s['stlo'], s['stla'], 0.02, ref_el=-1000, plot_map=False, plot_elevation=False)
        tr = tt_t(track, ref_el=-1000, source_depth_km=EVDP, ignore_small_islands=True)
        s['tp'] = tr['tt_t'] if tr else 0
        s['land_km'] = degrees2kilometers(tr['dist_land']) if tr and 'dist_land' in tr else 0
    except:
        s['tp'] = 0
        s['land_km'] = 0

stations.sort(key=lambda x: x['land_km'])

T0, T1 = 1200, 2160
fig, ax = plt.subplots(figsize=(18, 10))

n = len(stations)
ds = [s['land_km'] for s in stations]; dmin, dmax = min(ds), max(ds)
dr = max(dmax-dmin, 50); amp = dr / max(n-1, 1) * 0.12

for s in stations:
    y0 = s['land_km']
    tr = read(s['fn'])[0]
    d = tr.data.astype(np.float64); e = obs_env(d)
    t = np.arange(len(d))*tr.stats.delta
    m = (t >= T0) & (t <= T1)
    tv, wv, ev = t[m], d[m], e[m]
    mx = max(np.max(np.abs(wv)), 1e-30)
    wn = wv/mx; en = ev/mx
    ax.fill_between(tv, y0, y0+en*amp, color='#e74c3c', alpha=0.10, lw=0)
    ax.plot(tv, y0+en*amp, color='#e74c3c', lw=0.3, alpha=0.6)
    ax.plot(tv, y0+wn*amp, 'k-', lw=0.1, alpha=0.4)
    tp = s.get('tp', 0)
    if T0 <= tp <= T1:
        idx = np.argmin(np.abs(tv - tp))
        if idx < len(wn):
            ax.plot(tp, y0+wn[idx]*amp, 'o', color='#1f77b4', markersize=2, markeredgewidth=0, zorder=10)

yp = dr*0.05
ax.set_ylim(dmin-yp-amp, dmax+yp+amp*1.3)
ax.set_xlim(T0, T1)
ax.set_xlabel('Time from origin (s)', fontsize=12)
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.tick_params(labelsize=9)
ax.grid(True, axis='x', alpha=0.05, linestyle=':')
ax.set_ylabel('Epicentral distance (km)', fontsize=12)
ax.set_title(f'2014-01-13 Mw 6.3 Puerto Rico — All {n} stations', fontsize=13, fontweight='bold')

leg = [Line2D([0],[0],color='black',lw=0.3,label='2-8 Hz waveform'),
       Line2D([0],[0],color='#e74c3c',lw=0.8,label='Envelope'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor='#1f77b4',markersize=4,label='T-P theory')]
ax.legend(handles=leg, loc='upper right', fontsize=9)
plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.06)

fig.savefig(f"{OUT}/figure2a_record_section_v3_all.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\n✅ {OUT}/figure2a_record_section_v3_all.png")
