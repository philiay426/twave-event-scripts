#!/usr/bin/env python3
"""HRV 全时段三分量图 (2-8 Hz)"""
import sys, os, glob
sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from obspy import read
from obspy.signal.filter import bandpass

PROC_DIR = "/home/philiay/shared/case/sacprocess"
OUT_DIR  = "/home/philiay/shared/case/waveform_contrast/one"

FMIN, FMAX = 2.0, 8.0

# Load 3 components
z = read(f'{PROC_DIR}/IU.HRV.00.BHZ.M.2014.013.035104_proc.sac')[0]
h1 = read(f'{PROC_DIR}/IU.HRV.00.BH1.M.2014.013.035104_proc.sac')[0]
h2 = read(f'{PROC_DIR}/IU.HRV.00.BH2.M.2014.013.035104_proc.sac')[0]

delta = z.stats.delta

# Rotate BH1/BH2 to N/E (BH1=cmpaz=0→N, BH2=cmpaz=90→E)
n_raw = h1.data.astype(np.float64)
e_raw = h2.data.astype(np.float64)

# Bandpass all
z_bp = bandpass(z.data.astype(np.float64), FMIN, FMAX, 1/delta, corners=4, zerophase=True)
n_bp = bandpass(n_raw, FMIN, FMAX, 1/delta, corners=4, zerophase=True)
e_bp = bandpass(e_raw, FMIN, FMAX, 1/delta, corners=4, zerophase=True)

# BAZ = 188.6° → rotate to ZRT
baz = np.radians(188.6)
r = n_bp * np.cos(baz) + e_bp * np.sin(baz)
t = -n_bp * np.sin(baz) + e_bp * np.cos(baz)

min_len = min(len(z_bp), len(r), len(t))
z_bp, r, t = z_bp[:min_len], r[:min_len], t[:min_len]
ts = np.arange(min_len) * delta

# Normalize each component to [-1, 1] for display
comps = [
    ('Z (BHZ)', z_bp, 'black'),
    ('R (Radial)', r, 'green'),
    ('T (Transverse)', t, 'red'),
]

fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)

ymax = 0
for ax, (label, data, color) in zip(axes, comps):
    mx = np.max(np.abs(data)) * 1.1
    if mx > ymax: ymax = mx

for ax, (label, data, color) in zip(axes, comps):
    # Per-component normalization
    mx = np.max(np.abs(data))
    if mx > 0:
        data_plot = data / mx
    else:
        data_plot = data
    
    ax.plot(ts, data_plot, color=color, lw=0.3, alpha=0.7)
    ax.axhline(y=0, color='gray', lw=0.3)
    
    # T-P window region
    ax.axvspan(1606, 1656, color='#1f77b4', alpha=0.08, zorder=0)
    ax.text(1631, ax.get_ylim()[1]*0.92, 'T-P', fontsize=7, color='#1f77b4',
            ha='center', fontweight='bold')
    
    # P and S markers
    for ph, t0 in [('P', 313), ('S', 570)]:
        ax.axvline(x=t0, color='gray', lw=0.5, ls='--', alpha=0.4)
        ax.text(t0, ax.get_ylim()[1]*0.85, ph, fontsize=6, color='gray', ha='center')
    
    ax.set_ylabel(f'{label}\n(norm)', fontsize=8)
    ax.set_ylim(-1.5, 1.5)
    ax.tick_params(labelsize=7)

axes[-1].set_xlabel('Time from origin (s)', fontsize=10)
axes[-1].xaxis.set_major_locator(MultipleLocator(500))
axes[-1].xaxis.set_minor_locator(MultipleLocator(100))

fig.suptitle(
    'HRV (Harvard) — Full time 3-component  2-8 Hz BPF velocity\n'
    f'BAZ={188.6:.1f}°  |  T-P window 1606-1656s (blue shade)',
    fontsize=10, fontweight='bold', y=0.98
)

plt.subplots_adjust(hspace=0.12, left=0.07, right=0.97, top=0.92, bottom=0.06)

out = os.path.join(OUT_DIR, 'HRV_full_3component.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"✅ {out}")
