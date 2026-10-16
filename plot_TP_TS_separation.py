#!/usr/bin/env python3
"""
J59A 台站波形 → T-P / T-S 震相分离分析
"""
import sys
sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass, envelope

RAW = "/mnt/f/sharedate/case/TA.J59A..BHZ.M.2014.013.035104.SAC"
PROC = "/mnt/f/sharedate/case/process/TA.J59A..BHZ.M.2014.013.035104_proc.sac"
OUT = "/mnt/f/sharedate/case/J59A_TP_TS_separation.png"

# ===== 读取波形 =====
tr_raw = read(RAW)[0]
tr_proc = read(PROC)[0]

# 时间轴 (相对于发震时刻 origin time)
t_raw = tr_raw.times() + (tr_raw.stats.sac.b - 0)  # b 是相对发震的秒数

# 实际数据（用原始做带通，处理后已做）
data_raw = tr_raw.data.astype(np.float64)
data_proc = tr_proc.data.astype(np.float64)

# 若处理后长度不同，对齐
min_len = min(len(data_raw), len(data_proc))
data_raw = data_raw[:min_len]
data_proc = data_proc[:min_len]

# ===== 走时信息 =====
# 海洋段: 2289 km / 1.48 km/s = 1547 s
# 陆地 P: 524 km → TauP = 71 s
# 陆地 S: 524 km → TauP = 127 s
T_OCEAN = 1547.0   # s
T_LAND_P = 71.0     # s
T_LAND_S = 127.0    # s
T_TP = T_OCEAN + T_LAND_P   # 1618 s = 27.0 min
T_TS = T_OCEAN + T_LAND_S   # 1674 s = 27.9 min
T_DIFF = T_TS - T_TP         # 56 s

# 原始数据带通滤波 (1-8 Hz)
delta = tr_raw.stats.delta
data_bp = bandpass(data_raw, 1.0, 8.0, 1/delta, corners=4, zerophase=True)

# 包络
env_bp = envelope(data_bp)

# ===== 绘图 =====
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1, 0.6]})

t_min = t_raw / 60  # 分钟

# --- Panel 1: 原始波形 (灰色) ---
ax = axes[0]
ax.plot(t_min, data_raw, 'gray', lw=0.5, alpha=0.7, label='Raw BHZ')
ax.set_ylabel('Raw (counts)', fontsize=11)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.15, linestyle=':')

# --- Panel 2: 带通滤波 2-8 Hz + 理论到时 ---
ax = axes[1]
ax.plot(t_min, data_bp, 'k', lw=0.6, label='2-8 Hz BP')
ax.plot(t_min, env_bp, 'blue', lw=0.8, alpha=0.6, label='Envelope')

# 标注理论到时
ymin, ymax = ax.get_ylim()
for tt, label, color, ls in [
    (T_TP/60, f'T-P = {T_TP/60:.1f} min', 'red', '--'),
    (T_TS/60, f'T-S = {T_TS/60:.1f} min', 'darkorange', '-.'),
]:
    ax.axvline(x=tt, color=color, lw=1.5, ls=ls, alpha=0.8)
    ax.text(tt + 0.15, ymax * 0.85, label, color=color, fontsize=10,
            fontweight='bold', rotation=90, va='top')

ax.set_ylabel('1-8 Hz BP', fontsize=11)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.15, linestyle=':')

# --- Panel 3: 处理后波形 (去仪器响应+滤波) ---
ax = axes[2]
ax.plot(t_min, data_proc, 'teal', lw=0.6, label='Processed (rm inst. resp + 1-8 Hz)')

for tt, label, color, ls in [
    (T_TP/60, f'T-P', 'red', '--'),
    (T_TS/60, f'T-S', 'darkorange', '-.'),
]:
    ax.axvline(x=tt, color=color, lw=1.5, ls=ls, alpha=0.8)

ax.set_xlabel('Time after origin (min)', fontsize=12)
ax.set_ylabel('Processed', fontsize=11)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.15, linestyle=':')

# 时间窗口: 聚焦 T波到达前后
T_FOCUS = (900/60, 2600/60)  # 15-43 min
ax.set_xlim(T_FOCUS)

fig.suptitle(
    f'J59A (BHZ)  T-P / T-S Separation  |  '
    f'Δ=2814 km  |  ocean: 2289 km ({T_OCEAN/60:.0f} min)  |  '
    f'land: 524 km → T-P {T_LAND_P:.0f}s, T-S {T_LAND_S:.0f}s  |  '
    f'ΔT(P-S) = {T_DIFF:.0f} s',
    fontsize=9, fontweight='bold', y=0.99
)

plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"✅ {OUT}")

# ===== 统计 =====
print(f"\n========== T-P / T-S 震相分离分析 ==========")
print(f"  台站: J59A (43.46°N, 74.50°W)")
print(f"  震中距: 2814 km (25.3°)")
print(f"  海洋段: 2289 km → {T_OCEAN/60:.1f} min (声速 1.48 km/s)")
print(f"  陆地 P: 524 km → {T_LAND_P:.0f} s (TauP iasp91)")
print(f"  陆地 S: 524 km → {T_LAND_S:.0f} s (TauP iasp91)")
print(f"  T-P 到时: {T_TP/60:.1f} min ({T_TP:.0f} s)")
print(f"  T-S 到时: {T_TS/60:.1f} min ({T_TS:.0f} s)")
print(f"  ΔT(S-P):  {T_DIFF:.0f} s")
print(f"")

# 在理论到时附近搜索包络峰值
for name, tt_s, color in [('T-P', T_TP, 'red'), ('T-S', T_TS, 'darkorange')]:
    # 在理论到时 ±15s 窗口内找包络峰值
    idx_tt = int((tt_s - t_raw[0]) / delta) if t_raw[0] <= tt_s else 0
    if 0 < idx_tt < len(env_bp):
        win = int(15 / delta)
        lo = max(0, idx_tt - win)
        hi = min(len(env_bp), idx_tt + win)
        peak_idx = lo + np.argmax(env_bp[lo:hi])
        peak_tt = t_raw[peak_idx] / 60
        offset = peak_tt - tt_s/60
        print(f"  {name}: 理论 {tt_s/60:.2f} min, 包络峰值 {peak_tt:.2f} min, 偏差 {offset*60:+.0f} s")

plt.close()
