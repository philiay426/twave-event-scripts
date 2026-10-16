#!/usr/bin/env python3
"""
T波综合分析 — 使用预处理(去仪器响应)波形重新跑
与前版区别：波形+频谱用 _proc.sac（去仪器响应→速度）替代原始counts
"""
import sys, os
sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass, envelope
from obspy.geodetics.base import locations2degrees, degrees2kilometers
from scipy.signal import spectrogram as scipy_spectrogram

from t_phase_analysis.traveltime_profile import get_track, tt_t
from t_phase_analysis.snr import snr
from t_phase_analysis.tpef import tpef
from t_phase_analysis.waveform import plot_waveform_panel, plot_spectrogram_panel, compute_spectrogram

# ===== 参数 =====
CASE_DIR = "/mnt/f/sharedate/case"
PROC_DIR = "/mnt/f/sharedate/case/process"
OUT_DIR  = "/mnt/f/sharedate/case/analysis"
EVLO, EVLA = -66.87, 18.99
SOURCE_DEPTH_KM = 10.0

os.makedirs(OUT_DIR, exist_ok=True)

STATIONS = ['J59A']
ALL_RESULTS = {}

for sta in STATIONS:
    print(f"\n{'='*60}")
    print(f"  台站: {sta}")
    print(f"{'='*60}")

    raw_file = f"{CASE_DIR}/TA.{sta}..BHZ.M.2014.013.035104.SAC"
    proc_file = f"{PROC_DIR}/TA.{sta}..BHZ.M.2014.013.035104_proc.sac"

    # ── 读取 ──
    tr_raw = read(raw_file)[0]
    sac = tr_raw.stats.sac
    stla, stlo = sac.stla, sac.stlo

    if not os.path.exists(proc_file):
        print(f"  ❌ 预处理文件不存在: {proc_file}，跳过")
        continue

    try:
        tr_proc = read(proc_file)[0]
    except Exception as e:
        print(f"  ⚠ ObsPy读proc文件失败: {e}")
        print(f"  尝试用 debug_headers=True 绕过...")
        try:
            tr_proc = read(proc_file, debug_headers=True)[0]
        except Exception as e2:
            print(f"  ❌ 仍然失败: {e2}，跳过")
            continue
    print(f"  原始数据: {tr_raw.stats.npts} pts, start={tr_raw.stats.starttime}")
    print(f"  预处理后: {tr_proc.stats.npts} pts, start={tr_proc.stats.starttime}")

    # 从SAC头段读取真实origin time
    origin = UTCDateTime(year=sac.nzyear, julday=sac.nzjday,
                         hour=sac.nzhour, minute=sac.nzmin,
                         second=sac.nzsec, microsecond=sac.nzmsec*1000) + sac.o

    # ── 用预处理后的数据做时间轴 ──
    delta = tr_proc.stats.delta
    b = (tr_proc.stats.starttime - origin)  # seconds from origin
    npts = tr_proc.stats.npts
    t_rel = np.arange(npts) * delta + b  # seconds from origin

    # ── 走时计算 (不变) ──
    try:
        track = get_track(EVLO, EVLA, stlo, stla, 0.02,
                          ref_el=-1000, plot_map=False, plot_elevation=False)
    except Exception as e:
        print(f"  ⚠ get_track 失败: {e}")
        continue
    tt_result = tt_t(track, ref_el=-1000, source_depth_km=SOURCE_DEPTH_KM,
                     ignore_small_islands=True)
    if tt_result is None:
        print(f"  ⚠ tt_t 返回 None")
        continue

    tt_tp_s = tt_result['tt_t']
    tt_ocean_s = tt_result['tt_ocean']
    tt_land_s = tt_result['tt_land']
    tt_land_s_s = tt_result['tt_land_s']
    tt_ts_s = tt_ocean_s + tt_land_s_s
    dist_ocean_km = degrees2kilometers(tt_result['dist_ocean'])
    dist_land_km = degrees2kilometers(tt_result['dist_land'])
    dist_total_km = dist_ocean_km + dist_land_km

    print(f"  震中距: {dist_total_km:.0f} km,  海洋: {dist_ocean_km:.0f} km,  陆地: {dist_land_km:.0f} km")
    print(f"  理论 T-P: {tt_tp_s/60:.2f} min ({tt_tp_s:.0f} s)")
    print(f"  理论 T-S: {tt_ts_s/60:.2f} min ({tt_ts_s:.0f} s)")

    # ── SNR & TPEF (用预处理后数据) ──
    st_snr = read(proc_file)
    t_signal_center = origin + tt_tp_s
    t_noise_center = origin + 1300.0
    try:
        snr_val = snr(st_snr, t_signal=t_signal_center, t_noise=t_noise_center,
                      twin=15.0, method='std', plot_fig=False)
        snr_db = 10 * np.log10(max(snr_val, 1e-10))
        print(f"  SNR (std, ±15s): {snr_val:.2f} ({snr_db:.1f} dB)")
    except Exception as e:
        snr_val = np.nan
        snr_db = np.nan
        print(f"  ⚠ SNR 计算失败: {e}")

    # P/S-wave theoretical times from TauP
    from obspy.taup import TauPyModel
    dist_deg = locations2degrees(EVLA, EVLO, stla, stlo)
    taup = TauPyModel(model='iasp91')
    try:
        p_arr = taup.get_travel_times(source_depth_in_km=SOURCE_DEPTH_KM,
                                       distance_in_degree=dist_deg,
                                       phase_list=['P', 'S'])
        tt_p_taup = tt_s_taup = None
        for arr in p_arr:
            if arr.name == 'P' and tt_p_taup is None:
                tt_p_taup = arr.time
            elif arr.name == 'S' and tt_s_taup is None:
                tt_s_taup = arr.time
        if tt_p_taup is None:
            tt_p_taup = dist_total_km / 8.0
        if tt_s_taup is None:
            tt_s_taup = tt_p_taup * 1.82
    except:
        tt_p_taup = dist_total_km / 8.0
        tt_s_taup = tt_p_taup * 1.82

    t_t_utc = origin + tt_tp_s
    t_p_utc = origin + tt_p_taup

    try:
        tpef_result = tpef(st_snr, t_t=t_t_utc, t_p=t_p_utc, half_width=300,
                           plot_fig=False)
        if tpef_result:
            print(f"  TPEF: {tpef_result['tpef']:.2e},  SNR(energy): {tpef_result['SNR']:.1f},  "
                  f"T_amp: {tpef_result['T_amplitude']:.2e},  duration: {tpef_result['time_duration']:.1f}s")
        else:
            print(f"  ⚠ TPEF 返回 None")
            tpef_result = None
    except Exception as e:
        print(f"  ⚠ TPEF 计算失败: {e}")
        tpef_result = None

    # ── ★ 核心改动：用预处理后的数据做带通滤波+包络 ──
    data_proc = tr_proc.data.astype(np.float64)
    data_bp = bandpass(data_proc, 1.0, 8.0, 1/delta,
                       corners=4, zerophase=True)
    env = envelope(data_bp)

    # ── ★ 时频谱也改用预处理后数据 ──
    f_spec, t_spec, Sxx_db = compute_spectrogram(data_proc, tr_proc.stats.sampling_rate)
    t_spec_rel = t_spec + b  # seconds from origin

    # ── 绘制综合分析图 (4面板) ──
    fig = plt.figure(figsize=(16, 12))

    # Panel A: 波形 + 理论到时
    ax_wf = fig.add_axes([0.06, 0.72, 0.90, 0.24])
    plot_waveform_panel(ax_wf, t_rel, data_bp, env,
                        tt_p=tt_p_taup, tt_s=tt_s_taup, tt_t=tt_tp_s,
                        has_twave=True, label=f'{sta} (BHZ, deconvolved)',
                        arrival_labels=True, ylim_mult=1.5)

    # Panel B: 时频谱
    ax_spec = fig.add_axes([0.06, 0.42, 0.90, 0.26])
    plot_spectrogram_panel(ax_spec, t_spec_rel, f_spec, Sxx_db,
                           tt_p=tt_p_taup, tt_s=tt_s_taup, tt_t=tt_tp_s,
                           has_twave=True, freq_max=10)

    # Panel C: T波窗口放大
    ax_zoom = fig.add_axes([0.06, 0.18, 0.55, 0.20])
    t_mask = (t_rel >= tt_tp_s - 60) & (t_rel <= tt_ts_s + 60)
    if t_mask.sum() > 0:
        tz = t_rel[t_mask] / 60
        dz = data_bp[t_mask]
        ez = env[t_mask]
        ax_zoom.plot(tz, dz, 'k', lw=0.6, label='1-8 Hz BP')
        ax_zoom.plot(tz, ez, 'blue', lw=0.8, alpha=0.6, label='Envelope')
        yl = ax_zoom.get_ylim()
        for tt_s_mark, lbl, clr, ls in [
            (tt_tp_s/60, 'T-P', 'red', '--'),
            (tt_ts_s/60, 'T-S', 'darkorange', '-.'),
        ]:
            ax_zoom.axvline(x=tt_s_mark, color=clr, lw=1.5, ls=ls, alpha=0.8)
            ax_zoom.text(tt_s_mark + 0.05, yl[1]*0.9, lbl, color=clr,
                        fontsize=9, fontweight='bold', rotation=90, va='top')
        ax_zoom.set_xlabel('Time from origin (min)', fontsize=9)
        ax_zoom.set_ylabel('Velocity (m/s)', fontsize=9)
        ax_zoom.set_title(f'T-P / T-S zoom (±60s)  |  ΔT(S-P) = {tt_ts_s-tt_tp_s:.0f}s', fontsize=9)
        ax_zoom.legend(loc='upper right', fontsize=7)
        ax_zoom.grid(True, alpha=0.15, linestyle=':')

    # Panel D: 统计信息
    ax_stats = fig.add_axes([0.64, 0.18, 0.32, 0.20])
    ax_stats.axis('off')
    lines = [
        f"=== {sta}  T-wave Analysis (deconvolved) ===",
        f"",
        f"Location: {stla:.2f}°N, {stlo:.2f}°W  elev={sac.stel:.0f}m",
        f"Epicentral dist: {dist_total_km:.0f} km ({dist_deg:.2f}°)",
        f"",
        f"── Path ──",
        f"Ocean:  {dist_ocean_km:.0f} km → {tt_ocean_s/60:.1f} min (@ 1.48 km/s)",
        f"Land:   {dist_land_km:.0f} km → {tt_land_s/60:.1f} min (TauP iasp91 P)",
        f"",
        f"── Arrivals (theory) ──",
        f"T-P:  {tt_tp_s/60:.2f} min  ({tt_tp_s:.0f} s)",
        f"T-S:  {tt_ts_s/60:.2f} min  ({tt_ts_s:.0f} s)",
        f"ΔT:   {tt_ts_s-tt_tp_s:.0f} s",
        f"",
        f"── Quality ──",
        f"SNR (std):  {snr_val:.2f}  ({snr_db:.1f} dB)" if not np.isnan(snr_val) else "SNR:  N/A",
    ]
    if tpef_result:
        lines += [
            f"TPEF:       {tpef_result['tpef']:.2e}",
            f"T ampl:     {tpef_result['T_amplitude']:.2e} m/s",
            f"T duration: {tpef_result['time_duration']:.1f} s",
            f"SNR(energy):{tpef_result['SNR']:.1f}",
        ]
    for i, line in enumerate(lines):
        color = 'black'
        if line.startswith('==='):
            color = 'darkblue'
            fontweight = 'bold'
            fontsize = 11
        elif line.startswith('──'):
            color = 'gray'
            fontweight = 'bold'
            fontsize = 9
        else:
            fontweight = 'normal'
            fontsize = 8.5
        ax_stats.text(0.02, 0.97 - i*0.045, line, transform=ax_stats.transAxes,
                     fontsize=fontsize, fontweight=fontweight, color=color,
                     fontfamily='monospace', va='top')

    # ── 保存 ──
    out_png = f"{OUT_DIR}/{sta}_full_analysis.png"
    out_pdf = f"{OUT_DIR}/{sta}_full_analysis.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {out_png}")

    # 保存结果
    ALL_RESULTS[sta] = {
        'dist_km': dist_total_km, 'dist_deg': dist_deg,
        'tt_tp_s': tt_tp_s, 'tt_ts_s': tt_ts_s,
        'tt_ocean_s': tt_ocean_s, 'tt_land_s': tt_land_s,
        'dist_ocean_km': dist_ocean_km, 'dist_land_km': dist_land_km,
        'snr': snr_val, 'snr_db': snr_db,
        'tpef': tpef_result,
        'stla': stla, 'stlo': stlo,
    }

# ===== 汇总 =====
print(f"\n{'='*60}")
print(f"  汇总对比（预处理后数据）")
print(f"{'='*60}")
print(f"{'台站':<6} {'距离km':>8} {'海洋km':>8} {'陆地km':>8} {'T-P(min)':>9} {'T-S(min)':>9} {'ΔT(s)':>7} {'SNR(dB)':>8}")
print("-" * 75)
for sta in STATIONS:
    r = ALL_RESULTS.get(sta, {})
    if not r: continue
    snr_str = f"{r['snr_db']:>8.1f}" if not np.isnan(r['snr_db']) else f"{'N/A':>8}"
    print(f"{sta:<6} {r['dist_km']:>8.0f} {r['dist_ocean_km']:>8.0f} {r['dist_land_km']:>8.0f} "
          f"{r['tt_tp_s']/60:>9.2f} {r['tt_ts_s']/60:>9.2f} {r['tt_ts_s']-r['tt_tp_s']:>7.0f} "
          f"{snr_str}")
