#!/usr/bin/env python3
"""

使用方法 
# 1. 一键处理 raw/ 下全部 SAC
python3 process_obs.py --all

# 2. 只处理新增的（跳过已有 proc 的）
python3 process_obs.py --all --skip-existing

# 3. 指定单个或多个
python3 process_obs.py TA.J59A..BHZ.M.2014.013.035104.SAC  ****.SAC

流程：去均值、去趋势、去仪器响应 → OUT_RESP_DIR
      带通滤波2-8Hz → OUT_PROC_DIR
      时频分析、到时计算
      各步骤图 → IMG_DIR
      
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from obspy.io.sac import SACTrace
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel
from obspy.signal.invsim import simulate_seismometer

sys.path.insert(0, '/mnt/f/sharedate/T_Phase_Analysis-main')
from t_phase_analysis.traveltime_profile import get_track, tt_t
from t_phase_analysis.waveform import compute_spectrogram, plot_spectrogram_panel

# ===== 路径 =====
BASE_DIR = "/mnt/f/sharedate/case"
RAW_DIR = os.path.join(BASE_DIR, "raw")
SACPZ_DIR = os.path.join(RAW_DIR, "SACPZ")
OUT_RESP_DIR = os.path.join(BASE_DIR, "sacprocess", "remove_response")
OUT_PROC_DIR = os.path.join(BASE_DIR, "sacprocess")
IMG_DIR = os.path.join(OUT_PROC_DIR, "waveform_image")

os.makedirs(OUT_RESP_DIR, exist_ok=True)
os.makedirs(OUT_PROC_DIR, exist_ok=True)

# ===== 处理参数 =====
TAPER_MAX_PERCENTAGE = 0.05            # 5% cosine taper
PRE_FILT = (0.001, 0.005, 45.0, 50.0)  # 极宽 pre_filt，最小滤波效应
OUTPUT_UNIT = "VEL"                     # 输出速度 (m/s)
WATER_LEVEL = 60.0                      # water level (dB), 防止除零

BP_FREQMIN = 2.0   # 带通滤波低频
BP_FREQMAX = 8.0   # 带通滤波高频
BP_CORNERS = 4      # 滤波阶数

# ===== 辅助函数 =====

def parse_sacpz(filepath):
    """
    解析 SAC polezero 文件，返回 poles, zeros, A0, sensitivity.
    只读第**一个** response 段（多段拼接文件只取第一段）。
    SACPZ 格式:
      ZEROS N
      POLES M
      CONSTANT = A0 × sensitivity
      * SENSITIVITY = ...  (可选注释)
      * A0 = ...            (可选注释)
    """
    zeros = []
    poles = []
    constant = 1.0
    a0 = None
    sensitivity = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_zeros = False
    in_poles = False
    found_first = False  # 是否找到了第一个 ZEROS
    done = False         # 读完第一段后停止

    for line in lines:
        if done:
            break
        sline = line.strip()

        # ── 注释行：在第一段之前解析 A0 / SENSITIVITY ──
        if sline.startswith('*') and not found_first:
            if 'SENSITIVITY' in sline.upper():
                parts = sline.split()
                for i, p in enumerate(parts):
                    if 'SENSITIVITY' in p.upper() and i+2 < len(parts):
                        try:
                            sensitivity = float(parts[i+2])
                        except ValueError:
                            pass
            if sline.upper().lstrip('* ').startswith('A0'):
                parts = sline.split()
                for i, p in enumerate(parts):
                    if p.upper() == 'A0' and i+2 < len(parts):
                        try:
                            a0 = float(parts[i+2])
                        except (ValueError, IndexError):
                            pass

        # ── ZEROS ──
        if sline.startswith('ZEROS'):
            if not found_first:
                found_first = True
                in_zeros = True
                in_poles = False
            else:
                # 遇到第二个 ZEROS → 第一段结束
                done = True
            continue

        # ── POLES ──
        if sline.startswith('POLES') and found_first:
            in_zeros = False
            in_poles = True
            continue

        # ── CONSTANT ──
        if sline.startswith('CONSTANT') and found_first:
            parts = sline.split()
            if len(parts) >= 2:
                constant = float(parts[1])
            continue

        # ── 空行 / 注释分隔 → 可能结束 ──
        if sline == '' or sline.startswith('*') and found_first:
            # 如果已经在第一段内并且遇到注释行之间的空行，不结束
            # 但如果 ZEROS/POLES 之后紧跟空行+注释，需要判断
            continue

        # ── 数据行 (复数) ──
        parts = sline.split()
        if len(parts) >= 2 and found_first and not done:
            try:
                real = float(parts[0])
                imag = float(parts[1])
                if in_zeros:
                    zeros.append(complex(real, imag))
                elif in_poles:
                    poles.append(complex(real, imag))
            except ValueError:
                continue

    # 推导缺失值: CONSTANT = A0 × sensitivity
    if a0 is None and sensitivity is not None:
        a0 = constant / sensitivity
    elif sensitivity is None and a0 is not None:
        sensitivity = constant / a0
    elif a0 is None and sensitivity is None:
        # 无法分离，用 CONSTANT 当 gain，sensitivity=1.0
        # (ObsPy simulate 会保留物理单位，只要 gain 对)
        a0 = constant
        sensitivity = 1.0

    print(f"  [SACPZ] A0={a0:.4e}, sensitivity={sensitivity:.4e}, CONSTANT={constant:.4e}")
    return np.array(poles), np.array(zeros), a0, sensitivity

def remove_response_obs(tr, paz_file, pre_filt, water_level):
    """
    用 simulate_seismometer 去仪器响应，返回速度 (m/s)。
    （改用 simulate_seismometer 替代 tr.simulate + differentiate，
      避免 tr.simulate 的振幅压缩问题）
    """
    poles, zeros, a0, sensitivity = parse_sacpz(paz_file)

    data = tr.data.astype(np.float64)
    delta = tr.stats.delta

    data_out = simulate_seismometer(
        data, delta,
        paz_remove={'poles': poles, 'zeros': zeros, 'gain': a0, 'sensitivity': sensitivity},
        pre_filt=pre_filt,
        water_level=water_level,  # water_level in dB (e.g. 60.0 = -60 dB)
        paz_simulate=None,
        simulate_sensitivity=True,
    )

    # differentiate: displacement → velocity
    vel = np.gradient(data_out, delta)

    tr_out = tr.copy()
    tr_out.data = vel
    return tr_out


def apply_bandpass(tr, freqmin, freqmax, corners):
    """零相位带通滤波"""
    tr_copy = tr.copy()
    tr_copy.filter('bandpass', freqmin=freqmin, freqmax=freqmax,
                   corners=corners, zerophase=True)
    return tr_copy


def write_sac(tr, filepath):
    """用 SACTrace 写 SAC 文件，保留头段"""
    sac = SACTrace.from_obspy_trace(tr)
    sac.write(filepath)
    print(f"  → {filepath}")


def save_waveform_png(tr, origin, filepath, title, ylabel='Amplitude',
                      t_min=None, t_max=None, markers=None):
    """将波形保存为 PNG；markers=[(time_s, 'P', 'green'), ...]"""
    b = tr.stats.starttime - origin
    t = np.arange(tr.stats.npts) * tr.stats.delta + b
    data = tr.data

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t, data, 'k-', lw=0.4)
    ax.set_xlabel('Time from origin (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if t_min is not None: ax.set_xlim(left=t_min)
    if t_max is not None: ax.set_xlim(right=t_max)
    ax.grid(True, alpha=0.2)

    # 到时标注
    if markers:
        y_min, y_max = ax.get_ylim()
        # 不同震相用不同高度，避免重叠
        y_offsets = {'P': 0.92, 'S': 0.78, 'T-P': 0.92, 'T-S': 0.68}
        for t_arr, label, color in markers:
            y_pos = y_max * y_offsets.get(label, 0.92)
            ax.axvline(x=t_arr, color=color, lw=1.0, ls='--', alpha=0.7)
            ax.text(t_arr, y_pos, f'{label}={t_arr:.0f}s',
                    color=color, fontsize=7, fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📷 {filepath}")


def _compute_arrivals(stla, stlo, dist_deg):
    """计算 P/S (TauP) 和 T-P (tt_t) 到时；返回 dict"""
    arrivals = {}
    # TauP P/S
    taup = TauPyModel(model='iasp91')
    try:
        arrs = taup.get_travel_times(source_depth_in_km=10.0,
                                      distance_in_degree=dist_deg,
                                      phase_list=['P', 'S'])
        for a in arrs:
            if a.name == 'P' and 'P' not in arrivals: arrivals['P'] = a.time
            elif a.name == 'S' and 'S' not in arrivals: arrivals['S'] = a.time
    except:
        pass
    # tt_t T-P
    try:
        track = get_track(-66.87, 18.99, stlo, stla, 0.02,
                          ref_el=-1000, plot_map=False, plot_elevation=False)
        tr_tt = tt_t(track, ref_el=-1000, source_depth_km=10.0, ignore_small_islands=True)
        if tr_tt:
            arrivals['T-P'] = tr_tt['tt_t']
            arrivals['T-S'] = tr_tt['tt_ocean'] + tr_tt['tt_land_s']
    except:
        pass
    return arrivals


def save_spectrogram_png(tr, origin, stla, stlo, dist_km, dist_deg, filepath, sta,
                         t_min=-500, t_max=3500):
    """保存时频谱图，带 P/S/T 到时标注"""
    b = tr.stats.starttime - origin
    data = tr.data.astype(np.float64)
    f_spec, t_spec, Sxx_db = compute_spectrogram(data, tr.stats.sampling_rate)
    t_spec_rel = t_spec + b

    arrivals = _compute_arrivals(stla, stlo, dist_deg)
    tt_p = arrivals.get('P')
    tt_s = arrivals.get('S')
    tt_t_val = arrivals.get('T-P')
    has_twave = 'T-P' in arrivals

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_spectrogram_panel(ax, t_spec_rel, f_spec, Sxx_db,
                           tt_p=tt_p, tt_s=tt_s, tt_t=tt_t_val,
                           has_twave=has_twave, freq_max=10, cmap='inferno')
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel('Time from origin (s)', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_title(f'{sta} Spectrogram  |  {dist_km:.0f} km ({dist_deg:.2f}°)  |  '
                 f'deconvolved to velocity', fontsize=10, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {filepath}")


# ===== 预处理函数 =====

def process_station(sac_file):
    """
    处理单个 SAC 文件。
    sac_file: raw/ 下的 SAC 文件名 (不含路径)
    """
    raw_path = os.path.join(RAW_DIR, sac_file)
    if not os.path.exists(raw_path):
        print(f"  ❌ 文件不存在: {raw_path}")
        return False

    # 读 SAC
    st = read(raw_path)
    tr = st[0]
    sac_hdr = tr.stats.sac

    # 台站信息 — 安全读取，缺失的字段给默认值
    sac_hdr = tr.stats.sac
    net = (sac_hdr.knetwk or '').strip()
    sta = (sac_hdr.kstnm or '').strip()
    chn = (sac_hdr.kcmpnm or '').strip()
    # khole 可能不存在 (部分 SAC 文件无此字段)
    try:
        loc = sac_hdr.khole
    except AttributeError:
        loc = None
    if loc is None or str(loc).strip() in ('', '-12345', 'None'):
        loc = '--'
    else:
        loc = str(loc).strip()

    paz_file = os.path.join(SACPZ_DIR, f"SACPZ.{net}.{sta}.{loc}.{chn}")
    if not os.path.exists(paz_file):
        print(f"  ❌ SACPZ 文件不存在: {paz_file}")
        return False

    print(f"  台站: {net}.{sta}.{loc}.{chn}  |  SACPZ: OK")

    # 计算发震时刻 (用于时间轴)
    sac_hdr = tr.stats.sac
    origin = UTCDateTime(year=sac_hdr.nzyear, julday=sac_hdr.nzjday,
                         hour=sac_hdr.nzhour, minute=sac_hdr.nzmin,
                         second=sac_hdr.nzsec,
                         microsecond=(getattr(sac_hdr, 'nzmsec', 0) or 0) * 1000) \
             + (getattr(sac_hdr, 'o', 0) or 0)

    # 台站坐标 + 震中距 (优先 SAC 头段，缺失时读 BHZ 文件)
    try:
        stla = sac_hdr.stla
        stlo = sac_hdr.stlo
    except AttributeError:
        # 读同台站 BHZ 取坐标 — 尝试多种命名格式
        bhz_candidates = [
            f"{net}.{sta}.{loc}.BHZ.M.2014.013.035104.SAC",
            f"{net}.{sta}..BHZ.M.2014.013.035104.SAC",
            f"TA.{sta}..BHZ.M.2014.013.035104.SAC",
        ]
        stla = stlo = None
        for bc in bhz_candidates:
            bhz_file = os.path.join(RAW_DIR, bc)
            if os.path.exists(bhz_file):
                try:
                    bhz_tr = read(bhz_file, headonly=True)[0]
                    stla = bhz_tr.stats.sac.stla
                    stlo = bhz_tr.stats.sac.stlo
                    break
                except:
                    continue
        if stla is None:
            print(f"  ❌ 无法获取台站坐标")
            return False
    dist_deg = locations2degrees(18.99, -66.87, stla, stlo)
    from obspy.geodetics.base import degrees2kilometers
    dist_km = degrees2kilometers(dist_deg)

    # 理论到时 (共用)
    arrivals = _compute_arrivals(stla, stlo, dist_deg)
    parts = [f"{k}={v:.0f}s" for k, v in arrivals.items()]
    print(f"  到时: {'  '.join(parts)}")

    # ── 图片输出目录 (含通道名) — 只输出垂直分量 ──
    if 'BHZ' not in chn.upper():
        img_sta_dir = None  # skip non-vertical images
    else:
        img_sta_dir = os.path.join(IMG_DIR, f"{sta}_{chn}")
    
    if img_sta_dir:
        # ── 1. 保存原始波形图 ──
        save_waveform_png(tr, origin, os.path.join(img_sta_dir, 'raw.png'),
                          f'{sta} — Raw waveform', ylabel='Counts')

    # ── 2. 去均值 + 去趋势 ──
    tr.detrend('linear')
    tr.detrend('demean')

    # ── 3. 渐变 ──
    tr.taper(max_percentage=TAPER_MAX_PERCENTAGE, type='cosine')

    # ── 4. 去仪器响应 → 速度 ──
    print(f"  去仪器响应...")
    tr_resp = remove_response_obs(tr, paz_file, PRE_FILT, WATER_LEVEL)

    # ── 5. 输出去仪器响应后的波形文件 + 图片 ──
    base_name = sac_file.replace('.SAC', '').replace('.sac', '')
    resp_out = os.path.join(OUT_RESP_DIR, f"{base_name}_resp.sac")
    write_sac(tr_resp, resp_out)
    if img_sta_dir:
        save_waveform_png(tr_resp, origin, os.path.join(img_sta_dir, 'remove_response.png'),
                          f'{sta} — Remove response (velocity)', ylabel='Velocity (m/s)')

    # ── 6. 2-8 Hz 带通滤波 ──
    print(f"  2-8 Hz 带通滤波...")
    tr_proc = apply_bandpass(tr_resp, BP_FREQMIN, BP_FREQMAX, BP_CORNERS)

    # ── 7. 输出最终文件 + 图片 ──
    proc_out = os.path.join(OUT_PROC_DIR, f"{base_name}_proc.sac")
    write_sac(tr_proc, proc_out)
    if img_sta_dir:
        save_waveform_png(tr_proc, origin, os.path.join(img_sta_dir, 'BPF.png'),
                          f'{sta} — 2-8 Hz bandpass ({dist_km:.0f} km)', ylabel='Velocity (m/s)',
                          markers=[(v, k, {'P': '#2ca02c', 'S': '#ff7f0e', 'T-P': '#1f77b4', 'T-S': '#d62728'}[k])
                                   for k, v in arrivals.items()])

    # ── 8. 时频谱图 ──
    if img_sta_dir:
        save_spectrogram_png(tr_proc, origin, stla, stlo, dist_km, dist_deg,
                             os.path.join(img_sta_dir, 'spectrogram.png'), sta)

    return True


# ===== 主入口 =====
if __name__ == "__main__":
    import glob
    import argparse

    p = argparse.ArgumentParser(description='ObsPy SAC 预处理 (去仪器响应 + 带通滤波)')
    p.add_argument('files', nargs='*', help='SAC 文件名 (支持通配符)')
    p.add_argument('--all', action='store_true', help='处理 raw/ 下全部 SAC 文件')
    p.add_argument('--skip-existing', action='store_true', help='跳过已有 proc 文件的台站')
    args = p.parse_args()

    if args.all:
        sac_files = sorted([
            os.path.basename(f) for f in glob.glob(os.path.join(RAW_DIR, '*.SAC'))
        ])
    elif args.files:
        sac_files = []
        for arg in args.files:
            matched = glob.glob(os.path.join(RAW_DIR, arg))
            if matched:
                sac_files.extend([os.path.basename(f) for f in matched])
            else:
                full = os.path.join(RAW_DIR, arg)
                if os.path.exists(full):
                    sac_files.append(arg)
                else:
                    print(f"  ⚠ 找不到: {arg}")
    else:
        print(f"用法:")
        print(f"  {sys.argv[0]} --all                    处理 raw/ 下全部文件")
        print(f"  {sys.argv[0]} --all --skip-existing    跳过已有 proc 的台站")
        print(f"  {sys.argv[0]} TA.J59A..BHZ.SAC         处理单个文件")
        print(f"  {sys.argv[0]} TA.J59A..BHZ.SAC TA.K60A..BHZ.SAC  处理多个")
        sys.exit(1)

    if args.skip_existing:
        filtered = []
        for sf in sac_files:
            base = sf.replace('.SAC', '').replace('.sac', '')
            proc_out = os.path.join(OUT_PROC_DIR, f'{base}_proc.sac')
            if os.path.exists(proc_out):
                print(f"  ⏭ 跳过 (已有): {sf}")
            else:
                filtered.append(sf)
        sac_files = filtered

    if not sac_files:
        print("没有需要处理的文件。")
        sys.exit(0)

    print(f"准备处理 {len(sac_files)} 个文件")
    print(f"  原始数据: {RAW_DIR}/")
    print(f"  SACPZ:    {SACPZ_DIR}/")
    print(f"  去响应:   {OUT_RESP_DIR}/")
    print(f"  最终输出: {OUT_PROC_DIR}/")
    print(f"  pre_filt: {PRE_FILT}")
    print()

    ok = 0
    for sf in sac_files:
        print(f"── {sf} ──")
        try:
            if process_station(sf):
                ok += 1
                print(f"  ✅ 完成\n")
            else:
                print(f"  ❌ 失败\n")
        except Exception as e:
            print(f"  ❌ 异常: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"\n处理完成: {ok}/{len(sac_files)} 成功")
