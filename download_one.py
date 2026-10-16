#!/usr/bin/env python3
"""
下载 raw/ 中所有台站的 BHN/BHE 分量 + SACPZ 仪器响应文件
波形: TA.{STA}..{CHN}.M.2014.013.035104.SAC
响应: SACPZ/SACPZ.TA.{STA}.--.{CHN}
"""
import os
import sys
import glob
import time
import argparse
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

RAW_DIR = "/home/philiay/shared/case/raw"
SACPZ_DIR = os.path.join(RAW_DIR, "SACPZ")
NET = "TA"
T1 = UTCDateTime("2014-01-13T03:51:04")
T2 = UTCDateTime("2014-01-13T05:31:04")
CHANNELS = ["BHN", "BHE"]
TIMEOUT = 120

os.makedirs(SACPZ_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='下载 BHN/BHE 分量 + SACPZ')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--timeout', type=int, default=TIMEOUT)
    p.add_argument('--skip-waveforms', action='store_true', help='只下 SACPZ')
    p.add_argument('--skip-sacpz', action='store_true', help='只下波形')
    return p.parse_args()


def get_stations_list():
    bhz_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.BHZ*.SAC")))
    stations = []
    for f in bhz_files:
        parts = os.path.basename(f).split('.')
        if len(parts) >= 2:
            stations.append(parts[1])
    return sorted(set(stations))


def download_sacpz(client, stations, args):
    """下载所有台站指定通道的 SACPZ"""
    os.makedirs(SACPZ_DIR, exist_ok=True)
    ok, skip, fail = 0, 0, 0

    for sta in stations:
        for chn in CHANNELS:
            out_path = os.path.join(SACPZ_DIR, f"SACPZ.{NET}.{sta}.--.{chn}")

            if os.path.exists(out_path):
                print(f"  ⏭ SACPZ {sta:6s} {chn}  已存在")
                skip += 1
                continue

            if args.dry_run:
                print(f"  📋 SACPZ {sta:6s} {chn}  (dry-run)")
                continue

            print(f"  📄 SACPZ {sta:6s} {chn} ...", end=" ", flush=True)
            try:
                inv = client.get_stations(
                    network=NET, station=sta, channel=chn,
                    starttime=T1, endtime=T2, level="response",
                )
                if len(inv) == 0 or len(inv[0]) == 0:
                    print("❌ 无元数据")
                    fail += 1
                    continue

                # 写 SACPZ
                inv.write(out_path, format="SACPZ")
                print("✅")
                ok += 1
                time.sleep(0.5)

            except Exception as e:
                msg = str(e)[:80]
                print(f"❌ {msg}")
                fail += 1

    return ok, skip, fail


def download_waveforms(client, stations, args):
    """下载波形"""
    ok, skip, fail = 0, 0, 0
    for sta in stations:
        for chn in CHANNELS:
            out_name = f"TA.{sta}..{chn}.M.2014.013.035104.SAC"
            out_path = os.path.join(RAW_DIR, out_name)

            if os.path.exists(out_path):
                print(f"  ⏭ {sta:6s} {chn}  已存在")
                skip += 1
                continue

            if args.dry_run:
                print(f"  📋 {sta:6s} {chn}  (dry-run)")
                continue

            print(f"  ⬇ {sta:6s} {chn} ...", end=" ", flush=True)
            try:
                st = client.get_waveforms(
                    network=NET, station=sta, location="*",
                    channel=chn, starttime=T1, endtime=T2,
                )
                if len(st) == 0:
                    print("❌ 无数据")
                    fail += 1
                    continue

                tr = st[0]
                tr.write(out_path, format="SAC")
                print(f"✅ {tr.stats.npts} pts")
                ok += 1
                time.sleep(1)

            except Exception as e:
                print(f"❌ {str(e)[:80]}")
                fail += 1

    return ok, skip, fail


def main():
    args = parse_args()
    stations = get_stations_list()
    print(f"raw/ 下共 {len(stations)} 个台站")
    print(f"  通道: {CHANNELS}")
    print()

    client = Client("IRIS", timeout=args.timeout)

    w_ok = w_skip = w_fail = 0
    s_ok = s_skip = s_fail = 0

    if not args.skip_waveforms:
        print("── 下载波形 ──")
        w_ok, w_skip, w_fail = download_waveforms(client, stations, args)
        print()

    if not args.skip_sacpz:
        print("── 下载 SACPZ ──")
        s_ok, s_skip, s_fail = download_sacpz(client, stations, args)
        print()

    print(f"波形: 下载 {w_ok}, 跳过 {w_skip}, 失败 {w_fail}")
    print(f"SACPZ: 下载 {s_ok}, 跳过 {s_skip}, 失败 {s_fail}")


if __name__ == "__main__":
    main()
