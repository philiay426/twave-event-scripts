#!/usr/bin/env python3
"""
批量补下载全部台站的 BHN/BHE + SACPZ
1. 从 case/raw 拷已有的 11 台三分量
2. IRIS 下载剩余台站
"""
import os,sys,glob,time,argparse
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

RAW_OLD='/home/philiay/shared/case/raw'
RAW_NEW='/home/philiay/shared/2014-01-13-mw63-raw'
NET='TA';T1=UTCDateTime('2014-01-13T03:51:04');T2=UTCDateTime('2014-01-13T05:31:04')
CHANNELS=['BHN','BHE']

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--timeout',type=int,default=120)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--limit',type=int,default=0,help='最多下载 N 个台站 (0=全部)')
    return p.parse_args()

def main():
    args=parse_args()
    os.makedirs(RAW_NEW,exist_ok=True)
    os.makedirs(os.path.join(RAW_NEW,'SACPZ'),exist_ok=True)

    # ── Step 1: 拷已有 11 台 ──
    copied=0
    for chn in ['BHN','BHE']:
        for f in glob.glob(f'{RAW_OLD}/*.{chn}.*.SAC'):
            base=os.path.basename(f)
            dst=os.path.join(RAW_NEW,base)
            if not os.path.exists(dst):
                os.system(f'cp {f} {dst}')
                copied+=1
    for chn in ['BHN','BHE']:
        for f in glob.glob(f'{RAW_OLD}/SACPZ/SACPZ.*.{chn}'):
            base=os.path.basename(f)
            dst=os.path.join(RAW_NEW,'SACPZ',base)
            if not os.path.exists(dst):
                os.system(f'cp {f} {dst}')
    print(f'Copied {copied} existing files from case/raw/')

    # ── Step 2: 获取需要下载的台站 ──
    bhz_files=sorted(glob.glob(os.path.join(RAW_NEW,'*.BHZ*.SAC')))
    stations=set()
    for f in bhz_files:
        parts=os.path.basename(f).split('.')
        if len(parts)>=2:stations.add((parts[0],parts[1]))
    stations=sorted(stations)
    print(f'Total stations to process: {len(stations)}')

    # ── 检查缺失 ──
    missing=[]
    for net,sta in stations:
        for chn in CHANNELS:
            out=f'{net}.{sta}..{chn}.M.2014.013.035104.SAC'
            if not os.path.exists(os.path.join(RAW_NEW,out)):
                missing.append((net,sta,chn))
    print(f'Missing components: {len(missing)}')

    if args.limit>0:
        # Limit by unique stations
        sta_done=set()
        filtered=[]
        for net,sta,chn in missing:
            if len(sta_done)>=args.limit and sta not in sta_done:continue
            filtered.append((net,sta,chn))
            sta_done.add(sta)
        missing=filtered
        print(f'Limited to {args.limit} stations: {len(missing)} files')

    if args.dry_run:
        for net,sta,chn in missing[:10]:
            print(f'  {sta:6s} {chn}')
        print(f'  ... and {len(missing)-10} more')
        return

    # ── 下载 ──
    client=Client('IRIS',timeout=args.timeout)
    ok=skip=fail=0
    for net,sta,chn in missing:
        out_name=f'{net}.{sta}..{chn}.M.2014.013.035104.SAC'
        out_path=os.path.join(RAW_NEW,out_name)
        if os.path.exists(out_path):skip+=1;continue

        print(f'  ⬇ {sta:8s} {chn}',end=' ',flush=True)
        try:
            st=client.get_waveforms(network=net,station=sta,location='*',
                                     channel=chn,starttime=T1,endtime=T2)
            if len(st)==0:print('❌ no data');fail+=1;continue
            tr=st[0];tr.write(out_path,format='SAC')
            print(f'✅ {tr.stats.npts} pts');ok+=1
            time.sleep(0.3)
        except Exception as e:
            print(f'❌ {str(e)[:60]}');fail+=1;time.sleep(2)

        # Download SACPZ
        sacpz_out=os.path.join(RAW_NEW,'SACPZ',f'SACPZ.{net}.{sta}.--.{chn}')
        if not os.path.exists(sacpz_out):
            try:
                inv=client.get_stations(network=net,station=sta,channel=chn,
                                         starttime=T1,endtime=T2,level='response')
                if len(inv)>0 and len(inv[0])>0:
                    inv.write(sacpz_out,format='SACPZ')
            except:pass

    print(f'\nDownload: {ok} ok, {skip} skipped, {fail} failed')

if __name__=='__main__':main()
