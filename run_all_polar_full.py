#!/usr/bin/env python3
"""全台站极化分析 — 312 台三分量 → JSON + GMT 汇总图"""
import sys,os,json,numpy as np
sys.path.insert(0,'/mnt/f/sharedate/T_Phase_Analysis-main')
from obspy import read,UTCDateTime
from obspy.geodetics.base import locations2degrees,gps2dist_azimuth
from obspy.taup import TauPyModel

PROC='/mnt/f/sharedate/case/sacprocess'
OUT='/home/philiay/shared/oi/all_polar_analysis'
EVLO,EVLA=-66.87,18.99
os.makedirs(OUT,exist_ok=True)
taup=TauPyModel(model='iasp91')

# ── 找到所有有三量的台站 ──
import glob
bhz_files=sorted(glob.glob(f'{PROC}/*BHZ*_proc.sac'))
stations=[]
for f in bhz_files:
    base=os.path.basename(f)
    parts=base.split('.')
    if len(parts)<5:continue
    net,sta=parts[0],parts[1]
    bhn=f'{PROC}/{net}.{sta}..BHN.M.2014.013.035104_proc.sac'
    bhe=f'{PROC}/{net}.{sta}..BHE.M.2014.013.035104_proc.sac'
    if os.path.exists(bhn)and os.path.exists(bhe):
        stations.append((net,sta,f,bhn,bhe))
print(f'{len(stations)} stations with full 3-C')

# ── 批量极化 ──
results=[]
for idx,(net,sta,fz,fn,fe) in enumerate(stations):
    try:
        tr_z=read(fz)[0];tr_n=read(fn)[0];tr_e=read(fe)[0]
        t0=max(tr_z.stats.starttime,tr_n.stats.starttime,tr_e.stats.starttime)
        t1=min(tr_z.stats.endtime,tr_n.stats.endtime,tr_e.stats.endtime)
        tr_z.trim(t0,t1);tr_n.trim(t0,t1);tr_e.trim(t0,t1)
        fs=tr_z.stats.sampling_rate
        stla,stlo=tr_z.stats.sac.stla,tr_z.stats.sac.stlo
        d_deg=locations2degrees(EVLA,EVLO,stla,stlo)
        _,baz,_=gps2dist_azimuth(stla,stlo,EVLA,EVLO)
        arrs=taup.get_travel_times(source_depth_in_km=10,distance_in_degree=d_deg,phase_list=['P','S'])
        tp=ts=None
        for a in arrs:
            if a.name=='P'and tp is None:tp=a.time
            elif a.name=='S'and ts is None:ts=a.time
        # Flinn
        dz=tr_z.data.astype(np.float64);dn=tr_n.data.astype(np.float64);de=tr_e.data.astype(np.float64)
        win=3.0;ws=int(win*fs);step=max(ws//4,1);nw=(len(dz)-ws)//step+1
        rect=np.zeros(nw);azi=np.zeros(nw);tmid=np.zeros(nw)
        for i in range(nw):
            i0=i*step;i1=i0+ws;tmid[i]=(i0+i1)/2/fs
            X=np.column_stack([dz[i0:i1],dn[i0:i1],de[i0:i1]]).T
            X=X-X.mean(axis=1,keepdims=True);C=X@X.T/(ws-1)
            e,v=np.linalg.eigh(C);e=e[::-1];v=v[:,::-1]
            l1=e[0];l2=e[1];l3=e[2]
            rect[i]=1.0-(l2+l3)/(2*l1)if l1>0 else 0
            v1=v[:,0];az=np.degrees(np.arctan2(v1[2],v1[1]))
            if az<0:az+=360
            if az>=180:az-=180
            azi[i]=az
        p_az=s_az=None
        if tp:
            m=(tmid>=tp-5)&(tmid<=tp+5)&(rect>0.4)
            if m.sum()>0:p_az=float(np.median(azi[m]))
        if ts:
            m=(tmid>=ts-3)&(tmid<=ts+3)&(rect>0.4)
            if m.sum()>0:s_az=float(np.median(azi[m]))
        results.append({'net':net,'sta':sta,'lon':stlo,'lat':stla,'baz':baz,'p_az':p_az,'s_az':s_az})
    except Exception as e:
        pass
    if idx%50==0:print(f'  {idx}/{len(stations)}')

# ── 保存 ──
clean=[]
for r in results:
    d={}
    for k,v in r.items():
        if v is None:d[k]=-999
        elif isinstance(v,(np.floating,np.integer)):d[k]=float(v)
        else:d[k]=v
    clean.append(d)
with open(f'{OUT}/all_stations_polar.json','w')as f:json.dump(clean,f,indent=1)
print(f'Saved {len(clean)} results')
print(f'P_az valid: {sum(1 for r in clean if r["p_az"]!=-999)}')
print(f'S_az valid: {sum(1 for r in clean if r["s_az"]!=-999)}')
