#!/usr/bin/env python3
"""Quick fix: merge T-wave data back into full JSON"""
import json,os,sys,numpy as np
sys.path.insert(0,'/mnt/f/sharedate/T_Phase_Analysis-main')
from obspy import read,UTCDateTime
from t_phase_analysis.traveltime_profile import get_track,tt_t

OUT='/home/philiay/shared/oi/all_polar_analysis'
PROC='/mnt/f/sharedate/case/sacprocess'
EVLO,EVLA=-66.87,18.99

# Load full results
with open(f'{OUT}/all_stations_polar.json')as f:results=json.load(f)

# Find good stations (P_az near BAZ)
AZ_THRESH=35
good_idx=[]
for i,r in enumerate(results):
    p_az=r.get('p_az',-999)
    if p_az==-999:continue
    diff=abs(p_az-(r['baz']%180))
    if diff>90:diff=180-diff
    if diff<AZ_THRESH:good_idx.append(i)
print(f'Good: {len(good_idx)} — computing T-P/T-S')

updated=0
for idx in good_idx:
    if idx%20==0:print(f'  {idx}/{len(good_idx)}')
    r=results[idx]
    net,sta=r['net'],r['sta']
    try:
        tr_z=read(f'{PROC}/{net}.{sta}..BHZ.M.2014.013.035104_proc.sac')[0]
        tr_n=read(f'{PROC}/{net}.{sta}..BHN.M.2014.013.035104_proc.sac')[0]
        tr_e=read(f'{PROC}/{net}.{sta}..BHE.M.2014.013.035104_proc.sac')[0]
        t0=max(tr_z.stats.starttime,tr_n.stats.starttime,tr_e.stats.starttime)
        t1=min(tr_z.stats.endtime,tr_n.stats.endtime,tr_e.stats.endtime)
        tr_z.trim(t0,t1);tr_n.trim(t0,t1);tr_e.trim(t0,t1)
        fs=tr_z.stats.sampling_rate
        stla,stlo=tr_z.stats.sac.stla,tr_z.stats.sac.stlo

        track=get_track(EVLO,EVLA,stlo,stla,0.02,ref_el=-1000,plot_map=False,plot_elevation=False)
        tr_tt=tt_t(track,ref_el=-1000,source_depth_km=10,ignore_small_islands=True)
        if not tr_tt:continue
        tt_tp=tr_tt['tt_t'];tt_ts=tr_tt['tt_ocean']+tr_tt['tt_land_s']
        r['tt_tp']=tt_tp;r['tt_ts']=tt_ts

        # Flinn for T-P/T-S windows
        dz=tr_z.data.astype(np.float64);dn=tr_n.data.astype(np.float64);de=tr_e.data.astype(np.float64)
        win=3.0;ws=int(win*fs);step=max(ws//4,1);nw=(len(dz)-ws)//step+1
        rect=np.zeros(nw);azi180=np.zeros(nw);tmid=np.zeros(nw)
        for i in range(nw):
            i0=i*step;i1=i0+ws;tmid[i]=(i0+i1)/2/fs
            X=np.column_stack([dz[i0:i1],dn[i0:i1],de[i0:i1]]).T
            X=X-X.mean(axis=1,keepdims=True);C=X@X.T/(ws-1)
            e,v=np.linalg.eigh(C);e=e[::-1];v=v[:,::-1]
            l1=e[0];rect[i]=1.0-(e[1]+e[2])/(2*l1)if l1>0 else 0
            v1=v[:,0];az=np.degrees(np.arctan2(v1[2],v1[1]))
            if az<0:az+=360
            if az>=180:az-=180
            azi180[i]=az

        tp_az=ts_az=None
        m=(tmid>=tt_tp-5)&(tmid<=tt_tp+5)&(rect>0.4)
        if m.sum()>0:tp_az=float(np.median(azi180[m]))
        m=(tmid>=tt_ts-5)&(tmid<=tt_ts+5)&(rect>0.4)
        if m.sum()>0:ts_az=float(np.median(azi180[m]))
        r['tp_az']=tp_az;r['ts_az']=ts_az
        updated+=1
    except Exception as e:
        r['tp_az']=-999;r['ts_az']=-999

# Save
with open(f'{OUT}/all_stations_polar.json','w')as f:
    clean=[]
    for r in results:
        d={}
        for k,v in r.items():
            if v is None:d[k]=-999
            elif isinstance(v,(np.floating,np.integer)):d[k]=float(v)
            else:d[k]=v
        clean.append(d)
    json.dump(clean,f,indent=1)
tp_ok=sum(1 for r in results if r.get('tp_az',-999)!=-999)
print(f'\nUpdated {updated}, T-P valid: {tp_ok}, Saved {len(clean)}')
