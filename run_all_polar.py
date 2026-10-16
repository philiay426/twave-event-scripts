#!/usr/bin/env python3
"""批量极化分析 → 输出 JSON + GMT 汇总地图"""
import sys,os,json,numpy as np
sys.path.insert(0,'/mnt/f/sharedate/T_Phase_Analysis-main')
from obspy import read,UTCDateTime
from obspy.geodetics.base import locations2degrees,gps2dist_azimuth,degrees2kilometers
from obspy.taup import TauPyModel
from t_phase_analysis.traveltime_profile import get_track,tt_t

PROC='/home/philiay/shared/case/sacprocess'
OUTDIR='/home/philiay/shared/oi/all_polar_analysis'
EVLO,EVLA=-66.87,18.99
os.makedirs(OUTDIR,exist_ok=True)
taup=TauPyModel(model='iasp91')

stations=['Y59A','W57A','U55A','S53A','Q51A','N48A','N63A','K60A','J59A','H57A','F55A']
results=[]

for sta in stations:
    traces={};missing=False
    for chn in ['BHZ','BHN','BHE']:
        f=f'{PROC}/TA.{sta}..{chn}.M.2014.013.035104_proc.sac'
        if not os.path.exists(f):missing=True;break
        traces[chn]=read(f)[0]
    if missing:
        print(f'{sta}: ❌ missing 3C')
        continue
    t0=max(tr.stats.starttime for tr in traces.values())
    t1=min(tr.stats.endtime for tr in traces.values())
    for chn in traces:traces[chn].trim(t0,t1)
    tr_z=traces['BHZ'];fs=tr_z.stats.sampling_rate;sac=tr_z.stats.sac
    stla,stlo=sac.stla,sac.stlo
    dist_deg=locations2degrees(EVLA,EVLO,stla,stlo);dist_km=degrees2kilometers(dist_deg)
    _,baz,_=gps2dist_azimuth(stla,stlo,EVLA,EVLO)
    arrs=taup.get_travel_times(source_depth_in_km=10,distance_in_degree=dist_deg,phase_list=['P','S'])
    tt_p=tt_s=None
    for a in arrs:
        if a.name=='P'and tt_p is None:tt_p=a.time
        elif a.name=='S'and tt_s is None:tt_s=a.time
    dz=tr_z.data.astype(np.float64);dn=traces['BHN'].data.astype(np.float64);de=traces['BHE'].data.astype(np.float64)
    win=3.0;ws=int(win*fs);step=max(ws//4,1);nw=(len(dz)-ws)//step+1
    rect=np.zeros(nw);azi=np.zeros(nw);tmid=np.zeros(nw)
    for i in range(nw):
        i0=i*step;i1=i0+ws;tmid[i]=(i0+i1)/2/fs
        X=np.column_stack([dz[i0:i1],dn[i0:i1],de[i0:i1]]).T;X=X-X.mean(axis=1,keepdims=True)
        C=X@X.T/(ws-1);e,v=np.linalg.eigh(C);e=e[::-1];v=v[:,::-1]
        l1,l2,l3=e[0],e[1],e[2]
        rect[i]=1.0-(l2+l3)/(2*l1)if l1>0 else 0
        v1=v[:,0];az=np.degrees(np.arctan2(v1[2],v1[1]))
        if az<0:az+=360
        if az>=180:az-=180
        azi[i]=az
    p_az=s_az=t_az=None
    if tt_p:
        m=(tmid>=tt_p-5)&(tmid<=tt_p+5)&(rect>0.4)
        if m.sum()>0:p_az=float(np.median(azi[m]))
    if tt_s:
        m=(tmid>=tt_s-3)&(tmid<=tt_s+3)&(rect>0.4)
        if m.sum()>0:s_az=float(np.median(azi[m]))
    track=get_track(EVLO,EVLA,stlo,stla,0.02,ref_el=-1000,plot_map=False,plot_elevation=False)
    tr_tt=tt_t(track,ref_el=-1000,source_depth_km=10,ignore_small_islands=True)
    if tr_tt:
        tt_tp=tr_tt['tt_t']
        m=(tmid>=tt_tp-3)&(tmid<=tt_tp+3)&(rect>0.4)
        if m.sum()>0:t_az=float(np.median(azi[m]))
    results.append({'sta':sta,'lon':stlo,'lat':stla,'dist_km':dist_km,'baz':baz,
                    'p_az':p_az,'s_az':s_az,'t_az':t_az,'tt_p':tt_p,'tt_s':tt_s,'tt_tp':tt_tp if tr_tt else None})
    p_str=f'P_az={p_az:.0f}°'if p_az else'P_az=None'
    s_str=f'S_az={s_az:.0f}°'if s_az else'S_az=None'
    t_str=f'T_az={t_az:.0f}°'if t_az else'T_az=None'
    print(f'{sta}: {p_str}  {s_str}  {t_str}')

with open(f'{OUTDIR}/polarization_results.json','w')as f:
    clean=[]
    for r in results:
        d={}
        for k,v in r.items():
            if v is None: d[k]=-999
            elif isinstance(v,(np.floating,np.integer)): d[k]=float(v)
            else: d[k]=v
        clean.append(d)
    json.dump(clean,f,indent=2)
print(f'\nSaved {len(results)} results')
