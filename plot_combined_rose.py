#!/usr/bin/env python3
"""Top 10 stations — 合并偏振玫瑰图 (mod 180°)"""
import sys,os,numpy as np
sys.path.insert(0,'/mnt/f/sharedate/T_Phase_Analysis-main')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import read
from obspy.taup import TauPyModel
from obspy.geodetics.base import locations2degrees,gps2dist_azimuth

OUT='/home/philiay/shared/oi/all_polar_analysis'
PROC='/mnt/f/sharedate/case/sacprocess'
EVLO,EVLA=-66.87,18.99
taup=TauPyModel(model='iasp91')

TOP10=['R53A','R61A','L57A','K56A','L62A','U50A','X57A','W57A','J55A','U51A']

all_p_azi=[];all_s_azi=[];all_baz=[]

for sta in TOP10:
    print(f'{sta}...',end=' ',flush=True)
    try:
        tr_z=read(f'{PROC}/TA.{sta}..BHZ.M.2014.013.035104_proc.sac')[0]
        tr_n=read(f'{PROC}/TA.{sta}..BHN.M.2014.013.035104_proc.sac')[0]
        tr_e=read(f'{PROC}/TA.{sta}..BHE.M.2014.013.035104_proc.sac')[0]
        t0=max(tr_z.stats.starttime,tr_n.stats.starttime,tr_e.stats.starttime)
        t1=min(tr_z.stats.endtime,tr_n.stats.endtime,tr_e.stats.endtime)
        for tr in[tr_z,tr_n,tr_e]:tr.trim(t0,t1)
        fs=tr_z.stats.sampling_rate;sac=tr_z.stats.sac
        stla,stlo=sac.stla,sac.stlo
        d_deg=locations2degrees(EVLA,EVLO,stla,stlo)
        _,baz,_=gps2dist_azimuth(stla,stlo,EVLA,EVLO)
        arrs=taup.get_travel_times(source_depth_in_km=10,distance_in_degree=d_deg,phase_list=['P','S'])
        tp=ts=None
        for a in arrs:
            if a.name=='P'and tp is None:tp=a.time
            elif a.name=='S'and ts is None:ts=a.time

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

        def get_azi(tc,hw):
            if tc is None:return None
            m=(tmid>=tc-hw)&(tmid<=tc+hw)&(rect>0.4)
            return azi180[m]if m.sum()>0 else None

        p_azi=get_azi(tp,5);s_azi=get_azi(ts,3)
        if p_azi is not None:all_p_azi.extend(p_azi)
        if s_azi is not None:all_s_azi.extend(s_azi)
        all_baz.append(baz%180)
    except Exception as e:
        print(f'❌ {e}')

print(f'\nP: {len(all_p_azi)} pts  S: {len(all_s_azi)} pts')

# ── 合并玫瑰图 ──
fig,ax=plt.subplots(figsize=(9,8),subplot_kw={'projection':'polar'})

bins=np.arange(0,181,10)

if all_p_azi:
    hp,_=np.histogram(all_p_azi,bins=bins)
    ax.bar(np.radians(bins[:-1]+5),hp,width=np.radians(10),
           color='#2ca02c',alpha=0.6,edgecolor='white',lw=0.4,label=f'P (n={len(all_p_azi)})')

if all_s_azi:
    hs,_=np.histogram(all_s_azi,bins=bins)
    ax.bar(np.radians(bins[:-1]+5),hs,width=np.radians(10),
           color='#ff7f0e',alpha=0.5,edgecolor='white',lw=0.4,label=f'S (n={len(all_s_azi)})')

# 理论BAZ中位数方向
median_baz=np.median(all_baz)if all_baz else 0
ax.axvline(np.radians(median_baz),color='red',lw=1.5,ls='--',alpha=0.8,label=f'BAZ={median_baz:.0f}°')
ax.axvline(np.radians((median_baz+180)%180),color='red',lw=1.5,ls='--',alpha=0.8)

ax.set_theta_zero_location('N');ax.set_theta_direction(-1)
ax.set_title(f'Top 10 stations — Polarization Rose (mod 180°)\nP={len(all_p_azi)} pts  S={len(all_s_azi)} pts',
             fontsize=12,fontweight='bold',pad=20)
ax.legend(fontsize=9,loc='upper right',bbox_to_anchor=(1.15,1.05))

plt.tight_layout()
out=f'{OUT}/combined_rose_180.png';fig.savefig(out,dpi=200,bbox_inches='tight');plt.close(fig)
print(f'✅ {out}')
