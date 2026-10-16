#!/usr/bin/env python3
"""
批量走时分析 — 全部 343 台站
快速模式: 只读 SAC 头段 → TauP P/S + tt_t() T-P/T-S → GMT 汇总地图
适配当前环境
"""
import sys,os,json,numpy as np
sys.path.insert(0,'/home/sy/codex')
sys.path.insert(0,'/home/sy/codex/t_phase_analysis')
import pygmt
from obspy import read
from obspy.geodetics.base import locations2degrees,degrees2kilometers,gps2dist_azimuth
from obspy.taup import TauPyModel
from t_phase_analysis.traveltime_profile import get_track,tt_t

PROC='/mnt/f/sharedate/case/2014-01-13-mw63-process/sacprocess'
OUT='/mnt/f/sharedate/case/2014-01-13-mw63-process/process_flow/1_station_distribution'
EVLO,EVLA=-66.87,18.99
os.makedirs(OUT,exist_ok=True)
taup=TauPyModel(model='iasp91')

# ── 读取所有 SAC 头段 ──
sac_files=sorted([f for f in os.listdir(PROC) if f.endswith('_proc.sac') and 'BHZ' in f])
print(f'总计 {len(sac_files)} 个 BHZ 文件')

stations=[]
for f in sac_files:
    try:
        tr=read(os.path.join(PROC,f),headonly=True)[0];sac=tr.stats.sac
        stations.append({'sta':sac.kstnm.strip(),'net':sac.knetwk.strip(),
                         'lon':sac.stlo,'lat':sac.stla,'file':f})
    except:pass
print(f'读取 {len(stations)} 个台站')

# ── 去重 (同台站多 location) ──
seen=set();unique=[]
for s in stations:
    key=(s['net'],s['sta'])
    if key not in seen:seen.add(key);unique.append(s)
stations=unique
print(f'去重后 {len(stations)} 台')

# ── 批量 TauP P/S ──
print('计算 TauP P/S ...')
results=[]
for s in stations:
    d_deg=locations2degrees(EVLA,EVLO,s['lat'],s['lon'])
    d_km=degrees2kilometers(d_deg)
    _,baz,_=gps2dist_azimuth(s['lat'],s['lon'],EVLA,EVLO)
    try:
        arrs=taup.get_travel_times(source_depth_in_km=10,distance_in_degree=d_deg,phase_list=['P','S'])
        tp=ts=None
        for a in arrs:
            if a.name=='P'and tp is None:tp=a.time
            elif a.name=='S'and ts is None:ts=a.time
    except:tp=ts=None
    results.append({**s,'dist_deg':d_deg,'dist_km':d_km,'baz':baz,'tt_p':tp,'tt_s':ts})

# ── tt_t T-P/T-S (全量计算，不等间隔抽样) ──
results.sort(key=lambda r:r['dist_km'])
print(f'计算 T-P/T-S (全部 {len(results)} 台)...')
for i,r in enumerate(results):
    try:
        track=get_track(EVLO,EVLA,r['lon'],r['lat'],0.02,ref_el=-1000,
                        plot_map=False,plot_elevation=False)
        tr_tt=tt_t(track,ref_el=-1000,source_depth_km=10,ignore_small_islands=True)
        if tr_tt:
            r['tt_tp']=tr_tt['tt_t']
            r['tt_ts']=tr_tt['tt_ocean']+tr_tt['tt_land_s']
            r['dist_ocean']=degrees2kilometers(tr_tt['dist_ocean'])
            r['dist_land']=degrees2kilometers(tr_tt['dist_land'])
    except:r['tt_tp']=None;r['tt_ts']=None
    if (i+1)%50==0:print(f'  progress: {i+1}/{len(results)}')

# ── 保存 JSON ──
clean=[]
for r in results:
    d={}
    for k,v in r.items():
        if v is None:d[k]=-999
        elif isinstance(v,(np.floating,np.integer)):d[k]=float(v)
        else:d[k]=v
    clean.append(d)
with open(f'{OUT}/all_travel_times.json','w')as f:json.dump(clean,f,indent=1)
print(f'Saved {len(clean)} results to all_travel_times.json')

# ── GMT 地图: 台站按 T-P 到时着色 ──
valid_tp=[r for r in results if r.get('tt_tp')and r['tt_tp']>0]
print(f'T-P valid: {len(valid_tp)}')

region=[-95,-65,17,50]
fig=pygmt.Figure()
pygmt.config(MAP_FRAME_TYPE='plain')

fig.grdimage(grid='@earth_relief_30s',region=region,projection='M18c',
             cmap='gray',nan_transparent=True,transparency=60)
fig.coast(region=region,projection='M18c',shorelines='0.5p,gray50',
          borders=['1/0.3p,gray60'],water='white',frame=['WSen','xa5f1','ya5f1'])

# 震源
fig.plot(x=EVLO,y=EVLA,style='a0.35c',fill='red',pen='0.5p,darkred')
fig.text(x=EVLO+0.5,y=EVLA-0.5,text='Mw 6.3',font='10p,Helvetica-Bold,red')

# T-P 着色的台站
if valid_tp:
    tp_vals=[r['tt_tp']/60 for r in valid_tp]
    lons=[r['lon']for r in valid_tp];lats=[r['lat']for r in valid_tp]
    pygmt.makecpt(cmap='hot',series=[min(tp_vals),max(tp_vals)],reverse=True)
    import tempfile
    tf=tempfile.NamedTemporaryFile(mode='w',suffix='.txt',delete=False)
    for lo,la,tv in zip(lons,lats,tp_vals):
        tf.write(f'{lo} {la} {tv}\n')
    tf.close()
    fig.plot(data=tf.name,style='t0.25c',cmap=True,pen='0.2p,black')
    os.unlink(tf.name)
    fig.colorbar(cmap=True,position='jBL+o1c/0.6c+w6c/0.3c',
                 frame=['af','y+l"T-P travel time (min)"'])

# 其他台站 (灰色小点)
other=[r for r in results if not r.get('tt_tp')or r['tt_tp']<=0]
if other:
    fig.plot(x=[r['lon']for r in other],y=[r['lat']for r in other],
             style='c0.08c',fill='gray70',pen='0.1p')

fig.text(x=region[0]+1.5,y=region[2]+1,text=f'{len(results)} stations  |  TauP iasp91 + tt_t()  |  colored by T-P',
         font='7p,Helvetica-Oblique,gray40')

out_png=f'{OUT}/travel_time_map.png'
fig.savefig(out_png,dpi=300)
print(f'Saved {out_png}')
print('Done.')
