#!/usr/bin/env python3
"""GMT 汇总地图：台站 + P/T 波偏振方向 + 地形底图 (浅色)"""
import json,os,numpy as np
import pygmt

OUTDIR='/home/philiay/shared/oi/all_polar_analysis'
os.makedirs(OUTDIR,exist_ok=True)

# Load results
with open(f'{OUTDIR}/polarization_results.json')as f:
    results=json.load(f)

EVLO,EVLA=-66.87,18.99
region=[-86,-64,14,48]  # W,E,S,N — 宽到包含所有台站

# ── 准备台站数据 ──
sta_lon=[];sta_lat=[];sta_label=[]
p_dx=[];p_dy=[];t_dx=[];t_dy=[]
p_ok=0;t_ok=0

for r in results:
    if r['p_az']==-999 and r['t_az']==-999:continue
    sta_lon.append(r['lon']);sta_lat.append(r['lat']);sta_label.append(r['sta'])
    # P arrow
    if r['p_az']!=-999:
        rad=np.radians(r['p_az']);arr_len=1.5
        p_dx.append(arr_len*np.sin(rad));p_dy.append(arr_len*np.cos(rad))
        p_ok+=1
    else:
        p_dx.append(0);p_dy.append(0)
    # T arrow
    if r['t_az']!=-999:
        rad=np.radians(r['t_az']);arr_len=1.5
        t_dx.append(arr_len*np.sin(rad));t_dy.append(arr_len*np.cos(rad))
        t_ok+=1
    else:
        t_dx.append(0);t_dy.append(0)

print(f'Stations: {len(sta_lon)}, P arrows: {p_ok}, T arrows: {t_ok}')

# ── GMT 绘图 ──
fig=pygmt.Figure()
pygmt.config(MAP_FRAME_TYPE='plain',MAP_TICK_LENGTH_PRIMARY='4p',FONT_ANNOT_PRIMARY='9p')

# 地形 (浅色 — gray 色标, 30s 分辨率)
fig.grdimage(grid='@earth_relief_30s',region=region,projection='M14c',
             cmap='gray',nan_transparent=True,transparency=60)

# 海岸线、国界
fig.coast(region=region,projection='M14c',shorelines='0.5p,gray50',
          borders=['1/0.3p,gray60'],water='white',frame=['WSen','xa4f2','ya4f2'])

# 震源
fig.plot(x=EVLO,y=EVLA,style='a0.35c',fill='red',pen='0.5p,darkred')
fig.text(x=EVLO+0.5,y=EVLA-0.5,text='Mw 6.3',font='10p,Helvetica-Bold,red')

# 台站三角
fig.plot(x=sta_lon,y=sta_lat,style='t0.25c',fill='black',pen='0.3p')

# 台站标签
fig.text(x=[l+0.25 for l in sta_lon],y=[a+0.15 for a in sta_lat],
         text=sta_label,font='6p,Helvetica')

# P 波偏振箭头 (绿色)
p_start_lon=[];p_start_lat=[];p_end_lon=[];p_end_lat=[]
for i in range(len(sta_lon)):
    if p_dx[i]!=0 or p_dy[i]!=0:
        p_start_lon.append(sta_lon[i]);p_start_lat.append(sta_lat[i])
        p_end_lon.append(sta_lon[i]+p_dx[i]);p_end_lat.append(sta_lat[i]+p_dy[i])

if p_start_lon:
    fig.plot(x=p_start_lon,y=p_start_lat,style='c0.05c',fill='#2ca02c',pen='0.3p')
    for i in range(len(p_start_lon)):
        fig.plot(x=[p_start_lon[i],p_end_lon[i]],y=[p_start_lat[i],p_end_lat[i]],
                 pen='1.2p,#2ca02c')

# T 波偏振箭头 (蓝色)
t_start_lon=[];t_start_lat=[];t_end_lon=[];t_end_lat=[]
for i in range(len(sta_lon)):
    if t_dx[i]!=0 or t_dy[i]!=0:
        t_start_lon.append(sta_lon[i]);t_start_lat.append(sta_lat[i])
        t_end_lon.append(sta_lon[i]+t_dx[i]);t_end_lat.append(sta_lat[i]+t_dy[i])

if t_start_lon:
    for i in range(len(t_start_lon)):
        fig.plot(x=[t_start_lon[i],t_end_lon[i]],y=[t_start_lat[i],t_end_lat[i]],
                 pen='1.2p,#1f77b4,-')

# 图例
fig.plot(x=-84.5,y=14.5,style='r2.8c/1.5c',fill='white',pen='0.3p,gray50')
fig.plot(x=-84.3,y=14.8,style='c0.12c',fill='#2ca02c',pen='0.2p')
fig.text(x=-84.0,y=14.8,text='P-wave polarization',font='7p,Helvetica')
fig.plot(x=-84.3,y=14.5,style='c0.12c',fill='#1f77b4',pen='0.2p')
fig.text(x=-84.0,y=14.5,text='T-wave polarization',font='7p,Helvetica')
fig.text(x=-84.3,y=14.2,text='11 TA stations  |  Flinn (1965)  |  2014-01-13 Mw6.3',font='7p,Helvetica-Oblique,gray40')

out=f'{OUTDIR}/polarization_map.pdf'
fig.savefig(out)
print(f'✅ {out}')
out_png=f'{OUTDIR}/polarization_map.png'
fig.savefig(out_png)
print(f'✅ {out_png}')
