#!/usr/bin/env python3
"""GMT 偏振方向图 — 简洁版, 无插图"""
import json,os,numpy as np
import pygmt

OUT='/home/philiay/shared/oi/all_polar_analysis'
with open(f'{OUT}/all_stations_polar.json')as f:results=json.load(f)

EVLO,EVLA=-66.87,18.99
region=[-97,-62,14,52]
AZ_THRESH=35

good=[]
for r in results:
    p_az=r.get('p_az',-999)
    if p_az==-999:continue
    bf=r['baz']%180
    diff=abs(p_az-bf)
    if diff>90:diff=180-diff
    if diff<AZ_THRESH:good.append(r)
print(f'Good: {len(good)}')

from obspy.geodetics.base import degrees2kilometers
circles_deg=[5,10,15,20,25,30]

fig=pygmt.Figure()
pygmt.config(MAP_FRAME_TYPE='plain',FONT_ANNOT_PRIMARY='9p')
fig.grdimage(grid='@earth_relief_30s',region=region,projection='M16c',
             cmap='gray',nan_transparent=True,transparency=65)
fig.coast(region=region,projection='M16c',shorelines='0.5p,gray50',
          borders=['1/0.3p,gray60'],water='white',frame=['WSen','xa5f5','ya5f5'])

for d in circles_deg:
    fig.plot(x=EVLO,y=EVLA,style=f'E{degrees2kilometers(d)}k',pen='0.3p,gray60,-')
    fig.text(x=EVLO+degrees2kilometers(d)*0.68/111,y=EVLA,text=f'{d}°',font='6p,gray50')

fig.plot(x=EVLO,y=EVLA,style='a0.4c',fill='red',pen='0.8p,darkred')
fig.text(x=EVLO+1.8,y=EVLA-0.5,text='Mw 6.3',font='11p,Helvetica-Bold,red')
fig.text(x=EVLO+1.8,y=EVLA-1.2,text='18.99°N, 66.87°W',font='7p,Helvetica,darkred')

for phase,key,color,alen in [('T','tp_az','#1f77b4',0.6),
                               ('S','s_az','#ff7f0e',0.8),
                               ('P','p_az','#2ca02c',0.8)]:
    for r in good:
        az=r.get(key,-999)
        if az==-999:continue
        rad=np.radians(az)
        sx,sy=r['lon'],r['lat']
        ex=sx+alen*np.sin(rad);ey=sy+alen*np.cos(rad)
        fig.plot(x=[sx,ex],y=[sy,ey],pen=f'0.4p,{color}')

for r in good:
    label=r['sta'][:4]
    fig.plot(x=r['lon'],y=r['lat'],style='c0.08c',fill='black',pen='0.03p')
    fig.text(x=r['lon']+0.15,y=r['lat']+0.08,text=label,font='4p,Helvetica-Bold')

fig.plot(x=-95,y=14.3,style='r3c/2.4c',fill='white',pen='0.3p,gray50')
for i,(p,c)in enumerate([('P','#2ca02c'),('S','#ff7f0e'),('T','#1f77b4')]):
    y=15.9-i*0.55;fig.plot(x=-94.7,y=y,style='c0.1c',fill=c,pen='0.1p')
    fig.text(x=-94.0,y=y,text=f'{p}-wave',font='8p,Helvetica')
fig.text(x=-94.7,y=14.0,text=f'{len(good)} stations | P-BAZ < {AZ_THRESH}°',font='7p,Helvetica-Oblique,gray40')

out_png=f'{OUT}/polar_clean_map.png';fig.savefig(out_png);print(f'✅ {out_png}')
out_pdf=f'{OUT}/polar_clean_map.pdf';fig.savefig(out_pdf);print(f'✅ {out_pdf}')
