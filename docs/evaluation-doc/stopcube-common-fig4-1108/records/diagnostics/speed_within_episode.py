"""探索性：只用同时在早档 t∈[56,120) 与晚档 t≥200 都有点的局（58 局），同局内比较读速度 M 准确率的晚减早（按局重采样）。用法：在仓库根以 PYTHONPATH=仓库根 运行。"""
import json, numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.oracle_press_prob import bootstrap
src='v1-store/evaluation/stopcube-common-fig4-1108/explore/new'
s=hp.load_samples('v1-store/evaluation/stopcube-hover-probe-1105/hover_points.json'); P=np.load(src+'/predictions.npz'); A=json.load(open(src+'/angles.json'))
def comp(m,g):
    ch=A['selected'][f'{m}|{g}']['M']['layer_by_fold']; r=np.full(len(s['t']),np.nan)
    for f,l in enumerate(ch): r[s['outer']==f]=P[f'{m}|{g}|{l}|M'][s['outer']==f]
    return r
for m in ('mot','ctx'):
  for g in ('text','history'):
    p=comp(m,g); e=hp.readout_episodes(s,'M',np.where((s['t']>=56)&(s['t']<120),p,np.nan),s['labels']['M']); l=hp.readout_episodes(s,'M',np.where(s['t']>=200,p,np.nan),s['labels']['M'])
    k=np.isfinite(e)&np.isfinite(l); d=bootstrap(l[k]-e[k],2000,42)
    print(f"{m} {g}: same eps={k.sum()} early={e[k].mean():.2f} late={l[k].mean():.2f} late-early={d['mean']:+.3f}[{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]")
