import json,sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
for f in font_manager.findSystemFonts():
    if "NotoSerifCJK" in f.replace("-","") or "NotoSansCJK" in f.replace("-",""):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"]=["Noto Serif CJK JP","Noto Sans CJK JP","DejaVu Sans"]
R=sys.argv[1]; out=sys.argv[2]
d={m:json.load(open(f"{R}/{m}/readout.json")) for m in ("motion","wan")}
name={"motion":"MotionMem","wan":"Frozen Wan pooling"}
for task in ("PatternLock","RouteStick"):
    for grp in ("Bsame","Bcross"):
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
        for ax,m in zip(axes,("motion","wan")):
            mat=d[m]["summaries"][task]["groups"][grp]["matrix"]
            J=sorted(int(j) for j in mat); I=list(range(1,8))
            A=np.full((len(J),len(I)),np.nan)
            for a,j in enumerate(J):
                for b,i in enumerate(I):
                    k,n=mat[str(j)].get(str(i),[0,0])
                    if n>0: A[a,b]=k/n
                    if n>0: ax.text(b,a,f"{k}/{n}",ha="center",va="center",fontsize=8,color="white" if (k/n)<0.55 else "black")
            im=ax.imshow(A,cmap="viridis",vmin=0,vmax=1,aspect="auto",interpolation="nearest")
            for a,j in enumerate(J):
                if j<=7: ax.add_patch(plt.Rectangle((j-1-0.5,a-0.5),1,1,fill=False,ec="red",lw=1.5))
            ax.set_xticks(range(len(I)),[str(i) for i in I]); ax.set_yticks(range(len(J)),[str(j) for j in J])
            ax.set_xlabel("被读的子目标 i"); ax.set_ylabel("干预位置 j（换第 j 段 token）")
            ax.set_title(f"{name[m]}")
        fig.colorbar(im,ax=axes,shrink=0.8,label="比例")
        fig.suptitle(f"{task} · {grp}：对角（红框）= 第 j 步跟随供体的原始比例（未减基线）；i<j = 与原样同结果；i>j = 第 j 步走回自己后后续步正确",fontsize=10)
        fig.savefig(f"{out}/matrix_{task}_{grp}.png",dpi=130); plt.close(fig)
print("ok")
