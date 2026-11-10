"""画 Δ（替换回合按下步 − 原样按下步）对 Δ*（按 token 时钟预测的偏移）的散点，两模型并排。用法：uv run --no-project --with matplotlib python plot_delta.py <records 目录>"""
import json,sys,pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=pathlib.Path(sys.argv[1])
fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
col={"fast":"#2a6fdb","slow":"#d9480f","same":"#7a7a7a"}
for ax,m,title in zip(axes,("motion","wan"),("MotionMem (concat 1x1024)","Frozen Wan pooling")):
    d=json.loads((root/m/"readout.json").read_text())
    rows=[r for r in d["detail"]["delta_rows"] if r["delta"] is not None]
    beta=[l for l in d["lines"] if l.startswith("SPEED_FOLLOW ")][0].split()[2]
    for k in ("same","fast","slow"):
        rr=[r for r in rows if r["kind"]==k]
        ax.scatter([r["dstar"] for r in rr],[r["delta"] for r in rr],s=18,alpha=.65,c=col[k],label=f"{k} (n={len(rr)})")
    ax.plot([-300,300],[-300,300],ls="--",c="k",lw=1,label="y = x (fully follows token)")
    ax.axhline(0,c="#999",lw=.8)
    ax.set_title(f"{title}   {beta}")
    ax.set_xlabel("predicted shift  Δ* = T(M′,N) − T(M,N)  [steps]")
    ax.grid(alpha=.3)
axes[0].set_ylabel("observed shift  Δ = press_R − press_N  [steps]")
axes[0].legend(loc="upper left",fontsize=8)
fig.suptitle("StopCube: replace motion tokens with a different cube speed M′")
fig.tight_layout()
fig.savefig(root/"delta_vs_pred.png",dpi=130)
print("saved",root/"delta_vs_pred.png")
