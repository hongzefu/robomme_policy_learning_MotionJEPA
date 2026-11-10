"""主会话临时敏感性分析：把 W 起点收紧 2 步，重算新速度成功与基线。"""
import json,sys
d=json.load(open(sys.argv[1]))["detail"]
rows=d["success_rows"]
def inw(p,w,shift=0):
    return p is not None and w is not None and w[0]+shift<=p<=w[1]
for g in ("fast","slow"):
    rr=[r for r in rows if r["kind"]==g and not r.get("cat")=="overlap"]
    for sh in (0,2):
        k=sum(inw(r["p_R"],r["w_new"],sh) for r in rr); b=sum(inw(r["p_N"],r["w_new"],sh) for r in rr)
        t=sum(inw(r["p_R"],r["w_true"],sh) for r in rr)
        print(f"EDGE_SENS model={sys.argv[2]} group={g} shift={sh} new_speed={k}/{len(rr)} base={b}/{len(rr)} true_speed={t}/{len(rr)}")
