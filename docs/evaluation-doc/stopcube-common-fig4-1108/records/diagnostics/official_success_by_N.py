"""探索性：正式 3 seed 评测（official-stage2-1026，只读）的 StopCube 成功率按 N、M 拆分，o16cc（MotionMem）对 o16ctx（context split 4×256，motion 预算 384，即 MotionMem 的 split 变体；不是 FrameSamp）。
M、N 取自 stopcube-oracle-1104 的 oracle_table.json（按 episode_key，核对 env_seed 一致）。逐局按（seed, episode_key）配对，精确 McNemar。
用法：python official_success_by_N.py <official-stage2-1026 根> <oracle_table.json>
"""
import glob, json, math, sys
from collections import defaultdict

root, table = sys.argv[1], json.load(open(sys.argv[2]))["episodes"]

def load(run):
    out = {}
    files = glob.glob(f"{root}/{run}/run800/accepted/*.json") + glob.glob(f"{root}/{run}/run800-r2/accepted/*.json")
    for f in files:
        d = json.load(open(f))
        if d.get("task_name") != "StopCube":
            continue
        t = table[d["episode_key"]]
        assert t["env_seed"] == d["env_seed"], (d["episode_key"], t["env_seed"], d["env_seed"])
        if d["episode_key"] in out:
            assert out[d["episode_key"]][0] == int(d["outcome"] == "success"), ("同局两份记录结果不同", run, d["episode_key"])
        out[d["episode_key"]] = (int(d["outcome"] == "success"), t["move_interval"], t["target_pass"])
    return out

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

rows = []
for seed in (42, 7, 0):
    a, b = load(f"o16cc-60k-seed{seed}"), load(f"o16ctx-60k-seed{seed}")
    assert set(a) == set(b) and len(a) == 50, (seed, len(a), len(b))
    rows += [(seed, k, a[k][0], b[k][0], a[k][1], a[k][2]) for k in a]
print("episodes", len(rows), "MotionMem", sum(r[2] for r in rows), "FrameSamp(o16ctx)", sum(r[3] for r in rows))
for name, f in [("all", lambda r: True)] + [(f"N={n}", (lambda n: lambda r: r[5] == n)(n)) for n in (2, 3, 4, 5)] + \
        [("N<=3", lambda r: r[5] <= 3), ("N>=4", lambda r: r[5] >= 4)] + [(f"M={m}", (lambda m: lambda r: r[4] == m)(m)) for m in (60, 80, 120)]:
    sub = [r for r in rows if f(r)]
    mo, fs = sum(r[2] for r in sub), sum(r[3] for r in sub)
    bb = sum(1 for r in sub if r[2] and not r[3]); cc = sum(1 for r in sub if r[3] and not r[2])
    print(f"{name}: n={len(sub)} MotionMem {mo}/{len(sub)} ({mo/len(sub):.2f}) FrameSamp {fs}/{len(sub)} ({fs/len(sub):.2f}) discordant {bb}/{cc} p={mcnemar(bb, cc):.3g}")
