"""692 局评估身份闸（1117-e2e-enc-train-plan.html 第二部分一节 eval692_identity 行、三节 EVAL692_PLAN_IDENTITY / EVAL692_RESULT_IDENTITY）。

钉死的期望清单是同目录 ``eval692_identity_v1.json``：692 个 (task, difficulty, episode) 与 14 组逐组计数。期望**不得由本次动态
结果生成**：它由 ``build`` 子命令从 c0826be 上 Wan-9 本机 692 评估逐局记录生成，并与 MotionMEM 两次评估逐条交叉核对
（``primary700-motion50k-gl`` 去掉 BinFill 的 550 + ``binfilldemo-motion50k-gl`` 的 142），任一不一致即非零退出、停下报用户。

子命令（只用标准库）：
  build  --out <json>                          生成并交叉核对（只在制作清单时用；已存在即拒）
  verify-sources                               由三份留档记录现场重建，与提交的清单逐条比对（CROSSCHECK=PASS|FAIL）
  plan   --plan-dir <d> --shards <n>           分片计划 shard0..shard{n-1}.json 两两不交、并集与清单逐条相等、14 组计数相等
                                               → EVAL692_PLAN_IDENTITY=PASS manifest_sha=<s> shards=<n> union=692 per_group=14/14
  result --progress <merged/progress.json>     合并后逐局结果的身份集合与清单逐条相等
         （或 --eval-root <评估根>，读 <根>/merged/progress.json）
                                               → EVAL692_RESULT_IDENTITY=PASS manifest_sha=<s> episodes=692
少一条、多一条、等数量换一条都 FAIL（判定行附 missing / extra 数与前几条样例）。manifest_sha 为清单文件字节 sha256 前 16 位。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "eval692_identity_v1.json"
SRC_WAN9 = "docs/training-doc/local692-wanraw9s32-50k/records/merged-progress.json"
SRC_PRIMARY = "docs/training-doc/primary700-motion50k-gl/records/merged/progress.json"
SRC_BINFILL = "docs/training-doc/binfilldemo-motion50k-gl/records/merged/progress.json"


def keys_of_progress(progress: dict) -> set[str]:
    """progress.json（{ "Task/diff": {ep: 结果} }）→ {"Task/diff/ep"}。"""
    return {f"{g}/{int(e)}" for g, entries in progress.items() for e in entries}


def group_counts(keys) -> dict[str, int]:
    return dict(sorted(collections.Counter(k.rsplit("/", 1)[0] for k in keys).items()))


def load_manifest(path: pathlib.Path = MANIFEST) -> tuple[dict, str]:
    raw = path.read_bytes()
    m = json.loads(raw)
    keys = m["episodes"]
    if len(keys) != m["total"] or len(set(keys)) != len(keys) or group_counts(keys) != m["groups"]:
        raise SystemExit(f"清单自身不一致：{path}")
    return m, hashlib.sha256(raw).hexdigest()[:16]


def rebuild(repo: pathlib.Path = REPO) -> tuple[set[str], list[str]]:
    """由三份留档重建期望集合；返回 (集合, 交叉核对问题列表)。"""
    wan9 = keys_of_progress(json.loads((repo / SRC_WAN9).read_text()))
    prim = keys_of_progress(json.loads((repo / SRC_PRIMARY).read_text()))
    binf = keys_of_progress(json.loads((repo / SRC_BINFILL).read_text()))
    motionmem = {k for k in prim if not k.startswith("BinFill/")} | binf
    issues = []
    if len({k for k in prim if not k.startswith("BinFill/")}) != 550:
        issues.append(f"primary700 去 BinFill 后 {len({k for k in prim if not k.startswith('BinFill/')})} != 550")
    if len(binf) != 142 or any(not k.startswith("BinFill/") for k in binf):
        issues.append(f"binfilldemo 应为 142 条 BinFill，实得 {len(binf)}")
    if wan9 != motionmem:
        issues.append(f"Wan-9 与 MotionMEM 两次评估不一致：仅 Wan-9 {sorted(wan9 - motionmem)[:5]} 仅 MotionMEM {sorted(motionmem - wan9)[:5]}")
    if len(wan9) != 692 or len(group_counts(wan9)) != 14:
        issues.append(f"Wan-9 记录 {len(wan9)} 条 / {len(group_counts(wan9))} 组")
    return wan9, issues


def compare(got: set[str], want: list[str]) -> tuple[bool, str]:
    w = set(want)
    missing, extra = sorted(w - got), sorted(got - w)
    ok = not missing and not extra
    detail = "" if ok else f" missing={len(missing)} extra={len(extra)} missing_eg={missing[:3]} extra_eg={extra[:3]}"
    return ok, detail


def cmd_build(args) -> int:
    out = pathlib.Path(args.out)
    if out.exists():
        print(f"拒绝覆盖已存在的清单 {out}", file=sys.stderr)
        return 2
    keys, issues = rebuild()
    if issues:
        print(f"EVAL692_MANIFEST_BUILD=FAIL issues={issues}（停下报用户）", flush=True)
        return 1
    ordered = sorted(keys, key=lambda k: (k.rsplit("/", 1)[0], int(k.rsplit("/", 1)[1])))
    m = {"schema": "eval692_identity_v1", "total": len(ordered), "groups": group_counts(ordered),
         "sources": {"wan9_local692": f"c0826be:{SRC_WAN9}",
                     "motionmem_primary550": f"{SRC_PRIMARY}（去掉 BinFill 150 条）",
                     "motionmem_binfilldemo142": SRC_BINFILL},
         "crosscheck": "三份记录逐条相等（Wan-9 692 == MotionMEM 550+142）",
         "episodes": ordered}
    out.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    _, sha = load_manifest(out)
    print(f"EVAL692_MANIFEST_BUILD=PASS episodes={len(ordered)} groups={len(m['groups'])} manifest_sha={sha} out={out}", flush=True)
    return 0


def cmd_verify_sources(args) -> int:
    m, sha = load_manifest(pathlib.Path(args.manifest))
    keys, issues = rebuild()
    ok, detail = compare(keys, m["episodes"])
    ok = ok and not issues
    print(f"CROSSCHECK={'PASS' if ok else 'FAIL'} manifest_sha={sha} episodes={len(keys)}" + detail
          + (f" issues={issues}" if issues else ""), flush=True)
    return 0 if ok else 1


def cmd_plan(args) -> int:
    m, sha = load_manifest(pathlib.Path(args.manifest))
    plan_dir = pathlib.Path(args.plan_dir)
    union: set[str] = set()
    dup = 0
    reasons = []
    for k in range(args.shards):
        p = plan_dir / f"shard{k}.json"
        if not p.is_file():
            reasons.append(f"缺 {p.name}")
            continue
        groups = json.loads(p.read_text())["groups"]
        ks = {f"{g}/{int(e)}" for g, eps in groups.items() for e in eps}
        dup += len(ks & union)
        union |= ks
    stray = sorted(p.name for p in plan_dir.glob("shard*.json")
                   if p.stem[5:].isdigit() and int(p.stem[5:]) >= args.shards)
    if stray:
        reasons.append(f"多出分片文件 {stray[:3]}")
    if dup:
        reasons.append(f"分片相交 {dup} 条")
    ok, detail = compare(union, m["episodes"])
    pg_ok = sum(1 for g, n in group_counts(union).items() if m["groups"].get(g) == n)
    ok = ok and not reasons and pg_ok == len(m["groups"])
    print(f"EVAL692_PLAN_IDENTITY={'PASS' if ok else 'FAIL'} manifest_sha={sha} shards={args.shards} union={len(union)} "
          f"per_group={pg_ok}/{len(m['groups'])}" + detail + (f" reasons={reasons}" if reasons else ""), flush=True)
    return 0 if ok else 1


def cmd_result(args) -> int:
    m, sha = load_manifest(pathlib.Path(args.manifest))
    path = pathlib.Path(args.progress) if args.progress else pathlib.Path(args.eval_root) / "merged" / "progress.json"
    if not path.is_file():
        print(f"EVAL692_RESULT_IDENTITY=FAIL manifest_sha={sha} episodes=0 reason=缺 {path}", flush=True)
        return 1
    got = keys_of_progress(json.loads(path.read_text()))
    ok, detail = compare(got, m["episodes"])
    print(f"EVAL692_RESULT_IDENTITY={'PASS' if ok else 'FAIL'} manifest_sha={sha} episodes={len(got)}" + detail, flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True)
    sub.add_parser("verify-sources")
    p = sub.add_parser("plan"); p.add_argument("--plan-dir", required=True); p.add_argument("--shards", type=int, required=True)
    r = sub.add_parser("result"); g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--progress"); g.add_argument("--eval-root")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "verify-sources": cmd_verify_sources, "plan": cmd_plan, "result": cmd_result}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
