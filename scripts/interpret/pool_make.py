"""把一个或多个阶段的 run_local spec 按预计成本降序写成共享任务池（契约 C4，计划 1116-steer-speedup-fix-plan S-D）。

池目录布局（<O> 为 --out）：

    <O>/pool/<池名>/todo/<6位成本序号>-<spec 名>.<gl|aspen>.json   待领取（序号 000000 最贵，先领）
    <O>/pool/<池名>/claim/<6位成本序号>-<spec 名>/                 领取标记（pool_runner.sh 用 mkdir 原子领取）
    <O>/pool/<池名>/done/  完成：该键的 json 与 <键>.result（rc、站点、日志路径）
    <O>/pool/<池名>/fail/  失败：同上
    <O>/pool/<池名>/logs/  每 spec 一份运行日志
    <O>/pool/<池名>/manifest.json  每项的序号、阶段、成本估计与源 spec 路径

用法：

    pool_make.py make --out <O> --stage-spec <阶段> <GL glob> <aspen glob> [--stage-spec …] [--pool-name <名>]
                      [--cost-json <成本表>]
    pool_make.py status --out <O> --pool-name <名>

--stage-spec 可重复，多个阶段合并成一个池（池名缺省为各阶段名以 + 连接）；某站点没有 spec 时 glob 写 "-"。
GL 与 aspen 两套 spec 按 spec 名配对（文件名去掉 .json，再去掉可选的 .gl / .aspen 后缀）；只有一套的项只由该站点领取。
某站点的 glob 扫到带另一站点后缀（.gl.json / .aspen.json）的文件时自动剔除。

成本模型（每局秒数 × 阶段倍率 × j 倍率，再对 manifest 里的每一局求和）：
    seconds = Σ_episode difficulty_seconds[difficulty] × stage_mult[stage] × (1 + j_coef × j)
j 取 spec 的 "j" 键，没有则从 spec 名里的 "-j<数字>-" 解析，都没有记 0；difficulty 从 spec 的 manifest 读，
manifest 读不到时按 1 局 default_difficulty 计。--cost-json 可覆盖任一参数，也可用 spec_seconds 直接给某 spec 的秒数：
    {"difficulty_seconds": {"easy": 60, …}, "stage_mult": {"noise": 0.8}, "j_coef": 0.25,
     "default_difficulty": "medium", "spec_seconds": {"samep-j3-Sside-024": 900}}
末行判定行 POOL_MAKE=PASS stage=<阶段,…> specs=<n> gl=<n> aspen=<n> pool=<路径>。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import uuid
from pathlib import Path

SITES = ("gl", "aspen")
DEFAULT_COST = {
    # 上一轮（steer-4task-1115）定性：难度越高每局步数越多；xhard > hard > medium > easy
    "difficulty_seconds": {"easy": 60.0, "medium": 90.0, "hard": 120.0, "xhard": 150.0},
    "stage_mult": {},
    "j_coef": 0.25,
    "default_difficulty": "medium",
    "spec_seconds": {},
}
J_IN_NAME = re.compile(r"(?:^|[-_])j(\d+)(?:[-_.]|$)")
SITE_SUFFIX = re.compile(r"\.(gl|aspen)$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._+=-]+$")


def spec_name(path):
    """文件名去 .json，再去可选的 .gl/.aspen 后缀。"""
    stem = Path(path).name
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    return SITE_SUFFIX.sub("", stem)


def load_cost(path):
    cost = json.loads(json.dumps(DEFAULT_COST))
    if path:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(user) - set(cost)
        if unknown:
            raise SystemExit(f"--cost-json 含未知键 {sorted(unknown)}，只允许 {sorted(cost)}")
        for key, value in user.items():
            if isinstance(cost[key], dict):
                cost[key].update(value)
            else:
                cost[key] = value
    if cost["default_difficulty"] not in cost["difficulty_seconds"]:
        raise SystemExit(f"default_difficulty={cost['default_difficulty']!r} 不在 difficulty_seconds 里")
    return cost


def spec_j(spec, name):
    if "j" in spec:
        return int(spec["j"])
    match = J_IN_NAME.search(name)
    return int(match.group(1)) if match else 0


def spec_difficulties(spec, default):
    manifest = spec.get("manifest")
    if manifest:
        try:
            episodes = json.loads(Path(manifest).read_text(encoding="utf-8")).get("episodes") or []
        except (OSError, ValueError):
            episodes = []
        diffs = [ep.get("difficulty") or default for ep in episodes]
        if diffs:
            return diffs
    return [default]


def estimate_seconds(spec, name, stage, cost):
    if name in cost["spec_seconds"]:
        return float(cost["spec_seconds"][name])
    j = spec_j(spec, name)
    mult = float(cost["stage_mult"].get(stage, 1.0)) * (1.0 + float(cost["j_coef"]) * j)
    total = 0.0
    for diff in spec_difficulties(spec, cost["default_difficulty"]):
        if diff not in cost["difficulty_seconds"]:
            raise SystemExit(f"spec {name}: 难度 {diff!r} 不在成本表 difficulty_seconds 里")
        total += float(cost["difficulty_seconds"][diff]) * mult
    return total


def collect(stage_specs):
    """返回 [{name, stage, files: {site: 源路径}}]；同一 spec 名跨阶段重复即报错。"""
    items, seen = [], {}
    for stage, gl_glob, aspen_glob in stage_specs:
        if not SAFE_NAME.match(stage):
            raise SystemExit(f"阶段名 {stage!r} 含非法字符")
        by_name = {}
        for site, pattern in zip(SITES, (gl_glob, aspen_glob)):
            if pattern == "-":
                continue
            # 宽 glob（如 *.json）会同时扫到另一站点的 <名>.gl.json / <名>.aspen.json，按后缀剔除
            other = [o for o in SITES if o != site]
            matches = sorted(p for p in glob.glob(pattern) if not any(p.endswith(f".{o}.json") for o in other))
            if not matches:
                raise SystemExit(f"阶段 {stage} 站点 {site}: glob {pattern!r} 没有匹配任何文件")
            for path in matches:
                name = spec_name(path)
                if not SAFE_NAME.match(name):
                    raise SystemExit(f"spec 名 {name!r} 含非法字符（{path}）")
                files = by_name.setdefault(name, {})
                if site in files:
                    raise SystemExit(f"阶段 {stage} 站点 {site}: spec 名 {name} 出现两次（{files[site]}，{path}）")
                files[site] = str(Path(path).resolve())
        for name, files in sorted(by_name.items()):
            if name in seen:
                raise SystemExit(f"spec 名 {name} 同时出现在阶段 {seen[name]} 与 {stage}，池内键必须唯一")
            seen[name] = stage
            items.append({"name": name, "stage": stage, "files": files})
    return items


def write_json_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def cmd_make(args):
    cost = load_cost(args.cost_json)
    items = collect(args.stage_spec)
    if not items:
        raise SystemExit("没有任何 spec")
    stages = []
    for stage, _, _ in args.stage_spec:
        if stage not in stages:
            stages.append(stage)
    pool_name = args.pool_name or "+".join(stages)
    if not SAFE_NAME.match(pool_name):
        raise SystemExit(f"池名 {pool_name!r} 含非法字符")
    pool = Path(args.out) / "pool" / pool_name
    if pool.exists() and any(pool.iterdir()):
        raise SystemExit(f"池目录已存在且非空，拒绝覆盖：{pool}（换池名或先把旧池改名留档）")
    for item in items:
        # 估成本优先用 GL 版 spec（两套只差站点路径），缺则用 aspen 版
        src = item["files"].get("gl") or item["files"]["aspen"]
        spec = json.loads(Path(src).read_text(encoding="utf-8"))
        item["cost_seconds"] = estimate_seconds(spec, item["name"], item["stage"], cost)
    # 成本降序；同成本按阶段在命令行出现的顺序、再按名字，保证确定性
    order = {stage: i for i, stage in enumerate(stages)}
    items.sort(key=lambda it: (-it["cost_seconds"], order[it["stage"]], it["name"]))
    if len(items) > 10 ** 6:
        raise SystemExit("spec 超过 6 位序号上限")
    for sub in ("todo", "claim", "done", "fail", "logs"):
        (pool / sub).mkdir(parents=True, exist_ok=True)
    counts = {site: 0 for site in SITES}
    entries = []
    for seq, item in enumerate(items):
        key = f"{seq:06d}-{item['name']}"
        for site, src in sorted(item["files"].items()):
            spec = json.loads(Path(src).read_text(encoding="utf-8"))
            write_json_atomic(pool / "todo" / f"{key}.{site}.json", spec)
            counts[site] += 1
        entries.append({"seq": seq, "key": key, "name": item["name"], "stage": item["stage"],
                        "cost_seconds": round(item["cost_seconds"], 3), "sources": item["files"]})
    manifest = {"schema_version": 1, "pool": str(pool), "stages": stages, "cost_model": cost,
                "specs": len(entries), "sites": counts, "entries": entries}
    write_json_atomic(pool / "manifest.json", manifest)
    total = sum(e["cost_seconds"] for e in entries)
    print(f"POOL_COST total_seconds={total:.0f} max_seconds={entries[0]['cost_seconds']:.0f} "
          f"min_seconds={entries[-1]['cost_seconds']:.0f}")
    print(f"POOL_MAKE=PASS stage={','.join(stages)} specs={len(entries)} gl={counts['gl']} "
          f"aspen={counts['aspen']} pool={pool}")
    return 0


def pool_status(pool):
    pool = Path(pool)
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    keys = [e["key"] for e in manifest["entries"]]
    claimed = {p.name for p in (pool / "claim").iterdir()} if (pool / "claim").is_dir() else set()
    done = {p.name[: -len(".result")] for p in (pool / "done").glob("*.result")}
    fail = {p.name[: -len(".result")] for p in (pool / "fail").glob("*.result")}
    todo = [k for k in keys if k not in claimed]
    running = [k for k in keys if k in claimed and k not in done and k not in fail]
    return {"specs": len(keys), "todo": len(todo), "running": len(running), "done": len(done),
            "fail": len(fail), "running_keys": running, "fail_keys": sorted(fail)}


def cmd_status(args):
    pool = Path(args.out) / "pool" / args.pool_name
    st = pool_status(pool)
    for key in st["running_keys"]:
        print(f"POOL_RUNNING key={key}")
    for key in st["fail_keys"]:
        print(f"POOL_FAILED key={key}")
    verdict = "DONE" if st["todo"] == 0 and st["running"] == 0 else "OPEN"
    print(f"POOL_STATUS={verdict} pool={pool} specs={st['specs']} todo={st['todo']} running={st['running']} "
          f"done={st['done']} fail={st['fail']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make")
    make.add_argument("--out", required=True)
    make.add_argument("--stage-spec", nargs=3, action="append", required=True,
                      metavar=("STAGE", "GL_GLOB", "ASPEN_GLOB"))
    make.add_argument("--pool-name")
    make.add_argument("--cost-json")
    status = sub.add_parser("status")
    status.add_argument("--out", required=True)
    status.add_argument("--pool-name", required=True)
    args = parser.parse_args(argv)
    return cmd_make(args) if args.cmd == "make" else cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
