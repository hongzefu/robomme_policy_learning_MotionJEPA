"""1017-test-xhard5-12per 评估清单：从子模块 test metadata 导出 1200 局清单，并维护 GL 抢单用的 manifest.tsv。

两个子命令：

``export``——读 ``<metadata-dir>/record_dataset_<Task>_metadata.json``（16 个任务；结构
``{"env_id","record_count","records":[{"task","episode","seed","difficulty"}...]}``），产出：

* ``eval-testx-1200.jsonl``：每任务每档按 episode 升序取前 12 条（easy／medium／hard 取官方 ep0–49，
  xhard 五档取本次追加的 ep50 起）；行 ``{"task","episode","seed","difficulty"}``，
  按 (task, difficulty, episode) 排序；断言 rows=1200、cells=100、每格 ≥12；
* ``eval-testx-smoke2.jsonl``：两次冒烟（官方 79999 与 512+motion@10000）共用的同一份 2 局子清单
  （用户 2026-10-18「10k 100 局冒烟降低到和官方一样的2局」）：按排序取第一个 base 格与第一个 xhard 格的各第 1 条
  （合成与真实清单里都是 BinFill/easy ep0 与 BinFill/xhard12 ep50），两条分流路径各一局；断言为全量清单的子集；
* ``metadata/record_dataset_<Task>XHard_metadata.json``：14 个有 xhard 档的任务各一份 XHard 副本，
  只含 xhard 记录、记录的 task 改成 ``<Task>XHard``（``BenchmarkEnvBuilder`` 按 (task, episode) 查表，
  原任务名查不到 ``<Task>XHard`` 的键）；评估时经 ``XHARD_METADATA_DIR`` 指向这里；
* ``manifest_meta.json``：源 metadata 逐文件 sha256 与计数。
判定行 ``MANIFEST=PASS rows=1200 cells=100``、``SMOKE_SUBSET=PASS rows=2 subset_of_full=yes``、``XHARD_METADATA=PASS``。

``units``——往 ``manifest.tsv``（``gl_hold_pool.sh`` 的六列 TAB 格式：run_name / ckpt / policy /
policy_config / expect_ckpt_id / shard_index）追加单元。每个 ``--run`` 是
``run_name|ckpt绝对路径|policy|policy_config|step|分片数|motion(0/1)``，按给出的顺序逐 run 写 s0..s(n-1)；
worker 自上而下抢单，所以写入顺序就是领取优先级（计划 T9：正式轮 512 → 512+motion → 官方）。
``motion`` 列与 policy 名是否以 ``-motion`` 结尾必须一致——``gl_hold_pool.sh`` 靠这个后缀决定 resample 与
EXPECT_MOTION_STATS。``--stage smoke`` 要求本次恰好 2 个单元（官方 2 局 1 个 + 512+motion@10k 同 2 局 1 个）；
``--stage formal`` 要求恰好 3 个 run、各 10 片、motion 依次为 0 / 1 / 0。已存在的单元名拒绝重复写入。
判定行 ``UNITS=PASS stage=<s> added=<n> total=<m>``。

``export-official``——官方 800 局口径（1019，用户「a 同意开工」；与历史 eval-orig80k / eval-orig40k 同口径：
16 任务 × test split ep0–49）。读同一份 test metadata，每任务只取 ``episode < --episodes``（默认 50）的记录，
沿用 ``load_task`` 的「base 档必须 ep<50」断言，所以 ep50 起追加的 xhard 记录一律排除；断言每任务恰 50 条、
episode 恰为 0..49、总行数 = ``--expect-rows``（默认 800）。产出 ``eval-official-800.jsonl``（字段与排序同
``eval-testx-1200.jsonl``）与 ``manifest_meta.json``（metadata 目录、子模块 HEAD、逐文件 sha256、行数、清单 sha256）。
判定行 ``MANIFEST_OFFICIAL=PASS rows=800 tasks=16``。分片与单元沿用 ``make_shard_plan.py shards`` 与 ``units --stage any``。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess

#: 官网 Task 编号顺序（benchmark 仓库 P6）
TASK_ORDER = (
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube",
    "VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
)
BASE_DIFFICULTIES = ("easy", "medium", "hard")
XHARD_DIFFICULTIES = ("xhard12", "xhard14", "xhard16", "xhard18", "xhard20")
DIFFICULTY_ORDER = BASE_DIFFICULTIES + XHARD_DIFFICULTIES
BASE_EPISODE_LIMIT = 50
DEFAULT_METADATA_DIR = "third_party/robomme_benchmark/src/robomme/env_metadata/test"
FULL_NAME = "eval-testx-1200.jsonl"
SMOKE_NAME = "eval-testx-smoke2.jsonl"
OFFICIAL_NAME = "eval-official-800.jsonl"


def row_key(row):
    return (row["task"], row["difficulty"], int(row["episode"]))


def dump_jsonl(path: Path, rows) -> str:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_task(metadata_dir: Path, task: str):
    path = metadata_dir / f"record_dataset_{task}_metadata.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("env_id") != task:
        raise SystemExit(f"{path} 的 env_id={payload.get('env_id')!r} 不是 {task}")
    records = payload["records"]
    if payload.get("record_count") != len(records):
        raise SystemExit(f"{path} record_count={payload.get('record_count')} 与 records 条数 {len(records)} 不符")
    seen = set()
    for r in records:
        if r["task"] != task:
            raise SystemExit(f"{path} 里混入别的任务：{r}")
        ep = int(r["episode"])
        if ep in seen:
            raise SystemExit(f"{path} 里 episode {ep} 重复")
        seen.add(ep)
        d = r["difficulty"]
        if d not in DIFFICULTY_ORDER:
            raise SystemExit(f"{path} ep{ep} 的难度 {d!r} 不在 {DIFFICULTY_ORDER}")
        # 官方 ep0–49 只有 base 档；追加的 xhard 一律 ep≥50（计划 3.1 / T5）
        if (d in BASE_DIFFICULTIES) != (ep < BASE_EPISODE_LIMIT):
            raise SystemExit(f"{path} ep{ep} 的难度 {d} 与 episode 区间不符（base 必须 <50，xhard 必须 ≥50）")
    return records, hashlib.sha256(raw).hexdigest()


def cmd_export(args):
    metadata_dir = Path(args.metadata_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / FULL_NAME, out / SMOKE_NAME, out / "manifest_meta.json", out / "metadata"]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise SystemExit(f"拒绝覆盖已有产物：{existing}")

    full, smoke, sources, xhard_copies = [], [], {}, {}
    for task in TASK_ORDER:
        records, sha = load_task(metadata_dir, task)
        sources[task] = {"sha256": sha, "records": len(records)}
        by_diff = collections.defaultdict(list)
        for r in records:
            by_diff[r["difficulty"]].append({"task": task, "episode": int(r["episode"]),
                                             "seed": int(r["seed"]), "difficulty": r["difficulty"]})
        for d in DIFFICULTY_ORDER:
            if d not in by_diff:
                continue
            cell = sorted(by_diff[d], key=lambda r: r["episode"])
            if len(cell) < args.per_cell:
                raise SystemExit(f"{task}/{d} 只有 {len(cell)} 条，少于每格 {args.per_cell}")
            picked = cell[:args.per_cell]
            full += picked
            smoke.append(picked[0])
        xhard = sorted((r for r in records if r["difficulty"] in XHARD_DIFFICULTIES), key=lambda r: int(r["episode"]))
        if xhard:
            xhard_copies[task] = xhard

    full.sort(key=row_key)
    smoke.sort(key=row_key)
    smoke = [next(r for r in smoke if r["difficulty"] not in XHARD_DIFFICULTIES),
             next(r for r in smoke if r["difficulty"] in XHARD_DIFFICULTIES)]
    keys = [row_key(r) for r in full]
    if len(set(keys)) != len(keys):
        raise SystemExit("清单里有重复的 (task, difficulty, episode)")
    cells = collections.Counter((r["task"], r["difficulty"]) for r in full)
    if len(full) != args.expect_rows or len(cells) != args.expect_cells:
        raise SystemExit(f"MANIFEST=FAIL rows={len(full)} cells={len(cells)}（期望 {args.expect_rows}/{args.expect_cells}）")
    if set(cells.values()) != {args.per_cell}:
        raise SystemExit(f"MANIFEST=FAIL 每格条数不全是 {args.per_cell}：{dict(cells)}")
    full_set = set(keys)
    smoke_keys = {row_key(r) for r in smoke}
    if len(smoke) != 2 or not smoke_keys <= full_set:
        raise SystemExit(f"SMOKE_SUBSET=FAIL rows={len(smoke)} subset_of_full={'yes' if smoke_keys <= full_set else 'no'}")

    full_sha = dump_jsonl(out / FULL_NAME, full)
    smoke_sha = dump_jsonl(out / SMOKE_NAME, smoke)

    meta_dir = out / "metadata"
    meta_dir.mkdir()
    n_xhard = 0
    for task, recs in xhard_copies.items():
        name = f"{task}XHard"
        payload = {"env_id": name, "record_count": len(recs),
                   "records": [{"task": name, "episode": int(r["episode"]), "seed": int(r["seed"]),
                                "difficulty": r["difficulty"]} for r in recs]}
        (meta_dir / f"record_dataset_{name}_metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        n_xhard += len(recs)
    if len(xhard_copies) != args.expect_xhard_tasks:
        raise SystemExit(f"XHARD_METADATA=FAIL tasks={len(xhard_copies)}（期望 {args.expect_xhard_tasks}）")

    meta = {"metadata_dir": str(metadata_dir.resolve()), "sources": sources, "per_cell": args.per_cell,
            "rows": len(full), "cells": len(cells), "full": {"file": FULL_NAME, "sha256": full_sha},
            "smoke": {"file": SMOKE_NAME, "sha256": smoke_sha, "rows": len(smoke)},
            "xhard_metadata": {"tasks": sorted(xhard_copies), "records": n_xhard}}
    (out / "manifest_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"MANIFEST=PASS rows={len(full)} cells={len(cells)} sha256={full_sha}", flush=True)
    print(f"SMOKE_SUBSET=PASS rows={len(smoke)} subset_of_full=yes sha256={smoke_sha}", flush=True)
    print(f"XHARD_METADATA=PASS tasks={len(xhard_copies)} records={n_xhard} dir={meta_dir}", flush=True)


def git_head(path: Path):
    """metadata 目录所在 git 仓库（子模块）的 HEAD；取不到写 None，只作留档。"""
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def cmd_export_official(args):
    metadata_dir = Path(args.metadata_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = [str(p) for p in (out / OFFICIAL_NAME, out / "manifest_meta.json") if p.exists()]
    if existing:
        raise SystemExit(f"拒绝覆盖已有产物：{existing}")
    rows, sources = [], {}
    for task in TASK_ORDER:
        records, sha = load_task(metadata_dir, task)
        picked = sorted((r for r in records if int(r["episode"]) < args.episodes), key=lambda r: int(r["episode"]))
        eps = [int(r["episode"]) for r in picked]
        if eps != list(range(args.episodes)):
            raise SystemExit(f"MANIFEST_OFFICIAL=FAIL {task} 的 ep<{args.episodes} 记录不是恰好 0..{args.episodes - 1}："
                             f"{len(eps)} 条")
        sources[task] = {"sha256": sha, "records": len(records), "picked": len(picked)}
        rows += [{"task": task, "episode": int(r["episode"]), "seed": int(r["seed"]), "difficulty": r["difficulty"]}
                 for r in picked]
    rows.sort(key=row_key)
    if len(rows) != args.expect_rows:
        raise SystemExit(f"MANIFEST_OFFICIAL=FAIL rows={len(rows)}（期望 {args.expect_rows}）")
    sha = dump_jsonl(out / OFFICIAL_NAME, rows)
    meta = {"metadata_dir": str(metadata_dir.resolve()), "metadata_git_head": git_head(metadata_dir),
            "episodes_per_task": args.episodes, "sources": sources, "rows": len(rows), "tasks": len(TASK_ORDER),
            "difficulties": dict(collections.Counter(r["difficulty"] for r in rows)),
            "official": {"file": OFFICIAL_NAME, "sha256": sha}}
    (out / "manifest_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"MANIFEST_OFFICIAL=PASS rows={len(rows)} tasks={len(TASK_ORDER)} sha256={sha}", flush=True)


def parse_run(spec: str):
    parts = spec.split("|")
    if len(parts) != 7:
        raise SystemExit(f"--run 需要 7 段（run|ckpt|policy|config|step|分片数|motion）：{spec}")
    run, ckpt, policy, config, step, nshards, motion = parts
    if motion not in ("0", "1") or not nshards.isdigit() or int(nshards) < 1 or not step.isdigit():
        raise SystemExit(f"--run 字段非法：{spec}")
    if not ckpt.startswith("/") or Path(ckpt).name != step:
        raise SystemExit(f"ckpt 必须是绝对路径且末级目录名等于 step：{spec}")
    if policy.endswith("-motion") != (motion == "1"):
        raise SystemExit(f"policy 名 {policy!r} 与 motion={motion} 不一致（gl_hold_pool.sh 按 -motion 后缀切 resample）")
    return dict(run=run, ckpt=ckpt, policy=policy, config=config, step=step, nshards=int(nshards), motion=motion == "1")


def cmd_units(args):
    runs = [parse_run(s) for s in args.run]
    if args.stage == "formal":
        if [r["motion"] for r in runs] != [False, True, False] or any(r["nshards"] != 10 for r in runs):
            raise SystemExit("formal 轮必须恰好 3 个 run、各 10 片，顺序 512 → 512+motion → 官方（motion 0/1/0）")
    elif args.stage == "smoke":
        if len(runs) != 2 or any(r["nshards"] != 1 for r in runs) or [r["motion"] for r in runs] != [False, True]:
            raise SystemExit("smoke 轮必须恰好 2 个单元（官方 1 片 + 512+motion@10k 1 片，同一份 2 局子清单）")
    tsv = Path(args.tsv)
    existing = []
    if tsv.exists():
        existing = [line.split("\t") for line in tsv.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = {f"{c[0]}-s{c[5]}" for c in existing}
    lines = []
    for r in runs:
        for k in range(r["nshards"]):
            name = f"{r['run']}-s{k}"
            if name in names:
                raise SystemExit(f"单元 {name} 已在 {tsv} 里，拒绝重复写入")
            if args.eval_root:
                plan = Path(args.eval_root) / r["run"] / "plans" / f"shard{k}.json"
                if not plan.is_file():
                    raise SystemExit(f"单元 {name} 的分片计划不存在：{plan}")
            names.add(name)
            lines.append("\t".join([r["run"], r["ckpt"], r["policy"], r["config"], r["step"], str(k)]))
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with tsv.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"UNITS=PASS stage={args.stage} added={len(lines)} total={len(existing) + len(lines)} tsv={tsv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("out_dir")
    p.add_argument("--metadata-dir", default=DEFAULT_METADATA_DIR)
    p.add_argument("--per-cell", type=int, default=12)
    p.add_argument("--expect-rows", type=int, default=1200)
    p.add_argument("--expect-cells", type=int, default=100)
    p.add_argument("--expect-xhard-tasks", type=int, default=14)
    o = sub.add_parser("export-official")
    o.add_argument("out_dir")
    o.add_argument("--metadata-dir", default=DEFAULT_METADATA_DIR)
    o.add_argument("--episodes", type=int, default=BASE_EPISODE_LIMIT)
    o.add_argument("--expect-rows", type=int, default=800)
    u = sub.add_parser("units")
    u.add_argument("tsv")
    u.add_argument("--stage", choices=("smoke", "formal", "any"), required=True)
    u.add_argument("--run", action="append", required=True)
    u.add_argument("--eval-root", default="", help="给出则断言 <eval-root>/<run>/plans/shard<k>.json 存在")
    args = parser.parse_args()
    {"export": cmd_export, "export-official": cmd_export_official, "units": cmd_units}[args.cmd](args)


if __name__ == "__main__":
    main()
