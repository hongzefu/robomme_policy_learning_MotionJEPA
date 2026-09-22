"""把 fork 的 test/primary 700 条切成分片计划，或生成资源探针用的小计划。

分片口径：700 条按 ``(task, difficulty, episode)`` 排序后 ``idx % shards`` 发牌。
难度在这批候选里是**组属性**（整组 50 条同难度），不是 episode 属性，且 50 是 10 的倍数，
所以每片恰好从 14 个组各取 5 条，10 片的难度组构成完全相同——不需要额外分层。
（官方 test split 的难度是 4 周期 e,e,m,h，同样的发牌法会退化成每片只含 2 种难度，那是另一回事。）
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import robomme
from robomme.injection_candidates import load_candidates, candidate_key

#: 建环境最重的四组：demo 最长、步数上限最高，探针用它们逼出内存与耗时的最坏侧
HEAVIEST = (("VideoUnmaskSwap", "xhard"), ("VideoRepick", "xhard"),
            ("RouteStick", "xhard"), ("BinFill", "hard"))
DEFAULT_CANDIDATES = "third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl"


def load_primary(candidates: Path):
    benchmark_root = Path(robomme.__file__).resolve().parents[2]
    header, rows = load_candidates(candidates, repo_root=benchmark_root)
    primary = [r for r in rows if r["split"] == "test" and r["role"] == "primary"]
    if len(primary) != 700:
        raise SystemExit(f"test/primary 应为 700 条，实际 {len(primary)}")
    groups = collections.Counter((r["task"], r["difficulty"]) for r in primary)
    if len(groups) != 14 or set(groups.values()) != {50}:
        raise SystemExit(f"应为 14 组 × 50，实际 {dict(groups)}")
    primary.sort(key=candidate_key)
    return header, primary


def write_plan(path: Path, tag: str, candidates: Path, identity: str, rows: list) -> str:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[f"{row['task']}/{row['difficulty']}"].append(row["episode"])
    payload = {
        "shard_tag": tag,
        "candidates": str(candidates),
        "identity_sha256": identity,
        "groups": {g: sorted(eps) for g, eps in sorted(groups.items())},
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cmd_shards(args):
    candidates = Path(args.candidates).resolve()
    header, primary = load_primary(candidates)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shards = [primary[i::args.shards] for i in range(args.shards)]
    manifest = {"candidates": str(candidates), "identity_sha256": header["identity_sha256"],
                "shards": args.shards, "total": len(primary), "files": {}}

    seen = set()
    for i, rows in enumerate(shards):
        tag = f"s{i}"
        path = out / f"shard{i}.json"
        manifest["files"][path.name] = write_plan(path, tag, candidates, header["identity_sha256"], rows)
        keys = {candidate_key(r) for r in rows}
        if keys & seen:
            raise SystemExit(f"分片 {i} 与前面的分片相交")
        seen |= keys
        per_group = collections.Counter((r["task"], r["difficulty"]) for r in rows)
        if len(per_group) != 14 or set(per_group.values()) != {len(rows) // 14}:
            raise SystemExit(f"分片 {i} 的组分布不均：{dict(per_group)}")

    if len(seen) != len(primary):
        raise SystemExit(f"分片并集 {len(seen)} != {len(primary)}")
    (out / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per = len(primary) // args.shards
    print(f"PLAN_OK shards={args.shards} total={len(primary)} per_shard={per} "
          f"disjoint=True per_group={per // 14} identity={header['identity_sha256'][:12]}", flush=True)


def cmd_probe(args):
    candidates = Path(args.candidates).resolve()
    header, primary = load_primary(candidates)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_group = collections.defaultdict(list)
    for row in primary:
        by_group[(row["task"], row["difficulty"])].append(row)

    # A：14 组各 2 条 + 最重 4 组各再 1 条 = 32 集。32 > 27，故意跨过 Vulkan 静态 TLS 的默认红线；
    #    14 组全覆盖顺带把各组的 demo 长度形状写进共享 JAX 编译缓存。
    rows_a = [r for g in sorted(by_group) for r in by_group[g][:2]]
    rows_a += [by_group[g][2] for g in HEAVIEST]
    # B/C/D：只跑最重四组各 2 条，逼出内存与单集耗时的最坏侧
    rows_bcd = [r for g in HEAVIEST for r in by_group[g][:2]]

    for tag, rows in (("A", rows_a), ("BCD", rows_bcd)):
        path = out / f"probe-{tag}.json"
        sha = write_plan(path, f"probe-{tag}", candidates, header["identity_sha256"], rows)
        print(f"PROBE_PLAN {tag} episodes={len(rows)} groups={len({(r['task'], r['difficulty']) for r in rows})} "
              f"file={path} sha256={sha[:12]}", flush=True)


def cmd_task(args):
    """单任务分片计划（0922-binfill-demo-prefix-plan.md D 节）：BinFill 150 条切 10 片，每片 15 集。

    与 ``shards`` 的两点不同：只收一个任务的组，且**只收 demo 前缀预生成成功的集**——
    ``--demo-store`` 给出前缀库根时，按 ``index.json`` 里 ``demo_status == "ok"`` 过滤。
    planner 失败的集在这里就被排除，评测侧的 ``DemoPrefixStore.get`` 因此永远不会缺条目
    （缺了它会直接 raise，绝不静默无 demo 跑）。
    """
    candidates = Path(args.candidates).resolve()
    header, primary = load_primary(candidates)
    rows = [r for r in primary if r["task"] == args.task]
    if not rows:
        raise SystemExit(f"候选库里没有任务 {args.task}")

    allowed = None
    if args.demo_store:
        index = json.loads((Path(args.demo_store) / "index.json").read_text(encoding="utf-8"))
        allowed = {k for k, v in index.items() if v.get("demo_status") == "ok"}
        before = len(rows)
        rows = [r for r in rows if f"{r['task']}/{r['difficulty']}/{r['episode']}" in allowed]
        dropped = before - len(rows)
        if dropped:
            print(f"[plan] 按 demo 前缀库剔除 {dropped} 条（预生成未成功）", flush=True)

    if len(rows) < args.shards:
        raise SystemExit(f"只剩 {len(rows)} 条，少于 --shards={args.shards}")
    # 不要求整除：``rows[i::shards]`` 天然保证各片最多差 1 条。
    # planner 失败剔除后条数几乎不可能还是 10 的倍数，强行要求整除只会让计划出不来。
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=candidate_key)
    shards = [rows[i::args.shards] for i in range(args.shards)]
    manifest = {"candidates": str(candidates), "identity_sha256": header["identity_sha256"],
                "task": args.task, "shards": args.shards, "total": len(rows),
                "demo_store": args.demo_store, "files": {}}
    seen = set()
    for i, shard_rows in enumerate(shards):
        path = out / f"shard{i}.json"
        manifest["files"][path.name] = write_plan(
            path, f"s{i}", candidates, header["identity_sha256"], shard_rows)
        keys = {candidate_key(r) for r in shard_rows}
        if keys & seen:
            raise SystemExit(f"分片 {i} 与前面的分片相交")
        seen |= keys
    if len(seen) != len(rows):
        raise SystemExit(f"分片并集 {len(seen)} != {len(rows)}")
    (out / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_group = collections.Counter(r["difficulty"] for r in rows)
    print(f"PLAN_OK task={args.task} shards={args.shards} total={len(rows)} "
          f"per_shard={len(rows) // args.shards} disjoint=True by_difficulty={dict(sorted(per_group.items()))} "
          f"identity={header['identity_sha256'][:12]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("shards", "probe", "task"))
    parser.add_argument("out_dir")
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--task", default="BinFill", help="mode=task 时只收这一个任务")
    parser.add_argument("--demo-store", default="",
                        help="mode=task 时按该 demo 前缀库的 index.json 过滤，只收 demo_status=ok 的集")
    args = parser.parse_args()
    if args.mode == "shards":
        if 700 % args.shards or (700 // args.shards) % 14:
            raise SystemExit(f"--shards={args.shards} 无法把 14 组 × 50 均分")
        cmd_shards(args)
    elif args.mode == "task":
        cmd_task(args)
    else:
        cmd_probe(args)


if __name__ == "__main__":
    main()
