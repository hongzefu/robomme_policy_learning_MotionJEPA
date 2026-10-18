"""把 1017 的 1200 局清单（``make_manifest.py export`` 产出）切成分片计划。

分片口径（1017-test-xhard5-12per 计划 4.1）：清单按 ``(task, difficulty, episode)`` 排序后 ``idx % shards`` 发牌。
每格 12 条在排序后连续，发牌后每格在各片之间最多差 1 条；1200 / 10 = 每片恰 120 局。
不再读 0918 的候选库、不再写死 4 任务 / 700 条 / 50 的倍数。

每片写 ``<out_dir>/shard<i>.json``：
``{"shard_tag","manifest","manifest_sha256","identity_sha256","groups":{"任务/难度":[episode...]}}``，
``identity_sha256`` 与 ``manifest_sha256`` 相同（``merge_shards.py`` 沿用这个键做三模型清单一致性核对）。
``--rows N`` 只取排序后前 N 行（官方 2 局链路冒烟用）。

判定行：``SHARD_PLAN=PASS shards=<n> per_shard=<m> total=<t> cells=<c>``（各片总数不全相等时 per_shard 写成 min-max）。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def row_key(row):
    return (row["task"], row["difficulty"], int(row["episode"]))


def load_manifest(path: Path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    keys = [row_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"清单 {path} 里有重复的 (task, difficulty, episode)")
    rows.sort(key=row_key)
    return rows, hashlib.sha256(raw).hexdigest()


def write_plan(path: Path, tag: str, manifest: Path, sha: str, rows: list) -> str:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[f"{row['task']}/{row['difficulty']}"].append(int(row["episode"]))
    payload = {
        "shard_tag": tag,
        "manifest": str(manifest),
        "manifest_sha256": sha,
        "identity_sha256": sha,
        "groups": {g: sorted(eps) for g, eps in sorted(groups.items())},
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_shards(manifest: Path, out: Path, shards: int, rows_limit: int = 0, expect_per_shard: int = 0) -> dict:
    manifest = manifest.resolve()
    rows, sha = load_manifest(manifest)
    if rows_limit:
        rows = rows[:rows_limit]
    if shards < 1 or len(rows) < shards:
        raise SystemExit(f"清单只有 {len(rows)} 行，无法切成 {shards} 片")
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in out.glob("shard*.json"))
    if existing or (out / "plan_manifest.json").exists():
        raise SystemExit(f"拒绝覆盖已有分片计划：{out} {existing}")

    parts = [rows[i::shards] for i in range(shards)]
    all_cells = sorted({(r["task"], r["difficulty"]) for r in rows})
    per_cell = {c: [] for c in all_cells}
    files, seen = {}, set()
    for i, part in enumerate(parts):
        path = out / f"shard{i}.json"
        files[path.name] = write_plan(path, f"s{i}", manifest, sha, part)
        keys = {row_key(r) for r in part}
        if keys & seen:
            raise SystemExit(f"分片 {i} 与前面的分片相交")
        seen |= keys
        counts = collections.Counter((r["task"], r["difficulty"]) for r in part)
        for c in all_cells:
            per_cell[c].append(counts.get(c, 0))
    if len(seen) != len(rows):
        raise SystemExit(f"分片并集 {len(seen)} != {len(rows)}")
    sizes = [len(p) for p in parts]
    if max(sizes) - min(sizes) > 1 or (len(rows) % shards == 0 and len(set(sizes)) != 1):
        raise SystemExit(f"SHARD_PLAN=FAIL 各片总数不均：{sizes}")
    bad = {f"{t}/{d}": v for (t, d), v in per_cell.items() if max(v) - min(v) > 1}
    if bad:
        raise SystemExit(f"SHARD_PLAN=FAIL 这些格在各片之间差超过 1：{bad}")
    if expect_per_shard and set(sizes) != {expect_per_shard}:
        raise SystemExit(f"SHARD_PLAN=FAIL 每片应为 {expect_per_shard}，实际 {sizes}")

    summary = {"manifest": str(manifest), "manifest_sha256": sha, "identity_sha256": sha,
               "shards": shards, "total": len(rows), "cells": len(all_cells), "sizes": sizes, "files": files}
    (out / "plan_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per = str(sizes[0]) if len(set(sizes)) == 1 else f"{min(sizes)}-{max(sizes)}"
    print(f"SHARD_PLAN=PASS shards={shards} per_shard={per} total={len(rows)} cells={len(all_cells)} "
          f"disjoint=True manifest_sha256={sha[:12]}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("shards",))
    parser.add_argument("manifest", help="make_manifest.py export 产出的 jsonl（1200 局或 100 局冒烟子清单）")
    parser.add_argument("out_dir", help="通常是 v1-store/evaluation/<run>/plans")
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--rows", type=int, default=0, help="只取排序后前 N 行（官方 2 局链路冒烟）")
    parser.add_argument("--expect-per-shard", type=int, default=0, help="给出则断言每片恰为该数（正式轮 120）")
    args = parser.parse_args()
    make_shards(Path(args.manifest), Path(args.out_dir), args.shards, args.rows, args.expect_per_shard)


if __name__ == "__main__":
    main()
