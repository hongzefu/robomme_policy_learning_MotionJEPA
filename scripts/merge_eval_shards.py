#!/usr/bin/env python3
"""合并 test-hard 各片逐局结果并出判定行（benchmark 0927 计划 §6.4、§8.2 第 5 项）。

两种输入格式都认：MME-VLA 各片 ``<save_root>/mmevla-testhard/ckpt79999/seed7/episodes.jsonl``；
SimpleMemVLA 各片 ``results-r<轮>-shard<i>of10.jsonl``。每个身份取最后一条记录作终态（error 身份的重评以最后一次为准）。

    python scripts/merge_eval_shards.py --policy mmevla --identities <eval-identities-1100.jsonl> \
        --round 1 <片结果文件…>  [--round2 <第二轮片结果文件…>] [--specs-root <benchmark>/src/robomme_hard/env_metadata/test-hard]

判定行：``EVAL_ROUND<r>``（每片 55、每轮 550、每格 10、error_left、重跑次数 ≤ 55）、两轮都给时再出
``EVAL_IDENTITY_SET``、``EVAL_BINDING``、``EVAL_TIER_CAP``、``EVAL_DEMO_FRAMES``（参考层）与分档成功率表。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

NORMAL = ("success", "fail", "timeout")
TIER_MAX_STEPS = {"xhard1": 1500, "xhard2": 1700, "xhard3": 2000, "xhard4": 2600}
TOL = 1e-5


def load(paths: list[str]) -> tuple[dict, dict, dict]:
    """返回 (身份→终态记录, 片→身份集合, 片→基础设施重跑次数)。"""
    final, per_shard, retries = {}, collections.defaultdict(set), collections.Counter()
    for path in paths:
        seen = collections.Counter()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ident = rec.get("identity") or {}
            key = (rec["task"], ident.get("tier", rec.get("tier")), int(ident.get("seed", rec.get("seed", -1))),
                   ident.get("spec_sha256", rec.get("spec_sha256")))
            seen[key] += 1
            per_shard[path].add(key)
            final[key] = dict(rec, _key=key, _shard=path)
        retries[path] = sum(v - 1 for v in seen.values())
    return final, per_shard, retries


def round_gate(policy: str, r: int, final: dict, per_shard: dict, retries: dict) -> bool:
    normal = sum(1 for rec in final.values() if rec["status"] in NORMAL)
    error_left = len(final) - normal
    cells = collections.Counter((k[0], k[1]) for k in final)
    shard_sizes = sorted(len(v) for v in per_shard.values())
    worst = max(retries.values(), default=0)
    ok = (len(final) == 550 and normal == 550 and error_left == 0 and len(per_shard) == 10
          and set(shard_sizes) == {55} and set(cells.values()) == {10} and len(cells) == 55 and worst <= 55)
    print(f"EVAL_ROUND{r}={'PASS' if ok else 'FAIL'} policy={policy} episodes={len(final)} normal={normal} "
          f"error_left={error_left} retries={sum(retries.values())}/55(max_per_shard={worst}) shards={len(per_shard)} "
          f"per_shard={shard_sizes[0] if shard_sizes else 0}-{shard_sizes[-1] if shard_sizes else 0} shape=55x10")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--identities", required=True)
    ap.add_argument("--round1", nargs="+", required=True)
    ap.add_argument("--round2", nargs="*", default=None)
    ap.add_argument("--specs-root", default=None, help="包内 test-hard 目录；给了就出 EVAL_DEMO_FRAMES 参考层")
    ap.add_argument("--table", default=None, help="分档成功率表输出（json）")
    args = ap.parse_args()
    f1, s1, r1 = load(args.round1)
    ok = round_gate(args.policy, 1, f1, s1, r1)
    if not args.round2:
        return 0 if ok else 1
    f2, s2, r2 = load(args.round2)
    ok &= round_gate(args.policy, 2, f2, s2, r2)
    want = {(r["task"], r["tier"], int(r["seed"]), r["spec_sha256"])
            for r in map(json.loads, Path(args.identities).read_text().splitlines()) if r}
    both = {**f1, **f2}
    overlap = set(f1) & set(f2)
    missing, extra = want - set(both), set(both) - want
    ids_ok = not missing and not extra and not overlap and len(both) == 1100
    print(f"EVAL_IDENTITY_SET={'PASS' if ids_ok else 'FAIL'} policy={args.policy} rounds=2 shards=10 per_shard=55 "
          f"episodes={len(both)} missing={len(missing)} extra={len(extra)} dup={len(overlap)}")
    replay = injected = drift = unused = bad_mode = 0
    drift_max = 0.0
    cap_bad = 0
    for key, rec in both.items():
        b = rec.get("spec_binding") or {}
        good = (b.get("available") and b.get("mode") == "replay" and b.get("value_points", 0) > 0
                and b.get("spec_sha256") == key[3])
        replay += int(bool(good))
        bad_mode += int(not good)
        injected += int(b.get("injected_mismatch", 0) or 0)
        drift += int(b.get("recorded_drift", 0) or 0)
        drift_max = max(drift_max, float(b.get("recorded_max_abs", 0.0) or 0.0))
        unused += int(b.get("unused", 0) or 0)
        cap_bad += int(rec.get("max_steps") != TIER_MAX_STEPS.get(key[1]))
    bind_ok = replay == 1100 and injected == 0 and unused == 0 and bad_mode == 0
    print(f"EVAL_BINDING={'PASS' if bind_ok else 'FAIL'} policy={args.policy} episodes={len(both)} replay={replay} "
          f"injected_mismatch={injected} recorded_drift={drift} max_abs={drift_max:.2g} unused={unused}")
    print(f"EVAL_TIER_CAP={'PASS' if cap_bad == 0 else 'FAIL'} policy={args.policy} episodes={len(both)} mismatch={cap_bad}")
    if args.specs_root:
        old = {}
        for tier in TIER_MAX_STEPS:
            lines = (Path(args.specs_root) / tier / "specs.jsonl").read_text().splitlines()[1:]
            for row in map(json.loads, lines):
                ro = row.get("rollout") or {}
                if row.get("selected") and ro.get("status") == "ok":
                    old[(row["task"], tier, int(row["seed"]), row["spec_sha256"])] = ro.get("demo_frames")
        diffs = [abs(int(rec["demo_frames"]) - int(old[k])) for k, rec in both.items()
                 if rec.get("demo_frames") is not None and old.get(k) is not None]
        print(f"EVAL_DEMO_FRAMES=INFO policy={args.policy} episodes={len(diffs)} exact={sum(d == 0 for d in diffs)} "
              f"within_5={sum(d <= 5 for d in diffs)} max_diff={max(diffs, default=None)}")
    table = collections.defaultdict(lambda: collections.Counter())
    for key, rec in both.items():
        table[(key[0], key[1])][rec["status"]] += 1
    by_tier = collections.defaultdict(lambda: collections.Counter())
    for (task, tier), c in table.items():
        by_tier[tier].update(c)
    for tier in TIER_MAX_STEPS:
        c = by_tier[tier]
        n = sum(c.values())
        print(f"EVAL_TIER_SUCCESS policy={args.policy} tier={tier} episodes={n} success={c['success']} "
              f"rate={c['success'] / n if n else 0:.3f} fail={c['fail']} timeout={c['timeout']} error={c['error']}")
    if args.table:
        Path(args.table).write_text(json.dumps({f"{t}/{tier}": dict(c) for (t, tier), c in sorted(table.items())},
                                               ensure_ascii=False, indent=1) + "\n")
    return 0 if ok and ids_ok and bind_ok and cap_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
