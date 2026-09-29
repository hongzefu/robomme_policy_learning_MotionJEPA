#!/usr/bin/env python3
"""合并 test-hard 各片逐局结果并出判定行（v7：按身份清单推导各轮各片应评集合；v6 口径见 git 历史）。

两种输入格式都认：MME-VLA 各片 ``<save_root>/mmevla-testhard/ckpt79999/seed7/episodes.jsonl``；
SimpleMemVLA 各片 ``results-r<轮>-shard<i>of10.jsonl``。每个身份取最后一条记录作终态（error 身份的重评以最后一次为准）。
身份键为 ``(task, tier, seed)``（不要求 spec_sha256；清单行带了才核对）。

    python scripts/merge_eval_shards.py --policy mmevla --identities <v7 评估身份清单.jsonl> \
        --round1 <第一轮片结果文件…> [--round2 <第二轮片结果文件…>] [--specs-root <test-hard 规格目录>] [--table out.json]

清单每行 ``{task, episode, tier, seed, candidate|null, source_episode|null, round, shard}``。
判定行：``EVAL_ROUND<r>``（每个结果文件恰等于清单某一片、本轮集合等于清单本轮集合、error_left=0、每片重跑 ≤ 片大小）；
两轮都给时再出 ``EVAL_IDENTITY_SET``、``EVAL_BINDING``（replay／export 分类）、``EVAL_TIER_CAP``、
``EVAL_DEMO_FRAMES``（参考层）与 ``EVAL_TIER_SUCCESS`` 分档成功率表（xhard0..xhard4）。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

NORMAL = ("success", "fail", "timeout")
# 运行时优先从 robomme_hard 读；导入失败（如纯 CPU 合并环境）才用这张兜底表
FALLBACK_TIER_MAX_STEPS = {"xhard0": 1300, "xhard1": 1500, "xhard2": 1700, "xhard3": 2000, "xhard4": 2600}
XHARD0 = "xhard0"
EXPORT_SPEC_KIND = "native-parity/1"


def tier_max_steps() -> tuple[dict, str]:
    """返回 (档→步数上限, 来源)。"""
    try:
        from robomme_hard.env_record_wrapper.hard_specs import TIER_MAX_STEPS
        return dict(TIER_MAX_STEPS), "robomme_hard"
    except Exception:  # noqa: BLE001 - 合并脚本可在不装仿真的环境里跑
        return dict(FALLBACK_TIER_MAX_STEPS), "fallback"


def ident_key(rec: dict) -> tuple:
    """(task, tier, seed)：MME 记录从 identity 子字典取，SimpleMemVLA 记录取顶层字段。"""
    ident = rec.get("identity") or {}
    return (rec["task"], ident.get("tier", rec.get("tier")), int(ident.get("seed", rec.get("seed", -1))))


def load_manifest(path: str) -> dict:
    """返回 身份键→清单行；键重复即报错。"""
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["task"], r["tier"], int(r["seed"]))
        if key in rows:
            raise ValueError(f"清单身份重复：{key}")
        rows[key] = r
    return rows


def load(paths: list[str]) -> tuple[dict, dict, dict]:
    """返回 (身份→终态记录, 片文件→身份集合, 片文件→基础设施重跑次数)。"""
    final, per_shard, retries = {}, collections.defaultdict(set), collections.Counter()
    for path in paths:
        seen = collections.Counter()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = ident_key(rec)
            seen[key] += 1
            per_shard[path].add(key)
            final[key] = dict(rec, _key=key, _shard=path)
        retries[path] = sum(v - 1 for v in seen.values())
    return final, per_shard, retries


def round_gate(policy: str, r: int, manifest: dict, final: dict, per_shard: dict, retries: dict) -> bool:
    """本轮应评集合与各片划分都从清单推导（v6 写死的 550/55/10 改为清单给出的数）。"""
    want = {k for k, row in manifest.items() if int(row["round"]) == r}
    shards_want = collections.defaultdict(set)
    for k, row in manifest.items():
        if int(row["round"]) == r:
            shards_want[int(row["shard"])].add(k)
    normal = sum(1 for rec in final.values() if rec["status"] in NORMAL)
    error_left = len(final) - normal
    got = set(final)
    missing, extra = want - got, got - want
    # 每个结果文件必须恰好等于清单的某一片，且各片只出现一次；重跑次数上限＝本片局数
    matched, shard_bad, over_cap = set(), 0, 0
    for path, keys in per_shard.items():
        hit = [s for s, ks in shards_want.items() if ks == keys]
        if len(hit) != 1 or hit[0] in matched:
            shard_bad += 1
            continue
        matched.add(hit[0])
        over_cap += int(retries[path] > len(keys))
    shard_bad += len(set(shards_want) - matched)
    ep_bad = sum(1 for k, rec in final.items()
                 if k in manifest and "episode" in rec and int(rec["episode"]) != int(manifest[k]["episode"]))
    sizes = sorted(len(v) for v in shards_want.values())
    got_sizes = sorted(len(v) for v in per_shard.values())
    cells = collections.Counter((k[0], k[1]) for k in want)
    ok = (bool(want) and not missing and not extra and error_left == 0 and normal == len(want)
          and shard_bad == 0 and over_cap == 0 and ep_bad == 0)
    print(f"EVAL_ROUND{r}={'PASS' if ok else 'FAIL'} policy={policy} episodes={len(final)}/{len(want)} normal={normal} "
          f"error_left={error_left} retries={sum(retries.values())}(max_per_shard={max(retries.values(), default=0)},"
          f"over_cap={over_cap}) shards={len(per_shard)}/{len(shards_want)} shard_mismatch={shard_bad} "
          f"per_shard={got_sizes[0] if got_sizes else 0}-{got_sizes[-1] if got_sizes else 0}"
          f"(want {sizes[0] if sizes else 0}-{sizes[-1] if sizes else 0}) missing={len(missing)} extra={len(extra)} "
          f"episode_mismatch={ep_bad} shape={len(cells)}cells")
    return ok


def binding_gate(policy: str, manifest: dict, both: dict) -> bool:
    """回注绑定分类：非 xhard0 必须 replay、无注入不符、无未用取值点；xhard0 必须 export（native-parity/1）且无注入不符。"""
    n_replay_want = sum(1 for k in manifest if k[1] != XHARD0)
    n_export_want = sum(1 for k in manifest if k[1] == XHARD0)
    replay = export = injected = drift = unused = bad = cls_bad = layout_drift = 0
    drift_max = 0.0
    for key, rec in both.items():
        b = rec.get("spec_binding") or {}
        inj = int(b.get("injected_mismatch", 0) or 0)
        injected += inj
        drift += int(b.get("recorded_drift", 0) or 0)
        drift_max = max(drift_max, float(b.get("recorded_max_abs", 0.0) or 0.0))
        layout_drift += int(b.get("layout_drift", 0) or 0)
        want_cls = "export" if key[1] == XHARD0 else "replay"
        if rec.get("binding_class") is not None and rec["binding_class"] != want_cls:
            cls_bad += 1
        if key[1] == XHARD0:
            good = (b.get("available") and b.get("mode") == "export" and b.get("spec_kind") == EXPORT_SPEC_KIND
                    and inj == 0)
            export += int(bool(good))
        else:
            u = int(b.get("unused", 0) or 0)
            unused += u
            sha = (manifest.get(key) or {}).get("spec_sha256")
            good = (b.get("available") and b.get("mode") == "replay" and b.get("value_points", 0) > 0
                    and inj == 0 and u == 0 and (sha is None or b.get("spec_sha256") == sha))
            replay += int(bool(good))
        bad += int(not good)
    ok = (replay == n_replay_want and export == n_export_want and bad == 0 and cls_bad == 0 and injected == 0
          and len(both) == n_replay_want + n_export_want)
    print(f"EVAL_BINDING={'PASS' if ok else 'FAIL'} policy={policy} episodes={len(both)} replay={replay}/{n_replay_want} "
          f"export={export}/{n_export_want} bad={bad} class_mismatch={cls_bad} injected_mismatch={injected} "
          f"recorded_drift={drift} max_abs={drift_max:.2g} unused={unused} layout_drift={layout_drift}")
    return ok


def tier_cap_gate(policy: str, both: dict, caps: dict, caps_src: str) -> bool:
    """记录里有 max_steps 就逐局比对；没有（SimpleMemVLA）就退而核对 steps ≤ 上限 + 1（超上限那一步判 timeout）。"""
    mismatch = inferred = 0
    for key, rec in both.items():
        cap = caps.get(key[1])
        if cap is None:
            mismatch += 1
        elif rec.get("max_steps") is not None:
            mismatch += int(int(rec["max_steps"]) != cap)
        else:
            inferred += 1
            mismatch += int(int(rec.get("steps") or 0) > cap + 1)
    print(f"EVAL_TIER_CAP={'PASS' if mismatch == 0 else 'FAIL'} policy={policy} episodes={len(both)} mismatch={mismatch} "
          f"inferred_from_steps={inferred} caps={caps_src}")
    return mismatch == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--identities", required=True, help="v7 评估身份清单 jsonl")
    ap.add_argument("--round1", nargs="+", required=True)
    ap.add_argument("--round2", nargs="*", default=None)
    ap.add_argument("--specs-root", default=None, help="包内 test-hard 规格目录；给了就出 EVAL_DEMO_FRAMES 参考层")
    ap.add_argument("--table", default=None, help="分档成功率表输出（json）")
    args = ap.parse_args()
    manifest = load_manifest(args.identities)
    caps, caps_src = tier_max_steps()
    f1, s1, r1 = load(args.round1)
    ok = round_gate(args.policy, 1, manifest, f1, s1, r1)
    if not args.round2:
        return 0 if ok else 1
    f2, s2, r2 = load(args.round2)
    ok &= round_gate(args.policy, 2, manifest, f2, s2, r2)
    want = set(manifest)
    both = {**f1, **f2}
    overlap = set(f1) & set(f2)
    missing, extra = want - set(both), set(both) - want
    ids_ok = not missing and not extra and not overlap and len(both) == len(want)
    shard_sizes = sorted(collections.Counter((int(r["round"]), int(r["shard"])) for r in manifest.values()).values())
    n_shards = len({int(r["shard"]) for r in manifest.values()})
    print(f"EVAL_IDENTITY_SET={'PASS' if ids_ok else 'FAIL'} policy={args.policy} rounds=2 shards={n_shards} "
          f"per_shard={shard_sizes[0] if shard_sizes else 0}-{shard_sizes[-1] if shard_sizes else 0} "
          f"episodes={len(both)}/{len(want)} missing={len(missing)} extra={len(extra)} dup={len(overlap)}")
    bind_ok = binding_gate(args.policy, manifest, both)
    cap_ok = tier_cap_gate(args.policy, both, caps, caps_src)
    if args.specs_root:
        old = {}
        for tier in caps:
            path = Path(args.specs_root) / tier / "specs.jsonl"
            if not path.exists():
                continue
            for row in map(json.loads, [l for l in path.read_text().splitlines()[1:] if l.strip()]):
                ro = row.get("rollout") or {}
                if row.get("selected") and ro.get("status") == "ok":
                    old[(row["task"], tier, int(row["seed"]))] = ro.get("demo_frames")
        diffs = [abs(int(rec["demo_frames"]) - int(old[k])) for k, rec in both.items()
                 if rec.get("demo_frames") is not None and old.get(k) is not None]
        print(f"EVAL_DEMO_FRAMES=INFO policy={args.policy} episodes={len(diffs)} exact={sum(d == 0 for d in diffs)} "
              f"within_5={sum(d <= 5 for d in diffs)} max_diff={max(diffs, default=None)}")
    table = collections.defaultdict(collections.Counter)
    for key, rec in both.items():
        table[(key[0], key[1])][rec["status"]] += 1
    by_tier = collections.defaultdict(collections.Counter)
    for (task, tier), c in table.items():
        by_tier[tier].update(c)
    for tier in sorted(set(caps) | set(by_tier)):
        c = by_tier[tier]
        n = sum(c.values())
        print(f"EVAL_TIER_SUCCESS policy={args.policy} tier={tier} episodes={n} success={c['success']} "
              f"rate={c['success'] / n if n else 0:.3f} fail={c['fail']} timeout={c['timeout']} error={c['error']}")
    if args.table:
        Path(args.table).write_text(json.dumps({f"{t}/{tier}": dict(c) for (t, tier), c in sorted(table.items())},
                                               ensure_ascii=False, indent=1) + "\n")
    return 0 if ok and ids_ok and bind_ok and cap_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
