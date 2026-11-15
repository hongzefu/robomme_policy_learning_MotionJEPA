"""合并 10 个分片的结果，出 14 组成功率与两个口径的总成功率，并生成补跑计划。

两个总成功率都给，因为它们回答不同的问题：
* **宏平均**＝14 个「任务/难度」组成功率的均值。每组权重相同，与历史 log.json 的口径同族
  （那边是按任务平均）。组的样本数本来就都是 50，所以这里宏/微的差别只来自 error 条目。
* **微平均**＝成功数 / 计划总数。每条 episode 权重相同。

error 条目（建环境失败、step 期异常、单集墙钟超时）不计入成功数，但**计入微平均的分母**，
所以微平均是成功率的下界。补跑后重算会把它们填上。
"""
import argparse
import collections
import json
from pathlib import Path


def load_shards(run_root: Path, policy: str, ckpt: str, seed: str):
    shards = {}
    for shard_dir in sorted(p for p in run_root.iterdir() if p.is_dir() and p.name != "merged"):
        progress = shard_dir / policy / f"ckpt{ckpt}" / f"seed{seed}" / "progress.json"
        plan = shard_dir / policy / f"ckpt{ckpt}" / f"seed{seed}" / "plan.json"
        if not progress.is_file():
            print(f"[warn] 跳过没有 progress.json 的目录：{shard_dir.name}")
            continue
        motion = shard_dir / policy / f"ckpt{ckpt}" / f"seed{seed}" / "motion_stats.json"
        shards[shard_dir.name] = {
            "progress": json.loads(progress.read_text(encoding="utf-8")),
            "plan": json.loads(plan.read_text(encoding="utf-8")) if plan.is_file() else None,
            "motion": json.loads(motion.read_text(encoding="utf-8")) if motion.is_file() else None,
        }
    if not shards:
        raise SystemExit(f"{run_root} 下没有找到任何分片结果")
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="v1-store/evaluation/<run_name>")
    parser.add_argument("--policy", default="perceptual-framesamp-modul-8frame-8x8")
    parser.add_argument("--ckpt", default="59999")
    parser.add_argument("--seed", default="7")
    parser.add_argument("--expect-total", type=int, default=700)
    parser.add_argument("--candidates", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    shards = load_shards(run_root, args.policy, args.ckpt, args.seed)

    merged = collections.defaultdict(dict)
    owner = {}
    for name, payload in shards.items():
        for group, entries in payload["progress"].items():
            for episode, value in entries.items():
                key = (group, int(episode))
                if key in owner:
                    raise SystemExit(f"分片 {name} 与 {owner[key]} 都评了 {group} ep{episode}")
                owner[key] = name
                merged[group][episode] = value

    total = len(owner)
    if total != args.expect_total:
        raise SystemExit(f"合并后共 {total} 条，期望 {args.expect_total}")

    group_stats, errors = {}, []
    for group, entries in sorted(merged.items()):
        ok = sum(v is True for v in entries.values())
        err = [int(e) for e, v in entries.items() if v == "error"]
        errors += [{"group": group, "episode": e} for e in sorted(err)]
        evaluated = len(entries) - len(err)
        group_stats[group] = {
            "episodes": len(entries),
            "evaluated": evaluated,
            "successes": ok,
            "errors": len(err),
            # 组成功率的分母用计划集数，error 算失败；补跑后重算即修正
            "success_rate": ok / len(entries) if entries else 0.0,
        }

    rates = [s["success_rate"] for s in group_stats.values()]
    successes = sum(s["successes"] for s in group_stats.values())
    macro = sum(rates) / len(rates) if rates else 0.0
    micro = successes / total if total else 0.0

    by_task, by_difficulty = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    for group, stat in group_stats.items():
        task, difficulty = group.split("/")
        by_task[task][0] += stat["successes"]; by_task[task][1] += stat["episodes"]
        by_difficulty[difficulty][0] += stat["successes"]; by_difficulty[difficulty][1] += stat["episodes"]

    merged_dir = run_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "progress.json").write_text(
        json.dumps({g: dict(sorted(e.items(), key=lambda kv: int(kv[0]))) for g, e in sorted(merged.items())},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    log = {
        "success_rate": {g: s["success_rate"] for g, s in group_stats.items()},
        "macro_success_rate": macro,
        "micro_success_rate": micro,
        "episodes": total,
        "successes": successes,
        "errors": len(errors),
        "by_task": {t: {"successes": v[0], "episodes": v[1], "success_rate": v[0] / v[1]}
                    for t, v in sorted(by_task.items())},
        "by_difficulty": {d: {"successes": v[0], "episodes": v[1], "success_rate": v[0] / v[1]}
                          for d, v in sorted(by_difficulty.items())},
        "groups": group_stats,
    }
    (merged_dir / "log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    (merged_dir / "shards.json").write_text(json.dumps(
        {name: {"episodes": sum(len(e) for e in p["progress"].values()),
                "shard_tag": (p["plan"] or {}).get("shard_tag", ""),
                "identity_sha256": (p["plan"] or {}).get("identity_sha256", "")}
         for name, p in sorted(shards.items())}, indent=2, ensure_ascii=False), encoding="utf-8")
    (merged_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    # motion 窗统计合并（0922-binfill-demo-prefix-plan.md C 节）：只有 motion run 的分片才有这份文件
    motion_merged = {}
    for name, payload in shards.items():
        for key, stat in (payload["motion"] or {}).items():
            if key in motion_merged:
                raise SystemExit(f"分片 {name} 与别的分片都记了 {key} 的 motion 统计")
            motion_merged[key] = stat
    if motion_merged:
        (merged_dir / "motion_stats.json").write_text(
            json.dumps(dict(sorted(motion_merged.items())), indent=2, ensure_ascii=False), encoding="utf-8")
        downsampled = [k for k, s in motion_merged.items() if int(s["motion_downsample_steps"]) > 0]
        k_max = max(int(s["motion_k_max"]) for s in motion_merged.values())
        es_max = max(int(s["exec_start_idx"]) for s in motion_merged.values())
        log["motion"] = {
            "episodes": len(motion_merged),
            "budget": sorted({int(s["motion_budget"]) for s in motion_merged.values()}),
            "overflow": sorted({str(s["motion_overflow"]) for s in motion_merged.values()}),
            "k_max": k_max, "exec_start_idx_max": es_max,
            "episodes_downsampled": len(downsampled),
            "downsample_steps_total": sum(int(s["motion_downsample_steps"]) for s in motion_merged.values()),
        }
        # 分层成功率**只作描述、不作因果验收**：触发降级需要该集活过 16*(162-k_demo) 步，
        # 早早成功的集天然进不了触发组——两组本来就不可比（选择偏倚，不是处理效应）。
        merged_flat = {f"{g}/{e}": v for g, entries in merged.items() for e, v in entries.items()}
        for label, keys in (("downsampled", downsampled),
                            ("not_downsampled", [k for k in motion_merged if k not in set(downsampled)])):
            values = [merged_flat.get(k) for k in keys]
            done = [v for v in values if v is not True and v is not False]
            n_ok = sum(1 for v in values if v is True)
            log["motion"][f"{label}_n"] = len(keys)
            log["motion"][f"{label}_successes"] = n_ok
            log["motion"][f"{label}_errors"] = len(done)
        (merged_dir / "log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  MOTION_MERGED episodes={len(motion_merged)} k_max={k_max} es_max={es_max} "
              f"downsampled_episodes={len(downsampled)} "
              f"（分层成功率 {log['motion']['downsampled_successes']}/{log['motion']['downsampled_n']} vs "
              f"{log['motion']['not_downsampled_successes']}/{log['motion']['not_downsampled_n']}，"
              f"仅作描述：两组存在选择偏倚，不可当因果证据）")

    if errors:
        groups = collections.defaultdict(list)
        for item in errors:
            groups[item["group"]].append(item["episode"])
        candidates = args.candidates
        if not candidates:
            for payload in shards.values():
                if payload["plan"] and payload["plan"].get("candidates"):
                    candidates = payload["plan"]["candidates"]
                    break
        retry = {"shard_tag": "retry", "candidates": candidates,
                 "groups": {g: sorted(v) for g, v in sorted(groups.items())}}
        (merged_dir / "retry_plan.json").write_text(
            json.dumps(retry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"MERGE_OK shards={len(shards)} episodes={total} successes={successes} errors={len(errors)}")
    print(f"  macro_success_rate={macro:.4f}  micro_success_rate={micro:.4f}")
    for group, stat in group_stats.items():
        print(f"  {group:26s} {stat['successes']:3d}/{stat['episodes']:3d} = {stat['success_rate']:.3f}"
              f"{'  errors=' + str(stat['errors']) if stat['errors'] else ''}")
    if errors:
        print(f"  补跑计划已写入 {merged_dir / 'retry_plan.json'}（{len(errors)} 条）")


if __name__ == "__main__":
    main()
