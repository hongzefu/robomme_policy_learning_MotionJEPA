#!/usr/bin/env python3
"""hard 档评估汇总（留档 docs/training-doc/eval-hard-patternlock-routestick/）。

与 merge_eval_shards.py 的三点不同（故不复用它，也不改它——既有留档依赖其现有行为）：
  1. split 可选：merge_eval_shards.py 的 load_test_meta 把 `meta_dir / "test"` 写死，本轮要读 val。
  2. 分片单元由 --units 显式给：merge_eval_shards.py 的 stride 布局枚举 w0..w(shards-1)，本轮每个 unit 只有单片 w3。
  3. 期望集号可指定：merge_eval_shards.py 写死 range(episodes_per_task)，本轮是 hard 档的 {3,7,…,47}。

输入（每个 --units 一个「run × 任务 × split」评估单元）：
  <eval-root>/<run_name>-<shard>/ckpt<id>/seed<seed>/{progress.json, videos/*.mp4}
  <logs-dir>/<log_prefix>-<shard>.log            （eval_shard.sh 的 `=== EVAL_SHARD … split=… ===` 与 EXIT_CODE= 行）
  <metadata-dir>/<split>/record_dataset_<task>_metadata.json   （records[].{episode,seed,difficulty}）
输出：
  summary.txt / per_episode.json → --out-dir
判定行：
  HARD_ONLY=PASS|FAIL   所有实评集的难度均为 hard（难度取自视频文件名，是 env.unwrapped.difficulty 的回读值）
  EP_SET=PASS|FAIL      每个 unit 的实评集号恰为 --expect-episodes
  SPLIT_SEED=PASS|FAIL  split 确实生效（见 verdict_split_seed）
  HARD_EVAL=DONE|INCOMPLETE units=<n> episodes=<n>/<N> … <task>-<split>=<succ>/<n> … mean_rate=<r>
用法：
  UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/summarize_hard_eval.py \
    --units evhard-pl-test:PatternLock:test --units evhard-pl-val:PatternLock:val \
    --units evhard-rs-test:RouteStick:test --units evhard-rs-val:RouteStick:val \
    --ckpt-id 79999 --seed 42 \
    --metadata-dir /data/hongzefu/robomme_policy_learning-vqa-test/third_party/robomme_benchmark/src/robomme/env_metadata \
    --out-dir docs/training-doc/eval-hard-patternlock-routestick/records
注意 --metadata-dir 必须显式给：本仓库 third_party/robomme_benchmark/ 是未 init 的空 submodule。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))
# eval.py 的视频名：<task>_ep<k>_<flag>_<task_goal>_<difficulty>.mp4；task_goal 可能含下划线，故难度取最后一段。
_VIDEO_HEAD = re.compile(r"^(?P<task>[A-Za-z]+)_ep(?P<ep>\d+)_(?P<flag>[a-z]+)_")
_HARD = "hard"


def parse_unit(spec: str) -> dict:
    """--units 的格式 <run_name>:<task>:<split>。"""
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--units 需为 <run_name>:<task>:<split>，实为 {spec!r}")
    run_name, task, split = parts
    if split not in ("train", "val", "test"):
        raise argparse.ArgumentTypeError(f"split 只能是 train/val/test，实为 {split!r}")
    return {"run_name": run_name, "task": task, "split": split}


def load_meta(meta_dir: pathlib.Path, split: str, task: str) -> dict[int, dict]:
    """读某 split 的逐集元数据 → {episode: {seed, difficulty}}（等价于 BenchmarkEnvBuilder.resolve_episode 的数据源）。"""
    p = meta_dir / split / f"record_dataset_{task}_metadata.json"
    if not p.is_file():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {int(r["episode"]): {"seed": r.get("seed"), "difficulty": r.get("difficulty")}
            for r in payload.get("records", [])}


def read_videos(d: pathlib.Path) -> dict[int, dict]:
    """从视频文件名回读逐集的 flag 与 difficulty（difficulty 是 env 实际生效值，非我们传入的期望值）。"""
    out: dict[int, dict] = {}
    vd = d / "videos"
    if not vd.is_dir():
        return out
    for mp4 in sorted(vd.glob("*.mp4")):
        m = _VIDEO_HEAD.match(mp4.name)
        if not m:
            continue
        stem = mp4.name[: -len(".mp4")]
        out[int(m.group("ep"))] = {"flag": m.group("flag"), "difficulty": stem.rsplit("_", 1)[-1], "name": mp4.name}
    return out


def read_log(log: pathlib.Path) -> dict:
    """eval_shard.sh 的自证行与收尾行：split=（本轮实际传入的 split）、起止时刻、EXIT_CODE。"""
    info: dict = {"log": str(log), "exists": log.is_file(), "split": None, "start": None, "end": None, "exit_code": None}
    if not log.is_file():
        return info
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"=== EVAL_SHARD .* split=(?P<split>\w+) .* start=(?P<start>[\d-]+ [\d:]+) ===", line)
        if m:
            info["split"], info["start"] = m.group("split"), m.group("start")
        m = re.search(r"EVAL_RC=(\d+) end=([\d-]+ [\d:]+)", line)
        if m:
            info["end"] = m.group(2)
        m = re.match(r"EXIT_CODE=(\d+)", line)
        if m:
            info["exit_code"] = int(m.group(1))
    return info


def verdict_hard_only(units: list[dict]) -> tuple[bool, str]:
    """HARD_ONLY：所有实评集的难度必须是 hard。难度取自视频名（env 回读值），故本判据也实测覆盖了
    RouteStick.py 难度 fallback 里那行无条件 `self.difficulty = "easy"` 的硬覆盖。"""
    bad, n_checked, n_missing = [], 0, 0
    for u in units:
        for ep in sorted(u["episodes"]):
            v = u["videos"].get(ep)
            if v is None:
                n_missing += 1
                continue
            n_checked += 1
            if v["difficulty"] != _HARD:
                bad.append(f"{u['task']}-{u['split']}-ep{ep}={v['difficulty']}")
    ok = not bad and n_missing == 0 and n_checked > 0
    detail = f"checked={n_checked} non_hard={len(bad)} no_video={n_missing}"
    if bad:
        detail += " 非 hard: " + ",".join(bad[:10])
    return ok, detail


def verdict_ep_set(units: list[dict], expect: list[int]) -> tuple[bool, str]:
    """EP_SET：每个 unit 的实评集号必须恰为期望集合（hard 档 12 集）。"""
    bad = []
    for u in units:
        got = sorted(u["episodes"])
        if got != expect:
            miss = sorted(set(expect) - set(got))
            extra = sorted(set(got) - set(expect))
            bad.append(f"{u['task']}-{u['split']}(n={len(got)} missing={miss} extra={extra})")
    ok = not bad
    return ok, (f"units={len(units)} expect={len(expect)}集" + ("" if ok else " 不符: " + "; ".join(bad)))


def verdict_split_seed(units: list[dict]) -> tuple[bool, str]:
    """SPLIT_SEED：证明 --args.dataset_split 真的生效了——本轮最关键的判据。

    为什么需要它：若 split 参数没接通，val 会静默跑成 test 的那批环境，成功率数字看起来完全正常，
    但两组其实是同一批环境实例，「val vs test」的对照就是假的。这是最危险的静默失败模式。

    三层可用证据（已由调用方备妥，都在 units 的字段里）：
      1. u["log"]["split"]      —— eval_shard.sh 自证行 `=== EVAL_SHARD … split=… ===` 回读到的实际传入值，
                                   应等于 u["split"]。这证明「我们传对了」。
      2. u["meta"]              —— 该 split 元数据的 {episode: {seed, difficulty}}。同一 task 的 val 与 test
                                   在相同 episode 号上 seed 必须不同（实读基址：PatternLock test 650000 / val 1150000，
                                   RouteStick test 660000 / val 1160000，均为 BASE+100·ep）。这证明「两个 split 确实是两批环境」。
      3. u["videos"][ep]["name"] —— 视频文件名里含 task_goal 段。同一 task 同一 episode 号下，
                                   val 与 test 若真是不同环境实例，产物名不应完全相同。这是产物侧的直接证据（较弱，
                                   某些任务的 goal 可能是固定模板而与实例无关）。

    实现：第 1、2 层当硬判据，第 3 层只作观察项写进 detail（某些任务的 task_goal 是与实例无关的
    固定模板，此时 val/test 产物名天然相同，当硬判据会误报）。只传了一个 split 时第 2 层无从比对，
    标为「不适用」而非 FAIL。
    """
    # ── 第 1 层：日志自证行回读的 split 必须等于该 unit 声明的 split ──
    log_bad, log_missing = [], []
    for u in units:
        got = u["log"]["split"]
        if got is None:
            log_missing.append(f"{u['task']}-{u['split']}")
        elif got != u["split"]:
            log_bad.append(f"{u['task']}-{u['split']}(日志里是 {got})")

    # ── 第 2 层：同一 task 下，不同 split 在相同 episode 号上的 seed 必须两两不同 ──
    by_task: dict[str, list[dict]] = {}
    for u in units:
        by_task.setdefault(u["task"], []).append(u)
    seed_bad, n_pairs, n_cmp = [], 0, 0
    for task, us in by_task.items():
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a, b = us[i], us[j]
                if a["split"] == b["split"]:
                    continue
                n_pairs += 1
                shared = sorted(set(a["meta"]) & set(b["meta"]))
                n_cmp += len(shared)
                same = [e for e in shared if a["meta"][e].get("seed") == b["meta"][e].get("seed")]
                if not shared:
                    seed_bad.append(f"{task} {a['split']}vs{b['split']} 元数据为空、无法比对")
                elif same:
                    seed_bad.append(f"{task} {a['split']}vs{b['split']} 有 {len(same)}/{len(shared)} 集 seed 相同")

    # ── 第 3 层（观察项）：同 task 同 episode 下，不同 split 的视频名是否完全相同 ──
    dup_names = 0
    for task, us in by_task.items():
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a, b = us[i], us[j]
                if a["split"] == b["split"]:
                    continue
                for ep in set(a["videos"]) & set(b["videos"]):
                    if a["videos"][ep]["name"] == b["videos"][ep]["name"]:
                        dup_names += 1

    ok = not log_bad and not log_missing and not seed_bad
    if n_pairs == 0:
        detail = f"只有单个 split、跨 split seed 比对不适用；日志自证 {len(units) - len(log_missing)}/{len(units)} 个 unit 相符"
    else:
        seed_word = "全不同" if not seed_bad else "有撞车"
        detail = (f"日志自证 {len(units) - len(log_missing) - len(log_bad)}/{len(units)} 相符; "
                  f"跨 split seed 比对 {n_pairs} 对/{n_cmp} 集{seed_word}; 同名视频 {dup_names} 个(观察项)")
    if log_missing:
        detail += " | 日志缺 split 字段: " + ",".join(log_missing)
    if log_bad:
        detail += " | 日志 split 不符: " + ",".join(log_bad)
    if seed_bad:
        detail += " | seed 撞车: " + "; ".join(seed_bad)
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", action="append", required=True, type=parse_unit,
                    help="<run_name>:<task>:<split>，可重复")
    ap.add_argument("--shard", default="w3", help="分片名（本轮 hard 档恒为 w3：EP_START=3 EP_STRIDE=4）")
    ap.add_argument("--ckpt-id", type=int, default=79999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-prefix", default="", help="留空则用各 unit 的 run_name")
    ap.add_argument("--expect-episodes", default="3,7,11,15,19,23,27,31,35,39,43,47")
    ap.add_argument("--eval-root", default=str(_V1 / "evaluation"))
    ap.add_argument("--logs-dir", default=str(_V1 / "logs"))
    ap.add_argument("--metadata-dir", required=True,
                    help="必须显式给：本仓库 third_party/robomme_benchmark/ 是未 init 的空 submodule")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="中途看进度：不因不完整而返回非零")
    args = ap.parse_args()

    expect = [int(x) for x in args.expect_episodes.split(",")]
    eval_root, logs_dir, meta_dir = pathlib.Path(args.eval_root), pathlib.Path(args.logs_dir), pathlib.Path(args.metadata_dir)

    units: list[dict] = []
    for u in args.units:
        prefix = args.log_prefix or u["run_name"]
        d = eval_root / f"{u['run_name']}-{args.shard}" / f"ckpt{args.ckpt_id}" / f"seed{args.seed}"
        pj = d / "progress.json"
        episodes: dict[int, object] = {}
        if pj.is_file():
            for _task, eps in json.loads(pj.read_text(encoding="utf-8")).items():
                for e, v in eps.items():
                    episodes[int(e)] = v
        u.update({
            "dir": str(d),
            "episodes": episodes,
            "finished": (d / "log.json").is_file(),
            "videos": read_videos(d),
            "log": read_log(logs_dir / f"{prefix}-{args.shard}.log"),
            "meta": load_meta(meta_dir, u["split"], u["task"]),
        })
        units.append(u)

    ok_hard, d_hard = verdict_hard_only(units)
    ok_eps, d_eps = verdict_ep_set(units, expect)
    ok_split, d_split = verdict_split_seed(units)

    per_episode: list[dict] = []
    lines: list[str] = []
    n_total = n_succ = 0
    rates: list[float] = []
    for u in units:
        eps = u["episodes"]
        succ = sum(1 for v in eps.values() if v is True)
        n_total += len(eps)
        n_succ += succ
        if eps:
            rates.append(succ / len(eps))
        for ep in sorted(eps):
            v = eps[ep]
            vid = u["videos"].get(ep, {})
            meta = u["meta"].get(ep, {})
            per_episode.append({"task": u["task"], "split": u["split"], "episode": ep,
                                "seed": meta.get("seed"), "difficulty_meta": meta.get("difficulty"),
                                "difficulty_env": vid.get("difficulty"), "success": v, "flag": vid.get("flag"),
                                "run_name": u["run_name"], "video": vid.get("name")})
        rate = (succ / len(eps)) if eps else 0.0
        lines.append(f"{u['task']}-{u['split']:4s} | {succ}/{len(eps)} | {rate:.2%} | "
                     f"finished={u['finished']} exit={u['log']['exit_code']} log_split={u['log']['split']}")

    complete = all(u["finished"] for u in units) and ok_eps
    mean_rate = (sum(rates) / len(rates)) if rates else 0.0
    n_expect = len(expect) * len(units)
    verdicts = [f"HARD_ONLY={'PASS' if ok_hard else 'FAIL'} {d_hard}",
                f"EP_SET={'PASS' if ok_eps else 'FAIL'} {d_eps}",
                f"SPLIT_SEED={'PASS' if ok_split else 'FAIL'} {d_split}"]
    head = (f"HARD_EVAL={'DONE' if complete else 'INCOMPLETE'} units={len(units)} "
            f"episodes={n_total}/{n_expect} "
            + " ".join(f"{u['task']}-{u['split']}={sum(1 for v in u['episodes'].values() if v is True)}/{len(u['episodes'])}"
                       for u in units)
            + f" mean_rate={mean_rate:.4f}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = "\n".join([head, *verdicts, "", "任务-split | 成功/实评 | 成功率 | 分片状态", *lines, ""])
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (out_dir / "per_episode.json").write_text(json.dumps(per_episode, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)
    all_ok = complete and ok_hard and ok_eps and ok_split
    return 0 if (all_ok or args.allow_partial) else 1


if __name__ == "__main__":
    sys.exit(main())
