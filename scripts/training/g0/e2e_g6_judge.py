"""G6 测速与内存闸的分阶段判定与聚合（1117-e2e-enc-train-plan.html 第二部分一节 e2e_g6_judge.py 行、三节 G6、R10、RB-D1）。

两个子命令：
  stage      由 run_4task_e2eenc_50k.sh 在 --bench300 / --shmprobe / --ckptprobe 各自训练结束后调用，读该次的 runner 日志、
             记录目录（step_timing.jsonl、metrics.jsonl、gpu_util_15s_full.csv）、shmmon 日志，落一份证据
             <evidence-dir>/g6-<mode>-<attempt>.json（绑定 attempt id 与路径配置摘要 PATHCFG_SHA；同名已存在即拒），
             打印该阶段判定行与 G6_STAGE=<状态>。
  aggregate  只取路径配置摘要与当前一致的各模式最新通过尝试，聚合打印 G6 各行：
             BENCH300、GPU_UTIL、SHM_PEAK、GPU_PEAK、SMOKE_CKPT，末行 G6=<状态>。

状态词只用 R10 五个：PASS / RECORDED / ASK / ACCEPTED_OVER_LIMIT / FAIL。阈值（三节 G6，起跑前写死，R4 不事后放宽）：
  BENCH300：去掉前 skip 步（编译）的稳态 s/步均值 x；x ≤ 2.27 PASS；2.27 < x ≤ 3.0 用户预先接受，
            `BENCH300=ACCEPTED_OVER_LIMIT measured=<x> limit=2.27 accepted_by=user_preauth_3.0s_42h at=<美东>`；x > 3.0 → ASK。
  GPU_UTIL：bench 与 shmprobe 期间 8 卡 utilization.gpu 15 s 采样均值（去掉编译步窗口）≥ 85 → PASS，否则 RECORDED（不阻断）。
  SHM_PEAK：shmprobe 全程 /dev/shm 峰值 ≤ 330 GB 且最后 1500 步峰值相对此前峰值增长 ≤ 2% → PASS。
  GPU_PEAK：每卡 peak_bytes_in_use（GiB）≤ 72 → PASS；证据为日志里训练进程打印的
            `DEVICE_PEAK device=<i> peak_bytes_in_use=<n>` 行，三个阶段都没有该行 → FAIL（无证据不放行）。
  SMOKE_CKPT：ckptprobe 的 50/params 读回（runner 打印 CKPT50_READBACK：字节数 > 0、叶数 = Wan-9 叶数 + encoder 叶数、有限），
            且 step_timing 证明 step 50 之后又完成 ≥ 20 次更新；不得用最终 checkpoint 代替。
  各阶段另核 DataLoader 实际 prefetch_factor（日志 DATALOADER_PREFETCH_FACTOR=）等于本次配置值，不符即该阶段 FAIL。
只用标准库，纯 CPU；单测用假日志夹具（scripts/training/tests/test_e2e_runner_chain.py）。
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import pathlib
import re
import sys
import time

BENCH_LIMIT = 2.27
BENCH_PREAUTH_MAX = 3.0
UTIL_LIMIT = 85.0
SHM_LIMIT_GB = 330.0
SHM_GROWTH_LIMIT_PCT = 2.0
SHM_TAIL_STEPS = 1500
GPU_PEAK_LIMIT_GB = 72.0
CKPT_STEP = 50
CKPT_MIN_UPDATES_AFTER = 20
OK_STATES = ("PASS", "ACCEPTED_OVER_LIMIT", "RECORDED")
MODES = ("bench", "shmprobe", "ckptprobe")
ATTEMPT_RE = re.compile(r"^a[0-9]{2}$")


def now_et() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%dT%H:%M:%S%z")
    except Exception:  # noqa: BLE001  无 tzdata 时退化为 UTC 并写明
        return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ(UTC)")


def worst(states) -> str:
    states = list(states)
    if "FAIL" in states:
        return "FAIL"
    if "ASK" in states:
        return "ASK"
    return "PASS"


def bench_verdict(x: float | None, complete: bool, at: str) -> tuple[str, str]:
    """BENCH300 判定行（R10）：≤2.27 PASS；(2.27, 3.0] 用户预先接受；>3.0 ASK；数据不全 FAIL。"""
    if x is None or not complete or x != x:
        return "FAIL", f"BENCH300=FAIL reason=incomplete_or_missing s_per_step={x} limit={BENCH_LIMIT}"
    if x <= BENCH_LIMIT:
        return "PASS", f"BENCH300=PASS s_per_step={x:.4f} limit={BENCH_LIMIT}"
    if x <= BENCH_PREAUTH_MAX:
        return ("ACCEPTED_OVER_LIMIT",
                f"BENCH300=ACCEPTED_OVER_LIMIT measured={x:.4f} limit={BENCH_LIMIT} accepted_by=user_preauth_3.0s_42h at={at}")
    return "ASK", f"BENCH300=ASK measured={x:.4f} limit={BENCH_LIMIT} preauth_max={BENCH_PREAUTH_MAX}"


def util_verdict(mean: float | None) -> tuple[str, str]:
    if mean is None:
        return "FAIL", f"GPU_UTIL=FAIL reason=no_samples limit={UTIL_LIMIT:g}"
    st = "PASS" if mean >= UTIL_LIMIT else "RECORDED"
    return st, f"GPU_UTIL={st} util_mean={mean:.2f} limit={UTIL_LIMIT:g}"


def shm_verdict(peak: float | None, growth: float | None) -> tuple[str, str]:
    if peak is None or growth is None:
        return "FAIL", f"SHM_PEAK=FAIL reason=no_evidence peak_gb={peak} limit={SHM_LIMIT_GB:g} growth_last{SHM_TAIL_STEPS}={growth}"
    st = "PASS" if peak <= SHM_LIMIT_GB and growth <= SHM_GROWTH_LIMIT_PCT else "FAIL"
    return st, f"SHM_PEAK={st} peak_gb={peak:.1f} limit={SHM_LIMIT_GB:g} growth_last{SHM_TAIL_STEPS}={growth:.2f}%"


def gpu_peak_verdict(peak_gb: float | None) -> tuple[str, str]:
    if peak_gb is None:
        return "FAIL", f"GPU_PEAK=FAIL reason=no_peak_bytes_evidence limit={GPU_PEAK_LIMIT_GB:g}"
    st = "PASS" if peak_gb <= GPU_PEAK_LIMIT_GB else "FAIL"
    return st, f"GPU_PEAK={st} peak_gb={peak_gb:.2f} limit={GPU_PEAK_LIMIT_GB:g}"


# ─────────────────────────── 读证据 ───────────────────────────


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def log_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(errors="replace").splitlines() if path.is_file() else []


def last_value(lines: list[str], pat: str):
    rx = re.compile(pat)
    hit = None
    for ln in lines:
        m = rx.search(ln)
        if m:
            hit = m
    return hit


def device_peaks(lines: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for ln in lines:
        m = re.match(r"^DEVICE_PEAK device=(\d+) peak_bytes_in_use=(\d+)", ln)
        if m:
            d, b = int(m.group(1)), int(m.group(2))
            out[d] = max(out.get(d, 0), b)
    return out


def gpu_util_samples(csv_path: pathlib.Path, t0: float, t1: float) -> list[float]:
    vals = []
    if not csv_path.is_file():
        return vals
    with csv_path.open() as f:
        for row in csv.reader(f):
            try:
                ts = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()   # nvidia-smi 本地时间
                u = float(row[2])
            except (ValueError, IndexError):
                continue
            if t0 <= ts <= t1:
                vals.append(u)
    return vals


def shm_samples(path: pathlib.Path) -> list[tuple[float, float]]:
    out = []
    for ln in log_lines(path):
        m = re.match(r"^SHM_SAMPLE .*shm_used_gb=([0-9.]+).*epoch=([0-9.]+)", ln)
        if m:
            out.append((float(m.group(2)), float(m.group(1))))
    return out


def stage(args) -> int:
    if not ATTEMPT_RE.match(args.attempt):
        print(f"拒绝：attempt 非法 {args.attempt}", file=sys.stderr)
        return 2
    ev_dir = pathlib.Path(args.evidence_dir)
    ev_dir.mkdir(parents=True, exist_ok=True)
    ev_path = ev_dir / f"g6-{args.mode}-{args.attempt}.json"
    if ev_path.exists():
        print(f"拒绝覆盖已存在的 G6 证据 {ev_path}（重测请递增尝试编号）", file=sys.stderr)
        return 2
    at = args.now_et or now_et()
    run_log = pathlib.Path(args.run_log)
    rec = pathlib.Path(args.rec)
    lines = log_lines(run_log)
    states, out_lines, reasons = [], [], []

    # DataLoader 实际 prefetch_factor 读回（R8 (i)；不符即 FAIL）
    m = last_value(lines, r"^DATALOADER_PREFETCH_FACTOR=(\S+)")
    pf = m.group(1) if m else None
    pf_ok = pf is not None and pf == str(args.prefetch_expect)
    if not pf_ok:
        reasons.append(f"prefetch_factor={pf} expect={args.prefetch_expect}")
    states.append("PASS" if pf_ok else "FAIL")

    timing = sorted(read_jsonl(rec / "step_timing.jsonl"), key=lambda r: r["step"])
    done_m = last_value(lines, r"^TRAIN_DONE_EPOCH=(\d+)")
    start_m = last_value(lines, r"^TRAIN_START_EPOCH=(\d+)")
    win = timing[args.skip:]
    t0 = win[0]["wall_start"] if win else None
    t1 = float(done_m.group(1)) if done_m else (win[-1]["wall_end"] if win else None)
    util = gpu_util_samples(rec / "gpu_util_15s_full.csv", t0, t1) if t0 is not None and t1 is not None else []
    peaks = device_peaks(lines)
    peak_gb = max(peaks.values()) / 1024 ** 3 if peaks else None
    ev = {"mode": args.mode, "attempt": args.attempt, "run": args.run, "pathcfg": args.pathcfg,
          "pathcfg_sha": args.pathcfg_sha, "prefetch_factor": pf, "prefetch_expect": str(args.prefetch_expect),
          "run_log": str(run_log), "run_log_sha256": hashlib.sha256(run_log.read_bytes()).hexdigest() if run_log.is_file() else None,
          "rec": str(rec), "util_sum": sum(util), "util_n": len(util), "util_window": [t0, t1],
          "device_peak_bytes": {str(k): v for k, v in sorted(peaks.items())}, "gpu_peak_gb": peak_gb, "written_et": at}

    if args.mode == "bench":
        want = min(args.steps, len(timing)) if timing else args.steps
        complete = bool(timing) and len(timing) == args.steps and all(r.get("completed") for r in timing)
        s = [r["host_step_s"] - r.get("phases_s", {}).get("checkpoint", 0.0) for r in win]
        dl = [r.get("phases_s", {}).get("data_next", 0.0) for r in win]
        x = sum(s) / len(s) if s else None
        st, line = bench_verdict(x, complete, at)
        dl_frac = sum(dl) / sum(s) if s and sum(s) > 0 else float("nan")
        line += f" dl_wait_frac={dl_frac:.4f} steps={len(timing)}/{want} window_steps={len(win)} skip={args.skip}"
        ev.update(bench_status=st, s_per_step=x, dl_wait_frac=dl_frac, complete=complete, bench_line=line)
        states.append(st)
        out_lines.append(line)
    elif args.mode == "shmprobe":
        samples = shm_samples(pathlib.Path(args.shmmon_log)) if args.shmmon_log else []
        metrics = sorted((r for r in read_jsonl(rec / "metrics.jsonl") if r.get("step") is not None), key=lambda r: r["step"])
        split_step = args.steps - SHM_TAIL_STEPS
        split = next((r["wall_time"] for r in metrics if r["step"] >= split_step), None)
        t_start = float(start_m.group(1)) if start_m else (samples[0][0] if samples else None)
        t_end = t1 if t1 is not None else (samples[-1][0] if samples else None)
        run_s = [(t, g) for t, g in samples if t_start is None or t_end is None or t_start <= t <= t_end]
        peak = max((g for _, g in run_s), default=None)
        before = [g for t, g in run_s if split is not None and t < split]
        after = [g for t, g in run_s if split is not None and t >= split]
        growth = None
        if before and after and max(before) > 0:
            growth = max(0.0, (max(after) - max(before)) / max(before) * 100.0)
        if split is None:
            reasons.append(f"metrics.jsonl 无 step>={split_step} 的行，无法定位最后 {SHM_TAIL_STEPS} 步")
        if not samples:
            reasons.append(f"shmmon 日志无 SHM_SAMPLE 行: {args.shmmon_log}")
        st, line = shm_verdict(peak, growth)
        alerts = sum(1 for ln in log_lines(pathlib.Path(args.shmmon_log)) if ln.startswith("SHM_ALERT")) if args.shmmon_log else 0
        line += f" samples={len(run_s)} split_step={split_step} alerts={alerts}"
        ev.update(shm_status=st, shm_peak_gb=peak, shm_growth_pct=growth, shm_line=line, shmmon_log=args.shmmon_log)
        states.append(st)
        out_lines.append(line)
    else:  # ckptprobe
        rb = last_value(lines, r"^CKPT50_READBACK step=50 .*bytes=(\d+) leaves=(\d+) enc_leaves=(\d+) wan9_leaves=(\d+) finite=(\d)")
        after = sum(1 for r in timing if r["step"] > CKPT_STEP and r.get("completed"))
        p50 = pathlib.Path(args.run_root) / str(CKPT_STEP) / "params" if args.run_root else None
        ok, why = True, []
        if rb is None:
            ok = False; why.append("缺 CKPT50_READBACK 行")
        else:
            nbytes, leaves, enc, wan9, finite = (int(rb.group(i)) for i in range(1, 6))
            if nbytes <= 0: ok = False; why.append("50/params 字节数为 0")
            if enc <= 0 or leaves != wan9 + enc: ok = False; why.append(f"叶数 {leaves} != {wan9}+{enc}")
            if args.wan9_leaves is not None and wan9 != args.wan9_leaves: ok = False; why.append(f"wan9_leaves={wan9}")
            if finite != 1: ok = False; why.append("非有限值")
        if p50 is not None and not p50.is_dir():
            ok = False; why.append(f"缺 {p50}")
        if after < CKPT_MIN_UPDATES_AFTER:
            ok = False; why.append(f"step {CKPT_STEP} 之后完成更新 {after} < {CKPT_MIN_UPDATES_AFTER}")
        st = "PASS" if ok else "FAIL"
        line = (f"SMOKE_CKPT={st} step={CKPT_STEP} updates_after={after} "
                + (f"bytes={rb.group(1)} leaves={rb.group(2)} enc_leaves={rb.group(3)}" if rb else "readback=missing")
                + ("" if ok else f" reasons={why}"))
        ev.update(ckpt_status=st, ckpt_line=line, updates_after=after)
        states.append(st)
        out_lines.append(line)

    gst, gline = gpu_peak_verdict(peak_gb)
    out_lines.append(gline.replace("GPU_PEAK=", "GPU_PEAK_STAGE=", 1))
    stage_status = worst(states)
    ev.update(stage_status=stage_status, reasons=reasons, lines=out_lines)
    ev_path.write_text(json.dumps(ev, indent=1, ensure_ascii=False) + "\n")
    for ln in out_lines:
        print(ln, flush=True)
    util_mean = (sum(util) / len(util)) if util else float("nan")
    print(f"G6_STAGE={stage_status} mode={args.mode} attempt={args.attempt} pathcfg_sha={args.pathcfg_sha} "
          f"prefetch={pf}/{args.prefetch_expect} util_mean={util_mean:.2f} util_n={len(util)} evidence={ev_path}"
          + (f" reasons={reasons}" if reasons else ""), flush=True)
    return 1 if stage_status == "FAIL" else 0


def pick(evs: list[dict], mode: str, pathcfg_sha: str) -> dict | None:
    """该模式下路径配置摘要相同的尝试里取最新的通过尝试；没有通过的则取最新一次（用于如实报告其状态）。"""
    cands = sorted((e for e in evs if e.get("mode") == mode and e.get("pathcfg_sha") == pathcfg_sha),
                   key=lambda e: e["attempt"])
    ok = [e for e in cands if e.get("stage_status") in ("PASS",)]
    return ok[-1] if ok else (cands[-1] if cands else None)


def with_stage(e: dict, key: str, st: str, line: str) -> tuple[str, str]:
    """分项状态与该尝试的阶段状态取最坏：阶段 FAIL（如 prefetch 读回不符等非分项原因）时分项一律报 FAIL，
    阶段 ASK 时分项至少 ASK；不让回退选中的未通过尝试的阶段失败被分项 PASS 吞掉。"""
    stage_st = e.get("stage_status", "FAIL")
    w = st
    if stage_st == "FAIL" and st != "FAIL":
        w = "FAIL"
    elif stage_st == "ASK" and st not in ("FAIL", "ASK"):
        w = "ASK"
    if w == st:
        return st, line
    rest = line[len(f"{key}={st}"):] if line.startswith(f"{key}={st}") else " " + line
    return w, f"{key}={w}{rest} stage_status={stage_st} stage_reasons={e.get('reasons')}"


def aggregate(args) -> int:
    ev_dir = pathlib.Path(args.evidence_dir)
    evs = []
    for p in sorted(ev_dir.glob("g6-*-a[0-9][0-9].json")):
        try:
            evs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    chosen = {m: pick(evs, m, args.pathcfg_sha) for m in MODES}
    states, lines = [], []
    b = chosen["bench"]
    if b is None:
        st, line = "FAIL", f"BENCH300=FAIL reason=no_bench_evidence pathcfg_sha={args.pathcfg_sha}"
    else:
        st, line = with_stage(b, "BENCH300", b["bench_status"], b["bench_line"] + f" attempt={b['attempt']}")
    states.append(st); lines.append(line)
    n = sum(chosen[m]["util_n"] for m in ("bench", "shmprobe") if chosen[m])
    tot = sum(chosen[m]["util_sum"] for m in ("bench", "shmprobe") if chosen[m])
    st, line = util_verdict(tot / n if n else None)
    line += f" samples={n} stages={'+'.join(m for m in ('bench', 'shmprobe') if chosen[m]) or 'none'}"
    states.append("PASS" if st == "RECORDED" else st); lines.append(line)
    s = chosen["shmprobe"]
    if s is None:
        st, line = "FAIL", f"SHM_PEAK=FAIL reason=no_shmprobe_evidence pathcfg_sha={args.pathcfg_sha}"
    else:
        st, line = with_stage(s, "SHM_PEAK", s["shm_status"], s["shm_line"] + f" attempt={s['attempt']}")
    states.append(st); lines.append(line)
    peaks = [chosen[m]["gpu_peak_gb"] for m in MODES if chosen[m] and chosen[m].get("gpu_peak_gb") is not None]
    st, line = gpu_peak_verdict(max(peaks) if peaks else None)
    states.append(st); lines.append(line)
    c = chosen["ckptprobe"]
    if c is None:
        st, line = "FAIL", f"SMOKE_CKPT=FAIL reason=no_ckptprobe_evidence pathcfg_sha={args.pathcfg_sha}"
    else:
        st, line = with_stage(c, "SMOKE_CKPT", c["ckpt_status"], c["ckpt_line"] + f" attempt={c['attempt']}")
    states.append(st); lines.append(line)
    for ln in lines:
        print(ln, flush=True)
    g = worst(states)
    print(f"G6={g} pathcfg_sha={args.pathcfg_sha} attempts="
          + ",".join(f"{m}:{chosen[m]['attempt'] if chosen[m] else 'none'}" for m in MODES), flush=True)
    return 1 if g == "FAIL" else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage")
    st.add_argument("--mode", choices=MODES, required=True)
    st.add_argument("--attempt", required=True)
    st.add_argument("--run", required=True)
    st.add_argument("--pathcfg", required=True)
    st.add_argument("--pathcfg-sha", required=True)
    st.add_argument("--prefetch-expect", required=True)
    st.add_argument("--run-log", required=True)
    st.add_argument("--rec", required=True)
    st.add_argument("--evidence-dir", required=True)
    st.add_argument("--steps", type=int, required=True)
    st.add_argument("--skip", type=int, default=20)
    st.add_argument("--shmmon-log", default=None)
    st.add_argument("--run-root", default=None)
    st.add_argument("--wan9-leaves", type=int, default=None)
    st.add_argument("--now-et", default=None, help="单测用：固定 at= 时间")
    ag = sub.add_parser("aggregate")
    ag.add_argument("--pathcfg-sha", required=True)
    ag.add_argument("--evidence-dir", required=True)
    args = ap.parse_args(argv)
    return stage(args) if args.cmd == "stage" else aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
