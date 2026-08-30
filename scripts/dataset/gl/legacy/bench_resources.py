#!/usr/bin/env python3
"""本机 CPU/mem 档位扫描器：在不掉 GPU 利用率的前提下，找出能压到多低。

**为什么先在本机扫**：集群探针每档要占一张 A40 且要排队；本机扫一遍能把明显不可行的
档位（OOM、GPU 利用率塌方）先筛掉，只把 2–3 个候选送上集群做权威判定。

**压低档位的收益在哪**：chaijy2 全组共享配额只有 GPU 20 / MEM 960 G / **CPU 80**。
8 个 job 各要 4 CPU 就吃掉 CPU 配额的 40%，压到 2 CPU 只吃 20%。节点侧同理——实测
gl1500/1501/1508/1514 都有空闲 A40 却零空闲 CPU，CPU 比 GPU 更常成为卡点。

**限额怎么做到「真限」（本机实测坑）**：
  · 内存用 `systemd-run --user --scope -p MemoryMax=…`——实测生效（memory.max 确实被设上）。
  · CPU **不能**用 `-p AllowedCPUs=…`——实测在 user slice 下不生效（cpuset 控制器没下放，
    进程 affinity 仍是 0-31），必须改用 `taskset -c`。taskset 的语义与 Slurm 给
    `--cpus-per-task` 做的 cpuset 绑定一致。
  · `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 必须跟着档位一起改，否则线程超订会把
    「档位不够」和「线程打架」两个原因混在一起，档位结论就不可信。

**读数口径**：内存一律看 cgroup 的 anon+shmem 峰值，不看 MaxRSS（见 sample_summary.py）。

**输入侧刻意用本机 H5 原件**：本机扫档要隔离的是 CPU/内存这两个变量，用 NFS 输入会把
turbo 的带宽波动掺进来。集群探针那边读的是 turbo，两者定位不同——本机负责筛，集群负责定。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent

WRAPPER = r"""#!/usr/bin/env bash
# 由 bench_resources.py 生成。在 systemd scope + taskset 限额内跑一个档位。
set -uo pipefail
CG=$(sed -n 's/^0:://p' /proc/self/cgroup | head -n1)
CGDIR="/sys/fs/cgroup${CG}"
echo "CGROUP=$CGDIR"
echo "AFFINITY=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //')"
echo "MEMORY_MAX=$(cat "$CGDIR/memory.max" 2>/dev/null || echo 无)"

: > "$SAMPLOG"
(
    while true; do
        read -r UTIL MIB < <(nvidia-smi --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader,nounits -i "${BENCH_GPU:-0}" 2>/dev/null | head -1 | tr -d ',')
        CUR=$(cat "$CGDIR/memory.current" 2>/dev/null || echo 0)
        ANON=$(awk '$1=="anon"{print $2}'  "$CGDIR/memory.stat" 2>/dev/null || echo 0)
        FILE=$(awk '$1=="file"{print $2}'  "$CGDIR/memory.stat" 2>/dev/null || echo 0)
        SHM=$(awk '$1=="shmem"{print $2}'  "$CGDIR/memory.stat" 2>/dev/null || echo 0)
        echo "sample gpu_util=${UTIL:-0} gpu_mib=${MIB:-0} cg_current=$CUR cg_anon=$ANON cg_file=$FILE cg_shmem=$SHM"
        sleep "${SAMPLE_SEC:-2}"
    done
) >> "$SAMPLOG" 2>/dev/null &
SAMPLER=$!

"$PY" "$SCRIPT_DIR/build_shard.py" \
    --manifest "$MANIFEST" --raw_dir "$RAW_DIR" --out "$OUT" \
    --subset "$SUBSET" --shard_idx 0 --num_shards 1 --report_every 200
RC=$?
kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true
echo "OOM_KILL=$(awk '$1=="oom_kill"{print $2}' "$CGDIR/memory.events" 2>/dev/null || echo 0)"
echo "TIER_EXIT_CODE=$RC"
exit "$RC"
"""


def run_tier(args, cpus: int, mem_gb: int, wrapper: pathlib.Path) -> dict:
    tier = f"c{cpus}m{mem_gb}g"
    out_dir = pathlib.Path(args.bench_dir, f"out-{tier}")
    samplog = pathlib.Path(args.bench_dir, f"{tier}.samples")
    # 每档从干净输出目录起跑：留着上一档的产物会让 build_shard 的 resume 跳过工作，
    # 测出来的就不是这一档的真实速率了。
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        PY=args.python, SCRIPT_DIR=str(_HERE), MANIFEST=args.manifest,
        RAW_DIR=args.raw_dir, OUT=str(out_dir), SUBSET=args.subset,
        SAMPLOG=str(samplog), SAMPLE_SEC=str(args.sample_sec),
        BENCH_GPU=str(args.gpu),
        OMP_NUM_THREADS=str(cpus), MKL_NUM_THREADS=str(cpus),
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
    )
    cmd = [
        "systemd-run", "--user", "--scope", "-q",
        "-p", f"MemoryMax={mem_gb}G", "-p", "MemorySwapMax=0",
        "--", "taskset", "-c", f"0-{cpus - 1}", "bash", str(wrapper),
    ]
    print(f"\n=== 档位 {tier}: cpus={cpus} memmax={mem_gb}G ===", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - t0
    log = pathlib.Path(args.bench_dir, f"{tier}.log")
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")

    # 优先取稳态速率：整体 rate 里混着 ~36 s 的 SigLIP 加载 + XLA 编译固定开销，
    # 短跑时它能占到七成以上，会把所有档位的读数一起压低、掩盖档位间的真实差异。
    rate = None
    m = re.search(r"rate_steady=([0-9.]+) step/s", proc.stdout)
    if m:
        rate = float(m.group(1))
    rate_overall = None
    m = re.search(r"SHARD_DONE .*? rate=([0-9.]+) step/s", proc.stdout)
    if m:
        rate_overall = float(m.group(1))
    if rate is None:
        rate = rate_overall
    steps = None
    m = re.search(r"SHARD_DONE .*steps=(\d+)", proc.stdout)
    if m:
        steps = int(m.group(1))
    # OOM 判定不能只看 wrapper 打的 OOM_KILL：cgroup 触顶时 systemd 会终止**整个 scope**，
    # bash wrapper 自己也被杀，末尾那行 OOM_KILL=/TIER_EXIT_CODE= 根本没机会执行（实测如此，
    # 日志在 shard 起跑行后直接截断、退出码 -15）。所以再加一条「被信号终止且 wrapper 没走完」
    # 的旁证判定，否则会把内存不足误报成普通失败。
    oom = 0
    m = re.search(r"OOM_KILL=(\d+)", proc.stdout)
    if m:
        oom = int(m.group(1))
    wrapper_finished = "TIER_EXIT_CODE=" in proc.stdout
    killed_by_signal = proc.returncode < 0 or proc.returncode == 137
    mem_killed = bool(oom) or (killed_by_signal and not wrapper_finished)
    affinity = (re.search(r"AFFINITY=(.*)", proc.stdout) or [None, "?"])[1].strip()
    memmax = (re.search(r"MEMORY_MAX=(.*)", proc.stdout) or [None, "?"])[1].strip()

    summary = {}
    if samplog.exists():
        sys.path.insert(0, str(_HERE))
        from sample_summary import summarize
        summary = summarize(str(samplog), warmup=args.warmup)

    row = {
        "tier": tier, "cpus": cpus, "mem_gb": mem_gb,
        "returncode": proc.returncode, "oom_kill": oom,
        "mem_killed": mem_killed, "wrapper_finished": wrapper_finished,
        "wall_s": round(wall, 1), "steps": steps,
        "rate_step_per_s": rate, "rate_overall_step_per_s": rate_overall,
        "affinity": affinity, "memory_max": memmax,
        **summary,
    }
    verdict = "内存不足被终止" if mem_killed else ("失败" if proc.returncode else "OK")
    print(
        f"  {verdict}  rate={rate}  gpu_util_mean={summary.get('gpu_util_mean')}  "
        f"anon+shmem峰={summary.get('cg_workingset_peak_gib')}GiB  "
        f"affinity={affinity}  memory.max={memmax}",
        flush=True,
    )
    if args.keep_outputs == 0:
        shutil.rmtree(out_dir, ignore_errors=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--bench_dir", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sample_sec", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--keep_outputs", type=int, default=0)
    ap.add_argument("--cpu_tiers", default="8,4,2,1")
    ap.add_argument("--mem_tiers", default="32,16,12,8,6")
    ap.add_argument("--ref_cpu", type=int, default=8, help="扫 mem 时固定的 CPU 档")
    ap.add_argument("--ref_mem", type=int, default=32, help="扫 CPU 时固定的 mem 档")
    args = ap.parse_args()

    bench = pathlib.Path(args.bench_dir)
    bench.mkdir(parents=True, exist_ok=True)
    wrapper = bench / "_tier_wrapper.sh"
    wrapper.write_text(WRAPPER, encoding="utf-8")
    wrapper.chmod(0o755)

    cpu_tiers = [int(x) for x in args.cpu_tiers.split(",") if x]
    mem_tiers = [int(x) for x in args.mem_tiers.split(",") if x]

    rows: list[dict] = []
    # 两条一维扫描而不是全笛卡尔积：CPU 与 mem 的瓶颈机理互不耦合（一个卡计算/IO 线程，
    # 一个卡工作集），一维各扫一遍就能定位拐点，成本从 20 档降到 8 档。
    print("### 第一遍：固定 mem，扫 CPU ###")
    rows.extend(run_tier(args, c, args.ref_mem, wrapper) for c in cpu_tiers)
    print("\n### 第二遍：固定 CPU，扫 mem ###")
    rows.extend(run_tier(args, args.ref_cpu, m, wrapper) for m in mem_tiers)

    report = bench / "local_tier_bench.json"
    report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    base = next((r for r in rows if r["cpus"] == max(cpu_tiers) and r["mem_gb"] == args.ref_mem), None)
    print("\n=== 本机档位对照表 ===")
    hdr = f"{'档位':<10}{'状态':<6}{'step/s':>9}{'退化':>8}{'GPU均值':>9}{'anon+shmem':>12}{'file':>8}"
    print(hdr)
    for r in rows:
        state = "MEMKILL" if r.get("mem_killed") else ("FAIL" if r["returncode"] else "OK")
        deg = ""
        if base and base.get("rate_step_per_s") and r.get("rate_step_per_s"):
            deg = f"{(base['rate_step_per_s'] - r['rate_step_per_s']) / base['rate_step_per_s']:+.1%}"
        print(f"{r['tier']:<10}{state:<6}{r.get('rate_step_per_s') or 0:>9.3f}{deg:>8}"
              f"{r.get('gpu_util_mean') or 0:>9.1f}{r.get('cg_workingset_peak_gib') or 0:>12.2f}"
              f"{r.get('cg_file_peak_gib') or 0:>8.2f}")
    print(f"\n报告 -> {report}")
    print("BENCH_LOCAL_DONE")


if __name__ == "__main__":
    main()
