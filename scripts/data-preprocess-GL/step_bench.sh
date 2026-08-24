#!/usr/bin/env bash
# ── CPU / mem 档位实测：在不掉 GPU 利用率的前提下，尽可能压低 ─────────────────
#
# 压低的收益（实测数据支撑，不是拍脑袋）：
#   · chaijy2 全组共享配额 GPU 20 / MEM 960 G / **CPU 80**。8 个 job 各要 4 CPU 就吃掉
#     CPU 配额 40%，压到 2 CPU 只吃 20%——配额是组内共享的，留出余量才不会互相卡住。
#   · 节点侧：实测 gl1500/gl1501/gl1508/gl1514 都有空闲 A40 却**零空闲 CPU**，
#     CPU 比 GPU 更常成为「有卡却起不来」的真正原因。
#
# 压过头的风险：`MemoryBuffer.add_buffer` 每步只喂 **1 张图**给 SigLIP，GPU 单次前向极短，
#   流水线大概率本就偏 CPU/IO-bound——h5 解压、numpy 像素差、每步 np.save 588 KB 都在
#   CPU 上。所以必须实测，1 核档位很可能不达标。
#
# 三条选档判据（全过才采纳更省的档）：
#   ① step/s 相对最宽档位退化 ≤ 2%
#   ② GPU 平均利用率退化 ≤ 2 个百分点
#   ③ cgroup anon+shmem 峰值 ≤ 0.6 × 申请 mem（留 1.67× 裕度）
# ⚠ 判据③ 刻意不看 MaxRSS：greatlakes.md 已实证 MaxRSS 会精确贴住 --mem 申请上限，
#   那是页缓存填满配额的读数，不是真实工作集。
#
# 用法：
#   bash step_bench.sh local     # 本机扫档（不占集群配额），筛出候选
#   bash step_bench.sh cluster   # 候选档位各提一个 ≤30min 的 1-GPU 探针（限额内，无需放行）
#   bash step_bench.sh report    # 汇总两边结果
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

v1_prepare_dirs
v1_require_venv

STEP="${1:-local}"
PROBE_N="${PROBE_N:-3}"   # 探针/扫档用每任务前 N 个 episode。取 3（12 ep ≈4100 步）而不是 1：
                          # 单进程有 ~40s 的 SigLIP 加载 + XLA 编译固定开销，样本太小的话
                          # 稳态区间不够长，档位之间的差异会被启动噪声淹没。
SUBSET_PROBE="${V1_STORE}/subset_prefix${PROBE_N}.json"
# 候选档位：本机扫完再按判据挑，这里给的是默认候选集
CLUSTER_TIERS="${CLUSTER_TIERS:-4:16 2:12 2:8 1:8}"

ensure_subset() {
  [[ -f "${MANIFEST_PATH}" ]] || { echo "错误: 先跑 step_local_baseline.sh 生成清单" >&2; exit 1; }
  [[ -f "${SUBSET_PROBE}" ]] || "${PY}" "${V1_SCRIPT_DIR}/scan_manifest.py" sample \
      --manifest "${MANIFEST_PATH}" --out "${SUBSET_PROBE}" --mode prefix --n "${PROBE_N}"
}

do_local() {
  ensure_subset
  v1_require_cmd systemd-run taskset nvidia-smi
  echo "=== 本机扫档 ==="
  echo "  限额手段：内存用 systemd-run --scope -p MemoryMax（实测生效）；"
  echo "            CPU 用 taskset（实测 -p AllowedCPUs 在 user slice 下不生效，"
  echo "            cpuset 控制器没下放，进程 affinity 仍是 0-31）。"
  echo "  输入刻意用本机 H5 原件：本机扫档要隔离的是 CPU/内存两个变量，"
  echo "            走 NFS 会把 turbo 带宽波动掺进来；集群探针那边才读 turbo。"
  "${PY}" "${V1_SCRIPT_DIR}/bench_resources.py" \
      --manifest "${MANIFEST_PATH}" --raw_dir "${RAW_H5_LOCAL}" \
      --subset "${SUBSET_PROBE}" --bench_dir "${BENCH_DIR}" \
      --python "${PY}" --gpu "${BENCH_GPU:-0}" \
      --cpu_tiers "${CPU_TIERS:-8,4,2,1}" --mem_tiers "${MEM_TIERS:-32,16,12,8,6}"
}

do_cluster() {
  ensure_subset
  ssh -O check greatlakes >/dev/null 2>&1 || {
    echo "错误: greatlakes ControlMaster 不存活——按 greatlakes.md 建主连接（须先问用户验证方式）" >&2
    exit 1
  }
  local ids=()
  for tier in ${CLUSTER_TIERS}; do
    local cpus="${tier%%:*}" mem="${tier##*:}"
    local name="c${cpus}m${mem}g"
    local out="${BENCH_DIR}/probe-${name}"
    rm -rf "${out}"
    local exports="ALL,TIER_NAME=${name},RAW_DIR=${RAW_H5_TURBO},MANIFEST=${MANIFEST_PATH}"
    exports="${exports},SUBSET=${SUBSET_PROBE},OUT=${out}"
    echo "=== 提交探针 ${name}（${cpus} CPU / ${mem}G / 1 GPU / 30min）==="
    local jid
    jid="$(uv run --no-project --with pexpect python "${V1_SCRIPT_DIR}/gl_submit.py" \
        "sbatch --parsable --cpus-per-task=${cpus} --mem=${mem}G --job-name=v1-tierprobe-${name} \
         --export=${exports} scripts/data-preprocess-GL/gl_probe.sbatch" \
        | grep -Eo '^[0-9]+$' | tail -1)"
    [[ -n "${jid}" ]] || { echo "错误: 探针 ${name} 未取得 jobid" >&2; exit 1; }
    echo "  jobid=${jid}  日志=${LOGS_DIR}/v1-tierprobe-${name}-${jid}.log"
    ids+=("${jid}")
  done
  printf '%s\n' "${ids[@]}" > "${BENCH_DIR}/cluster_probe_jobids.txt"
  cat <<EOF

已提交 ${#ids[@]} 个探针（本段不等待）。监控建议——每份日志各挂一个 Monitor：
  tail -n +1 -F ${LOGS_DIR}/v1-tierprobe-*-<jobid>.log | grep --line-buffered -E \\
    "JAXCHK|TIER_SUMMARY|SHARD_DONE|PROBE_EXIT_CODE|Error|Traceback|CANCELLED|out of memory"
sacct 兜底： uv run --no-project --with pexpect python ${V1_SCRIPT_DIR}/gl_submit.py \\
    "sacct -j $(IFS=,; echo "${ids[*]}") --format=JobID,State,Elapsed,Submit,Start,ExitCode -X"
全绿后： bash ${V1_SCRIPT_DIR}/step_bench.sh report
EOF
}

do_report() {
  echo "=== 档位汇总（本机 + 集群）==="
  "${PY}" - "${BENCH_DIR}" "${LOGS_DIR}" <<'PYREP'
import json, pathlib, re, sys
bench, logs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

local_p = bench / "local_tier_bench.json"
if local_p.exists():
    rows = json.loads(local_p.read_text())
    base = max((r for r in rows if r.get("rate_step_per_s")),
               key=lambda r: r["rate_step_per_s"], default=None)
    print("\n## 本机扫档")
    print(f"{'档位':<10}{'状态':<6}{'step/s':>9}{'退化':>9}{'GPUμ':>8}{'anon+shmem':>12}{'file':>8}")
    for r in rows:
        st = "MEMKILL" if r.get("mem_killed") else ("FAIL" if r["returncode"] else "OK")
        deg = ""
        if base and r.get("rate_step_per_s"):
            deg = f"{(base['rate_step_per_s'] - r['rate_step_per_s']) / base['rate_step_per_s']:+.1%}"
        print(f"{r['tier']:<10}{st:<6}{r.get('rate_step_per_s') or 0:>9.3f}{deg:>9}"
              f"{r.get('gpu_util_mean') or 0:>8.1f}{r.get('cg_workingset_peak_gib') or 0:>12.2f}"
              f"{r.get('cg_file_peak_gib') or 0:>8.2f}")
else:
    print("（本机扫档结果缺失，先跑 step_bench.sh local）")

print("\n## 集群探针")
found = False
for log in sorted(logs.glob("v1-tierprobe-*.log")):
    txt = log.read_text(errors="replace")
    s = re.search(r"TIER_SUMMARY (.*)", txt)
    r = re.search(r"SHARD_DONE .*rate=([0-9.]+) step/s", txt)
    rc = re.search(r"PROBE_EXIT_CODE=(\d+)", txt)
    jax_ok = "JAXCHK OK" in txt
    if s or rc:
        found = True
        print(f"\n- {log.name}  jax={'OK' if jax_ok else '未通过'}  "
              f"exit={rc.group(1) if rc else '?'}  rate={r.group(1) if r else '?'} step/s")
        if s:
            print(f"  {s.group(1)}")
if not found:
    print("（暂无探针日志，先跑 step_bench.sh cluster 并等 job 结束）")
PYREP
}

case "${STEP}" in
  local)   do_local ;;
  cluster) do_cluster ;;
  report)  do_report ;;
  *) echo "用法: $0 [local|cluster|report]" >&2; exit 1 ;;
esac
