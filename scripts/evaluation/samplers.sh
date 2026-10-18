#!/usr/bin/env bash
# 评测 job 的资源采样器。骨架沿用 scripts/training/prod/gl_train_prod.sbatch 的 start_samplers()，
# 两处按评测需求改动：
#   1. 内存通道间隔可调（探针用 2s）——checkpoint 恢复的峰值只持续约 26 秒，15s 有漏峰风险；
#   2. 增采 cgroup cpu.stat 的 usage_usec（推核当量）与逐进程 RSS（区分 server / client 归因）。
# 逐进程走 cgroup.procs 而不是 pgrep -f：AGENTS 7 禁止裸 pgrep -f，pattern 会匹配到采样器自己。
# 判 OOM 一律看 cgroup anon(+shmem)，不看 MaxRSS——它会精确贴死 --mem 上限（greatlakes.md 铁律）。
SAMPLER_PIDS=()

stop_samplers() {
    [ "${#SAMPLER_PIDS[@]}" -eq 0 ] && return 0
    kill "${SAMPLER_PIDS[@]}" 2>/dev/null || true
    SAMPLER_PIDS=()
}

start_samplers() {   # $1=RECORD_DIR  $2=内存/CPU 采样间隔秒（默认 15）
    local rd="$1" iv="${2:-15}"
    local mp="${MOUNT_POINT:-/nfs/turbo/coe-chaijy-unreplicated}"
    local cg="/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)"
    mkdir -p "$rd"
    echo "CGROUP=$cg SAMPLE_INTERVAL=${iv}s" | tee "$rd/sampler.meta"

    # ① cgroup 内存 + CPU：定档的唯一依据（时间,anon,file,shmem,pgmajfault,cpu_usage_usec）
    ( while true; do
        awk -v t="$(date +%s.%N)" '
          FILENAME ~ /memory.stat$/ { if($1=="anon")a=$2; if($1=="file")f=$2; if($1=="shmem")s=$2; if($1=="pgmajfault")m=$2 }
          FILENAME ~ /cpu.stat$/ { if($1=="usage_usec")u=$2 }
          END { print t","a","f","s","m","u }
        ' "$cg/memory.stat" "$cg/cpu.stat" 2>/dev/null
        sleep "$iv"
      done ) >> "$rd/cgroup_mem_cpu.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)

    # ② cgroup 当前用量与高水位（内核 >=5.19 才有 memory.peak，没有则该列为空）
    ( while true; do
        printf '%s,%s,%s\n' "$(date +%s)" \
          "$(cat "$cg/memory.current" 2>/dev/null)" "$(cat "$cg/memory.peak" 2>/dev/null)"
        sleep "$iv"
      done ) >> "$rd/cgroup_peak.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)

    # ③ 逐进程 RSS（时间,pid,名字,VmRSS_kB）：把峰值归因到 server 还是 client
    ( while true; do
        t=$(date +%s)
        while read -r p; do
          [ -r "/proc/$p/status" ] || continue
          awk -v t="$t" -v p="$p" '/^Name:/{n=$2} /^VmRSS:/{r=$2} END{if(r!="")print t","p","n","r}' "/proc/$p/status"
        done < "$cg/cgroup.procs" 2>/dev/null
        sleep "$iv"
      done ) >> "$rd/proc_rss.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)

    # ④ GPU dense 500ms（AGENTS 16：步时数秒量级必须密集采样，禁以中位数下结论）
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits -lms 500 >> "$rd/gpu_util_dense.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)

    # ⑤ NFS server_read 计数（纯 awk 解析，不起 python 子进程）：取证 checkpoint 恢复带宽
    ( while true; do
        awk -v mp="$mp" -v t="$(date +%s)" '
          /^device / { inm = (index($0, " mounted on " mp " ") > 0); next }
          inm && /^[[:space:]]*bytes:/ { print t","$2","$6; exit }
        ' /proc/self/mountstats
        sleep 15
      done ) >> "$rd/nfs_read.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
}
