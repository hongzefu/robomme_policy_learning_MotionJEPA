#!/usr/bin/env bash
# ── 提交四任务全量构建：8×1GPU array + afterok finalize ────────────────────────
#
# ⚠ 审批闸门：8 并发 GPU 与长 walltime **超出 greatlakes.md 的调试限额**
#   （硬规则：≤2 GPU、≤00:30:00）。必须由用户逐次显式放行：
#       CONFIRM_FULL=yes bash step1_submit.sh
#
# pre-flight 九项，任一失败拒绝提交——否则 8 个 job 排队几小时后集体炸，
# 还要人工清 8 个 claim。
#
# 续跑（某分片超时/被杀）三步：
#   ① rm <OUT>/_claims/_claim_shard<i>of8.json
#   ② 重提该分片： sbatch --array=<i> --export=ALL,REQUIRE_EMPTY=0,<三路径> ... gl_build_dataset.sbatch
#   ③ 连 finalize 一起重提： sbatch --dependency=afterok:<原AID>:<新JOBID> ... gl_finalize.sbatch
#      （任一分片失败会让原 finalize 因 afterok 不满足被 kill_invalid_depend 自动 CANCELLED，
#        此时连日志文件都不会生成，判死只能靠 sacct）
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

v1_prepare_dirs
v1_require_cmd jq uv
v1_require_venv

NUM_SHARDS="${NUM_SHARDS:-8}"
TIER_CPUS="${TIER_CPUS:-2}"          # 2026-08-23 档位实测定案值（复测入口 legacy/step_bench.sh）
TIER_MEM_GB="${TIER_MEM_GB:-24}"     # 同上定案值。全量实测最重分片 anon 峰 14.78 GiB，压到 16G 几乎必然 OOM
RATE="${RATE:-0}"                    # A40 实测 step/s（定案值 28.913，来自探针）；0 表示未给，pre-flight 会拒绝
WALLTIME="${WALLTIME:-}"             # 留空则按 RATE 反算
FTIME="${FTIME:-03:00:00}"   # 完整性核验要在 NFS 上 scandir 39.5 万个 pkl + 1600 个目录
SPOT_CHECK="${SPOT_CHECK:-256}"
JOB="${JOB:-v1-4task-build}"
OUT="${OUT:-${GL_DATASET}}"
FAIL=0

echo "[提交] OUT=${OUT}  array=0-$((NUM_SHARDS - 1))  档位=${TIER_CPUS} CPU / ${TIER_MEM_GB}G"
echo "[提交] pre-flight："

# ① 清单
if [[ -f "${MANIFEST_PATH}" ]]; then
  EP_COUNT="$(jq -r '.totals.episodes' "${MANIFEST_PATH}")"
  TOTAL_STEPS="$(jq -r '.totals.timesteps' "${MANIFEST_PATH}")"
  MSHARDS="$(jq -r '.num_shards' "${MANIFEST_PATH}")"
  if [[ "${EP_COUNT}" -eq "${EXPECTED_TOTAL_EPISODES}" ]]; then
    echo "  ✓ 清单 episode=${EP_COUNT} timestep=${TOTAL_STEPS}"
  else
    echo "  ✗ 清单 episode=${EP_COUNT} ≠ 期望 ${EXPECTED_TOTAL_EPISODES}"; FAIL=1
  fi
  [[ "${MSHARDS}" -eq "${NUM_SHARDS}" ]] || { echo "  ✗ 清单 num_shards=${MSHARDS} ≠ ${NUM_SHARDS}"; FAIL=1; }
  "${PY}" -c "
import sys; sys.path.insert(0, '${V1_SCRIPT_DIR}')
from scan_manifest import load_manifest; load_manifest('${MANIFEST_PATH}')" \
    && echo "  ✓ 清单 sha256 自洽" || { echo "  ✗ 清单 sha256 不符"; FAIL=1; }
else
  echo "  ✗ 缺清单 ${MANIFEST_PATH}——先跑 step0_setup_turbo.sh manifest"; FAIL=1; TOTAL_STEPS=0
fi

# ② 输入清单与 H5
[[ -f "${INPUT_MANIFEST_PATH}" ]] && echo "  ✓ 输入 sha256 清单在位" \
  || { echo "  ✗ 缺 ${INPUT_MANIFEST_PATH}——先跑 step0_setup_turbo.sh h5"; FAIL=1; }
v1_validate_raw_h5 "${RAW_H5_TURBO}" && echo "  ✓ turbo H5 四件齐全"

# ③ 输出目录洁净（续跑时置 REQUIRE_EMPTY=0 跳过）
REQUIRE_EMPTY="${REQUIRE_EMPTY:-1}"
if [[ "${REQUIRE_EMPTY}" = "1" ]]; then
  N_RESIDUE=0
  [[ -d "${OUT}/features" ]] && N_RESIDUE="$(find "${OUT}/features" -maxdepth 1 -type d -name 'episode_*' | wc -l)"
  if [[ "${N_RESIDUE}" -eq 0 ]]; then echo "  ✓ 输出库为空"
  else echo "  ✗ ${OUT}/features 已有 ${N_RESIDUE} 个 episode 目录（续跑请置 REQUIRE_EMPTY=0）"; FAIL=1; fi
fi

# ④ 无 claim/tmp 残留
N_CLAIM=0
[[ -d "${OUT}/_claims" ]] && N_CLAIM="$(find "${OUT}/_claims" -name '_claim_*' | wc -l)"
if [[ "${N_CLAIM}" -eq 0 ]]; then echo "  ✓ 无 claim 残留"
else echo "  ✗ 有 ${N_CLAIM} 个 claim 残留，确认对应 job 已死后手动删除"; FAIL=1; fi

# ⑤ ControlMaster
if ssh -O check greatlakes >/dev/null 2>&1; then echo "  ✓ greatlakes ControlMaster 存活（免认证）"
else echo "  ✗ ControlMaster 不存活——按 greatlakes.md 建主连接（须先问用户验证方式）"; FAIL=1; fi

# ⑥ 模型
v1_require_models 0 && echo "  ✓ 项目内 SigLIP 就位"

# ⑦ walltime 裕度 ≥1.2×：GPU 与 I/O 两条估算取大者
#    单分片探针测不出 8 路并发下的 turbo 带宽争用，只按探针速率反算会严重低估。
#    全量 I/O = 读 321 GB 原始 H5 + 写 ≈(总步数 × 每步字节)；8 个分片并发共享同一个卷，
#    所以 I/O 侧的耗时对每个分片都是「全量字节 ÷ 卷带宽」，不再除以 8。
IO_BW_MBPS="${IO_BW_MBPS:-132}"          # turbo 卷实测天花板（greatlakes.md）
BYTES_PER_STEP="${BYTES_PER_STEP:-932154}"   # 910 KiB/step，legacy/step_local_baseline.sh 实测校准值
                                             # （token_emb 的 602,144 B + data/*.pkl 里的原图与 wrist 图）
RAW_BYTES="${RAW_BYTES:-344999999999}"       # 321 GiB 原始 H5
if [[ "${RATE}" != "0" && "${TOTAL_STEPS}" -gt 0 ]]; then
  read -r GPU_EST IO_EST EST_SEC <<< "$("${PY}" -c "
gpu = ${TOTAL_STEPS} / ${NUM_SHARDS} / ${RATE}
io  = (${TOTAL_STEPS} * ${BYTES_PER_STEP} + ${RAW_BYTES}) / (${IO_BW_MBPS} * 1e6)
print(int(gpu), int(io), int(max(gpu, io)))")"
  printf "  · GPU 侧估算 %dh%02dm（%s step ÷ %s 片 ÷ %s step/s）\n" \
    $((GPU_EST / 3600)) $((GPU_EST % 3600 / 60)) "${TOTAL_STEPS}" "${NUM_SHARDS}" "${RATE}"
  printf "  · I/O 侧估算 %dh%02dm（读 321GB + 写 %d×%dB，÷ %s MB/s 卷带宽，8 路共享不除以片数）\n" \
    $((IO_EST / 3600)) $((IO_EST % 3600 / 60)) "${TOTAL_STEPS}" "${BYTES_PER_STEP}" "${IO_BW_MBPS}"
  if [[ -z "${WALLTIME}" ]]; then
    WALLTIME="$("${PY}" -c "
s = int(${EST_SEC} * 1.5)
print(f'{s // 3600:02d}:{s % 3600 // 60:02d}:00')")"
    echo "  · walltime 未给定，按两者取大 ×1.5 反算 = ${WALLTIME}"
  fi
  WT_SEC="$("${PY}" -c "
h, m, s = '${WALLTIME}'.split(':'); print(int(h) * 3600 + int(m) * 60 + int(s))")"
  MARGIN="$("${PY}" -c "print(f'{${WT_SEC} / max(1, ${EST_SEC}):.2f}')")"
  echo "  · 取大者 $((EST_SEC / 3600))h$((EST_SEC % 3600 / 60))m vs 申请 ${WALLTIME}，裕度 ${MARGIN}×"
  "${PY}" -c "import sys; sys.exit(0 if ${MARGIN} >= 1.2 else 1)" \
    && echo "  ✓ walltime 裕度 ≥1.2×" \
    || { echo "  ✗ walltime 裕度 <1.2×——提高 WALLTIME 或 NUM_SHARDS（超时=每片一轮重提周转）"; FAIL=1; }
else
  echo "  ✗ 未提供 RATE（A40 实测 step/s）——定案值 RATE=28.913；数据形制变了则先跑 legacy/step_bench.sh cluster 重测"; FAIL=1
fi

# ⑧ 配额：8×档位必须装得下 chaijy2 的剩余 CPU / MEM
# ⚠ 退出码必须「先捕获再判」。原写法末尾一个 `|| true` 兼了两件事：
#   ① 不让 set -euo pipefail 把脚本打死；② 查询失败时静默跳过该项。
#   ② 是 fail-open —— ControlMaster 掉线或 sacctmgr 抽风时，第 ⑧ 项悄悄变成「通过」，
#   配额真不够也照样提交 8 个 GPU job 去挤爆组内共享额度。拆开：仍不打死脚本，但判 FAIL。
QUOTA=""; QUOTA_RC=0
QUOTA="$(uv run --no-project --with pexpect python "${V1_SCRIPT_DIR}/gl_submit.py" \
  "sacctmgr -nP show assoc account=chaijy2 format=GrpTRES 2>/dev/null | grep -m1 . ; \
   echo ---; squeue -A chaijy2 -t RUNNING -h -O 'tres-alloc:300'" 2>/dev/null)" || QUOTA_RC=$?
NEED_CPU=$((NUM_SHARDS * TIER_CPUS)); NEED_MEM=$((NUM_SHARDS * TIER_MEM_GB)); NEED_GPU="${NUM_SHARDS}"
echo "  · 本次需求：GPU ${NEED_GPU} / CPU ${NEED_CPU} / MEM ${NEED_MEM}G"
if [[ "${QUOTA_RC}" -eq 0 && -n "${QUOTA}" ]]; then
  "${PY}" "${V1_SCRIPT_DIR}/check_quota.py" --quota_text "${QUOTA}" \
      --need_gpu "${NEED_GPU}" --need_cpu "${NEED_CPU}" --need_mem_gb "${NEED_MEM}" || FAIL=1
elif [[ "${ALLOW_QUOTA_SKIP:-}" == "yes" ]]; then
  echo "  ⚠ 配额查询失败（rc=${QUOTA_RC}），已按 ALLOW_QUOTA_SKIP=yes 显式豁免——"
  echo "    提交后必须自行用 skill greatlakes-usage 复核，本项不构成绿灯"
else
  echo "  ✗ 配额查询失败（rc=${QUOTA_RC}）——集群不可达或 ControlMaster 掉了。"
  echo "    确需带病提交： ALLOW_QUOTA_SKIP=yes CONFIRM_FULL=yes ... bash $0"
  FAIL=1
fi

# ⑨ 审批闸门
if [[ "${CONFIRM_FULL:-}" != "yes" ]]; then
  echo
  echo "⛔ 全量构建是审批点：${NUM_SHARDS}×1GPU 并发与 ${WALLTIME:-未定} walltime 超出"
  echo "   greatlakes.md 的调试限额（≤2 GPU、≤00:30:00），须用户明示批准："
  echo "     CONFIRM_FULL=yes RATE=<实测> TIER_CPUS=${TIER_CPUS} TIER_MEM_GB=${TIER_MEM_GB} bash $0"
  exit 1
fi
[[ "${FAIL}" -eq 0 ]] || { echo "pre-flight 未全绿，拒绝提交"; exit 1; }

EXPORTS="ALL,NUM_SHARDS=${NUM_SHARDS},RAW_DIR=${RAW_H5_TURBO},OUT=${OUT}"
EXPORTS="${EXPORTS},MANIFEST=${MANIFEST_PATH},INPUT_MANIFEST=${INPUT_MANIFEST_PATH}"
EXPORTS="${EXPORTS},SPOT_CHECK=${SPOT_CHECK},REQUIRE_EMPTY=${REQUIRE_EMPTY}"

# ⚠ 提交结果必须「先整份捕获、再单独解析」。原写法把提交与解析写成一条管道
#   `AID="$(... | grep -Eo '^[0-9]+$' | tail -1)"`：grep 无匹配即退出码 1，pipefail 让
#   整条管道非零，命令替换赋值随之失败，set -e 当场杀掉脚本——紧跟其后那句自己写的
#   `✗ 未取得 array jobid` 诊断**永远执行不到**。下面 `|| RC=$?` 是这条链路的关键。
echo "[提交] array..."
ARRAY_OUT=""; ARRAY_RC=0
ARRAY_OUT="$(uv run --no-project --with pexpect python "${V1_SCRIPT_DIR}/gl_submit.py" \
  "sbatch --parsable --array=0-$((NUM_SHARDS - 1)) --cpus-per-task=${TIER_CPUS} \
   --mem=${TIER_MEM_GB}G --time=${WALLTIME} --job-name=${JOB} --export=${EXPORTS} \
   scripts/dataset/gl/gl_build_dataset.sbatch" 2>&1)" || ARRAY_RC=$?
# grep 整行锚定，混进来的 stderr（如 `[remote exit 5]`）不会被误认成 jobid
AID="$(printf '%s\n' "${ARRAY_OUT}" | grep -Eo '^[0-9]+$' | tail -1 || true)"
if [[ "${ARRAY_RC}" -ne 0 || -z "${AID}" ]]; then
  echo "✗ array 提交失败（rc=${ARRAY_RC}），未取得 jobid。"
  echo "  **本次没有提交任何作业**，排除故障后原样重跑本脚本即可。"
  echo "  提交器原始输出："
  printf '%s\n' "${ARRAY_OUT}" | sed 's/^/    /'
  echo "SUBMITTED_NONE"
  exit 1
fi
echo "  array jobid = ${AID}"

echo "[提交] finalize（afterok:${AID}）..."
FIN_OUT=""; FIN_RC=0
FIN_OUT="$(uv run --no-project --with pexpect python "${V1_SCRIPT_DIR}/gl_submit.py" \
  "sbatch --parsable --dependency=afterok:${AID} --time=${FTIME} \
   --job-name=${JOB}-final --export=${EXPORTS} \
   scripts/dataset/gl/gl_finalize.sbatch" 2>&1)" || FIN_RC=$?
FID="$(printf '%s\n' "${FIN_OUT}" | grep -Eo '^[0-9]+$' | tail -1 || true)"
if [[ "${FIN_RC}" -ne 0 || -z "${FID}" ]]; then
  # 这是真正危险的半途状态：8 个 GPU job 已经在跑，却没有收尾守卫绑在后面。
  # 重跑本脚本会二次提交 array（且第 ③ 项输出洁净检查会拦住你）。必须手工补交。
  cat <<EOF
✗ finalize 提交失败（rc=${FIN_RC}），但 array ${AID} **已经提交、可能正在跑**。
  这是半途状态，重跑本脚本会二次提交 array。正确做法是手工补交 finalize：

  ① 先确认 array 状态：
     uv run --no-project --with pexpect python ${V1_SCRIPT_DIR}/gl_submit.py \\
       "sacct -j ${AID} --format=JobID,State,Elapsed,ExitCode -X"
  ② 补交 finalize（即 README「续跑与故障处理」三步的第三步）：
     uv run --no-project --with pexpect python ${V1_SCRIPT_DIR}/gl_submit.py \\
       "sbatch --parsable --dependency=afterok:${AID} --time=${FTIME} \\
        --job-name=${JOB}-final --export=${EXPORTS} \\
        scripts/dataset/gl/gl_finalize.sbatch"
  ③ 若决定不补交，必须先取消 array，避免留下一份从未过收尾守卫
     （完整性 / stats / provenance / 同架构零容差抽检）的库：
     uv run --no-project --with pexpect python ${V1_SCRIPT_DIR}/gl_submit.py "scancel ${AID}"

  提交器原始输出：
EOF
  printf '%s\n' "${FIN_OUT}" | sed 's/^/    /'
  echo "SUBMITTED_PARTIAL array=${AID} finalize=NONE"
  exit 1
fi
echo "  finalize jobid = ${FID}"

cat <<EOF

[提交] 完成（sbatch 异步，本段不等待）。
  分片日志： ${LOGS_DIR}/${JOB}-${AID}_<0..$((NUM_SHARDS - 1))>.log
  finalize： ${LOGS_DIR}/${JOB}-final-${FID}.log
  监控：每份日志各挂一个 Monitor，过滤
    "PROGRESS|EPISODE|SHARD_EXIT_CODE|FINALIZE_EXIT_CODE|Error|Traceback|CLAIM_EXISTS|CANCELLED|out of memory"
  sacct 兜底（finalize 因依赖失败被 CANCELLED 时不会有日志文件）：
    gl_submit.py "sacct -j ${AID},${FID} --format=JobID,State,Elapsed,ExitCode -X"
全绿后： bash ${V1_SCRIPT_DIR}/step2_verify.sh
SUBMITTED array=${AID} finalize=${FID}
EOF
