# 原版80k P1入口两步验证：正式起跑前记录（未运行）

**P1尚未运行，也没有P1通过结论。** 两组正式CPU输入已在真实INPUT_HEAD `3a1582db39c723c735e04752e5027bfe40ecc3e1` 完成并分别通过，具体证据如下。依据[主计划C节](../../../0925-orig-80k-full-counting-4plus4-plan.md)，接下来准备两库各自A/B的两步入口验证；本文件与[100步起跑档案](train100-launch.md)及工具补齐须先共同形成clean Beta，之后将其完整40位字面量填入 `TRAIN_HEAD`。当前不预填未来Beta、P1标签、会话或运行结果，也不把INPUT_HEAD冒称为P1启动版本。

用户已明确「允许仅对拍关闭 W&B」；本P1两侧均使用既有驱动的 `--no-wandb-enabled`，五标量和全部参数/EMA/优化器/step叶摘要留本地。只把 `num_train_steps` 覆盖为2、log_interval为1、save_interval为50；保留b64/fsdp4/workers4/seed42、modul 512/4×4、原warmup/lr/EMA/keep参数。本阶段不调用任何judge，不产生100步baseline，不宣称A/B轨迹逐位等价，也不替代后续100步、单跑/并跑、300步perf或80k。

## 已完成INPUT及跨版本沿用

正式INPUT已完成，全部采集/judge会话已经结束。真实 **`INPUT_HEAD=3a1582db39c723c735e04752e5027bfe40ecc3e1`**，A项目源码固定 **`ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`**；这两个提交分别是已完成的B侧取证版本和上游版本，不是尚待创建的TRAIN_HEAD。原始取证根为：

`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-equiv-0925/input-20260926T170654Z`

每组下的 `a/`、`b/` 分别是A/B独立记录。两份真实judge均输出 `comparison=host_numeric_exact_motion_none_v1`；覆盖及结果如下，边界样本数与执行样本总数分开记录：

| 组 | episode | 全量执行样本身份 | 边界样本 | 每侧真实batch | 完整sampler epoch | 实际判定 |
|---|---:|---:|---:|---:|---:|---|
| full | 1600 | 476857 | 6906 | 104 | 2 | INPUT_EQ=PASS |
| count | 400 | 189035 | 2400 | 104 | 2 | INPUT_EQ=PASS |

两份judge原始日志位于主仓 `v1-store/logs/`，文件分别为 `orig80k-input-full-judge-20260926T170654Z.log` 和 `orig80k-input-count-judge-20260926T170654Z.log`；各自的COMMAND_EXIT、TEE_EXIT、FOOTER_PRINTF_EXIT、FOOTER_TEE_EXIT和EXIT_CODE均唯一为0。已核对的原始摘要为：

| 原始证据 | SHA256 |
|---|---|
| full/a/record_manifest.json | `b95d89845d2e7219a6102cd05ffe26d5941ebcf66179f5c423539e1d710dbc05` |
| full/b/record_manifest.json | `e70f4dc1c66e60ac97b1dd3f7fc116a82bb783f4bb4e00dd34c236572808ddf0` |
| count/a/record_manifest.json | `4adabc7d5e4f959682530360c6ea06acb30222cc5419ffd16b4e19ec9706c24e` |
| count/b/record_manifest.json | `ac66670e25f5c9f9b01e088b5c15d1577f62d6cc11563c67c4be54b6fbc69430` |
| full原始judge日志 | `282a827f091ae72f4ad390f54fc1be3e2ee829f45c18ec55b26e5da4fa9cec32` |
| count原始judge日志 | `6b69b0cdaf34e04e086324d014540d47d9a6dbc71088cb7c96846b3d9a2b91b0` |

原采集的实际版本与范围见[INPUT档案](README.md)及[其启动记录](launch.md)。**沿用到未来TRAIN_HEAD前仍须实查，不因INPUT已PASS自动放行新版本。** 执行方须保存INPUT_HEAD到TRAIN_HEAD的固定提交差异，逐项核对输入取证工具及协议、父进程/worker实际项目模块、Python与依赖版本、输入参数/history、数据source/manifest/provenance/packed元信息、norm/tokenizer及初始化资产指纹不变，并记录实际对照结果。本文件未预填这项跨版本复核PASS，也不以“改的是文档或无关工具”代替证据。

原输入记录继续绑定真实INPUT_HEAD；不能改写其HEAD或工具SHA，重新judge旧记录也不等于新提交重采。取证工具改变须两侧同工具重采；输入相关指纹变化或影响不明先报告，再按受影响范围确定重测。**INPUT的受控跨版本沿用不豁免100步单跑/并跑的同HEAD要求：S1、S3、S2必须同一TRAIN_HEAD及完整环境指纹。**

## 入口、资产与真正的保存行为

复用 `scripts/training/tests/run_entry_equiv.sh` 的 `run-a` / `run-b`，harness为同目录 `entry_equiv.py`。A固定上游 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，由独立worktree的uv `.venv/bin/python`运行其 `scripts/train.py::__main__`，执行2步tentative再2步main；B由主仓库uv环境运行当前 `scripts/training/train.py`，仅2步main。B的 `UV_PROJECT_ENVIRONMENT` 显式指向主仓库 `.venv`，`uv run --no-sync` 不同步依赖；harness还会检查实际 `sys.prefix` 归属。A/B不共用环境，不修改上游源码、不设置HOME。

| 组 | source / B packed（均相对主仓库v1-store/datasets） | norm父目录（相对v1-store/train-assets） | norm SHA256 |
|---|---|---|---|
| full | `16task-pub-1600ep/source` / `16task-pub-1600ep/framesamp` | `mme_vla_suite` | `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5` |
| count | `4task-counting-pub-400ep/source` / `4task-counting-pub-400ep/framesamp` | `mme_vla_suite/4task-counting-pub-400ep` | `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9` |

两侧history均为 `perceptual-framesamp-modul.yaml`，SHA256为 `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`。A实际消费source；B显式绑定同库source/episode manifest并消费packed。初始化参数与tokenizer分别为主仓库 `v1-store/models/openpi-assets/checkpoints/pi05_base/params`、`v1-store/models/big_vision/paligemma_tokenizer.model`，沿用已核实资产，不重新下载。正式起跑还须核对资产锁、两库packed verified/full、无pack.lock、两组CPU输入记录及norm/history实际摘要。

`entry_equiv.py::_run()` 把 `openpi.training.checkpoints.save_state` 临时替换为 `_summarize_state()`；它读取真实TrainState各叶并计算摘要，**不调用原始save_state、不落真实checkpoint权重**。两段A末步各触发一次step=1摘要，必须保留 `state_digests.jsonl` 的 `[1,1]`；B为 `[1]`。初始化仍会创建checkpoint工作目录，不能把“不落权重”写成“不产生目录”。`_CheckpointOwnership` 拒绝首轮复用已有根；只允许A第二段重建它在同一进程中创建且inode归属未变的tentative目录，B禁止overwrite。此内部受控动作不授权清空任何旧产物。

## 起跑前置和四个任务的顺序

两组CPU采集与judge已经完成；P1起跑前仍须完成上节INPUT沿用复核，再核对主副本与A源码clean、独立解释器、资产及磁盘预算。资源预检重新记录 `nvidia-smi` 的设备/活动进程、可用内存/SHM、`df -B1 /scratch` 和本轮活动任务；本档案不预写资源充足结论。P1起跑后全阶段冻结主副本源码和两侧 `.venv`，不并行进行本轮建库或其它GPU训练/测速。

| 阶段 | 拟会话名 | GPU | 依赖 |
|---|---|---|---|
| A-full | `orig80k-p1-full-a-$P1_TAG` | 0,1,2,3 | 输入及环境闸门通过 |
| A-count | `orig80k-p1-count-a-$P1_TAG` | 4,5,6,7 | 同上，可与A-full并行 |
| B-full | `orig80k-p1-full-b-$P1_TAG` | 0,1,2,3 | 两个A均成功、已退出、记录已核验；本轮独占运行 |
| B-count | `orig80k-p1-count-b-$P1_TAG` | 4,5,6,7 | B-full成功、已退出且已核验；本轮独占运行 |

任务预计可能超过5分钟，均以独立detached tmux运行并保留记录。会话全名、tmux pane PID、harness PID、起止时间及命令文件SHA在现场记录；只按完整名 `tmux has-session -t '=完整会话名'`检查，不执行全局清理。任一失败停止依赖步骤并保留全部文件，不自动改名、覆盖重试、降batch/workers或放宽有限值/来源判据。

## 待填公共设置

以下命令仅作为待执行正文；不得整页自动执行。唯一需要手工填入的命令占位为 `TRAIN_HEAD` 与 `P1_TAG`。四个拟exp_name均含独立P1前缀和同一唯一标签，不能复用已完成的20步、CPU输入或未来100步名称。本P1档案须与100步档案及工具补齐一并提交为clean Beta，之后填实际锚点；起跑现场再保存展开后的具体名称和命令，不用当前开发HEAD动态代替期望提交。

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
UP="$STORE/worktrees/orig-ecf086c"
UPSTREAM_HEAD=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b
TRAIN_HEAD='<待两份launch及工具补齐的clean Beta提交后填写40位Git SHA>'
P1_TAG='<本轮唯一UTC标签>'
P1_ROOT="$STORE/bench/orig80k-p1-$P1_TAG"
COMMAND_ROOT="$STORE/bench/orig80k-p1-commands-$P1_TAG"
P1_CACHE="$STORE/cache/jax/orig80k-p1-$P1_TAG"
DRIVER="$MAIN/scripts/training/tests/run_entry_equiv.sh"
FULL="$STORE/datasets/16task-pub-1600ep"
COUNT="$STORE/datasets/4task-counting-pub-400ep"
FULL_SOURCE="$FULL/source"
COUNT_SOURCE="$COUNT/source"
FULL_DATA="$FULL/framesamp"
COUNT_DATA="$COUNT/framesamp"
FULL_ASSETS="$STORE/train-assets/mme_vla_suite"
COUNT_ASSETS="$STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep"
HISTORY_YAML=perceptual-framesamp-modul.yaml

[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$P1_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
test "$(git -C "$MAIN" rev-parse HEAD)" = "$TRAIN_HEAD" || exit 2
test -z "$(git -C "$MAIN" status --porcelain)" || exit 2
test "$(git -C "$UP" rev-parse HEAD)" = "$UPSTREAM_HEAD" || exit 2
test -z "$(git -C "$UP" status --porcelain)" || exit 2
test -x "$UP/.venv/bin/python" && test -x "$MAIN/.venv/bin/python" || exit 2
for path in "$P1_ROOT" "$COMMAND_ROOT" "$P1_CACHE"; do
  test ! -e "$path" && test ! -L "$path" || exit 2
done
test -d "$STORE/logs" && test ! -L "$STORE/logs" || exit 2
for side in full-a full-b count-a count-b; do
  exp="p1-orig80k-$side-$P1_TAG"
  for path in "$STORE/cache/cuda/$exp" "$STORE/entryeq/wandb/$exp" \
    "$STORE/cache/wandb/$exp" "$STORE/cache/wandb-config/$exp" "$STORE/cache/wandb-data/$exp"; do
    test ! -e "$path" && test ! -L "$path" || exit 2
  done
done
printf '%s  %s\n' \
  f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5 "$FULL_ASSETS/robomme/norm_stats.json" \
  a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9 "$COUNT_ASSETS/robomme/norm_stats.json" \
  823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a "$MAIN/src/mme_vla_suite/models/config/robomme/$HISTORY_YAML" \
  823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a "$UP/src/mme_vla_suite/models/config/robomme/$HISTORY_YAML" \
  | sha256sum --check --strict || exit 2

P1_ENV=(env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME
  -u JAX_COMPILATION_CACHE_DIR -u TRAIN_RECORD_DIR -u TRAIN_FINAL_RECORD_DIR -u TRAIN_TIMING_STEPS
  JAX_ENABLE_X64=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
  WORKTREE="$UP" A_PYTHON="$UP/.venv/bin/python" UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
  HEAD_A="$UPSTREAM_HEAD" HEAD_B="$TRAIN_HEAD"
  STEPS=2 BATCH=64 FSDP=4 SAVE_INTERVAL=50 HISTORY_YAML="$HISTORY_YAML"
  MODE=upstream EXPECT_TENTATIVE_A=2 EXPECT_STATE_STEPS=1 ANCHOR_SHA256=)
FULL_ENV=(RECORD_ROOT="$P1_ROOT/full" RECORD_A="$P1_ROOT/full/a" RECORD_B="$P1_ROOT/full/b"
  EXP_A="p1-orig80k-full-a-$P1_TAG" EXP_B="p1-orig80k-full-b-$P1_TAG"
  DATA_A="$FULL_SOURCE" DATA_B="$FULL_DATA" ASSETS_DIR="$FULL_ASSETS"
  FRAMESAMP_SOURCE="$FULL_SOURCE" FRAMESAMP_MANIFEST="$FULL/meta/episode_manifest.json"
  CHECKPOINT_A="$P1_ROOT/checkpoints/full-a" CHECKPOINT_B="$P1_ROOT/checkpoints/full-b"
  JAX_CACHE_A="$P1_CACHE/full-a" MMEVLA_JAX_CACHE_DIR="$P1_CACHE/full-b")
COUNT_ENV=(RECORD_ROOT="$P1_ROOT/count" RECORD_A="$P1_ROOT/count/a" RECORD_B="$P1_ROOT/count/b"
  EXP_A="p1-orig80k-count-a-$P1_TAG" EXP_B="p1-orig80k-count-b-$P1_TAG"
  DATA_A="$COUNT_SOURCE" DATA_B="$COUNT_DATA" ASSETS_DIR="$COUNT_ASSETS"
  FRAMESAMP_SOURCE="$COUNT_SOURCE" FRAMESAMP_MANIFEST="$COUNT/meta/episode_manifest.json"
  CHECKPOINT_A="$P1_ROOT/checkpoints/count-a" CHECKPOINT_B="$P1_ROOT/checkpoints/count-b"
  JAX_CACHE_A="$P1_CACHE/count-a" MMEVLA_JAX_CACHE_DIR="$P1_CACHE/count-b")
```

驱动明确设置 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`CUDA_VISIBLE_DEVICES=$GPUS`、`UV_LINK_MODE=copy`、`UV_CACHE_DIR=$STORE/cache/uv`、`OPENPI_DATA_HOME=$STORE/models`、`XDG_CACHE_HOME=$STORE/cache`、`HF_HOME=$STORE/cache/huggingface`，并把CUDA与W&B各类缓存按exp_name隔离到以上新路径。这里记录驱动的实际值，不误写成CPU输入阶段的另一套HF/XDG路径。A/B的JAX cache由harness只转接cache配置项，`jax_cache.json`验实；A清除packed来源环境，B显式注入本组source/manifest。`EXPECT_STATE_STEPS=1` 只固化驱动参数，P1绝不调用其judge；不能把A的两个state step=1去重。

## 保存命令文件与完整退出记录

下述wrapper沿用已验证的CPU INPUT协议：保存展开后的同一命令文件，以SHA256作为外部参数传入，启动后重新核对文件SHA。正文tee与footer tee分别捕获PIPESTATUS；只有检查footer后才直接追加唯一终态。命令文件不作为独立脚本复制到docs归档；归档正文保留实际命令及真实路径/SHA。日志不放harness记录目录，避免混淆量具自产证据。

```bash
launch_p1() {
  local session="$1" log="$2" cwd="$3"
  shift 3
  test ! -e "$log" && test ! -L "$log" || return 2
  if tmux has-session -t "=$session" 2>/dev/null; then
    printf '拒绝复用会话 %s\n' "$session" >&2
    return 2
  fi
  local body='set -o pipefail
p1_session=$1
p1_head=$2
p1_cwd=$3
p1_log=$4
p1_command_file=$5
p1_command_sha=$6
p1_wrapper_pid=$BASHPID
shift 6
( set -o noclobber; : > "$p1_log" ) || exit 2
(
  printf "SESSION=%s\nWRAPPER_PID=%s\nBODY_PID=%s\nTRAIN_HEAD=%s\nCOMMAND_FILE=%s\nCOMMAND_FILE_SHA256_EXPECTED=%s\nSTART_UTC=%s\n" \
    "$p1_session" "$p1_wrapper_pid" "$BASHPID" "$p1_head" "$p1_command_file" "$p1_command_sha" "$(date -u +%FT%TZ)"
  p1_actual_sha=$(sha256sum -- "$p1_command_file") || exit 2
  p1_actual_sha=${p1_actual_sha%% *}
  printf "COMMAND_FILE_SHA256_ACTUAL=%s\n" "$p1_actual_sha"
  [[ "$p1_actual_sha" == "$p1_command_sha" ]] || { printf "COMMAND_FILE_SHA256_MISMATCH\n" >&2; exit 2; }
  cd "$p1_cwd" || exit 2
  printf "COMMAND="
  printf "%q " "$@"
  printf "\n"
  "$@"
) 2>&1 | tee "$p1_log"
codes=("${PIPESTATUS[@]}")
rc=${codes[0]}
if [ "${codes[1]}" -ne 0 ]; then rc=${codes[1]}; fi
printf "SESSION_END=%s\nTRAIN_HEAD_END=%s\nEND_UTC=%s\nCOMMAND_EXIT=%s\nTEE_EXIT=%s\n" \
  "$p1_session" "$p1_head" "$(date -u +%FT%TZ)" "${codes[0]}" "${codes[1]}" | tee -a "$p1_log"
ending=("${PIPESTATUS[@]}")
if [ "${ending[0]}" -ne 0 ]; then rc=${ending[0]}; fi
if [ "${ending[1]}" -ne 0 ]; then rc=${ending[1]}; fi
if ! printf "FOOTER_PRINTF_EXIT=%s\nFOOTER_TEE_EXIT=%s\nEXIT_CODE=%s\n" \
  "${ending[0]}" "${ending[1]}" "$rc" >> "$p1_log"; then
  printf "最终退出记录追加失败：%s\n" "$p1_log" >&2
  exit 1
fi
exit "$rc"'
  local command_file="$COMMAND_ROOT/$session.command"
  mkdir -p "$COMMAND_ROOT" || return 2
  (
    set -o noclobber
    {
      printf '%q ' bash -c "$body" -- "$session" "$TRAIN_HEAD" "$cwd" "$log"
      printf '"$0" "$1" '
      printf '%q ' "$@"
      printf '\n'
    } > "$command_file"
  ) || return 2
  local command_sha
  command_sha=$(sha256sum -- "$command_file") || return 2
  command_sha=${command_sha%% *}
  printf 'COMMAND_FILE=%s\nCOMMAND_FILE_SHA256=%s\n' "$command_file" "$command_sha"
  local command
  printf -v command '%q ' bash "$command_file" "$command_sha"
  tmux new-session -d -s "$session" "$command"
}

require_p1_log() {
  local log="$1" tentative="$2" key
  test -f "$log" || return 1
  for key in COMMAND_EXIT TEE_EXIT FOOTER_PRINTF_EXIT FOOTER_TEE_EXIT EXIT_CODE; do
    test "$(rg -c "^${key}=" "$log")" = 1 || return 1
    rg -q "^${key}=0$" "$log" || return 1
  done
  test "$(rg -c '^ENTRY_RUN=' "$log")" = 1 || return 1
  rg -q '^ENTRY_RUN=OK$' "$log" || return 1
  test "$(rg -c '^ENTRY_FINITE=' "$log")" = 1 || return 1
  rg -q '^ENTRY_FINITE=PASS$' "$log" || return 1
  test "$(rg -c '^SEGMENTS ' "$log")" = 1 || return 1
  rg -q "^SEGMENTS tentative_rows=$tentative main_rows=2$" "$log" || return 1
}
```

`TRAIN_HEAD/END` 是声明的B侧锚点，不是假称实时读取；各侧实际HEAD/clean起止证据由harness的两份provenance产生。`launch_p1` 返回只说明tmux会话创建成功，不能据此推进下一阶段。`require_p1_log`仅核日志，不代替下文JSON、来源及不落权重检查。

## 分阶段待执行命令

第一阶段：正式前置均满足后，A-full与A-count可先后发出并并行运行。两项均只运行官方入口的tentative2/main2。

```bash
launch_p1 "orig80k-p1-full-a-$P1_TAG" "$STORE/logs/orig80k-p1-full-a-$P1_TAG.log" "$MAIN" \
  "${P1_ENV[@]}" "${FULL_ENV[@]}" GPUS=0,1,2,3 EXECUTION_ROLE=upstream bash "$DRIVER" run-a
launch_p1 "orig80k-p1-count-a-$P1_TAG" "$STORE/logs/orig80k-p1-count-a-$P1_TAG.log" "$MAIN" \
  "${P1_ENV[@]}" "${COUNT_ENV[@]}" GPUS=4,5,6,7 EXECUTION_ROLE=upstream bash "$DRIVER" run-a
```

第二阶段：必须先确认两个A会话及其harness均已结束，两个A日志和下述JSON验收全部通过；重新记录没有本轮其他GPU负载后，B-full独占运行。不把下面日志检查成功当作省略其余验收的授权。

```bash
require_p1_log "$STORE/logs/orig80k-p1-full-a-$P1_TAG.log" 2 || exit 2
require_p1_log "$STORE/logs/orig80k-p1-count-a-$P1_TAG.log" 2 || exit 2
launch_p1 "orig80k-p1-full-b-$P1_TAG" "$STORE/logs/orig80k-p1-full-b-$P1_TAG.log" "$MAIN" \
  "${P1_ENV[@]}" "${FULL_ENV[@]}" GPUS=0,1,2,3 EXECUTION_ROLE=solo bash "$DRIVER" run-b
```

第三阶段：B-full已退出并完成全部验收后，再单独运行B-count。B两侧各自仅main2；P1的 `execution_role=solo` 不使其成为后续100步单跑基线。

```bash
require_p1_log "$STORE/logs/orig80k-p1-full-b-$P1_TAG.log" 0 || exit 2
launch_p1 "orig80k-p1-count-b-$P1_TAG" "$STORE/logs/orig80k-p1-count-b-$P1_TAG.log" "$MAIN" \
  "${P1_ENV[@]}" "${COUNT_ENV[@]}" GPUS=4,5,6,7 EXECUTION_ROLE=solo bash "$DRIVER" run-b
```

末侧结束后检查 `require_p1_log "$STORE/logs/orig80k-p1-count-b-$P1_TAG.log" 0`，再汇总四侧JSON与所有退出记录。监听只使用本轮精确日志；示例 `tail -n +1 -F "$STORE/logs/orig80k-p1-full-a-$P1_TAG.log" | grep --line-buffered -E 'SEGMENTS|ENTRY_|TrainState|Traceback|Error|EXIT_CODE='`。保留完整原始日志及去tqdm后的清洗版，失败、摘要、分段和终态不能丢失。

## P1验收：当前工具的实际文件和字段

每组记录位于 `$P1_ROOT/{full,count}/{a,b}`。以下是已有量具字段，不是新增或假定已有的“P1 judge”接口：

| 文件/判定 | A要求 | B要求 |
|---|---|---|
| `run_status.json` | `exit_code=0`、`entry_run_ok=true` | 同左 |
| `metrics_all.jsonl` | 4行，step顺序 `[0,1,0,1]` | 2行，`[0,1]` |
| `metrics_tentative.jsonl` | 2行，`[0,1]` | 不存在（非空即失败） |
| `metrics.jsonl` | main恰2行，`[0,1]` | 同左 |
| `state_digests.jsonl` | 两行，step `[1,1]`，按出现顺序保留两段 | 一行，step `[1]` |
| `init_ownership.json` | `upstream=true`、2次events，首次effective_overwrite=false，第二次仅自身目录true | `upstream=false`、1次events、effective_overwrite=false |
| 日志 | 唯一 `SEGMENTS tentative_rows=2 main_rows=2`、`ENTRY_FINITE=PASS`、`ENTRY_RUN=OK` | tentative_rows=0，其余同左 |

每行metrics完整含 `loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm`，各自 `finite=true` 且dec/hex一致；状态每行 `finite=true`、`n_leaves>0`、`finite_leaves=n_leaves`、`nonfinite_keys=[]`，完整保留 `strict_global/g0_global/treedef_sha/keyset_sha/per_leaf_strict/per_leaf_g0`。harness会拒绝非有限值，但P1不比较A/B的摘要是否相等。A两个相同step编号是官方双段的预期行为，不能去重或交给100步judge。

`provenance_start.json` 与 `provenance.json` 的 `git_head_of_cwd` 分别固定上游SHA/TRAIN_HEAD，`git_porcelain_of_cwd`为空、`git_root/cwd/expect_root`均为对应根，entry和全部项目module路径/SHA真实来自对应侧；B不得混入A worktree。`runtime_fingerprint.json` / `runtime_fingerprint_end.json` 完全一致，核实设备四卡、JAX x64=false、确定性flags、Python/包版本、同组source/manifest/norm/tokenizer/初始化索引与history摘要。`resolved_config.json.fields`明确为2步、b64、workers4、fsdp4、seed42、log1/save50、wandb_enabled=False及modul，原warmup/lr/EMA/keep未改；不能只看文件存在。

`harness_meta.json`保留实际entry、argv_tail、expect_root/forbid_roots、expect_steps=2、execution_role、harness SHA与PID；`jax_cache.json.actual`应为本侧新cache、calls记录上游/当前入口请求及实际转接。检查四个checkpoint工作根不含数字step checkpoint目录、`params`大权重载荷或真实保存完成标记，日志对应次数均为「TrainState 摘要已记…不落真权重」。不得调用 `check_orig80k_completion.py` 冒充P1保存恢复验收；P1本来不落真实权重。

所有工具成功行之外，还必须核对每侧唯一 `COMMAND_EXIT=0`、`TEE_EXIT=0`、`FOOTER_PRINTF_EXIT=0`、`FOOTER_TEE_EXIT=0`、`EXIT_CODE=0`，缺失、重复或非零均不通过。将完整命令与文件SHA、所有小JSON/JSONL、清洗日志、来源/资源快照归档；源码/驱动由最终TRAIN_HEAD还原，不复制独立.sh/.yaml/运行时.command文件。P1全部验收通过后，才按[100步正式起跑档案](train100-launch.md)进入下一阶段；不自动启动或清理任何已有产物。

## 提交与起跑前尚需核实

1. 本文件尚未填实际TRAIN_HEAD/P1_TAG。两组INPUT实际版本、路径、覆盖及摘要已登记，但到TRAIN_HEAD的工具/项目模块/依赖/数据资产指纹沿用复核仍须执行方在起跑前实查并留证，不用三样本或20步可读性代替。
2. 核对四个拟名称及所有输出/cache/log目录确实不存在、活动进程/磁盘预算允许本轮顺序。现有harness来源及finite检查不能替代现场GPU空闲、预算和资产锁预检。
3. 正式launch须复核现有驱动及harness在最终TRAIN_HEAD的实现，尤其B的独立uv环境、A仅覆盖本次tentative目录、两段状态完整保留和唯一终态协议。若实际实现或现场条件不能满足本文任一项，立即报告；不自行减小参数、改变保存接口或放宽判据。

本文件是正式起跑前记录；当前只核对文档、静态接口/参数与Bash语法，P1仍未执行，没有真实P1通过或训练等价结论。
