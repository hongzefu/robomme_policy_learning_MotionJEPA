# 原版80k训练100步2a/2b：正式起跑前记录（未运行）

**P1及100步训练对拍尚未运行，当前没有100步或并跑等价结论。** 两组正式CPU输入已在真实INPUT_HEAD **`3a1582db39c723c735e04752e5027bfe40ecc3e1`** 完成并分别通过，全部会话已结束。该提交也是本文已有接口的静态核对基线，不能当作尚待创建的TRAIN_HEAD。本文件与[p1-launch.md](p1-launch.md)及本轮工具补齐须共同形成clean Beta；实际100步TRAIN_HEAD在该提交创建后填40位字面量，具体标签、名称与命令展开在起跑现场记录。

依据[主计划第2步与C节](../../../0925-orig-80k-full-counting-4plus4-plan.md)和[P1正式起跑档案](p1-launch.md)，在已完成INPUT的跨版本沿用复核及P1实际验收通过后，再按“两组上游A并行 → A全部完成 → B-full独占S1 → B-count独占S3 → 新B-full+B-count并行S2”执行。2a比较上游A与各自B单跑；2b复用S1/S3，分别比较同入口S2。六次训练及四份judge的B侧代码和共用harness必须始终属于同一 `TRAIN_HEAD`；A项目源码固定上游 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，不把它误写为B侧HEAD。

用户已批准「允许仅对拍关闭 W&B」；P1/100步A/B关闭在线W&B，完整五标量及参数/EMA/优化器/step状态留在本地。20步可读性、300步perf和80k仍开启。这里只在对拍入口覆盖100步、log1/save50，保持global b64/fsdp4/workers4/seed42、modul 512/4×4、原warmup/lr/EMA/keep和确定性档。perf磁盘采样虽已获准，本阶段不运行perf，不把状态摘要器当作真实checkpoint保存。

## 1. 真实入口和100步状态口径

现有 `scripts/training/tests/run_entry_equiv.sh` 的 `run-a/run-b/judge` 调用 `entry_equiv.py::_run()/_judge()`。A使用独立uv解释器执行官方 `scripts/train.py::__main__`：tentative的step0…11共12步，随后官方20秒间隔和main0…99；B只执行当前 `scripts/training/train.py` 的main0…99。

上游100步tentative在step11命中 `tentative_run and step > tentative_run_step`，在保存分支之前break，因此该段没有state摘要；main仅在50和99调用摘要器。每份100步run的 `state_digests.jsonl` 应恰为两行、step顺序 `[50,99]`，不能照抄P1的A侧 `[1,1]` 或人为删除额外行来通过。

harness临时替换 `openpi.training.checkpoints.save_state`，记录真实 `params/ema_params/opt_state/step` 的逐叶dtype、shape、原始字节摘要和有限性，**不调用原save_state、不落真实权重**。仍会创建checkpoint工作目录；`_CheckpointOwnership`只允许A正式段重新初始化本进程刚创建、实体路径与inode未变的tentative目录，首轮A和所有B均拒绝旧目录。它不授权删除旧run，也不能使用完成器 `check_orig80k_completion.py` 为本阶段冒充真实恢复验收。

## 2. 已完成INPUT与其他起跑前置

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

P1仍未运行，未来TRAIN_HEAD及本轮名称仍待填。以下其余条件全部满足后，才能使用后文阶段命令；几个目录存在或只看到一个PASS字符串不算验收完成。

| 前置 | 正式起跑前要核对并留档 |
|---|---|
| 两库及资产 | full1600/768897帧/476857样本，count400/189035样本；来源pin、packed `verified/full`、无pack.lock、严格SUBSET零失配；norm/history/初始化权重与tokenizer资产锁及实际摘要 |
| 正式INPUT两组 | 已完成：上表已登记实际INPUT_HEAD、四侧record_manifest与两份judge日志SHA、各组覆盖；原始记录保持该版本，起跑前复核其文件清单和摘要完整 |
| INPUT跨HEAD沿用 | 若TRAIN_HEAD与真实INPUT_HEAD不同，按已提交INPUT launch核对取证工具、输入模块、依赖、数据和资产指纹与固定提交差异；不改写原记录HEAD，不能仅重新judge就声称新版本重采 |
| P1四侧 | 登记实际P1版本、记录/日志路径与SHA；A各tentative2+main2，B各main2，有限值/来源/配置通过；A状态原始两行 `[1,1]`、B一行 `[1]`，真实 `run_status`和强wrapper终态通过；P1不运行轨迹judge |
| 正式100步档案 | 本文件与p1-launch.md及本轮工具补齐须先共同提交clean Beta；TRAIN_HEAD填该提交完整字面量，运行期间不能修改tracked或共享.venv |
| 新输出和资源 | 六run、四judge、各自日志/命令/cache/记录/checkpoint工作根全新；八卡设备UUID、空闲和其他进程、内存/SHM、`df -B1 /scratch`及本轮摘要/缓存/临时输出预算现场留证，仍满足计划空间门槛 |

P1与100步的量具/入口/依赖若在两阶段之间发生影响行为的变化，先报告并按原闸门确定复测范围；不能自行声称旧P1已验证新实现。INPUT结束后已解除其运行冻结以补齐工具和档案，P1及100步各自正式起跑仍必须从clean Beta开始并冻结源码和两侧环境。所有缺项、字段差异或非零终态均先停在对应前置。

## 3. 数据、资产、名称与版本绑定

| 组 | A source / B packed（相对主仓库v1-store/datasets） | norm父目录（相对v1-store/train-assets） | norm SHA256 |
|---|---|---|---|
| full | `16task-pub-1600ep/source` / `16task-pub-1600ep/framesamp` | `mme_vla_suite` | `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5` |
| count | `4task-counting-pub-400ep/source` / `4task-counting-pub-400ep/framesamp` | `mme_vla_suite/4task-counting-pub-400ep` | `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9` |

full规范manifest SHA为 `fb1bdbcbcc176d15475332b728fa218bb545b4a8c9dcaa9ce1ec9e0bb17f174f`，count为 `6b309822aa604a31f6eb0dfb5f02d5482573dc5fa11dc5c4f917880aa92ba986`；文件自身SHA分别为 `222d84f58fb82073603fda19992c4f82e542475294176bea99e8868036b8a11a`、`fa0535335751910451c483d198cda151f04d25df36b75467c7460d118e25ad52`。两侧history均为 `perceptual-framesamp-modul.yaml`，SHA `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`。norm的assets-dir传上表父目录，asset-id为robomme，不再追加一层robomme。

初始化为 `$STORE/models/openpi-assets/checkpoints/pi05_base/params`，tokenizer为 `$STORE/models/big_vision/paligemma_tokenizer.model`。harness只hash初始化原生索引，不能用它替代前置完整资产锁校验；A/B共享只读资产，不下载或同步依赖。

候选exp_name使用独立100步前缀；`EQ100_TAG`在正式起跑现场选唯一UTC标签：

| 训练任务 | exp_name候选 | 记录相对EQ100_ROOT | GPU / role |
|---|---|---|---|
| 上游full A | `eq100-orig80k-full-a-$EQ100_TAG` | `full/a` | 0–3 / upstream |
| 上游count A | `eq100-orig80k-count-a-$EQ100_TAG` | `count/a` | 4–7 / upstream |
| B-full S1 | `eq100-orig80k-full-s1-$EQ100_TAG` | `full/solo` | 0–3 / solo |
| B-count S3 | `eq100-orig80k-count-s3-$EQ100_TAG` | `count/solo` | 4–7 / solo |
| B-full S2 | `eq100-orig80k-full-s2-$EQ100_TAG` | `full/concurrent` | 0–3 / concurrent |
| B-count S2 | `eq100-orig80k-count-s2-$EQ100_TAG` | `count/concurrent` | 4–7 / concurrent |

这些不是已经创建的run。S1/S3与各自S2必须使用相同物理GPU UUID、同TRAIN_HEAD/模块、依赖、数据/资产和有效计算环境；只有已登记的run身份、输出与cache路径差别可接受。A两组可并行；S1、S3必须依次独占本轮GPU训练阶段，记录机器其他负载，不并行本轮建库、perf或其它训练。

## 4. 待填公共环境与新输出守卫

代码块仅供审阅和分阶段执行，不得整页自动运行。命令占位为 `TRAIN_HEAD` 与 `EQ100_TAG`，前置INPUT/P1实际证据另按第2节填实。下面新根检查只在本轮第一次准备时执行；创建产物后不重放整页、不自动换标签重跑失败任务。

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
UP="$STORE/worktrees/orig-ecf086c"
UPSTREAM_HEAD=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b
TRAIN_HEAD='<待两份launch及工具补齐的clean Beta提交后填写40位Git SHA>'
EQ100_TAG='<本轮唯一UTC标签>'
EQ100_ROOT="$STORE/bench/orig80k-eq100-$EQ100_TAG"
COMMAND_ROOT="$STORE/bench/orig80k-eq100-commands-$EQ100_TAG"
EQ100_CACHE="$STORE/cache/jax/orig80k-eq100-$EQ100_TAG"
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
[[ "$EQ100_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
require_eq100_head() {
  test "$(git -C "$MAIN" rev-parse HEAD)" = "$TRAIN_HEAD" || return 1
  test -z "$(git -C "$MAIN" status --porcelain)" || return 1
  test "$(git -C "$UP" rev-parse HEAD)" = "$UPSTREAM_HEAD" || return 1
  test -z "$(git -C "$UP" status --porcelain)" || return 1
}
require_eq100_head || exit 2
test -x "$UP/.venv/bin/python" && test -x "$MAIN/.venv/bin/python" || exit 2
for parent in "$STORE" "$STORE/bench" "$STORE/cache" "$STORE/cache/jax" "$STORE/logs"; do
  test -d "$parent" && test "$(realpath "$parent")" = "$parent" || exit 2
done
for path in "$EQ100_ROOT" "$COMMAND_ROOT" "$EQ100_CACHE"; do
  test ! -e "$path" && test ! -L "$path" || exit 2
done
for side in full-a count-a full-s1 count-s3 full-s2 count-s2; do
  exp="eq100-orig80k-$side-$EQ100_TAG"
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

EQ100_ENV=(env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME
  -u JAX_COMPILATION_CACHE_DIR -u TRAIN_RECORD_DIR -u TRAIN_FINAL_RECORD_DIR -u TRAIN_TIMING_STEPS
  JAX_ENABLE_X64=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
  WORKTREE="$UP" A_PYTHON="$UP/.venv/bin/python" UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
  HEAD_A="$UPSTREAM_HEAD" HEAD_B="$TRAIN_HEAD"
  STEPS=100 BATCH=64 FSDP=4 SAVE_INTERVAL=50 HISTORY_YAML="$HISTORY_YAML"
  MODE=upstream EXPECT_TENTATIVE_A=12 EXPECT_STATE_STEPS=50,99
  ANCHOR_SHA256= CONCURRENT_PEER_DIR=)
FULL_ENV=(RECORD_ROOT="$EQ100_ROOT/full"
  RECORD_A="$EQ100_ROOT/full/a" RECORD_B="$EQ100_ROOT/full/solo"
  EXP_A="eq100-orig80k-full-a-$EQ100_TAG" EXP_B="eq100-orig80k-full-s1-$EQ100_TAG"
  DATA_A="$FULL_SOURCE" DATA_B="$FULL_DATA" ASSETS_DIR="$FULL_ASSETS"
  FRAMESAMP_SOURCE="$FULL_SOURCE" FRAMESAMP_MANIFEST="$FULL/meta/episode_manifest.json"
  CHECKPOINT_A="$EQ100_ROOT/checkpoints/full-a" CHECKPOINT_B="$EQ100_ROOT/checkpoints/full-s1"
  JAX_CACHE_A="$EQ100_CACHE/full-a" MMEVLA_JAX_CACHE_DIR="$EQ100_CACHE/full-s1")
COUNT_ENV=(RECORD_ROOT="$EQ100_ROOT/count"
  RECORD_A="$EQ100_ROOT/count/a" RECORD_B="$EQ100_ROOT/count/solo"
  EXP_A="eq100-orig80k-count-a-$EQ100_TAG" EXP_B="eq100-orig80k-count-s3-$EQ100_TAG"
  DATA_A="$COUNT_SOURCE" DATA_B="$COUNT_DATA" ASSETS_DIR="$COUNT_ASSETS"
  FRAMESAMP_SOURCE="$COUNT_SOURCE" FRAMESAMP_MANIFEST="$COUNT/meta/episode_manifest.json"
  CHECKPOINT_A="$EQ100_ROOT/checkpoints/count-a" CHECKPOINT_B="$EQ100_ROOT/checkpoints/count-s3"
  JAX_CACHE_A="$EQ100_CACHE/count-a" MMEVLA_JAX_CACHE_DIR="$EQ100_CACHE/count-s3")
FULL_S2_ENV=("${FULL_ENV[@]}" RECORD_B="$EQ100_ROOT/full/concurrent"
  EXP_B="eq100-orig80k-full-s2-$EQ100_TAG"
  CHECKPOINT_B="$EQ100_ROOT/checkpoints/full-s2" MMEVLA_JAX_CACHE_DIR="$EQ100_CACHE/full-s2")
COUNT_S2_ENV=("${COUNT_ENV[@]}" RECORD_B="$EQ100_ROOT/count/concurrent"
  EXP_B="eq100-orig80k-count-s2-$EQ100_TAG"
  CHECKPOINT_B="$EQ100_ROOT/checkpoints/count-s2" MMEVLA_JAX_CACHE_DIR="$EQ100_CACHE/count-s2")
```

驱动固定设置确定性 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、显存比例0.95、`CUDA_VISIBLE_DEVICES=$GPUS`、`UV_LINK_MODE=copy`、`UV_CACHE_DIR=$STORE/cache/uv`、`OPENPI_DATA_HOME=$STORE/models`、`XDG_CACHE_HOME=$STORE/cache`及`HF_HOME=$STORE/cache/huggingface`。CUDA cache按exp_name落 `$STORE/cache/cuda/<exp>`；W&B目录按exp_name分别落 `$STORE/entryeq/wandb`、`cache/wandb`、`cache/wandb-config`、`cache/wandb-data`，即使在线W&B关闭也不复用旧路径。A/B的JAX cache各自独立，由harness转接入口的cache设置并记录 `jax_cache.json`，不覆盖HOME。

A由uv管理的独立 `$UP/.venv/bin/python`执行上游入口，清除packed来源变量；B由主仓库uv环境执行，`UV_PROJECT_ENVIRONMENT=$MAIN/.venv`且`--no-sync`，显式绑定本组source/manifest。CPU平台变量在GPU命令外层清除，JAX_ENABLE_X64明确为0。其它影响计算的继承变量（如MKL线程、matmul precision、TF32、PYTHONHASHSEED）按实际指纹记录，同组各run须一致；发现不一致停止，不自行加未约定覆盖来修饰结果。

## 5. 保存命令文件与完整终态

沿用[P1正式起跑档案](p1-launch.md)中与CPU方案一致的强wrapper协议：展开后的同一命令先独占写入ignored命令文件，现场记录其SHA，再交tmux执行并重核SHA。主体和末尾printf/tee的真实状态均留证，检查末尾tee之后才追加唯一综合EXIT_CODE；不能只看某个工具PASS。下面仍是待执行代码，本次正式落档没有启动它或把静态检查作为训练实测。

```bash
launch_eq100() {
  local session="$1" log="$2" cwd="$3"
  shift 3
  test ! -e "$log" && test ! -L "$log" || return 2
  if tmux has-session -t "=$session" 2>/dev/null; then
    printf '拒绝复用会话 %s\n' "$session" >&2
    return 2
  fi
  local body='set -o pipefail
eq100_session=$1
eq100_head=$2
eq100_cwd=$3
eq100_log=$4
eq100_command_file=$5
eq100_command_sha=$6
eq100_wrapper_pid=$BASHPID
shift 6
( set -o noclobber; : > "$eq100_log" ) || exit 2
(
  printf "SESSION=%s\nWRAPPER_PID=%s\nBODY_PID=%s\nTRAIN_HEAD=%s\nCOMMAND_FILE=%s\nCOMMAND_FILE_SHA256_EXPECTED=%s\nSTART_UTC=%s\n" \
    "$eq100_session" "$eq100_wrapper_pid" "$BASHPID" "$eq100_head" "$eq100_command_file" "$eq100_command_sha" "$(date -u +%FT%TZ)"
  eq100_actual_sha=$(sha256sum -- "$eq100_command_file") || exit 2
  eq100_actual_sha=${eq100_actual_sha%% *}
  printf "COMMAND_FILE_SHA256_ACTUAL=%s\n" "$eq100_actual_sha"
  [[ "$eq100_actual_sha" == "$eq100_command_sha" ]] || { printf "COMMAND_FILE_SHA256_MISMATCH\n" >&2; exit 2; }
  cd "$eq100_cwd" || exit 2
  printf "COMMAND="
  printf "%q " "$@"
  printf "\n"
  "$@"
) 2>&1 | tee "$eq100_log"
codes=("${PIPESTATUS[@]}")
rc=${codes[0]}
if [ "${codes[1]}" -ne 0 ]; then rc=${codes[1]}; fi
printf "SESSION_END=%s\nTRAIN_HEAD_END=%s\nEND_UTC=%s\nCOMMAND_EXIT=%s\nTEE_EXIT=%s\n" \
  "$eq100_session" "$eq100_head" "$(date -u +%FT%TZ)" "${codes[0]}" "${codes[1]}" | tee -a "$eq100_log"
ending=("${PIPESTATUS[@]}")
if [ "${ending[0]}" -ne 0 ]; then rc=${ending[0]}; fi
if [ "${ending[1]}" -ne 0 ]; then rc=${ending[1]}; fi
if ! printf "FOOTER_PRINTF_EXIT=%s\nFOOTER_TEE_EXIT=%s\nEXIT_CODE=%s\n" \
  "${ending[0]}" "${ending[1]}" "$rc" >> "$eq100_log"; then
  printf "最终退出记录追加失败：%s\n" "$eq100_log" >&2
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

require_eq100_exit() {
  local log="$1" key
  test -f "$log" && test ! -L "$log" || return 1
  for key in COMMAND_EXIT TEE_EXIT FOOTER_PRINTF_EXIT FOOTER_TEE_EXIT EXIT_CODE; do
    test "$(rg -c "^${key}=" "$log")" = 1 || return 1
    rg -q "^${key}=0$" "$log" || return 1
  done
}

require_eq100_run_log() {
  local log="$1" tentative="$2"
  require_eq100_exit "$log" || return 1
  test "$(rg -c '^ENTRY_RUN=' "$log")" = 1 || return 1
  rg -q '^ENTRY_RUN=OK$' "$log" || return 1
  test "$(rg -c '^ENTRY_FINITE=' "$log")" = 1 || return 1
  rg -q '^ENTRY_FINITE=PASS$' "$log" || return 1
  test "$(rg -c '^SEGMENTS ' "$log")" = 1 || return 1
  rg -q "^SEGMENTS tentative_rows=$tentative main_rows=100$" "$log" || return 1
}

require_eq100_judge_log() {
  local log="$1" mode="$2"
  require_eq100_exit "$log" || return 1
  for key in ENTRY_EQ ENTRY_FINITE ENTRY_PROVENANCE; do
    test "$(rg -c "^${key}=" "$log")" = 1 || return 1
    rg -q "^${key}=PASS$" "$log" || return 1
  done
  test "$(rg -c '^ENTRY_SCALARS ' "$log")" = 1 || return 1
  rg -q '^ENTRY_SCALARS steps=100 keys=5 hex_mismatch=0$' "$log" || return 1
  test "$(rg -c '^ENTRY_STATE_DIGEST ' "$log")" = 1 || return 1
  rg -q '^ENTRY_STATE_DIGEST rows=2 mismatch=0$' "$log" || return 1
  test "$(rg -c '^ENTRY_RESOLVED_CFG ' "$log")" = 1 || return 1
  rg -q "^ENTRY_RESOLVED_CFG mismatch=0 mode=$mode$" "$log" || return 1
  if [ "$mode" = same-entry ]; then
    test "$(rg -c '^ENTRY_CONCURRENT=' "$log")" = 1 || return 1
    rg -q '^ENTRY_CONCURRENT=PASS ' "$log" || return 1
  elif [ "$mode" != upstream ]; then
    return 1
  fi
}
```

`TRAIN_HEAD/END`是声明的B侧锚点，实际起止HEAD/clean由harness provenance另验；不能把声明字段当作一次新的git观测。`launch_eq100`返回只代表tmux创建，`require_eq100_*_log`仅是终态检查，不代替第8节全部JSON/来源/内容验收。每个stage推进前，还须确认对应完整tmux名称和harness PID已结束；会话消失本身不算成功。

## 6. 2a：两组A完成后依次S1、S3，再判上游等价

先完成全部前置，记录八卡、主机负载与磁盘，启动两组A。两个命令可连续发出，互不等待；A各自执行tentative12+main100。

```bash
require_eq100_head || exit 2
launch_eq100 "orig80k-eq100-full-a-$EQ100_TAG" "$STORE/logs/orig80k-eq100-full-a-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${FULL_ENV[@]}" GPUS=0,1,2,3 EXECUTION_ROLE=upstream bash "$DRIVER" run-a
launch_eq100 "orig80k-eq100-count-a-$EQ100_TAG" "$STORE/logs/orig80k-eq100-count-a-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${COUNT_ENV[@]}" GPUS=4,5,6,7 EXECUTION_ROLE=upstream bash "$DRIVER" run-a
```

等两个A的日志、JSON、分段、状态、finite和来源全部验收通过，且两个A真实进程均已结束后，重新核对本轮没有其他GPU任务，再独占运行S1。下面仅列日志与HEAD的可执行检查，不能据此省略上述验收和资源观察。

```bash
require_eq100_run_log "$STORE/logs/orig80k-eq100-full-a-$EQ100_TAG.log" 12 || exit 2
require_eq100_run_log "$STORE/logs/orig80k-eq100-count-a-$EQ100_TAG.log" 12 || exit 2
require_eq100_head || exit 2
launch_eq100 "orig80k-eq100-full-s1-$EQ100_TAG" "$STORE/logs/orig80k-eq100-full-s1-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${FULL_ENV[@]}" GPUS=0,1,2,3 EXECUTION_ROLE=solo bash "$DRIVER" run-b
```

S1完成、进程退出、全部记录核验后，再独占运行S3。S1就是后续full的单跑基线，不另造或覆盖一份“看起来相同”的记录。

```bash
require_eq100_run_log "$STORE/logs/orig80k-eq100-full-s1-$EQ100_TAG.log" 0 || exit 2
require_eq100_head || exit 2
launch_eq100 "orig80k-eq100-count-s3-$EQ100_TAG" "$STORE/logs/orig80k-eq100-count-s3-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${COUNT_ENV[@]}" GPUS=4,5,6,7 EXECUTION_ROLE=solo bash "$DRIVER" run-b
```

S3完成并通过全部验收后，运行两组upstream judge。这里A是上游记录，B是刚才的solo记录；显式保留异根判据、上游HEAD及tentative12，不能从same-entry设置反向继承。两个judge各自单独日志和tmux，可并行；必须都完成成功才进入S2。

```bash
require_eq100_run_log "$STORE/logs/orig80k-eq100-full-s1-$EQ100_TAG.log" 0 || exit 2
require_eq100_run_log "$STORE/logs/orig80k-eq100-count-s3-$EQ100_TAG.log" 0 || exit 2
require_eq100_head || exit 2
launch_eq100 "orig80k-eq100-full-upstream-judge-$EQ100_TAG" "$STORE/logs/orig80k-eq100-full-upstream-judge-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${FULL_ENV[@]}" JAX_PLATFORMS=cpu GPUS=0,1,2,3 \
  MODE=upstream HEAD_A="$UPSTREAM_HEAD" HEAD_B="$TRAIN_HEAD" \
  EXPECT_TENTATIVE_A=12 EXPECT_STATE_STEPS=50,99 ANCHOR_SHA256= CONCURRENT_PEER_DIR= \
  RECORD_A="$EQ100_ROOT/full/a" RECORD_B="$EQ100_ROOT/full/solo" bash "$DRIVER" judge
launch_eq100 "orig80k-eq100-count-upstream-judge-$EQ100_TAG" "$STORE/logs/orig80k-eq100-count-upstream-judge-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${COUNT_ENV[@]}" JAX_PLATFORMS=cpu GPUS=4,5,6,7 \
  MODE=upstream HEAD_A="$UPSTREAM_HEAD" HEAD_B="$TRAIN_HEAD" \
  EXPECT_TENTATIVE_A=12 EXPECT_STATE_STEPS=50,99 ANCHOR_SHA256= CONCURRENT_PEER_DIR= \
  RECORD_A="$EQ100_ROOT/count/a" RECORD_B="$EQ100_ROOT/count/solo" bash "$DRIVER" judge
```

`MODE=upstream`展开为 `judge --mode upstream --expect-steps 100 --expect-tentative-a 12 --expect-state-steps 50,99 --expect-head-a <上游SHA> --expect-head-b <TRAIN_HEAD>`；显式空ANCHOR不会传 `--expect-sha256`，输出 `anchor=SKIPPED`，不借用历史1000步/A40锚点。

judge只处理已有记录，但现有实现会向两个记录目录写 `scalars_hex.tsv`，所以不能描述成完全只读。命令设置 `JAX_PLATFORMS=cpu`；驱动仍会按GPUS设置CUDA_VISIBLE_DEVICES，不能声称该变量为空，judge本身不启动训练或GPU计算。JSON中的原训练runtime指纹不由judge当前CPU环境代替。

## 7. 2b：新S2并行，再判同入口单跑/并跑

两份2a judge、所有四份上游/solo训练记录及强终态均PASS之后，再新起S2两侧；记录、checkpoint工作根和cache与S1/S3完全分开。两条命令连续发出，不等待其中一侧结束；实际训练重叠由后续judge验证。

```bash
require_eq100_judge_log "$STORE/logs/orig80k-eq100-full-upstream-judge-$EQ100_TAG.log" upstream || exit 2
require_eq100_judge_log "$STORE/logs/orig80k-eq100-count-upstream-judge-$EQ100_TAG.log" upstream || exit 2
require_eq100_head || exit 2
launch_eq100 "orig80k-eq100-full-s2-$EQ100_TAG" "$STORE/logs/orig80k-eq100-full-s2-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${FULL_S2_ENV[@]}" GPUS=0,1,2,3 EXECUTION_ROLE=concurrent bash "$DRIVER" run-b
launch_eq100 "orig80k-eq100-count-s2-$EQ100_TAG" "$STORE/logs/orig80k-eq100-count-s2-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${COUNT_S2_ENV[@]}" GPUS=4,5,6,7 EXECUTION_ROLE=concurrent bash "$DRIVER" run-b
```

等S2两侧均已退出并各自完成100步、finite/来源/状态及完整终态验收，再启动两个same-entry judge。其参数名A/B只是左右操作数：**A=对应solo基线，B=对应concurrent记录**，两侧expect-head都必须为TRAIN_HEAD，tentative期望0，peer是另一库的S2，不是自身、solo或上游记录。

| 判定 | MODE | HEAD_A / HEAD_B | RECORD_A → RECORD_B | CONCURRENT_PEER_DIR |
|---|---|---|---|---|
| full 2a | upstream | 上游 / TRAIN_HEAD | full/a → full/solo | 空 |
| count 2a | upstream | 上游 / TRAIN_HEAD | count/a → count/solo | 空 |
| full 2b | same-entry | TRAIN_HEAD / TRAIN_HEAD | full/solo → full/concurrent | count/concurrent |
| count 2b | same-entry | TRAIN_HEAD / TRAIN_HEAD | count/solo → count/concurrent | full/concurrent |

```bash
require_eq100_run_log "$STORE/logs/orig80k-eq100-full-s2-$EQ100_TAG.log" 0 || exit 2
require_eq100_run_log "$STORE/logs/orig80k-eq100-count-s2-$EQ100_TAG.log" 0 || exit 2
require_eq100_head || exit 2
launch_eq100 "orig80k-eq100-full-same-judge-$EQ100_TAG" "$STORE/logs/orig80k-eq100-full-same-judge-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${FULL_ENV[@]}" JAX_PLATFORMS=cpu GPUS=0,1,2,3 \
  MODE=same-entry HEAD_A="$TRAIN_HEAD" HEAD_B="$TRAIN_HEAD" \
  EXPECT_TENTATIVE_A=0 EXPECT_STATE_STEPS=50,99 ANCHOR_SHA256= \
  RECORD_A="$EQ100_ROOT/full/solo" RECORD_B="$EQ100_ROOT/full/concurrent" \
  CONCURRENT_PEER_DIR="$EQ100_ROOT/count/concurrent" bash "$DRIVER" judge
launch_eq100 "orig80k-eq100-count-same-judge-$EQ100_TAG" "$STORE/logs/orig80k-eq100-count-same-judge-$EQ100_TAG.log" "$MAIN" \
  "${EQ100_ENV[@]}" "${COUNT_ENV[@]}" JAX_PLATFORMS=cpu GPUS=4,5,6,7 \
  MODE=same-entry HEAD_A="$TRAIN_HEAD" HEAD_B="$TRAIN_HEAD" \
  EXPECT_TENTATIVE_A=0 EXPECT_STATE_STEPS=50,99 ANCHOR_SHA256= \
  RECORD_A="$EQ100_ROOT/count/solo" RECORD_B="$EQ100_ROOT/count/concurrent" \
  CONCURRENT_PEER_DIR="$EQ100_ROOT/full/concurrent" bash "$DRIVER" judge
```

`MODE=same-entry`只切换judge规则，不调用 `run-a`冒充左侧B基线；不要把 `WORKTREE`改成主仓库，否则run-b的 `--forbid-root`会错误指向自己。六个训练任务仍按各自真实根运行，B始终保留禁止导入A worktree的来源检查。env数组后的重复键是明确的本次覆盖，防止沿用EQ100_ENV中的上游HEAD、tentative12或full/a路径；四个judge的实际argv须现场留证并再次核对上表。

`_concurrent_overlap()`从各自main `metrics.jsonl.wall_time`构造step10…99的主机完成间隔 `[time(step-1), time(step)]`。两窗口交集须覆盖**两侧各至少50个完整间隔**，不是共50步，也不是两个进程同时存在即可。记录 `ENTRY_CONCURRENT=PASS` 的 `window_steps/begin/end/seconds/complete_steps`，同时绑定互不重叠的两组四卡UUID、同仓库及同HEAD。该时间不是GPU内核计算耗时或稳态吞吐。

若编译错开导致重叠不足，即使标量/状态全相同也不能判2b通过：保留本次无效证据并立即报告。计划允许独立同配置预热后重跑，但此处不自动启动预热或新一轮、不复用记录/cache或改阈值；先形成明确的新名称、版本和执行记录。

## 8. 每侧内容及四份judge的验收

| 记录/条件 | 上游A（两份） | B solo/S2（四份） |
|---|---|---|
| run_status | `exit_code=0, entry_run_ok=true` | 同左 |
| metrics_all | 112行：0…11后0…99 | 100行：0…99 |
| metrics_tentative | 恰12行、step0…11 | 不存在或空，非零行失败 |
| metrics | main恰100行、step0…99，原始行无重复/缺步 | 同左 |
| state_digests | 恰两行、step顺序50、99，无重复/缺行 | 同左 |
| init_ownership | upstream=true、2次；首次不覆盖，第二次仅本进程tentative | upstream=false、1次、effective_overwrite=false |
| harness role | upstream | S1/S3为solo，S2为concurrent |

每步五标量 `loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm`保留dec/hex、finite=true，十进制回转hex自洽。每行状态保留 `strict_global/g0_global/treedef_sha/keyset_sha/per_leaf_strict/per_leaf_g0`、n_leaves>0、finite_leaves=n_leaves、nonfinite_keys=[]、finite=true；原始状态包括参数、EMA、优化器和step。训练对拍仍严格逐位，不因CPU输入允许主机dtype不同而放宽训练结果。

`provenance_start.json/provenance.json`必须绑定实际A上游或B TRAIN_HEAD，clean，entry/root/cwd和全部已记录项目模块在所属根内；B不能落入A worktree。起止runtime_fingerprint一致，记录四卡UUID、Python/依赖/uv.lock、确定性flags/x64、history、source/manifest/pin、norm/tokenizer及初始化原生索引。A/B共用同份harness SHA；真正独立环境由sys.prefix检查，不能只看路径变量。resolved_config验证100步/b64/workers4/fsdp4/seed42/log1/save50/no-W&B/modul及原warmup/lr/EMA/keep，不能仅因配置文件存在就通过。

upstream判定的配置白名单为exp_name/dataset_path/checkpoint_base_dir/overwrite；入口路径差异另在来源中核对。同入口配置白名单仅exp_name/checkpoint_base_dir，数据、overwrite、模型和超参不在白名单；runtime指纹必须完全相同，cache路径的区别不被伪装成来源/资产差异。

每份judge须同时得到以下真实输出与强wrapper状态；任一缺失、重复、非有限或失配停止后续，不手工删行或改JSON：

```text
ENTRY_SCALARS steps=100 keys=5 hex_mismatch=0
ENTRY_STATE_DIGEST rows=2 mismatch=0
ENTRY_FINITE=PASS
ENTRY_RESOLVED_CFG mismatch=0 mode=upstream 或 same-entry
ENTRY_PROVENANCE=PASS
ENTRY_EQ=PASS
COMMAND_EXIT=0
TEE_EXIT=0
FOOTER_PRINTF_EXIT=0
FOOTER_TEE_EXIT=0
EXIT_CODE=0
```

same-entry还须唯一 `ENTRY_CONCURRENT=PASS`，并核对两侧完整间隔计数均≥50；upstream每侧scalar投影SHA应一致且标记anchor=SKIPPED。末两份judge结束且原始证据全部核验后，收尾执行以下日志与HEAD检查；当前不预填任何返回值。

```bash
require_eq100_judge_log "$STORE/logs/orig80k-eq100-full-same-judge-$EQ100_TAG.log" same-entry || exit 2
require_eq100_judge_log "$STORE/logs/orig80k-eq100-count-same-judge-$EQ100_TAG.log" same-entry || exit 2
require_eq100_head || exit 2
```

## 9. 观察、冻结、归档与失败交接

全部任务预计可能超过5分钟，均使用detached tmux、完整会话名、行缓冲日志观察。六次训练和四份judge都留命令文件路径/真实SHA、wrapper/body PID、harness PID、实际UTC、源HEAD/clean、GPU UUID和全部终态。只按完整名称 `tmux has-session -t '=完整会话名'`检查；示例监听 `tail -n +1 -F <日志> | grep --line-buffered -E 'SEGMENTS|ENTRY_|TrainState|Traceback|Error|EXIT_CODE='`，不全局kill，不把监听进程退出0当作包装退出证据。

同一轮从两个A起跑到四份judge结束期间，不改tracked、不同步两个.venv、不改数据/统计量、不调整环境。S1/S3复用必须满足同TRAIN_HEAD及全部指纹；中途改代码或依赖使基线失效，先报告，不临时略过same-entry的HEAD/来源判据。其余组失败时停止受依赖的后续阶段，保留仍运行任务的精确归属；任何终止/重跑另按已授权范围处理，不自动清理其他会话。

通过结论仅限本机A100确定性档和这两组前100步。归档六侧完整小JSON/JSONL、四份judge原始/清洗日志、所有阶段强终态与环境/资源快照；Git可还原源码和配置不复制独立.sh/.yaml/command正文。judge产生的scalars_hex.tsv也留档。原始日志拆CR、去进度态时核对关键行完整，不能改写结果；所有清洗/归档SHA区别记录。

先确认正式归档完整，再按计划和明确归属处理本轮临时checkpoint工作目录/cache；本阶段起跑文档不授权自动删除，也不把A独立worktree自动移除。本阶段不测真实保存峰值、不恢复checkpoint，不自动启动perf或80k。

## 10. Beta提交与正式起跑前仍需完成

1. 两组正式INPUT已取得本节记录的实际PASS；执行方仍须实查其到TRAIN_HEAD的输入工具/项目模块/依赖/数据资产指纹沿用条件。P1尚未运行，必须取得其真实通过记录并补齐路径、HEAD、SHA，不使用起跑文档或dry作为实跑证据。
2. 本P1/100步两份launch及工具补齐共同提交clean Beta后，才填写TRAIN_HEAD；EQ100_TAG与所有具体名称起跑时选定并检查全新。六次训练及四judge全过程冻结同一B侧锚点，A始终固定上游。
3. 复核最终TRAIN_HEAD的驱动/量具接口与静态基线 `3a1582db39c723c735e04752e5027bfe40ecc3e1` 一致，特别是same-entry记录映射、tentative12、state50/99、50完整间隔重叠与strong footer判据；变化或不能满足时立即报告，不自行修改方案。
4. 本文件只做静态接口、路径、命令文本和Bash语法核验，不运行dry子命令、训练、judge、恢复或新预检，不产生本次100步成功结论。
