# 原版80k正式CPU输入对拍：预先启动记录（尚未运行）

本档案在正式 CPU 输入对拍前预建；四份 collector 与两份 judge 均尚未执行，INPUT_EQ、P1、100 步训练对拍和 perf 的本阶段结果均待验证。数据组已完成归档提交 `ec6c9e35784a96f636a587dd737dffa294b616ad`，构建及两次 20 步检查的实际 Beta 为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`；两者都是前置证据版本，**均不是尚未启动的本次 INPUT_HEAD**。本 launch 和 [README](README.md) 须先提交，再从包含本 launch 的 clean 提交运行。`INPUT_HEAD` 必须在该提交创建后填入完整 **40 位 Git SHA 字面量**；64 位只用于文件 SHA256。实际标签、展开后的完整命令、会话名、起止版本及退出记录在启动现场留证，结果回写另行提交，不预写尚未发生的会话或通过结论。

从内容依赖看，本阶段仍独立于未答复的 P1/100 步 W&B 和 perf 磁盘采样问题。`check_orig80k_inputs.py::collect()` 只构造目标侧 Dataset、transforms、真实 TorchDataLoader 并做 CPU 输入取证，不调用训练入口、`wandb.init()`、模型初始化、优化器更新或 checkpoint 保存；`judge()` 只读记录并打印判定。因此这两个未决问题不影响CPU输入内容本身，但仍阻止跳入对应的后续训练/测速阶段。本阶段不调用 `run_entry_equiv.sh` 或 `run_orig80k.sh`，也不修改训练参数、真正输入或依赖。

**当前另有第三项前置处置待答：历史最外包装终态的证据边界；CPU输入采集暂缓启动。** 已向用户询问是“补记限制继续CPU输入”，还是“先制定额外复验方案”，目前没有答复，不默认接受任一选项。历史说明见[exit-record-audit.md](exit-record-audit.md)，该说明由根代理另行补充。数据及两次20步的子任务和产物验收PASS未被该反例证伪，但历史最外footer与整体进程真实退出0不能仅凭日志独立证明。下方六份命令仅保留为待执行模板；第三项处置明确并落实前，不调用launch_input、collect或judge。前两项与第三项的依赖关系分开记录，任何一项均未被写成用户已批准。

预建审查已复现新INPUT启动包装的一项缺口：footer tee可完整写出成功终态后再返回非零。当前包装已改为先检查footer printf/tee，再受检查地直接append两项footer状态与唯一综合退出码。实现方与独立审查方各用6例纯shell替身验证，正常、主命令exit7、正文tee失败、footer完整输出后失败、自SHA不匹配及日志冲突均通过；footer失败测试分别保留非零终态1和29，没有成功EXIT_CODE。测试未运行真实tmux、collector或训练，临时载体已清理。这是新包装的验证，不是历史真实退出状态的独立观测，也不改变本阶段采集、数值、并行或参数判据。

## 已核实的输入与资产

以下路径均以主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA` 为根。两库 source/framesamp 是实体目录，4×4 packed metadata 均为 `verified`。前置结果已在数据组完整提交 `ec6c9e35784a96f636a587dd737dffa294b616ad` 归档：[公开16任务库](../../dataset-build-doc/16task-pub-1600ep/README.md)、[counting四任务库及严格子集检查](../../dataset-build-doc/4task-counting-pub-400ep/README.md)、[full20真实保存/恢复](../smoke-orig80k-full-0925/README.md)、[count20真实保存/恢复](../smoke-orig80k-count-0925/README.md)。这些已验收的数据/产物范围不替代本次完整输入取证，也不自动补齐历史最外终态的独立证据；原始记录保持不变。阶段关系遵循[原版80k计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)，实际CPU起跑目前等待上述历史边界处置。

| 组 | 库路径（相对 `v1-store/datasets/`） | 总帧 / 执行样本 | manifest 内嵌 SHA256 |
|---|---|---|---|
| full | `16task-pub-1600ep` | 768897 / 476857 | `fb1bdbcbcc176d15475332b728fa218bb545b4a8c9dcaa9ce1ec9e0bb17f174f` |
| count | `4task-counting-pub-400ep` | 189035 / 189035 | `6b309822aa604a31f6eb0dfb5f02d5482573dc5fa11dc5c4f917880aa92ba986` |

同组 A/B 使用同一 source、episode manifest、input manifest 和 norm_stats。A 读取 `<lib>/source`；B 读取 `<lib>/framesamp`，并由 `load_side()` 显式绑定同一 source/manifest。`--assets-dir` 传父目录，工具自行拼接 `robomme/norm_stats.json`：

| 组 | `--assets-dir`（相对 `v1-store/train-assets/`） | 实测 norm_stats SHA256 |
|---|---|---|
| full | `mme_vla_suite` | `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5` |
| count | `mme_vla_suite/4task-counting-pub-400ep` | `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9` |

A 固定 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，工作树为 `v1-store/worktrees/orig-ecf086c`，使用自身 uv 管理的 `.venv/bin/python`；不重建或同步该环境。B 使用主副本 uv 环境，`uv run --no-sync` 禁止触发依赖同步。取证工具本身由同一主副本文件提供；它不向 `sys.path` 注入 B 的项目源码，A/B 项目模块必须分别来自各自根。B 显式 `--forbid-root` A，judge 还会独立排除 B 父进程及 worker 混入 A 的模块。

## 取证范围与判据

`full_identity()` 解码全量 pkl 核实文件集合及 `(h5_file, raw_ep_idx, step_idx)` 身份，并与 episode 编号/偏移互校；不是只看三样本或前 100 批。

`sample_plan()` 对每集取合法执行步集合 `{exec_start_idx, exec_start_idx+1, num_timesteps-1, 30, 31, 32}`，排序去重且只保留 `exec_start_idx <= step < num_timesteps` 的项。它覆盖执行首尾、执行交界及可用的 31/32/33 帧历史边界，不把不属于训练样本的 demo 帧强行加入。逐点同时记录真实 raw Dataset 与完整 transforms 后的输入；实际定点样本总数以 collector 输出为准。

每侧真实 loader 均固定 b64/workers4/seed42、`drop_last=True`、modul 4×4、action_horizon20。索引探针与内容取证分别由目标侧原生 loader 工厂创建；两遍完整消费真实 sampler，必须记录并一致比较前两个完整 epoch 的全部排列、丢弃尾样本、generator 起止状态与实际 base_seed。内容遍只转发已登记 BatchSampler 位置，不另造乱序序列，也不转换样本 dtype：

| 组 | 每 epoch 完整批数 / 丢弃尾样本 | 第一个 epoch 内容批次（从 0 编号） | 第二个 epoch 内容批次 |
|---|---|---|---|
| full | 7450 / 57 | 0…99、7448、7449 | 0、1 |
| count | 2953 / 43 | 0…99、2951、2952 | 0、1 |

每侧共取 104 个真实批次，即 6656 个批次内样本位置；它们不等于全量两 epoch 的内容解码。两 epoch 的完整索引证据必须齐全，不能只比较选中的 104 批。内容取证终点明确为 `collate_before_jax`；四 worker 的独立环境与模块来源也留证。

当前输入协议为 `schema=2`、`metadata.contract.input_comparison=host_numeric_exact_motion_none_v1`。原始 dtype、shape、raw SHA 和 ALL 字段均保留；主机 dtype 可不同，但支持的数值叶必须按无损编码精确相同，signed zero、有限性、shape 和顺序仍严格。用户批准的新增 schema 白名单仅为 `motion_emb`、`motion_pos`、`motion_mask`、`mem_order` 的缺失与严格 `None` 等价，完整登记在 `equivalent_motion_none_keys`；任何非 None 值或其他未获准字段差异失败。既有废弃 recurrent None 键仍按工具中明确登记的旧规则处理。这里不触及训练五标量和参数/EMA/优化器状态的逐位判据。

两侧 collector 的 `INPUT_COLLECT=PASS` 只是各自取证完整；同组两侧记录须再经 judge 输出 `INPUT_EQ=PASS ... comparison=host_numeric_exact_motion_none_v1`。两个组独立判定；任何失败保留原目录与完整日志，不改名、不补键、不放宽范围。此前三样本证据与工具单测不能代替本阶段。

## CPU、内存与输出预检

并行调度前的只读快照：`nproc=96`，`MemAvailable=1120059940 kB`（约 1068.17 GiB），`/dev/shm` 为 tmpfs、可用 `602265333760 B`（约 560.90 GiB）；负载均值为 0.12/0.28/1.03。tmux server 与当前工具同属 `session-9.scope`，该层及可见祖先的 `memory.max/high=max`、`cpu.max=max 100000`，可用 CPU 为 0–95；用户 cgroup 的 `pids.max=1384119`、当时 `pids.current=1524`，tmux 的文件描述符软限为 8192，当前 cgroup 的 OOM 事件为 0。较高的 `memory.current` 主要是文件页缓存，不能当作同量的不可回收匿名内存。这些不是资源保证，起跑前应重新记录相关限制、可用内存/SHM、负载及本轮活动进程，并确认无本轮并行构建/训练负载。

正式 CPU 输入阶段采用 **full-A、full-B、count-A、count-B 四份独立 collector 并行**。它们只读既定共享数据和资产，各自使用独立输出、日志、tmux 会话与 JAX 编译缓存；同组 judge 仍须等待本组 A/B 全部成功且记录完整；该组满足条件后即可执行 judge，无须等待无关的另一组结束。这只改变调度，不改变 b64/workers4、已核实的原生 `prefetch_factor=2`、seed42、两 epoch 范围或判据，也不形成任何性能保证。每个 loader 最多 8 个预取任务；上游历史读取线程池的代码上限为 36，但本轮最多 32 帧，因此每样本实际最多 32 个读取线程，两份 A 在四 worker 阶段合计可达 256 个历史读取线程，另有 JAX CPU 原生线程，不能凭 96 个 CPU 断言没有瓶颈。

按当前 shape/dtype 计算，B 的 b64 模型输入数组载荷约 244.77 MiB/批；A 无短历史混入时相同，混入短历史使 image/pos 提升为 float64 时约 724.77 MiB/批。四侧各按 8 个预取加 1 个消费批计算，载荷合计约 17.04 GiB；考虑 worker collate、A 的 pickle 序列化/接收和 B 的共享内存转移重叠，复制包络估算约 43.54 GiB，四父进程的有限值检查临时数组另约 2.25 GiB，两份 B 的 worker pos/state 表合计约 0.56 GiB。因此应用数据工作集可按 **约 50 GiB 量级**估计，另有未实测的 Python/JAX/Torch 常驻、分配器缓存和索引对象，不能称为实测峰值或硬上限。两份 B 的 SHM 按每侧 8 个预取加 2 个消费/交接批保守估算合计 **约 4.78 GiB**；A 的原生 NumPy collate 主要承担 pickle 内存/传输开销。

`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1` 四侧一致。运行中以本轮明确的 collector PID 及其子进程树观察 RSS/PSS、线程/文件描述符数，同时监测 MemAvailable、SHM 使用量、CPU/IO 压力、各侧进度和退出状态；共享页不能重复累计后冒充物理独占内存。若资源不足或报错，停止并报告，不降低 batch/workers/prefetch 来凑通过。本阶段不另设未经确认的固定内存容量闸门，不把并行输入取证当作吞吐基准。

四个 `--out` 必须全新且不存在，不能先 mkdir 这些目录，也不能把 tee 日志放进去。`record_manifest.json` 会绑定 collector 输出目录内全部文件，收尾后不要追加日志或判定文本到该目录。日志统一放 `v1-store/logs/`，judge 输出也独立留日志。缓存逐项指向 scratch；清除 `PYTHONPATH/PYTHONHOME`，**不覆盖用户的 `HOME`**。显式清除 `JAX_PLATFORM_NAME` 和遗留 `XLA_FLAGS`，设置 `CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`、`JAX_ENABLE_X64=0`、`HF_HUB_OFFLINE=1`。source、norm_stats 和 tokenizer 仍为已核实的只读共享资产；六个任务分别使用自己命名的 JAX 缓存目录。现有 judge 比较输入 contract、Python/依赖、来源/资产摘要及输入内容，不比较编译缓存目录；独立缓存路径和共同的 HF 离线设置不改变这些契约，不能借此改变实际资产路径或跳过来源检查。

先检查双方 clean HEAD、A 固定提交、两侧独立解释器、数据/资产摘要、packed verified/full 与无 pack.lock。四个 collector 运行期间冻结主副本源码和两个 `.venv`，不提交归档或修改工具。预计超过 5 分钟，一律 detached tmux，记录完整会话名及退出码；只按清单精确匹配会话，不做全局清理。

## 公共命令设置与有退出记录的启动包装

以下仅为待执行模板；当前须先等待用户对历史最外终态边界的处置答复并落实，不能直接启动。之后再将所有占位符填实。`INPUT_HEAD` 在包含正式 launch 的提交创建后手动填完整字面量；不要用 `INPUT_HEAD=$(git rev-parse HEAD)` 代替期望锚点。`INPUT_TAG` 一经选定即同时用于六个会话、日志及命令文件，起跑前确认没有重名记录。实际执行先逐份保存展开后的同一命令文件，现场计算其SHA256，再以短命令交给tmux运行，避免长内联命令限制。文件路径和真实SHA在日志现场头记录，不能提前编造当前尚未创建的文件摘要，也不将这些运行时命令文件复制到本档案目录。

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
UP="$STORE/worktrees/orig-ecf086c"
UPSTREAM_HEAD=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b
INPUT_HEAD='<待包含launch的提交创建后填写40位Git SHA>'
INPUT_TAG='<本次唯一UTC标签>'
INPUT_ROOT="$STORE/bench/orig80k-equiv-0925/input-$INPUT_TAG"
COMMAND_ROOT="$STORE/bench/orig80k-equiv-0925/commands-$INPUT_TAG"
INPUT_TOOL="$MAIN/scripts/training/tests/check_orig80k_inputs.py"
FULL="$STORE/datasets/16task-pub-1600ep"
COUNT="$STORE/datasets/4task-counting-pub-400ep"
FULL_ASSETS="$STORE/train-assets/mme_vla_suite"
COUNT_ASSETS="$STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep"

[[ "$INPUT_HEAD" =~ ^[0-9a-f]{40}$ ]] || { printf 'INPUT_HEAD尚未填实\n' >&2; exit 2; }
[[ "$INPUT_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { printf 'INPUT_TAG尚未填实或含非法字符\n' >&2; exit 2; }
test "$INPUT_HEAD" = "$(git -C "$MAIN" rev-parse HEAD)" || exit 2
test -z "$(git -C "$MAIN" status --porcelain)" || exit 2
test "$UPSTREAM_HEAD" = "$(git -C "$UP" rev-parse HEAD)" || exit 2
test -z "$(git -C "$UP" status --porcelain)" || exit 2
test ! -e "$INPUT_ROOT" && test ! -L "$INPUT_ROOT" || exit 2
test ! -e "$COMMAND_ROOT" && test ! -L "$COMMAND_ROOT" || exit 2
test -d "$STORE/logs" && test ! -L "$STORE/logs" || exit 2

INPUT_ENV=(env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORM_NAME -u XLA_FLAGS
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=0
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  OPENPI_DATA_HOME="$STORE/models" UV_CACHE_DIR="$STORE/cache/uv"
  XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
  HF_HUB_OFFLINE=1)

launch_input() {
  local session="$1" log="$2" cwd="$3"
  shift 3
  test ! -e "$log" && test ! -L "$log" || return 2
  if tmux has-session -t "=$session" 2>/dev/null; then
    printf '拒绝复用会话 %s\n' "$session" >&2
    return 2
  fi
  local body='set -o pipefail
input_session=$1
input_head=$2
input_cwd=$3
input_log=$4
input_command_file=$5
input_command_sha=$6
input_wrapper_pid=$BASHPID
shift 6
( set -o noclobber; : > "$input_log" ) || exit 2
(
  printf "SESSION=%s\nWRAPPER_PID=%s\nBODY_PID=%s\nINPUT_HEAD=%s\nCOMMAND_FILE=%s\nCOMMAND_FILE_SHA256_EXPECTED=%s\nSTART_UTC=%s\n" \
    "$input_session" "$input_wrapper_pid" "$BASHPID" "$input_head" "$input_command_file" "$input_command_sha" "$(date -u +%FT%TZ)"
  input_actual_sha=$(sha256sum -- "$input_command_file") || exit 2
  input_actual_sha=${input_actual_sha%% *}
  printf "COMMAND_FILE_SHA256_ACTUAL=%s\n" "$input_actual_sha"
  [[ "$input_actual_sha" == "$input_command_sha" ]] || { printf "COMMAND_FILE_SHA256_MISMATCH\n" >&2; exit 2; }
  cd "$input_cwd" || exit 2
  printf "COMMAND="
  printf "%q " "$@"
  printf "\n"
  "$@"
) 2>&1 | tee "$input_log"
codes=("${PIPESTATUS[@]}")
rc=${codes[0]}
if [ "${codes[1]}" -ne 0 ]; then rc=${codes[1]}; fi
printf "SESSION_END=%s\nINPUT_HEAD_END=%s\nEND_UTC=%s\nCOMMAND_EXIT=%s\nTEE_EXIT=%s\n" \
  "$input_session" "$input_head" "$(date -u +%FT%TZ)" "${codes[0]}" "${codes[1]}" | tee -a "$input_log"
ending=("${PIPESTATUS[@]}")
if [ "${ending[0]}" -ne 0 ]; then rc=${ending[0]}; fi
if [ "${ending[1]}" -ne 0 ]; then rc=${ending[1]}; fi
if ! printf "FOOTER_PRINTF_EXIT=%s\nFOOTER_TEE_EXIT=%s\nEXIT_CODE=%s\n" \
  "${ending[0]}" "${ending[1]}" "$rc" >> "$input_log"; then
  printf "最终退出记录追加失败：%s\n" "$input_log" >&2
  exit 1
fi
exit "$rc"'
  local command_file="$COMMAND_ROOT/$session.command"
  mkdir -p "$COMMAND_ROOT" || return 2
  (
    set -o noclobber
    {
      printf '%q ' bash -c "$body" -- "$session" "$INPUT_HEAD" "$cwd" "$log"
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
```

公共设置只检查两个新根；启动包装会把命令文件保存在独立的 `COMMAND_ROOT`，collector自行创建各自输出及父路径，命令文件不会混入任何collector清单。命令文件中引用自身执行路径及外部传入的期望SHA，避免把文件自身摘要写回正文而改变它；运行前再次核对实际SHA，只有一致才调用collector/judge。四份独立任务可依次发出启动命令后并行运行，无须等待另一collector完成。六份环境均固化在各自命令文件中，不依赖既有tmux server是否继承当前shell的自定义环境变量。

日志现场头必须包含会话全名、wrapper/body PID、声明的 `INPUT_HEAD`、命令文件路径及期望/实际SHA、`START_UTC`。尾部先经tee记录 `END_UTC`、命令退出码和正文 `TEE_EXIT`，立即捕获该尾部printf与tee状态；之后才用受检查的直接append写 `FOOTER_PRINTF_EXIT`、`FOOTER_TEE_EXIT` 和唯一综合 `EXIT_CODE`，不能先写成功终态再检查tee。任一写入或退出记录失败均不得视作成功。这里的 `INPUT_HEAD` 是本CPU阶段的B侧锚点；A侧实际 `--expect-head` 仍固定上游完整提交，不被头部字段替代。当前这些值均待运行现场产生，并无实际启动记录。

`launch_input` 返回只表示会话创建，不表示collector或judge成功。进入同组judge前，须核对本组A/B各自日志中的 `INPUT_COLLECT=PASS`，以及**唯一 `EXIT_CODE=0`、唯一 `COMMAND_EXIT=0`、唯一正文 `TEE_EXIT=0`、唯一 `FOOTER_PRINTF_EXIT=0`、唯一 `FOOTER_TEE_EXIT=0`**；缺失、重复或非零均不放行。两份judge自身也使用同一退出记录要求，再结合各自唯一 `INPUT_EQ` 成功判定，不能只看到工具PASS或正文tee成功就宣布阶段完成。

## 四份 collector 命令

1. full A；从 A 根启动 A 的独立解释器，与其余三份 collector 独立并行。

```bash
launch_input "orig80k-input-full-a-$INPUT_TAG" "$STORE/logs/orig80k-input-full-a-$INPUT_TAG.log" "$UP" \
  "${INPUT_ENV[@]}" JAX_COMPILATION_CACHE_DIR="$STORE/cache/orig80k-input-$INPUT_TAG/full-a" \
  "$UP/.venv/bin/python" -B "$INPUT_TOOL" collect \
  --side upstream --expect-root "$UP" --expect-head "$UPSTREAM_HEAD" \
  --dataset-path "$FULL/source" --source "$FULL/source" \
  --manifest "$FULL/meta/episode_manifest.json" --input-manifest "$FULL/meta/input_manifest.json" \
  --assets-dir "$FULL_ASSETS" --out "$INPUT_ROOT/full/a"
```

2. full B；从主副本启动，不等待 full A 完成。

```bash
launch_input "orig80k-input-full-b-$INPUT_TAG" "$STORE/logs/orig80k-input-full-b-$INPUT_TAG.log" "$MAIN" \
  "${INPUT_ENV[@]}" JAX_COMPILATION_CACHE_DIR="$STORE/cache/orig80k-input-$INPUT_TAG/full-b" \
  uv run --no-sync python -B "$INPUT_TOOL" collect \
  --side current --expect-root "$MAIN" --expect-head "$INPUT_HEAD" --forbid-root "$UP" \
  --dataset-path "$FULL/framesamp" --source "$FULL/source" \
  --manifest "$FULL/meta/episode_manifest.json" --input-manifest "$FULL/meta/input_manifest.json" \
  --assets-dir "$FULL_ASSETS" --out "$INPUT_ROOT/full/b"
```

3. counting A；与 full 组及 counting B 独立并行。

```bash
launch_input "orig80k-input-count-a-$INPUT_TAG" "$STORE/logs/orig80k-input-count-a-$INPUT_TAG.log" "$UP" \
  "${INPUT_ENV[@]}" JAX_COMPILATION_CACHE_DIR="$STORE/cache/orig80k-input-$INPUT_TAG/count-a" \
  "$UP/.venv/bin/python" -B "$INPUT_TOOL" collect \
  --side upstream --expect-root "$UP" --expect-head "$UPSTREAM_HEAD" \
  --dataset-path "$COUNT/source" --source "$COUNT/source" \
  --manifest "$COUNT/meta/episode_manifest.json" --input-manifest "$COUNT/meta/input_manifest.json" \
  --assets-dir "$COUNT_ASSETS" --out "$INPUT_ROOT/count/a"
```

4. counting B；从主副本启动，不等待其他 collector 完成。

```bash
launch_input "orig80k-input-count-b-$INPUT_TAG" "$STORE/logs/orig80k-input-count-b-$INPUT_TAG.log" "$MAIN" \
  "${INPUT_ENV[@]}" JAX_COMPILATION_CACHE_DIR="$STORE/cache/orig80k-input-$INPUT_TAG/count-b" \
  uv run --no-sync python -B "$INPUT_TOOL" collect \
  --side current --expect-root "$MAIN" --expect-head "$INPUT_HEAD" --forbid-root "$UP" \
  --dataset-path "$COUNT/framesamp" --source "$COUNT/source" \
  --manifest "$COUNT/meta/episode_manifest.json" --input-manifest "$COUNT/meta/input_manifest.json" \
  --assets-dir "$COUNT_ASSETS" --out "$INPUT_ROOT/count/b"
```

每份日志应有唯一 `INPUT_COLLECT=PASS`，并满足上述命令、正文tee、两项footer状态及唯一综合退出码均为0的放行要求；逐侧核对 full 的 476857、count 的 189035 全量身份记录、各 104 批以及两个完整 epoch。是否最终等价必须等以下对应judge，不能凭collector成功提前宣布。

## 两份 judge 命令

每个judge仅在本组两份collector满足全部工具/退出状态要求且记录清单完整后执行，日志仍放collector输出之外。judge也须核对其命令、正文tee、footer printf/tee及唯一综合退出码均为0。

```bash
launch_input "orig80k-input-full-judge-$INPUT_TAG" "$STORE/logs/orig80k-input-full-judge-$INPUT_TAG.log" "$MAIN" \
  "${INPUT_ENV[@]}" JAX_COMPILATION_CACHE_DIR="$STORE/cache/orig80k-input-$INPUT_TAG/full-judge" \
  uv run --no-sync python -B "$INPUT_TOOL" judge \
  --a-dir "$INPUT_ROOT/full/a" --b-dir "$INPUT_ROOT/full/b" \
  --expect-head-a "$UPSTREAM_HEAD" --expect-head-b "$INPUT_HEAD"
```

```bash
launch_input "orig80k-input-count-judge-$INPUT_TAG" "$STORE/logs/orig80k-input-count-judge-$INPUT_TAG.log" "$MAIN" \
  "${INPUT_ENV[@]}" JAX_COMPILATION_CACHE_DIR="$STORE/cache/orig80k-input-$INPUT_TAG/count-judge" \
  uv run --no-sync python -B "$INPUT_TOOL" judge \
  --a-dir "$INPUT_ROOT/count/a" --b-dir "$INPUT_ROOT/count/b" \
  --expect-head-a "$UPSTREAM_HEAD" --expect-head-b "$INPUT_HEAD"
```

两份成功判定分别应报告 episodes=1600/samples=476857 与 episodes=400/samples=189035，均为 batches=104、epochs=2、`comparison=host_numeric_exact_motion_none_v1`。记录每侧实际定点样本数、耗时、启动提交、模块/依赖/资产摘要及全部退出状态。原始日志与清洗日志分别保留；清洗不能遗漏失败或完成行。

监听使用当前宿主流式工具；shell 过滤示例为 `tail -n +1 -F <该侧日志> | grep --line-buffered -E 'INPUT_|Traceback|Error|EXIT_CODE='`。检查会话用 `tmux has-session -t '=完整会话名'`，任务结束须再读对应日志退出码；会话消失本身不代表成功。归档 collector 原始记录时连同 `record_manifest.json` 保持字节不变，解释和 judge 日志写在独立的 result/records 层；不往被清单绑定的 collector 子目录加文件。

## 后续提交变化与证据沿用边界

本阶段 `meta.json`、起止 provenance、父进程/worker 模块摘要、common_files、packed side_files 和 record_manifest 均绑定实际 `INPUT_HEAD`。已完成证据始终只能如实称作“在该 INPUT_HEAD 采集”；禁止修改记录中的 HEAD、harness SHA 或文件清单，把旧取证伪装成新提交实跑。judge 的 `--expect-head-b` 必须继续传采集时的 INPUT_HEAD，而不是之后的 TRAIN_HEAD。

当前 judge 校验的是记录中声明的提交和取证文件，**不会自动把这些记录与主副本未来 HEAD 的全部源码重新比较**；它也没有“忽略 HEAD 差异并迁移记录”的接口。后续仅改文档或未被输入链使用的测速工具时，原始数据/身份、完整 sampler 序列、样本/batch 内容证据不因此被改写或消失，可作为原 INPUT_HEAD 的证据留用。但不能仅凭再次对旧记录得到 PASS，就声称新提交已经重跑过输入取证。

拟在未来训练 launch 中沿用这份输入证据时，应同时记 INPUT_HEAD 与实际 TRAIN_HEAD，并核对二者之间的差异确实限于文档或输入链之外的工具：取证脚本内容及协议、父进程/worker 记录的项目模块、Python构建和依赖版本、输入参数、history、source/manifest/provenance、norm_stats、tokenizer 与 packed metadata 均未改变。对照依据是已记录的文件/版本指纹和固定提交 diff，不把“文件名看起来无关”作为充分证据。任何指纹变化、输入链改动或影响是否独立无法确认时，不自动沿用，先报告并按受影响范围重新取证；不得通过重写 expected head、减少输入字段或更改比较契约绕过。

A 固定上游源码、A独立环境、数据和工具均未改变时，A 的既有取证仍是该上游环境的有效历史基线；若取证工具本身改变，judge 明确要求 A/B 的 `harness.sha256` 一致，不能混用新旧工具记录，需两侧重采。数据/资产改变时，其对应组的 A/B 都受影响；仅重新运行 judge 不产生新输入证据。

训练基线另受更严格的现有约束：计划要求 S1/S3 与 S2 同 HEAD，`entry_equiv.py --mode same-entry` 也直接比较两侧提交和模块摘要。因此 100 步单跑基线一旦与 S2 的 HEAD 不同，即便只改了文档，也不能擅自让 same-entry 忽略它；应先统一验证用 clean HEAD，再按既定判据取得可比较基线。本档案不新增跨 HEAD 放行规则，也不以 CPU INPUT_EQ 替代 P1、100 步、并跑、perf 或正式完成验收。
