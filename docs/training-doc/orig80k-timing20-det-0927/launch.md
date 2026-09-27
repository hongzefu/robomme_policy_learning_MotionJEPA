# 确定性档20步：两库测速包装off/on真保存启动候选（待Beta，尚未运行）

已形成并推送的实现锚点为 `aec86db64e5178e63d9e7f77d3f5bc235db16390`；它锁定本轮已批准档位实现，377项核心合测通过。下一档案Beta确定为 `commitV11.16Beta`，其实际TRAIN_HEAD仍待生成；实施锚点不能冒充尚未发生的训练起跑版本。

准备标签为 `20260927T012426Z`，**不是实际起跑时间**。本候选仅准备启动材料；四个run、两个CPU judge均未运行，TRAIN_HEAD、START/END_UTC、PID、UUID、W&B ID及结果全部待真实记录。预计由主代理选择 `commitV11.16Beta`，最终编号与完整40位SHA以实际Git提交为准；不能用准备时HEAD或未来假造SHA填充。

用户已同意将本轮20步包装正确性对照切到既有100步使用的确定性档，保持逐位判据，perf/prod仍用原normal环境。本阶段显式选择 `ORIG80K_TIMING_EQ_PROFILE=deterministic100`，runner据此设置精确 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`。这不是已证明的根因修复，也不保证逐位通过；旧normal失败与off复验只引用[原off/on结果](../orig80k-timing20-0926/result.md)及[off复验结果](../orig80k-off-repeat-0927/result.md)，其原HEAD、数值和FAIL不改写。

前置INPUT的实际HEAD为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`；P1/100量具与B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。旧INPUT/P1最外退出限制按用户「同样补记限制，沿用INPUT结果（推荐）」及「补记P1限制，补独立退出取证后继续（推荐）」保留；六份100步run及四judge已有原版本通过证据。新Beta须核对旧证据的模块/依赖/数据沿用范围和本轮已批准的档位工具改动，不能宣称在新HEAD重训过这些历史阶段。新schema2要求显式profile，不以缺字段兼容来重判旧normal记录。

用户要求「两库各跑一对 20 步（推荐）」「尽可能并行做」「你有8张卡」。full固定0–3、count固定4–7，两条链分别off→真实保存/CPU恢复/独立退出验收→on→同样验收→正式CPU judge，各库不等待无关阶段。同一库off/on不重叠。保持b64/fsdp4/workers4/seed42、modul512/4×4/max32、原lr/warmup/EMA、W&B online；smoke仅覆盖steps20，log100/save10000/keep10000保持，末步19真保存及实际恢复不变。

本阶段只验正确性，不报告吞吐或ETA。用户「继续工作 一路做到起泡前 有问题问用户」限定推进至正式80k起跑前；即使本阶段通过，也只能证明确定性档内的包装对照，不能改写normal失败、替代normal300步perf、预算或正式起跑前全部闸门。本文包含完整控制函数和六任务CLI，不能整页直接执行。

## 四个名称、路径与范围

| 库 | off run | on run | GPU | assets父目录 |
|---|---|---|---|---|
| full | `timing20-full-off-0926-$TAG` | `timing20-full-on-0926-$TAG` | `0,1,2,3` | `v1-store/train-assets/mme_vla_suite` |
| counting | `timing20-count-off-0926-$TAG` | `timing20-count-on-0926-$TAG` | `4,5,6,7` | `v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep` |

8张GPU按下表固定分配；两个CPU恢复和两个CPU judge允许在各自依赖就绪时及时运行或并行，不借用另一库GPU，也不因另一库尚未结束而空等：

| 资源 | full链 | counting链 | 本链放行条件 |
|---|---|---|---|
| GPU `0,1,2,3` | full-off → full-on | 不使用 | off训练、恢复及独立退出全部验收后才启动full-on |
| GPU `4,5,6,7` | 不使用 | count-off → count-on | off训练、恢复及独立退出全部验收后才启动count-on |
| CPU真实恢复 | 各run末步19保存后由其wrapper串行执行 | 各run末步19保存后由其wrapper串行执行 | `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`；本run恢复及wrapper独立退出通过前不进入本库下一run |
| CPU judge | full两run各自验收后立即可启动 | count两run各自验收后立即可启动 | 每库judge独立capture/receipt、父返回码及内容验收；两库全通过才完成本阶段 |

库分别是主仓 `v1-store/datasets/16task-pub-1600ep` 与 `4task-counting-pub-400ep`，由runner消费 `<lib>/framesamp` 并显式绑定同库source/episode manifest。full使用原版norm SHA `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`；counting为 `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。history `perceptual-framesamp-modul.yaml` 的期望SHA为 `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`。

每run由既有runner独占创建 `v1-store/bench/orig80k/<run>`、`v1-store/train-runs/mme_vla_suite/<run>` 和 `v1-store/logs/<run>.driver.log`，不得提前mkdir记录根、复用名称、resume或overwrite。runner按run名隔离JAX/CUDA/W&B缓存并检查GPU、完整配置、资产及smoke的300GiB磁盘最低保护。整个阶段还须按实际剩余空间核算四份新权重与既有数据/日志占用；本启动档不把300GiB最低保护冒称完整峰值预算，不自动删除任何产物。

## 两个正式模板与真实调用

正式源 `scripts/training/tests/timing20-wrapper.sh` 将 runner → CPU completion 串行运行。其内部实际命令分别为：

```bash
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME -u PYTHONPATH -u PYTHONHOME \
  TRAIN_HEAD="$TRAIN_HEAD" HISTORY_CONFIG_SHA256="$HISTORY_SHA" NORM_STATS_SHA256="$NORM_SHA" \
  ORIG80K_SMOKE_EQ_MODE="$TIMING" \
  bash "$MAIN/scripts/training/prod/run_orig80k.sh" smoke "$RUN" "$GPUS" "$LIB" "$ASSETS"

env -u JAX_PLATFORM_NAME -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  uv run --no-sync python "$MAIN/scripts/training/tests/check_orig80k_completion.py" \
  --mode smoke --records "$REC" --run-root "$RUN_ROOT" --log "$DRIVER_LOG" \
  --run "$RUN" --head "$TRAIN_HEAD" --out "$REC/completion.json"
```

外层日志为 `v1-store/logs/orig80k-<run>.wrapper.log`，completion自己的tee日志为 `<records>/completion.summary.log`。外层不追加或覆盖driver.log。runner失败时不执行恢复；恢复或其tee失败时不生成成功终态。恢复成功后读取其真实UUID、SHA并记录 `WRAPPER_RUN_UUID`、`WRAPPER_COMPLETION_SHA256`，再核HEAD/clean。只清除CPU平台遗留，不覆盖继承的 `JAX_ENABLE_X64`；若其实际启用，runner必须拒绝，不悄悄改输入。W&B凭据由runner现有机制读取，不回显凭据。

每份运行日志中的下列8项仍必须各自唯一且为0，正是当前 `check_orig80k_timing_equiv.py::WRAPPER_EXITS` 的协议；命令、身份读取和末次版本核对的返回值也分别记录。但最终direct append完整写0后仍可能返回非零，**仅这些字段不能证明最外进程实际退出0**。本修订保留原wrapper，再用冻结f8保留原生pane退出并联合验收。原始 `EXIT_CODE` 归driver所有，不能混作外层退出；不得只凭当前Python judge接受旧日志而跳过独立退出层。

```text
RUNNER_EXIT_CODE=0
COMPLETION_EXIT_CODE=0
COMPLETION_TEE_EXIT_CODE=0
WRAPPER_COMMAND_EXIT=0
WRAPPER_TEE_EXIT=0
WRAPPER_FOOTER_PRINTF_EXIT=0
WRAPPER_FOOTER_TEE_EXIT=0
WRAPPER_EXIT_CODE=0
```

`timing20-judge-wrapper.sh` 用同一外层故障传播函数包装独立CPU judge。它校验运行命令文件和通用wrapper文件的SHA；任一错误不得只凭 `TIMING_EQ=PASS` 文字放行。四个run与两个judge均由冻结f8先门闩启动，再设置本窗口保留、落身份文件并核SHA，最后释放。所有命令/identity/receipt及父进程sidecar保留在v1-store内，不能在验收前改写或删除。

## 待填设置与命令文件生成

先在新Beta的clean工作区独立运行只读预检，再执行下面会创建COMMAND_ROOT的设置块；两者顺序不能颠倒。预检实体为 `v1-store/bench/orig80k-build-preflight-0925/check_timing20_deterministic_preflight_20260927.py`，SHA为 `a3561665c4a92df7d2fef8660be9d04a77bd91186642878d4ffcfc132594b5d5`。实际调用为 `.venv/bin/python -B v1-store/bench/orig80k-build-preflight-0925/check_timing20_deterministic_preflight_20260927.py --implementation-head aec86db64e5178e63d9e7f77d3f5bc235db16390 --head <实际新Beta完整SHA> --tag 20260927T012426Z`，另显式设定 `UV_CACHE_DIR=v1-store/cache/uv`。留存其JSON、stderr和父进程观察到的真实返回码；只有 `status=PASS` 且真实返回0，才可创建本阶段命令根。它要求所有新输出均不存在，不能在生成命令根后再执行。此预检只核沿用边界与CPU配置，不代替下面四次GPU训练及两次正式judge。

本阶段Beta须共同锁定 `scripts/training/tests/timing20-wrapper.sh`、`timing20-judge-wrapper.sh`、`tmux_exit_guard.py`及本文完整控制函数。两模板逐字SHA固定为45b4…/17c0…；guard正式源和旧v1-store runtime均为f8。实际guard继续执行原v1-store实体，不能只把GUARD改成正式Git路径而放宽现有边界。下列父适配每次launch/capture/verify都从实际TRAIN_HEAD读取三份Git源，与工作树逐字核对，再把guard与runtime逐字核对；`git_source_binding`随launch/capture父记录留档并在后续verify重核。正式TRAIN_HEAD尚待Beta生成后填写；准备TAG固定，full-off→full-on→full-judge与count-off→count-on→count-judge两条链独立推进。

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
TRAIN_SCRIPTS="$MAIN/scripts/training/tests"
WRAPPER_TEMPLATE="$TRAIN_SCRIPTS/timing20-wrapper.sh"
JUDGE_TEMPLATE="$TRAIN_SCRIPTS/timing20-judge-wrapper.sh"
WRAPPER_SHA=45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f
JUDGE_SHA=17c0d6deea3ab3cd132bf406c968097ff9e7b455c9f22f5e3614c8641d7aafce
GUARD_TOOL="$STORE/bench/orig80k-build-preflight-0925/tmux-exit-guard-candidate-20260926/tmux_exit_guard.py"
GUARD_SHA256=f8dada4025dc49f260697e2fb768e922cbea7b8f98afe5cc8b98555648bff52b
TRAIN_HEAD='<待commitV11.16Beta生成后填写实际完整40位SHA>'
TAG=20260927T012426Z
COMMAND_ROOT="$STORE/bench/orig80k-timing20-commands-$TAG"
RESULT_ROOT="$STORE/bench/orig80k-timing20-results-$TAG"
COMMON_WRAPPER="$COMMAND_ROOT/common-wrapper.sh"
FULL="$STORE/datasets/16task-pub-1600ep"
COUNT="$STORE/datasets/4task-counting-pub-400ep"
FULL_ASSETS="$STORE/train-assets/mme_vla_suite"
COUNT_ASSETS="$STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep"
FULL_NORM=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
COUNT_NORM=a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9
FULL_OFF="timing20-full-off-0926-$TAG"
FULL_ON="timing20-full-on-0926-$TAG"
COUNT_OFF="timing20-count-off-0926-$TAG"
COUNT_ON="timing20-count-on-0926-$TAG"
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ && "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
test "$(git -C "$MAIN" rev-parse HEAD)" = "$TRAIN_HEAD" || exit 2
test -z "$(git -C "$MAIN" status --porcelain)" || exit 2
for path in "$COMMAND_ROOT" "$RESULT_ROOT"; do
  test ! -e "$path" && test ! -L "$path" || exit 2
done
for run in "$FULL_OFF" "$FULL_ON" "$COUNT_OFF" "$COUNT_ON"; do
  for path in "$STORE/bench/orig80k/$run" "$STORE/train-runs/mme_vla_suite/$run" \
    "$STORE/logs/$run.driver.log" "$STORE/logs/orig80k-$run.wrapper.log" \
    "$STORE/cache/jax/$run" "$STORE/cache/cuda/$run" "$STORE/logs/wandb/$run" \
    "$STORE/cache/wandb/$run" "$STORE/cache/wandb-config/$run" "$STORE/cache/wandb-data/$run" \
    "$STORE/cache/xdg-data/$run"; do
    test ! -e "$path" && test ! -L "$path" || exit 2
  done
done
actual=$(sha256sum -- "$GUARD_TOOL") || exit 2
test "${actual%% *}" = "$GUARD_SHA256" || exit 2
actual=$(sha256sum -- "$WRAPPER_TEMPLATE") || exit 2
test "${actual%% *}" = "$WRAPPER_SHA" || exit 2
mkdir -p "$COMMAND_ROOT" || exit 2
(set -o noclobber; cat "$WRAPPER_TEMPLATE" > "$COMMON_WRAPPER") || exit 2
actual=$(sha256sum -- "$COMMON_WRAPPER") || exit 2
test "${actual%% *}" = "$WRAPPER_SHA" || exit 2
```

`COMMON_WRAPPER`是原run模板的独立实体副本，完整字节SHA仍为45b4…，**绝不经过下面的set前缀生成器**。judge source它时只能装载函数，不能改变judge自己的位置参数。各自runtime command必须有两行前缀：第一行是 `set -- "$1" <固定参数...>`；第二行run固定为 `export ORIG80K_TIMING_EQ_PROFILE=deterministic100`，judge固定为 `unset ORIG80K_TIMING_EQ_PROFILE`。随后才逐字附上原模板；`$1`在执行时来自guard传入的整份runtime新SHA，不能在生成时替换成模板SHA。固定参数逐个由Bash `printf %q`生成。不能只在控制shell export并假定tmux继承；run把选择写进自身命令，judge明确清掉选择，仅影响CPU读取环境，不修改所读的det记录。guard只需执行 `bash <runtime-command> <runtime-sha>`，run模板最终获得10参数、judge模板9参数。

```bash
write_bound_runtime() {
  local template=$1 expected=$2 command_file=$3 role=$4 actual directive
  shift 4
  test "$#" -gt 0 || return 2
  case "$role" in
    run) directive='export ORIG80K_TIMING_EQ_PROFILE=deterministic100' ;;
    judge) directive='unset ORIG80K_TIMING_EQ_PROFILE' ;;
    *) return 2 ;;
  esac
  actual=$(sha256sum -- "$template") || return
  [[ "${actual%% *}" == "$expected" ]] || return 2
  (
    set -o noclobber
    {
      printf 'set -- "$1"' || exit 2
      printf ' %q' "$@" || exit 2
      printf '\n%s\n' "$directive" || exit 2
      cat "$template" || exit 2
    } > "$command_file"
  ) || return 2
  bash -n "$command_file" || return 2
}
```

以下父进程适配与root已用的 `.exit.capture.json`保持同一schema：真实 `subprocess.run.returncode`、identity/receipt/guard/stdout/stderr的路径/bytes/SHA、实际argv和UTC。`verify`只读已有回执，不查询tmux。capture先probe；活任务返回3而不占最终文件，若后续真正capture又返回3也不创建最终sidecar。父进程本身的实际执行退出码仍由现场调用工具/controller保存，不从它写出的JSON推断。

```bash
guard_timing_action() {
  local action="$1" session="$2" command_sha=""
  if [ "$#" -eq 3 ]; then command_sha="$3"; fi
  env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 \
    "$MAIN/.venv/bin/python" -B - "$action" "$session" "$command_sha" "$TRAIN_HEAD" \
    "$MAIN" "$COMMAND_ROOT" "$GUARD_TOOL" "$GUARD_SHA256" <<'PY'
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

action, session, command_sha, head, main, command_root, tool_path, tool_sha = sys.argv[1:]
main, command_root, tool_path = Path(main), Path(command_root), Path(tool_path)
if not (tool_path == tool_path.resolve() and tool_path.is_relative_to(main / "v1-store")
        and hashlib.sha256(tool_path.read_bytes()).hexdigest() == tool_sha):
    raise SystemExit("加载前冻结guard实体路径/SHA不符")
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=main, text=True).strip() != head:
    raise SystemExit("正式Beta HEAD不符")
if subprocess.check_output(["git", "status", "--porcelain"], cwd=main, text=True):
    raise SystemExit("正式Beta工作区不clean")
source_specs = {
    "scripts/training/tests/tmux_exit_guard.py": tool_sha,
    "scripts/training/tests/timing20-wrapper.sh": "45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f",
    "scripts/training/tests/timing20-judge-wrapper.sh": "17c0d6deea3ab3cd132bf406c968097ff9e7b455c9f22f5e3614c8641d7aafce",
}
git_source_binding = {"head": head, "files": {}}
for relative, expected in source_specs.items():
    working_path = main / relative
    if not (working_path == working_path.resolve() and working_path.is_file()):
        raise SystemExit("正式源必须为固定实体文件: " + relative)
    committed = subprocess.check_output(["git", "show", head + ":" + relative], cwd=main)
    working = working_path.read_bytes()
    if committed != working or hashlib.sha256(committed).hexdigest() != expected:
        raise SystemExit("正式Beta源码字节或期望SHA不同: " + relative)
    if relative.endswith("/tmux_exit_guard.py") and committed != tool_path.read_bytes():
        raise SystemExit("正式Beta guard与冻结runtime副本字节不同")
    git_source_binding["files"][relative] = {
        "git_path": relative, "working_path": str(working_path),
        "bytes": len(committed), "sha256": expected}
git_source_binding["runtime_guard"] = {
    "path": str(tool_path), "bytes": len(tool_path.read_bytes()), "sha256": tool_sha}

spec = importlib.util.spec_from_file_location("frozen_timing_exit_guard", tool_path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
require = guard.require
tool_ref = guard.file_reference(guard.entity_path(tool_path, main))
require(tool_ref["sha256"] == tool_sha, "冻结guard实际SHA不同")
require(action in {"launch", "capture", "verify"}, "未知阶段动作")
command = command_root / (session + ".command")
identity_path = command_root / (session + ".identity.json")
launch_meta_path = command_root / (session + ".launch.process.json")
capture_meta_path = command_root / (session + ".exit.capture.json")
receipt_path = command_root / (session + ".exit.json")

def new_bytes(path, raw):
    guard.entity_path(path, main, exists=False)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())

def check_ref(value):
    guard.check_reference(value, main)
    return value

def read_launch():
    meta = guard.read_json(guard.entity_path(launch_meta_path, main))
    require(meta.get("git_source_binding") == git_source_binding, "launch正式Git源/runtime绑定不同")
    require(meta["schema"] == 1 and meta["session"] == session
            and type(meta["actual_exit_code"]) is int and meta["actual_exit_code"] == 0
            and meta["observer"] == "root subprocess.run.returncode", "launch外部实际返回码缺失")
    for key in ("command_file", "guard_tool", "stdout", "stderr", "identity"):
        check_ref(meta[key])
    require(meta["guard_tool"] == tool_ref and meta["command_file"] == guard.file_reference(command)
            and meta["identity"]["path"] == str(identity_path), "launch来源不匹配")
    identity = guard.read_identity(identity_path, meta["identity"]["sha256"])
    require(identity["session"] == session and identity["label"] == session
            and identity["train_head"] == head and identity["command_file"] == meta["command_file"],
            "identity归属不同")
    expected = [sys.executable, "-B", str(tool_path), "launch", "--session", session,
                "--train-head", head, "--label", session, "--command-file", str(command),
                "--command-sha256", meta["command_file"]["sha256"], "--identity-out", str(identity_path)]
    require(meta["argv"] == expected, "launch实际ARGV不同")
    payload = guard.read_json(Path(meta["stdout"]["path"]))
    require(payload["status"] == "RELEASED" and payload["created"] is True
            and payload["released"] is True and payload["release_uncertain"] is False
            and payload["identity"] == meta["identity"] and payload["tool"] == tool_ref
            and payload["command_file"] == meta["command_file"], "launch未证实先落身份后释放")
    return meta

def verify_capture(launch):
    meta = guard.read_json(guard.entity_path(capture_meta_path, main))
    require(meta.get("git_source_binding") == git_source_binding, "capture正式Git源/runtime绑定不同")
    require(meta["schema"] == 1 and meta["session"] == session
            and type(meta["actual_exit_code"]) is int and meta["actual_exit_code"] == 0
            and meta["observer"] == "root subprocess.run.returncode", "capture外部实际返回码缺失")
    for key in ("identity", "receipt", "guard_tool", "stdout", "stderr"):
        check_ref(meta[key])
    expected = [sys.executable, "-B", str(tool_path), "capture",
                "--identity", str(identity_path), "--identity-sha256", launch["identity"]["sha256"],
                "--out", str(receipt_path)]
    require(meta["argv"] == expected and meta["identity"] == launch["identity"]
            and meta["guard_tool"] == tool_ref and meta["receipt"]["path"] == str(receipt_path),
            "capture来源或ARGV不同")
    receipt = guard.validate_receipt(receipt_path, identity_path=identity_path,
                                    identity_sha256=launch["identity"]["sha256"])
    require(guard.read_json(Path(meta["stdout"]["path"])) == receipt, "capture stdout与回执不同")
    print("NATIVE_STAGE_EXIT=PASS session=" + session, flush=True)

if action == "verify":
    verify_capture(read_launch())
    raise SystemExit(0)

if action == "launch":
    require(guard.file_reference(guard.entity_path(command, main))["sha256"] == command_sha,
            "原命令SHA不同")
    argv = [sys.executable, "-B", str(tool_path), "launch", "--session", session,
            "--train-head", head, "--label", session, "--command-file", str(command),
            "--command-sha256", command_sha, "--identity-out", str(identity_path)]
    stdout_path = command_root / (session + ".launch.stdout")
    stderr_path = command_root / (session + ".launch.stderr")
    meta_path = launch_meta_path
else:
    launch = read_launch()
    probe = guard.probe_identity(identity_path, launch["identity"]["sha256"])
    if probe["status"] == "PENDING":
        print("NATIVE_STAGE_EXIT=PENDING session=" + session, flush=True)
        raise SystemExit(3)
    argv = [sys.executable, "-B", str(tool_path), "capture",
            "--identity", str(identity_path), "--identity-sha256", launch["identity"]["sha256"],
            "--out", str(receipt_path)]
    stdout_path = command_root / (session + ".exit.capture.stdout")
    stderr_path = command_root / (session + ".exit.capture.stderr")
    meta_path = capture_meta_path
for path in (stdout_path, stderr_path, meta_path):
    guard.entity_path(path, main, exists=False)
guard.entity_path(identity_path if action == "launch" else receipt_path, main, exists=False)
completed = subprocess.run(argv, capture_output=True, check=False)
if action == "capture" and completed.returncode == 3:
    # probe与capture之间若仍判alive，保留PENDING原输出且不预占最终侧证。
    require(not os.path.lexists(receipt_path), "PENDING异常占用了最终回执")
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    raise SystemExit(3)
new_bytes(stdout_path, completed.stdout)
new_bytes(stderr_path, completed.stderr)
meta = {"schema": 1, "session": session, "actual_exit_code": completed.returncode,
        "git_source_binding": git_source_binding,
        "identity": guard.file_reference(identity_path) if identity_path.is_file() else None,
        "guard_tool": tool_ref, "stdout": guard.file_reference(stdout_path),
        "stderr": guard.file_reference(stderr_path), "argv": argv,
        "observed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "observer": "root subprocess.run.returncode"}
if action == "launch":
    meta["command_file"] = guard.file_reference(command)
else:
    meta["receipt"] = guard.file_reference(receipt_path) if receipt_path.is_file() else None
new_bytes(meta_path, (json.dumps(meta, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
require(guard.file_reference(tool_path) == tool_ref, "执行期间冻结guard变化")
if completed.returncode:
    raise SystemExit(completed.returncode)
if action == "launch":
    read_launch()
    print("GUARDED_STAGE_RELEASED session=" + session, flush=True)
else:
    verify_capture(launch)
PY
}

capture_timing_stage() {
  guard_timing_action capture "$1"
}

launch_file() {
  local template=$1 expected=$2 session=$3 command_file command_sha role directive
  shift 3
  [[ "$session" =~ ^orig80k-timing20-[A-Za-z0-9_-]+$ ]] || return 2
  command_file="$COMMAND_ROOT/$session.command"
  if [[ "$template" == "$WRAPPER_TEMPLATE" && "$expected" == "$WRAPPER_SHA" ]]; then
    role=run
    directive='export ORIG80K_TIMING_EQ_PROFILE=deterministic100'
  elif [[ "$template" == "$JUDGE_TEMPLATE" && "$expected" == "$JUDGE_SHA" ]]; then
    role=judge
    directive='unset ORIG80K_TIMING_EQ_PROFILE'
  else
    return 2
  fi
  write_bound_runtime "$template" "$expected" "$command_file" "$role" "$@" || return
  command_sha=$(sha256sum -- "$command_file") || return
  command_sha=${command_sha%% *}
  printf 'TEMPLATE_SHA256=%s\nCOMMAND_FILE=%s\nCOMMAND_SHA256=%s\nRUNTIME_PREFIX_LINES=2\nRUNTIME_PROFILE_DIRECTIVE=%s\n' "$expected" "$command_file" "$command_sha" "$directive" || return
  guard_timing_action launch "$session" "$command_sha"
}

require_deterministic_profile() {
  env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 \
    "$MAIN/.venv/bin/python" -B - "$1" "$2" "$TRAIN_HEAD" "$TAG" "$STORE" "$COMMAND_ROOT" "$RESULT_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

session, stage, head, tag, store, commands, results = sys.argv[1:]
store, commands, results = Path(store), Path(commands), Path(results)
flags = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
profile = {"name": "deterministic100", "xla_flags": flags}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    require(path.is_file() and not path.is_symlink(), "缺少实体证据: " + str(path))
    return json.loads(path.read_bytes())

command = commands / (session + ".command")
require(command.is_file() and not command.is_symlink(), "缺少实体运行命令")
parts = command.read_bytes().split(b"\n", 2)
require(len(parts) == 3 and parts[0].startswith(b'set -- "$1" '), "固定参数前缀缺失")
if stage == "run":
    expected_line = b"export ORIG80K_TIMING_EQ_PROFILE=deterministic100"
    expected_sha = "45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f"
elif stage == "judge":
    expected_line = b"unset ORIG80K_TIMING_EQ_PROFILE"
    expected_sha = "17c0d6deea3ab3cd132bf406c968097ff9e7b455c9f22f5e3614c8641d7aafce"
else:
    raise ValueError("未知阶段")
require(parts[1] == expected_line, "显式profile环境前缀不同")
require(hashlib.sha256(parts[2]).hexdigest() == expected_sha, "前缀后的原模板字节不同")
if stage == "run":
    choices = {
        "orig80k-timing20-" + group + "-" + timing + "-0926-" + tag: (group, timing)
        for group in ("full", "count") for timing in ("off", "on")}
    require(session in choices, "不属于本轮四run")
    group, timing = choices[session]
    run = session.removeprefix("orig80k-")
    records = store / "bench/orig80k" / run
    launch = read(records / "launch.json")
    metadata = read(records / "timing_eq/metadata.json")
    require((launch.get("mode"), launch.get("run_name"), launch.get("head")) ==
            ("smoke", run, head), "启动身份不同")
    require((metadata.get("schema"), metadata.get("status"), metadata.get("timing"),
             metadata.get("run_name"), metadata.get("head"), metadata.get("records")) ==
            (2, "PASS", timing, run, head, str(records)), "共同取证身份不同")
    require(launch.get("timing_eq_profile") == metadata.get("timing_eq_profile") == profile,
            "本阶段必须是显式deterministic100记录")
    environment = launch["environment"]
    require(environment.get("ORIG80K_TIMING_EQ_PROFILE") == "deterministic100"
            and environment.get("ORIG80K_SMOKE_EQ_MODE") == timing
            and environment.get("XLA_FLAGS") == flags
            and metadata["fingerprint"]["environment"].get("XLA_FLAGS") == flags,
            "实际档位环境记录不符")
    if timing == "on":
        speed_start = read(records / "speed_start.json")
        speed_run = read(records / "speed_run.json")
        source_sha = hashlib.sha256(
            (store.parent / "scripts/training/tests/check_orig80k_speed.py").read_bytes()).hexdigest()
        for snapshot in (speed_start, speed_run):
            require((snapshot.get("schema"), snapshot.get("head"), snapshot.get("mode"),
                     snapshot.get("steps")) == (2, head, "smoke", 20),
                    "on测速起止schema或20步身份不同")
            require(snapshot.get("timing_eq_profile") == profile and snapshot.get("xla_flags") == flags,
                    "on测速起止必须是显式deterministic100及精确flags")
            require(snapshot.get("argv") == launch["argv"]
                    and snapshot.get("wrapper_sha256") == source_sha
                    and snapshot.get("profiler") is False, "on测速来源或本run实际参数不同")
        require(speed_run.get("success") is True and speed_run.get("entry_calls") == 1
                and speed_run.get("sampler_stopped") is True and not speed_run.get("sampling_error"),
                "on测速入口或主机采样未正常完成")
        require(speed_run["loader"]["batch_size"] == 64 and speed_run["loader"]["workers"] == 4,
                "on实际loader不同")
        require(len(speed_run["save_calls"]) == 1 and speed_run["save_calls"][0]["step"] == 19
                and speed_run["save_calls"][0]["state_step"] == 20, "on未转发末步真实保存")
else:
    choices = {"orig80k-timing20-" + group + "-judge-" + tag: group for group in ("full", "count")}
    require(session in choices, "不属于本轮两judge")
    result = read(results / (choices[session] + ".json"))
    require(result.get("schema") == 2 and result.get("status") == "PASS"
            and result.get("head") == head and result.get("timing_eq_profile") == profile,
            "正式judge必须显式返回deterministic100 PASS")
    require((result.get("fetched_batches"), result.get("used_batches"), result.get("scalar_steps"),
             result.get("state_step"), result.get("loop_step")) == (21, 20, 20, 20, 19),
            "正式judge范围不符")
print("DETERMINISTIC_PROFILE=PASS session=" + session, flush=True)
PY
}

require_timing_stage() {
  local session=$1 log=$2 stage=$3 marker=$4 key value actual
  guard_timing_action verify "$session" || return
  test -f "$log" && test ! -L "$log" || return 2
  for key in WRAPPER_COMMAND_EXIT WRAPPER_TEE_EXIT WRAPPER_FOOTER_PRINTF_EXIT WRAPPER_FOOTER_TEE_EXIT WRAPPER_EXIT_CODE; do
    test "$(rg -c "^${key}=" "$log")" = 1 || return 2
    rg -q "^${key}=0$" "$log" || return 2
  done
  if [ "$stage" = run ]; then
    for key in RUNNER_EXIT_CODE COMPLETION_EXIT_CODE COMPLETION_TEE_EXIT_CODE WRAPPER_IDENTITY_EXIT_CODE WRAPPER_VERSION_EXIT_CODE; do
      test "$(rg -c "^${key}=" "$log")" = 1 || return 2
      rg -q "^${key}=0$" "$log" || return 2
    done
  elif [ "$stage" = judge ]; then
    test "$(rg -c '^JUDGE_EXIT_CODE=' "$log")" = 1 || return 2
    rg -q '^JUDGE_EXIT_CODE=0$' "$log" || return 2
  else
    return 2
  fi
  for key in WRAPPER_SESSION WRAPPER_TRAIN_HEAD WRAPPER_COMMAND_FILE; do
    test "$(rg -c "^${key}=" "$log")" = 1 || return 2
    value=$(sed -n "s/^${key}=//p" "$log") || return
    case "$key" in
      WRAPPER_SESSION) test "$value" = "$session" || return 2 ;;
      WRAPPER_TRAIN_HEAD) test "$value" = "$TRAIN_HEAD" || return 2 ;;
      WRAPPER_COMMAND_FILE) test "$value" = "$COMMAND_ROOT/$session.command" || return 2 ;;
    esac
  done
  actual=$(sha256sum -- "$COMMAND_ROOT/$session.command") || return
  actual=${actual%% *}
  for key in WRAPPER_COMMAND_SHA256_EXPECTED WRAPPER_COMMAND_SHA256_ACTUAL; do
    test "$(rg -c "^${key}=" "$log")" = 1 || return 2
    test "$(sed -n "s/^${key}=//p" "$log")" = "$actual" || return 2
  done
  test "$(rg -c "$marker" "$log")" = 1 || return 2
  require_deterministic_profile "$session" "$stage"
}
```

这些检查只是输出/版本预检，不读取一个不存在的“统一P1/100闸门JSON接口”。正式执行者必须列清P1按用户批准口径保留的记录/限制、100步真实通过及独立退出证据，并引用INPUT已批准补记限制及沿用的决定；缺项时不执行下面的训练命令。runner仍会独立做完整preflight；guard的RELEASED只说明门闩释放，不能代替后续capture/receipt/原日志联合验收。

## 四个独立run命令及两个judge

先背靠背调用两个off的guard launch；launch只等待本任务身份持久化及门闩释放，不等待训练结束，因此full-off与count-off并行占用0–3和4–7两组卡。随后各库只依赖自己的前一阶段，以下代码块按对应退出事件执行，**正文排列顺序不是跨库等待屏障，不得把整页一次性运行**。每个run都等真实保存/恢复、独立pane退出、capture父进程实际返回及原日志核验完成后才进入本库下一run；不等待另一库。六个任务都经guard门闩进入各自detached tmux，同时记录原生window/pane/PID与wrapper/harness PID。

两个off的共同起点：

```bash
launch_file "$WRAPPER_TEMPLATE" "$WRAPPER_SHA" "orig80k-$FULL_OFF" \
  "$TRAIN_HEAD" "$FULL_OFF" off 0,1,2,3 "$FULL" "$FULL_ASSETS" "$FULL_NORM" \
  "orig80k-$FULL_OFF" "$STORE/logs/orig80k-$FULL_OFF.wrapper.log" || exit 2
launch_file "$WRAPPER_TEMPLATE" "$WRAPPER_SHA" "orig80k-$COUNT_OFF" \
  "$TRAIN_HEAD" "$COUNT_OFF" off 4,5,6,7 "$COUNT" "$COUNT_ASSETS" "$COUNT_NORM" \
  "orig80k-$COUNT_OFF" "$STORE/logs/orig80k-$COUNT_OFF.wrapper.log" || exit 2
```

收到full-off准确pane的退出事件后，实际capture再只读verify；通过后立即启动full-on，不等待count-off、count-on或count judge。任一非零或PENDING都停止对应依赖步骤；最终回执文件不覆盖重捕获，重复验收使用verify：

```bash
capture_timing_stage "orig80k-$FULL_OFF" || exit 2
require_timing_stage "orig80k-$FULL_OFF" "$STORE/logs/orig80k-$FULL_OFF.wrapper.log" run \
  '^TIMING_EQ_RUN=PASS timing=off fetched=21 used=20 state_step=20 real_save=1$' || exit 2
launch_file "$WRAPPER_TEMPLATE" "$WRAPPER_SHA" "orig80k-$FULL_ON" \
  "$TRAIN_HEAD" "$FULL_ON" on 0,1,2,3 "$FULL" "$FULL_ASSETS" "$FULL_NORM" \
  "orig80k-$FULL_ON" "$STORE/logs/orig80k-$FULL_ON.wrapper.log" || exit 2
```

收到count-off准确pane的退出事件后，独立验收并立即启动count-on，不等待full链；可早于或晚于上一段full-off事件发生：

```bash
capture_timing_stage "orig80k-$COUNT_OFF" || exit 2
require_timing_stage "orig80k-$COUNT_OFF" "$STORE/logs/orig80k-$COUNT_OFF.wrapper.log" run \
  '^TIMING_EQ_RUN=PASS timing=off fetched=21 used=20 state_step=20 real_save=1$' || exit 2
launch_file "$WRAPPER_TEMPLATE" "$WRAPPER_SHA" "orig80k-$COUNT_ON" \
  "$TRAIN_HEAD" "$COUNT_ON" on 4,5,6,7 "$COUNT" "$COUNT_ASSETS" "$COUNT_NORM" \
  "orig80k-$COUNT_ON" "$STORE/logs/orig80k-$COUNT_ON.wrapper.log" || exit 2
```

full-on结束并独立验收后，立即执行full CPU judge，不等待count链。COMMON参数始终传原模板实体副本，不传任一加过两行前缀的run命令文件；judge命令自身第二行明确unset selector：

```bash
capture_timing_stage "orig80k-$FULL_ON" || exit 2
require_timing_stage "orig80k-$FULL_ON" "$STORE/logs/orig80k-$FULL_ON.wrapper.log" run \
  '^TIMING_EQ_RUN=PASS timing=on fetched=21 used=20 state_step=20 real_save=1$' || exit 2
launch_file "$JUDGE_TEMPLATE" "$JUDGE_SHA" "orig80k-timing20-full-judge-$TAG" \
  "$COMMON_WRAPPER" "$WRAPPER_SHA" "$TRAIN_HEAD" "$FULL_OFF" "$FULL_ON" \
  "$RESULT_ROOT/full.json" "orig80k-timing20-full-judge-$TAG" \
  "$STORE/logs/orig80k-timing20-full-judge-$TAG.wrapper.log" || exit 2
```

count-on结束并独立验收后，立即执行counting CPU judge，不等待full链；可与full CPU judge并行。两份judge的结果路径都位于各自run records之外：

```bash
capture_timing_stage "orig80k-$COUNT_ON" || exit 2
require_timing_stage "orig80k-$COUNT_ON" "$STORE/logs/orig80k-$COUNT_ON.wrapper.log" run \
  '^TIMING_EQ_RUN=PASS timing=on fetched=21 used=20 state_step=20 real_save=1$' || exit 2
launch_file "$JUDGE_TEMPLATE" "$JUDGE_SHA" "orig80k-timing20-count-judge-$TAG" \
  "$COMMON_WRAPPER" "$WRAPPER_SHA" "$TRAIN_HEAD" "$COUNT_OFF" "$COUNT_ON" \
  "$RESULT_ROOT/count.json" "orig80k-timing20-count-judge-$TAG" \
  "$STORE/logs/orig80k-timing20-count-judge-$TAG.wrapper.log" || exit 2
```

full judge结束后完成本库的capture与联合验收，不等待count judge：

```bash
capture_timing_stage "orig80k-timing20-full-judge-$TAG" || exit 2
require_timing_stage "orig80k-timing20-full-judge-$TAG" \
  "$STORE/logs/orig80k-timing20-full-judge-$TAG.wrapper.log" judge \
  '^TIMING_EQ=PASS fetched=21 used=20 scalar_steps=20 full_state=bitwise real_restore=both$' || exit 2
```

count judge结束后完成本库的capture与联合验收，不等待full judge；两个judge及其独立退出证据、显式deterministic100记录都通过才可汇总本阶段：

```bash
capture_timing_stage "orig80k-timing20-count-judge-$TAG" || exit 2
require_timing_stage "orig80k-timing20-count-judge-$TAG" \
  "$STORE/logs/orig80k-timing20-count-judge-$TAG.wrapper.log" judge \
  '^TIMING_EQ=PASS fetched=21 used=20 scalar_steps=20 full_state=bitwise real_restore=both$' || exit 2
```

judge wrapper实际传完整 `--off-records/--on-records`、两个 `completion.json`、两个对应wrapper日志、`--head`和独立`--out`；使用 `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`。两组judge各自必须有唯一内容成功行、`JUDGE_EXIT_CODE=0`及5项外层WRAPPER退出各唯一0，结果JSON必须schema2、status为PASS且timing_eq_profile精确为deterministic100及两项flags；同时独立receipt须原生 `pane_dead=1/pane_dead_status=0`、无信号，父进程捕获的capture实际返回码为0且所有绑定一致。任一失败保留现场、停止依赖步骤，不改容差、不重写日志、不用已有可读性20步替代。

保留所有窗口直至命令/identity/capture/receipt/日志及结果已核验并归档。之后由根代理按本轮完整会话清单逐个精确清理，清理前后核对tmux列表，只移除该任务；失败或归属不明时保留并报告。本启动档不自动kill任何session，也不使用session消失作为完成证明。

## 验收依据与限制

`timing_eq/manifest.json`绑定metadata、fetch_inputs、model_inputs、full_state四份记录，目录不加其他文件。每run严格为21次取批、20次模型输入/更新、一次真实save返回；最后一批未用于训练。完整状态step20/loop19的params、EMA、optimizer、step及其结构按dtype/shape/bytes逐位比较；CPU输入阶段允许主机dtype不同及四None键等价的规则不应用到本对照。五标量由log100的step0加final尾窗1…19拼齐，不能擅自覆盖log_interval。

completion必须是真实恢复的smoke20、checkpoint集合`[19]`，UUID/HEAD/run一致、全部11项小文件证据完整，恢复EMA与训练末步及共同完整状态摘要一致。两侧launch与共同取证metadata必须均为显式deterministic100且实际XLA_FLAGS精确对应；on两份speed JSON均schema2并与launch/metadata逐项相同。off没有speed_start/speed_run/step_timing/host_samples/disk_samples五种文件；on为真实 `speed.run(mode=smoke)`，同步只在step19保存前，包含0.5秒磁盘采样且有效样本落在原生commit之前。没有保存窗口样本或采样错误就失败，不能用终态大小放宽。

两侧同一额外观测层的device_get/CPU哈希会带来同步和内存/耗时开销，量具记录该开销；**20步不用于吞吐、ETA或无观测运行证明**。真实300步perf、正式预算和两个80k仍需各自后续闸门。缓存/临时权重未经后续明确清理动作不得自动删除；本阶段原始记录、命令文件、driver、wrapper、completion及judge记录保留，归档只复制Git不能还原的实测记录，不把候选.sh复制为docs脚本附件。

## 控制文本的验证与真实运行边界

两模板及两份测试已原样纳入 `scripts/training/tests/`：模板SHA保持45b4…/17c0…；传播测试SHA为 `1b128797326af00614b3b948a75cee05315f0d141274be8c09f9edf0fff97444`，参数测试SHA为 `ae9be2bb80b9f4956f1690494ad6dfab0a2b26156eb4a68c995aa972abb4266f`。已有11项传播和11项参数测试的原验证见[模板纳入检查](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-integration-candidate-00bdabc-20260926/validation.json)，f8原18项stdlib验证见[guard纳入检查](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/tmux-exit-guard-integration-candidate-00bdabc-20260926/validation.json)。这些是控制模板验证，不是本阶段真实训练结果。

本档案沿用Git源/工作树/runtime字节检查及外部父记录；两模板与f8保持原字节，仅在runtime原模板前增加固定参数及档位选择两行，并在联合验收中核对原模板余体SHA、launch/metadata与judge结果的显式deterministic100。runner/contract/speed/observer的已批准档位实现由新Beta锁定，训练模型源码和原超参不改。本轮两行前缀必须使用本候选新增的参数/环境绑定小验证，旧单前缀测试不能直接代替。正式Beta后需要核对实际source绑定、所有命令/输出冲突、GPU空闲、磁盘预算及前置指纹后才能起跑。新Beta SHA、实际会话身份与运行结果全部待真实填写；本档案准备时没有launch/capture/训练或恢复结果。

原11项传播测试未覆盖最终append完整写0后自身失败，不能扩写成“最外退出已被日志证明”。本阶段所有六任务必须同时通过独立pane回执、capture父真实返回、父适配宿主状态和原内容验收；PENDING或任一非零不放行。发生失败保留全部产物，停止依赖阶段并报告，不改名、覆盖、删日志或自行放宽判据。

四个run README见[full-off](../timing20-full-off-0926-20260927T012426Z/README.md)、[full-on](../timing20-full-on-0926-20260927T012426Z/README.md)、[count-off](../timing20-count-off-0926-20260927T012426Z/README.md)、[count-on](../timing20-count-on-0926-20260927T012426Z/README.md)。只有两库judge及全部独立退出证据都通过，才形成本阶段结果；300步perf、正式预算和80k起跑前准备仍各自有后续闸门。
