# 两库追加20步off重复性诊断：已执行，结果DIFFER

用户已批准当前off/on取证结束后，两库各追加一次同配置20步off，4+4并行，严格判据不变，仅作诊断。根代理已采用 `timing20-full-off-0926-20260927T003051Z`、`timing20-count-off-0926-20260927T003051Z`；TAG是准备标签，不是实际起跑时间。本档案用于新Beta起跑前锚定；尚未执行新任务，模板/量具/argv/容差/依赖保持原样。

**实测追加**：上述启动前模板由实际Beta`b60ec2b59e0ba012cae0998c5f8713b2aef07b2e`锁定，两个run均于2026-09-27 00:54:30 UTC起跑，真实保存恢复与原生退出0通过。独立CPU比较于01:05:59完成，进程0、严格结果DIFFER/FAIL；每库80/100标量与147/201状态叶不等，见[result](result.md)。下方保留原模板与占位规则，实际调度仅把TRAIN_HEAD展开为b60并留SHA，旧输出已存在不得重跑覆盖。下一阶段的确定性档是其后新授权，不属于本次normal执行。

旧四run真实保存/恢复及原生退出已通过，两正式off/on judge实际FAIL“20步五标量不是逐位一致”，原生退出1/capture1；完整核验报告SHA为 `113cec3b17a19e85bd55c7a2b4420f150eac17dbea69a2b3c3e6c38d62265e99`，已由`c6d726f3834f871fcef10f33fa2530321bee534f`归档，见[原失败结果](../orig80k-timing20-0926/result.md)。每库21 fetch/20 model+RNG一致，100个标量80个不同，末态147/201叶不同。追加off不改写这些失败；[机理说明](../orig80k-timing20-0926/records/diagnostics/timing20-step0-mechanism-note-20260927.md)没有已证根因，初始完整state未摘要、相同param_norm仅为kernel聚合的边界仍保留。

## 起跑前版本、参数与新输出

先结束并归档当前V11.14失败结果，再以新 `commitV11.15Beta`锁定本launch与两份run README；完整SHA待真实提交后填写。旧off基线固定 `b0efbde61e38411fb1b9114eec8485d36a4ee9a0`，新Beta相对它只能有本轮docs/计划变化。三规则仍固定d710；训练源码、runner、speed/observer、模板、依赖、输入/资产指纹逐字不变。新旧HEAD分别保留，不改BASE，不改变正式judge的同HEAD要求。

两run均smoke20、b64/fsdp4/workers4/seed42、原modul512/4×4/max32、warmup10000、peak=decay5e-5、decay_steps100000、AdamW clip1.0、EMA0.999、log100/save10000/keep10000，W&B online，normal env。只清遗留CPU平台，不设置确定性XLA_FLAGS、不覆盖JAX_ENABLE_X64绕过拒绝，真实保存19并CPU恢复。GPU固定full0–3、count4–7。

以下为未来同一控制shell中的设置，当前不执行：

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
OLD_HEAD=b0efbde61e38411fb1b9114eec8485d36a4ee9a0
RULES=d710d8489b88aa75770af3452fd7c9b374deaef7
TRAIN_HEAD='<待commitV11.15Beta真实完整40位SHA>'
TAG=20260927T003051Z
COMMAND_ROOT="$STORE/bench/orig80k-offrepeat-commands-$TAG"
RESULT_ROOT="$STORE/bench/orig80k-offrepeat-results-$TAG"
WRAPPER_TEMPLATE="$MAIN/scripts/training/tests/timing20-wrapper.sh"
WRAPPER_SHA=45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f
GUARD_TOOL="$STORE/bench/orig80k-build-preflight-0925/tmux-exit-guard-candidate-20260926/tmux_exit_guard.py"
GUARD_SHA256=f8dada4025dc49f260697e2fb768e922cbea7b8f98afe5cc8b98555648bff52b
FULL_NEW="timing20-full-off-0926-$TAG"
COUNT_NEW="timing20-count-off-0926-$TAG"
FULL="$STORE/datasets/16task-pub-1600ep"
COUNT="$STORE/datasets/4task-counting-pub-400ep"
FULL_ASSETS="$STORE/train-assets/mme_vla_suite"
COUNT_ASSETS="$STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep"
FULL_NORM=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
COUNT_NORM=a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ && "$TRAIN_HEAD" != "$OLD_HEAD" ]] || exit 2
test "$(git -C "$MAIN" rev-parse HEAD)" = "$TRAIN_HEAD" || exit 2
test -z "$(git -C "$MAIN" status --porcelain)" || exit 2
git -C "$MAIN" merge-base --is-ancestor "$OLD_HEAD" "$TRAIN_HEAD" || exit 2
git -C "$MAIN" diff --exit-code "$RULES" "$TRAIN_HEAD" -- AGENTS.md CLAUDE.md greatlakes.md || exit 2
git -C "$MAIN" diff --exit-code "$OLD_HEAD" "$TRAIN_HEAD" -- src packages shared scripts pyproject.toml uv.lock || exit 2
changed=$(git -C "$MAIN" diff --name-only --no-renames "$OLD_HEAD" "$TRAIN_HEAD") || exit 2
while IFS= read -r path; do
  case "$path" in docs/*|0925-orig-80k-full-counting-4plus4-plan.md|'') ;;
    *) printf '拒绝非文档差异：%s\n' "$path" >&2; exit 2 ;;
  esac
done <<< "$changed"
for path in "$COMMAND_ROOT" "$RESULT_ROOT"; do
  test ! -e "$path" && test ! -L "$path" || exit 2
done
for run in "$FULL_NEW" "$COUNT_NEW"; do
  for path in "$STORE/bench/orig80k/$run" "$STORE/train-runs/mme_vla_suite/$run" \
    "$STORE/logs/$run.driver.log" "$STORE/logs/orig80k-$run.wrapper.log" \
    "$STORE/cache/jax/$run" "$STORE/cache/cuda/$run" "$STORE/logs/wandb/$run" \
    "$STORE/cache/wandb/$run" "$STORE/cache/wandb-config/$run" "$STORE/cache/wandb-data/$run" \
    "$STORE/cache/xdg-data/$run"; do
    test ! -e "$path" && test ! -L "$path" || exit 2
  done
done
mkdir "$COMMAND_ROOT" "$RESULT_ROOT" || exit 2
```

原两off的manifest、completion、原生退出及独立复核已归档的小记录路径/UUID/SHA须由根代理在新Beta前固定，并核对当前实体；不以现场重算值重新定义旧基线。复验前仍核数据/资产/依赖指纹、指定GPU空闲和新增两份checkpoint/临时/cache/log预算，不能将300GiB最低保护当完整预算。禁止预建run/records根或删除冲突产物。

旧基线的固定路径、UUID、manifest/completion/identity/native/capture摘要，以及原独立报告，已写入[baseline-pins.json](records/baseline-pins.json)。此文件与启动档一起入新Beta；起跑前重新核原实体bytes/SHA及旧独立报告，不以现场新摘要重新定义旧基线。旧记录继续保留b0真实HEAD。

## 原样模板、参数绑定和独立退出

run模板完整SHA保持45b4…，不复制其源码到docs附件。runtime只在v1-store新命令根创建，第一行以printf %q绑定固定参数，再逐字附原模板；传guard的是整份runtime新SHA。两新run没有on模式或正式judge，下面复用旧控制函数的run分支；未调用的judge分支不构成任何比较授权。

```bash
write_bound_runtime() {
  local template=$1 expected=$2 command_file=$3 actual
  shift 3
  test "$#" -gt 0 || return 2
  actual=$(sha256sum -- "$template") || return
  [[ "${actual%% *}" == "$expected" ]] || return 2
  (
    set -o noclobber
    {
      printf 'set -- "$1"' || exit 2
      printf ' %q' "$@" || exit 2
      printf '\n' || exit 2
      cat "$template" || exit 2
    } > "$command_file"
  ) || return 2
  bash -n "$command_file" || return 2
}
```

以下父适配保持既有协议；仅移除本诊断不使用的judge模板源码依赖，绑定正式Beta guard/run模板与工作树、原f8 runtime。每次动作要求新HEAD/clean；旧基线退出回执保持原b0ef及原工具路径，由既有归档检查链引用，不用本新HEAD控制器去伪装旧回执。

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
  local template=$1 expected=$2 session=$3 command_file command_sha
  shift 3
  [[ "$session" =~ ^orig80k-timing20-[A-Za-z0-9_-]+$ ]] || return 2
  command_file="$COMMAND_ROOT/$session.command"
  write_bound_runtime "$template" "$expected" "$command_file" "$@" || return
  command_sha=$(sha256sum -- "$command_file") || return
  command_sha=${command_sha%% *}
  printf 'TEMPLATE_SHA256=%s\nCOMMAND_FILE=%s\nCOMMAND_SHA256=%s\n' "$expected" "$command_file" "$command_sha" || return
  guard_timing_action launch "$session" "$command_sha"
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
}
```

## 两个off并行启动与分别收尾

只有上述前置实际通过后，连续发出两个新launch；不等待full训练结束才起count。原45b模板内每run真实训练后立即串行CPU恢复，与旧off流程相同；不改成perf的“两training done才恢复”流程。两个wrapper都通过真实恢复与独立退出后，才做后面的纯CPU重复性比较。

```bash
launch_file "$WRAPPER_TEMPLATE" "$WRAPPER_SHA" "orig80k-$FULL_NEW" \
  "$TRAIN_HEAD" "$FULL_NEW" off 0,1,2,3 "$FULL" "$FULL_ASSETS" "$FULL_NORM" \
  "orig80k-$FULL_NEW" "$STORE/logs/orig80k-$FULL_NEW.wrapper.log" || exit 2
launch_file "$WRAPPER_TEMPLATE" "$WRAPPER_SHA" "orig80k-$COUNT_NEW" \
  "$TRAIN_HEAD" "$COUNT_NEW" off 4,5,6,7 "$COUNT" "$COUNT_ASSETS" "$COUNT_NORM" \
  "orig80k-$COUNT_NEW" "$STORE/logs/orig80k-$COUNT_NEW.wrapper.log" || exit 2
```

收到对应pane结束事件后分别执行；PENDING或非零不放行，不忙轮询、不覆盖已生成回执：

```bash
capture_timing_stage "orig80k-$FULL_NEW" || exit 2
require_timing_stage "orig80k-$FULL_NEW" "$STORE/logs/orig80k-$FULL_NEW.wrapper.log" run \
  '^TIMING_EQ_RUN=PASS timing=off fetched=21 used=20 state_step=20 real_save=1$' || exit 2
```

```bash
capture_timing_stage "orig80k-$COUNT_NEW" || exit 2
require_timing_stage "orig80k-$COUNT_NEW" "$STORE/logs/orig80k-$COUNT_NEW.wrapper.log" run \
  '^TIMING_EQ_RUN=PASS timing=off fetched=21 used=20 state_step=20 real_save=1$' || exit 2
```

这里TIMING_EQ_RUN仅为单run采集/真保存完成，不是两次运行相等。保留wrapper/driver独立日志和真实进程返回；只在原生pane、capture父实际返回、宿主父适配返回、完成器/tee和原日志全通过后接受该run。原append盲区不因日志0消失。所有窗口按本轮精确清单保留至证据核验，不自动清理。

## 独立纯CPU比较方案：保留两真实HEAD，不调用正式judge

以下正文只在两新run各自真恢复及独立退出通过后执行。它调用现有read_side，分别传旧b0ef和新真实Beta，严格验证每侧自身文件/配置/来源/finite/恢复；完整配置只允许现有IDENTITY_FIELDS的exp_name、derived checkpoint_dir两项不同。原HEAD分别留证；provenance仅在单独比较投影排除已核实的git_head_of_cwd，其他字段、模块和SHA全部严格相同。原记录不改、不归一化头字段再调用compare_sides，也不改正式judge跨HEAD判据。

read_side的validate_provenance对各自存档snapshot调用_assert_clean_snapshot(snapshot, passed_head)，随后重核当前模块字节；它没有要求当前HEAD等于旧passed_head，因此仅docs的新Beta可分别读取旧b0与新记录，不需改记录或猴补检查。compare_sides另要求同HEAD并比较完整provenance，本诊断不调用它。新旧受保护源码/Git blob相等的前置仍必须先通过。

两库都比较21次fetch tree、20次model tree/RNG、20×5原始hex、完整state的groups/treedef/EMA/路径/step；不设容差。结果写全新独立JSON，read_validation_completed=true表示两侧验证及比较正常读完，进程返回0；comparison_status另为PASS或FAIL、status另记MATCH或DIFFER，不一致就是独立诊断比较FAIL，不能从进程0推导相等。read_side/配置/IO异常则进程非0、不能接受其为完整比较。根代理须另外捕获该短CPU命令实际返回及stdout/stderr；若用tee同时记录其返回。JSON可读不代替真实返回，任何结果都不改原off/on失败或放行perf。

```bash
cd "$MAIN"
env -u JAX_PLATFORM_NAME -u PYTHONPATH -u PYTHONHOME \
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  UV_PROJECT_ENVIRONMENT="$MAIN/.venv" UV_CACHE_DIR="$STORE/cache/uv" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync python - "$MAIN" "$OLD_HEAD" "$TRAIN_HEAD" "$TAG" "$RESULT_ROOT/repeat-comparison.json" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

main, old_head, new_head, new_tag, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5])
assert old_head == "b0efbde61e38411fb1b9114eec8485d36a4ee9a0" and new_head != old_head
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=main, text=True).strip() == new_head
assert subprocess.check_output(["git", "status", "--porcelain"], cwd=main, text=True) == ""
sys.path.insert(0, str(main / "scripts/training/tests"))
import check_orig80k_timing_equiv as eq
config_record, _, _ = eq.training_imports()
assert config_record.IDENTITY_FIELDS == ("train_config.fields.exp_name", "derived.checkpoint_dir")
assert out.is_absolute() and out == out.resolve() and out.is_relative_to(main / "v1-store")
assert out.parent.is_dir() and not out.exists() and not out.is_symlink()
results = {}
for group in ("full", "count"):
    old_run = f"timing20-{group}-off-0926-20260926T231534Z"
    new_run = f"timing20-{group}-off-0926-{new_tag}"
    sides = []
    for run, head in ((old_run, old_head), (new_run, new_head)):
        records = main / "v1-store/bench/orig80k" / run
        side = eq.read_side(records, records / "completion.json",
                            main / "v1-store/logs" / ("orig80k-" + run + ".wrapper.log"), "off", head)
        assert side["metadata"]["run_name"] == run and side["metadata"]["head"] == head
        sides.append(side)
    old, new = sides
    left, right = old["metadata"], new["metadata"]
    assert left["run_uuid"] != right["run_uuid"] and left["records"] != right["records"]
    identity_differences = config_record.compare_records(left["complete"], right["complete"],
                                                         config_record.IDENTITY_FIELDS)
    # 各自真实HEAD已先核对；只在比较投影中排除这一个声明身份字段，原记录不改。
    def provenance_payload(meta, head, name):
        value = meta[name]
        assert value["git_head_of_cwd"] == head and value["git_porcelain_of_cwd"] == ""
        return {key: item for key, item in value.items() if key != "git_head_of_cwd"}
    checks = {
        "fingerprint": left["fingerprint"] == right["fingerprint"],
        "toolchain": left["toolchain"] == right["toolchain"],
        "loader": left["loader"] == right["loader"],
        "original_save": left["original_save"] == right["original_save"],
        "provenance_start": provenance_payload(left, old_head, "provenance_start") ==
                            provenance_payload(right, new_head, "provenance_start"),
        "provenance_end": provenance_payload(left, old_head, "provenance_end") ==
                          provenance_payload(right, new_head, "provenance_end"),
        "fetch_trees": [row["tree"] for row in old["fetches"]] == [row["tree"] for row in new["fetches"]],
        "model_trees_rng": [(row["tree"], row["rng"]) for row in old["models"]] ==
                           [(row["tree"], row["rng"]) for row in new["models"]],
        "scalar_hex_20x5": old["scalars"] == new["scalars"],
    }
    for key in ("groups", "original_treedefs", "ema_leaves", "ema_leaf_paths", "state_step", "loop_step"):
        checks["state_" + key] = old["state"][key] == new["state"][key]
    results[group] = {
        "repeat_equal": all(checks.values()), "checks": checks,
        "old": {"head": old_head, "run_name": old_run, "run_uuid": left["run_uuid"], "bindings": old["bindings"]},
        "new": {"head": new_head, "run_name": new_run, "run_uuid": right["run_uuid"], "bindings": new["bindings"]},
        "allowed_config_identity_differences": identity_differences,
        "scalar_mismatches": [{"step": a["step"], "key": key, "old_hex": a["hex"][key], "new_hex": b["hex"][key]}
            for a, b in zip(old["scalars"], new["scalars"], strict=True)
            for key in sorted(a["hex"]) if a["hex"][key] != b["hex"][key]],
    }
matched = all(value["repeat_equal"] for value in results.values())
report = {"schema": 1, "diagnostic_only": True, "read_validation_completed": True,
          "comparison_status": "PASS" if matched else "FAIL", "status": "MATCH" if matched else "DIFFER",
          "old_head": old_head, "new_head": new_head, "results": results,
          "scope": "同库旧off/新off严格重复性诊断；不代替或改写原off/on正式judge，不证明一般确定性"}
with out.open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    stream.write("\n")
print("OFF_REPEAT_DIAGNOSTIC_READ=COMPLETE", flush=True)
print("OFF_REPEAT_DIAGNOSTIC_EQ=" + report["comparison_status"], flush=True)
raise SystemExit(0)
PY
```

此处是可审阅的内联CPU方案，不新增仓库工具或诊断框架；当前未执行。一次重复相同不能证明一般确定性；重复不同也不直接定位初始state、编译算法或线程时序。旧off/on正式FAIL、已有完整取证与初态未记录的边界不变。结果归档后由根代理与用户判断下一步，perf继续冻结。

两run十二节候选见[full](../timing20-full-off-0926-20260927T003051Z/README.md)和[count](../timing20-count-off-0926-20260927T003051Z/README.md)。当前没有新Beta、运行身份、保存恢复结果或重复比较结论；本轮仍止步正式80k起跑前。
