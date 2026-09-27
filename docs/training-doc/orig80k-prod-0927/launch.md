# 两个原版normal80k：起跑前launch（尚未启动）

最新授权：用户明确「确认无误后可以直接开始两个训练」。共同V11.18Beta、clean状态和全部起跑前检查真实通过后，由root直接启动两条既定80k，无需再次请求批准。此前stop-before要求仅作为历史决定保留；检查器本身仍只读并返回STOP_BEFORE_PROD_LAUNCH，root随后独立调用f8 launch。当前两个prod尚未启动。

用户已确认两个run名和4+4分组；两者由同一真实Beta锁定，从同一clean HEAD起跑。normal perf与B预算结果已归档到 `bafecc658c45fbd940fb1655e7057d2b1d2c0593`。本启动档在起跑前固定配置和命令，后续真实时间、UUID、PID与W&B ID记入各run的launch/final/start记录及控制账本；正式长训练期间源码与本档案冻结，正文的准备状态不作为实时运行状态。

| 固定run_name | GPU | lib | assets父目录 |
|---|---|---|---|
| `v2-orig-16task-pub1600ep-modul-b64-80k` | `0,1,2,3` | `v1-store/datasets/16task-pub-1600ep` | `v1-store/train-assets/mme_vla_suite` |
| `v2-orig-counting-pub400ep-modul-b64-80k` | `4,5,6,7` | `v1-store/datasets/4task-counting-pub-400ep` | `v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep` |

[full十二节README](../v2-orig-16task-pub1600ep-modul-b64-80k/README.md)、[count十二节README](../v2-orig-counting-pub400ep-modul-b64-80k/README.md)与[起跑前清单](checklist.md)列明原有闸门。现行实现锚点为 `aec86db64e5178e63d9e7f77d3f5bc235db16390`，确定性20实际Beta为 `c71d5255597db2f26930b7b5684a1d5b2994cf75`；均不能填成未来共同V11.18Beta。数据49a、INPUT3a、P1/100的00bd/ecf保持原版本；新Beta对实际perf来源与受保护训练代码、依赖/数据/资产的兼容性须按证据核对。

## normal环境与原参数

prod只调用现有 `run_orig80k.sh prod <run> <gpus> <lib> <assets>` 五参数接口；不追加steps、batch、lr或worker覆盖。固定80000步、b64、fsdp4、workers4、seed42、modul budget512/4×4=16 token每帧/最多32 memory帧、memory_token_dim1024、streaming_obs_horizon16、warmup10000、peak_lr=decay_lr=5e-5、decay_steps100000、AdamW clip1.0、EMA0.999、log100、save10000、keep10000。pi05_base初始化、W&B online/project openpi保持。

实际命令明确清除 `ORIG80K_TIMING_EQ_PROFILE`、`ORIG80K_SMOKE_EQ_MODE`、`XLA_FLAGS`、`TRAIN_TIMING_STEPS`、`JAX_PLATFORMS/JAX_PLATFORM_NAME`及Python路径污染。不依赖控制shell/tmux继承。继承的 `JAX_ENABLE_X64`不覆写，实际True由runner拒绝。prod不经过speed包装，也不装20步共同取证层或100步摘要保存替身。

## 共用预算必须来自真实normal300

两侧使用**同一** `ORIG80K_DISK_BUDGET_JSON`绝对实体路径与已审核固定64位 `ORIG80K_DISK_BUDGET_SHA256`。不得在起跑时现算摘要并默认当作期望，不接受仅launch记录或旧normal缺profile记录。

来源必须是两不同normal300 run、共同真实perf HEAD、各自state300/checkpoint299真恢复、原始磁盘采样、联合报告及清理前checkpoint measurement receipt。launch与两份speed起止记录都明确normal，speed schema2；联合报告各侧normal/实际flags与本run一致。预算schema2是独立格式，不能凭空要求report外层有schema字段。

B档已由用户确认：每份checkpoint按16GiB、两侧共16份保留，即checkpoint_total_bytes=274877906944；增长余量84798963712B；保存采样余量34359738368B（32GiB），另加真实E合计5531541504B，save_temporary_peak_bytes=39891279872；日志/cache余量68719476736B（64GiB）。三项合计383488663552B，约357.151649GiB。预算实体固定为 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-perf300-results-20260926T232339Z/prod-disk-budget.json`，真实budget SHA `3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20`，margin-source SHA `09ea4e4739e11e2851a013b00e9a2f3322d0b6d9b7859f9b749495be90d375bf`。第8阶段于04:20:41完成正式预算PASS，所需383488663552B、当时可用708412715008B，原生/capture/父宿主均0；该可用量不是未来正式起跑保证，最终独立复核已通过，已归档到 `bafecc658c45fbd940fb1655e7057d2b1d2c0593`。

保留原schema2与公式，不选新倍率：

```text
retained_bytes = 8*C_full + 8*C_counting
checkpoint_total_bytes = retained_bytes + checkpoint_growth_margin_bytes
save_temporary_peak_bytes = max(0,P_full-F_full) + max(0,P_counting-F_counting) + save_sampling_margin_bytes
required_bytes = max(300*2^30,
    checkpoint_total_bytes + save_temporary_peak_bytes + logs_cache_margin_bytes)
```

C为单份299实测分配量；P/F为同一run根的采样最大/最终分配量。0.5秒只读采样用st_blocks*512并按设备/inode去重，观察最大值只是连续峰值的下界；不得因本次保存错峰或采样增量0而缩减保守余量。B档及预算已正式验收，真实budget SHA `3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20`，margin-source SHA `09ea4e4739e11e2851a013b00e9a2f3322d0b6d9b7859f9b749495be90d375bf`。第8阶段于04:20:41完成正式预算PASS，所需383488663552B、当时可用708412715008B，原生/capture/父宿主均0；该可用量不是未来正式起跑保证，最终独立复核已通过，已归档到 `bafecc658c45fbd940fb1655e7057d2b1d2c0593`。 本稿不清理权重/cache。

## 最终只读/CPU起跑前检查（检查器不launch，root随后起跑）

八阶段normal perf已正式、原生、父/宿主及独立验收通过，源HEAD固定70a；完成快照原SHA为 `2795d949fe06b50a6a81c316fb236f3f13700f0bf36acbed375bb16939697d22`。先完成V11.17归档，再生成共同V11.18Beta；以下CLI的两提交SHA与最终索引SHA只能由真实记录填写。检查结果不是启动命令，**检查器到此停止；全部检查真实通过后root直接独立调用下文launch**。

只读检查器位于 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/prod-normal-ready-candidate-20260927/import-ready/check_prod_ready.py`，SHA `af8f32b10f51ff9062db919443e5892f74a46f112bc268e027251f47c83ee12b`。它沿用冻结5aba的113来源、两uv环境208依赖、保护runtime/资产和设备检查；核八阶段不可变快照及独立报告/原生f8回执，原70a记录不被新HEAD重判或改写。CPU子进程只调用现有 `make_train_args/parse_in_process`，与已通过的4cf预览完整配置一致后，再调用 `validate_disk_budget("prod",...)`重新核原B预算和**当时**可用空间；不调用会创建正式launch记录的 `check_launch`。

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
PRE="$MAIN/v1-store/bench/orig80k-build-preflight-0925"
READY_HEAD='<共同V11.18Beta实际完整40位SHA>'
ARCHIVE_HEAD=bafecc658c45fbd940fb1655e7057d2b1d2c0593
PERF_INDEX_SHA=2f5a044e03a60c58541b45cff3790b34d7298c5f3b3e232d08ebc41448569034
"$MAIN/.venv/bin/python" -B \
  "$PRE/prod-normal-ready-candidate-20260927/import-ready/check_prod_ready.py" \
  --archive-head "$ARCHIVE_HEAD" --head "$READY_HEAD" \
  --perf-controller \
    "$MAIN/docs/training-doc/orig80k-perf300-0927/records/controller.completed.snapshot.json" \
    "$PRE/perf300-completed-controller-snapshot-20260927.json" \
    2795d949fe06b50a6a81c316fb236f3f13700f0bf36acbed375bb16939697d22 \
  --perf-index \
    "$MAIN/docs/training-doc/orig80k-perf300-0927/records/final-perf-archive-index.json" \
    "$PRE/perf300-archive-staging-20260927/final-perf-archive-index.json" \
    "$PERF_INDEX_SHA"
```

stdout/stderr须独占保存，根Python和宿主tools实际返回分别记录。仅当status=PASS、stage=STOP_BEFORE_PROD_LAUNCH、实际返回0且所有引用固定时，该检查完成；没有隐含prod启动动作。当前只验证过该工具的AST/help和纯小边界，**未在未来Beta执行完整main/CPU/预算复核**。

正式session准备标签固定为 `20260927-ready`，不是起跑UTC：`orig80k-prod-full-20260927-ready` 与 `orig80k-prod-count-20260927-ready`；命令根为 `v1-store/bench/orig80k-prod-commands-20260927-ready`。检查先确认其全新；真正runtime文件生成、整体SHA固定与launch须有后续明确起跑安排。本准备代理不创建它们或正式run/records/driver；root在全部检查通过后的起跑步骤创建。

## 两份正式runtime命令正文（检查通过后由root执行）

下列代码块分别成为两个新命令文件的正文。正式生成时替换待填值为同一Beta、同一预算路径与SHA；路径/内容经父适配固定完整SHA，runtime还须核对guard传入的自身SHA。不提前mkdir run/records/cache，不用resume/overwrite，保留冲突现场。

`v2-orig-16task-pub1600ep-modul-b64-80k`：

```bash
set -euo pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
RUN=v2-orig-16task-pub1600ep-modul-b64-80k
TRAIN_HEAD='<待共同V11.18Beta实际完整40位SHA>'
ORIG80K_DISK_BUDGET_JSON=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-perf300-results-20260926T232339Z/prod-disk-budget.json
ORIG80K_DISK_BUDGET_SHA256=3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20
COMMAND_SHA=${1-}
[[ "$COMMAND_SHA" =~ ^[0-9a-f]{64}$ ]] || exit 2
actual=$(sha256sum -- "$0")
test "${actual%% *}" = "$COMMAND_SHA"
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ && "$ORIG80K_DISK_BUDGET_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
cd "$MAIN"
test "$(git rev-parse HEAD)" = "$TRAIN_HEAD"
test -z "$(git status --porcelain)"
exec env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  -u ORIG80K_TIMING_EQ_PROFILE -u ORIG80K_SMOKE_EQ_MODE -u XLA_FLAGS -u TRAIN_TIMING_STEPS \
  UV_PROJECT_ENVIRONMENT="$MAIN/.venv" UV_CACHE_DIR="$STORE/cache/uv" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TRAIN_HEAD="$TRAIN_HEAD" \
  HISTORY_CONFIG_SHA256=823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a \
  NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5 \
  ORIG80K_DISK_BUDGET_JSON="$ORIG80K_DISK_BUDGET_JSON" \
  ORIG80K_DISK_BUDGET_SHA256="$ORIG80K_DISK_BUDGET_SHA256" \
  bash "$MAIN/scripts/training/prod/run_orig80k.sh" prod "$RUN" 0,1,2,3 \
  "$STORE/datasets/16task-pub-1600ep" "$STORE/train-assets/mme_vla_suite"
```

`v2-orig-counting-pub400ep-modul-b64-80k`：

```bash
set -euo pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
RUN=v2-orig-counting-pub400ep-modul-b64-80k
TRAIN_HEAD='<待共同V11.18Beta实际完整40位SHA>'
ORIG80K_DISK_BUDGET_JSON=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-perf300-results-20260926T232339Z/prod-disk-budget.json
ORIG80K_DISK_BUDGET_SHA256=3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20
COMMAND_SHA=${1-}
[[ "$COMMAND_SHA" =~ ^[0-9a-f]{64}$ ]] || exit 2
actual=$(sha256sum -- "$0")
test "${actual%% *}" = "$COMMAND_SHA"
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ && "$ORIG80K_DISK_BUDGET_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
cd "$MAIN"
test "$(git rev-parse HEAD)" = "$TRAIN_HEAD"
test -z "$(git status --porcelain)"
exec env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  -u ORIG80K_TIMING_EQ_PROFILE -u ORIG80K_SMOKE_EQ_MODE -u XLA_FLAGS -u TRAIN_TIMING_STEPS \
  UV_PROJECT_ENVIRONMENT="$MAIN/.venv" UV_CACHE_DIR="$STORE/cache/uv" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TRAIN_HEAD="$TRAIN_HEAD" \
  HISTORY_CONFIG_SHA256=823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a \
  NORM_STATS_SHA256=a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9 \
  ORIG80K_DISK_BUDGET_JSON="$ORIG80K_DISK_BUDGET_JSON" \
  ORIG80K_DISK_BUDGET_SHA256="$ORIG80K_DISK_BUDGET_SHA256" \
  bash "$MAIN/scripts/training/prod/run_orig80k.sh" prod "$RUN" 4,5,6,7 \
  "$STORE/datasets/4task-counting-pub-400ep" "$STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep"
```

runner创建本run的 `v1-store/bench/orig80k/<run>`、`train-runs/mme_vla_suite/<run>`和 `logs/<run>.driver.log`，按run名隔离JAX/CUDA/W&B/XDG目录，并核实际四卡、完整配置、资产、预算与W&B。唯一driver `EXIT_CODE`归runner所有，外层不得追加另一个终态。full norm始终f332原版，不能改用full自算c5；count使用已授权重算并验收的a770 norm。

## f8与独立退出：复用既有父适配，不造新机制

冻结guard为 `v1-store/bench/orig80k-build-preflight-0925/tmux-exit-guard-candidate-20260926/tmux_exit_guard.py`，SHA256 `f8dada4025dc49f260697e2fb768e922cbea7b8f98afe5cc8b98555648bff52b`。正式Git源、工作树和该runtime副本须逐字相同。

可复用[normal perf候选父适配](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-schema2-candidate-20260927/launch.md)内的 `guard_perf_action()`，该候选launch SHA为 `7fa32825c58851874430d4fc236018beb4de96df4bd80ad70133dff10ad8f2f3`。它的launch/capture/verify逻辑对session不限定perf前缀，可直接以两prod session和新的COMMAND_ROOT调用；不要复用有perf前缀检查的 `launch_perf_stage()`来绕过名字约束。上面prod runtime用exec转交原runner，由f8保留其真实最外退出，runner自己负责pipefail/tee/driver终态。

下面已纳入与70a perf逐字相同的完整父适配正文；正式prod Beta须同时锁定它与本两runtime正文。只读检查与起跑接口分开：本检查器只完成前者；root在共同Beta与全部检查通过后依据用户最新授权调用launch。实际TRAIN_HEAD、预算期望SHA和命令整体SHA须在真实Beta/预算产生后填入。

```bash
guard_perf_action() {
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

spec = importlib.util.spec_from_file_location("frozen_perf_exit_guard", tool_path)
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
```

以下是未来调用设置与两个既定session，在已获用户授权且最终检查真实通过后由root执行：

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
TRAIN_HEAD='<共同V11.18Beta实际完整40位SHA>'
TAG=20260927-ready
COMMAND_ROOT="$STORE/bench/orig80k-prod-commands-$TAG"
GUARD_TOOL="$STORE/bench/orig80k-build-preflight-0925/tmux-exit-guard-candidate-20260926/tmux_exit_guard.py"
GUARD_SHA256=f8dada4025dc49f260697e2fb768e922cbea7b8f98afe5cc8b98555648bff52b
FULL_SESSION="orig80k-prod-full-$TAG"
COUNT_SESSION="orig80k-prod-count-$TAG"
FULL_COMMAND_SHA='<full runtime完整已固定SHA>'
COUNT_COMMAND_SHA='<count runtime完整已固定SHA>'
# 父适配读取COMMAND_ROOT/<session>.command；两文件必须独占新建且正文如上。
guard_perf_action launch "$FULL_SESSION" "$FULL_COMMAND_SHA" || exit 2
guard_perf_action launch "$COUNT_SESSION" "$COUNT_COMMAND_SHA" || exit 2
```

两次launch只等本会话身份持久化及释放，因此可4+4同启。f8先为自有窗口设置remain-on-exit、落identity再释放；RELEASED不代表训练结束。实际结束事件后单独调用同函数 `capture`和只读 `verify`，不在launch后连续capture；PENDING不放行也不预占最终sidecar。

完整成功证据必须包含原生pane_dead_status0/无信号、f8真实capture返回0、父适配launch/capture argv和stdout/stderr/identity/receipt/command/SHA绑定；根调度Python实际返回及宿主tools实际返回还须分别留证，不能从内层JSON反推最外wait。日志0和会话消失都不单独放行。失败保留全部产物，停止依赖；只在验收后按本轮精确清单处理窗口，不自动清理或全局kill。

## 正式完成判据（本轮尚无这些结果）

每侧checkpoint恰为 `10000,20000,30000,40000,50000,60000,70000,79999`，常规metrics为 `range(0,80000,100)`共800条，尾窗79901…79999共99步五标量全部有限。末态state80000/loop79999及run/HEAD/UUID/异步wait必须一致；真实恢复79999/params并逐叶核EMA。此处不填EMA叶数、权重大小或完成PASS。

未来真实结束后，在独立CPU任务中执行既有 `check_orig80k_completion.py --mode prod --records <本runREC> --run-root <本run根> --log <本rundriver> --run <固定名称> --head <实际TRAIN_HEAD> --out <本runREC/completion.json>`。out独占新建，completion及tee和该CPU任务独立退出分别取证；不把本准备工作说成完成器已运行。

## 本轮交付与独立起跑步骤

只提供两run README、本launch与既定[检查清单](checklist.md)，做语法、参数、路径和旧超参对照。normal perf七实测阶段及B档数值已有证据；第8预算及独立复核已通过；V11.17已归档到bafecc658c45fbd940fb1655e7057d2b1d2c0593；共同V11.18Beta/clean与现场复核在本启动档提交后执行。此前「继续工作 一路做到起泡前 有问题问用户」的stop-before是历史范围；用户最新「确认无误后可以直接开始两个训练」已授权root在全部检查通过后执行本两条launch。

## 已完成CPU参数预览与本轮包装例外

70a下两个prod配置已实际CPU解析PASS，报告 [prod-config-preview-20260927.json](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/prod-config-preview-20260927.json) SHA `4cf995473747d6ad9e462e4dda710f433fe6409b01e71566c54b8b2c3643f7a6`。实际80000步、normal、x64 False，参数保持原版，主机返回0；六正式输出未创建。后续仅docs/本根计划变化时保留该预览原HEAD与字节，通过源码/依赖和真实资产引用桥接复核；不是直接沿用预览当作prod已放行。

用户已批准B档16/32/64GiB，并仅对本轮normal perf八stage wrapper保留原字节、豁免COMMAND末尾1B空格检查；原日志及SHA不改，数值、参数、来源和独立退出判据全部保持。该范围不自动扩大为未来prod的任意日志宽容。

准备阶段的[候选验证](records/preparation/candidate-validation.json)、[只读预检小例](records/preparation/preflight-small-tests.json)和[独立静态复核](records/preparation/independent-static-review.json)原字节归档。它们记录各自候选时点的源SHA；本正式稿只回填已经形成的bafecc6归档事实和展示说明，不改两runtime及父适配正文。真正的共同Beta预检和启动结果在该Beta生成后另行留存，不能用这些准备记录替代。
