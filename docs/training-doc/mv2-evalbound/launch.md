# 1600 集 motion 配置：评估 demo 长度预检

本任务只读取 test 配置并实际 reset 环境，检验 budget=160 能否覆盖未来评估；不加载策略，不计算成功率。用户原话「采用现有仿真环境和单卡渲染，保留 200 次真实 reset」。

## 版本与配置

从本页及验证工具提交后的 clean HEAD 启动，完整提交通过 CHECK_HEAD 参数固定并写入 launch.actual.json；结束再次核对 HEAD 和 porcelain。独立 RoboMME 依赖提交为 856bc3a189d4172f3f47dbee4424d585f8d78db3，每个任务检查其工作区干净，记录实际依赖版本及对应 test metadata SHA。

由 uv 调用现有 /scratch/hongze/micromamba/envs/robomme/bin/python，四任务各启一个独立进程，避免反复重建 Vulkan 上下文积累 TLS 问题。仅占 GPU 7，GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192；uv/HF/XDG/CUDA/ManiSkill 缓存均落 v1-store/cache，不改 HOME，不下载或修改依赖。存储为 AWS 本地 NVMe RAID /dev/md0 XFS，GPU 为 A100-SXM4-80GB。

## 验证与产物

每任务遍历全部 test episode 0–49，执行 EnvRunner.make_env 与 get_init_obs，记录 es=len(images)-1、渲染设备和耗时。四任务 BinFill、RouteStick、VideoRepick、VideoUnmaskSwap 合计 200 次 reset。以 max_steps=1300 独立计算 demo 窗加执行窗，输出 EVAL_ES_BOUND；任何缺集、重复、超预算或任务退出非零均阻断。

产物为 v1-store/reports/motion/mv2-evalbound 下的任务 JSON、日志、aggregate.json 与 launch.actual.json；驱动日志为 v1-store/logs/mv2-evalbound.driver.log。tmux 全名 mv2-evalbound。本任务完成、单卡进程退出后才启动独占 8 卡的 Wan 建库。正式八卡训练尚未启动。

预检脚本已在修补前 clean df6fdcc4f9424f00901a107a66d98e19e679a7ba 完成 BinFill 第 0 集真实 reset：es=0、k=80、实际渲染设备 cuda:0（可见物理 GPU 7），单次 make_env/reset 16.25 秒、EXIT_CODE=0；这次短测不替代完整 200 集判定。

## 启动命令

下面完整外壳保存为 v1-store/logs/mv2-evalbound-runner.sh；用已记录的完整提交运行：
`tmux new-session -d -s mv2-evalbound 'bash v1-store/logs/mv2-evalbound-runner.sh <CHECK_HEAD>'`。

```bash
#!/usr/bin/env bash
# 真实 test 集 reset 预检；每任务独立进程，仅 GPU 7 渲染，不加载策略。
set -uo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA || exit 1
CHECK_HEAD="${1:?必须给出完整提交}"
LOG="$PWD/v1-store/logs/mv2-evalbound.driver.log"
test ! -e "$LOG" || exit 2
body() (
  set -euo pipefail
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
  source scripts/dataset/paths.sh
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
  export CUDA_VISIBLE_DEVICES=7 CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export MS_ASSET_DIR="$V1_STORE/cache/maniskill/data"
  export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192
  export XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  ROBOMME_PY=/scratch/hongze/micromamba/envs/robomme/bin/python
  OUT="$V1_STORE/reports/motion/mv2-evalbound"
  test ! -e "$OUT"
  test "$(nvidia-smi --id=7 --query-gpu=memory.used --format=csv,noheader,nounits)" = 0
  mkdir -p "$OUT"
  printf 'CHECK_HEAD=%s\nSTART_UTC=%s\n' "$CHECK_HEAD" "$(date -u +%FT%TZ)"
  uv run --no-sync python - "$OUT" "$CHECK_HEAD" "$BASHPID" <<'PY'
import datetime,hashlib,json,pathlib,subprocess,sys
out,head,pid=sys.argv[1:]
record=dict(head=head,status=subprocess.check_output(['git','status','--porcelain'],text=True),
            pid=int(pid),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            uv_lock_sha256=hashlib.sha256(pathlib.Path('uv.lock').read_bytes()).hexdigest(),
            command='bash v1-store/logs/mv2-evalbound-runner.sh '+head,
            gpu=7,storage='AWS 本地 NVMe RAID /dev/md0 XFS',tmux='mv2-evalbound')
with (pathlib.Path(out)/'launch.actual.json').open('x') as f: json.dump(record,f,ensure_ascii=False,indent=2)
PY
  reports=()
  for task in BinFill RouteStick VideoRepick VideoUnmaskSwap; do
    set +e
    uv run --no-project --python "$ROBOMME_PY" python scripts/training/tests/eval_es_bound.py reset --task "$task" --out "$OUT/$task.json" --budget 160 --max-steps 1300 2>&1 | tee "$OUT/$task.log"
    code=${PIPESTATUS[0]}
    set -e
    printf 'TASK_EXIT_CODE=%s task=%s\n' "$code" "$task" | tee -a "$OUT/$task.log"
    test "$code" = 0
    reports+=("$OUT/$task.json")
  done
  uv run --no-sync python scripts/training/tests/eval_es_bound.py aggregate --reports "${reports[@]}" --out "$OUT/aggregate.json" --budget 160 --max-steps 1300
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
)
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
```
