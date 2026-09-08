# vulkan-fix-30ep —— 结果

> 起跑 commit `6a40b20`（修法在 commitV7.2 `8c20c78`）。2026-09-08 01:01:22 → 01:17:30（16 min），环境 B，GPU 0（policy + 仿真）+ GPU 1（sidecar），tmux `tic-vulkanfix`（结束自行退出）。
> 原始日志 `records/eval-probe.txt`（驱动）、`records/eval.txt`（`eval.py` 全量输出）、`records/server.txt`、`records/progress.json`。

## 判定行

```
VULKAN_FIX_30EP=PASS episodes_setup=30 progress_entries=30 incompatible_driver=0 error_lines=0 eval_rc=0 tunable=glibc.rtld.optional_static_tls=8192
EXIT_CODE=0
```

## 结论

加了 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192` 之后，同一个 `eval.py` 进程用生产 policy server 连评 ButtonUnmask test split 30 集：30 次 `make_env` 全部建成（`setup finished` 30 行），第 28 次不再抛 `ErrorIncompatibleDriver`，`progress.json` 30 条、零 `"error"`，`eval.py` 退出码 0。对照：未加环境变量时同一路径在 `eval-official-framesamp-context`（w0/w1）与 `tic-vulkan-makeenv`（基线）都精确崩在第 28 次。

**「每个 `eval.py` 进程 ≤ 27 集」红线自 commitV7.2 起解除**：四个 `eval.py` 启动点（`legacy-eval/eval_shard.{local,remote}.sh`、`tests/run_t3_eval_obs.sh`、`eval.sh`）默认带该环境变量，按每轮 64 B 外推可支撑约 147 集，覆盖单进程 50 集。成功率不作结论（单 seed、单任务、30 集成功 5）。

## 异常处置

无。单次运行、未续评；server 由脚本 kill 自己起的 PID，tmux 会话随脚本退出。
