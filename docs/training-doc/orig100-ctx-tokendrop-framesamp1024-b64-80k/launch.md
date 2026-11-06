# Context 1024 双四卡 80k：起跑档案

本档案同时覆盖两条并行 run，二者只差记忆取法。

## 一、目的

按官方 `mme_vla_suite`（batch 64、80k）训 context 模式的 tokendrop 与 framesample，只把记忆预算 `budget` 从 512 改为 1024。

用户决策原话：

1. 「给出方案 训context模式下的tokendrop和framesample 但是1024 尽可能和原版设计对齐 就是robomme policy learning原版 batch64 80k的设计」
2. 「我们现在是八卡，所以说就会分两个四卡来做。」
3. 「同意开始，然后你开始训练了之后，你要看一下 GPU 占有率是不是有问题」
4. 「先暂缓，你目前只做 CPU 的工作，GPU 的都不做」
5. 「可以继续做了GPU 也继续」

计划：仓库根目录 `1106-ctx-tokendrop-framesample-1024-b64-80k-plan.html`。

## 二、版本与代码状态

- 环境 B：AWS 8×A100-80GB，存储为本地 NVMe RAID（`/dev/md0`）。
- 分支 `v8-ctx-td-fs-1024-b64-80k`；Beta commit 即本文件所在提交，运行前复制到只读副本 `/scratch/hongze/worktrees/c1k-<Beta 短 sha>`。
- 解释器为主检出 `.venv`，`PYTHONPATH` 指向只读副本，日志首行 `MME_VLA_SUITE_IMPORT=` 可核对。

## 三、启动与配置还原

| run | GPU | history YAML | `--dataset-path` |
|---|---|---|---|
| `orig100-ctx-tokendrop1024-b64-80k` | 0–3 | `perceptual-tokendrop-context-1024.yaml` | `v1-store/datasets/16task-orig100-1600ep-9e109419/framesamp-8x8` |
| `orig100-ctx-framesamp1024-b64-80k` | 4–7 | `perceptual-framesamp-context-1024.yaml` | `v1-store/datasets/16task-orig100-1600ep-9e109419/framesamp-4x4` |

```bash
CODE_ROOT=<只读副本> bash <只读副本>/scripts/training/run_ctx1024_b64_80k.sh <td|fs> prod
```

超参全部取官方条目，命令行不覆盖：batch 64、`fsdp_devices` 4、`num_workers` 4、80000 步、warmup 10000、peak = decay = 5e-5、`decay_steps` 100000、AdamW clip 1.0、EMA 0.999、每 10000 步保存、seed 42、`pi05_base` 初始化。

## 四、数据

16 任务 × 100 集（`16task-orig100-1600ep-9e109419`），476857 个执行样本，768897 行。两条 run 使用同一 `source` 与同一 `norm_stats`（md5 `cf425386…`）。8×8 库由本轮打包，`status=verified`。不使用 motion/memory encoder，P-LINEAGE 不适用。

## 五、起跑前实测

| 判定 | 结果 |
|---|---|
| A1 | `LOADER_EQ=PASS td=1000 fs=1000 mismatch=0`；`REG512=PASS n=200 mismatch=0` |
| A2 | `ONLINE_TD_REPLAY=PASS episodes=16 mismatch=0` |
| 合并审查 | 三个分支 `PRE_MERGE_REVIEW=PASS findings=0`，合并后短测通过 |
| A3 短测 | 双四卡各 20 步，`EXIT_CODE=0` |
| A3 测速 | 双四卡各 300 步，`EXIT_CODE=0` |

测速（双四卡同时运行）：tokendrop 2.643 s/步，framesample 2.706 s/步；外推 80k 约 58.8 小时、60.1 小时。

GPU 采样在第 200 步后启动，单个 `nvidia-smi` 进程，间隔 1 秒，共 240 次：8 张卡 util 均值 99.8–99.9%，0% 占比 0，低于 50% 占比 0，平均功耗 358–367 W。步速在开启采样前后无差异。显存读数 79.4 GB 是 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` 的预分配，不代表真实用量。

## 六、产物与上传

- checkpoint：`v1-store/train-runs/mme_vla_suite/<run>/`，共 8 份（10000…70000、79999）。
- 日志：`v1-store/logs/<run>.log`；上传日志 `v1-store/logs/<run>-inc.log`。
- 公开 bucket：`HongzeFu/robomme-vla-orig100-ctx-tokendrop1024-b64-80k-v1`、`HongzeFu/robomme-vla-orig100-ctx-framesamp1024-b64-80k-v1`。

## 七、tmux 会话清单

`c1k-td-train`、`c1k-fs-train`、`c1k-td-inc`、`c1k-fs-inc`。本轮已结束的：`c1k-pack8x8`、`c1k-verify8x8`、`c1k-verify8x8b`、`c1k-td-smoke`、`c1k-fs-smoke`、`c1k-td-perf`、`c1k-fs-perf`。

## 八、盯盘项

`EXIT_CODE=`、`Traceback`、`nan`、`out of memory`、`RESOURCE_EXHAUSTED`、`Killed`；上传日志另盯 `INC_UPLOADED`、`UPLOAD_FAIL`、`HF_VERIFY`。

## 九、已知未验证

- 上传脚本的真实 `hf` CLI 路径只用假 CLI 测过，第一份 checkpoint（第 10000 步）才实际验证。
- 在线 tokendrop 没有用真实模型跑过，要到 eval 才验证。
