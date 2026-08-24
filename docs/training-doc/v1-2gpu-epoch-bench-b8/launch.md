# v1-2gpu-epoch-bench-b8 起跑留档

（目录名按 AGENTS.md 第 12 条取最终成功 run_name `v1-2gpu-epoch-bench-b8`；探档期失败档 b64/b32/b16 的轮次记录一并留在本文件。）

按 AGENTS.md 第 12 条：正式训练从 clean HEAD 启动，起跑前记录可复现的 commit、命令、配置、数据来源与输出路径。本基准是「OOM 自动降档」系列，run_name 按档位命名为 `v1-2gpu-epoch-bench-b<batch>`。

## 轮次记录

- **第 1 轮**（commit `fb4f03a`，梯子 64→32→16）：2026-08-24 约 13:35 起跑，**三档全部 OOM**——失败张量 17.62 GiB（b64）/ 12.61 GiB（b32）/ 10.38 GiB（b16）。定性：2 卡 FSDP 下每卡驻留约 28 GB 参数+优化器+EMA 状态，43.7 GB 显存池的剩余空间装不下固定底座约 8 GiB 的激活张量（官方 4 卡每卡状态减半，故能跑 batch 64）。另暴露驱动脚本 `grep -q` + pipefail 的 SIGPIPE 竞态：b16 的 OOM 被误判为「非 OOM 失败」。
- **第 2 轮**（commit `891d6e3`，修复误判 bug、梯子延伸至 8→4→2，`BATCHES="8 4 2"`）：**b8 不 OOM，300 步训练本体全部完成**（loss 0.58→0.04 全有限、12 次参数校验和齐全、单次约 47 s），但 `metrics.jsonl` 一行未写、结果判定 fail-loud 退出——`train.main` 里的 `wandb.init(mode="disabled")` 会把 wandb 模块级 `log` 重新赋值成 run stub，盖掉了 bench 入口装在 `wandb.log` 上的记录器。
- **第 3 轮**（修复方式：`train` 模块的全局名 `wandb` 替换为代理对象，`log` 先记录再转发、其余属性透传，`wandb.init` 改真模块属性影响不到代理；commit 见 git log，`BATCHES="8 4 2"`）：结果见同目录 `result.md`。

## 可复现信息

- **环境**：本机，2× RTX 6000 Ada 46 GB，驱动 570.211.01 / CUDA 12.8
- **启动命令**（仓库根目录；第 2 轮在 tmux 命令前加 `BATCHES="8 4 2"`）：
  ```bash
  tmux new-session -d -s epoch-bench \
    "set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/smoke-local/run_2gpu_epoch_bench.sh 2>&1 \
     | tee v1-store/logs/2gpu-epoch-bench-driver.log; \
     echo \"EXIT_CODE=\$?\" >> v1-store/logs/2gpu-epoch-bench-driver.log"
  ```
- **入口链**：`scripts/smoke-local/run_2gpu_epoch_bench.sh` → `scripts/smoke-local/bench_train_steps.py` → 原版 `train.main(config)`（训练循环零改动，`wandb.log` 与 `_checkpoints.save_state` 被替换为记录器，详见 `scripts/smoke-local/README.md`）

## 配置（与官方 finetune_mme_vla_suite.sh 的差异逐项见 scripts/smoke-local/README.md）

- 具名配置 `mme_vla_suite`；batch 按降档梯子取当轮首档（OOM 自动降档，run_name 随档位变化）
- `--num-workers 4`、`--num-train-steps 300`、`--log-interval 1`、`--save-interval 25`（参数校验和间隔，**不落 checkpoint**）
- `--seed 42`、`--fsdp-devices 2`、`CUDA_VISIBLE_DEVICES=0,1`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- `--model.use-history --model.history-config perceptual-framesamp-context.yaml`
- `--no-wandb-enabled`（wandb 关闭）
- 权重初始化：`--weight-loader.params-path v1-store/models/openpi-assets/checkpoints/pi05_base/params`

## 数据来源

- 数据集：`v1-store/datasets/4task-gl`（GL 构建的全量库，`execution_samples=395289`，构建溯源见其 `meta/provenance.json`）
- norm stats：`v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`

## 输出路径

- 一致性记录（仅成功档）：git 归档于本目录 `records/`（`metrics.jsonl`、`param_checksums.jsonl`、`env.json`；格式见 `scripts/smoke-local/README.md`）；`v1-store/bench/2gpu-epoch-bench/v1-2gpu-epoch-bench-b8/` 为内容一致的工作副本；OOM 档的记录目录随即删除
- 日志（保留）：`v1-store/logs/2gpu-epoch-bench-driver.log`、`v1-store/logs/v1-2gpu-epoch-bench-b<batch>.log`
- run 目录 `v1-store/train-runs/mme_vla_suite/v1-2gpu-epoch-bench-b<batch>` 与 `~/.cache/jax_v1-2gpu-epoch-bench-b<batch>`：跑完即删（无 checkpoint 产出）

## 结果

见同目录 `result.md`（跑完后归档：稳态 s/step、1 epoch 外推、loss 统计、校验和摘要；jsonl 记录本体留在 v1-store，此处只归档 Git 无法还原的指标摘要，不归档权重）。
