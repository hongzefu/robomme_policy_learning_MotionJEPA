# Context framesample 1024 八卡 b64 80k：训练档案

## 一、目的与用户指令

按官方 `mme_vla_suite`（batch 64、80k）训练 context 模式的 framesample，记忆预算 1024。这一条就是 11-06 叫停的 framesample 臂，当时在 4 卡上停在 14.9k，这次在 8 卡上从头训满。

用户原话：

1. 「给出方案 跑完eval完立刻开始 context1024的训练 之前没跑的那个 写html方案」
2. 选择题答复：「8卡跑framesample1024」
3. 「方案今晚就要定好 然后就在这个session运行 但是要监控直到ok了再开始训练 不能再来问我的那种」

计划：仓库根目录 `1109-ctx-framesamp1024-8gpu-b64-80k-plan.html`（定稿 aa810db），自动处理口径 D1–D6 见计划 §8。

## 二、版本与环境

- 分支 `v12-ctx-fs1024-8gpu-b64-80k`；Beta `a4293f4`（commitV11.90Beta）；只读副本 `/scratch/hongze/worktrees/fs8-a4293f4`。
- 日志首行 `MME_VLA_SUITE_IMPORT=/scratch/hongze/worktrees/fs8-a4293f4/src/mme_vla_suite/__init__.py`。
- 解释器：专用 venv `v1-store/envs/fs8-train`（`uv sync --frozen --no-install-project`），关键包与主检出 `.venv` 同版本（jax/jaxlib 0.5.3、flax 0.10.2、orbax 0.11.13、numpy 1.26.4、torch 2.7.1、transformers 4.53.2）。
- 硬件：环境 B，8×A100-80GB。`TRAIN_MESH shape=(2, 4) devices=8 fsdp_devices=4 batch_size=64 num_workers=4`。

## 三、配置

TrainConfig（日志原样）：`mme_vla_suite`，`batch_size=64`、`num_train_steps=80000`、`CosineDecaySchedule(warmup_steps=10000, peak_lr=5e-05, decay_steps=100000, decay_lr=5e-05)`、`AdamW(clip_gradient_norm=1.0)`、`ema_decay=0.999`、`seed=42`、`save_interval=10000`、`fsdp_devices=4`、`pi05_base` 初始化，history YAML `perceptual-framesamp-context-1024.yaml`，数据 `v1-store/datasets/16task-orig100-1600ep-9e109419/framesamp-4x4`。命令行没有覆盖任何超参。

与原版和上次 4 卡 fs 臂相比，唯一的差别是卡数：4 卡（mesh 1×4，每卡 16）变为 8 卡（mesh 2×4，每卡 8）。单进程下 `local_batch = 64 // process_count`，每步 batch 的组成只由 seed 决定。

## 四、时间线（美东）

| 时刻 | 事件 |
|---|---|
| 11-09 00:07 | `fs8-chain` 起跑，开始等待 |
| 11-09 06:08 | w768 评估 `EVAL692=PASS episodes=692 success=298 rate=0.4306` |
| 11-09 06:09 | `FS8_TRIGGER=PASS eval=PASS gpus_idle=8 free_gb=382` |
| 11-09 06:13 | `FS8_SMOKE=PASS steps=20 mesh=2x4 s_per_step=1.6s/it`，正式起跑 |
| 11-09 10:20 | `FS8_FIRST_CKPT step=10000`，`fs8-chain` `EXIT_CODE=0` |
| 11-10 14:58 | 训练结束，`EXIT_CODE=0`，墙钟 32:43:02 |

稳态步速 1.4–1.5 s/步，8 卡 GPU 利用率 99%。上次 4 卡为 2.706 s/步。

## 五、训练曲线（日志抽样）

| step | loss | grad_norm |
|---|---|---|
| 1000 | 0.0224 | 0.4207 |
| 10000 | 0.0069 | 0.0500 |
| 20000 | 0.0042 | 0.0345 |
| 40000 | 0.0025 | 0.0312 |
| 60000 | 0.0018 | 0.0258 |
| 79900 | 0.0016 | 0.0252 |

全程没有 nan、OOM、Traceback。

## 六、验收

| 判定 | 结果 |
|---|---|
| A4 训练完成 | `CTX1024_TRAIN=PASS run=orig100-ctx-framesamp1024-b64-80k-8gpu step=79999`：`EXIT_CODE=0` 恰 1 行，checkpoint 10000…70000、79999 共 8 份（87 GB） |
| 79999 EMA CPU 恢复 | `CKPT_CPU_RESTORE=PASS leaves=55 params=3359793680 nonfinite_leaves=0` |
| 上传 | `INC_UPLOADED` 8 份；全量匿名回读 `HF_VERIFY=PASS bucket=HongzeFu/robomme-vla-orig100-ctx-framesamp1024-b64-80k-8gpu-v1 steps=8 mismatches=0`，上传会话 `EXIT_CODE=0` |

## 七、产物

- checkpoint：`v1-store/train-runs/mme_vla_suite/orig100-ctx-framesamp1024-b64-80k-8gpu/`
- 公开 bucket：`HongzeFu/robomme-vla-orig100-ctx-framesamp1024-b64-80k-8gpu-v1`
- 日志：`v1-store/logs/orig100-ctx-framesamp1024-b64-80k-8gpu.log`、`…-8gpu-inc.log`、`…-8gpu-smoke.log`、`fs8-chain.log`

## 八、事故与备注

- 11-08 美东 23:56，写入子代理 S1 清理测试残留时用 `ps | grep "sleep 300"` 找到 PID 再 kill，误杀了 w768 上传会话 `w768-ckptup` 的 sleep 子进程，使其以 143 退出。w768 会话已续传恢复，w768 训练不受影响。此后规定子代理只能对自己用 `$!` 记录的 PID 发信号。
- 本任务不含评估（D3）；成绩要另出评估方案。对照参考：timepool1024 context 本机 35.46±0.79，concat 1×1024 GL 37.33±0.59。
- 旧 run 目录 `orig100-ctx-framesamp1024-b64-80k` 与其 bucket 中的第 10000 步副本未动。
