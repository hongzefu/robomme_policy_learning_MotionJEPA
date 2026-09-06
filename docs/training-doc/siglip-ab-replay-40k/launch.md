# siglip-ab-replay-40k — 起跑记录

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。诊断 run，按 AGENTS 17 视作完整运行留档。

起跑时间：2026-09-06 00:5x（训练 `awsprod40k-b128-motion` 于 00:46 落最终 checkpoint `39999`、`EXIT_CODE=0` 后立即起）
起跑 commit：权重核对 `c01e45d`、五臂重放 `034c18f`（三次脚本接线修正 `8f0cdd7` → `f1817f7` → `034c18f`，均不改被测链路；worktree 分支 `v2-motionmem-sgab`，基于 `v2-motionmem` HEAD `b12f1a0`；代码 clean，留档文件未提交）
进程：本会话 `run_in_background` 直接起（tmux 启动被 worktree 隔离检查拦下），日志各自写 `EXIT_CODE=`。本轮未起任何 tmux 会话。

## 这条 run 回答什么

对抗审计（`v1-store/reports/audit/train-infer-consistency-b12f1a0.md`，锚 `b12f1a05…`）确认的 D-A1：
训练建库时记忆帧 SigLIP 特征由离线 f32 权重 `v1-store/models/pi05_vision_encoder/siglip_params.pkl`
（`dataset_builder/siglip_tokenizer.py::SigLipTokenizer`）算出并打包进 framesamp 库；在线推理由 checkpoint 内 bf16 的
`PaliGemma.img`（`history_pi0.py::HistoryPi0.vision_encode`）现场算。用户要求正式评估前先用 1 个 episode 把两者差异量化。

两个脚本、三个问题：

| 问题 | 脚本 | 判定行 |
|---|---|---|
| 两份权重同源吗 | `scripts/training/g0/compare_siglip_weights.py`（纯 CPU） | `PARAM_SAME_ENC`（pkl vs `pi05_base` img 逐叶 max_abs == 0） |
| 线上 img 就是训练看到的那份吗 | 同上 | `CKPT_IMG_FROZEN`（`base.astype(bf16)` vs `39999` img 逐叶 sha） |
| bf16 舍入让记忆 token / 动作偏了多少 | `scripts/training/g0/compare_siglip_replay.py`（GPU 0） | `MEM_A1_VS_STORE` / `MEM_S_VS_TRAINSET` / `MEM_C_VS_B` / `MEM_S_VS_B` / `MEM_S_VS_A` / `MEM_A_VS_B` / `ACT_S_VS_B` / `ACT_DETERMINISM` |

五臂共用 `FrameSampMemory.add_buffer` 的同一套预处理（`/255*2-1 → resize_with_pad(224) → enc → pool_tokens_to_size(16)`，
与离线建库 `mem_buffer.py::MemoryBuffer.add_buffer` 逐字相同）：
S = 训练库 `image_emb_4x4` 行（训练真值）；A1 = 离线 `jax.jit(SigLipTokenizer().__call__)`（f32 pkl）逐帧 batch=1（复现建库喂法，只走帧路）；
A = 同一 f32 编码器按 eval 节奏成批；B = 生产 `policy._vision_encode`（`create_trained_policy(get_config("mme_vla_suite"), <run>/39999)` 的 bf16 img，成批）；
C = A 的权重 `astype(bf16)` 再成批算（诊断臂：C == B 逐位 ⇒ A/B 之差完全由权重 dtype 解释）。
对拍对：S-B（训练/推理真实差距）、S-A（纯批形状）、A-B（纯 dtype）、A1==S 与 C==B（逐位，脚本正确性前提）。
运动路各臂固定为训练库该 episode 的真 motion 行（`MotionStore.rows` 按起点帧号查表；P5 已证与真 sidecar 逐位同），不起 sidecar。
第一轮只有 A/B/C 三臂（`records/replay-v1-3arm/`），发现 A 与库 0/227 逐位后加 S、A1 两臂拆分来源。

## 口径

- episode：`VideoUnmask` raw_ep 0（清单 g=200，es=66，T=239，`total_sample_offset=66566`；有 demo 段，两段窗口都覆盖）。
- 节奏复刻 `examples/robomme/eval.py`：首批 `[0, 66]` 传 `exec_start_idx=66`，之后每批 16 帧传 0；infer 时刻 t = 66, 82, …, 226 共 11 个点。
- 动作：`policy._sample_actions(jax.random.key(0), obs, noise=N_s)`，`N_s = normal(key(s), (1,20,32))`，s ∈ {0,1,2}；
  各对差的 RMS 与「S 臂换 noise seed 的两两 RMS」（采样噪声量级）及 `norm_stats.actions.std` 均值作参照；S 臂同噪声重跑逐位作确定性自检。
- 观测：`observation/image = front_rgb[t]`、`observation/wrist_image = wrist_rgb[t]`、`observation/state = joint‖gripper[:1]`、`prompt = setup/task_goal`，
  经生产 `policy._input_transform`（与 `eval.py` 的 element 同键）。
- 训练侧对照：`FrameSampDataset[idx(g,t)]`（`_create_framesamp_dataset(400ep/framesamp, norm_stats 取自 policy, history_config.resolved.yaml, 20)`，
  `MMEVLA_MOTION_STORE=400ep/motion`），S 侧装配八键逐位比。
- 数据：`v1-store/datasets/4task-motion-400ep/{framesamp,motion}`；原始 h5 `/scratch/hongze/robomme_data_h5/record_dataset_VideoUnmask.h5`。
- checkpoint：`v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999`（EMA 树，bf16 restore）。
- 底层存储介质：AWS 本地 NVMe RAID（`/dev/md0`）；耗时只作量级。

## 启动命令

`run_weights.sh`（CPU）与 `run_replay.sh`（GPU 0，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`）同目录，均在 worktree 内运行、
`UV_PROJECT_ENVIRONMENT` 指主副本 `.venv`、`MMEVLA_V1_STORE` / `OPENPI_DATA_HOME` 指主副本 `v1-store`。

## 输出

- `records/weights.txt`、`records/weights.json`
- `records/replay.txt`、`records/replay/summary.txt`、`records/replay/per_point.json`（五臂）
- `records/replay-v1-3arm.txt`、`records/replay-v1-3arm/`（第一轮三臂）
