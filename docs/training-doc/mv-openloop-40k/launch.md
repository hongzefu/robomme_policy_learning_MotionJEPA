# mv-openloop-40k — 起跑记录

**环境 B（AWS 单机 8×A100-SXM4-80GB）**。诊断 run（AGENTS 17），motion 利用率评估阶段 0 + 阶段 1（零 rollout）。正本 `docs/motion-utilization.md`，工具 `scripts/motion-variance/`（commitV8.0）。

起跑时间：<START_TIME>（矩阵后起跑，result.md 记实际时间）
起跑 commit：代码 `5620f662968ff68d278add6339a818c0be443490`（commitV8.0）；本留档在起跑前提交，工作区 clean
被评对象：`v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999`（bf16 加载，生产口径 `create_trained_policy`）
数据来源：训练库 `v1-store/datasets/4task-motion-400ep`（帧特征 / motion 表 / 真值动作）+ 原始 H5 `/scratch/hongze/robomme_data_h5`（前 n 帧原图，按真实评估节奏喂）；donor bank `v1-store/reports/motion-variance/bank-lib`（开环映射 `ol<seed>|task|recv_g`，排除自身）

## 口径

- 决策点：6 集 `ButtonUnmask:longest, ButtonUnmaskSwap:3, VideoUnmask:3, VideoUnmaskSwap:5, VideoUnmaskSwap:31, VideoUnmaskSwap:3`（覆盖 4 任务与 es ∈ {0, 66, 114, 168, 216}），每集全部决策时刻 `expected_taus(es, T)`（首批整段 demo，之后每批 16 帧）。
- 阶段 0：每点 × noise seed {0,1,2,3,4}，`none / mask / swap` 三种输入的最终动作（`f_sample`，生产 bf16 路径，与 `policy._sample_actions` 逐位相同）；标尺 = noise seed 两两差；`OL_ZEROK_NULL` 阻断。
- 阶段 1：每集 cold / early / mid / late 4 点（24 点），18 层展开路径内自洽比较：attention 富集（逐 query 合法-key 均匀基准）、`step_only/both/kv_vzero/kv_donor` 逐层干预的完整 10 步最终动作差、固定 `(obs, x_t, t, u_t)` 的逐层入口梯度（t ∈ {0.1, 0.5, 0.9}）。展开路径与 `nn.scan` 在 bf16 下不逐位（`UNROLL_VS_SCAN_BF16 rel_fro≈3.4e-3`），语义等价由 f32 闸 `UNROLL_VS_SCAN_F32` / `LAYER_SELFCHECK_F32` 判定（见 `records/adapter_selftest.log`）。
- 前提：GPU 后端硬闸、`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、`MMEVLA_MOTION_STORE` 显式指 400ep 库。

## 启动命令

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
tmux new-session -d -s mv-openloop -c "$PWD" \
  "set -o pipefail; CUDA_VISIBLE_DEVICES=<GPU> XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
   UV_LINK_MODE=copy PYTHONUNBUFFERED=1 uv run --no-sync python scripts/motion-variance/run_open_loop.py --stage 0,1 \
   --out v1-store/reports/motion-variance/open_loop.json 2>&1 | tee v1-store/logs/mv-openloop.log; echo EXIT_CODE=\$?"
```

## tmux 会话清单

本轮只起 `mv-openloop` 一个会话；用户既有会话一律不动。

## 输出

- `v1-store/reports/motion-variance/open_loop.{json,npz,summary.txt}`、`adapter_selftest_*.{log,json}`（不进 git）
- 留档 `records/{open_loop.summary.txt,open_loop.json,adapter_selftest.log}`；图 `docs/motion-utilization/figures/fig4-*`、`fig5-*`
