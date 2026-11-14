# v2-1600ep-m8x8-ctx-b128-50k-gl 中途快照（2026-11-14 美东 18:03，训练未完成）

> 这是训练进行中的留档，供其他 agent 引用已落盘的中间 ckpt。训练完成后另写 `result.md`（两段式第二段），本文件不再改写。起跑档案见同目录 [`launch.md`](launch.md)，代码锚点 Beta `af56267`。

## 一、现状

- 模式：context，memory budget 512（8 帧 × 8×8）；四任务 × 400 episode 库 `4task-v2-1600ep-604f16da`（BinFill / RouteStick / VideoRepick / VideoUnmaskSwap）；无 motion（`motion_enabled=false`）。
- batch 128（每卡 16 × 4 卡 × 梯度累积 2），共 50k 步；配置 `mme_vla_suite_b128_60k`（warmup 5000、cosine decay 到 60000、峰值 lr 5e-5、seed 42），与对应 modulation 对照 run 的学习率曲线相同。
- GreatLakes 占位 65943150 / gl1513，aspen tmux `g4c-train-512`，2026-11-12 05:29:50 美东起跑。
- 2026-11-14 18:03 美东：第 22800 步，9.5 s/步，预计 2026-11-17 约 16:30 完成；日志里 `Traceback` / `RESOURCE_EXHAUSTED` / `nan` 出现 0 次。
- 指标文件最后一行：step 22700，loss 0.0037，grad_norm 0.0294。

## 二、已落盘 ckpt（可直接引用）

根目录 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-ctx-b128-50k-gl/`（turbo，aspen 与 GL 都能读）：

| step | 路径 | 落盘时刻（美东） | 大小 |
|---|---|---|---|
| 5000 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-ctx-b128-50k-gl/5000`（`params/` + `assets/robomme/norm_stats.json` + `_CHECKPOINT_METADATA`） | 2026-11-12 18:52 | 13G |
| 10000 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-ctx-b128-50k-gl/10000`（`params/` + `assets/robomme/norm_stats.json` + `_CHECKPOINT_METADATA`） | 2026-11-13 08:12 | 13G |
| 15000 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-ctx-b128-50k-gl/15000`（`params/` + `assets/robomme/norm_stats.json` + `_CHECKPOINT_METADATA`） | 2026-11-13 21:27 | 13G |
| 20000 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-ctx-b128-50k-gl/20000`（`params/` + `assets/robomme/norm_stats.json` + `_CHECKPOINT_METADATA`） | 2026-11-14 10:41 | 13G |

- 推理与评估要用 history config `perceptual-framesamp-context-8frame-8x8.yaml`（resolved sha256 `9fa83497c0ce1e5b493b248c79e3c9b67efe03f9c7de2b84bf192a4932050d15`，run 目录有 `history_config.resolved.yaml` 原文）。
- `norm_stats.json` sha256 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`，两条 run 相同。
- 数据来源记录见 run 目录 `motion_provenance.json`：framesamp 根 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-8x8`，manifest sha256 `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918`。
- `resume/`（完整训练状态，约 45 GiB，现为 step 20001）只用于续跑。它每 5000 步被原子替换一次，**不要当评估 ckpt 读**。
- ⚠ 中间 ckpt 处在学习率衰减中途，不能当成 50k 最终结果，也不能直接和其他 run 的 50k 数字比较；跨 run 比较要用同一步数（例如 ctx1024 `v2-1600ep-m16x8x8-ctx-b128-50k-gl` 的同步 ckpt）。
- 训练还在写这个目录（每 5000 步新增一份）。请只读，不要在里面新建、删除或移动文件。

## 三、loss 走势（metrics.jsonl，每 2500 步抽一点）

| step | loss | grad_norm |
|---|---|---|
| 0 | 0.5297 | 60.0217 |
| 2500 | 0.0108 | 0.0647 |
| 5000 | 0.0088 | 0.0523 |
| 7500 | 0.0071 | 0.0431 |
| 10000 | 0.0060 | 0.0378 |
| 12500 | 0.0054 | 0.0362 |
| 15000 | 0.0046 | 0.0308 |
| 17500 | 0.0046 | 0.0339 |
| 20000 | 0.0041 | 0.0332 |
| 22500 | 0.0039 | 0.0300 |

完整逐步指标：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/train-records/v2-1600ep-m8x8-ctx-b128-50k-gl/attempt01/metrics.jsonl`；日志：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/v2-1600ep-m8x8-ctx-b128-50k-gl.log`。

## 四、还没有的

- 这些中间 ckpt 都没做过闭环评估，成功率未验证。
- 50k 最终 ckpt、`result.md`、`TRAIN_FINAL_RECORD` 判定行都要等训练结束后补上。
