# v2-1600ep-m8x8-ctx-b128-50k-gl 训练结果（第二段，2026-11-17 美东）

> 第一段（版本、启动、配置还原、实际起跑）见同目录 [`launch.md`](launch.md)；中途快照见 [`interim-1114.md`](interim-1114.md)。代码锚点 Beta `af56267`（`commitV4.6.5Beta`），计划 `1111-gl-4task400-ctx512-1024-b128-50k-plan.html`。闭环评估另见 [`../gl692-ctx512-50k/`](../gl692-ctx512-50k/launch.md)。

## 一、一句话结论与指标速览

context 模式 budget 512（8 帧 × 8×8，无 motion）在四任务 × 400 库上，以 bs128（每卡 16 × 4 卡 × 梯度累积 2）训练满 50000 步，正常退出，`EXIT_CODE=0`。10 份 ckpt（5000–50000）都已落盘，全程没有 `Traceback`、`RESOURCE_EXHAUSTED` 或 `nan`。最终 loss 0.0022，grad_norm 0.0235。成功率由 692 局评估给出，见评估档案；本档案不含成功率。

判定行：`TRAIN_DONE=PASS run=v2-1600ep-m8x8-ctx-b128-50k-gl steps=50000 ckpts=10 nonfinite=0 exit=0`

## 七、训练过程行为

- **起止时刻**：2026-11-12 05:29:50 美东起跑，2026-11-17 18:10:25 美东结束。tqdm 记录的墙钟为 132:37:25，即约 5.5 天。
- **步速**：全程稳定在 9.5 s/步，与 V7 冒烟实测的 9.4 s/步一致。每 5000 步保存一次 EMA 参数与 `resume/`，其中 resume 约 45.4 GiB，写盘约 1 分钟。最后一步含最终保存，tqdm 显示 88.2 s/it。
- **数值稳定**：日志里 `Traceback` / `RESOURCE_EXHAUSTED` / `nan` 均为 0 次。`metrics.jsonl` 共 501 行（step 0–50000，每 100 步一行）。
- **loss 走势**（每 2500 步抽一点，前半段见 interim-1114.md）：

| step | loss | grad_norm |
|---|---|---|
| 0 | 0.5297 | 60.0217 |
| 10000 | 0.0060 | 0.0378 |
| 20000 | 0.0041 | 0.0332 |
| 25000 | 0.0036 | 0.0276 |
| 27500 | 0.0034 | 0.0299 |
| 30000 | 0.0032 | 0.0291 |
| 32500 | 0.0030 | 0.0282 |
| 35000 | 0.0028 | 0.0289 |
| 37500 | 0.0027 | 0.0266 |
| 40000 | 0.0025 | 0.0253 |
| 42500 | 0.0025 | 0.0253 |
| 45000 | 0.0024 | 0.0269 |
| 47500 | 0.0022 | 0.0248 |
| 50000 | 0.0022 | 0.0235 |

- **ckpt 落盘**（`<GL_REPO>/v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-ctx-b128-50k-gl/<step>`，每份 13G，均含 `_CHECKPOINT_METADATA`）：5000 11-12 18:52、10000 11-13 08:12、15000 11-13 21:27、20000 11-14 10:41、25000 11-14 23:56、30000 11-15 13:10、35000 11-16 02:23、40000 11-16 15:39、45000 11-17 04:53、50000 11-17 18:07（均为美东）。
- **一致性**：50000 的 `assets/robomme/norm_stats.json` sha256 为 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`，与起跑口径一致；history config resolved sha256 为 `9fa83497c0ce1e5b493b248c79e3c9b67efe03f9c7de2b84bf192a4932050d15`。

## 八、训练后评估

按计划 `1116-gl-4task400-ctx-692-eval-plan.html`，在原占位 65943150 内评 50000 步 ckpt，跑 692 局、seed 7，评估档案为 `docs/training-doc/gl692-ctx512-50k/`。评估链 tmux `g4c-ev512-chain` 在训练退出后自动接上：

- g6b 冒烟：`EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS peak_mem_gb=35.6`。
- 压测：`EVAL_CAPACITY=PASS per_gpu=2 peak_mem_gb=35.6 free_gb=9.4`、`HOST_MEM=PASS projected_gb=92.4 limit=115`、`CAP_TIMEOUT=PASS wall_s=124`。
- 正式评估：2026-11-17 18:39 美东开始，每卡 2 片 × 显存比例 0.38，共 8 片。

成功率与对照比较写进评估档案的 result.md。

## 九、用户决策记录

- 「我要训练的是这个四 task……context 模式下的这个呃 memory budget 五一二和一零二四，然后都是 batch size 一二八，然后都是五十 k step。」「然后两个都 query 四卡」「对的，要用梯度累积来」「好的就保持吧」（起跑前，见 launch.md）。
- 「给出方案 跑完了直接在本地原job评估 参考aws的评估方式」：训练结束后**不 scancel 占位 65943150**，转交 692 局评估使用；按 1116 计划 R3，只有 `EVAL692=PASS` 且 `new_errors=0` 后才 scancel。

## 十、计划外事件与处置

训练本身没有计划外事件。评估链在 g6b 关先后遇到三个工程故障，均已修复，不影响训练产物：

1. `env` 命中 `~/.local/bin/env`，改用 `/usr/bin/env`；
2. server 缺仓库根 PYTHONPATH；
3. 合并 95faf61 遗留的 eval_v17 客户端依赖错配，由 `4ba8b49` 修复。

细节见评估档案与 `4ba8b49` 的 commit body。

## 十一、结论与下一步

训练完成，可作为 context 512 @50000 的最终 ckpt。下一步等 692 局评估出结果，与 modulation 对照 `v2-1600ep-m8x8-modul-b128-60k` @50000（211/692 = 30.49%）比较。ctx1024 训练预计 11-20 约 06:00 美东结束。

## 十二、归档文件清单

本档案只引用 turbo 上的原始文件，不另拷贝：日志 `<GL_REPO>/v1-store/logs/v2-1600ep-m8x8-ctx-b128-50k-gl.log`，逐步指标 `<GL_REPO>/v1-store/train-records/v2-1600ep-m8x8-ctx-b128-50k-gl/attempt01/metrics.jsonl`（另有同目录 `run_meta.json`、`runtime.json`）。`<GL_REPO>` = `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA`。
