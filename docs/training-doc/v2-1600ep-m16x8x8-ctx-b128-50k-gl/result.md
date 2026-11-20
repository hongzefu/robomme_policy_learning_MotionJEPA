# v2-1600ep-m16x8x8-ctx-b128-50k-gl 训练结果（第二段，2026-11-20 美东）

> 第一段（版本、启动、配置还原、实际起跑）见同目录 [`launch.md`](launch.md)；中途快照见 [`interim-1114.md`](interim-1114.md)。代码锚点 Beta `af56267`（`commitV4.6.5Beta`），计划 `1111-gl-4task400-ctx512-1024-b128-50k-plan.html`。692 局评估另见 [`../gl692-ctx1024-50k/`](../gl692-ctx1024-50k/launch.md)。

## 一、一句话结论与指标速览

context 模式 budget 1024（16 帧 × 8×8，无 motion）在四任务 × 400 库上，以 bs128（每卡 16 × 4 卡 × 梯度累积 2）训练满 50000 步，正常退出，`EXIT_CODE=0`。10 份 ckpt（5000–50000）都已落盘，全程没有 `Traceback`、`RESOURCE_EXHAUSTED` 或 `nan`。最终 loss 0.0020，grad_norm 0.0223。成功率由 692 局评估给出，见评估档案；本档案不含成功率。

判定行：`TRAIN_DONE=PASS run=v2-1600ep-m16x8x8-ctx-b128-50k-gl steps=50000 ckpts=10 nonfinite=0 exit=0`

## 七、训练过程行为

- **起止时刻**：2026-11-12 06:57 美东起跑，tqdm 从 07:00:25 开始计时；2026-11-20 06:30:47 美东结束。tqdm 记录的墙钟为 191:30:22，约 8.0 天。
- **步速**：全程稳定在 13.7 s/步，与 V7 冒烟实测的约 13.5 s/步一致；比 ctx512 的 9.5 s/步慢约 44%，与 context 序列长度加倍相符。每 5000 步保存一次 EMA 参数与 `resume/`。最后一步含最终保存，tqdm 显示 89.2 s/it。
- **数值稳定**：日志里 `Traceback` / `RESOURCE_EXHAUSTED` / `nan` 均为 0 次。`metrics.jsonl` 共 501 行（step 0–50000，每 100 步一行），loss 与 grad_norm 非有限值 0 个。
- **loss 走势**（每 5000 步抽一点）：

| step | loss | grad_norm |
|---|---|---|
| 0 | 0.4507 | 17.3486 |
| 5000 | 0.0086 | 0.0498 |
| 10000 | 0.0059 | 0.0365 |
| 15000 | 0.0045 | 0.0310 |
| 20000 | 0.0039 | 0.0326 |
| 25000 | 0.0033 | 0.0265 |
| 30000 | 0.0029 | 0.0264 |
| 35000 | 0.0027 | 0.0259 |
| 40000 | 0.0022 | 0.0248 |
| 45000 | 0.0022 | 0.0252 |
| 50000 | 0.0020 | 0.0223 |

- **ckpt 落盘**（`<GL_REPO>/v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m16x8x8-ctx-b128-50k-gl/<step>`，每份 13G，均含 `_CHECKPOINT_METADATA`）：5000 11-13 02:07、10000 11-13 21:14、15000 11-14 16:23、20000 11-15 11:29、25000 11-16 06:36、30000 11-17 01:43、35000 11-17 20:50、40000 11-18 15:55、45000 11-19 11:01、50000 11-20 06:28（均为美东）。
- **一致性**：50000 的 `assets/robomme/norm_stats.json` sha256 为 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`，与 ctx512 及起跑口径一致；history config resolved sha256 为 `70208fa373a470b5e0a19385d4a1db0984d3ed3bc70b8a635fe4f4cedf31e016`。

## 八、训练后评估

按计划 `1116-gl-4task400-ctx-692-eval-plan.html`，在原占位 65943151 内评 50000 步 ckpt，跑 692 局、seed 7，评估档案为 `docs/training-doc/gl692-ctx1024-50k/`。评估链 tmux `g4c-ev1024-chain`（已换上 ctx512 评估时修好的脚本，评估代码 `4ba8b49`）在训练退出后自动接上：

- 06:31 `TRAIN_DONE=PASS ckpts=10`。
- g6b 冒烟：`EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS peak_mem_gb=35.6`。
- 压测：`EVAL_CAPACITY=PASS per_gpu=2 peak_mem_gb=35.6 free_gb=9.4 host_delta_gb=25.5`、`HOST_MEM=PASS projected_gb=102.0 limit=115`、`CAP_TIMEOUT=PASS wall_s=122 projected_chunk_s=4880 limit=14400`。
- 正式评估：2026-11-20 06:38 美东开始，每卡 2 片 × 显存比例 0.38，共 8 片；`PLAN_OK shards=8 total=692 groups=14`、`PRESEED=PASS episodes=9 placed=9 duplicates=0`。

结果（07:33 美东）：`EVAL692=PASS episodes=692 success=160 rate=0.2312 new_errors=0`，即 160/692 = 23.12%，modulation 1024 对照 42.05%（−18.93 pp）；07:34 按 R3 scancel 65943151。逐格对照见评估档案 result.md。

## 九、用户决策记录

- 「我要训练的是这个四 task……context 模式下的这个呃 memory budget 五一二和一零二四，然后都是 batch size 一二八，然后都是五十 k step。」「然后两个都 query 四卡」「对的，要用梯度累积来」「好的就保持吧」（起跑前，见 launch.md）。
- 「给出方案 跑完了直接在本地原job评估 参考aws的评估方式」：训练结束后**不 scancel 占位 65943151**，转交 692 局评估使用；按 1116 计划 R3，只有 `EVAL692=PASS` 且 `new_errors=0` 后才 scancel。

## 十、计划外事件与处置

训练本身没有计划外事件。评估链所用脚本的三处工程修复（`/usr/bin/env`、`PYTHONPATH`、`4ba8b49`）在 ctx512 评估时已完成，见 `../gl692-ctx512-50k/result.md` ⑩。

## 十一、结论与下一步

训练完成，可作为 context 1024 @50000 的最终 ckpt。692 局评估 160/692 = 23.12%，比 modulation 对照 `v2-1600ep-m16x8x8-modul-b128-80k` @50000（291/692 = 42.05%）低 18.93 pp，与 context 512（22.83%）基本相同。

## 十二、归档文件清单

本档案只引用 turbo 上的原始文件，不另拷贝：日志 `<GL_REPO>/v1-store/logs/v2-1600ep-m16x8x8-ctx-b128-50k-gl.log`，逐步指标 `<GL_REPO>/v1-store/train-records/v2-1600ep-m16x8x8-ctx-b128-50k-gl/attempt01/metrics.jsonl`（另有同目录 `run_meta.json`、`runtime.json`）。`<GL_REPO>` = `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA`。

## HF 备份（2026-11-20 美东）

10 份 ckpt（5000–50000，原样 orbax 目录）与 run 根平文件已上传到**公开** bucket `HongzeFu/robomme-vla-4task-m16x8x8-ctx-b128-50k-gl-v1`，根下附 `SHA256SUMS`；`resume/` 未上传。上传与读回都在 GL standard 纯 CPU job 66535203 里做（脚本 `scripts/training/gl/hf_upload_verify_ckpt.sh`），读回为逐对象 `hf buckets cp <uri> -` 流式算 sha256 与字节数：

```
HF_VERIFY=PASS objects=186 sha_match=186 size_equal=186 missing=0 extra=0 bucket=HongzeFu/robomme-vla-4task-m16x8x8-ctx-b128-50k-gl-v1 bytes=115744430385
```

清单与清洗后的 job 日志见 `records/hf-upload/`。首次提交时 bucket 建成 private，账号 private 存储已满（约 1 TB），17:00 前后被 403 拒绝；按用户「上传公开的！！！！」改为公开后续传完成。
