# mixed8-200 32 帧 8×8 context 2048 80k：中止，作测速留档

## 一句话结论

context 2048（32 帧 × 8×8 记忆 token 拼进主干前缀）在 4 张 A100-80GB、global batch 128（每卡 32）下能稳定训练、不 OOM，但稳态 **10.09 s/步**，是同卡同 batch modulation 2048（1.9 s/步）的约 5.3 倍，80k 推算约 9.3 天。用户在第 455 步（美东 2026-10-18 19:09:36）叫停：「停止context2048 作为测速留档 大产物删除」。本 run 未产出任何 checkpoint，HF bucket 未创建。

| 指标 | 值 |
|---|---|
| 起跑 | V11.27Beta `42b2523`，美东 2026-10-18 17:43:57，tmux `m2k-train-ctx`，GPU 0–3 |
| 首次编译 | 约 9 分钟（Step 0 约美东 17:53） |
| 叫停 | 美东 2026-10-18 19:09:36，`tmux kill-session -t '=m2k-train-ctx'`（强杀，日志无 `EXIT_CODE=`），最后进度 455/80000 |
| 稳态步时 | 10.09 s/步（tqdm：step 100 美东 18:09:52 → step 400 美东 19:00:19，300 步 3027 s）；tqdm rate 全程 10.1 s/it |
| 每卡显存峰值 | 79445 MiB（20 步短测，`XLA_PYTHON_CLIENT_PREALLOCATE=false`，NVML 500 ms），约占 80 GiB 的 97% |
| 吞吐 | 128 / 10.09 ≈ 12.7 样本/s |
| wandb | `i4iynv4i`（项目 robomme-framesamp） |

## 训练过程

| step | loss | grad_norm | mem_enc_norm |
|---|---|---|---|
| 0 | 0.4653 | 13.1289 | 6.6757 |
| 100 | 0.2108 | 4.5516 | 2.4076 |
| 200 | 0.0534 | 0.8935 | 0.4857 |
| 300 | 0.0342 | 0.6905 | 0.3496 |
| 400 | 0.0283 | 0.5458 | 0.2697 |

全程无 Traceback、RESOURCE_EXHAUSTED、out of memory、NaN。前 400 步处于 warmup（5000 步），loss 只作有限性与下降趋势参考，不作效果结论。

## 同环境对照（环境 B，8×A100-SXM4-80GB，AWS 本地 NVMe RAID `/dev/md0`，同库 `16task-mixed8-200-3200ep-e3c7564f`，fsdp4/w8，batch 128）

| run | 记忆 | 接入 | 稳态步时 | 每卡峰值 | 来源 |
|---|---|---|---|---|---|
| 本 run | 2048 | context | 10.09 s/步 | 79445 MiB | 本档案 |
| `mixed8-200-m32x8x8-modul-b128-80k` | 2048 | modulation | 1.84 s/步（20 步短测）；正式训练 1.9 s/步 | 67907 MiB | 同时并行在 GPU 4–7 |
| `mixed8-200-m8x8-modul-b128-80k` | 512 | modulation | 1.718 s/步 | — | commitV11.24 |

两条 2048 在 G4 短测时 4+4 同时跑，正式训练时也同时跑；本 run 与 modulation 2048 的步时是同一时段、同一 IO 争用条件下测得的。context 慢的来源按机制推断：2048 个记忆 token 拼进主干前缀，前缀从约 1108 变为约 2644，注意力项按平方粗估约 5.7 倍；未做逐层剖析，属推断。

## 产物处置

- 本 run 目录 `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m32x8x8-context-b128-80k/` 只有 history_config 与 provenance 等元数据（20K），保留；日志 `v1-store/logs/mixed8-200-m32x8x8-context-b128-80k.log` 保留。
- 用户「大产物删除」：删除本计划三份短测 checkpoint（`m2k-smoke-ctx-20261018T205822Z/19`、`m2k-smoke-modul-20261018T205822Z/19`、`m2k-ckpt100-ctx-20261018T211447Z/100`，各约 11–12G，合计约 34G）；三个 run 目录的元数据与日志保留。
- 增量上传会话 `m2k-inc-ctx` 在训练会话消失后自行收尾（无 step 可传，未建 bucket `HongzeFu/robomme-vla-mixed8-200-context-2048-80k-v1`）；正式上传驱动 `run_mixed8_200_vla_ctx2048_80k_ckpt_export.sh` 不再运行。

## 用户决定

1. 「太慢或太慢就不要管了」（起跑前裁决，按此照常起跑）。
2. 「停止context2048 作为测速留档」「大产物删除」（美东 2026-10-18 19:09 前后）。

## 下一步

GPU 0–3 空出，不自动挪用。modulation 2048 继续跑，结束后按计划收尾。
