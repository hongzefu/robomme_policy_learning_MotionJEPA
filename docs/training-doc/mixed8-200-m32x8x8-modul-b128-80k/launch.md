# mixed8-200 32 帧 8×8 modulation 2048（无 motion）80k 正式训练

本页是 commitV11.28Beta 起跑前档案（AGENTS 第 12 条两段式第一段）；跑完后结果回写同号正式 commit 与 result.md。起跑 HEAD 即本 Beta 提交，由 runner 打印的 `GIT_HEAD=` 固定，本页不预填自身 sha。

实施依据：根目录 [`1018-mixed8-2048-context-modul-plan.md`](../../../1018-mixed8-2048-context-modul-plan.md) @ `fa14d0a`。另一条并行 run 见 [mixed8-200-m32x8x8-context-b128-80k](../mixed8-200-m32x8x8-context-b128-80k/launch.md)。

用户原话（按时间）：

1. 「给出方案 训练2048contetxt和modulation 先用subagent检查一下是否都可以实现 还是4卡分别跑 其他各个配置都都是一样的」
2. 「然后也是边训练边上传。」
3. 「本次不再需要建 HTML 计划，直接建 Markdown 计划。还是写在policy库的根目录」
4. 「都是三十二帧的八乘八。」
5. 「再审查一下，RAM 名和 GPU 名和上传位置都你定，太慢或太慢就不要管了。然后 OOM 是不可能的，应该是都会在的我之前跑过。如果oom就报错。上一轮回读副本删除，状态机登登记。」
6. 「开工」

## 版本与代码状态

代码为 commitV11.26 两次合入（`bb74c95` sub/S1 训练侧、`870857f` sub/S2 上传侧）之后的主检出；`src/mme_vla_suite/training/config.py` 不改，训练计算、dataloader 读取逻辑不改，只在 `FrameSampDataset.__init__` 形制守卫新增 context 2048 精确分支（对 modulation 2048 无影响）。uv.lock sha256 `02cbc3ba67a9024f8afb9e31f60661c9abdcc3eb680ae80a8ee464c639327221`，`uv run --no-sync`。配置还原：`git show <Beta>:src/mme_vla_suite/training/config.py` 中 `mme_vla_suite_b128_80k` 条目加下述 CLI 覆盖。

代码冻结：第一条 run 起跑前对被跟踪的 `src`/`scripts`/`packages` 文件上只读锁，原权限记入 `v1-store/m8v-scripts/m2k-chmod-before.txt`；两条训练都出现 `EXIT_CODE=` 后按原权限恢复。冻结期内只提交 `docs/`。

## 配置与启动覆盖

- run 名 `mixed8-200-m32x8x8-modul-b128-80k`；config `mme_vla_suite_b128_80k`（80k 步、warmup 5000、peak_lr=decay_lr=5e-5、AdamW clip 1.0、EMA 0.999、每 5000 步保存、batch 128、seed 42、pi05_base 初始化），与上一轮 512 两条一致。
- history yaml `perceptual-framesamp-modul-32frame-8x8.yaml`，文件 sha256 `42813c7e06840e511f8a382f14954edcaad8d4fd5b320b4323fea94979cda4de`；resolved sha256 `91512306b3aaaaa541d248bc5dfaf8be2849632cd3b9e89a226207c09c6501c0`（取自 20 步短测 run 的 `history_config.resolved.sha256`）。32 帧 × 8×8 = 2048 token，无 motion。`integration_type: modulation`，`memory_token_dim: 1024`；现成 yaml 不改（与 8 帧 modul 版只差 `budget: 2048`，无 motion 节）。
- 4 卡：`CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_DEVICES=4 NUM_WORKERS=8`；全局 batch 128（每卡 32）。
- 库、assets、`MMEVLA_FRAMESAMP_SOURCE/MANIFEST` 由 `scripts/dataset/hf_export/mixed8_200_runner/_train_common.sh` 固定指向 `16task-mixed8-200-3200ep-e3c7564f`；wandb：`v1-store/secrets/wandb.env` 在位即开启（项目 robomme-framesamp），与上一轮相同。

```bash
tmux new-session -d -s m2k-train-modul "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/mixed8_200_runner/train-modul2048-80k.sh mixed8-200-m32x8x8-modul-b128-80k"
```

## 数据与身份

| 对象 | 值 |
|---|---|
| 库 | `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f`（16 任务 × 200 = 3200 集） |
| canonical manifest | `d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad` |
| framesamp-8x8 | steps 2,145,548 / exec samples 1,383,846；store_meta `ace85246419836d343b925ac45591e3fd75eb2f054c16eaafafb3b35d6ca9b79` |
| norm_stats | `36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394` |
| 短历史 | `SHORT_HIST=43413/1383846`（约 3.1% 执行样本不满 32 帧，走既有补零路径） |

checkpoint 根 `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m32x8x8-modul-b128-80k`，预期 5000…75000 每 5000 一份加最终 79999；日志 `v1-store/logs/mixed8-200-m32x8x8-modul-b128-80k.log`（runner 拒绝覆盖已有日志，并自写 `END_UTC=`、`EXIT_CODE=`）。

## 起跑前闸门（环境 B，8×A100-SXM4-80GB，AWS 本地 NVMe RAID `/dev/md0`）

- G0 守卫（主检出合并后，mixed8 真实库，一次性脚本留存于 `v1-store/m8v-scripts/m2k-evidence/g0-guard-check-postmerge.py`）：`GUARD_CTX2048_POSTMERGE=PASS accept=1 reject_type=3/3 reject_combo=7/7 legacy_accept=5/5 modul32_reject=3/3`。
- G3 在线装配：`ONLINE_ASM_CTX2048=PASS steps=5 mem_len=2048`。
- G4 20 步短测（4+4 同时，`XLA_PYTHON_CLIENT_PREALLOCATE=false`，NVML 500 ms，美东 2026-10-18 16:58 起）：`SMOKE20_MODUL=PASS oom=0 loss_finite=1 exit_code=0 peak_mib=67907 steady_s_per_step=1.84 window=step13-19 est_80k_days=1.7`。Step 0：`Step 0: grad_norm=1.3978, llm_grad_norm=1.3848, loss=0.0874, mem_enc_norm=0.1188, param_norm=1815.5465`。稳态 1.84 s/步（20 步短测 step13–19 差分，窗口仅 4 个 Progress 采样点），80k 推算约 41 h；首次编译约 2 分钟。
- G5（100 步保存加载）只对 context 做；本 run 的 history yaml 与 2048 modulation 结构在旧 4 任务库上已完整训练过（`v2-1600ep-m32x8x8-modul-b128-80k`）。

### 实施中调整（AGENTS 第 2 条，已告知用户）

| 闸门 | 旧口径 | 新口径 | 理由与对抗验证 |
|---|---|---|---|
| G0 | `pytest scripts/training/tests/test_pack_guards.py` 全过 | mixed8 真实库只读守卫核对 + 新增测试代码静态审查；新增 pytest 用例本机未执行 | 环境 B 缺 fixture `mini_stores` 源数据：BASE `e65b5df` 上即「1 failed, 2 passed, 40 errors」，全为 `FileNotFoundError: v1-store/episode_manifest.json`（failed 的 `test_g6a_exec_lookup_formula` 同因）。`ADJUST_REVIEW=PASS scale=same findings=3` |
| G4 | `check_32frame_context.py smoke-summary` 出判定行 | 主会话一次性脚本按同一判据从原始 CSV 与日志计算，输出留存于 `v1-store/m8v-scripts/m2k-evidence/g4-smoke-summary.txt` | smoke-summary 两处解析缺陷：CSV 表头 `index` 在第 2 列未被识别、tqdm 正则不匹配 openpi `Progress on:` 格式，导致 `peak_mib=nan` 判 FAIL；训练本身无误。`ADJUST_REVIEW=PASS scale=same findings=2`。该子命令后续不再使用，缺陷不修，复用前须先修 |

## 训练中增量上传

| 项 | 值 |
|---|---|
| bucket | `HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1`（公开） |
| 增量会话 | tmux `m2k-inc-modul`，日志 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m2k-inc-modul.log` |
| 命令 | `bash v1-store/m8v-scripts/m8v-ckpt-inc.sh mixed8-200-m32x8x8-modul-b128-80k HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1 hf-ckpt-mixed8-200-vla-modul2048-80k scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_modul2048_80k.json m2k-train-modul`（脚本全文见上一轮 [512 档案](../mixed8-200-m8x8-modul-b128-80k/launch.md)「m8v-ckpt-inc.sh 全文」） |
| 正式驱动 | `scripts/dataset/hf_export/run_mixed8_200_vla_modul2048_80k_ckpt_export.sh`：训练 `EXIT_CODE=0`、增量会话 `INC_END` 且已退出后，先 `SMOKE=1` 再全量 |
| 完成判据 | 正式驱动 `RESULT=PASS`、`SHA256_PRE_POST_DIFF=0`、`BUCKET_PUBLIC=True`、`全部完成`；`INC_END` 只表示增量会话结束 |

## 本轮 tmux 会话清单

| 会话名 | 用途 |
|---|---|
| `m2k-smoke-nvml` / `m2k-smoke-ctx` / `m2k-smoke-modul` | G4 短测（已结束，nvml 已按清单清理） |
| `m2k-ckpt100-ctx` | G5 context 100 步保存加载 |
| `m2k-train-modul` | 本 run 80k 训练 |
| `m2k-inc-modul` | 本 run 增量上传 |

## 监控与用户决定

主会话对训练日志与增量日志各挂一个 Monitor，过滤 `EXIT_CODE=|Traceback|nan|NaN|out of memory|RESOURCE_EXHAUSTED|Killed`（增量日志另加 `INC_UPLOADED|INC_END|UPLOAD_FAIL|HYGIENE=FAIL`）。OOM、崩溃、NaN、掉卡：停下报错交用户，不 resume、不自动重跑、不换卡数；步时慢不停。
