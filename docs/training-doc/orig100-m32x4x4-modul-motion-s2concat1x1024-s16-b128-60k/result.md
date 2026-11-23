# orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k 结果档案（第二段）

## ① 一句话结论与指标速览

官方 16任务×100 上 modulation + motion（每窗 1×1024 concat、窗口步长 16、budget 192）训满 60k，官方 test 800 局三 seed（42/7/0）总体成功率 **44.79 ± 1.25**（366/362/347 局，45.75 / 45.25 / 43.38%，ddof=1）。并列参考：10-30 modulation split（4×256、stride 32）在 GL A40 上 **43.42 ± 0.52**，论文 FrameSamp+Modul 44.51 ± 0.77。本条在本机 A100 上用 `scripts/evaluation/local_eval.py` 评，平台不同，只作跨平台参考，不做显著性判读（用户 11-17 约定）。

| 指标 | 数值 |
|---|---|
| 训练 | 60000 步，`S16_RUN=PASS steps=60000 step=59999`，退出 0；稳态 0.971 s/步；Step 0 loss 0.0920 → Step 59900 loss 0.0013 |
| ckpt 上传 | `HF_VERIFY=PASS bucket=HongzeFu/robomme-vla-orig100-modul-s2concat-s16-60k-v1 objects=283 bytes=143020572846 mismatches=0`（公开桶，12 份） |
| seed 42 | `COMPUTE_DONE seed=42 completed=800 success=366 fail=395 timeout=39` |
| seed 7 | `COMPUTE_DONE seed=7 completed=800 success=362 fail=375 timeout=63` |
| seed 0 | `COMPUTE_DONE seed=0 completed=800 success=347 fail=411 timeout=42` |
| 三 seed | **44.79 ± 1.25**（easy 52.00±1.82、medium 46.35±2.71、hard 27.60±2.08） |

## ⑦ 训练过程

- 起跑：2026-11-22 美东 17:28（锚点 `commitV11.94Beta` fdce0f68），训完 11-23 09:41（ckpt 59999 落盘），收尾检查 09:43 `TRAIN60K=PASS`。
- loss 单调平稳下降，无 nan/inf/OOM（训练日志诊断正则零命中）；grad_norm 1.29 → 0.02；8 卡利用率全程 99–100%。逐步指标 `records/train-metrics.jsonl`（600 行，每 100 步），清洗日志 `records/train.summary.log`。
- 步速 0.971 s/步，与旧 split（0.970）几乎相同：记忆序列 704（512+192）比 split 的 896 更短，motion 步长减半未拖慢训练。
- stage2 motion 快照与 provenance 已写入 run 目录（`motion_snapshot.ready.json`、`motion_provenance.json`），即 S5 对 `train.py` 布局判等的修复生效。

## ⑧ 训练后评估（官方 test 800 局 × 3 seed，ckpt 59999，每局最多 1300 步，本机 8 卡×4 worker）

| 任务 | 本条三 seed（%） | seed42 / 7 / 0 | 参考：split 三 seed（GL A40） |
|---|---|---|---|
| BinFill | 42.00 ± 2.00 | 42 / 44 / 40 | 34.00 ± 5.29 |
| ButtonUnmask | 28.67 ± 2.31 | 26 / 30 / 30 | 25.33 ± 5.03 |
| ButtonUnmaskSwap | 15.33 ± 3.06 | 16 / 12 / 18 | 20.67 ± 1.15 |
| InsertPeg | 10.67 ± 2.31 | 12 / 8 / 12 | 10.67 ± 2.31 |
| MoveCube | 78.67 ± 4.62 | 84 / 76 / 76 | 84.00 ± 8.00 |
| PatternLock | 53.33 ± 2.31 | 52 / 56 / 52 | 46.67 ± 5.03 |
| PickHighlight | 28.67 ± 6.11 | 34 / 22 / 30 | 28.00 ± 4.00 |
| PickXtimes | 74.67 ± 7.57 | 66 / 80 / 78 | 72.67 ± 4.16 |
| RouteStick | 80.00 ± 4.00 | 84 / 80 / 76 | 68.67 ± 1.15 |
| StopCube | 64.67 ± 16.29 | 76 / 72 / 46 | 64.67 ± 6.11 |
| SwingXtimes | 96.67 ± 1.15 | 96 / 98 / 96 | 95.33 ± 1.15 |
| VideoPlaceButton | 40.67 ± 5.03 | 40 / 46 / 36 | 32.00 ± 2.00 |
| VideoPlaceOrder | 19.33 ± 1.15 | 20 / 18 / 20 | 32.67 ± 4.62 |
| VideoRepick | 26.00 ± 2.00 | 26 / 24 / 28 | 24.00 ± 4.00 |
| VideoUnmask | 30.67 ± 2.31 | 32 / 32 / 28 | 30.67 ± 1.15 |
| VideoUnmaskSwap | 26.67 ± 1.15 | 26 / 26 / 28 | 24.67 ± 2.31 |
| **总体** | **44.79 ± 1.25** | 45.75 / 45.25 / 43.38 | **43.42 ± 0.52** |

split 列取自 `c37b96ec:docs/evaluation-doc/o16-3seed-1031/result.md` ②节。评估用时：seed 42 09:43–11:23、seed 7 11:25–13:05、seed 0 13:07–14:50（每 seed 约 1 h 40 min，在线编码窗口约为 stride 32 的两倍）。汇总原件 `records/three_seed_summary.{txt,json}`、`records/seed42_summary.{txt,json}`。

## ⑨ 用户决策记录

1. 11-21「给出方案 重新训modulation + motion 但是这次stride 16 且不要split4 先只eval一个seed 在评估结束后立刻开始训 之前可以写代码 还是60k」
2. 11-21「budget 96翻倍！」
3. 11-21「磁盘给出清理的方案 现在就清理不用开工后再做 不要影响eval」「同意 开始删除」——删本地 4task rgb-state 1.8T（HF 有副本）。
4. 11-21「先进行一次workflow对抗验证 除非有影响训练语义的内容 否则你自己决策了 同意开工 可以push可以登记状态机」
5. 11-22「你需要挂一个兜底的监控机制……」「你的定时巡检能不能挂成每1-2小时巡检一次」「把这个写入 Agent MetaRules 以后，这种长的尤其是训练任务都要挂好每小时的巡检」——正本 f112204。
6. 11-22「你现在结束后会 evaluate 几个 seed，改为 evaluate 三个 seed 和其他的都一样」
7. 11-22「为什么还没有起跑训练，你到底在搞什么」「你为什么是三张卡在跑？……每次都要先查清楚了再开始啊！」

## ⑩ 计划外事件与处置

| # | 时间（美东） | 事件 | 处置 |
|---|---|---|---|
| r1 | 11-22 00:30 | 主检出切换后 `.agentmetarules` 子模块指针未对齐，`head_check` FAIL；单次 `grep -m1` 等待漏报，8 卡空转约 12.5 h，13:07 用户来问才发现 | 手工对齐子模块；加每小时 CronCreate 兜底巡检 + health.sh，并写入 AgentMetaRules 正本 |
| r2 | 13:08–13:48 | E-G3：闸门脚本写死的 MJ 工作树已前进到 b7cb8a2，`rgb_gtxyz_reader.py` 与钉版 b5d23a5 不同 | 新建只读钉版工作树 `/scratch/hongze/worktrees/mj-pin-b5d23a5`，原命令只换 `--mj-root` |
| r2→r3 | 14:50–15:52 | G4 评估侧逐任务串行 reset，每任务 2–21 min、7 卡空闲 | 停 r2，r3 复用 5 任务、其余 11 任务 8 卡并行（脚本不支持按局分片，末尾 3 任务仅 3 卡） |
| r3 | 16:42 | G5：`motion_client.validate_stage2_evidence` 缓存分支未传 `grid_subsample`，grid16 G0 奇数格被拒（G6 同因） | S6 修复（commitV11.91）→ 新锚点 commitV11.92Beta |
| r4 | 17:06 | G6：DATA_LINEAGE 读 encoder data_root（16views 本地 11-11 已删）下 build_spec.json，FileNotFoundError | S7 修复（commitV11.93）改读 keep-meta 副本，三源逐文件比对不变 → 新锚点 commitV11.94Beta |
| — | 11-23 09:43 | 健康脚本两次误报（`TRAIN_ES_BOUND_NOTE … 找不到`、`PASS … 失败=[]`）与收尾空档 GPU 空闲告警 | 排除 `_NOTE`/`=PASS` 行；收尾空档核实后未告警 |

闸门全部判定行：E `RAW_WINDOWS_READY=PASS windows=64`、`G2_INPUT_IDENTITY=PASS windows=64`；G2 `EVEN_GRID_BITEXACT=PASS mode=bitexact segments=2496 compared=23253 mismatches=0`；G3 `VERIFY_MOTION=PASS scanned=45228 mismatches=0`；G4 `TRAIN_ES_BOUND=PASS max_windows=86 budget=192 headroom=106`、`EVAL_ES_BOUND=PASS episodes=800 es_max=1135 headroom=42`；G5 `ONLINE_STAGE2=PASS windows=64 odd_grid=32 demo_tail=16 e2e_max_abs=0.0041`；G6 `LAUNCH_CONTRACT=PASS checks=51 failed=0`、`DATA_LINEAGE=PASS encoder_ckpt=27daa832 files=16 mismatches=0`；G7 `SMOKE20=PASS`。各轮链日志清洗件见 `records/o16s16-chain*.summary.log`。

## ⑪ 结论与下一步

- 训练与三 seed 评估全部完成；ckpt 12 份公开桶读回 PASS。三 seed 44.79 ± 1.25，与 split（43.42 ± 0.52，GL A40）及论文 FrameSamp+Modul（44.51 ± 0.77）并列；平台不同，同平台对照需在本机补评旧 split 59999（约 3×1h40m，用户未要求）。
- 待清理（另行处理，不影响本结果）：链脚本补同步 `.agentmetarules`、闸门脚本 MJ 路径改钉版工作树、`eval_es_bound.py` 增按局分片；S1–S7 子代理 worktree 与 `mj-pin-b5d23a5` 清理；v19 分支规则文件随下次开工同步。

## ⑫ 归档文件清单（records/）

`three_seed_summary.{txt,json}`、`seed42_summary.{txt,json}`、`train-metrics.jsonl`（600 行）、`train.summary.log`、`o16s16-chain.r1-headcheck-fail / r2-eg3-fail / r2-g4-serial / r3-g5-fail / r4-g6-fail .summary.log`、`o16s16-chain.summary.log`（r5）、`o16s16-inc.summary.log`、`o16s16-eval3.summary.log`。
