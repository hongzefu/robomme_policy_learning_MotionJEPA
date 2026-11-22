# orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k 起跑档案（第一段）

## ① 目的与一句话

在官方 RoboMME 16任务×100 上重训 modulation + motion：motion 每窗 1×1024（concat，不 split4），窗口步长 16（旧 32），budget 192（96 翻倍），其余与 10-30 的 `orig100-m32x4x4-modul-motion-s2split4x256-b128-60k` 完全相同；训满 60k 后只评官方 test seed 42（800 局）。方案 `1122-orig16-modul-s2concat-s16-60k-plan.md`（定稿 b28a9a5d，本档案随 Beta 提交）。

用户原话（2026-11-21 美东）：
1. 「给出方案 重新训modulation + motion 但是这次stride 16 且不要split4 先只eval一个seed 在评估结束后立刻开始训 之前可以写代码 还是60k」
2. 「budget 96翻倍！」
3. 「先进行一次workflow对抗验证 除非有影响训练语义的内容 否则你自己决策了 同意开工 可以push可以登记状态机」

## ② 版本与代码状态

- 分支 `v19-orig16-modul-s2concat-s16-60k`（自 `origin/v4-gl-orig100-1031@1c652dc7`），Beta 锚点 = 打本档案的 Beta 提交（`commitV11.90Beta`，完整 sha 见 `git log`，触发链以 `TRAIN_HEAD` 精确核对）。
- 本轮代码改动（全部 `--no-ff` 合入，合并前后两次审查）：
  - S1 `commitV11.85`：grid16 stage2 布局 `motion-stage2-1x1024-grid16-demopad17-v1`，导出/打包/校验按 `grid_subsample` 参数化。
  - S2 `commitV11.86`：在线与检查脚本 stride/budget/布局读 yaml/spec/store meta。
  - S3 `commitV11.87`：新 yaml、启动合同按 run 登记、runner `scripts/training/run_orig16_modul_s16_60k.sh`、上传包装器。
  - S4 `commitV11.88`：本机官方 800 局评估入口 `scripts/evaluation/local_eval.py`（profile `modulcat16`、`--seeds`）。
  - S5 `commitV11.89`：train.py stage2 快照判定、policy_config 传 grid_stride、去过渡回退、闸门脚本 `scripts/training/o16s16_gates.sh`、G0 抽样与 validate_stage2 的 `--grid-subsample`、G2 先严后宽。
- 训练语义：新 yaml 相对 `perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-orig16.yaml` 只差 `motion.stride 32→16`、`motion.budget 96→192`、`motion.store_path→motion-stage2-1024-grid16`（`YAML_DIFF=PASS keys=3`，启动合同 `STRIDE_PAIR_CONFIG` 机械核对）。

## ③ 启动与配置还原

- 配置：`git show <Beta>:src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-s16-orig16.yaml`；训练 config `mme_vla_suite_orig16_b128_60k`（8 卡 FSDP8、B128、workers16、seed42、60k、预热 5k、lr 5e-5、EMA 0.999、pi05_base），入口不加任何超参覆盖。
- 触发链 `v1-store/jobs/o16s16/chain.sh`（tmux `o16s16-chain`，日志 `v1-store/logs/o16s16-chain.log`），顺序：
  1. 只读等待 `v1-store/jobs/t1s-1120/T1S_CLOSEOUT_DONE`（t1s 会话收尾标记，跨会话约定 `req=f8ace2-11212205-1`）＋ G3 子集下载 `EXIT_CODE=0` ＋ 8 卡显存 <2000 MiB。
  2. 主检出 `git checkout --detach <Beta>`、`git submodule update --init third_party/robomme_benchmark`、`uv sync --frozen`。
  3. `GATE_GPU=0 ONLINE_GPU=1 bash scripts/training/o16s16_gates.sh`（E 证据重建 → G2 导出 → G3 打包 → G4 预算 → G5 在线一致 → G6 启动合同含 `DATA_LINEAGE`）。
  4. G7：`MAX_STEPS=20 RUN_NAME=o16-smoke-s2cat-s16 bash scripts/training/run_orig16_modul_s16_60k.sh`。
  5. 正式：`TRAIN_HEAD=<Beta> FINAL_ROOT=v1-store/validation/orig16-modul-s16-60k bash scripts/training/run_orig16_modul_s16_60k.sh`（训练日志 `v1-store/logs/orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k.log`）。
  6. 评估：`uv run --no-sync python scripts/evaluation/local_eval.py --profile modulcat16 --seeds 42`（输出 `v1-store/evaluation/orig100-modul-s2concat1x1024-s16-60k-1122`）。
- 增量上传 `v1-store/jobs/o16s16/inc.sh`（tmux `o16s16-inc`，日志 `v1-store/logs/o16s16-inc.log`）：训练日志出现后跑 `run_orig16_stage2_ckpt_inc.sh`，训练结束后跑 `run_orig16_modul32x4x4_concat_s16_60k_ckpt_export.sh`；公开桶 `HongzeFu/robomme-vla-orig100-modul-s2concat-s16-60k-v1`，本地 12 份全留。

## ④ 数据集与划分

- 训练库 `v1-store/datasets/16task-orig100-1600ep-9e109419`（与旧 split 同库，norm_stats `c5d45b4b…`）。
- motion：复用本机 grid16 Wan latent `v1-store/cache/orig16-wan-timepool1024-recovery/wan-latents/`（45,228 窗），stage2 encoder `stage2-full144-1025-a/checkpoint_epoch_105.pt`（sha256 `27daa832…`）对全部窗口重导出到 `motion-tokens-stage2-grid16/`，打包为 `motion-stage2-1024-grid16/`。证据重建到 `v1-store/validation/orig16-stage2-grid16/`；G3 所需 RGB 用 `plan_g3_subset.py` 从公开桶 `robomme-16task-rgb-gtxyz-stride2-16views-fp16-v1` 只读下载 32 窗子集（14 片、32,476,096,020 字节）到 `v1-store/external/rgb-gtxyz-g3-subset`。
- DATA_LINEAGE：encoder 训练源与 VLA 库同为官方 16×100 源 H5（G6 在起跑前机械核对）。

## ⑤ 关键超参

与旧 split run 相同，唯三差异见 ②。记忆序列 512 帧 token + 192 motion 槽 = 704。

## ⑥ 硬件与预计耗时

AWS p4de 8×A100-80GB，本地 NVMe RAID `/dev/md0`。G2–G7 约 40 分钟；60k 约 16 h（参照旧 split 0.97 s/步）；seed 42 评估约 1 h（在线编码窗口约翻倍，G5 实测后修正）。

## 本轮 tmux 会话清单

`o16s16-g3fetch`（G3 子集下载）、`o16s16-chain`（触发链）、`o16s16-inc`（增量上传）；评估由 local_eval 起的 `o16s16-ev-*`。清理只按本清单，`tmux kill-session -t '=名'` 一次一个。

## 盯盘项

各日志单独挂 Monitor：chain（`O16S16_|SMOKE20|TRAIN60K|S16_RUN|=FAIL|拒绝|失败|错误|找不到|Traceback|EXIT_CODE=`）、训练日志（`nan|NaN|inf|RESOURCE_EXHAUSTED|out of memory|Killed|Step [0-9]*000:|EXIT_CODE=`）、inc（`INC_|HF_VERIFY|FAIL|403|EXIT_CODE=`）、g3fetch。

## 起跑锚点变更与续跑记录（2026-11-22 美东）

| 轮次 | 时间（美东） | 结果 | 处置 |
|---|---|---|---|
| r1 | 11-22 00:30 | `O16S16_CHAIN=FAIL at=head_check`：主检出切换后 `.agentmetarules` 子模块指针未对齐 | 手工 `git submodule update --init .agentmetarules`；单次 `grep -m1` 等待漏报，GPU 空转约 12.5 h，至 13:07 才发现，之后加每小时兜底巡检 |
| r2 | 13:08–13:48 | E-G3 FAIL：闸门脚本写死的 MJ 工作树已前进到 b7cb8a2，`rgb_gtxyz_reader.py` 与钉版 b5d23a5 不同 | 新建只读钉版工作树 `/scratch/hongze/worktrees/mj-pin-b5d23a5`，原命令只换 `--mj-root` |
| r2→r3 | 13:50–15:52 | E 通过；G2 `EVEN_GRID_BITEXACT=PASS mode=bitexact compared=23253 mismatches=0`；G3 `VERIFY_MOTION=PASS scanned=45228`；G4 训练侧 `TRAIN_ES_BOUND=PASS max_windows=86 budget=192 headroom=106`；评估侧逐任务串行过慢 | 15:52 停 r2，r3 复用 5 任务、其余 11 任务 8 卡并行 |
| r3 | 15:52–16:42 | `EVAL_ES_BOUND=PASS tasks=16 episodes=800 es_max=1135 budget=192 headroom=42`；G5 FAIL：`motion_client.validate_stage2_evidence` 缓存分支未传 grid_subsample | S6 修复（commitV11.91），本提交为新锚点 |
| r4 | 16:50–17:06 | G5 `ONLINE_INPUT_IDENTITY=PASS windows=64`、`ONLINE_STAGE2=PASS windows=64 odd_grid=32 demo_tail=16 e2e_max_abs=0.0041`；G6 51 项中 50 项 PASS，CHECK_DATASET FAIL：encoder data_root（16views 本地已删）下 build_spec.json 不存在 | S7 修复（commitV11.93）：DATA_LINEAGE 改读 keep-meta 副本，三源逐文件比对不变；真实数据 `DATA_LINEAGE=PASS files=16 mismatches=0`；本提交（commitV11.94Beta）为新锚点 |
| r5 | 本提交后 | 链脚本 `v1-store/jobs/o16s16/chain5.sh`：主检出 detached 到本锚点 → G6 → G7 → 60k → seed42（G5 产物不绑定 HEAD，不重跑） | — |
