# 官方16×100 modulation + concat（不 split）60k：正式启动档案

## 一、目的与状态

按[计划](../../../1030-orig16-context-modul-nosplit-60k-plan.md)补训 concat 读法（每窗 `1×1024`，`stage2_pool4_concat_v1`），与已完成的 split 对照 [orig100-m32x4x4-modul-motion-s2split4x256-b128-60k](../orig100-m32x4x4-modul-motion-s2split4x256-b128-60k/result.md) 组成 2×2。本条为第二条，context 正常结束且无待决失败后接续。本档案随正式 Beta 提交；起跑后实际时间、句柄与 Step0 数值以「实际起跑回填」小节补写，训练未完成前不记作完成。

用户原话（2026-10-30 美东）：「给出方案 现在训了context和modul split的模型 同时再训2个不split的」「同意增量上传诊断失败后就问用户 改写md」「我希望开工的时候只有一份仓库和一份temp」「和上次一样」「其他你来决策」「开工」。诊断、短测、打包、上传、回读、恢复任何失败都写 `WAIT_USER_DECISION` 并暂停相关后续步骤，询问用户。

## 二、版本与可复现锚点

| 对象 | 身份 |
|---|---|
| 计划 | `1030-orig16-context-modul-nosplit-60k-plan.md`（修订 1fe56c6，随 commitV11.68 合入） |
| 代码 | commitV11.68 `6cd4d6cd11ab6e718d05af5b8dc9cda8c656599c`（启动合同四项映射、concat 串行入口、导出链、rg→grep） |
| 两条 20 步与严格恢复 | 在 `6cd4d6c` clean HEAD 上执行，结果见第四节 |
| 正式训练 | 本档案所在 Beta 提交；完整 SHA 由 `TRAIN_HEAD` 写入日志及本地 HOLD |
| encoder | `stage2-full144-1025-a/checkpoint_epoch_105.pt#model`，SHA `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035` |

## 三、完整启动与还原

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
export TRAIN_HEAD=$(git rev-parse HEAD)
bash v1-store/logs/o16-concat-20261030/formal-chain.sh                  # tmux o16-concat-train-chain-<美东戳>
ROLE=ctx   CHAIN_SESSION=<chain会话名> bash v1-store/logs/o16-concat-20261030/checkpoint-backup.sh
ROLE=modul CHAIN_SESSION=<chain会话名> bash v1-store/logs/o16-concat-20261030/checkpoint-backup.sh
```

chain 调用 `scripts/training/run_orig16_concat_ctx_then_modul_60k.sh`（`FINAL_ROOT=v1-store/validation/orig16-concat-60k`），再逐条调用单条入口 `run_orig16_stage2_60k.sh`，后者写完整 `TRAIN_CMD`。配置还原：`git show <TRAIN_HEAD>:src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-orig16.yaml` 加训练配置 `mme_vla_suite_orig16_b128_60k`；无额外 CLI 覆盖超参。

## 四、数据、同源与起跑前判定

唯一库 `16task-orig100-1600ep-9e109419`（1600 集、476857 执行样本、`framesamp-4x4`、motion 表 23253 行、训练最大 44 窗）。起跑前判定（原文）：

```text
DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=31d6ab0f845336e76dc25db64170a0bc4ac59ecf366c161025b9800285c74467 vla_input_pin=5d1bfed54377c22e68bc1883ebd35fdf77372129ac1565dd6a78601c1d9965f9 files=16 mismatches=0
PAIRED_RUNS_SAME_LIB=PASS input_pin=5d1bfed54377c22e68bc1883ebd35fdf77372129ac1565dd6a78601c1d9965f9
NOSPLIT_CONFIG=PASS fusion=modul diff_keys=motion.readout,motion.num_tokens,motion.dim,motion.budget
MOTION_LAYOUT=PASS readout=concat tokens_per_window=1 budget=96 total_tokens=608
NOSPLIT_INPUT=PASS samples=4 windows=89 mismatches=0
LAUNCH_CONTRACT=PASS checks=50 failed=0
```

## 五、固定配置与交付

run `orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k`；融合 modulation，`memory_token_dim=1024`；512 帧位（32 帧×4×4）+ 96 motion 位（96 窗×1 token×1024 维），总 608。B128、FSDP8、workers16、seed42、60000 步，warmup5000、peak/decay_lr 5e-5、EMA0.999，从 `pi05_base` 新初始化。保存 5000…55000、59999 共 12 份到 `v1-store/train-runs/mme_vla_suite_orig16_b128_60k/orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k`；增量公开备份到 `HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2concat1x1024-60k-v1`，本地 stage `v1-store/exports/hf-ckpt-orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k/stage`。G12（`check_stage2_checkpoint.py --mode delivery --delivery-step 59999`）在两条训练结束后从 stage 恢复；HF 全量回读与 G12 均 PASS 后按用户「以后都要删除」删除本地 stage/cache。

## 六、硬件、句柄与监督

AWS 8×A100-SXM4-80GB、本地 NVMe RAID；训练 8 卡独占串行，上传不占 GPU。参考 split 端到端约 40.6 h（context）/16.3 h（modulation），concat 实际步时起跑后回填。起跑后主检出写 HOLD 冻结，运行态写开发副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp` 分支 `dev-o16-concat-20261030`。

## 七、两条 20 步短测（美东 2026-10-30 15:56–16:15，代码 6cd4d6c clean）

脚本 `v1-store/logs/o16-concat-20261030/smoke.sh`（tmux `o16-concat-smoke-10301556`，日志 `smoke.log`）：8 卡、`MAX_STEPS=20`，随后父 GPU0／在线 GPU1 从 `19/` 严格恢复。

```text
o16-smoke-concat-ctx-10301556:   LAUNCH_CONTRACT=PASS checks=50 failed=0; TRAIN_FINAL_RECORD=PASS tail_steps=19 ema_leaves=59; EXIT_CODE=0
  SAVE_LOAD=PASS missing=0 extra=0 shape_mismatch=0 dtype_mismatch=0 ema_mismatch=0 reference_output_max_abs=0 online_infer=1
o16-smoke-concat-modul-10301556: LAUNCH_CONTRACT=PASS checks=50 failed=0; TRAIN_FINAL_RECORD=PASS tail_steps=19 ema_leaves=65; EXIT_CODE=0
  SAVE_LOAD=PASS missing=0 extra=0 shape_mismatch=0 dtype_mismatch=0 ema_mismatch=0 reference_output_max_abs=0 online_infer=1
SMOKE_ALL=PASS runs=2 train_head=6cd4d6cd11ab6e718d05af5b8dc9cda8c656599c
```

Step0：ctx loss 0.5561、grad_norm 23.1155；modul loss 0.0923、grad_norm 1.3414，均有限，与 split 对照 Step0（ctx 0.6093／modul 0.0918）同量级。报告在 `v1-store/validation/orig16-concat-60k/o16-smoke-concat-*-10301556/`；短测 checkpoint（11G＋12G）按用户「以后都要删除」于通过后删除，日志与报告保留。
