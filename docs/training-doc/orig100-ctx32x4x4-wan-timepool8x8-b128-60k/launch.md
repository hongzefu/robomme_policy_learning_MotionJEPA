# 官方16×100 context + Wan 时间池化 1024 60k：正式启动档案

## 一、目的与状态
替换 stage2 encoder 为零参数池化，其余与 `orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k` 一致，重新训练。计划：根目录 `1106-orig16-context-wan-timepool1024-60k-plan.html`（`ed5714c`）。

用户原话：「按照时间优先 8→1取均值 32×32→8×8 [1,16,8,8] 1024的机制 相当于替换我们的encoder 其他全部保持一致 都是context b128 60k」「只做pooling 不要用接续的wan」「仍去掉首帧，只平均后8个」；裁决「全归 E 一个脚本」「并入 smoke 句柄」；开工令「开工」「同意」。

## 二、版本与锚点
正式训练 HEAD 为本档案所在 Beta 提交，完整 SHA 由 `TRAIN_HEAD` 写入日志。建库与短测在 `19be5e9`；Beta 相对它只改文档，`src`、`scripts` 零差异。新表 `train_head=19be5e9`，表 SHA `df60d2b523c4b85a1ea27d7d5aa79b10d4b05f728da119830cbe2b713a8e3ca4`，索引 SHA `6f92f59547f9ac3965ceda844e500d51c106f3b831728b0114f8b62ff6ef45fc`。不加载任何 stage2 checkpoint。

## 三、完整启动与还原
```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
TRAIN_HEAD=$(git rev-parse HEAD) \
HISTORY=perceptual-framesamp-ctx-32frame-4x4-wan-timepool1024-orig16.yaml \
RUN_NAME=orig100-ctx32x4x4-wan-timepool8x8-b128-60k \
FINAL_RECORD_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/orig16-wan-timepool1024/final-record \
bash scripts/training/run_orig16_timepool_60k.sh
```
配置还原：`git show <TRAIN_HEAD>:src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-32frame-4x4-wan-timepool1024-orig16.yaml`，TrainConfig 为 `mme_vla_suite_orig16_b128_60k`，覆盖项见日志 `TRAIN_CMD=`。

## 四、数据与起跑前判定
库 `16task-orig100-1600ep-9e109419`，1600 集、476,857 执行样本、`framesamp-4x4`；motion 表 `motion-wan-timepool1024`，23,253 行，每行 1×1024。原文判定见 `v1-store/logs/orig16-wan-timepool1024/build.log`：
```text
DATA_LINEAGE=PASS encoder_kind=wan_fixedpool1024 encoder_ckpt=none files=16 mismatches=0
WAN_RESTORE_EQ=PASS windows=45228 input_mismatches=0 latent_mismatches=0
WINDOW_EQ=PASS windows=23253 copy=21814 fill=1439 demo_tail=386 exec_fill=1053 max_visible=44
POOL_CONTRACT=PASS shape=1x1024 dtype=float32 params=0 raw=1 drop_first=1
POOL_TABLE_EQ=PASS rows=23253 mismatches=0 nonfinite=0 bytes=95244288
VLA_PARAM_EQ=PASS total=3363663120 trainable=2948859424 frozen=414803696 leaves=59 trainable_leaves=36
VLA_INIT_EQ=PASS leaves=59 mismatches=0 seed=42
TRAIN_CONFIG_EQ=PASS
NON_MOTION_EQ=PASS mismatches=0
POOL_SHARED_EQ=PASS max_abs=0 bitwise_equal=1
LAUNCH_CONTRACT=PASS checks=60 failed=0
ONLINE_WINDOW_EQ=PASS windows=64 input_mismatches=0 cross_window_state=0
ONLINE_NUMERICS=RECORDED copy_tf32=true fill_online_tf32=false
SAVE_LOAD=PASS checkpoint_step=19 state_step=20
```
G7 误差：copy 路线 latent 相对 L2 约 2.7e-4、池化后约 1.9e-4、余弦大于 0.99999996；fill 路线 latent 逐位相等。64 窗覆盖 demo 完整 10、demo 补尾 8、exec copy 30、exec fill 16；8 个任务缺 demo 类别，已记入记录。

## 五、计划外事件
短测首次 G8 为 `SAVE_LOAD=FAIL`（参考输出 `max_abs=0.01947`），EMA 字节比对已通过。原因：`smoke.sh` 未设旧短测使用的 `JAX_COMPILATION_CACHE_DIR` 等环境，推理程序被重新编译。代码未改，补环境后在同一 checkpoint 上重跑通过，首次记录保留为 `SAVE_LOAD.attempt-*.json`，短测日志原样记 `EXIT_CODE=1`。计划写的是修复后用新 smoke 名重测；本次未改代码、未改权重，没有重训 20 步。

## 六、硬件与句柄
AWS 8×A100-SXM4-80GB，本地 NVMe RAID，8 卡独占。短测稳定段约 1.9 秒/步，参考旧同配置 31 小时 10 分。本轮 tmux：`o16-tp-fetch-11062053`、`o16-tp-restore-11062118`、`o16-tp-build-11062215`、`o16-tp-smoke-11062246`、`o16-tp-g8retry-11062305`、`o16-tp-train-11062312`。起跑后主检出写 HOLD 冻结，结束前不提交。
