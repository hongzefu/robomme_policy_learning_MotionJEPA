# 8 帧 × 8×8 实施：参考工具与起跑核对

用户本轮指令：“开始实现 有问题越早问用户越好 一口气全做完！”；补充约束：“注意你只能用4个gpu”。全部 GPU 进程仅允许使用物理 GPU 4、5、6、7，阶段一冒烟使用 4、5。目标计划为根目录 `8frame-8x8-training-plan.md`，本轮授权已经从计划文档扩展为按该计划实施、建库和验收。

用户对摘要口径的明确决定：“采用文件 SHA＋解析后数组摘要双重检查”。JSON 原始文件 SHA 与数组原始字节摘要属于不同哈希域，分别记录；训练真实 loader 接收的 `data_config.norm_stats` 与指定文件解析出的 state/actions 各四个数组逐项比较，文件 SHA 仍独立锁定。正式 400ep 验收的文件 SHA 为 `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`。

## 版本与环境

开始实施时 HEAD 为 `ad82b6034d3e825efa74915875014e19428738d0`，`git status --porcelain` 为空。仓库位于 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，属于环境 B；存储为 `/dev/md0`、XFS、本地 NVMe RAID。实测 8 × A100-SXM4-80GB，任务仅使用后四张卡；`/data/hongzefu`、NFS 路径和 `~/.ssh/config` 均不存在。磁盘可用约 4.1 TiB。

40ep 与 400ep 的 framesamp/motion 库均为 verified，分别有 13,756 / 123,044 帧和 772 / 6,832 个 motion token。两份 `framesamp-8x8/` 均不存在。源库、旧库与模型目录均为 scratch 内的实体路径，未开始下载或重抽特征。

两份 norm_stats 文件 SHA 分别为 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`（旧 40ep）与上述 400ep 文件。`train.py mme_vla_suite --help` 已确认支持 `--data.assets.assets-dir` 和 `--data.assets.asset-id`，无需新增全局配置条目。

## 阶段一实施与验证

参考 Dataset 直接读取源 npy，复用既有采样与 motion 协议，独立实现特征装配与补零调用；生产 `framesamp_store.py`、`framesamp_dataset.py` 与在线 `framesamp_memory.py` 在此阶段保持原版本。两份新 YAML 只增加 8×8 档所需的 `token_per_image: 64`，开启 motion 的新 YAML 将 `online_gpu` 设为 4。缓存覆盖通过 `MMEVLA_JAX_CACHE_DIR` 控制，训练超参仍由启动命令覆盖。

已通过的轻量验证：`test_padding_dtype.py` 在真实 40ep 清单上 14 项通过，耗时 5.80 秒；4×4 参考 Dataset 与 packed Dataset 在 motion 开、关两种配置下各取 16 个真实边界样本，所有键的类型、形状和 raw 摘要一致；真实 transforms + collate 冒烟对拍 11 个样本与 4 个 batch 一致，另核对全部 11,530 个执行样本的身份。小规模 fixture 不替代阶段四的完整 200 batch 验收。

`gate_8x8.py --self-test` 已验证六类负例会失败：导入来源错误、遗漏 None 键、遗漏状态摘要步、错误 norm_stats、NaN 标量、旧 checkpoint。完整 TrainState 摘要分别记录 `phase`、`loop_step` 和 `state_step`，并记录每个叶子的有限性。候选 checkpoint 的 params 叶子数与该 run 实际参数子树计数比较，避免把全树叶子数混作参数数。

## GPU 冒烟命令与判据

先提交阶段一代码形成 clean HEAD，再运行下列 100 步回归，通过后将该代码提交确认为 `REF`。临时 run 使用已授权的 `t8-c32-s100`，tmux 会话全名同为 `t8-c32-s100`；结束后仅清理本轮核实的临时 run 和取证目录。该冒烟用于排错与历史标量摘要核对，不替代 12 条正式 1000 步轨迹。

```bash
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export CUDA_VISIBLE_DEVICES=4,5
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/t8-c32-s100"
export BENCH_RECORD_DIR="$V1_STORE/bench/8x8/t8-c32-s100"
export BENCH_DIGEST_INTERVAL=25 BENCH_EXTRA_DIGEST_STEPS=99
export BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1
uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite \
  --exp-name t8-c32-s100 --assets-base-dir "$V1_STORE/train-assets" \
  --checkpoint-base-dir "$V1_STORE/train-runs/t8-c32-s100" \
  --batch-size 8 --num-workers 4 --num-train-steps 100 --log-interval 1 \
  --save-interval 1 --seed 42 --fsdp-devices 2 \
  --dataset-path "$V1_STORE/datasets/4task-motion-40ep/framesamp" \
  --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
  --model.use-history --model.history-config perceptual-framesamp-context.yaml --no-wandb-enabled
uv run --no-sync python scripts/training/tests/project_scalars.py \
  "$BENCH_RECORD_DIR/metrics.jsonl" "$BENCH_RECORD_DIR/scalars_hex.tsv"
```

历史基线为 `aws-t2-ref-s100`、代码 `c5925d96305f771058e2206ae89461269af9d97c`。对该基线已有指纹字段逐项比对通过，唯一差异是计划允许的 `CUDA_VISIBLE_DEVICES: 0,1 → 4,5`。两者 `uv.lock` SHA 均为 `02cbc3ba67a9024f8afb9e31f60661c9abdcc3eb680ae80a8ee464c639327221`。本次新增记录 `flax=0.10.2`、`optax=0.2.4`、解析后 norm_stats 数组、manifest 与 motion meta；这些新增字段不冒充历史 run 已记录的事实，12 条正式轨迹将同场重新采集。

冒烟判据：日志 `EXIT_CODE=0`、100 行逐步指标、投影文件与历史基线 scalars 文件 SHA 完全相同。训练由 detached tmux、`PYTHONUNBUFFERED=1`、`set -o pipefail` 和 `tee` 执行，日志为 `v1-store/logs/t8-c32-s100.log`。完整 1000 步、输入和推理验收仍待后续阶段完成。
