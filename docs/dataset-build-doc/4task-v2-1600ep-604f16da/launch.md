# 新版四任务 1600 集正式建库启动说明

## 结论与实施范围

按照 [合并建库计划](../../../v2-4task-h5-merge-plan.md) 交付全部 1600 条 primary 的物理合并 H5、SigLIP source、4×4 与 8×8 framesamp，以及这批数据独立的 norm_stats；最后在关闭 motion 的配置上跑 20 步训练可读性验证。Wan 抽取、motion 编码和 sim 评估均不属于本轮。正式运行须等 28 集冒烟全部通过，本文件先记录命令与口径，实际启动版本和时间另由阶段日志固定。

## 版本与代码状态

实现提交 `a0cdfe751fcd5bd4ce9fc7e4d5037dee03a4246e`（V9.4）包含新脚本、测试及用户批准的 `finalize_checks.check_inputs` 四文件并行 sha256。其余建库入口和训练链路未改。74 项测试、Ruff、diff 检查及一条真实 200 步 H5 对拍通过。各阶段日志先打印完整 `START_HEAD` 和 `git status --porcelain`；HF 在途工作按用户“继续工作 忽略huggingface的任务 但是要注意git”排除，并核查实际消费代码不变。SigLIP 到 finalize 期间本轮不提交。

## 数据来源与排序

源根 `/scratch/hongze/robomme-4task-h5-20260912-v2`；来源 `HongzeFu/robomme-4task-h5-20260912-v2`，revision `604f16da36d6b6d175884df8fb687dc08e0a36eb`，MANIFEST sha256 `df992cdf5a768ae0f368e520b0d7be28a68ed4ce6ca6201967cf5d6654cf2bb9`。按任务分组，再按 easy、medium、hard、xhard 与原 episode 编号排序，编为每任务 `episode_0..399`。原 episode 号和 seed 不能单独作为全局身份；每行 map 保留难度、member、src_group 和原始身份。

| 任务 | 难度 | 条数 | 合并后的 raw_ep_idx 闭区间 |
|---|---|---:|---|
| BinFill | easy | 134 | 0–133 |
| BinFill | medium | 133 | 134–266 |
| BinFill | hard | 133 | 267–399 |
| RouteStick | easy | 100 | 0–99 |
| RouteStick | medium | 100 | 100–199 |
| RouteStick | hard | 100 | 200–299 |
| RouteStick | xhard | 100 | 300–399 |
| VideoRepick | easy | 134 | 0–133 |
| VideoRepick | medium | 133 | 134–266 |
| VideoRepick | xhard | 133 | 267–399 |
| VideoUnmaskSwap | easy | 100 | 0–99 |
| VideoUnmaskSwap | medium | 100 | 100–199 |
| VideoUnmaskSwap | hard | 100 | 200–299 |
| VideoUnmaskSwap | xhard | 100 | 300–399 |

该表来自固定 MANIFEST 的 primary 数量；正式 plan 后还要与四份 episode map 核对。下游 `canonical_order` 以实际清单为准，不按 CLI 的任务顺序推断 global episode 区间。

## 路径、环境与资源

合并根为 `v1-store/raw-h5/4task-20260912-v2`，库为 `v1-store/datasets/4task-v2-1600ep-604f16da`，统计量为 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`。三处均为本轮全新输出。`extracted/` 保留；不覆盖旧库、旧统计量或既有 run。

AWS 8×A100-SXM4-80GB，存储为 `/dev/md0` XFS 本地 NVMe RAID，开工可用约 4.19 TB。本轮 GPU 限 4–7；合并 4 进程、verify 与打包 48 进程。CPU 阶段用 `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu` 隔离，尤其 norm_stats。所有缓存逐项放 `v1-store/cache/`，不覆盖 HOME。

GPU 0–3 的训练参考吞吐 874.0 samples/s；观察到新 epoch 低于 865.26 时暂停本轮活动阶段。此处只评价资源共处，不据训练侧相对 loss 判断模型质量。日志保存新 epoch 吞吐与阶段时间，空间按实际产物增长复核。

## 命令与配置还原

每个阶段使用独立 detached tmux，会话前缀 `v2b-`、后缀 `20260914T174147Z`；只清理本轮实际记录的完整会话名。每个会话只 source 一次 `paths.sh`，使用 `PYTHONUNBUFFERED=1`、`set -o pipefail` 和 `tee`，结束写 `EXIT_CODE=`。基础变量如下；完整阶段命令沿用计划第一部分第 5 节，只有 smoke 的 `--per-group 2` 在正式构建中省略。

```bash
source scripts/dataset/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SRC=/scratch/hongze/robomme-4task-h5-20260912-v2
RAW=$V1_STORE/raw-h5/4task-20260912-v2
LIB=$V1_STORE/datasets/4task-v2-1600ep-604f16da
TASKS=BinFill,RouteStick,VideoUnmaskSwap,VideoRepick
MANI=$LIB/meta/episode_manifest.json
INMANI=$LIB/meta/input_manifest.json
uv run --no-sync python scripts/dataset/merge_v2_h5.py plan --snapshot "$SRC/snapshot" --control "$SRC/control" --extracted "$SRC/extracted" --out "$RAW" --tasks "$TASKS" --procs 48
uv run --no-sync python scripts/dataset/merge_v2_h5.py merge --out "$RAW" --extracted "$SRC/extracted" --procs 4 --check-source-sha256
uv run --no-sync python scripts/dataset/merge_v2_h5.py verify --out "$RAW" --extracted "$SRC/extracted" --level full --procs 48
uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --tasks "$TASKS" --episodes-per-task 400 --num_shards 1 --out "$MANI"
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 4,5,6,7 --raw-dir "$RAW"
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-8x8-v1 --reader decode --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp-8x8" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp-8x8" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py report --store "$LIB/framesamp-8x8"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/xgrid_pos_check.py --store-4x4 "$LIB/framesamp" --store-8x8 "$LIB/framesamp-8x8" --source "$LIB/source" --image-spot 512
```

## 内容与结构保证

原单 episode H5 经 `Group.copy` 物理拷贝到合并 H5，仅顶层 episode 名改变；每个 dataset 的 shape、dtype 和数值保持。照片为 `(256,256,3) uint8`，其余键保留原始 dtype，包括跨任务的 float32/float64 差异。验证要求全部 dataset 数值相等，浮点 NaN 与正负零按相等，同时校验 attrs、名字集合、链接类型和布局。并非承诺 H5 容器字节完全相同。源 SHA256 独立重算回写 map，完成标记绑定 pin、map 摘要、选择规则和四个输出摘要。

## 用户决策与关联限制

用户批准“允许最小改动，实现四文件并行”，因此阶段 7 的输入 SHA256 并行是本轮唯一既有链路代码改动。库名、1600 primary、保留 extracted、GPU 4–7、两档 framesamp、暂不做 motion 和 sim 评估均沿用计划。正式短训练只在 CLI 覆盖为 20 步，默认 batch 64、worker 4、FSDP 4 和学习率配置保持，显式指定新库 norm_stats；实际完整启动命令和临时 run 名在起跑前补记。

## 归档与验收

归档四份 episode map、source pin、MERGE_DONE、input_manifest、两档 store_meta、norm_stats SHA256 和清洗阶段日志，不归档 H5、权重、features 或配置脚本拷贝。结果需逐项列出真实判定行、阶段耗时、产物字节量、来源 manifest 摘要、canonical_order 和训练可读性结果；未执行项不标通过。
