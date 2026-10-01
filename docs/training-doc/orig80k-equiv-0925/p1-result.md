# 原版80k P1入口/记录验收结果与外层退出证据限制

本文件归档实际P1入口与记录验收，以及用户批准的最外退出证据限制。所有原始记录、日志和无损包保持原字节；正式导入使用189项白名单并逐项核对SHA/bytes，见[导入回执](records/post-eq100-import-receipt.json)。后续100步结果见[独立结果档](train100-result.md)。

## 1. 结论与指标速览

**full/count四侧P1的入口、finite、分段、配置和来源记录验收通过；最外wrapper进程的真实退出码未独立留存。** A各完成tentative2与main2，B各完成main2；内层状态及原始标量/状态证据保留。日志中的退出0不能补成最外进程独立wait回执。当前没有真实训练失败证据；P1也没有执行A/B轨迹相等judge或真实checkpoint保存/恢复。

| 侧 | 真实执行段 | metrics_all / main行数 | state_digests的step顺序 | 每行叶数 | 入口/记录验收 |
|---|---|---:|---|---|---|
| full-A | tentative2 + main2 | 4 / 2 | `[1,1]` | `[201,201]` | PASS |
| count-A | tentative2 + main2 | 4 / 2 | `[1,1]` | `[201,201]` | PASS |
| full-B | main2 | 2 / 2 | `[1]` | `[201]` | PASS |
| count-B | main2 | 2 / 2 | `[1]` | `[201]` | PASS |

根代理四侧最终报告SHA256为 `eda0e63e5a81e041474a45dc207c5987a1cc0a2ff1a7fa5789db04cc6d415dcf`；独立四侧汇总为 `92e36eaa3cbf1565b37737aad9b7c7ae30f91cab6b42833ae18f6740cd9670b7`。独立逐侧验收器自身真实返回码均为0，说明其记录检查通过；它没有追取已结束P1最外wrapper的独立退出码。两份历史报告及原始证据不改写，其PASS按此限定范围解读。

## 2. 版本与代码状态

主仓实际 `TRAIN_HEAD=00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，提交题为 `commitV11.13Beta: 锚定原版双库训练对拍并归档正式输入验收`。A固定独立上游 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，B与共同harness来自本次TRAIN_HEAD。四侧起跑/收尾的HEAD、clean状态、入口及已加载项目模块来源均验收通过；A收尾记录49个项目模块，B为52个。

A解释器为主仓 `v1-store/worktrees/orig-ecf086c/.venv/bin/python`，B使用主仓独立uv `.venv`。起跑前及验收时核对相同Python 3.11.15构建和208个分发包，包版本清单SHA为 `a31fe021dbb20937aa93ce68d59a7d2f7d38d92a7e9514256ca2da617f5274dc`。共同harness `entry_equiv.py` 的实际SHA为 `92b17055af8f185175aabfc65aee1e94287d6f188a68bc7c6f1dcf5c98f99ce6`。

`sys.prefix`没有单独落入训练JSON；证据由固定harness入口强制断言、实际driver/outer命令、所属项目模块及验收器对两侧独立解释器的只读查询共同支撑。文档不补造未记录的字段。P1期间tracked源码和两侧venv保持冻结；本结果草案与打包仅写ignored路径。

## 3. 启动与配置还原

实际标签为 `20260926T195426Z`。记录根是主仓 `v1-store/bench/orig80k-p1-20260926T195426Z`，四侧子目录为 `full/a`、`count/a`、`full/b`、`count/b`；checkpoint工作目录为同根 `checkpoints/<side>/mme_vla_suite/<run>`。A的 `execution_role=upstream`，两份B为 `solo`，已由最终报告绑定的 `harness_meta.json`核实。

| 侧 | 实际run_name | 完整tmux会话 | wrapper / body / harness PID |
|---|---|---|---|
| full-A | `p1-orig80k-full-a-20260926T195426Z` | `orig80k-p1-full-a-20260926T195426Z` | 582616 / 582619 / 582619 |
| count-A | `p1-orig80k-count-a-20260926T195426Z` | `orig80k-p1-count-a-20260926T195426Z` | 582631 / 582633 / 582633 |
| full-B | `p1-orig80k-full-b-20260926T195426Z` | `orig80k-p1-full-b-20260926T195426Z` | 679414 / 679416 / 679425 |
| count-B | `p1-orig80k-count-b-20260926T195426Z` | `orig80k-p1-count-b-20260926T195426Z` | 686819 / 686821 / 686830 |

可用下列只读命令还原所跑接口与启动约定；实际展开值还由每侧 `harness_meta.argv_tail`、原日志 `COMMAND=`和命令文件SHA绑定：

```bash
git show 00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8:docs/training-doc/orig80k-equiv-0925/p1-launch.md
git show 00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8:scripts/training/tests/run_entry_equiv.sh
git show 00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8:scripts/training/tests/entry_equiv.py
git show ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b:scripts/train.py
```

以下为四份已执行的 `COMMAND`正文，顺序为full-A、count-A、full-B、count-B；只去除了日志输出末尾的一个分隔空格，参数完全一致。outer日志写出的终态字段、原始命令文件路径/SHA保留在各自日志与引用清单中；最外进程的独立退出观测缺失。

```bash
env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR -u TRAIN_RECORD_DIR -u TRAIN_FINAL_RECORD_DIR -u TRAIN_TIMING_STEPS JAX_ENABLE_X64=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 WORKTREE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c A_PYTHON=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c/.venv/bin/python UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv HEAD_A=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b HEAD_B=00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8 STEPS=2 BATCH=64 FSDP=4 SAVE_INTERVAL=50 HISTORY_YAML=perceptual-framesamp-modul.yaml MODE=upstream EXPECT_TENTATIVE_A=2 EXPECT_STATE_STEPS=1 ANCHOR_SHA256= RECORD_ROOT=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/full RECORD_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/full/a RECORD_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/full/b EXP_A=p1-orig80k-full-a-20260926T195426Z EXP_B=p1-orig80k-full-b-20260926T195426Z DATA_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep/source DATA_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep/framesamp ASSETS_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite FRAMESAMP_SOURCE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep/source FRAMESAMP_MANIFEST=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep/meta/episode_manifest.json CHECKPOINT_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/checkpoints/full-a CHECKPOINT_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/checkpoints/full-b JAX_CACHE_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/orig80k-p1-20260926T195426Z/full-a MMEVLA_JAX_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/orig80k-p1-20260926T195426Z/full-b GPUS=0\,1\,2\,3 EXECUTION_ROLE=upstream bash /scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/tests/run_entry_equiv.sh run-a
env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR -u TRAIN_RECORD_DIR -u TRAIN_FINAL_RECORD_DIR -u TRAIN_TIMING_STEPS JAX_ENABLE_X64=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 WORKTREE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c A_PYTHON=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c/.venv/bin/python UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv HEAD_A=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b HEAD_B=00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8 STEPS=2 BATCH=64 FSDP=4 SAVE_INTERVAL=50 HISTORY_YAML=perceptual-framesamp-modul.yaml MODE=upstream EXPECT_TENTATIVE_A=2 EXPECT_STATE_STEPS=1 ANCHOR_SHA256= RECORD_ROOT=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/count RECORD_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/count/a RECORD_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/count/b EXP_A=p1-orig80k-count-a-20260926T195426Z EXP_B=p1-orig80k-count-b-20260926T195426Z DATA_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep/source DATA_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep/framesamp ASSETS_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep FRAMESAMP_SOURCE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep/source FRAMESAMP_MANIFEST=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep/meta/episode_manifest.json CHECKPOINT_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/checkpoints/count-a CHECKPOINT_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/checkpoints/count-b JAX_CACHE_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/orig80k-p1-20260926T195426Z/count-a MMEVLA_JAX_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/orig80k-p1-20260926T195426Z/count-b GPUS=4\,5\,6\,7 EXECUTION_ROLE=upstream bash /scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/tests/run_entry_equiv.sh run-a
env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR -u TRAIN_RECORD_DIR -u TRAIN_FINAL_RECORD_DIR -u TRAIN_TIMING_STEPS JAX_ENABLE_X64=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 WORKTREE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c A_PYTHON=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c/.venv/bin/python UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv HEAD_A=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b HEAD_B=00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8 STEPS=2 BATCH=64 FSDP=4 SAVE_INTERVAL=50 HISTORY_YAML=perceptual-framesamp-modul.yaml MODE=upstream EXPECT_TENTATIVE_A=2 EXPECT_STATE_STEPS=1 ANCHOR_SHA256= RECORD_ROOT=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/full RECORD_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/full/a RECORD_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/full/b EXP_A=p1-orig80k-full-a-20260926T195426Z EXP_B=p1-orig80k-full-b-20260926T195426Z DATA_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep/source DATA_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep/framesamp ASSETS_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite FRAMESAMP_SOURCE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep/source FRAMESAMP_MANIFEST=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep/meta/episode_manifest.json CHECKPOINT_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/checkpoints/full-a CHECKPOINT_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/checkpoints/full-b JAX_CACHE_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/orig80k-p1-20260926T195426Z/full-a MMEVLA_JAX_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/orig80k-p1-20260926T195426Z/full-b GPUS=0\,1\,2\,3 EXECUTION_ROLE=solo bash /scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/tests/run_entry_equiv.sh run-b
env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR -u TRAIN_RECORD_DIR -u TRAIN_FINAL_RECORD_DIR -u TRAIN_TIMING_STEPS JAX_ENABLE_X64=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 WORKTREE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c A_PYTHON=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c/.venv/bin/python UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv HEAD_A=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b HEAD_B=00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8 STEPS=2 BATCH=64 FSDP=4 SAVE_INTERVAL=50 HISTORY_YAML=perceptual-framesamp-modul.yaml MODE=upstream EXPECT_TENTATIVE_A=2 EXPECT_STATE_STEPS=1 ANCHOR_SHA256= RECORD_ROOT=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/count RECORD_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/count/a RECORD_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/count/b EXP_A=p1-orig80k-count-a-20260926T195426Z EXP_B=p1-orig80k-count-b-20260926T195426Z DATA_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep/source DATA_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep/framesamp ASSETS_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep FRAMESAMP_SOURCE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep/source FRAMESAMP_MANIFEST=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep/meta/episode_manifest.json CHECKPOINT_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/checkpoints/count-a CHECKPOINT_B=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/checkpoints/count-b JAX_CACHE_A=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/orig80k-p1-20260926T195426Z/count-a MMEVLA_JAX_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/orig80k-p1-20260926T195426Z/count-b GPUS=4\,5\,6\,7 EXECUTION_ROLE=solo bash /scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/tests/run_entry_equiv.sh run-b
```

A用官方 `scripts/train.py::__main__` 双段，B用当前 `scripts/training/train.py` 单段。`_CheckpointOwnership`首轮拒绝复用已有输出；A第二次初始化只允许重建本进程刚创建且inode归属未变的tentative目录。实际A初始化各2次、effective_overwrite为False/True；B各1次且False。

## 4. 数据集与划分口径

full为公开16任务各100集，1600集、768897帧、476857执行样本；counting为同一公开集合的BinFill、PickXtimes、StopCube、SwingXtimes各100集，400集、189035帧及执行样本。P1未重建、重划分或裁剪数据。两库路径分别为主仓 `v1-store/datasets/16task-pub-1600ep`、`4task-counting-pub-400ep`。

A消费同库source，经上游Dataset/transforms与原生collate；B消费4×4 packed帧采样库，经当前FrameSampDataset/transforms及共享内存collate。两侧均交由JAX四卡路径训练。full使用原版norm `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`，counting使用自算norm `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`，共享tokenizer与pi05_base初始化资产。

CPU INPUT的真实版本始终是 `3a1582db39c723c735e04752e5027bfe40ecc3e1`。P1起跑前的 `p1-preflight-20260926T195426Z.json`记录其至00bd的提交差异、113项输入文件引用复核、相同Python/依赖与完整初始化/tokenizer资产校验；两组原INPUT PASS由此按已批准边界沿用，原HEAD/provenance没有重写。P1不声称在00bd重新执行过全量CPU输入采集。详见[原INPUT结果](result.md)。

## 5. 关键超参与验证覆盖项

| 项目 | 本次实际值及落点 |
|---|---|
| 配置条目 | 原版 `mme_vla_suite` |
| 短验证覆盖 | 入口argv的steps2、log_interval1、save_interval50；未修改全局默认 |
| batch / FSDP / workers / seed | 64 / 4 / 4 / 42 |
| history | `perceptual-framesamp-modul.yaml`，budget512、4×4、每帧16 token，motion关闭 |
| history SHA | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| 学习率 | 原warmup10000、peak/decay均5e-5、decay_steps100000 |
| 优化器 / EMA / keep | 原AdamW、clip1.0 / 0.999 / 10000 |
| W&B | 两侧均按批准传 `--no-wandb-enabled`，本地标量/状态仍完整记录 |
| 确定性档 | `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` |
| x64及线程 | `JAX_ENABLE_X64=0`，实际x64禁用；OMP/OPENBLAS均1 |

GPU命令清除 `JAX_PLATFORMS/JAX_PLATFORM_NAME`和Python外部路径；A/B环境独立且不进行依赖同步。JAX/CUDA/W&B缓存按各自新run隔离在v1-store内，`jax_cache.json`验实转接结果，不覆盖HOME。其余有效计算环境、实际GPU UUID、完整配置和源/资产摘要由起止fingerprint记录并核对。

## 6. 硬件、耗时与观测成本

环境为AWS单机8×A100-SXM4-80GB，scratch为 `/dev/md0` XFS本地NVMe。full使用GPU0–3，counting为4–7；两A并行，全部A结束后B-full单跑，再B-count单跑。起跑前scratch可用851098050560 B，八卡空闲快照及GPU UUID保存在本轮preflight，属于当时现场，不是后续资源保证。

下表时间严格来自wrapper日志的 `START_UTC/END_UTC`，并与根/独立报告一致；是包括启动、编译、数据读取、两类状态哈希及收尾的日志记录窗口。END_UTC不是独立wait观测的最外进程精确退出时刻。

| 侧 | START_UTC（日志） | END_UTC（日志） | 记录窗口耗时 |
|---|---|---|---:|
| full-A | 2026-09-26T19:56:57Z | 2026-09-26T20:31:18Z | 2061秒（34分21秒） |
| count-A | 2026-09-26T19:56:57Z | 2026-09-26T20:30:32Z | 2015秒（33分35秒） |
| full-B | 2026-09-26T20:32:04Z | 2026-09-26T20:40:08Z | 484秒（8分04秒） |
| count-B | 2026-09-26T20:40:40Z | 2026-09-26T20:48:32Z | 472秒（7分52秒） |

整个P1日志记录窗口为3095秒。A双段与B单段的工作量、调度和编译/缓存条件不同，不能据此推算loader速度比、GPU瓶颈或正式训练ETA。

[`p1-verifier-evidence-20260926.json`](records/p1/background/p1-verifier-evidence-20260926.json)按初始化shape/dtype与Optax模板推得完整state逻辑载荷50092740044 B（约50.09 GB/46.65 GiB）、预计201叶。它未扣数组别名或已有host缓存，不能当作RSS峰值、实际DMA字节或checkpoint体积。harness每叶读取host值、检有限性，再顺序执行两次tobytes/哈希；JAX `ArrayImpl._value`可能把host数组缓存于仍存活的叶对象，这些机制说明额外成本但不构成吞吐实测。

[20:11 UTC附近的一组py-spy瞬时取栈](records/p1/background/p1-stack-observation-20260926T2011.json)显示full-A当时在状态有限值检查、count-A在device_get组装host值。取栈为nonblocking、非连续profile且未采locals，有瞬态一致性限制；不能推断全程占比、剩余时间或完成时刻。本结果未复用模板比例或四位小数日志作任何性能结论。

## 7. 训练过程行为与记录完整性

A的metrics_all按序为0、1、0、1；metrics_tentative保留前2行，metrics保留main后2行。B只有0、1两行。所有记录的五标量 `loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm`均有限，dec/hex自洽。

`state_digests.step`是调用摘要器时的loop编号：A保留两行 `[1,1]`，B保留 `[1]`，没有去重。每行均201叶、finite_leaves=201、nonfinite_keys为空，覆盖params、EMA、opt_state和step四棵树。strict摘要纳入key/dtype/shape/bytes，另保留g0口径、全局、keyset和treedef摘要；验收器重算全局与keyset，treedef本身只有摘要，未据此重建原树或恢复真实数组。

四侧日志均有唯一 `ENTRY_FINITE=PASS`、`ENTRY_RUN=OK`和相应 `SEGMENTS`行；五项退出字段的原文均各一次为0，但须按其实际观测层区分：

| 证据 | 原始记录 | 能支持的结论 |
|---|---|---|
| `run_status.json`、`ENTRY_RUN/ENTRY_FINITE/SEGMENTS` | 完整且通过 | 入口、分段及标量/状态有限性记录通过 |
| `COMMAND_EXIT` | 四侧各唯一0 | 已捕获的主体子命令返回状态 |
| `TEE_EXIT` | 四侧各唯一0 | 主体tee管道的已捕获状态 |
| `FOOTER_PRINTF_EXIT`、`FOOTER_TEE_EXIT` | 四侧各唯一0 | 最后直接append之前的footer管道状态 |
| `EXIT_CODE` | 四侧日志各写出0 | 写入日志的综合值；不能单独证明写入动作之后最外进程实退0 |
| 最外wrapper的独立wait或`pane_dead_status` | 未留存；P1四侧已结束 | 不能事后追补或按日志0补造 |

A每侧两条201叶状态完成打印，B各一条，与JSON行数一致。**这些状态摘要打印没有时间戳**；本文不把相邻tqdm/logger时间回填为摘要的精确完成时间。

## 8. 训练后验收与尚未覆盖的验证

P1验收由ignored只读检查器 `check_p1_records_20260926.py`执行，脚本SHA `ce469d04fc29e81fb383a1b514f1602594f88fae08f77a780e3c77f6abf4e728`。根代理与独立验收均核对实际记录，独立逐侧验收命令真实返回0；其此前7项小夹具自检仅为量具覆盖。入口/记录通过来自结束后的真实四侧记录，最外进程独立退出码的缺口另按第10节处理。

本阶段替换save_state为完整状态摘要器，没有真实权重写入，也未调用真实恢复完成器或策略评估。P1没有执行 `ENTRY_EQ` 轨迹judge；日志用于观察的四位小数不能证明相等。上游/当前100步、同入口单跑/并跑必须另按完整hex、状态、配置与来源判据实跑。后续两库各一对真保存20步包装off/on、300步perf及正式80k也不由P1自动放行。后续同HEAD100步六run、两组upstream及两组same-entry已另行完成并通过独立验收，结果见[100步结果正文草案](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-eq100-20260926T210415Z/archive-staging/train100-result-draft-20260926.md)，不并入本P1结论。

## 9. 用户决定记录与本轮边界

用户原始目标为「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」，并要求「开始实现该计划 有问题立刻问用户 不要自己决策」。两库范围、原版训练参数、各自norm和分阶段闸门沿用。

用户明确「允许仅对拍关闭 W&B」，因此P1/100步两侧关闭在线W&B、完整本地证据保留；真保存20步、300步perf和80k仍保持W&B开启。CPU阶段「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」与「允许这四键缺失与 None 等价」只在已批准输入比较范围生效，训练轨迹的逐位要求未降低。

用户此前原话为「继续工作 一路做到起泡前 有问题问用户」。针对新发现的最外退出证据缺口，用户随后明确批准「补记P1限制，补独立退出取证后继续（推荐）」。按此保留已完成P1的原输入/标量/状态/来源验收，补记不能追取最外退出码的限制，并为进行中及后续任务补充独立退出取证后继续推进。训练配置、逐位判据、源码与环境不变，本轮仍推进至**正式80k起跑前**，遇到问题向用户询问。用户随后于22:11:03另批准「同样补记限制，沿用INPUT结果（推荐）」；这是旧CPU输入最外退出缺项的独立沿用决定，不倒填成P1起跑前已有该项外层退出批准。

## 10. 计划外观察、处置及日志清洗

A末步摘要期间曾观察到较高host内存和较长等待，主代理以只读源码/模板和一次非阻塞栈取样补充定位，未据此改batch、workers、精度或有限值判据，也没有提前宣称摘要完成。入口和记录结果由结束后的实际日志/JSON验收确定；中途PENDING或模板估算不改写为当时已PASS。

后续[纯shell故障注入结果](records/p1/exit-evidence/perf-wrapper/results.tsv)共16例，15例符合预期、1例标记DEFECT，文件SHA256为 `7c5ac6e50803708cd36020552e924c0993cfafcfdf00bd285b008491eb72f7ea`。`append_after`模拟最终append完整写出 `WRAPPER_EXIT_CODE=0`后返回37，wrapper实际返回1，但旧的仅日志checker返回0、仍看到一条成功终态。这证明日志不能独立证明写入完成后的最外进程退出状态；**没有真实训练失败证据**，也不把模拟的1或37写成任何P1任务的实际退出码。

P1四侧已经结束，未保留最外wait或已结束pane的真实退出状态，现不能追取。按用户决定，历史报告、原始日志、JSON与tar包全部保留原字节；只在结果说明与引用清单中补记证据限制，不补造独立wait、不重新标注实际训练失败。

根代理已于2026-09-26 **21:18:27 UTC**对当前两A100各自窗口设置 `remain-on-exit on`：[full-A配置回执](records/train100/background/eq100-full-a-exit-retention-20260926T210415Z.json)绑定窗口`@552`/pane`%552`，[count-A配置回执](records/train100/background/eq100-count-a-exit-retention-20260926T210415Z.json)绑定`@553`/`%553`，两次设置命令返回0。配置快照当时均为pane仍存活、退出字段为空，**这些设置成功记录不是训练退出结论**。两A100结束后已另取得与身份绑定的真实pane退出0和外部capture实际0；后续S1/S3/S2及四judge均通过冻结f8门闩和独立退出链验收。配置当时的alive快照仍保留原样，不能回写成当时已完成。该操作仅改变本窗口结束后的保留方式，不改训练参数、逐位判据、代码或venv。

P1原日志含独立 `tqdm_logging.py` 的 `Progress on:`行：A每份8条，B每份4条。归档只删除这些明确进度行并将CRLF/CR转LF；所有其余行逐行保留，包括shape、配置、HLO百分号文本、Step标量、分段、状态打印、失败/警告和退出字段。另在git-ready副本各删除 `COMMAND=`末尾一个未转义ASCII分隔空格，解析argv相同；其余关键行保持原字节，原日志与先前stage/checks不变。

| 侧 | 原日志B | 清洗stage B | git-ready B | 删除tqdm行 |
|---|---:|---:|---:|---:|
| full-A | 86803 | 85783 | 85782 | 8 |
| count-A | 84947 | 83927 | 83926 | 8 |
| full-B | 44508 | 43998 | 43997 | 4 |
| count-B | 44572 | 44062 | 44061 | 4 |

所有tar只规范化成员顺序、uid/gid、mode/mtime和gzip头；JSON/JSONL成员字节不变，没有复制.sh/.yaml/.command或权重。P1的A双段原始state没有被去重。

## 11. 当前结论与下一步

P1四侧入口、finite、分段、配置与来源记录验收已通过；最外实际退出码缺少独立证据，按用户批准补记后继续。后续两A100、S1/S3/S2及四份正式judge已经完成并通过，独立原生退出和capture链另在100步档案记录；这些新证据不补造历史P1缺失的外层回执。接下来仍须完成已批准的两库包装off/on20步真保存对照、300步perf、真实保存恢复及有明确余量来源的磁盘预算，再整理正式80k起跑前资料。

CPU INPUT保持原3a版本，P1保持00bd/上游ecf版本；任何后续沿用均须保留原证据版本并核对影响范围。100步S1/S3/S2的同HEAD及完整环境指纹要求继续有效，不因结果归档而改写或绕过。P1和100步运行及独立验收期间保持源码与venv冻结；主代理于2026-09-26 23:13:22 UTC解除冻结开始下一轮代码/文档整合，不改变旧00bd/ecf运行锚点。本归档更新不改变原始运行记录，也不启动新训练或清理临时产物。

## 12. 归档文件清单与最终引用

四侧共50份原始JSON/JSONL、526374 B，分别保留在P1根的 `full/a`、`count/a`、`full/b`、`count/b`。四个无损包及检查链已按白名单复制到正式档案，目标与来源字节完全一致。

| 侧 | 原文件数 / payload B | 无损包B | 无损包SHA256 |
|---|---:|---:|---|
| full-A | 13 / 162864 | 43639 | `c17af1396199a42cb326e3a290103ab0ca59b1e3576d15be93b29666405dd9ba` |
| count-A | 13 / 163057 | 43565 | `067b62d5482295aea0d4a93d1dfb3d5c777e83b95cf9ec54ec27ac8c3c545374` |
| full-B | 12 / 100125 | 26792 | `10852979af7d6dd5c1c60e4a0e220938fab812024e9baa380c7710f3f83ad616` |
| count-B | 12 / 100328 | 26747 | `27cd6c9d03fc51d29b2c63449109ff4d00d0e28e646ec1891bd5de1e11171a3b` |

可直接查看[full-A检查](records/p1/full-a.archive_checks.json)、[count-A检查](records/p1/count-a.archive_checks.json)、[full-B检查](records/p1/full-b.archive_checks.json)、[count-B检查](records/p1/count-b.archive_checks.json)，以及对应[full-A](logs/p1/full-a.summary.log)、[count-A](logs/p1/count-a.summary.log)、[full-B](logs/p1/full-b.summary.log)、[count-B](logs/p1/count-b.summary.log) Git日志。各自final-log-checks同时绑定原日志、stage检查和Git副本。所有包逐成员解码SHA/bytes、gzip CRC和原记录不变检查均通过。

[p1-archive-references.json](records/p1/p1-archive-references.json)完整列出四侧原文件、原日志、包、清洗/Git日志及检查文件的路径/bytes/SHA，并绑定根四侧报告、独立汇总及其逐侧报告、preflight/资产/机制背景；新增纯shell反例与两份100步窗口保留配置回执。清单的PASS仅表示引用和原字节校验通过，不表示补齐P1最外独立退出证据。正式记录位于 `records/p1/`，Git日志位于 `logs/p1/`，未覆盖既有CPU INPUT同名包；展示链接按正式目标转换，原清单内的实际来源路径不改。本引用清单不打包仍可能更新的控制器/资源JSONL，也不复制源码、运行命令载体、venv或缓存。
