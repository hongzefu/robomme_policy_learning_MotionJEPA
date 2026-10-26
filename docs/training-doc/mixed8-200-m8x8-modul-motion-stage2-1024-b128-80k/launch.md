# mixed8 阶段二单 token VLA：实施与起跑档案

当前处于实施阶段，G0已通过，正式训练尚未启动。计划为根目录 `1026-vla-stage2-memory-1024-train-plan.md`，开工锚点 `1383aba0fd77b259f0b32eadb47ea7792190145e`。实际训练 Beta、完整命令和启动时间将在通过 G0–G7 后填写，不能以本档案创建时间代替起跑时间。

## 用户授权与范围

1. 用户原话：「/scratch/hongze/robomme_policy_learning_MotionJEPA/1026-vla-stage2-memory-1024-train-plan.md 开工 有问题问用户 所有subagebt必须全部astra high」。主run、建库、短测、80k训练、边训边传、MJ上传和60k交付检查已授权；备选B与benchmark仍不在范围内。
2. 用户原话：「同意推送 要和github同步」。已建立 `origin/v4-256concat1` 跟踪，后续本次实施提交通过审查后立即同步。

## 版本与代码状态

| 对象 | 锚点与状态 |
|---|---|
| policy | `v4-256concat1@1383aba`，开工时干净；功能编号接续 `commitV11.37` |
| MotionJEPA | `v12-3decoder@6c79ebb3d9c2f8a6a45dfcd012954af688c962b0`；只新增导出模块、CLI及测试，集成到新分支 `v12-3decoder-memexport` |
| 独立开工审查 | `PLAN_CONSISTENCY_REVIEW=PASS base=1383aba0fd77b259f0b32eadb47ea7792190145e findings=0 blocking=0` |
| 共享字段 | 按计划具体化并经独立审查；协调文档在 `v1-store/validation/stage2-memory/contracts-20261026.md`，生产代码不依赖它 |
| 正式训练 Beta | 尚未生成 |

S1已合入并推送 MotionJEPA `v12-3decoder-memexport@b5d23a5fab55be16e580c5b77a8f04d68b3170c2`；合前独立审查通过，合后CPU `34 passed`。主MJ检出保持原分支。G0入口已合入并推送policy `e44ee413beedbf38dbf56f1a75bc17a50eadc627`，合后CPU `39 passed`。

全部子代理使用 `gpt-6-astra`、`high`。代码代理独立worktree，运行代理不改代码；合并保留子提交，合前固定BASE/TIP审查、合后定向验证。

## 数据与权重身份

- 数据父目录：`v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/`；帧路消费既有 `framesamp-8x8`。
- latent恢复根：`v1-store/restore/mixed8-200-vla-latents-20261021-a/`；receipt记录 `status=PASS`、4,995段、128,342旧窗。新表预期只选偶数格，65,419行、每行4,096字节；这些目标尚待运行验证。
- 原始输入：exec读父目录 `source/data/*.pkl` 的 `image`；demo读 `/scratch/hongze/dataset-local-a100-16task-mixed8-200/data-raw/` 内无损H5。G0须核64窗逐窗SHA，不能用目录存在替代同源验证。
- encoder：`/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`，314,265,763字节；现场SHA256为 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`。G3须再次核验并证明实际加载 `model` EMA。
- Wan VAE：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`；历史权重SHA256 `9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0`，缓存不重建。

## 参数与环境

| 项 | 固定口径 |
|---|---|
| 硬件与盘 | 环境B，8×A100-SXM4-80GB；AWS本地NVMe RAID `/dev/md0`，XFS；开工时剩余约1.6 TiB |
| encoder | fp32、B=1、eval、no_grad、autocast关闭、双TF32关闭；4×256按源时间[0,3,5,8]拼为1×1024 |
| motion | 33帧、stride32、budget96；demo末窗至少17真实帧并重复末帧补齐，exec不补帧 |
| 帧路 | 8帧×8×8=512 token；总memory 608，modulation |
| 训练 | `mme_vla_suite_b128_80k`，batch128、FSDP8、80,000步；基础权重pi05_base、EMA0.999；5000…75000及79999共16份 |
| 短测 | 同配置20步，仅run、步数和记录路径不同；保存前EMA参考与step19严格回读 |
| GPU隔离 | 正式训练独占0–7；G12恢复排在G8后，除非另有已授权空闲机器 |

## 执行顺序与预算

G0原帧核验 → 4段抽取冒烟 → 全量抽取与G1/G2/G3 → pack → 独立全表重算 → verify → G6预算 → G4真实在线 → G7短测与恢复 → Beta正式训练。MJ上传可在自身SMOKE通过后与准备阶段并行。60k只有G12通过才可交测试。

| 阶段 | 上限 |
|---|---|
| 4段抽取冒烟 | 5分钟 |
| 全量抽取／全表独立重算 | 各2小时 |
| G2–G4 | 合计2小时 |
| 20步短测与保存回读 | 合计30分钟；超过5分钟保留诊断档案 |
| 正式训练 | 30小时 |
| MJ／VLA最终上传 | 各24小时；同一网络对象最多3次尝试、间隔60秒 |

任何正式闸门FAIL留证并交用户，不放宽阈值、不自动重跑训练或抽取。吞吐按G7实测更新；旧0.96秒/步仅为不同motion配置的估算依据。

## tmux与日志清单

下表为计划预留会话，当前均未启动。实际起跑时逐项登记时间、版本、完整命令和退出码。日志根为 `v1-store/logs/vla-s2-20261026/`。

| 会话 | 用途 | 状态 |
|---|---|---|
| cc-vla-s2-export | 冒烟／全量抽取 | 未启动 |
| cc-vla-s2-pack | 打包／全表比较／verify | 未启动 |
| cc-vla-s2-parity | 在线对拍 | 未启动 |
| cc-vla-s2-smoke | 20步短测与回读 | 未启动 |
| cc-vla-s2-train | 正式80k | 未启动 |
| cc-vla-s2-inc | 根快照及step增量上传 | 未启动 |
| cc-vla-s2-ready60 | 60k交付检查 | 未启动 |
| cc-vla-s2-ckptup | VLA最终上传／匿名全量回读 | 未启动 |
| cc-vla-s2-mjup | MJ两源上传／匿名全量回读 | 未启动 |

## 产物与归档

新token根 `motion-tokens-stage2/`、新表根 `motion-stage2-1024/` 均位于数据父目录。验证证据在 `v1-store/validation/stage2-memory/`。正式run在 `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k/`。交付bucket分别为 `HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1` 和 `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1`。

本目录records只保存无法由git还原的清洗日志与验证摘要，不复制脚本、YAML或权重。历史测速run `cc-speed8-motion-b128-20261026-a` 的step119仅在本次收尾按已批准精确路径清理；此时尚未删除。

## G0实际核验

从clean `e44ee413beedbf38dbf56f1a75bc17a50eadc627` 单次CPU运行，美东2026-10-26 13:41:06–13:41:19，耗时13秒，`EXIT_CODE=0`。原始日志 `v1-store/logs/vla-s2-20261026/g0.log`；实际判定：`RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=16 sha_mismatches=0`。64个唯一偶数格窗，demo22、exec42，16任务各4窗；无缺件、未重试。

入口为 `scripts/dataset/check_stage2_validation_inputs.py`，参数按本档案列出的manifest、恢复latent、source和demo根，输出 `v1-store/validation/stage2-memory/windows.json`。清单430170字节，SHA256 `4cb05b6b3097b22a8594576def1505957399a924d4189591980292e3bd320d30`；完整样本身份与SHA存入本档案 `records/windows.json`。该结果只证明原始样本来源，不替代G2/G4数值对拍或G5全表验证。

## 实施中的具体化

- G0检查与启动检查拆成独立不相交工作树，S2/S3保留其余写集；S4按在线、数值、恢复拆分，模型均Astra high。
- stage2成功响应新增小端uint32长度字段，4096只指载荷；默认3072分支和公共错误格式不变。`PROTOCOL_ADJUSTMENT_REVIEW=PASS`，数值判据不变；不把流式socket误述成能立即识别任意尾随字节。
- G4通过后生成便携真实demo/exec输入，S2随run快照保存、S3随清单发布；恢复不依赖原机隐藏pkl/H5。VAE与tokenizer来源身份显式记录，不另复制整VAE。
- 旧motion metadata测试的旧三layout参数集显式固定，避免自动把stride16夹具套给stage2；`TEST_ADJUSTMENT_REVIEW=PASS`，新stage2仍保留独立完整正负例。

## 全量构建锚点与确定命令

本节随 `commitV11.45Beta` 固定全量构建起点；该版本可用 `git log --format='%H %s' --grep='^commitV11.45Beta:'` 唯一恢复，运行日志记录完整SHA。它不是正式VLA训练Beta。全部代码合并后，分组CPU测试和真实最小验证见 `records/prebuild-verdicts.json`；曾出现启动测试旧夹具缺layout，已补真实stage2认证正负例并复测37项通过。独立torch环境正式补齐 `ml-dtypes==0.4.1`，隔离import正向通过，锁与SOURCE_PIN同步更新。

4段GPU冒烟于美东13:54:02–13:54:16通过：128窗、4段、`EXIT_CODE=0`，GPU已释放；它不替代G1。后111窗跨三段写入间隔约1.98秒，约0.01784秒/窗，仅短区间观测；单遍65419窗约19.45分钟的数字只是外推。

运行环境固定如下，所有缓存均位于scratch；不依赖原MJ检出的当前分支：

```bash
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
MJ_SRC=/scratch/hongze/worktrees/vla-s2-memexport-20261026
D="$P/v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f"
LAT="$P/v1-store/restore/mixed8-200-vla-latents-20261021-a/wan-latents"
VALID="$P/v1-store/validation/stage2-memory"
CKPT=/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt
CKPT_SHA=27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035
VAE_SNAPSHOT="$P/v1-store/cache/hf/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"
unset PYTHONPATH PYTHONHOME NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PROJECT_ENVIRONMENT="$P/v1-store/venvs/wan-stage2"
export HF_HOME="$P/v1-store/cache/hf" HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$P/v1-store/cache" TORCH_HOME="$P/v1-store/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$P/v1-store/cache/torchinductor-stage2"
export TRITON_CACHE_DIR="$P/v1-store/cache/triton-stage2" CUDA_CACHE_PATH="$P/v1-store/cache/cuda-stage2"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
```

`cc-vla-s2-export` 执行以下固定入口，抽取独立上限7200秒；G2/G3/G4共用两小时数值预算，逐阶段累计账目。任何失败不自动重跑。所有阶段均通过pipefail/tee记录真实退出码；源码、配置和锁在本阶段保持不变。

```bash
uv run --frozen --no-sync --project "$P/scripts/dataset/wan_stage2" python \
  "$MJ_SRC/scripts/stage2_memory_export.py" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" \
  --latents "$LAT" --manifest "$D/meta/episode_manifest.json" \
  --out "$D/motion-tokens-stage2" --grid-subsample 2 --batch 1 --device cuda:0 --check-dropout 64

uv run --frozen --no-sync --project "$P/scripts/dataset/wan_stage2" python \
  "$P/scripts/dataset/wan_stage2/validate_stage2.py" --gate G2 --samples "$VALID/windows.json" \
  --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --mj-root "$MJ_SRC" \
  --vae-snapshot "$VAE_SNAPSHOT" --out "$VALID/G2.json"

uv run --frozen --no-sync --project "$P/scripts/dataset/wan_stage2" python \
  "$P/scripts/dataset/wan_stage2/validate_stage2.py" --gate G3 --samples "$VALID/windows.json" \
  --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --mj-root "$MJ_SRC" \
  --rgb-root /scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1 --out "$VALID/G3.json"
```

打包和全表比较在前述阶段完成后，于 `cc-vla-s2-pack` 串行执行；实际命令和闸门记录随后补入。当前训练yaml的SHA仍保留占位，须G3与新库验证后填入，不能提前启动VLA。
