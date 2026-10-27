# mixed8 阶段二单 token VLA：实施与起跑档案

本轮已按用户指令于美东2026-10-27 12:17:53停止，以完整70000 checkpoint为最终交付；14份checkpoint保留，70k远端回读及严格在线恢复已通过，最终全量备份核验进行中。原80k配置和run名保持历史身份，真实训练退出143。计划为根目录 `1026-vla-stage2-memory-1024-train-plan.md`，开工锚点 `1383aba0fd77b259f0b32eadb47ea7792190145e`。以下按时间保留实施及起跑记录，最新结论见`result.md`。

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

## 用户G2例外后的打包续接

用户于美东15:21明确「可以忽略 继续做」，仅接受已报告的G2数值失败；原G2.json和阈值不改，新WAIVED记录保持原失败UTF8字节和精确三SHA。G3随后单次32秒通过，65项EMA和32窗两路调用均精确一致。消费链合后235项CPU测试、记录CLI合后17项CPU测试及独立torch环境入口导入通过；两块均独立合前审查PASS。

本节随 `commitV11.48Beta` 固定后续版本，原S1 `b5d23a5`、全量token、恢复latent和G0均不变。此Beta仅续接打包验收，不是正式VLA训练Beta。运行代理在 `cc-vla-s2-pack` 以明确完整 `STAGE2_PACK_HEAD` 执行主仓 `v1-store/logs/vla-s2-20261026/run-pack.sh`；该脚本固定下列顺序，首个失败停止，不自动重试或重抽：

1. 锁定torch环境运行 `accept_stage2_g2_exception.py --failure "$VALID/G2.json" --samples "$VALID/windows.json" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --out "$VALID/G2-waiver.json" --device cuda:0`，只加载并核来源身份，不执行窗编码；打印 `G2_USER_EXCEPTION=ACCEPTED original_status=FAIL numeric_validation_complete=0`。耗时计入G2–G4原7200秒预算，G2/G3已有65秒。
2. 主环境 `pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --manifest "$D/meta/episode_manifest.json" --tokens "$D/motion-tokens-stage2" --latents "$LAT" --out "$D/motion-stage2-1024" --encoder-run-dir /scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a --raw-dir "$P/v1-store/raw-h5/16task-mixed8-200-20261014" --restore-receipt "$P/v1-store/restore/mixed8-200-vla-latents-20261021-a/restore_receipt.json" --expected-ckpt-sha256 "$CKPT_SHA" --stage2-validation "$VALID" --stage2-source-pin "$P/scripts/dataset/wan_stage2/SOURCE_PIN.json"`。
3. GPU0锁定子环境 `compare_stage2.py --store "$D/motion-stage2-1024" --latents "$LAT" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --scope full --batch 1 --device cuda:0`，独立重算65419行，上限7200秒。
4. 主环境 `pack_motion_store.py verify --store "$D/motion-stage2-1024" --resume --check-padding`，只有当前全表认证成功才接管已结束pack进程的锁。

上述路径变量与上一节相同，CPU阶段不暴露GPU、GPU阶段只用物理0；日志统一 `v1-store/logs/vla-s2-20261026/pack.log`，目标motion-stage2-1024必须全新。WAIVED证据沿用G2键进入原12文件快照并随交付发布；其他来源检查、G4/G5/G6/G7门槛不变。

## 用户G5省时裁决后的实际续接

用户明确「同意按照你说过的更省时方案 做」「encoder只抽机小部分」，随后「同意继续」。原全量encoder比较已按精确PID停止，退出130，不计为验证失败或全表PASS；packed表保留。新合同与比较器独立合前PASS、合后95＋53项CPU测试通过。真实index在主NumPy1.26.4和独立torch NumPy2.4.4均生成相同样本SHA `3d9a162d3528184a638750abbf69cdd385a21968349459964409a5baf7477936`：16任务各64，共1,024窗，demo332、exec692、补帧68。

`commitV11.51Beta` 固定本次抽样续接版本；只运行已有表上的 `compare_stage2.py --scope sampled --batch 1 --device cuda:0`（其余store/latents/ckpt/SHA同上），随后主环境 `pack_motion_store.py verify --store "$D/motion-stage2-1024" --resume --check-padding`。不再次执行pack或抽取。入口 `v1-store/logs/vla-s2-20261026/run-sampled-verify.sh`，调用方显式传完整 `STAGE2_VERIFY_HEAD`；tmux仍用已结束可复用的 `cc-vla-s2-pack`，新日志为 `sampled-verify.log`，保留旧pack.log。

CPU全量memcmp表与所有S1 token并核文件/metadata/sidecar/latent SHA，最多4个散列线程；GPU只计算1024个选中窗口。分别出 `PACKED_TOKEN_BYTES=PASS compared=65419` 与 `ENCODER_SAMPLE_BITEXACT=PASS compared=1024 total=65419`；认证明确scope=sampled，原文/SHA/覆盖随stage2_compat进入run与交付，不能宣称全表encoder数值一致。verify仍全表padding/行摘要，原pack死进程锁只用resume接管。沿用原2小时总验证上限，不扩大预算；首错停止，不自动重试。

## G4与首次G7实测（美东2026-10-26）

G4在`50805050ef0b940cb75417028a9ff24abc59a704`通过，64窗同latent与端到端最大误差均0，真实握手、4096字节响应、连续exec和reset均通过，耗时129秒。原uv启动器PID误判已修正，首次失败证据保留。

G7诊断run `cc-vla-s2-smoke-10261649-a` 从相同clean HEAD在美东16:51:08启动，20步训练417秒成功退出，step0 loss0.08711，后19步loss均值0.088020629且全部有限。W&B实际账号hongzefu、entity `hongzefu-university-of-michigan`、run `3gfhbs34`。step19保存完整；首次独立恢复65叶原始EMA逐字节一致，但动作输出最大差0.0048828125，`SAVE_LOAD=FAIL`，美东16:59:37结束。总耗时509秒计入原30分钟预算。训练未重跑，checkpoint及保存前参考保留；仅复验恢复检查，正式训练尚未启动。

## G7通过与正式训练冻结

首次恢复没有消费训练自定义的JAX编译缓存变量。原始权重无差异；输入往返及两种BF16转换CPU核验均无差异。在同一`5080505`、同一step19及原保存前参考下，仅显式设置标准`JAX_COMPILATION_CACHE_DIR`为训练缓存，复验于美东17:04:34–17:06:33完成：`SAVE_LOAD=PASS missing=0 extra=0 shape_mismatch=0 dtype_mismatch=0 ema_mismatch=0 reference_output_max_abs=0 online_infer=1`。20步没有重跑，G7累计628秒，未放宽阈值。此逐位结论限定同一编译缓存，不声称冷缓存或异机必定逐位相同。

正式Beta固定80,000步、B128、W16、8卡FSDP8、pi05_base、motion96×1024、帧路512、EMA0.999及原学习率。正式入口不设MAX_STEPS，不从短测checkpoint续训。W&B使用hongzefu、entity `hongzefu-university-of-michigan`、project `robomme-framesamp`。源码只读锁覆盖git跟踪的src/scripts实体文件，原权限清单在`v1-store/logs/vla-s2-20261026/formal-readonly-modes.json`；冻结标记保存正式Beta完整SHA，运行期状态在`-temp`开发分支维护。

完整启动命令，SHA在调用时替换为真实正式Beta，包装器强制核对clean HEAD：

```bash
STAGE2_FORMAL_HEAD=<正式Beta完整SHA> bash v1-store/logs/vla-s2-20261026/run-formal80k.sh
# 包装器调用以下正式入口；外层timeout上限108000秒
RUN_NAME=mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k TRAIN_HEAD=<正式Beta完整SHA> FINAL_RECORD_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/stage2-memory/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k/final-record bash scripts/training/run_mixed8_stage2_1024_80k.sh
# 独立CPU会话先上传根快照，随后每5000步增量上传
bash scripts/dataset/hf_export/run_mixed8_stage2_1024_ckpt_inc.sh
```

正式会话`cc-vla-s2-train`，训练日志`v1-store/logs/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k.log`；增量会话`cc-vla-s2-inc`，日志`v1-store/logs/vla-s2-20261026/vla-inc.log`。实际完整Beta、进程和时间由日志及冻结开发副本状态记录。首次有限loss才报告正式起跑；训练上限30小时，G12 GPU恢复等训练释放8卡后执行。

## 正式起跑实测

正式Beta为`5730a0cfc7a62036585362a03a8c922f14b4d94f`，已同步GitHub。美东2026-10-26 17:08:59启动，实际Python PID2780633，tmux `cc-vla-s2-train`；首个loss在17:13:20为0.0871，梯度有限。W&B正式ID `al3p8t77`，账号hongzefu。`cc-vla-s2-inc`在17:09:19启动，根17文件上传及匿名回读PASS。432份跟踪源码/脚本已只读冻结。启动后两路独立只读审计均PASS，完整现场记录见`records/startup-audit.json`；5k/60k/80k产物尚未到达验收时机。
