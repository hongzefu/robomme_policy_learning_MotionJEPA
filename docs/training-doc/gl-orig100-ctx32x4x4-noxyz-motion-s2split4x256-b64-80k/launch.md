# noxyz encoder 的 orig100 context VLA 启动记录

## 一、目的与当前阶段

计划正式 run 名为 `gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k`，先在同一训练 Beta 上做本标签的 64 档 20 步冒烟。当前已完成两 motion 表及真实 CPU 启动契约，正式 80k 尚未开始。只有两个 run 的 64 档均 PASS、SAVE_LOAD/100步测时判定齐全且第一阶段完成后，才直接正式起跑；任一64档OOM则补32/16各20步，只记录显存诊断，随后主会话取消两占位交用户，不自动降档训练。

## 二、版本、对象与起跑状态

代码沿实施分支 `v4-gl-orig100-1031`，起跑前提交 `commitV11.83Beta`，部署 G 到该完整 SHA、确认 clean 与包导入路径，再执行占位内工作负载；实际完整 SHA 从 `TRAIN_HEAD` 及日志读回。此 Beta 覆盖冒烟、测时与正式训练。数据建库 Beta 是 `4c05451082ea9617cf7c721d769f2be84f851471`，已在 commitV11.82 回写完成，两个锚点分别记录。

encoder 为 `aws-stage2-noxyz-1101-a/checkpoint_epoch_105.pt#model`，checkpoint SHA `6acc92af239e9b9a4c58ced4c5123bc25183188ec518b10734e917879240d00d`，原始 config SHA 已独立发布清单核验。运动表 `motion-stage2-1024-noxyz` 在共享训练库内，23,253 行、95,244,288 字节、verified；证据目录 `evidence-noxyz` 含5份闸门证据及两份在线输入。SOURCE_PIN 的实际包提交 b5 与固定4a入口来源分开记录。详情见相应建库档案。

## 三、数据、参数与资源

| 项 | 固定值 |
|---|---|
| 数据 | orig100 16任务×100集，共1600集；476857执行样本，同一episode/input manifest与norm stats |
| 记忆 | ctx，32帧×16视觉token=512；motion每窗4×256，预算384；合计896，模型memory宽度2048 |
| 训练 | global64，FSDP4，每卡16；workers8，无梯度累积；seed42 |
| 日程 | 80000步，warmup10000，decay80000，peak_lr=decay_lr=5e-5，EMA0.999 |
| 保存 | save_interval=keep_period=5000；16份，步集合5000到75000每5000一份，加79999 |
| 占位 | 65209321，节点gl1519，chaijy2/spgpu，单节点单任务，4×A40/16CPU/128G/176h |
| 输出 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_orig16_b64_80k/gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k/` |
| 规范缓存 | G/v1-store/cache；显式HF_HOME/HF_HUB_CACHE/XDG/TORCH，CUDA/JAX按run分离；HOME保持 |
| 预算 | 原176h；V7以实际剩余时间算余量，低于10%按计划自动另申估计总时长×1.2的新占位，不续跑 |

## 四、部署、诊断与完整命令

下列 `BETA` 必须换成本次已经推送的完整40位训练 Beta。主会话从有凭据的Aspen对G执行fetch与detached checkout；计算节点不安装依赖。已运行资源的只读探针确认每job实际NVML可见4张A40、Torch CUDA_COUNT=4、包路径G/src；部署后还须现场核Beta、clean与包路径。

两运行载体已做独立静态审查：step SHA `5b2c55e2c0dccae6ecc7d384e1c2d3eb616d79884a3ce5c7c9cac2b193e011b0`，controller SHA `7cc76f8d73bfa68235f65b1e657ba2de0e306f25e5cd5a990ab12d4fc45f8bc6`，`RUNTIME_REVIEW=PASS type=gl_smoke findings=0`。controller 只做64与必要32/16诊断，不启动正式训练、不取消job。NVML只在诊断短测使用单持久进程查询分配内四卡，500ms；V7与正式训练不挂它，显存采样不当作零干扰性能证据。

```bash
# Aspen detached tmux：下列BETA为训练Beta完整SHA
ssh greatlakes "bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/o16abl-smoke-controller.sh noxyz BETA"
```

外层 tmux 为 `o16abl-smoke-noxyz`，以 pipefail/PYTHONUNBUFFERED/tee 落 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-smoke-noxyz.log` 并记录 EXIT_CODE。占位内20步实际工作负载是：

```bash
G=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
srun --jobid=65209321 --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
  env CONFIG=mme_vla_suite_orig16_b64_80k ENCODER_RUN=aws-stage2-noxyz-1101-a \
  HISTORY=perceptual-framesamp-ctx-32frame-4x4-noxyz-motion-stage2-4x256-orig16.yaml \
  RUN_NAME=o16gl-smoke-noxyz-b64-1102-a GPUS=4 GLOBAL_BATCH=64 MAX_STEPS=20 TRAIN_HEAD=BETA \
  MMEVLA_STAGE2_EVIDENCE_ROOT="$G/v1-store/datasets/16task-orig100-1600ep-9e109419/evidence-noxyz" \
  FINAL_RECORD_DIR="$G/v1-store/validation/o16abl/o16gl-smoke-noxyz-b64-1102-a/final-record" \
  HF_HOME="$G/v1-store/cache/hf" HF_HUB_CACHE="$G/v1-store/cache/hf/hub" \
  bash "$G/scripts/training/gl/run_in_hold_o16.sh"
```

OOM 后的两诊断仅将上面batch及run名改32/16，仍20步、新配置、4卡、同encoder/yaml和Beta，各有新final-record与显存文件。失败时未证实的完成步数/有限性记 unknown，不把请求20步写成实际完成20步。最终记录验收与训练返回码独立检查，缺显存采样不得伪PASS。

## 五、保存恢复、测时与正式起跑

64通过后，同占位执行 `check_stage2_checkpoint.py --mode smoke`：checkpoint取上述run的 `19` 子目录，reference取该final-record内 `pre_save_reference.json`，`--online-gpu 1`，父进程 `CUDA_VISIBLE_DEVICES=0`，显式 `MMEVLA_V1_STORE=$G/v1-store OPENPI_DATA_HOME=$G/v1-store/models`；完整结果与失败原文记录。该合同要求的GPU0/1为job可见命名空间，实际UUID/PID由验收器记录。

100步测时仍用该history/global64/4卡，run名 `o16gl-smoke-noxyz-timing100-1102-a`，`MAX_STEPS=100 TRAIN_TIMING_STEPS=100`，单独final-record；从生产 StepTiming 输出取21–98步的主线程与设备trace窗口，估计80k耗时。正式起跑前重新读该JobID的squeue剩余时间；余量不足10%按K7换新占位。

两个run第一阶段完成后，正式命令如下，保持MAX_STEPS/TRAIN_TIMING_STEPS未设置：

```bash
G=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
srun --jobid=65209321 --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
  env CONFIG=mme_vla_suite_orig16_b64_80k ENCODER_RUN=aws-stage2-noxyz-1101-a \
  HISTORY=perceptual-framesamp-ctx-32frame-4x4-noxyz-motion-stage2-4x256-orig16.yaml \
  RUN_NAME=gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k GPUS=4 GLOBAL_BATCH=64 TRAIN_HEAD=BETA \
  MMEVLA_STAGE2_EVIDENCE_ROOT="$G/v1-store/datasets/16task-orig100-1600ep-9e109419/evidence-noxyz" \
  FINAL_RECORD_DIR="$G/v1-store/validation/o16abl/gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k/final-record" \
  HF_HOME="$G/v1-store/cache/hf" HF_HUB_CACHE="$G/v1-store/cache/hf/hub" \
  bash "$G/scripts/training/gl/run_in_hold_o16.sh"
```

正式 tmux `o16abl-train-noxyz`；失败重起只能按已登记attempt02及新目录，原失败产物保留。完成后核退出0、final record、精确16份步集合、79999严格恢复，再由主会话逐个释放本轮占位。公开上传与集群CPU读回验证按计划长尾执行，未读回PASS不写成已备份。

## 六、用户决策与盯盘

用户原话：「同意修订 继续做」「不需要再确认，你可以直接开始了。」「继续工作」。一次开工令覆盖全程，两个run64档通过后直接继续；64失败的裁决边界保留。每份日志关注训练步、finite、OOM、退出、最终checkpoint状态；正式性能与旧B128/60k的比较按用户决定只作探索性口径。后续实测追加result.md与records，不把当前启动计划写成已运行或已成功。

