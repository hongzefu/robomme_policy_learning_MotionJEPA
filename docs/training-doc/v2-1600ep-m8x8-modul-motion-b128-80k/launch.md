# modulation 8×8 motion 80k 正式训练

**结束状态已回写**：80k于2026-09-19 03:24:35 UTC（9月18日23:24:35 EDT）完成，EXIT_CODE=0，最终79999 checkpoint真实加载通过；详见[result.md](result.md)。以下保留原起跑记录与配置，起跑Beta不改写。

**已于2026-09-18 05:47:05 UTC正式起跑。** 训练锚点为 `2f10473161b760f16d9240d3c2959ff326cde66b`（commitV10.2Beta），30项preflight全部通过，实际8卡/b128/fsdp8/w16。训练PID545524，tmux为mv2-prod；[WandB运行](https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/uzv8avpq)已同步。实际命令与起跑状态见 [launch.actual.json](records/launch.actual.json)，过程见 [result.md](result.md)。本页以下保留Beta固定的配置。

本页固定用户已确认的正式run `v2-1600ep-m8x8-modul-motion-b128-80k`。V8、V-online和八卡smoke全部验收已通过，此页为Beta起跑前档案；实际启动版本、时刻和进程随启动记录固定。用户原话：「开始实施 有问题尽早问用户」「采用 v2-1600ep-m8x8-modul-motion-b128-80k」。

实施依据为 [0916计划](../../../0916-motion-modul-8x8-plan.md)。前置证据见 [建库结果](../../dataset-build-doc/4task-v2-1600ep-motion-demopad17/result.md)、[V8逐位对拍](../mv2-aa/result.md)、[V-online三档](../mv2-online/result.md) 与 [八卡smoke结果](../smoke-m8x8-modul-motion-20260918T050716Z/result.md)。

## 版本与代码状态

生产改动为 `bec052e5ac76e338cfbe283bf7c3c434f05112b7`，验证工具为 `c0be292c40c4a971161e7233540f4dc5c0c1c7cc`。关闭态BASE为 `2126b1b1c436166662ff89a629985ecd4524fc42`；V1/V2/V6/V7前后逐位验证均通过。位置表验证及V5预训练兼容初始化修补为 `f6915f2a09443d48c1bb57e9b5f83e95400704fd`；终止验证参考修补为 `782696c231aace21c20200638ae302a5a1c7f277`，生产eval行为保持原样。

正式训练必须从起跑前Beta提交后的clean HEAD启动。确切TRAIN_HEAD由启动命令的完整提交参数、preflight的CHECK_REPO_HEAD和 `records/launch.actual.json` 固定；本页不预填自身提交SHA。源码和参数可用 `git show <TRAIN_HEAD>:src/mme_vla_suite/training/config.py`、`git show <TRAIN_HEAD>:src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml` 还原。依赖uv.lock SHA固定为 `02cbc3ba67a9024f8afb9e31f60661c9abdcc3eb680ae80a8ee464c639327221`。

## 配置与启动覆盖

新具名配置 `mme_vla_suite_b128_80k` 明确承载训练步数80000、decay_steps80000、fsdp_devices8和num_workers16四项。batch128、seed42、EMA0.999、AdamW裁剪1.0、warmup5000、peak_lr=decay_lr=5e-5、保存间隔5000、keep_period5000、pi05_base及冻结过滤保持60k基线口径；80000步学习率与60k条目逐步相同的验证已通过。正式模型为完整生产VLM及动作专家，V5中的gemma_150m替身仅用于验证。

runner只覆盖exp_name、assets_base_dir、新库norm assets绝对路径、asset_id=robomme、checkpoint_base_dir、新库framesamp-8x8路径和history YAML。YAML开启motion、预算160；8帧×64 token构成512位帧记忆，motion加入后记忆总长672、宽1024，主干prefix仍576位。demo至少17个真实帧、repeat_last补满33，exec恒需完整33帧，stride16，超预算直接报错。

训练入口命令为 `bash v1-store/logs/mv2-prod-runner.sh <TRAIN_HEAD> 6c9f165f928786e456b98c2448a60d6be2c2fd3e935882d9fc08e439964f1d20 d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268`，tmux全名 `mv2-prod`。使用现有uv环境的 `uv run --no-sync`，禁止依赖同步改变主环境。WandB使用已配置的本机凭据文件，项目robomme-framesamp，凭据值不进入日志。

## 数据与产物

四任务BinFill、RouteStick、VideoRepick、VideoUnmaskSwap各400集，共1600集、605611个执行样本。实体数据根 `v1-store/datasets/4task-v2-1600ep-604f16da`，帧库framesamp-8x8有1192918行，motion库有71316行（demo35913、exec35403），其中1600个demo补帧窗。VAE保持fp32，encoder依checkpoint使用bf16 autocast，motion输出保存float32。encoder为 `wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt`，ckpt SHA为 `0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca`。

原始输入实体目录为 `v1-store/raw-h5/4task-20260912-v2`，四个H5已完成完整SHA重锚。构建顺序按用户决定采用Wan→encode→oracle重算/汇总→pack/verify→compare，清单、构建和oracle的raw_dir三方强校验全部通过。

| 对象 | SHA256 |
|---|---|
| 清单文件 | df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482 |
| canonical manifest | 4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918 |
| framesamp-8x8 metadata | f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a |
| motion metadata | d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268 |
| motion整表 | 03fb46e150dd9015f264f9bd8f08b35883d84da40e9e8080147d2af5fef9b6c5 |
| norm_stats | 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 |
| history YAML | 6c9f165f928786e456b98c2448a60d6be2c2fd3e935882d9fc08e439964f1d20 |

norm_stats来自 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`，沿用原件，不因开启motion重算。表路径来自YAML，`MMEVLA_MOTION_STORE`保持unset。checkpoint根为 `v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-motion-b128-80k`，预期5000到75000每5000一步及最终79999。日志为 `v1-store/logs/v2-1600ep-m8x8-modul-motion-b128-80k.log`，指标与trace位于同名 `v1-store/bench/` 子目录，JAX缓存为同名 `v1-store/cache/jax/` 子目录。拒绝覆盖既有run与日志。

## 起跑条件与监控

环境B，8×A100-SXM4-80GB，AWS本地NVMe RAID `/dev/md0` XFS。启动前重新确认scratch可用≥400G、共享内存余量、GPU0–7显存占用之和为0，并核本轮前置tmux均已退出。独立输出GPU_IDLE判定；runner主体使用启用 `set -euo pipefail` 的子shell，任何起跑检查非零即止，preflight和train.py消费同一个TRAIN_ARGS数组。preflight必须30项通过。

全程GPU采样为15秒；首段另起 `mv2-dense`，500ms采集八卡timestamp/index/utilization/memory。`TRAIN_TIMING_STEPS=300`启用前300步主线程分段计时与JAX设备trace，不增加逐步强制同步，trace结束时仅等待一次。性能窗口为第100–299步：报告平均步时、吞吐、主线程取batch等待比例、设备kernel分类、GPU利用率均值与0%比例，并按慢步/其他步分层。异步dispatch、数据等待和设备计算可能重叠，不能相加成互斥百分比；计时与采样范围须完全覆盖八卡。

八卡smoke暴露并闭合了查看器JSON事件上限问题：完整原始XPlane成功恢复20步和全部GPU事件。正式入口请求采集全部前300步，`StepTiming`仅在导出时将查看器上限至少提高到一亿并恢复环境，汇总器检查上限与步骤/GPU覆盖、仅计物理stream。**实际正式采集另遇设备层覆盖缺口：主线程300步完整，GPU原始事件仅到第25步附近，详见result.md；不能把查看器上限修复等同于300步设备采集完整。** 指标与原始XPlane都保留在记录目录。本轮密集采样采用原定1800秒自然到时路径，未提前发信号，采样器真实退出码单独留存。

正式和密集采样入口的精确源码见 [prod-runner.sh](records/prod-runner.sh) 与 [dense-runner.sh](records/dense-runner.sh)。独立开发副本准备记录见 [devcopy.summary.log](records/devcopy.summary.log)。

正式稳定起步后保存源码SHA并锁定主副本src/scripts/packages只读。开发和起跑后文档回写使用 `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp` 独立克隆及独立uv环境；共享v1-store仅沿用已批准的指向主副本实体目录的symlink。克隆继承主副本既有GitHub地址和身份，不修改主副本的upstream或凭据。主副本在训练期间保持TRAIN_HEAD及clean状态，开发副本仅提交本轮档案。

开发副本已于2026-09-18 04:47:14→04:47:19 UTC从clean `782696c231aace21c20200638ae302a5a1c7f277` 预先建立，tmux为mv2-devcopy，EXIT_CODE=0。`uv sync --offline --frozen` 从现有缓存安装208包，独立venv及mme_vla_suite/openpi/openpi_client/JAX导入路径全部位于temp。正式起跑前必须将该副本快进到最终TRAIN_HEAD，准确结果随启动记录固定。

## 用户决定与结论边界

用户确认「采用建议顺序，保留三方强校验」「一并修补，在修补后的同一驱动上取前后基线」「纳入修补，确保所有起跑检查失败即停」「采用真实 skipped 计数及完整性验收」「纳入计时与 trace，按计划提供实测分解」。GPU位置表供CPU验证、现有仿真环境单卡200次reset、V5兼容权重加载、35代表例及终止参考修正也均已按用户确认实施，记录见各验证档案。

已完成的验证证明实现和可复现性，不证明motion提升策略成功率。正式策略评估另起计划，保留同权重同budget的motion全遮消融，以及新run的60000与关闭态基线59999的同seed/test/1300步比较。当前四任务200次真实reset的最长es为383，1300步上限需求103窗，budget160余量57；改变任务或评估口径后需重测。
