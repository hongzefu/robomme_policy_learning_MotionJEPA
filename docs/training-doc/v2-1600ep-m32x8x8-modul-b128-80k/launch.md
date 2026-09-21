# 无motion、32帧×8×8 modulation 80k正式训练

**更新：全部前置条件已满足，固定 commitV11.2Beta 后执行 V10。** [测速报告](../perf-m32x8x8-modul-b128-20260921T051856Z/result.md)为READY，GPU稳态均值98.6104%>50%，正确性与容量全部通过，无未解决问题。条件判定见 [conditions.met.json](records/conditions.met.json)，正式授权记录待Beta实际SHA生成后写入忽略目录。下面保留预建档时的决定与完整启动约定；实际起跑状态以随后写入的records为准。

**准备中，尚未起跑。** 用户已指定新run_name `v2-1600ep-m32x8x8-modul-b128-80k`，并明确：「1000步测速后如果占用率高于50% 并且没有其他的问题 可以直接启动训练」。采用800步稳态窗口内八卡总体平均GPU利用率严格大于50%、全部正确性与容量检查通过、SPEED_REPORT=READY且无未解决问题的口径。条件成立后生成绑定实际报告/runner/TRAIN_HEAD的授权记录，再执行V10，不重复询问。条件未满足则不启动。

## 版本与代码状态

正式启动前须固定Beta与clean TRAIN_HEAD；实际完整hash及UTC由runner和起跑记录写入，不预填本文自身hash。PERF_HEAD到TRAIN_HEAD之间只允许文档变化，src/scripts/packages/pyproject.toml/uv.lock必须相同。源码、配置与runner可用 `git show <TRAIN_HEAD>:scripts/training/prod/run_modul2048.sh`、`git show <TRAIN_HEAD>:src/mme_vla_suite/training/config.py` 和同提交的新history YAML还原。

生产改动始于commitV11.0，CAND0c877c7495dfe5db8b83f033442013c6d6fd8552完成实际训练取证；后续仅修验证器的权重JIT参数和同路径扰动基线。正式模型为完整生产VLM与动作专家，不使用V5的gemma_150m替身。正式从pi05_base独立初始化，不接续验证或测速checkpoint。

## 实际配置与数据

原样使用mme_vla_suite_b128_80k：batch128、80000步、worker16、FSDP8、seed42、warmup5000、peak_lr=decay_lr=5e-5、decay_steps80000、EMA0.999、梯度裁剪1.0、log_interval100、save_interval/keep_period5000。正式CLI只指定run、资产根、数据路径与2048 YAML；所有训练超参保持具名默认。两次独立CLI预解析一致，预览config SHA为b2acd4a95c6c9fb5649fe0791b24ab891292ffeaa56befb747221cf90179bbb8；起跑门仍重新解析实际TRAIN_ARGS，预览不是授权记录。

数据根为主仓v1-store/datasets/4task-v2-1600ep-604f16da，使用其source/meta和verified/full framesamp-8x8，1600集、605611执行样本、1192918帧；本库执行样本均满32帧。清单file SHA为df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482，canonical SHA为4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918；norm_stats文件SHA为856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。新YAML raw SHA为42813c7e06840e511f8a382f14954edcaad8d4fd5b320b4323fea94979cda4de，budget2048/token_per_image64/num_views1/memory_token_dim1024、modulation、motion关闭。

## 启动与监控

目标为本机8×A100-SXM4-80GB、/dev/md0 XFS NVMe RAID。起跑前重查八卡显存全0、其他本轮GPU任务结束、scratch余量≥400GB、输出根全新、代码clean、数据及报告/runner摘要匹配。实际输出为v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m32x8x8-modul-b128-80k，禁止overwrite/resume。预期保存5000至75000每5000一步和最终79999。

入口为 `bash scripts/training/prod/run_modul2048.sh prod v2-1600ep-m32x8x8-modul-b128-80k <TRAIN_HEAD> 42813c7e06840e511f8a382f14954edcaad8d4fd5b320b4323fea94979cda4de <APPROVAL_JSON> <REPORT_JSON>`，占位符仅在条件成立后填入实际值。tmux全名m2048-prod，runner保留内层set-euo、外层pipefail/tee/EXIT_CODE；XLA_FLAGS unset，TRAIN_TIMING_STEPS=0，无profiler。W&B沿用现有本机配置，凭据不进入档案或日志。

runner以独立精确PID持续500ms采集八卡GPU数据并在训练退出时回收，日志和采样落本run的v1-store路径。正式约300步后用原生metrics时间戳与GPU样本复核测速ETA，不添加逐步同步或trace。稳定后保持主副本训练源码与环境不变；起跑后文档在本轮独立副本归档，不触碰已有-temp的其他代理现场。训练结束的完整评估按计划另行开展，本轮不据短测宣称策略质量。
