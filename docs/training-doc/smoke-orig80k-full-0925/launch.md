# 公开16任务库20步可读性检查：已完成的启动记录

本launch在起跑前随Beta提交，运行结束后回填实际起止、参数和验收；当时的启动约定保留在下一节。用户要求「恢复计划中的建库，严格按前置闸门推进」，根计划第1步包含正式库完成后的20步可读性、真实保存和加载。本run已全部通过，详见[十二节结果](README.md)；它不是80k正式训练，也不代替正式输入、上游100步或并跑/perf验证。

## 起跑前约定与实际版本

起跑前约定从包含本档案的`commitV11.11Beta` clean HEAD启动，输出根全新，失败保留，不使用overwrite/resume。实际启动和结束锚点均为完整 **`49a333eb18e8d6ff1143bf7871ef7c498ab91579`**；driver与wrapper检查HEAD和clean状态，实际解析配置、开始/末步/恢复记录均绑定该提交。结果随本组`commitV11.11`配对归档，不预填最终提交SHA，也不把归档后的版本当成训练代码。

数据为`v1-store/datasets/16task-pub-1600ep/framesamp`。起跑前已通过完整库来源pin、SigLIP、finalize、packed全量verify及统计量比较。原先约定使用原版norm_stats，实际仍使用f332文件；完整库自算c5文件只作比较，未替换训练资产。

模式`smoke`、run_name=`smoke-orig80k-full-0925`、物理GPU`0,1,2,3`。仅将步数覆盖为20；其余仍为`mme_vla_suite`、global batch64、FSDP4、workers4、seed42、原版lr/warmup/EMA、log100/save10000/keep10000、modul512/4×4，无motion。history SHA为`823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`，实际norm_stats SHA为`f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。W&B按原版开启，凭据未写入档案。

## 实际启动与会话

以下为已经执行的实质命令；完整展开argv及配置见[run_meta.json](records/run_meta.json)与[launch.actual.json](records/launch.actual.json)。训练只清除CPU平台选择遗留；实际`JAX_ENABLE_X64`未设置、解析值为false，没有通过覆盖它绕过runner拒绝。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  TRAIN_HEAD=49a333eb18e8d6ff1143bf7871ef7c498ab91579 \
  HISTORY_CONFIG_SHA256=823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a \
  NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5 \
  bash scripts/training/prod/run_orig80k.sh \
  smoke smoke-orig80k-full-0925 0,1,2,3 \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite
```

实际detached tmux全名为`orig80k-full-read20-20260926T004046Z`，pane `%539`，pane/包装PID `3915445`；run UUID为`a56dd89a-ffbe-415d-9dd1-d5e205e14c25`。2026-09-26 UTC整体时间`00:41:03→00:47:20`，训练runner为`00:41:03→00:46:19`，随后CPU完成器至`00:47:20`。采样、真实训练等其余精确PID见[wrapper日志](records/wrapper.summary.log)和[final开始记录](records/final/start.json)。

实际driver日志为`v1-store/logs/smoke-orig80k-full-0925.driver.log`，wrapper日志为`v1-store/logs/orig80k-full-read20-20260926T004046Z.wrapper.log`。runner独占driver终态，外层没有往driver追加；wrapper保存其主体任务、正文tee返回码及预先计算的日志终态，最外层footer证据边界见下文。checkpoint根`v1-store/train-runs/mme_vla_suite/smoke-orig80k-full-0925`、记录根`v1-store/bench/orig80k/smoke-orig80k-full-0925`起跑时全新，现已生成结果，未覆盖重跑。

## 实际验收与归档

训练成功后同一tmux串行以`CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`调用`check_orig80k_completion.py --mode smoke`，完整`--run/--head/--records/--run-root/--log/--out`参数及实际命令在[结果正文](README.md)第3节；`--out`为本run记录根下`completion.json`，完成器日志单独tee，两个返回码分别核对。

实际验收为20次更新、唯一末步19、普通step0和尾窗1–19完整有限、四卡与x64关闭、GPU采样覆盖及异步保存完成。CPU完成器真实加载19/params，与本次EMA的61个叶逐项比较，23个bf16叶与38个f32叶的dtype/shape/字节摘要全部相等。日志保留`RUN_COMPLETED=PASS run=smoke-orig80k-full-0925 checkpoints=1 final=19`与`READABILITY20=PASS`；整个runner的真实返回0由外层捕获，恢复器及各主体管道也分别返回0。原版训练标量/状态对上游的bitwise要求没有被这道自恢复检查替代。

最外层wrapper先写终态文本、再检查footer自身printf/tee，最后状态未独立持久化；末行`EXIT_CODE=0`不能单独证明包装进程实际退出0，详见[历史退出记录审计](../orig80k-equiv-0925/exit-record-audit.md)。没有证据表明历史footer失败，训练、保存和CPU恢复PASS保留，原始records不改写。

本目录已归档16个records文件，共348373 B，最终[archive_checks.json](records/archive_checks.json) SHA为`2a3bce6eee4f2bcb4ec4e50f481b6df995747504845eb8f50b943b59cd744f8d`。正式副本只对.log去行尾空格/制表符，train/wrapper两log共去168个字符；原始和ignored暂存未改，规范化后逐行一致，没有指标变化。权重留在运行目录，不复制脚本、yaml或独立bash载体。

正式INPUT_EQ、上游100步、单跑/并跑、perf及80k尚未通过本run验证；对拍W&B关闭覆盖和保存期磁盘峰值采样两项仍待用户答复，未虚构批准。20步不作吞吐、ETA或模型质量结论。
