# 官方16×100 context训练启动档案

当前状态：已获开工授权，尚未启动；等待[建库](../../dataset-build-doc/16task-orig100-1600ep-9e109419/launch.md)、G0–G6及两条真实20步短测通过。正式启动时补充训练Beta完整SHA、时间、实际设备和句柄，不把建库Beta冒充训练版本。

固定run为`orig100-ctx32x4x4-motion-s2split4x256-b128-60k`。训练配置`mme_vla_suite_orig16_b128_60k`，8卡单进程FSDP8、全局批量128、16 workers、seed42、60,000步、EMA0.999；warmup5000、学习率5e-5、每5000步保存及最终59999。帧32×4×4=512 token，motion每窗4×256、96窗=384 token，总896；context宽度2048。

数据为官方16×100，库`16task-orig100-1600ep-9e109419`的`framesamp-4x4`、完整source及ep105 EMA导出的motion；实际摘要在建库后回填启动合同。所有数值/数据/seed/顺序与后继modulation相同，配置仅`integration_type`和`memory_token_dim`不同。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRAIN_HEAD=<训练Beta完整SHA> \
FINAL_ROOT=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/orig16-stage2 \
bash scripts/training/run_orig16_ctx_then_modul_60k.sh
```

正式入口从`history_config`选择帧表；W&B沿既有凭据存在分支。日志`v1-store/logs/orig100-ctx32x4x4-motion-s2split4x256-b128-60k.log`，checkpoint根`v1-store/train-runs/mme_vla_suite_orig16_b128_60k/orig100-ctx32x4x4-motion-s2split4x256-b128-60k`，终步59999。训练成功退出、最终记录有限且59999已保存后立即接modulation，不等上传/G12；异常停止接续。

公开bucket为`HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2split-60k-v1`，12份checkpoint增量上传后全量匿名读回；无集群环境，读回在本机执行。两条59999恢复均在modulation完成后执行。本计划不评估，不预先承诺实测步时或显存。

用户原话：「开工 有问题问用户 尽可能不要阻塞 所有subagent全部astra high」「登记三项任务（推荐）」。运行、失败及耗时在实际起跑后补记，结果与不可由Git还原的证据最终写入`result.md`、`records/`。
