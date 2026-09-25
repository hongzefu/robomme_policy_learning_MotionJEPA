# 公开16任务库20步可读性检查：预建记录

用户已要求「恢复计划中的建库，严格按前置闸门推进」，根计划第1步包含正式库完成后的20步可读性、保存和加载检查。本run尚未启动，不是80k正式训练，也不代替上游100步对拍。

## 版本、数据与参数

从本轮包含本档案的 `commitV11.11Beta` clean HEAD启动，实际完整 `TRAIN_HEAD` 字面量写入driver日志、launch.json及final记录。数据为 `v1-store/datasets/16task-pub-1600ep/framesamp`，须先完成本库来源pin、SigLIP、finalize、packed全量verify及统计量比较；本run使用原版norm_stats，不用完整库自算文件。

模式 `smoke`、run_name=`smoke-orig80k-full-0925`、GPU0–3。仅覆盖步数为20；配置仍为mme_vla_suite、global batch64、FSDP4、workers4、seed42、原版lr/warmup/EMA、log100/save10000/keep10000、modul 512/4×4，无motion。history SHA为 `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`，norm_stats SHA为 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。

## 启动与验收

入口为 `scripts/training/prod/run_orig80k.sh smoke smoke-orig80k-full-0925 0,1,2,3 <主仓库>/v1-store/datasets/16task-pub-1600ep <主仓库>/v1-store/train-assets/mme_vla_suite`，环境提供上述完整 `TRAIN_HEAD/HISTORY_CONFIG_SHA256/NORM_STATS_SHA256`。所有实际路径使用绝对路径；展开后的argv由runner解析并留档。W&B保持原版开启，使用既有凭据环境文件，不在日志或档案输出凭据内容。

即使只有20步也按可能超过5分钟的验证管理：detached tmux、唯一会话名和精确PID留档、tee及最终EXIT_CODE。输出根为 `v1-store/train-runs/mme_vla_suite/smoke-orig80k-full-0925`，记录为 `v1-store/bench/orig80k/smoke-orig80k-full-0925`；启动前必须不存在，失败保留现场，不用overwrite/resume。

用 `check_orig80k_completion.py --mode smoke` 按本run/head/records/run-root/driver.log验收：20次更新、末步19、普通日志step0及尾窗1…19完整有限，真实四卡及x64关闭、GPU采样有效、异步保存完成；真实加载19/params并与本次EMA摘要一致。成功后归档指标和清洗日志，权重留在v1-store并按已授权临时产物纪律处理，不能删除其他run。实际启动时间、耗时和结果待执行后回填。
