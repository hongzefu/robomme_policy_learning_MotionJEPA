# 公开counting库20步可读性检查：预建记录

用户已要求「恢复计划中的建库，严格按前置闸门推进」，根计划第1步包含正式库完成后的20步可读性、保存和加载检查。本run尚未启动，不是80k正式训练，也不代替上游100步对拍。

## 版本、数据与参数

从本轮包含本档案的 `commitV11.11Beta` clean HEAD启动，实际完整 `TRAIN_HEAD` 字面量写入driver日志、launch.json及final记录。数据为 `v1-store/datasets/4task-counting-pub-400ep/framesamp`，须先完成来源、构建、finalize、pack/verify、独立norm_stats及全量子集对拍；不得用完整库统计量替代counting统计量。

模式 `smoke`、run_name=`smoke-orig80k-count-0925`、GPU4–7。仅覆盖步数为20，其余与mme_vla_suite原版相同：global batch64、FSDP4、workers4、seed42、原版lr/warmup/EMA、log100/save10000/keep10000、modul 512/4×4，无motion。history SHA为 `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`；counting统计量尚未生成，`NORM_STATS_SHA256`须待本库CPU统计完成、结构和有限性检查及来源记录后填写完整字面量，不预造值。

## 启动与验收

入口为 `scripts/training/prod/run_orig80k.sh smoke smoke-orig80k-count-0925 4,5,6,7 <主仓库>/v1-store/datasets/4task-counting-pub-400ep <主仓库>/v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep`，环境提供完整 `TRAIN_HEAD/HISTORY_CONFIG_SHA256/NORM_STATS_SHA256`。`assets-dir`指统计量父目录，asset-id仍为robomme；所有实际路径使用绝对路径。W&B保持原版开启，凭据不输出到日志或档案。

按可能超过5分钟的验证管理：detached tmux、唯一会话和精确PID、tee和最终EXIT_CODE。输出根为 `v1-store/train-runs/mme_vla_suite/smoke-orig80k-count-0925`，记录为 `v1-store/bench/orig80k/smoke-orig80k-count-0925`；必须全新，失败保留，不覆盖/续跑。

使用 `check_orig80k_completion.py --mode smoke` 核对20次更新、末步19、step0及尾窗1…19完整有限、真实四卡/x64关闭、有效GPU采样、异步保存完成和真实恢复权重等于本次EMA。指标和清洗日志归档，权重不进Git；本检查不宣称counting与完整run的loss可直接比较。实际起止、耗时、判定和后续清理待完成后回填。
