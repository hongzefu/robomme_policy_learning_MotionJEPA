# 官方16×100 context + Wan 时间池化 1024 60k：结果

## 一、结论与指标
零参数池化替换 stage2 encoder 后，context B128 60k 全程完成：60,000 步，真实退出 0，12 份 checkpoint 齐全，59999 严格恢复通过。VLA 参数与旧 concat 实验逐叶一致（3,363,663,120）。耗时 31 小时 9 分 45 秒（旧 concat 31 小时 10 分 47 秒）。训练 loss 只是过程参考，不代表任务成功率；本任务不含策略评估。

| 判定 | 原文 |
|---|---|
| 训练记录 | `TRAIN_FINAL_RECORD=PASS state_step=60000 checkpoint_step=59999` |
| checkpoint 清单 | `CKPT_INVENTORY=PASS steps=12` |
| 严格恢复 | `RESTORE_POOL=PASS` |
| 训练退出 | `EXIT_CODE=0`，无 `TRAIN_NUMERICS=FAIL`、无 OOM |

## 二、起止与硬件
美东 2026-11-06 23:12:50 起跑（UTC 11-07 04:12:50），美东 2026-11-08 06:22:35 结束（UTC 11-08 11:22:35）。8×A100-80GB，稳态约 1.9 秒/步。

## 三、训练过程（训练日志标量，非评估指标）
| 步 | timepool loss / grad_norm | 旧 concat 对照 loss / grad_norm |
|---|---|---|
| 0 | 0.5053 / 43.4176 | 0.5556 / 23.1226 |
| 5000 | 0.0074 / 0.0543 | 0.0070 / 0.0536 |
| 30000 | 0.0020 / 0.0220 | 0.0020 / 0.0238 |
| 59900 | 0.0013 / 0.0202 | 0.0013 / 0.0216 |

末 99 步均值：loss 0.001302，grad_norm 0.02026。起始 grad_norm 约为旧实验的 2 倍：raw Wan 特征幅度和 learned encoder 不同，计划第 13 节已列为风险。5000 步后两者轨迹接近。

## 四、计划外事件与处置
1. **短测 G8 首次失败**：参考输出 `max_abs=0.01947`，原因是 `smoke.sh` 缺编译缓存环境。补环境后在同一 checkpoint 上通过，没有重训 20 步（详见 launch.md 第五节）。
2. **G9 首次 `CKPT_INVENTORY=FAIL`**：验证器把 `train_state` 列为必需项，但本仓库 `save_state` 只保存 `assets` 与 `params`，旧实验也一样。修复提交 `b3c1e90` 经合前、合后审查后重跑通过。首次日志保留为 `records/g9-final.attempt1.summary.log`，首次记录保留为 `*.attempt-*.json`。
3. **备份**：计划写的是沿用旧导出脚本，但旧脚本绑定 stage2，所以新写了独立工具（`7512410`）。训练期间从固定源码树 `/scratch/hongze/worktrees/tp-backup-2065b73` 增量上传，12 份均已上传并逐组匿名读回，最终全量读回作为长尾。bucket：`HongzeFu/robomme-vla-orig100-ctx32x4x4-wan-timepool8x8-60k-v1`（公开）。

## 五、下一步
- 策略成功率评估不在本任务内。已有另一个会话在 `v10-ctx-wan-timepool-eval` 分支按其计划进行 3 seed 评估。
- 本地大文件是否删除，待用户裁决：恢复产物 `published` 362 GB 与 `wan-latents` 25 GB；导出暂存 `v1-store/exports/hf-ckpt-<run>` 130 GB，等备份全量读回通过后可删。训练 run 根 130 GB 是评估要读的，不在清理范围。

## 六、归档文件
`records/train.summary.log`、`build.summary.log`、`smoke-11062246.summary.log`、`save-load-retry-11062246.summary.log`、`g9-final.attempt1.summary.log`、`g9-final-rerun.summary.log`；验证 JSON 留在 `v1-store/validation/orig16-wan-timepool1024/`。
