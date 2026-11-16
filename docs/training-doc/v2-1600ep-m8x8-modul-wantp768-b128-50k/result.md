# v2-1600ep-m8x8-modul-wantp768-b128-50k 训练结果

## ① 一句话结论

50,001 步正常跑完，10 份 checkpoint（5000…50000）齐全，最终 50000 真实加载通过：`TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10`，`EXIT_CODE=0`。策略成功率见本机 692 局评估档案（训练结束后自动接续，评估另档）。

## ⑥ 硬件与耗时

AWS 8×A100-SXM4-80GB，本地 NVMe RAID `/dev/md0`，FSDP8、batch 128、workers 16。tmux `w768-train` 起于 2026-11-08T18:29:42Z（美东 11-08 13:29:42），`Step 0` 于 18:33:59Z（含编译约 4 分 17 秒），`END_UTC=2026-11-09T08:00:21Z`（美东 11-09 03:00:21），总墙钟 13 h 30 m 39 s。稳态约 0.99–1.03 s/步（Step 200→1000 实测 1.03 s/步，后段 tqdm 报 1.0 it/s），与上次同配置 80k run 的 0.99 s/步一致。编译期出现 XLA `slow_operation_alarm`（常量折叠单次约 9.4 s），仅为提示。

## ⑦ 训练过程行为

| step | loss | grad_norm |
|---:|---:|---:|
| 0 | 0.0887 | 1.1104 |
| 5000 | 0.0084 | 0.0590 |
| 10000 | 0.0053 | 0.0361 |
| 15000 | 0.0043 | 0.0311 |
| 20000 | 0.0034 | 0.0283 |
| 25000 | 0.0032 | 0.0288 |
| 30000 | 0.0028 | 0.0263 |
| 35000 | 0.0026 | 0.0262 |
| 40000 | 0.0024 | 0.0242 |
| 45000 | 0.0022 | 0.0240 |
| 50000 | 0.0020 | 0.0232 |

全程无 NaN、OOM、进程被杀或报错；Monitor 每 5000 步记录一次，按用户裁决「训练中途异常停下等你」未触发。

最终加载：`CKPT_LOAD=PASS step=50000 leaves=65 resolved_sha256=4e70e93373d5b20061a4067c911021100375d2bc28cff276acaf7559b2441910`（与 check_checkpoint 白名单一致）。

## ⑩ 计划外事件与处置

1. **HF 私有存储额度**：ckpt 上传到第 5 份（step 25000）时 HF 返回 403（需开自动充值）。原因是该 bucket 按计划建为私有，而账户以往 37 个 bucket 全为公开（约 9.8 TB），私有额度很小。用户裁决「改成公开（跟以往一样」，主会话以仓库既定 `v1-store/secrets/hf.env` 令牌调 `PUT /api/buckets/<id>/settings {"private": false}`，返回 200，匿名访问由 401 变 200；脚本第 5 次重试即续传成功。
2. **上传会话被外部 SIGTERM**：美东 11-08 23:56:38，另一会话（context1024 训练）的写入子代理清理自身测试残留时以 `ps | grep "sleep 300"` 匹配后 kill，误中 `w768-ckptup` 等待落盘的 `sleep 300`，脚本以 143 退出。对方已承认并改为只清理 `$!` 记录的 PID。主会话整脚本重跑续传（已传 7 份按 `verified/<step>.done` 跳过），未丢数据。训练不受影响。

## ⑪ 结论与下一步

训练侧交付完成；下游评估任务 `4task-wantp768-eval` 已于美东 11-09 03:01 起跑（tmux `w768-eval`，`PRESEED=PASS episodes=9 placed=9 duplicates=0`）。ckpt 上传长尾 `w768-ckptup` 剩最后一份 step 50000，完成行 `HF_VERIFY=PASS ckpts=10`（bucket `HongzeFu/robomme-vla-4task-modul-wantp768-50k-v1`，公开）。

## ⑫ 归档文件

日志留在 `v1-store/logs/w768-train.log`（含 tqdm，未清洗进 git）；本档案只记判定行与汇总数字。
