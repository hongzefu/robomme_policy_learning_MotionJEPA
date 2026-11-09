# 官方16×100 context + concat（不 split）60k：结果

## 一、一句话结论

正式 60000 步完成（美东 10-30 16:15:48 → 10-31 23:26:35，墙钟 31 小时 10 分 47 秒，稳态约 1.9 秒/步），退出 0；12 份 checkpoint 全量公开上传并匿名回读零失配；59999 从实体 stage 严格恢复与真实在线推理通过。同批计划的 modulation + concat 按用户决定放弃、未训练。

```text
LAUNCH_CONTRACT=PASS checks=50 failed=0
TRAIN_FINAL_RECORD=PASS run=orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k tail_steps=99 ema_leaves=59
EXIT_CODE=0
HF_VERIFY=PASS bucket=HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1 objects=273 bytes=139451480655 mismatches=0
CKPT_UPLOAD=PASS steps=12
CKPT_DELIVERY_READY=PASS step=59999 root_complete=1 sha256_mismatch=0 strict_restore=1 online_infer=1
```

## 二、版本、数据与配置

启动与配置还原见 [launch.md](launch.md)。训练 Beta `commitV11.69Beta 6424d7f39569739eae1ca6cdbb50c7f26897e738`（代码同 commitV11.68 `6cd4d6c`），库 `16task-orig100-1600ep-9e109419`，`DATA_LINEAGE=PASS ... files=16 mismatches=0`。concat 读法每窗 1×1024、96 窗、总记忆槽 608；context 融合、`memory_token_dim=2048`。

## 三、训练过程（训练日志标量，非评估指标）

| 步 | concat loss / grad_norm | split 对照 loss / grad_norm |
|---|---|---|
| 0 | 0.5556 / 23.1226 | 0.6093 / 22.0420 |
| 5000 | 0.0070 / 0.0536 | 0.0067 / 0.0595 |
| 30000 | 0.0020 / 0.0238 | 0.0019 / 0.0222 |
| 59900 | 0.0013 / 0.0216 | 0.0012 / 0.0199 |

全程标量有限，无 OOM。稳态约 1.9 秒/步、每卡约 41.8 GB；split 对照约 2.4 秒/步、墙钟 40 小时 34 分，concat 少 288 个记忆槽，端到端快约 23%。训练 loss 只作过程参考，不能代替统一 benchmark 下的任务成功率比较；正式评估另列任务。

## 四、交付与清理

增量上传 5000…55000、59999 共 12 份到公开 bucket，最终清单 273 对象全量匿名回读零失配；G12 从 `v1-store/exports/hf-ckpt-<run>/stage/59999` 实体包恢复。两项均 PASS 后按用户「现在你来删除」「以后都要删除」删除本地 stage（139451543272 字节、273 文件）与 cache，删除记录在 `v1-store/exports/hf-ckpt-<run>/logs/DELETED_LOCAL_STAGE.txt`；训练 run 本体保留在 `v1-store/train-runs/mme_vla_suite_orig16_b128_60k/<run>`。两条 20 步短测 checkpoint 亦已删除。

## 五、用户决策记录

- 2026-10-30：「给出方案 现在训了context和modul split的模型 同时再训2个不split的」「同意增量上传诊断失败后就问用户 改写md」「我希望开工的时候只有一份仓库和一份temp」「其他你来决策」「开工」。
- 2026-10-31 美东 22:2x：「modulation concat不训练放弃 context训练完毕就收尾」。
- 同日：「同意 一路继续做下去 尽可能不要阻塞 有问题留着明天早上再问用户」「明天早上10点前不要有问题」。

## 六、计划外事件与处置

- 本机无 `rg` 可执行文件，旧入口诊断会静默失效；commitV11.68 改为 GNU grep（ADJUST_REVIEW=PASS）。
- 放弃 modulation：停掉其备份会话（未上传任何对象），在其日志路径放 `ABANDONED_BY_USER` 占位；串行入口接续时被单条入口 noclobber 拒绝（退出 2、未占 GPU、未建 run），链写 `WAIT_USER_DECISION ... stage=train_exit` 后以 `EXIT_CODE=2` 结束——为预期留证，不是故障。

## 七、归档

`records/train.summary.log`（清洗后训练日志，600 个 Step 行）、`records/checkpoint-backup.summary.log`、`records/delivery.json`。
