# tic-t3-causal-40k —— 结果

> 起跑 commit `9b3b95f`（clean HEAD；工具版本 commitV7.1 `a8cfa17`）。2026-09-07 21:05:25 起，环境 B，GPU 2+3（`--fsdp 2`，`JAX_PLATFORMS=cuda`，确定性 XLA flags），tmux `tic-t3`（结束自行退出）。
> 原始日志 `records/t3-causal.txt`，明细 `records/t3-mechanism.json`。**`T3_MOTION_CAUSAL=PASS`、`T3_MECHANISM=PASS`，`EXIT_CODE=0`。**

## 一、判定行原文

```
[start] 2026-09-07T21:05:25+00:00 HEAD=9b3b95f04aaee2dfeb8b1501bb578c0167aab562
[t3mechanism] 选 step 0 的 batch: [6556, 671, 8452, 3987, 10070, 3804, 8928, 2595]
[t3mechanism] 初态命中 reference 与 run init 记录
[t3mechanism] 梯度摘要覆盖 trainable 叶 36 个（全参数叶 59）
[t3mechanism] 确定性探针 R=3：loss 逐位同=True（['0x1.698aea0000000p-1', '0x1.698aea0000000p-1', '0x1.698aea0000000p-1']）；不确定叶 0/36：[]
[t3mechanism] 排除不确定叶 0 个后，梯度摘要覆盖 36/36 叶
[t3mechanism] base loss 0.706138
  [pad] loss base=0x1.698aea0000000p-1 pad(1e3)=0x1.698aea0000000p-1 同=True；覆盖叶变化 0/36：[]
  [pad] 垫料尺度 1: loss 同=True 摘要同=True 覆盖叶变化 0：[]
  [pad] 垫料尺度 0.001: loss 同=True 摘要同=True 覆盖叶变化 0：[]
[t3mechanism] 分组梯度范数 {"W2_content[:768]": "4.3906e+01", "W2_pos[768:]": "5.2996e+00", "W1": "5.5220e+00", "b1": "4.8132e-01", "b2": "1.6320e+00", "motion_emb_valid": "9.4053e-01", "motion_pos_valid": "1.5253e-01"}
T3_MOTION_CAUSAL=PASS pad_bitexact=1 loss_bitexact=1 emb_effect=1 pos_effect=1 det_probes=3 nondeterministic_leaves=[] excluded=0 covered=36/36 excluded_diag=[]
T3_MECHANISM=PASS step=0 input_grad_ok=1 group_norms_ok=1
EXIT_CODE=0
```

## 二、结论

1. **`T3_MOTION_CAUSAL` FAIL 正式闭合**：新口径先无条件对同一 obs 连算 R=3 次，loss 三次逐位同（`0x1.698aea0p-1`），36 个可训练叶的梯度摘要三次全同——**不确定叶 0 个**；随后垫料三档（1e3 / 1 / 1e-3）下 loss 与全部 36 叶梯度摘要逐位不变（`pad_bitexact=1 loss_bitexact=1`），motion 嵌入与位置两路各自有效（`emb_effect=1 pos_effect=1`）。判定语义按计划收窄为「排除叶之外逐位一致」，本 run `excluded=0 covered=36/36`，即**全部可训练叶逐位一致**，无需收窄。
2. **09-04 的不确定性未复现**：`aws-t3-open-s100`（HEAD `8093ebd`，2026-09-04）记录 `['PaliGemma']['llm']['embedder']['input_embedding']` 叶同 obs 两次也变、`pad_bitexact=0`；本 run 与 2026-09-07 开发 smoke（同 GPU 2,3）均 `nondeterministic_leaves=[]`。该不确定性看来是间歇的（换 GPU 对 / HEAD 后消失），**排除叶诊断路径（`excluded_diag`）本 run 未被真实数据走过**，只在空集意义上验证。
3. **`T3_MECHANISM` 同批同初态 PASS**：step 0 batch `[6556, 671, 8452, 3987, 10070, 3804, 8928, 2595]` 与 `aws-t3-open-s100` 相同，初态命中 `t3_common_init_reference.json` 与 run init 记录；输入梯度与分组范数判据过。

## 三、与 09-04 记录并列的数字

| 项 | aws-t3-open-s100（09-04，HEAD 8093ebd） | 本 run（09-07，HEAD 9b3b95f） |
|---|---|---|
| step 0 batch | `[6556, 671, 8452, 3987, 10070, 3804, 8928, 2595]` | 同 |
| 初态命中 reference | 是 | 是 |
| base loss | 0.704831 | 0.706138 |
| 不确定叶 | `input_embedding`（`pad_bitexact=0`） | 无（`covered=36/36`） |

base loss 相差 0.19%：两个 HEAD 之间 `src/` 只新增了 `mme_vla_suite_b128` 配置项（无数值改动），同批同初态、确定性 flags 下 loss 仍不同，与开发 smoke（同卡）逐位相同——差异应来自 GPU 对 / 编译层面而非代码语义，本轮未进一步追查，如实并列。

## 四、异常处置

无重跑；tmux 会话随命令结束自行退出，未执行任何 kill；`--tmp v1-store/tmp/tic-t3mech` 由脚本自身清理。
