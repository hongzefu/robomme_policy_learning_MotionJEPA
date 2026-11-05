# StopCube 真实 rollout 起跑档案

## 目的与授权

计划见 [1105-stopcube-real-rollout-plan.html](../../../1105-stopcube-real-rollout-plan.html)，实施中调整见其第十节。

用户原话：「就还是这四个问题全部用真实 rollout 来做……触发时刻分布是否在窗口内，按的那一刻有没有看 motion，以及模型知不知道自己离按有几步和离正确窗口还有几步。我们都只做真实的 rollout，给我一个完整的方案」「同意开工 先别加了」「都按建议改」「开工了！！」。

状态机：用户未答复是否登记，未登记，按人工交接。

## 运行环境

环境 A，sled-aspen，两张 RTX A6000，不连接集群。基线 ctx 用 GPU0（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`），motion 用 GPU1（`0.55`）。输入和产物都在本机 `/data` NVMe。`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`。现场有其他用户的小显存进程，没有操作。

## 版本

| 阶段 | 版本 |
| --- | --- |
| 第 16 局 rollout 冒烟 | `7822a6aad15fbd5d6d1641ee4cc0dacfe6ce58ee` |
| 第 16 局离线冒烟（首轮失败） | `7822a6a`、`a904e8d` |
| 第 16 局离线冒烟（基线续跑） | `41fc7b6149973d173bab8d5c61441df7ef3ddc05` |
| 正式 50 局 rollout、全量离线、分析 | `41fc7b6149973d173bab8d5c61441df7ef3ddc05` |

执行树是干净的分离工作树 `v1-store/worktrees/stopcube-rollout-run`，脚本起跑前核对 HEAD 和 `git status --porcelain` 为空。benchmark 使用主检出 `third_party/robomme_benchmark`，gitlink `856bc3a189d4172f3f47dbee4424d585f8d78db3`。

## 被跑对象

| 标记 | 权重 | 配置 | model_kind |
| --- | --- | --- | --- |
| 基线 ctx | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | `mme_vla_suite_b128` | `history` |
| motion mot | `v1-store/assets/vla-o16cc-59999/59999` | `mme_vla_suite_orig16_b128_60k` | `stage2_pool4_concat` |

模型 seed 42；清单 `v1-store/evaluation/stopcube-oracle-1104/manifest-all.json`，封签 `05313f4116785920f732984648e7958aecbeeb07592379ebcf6997539b3e2fdd`，50 局。窗口表 `v1-store/evaluation/stopcube-oracle-1104/oracle/oracle_table.json`，sha256 `6a9647cd6e7442136221b2d5c7f3d3d3ed4ed60e734eb02aaa8984798c6322a3`，只取窗口、M、N、悬停高度。

## 代码改动

W1 输入记录（`commitV4.1.25`），W2 隐状态捕获（`V4.1.26`），W3 触发分布与对齐（`V4.1.27`），W4 离线入口（`V4.1.28`），W5 注意力与探针（`V4.1.29`），W6、W7 隐状态闸门（`V4.1.30`、`V4.1.31`）。全部经合并前后审查，`src/`、`third_party/`、`scripts/training/`、`scripts/evaluation/` 零改动。

## 完整命令

产物根 `O=v1-store/evaluation/stopcube-rollout-1105`。

```bash
# 正式 rollout（规格 $O/spec-all-<模型>.json：stopcube_mode=normal、rollout_record=true）
python -m scripts.interpret.run_local --spec "$O/spec-all-<模型>.json"

# 全量离线
python -m scripts.interpret.rollout_offline --attn-root "$O/<模型>/attn" --run-root "$O/<模型>/run" \
  --output "$O/<模型>/offline" --ckpt <权重> --config <配置> --seed 42 --query-batch-size 2 --parity first

# 分析
python -m scripts.interpret.analysis.rollout_trigger --root "$O" --table <窗口表>
<sim 环境 python> -m scripts.interpret.analysis.rollout_attention heights --root "$O" --table <窗口表> --workers 4
python -m scripts.interpret.analysis.rollout_attention summarize --root "$O" --table <窗口表>
python -m scripts.interpret.analysis.rollout_probe --root "$O"
```

冒烟另加 `--episodes StopCube:16`，离线冒烟用 `--parity all`。

## 时间线（美东 2026-11-05）

| 时刻 | 事件 |
| --- | --- |
| 12:13 | 第 16 局 rollout 冒烟起跑，两边约 2 分钟完成 |
| 12:15 | 离线冒烟首轮，`HIDDEN_PARITY` 因逐位判据失败，停止放大 |
| 12:18～12:27 | 诊断：f32 下开关差异约 1e-7；隐状态精度补测 |
| 12:31 | 新判据重跑，基线单头差 0.0066 超 0.005，再次停下测分布 |
| 12:40 | 单头容差定为 0.02，基线续跑通过 |
| 12:43 | 正式 50 局起跑；基线约 17 分钟结束，motion 约 30 分钟结束 |
| 13:00、13:14 | 基线、motion 全量离线起跑，各用时 2360.9、2476.7 秒 |
| 13:56 | S5 分析起跑，约 5 分钟结束 |

## tmux 会话清单

`sr-smoke-ctx-a`、`sr-smoke-mot-a`、`sr-smoff-ctx-a`、`sr-smoff-mot-a`、`sr-smoff-ctx-b`、`sr-smoff-mot-b`、`sr-smoff-ctx-c`、`sr-diag-ctx-a`、`sr-diag-mot-a`、`sr-diag-ctx-b`、`sr-diagh-ctx-a`、`sr-diagh-mot-a`、`sr-diagh-ctx-b`、`sr-diagh-mot-b`、`sr-diagh-mot-c`、`sr-diaga-ctx-a`、`sr-roll-ctx-a`、`sr-roll-mot-a`、`sr-off-ctx-a`、`sr-off-mot-a`、`sr-analysis-a`。全部自然退出，没有手动清理。

## 产物路径

原始数组、轨迹和视频留在 `$O/<模型>/`，共约 8.8 GB，不进 git。git 只归档 `records/` 下的统计 JSON、图、清洗后日志和一次性诊断脚本。
