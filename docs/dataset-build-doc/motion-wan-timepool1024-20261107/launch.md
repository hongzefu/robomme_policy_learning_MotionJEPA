# 官方16×100 Wan 时间优先池化 1024 motion 表：启动说明

## 一、目的与来源
替换 stage2 encoder，用零参数池化生成 `motion-wan-timepool1024`：同一 33 帧窗口独立 Wan 编码，丢第 0 个 latent，后 8 个沿时间取均值，32×32 空间池化到 8×8，展平为每窗 1×1024 float32。计划：根目录 `1106-orig16-context-wan-timepool1024-60k-plan.html`（`ed5714c`）。

用户原话：「按照时间优先 8→1取均值 32×32→8×8 [1,16,8,8] 1024的机制」「只做pooling 不要用接续的wan」「仍去掉首帧，只平均后8个」；开工令「开工」，修订后「同意」。

## 二、运行环境
环境 B：AWS 单机 8×A100。本阶段全程 CPU（恢复阶段已结束，用过 8 卡）。

## 三、输入与输出
| 项 | 值 |
|---|---|
| 源 H5 | `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5` |
| 旧计划与收据（只读） | `v1-store/validation/orig16-stage2/wan-cache-plan.json`、`wan-cache-import.json` |
| 恢复根 | `v1-store/cache/orig16-wan-timepool1024-recovery/`（`published`、`wan-latents`、`wan-cache-import.restored.json`） |
| 导出 | 同根 `wan-timepool-export/` |
| 新表 | `v1-store/datasets/16task-orig100-1600ep-9e109419/motion-wan-timepool1024/` |
| 闸门记录 | `v1-store/validation/orig16-wan-timepool1024/` |

## 四、已完成的前置阶段
- 下载：美东 2026-11-06 20:49 起，16 路，`WAN_CACHE_FETCH=PASS`，退出 0，`v1-store/logs/orig16-wan-timepool1024/fetch.log`。下载方式与计划差异（16 路 SDK、`.venv` 解释器）已获用户同意。
- 恢复：美东 21:18–22:08，8 卡，`WAN_CACHE_IMPORT=PASS copied=42404 filled=2824 total=45228 input_mismatches=0 nonfinite=0`，退出 0，`restore.log`。

## 五、执行顺序
脚本 `v1-store/logs/orig16-wan-timepool1024/build-stages.sh`（不进 git，运行载体）按序：G0 同源 → G3 机制 → G1 恢复对拍 → export → pack → 表核对 → G2 窗口 → G4 全表 → 记录绑定 → verify 封口 → G5 参数／初始化／配置／样本 → G6 同 latent → 起跑合同 dry-run。解释器：export 用 `v1-store/venvs/wan-stage2`（与在线 sidecar 同环境）；pack、verify、闸门用主 `.venv`。

依用户裁决，G7 在 20 步短测句柄内执行。`train_head` 口径：表记建库 Beta；正式训练 HEAD 必须是其后代且 `src`、`scripts` 零差异。

## 六、本轮 tmux 会话清单
`o16-tp-fetch-11062053`、`o16-tp-restore-11062118`；建库、短测、训练、备份会话见后续记录。
