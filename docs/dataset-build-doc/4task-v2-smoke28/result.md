# 28 集完整冒烟通过

2026-09-14 17:43:02–17:48:22 UTC，墙钟 320.90 秒，`EXIT_CODE=0`。14 个任务难度组各选两条，得到 28 集、21,071 步、11,110 个执行样本；全量 H5 验真、SigLIP、finalize、两档 framesamp 和独立跨网格校验全部通过。完整命令见 [launch.md](launch.md)，清洗日志见 [records/smoke.summary.log](records/smoke.summary.log)。

## 实测判定

```text
PLAN_OK tasks=4 selected=28 role=primary:28 max_timesteps=2304 prechecks=PASS
MERGE_VERIFY=PASS level=full tasks=4 episodes=28 timesteps=21071 role=primary:28 mismatches=0
STAGE_DONE stage=siglip workers=4 items=28 elapsed=123s
FINALIZE_EXIT_CODE=0
VERIFY_PACK=PASS scanned=21071 mismatches=0
VERIFY_PACK=PASS scanned=21071 mismatches=0
IMAGE_NPY_SPOT=PASS frames=512 mismatches=0
MOTION_POS_XGRID=PASS t=2078 npy=64 mismatches=0
EXIT_CODE=0
```

两行 VERIFY_PACK 按顺序分别属于 4×4 和 8×8，均为全行扫描。source 产物完整性为 feature 目录缺失 0、pkl 实得/期望 11,110/11,110、sidecar 4、覆盖 episode 28、残留 claim 0。MERGE_DONE 记录源 SHA256 独立核对 28 条，role 全为 primary。`plan` 的前检对象仍是完整 1600 条源，2304 是源全集最长步数；2078 是本次选中子集的位置表长度，两者口径不同。

## 耗时与占用

| 阶段 | 墙钟秒 |
|---|---:|
| 全源前检与选取 | 9 |
| 四任务合并及源 SHA256 | 21 |
| full 验真及输出 SHA256 | 37 |
| scan | 3 |
| 输入指纹 | 33 |
| SigLIP 四卡 | 123 |
| finalize，含 1024 条抽检 | 79 |
| 4×4 pack / verify | 3 / 1 |
| 8×8 pack / verify | 7 / 3 |

清理前 `du -sb`：合并目录 13,985,577,056 B；source 17,100,944,336 B；4×4 1,484,079,996 B；8×8 5,933,220,768 B。这些含目录与元数据，不是纯特征 payload。按本次合并体量和 21 秒外推，正式约 790 GB 的同等吞吐合并约 20 分钟；这是预估，正式时间必须实测，不能把缓存和文件长度差异忽略。

## 版本、并行工作与资源观察

实际启动提交 `73d8748370e94e7eb4671739a0247b881a1d185a`。启动工作区有 HF 导出文件修改/新增，按用户“继续工作 忽略huggingface的任务 但是要注意git”排除。HF 任务期间另有提交；四个 worker 在结束处采集的 `git_commit` 均为 `d9e1f5d9cb89e5fedc3830173c5a3acb2984132e`，8×8 packer 记录 `0630a36d1ffc0bc10ca47ea0ac32ec0df645248f`。对实际消费的 `scripts/dataset/`（排除 hf_export）、`src/`、`pyproject.toml` 与 `uv.lock` 核对，较启动版本无改动；本轮未在 SigLIP 到 finalize 之间提交。不能把结束时读取的 HEAD 当作启动 HEAD。

GPU 仅使用 4–7，GPU 0–3 的训练持续运行。最近可用训练记录 epoch 55 为 873.8304346855721 samples/s、873.3227519989014 秒；本次短阶段没有覆盖完整新 epoch，不能据此宣称共处吞吐已通过完整 epoch 检验。正式阶段继续跟踪新增 epoch，低于 865.26 samples/s 时暂停本轮活动阶段。

## 归档和后续

清单 manifest sha256 为 `4d4468fd85a91ded763edb3a631887c33503c48ef75a14b0d62e3ae5cc01a8e9`，实际 canonical_order 为 BinFill、RouteStick、VideoRepick、VideoUnmaskSwap。records 保留两份 manifest、四份 episode map、source pin、MERGE_DONE、任务目标变体、source stats/provenance、两档 store_meta 和清洗日志。全部均属于 `4task-v2-smoke28`，不能与正式库同名文件混用。

按已授权清理本轮两个冒烟数据目录，源 extracted 和归档保留。下一步为 1600 集正式合并、建库及新 norm_stats 的训练可读性验证。本轮未执行 Wan、motion 或官方 sim 评估。
