# 4task-gl-400ep

四任务（`ButtonUnmask` / `VideoUnmask` / `ButtonUnmaskSwap` / `VideoUnmaskSwap`）
各 400 episodes 的预处理库，由 GreatLakes 上 8 个 1-GPU job array 并行产出。

## 状态

进行中。

## 正文在别处，本档案只放索引与不可由 Git 还原的结果

- **方案与实测报告**：[`docs/v1-gl-dataset-consistency-report.md`](../../v1-gl-dataset-consistency-report.md)
- **CPU/mem 档位实测**：[`docs/v1-gl-resource-tier-bench.md`](../../v1-gl-resource-tier-bench.md)
- **实现**：[`scripts/data-preprocess-GL/`](../../../scripts/data-preprocess-GL/README.md)

## 关键坐标

| 项 | 值 |
|---|---|
| 输入（原件，永久保留） | `/data/hongzefu/robomme_data_h5_v2_4env400ep`（321 GB） |
| 输入（集群暂存，验收后删） | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_data_h5_v2_4env400ep` |
| 产物 | `<repo>/v1-store/datasets/4task-gl/` |
| 分片清单 | `<repo>/v1-store/episode_manifest.json` |
| 比对报告 | `<repo>/v1-store/reports/layer{1,2,3}_*.json` |

## 起跑记录

> 待回填：commit、清单 sha256、array/finalize jobid、档位、walltime、实测耗时、
> 各层判定行、产物体积。
