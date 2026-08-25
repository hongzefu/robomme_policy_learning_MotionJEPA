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

## 验证复跑记录（2026-08-24，对抗审查修复后）

按 AGENTS.md 第 17 条（超 5 分钟的诊断 run 同等适用第 12 条留档要求）记录。
起因：对抗审查确认 13 项守卫缺陷并修复（`commitV1.12`–`V1.15` 与随后的 docs commit），
需重新取得「本地真值」资格并确认收紧后的判据不会误判已交付库。
完整分析与逐 key 数字见一致性报告第八节，本节只放坐标与判定。

| 项 | 值 |
|---|---|
| 起跑 commit | 第一层 `29c8c9b`（clean HEAD）；第二/三层同一 HEAD，工作区仅有**纯注释/文档**改动（后并入 `a6e2b3c`），无行为差异 |
| 执行环境 | 本机 sled-vail，`BENCH_GPU=1`（RTX 6000 Ada），tmux detached + `tee`；`RAW_H5_DIR` 保持默认 turbo 副本 |
| 第一层入口 | `legacy/step_local_baseline.sh` |
| 第一层判定 | **`COMPARE_RESULT=bitexact PASS` / `LAYER1_PASS` / `EXIT_CODE=0`**；12 episode / 3,862 步，9 key 全逐字节相同，`fails`/`errors` 空 |
| 第一层耗时 | 约 14 min（参照系 4.9 min + 四分片约 6 min + 对拍校准约 3 min） |
| 第二/三层入口 | `step2_verify.sh` |
| 第二/三层判定 | **`crossarch PASS` / `downstream PASS` / `VERIFY_PASS` / `EXIT_CODE=0`**；`has_nonfinite_any: false`，`pos_emb_*` 与 `ds_pos_emb` 逐位相同，`ds_frames_present` 376/376 |
| 第二/三层耗时 | 约 21 min（参照库 `--resume` 全跳过 `skipped=47`，耗时 0.5 s） |
| 存量补检 | 1600 个 `kept_indices.json` 全量 `json.loads`：**1600/1600 合法、坏 0、`.tmp` 残留 0**，11 s |
| 产物体积 | `v1-store/datasets/4task-gl` = 678 G（`du -sh`）；体积校准 `CALIBRATION_BYTES_PER_STEP=932154` |
| 日志 | `v1-store/logs/relayer1_20260824_2057.log`、`relayer23_20260824_2111.log`（`v1-store/` 不入 git） |
| 比对报告 | `v1-store/reports/layer{1,2,3}_*.json`（同上，不入 git；关键数字已转录进一致性报告 8.5） |

**结论：已交付的 1600 episode 数据集维持原状、不需重跑**——审查确认的缺陷全属
「守卫失灵」而非「计算出错」，五条证据见一致性报告 8.1。
