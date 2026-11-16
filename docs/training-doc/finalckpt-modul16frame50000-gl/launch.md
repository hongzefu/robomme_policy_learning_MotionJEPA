# ev1k4k 起跑留档（1024 / 4096 两个新 run × {50000, 79999} × 692 条，四个 run 共用）

结果见各 run 的 `result.md` 与主结果页 [`../binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)。
计划见 `~/.claude/plans/greatlake-8-48-cancel-concurrent-moon.md`（会话外计划文件，未进仓库）。

## 用户指令原话

1.「现在的eval做了哪些 我还刚才上传了1024 4096 如何用同样方式eval？目前还有老binfill混在结果里吗」
2.「给出在greatlake上跑的方案 还是8卡占48小时 跑完再cancel」
3.（AskUserQuestion 拍板）评 50000 + 79999 两档；run 名 `finalckpt-modul16frame{50000,79999}-gl` / `finalckpt-modul64frame{50000,79999}-gl`；
   `EXPECT_DEMO_INJECT` 补进透传白名单。
4.「立刻开始占卡」

## 被评权重

| run_name | checkpoint | POLICY | POLICY_CONFIG | budget |
|---|---|---|---|---|
| `finalckpt-modul16frame50000-gl` | `robomme-vla-modul-1024-80k-v1/50000` | `perceptual-framesamp-modul-16frame-8x8` | `mme_vla_suite_b128_80k` | 1024（16 帧） |
| `finalckpt-modul16frame79999-gl` | `robomme-vla-modul-1024-80k-v1/79999` | 同上 | 同上 | 1024 |
| `finalckpt-modul64frame50000-gl` | `robomme-vla-modul-4096-80k-v1/50000` | `perceptual-framesamp-modul-64frame-8x8` | 同上 | 4096（64 帧） |
| `finalckpt-modul64frame79999-gl` | `robomme-vla-modul-4096-80k-v1/79999` | 同上 | 同上 | 4096 |

来源 bucket：`HongzeFu/robomme-vla-modul-1024-80k-v1`（训练 run `v2-1600ep-m16x8x8-modul-b128-80k`）与
`HongzeFu/robomme-vla-modul-4096-80k-v1`（`v2-1600ep-m64x8x8-modul-b128-80k`），各 16 个 step、177 GiB。
两者冻结配置与 2048 run 只差 `budget` 一个键（`token_per_image 64`、`modulation`、`frame_sampling`、motion 关）。
`history_config.resolved.sha256` 分别为 `9d7a18fe…5390b` 与 `9df78b4d…f96a3`，均与各自 `motion_provenance.json.resolved_sha256` 一致；
`norm_stats.json` 四个 step 全部为 `856c75ea…d173`（与既有三条 run 同一份）。

下载：`download.sh` 每 bucket 先全量拉 50000（连根文件），再增量拉 79999；四次均 `CHECKPOINT_DOWNLOAD_PASS mode=sha256`。
日志 `v1-store/logs/policy-eval/dl-{1024,4096}-50000-79999.log`。

## 口径

与 `finalckpt-*`（上一轮）**逐字相同**：seed 7、`max_steps=2000`、`EPISODE_WALL_S=2400`、`EVAL_TIMEOUT=14400`、
1 卡 / 1 CPU / 24G、benchmark gitlink `b4e97f2`、候选库 identity `3b4de03a0b46…`、
`DEMO_PREFIX_STORE=v1-store/demo-prefix/binfill-3b4de03a0b46`、`EXPECT_DEMO_INJECT=1`、`ALLOW_RESUME=1`、`CHUNK_EPISODES=20`、
`MMEVLA_ENC_CHUNK` 不设。四条都不是 motion run，`gl_hold_pool.sh` 自动 `EXPECT_MOTION_STATS=0`。

692 条计划直接复制上一轮 `final-ckpt-queue/plans/`（不重新生成），`plan_manifest.json` identity `3b4de03a0b46…`、total 692。

## 本轮代码改动（commitV10.10）

1. `scripts/evaluation/check_checkpoint.py` 的 `RUNS` 表登记两个新 bucket（照 2048 条目，只改 `budget` 与 `resolved_sha256`）。
2. `scripts/evaluation/gl_hold_queue.sh` 的 PASS 透传白名单补 `EXPECT_DEMO_INJECT`——上一轮它靠 srun 默认带提交端环境才到达计算节点，
   不在显式白名单里；补上后不再依赖这一默认行为。

## 预标的 9 条卡死 episode

与上一轮同一份清单（`primary700-nomotion50k-gl/merged/errors.json`），同一份 692 计划下分布 s0/s1/s4/s6/s7，
四个 run 各预标 9 条，`preseed.py` 输出 4 行 `PRESEED_OK`。清单见 `records/preseeded-errors.json`。

## 占位作业

2026-09-25T17:02:55Z 在 `slurm-holds/` 提交 8 个占位作业（模板 `hold_job.sbatch`：1×A40 / 1 CPU / 24G / `--time=2-00:00:00` / `--gpu_cmode=shared`）：

| 作业名 | jobid | 节点 |
|---|---|---|
| ev1k4k-hold-1 | 61884194 | gl1510 |
| ev1k4k-hold-2 | 61884195 | gl1510 |
| ev1k4k-hold-3 | 61884196 | gl1513 |
| ev1k4k-hold-4 | 61884197 | gl1513 |
| ev1k4k-hold-5 | 61884198 | gl1517 |
| ev1k4k-hold-6 | 61884200 | gl1517 |
| ev1k4k-hold-7 | 61884201 | gl1526 |
| ev1k4k-hold-8 | 61884202 | gl1526 |

提交时账号下没有其他作业；8 个全部在 1 分钟内 RUNNING。**跑完合并后逐个 `scancel <jobid>` 释放**，见 result 页「收尾」。

## 64 帧显存冒烟（起正式前）

在占位作业 `61884194`（gl1510，A40 46 GB）上对 4096 的 **50000**（架构与 79999 相同，先下完的那个）跑
`smoke_demo_inject.py verify` 的历史最坏集 `BinFill/hard/202`（D=1107，首批 1108 帧），`MMEVLA_ENC_CHUNK` 不设：

    SMOKE_INJECT=PASS D=1107 exec_start_idx=1107 first_batch_frames=1108 enc_chunk=0 motion_enabled=false
                      first_add_buffer_seconds=6.0 history_feats=1108 steps_fed=0

1108 帧一次性过 SigLIP **不 OOM、6.0 s**（上轮 512+motion 的 179.9 s 里绝大部分是 sidecar 逐窗编码，非 motion 模型没有这一项）。
**口径限制**：该冒烟只覆盖 SigLIP 编码与帧路键集合，不走主干 `infer`；64 帧 / 4096 token 主干序列的显存由正式跑的首批分片实证——
manifest 把 `modul64frame` 两 run 排在前面，8 个 worker 起跑即全部落在 64 帧分片上，若 OOM 会在几分钟内以 `rc≠0` 暴露。
日志 `v1-store/logs/policy-eval/ev1k4k-smoke64.log`，产物 `v1-store/demo-prefix/smoke-1k4k/`（不进 git）。

## 执行

四个 checkpoint 起跑前本机 `check_checkpoint.py` 全部 PASS（`PARAM_TREE_EXACT n_model=61 motion_leaves=0`、
`HISTORY_CONFIG=PASS NORM_STATS=PASS MOTION_DISABLED=PASS`，budget 分别钉死为 1024 / 4096），2048/79999 回归仍 PASS；
`test_control.py` 6 项 OK、`test_result.py` 4 项 OK；`DRY_RUN=1` 干跑 `gl_hold_pool.sh` 抢完 40 单元、顺序为 64frame-79999 → 64frame-50000 → 16frame-79999 → 16frame-50000。

登录节点一个 detached tmux `ev-1k4k-pool` 跑 `v1-store/evaluation/ev1k4k-queue/pool-run.sh`，内部对 8 个 jobid 各起一个
`gl_hold_pool.sh` worker（MANIFEST 模式），40 行 manifest 见 `records/manifest.tsv`。汇总日志
`v1-store/logs/policy-eval/ev1k4k-pool-summary.log`，结束判定行 `EV1K4K_POOL_ALL_DONE workers=8 failed_workers=N`。
起跑 HEAD 为本 commit（clean）；**评测期间不再 commit**（`gl_eval_shard.sbatch` 的 `EXPECTED_GIT_HEAD` 闸会拒后续单元）。

执行结果、验收表与 scancel 记录写在 `result.md`。

## 产物

`v1-store/evaluation/ev1k4k-queue/{manifest.tsv,plans/,preseed.py,pool-run.sh,claims/,done/}`（不进 git）；
本目录 `records/` 存 `manifest.tsv`、`preseeded-errors.json`、`shard-checks.json`、`merged/`。
本轮起过的 tmux 会话（本机）：`ev-1k4k-dl-1024`、`ev-1k4k-dl-4096`；（登录节点）：`ev-1k4k-pool`。
