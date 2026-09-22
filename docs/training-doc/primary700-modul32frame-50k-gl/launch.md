# primary700-modul32frame-50k-gl 起跑留档（32 帧 / 感知预算 2048，modulation 注入，motion 关，step 50000）

**环境 A**（仓库工作副本 turbo，分支 `v2-eval-0918-motion`）。本 run 与既有的
[`primary700-nomotion50k-gl`](../primary700-nomotion50k-gl/launch.md)（8 帧 / 512）、
[`primary700-motion50k-gl`](../primary700-motion50k-gl/launch.md)（8 帧 / 512 + motion）构成 50000 步上的三方对照，
回答「把感知上下文从 8 帧 / 512 提到 32 帧 / 2048，能不能顶上 motion memory 的收益」。

## 用户指令原话（按时间）

1. 「新的2048无motion也放在bucket上了 也进行eval」「query 4张卡的job来跑」
2. 「不要动现有job」「你只能用新的」
3. 「不要提交4卡！！！你这个是eval 尽可能拆分容易排队！！和上次我提交的eval用的job规模一致！只是不再提交10卡！」
4. 「slurm job可以先发了」
5. 「不要以提交job的形式 改为占用4卡 48h的方案」
6. 「优先只跑ctx2048」「跑完报告 并且同步跑motion80k」
7. 「ctx2048这个是modulation注入？？」「不要叫ctx2048会引起歧义！！！」→ 改名为本 run 名
8. AskUserQuestion 拍板：评 50000（与已跑完两条同步数）；motion-80k「只增加带motion的 79999 其他的先不测」；
   9 条卡死 episode「预先标 error 跳过」；run 名 `primary700-modul32frame-50k-gl`

## 命名说明（为什么不叫 ctx2048）

初版 run 名用过 `primary700-ctx2048-50k-gl`，用户指出会引起歧义并要求改名。原因是 `ctx` 与本仓库另一种**注入方式**撞名：
`src/mme_vla_suite/models/config/robomme/` 下 `perceptual-framesamp-**context**-*.yaml` 与
`perceptual-framesamp-**modul**-*.yaml` 是两套并列配置。**本 run 是 modulation 注入**，与对照的另两条完全一致
（三条 run 的 run 内快照 `integration_type` 都是 `modulation`，逐条核实过），差别只在感知预算与 motion 开关。
现名 `modul32frame` 与 `POLICY=perceptual-framesamp-modul-32frame-8x8` 逐字对应。改名时结果目录、队列 claims/done、
manifest、预标记清单、分片日志一并同步改掉。

## 被评权重

- bucket `HongzeFu/robomme-vla-modul-2048-80k-v1` 的 `50000/`（run `v2-1600ep-m32x8x8-modul-b128-80k`，16 个 step、178 GiB，
  80,000 步跑满正常退出）。本地落点 `v1-store/models/robomme-vla-modul-2048-80k-v1/`，`download.sh` 全量模式按 bucket 的
  `SHA256SUMS.pre.txt` 校验：`CHECKPOINT_DOWNLOAD_PASS mode=sha256 files=29`（`records/dl-modul32frame-50000.txt`）。
- run 快照 `history_config.resolved.yaml` sha256 `91512306b3aaaaa541d248bc5dfaf8be2849632cd3b9e89a226207c09c6501c0`
  （与 bucket 根 `history_config.resolved.sha256`、`motion_provenance.json.resolved_sha256` 三方一致）：
  `budget: 2048`（= 32 帧 × `token_per_image: 64`）、`integration_type: modulation`、`perceptual_memory.type: frame_sampling`、
  `streaming_obs_horizon: 16`、`memory_token_dim: 1024`，**无 `motion:` 节**（`motion_enabled: false`、`encoder`/`vae` 均 null）。
  与 8 帧档的 `perceptual-framesamp-modul-8frame-8x8.yaml` 逐键比对，**唯一差异就是 budget 512 → 2048**。
- `norm_stats.json` sha256 `856c75ea…`，与另两条 run 同一份（bucket 全 16 个 step 也都是这个值）。
- `check_checkpoint.py`：`PARAM_TREE_EXACT=PASS n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0 motion_leaves=0`、
  `HISTORY_CONFIG=PASS NORM_STATS=PASS MOTION_DISABLED=PASS config=mme_vla_suite_b128_80k step=50000 budget=2048`。
  **61 叶与 8 帧档完全相同**——budget 只改运行期序列长度，不进参数形状（详见下节）。

## 为什么 budget 2048 不需要改模型侧代码

起跑前做过一轮只读调查，结论是评估链路对 budget 完全由快照驱动，只有一处硬编码要改：

- **参数树不变**：budget 只出现在「序列长度」位置——`history_pi0.py::inputs_spec` 的 `static_image_emb [b, budget, 2048]`、
  `percep_mem.py::PerceptualMemory.__call__` 的运行期 `assert … == self.config.budget`。位置编码表 `PosEmb3D` 是纯函数算出的**无参数**张量，
  行数只由 `max_steps` 决定；modulation 下记忆 token 走交叉注意力、不拼进主序列，RoPE 位置号也不受影响。所以 61 叶、逐叶形状全不变。
- **取帧无上界风险**：`FrameSampMemory.get_frame_sampling_indices` → `even_sampling_indices`，`max_size = budget // token_per_image = 32`，
  `step_idx ≥ 32` 时 `linspace(0, step_idx, 32)` 等距重采样，**永远恰好 32 帧**，没有 motion 那种「超预算即 raise」的零截断契约。
  唯一硬上界是 `FrameSampMemory.add_buffer` 的 `max_steps=4096`，本轮最大全域帧号（502 demo + 2000 exec + 16）≈ 2518，留 1.6× 余量。
- **唯一阻塞**：`check_checkpoint.py` 原本写死 `assert cfg.budget == 512`，会在起 server 前把分片打死。已改为按 RUNS 条目取期望 budget
  （commitV10.5），保持「budget 是模型语义参数、参数树对它失明、必须显式钉死」的原则。
- `POLICY` 全链路只当结果目录层级名与 `--args.policy_name`，没有任何按名字推断 motion 的逻辑，故 `…-modul-32frame-8x8` 安全。

## 口径（与对照两条逐字相同）

同一套 10 个分片计划（`primary700-gl/plans/shard{0..9}.json`，`plan_manifest` 身份 sha `3b4de03a…`）、策略 seed 7、
`max_steps=2000`、`EPISODE_WALL_S=2400`、`EVAL_TIMEOUT=14400`、`CHUNK_EPISODES=20`、benchmark gitlink `b4e97f2`、
每片 1×A40 / 1 CPU / 24G。与对照的差异只有四个值：

| | 本 run | 8 帧 / 512 | 8 帧 / 512 + motion |
|---|---|---|---|
| `RUN_NAME` | `primary700-modul32frame-50k-gl` | `primary700-nomotion50k-gl` | `primary700-motion50k-gl` |
| `CKPT` | `…/robomme-vla-modul-2048-80k-v1/50000` | `…/robomme-vla-modul-60k-v1/50000` | `…/robomme-vla-modul-motion-80k-v1/50000` |
| `POLICY` | `perceptual-framesamp-modul-32frame-8x8` | `perceptual-framesamp-modul-8frame-8x8` | `perceptual-framesamp-modul-8frame-8x8-motion` |
| `POLICY_CONFIG` | `mme_vla_suite_b128_80k` | `mme_vla_suite` | `mme_vla_suite_b128_80k` |

（`POLICY_CONFIG` 的两个名字在评估侧等价：四条 TrainConfig 的 `model` 字段完全一致且 `history_config=None`，模型超参全部来自 run 快照；
norm_stats 取自 checkpoint 自带的那份。本 run 与 motion run 同属 b128 / 80k / 1600ep 新库，故用 `mme_vla_suite_b128_80k`。）

## 预先标记的 9 条卡死 episode

`VideoRepick/easy 189`、`VideoRepick/medium 155 / 167 / 172`、`VideoRepick/xhard 168 / 197 / 201`、
`VideoUnmaskSwap/xhard 139 / 164`——前三轮（老基线 59999、8 帧 512 @50000、motion @50000）逐条相同，卡在环境
`make_env`/reset 的 native 调用里，`eval.py` 的 `SIGALRM` 单集墙钟打不断，上一轮靠看门狗现场砍 + 续跑，代价是 13 次重跑。
本轮经用户批准**起跑前**按分片把它们写进各自 `progress.json` 为 `"error"`（配 `ALLOW_RESUME=1`），清单见
`records/preseeded-errors.json`，分布 s2=1 / s4=4 / s5=1 / s8=1 / s9=2，合计 9 条。

机制已验证：`eval.py::setup_save_directory` 只在 `--overwrite` 时 `rmtree`（本链路不传），`setup_log_dict` 在带
`--args.episode_plan` 时**不**清除 error 条目，`check_shard.py` 把 error 排除在「视频数 == 非 error 集数」之外。
效果：本轮 10 片一次过，**零重跑、零新增卡死**（上一轮同样 9 条要花 13 次续跑、约 1.5 h）。

## 作业与执行

用户先后否决了两种形态（`--array=0-19%4` 的细粒度排队、4 卡整包），最终定为**自己占 4 张卡 48 h**：
新提交 `ev2048-hold-1..4`（61721503–61721506），各 1×A40 / 1 CPU / 24G / 2 天 / `--gpu_cmode=shared`，
全部立即 RUNNING（gl1513 ×3、gl1523 ×1）。用户原有的 4 个 `hold-4cpu-*` 作业一概不碰。
登录节点 4 个 worker（tmux `ev-2048-w1..4`）跑 `gl_hold_pool.sh <jobid>` 的 **manifest 模式**从共享队列抢单，
另一个 tmux `ev-2048-watchdog` 兜新出现的卡死（全程未触发）。

HEAD `e9c46c6`（clean）。20 行 manifest（本 run 10 片在前、motion80k 10 片在后）见 `records/manifest.tsv`。

| 分片 | 作业 / 节点 | 起 → 止（UTC） | 分钟 | 评/成功/error |
|---|---|---|---|---|
| s0 | 61721504 / gl1513 | 16:32 → 17:10 | 38 | 70 / 23 / 0 |
| s1 | 61721506 / gl1523 | 16:32 → 17:13 | 41 | 70 / 26 / 0 |
| s2 | 61721503 / gl1513 | 16:32 → 17:09 | 37 | 69 / 27 / 1 |
| s3 | 61721505 / gl1513 | 16:32 → 17:08 | 36 | 70 / 23 / 0 |
| s4 | 61721505 / gl1513 | 17:08 → 17:44 | 36 | 66 / 22 / 4 |
| s5 | 61721503 / gl1513 | 17:09 → 17:44 | 35 | 69 / 22 / 1 |
| s6 | 61721504 / gl1513 | 17:10 → 17:49 | 39 | 70 / 24 / 0 |
| s7 | 61721506 / gl1523 | 17:13 → 17:53 | 40 | 70 / 30 / 0 |
| s8 | 61721503 / gl1513 | 17:44 → 18:19 | 35 | 69 / 25 / 1 |
| s9 | 61721505 / gl1513 | 17:44 → 18:18 | 34 | 68 / 22 / 2 |

10 片全部 `SHARD_PASS`、rc=0，总墙钟 **1 h 47 min**（16:32:18Z → 18:19:46Z，4 卡跑 10 片，3 轮）。

## 意外与处置

1. **方案两次改向**：先按「尽可能拆分容易排队」提交了 `sbatch --array=0-19%4`（61721490），用户改口要占卡后 `scancel` 撤掉，
   改为 4 个 hold 作业 + manifest 模式 worker。
2. **`srun` 泄漏 `MANIFEST`**：worker 环境里的 `MANIFEST` 被 srun 整个带进 step，计算节点上的 `gl_eval_shard.sbatch` 误判成
   array 模式、因没有 `SLURM_ARRAY_TASK_ID` 秒退，20 个单元 2 秒内全 rc=1。本机 `DRY_RUN=1` 干跑测不出来（那条路径不走 srun）。
   修为 `/usr/bin/env … -u MANIFEST`（commit `e9c46c6`），清空队列后重起，分片本体当时未启动、预标记完好。
3. **改名**：见「命名说明」。
4. **全量作业被取消**：18:24:09Z 账号下全部作业被 `CANCELLED by 114466650`（含用户自己已跑 20 h 的 4 个 `hold-4cpu-*`），
   本 run 当时已全部跑完合并、不受影响；同批的 motion80k 被打断，另行续跑。

## 产物

`records/merged/*.json`（合并结果）、`records/shards/s*-{check.json,checkpoint.txt,server.excerpt.txt,driver.excerpt.txt}`、
`records/preseeded-errors.json`、`records/manifest.tsv`、`records/watchdog.sh.txt`、`records/dl-modul32frame-50000.txt`。
视频与 server 全量日志留在 `v1-store/`（不进 git）。结果见 [`result.md`](result.md)。
