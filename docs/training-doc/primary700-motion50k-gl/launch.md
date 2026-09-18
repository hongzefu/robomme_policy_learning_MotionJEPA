# primary700-motion50k-gl 起跑留档（冒烟阶段，正式 700 条未起跑）

**环境 A**（仓库工作副本 turbo，分支 `v2-eval-0918-motion`）。本档只记录到「集群两集冒烟通过」为止；
正式 700 条评测按用户 2026-09-18 指令「只做到集群冒烟为止之后收尾commit并且记录」**本轮不起跑**，起跑方案见文末。

## 用户指令原话（按时间）

1. 「我需要打通带motion的eval 找最新的实现 …/tree/v2-motionmem 已经训到50k了在huggingface bucket
   HongzeFu/robomme-vla-modul-motion-80k-v1 跑通50k的eval 改为使用 [gpu-hold-01..08 八个已占用作业] 这几个已经占用的直接跑
   参考 robomme-eval-GL 的直接跑方式 注意不能使用gpu01 这两个其他的job要用 告诉我你两个repo需要修改什么
   benchmark需要增加branch 因为policy learning的branch改变了」
2. 「注意这里我意思是不能使用hold的卡1 2」
3. 「已经都上传好了」（bucket `HongzeFu/motionjepa-wan-full1600-72ep-v1` 全部 36 个 epoch 17.26 GB；VLA bucket 传到 10/16 step）
4. AskUserQuestion 拍板：provenance 仅放行 `gpu_name / compute_cap / sm_count` 三个硬件键；正式 run_name `primary700-motion50k-gl`
5. 「只做到集群冒烟为止」「只做到集群冒烟为止之后收尾commit并且记录」

计划正本：`~/.claude/plans/motion-eval-https-github-com-hongzefu-r-cozy-ullman.md`（已批准）。

## 被评权重与口径

- bucket `HongzeFu/robomme-vla-modul-motion-80k-v1` 的 `50000/`（run `v2-1600ep-m8x8-modul-motion-b128-80k`，训练 commit
  `2f10473`，8×A100，训练时仍在跑：留档时 55.8k/80k）。本地落点 `v1-store/models/robomme-vla-modul-motion-80k-v1/`，
  25 个文件（`50000/` 20 个 + 根 5 个）按 bucket tree 的 size 逐文件核对，本地 sha256 见 `records/vla-50000-SHA256SUMS.local.txt`
  （bucket 根没有 SHA256SUMS.pre.txt，60k bucket 有）。
- run 快照 `history_config.resolved.yaml` sha256 `9650f225fe42b802262f93b71139cebbda934dc5ccfaa849fa4dadaf30d73110`，与
  `history_config.resolved.sha256`、`motion_provenance.json.resolved_sha256` 三方一致；去空白后与仓库
  `perceptual-framesamp-modul-8frame-8x8-motion.yaml` 逐行相同：`budget 512 / token_per_image 64 / modulation /
  motion.enabled true / motion.budget 160 / stride 16 / window_frames 33 / demo_min_real_frames 17 / repeat_last /
  source_run wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder / online_gpu 4`。
- `norm_stats.json` sha256 `856c75ea…`，与 60k 基线同一份。
- 口径与非 motion 基线 `primary700-gl`（27.57%，`v1-store/evaluation/primary700-gl/merged/log.json`）逐项相同：
  同一套 10 个分片计划、策略 seed 7、`max_steps=2000`、benchmark gitlink `b4e97f2`、注入候选 identity `3b4de03a…`。
  只有单集墙钟 `EPISODE_WALL_S` 从默认 900 放宽到 2400（它只兜死锁不属口径，见下方计时）。

## 两个仓库的改动

- **本仓库** `c0bb5eb`（commitV10.3）+ `f8697f8`（fix）：`src/` 两个口子（`MMEVLA_MOTION_PROV_RELAX` 放行开关、
  `MMEVLA_MOTION_ONLINE_GPU` sidecar 卡号覆盖）+ 评估脚手架参数化 + `gl_hold_queue.sh` + 资产锁指向新 bucket。
  详见两条 commit message 与 `scripts/evaluation/README.md`「motion 轮」一节。`src/` 里 motion 在线推理路本来就在
  （eval 分支基点 `f6915f2` = `v2-motionmem` tip；`origin/v2-motionmem` 多出的 7 个 commit 不含 `src/`），未从训练分支搬代码。
- **benchmark fork** `hongzefu/robomme_benchmark_MotionJEPA`：从锁定 `b4e97f22fe007078e297205898c07c1acbc69165` 新建并推送
  `PolicyEvalThirdParty-v2-eval-0918-motion`（与 `PolicyEvalThirdParty-v2-eval-0917` 同一 commit）。**未改 benchmark 源码**：
  帧已是 256×256 uint8（`RENDER_PASS`），`exec_start_idx` 由主仓库 `examples/robomme/eval.py` 算并下传。主仓库 gitlink 不变，
  `verify_sources.py` 的 `PINNED` 不变，`SOURCES_PASS`。

## 资产落位（全部在 NFS 副本 `v1-store/`，集群节点可见）

| 资产 | 落点 | 来源与核验 |
|---|---|---|
| sidecar venv | `v1-store/venvs/wan` | `setup.sh wan`，NFS 3.11.14 解释器，`WAN_VENV_OK torch=2.9.0+cu128 diffusers=0.39.0`，约 4 分钟（`records/setup-wan.log`） |
| Wan2.1 VAE | `v1-store/cache/hf/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers` | 本机 `/data` 缓存 rsync（507 MB），`fetch_assets.py verify --level full` ✓ |
| MotionJEPA encoder | `v1-store/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a/{checkpoint_epoch_72.pt,config.yaml}` | bucket `HongzeFu/motionjepa-wan-full1600-72ep-v1`，`download.sh` 按其 `SHA256SUMS.pre.txt` 校验 `CHECKPOINT_DOWNLOAD_PASS mode=sha256 files=2`；`fetch_assets.py verify` ✓（sha `0c198629…` / `4a505440…`） |
| VLA 50000 | `v1-store/models/robomme-vla-modul-motion-80k-v1/` | `download.sh … 50000`，`CHECKPOINT_DOWNLOAD_PASS mode=size files=25`，11.07 GiB（`records/dl-vla-50000.log`） |

`check_checkpoint.py`（服务端环境、CPU）对 50000：`BUDGET_CONSISTENT=PASS budget=160`、
`PARAM_TREE_EXACT=PASS n_model=65 n_ckpt=65 missing=0 extra=0 shape_mismatch=0 motion_leaves=4`、
`HISTORY_CONFIG=PASS NORM_STATS=PASS MOTION_ENABLED=PASS config=mme_vla_suite_b128_80k`。

## 冒烟一：本机 RTX 6000 Ada（GPU1）——被 provenance 守卫拦下，属预期

tmux `ev-mot-smoke-local`，HEAD `c0bb5eb`，2026-09-18 21:27:21Z → 21:30:13Z，`EXIT_CODE=1`。
SOURCES / RENDER / BUDGET_CONSISTENT / PARAM_TREE 全过，sidecar 握手成功（warmup 3.20 s），
`MOTION_PROV_RELAXED` 恰 6 行（gpu_name Ada≠A100、compute_cap 8.9≠8.0、sm_count 142≠108，vae / encoder 各一组），
随后 **`driver: sidecar='570.211.01' store='595.71.05'` 硬拒**——`driver` 不在用户批准的三键内，守卫按设计 raise。
结论：本机驱动与训练侧不同，本机不能跑 motion sidecar（除非另行批准放行 `driver`）；集群 gl1523 驱动 595.71.05 与训练侧一致，
冒烟改在集群做。日志 `records/smoke-local-*.log`。

## 冒烟二：集群 gpu-hold-08（job 61495578，gl1523，1×A40 / 1 CPU / 24G）

「直接跑」= 登录节点 tmux `ev-mot-smoke-gl` 内 `gl_hold_queue.sh 61495578 primary700-motion50k-smoke-gl <ckpt> <plans> <HEAD> smoke`
→ `srun --jobid=61495578 --overlap --exact --gpu_cmode=shared --cpus-per-task=1 --gpus-per-node=1 --time=01:30:00`
→ `gl_eval_shard.sbatch`（`SHARD_INDEX=smoke`）→ `run_shard.sh`。环境变量：
`CHUNK_EPISODES=1 POLICY=perceptual-framesamp-modul-8frame-8x8-motion POLICY_CONFIG=mme_vla_suite_b128_80k EXPECT_CKPT_ID=50000
EPISODE_WALL_S=2400 EVAL_TIMEOUT=14400 MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count`。
计划 `plans/shardsmoke.json`（= `smoke2.json`，与基线冒烟同两条：RouteStick/xhard 115 + BinFill/hard 153）。

- **attempt1**（HEAD `c0bb5eb`，21:30:51Z → 21:33:24Z，rc=127）：一路走到 `SERVER_READY` 之前的端口归属断言，
  `HEXPORT4781: command not found`——commitV10.3 前移 `collect_tree()` 时把 `HEXPORT=$(…)` 写丢了等号。已修（`f8697f8`），
  日志 `records/smoke-gl-attempt1-driver.log`。
- **attempt2**（HEAD `f8697f8`，clean，21:34:45Z → 21:42:42Z，**7 分 57 秒**，`EXIT_CODE=0`，`QUEUE_DONE failed=0`）：

```
SHARD_PASS {"pipeline_pass": true, "episodes": 2, "evaluated": 2, "successes": 1, "errors": [],
            "groups": {"BinFill/hard": {"episodes": 1, "successes": 0}, "RouteStick/xhard": {"episodes": 1, "successes": 1}}}
```

| 条目 | 结果 | demo 帧（exec_start_idx） | 视频帧 | 视频字节 |
|---|---|--:|--:|--:|
| RouteStick/xhard ep115 | **success** | 450 | 921 | 2,632,028 |
| BinFill/hard ep153 | fail | 0 | 1041 | 3,406,760 |

（非 motion 基线冒烟同两条是 xhard fail / hard success，方向相反；两集样本不构成任何结论。）

**sidecar 与 provenance**：握手 warmup 2.32 s；`MOTION_PROV_RELAXED` 恰 6 行——
`gpu_name 'NVIDIA A40' vs 'NVIDIA A100-SXM4-80GB'`、`compute_cap '8.6' vs '8.0'`、`sm_count 84 vs 108`（vae / encoder 各一组）；
driver 595.71.05、torch 2.9.0+cu128、cuda 12.8、cudnn 91002、diffusers 0.39.0、VAE state sha `9980d252…`、encoder ckpt sha
`0c198629…` 全部与训练侧相等。`SERVER_READY port=18305 own=1 tree=2253282 2253289 2253668 2253672`（后两级即 sidecar 的
uv → python），收尾 `cleanup()` 按树从叶到根 TERM，作业内无残留 step。

**计时（A40，`records/smoke-gl-server.excerpt.log` 的 TIMING 行）**：

| 环节 | 实测 |
|---|---|
| 首批 `add_buffer`（451 帧 demo，28 窗 + SigLIP 首编译） | 49,257 ms |
| 首次 `infer`（编译） | 43,274 ms |
| 16 帧 `add_buffer` 稳态（n=93） | median **1,709 ms**，mean 1,687，p90 1,715，max 1,722 |
| sidecar 单窗（sidecar 自报） | 1,636–1,643 ms |
| `infer` 稳态（n=94） | median **126 ms**，max 126 |
| 单集墙钟上界外推 | 2000 步 ≈ 125 窗 × 1.71 s + 125 × 0.126 s + 仿真 ≈ 4–5 min，远小于 2400 s |

对照：非 motion 基线在本机 Ada 稳态 `add_buffer` 29 ms / `infer` 68 ms；A100 独占卡 sidecar 1.46 s/窗。
A40 sidecar 每窗 1.64 s（关 TF32 纯 fp32），16 步一窗折合每步 +0.11 s。

**资源**：cgroup anon 峰 **15.71 GiB**（基线 s0 峰 11.1 GiB，sidecar torch 进程约 +4.6 GiB；作业上限 24G 有余）；
CPU 核当量 0.87（1 CPU 档，与基线 s0 的 0.90 持平）；GPU 显存峰 **35,756 MiB / 46,068**（policy 0.7 预留 + sidecar + 仿真）；
GPU util 峰 100%。采样 `records/smoke-gl-cgroup_mem_cpu.csv`。

冒烟产物已按 AGENTS 6 清理（`v1-store/evaluation/primary700-motion50k-smoke-gl` 删除），日志与 JSON 全部在 `records/`。

## 意外与处置

1. 第一次 rsync Wan VAE 时 Bash 工作目录残留在 `third_party/robomme_benchmark`，578 MB 落进了子模块的 `v1-store/`，
   主仓库 status 出现 ` M third_party/robomme_benchmark`。已删该误建目录、重新复制到主仓库路径并 verify 通过。
2. `test_control.py` 在 HEAD 上本就 5 项 `NameError: summarize`（`eval.py` 顶层新增了 `summarize/load_plan/episode_deadline`
   而测试抽取集合未更新），顺手补齐；`test_result.py` 只能以脚本方式跑（内部 `from check_result import …`）。
3. 本机驱动 570.211.01 ≠ 训练侧 595.71.05，本机冒烟被守卫硬拒（见冒烟一）。
4. attempt1 的 `HEXPORT` 等号丢失（见冒烟二）。
5. NFS 删目录偶发 `Directory not empty`（silly-rename 延迟），稍后重删即可。

## 正式起跑方案（已批准但本轮不执行）

可用作业 gpu-hold-03…08（用户：01、02 保留）。沿用 `primary700-gl/plans/shard{0..9}.json`，6 个作业串行排队：

| 作业 | jobid | 节点 | 分片 |
|---|---|---|---|
| gpu-hold-03 | 61495431 | gl1517 | s0 → s6 |
| gpu-hold-04 | 61495432 | gl1523 | s1 → s7 |
| gpu-hold-05 | 61495575 | gl1526 | s2 → s8 |
| gpu-hold-06 | 61495576 | gl1522 | s3 → s9 |
| gpu-hold-07 | 61495577 | gl1522 | s4 |
| gpu-hold-08 | 61495578 | gl1523 | s5 |

```bash
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
CKPT=$REPO/v1-store/models/robomme-vla-modul-motion-80k-v1/50000
PLANS=$REPO/v1-store/evaluation/primary700-motion50k-gl/plans      # 先 cp primary700-gl/plans/shard*.json 与 plan_manifest.json 进来
HEAD=$(git -C $REPO rev-parse HEAD)                                  # 必须 clean
ssh greatlakes "tmux new-session -d -s ev-mot-h03 \"cd $REPO; STEP_TIME=12:00:00 POLICY=perceptual-framesamp-modul-8frame-8x8-motion POLICY_CONFIG=mme_vla_suite_b128_80k EXPECT_CKPT_ID=50000 EPISODE_WALL_S=2400 EVAL_TIMEOUT=14400 MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count bash scripts/evaluation/gl_hold_queue.sh 61495431 primary700-motion50k-gl $CKPT $PLANS $HEAD 0 6 2>&1 | tee v1-store/logs/policy-eval/primary700-motion50k-gl.h03.queue.log\""
# h04: 61495432 → 1 7；h05: 61495575 → 2 8；h06: 61495576 → 3 9；h07: 61495577 → 4；h08: 61495578 → 5
```

按冒烟稳态外推每片 70 集 ≈ 1.5–3 h，两片串行的作业 3–6 h；作业 2 天时限已用约 2 h。
合并：`merge_shards.py v1-store/evaluation/primary700-motion50k-gl --policy perceptual-framesamp-modul-8frame-8x8-motion --ckpt 50000`。

## tmux 会话清单（清理唯一依据）

本机：`ev-mot-setup-wan`、`ev-mot-dl-vla`、`ev-mot-smoke-local`（均已自然退出）；登录节点：`ev-mot-smoke-gl`（已自然退出）。
正式起跑时将用登录节点 `ev-mot-h03…h08`。
