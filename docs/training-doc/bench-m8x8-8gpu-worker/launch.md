# bench-m8x8-8gpu-worker — 起跑记录

## 目的

为 `0916-motion-modul-8x8-plan.md` 的正式 80k run 改用 8 卡（用户 09-17 拍板：mesh (1,8)——runner 命令行 `--fsdp-devices 8` 覆盖（config 默认 4 不动），8 卡可见、per-device batch 16）钉死两个未知数：

1. **8 卡 mesh (1,8) 在本布局（8 帧 × 8×8、modulation、1600ep 新库）上的真实步时**。仓库既有 8 卡数字只有旧布局（32 帧 context、400ep）的 `bench-b128-util` B4/B5 与 `awsprod40k-b128-motion`，本布局从未在 8 卡上起过。
2. **`num_workers` 取 8 还是 16**。bench-b128-util 实测 8 卡时供给成瓶颈，但只测过 8 卡 w16 / w32，从未测过 8 卡 w8；4 卡基线 `v2-1600ep-m8x8-modul-b128-60k` 用 w8（2.34 s/步、util 均值 72%）。

用户在 (1,8) 两档跑完后追加一组 **mesh (2,4) 对照**（`fsdp_devices=4`、取 (1,8) 里最快的 w16），共三档。三档都用**改动前的关闭态配置**（无 motion 节），所以绝对步时是开启态的下界（开启态每样本多交付 658 KB、worker 多常驻 219 MB motion 表）；w8 与 w16 的相对排序可借用。

## 环境

- 环境 B（AWS 单机）。仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`。
- 8 × NVIDIA A100-SXM4-80GB，96 vCPU，1121 GB RAM，起跑时 8 卡显存占用全为 0。
- 存储介质：**AWS 本地 NVMe RAID（`/dev/md0`）**。不得与环境 A 或旧本机 `/data` 数字混比（AGENTS.md 第 13 条）。

## commit

起跑 commit：`9cbb94e3779cd904fff051a3fa38f156c5b8c47c`（= 本 launch.md 入库的 docs commit；clean HEAD，`CHECK_REPO_HEAD` / `CHECK_REPO_CLEAN` 由 preflight 实测 PASS）。第一次以 `b4397d5` 起跑时 launch.md 尚未入库，被 `CHECK_REPO_CLEAN` 拦下（`records/driver.f8.repoclean-fail.log`），随后入库重起。本轮无代码改动，runner 与驱动脚本落 `v1-store/logs/`（副本见 `records/`）。

## 配置档位

config 条目 `mme_vla_suite_b128_60k`（与 4 卡基线同一条目），命令行覆盖：`--num-train-steps 300 --log-interval 10 --no-wandb-enabled --num-workers {8,16} --fsdp-devices {8,4}`（fsdp 值是 runner 第 4 个参数，默认 8）。

- history config：`perceptual-framesamp-modul-8frame-8x8.yaml`（sha256 `5b5ac2f8…`，基线同一份，motion 关闭）
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，`--fsdp-devices 8` ⇒ `sharding.make_mesh` 给 mesh (1,8)（纯 FSDP，参数与优化器状态每卡持 1/8）；global batch 128、per-device 16
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`OMP_NUM_THREADS=1`、`MMEVLA_JAX_CACHE_DIR=v1-store/cache/jax/bw-8gpu-f8`（两档共用编译缓存；稳态窗口从 it100 起、不含编译）
- 数据集 `v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-8x8`；norm_stats sha256 `856c75ea…`（基线同一份）
- `TRAIN_RECORD_DIR=v1-store/bench/bench-m8x8-8gpu-f8-w<N>`，`metrics.jsonl` 每 10 步一行带 `wall_time`，是步时的唯一数据源
- 两档**串行**（bench-b128-util 实测两条 run 并行互扰 +11–13%）

## 命令

tmux 会话 `bw-8gpu-f8`（(1,8) w8→w16 串行）与 `bw-8gpu-f4`（(2,4) w16 对照）。本轮起过的会话清单：`bw-8gpu`（(2,4) w8 首次尝试，已中止）、`bw-8gpu-f8`、`bw-8gpu-f4`，三者都随驱动脚本自然退出，未执行任何 kill。用户自有会话 `0`、`1`、`claude-private`、`codex`、`codex-repo`、`codex2` 一律不动）。

```bash
tmux new-session -d -s bw-8gpu-f8 "bash v1-store/logs/bw-8gpu-driver.sh"
# 驱动：for W in 8 16 → bash v1-store/logs/bw-8gpu-runner.sh $W 9cbb94e3779cd904fff051a3fa38f156c5b8c47c 5b5ac2f8… 8
# 对照：tmux new-session -d -s bw-8gpu-f4 "bash v1-store/logs/bw-8gpu-f4-driver.sh 16"  → runner 第 4 参数 4（mesh (2,4)）
# runner：GPU_IDLE（8 卡 memory.used 求和为 0）→ preflight_train_launch.py（与 train 共用同一 TRAIN_ARGS）
#        → nvidia-smi --id=0..7 -lms 500 密采 → train.py；日志 v1-store/logs/bw-8gpu-f8-w<N>.log，收尾 EXIT_CODE=
```

判定行：`PREFLIGHT=PASS n=25`、`EXIT_CODE=0`（每档）、`DRIVER_ALL_DONE`；分析脚本 `records/bw_analyze.py` 输出 `BENCH_8GPU workers=<N> step_mean_s=… util_mean=… zero_share=… slow_util=… other_util=…`。

判读口径（AGENTS.md 第 16 条）：稳态窗口 = `metrics.jsonl` 的 step 100 → 末行；util 以 500 ms 密采在该窗口内的**均值、0% 占比、慢步（区间步时 > 1.5× 中位）/非慢步分层均值**为准，禁止以中位数作结论。

## 清理

两档跑完后删除 run 产物 `v1-store/train-runs/mme_vla_suite_b128_60k/bench-m8x8-8gpu-f{8-w8,8-w16,4-w16}/`（末步 ckpt）；`v1-store/bench/bench-m8x8-8gpu-f{8-w8,8-w16,4-w16}/` 的 `metrics.jsonl` 与密采 csv 拷入 `records/` 后删除；tmux 会话随驱动退出自然结束，无需 kill。

## 首次尝试（mesh (2,4)，已中止）

06:47:02Z 曾以 `fsdp_devices=4`（mesh (2,4)）起过 w8 档（会话 `bw-8gpu`，run `bench-m8x8-8gpu-w8`），preflight PASS、编译通过、第 0 步 loss 0.0894 与 4 卡基线第 0 步 0.08937 一致，跑到约 it8 时用户改决策为 mesh (1,8)，06:51Z 经 Ctrl-C 中止，run 产物与 bench 目录已删。保留证据：`records/bw-8gpu-driver.mesh24-aborted.log`（含 XLA `Involuntary full rematerialization` 告警——(2,4) 下 `{devices=[8,1,1]}` 到 `{devices=[1,1,4,2]}` 的重分片，`mem_encoder.py` / `history_pi0.py` 多处），供与 (1,8) 对照。
