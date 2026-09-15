# 1600 集新库 modulation 8×8 正式训练

> **已启动，训练进行中。** 起跑时间 `2026-09-15T05:48:27Z`，训练 HEAD 为 `dd07f18fc385b01eb52db7563fe5f202997b9706`。主副本代码已锁只读，运行状态见 [result.md](result.md)，实际启动和隔离记录见 [records/launch.actual.json](records/launch.actual.json)。[wandb 运行](https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/6ubtaf9l) 已在线同步。

正式 run_name 为 `v2-1600ep-m8x8-modul-b128-60k`，用户已在本轮再次明确确认。使用物理 GPU 4,5,6,7、batch 128、worker 8、fsdp 4、60k 步、warmup 5k、peak/decay lr 均为 5e-5、EMA 0.999、seed 42；wandb 开启，project 为 `robomme-framesamp`。超参来自新条目 `mme_vla_suite_b128_60k`，不修改既有默认条目，不做学习率线性缩放。

## 起跑版本与命令

训练从主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA` 的 clean HEAD 启动。功能源码含配置 `e0bcb45`、守卫 `81ba002` 和取证工具 `63858f5`。最终 `TRAIN_HEAD` 为提交全部起跑证据之后、建立开发副本之前的 HEAD；准确值由正式 runner 的 `TRAIN_HEAD=`、preflight 的 `CHECK_REPO_HEAD` 和运行记录固化。起跑前留档不嵌入自身提交 SHA，避免补写 SHA 后又改变 HEAD。

完整 shell 命令见 `records/prod-runner.sh`，smoke 命令见 `records/smoke-runner.sh`。两者均从主副本 `source scripts/training/paths.sh`，缓存位于 `v1-store/cache/`，只使用现有 `.venv` 的 `uv run --no-sync`。preflight 与训练共用同一个 `TRAIN_ARGS` 数组，显式传入绝对 Dataset/assets/checkpoint 路径。

实际启动额外在 runner 外层显式设置 `CUDA_CACHE_PATH`、`WANDB_DATA_DIR`、`XDG_DATA_HOME` 到主副本 `v1-store/cache/`，完整值已记录在 `launch.actual.json`，确保这些缓存也位于存储边界内。正式 preflight 的 25 条原文见 [records/preflight.log](records/preflight.log)，全部通过。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
TRAIN_HEAD=$(git rev-parse HEAD)
HC_SHA=$(sha256sum src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml | cut -d' ' -f1)
tmux new-session -d -s m8-prod "bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8-prod-runner.sh '$TRAIN_HEAD' '$HC_SHA'"
tmux has-session -t m8-prod
```

正式 runner 不带 smoke 的 20 步或关闭 wandb 覆盖。启动前确认 GPU 4–7 显存均为 0、run 根不存在、HEAD 相符且工作区干净。`PREFLIGHT=PASS n=25` 必须成立；计划旧稿的 23 项已按实际脚本纠正，全部 25 条检查保留。

## 数据、配置与输出

数据为本机新四任务 BinFill、RouteStick、VideoRepick、VideoUnmaskSwap，共 1600 集、605611 个执行样本。根为 `v1-store/datasets/4task-v2-1600ep-604f16da`；packed 布局 `framesamp-8x8-v1`，状态 `verified`，1192918 个帧行。训练设置 `MMEVLA_FRAMESAMP_SOURCE=<数据根>/source` 与 `MMEVLA_FRAMESAMP_MANIFEST=<数据根>/meta/episode_manifest.json`；motion 关闭。

| 对象 | SHA256 |
|---|---|
| `episode_manifest.json` 文件 | `df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482` |
| store 记录的 canonical manifest | `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918` |
| `framesamp-8x8/meta/store_meta.json` 文件 | `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` |
| 新库 `norm_stats.json` | `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173` |
| `perceptual-framesamp-modul-8frame-8x8.yaml` | `5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec` |

norm_stats 显式来自 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`。预训练权重为本机 `v1-store/models/openpi-assets/checkpoints/pi05_base/params`，tokenizer 位于同一模型根。没有下载原始 H5，也不依赖环境 A 的 turbo 产物。

checkpoint 根为 `v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-modul-b128-60k`，预期目录 `{5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,59999}`。记录目录 `v1-store/bench/v2-1600ep-m8x8-modul-b128-60k`，日志 `v1-store/logs/v2-1600ep-m8x8-modul-b128-60k.log`；权重不进入 Git。

## 起跑前验证

V1 配置由 `e0bcb45` 固化。V2–V4 见 [守卫验证](../t8-c8-guard-s100/result.md)：3454 个样本/19 键零差异；100 步五标量、800 个训练索引、6 份输入和完整状态逐位一致。比较器另验证含预取的 872 个索引。

```text
DS_EQUIV=PASS samples=3454 keys=19 mismatches=0
GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=6 state_digest_rows=6
```

V5–V6 的最终判定引用 [modulation 追溯对拍](../m8-modul-retro/result.md)，必须两档 A/A 与 A/B 均通过后才能启动正式 run。V7–V8 的 smoke 判定在启动正式训练前补齐：20 步有限、退出 0、新库 norm_stats 随 checkpoint 保存、motion 关闭、参数树双向精确匹配 61/61，以及六条 modulation 专属参数路径全部存在。V9 逐字保留 25 条 preflight 结果；V10 在训练结束后检查代码未变。

2026-09-15 实施结果：V5–V6 两档四组比较均通过，`MODUL_RETRO=PASS profiles=2 comparisons=4 init_leaves=61 grad_leaves=38`。V7–V8 的 [smoke 验收](../smoke-m8x8-modul-20260915T054007Z/result.md) 也全部通过，实际运行 5 分 18 秒；本次已按完整 run 留档。smoke 的 25 条 preflight 原文与 20 条 Step 行在其 `records/train.log`，正式起跑时再次检查最终 HEAD。

```text
GRAD_EQ=PASS kinds=3 leaves=38 mismatches=0
PREFLIGHT=PASS n=25
SMOKE20=PASS steps=20 finite=1 exit_code=0
PARAM_TREE_EXACT=PASS config=mme_vla_suite_b128_60k history_config=perceptual-framesamp-modul-8frame-8x8.yaml yaml_sha256=5b5ac2f85302d4d8 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
MEM_PARAMS=PASS n=6
```

## 与官方及上次 run 的关系

官方口径为 b64、80k、lr 5e-5、warmup 10k；本 run 为 b128、60k、lr 5e-5、warmup 5k，总样本数 7.68M，约新库 12.7 epoch。学习率按用户决定保持，不以 batch 翻倍推导新的覆盖值。

| 项 | `awsprod40k-b128-motion` | 本 run |
|---|---|---|
| 数据 | 旧四任务 400 集，101066 样本 | 新四任务 1600 集，605611 样本 |
| 记忆 | 32 帧 4×4 | 8 帧 8×8 |
| 接入与 motion | context，motion 开 | modulation，motion 关 |
| 步数 / lr | 40k / 1e-4 | 60k / 5e-5 |
| GPU / worker | 8 卡 / 16 | 4 卡 / 8 |
| norm_stats | `robomme-400ep` | `4task-v2-1600ep-604f16da` |

两条生产 run 不逐项可比。motion 关闭时模型新增参数层不创建，数据侧仍需本轮 `81ba002` 放宽 `FrameSampDataset.__init__` 的形制守卫；不能写成数据侧零改动。运行时等价主张限于已通过的 F2/F3 与固定 batch 追溯范围，不把模型对拍扩张为全部新库装配证明。

## 训练隔离、监控与收尾

启动前从 clean `TRAIN_HEAD` 以 `git clone` 建立 `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`；其 origin 指向同一 GitHub SSH 地址，沿用用户本轮批准的凭据方式。开发副本必须有自己的 `.venv`，`v1-store` 是指向主副本的可写 symlink。若开发副本已存在先问用户，不覆盖。

实际已完成：开发环境以 `uv sync --frozen --python <主副本>/.venv/bin/python` 建立，`sys.prefix` 和 `mme_vla_suite`、`openpi`、`openpi_client`、JAX 的导入路径均属于开发副本；两份 `.venv` 是不同实体目录。克隆的 `.git/info/exclude` 增加 `/v1-store`，只忽略本克隆中的符号链接。主副本 `.venv` 未同步或修改。

开发副本的仓库级 Git 配置还需显式沿用主副本的 SSH 身份：在开发副本执行 `git config core.sshCommand 'ssh -i ~/.ssh/id_ed25519_hongzefu -o IdentitiesOnly=yes'`。本次 clone 未继承该配置，首次推送被 GitHub 按默认 `yinpei-tri` 身份拒绝；按 AGENTS 第 11 条停止，用户确认后补设配置，留档提交 `739aca3` 已于 2026-09-15 成功推送。该配置只用于 GitHub 同步。

正式训练稳定起步后记录 `framesamp_dataset.py`、`train.py`、`history_pi0.py` 的 SHA256，并执行 `chmod -R a-w src scripts packages`。此后代码、留档编辑和环境安装均在开发副本进行，主副本不改文件、不 pull、不运行 uv sync/add/pip。开发副本禁止对共享数据执行带 `--force` 或破坏性输出根的命令。

训练连续推进超过 25 步后已执行只读锁定，目录权限为 `dr-xr-xr-x`，主副本 Git 状态仍干净；锁定摘要见 [records/lock_sha256.txt](records/lock_sha256.txt)。这证明起跑时的隔离已生效，训练结束后的 V10 复查仍待执行。

本次正式运行的 tmux 清单是 `m8-prod` 与 `m8-prod-dense`，smoke 使用 `m8-smoke`；不操作其他会话。训练有精确 PID 的 15 秒采样器并由 EXIT trap 回收；另用 500ms 密采记录前 30 分钟。每一级日志过滤均行缓冲，存活以 `tmux has-session` 判断。300 步后按实际稳定段重新估算 ETA；计划的约 65 小时仅为旧档位外推。利用率结论使用均值、0% 占比及慢步/非慢步分层，不以中位数作结论。

500ms 采样单独由 `timeout 1800` 管理；正常达到 30 分钟时，其专属 `.status` 文件会记录 `EXIT_CODE=124`，与正式训练日志的退出码是两个对象。训练 PID 和 15 秒采样器 PID 均已记录在 `launch.actual.json`。

保护边界：preflight 与 train 必须使用同一 argv 数组；chmod 不能阻止 root 或主动恢复写权限；环境保护依赖禁止主副本 uv 操作；失败可能留下半截 run 根与 wandb run。任何正式失败不自动覆盖、清空或复用名称，先把原因及残留交用户决定。

训练结束后恢复 `src/scripts/packages` 的用户写权限，确认 `git status --porcelain` 为空且三个源码 SHA 不变，记录 12 个 checkpoint、退出码、里程碑 loss、稳态吞吐及利用率统计，再归档 `result.md` 和日志/指标；不提交权重。评估不在本轮范围。
