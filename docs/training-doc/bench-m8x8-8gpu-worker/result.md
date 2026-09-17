# bench-m8x8-8gpu-worker — 结果

三档全部 `PREFLIGHT=PASS n=25`、`EXIT_CODE=0`、300 步跑完、无 OOM。起跑 commit `9cbb94e`。判定行与原始数据在 `records/`（`judgement_lines.txt`、`analysis.json`、各档 `metrics.jsonl` / `gpu_util_lms500.csv` / `train.log.gz`）。

## 结果表（稳态 = `metrics.jsonl` 的 step 100 → 290，共 190 步；util 为 500 ms 密采在该窗口内的统计）

| 档 | mesh | worker | 步时均值 | 步时中位 | samples/s | util 均值 | 0% 采样占比 | 慢步区间 | 显存峰值 | 80k 外推 |
|---|---|---|---|---|---|---|---|---|---|---|
| f8-w8 | (1,8) | 8 | **1.847 s** | 1.648 s | 69.3 | 51.02% | 45.47% | 0/19 | 81,009 MiB | 41.0 h |
| f8-w16 | (1,8) | 16 | **1.818 s** | 2.114 s | 70.4 | 51.64% | 43.49% | 0/19 | 80,989 MiB | 40.4 h |
| f4-w16 | (2,4) | 16 | **1.833 s** | 2.274 s | 69.8 | 52.55% | 43.61% | 0/19 | 80,163 MiB | 40.7 h |
| 4 卡基线 `v2-1600ep-m8x8-modul-b128-60k` | (1,4) | 8 | 2.340 s | — | 54.7 | 72.39% | 25.39% | — | — | 52.0 h |

- 慢步定义：10 步区间步时 > 1.5 × 中位。三档都是 0 个慢步区间，所以「慢步 util」为空、非慢步 util 等于全窗口均值；分层判据在本次没有区分度，不构成结论依据。
- 中位数三档差异大（1.65 / 2.11 / 2.27）而均值相同：区间步时呈周期性交替（f8-w8 每 30 步里 20 步 1.64 s、10 步 2.40 s），是 DataLoader 多 worker 轮转造成的节律，不是负载差异。按 AGENTS.md 第 16 条以均值为准。
- 8 卡 util 逐卡均匀（51–53%），不存在某张卡拖后腿。
- step 290 的 loss：0.02320 / 0.02316 / 0.02319（4 卡基线同 step 无记录）。三档不逐位相同、统计上一致，符合「同配方、不同归约次序」的预期。第 0 步 loss 三档与 (2,4) 首次尝试都是 0.0894，与 4 卡基线 0.08937 一致，印证每步样本相同。
- 前 100 步（含编译）三档都是 179–180 s，共用 `MMEVLA_JAX_CACHE_DIR` 没有缩短它，说明这 3 min 的大头不在 XLA 编译缓存能覆盖的部分。
- XLA `Involuntary full rematerialization` 告警：(1,8) 两档 0 条；(2,4) 5 条（首次尝试那份日志更多）——编译期一次性告警，从步时看没有可测代价。

## 三条结论

**1. 8 卡比 4 卡只快约 22%，不是 2 倍。** 1.82 s vs 2.34 s。GPU 在 8 卡下 43–45% 的采样处于 0%，util 均值从 72% 掉到 51%——计算变快了，等数据的时间没变。

**2. mesh (1,8) 与 (2,4) 无差别，worker 8 与 16 无差别。** 三档相差 ≤ 1.6%，在 300 步 bench 的噪声范围内。mesh 之间通信量一阶相等（每卡每步约 12 GB，NVSwitch 下几十 ms）；worker 从 8 加到 16 只提升了 worker 侧供给峰值，而瓶颈不在 worker。

**3. 瓶颈是主进程的数据路，任何配置共用。** `src/openpi/training/data_loader.py::TorchDataLoader.__iter__` 对每个 batch 在主进程主线程上串行做「从 worker 经 Pipe 收 pickle」+「`jax.make_array_from_process_local_data` 拷到 8 卡」；8×8 帧库每样本约 3.7 MB（`image_emb` 64×2048 bf16 × 8 帧 = 2.1 MB，`pos_emb` 64×768 f32 × 8 帧 = 1.6 MB），b128 一个 batch 约 500 MB，bench-b128-util 实测该 Pipe 约 398 MB/s ⇒ 约 1.3 s/batch 加拷贝。每步 ≈ max(GPU 计算, 数据路)：4 卡时计算 2.3 s 长于数据路，GPU 忙；8 卡时计算约 1.2 s 短于数据路，GPU 等。与 modulation / context、motion 开关无关（喂进去的 batch 相同；motion 开启再加 658 KB/样本，数据路只会更长）。

## 对 0916 motion 计划的输入

- 正式 80k run 用 8 卡：步时按 **1.82 s（关闭态实测）** 计为下界，开启态多交付 84 MB/batch 会再慢一些，80k ≈ 40.4 h + 编译与 16 次 ckpt ≈ **41 h**（4 卡 52 h）；300 步处仍按第 16 条重估。
- mesh 与 worker 的取值对速度无影响，按实现简洁度与先例定即可；本次用户拍板 (1,8)（`--fsdp-devices 8` 命令行覆盖）+ `--num-workers 16`，config 默认 `fsdp_devices=4` / `num_workers=8` 不动。
- 想真正吃满 8 卡要改数据路（后台线程做 device 预取；`pos_emb` 全库只有 2,304 行，可传行号在 device 上索引，每样本省 1.6 MB），属 dataloader 改动，按 AGENTS.md 第 18 条另立任务，不夹进 motion 这轮。

## 口径声明

存储介质 AWS 本地 NVMe RAID（`/dev/md0`）；global batch 128 / per-device 16 / seed 42 / `--log-interval 10` / 非确定性 XLA / 8 卡独占、机器无其他任务；关闭态（无 motion 节）。不得与环境 A 或旧本机 `/data` 数字混比；与 4 卡基线的对照是同介质、同库、同 YAML，只有卡数、mesh、worker 三项不同。

## 清理

三档 run 产物（各 12 G 末步 ckpt）与 `v1-store/bench/bench-m8x8-8gpu-*` 已删除；tmux 会话 `bw-8gpu`、`bw-8gpu-f8`、`bw-8gpu-f4` 均随驱动退出自然结束（未执行 kill）；runner / 驱动脚本留在 `v1-store/logs/`（副本在 `records/`）。
