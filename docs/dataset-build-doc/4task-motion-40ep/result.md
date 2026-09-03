# 4task-motion-40ep 构建结果留档（S0 先验与 oracle → S1 重抽与建库）

起跑口径见同目录 [`launch.md`](launch.md)。本文件按阶段追加实测判定行与数字；`v1-store/` 不进 git，关键行内联于此。

## S0 结果（2026-09-03）

**起跑 HEAD**：`6176f09cdee083d49358bf723cf6e69b9b0d44f0`（commitV6.1；相对代码锚点 `46ba954`，`src/` 与 SigLIP oracle
所用旧脚本零改动）。全部判定行 PASS：

```
CROSSCHECK=PASS                                                        （MotionJEPA crosscheck.py --vae_check，GPU1，2026-09-03 12:02:48 起，≈75 s）
A3_CROSSGPU=PASS compared=64 latent_bitwise=64 token_bitwise=64 max_abs_diff=0.000e+00
A4_DUALVENV=PASS compared=64 latent_bitwise=64 token_bitwise=64 max_abs_diff=0.000e+00
PROBE_BENCH=PASS windows=20 ms_per_window=850.7 peak_mib=1714 rerun_bitwise=20/20
SHARD_DONE shard=0 episodes=40 skipped=0 steps=13756 elapsed=248.4s rate=55.382 step/s steady_steps=13465 rate_steady=91.111 step/s   （O1）
Time taken: 2.77 minutes                                              （O2）
EXIT_CODE=0                                                            （motion-siglip-oracle / motion-s0-probes 两条 tmux 链尾行）
```

### SigLIP oracle O1 / O2（tmux `motion-siglip-oracle`，GPU1，11:54:43 → 12:02:13）

| 项 | O1 `oracle/siglip-shard1`（`build_shard.py --num_shards 1 --shard_idx 0 --subset`） | O2 `oracle/siglip-serial`（`build_dataset.py --max_episodes 10`） |
|---|---|---|
| episode / pkl | 40 个 `features/episode_*`、11,530 个 `data/*.pkl` | 40 / 11,530；`meta/stats.json` = `{"execution_samples": 11530, "total_samples": 13756}` |
| 耗时 | 248.4 s（含约 36 s 启动 + XLA 编译），稳态 91.1 step/s | 2.77 min（含启动） |
| 体积 | 13 GB | 13 GB |
| 指纹（sidecar `meta/_shard0of1.json`） | `manifest_sha256=4de8a0fc…`、`gpu_device_kind=NVIDIA RTX 6000 Ada Generation`、jax/jaxlib 0.5.3、`git_commit=6176f09…` | O2 stdout（40 行 `Episode g: timesteps=…`）留在 `oracle/siglip-serial.build.log` 供 `--a_untouched_log` 反查编号 |

### MotionJEPA crosscheck（`oracle/wan-mj/crosscheck.json`）

encoder 段 24 块：[0] 三方 77 张量逐位相同、ckpt `bae96037…` epoch 72；[1]–[5]、[8]、[9] 24/24 逐位；[6] batch 1 vs 8 与 [7] bf16 vs fp32
只报告（min_cos 0.99996245 / 0.99996634，max|Δ| 3.125e-2 / 3.416e-2，与 README 4.3 记录同值）；[10] 三个环境变量未设、
torch 2.9.0+cu128 / cudnn 91002。VAE 段 8 窗：[V7] 指纹 `9980d252…` == 记录值；[V1]–[V6] 8/8 逐位。

### Ada 探针（`oracle/probe/`；`scripts/dataset/wan/probe_wan.py`，窗口取 `ButtonUnmask.h5 episode_0`）

| 探针 | 实测 |
|---|---|
| **A2 计时**（20 窗，起点 0,12,…,228，预热 1 窗后逐窗 cuda 同步计时） | 合计 **850.7 ms/窗**（VAE 段 845.6，min 829.9 / max 859.5；encoder 段 5.0）；`max_memory_allocated` **1,714 MiB**；同设置重跑 20/20 逐位。相对 A40 先验 1.57 s/窗快 1.85×。同批 A3 GPU0 侧 64 窗含落盘 793 ms/窗 |
| **A2 漂移**（只记录，生产 / 在线均不启用） | TF32（matmul+cudnn 开）：latent min_cos 0.99999995、max rel 2.30e-3；token min_cos 0.9999798、max|Δ| 3.125e-2（1 个 bf16 ULP）、逐位 0/20；334.9 ms/窗（2.54×）。VAE bf16 autocast：latent min_cos 0.9999925、max rel 3.18e-2；token min_cos 0.99981、max|Δ| 8.2e-2、逐位 0/20；550.4 ms/窗（1.55×） |
| **A3 跨卡**（64 窗，起点 0,4,…,252，复制件，GPU1 vs GPU0） | latent 64/64、token 64/64 逐位，max|Δ| 0；两卡 uuid 不同（provenance 差异仅 gpu_uuid / pid，白名单键全等） |
| **A4 双 venv**（同 64 窗，MotionJEPA `.venv` 原版模块 vs `v1-store/venvs/wan` 复制件，同卡 GPU1） | latent 64/64、token 64/64 逐位，max|Δ| 0；provenance 白名单（torch / cuda / cudnn / diffusers / gpu / driver / flags / env / module_sha256 / encoder_src_sha256 / vae_state_sha256 / checkpoint_sha256 …）逐键相等；`module_sha256 == SOURCE_PIN.source_sha256 == af67fdd9…` |

### 资产与环境

- 复制件 `scripts/dataset/wan/wan_motion_infer.py` sha256 `af67fdd9…` == 源；`SOURCE_PIN.json` 记 `2a484ad`。
- ckpt `bae96037…`、`config.yaml` `99548a6c…`、VAE blob `d6e524b3…` 两侧 sha256 相同（`v1-store/external/motionjepa/wan-v8-filter10-72ep-a/SHA256SUMS.src-vs-copy.txt`）。
- wan 子 venv：`uv lock` 解析 91 包 12.9 s，`uv sync` 装 88 包 2.4 s（缓存命中；首次误建到错误路径 7.5 GB 已删除重建），
  `check_versions()` 通过（torch 2.9.0+cu128 / cudnn 91002 / diffusers 0.39.0），`encoder_src_sha256 d00267e1…` 与 MotionJEPA 树内同值。
- 新清单（新 CLI，`scan_manifest.py build --tasks … --episodes-per-task 10 --num_shards 1`）：40 ep / 13,756 timestep / 11,530 exec，
  sha256 `fee2777f58bf0e83b20fc95fff98a6b5871bfb2de10f967da39aecfccba892b6`；与 oracle 子集 40 条五字段逐条一致（A6 第一半）；
  按 `motion_store.build_index_entries` 现算 motion 表 **772 行 = exec 658 + demo 114**，单样本最大合法起点数 34，均与计划一致。

### 与计划的偏差

- 计划 1.3 预估 Wan 抽取按 1.64 s/窗；Ada 实测 0.85 s/窗（单卡 772 窗 ≈ 11 min，双卡 ≈ 5.5 min）。
- `crosscheck.py` 起手 `[env]` 打印的 `cudnn.allow_tf32=True` 是进程初始默认值，随后 `pin_numerics()` 钉为 False（[10] 读回值全对），不是异常。
- `uv run`（不带 `--no-sync`）在 `pyproject.toml` 改动后会重建 openpi editable 包，不改根 `uv.lock`；后续命令一律加 `--no-sync`。
