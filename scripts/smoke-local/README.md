# smoke-local：本机 2 GPU epoch 基准 + 一致性检验记录底座

本目录五个文件：

| 文件 | 作用 |
|---|---|
| `run_2gpu_epoch_bench.sh` | 驱动脚本：2 卡跑 `STEPS`（默认 300）步，**batch 固定 8**（2 卡实测唯一确认可跑档，64/32/16 均 OOM；改档属超参变更须先确认并重新实测），算稳态 s/step 并外推 1 epoch 时长，留下一致性检验记录；任何失败 fail-loud 不重试。身份拆分：`EXP_NAME` 决定 jax 编译缓存目录（对拍两轮共用即共享编译产物与 per-fusion autotune），`RUN_TAG` 决定记录/日志/checkpoint 目录（每轮各异）；`KEEP_JAX_CACHE=1` 收官保留缓存，`XLA_FLAGS` 外部注入并留档 |
| `bench_train_steps.py` | 训练入口：只调一次 `train.main(config)`，训练循环一行不改；靠 monkeypatch 把逐步标量、完整 TrainState 摘要（params/opt_state/EMA/step，步 0 必记）、输入 batch 摘要写成 jsonl，另留档编译缓存事件计数与真实 argv。速度口径开关：`SAVE_INTERVAL=0` 禁 TrainState 摘要、`BATCH_DIGESTS=0` 禁输入摘要（speed 链 run 专用，见 `v1-gradient-baseline.md` 符号总表） |
| `check_baseline_env.py` | 环境指纹 `dump`（起跑留档进 env.json）/ `check`（**引用任何基线产物前的强制 preflight**，输出 `BASELINE_ENV=PASS|FAIL`）/ `manifest`（生成 `BASELINE_MANIFEST.json` 防产物腐烂） |
| `compare_baseline.py` | 两份 records 目录对拍：逐步标量 hex diff、`state_digest`/`batch_digest` diff、rel 分布与包络（量化判据权威版本见 `v1-gradient-baseline.md` 六节），输出 `DET_CHECK=` / `QUANT_EQUIV=` 判定行 |

一句话定位：**测「本机 2 卡、数据在 NFS turbo 时 1 个 epoch 要多久」，同时为将来
修改 dataloader 后的一致性检验（数据允许等价但不逐位相同）留下可逐位比对的轨迹记录。**
不落任何 checkpoint。本机数字按仓库 AGENTS.md 第 13 条只作估算，不作正式吞吐结论。

---

## 一、本基准与官方默认训练的逐项差异

官方口径 = `scripts/finetune_mme_vla_suite.sh` → `scripts/train.py` 的 `mme_vla_suite`
具名配置（batch 64、4 卡 `fsdp_devices=4`、`num_workers=4`、80k steps、wandb 开启）。
总原则：**训练循环、模型、loss、优化器、lr schedule、seed 的代码一行不改**；所有差异
要么是资源现实（2 卡），要么「只截断长度 / 只加观测」。分三类逐项说明。

### A. 对训练数值轨迹有实质影响的差异

| # | 差异 | 为什么有影响 | 对一致性检验的含义 |
|---|---|---|---|
| A1 | 4 卡 `fsdp_devices=4` → 2 卡 `fsdp_devices=2` | mesh 从 (1,4) 变 (1,2)，参数/激活分片与跨卡归约顺序不同；浮点加法不结合，**loss/梯度与 4 卡 run 必然逐位不同**（数学期望等价：同一全局 batch、同一组样本） | 本机 A/B 检验不受影响——比较的是两条 2 卡轨迹；但**本目录的记录不能拿去和任何 4 卡 run 逐位比对** |
| A2 | batch：官方全局 64（per-device 16）→ **实测只能跑全局 8（per-device 4）** | 首档设计本是「全局 64 不变、per-device 16→32」——那样样本集合相同、仅逐位不同；但 2 卡下 64/32/16 全部 OOM（见 A3），实际落在 batch 8，**全局 batch 也变了**，属实质超参变更 | epoch 外推按 batch 8 换算并标注非官方口径；A/B 两边同用 batch 8 即无影响 |
| A3 | batch 8 的显存依据（为什么不是官方 64） | 2026-08-24 实测 2 卡下 64/32/16 全部 OOM（失败张量 17.62/12.61/10.38 GiB：每卡驻留约 28 GB 参数+优化器+EMA，激活固定底座约 8 GiB——官方 4 卡下每卡状态减半所以能跑 batch 64），batch 8 以 300 步全程验证可跑。早期版本曾内置 OOM 自动降档梯子，实测钉死后已移除、脚本写死 batch 8 | 改档位属实质超参变更：须先与用户确认并重新实测显存；A/B 两边用同一档即可 |
| A4 | history 变体：官方脚本默认 `perceptual-framesamp-modul` → 本基准锁死 `perceptual-framesamp-context` | 模型结构分支不同（14 变体之一），有意选择：与此前本机 smoke、GL 数据集验证、已就绪的 norm stats 同口径 | 入口有护栏强制该变体，A/B 天然同口径 |
| A5 | 硬件：RTX 6000 Ada + 本机驱动 vs 集群 A40 | 不同 GPU/驱动/XLA 编译选择 → 浮点行为逐位不同 | 跨机器逐位比对不成立；`env.json` 留档硬件与环境，A/B 前逐项核对 |

### B. 不改训练数值、只影响吞吐或计时口径的差异

| # | 差异 | 为什么没有数值影响 |
|---|---|---|
| B1 | `num_train_steps` 80,000 → 300 | 只截断长度。lr schedule 是 CosineDecay（warmup/peak/decay 参数在具名配置里写死），**不依赖 `num_train_steps`**——前 300 步每步的 lr 与计算和官方 80k run 的前 300 步完全一致 |
| B2 | `log_interval` 100 → 1 | log 只做 host 端聚合与打印（`wandb.log`），不进计算图；官方 100 步记一次均值，本基准逐步单值。代价是每步一次标量 device_get，毫秒级 |
| B3 | `save_state` 被换成参数校验和、`save_interval` 10,000 → 25 | 校验和只读参数、不写训练状态，对轨迹零影响；但每次把 ~14 GB 参数拉回 host 需数十秒，**稳态统计已剔除校验和步及其下一步**（正式训练没有这项开销） |
| B4 | 跳过 tentative 预热（不走 `train.py` `__main__` 的双跑结构） | tentative 那次独立初始化、状态全部丢弃，本来就不影响正式轨迹；代价是第一步吃满 JIT 编译时间，稳态统计已丢弃前 `WARMUP_STEPS`（默认 50）步 |
| B5 | 数据经 NFS turbo 读取 | IO 路径只影响吞吐。turbo 单流实测 ~132 MB/s；framesamp 每样本读 ~32 个 token_emb（约 19-26 MB），batch 64 一步 ~1.2-1.6 GB，IO 很可能是瓶颈；page cache 可能使 300 步稳态偏乐观 |

### C. 完全无影响的差异（纯路径 / 工程卫生）

| # | 差异 | 说明 |
|---|---|---|
| C1 | wandb 开启 → `--no-wandb-enabled` + `wandb.log` 替换为 jsonl 记录器 | disabled 模式下 `wandb.log` 本来就是 no-op，记录器只是把丢掉的数据接住；不进计算图 |
| C2 | 路径显式化：`--assets-base-dir`/`--checkpoint-base-dir` 指到 `v1-store`、`--dataset-path` 指 GL 全量库 | 纯目录选择；norm stats 用 GL 数据集上已算好的那份，与数据集自洽 |
| C3 | `uv run` → 直接 `${PY}`（turbo 仓库 `.venv` 的解释器） | 同一 venv，无差异 |
| C4 | bench 入口的 fail-loud 护栏（≤500 步、强制关 wandb、禁 overwrite/resume、锁 history_config、强制 `log_interval=1`） | 只在启动前检查参数，不参与训练 |
| C5 | `seed=42`、`num_workers=4`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、EMA、优化器、loss | 与官方完全相同。同数据集同 seed 下 dataloader 的逐 epoch shuffle 顺序也与官方一致（torch generator 按 `config.seed` 固定） |

**结论**：会让数值逐位偏离官方 4 卡 run 的只有 A 组（2 卡切分、per-device batch、
本机硬件、有意选择的变体）；B/C 组要么只截断长度、要么只加观测。而一致性检验是
本机 2 卡内部的 A/B，A 组偏离对它无伤害——前提是 A/B 两边沿用同一套 `env.json` 环境。

---

## 二、一致性检验：三级比较协议与记录文件格式

未来修改 dataloader（数据允许「等价但不逐位相同」）后，与本基准留下的记录做逐步
对比。**由便宜到贵三级，逐级下钻，前一级通过就不必做后一级：**

### 第 1 级：loss 曲线（最便宜，每步一个标量）

前几百步逐步对比 `metrics.jsonl` 的 `loss.hex`。确定性设置下应 **bitwise 相同**。
确定性口径（对应 torch 世界的 `use_deterministic_algorithms` / 固定 cudnn / 关 TF32，
本仓库是 JAX 栈）：
- 固定 `--seed 42`（模型初始化、dataloader shuffle 都由它决定）；
- A/B 两边**同机、同驱动、同 `XLA_FLAGS`、同 mesh（2 卡 fsdp=2）、同 batch 档**——
  逐项核对两份 `env.json` 相同（TF32 等矩阵精度行为由 XLA 按硬件+flags 决定，
  两边同设即抵消）；
- 确定性档由调用方经 `XLA_FLAGS` 注入（2026-08-26 起支持；正确性对拍 run 跑在
  `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` 档，**偏离官方生产
  口径是有意的**——G0 的语义是「受控确定性档位下的当前训练语义」；speed 链 run 不注入
  flags，保持生产口径。两族 run 的权威口径见 `v1-gradient-baseline.md` 符号总表）。
- **2026-08-24 实测：同配置同 seed 重跑两轮，参数校验和逐步全不相同——本机默认
  设置下并非 bitwise 确定**（疑因每轮删了 jax 编译缓存、XLA 重新 autotune 选中
  不同 kernel/归约实现）。
- **2026-08-26 四档确定性实验定档（D0/D1/D2/D2-cold 各两轮 100 步，commit V2.1，
  留档 `docs/training-doc/v1-det-*/`，权威结论见 `docs/v1-determinism-conclusions.md`）**：
  - D0（删缓存、无 flags）FAIL：从步 0 起逐步全分歧，loss rel 噪声底 median 2.7e-3 /
    max 4.6e-2——上面 2026-08-24 的观察被系统复现并量化；
  - D1（共用编译缓存、无 flags）FAIL 但仅差 ULP：100 步中 2 步 llm_grad_norm 差一个
    最低位，末段分歧只有 embedder input_embedding 一族（scatter-add atomics）；
  - D2（+`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`，共用缓存）
    **PASS**：两轮 100 步逐位一致；
  - **D2-cold（同 flags、独立空缓存强制重编译两次）PASS**：独立编译两次仍逐位一致，
    且与 D2 轮交叉对拍亦 PASS——正确性 A/B 一律注入 D2 flags 即可跨编译对拍 bitwise。

### 第 2 级：梯度范数 + 参数校验和（便宜，且校验和是累积效应）

- 每步的 `grad_norm` / `llm_grad_norm` / `mem_enc_norm`（`metrics.jsonl`，hex 精度）
  ——相当于 `sum(p.grad.norm())` 口径的逐步指纹；
- 每 `save_interval`（默认 25）步 + 最后一步，对**全部参数（params 与 ema_params）**
  逐叶子 sha256（`param_checksums.jsonl`）。参数是累积效应：**300 步后
  `global_digest` 相同，基本就能证明整条轨迹一致**，比逐元素比梯度便宜几个量级。

### 第 3 级：逐元素梯度对比（最贵，只在前两级发现分叉后用）

用于二分定位是哪一层、哪个模块开始分叉。本目录不实现，方法：
1. 用 `param_checksums.jsonl` 的 `per_leaf` 摘要先按模块缩小范围（哪些叶子的
   digest 先开始不同、从哪个 step 开始）；
2. 找到「两边 global_digest 仍一致的最近记录 step」，从该状态出发、喂**同一个
   固定 batch**，重算一步梯度做逐元素 diff（梯度可从状态+batch 重算，无需训练时落盘
   ——逐步存完整梯度每步 ~14 GB，不可行）。

### 记录文件（`v1-store/bench/2gpu-epoch-bench/<run_name>/`）

**`metrics.jsonl`** —— 每训练步一行（`--log-interval 1` 保证逐步）：

```json
{"step": 7, "wall_time": 1756056000.123456,
 "loss":          {"dec": 0.4427, "hex": "0x1.c55f2dd3c9052p-2"},
 "grad_norm":     {"dec": 33.8064, "hex": "0x1.0e73a6f8b21c4p+5"},
 "llm_grad_norm": {"dec": 26.6147, "hex": "0x1.a9d5e0c4f1832p+4"},
 "mem_enc_norm":  {"dec": 19.6468, "hex": "0x1.3a5921bb64e08p+4"},
 "param_norm":    {"dec": 1803.0919, "hex": "0x1.c2c5e17a4b990p+10"}}
```

- `dec` 供人读，`hex` 是 `float.hex()` 的位级精度，**bitwise 对比一律用 hex**；
- `wall_time` 为该步 log 时刻的 unix 秒，逐步差即步时（驱动脚本用它算稳态，
  剔除前 `WARMUP_STEPS` 步与校验和步）。

**`param_checksums.jsonl`** —— 每个校验和步一行：

```json
{"step": 25, "wall_time": 1756056100.5, "checksum_seconds": 21.3, "n_leaves": 1234,
 "global_digest": "3f8a…(sha256)", "state_digest": "b02c…(sha256)",
 "per_leaf": {"params.PaliGemma.llm…kernel": "9c1d…", "opt_state.…": "4e6a…",
              "ema_params.…": "77b2…", "step": "a1f0…"}}
```

- `state_digest`：**完整 TrainState**（params/opt_state/EMA/step）全部叶子按 key
  排序后拼接再 sha256——**判整体一致的主列**（Adam 动量是最灵敏的累积量）；
- `global_digest`：只覆盖 params+ema 的旧口径列，供与 2026-08-24 之前的记录续比；
- `per_leaf`：`{params|opt_state|ema_params|step}` 前缀 + pytree 路径 → 该叶子
  （dtype+shape+字节）的 sha256——**定位分叉模块用这一层**；
- 触发步：步 0（init 后立即记，`init_train_state` 包装所致）、
  `step % save_interval == 0 (step>0)` 及最后一步。

**`batch_digests.jsonl`** —— 输入侧摘要（collate 后、device_put 前的 host 侧 batch，
逐键 `sha256(dtype‖shape‖bytes)`），步 0/1/2 + 每 `save_interval` + 末步。与
XLA/缓存/驱动无关、跨计算图（HLO）永远逐位可比——改输入签名场景对拍的主判据：

```json
{"step": 0, "wall_time": 1756056000.2, "digest_seconds": 0.4, "n_keys": 18,
 "batch_digest": "c9d2…(sha256)", "per_key": {"['actions']": "0b3f…"}}
```

**`run_meta.json`** —— 训练进程真实 `sys.argv` + jax.monitoring 事件计数（编译
缓存命中/编译次数——判定「这轮是热缓存还是冷编译」的留档事实）；驱动脚本收官时
并入 `env.json` 的 `run_meta` 键。

**`env.json`** —— 启动时留档：git HEAD 与 dirty 标记、batch/steps/workers/seed/
fsdp/变体、`XLA_FLAGS`、`XLA_PYTHON_CLIENT_MEM_FRACTION`、`CUDA_VISIBLE_DEVICES`、
主机名、Python/jax 版本、GPU 型号与驱动。**A/B 对比前必须逐项核对相同。**

### 最小对比命令

```bash
# 第 1 级：loss 曲线逐步 bitwise 对比（无输出 = 完全一致）
diff <(jq -r '"\(.step) \(.loss.hex)"' A/metrics.jsonl) \
     <(jq -r '"\(.step) \(.loss.hex)"' B/metrics.jsonl)

# 第 2 级：参数全局摘要对比
diff <(jq -r '"\(.step) \(.global_digest)"' A/param_checksums.jsonl) \
     <(jq -r '"\(.step) \(.global_digest)"' B/param_checksums.jsonl)

# 分叉定位：找第一个不一致的叶子（对某个具体 step）
"${PY}" - A/param_checksums.jsonl B/param_checksums.jsonl <<'PYEOF'
import json, sys
A = {r["step"]: r for r in map(json.loads, open(sys.argv[1]))}
B = {r["step"]: r for r in map(json.loads, open(sys.argv[2]))}
for step in sorted(set(A) & set(B)):
    bad = [k for k in A[step]["per_leaf"] if A[step]["per_leaf"][k] != B[step]["per_leaf"].get(k)]
    print(step, "一致" if not bad else f"分叉 {len(bad)} 叶子, 首个: {bad[0]}")
PYEOF
```

---

## 三、怎么跑

预计单档 300 步 ≥ 20 分钟（IO 瓶颈下更久），按 AGENTS.md 第 7 条放 tmux：

```bash
tmux new-session -d -s epoch-bench \
  "set -o pipefail; PYTHONUNBUFFERED=1 \
   bash scripts/smoke-local/run_2gpu_epoch_bench.sh 2>&1 \
   | tee v1-store/logs/2gpu-epoch-bench-driver.log; \
   echo \"EXIT_CODE=\$?\" >> v1-store/logs/2gpu-epoch-bench-driver.log"
```

结束后日志尾部有 `RESULT` 两行（稳态 s/step 与 epoch 外推）与 `BENCH_PASS`。
可调环境变量：`STEPS`（≤500）、`SAVE_INTERVAL`、`WARMUP_STEPS`、`DATASET_PATH`。
run 目录与 `~/.cache/jax_<exp_name>` 跑完即删；记录目录与日志保留。
