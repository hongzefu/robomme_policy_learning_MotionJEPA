# dtype-unify：`right_padding_token_emb` dtype 统一修复的验证工具

本目录只服务一件事：证明「给三个 `np.zeros` 补上 `dtype=` 参数」这个改动**没有改变
训练语义**。它不参与训练，不被训练进程 import，产物全部落在 `v1-store/` 下。

计划与判据的权威载体是仓库根目录 [`v1-dtype-unify-plan.md`](../../v1-dtype-unify-plan.md)；
基线链符号（G0b / G1 / speed 链）与登记簿在 [`v1-gradient-baseline.md`](../../v1-gradient-baseline.md)。
本文件只写「怎么用这几个脚本、它们各自判什么」。

---

## 一、要修的是什么，为什么需要这套工具

`src/mme_vla_suite/shared/data_utils.py` 的 `right_padding_token_emb` 在短样本分支
用 `np.zeros(...)` 造填充块却没指定 dtype，numpy 默认给 float64，于是 `np.concatenate`
把整条序列一起抬成 f64：

| 场景 | `static_image_emb` | `static_pos_emb` | `static_state_emb` |
|---|---|---|---|
| 短样本（`step_idx ≤ 30`，占 6.27%） | bf16 → **f64** | f32 → **f64** | f32 → f64 |
| 满长样本（`step_idx ≥ 31`） | bf16（纯切片） | f32 | f32 |

`max_size = budget // (token_per_image × num_views) = 512 // 16 = **32**`，所以边界
落在 `step_idx` 的 30 / 31 之间。collate 的 `np.stack` 会把「含任一短样本」的整个
batch 抬成 f64（b8 概率 40.4%），剩下的满长 batch 仍以 bf16 交付——**同一次训练里
dtype 随 batch 摆动，XLA 因此编译两份产物**。修复就是把那 1.6% 的行为推广到 100%。

模型第一层 `nnx.Linear` 显式 `dtype=bfloat16`，flax `promote_dtype` 在任何算术前统一
转 bf16，而 bf16→f32→f64 是精确升位——三种交付进投影层的张量逐位相同。所以这是个
**纯搬运浪费**，但「预期不改变数值」必须被证明，不能被假设。这套工具就是证明。

---

## 二、四个脚本

| 文件 | 职责 |
|---|---|
| `_common.py` | 公共层：位型容器读写、raw/canonical 摘要口径、定点样本集与定点 batch 的构造 |
| `dump_fixture_samples.py` | 取证：在修复前 / 修复后两个 clean HEAD 上各跑一次，落定点样本与定点 batch 的摘要（memory 四键另落数组本体） |
| `single_step_grad.py` | 取证：固定初始 state + 三个定点 batch，只算一步梯度并落盘 |
| `compare_dtype_fix.py` | 判定：离线对拍两侧产物，输出 `COMPARE_DTYPE=` / `COMPARE_GRAD=` 判定行 |
| `test_padding_dtype.py` | 自检：纯函数位型断言 + 摘要口径守卫 + 位型容器 round-trip |
| `run_dtype_dump.sh` | dump 驱动：固化路径与参数，两侧取证只剩 `RUN_TAG` 一个变量；起跑前查 clean HEAD |
| `run_dtype_grad.sh` | 单步梯度驱动：同上，另把确定性档 `XLA_FLAGS` 与编译缓存软链写死 |

### 为什么位型容器不用 npy/npz

`np.save` 会把 `ml_dtypes.bfloat16` 写成 `V2` void 类型，`np.load` 读回即丢逻辑类型——
用它存 fixture 会让对拍读到错误对象却照常给出判定。本目录改为**每键一个 `.bin`
（原始字节，C-order）+ 旁置 JSON**（记 shape / 逻辑 dtype / 字节序 / 键名 / 双口径 sha），
读回按 JSON 以 `np.frombuffer` + `reshape` 重建，并在写盘后立即读回做 round-trip 守卫，
守卫失败即 fail-loud——半套 fixture 比没有 fixture 更危险。同一格式在
`bench_train_steps.py` 的 `STATE_DUMP_STEPS` 里已经用过一遍（`state_step_<N>.bin`）。

### 摘要的两个口径

与 `scripts/smoke-local/bench_train_steps.py` 逐字相同，`test_padding_dtype.py` 里有
机器守卫盯着它们不漂移：

- **raw** = `sha256(dtype‖shape‖bytes)`——「应逐字节不变」的键用它（memory 之外的全部键）；
- **canonical** = 浮点键升 f32 后 `sha256("f32"‖shape‖bytes)`，dtype 不入域——**跨 dtype
  比数值**用它。canonical 相等 ⟺ 「`astype(f32)` 后 `view(uint32)` 逐位相同」，正是本
  计划判据 2 的机器形式。

### 定点集怎么来的（不靠 shuffle 撞边界）

由 `v1-store/episode_manifest.json` 正向公式精确算出：

```
dataset_index = ep.exec_sample_offset + (step_idx - ep.exec_start_idx)
```

- 短样本档 `step_idx ∈ {0,1,2,29,30}` 各 200 + 满长边界档 `{31,32,33}` 各 200 + 固定
  seed 随机 1,000 ≈ **2,600 个样本**；
- 定点 batch 200 个，四种组成各 50：`mixed1`（1 短 + 7 满长）/ `allshort` / `allfull` /
  `random`——判据 4「dtype 不随 batch 组成摆动」靠这四档同时成立才有意义。

> ⚠ **短样本档只能取自 800 个 `exec_start_idx == 0` 的 Button 系 episode**。Video 系
> （`VideoUnmask` / `VideoUnmaskSwap`）的 `exec_start_idx` 最小 66，其样本 `step_idx`
> 恒 ≥ 66、永远走满长分支，**根本产生不出短样本**。这是数据事实、不是取样偏置，但
> 留档必须写明；满长档与随机 1,000 自然覆盖两系。

dump 工具对每个样本都做同源自校验：manifest 算出的 `(epis_idx, step_idx)` 必须与 pkl
里记的一致，且数据集 `meta/provenance.json` 的 `manifest_sha256` 必须与 manifest 实物
相符——这两条能抓住「manifest 过期」与「数据集与 manifest 不同源」。

---

## 三、怎么跑

所有命令的 **cwd 必须是仓库根**（`get_history_config` 按相对路径加载 yaml），且必须
显式给 `--dataset-path`（代码默认的 `data/robomme` 在本仓库不存在）。

### 1. 自检（随时可跑，约 1 分钟）

```bash
JAX_PLATFORMS=cpu UV_LINK_MODE=copy uv run pytest scripts/dtype-unify/ -q
```

修复尚未落地时，纯函数 dtype 断言会自动 **skip** 并说明原因；修复落地后自动转为真
断言，不需要改测试文件。

### 2. dump（修复前后各一次）

正常走驱动脚本（它会先查 clean HEAD、拒绝覆盖既有产物、跑完清理 checkpoint 空壳）：

```bash
RUN_TAG=v1-dtype-p3-dump-pre bash scripts/dtype-unify/run_dtype_dump.sh
```

底层命令（调试时用）：

```bash
DTYPE_DUMP_DIR=v1-store/dtype-unify/<run_name> JAX_PLATFORMS=cpu UV_LINK_MODE=copy \
uv run scripts/dtype-unify/dump_fixture_samples.py mme_vla_suite \
  --exp-name <exp> --assets-base-dir v1-store/train-assets \
  --checkpoint-base-dir v1-store/train-runs/<exp> \
  --dataset-path v1-store/datasets/4task-gl \
  --model.use-history --model.history-config perceptual-framesamp-context.yaml \
  --no-wandb-enabled
```

可调环境变量：`DTYPE_DUMP_MODE`（`samples` / `batches` / `both`，默认 both）、
`DTYPE_DUMP_ARRAYS`（`1`/`0`，默认 1，是否落 memory 四键数组本体）、
`DTYPE_DUMP_LIMIT`（>0 时每组只取前 N 个，**仅供冒烟**，正式取证不设）。
输出目录已存在且非空即拒跑——不覆盖既有取证产物。

### 3. 单步梯度（修复前后各一次，需 2 卡）

正常走驱动脚本：

```bash
RUN_TAG=v1-dtype-p3-dump-pre \
GRAD_ARRAYS_DIR=/data/hongzefu/v1-baselines/dtype-p5-grad-pre \
bash scripts/dtype-unify/run_dtype_grad.sh
```

底层命令（调试时用）：

```bash
DTYPE_GRAD_DIR=<records 目录> \
DTYPE_BATCH_FIXTURE_DIR=<batch fixture 目录> \
DTYPE_GRAD_ARRAYS_DIR=<梯度数组目录，不设则只落摘要与统计> \
DTYPE_BASELINE_CHECKSUMS=docs/training-doc/v1-grad-baseline-g0b/records/r1/param_checksums.jsonl \
XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" \
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 UV_LINK_MODE=copy \
uv run scripts/dtype-unify/single_step_grad.py mme_vla_suite --exp-name <exp> ...
```

`DTYPE_GRAD_KINDS` 可限定只跑其中某几种 batch（逗号分隔），仅供冒烟。

两点值得单独说：

- **初始 state 的同源性**不靠加载那份 45.4 GiB 的 `state_step_0.bin`，而是用同 seed /
  同 config 现场 `init_train_state`，再把逐叶 sha256 与 G0b r1 的 step 0 `per_leaf`
  逐条比对——全等即同源（G0b 的步 0 摘要本来就是 init 后立即记的）。
- **梯度的取法**：`train_step` 用 `nnx.value_and_grad` 算出 grads 却不返回，本脚本
  复刻了它的梯度段（并在 `sharding.set_mesh` 上下文里调用，否则模型内部的
  activation sharding constraint 失效、HLO 与训练不同）。复刻有漂移风险，因此照
  bench 既有做法加了源码指纹护栏：`train.train_step` 的源码不含预期子串即当场报错。

### 4. 对拍

```bash
# 第一块（样本级 + batch 级 + 纯函数位型测试）
JAX_PLATFORMS=cpu UV_LINK_MODE=copy uv run scripts/dtype-unify/compare_dtype_fix.py \
  <修复前 dump 目录> <修复后 dump 目录> --report <报告.json>

# 单步梯度
uv run scripts/dtype-unify/compare_dtype_fix.py \
  --grad-a <修复前 grad records> --grad-b <修复后 grad records> --report <报告.json>
```

判定行：

```
COMPARE_DTYPE=PASS samples=2600 batches=200 mismatches=0
COMPARE_GRAD=PASS kinds=3 mismatches=0
```

---

## 四、判据

第一块四条，全部零容差：

1. 全键 shape 相同；
2. 数值 canonical 一致（等价于 `astype(f32)` 后逐位相同）；
3. dtype 变化逐键清单与预期完全一致——`static_image_emb` 短样本 f64→bf16 / 满长
   bf16 不变，`static_pos_emb` 短 f64→f32 / 满长 f32 不变，`static_state_emb` **恒 f64
   不变**，`static_mask` 恒 bool，**memory 之外全部键的 dtype 与 raw 摘要都必须完全
   相同**（任何变化都是改动越界）；
4. batch 级 memory 键 dtype 恒定，不随 batch 组成摆动。

外加**归一化前纯函数位型测试**：`static_state_emb` 交付键经 `_normalize_state`
（norm stats 为 f64）恒为 f64，第三处 `np.zeros` 的修复在交付键和梯度上都不可观测——
这个纯函数测试是它**唯一**的有效证据，同时也是 modulation / expert 变体与在线评估
路径（同一函数、同一修复、不在本计划验收范围）的函数级证据。

单步梯度三个 batch 的分工：

- `mixed1`（1 短 + 7 满长）——**主判据**，唯一存在 dtype 差异的典型场景；
- `allshort`（全短样本）——差异密度最大化；
- `allfull`（全满长）——**阴性对照**：两侧本就同为 bf16 交付，若它不逐位相同，说明
  改动越界（与 dtype 无关），必须立刻停下排查。

---

## 五、产物与清理

- dump 产物落 `v1-store/dtype-unify/<run_name>/`（`samples/` + `batches/` +
  `fixture_plan.json` + `DUMP_MANIFEST.json`），**不进 git**，对拍报告写进
  `docs/training-doc/<run_name>/` 后即可清理；
- 三个定点 batch 的 fixture（修复后一侧）落 `v1-store/fixtures/`，**长期保留**，逐文件
  sha256 清单进 git——它按基线计划五节升格为常规回归闸，后续任何 commit 花约 2 分钟
  即可重锚；
- 修复前一侧的梯度数组（约 11 GB × batch 数）落本机 `/data/hongzefu/v1-baselines/`
  （不留 NFS，沿用 G0b state dump 的先例），**验收通过后删除数组本体**，保留逐叶
  sha256 清单与逐叶统计。
