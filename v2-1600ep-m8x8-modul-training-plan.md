# 1600 集新库 · 8 帧 8×8 · modulation · 无 motion · b128 60k · lr 5e-5 正式训练计划

> **实施过程档案**（2026-09-15 第五稿，用户批准）。结果以 [`docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/`](docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md) 为准；本文按 `AGENTS.md` 第 2 条分两部分，另含 F（守卫放宽与增补验证）、G（worktree 隔离）、H（审计结论）、I（本文固化）四节。
>
> **修订史**：第一稿 b64 80k（官方口径）；第二稿 b128 40k lr 1e-4；第三稿 b128 60k lr 5e-5（用户「lr 等效比官方低一倍、40k 延长到 60k」）；第四稿加 F 节（smoke 被 dataset G13 守卫挡下）；第五稿加 H（motion 接入不影响无 motion 的 modulation 的审计结论）、G（不在主工作副本起训练，改从 detached worktree 快照起跑）、I（本文固化到根目录）。
>
> **进度**：步骤 1 已完成（`commitV9.5` = `e0bcb45`）；`framesamp_dataset.py` 守卫白名单已改在工作区、未提交；smoke 临时产物已清理；其余步骤待用户指令后执行。

环境判定：**环境 B（AWS 单机）**。仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`（HEAD `ba0f450`，clean），8 × A100-SXM4-80GB 当前全空，`/scratch` 余 1.4 T。无 turbo、无 GreatLakes。

用户已拍板（2026-09-15）：GPU **4,5,6,7**；wandb **开启**；batch **128**；步数 **60k**；**lr peak = decay = 5e-5**（b128 下等效为官方 5e-5@b64 的一半，不做线性缩放）；改动落在**新配置条目 `mme_vla_suite_b128_60k`**。run_name 随步数改为 **`v2-1600ep-m8x8-modul-b128-60k`**（原拍板 `…-40k` 已不适用，起跑前再确认一次）。

---

## 第一部分（给人看）

### Context：为什么做这件事

上一条生产 run `awsprod40k-b128-motion`（2026-09-04→06）用旧 400 ep 四任务库、32 帧 × 4×4、context + motion。2026-09-14 新库 `4task-v2-1600ep-604f16da`（BinFill / RouteStick / VideoRepick / VideoUnmaskSwap 各 400 集，605,611 执行样本）建成并通过 20 步可读性验收；同日 8 帧 × 8×8 布局（`framesamp-8x8` 库，292 G，`VERIFY_PACK=PASS`）完成逐位验收。本轮在新库上训练第一条 **8 帧 × 8×8 + modulation + 无 motion** 的正式模型，作为后续对照基线。lr 取比线性缩放保守一档、步数从 40k 延长到 60k，以补偿更小步长。

### 结论先行：两处配置改动、零代码改动；先 20 步 smoke 再起正式 run

**1. 改动只有两个配置文件。**
- 新 YAML `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml`：以 `perceptual-framesamp-modul.yaml` 为底只把 `token_per_image` 16 → 64。现有 modul.yaml 是 32 帧 × 4×4，context-8frame-8x8.yaml 是 8×8 但 context，各差一两行。
- 新条目 `mme_vla_suite_b128_60k`（`src/mme_vla_suite/training/config.py`）：复制 `mme_vla_suite_b128`，改 `num_train_steps 60_000`、`peak_lr = decay_lr = 5e-5`、`decay_steps 60_000`（peak == decay 时余弦段为常数，改它只为自洽）、`data.assets.assets_dir` 指到新库 norm_stats 目录。`mme_vla_suite`（官方参照）与 `mme_vla_suite_b128`（上次生产口径）一字不动。

代码侧已就位：dataset `_req` 接受 `(512, 64, 1)` 档并核 `store_meta.spec.tokens_per_frame == 64`，`_max_frames = 8`；`PerceptualMemory` 只断言 `budget == 512`；`MemoryAttention` 对 token 数无假设；motion 节整节缺省即关闭态。

**2. 学习率口径（与官方对照）。** 官方 b64：warmup 10k、peak = decay 5e-5、80k 步。本 run b128：warmup 5k（= 640k 样本，与官方 warmup 样本数相同）、peak = decay 5e-5、60k 步。曲线形状相同（线性升、然后恒定不衰减）。batch 翻倍而 lr 不变，等效步长是线性缩放（1e-4）的一半；总样本 60k × 128 = 7.68M，是官方 5.12M 的 1.5 倍，对新库约 **12.7 epoch**。更新次数 60k，官方 80k。

**3. 为什么必须先 smoke。** modulation 分支在本仓库从未训练过（`v1-postclean-g3` 登记 UNVERIFIED，只有环境 A 用官方 modul 权重评估过）；t8 那轮 12 条轨迹只覆盖 context（C8）与 context+motion（M8）。smoke 用真实 b128、真实 4 卡 fsdp4 跑 20 步，回答两件事：modulation × 8×8 能否编译并出有限数；per-device 32 在 modulation 下是否 OOM（旁证：`bench-b128-util` 在 context+motion、608 位进 prefix 的更重口径下 4 卡 b128 300 步无 OOM；modulation 记忆不进 prefix，激活只会更小）。≤ 5 分钟，临时 run 跑完即删。

**4. 预期。** 4 卡 b128 稳态 3.831 s/step（`bench-b128-util`，util 均值 94.75%）；60k ≈ 63.9 h，加编译与 12 次存盘约 **64.5 h（2.7 天）**。checkpoint 12 × 11 G ≈ 132 G，磁盘余 1.4 T。起跑后以前 300 步稳态复核 ETA。

**5. 与上次生产 run 的差异表**（引用结果须带此声明，两条不逐项可比）

| 项 | awsprod40k-b128-motion | 本 run |
|---|---|---|
| 数据 | 400 ep 旧四任务，101,066 样本 | 1600 ep 新四任务，605,611 样本 |
| 记忆布局 | 32 帧 × 4×4，`framesamp` | 8 帧 × 8×8，`framesamp-8x8` |
| 接入 | context，608 位进 prefix | modulation，cross-attention，`memory_token_dim` 1024 |
| motion | 开 | 关 |
| 步数 / lr | 40k / 1e-4 | 60k / 5e-5 |
| 卡 / worker | 8 卡 w16，mesh (2,4) | 4 卡 w8，mesh (1,4)，per-device 32 |
| norm_stats | `robomme-400ep` | `4task-v2-1600ep-604f16da`（`856c75ea…`） |
| 新增可训练参数 | motion 两层 3.35 M | `MemoryAttention`（q/kv/out einsum）+ 每层 `MemoryRMSNorm` modulation Dense；`encoder_static` 2816→1024 |

**6. 第四稿新增：一处 dataloader 守卫要放宽，因此多跑三项验证。** 第三稿的 smoke 起跑后在 `FrameSampDataset.__init__`（`src/mme_vla_suite/training/framesamp_dataset.py`）被一条有意加的守卫挡下：`integration_type='modulation' != 'context'`。这条和它下一行 `memory_token_dim == 2048` 是 commitV3.1 按 Codex 审计 G13 加的，目的是拒绝「同形的 modul 配置」；它们只在构造时拒绝配置，后面的取样、填充、store 读取一行都不读这两个键。训练链路上没有别的 context 专属守卫（`training/`、`train.py`、`compute_norm_stats.py`、`g0/`、`policies/` 全部 grep 过，只此两行；模型侧 `history_pi0.py` 本来就接受 modulation）。

处理方式是把两条守卫合成一条**成对白名单**：只允许 `(context, 2048)` 或 `(modulation, 1024)`，expert 与错配对（例如 modulation 配 2048）照样拒。这是对 dataloader 文件的改动，按 AGENTS 第 18 条要证明「不改数」，所以比第三稿多跑三项，总共约 20–25 分钟：

- **G13 测试改写**（`scripts/training/tests/test_pack_guards.py`）：原测试断言 modul.yaml 必拒，改成「modul.yaml 通过、expert.yaml 拒、错配对拒」三条。该 pytest 依赖 `v1-store/datasets/ref-shard` 迷你库，本环境没有，跑不了；三条断言改在 1600ep 库上用一次性脚本执行，留档写明「本环境未执行 pytest」。
- **第一块（轻量对拍，CPU 约 3–5 分钟）**：用 `perceptual-framesamp-modul-8frame-8x8.yaml` 和 `perceptual-framesamp-context-8frame-8x8.yaml` 在同一 `framesamp-8x8` 库上各构造一个 Dataset，对同一组 256 个索引逐样本逐键比 dtype、shape、raw sha256，判定行 `DS_EQUIV=PASS samples=<n> keys=<k> mismatches=0`。这直接证明 integration_type 不影响交付内容。
- **第二块（真实训练梯度一致，GPU 4,5 约 10–15 分钟）**：复用 09-14 已固化的 `t8-c8-b` 轨迹（400ep 库、context 8×8、batch 8、fsdp 2、seed 42、确定性 XLA），在改守卫后的 HEAD 上照抄其命令跑前 100 步，先过 `check_baseline_env.py check` 指纹 preflight（`BASELINE_ENV=PASS`，jax / 驱动 / norm_stats / pi05_base / 库摘要逐项同），再把逐步五标量 hex 与 `docs/training-doc/t8-c8-b/records/scalars_hex.tsv` 前 100 行逐字节比，判定行 `GUARD_GRAD_100=PASS steps=100 mismatches=0`。这证明 context 链在守卫改动前后逐位相同。
- 三项全 PASS 才提交 `commitV9.6`（守卫 + 测试改写），push，然后才回到 smoke。任一 FAIL 即停、把原文交你处置，不放宽判据。

**7. 审计结论：motion 接入不影响「无 motion 的 modulation」——源码级成立，运行时证据只覆盖 context。** 按 motion 接入 commit `06220c4`（commitV6.5）的 diff 原文逐层核：
- **模型侧参数树不变**：`PerceptualMemory.__init__` 的两层新参数 `motion_pos_proj`、`motion_encoder_static` 只在 `motion.enabled` 为真时创建，且建在 `feature_encoder` 之后（nnx 默认 RNG 流按调用顺序 fold_in，帧路初始化值不变）。modul-8x8 YAML 没有 `motion` 节，`_motion_enabled` 为 False。
- **`embed_memory` 关闭态是编译期早返回，函数体与接入前逐字相同**（接入前四行：`tokens = mem_encoder(...)`、`input_mask = static_mask`、两个全 False 列表、return；接入后早返回分支内正是这四行）。modulation 分支在 `compute_loss` / `sample_actions` 里只取前两个返回值 `mem_seq, mem_mask` 喂 `PaliGemma.llm(..., mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])`。
- **modulation 的 prefix 不含记忆区**：`embed_prefix` 只在 `integration_type == "context"` 时才把记忆 token 拼进 prefix；modulation 的 prefix 是 img 512 + prompt，`make_attn_mask` 两参数版无 `na_mask`。608 位交错、`take_along_axis`、RoPE 位次变化对 modulation 不可达。
- **cross-attention 本体零改动**：接入前后 `history_gemma.py`（`MemoryAttention` / `MemoryRMSNorm`）、`integration/utils.py`、`openpi/models/gemma.py`、`representation/mem_encoder.py` 四文件 `git diff` 为空。
- **两道显式闸**：`HistoryPi0.__init__` 里 `motion_enabled and integration_type != "context"` 即 raise；`inputs_spec` 与 `PerceptualMemory` 的 `motion_enabled` 不一致即 raise。开启态混进 modulation 会在建模型时报错。
- **数据侧交付不读 `integration_type`**：`FrameSampDataset.__getitem__` 的 motion 代码全在 `if self._motion_enabled` 内，关闭态只在样本末尾按 `_NONE_KEYS` 追加四个 None（与旧路径「尾部补空键」逐字一致），`HistAugObservation.from_dict` 用 `data.get(key, None)` 接住，None 在 jax pytree 里是空节点、不进数值图。context 与 modulation 拿到同一份 batch。
- **证据边界**：运行时逐位证据只在 context 关闭态取过（环境 A `motion-t1-closed` 对 G0b 逐位同、`motion-t2-ref/cand`；环境 B `aws-t3-closed-s100`、t8 C8/C32）。modulation 关闭态没有单独跑过对拍，`v1-postclean-g3` 登记的 UNVERIFIED 状态未变。F 节第一块（两份 YAML 的 Dataset 逐字节对拍）把数据侧从论证变成实测；模型侧仍是源码级——若要实测，唯一对照是「接入前 HEAD `06220c4~1` vs 当前 HEAD 各跑官方 modul.yaml N 步比标量 hex」（接入前 HEAD 不认 8×8 库，只能在 400ep 4×4 库跑），属另立任务，不在本计划内。

**8. 隔离：训练不从主工作副本起跑，而从 detached worktree 快照起跑。** 主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA` 在 65 小时训练期间随时可能有新 commit（留档、评估、其他任务）；Python 在 dataloader worker 重建、checkpoint 保存等时刻会重新 import 源文件，主树被改就可能把新代码带进正在跑的训练。做法沿用 t8 那轮 REF worktree 的既有口径：起跑 commit 记为 `TRAIN_HEAD`，`git worktree add --detach v1-store/worktrees/train-v2-1600ep-m8x8-modul-b128-60k $TRAIN_HEAD` 建只读快照；训练命令 `cd` 到快照根、`PYTHONPATH=<快照>/src`（压过主树 `.venv` 里 editable 安装的绝对路径 `.pth`）、`UV_PROJECT_ENVIRONMENT=<主树>/.venv`（快照里没有 venv，复用主树的）、`uv run --no-sync`；`source` 的是**主树**的 `scripts/training/paths.sh`（它按自身位置解析 `REPO_ROOT`，所以 `V1_STORE` / `OPENPI_DATA_HOME` 仍指主树 `v1-store`），数据、权重、checkpoint、日志全部走主树 `v1-store` 绝对路径；`get_history_config` 按 cwd 相对路径读 YAML，cwd 是快照，读到的是快照里的 YAML。起跑前一行 preflight 打印并断言 `mme_vla_suite`、`openpi` 两个包的 `find_spec().origin` 与 `train.py` 路径都落在快照内，写进日志与 `launch.md`。smoke 也从快照跑，同时验证隔离机制本身。训练结束、`result.md` 提交后再 `git worktree remove` 快照。已知残余：`packages/openpi-client` 是主树 editable 安装（policy 服务用，训练不 import）；`third_party/` 子模块快照里未初始化，训练不需要。

**当前状态**：`commitV9.5` 已 push；守卫白名单已写进工作区但未提交；smoke 临时产物与 tmux `m8-smoke` 已清理，GPU 4–7 空闲。

### 执行顺序（第五稿）

1. 新 YAML + 新条目 → 验证 → `commitV9.5` → push。**（已完成，`e0bcb45`）**
2. 守卫白名单 + G13 测试改写 → 第一块对拍 → 第二块 100 步 → `commitV9.6` → push。（第二部分 F 节）
3. 本计划固化为根目录 `v2-1600ep-m8x8-modul-training-plan.md` + `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md` 初稿 + `docs/training-doc/README.md` 加行 → `docs:` commit → push → 记 `TRAIN_HEAD` → 建 worktree 快照。（第二部分 G、I 节）
4. 从快照起 smoke（tmux `m8-smoke`）→ 核判据（含导入来源断言）→ 删临时产物。（B 节）
5. `launch.md` 补 smoke 与步骤 2 的判定行摘录 → `docs:` commit → push（主树；快照仍停在 `TRAIN_HEAD`，训练代码不受影响）→ 从快照起正式 run（tmux `m8-prod`）→ 挂 Monitor。（C 节）
6. 跑完（约 65 h 后）：`result.md` + `records/` → `docs:` commit → push → `git worktree remove` 快照。评估另立任务。

---

## 第二部分（技术细节，供 agent 追踪）

### A. 配置改动（两文件，一次 commit）

**A1. 新 YAML** `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml`：

```yaml
# 8 帧 × 8×8（token_per_image 64 × 8 帧 = 512 位）+ modulation 接入，无 motion 节（关闭态）。
# 以 perceptual-framesamp-modul.yaml 为底，唯一差异 token_per_image: 16 → 64；对应库布局 framesamp-8x8-v1。
budget: 512
num_views: 1
token_per_image: 64
streaming_obs_horizon: 16
pool_type: mean
use_pos_emb: true
use_state_emb: false
memory_feature:
  img:
    net: identity
    input_dim: 2048
  pos:
    input_dim: 768
    hidden_dim: 768
  state:
    input_dim: 8
    hidden_dim: 512
integration_type: modulation
memory_token_dim: 1024
representation_type: perceptual
perceptual_memory:
  type: frame_sampling
```

**A2. 新条目** 追加在 `_CONFIGS` 列表 `mme_vla_suite_b128` 之后（`src/mme_vla_suite/training/config.py`），带中文注释说明与 b128 条目的差异：

```python
    # 环境 B 新库 1600ep 档（2026-09-15，用户拍板）：与 mme_vla_suite_b128 的差异只有四项——
    #   num_train_steps 40_000 → 60_000；peak_lr / decay_lr 1e-4 → 5e-5（b128 下不做线性缩放，等效为官方
    #   5e-5@b64 的一半，用延长步数补偿）；decay_steps 50_000 → 60_000（peak==decay 时余弦段为常数，只为自洽）；
    #   data.assets 指到 1600ep 新库的 norm_stats。warmup 5_000 不变（= 640k 样本，与官方 10k×64 同）。
    TrainConfig(
        name="mme_vla_suite_b128_60k",
        model=history_pi0.HistoryPi0Config(pi05=True, action_horizon=20, use_history=True,
                                           history_config=None, discrete_state_input=False),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            assets=AssetsConfig(assets_dir="v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da",
                                asset_id="robomme"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=60_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params")),
        num_train_steps=60_000,
        save_interval=5_000,
        keep_period=5_000,
        project_name="robomme-framesamp",
        num_workers=8,
        ema_decay=0.999,
        fsdp_devices=4,
    ),
```

**验证（≤ 2 分钟，CPU）**：
```bash
diff src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml \
     src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml   # 期望：仅注释 + token_per_image 一行
JAX_PLATFORMS=cpu uv run --no-sync python - <<'PY'
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.config import get_config
c=get_history_config('perceptual-framesamp-modul-8frame-8x8.yaml')
assert (c.budget,c.token_per_image,c.num_views,c.integration_type,c.memory_token_dim)==(512,64,1,'modulation',1024)
assert getattr(c,'motion',None) is None
t=get_config('mme_vla_suite_b128_60k'); s=t.lr_schedule
assert (t.batch_size,t.num_train_steps,s.warmup_steps,s.peak_lr,s.decay_lr,t.save_interval,t.keep_period,t.num_workers,t.fsdp_devices,t.ema_decay)==(128,60000,5000,5e-5,5e-5,5000,5000,8,4,0.999)
b=get_config('mme_vla_suite_b128'); assert (b.num_train_steps,b.lr_schedule.peak_lr)==(40000,1e-4)   # 旧条目未动
sched=s.create(); import jax.numpy as jnp
assert abs(float(sched(5000))-5e-5)<1e-12 and abs(float(sched(59999))-5e-5)<1e-12   # warmup 后恒定
print('CONFIG_OK')
PY
git diff --check
```
commit：`git add` 仅这两个文件；subject `commitV9.5: 新增 modulation 8帧8×8 配置与 b128 60k lr5e-5 训练条目`；`git push`。

### B. smoke（20 步、真实 b128 / 4 卡、临时 run，跑完删）

run_name `smoke-m8x8-modul-$(date -u +%Y%m%dT%H%M%SZ)`，tmux `m8-smoke`，日志 `v1-store/logs/m8-smoke.log`。**从 worktree 快照起跑（G 节），不在主树跑。** 前置：主树 `git status --porcelain` 空且 HEAD == `TRAIN_HEAD`；快照存在且 `git -C <快照> rev-parse HEAD == TRAIN_HEAD`；`nvidia-smi --id=4,5,6,7` 显存 0；run 根不存在。

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
WT=$MAIN/v1-store/worktrees/train-v2-1600ep-m8x8-modul-b128-60k
source "$MAIN/scripts/training/paths.sh"    # 主树 paths.sh：V1_STORE / OPENPI_DATA_HOME 指主树 v1-store，不覆盖 HOME
cd "$WT"                                     # cwd = 快照：get_history_config 按 cwd 相对路径读快照里的 YAML
export PYTHONPATH="$WT/src" UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset JAX_PLATFORMS MMEVLA_MOTION_STORE
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# 导入来源断言（G 节）：三个路径都必须以 $WT 开头，否则不起跑
JAX_PLATFORMS=cpu uv run --no-sync python - "$WT" <<'PY'
import importlib.util, pathlib, sys
wt = pathlib.Path(sys.argv[1]).resolve()
o = {n: pathlib.Path(importlib.util.find_spec(n).origin).resolve() for n in ("mme_vla_suite", "openpi")}
o["train.py"] = (pathlib.Path("scripts/training/train.py")).resolve()
bad = {k: str(v) for k, v in o.items() if wt not in v.parents}
print("IMPORT_ORIGIN=" + ("PASS" if not bad else "FAIL") + " " + " ".join(f"{k}={v}" for k, v in o.items()))
sys.exit(1 if bad else 0)
PY
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN" TRAIN_RECORD_DIR="$V1_STORE/bench/$RUN"
export MMEVLA_FRAMESAMP_SOURCE="$V1_STORE/datasets/4task-v2-1600ep-604f16da/source"
export MMEVLA_FRAMESAMP_MANIFEST="$V1_STORE/datasets/4task-v2-1600ep-604f16da/meta/episode_manifest.json"
export WANDB_MODE=disabled
test -z "$(git -C "$MAIN" status --porcelain)"; test "$(git -C "$WT" rev-parse HEAD)" = "$TRAIN_HEAD"; test ! -e "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN"
set -o pipefail
uv run --no-sync python scripts/training/train.py mme_vla_suite_b128_60k \
  --exp-name "$RUN" --num-train-steps 20 --log-interval 1 \
  --assets-base-dir "$V1_STORE/train-assets" \
  --checkpoint-base-dir "$V1_STORE/train-runs" \
  --dataset-path "$V1_STORE/datasets/4task-v2-1600ep-604f16da/framesamp-8x8" \
  --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml \
  --no-wandb-enabled 2>&1 | tee -a "$LOG"; echo "EXIT_CODE=${PIPESTATUS[0]}" | tee -a "$LOG"
```
（`--num-train-steps 20 --log-interval 1 --no-wandb-enabled` 为 smoke 专用覆盖，正式 run 不带。）

Monitor：
```bash
tail -n +1 -F v1-store/logs/m8-smoke.log | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "Integration Type|Step 19:|EXIT_CODE=|Error|Traceback|RESOURCE_EXHAUSTED|out of memory"
```

判据（全部满足才进入 C）：
- 日志含 `Integration Type: modulation`；`EXIT_CODE=0`；`Step 0`…`Step 19` 的 loss / grad_norm / mem_enc_norm 全部有限，无 `RESOURCE_EXHAUSTED`。
- run 根 `history_config.resolved.yaml` 含 `token_per_image: 64`、`integration_type: modulation`；`motion_provenance.json` 的 `motion_enabled=false`、framesamp manifest sha = `4cd5a170…`。
- checkpoint 19 的 `assets/robomme/norm_stats.json` sha256 == `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`（证明条目内 `assets_dir` 生效）。
- 参数树：`JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/legacy-eval/check_ckpt_param_tree.py --config mme_vla_suite_b128_60k --ckpt-dir <run根>/19 --out <scratchpad>/param_tree.json` 得 `PARAM_TREE_EXACT=PASS`，json 中出现 `q_einsum_mem` / `kv_einsum_mem` / `out_einsum_mem` / `mem_rms_norm` 路径。
- 显存以「20 步跑完、退出 0」为 OOM 判据；`nvidia-smi` 数字只记录不判读。

清理：`rm -rf` 精确路径 `v1-store/train-runs/mme_vla_suite_b128_60k/$RUN`、`v1-store/bench/$RUN`、`v1-store/cache/jax/$RUN`；`tmux kill-session -t m8-smoke`（删前删后各 `tmux ls`）。判定行与 20 条 Step 行摘录进正式 run 的 `launch.md`「起跑前 smoke」节，不单独建目录。

### C. 正式 run：`v2-1600ep-m8x8-modul-b128-60k`

**起跑前**：写 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md`（起跑 commit、差异表、lr 口径说明、完整命令、数据三件套 sha、输出路径、判据、smoke 摘录、tmux 清单 `m8-prod`）；`docs/training-doc/README.md` 表格加一行；`git add` 两文件 → `docs: v2-1600ep-m8x8-modul-b128-60k 起跑留档` → push。起跑 HEAD clean 且含该 commit。

**命令**（tmux `m8-prod`，日志 `v1-store/logs/v2-1600ep-m8x8-modul-b128-60k.log`；**从 worktree 快照起跑，G 节**）：
```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=v2-1600ep-m8x8-modul-b128-60k
WT=$MAIN/v1-store/worktrees/train-$RUN
LOG=$MAIN/v1-store/logs/$RUN.log
source "$MAIN/scripts/training/paths.sh"            # V1_STORE / OPENPI_DATA_HOME 指主树 v1-store
set -a; . "$MAIN/v1-store/secrets/wandb.env"; set +a   # WANDB_API_KEY / WANDB_ENTITY，不进日志
cd "$WT"
export PYTHONPATH="$WT/src" UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset JAX_PLATFORMS MMEVLA_MOTION_STORE WANDB_MODE
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN" TRAIN_RECORD_DIR="$V1_STORE/bench/$RUN"
export MMEVLA_FRAMESAMP_SOURCE="$V1_STORE/datasets/4task-v2-1600ep-604f16da/source"
export MMEVLA_FRAMESAMP_MANIFEST="$V1_STORE/datasets/4task-v2-1600ep-604f16da/meta/episode_manifest.json"
mkdir -p "$TRAIN_RECORD_DIR"
test "$(git -C "$WT" rev-parse HEAD)" = "$TRAIN_HEAD"; test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN"
printf 'TRAIN_HEAD=%s\nWORKTREE=%s\nSTART_UTC=%s\n' "$(git -C "$WT" rev-parse HEAD)" "$WT" "$(date -u +%FT%TZ)" | tee -a "$LOG"
# 导入来源断言（同 B 节那段 python，IMPORT_ORIGIN=PASS 才继续）
nvidia-smi --id=4,5,6,7 --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 15 > "$TRAIN_RECORD_DIR/gpu_util_15s_full.csv" &   # 记 PID
set -o pipefail
uv run --no-sync python scripts/training/train.py mme_vla_suite_b128_60k \
  --exp-name "$RUN" \
  --assets-base-dir "$V1_STORE/train-assets" \
  --checkpoint-base-dir "$V1_STORE/train-runs" \
  --dataset-path "$V1_STORE/datasets/4task-v2-1600ep-604f16da/framesamp-8x8" \
  --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml 2>&1 | tee -a "$LOG"
echo "EXIT_CODE=${PIPESTATUS[0]}" | tee -a "$LOG"
```
未覆盖项全走条目默认：`num_train_steps 60000`、`batch_size 128`、lr 5e-5 / warmup 5k、`num_workers 8`、`fsdp_devices 4`、`seed 42`、`log_interval 100`、`save_interval/keep_period 5000`、wandb 开（project `robomme-framesamp`，run 名同 exp-name）。前 30 min 另起 `-lms 500` 密采（AGENTS 16）。

Monitor：
```bash
tail -n +1 -F v1-store/logs/v2-1600ep-m8x8-modul-b128-60k.log | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "Step [0-9]*00:|Finished asynchronous save|EXIT_CODE=|Error|Traceback|RESOURCE_EXHAUSTED|nan"
```
起跑后 300 步复核稳态 s/step（tqdm `Progress on` 行时间戳差分，同 `bench-b128-util` 口径），在 launch.md 补 ETA。

**输出**：checkpoint `v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-modul-b128-60k/{5000,…,55000,59999}`（12 个）；日志与 GPU csv 落 `records/`；wandb run `robomme-framesamp/v2-1600ep-m8x8-modul-b128-60k`。

**跑完**：`result.md`（起止、EXIT_CODE、12 ckpt 齐全、loss 里程碑、稳态 s/step、util 均值 / 0% 占比 / 慢步分层、显存口径声明、下一步）；`records/` 收清洗日志、两份 GPU csv；不归档权重。`docs:` commit + push。

### D. 关键文件与复用

- 配置：`src/mme_vla_suite/training/config.py`（新增条目；`mme_vla_suite` 与 `mme_vla_suite_b128` 不动）；`src/openpi/training/optimizer.py::CosineDecaySchedule`（不动）。
- 训练入口：`scripts/training/train.py`（`init_history_config` 写 provenance）；环境变量解析 `src/mme_vla_suite/training/dataloader.py::_create_framesamp_dataset`。
- 路径源：`scripts/training/paths.sh`。
- 参数树核对：`scripts/training/legacy-eval/check_ckpt_param_tree.py`。
- 留档样板：`docs/training-doc/awsprod40k-b128-motion/{launch,result}.md`、`docs/training-doc/v2b-read20-20260914T174147Z/`。

### F. 守卫放宽与增补验证（第四稿新增；插在步骤 1 与步骤 2 之间）

**为什么要多跑。** `framesamp_dataset.py::FrameSampDataset.__init__` 的 `_req` 里有两条 commitV3.1 按 Codex 审计（G13）加的守卫：`integration_type == "context"`、`memory_token_dim == 2048`，目的是挡住「同形的 modul 配置」。它们只在 `__init__` 拒绝配置，后续交付路径（`__getitem__`、`_pad`、store 读取）不读这两个键。放宽它们是对 dataloader 文件的改动，按 AGENTS 第 18 条要证明不改数，所以比第三稿多跑三项验证（F2–F4），再重跑 smoke。训练链路上没有其他 context 专属守卫（`grep` 过 `training/`、`train.py`、`compute_norm_stats.py`、`g0/`、`policies/`，只此两行；`history_pi0.py` 自身接受 modulation）。

**F1. 守卫改成成对白名单**（已写入工作区，未 commit）：

```python
        _req((str(hc.integration_type), int(hc.memory_token_dim)) in {("context", 2048), ("modulation", 1024)},
             f"(integration_type, memory_token_dim)=({hc.integration_type!r}, {hc.memory_token_dim}) "
             f"不在支持的 (context,2048)/(modulation,1024) 档位中")
```
context 对 `budget/token_per_image/num_views/feature dims/use_state_emb` 的其余断言原样保留。expert（`memory_token_dim 1024` 但 `integration_type expert`）与错配对（如 `modulation,2048`）仍拒。

**F1b. G13 测试改写** `scripts/training/tests/test_pack_guards.py::test_g13_modul_config_rejected` → 改为三条断言：`perceptual-framesamp-modul.yaml` 现在**通过**构造（4×4 迷你库、`(modulation,1024)`）；`perceptual-framesamp-expert.yaml` 拒；用 `OmegaConf` 把 modul.yaml 的 `memory_token_dim` 改成 2048 的错配对拒。该测试依赖 `v1-store/datasets/ref-shard` 迷你库，**本环境不存在**，pytest 跑不了；同样三条断言改在 1600ep 库上用一次性脚本执行（F2 里一并做），测试文件的改写只保证逻辑正确、在留档里写明「本环境未执行 pytest」。

**F2. 第一块：轻量对拍 + 三条守卫断言（CPU，约 3–5 分钟）。** 一次性脚本放 scratchpad、`JAX_PLATFORMS=cpu uv run --no-sync`，cwd 仓库根：
- 守卫：`get_history_config` 分别加载 modul-8frame-8x8（期望通过）、expert（期望 `ValueError` 含「形制断言失败」）、modul-8frame-8x8 且 `memory_token_dim` 改 2048（期望拒）；另在 4×4 库 `framesamp/` 上加载 `perceptual-framesamp-modul.yaml`（期望通过）。
- 对拍：`_create_framesamp_dataset`（`src/mme_vla_suite/training/dataloader.py`）分别用 `perceptual-framesamp-modul-8frame-8x8.yaml` 与 `perceptual-framesamp-context-8frame-8x8.yaml` 在 `framesamp-8x8` 库上构造 Dataset（`MMEVLA_FRAMESAMP_SOURCE/MANIFEST` 同正式 run），取 256 个索引（`np.random.default_rng(20260915).choice(605611, 256)` 并加 0、605610 与每个 episode 边界附近各 1 个），逐样本逐键比 `dtype / shape / raw sha256`。判定行 `DS_EQUIV=PASS samples=<n> keys=<k> mismatches=0`；任一不等即停。

**F3. 第二块：复用 t8-c8-b 固化轨迹前 100 步（GPU 4,5，约 10–15 分钟）。** 完全照抄 `docs/training-doc/t8-c8-b/launch.md` 的环境与命令，只改：run 名 `t8-c8-guard-s100`、`--num-train-steps 100`、`BENCH_CHECKSUM=0 BENCH_BATCH_DIGESTS=1`（与用户 09-14 批准的 `-s100` 排错口径相同）、去掉 `BENCH_SAVE_FINAL_CKPT`。要点：
- 数据：400ep 库 `v1-store/datasets/4task-motion-400ep/framesamp-8x8`，norm_stats `robomme-400ep`（sha `750a8e9b…`），YAML `perceptual-framesamp-context-8frame-8x8.yaml`，batch 8 / worker 4 / seed 42 / fsdp 2 / `--log-interval 1`，`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、`CUDA_VISIBLE_DEVICES=4,5`、`WANDB_MODE=disabled`。
- 指纹 preflight（AGENTS 18 第三段）：`check_baseline_env.py dump` → `check --base v1-store/bench/8x8/t8-c8-b --record-dir <rec> --steps 100 --batch-size 8 --dataset <400ep 8x8> --allow-difference dataset.store_meta_sha256`，要求 `BASELINE_ENV=PASS`（基线 env.json 的 fingerprint：jax 0.5.3、A100 驱动 595.71.05、norm_stats 与 pi05_base 摘要等；`v1-store/bench/8x8/t8-c8-b/` 产物齐全，`scalars_hex.tsv` 1001 行且与 docs 归档 `cmp` 相同）。FAIL 即停、不放宽。
- 判据：`project_scalars.py metrics.jsonl → scalars_hex.tsv`，前 100 行（step 0–99）五列 hex 与 `docs/training-doc/t8-c8-b/records/scalars_hex.tsv` 前 100 行逐字节相同；`batch_digests.jsonl` 前 100 步摘要与基线相同。判定行 `GUARD_GRAD_100=PASS steps=100 mismatches=0`。
- 清理：`v1-store/bench/8x8/t8-c8-guard-s100`、`v1-store/train-runs/t8-c8-guard-s100`、`v1-store/cache/jax/t8-c8-guard-s100`、tmux `m8-guard`。≤ 15 分钟不建独立留档，判定行摘录进正式 run 的 `launch.md`。

**F4. commit + push**：F2、F3 全 PASS 后 `git add` 仅 `framesamp_dataset.py` 与 `test_pack_guards.py`，subject `commitV9.6: dataset 形制守卫放宽为 (integration_type, memory_token_dim) 成对白名单`，body 写两块判定行原文与耗时；push。之后回到步骤 2（smoke）→ 3 → 4，不变。

**对应第一部分「执行顺序（第四稿）」的步骤 2**：F1/F1b → F2 → F3 → F4，多出的墙钟约 20–25 分钟；之后步骤 3 smoke（B 节）→ 步骤 4 正式 run（C 节）→ 步骤 5 收尾。

### G. 隔离机制：detached worktree 快照（步骤 3 建、步骤 4/5 用、步骤 6 删）

**建**（步骤 3，`docs:` commit push 之后、主树 clean）：
```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
test -z "$(git status --porcelain)"
TRAIN_HEAD=$(git rev-parse HEAD); echo "TRAIN_HEAD=$TRAIN_HEAD"       # 写进 launch.md
git worktree add --detach v1-store/worktrees/train-v2-1600ep-m8x8-modul-b128-60k "$TRAIN_HEAD"
git worktree list                                                      # 确认新行且为 detached
```
`v1-store/worktrees/` 已有先例（`official-89efeaab`），整体不进 git。快照里没有 `v1-store/`、`.venv`、`third_party/` 子模块，训练不需要它们（数据 / 权重 / checkpoint 全走主树 `V1_STORE` 绝对路径，venv 由 `UV_PROJECT_ENVIRONMENT` 指主树）。

**用**（B、C 节命令已改）：三件事缺一不可——`cd $WT`（YAML 从快照读）、`PYTHONPATH=$WT/src`（`mme_vla_suite` 与 `openpi` 都从快照 import，压过主树 `.venv/…/_editable_impl_openpi.pth` 的绝对路径）、`UV_PROJECT_ENVIRONMENT=$MAIN/.venv` + `uv run --no-sync`（不让 uv 在快照里另建 venv）。起跑前 `IMPORT_ORIGIN=PASS` 断言三个路径都在 `$WT` 下；`launch.md` 记 `TRAIN_HEAD`、`WORKTREE`、三个 origin 原文。主树期间可以照常 commit / push 文档，快照始终停在 `TRAIN_HEAD`。

**禁**：训练期间不 `git worktree remove` / `prune`、不在快照里改文件、不 `git checkout` 快照；`git clean -x` 全仓禁止（AGENTS 19 附）。

**删**（步骤 6，`result.md` 提交之后）：`git worktree remove v1-store/worktrees/train-v2-1600ep-m8x8-modul-b128-60k && git worktree prune && git worktree list`。删前 `tmux has-session -t m8-prod` 必须已不存在。

**已知残余**：`packages/openpi-client` 是主树的 editable 安装（policy server 用，`train.py` 不 import）；`scripts/training/paths.sh` 用的是主树那份（只导出路径变量，不含训练逻辑）。两者写进 launch.md。

### H. 审计结论固化（第一部分第 7 点的落点）

结论正文见第一部分第 7 点。固化位置：根目录计划文件（I 节）第一部分同款一节；`launch.md`「与官方 / 上次 run 的关系」一节引用该节并写明「modulation 关闭态无运行时对拍，本 run 不据此宣称与接入前逐位等价」。不改 `docs/motion-memory.md` 等正本（评审性结论，非链路事实；正本改动另立）。

### I. 计划固化到仓库根目录（步骤 3）

新建 `v2-1600ep-m8x8-modul-training-plan.md`（与 `8frame-8x8-training-plan.md` 同级、同体例）：内容 = 本计划文件全文（去掉 harness 版本注，保留第一 / 第二部分与 F、G、H 节），文首加一行状态说明「实施过程档案；结果以 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/` 为准」。与 `launch.md` 初稿、`docs/training-doc/README.md` 加行一起 `git add` 三个文件，subject `docs: v2-1600ep-m8x8-modul-b128-60k 计划固化与起跑留档初稿`，push。之后再记 `TRAIN_HEAD`（保证快照里含这份计划）。验证：`git diff --check`；Markdown 链接 `docs/training-doc/README.md` → 新 run 目录可解析。

### E. 红线与不做的事

- 不改 dataloader / 模型代码（AGENTS 18 不触发）；不动既有两个条目。
- 训练从 worktree 快照起跑（G 节），主树期间的 commit 不影响在跑训练；但仍不在快照里改任何文件，不建库、不起第二个 GPU 大任务。
- tmux 只按确切名 `m8-smoke` / `m8-prod` kill；用户会话 `0`、`1`、`claude-private`、`codex`、`codex-repo` 一律不动。
- 评估（legacy-eval / motion-variance 对 modulation + 8×8 的 prefix 长度、VideoRepick 驱动）不在本轮范围。
