# 新库 motion 表构建 + modulation 8×8 接入 motion 计划（v2-motion）

> **状态：方案已固化，待 budget 与 run_name 拍板；未实施。** 2026-09-16 起草，2026-09-17 按用户要求重写第一部分。环境判定：**环境 B（AWS 单机 8×A100-SXM4-80GB）**，主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`。基线 `v2-1600ep-m8x8-modul-b128-60k` 已于 `2026-09-16T20:53:31Z` 以 `EXIT_CODE=0` 跑完，主副本已解锁、开发副本 `-temp` 已删除（`5432ba2`），**8 卡全空，一切工作回主副本进行**。无 turbo、无 GreatLakes。
>
> 本文按 `AGENTS.md` 第 2 条分两部分。正本 `docs/motion-memory.md` 写的是 context + 32 帧 4×4 + 旧四任务口径，本文只写**与其不同**的部分；实施后正本另立 `docs:` 更新。

**用户已拍板**

| 项 | 决定 | 日期 |
|---|---|---|
| 接入目标 | **modulation 8×8**，与已跑完的基线 `v2-1600ep-m8x8-modul-b128-60k` 同 YAML、只多 `motion` 节 | 09-16 |
| demo 段窗口 | **全覆盖 + 真实帧 ≥ 17**：起点 `s ∈ range(0, es, 16)` 且 `es − s ≥ 17`；窗口越过 `es − 1` 的部分用第 `es − 1` 帧重复填充凑满 33 帧 | 09-16 |
| exec 段窗口 | **不变**：尾端 ≤ 当前帧的完整 33 帧窗，不补帧 | 09-16 |
| 新库 motion 表 | **单独构建**（1600 集新库当前没有 motion 表） | 09-16 |
| 正式 run 步数 | **80k 步**（用户原话「一路做到开始 80k 训练」）；落点为新具名配置 `mme_vla_suite_b128_80k`，`mme_vla_suite_b128_60k` 与官方条目一字不动 | 09-17 |
| `motion.budget` | **待拍板**（本文推荐 160，见「budget」节；表的构建不依赖 budget） | — |
| `run_name` | **待确认**（本文建议 `v2-1600ep-m8x8-modul-motion-b128-80k`，起跑前按 `AGENTS.md` 第 6 条确认） | — |

---

## 第一部分（给人看）

### 这轮做什么

给 1600 集新库（`v1-store/datasets/4task-v2-1600ep-604f16da`：BinFill / RouteStick / VideoRepick / VideoUnmaskSwap，605,611 个执行样本）建一张 motion token 表，然后用已跑完的 modulation 8×8 基线配置**只加一个 `motion` 节**，训练一条 80k 步的「+motion」模型。全文分五块：① demo 段补帧规则（数轴）→ ② 建表五步、每步做没做 → ③ 训练链路改什么（dataloader + 模型）→ ④ 推理侧改什么 → ⑤ budget 表 → ⑥ 从现在到 80k 起跑的 commit 与执行顺序。

### 一、demo 段怎么补：数轴

**先说清楚窗口是什么。** 一个 motion 窗口 = 连续 33 帧 `[s, s+32]`，喂给冻结的 MotionJEPA encoder 得到一个 768 维 token。起点 `s` 钉在段内网格 `0, 16, 32, …` 上（`motion_store.GRID_STRIDE = 16`、`WINDOW_FRAMES = 33`），demo 段和 exec 段各自从段起点算网格，窗口不跨段。

**现行规则的问题只出在 demo 段尾部。** 以 RouteStick 一集 `es = 100` 为例（demo 段帧号 0–99，exec 段从帧 100 起）：

```
demo 帧号   0        16       32       48       64       80       96  99 │100 exec →
            ├────────┼────────┼────────┼────────┼────────┼────────┼──┤
现行规则    [0 ══════════════ 32]                                        s=0   真 33
                     [16 ═════════════ 48]                               s=16  真 33
                              [32 ═════════════ 64]                      s=32  真 33
                                       [48 ═════════════ 80]             s=48  真 33
                                                [64 ═════════════ 96]    s=64  真 33
                                                         ✗ s=80: 80+32=112 > 99，整窗越段，不编
                                                            → 帧 97–99 的运动没有任何窗覆盖
```

现行合法条件是 `s + 32 ≤ es − 1`（`motion_store.visible_motion_rows` 的 demo 分支），越段的窗直接不编。demo 尾部没被任何窗覆盖的帧数 = `(es − 1) − (s_last + 32)`，在新库上于 0–15 之间均匀分布——也就是说，demo 段最后那 0–15 帧的运动（往往是演示动作的收尾）模型看不到。

**新规则：起点只要还剩 ≥ 17 帧真实帧就编一窗，越段的部分用 demo 最后一帧填满。** 同一集：

```
demo 帧号   0        16       32       48       64       80       96  99 │100 exec →
            ├────────┼────────┼────────┼────────┼────────┼────────┼──┤
新规则      前 5 窗与现行完全相同（字节不变）
                                                         [80 ═══════ 99│99 99 … 99]   s=80  真 20 + 补 13（第 99 帧重复 13 次）
                                                                  ✗ s=96: 真实帧只剩 4 < 17，不编
```

合法条件改为 `s + 16 ≤ es − 1`（即 `es − s ≥ 17`）；窗口切片是 `frames[s : min(s+33, es)]`，不足 33 帧的用 `frames[es−1]` 重复补齐。补的是**静止帧**：补帧段内没有运动，encoder 看到的是「前 20 帧在动、后 13 帧定格」，语义就是「这段动作到此结束」。

**为什么恰好每集多 1 窗、且尾部一定被盖住。** 现行接受 `s ≤ es − 33`，新规则接受 `s ≤ es − 17`，多出来的区间 `(es − 33, es − 17]` 宽度恰好 16 = 一个网格步，所以里面**有且只有一个**网格点。这个新窗起点 `s_new > es − 33`，即 `s_new + 32 ≥ es`，窗口右端一定覆盖到第 `es − 1` 帧——demo 段每一帧都至少落在一个窗里。换一集 VideoUnmaskSwap `es = 114` 验证：现行 `s ≤ 81` → 6 窗（0…80）；新规则 `s ≤ 97` → 7 窗，新增 `s = 96`，真 18 帧 + 补 15 帧。

**exec 段一个字不改。** exec 的合法条件仍是 `u + 32 ≤ t − es`（窗口尾端不能超过当前帧 t），因为 exec 是在线增量到货、当前帧之后的帧根本不存在，没有「段尾」可补；训练侧和在线侧本来就同式。

**公式落地。** 统一写成 `seg_num_grid(L, min_real) = len(range(0, max(0, L − (min_real − 1)), 16))`，demo 传 `min_real = 17`、exec 传 `33`（exec 与现行 `len(range(0, max(0, L − 32), 16))` 逐字相等）。新库按清单实算（2026-09-17 复核）：

| 项 | 现行 | 新规则 |
|---|---:|---:|
| demo 窗 | 34,313 | **35,913** |
| exec 窗 | 35,403 | 35,403 |
| 总行数 | 69,716 | **71,316** |
| 补帧窗 | 0 | **1,600**（= 集数，每集恰 1 个） |

四处代码各自实现这一个公式（`datastore/motion_store.py::seg_num_grid`、`wan/wan_common.py::seg_num_grid`、`wan/oracle_driver.py` 独立重算、`policies/framesamp_memory.py::_encode_ready_windows` / `visible_motion_frames`），**必须四处同改**，第二部分 A 节逐处列出。

### 二、建表流程：五步，每步做没做

建表链路照抄 400ep 库的留档 `docs/dataset-build-doc/4task-motion-400ep/launch.md`（`run_local.py --stage wan` → `--stage encode` → `pack_motion_store.py pack | verify` → oracle 对拍 → 账目检查），**链路不改，只换口径与规模**。现状：新库目录下只有 `framesamp` / `framesamp-8x8` / `source` / `meta` / `logs`，**motion 相关五样（`wan-latents` / `motion-tokens` / `motion` / `oracle` / 留档）一样都没有**。原始 H5 在 `v1-store/raw-h5/4task-20260912-v2/record_dataset_{BinFill,RouteStick,VideoRepick,VideoUnmaskSwap}.h5`（四个合并文件 793.8 GB）；wan 子 venv `v1-store/venvs/wan` 完好；encoder 权重 `v1-store/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt`（sha256 `bae96037…c15a`，`ASSETS_LOCK.json` 钉死）。

| 步 | 做什么 | 状态 | 产物 / 判定行 | 预计耗时（8 卡） |
|---|---|---|---|---|
| **1. 改代码** | 四处网格公式改 `min_real` 参数化；`extract_wan.py` 切窗时对 demo 段补帧；`oracle_driver.py` 独立实现补帧并支持抽样；`motion_checks.py` 加 `a11` 补帧账；`pack_motion_store.py` 写新布局名与三新键 | **未做** | `commitV10.0`，工作区必须干净才能进第 2 步 | CPU 单测 < 5 min |
| **2. Wan 抽取** | `run_local.py --stage wan`：每段读 H5 帧 → 逐网格切 33 帧窗（demo 尾窗补帧）→ 冻结 Wan VAE 编成 `(9,16,32,32)` f32 latent → `wan-latents/<段>.bin` + `metadata.json`（每窗记 `real_frames` / `pad_frames` / 输入 sha256） | **未跑** | `STAGE_DONE stage=wan workers=8 items=3200`（demo 1600 段 + exec 1600 段）；71,316 × 589,824 B ≈ **39.2 GiB** | 400ep 实测 0.69 窗/s/worker → ≈ **3.6 h** |
| **3. encode** | `run_local.py --stage encode`：每窗 latent → MotionJEPA encoder → 768 维 f32 token，`motion-tokens/<段>.f32.bin`（行序 = 网格序） | **未跑** | `STAGE_DONE stage=encode workers=8 items=3200`；**与第 2 步之间零 commit**（`gather_provenance` 断言 wan / encode 两阶段 `git_commit` 唯一，400ep 曾因跨 commit 被拒、重抽一遍） | 7.7 窗/s/worker → ≈ **19 min** |
| **4. pack / verify** | 按段基址拼成整表 `motion/motion.f32.bin`（71,316 × 3,072 B ≈ **209 MiB**）+ `meta/motion_index.json`（段基址表）+ `meta/store_meta.json`（契约：`layout` / `grid_stride` / `window_frames` / 三新键 / provenance） | **未跑** | `PACK_MOTION_DONE=1` → `VERIFY_MOTION=PASS scanned=71316 mismatches=0`（`state` 从 `packed` 翻 `verified`，训练侧 `require_verified` 才放行） | 分钟级 |
| **5. oracle 与账目** | D3：MotionJEPA 仓库自己的 encoder 对 71,316 行**全量**重算逐位比；D2：MotionJEPA 仓库自己的 VAE 对**抽样**窗（1,600 个补帧窗全部 + 其余随机 10% ≈ 6,972 窗，seed 0）从 H5 独立切窗、独立补帧、重编逐位比；账目 `a7`（字节数）/ `a9set`（500 个 (g,t) 的可见集合 == 独立实现）/ `a10`（行数 71,316 = exec 35,403 + demo 35,913）/ **`a11`（新增：补帧账，补帧窗 == 1,600、每窗 `real + pad == 33`、exec 段 `pad == 0`）** | **未跑**（`--sample-spec` 与 `a11` 属第 1 步新代码） | `ORACLE_ENCODER=DONE rows=71316` + `ENCODER_BITEXACT=PASS compared=71316 mismatches=0`；`ORACLE_VAE=DONE windows=<抽样数> frame_mismatches=0` + `WAN_BITEXACT=PASS compared=<抽样数> frame_mismatches=0 latent_mismatches=0`；`A7_BYTES=PASS` / `A9_INDEXSET=PASS samples=500 mismatches=0` / `A10_ROWS=PASS rows=71316 exec=35403 demo=35913` / `A11_PAD=PASS padded=1600` | D3 ≈ 19 min（单卡）；D2 抽样 ≈ 8,600 窗，8 片 ≈ 1.2 h |

400ep 库的 `a5`（40ep 帧与 400ep 帧同源）和 `a6`（40ep 清单 vs 400ep 清单）是两库交叉检查，新库没有对照库，**不适用**。第 2–5 步全部在主副本 clean HEAD 上跑，tmux 前缀 `mv2-`，完整命令与 Monitor 过滤行见第二部分 B 节；跑完在 `docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/` 留档。

新表布局名 **`motion-768-grid16-demopad17-v1`**，`store_meta.json` 与 `motion_index.json` 新增 `demo_min_real_frames: 17` / `exec_min_real_frames: 33` / `demo_tail_pad: repeat_last` 三键；`motion_store.py` 改成规格表（旧布局 `motion-768-grid16-v1` 对应 `(33, "none")`），40ep / 400ep 旧表照旧可读、可跑旧测试。

### 三、训练链路改什么：dataloader 与模型

**改动前后链路图（`AGENTS.md` 第 18 条）。** 帧路（SigLIP 8×8 packed 库 → 512 位记忆）一跳不动，下面只画运动路：

```
改动前（context 32 帧 4×4 口径，新库上根本跑不起来：无表 + budget 96 溢出）
  h5 帧 ─→ 33 帧窗(仅整窗) ─→ Wan VAE (9,16,32,32) f32 ─→ encoder 768 f32 ─→ motion 表 (69,716 行 × 3,072 B)
  __getitem__: visible_motion_rows(entry, t)  demo: s+32 ≤ es−1 │ exec: u+32 ≤ t−es
     → rows (k,) → 查表 motion_emb (k,768) f32 / motion_pos = pos 表[f,0,:256] (k,256) f32
     → _pad_motion 右填充到 budget=96 → motion_mask (96,) bool → mem_order = memory_order(帧时刻, 16, 窗时刻) (512+96,)
  RepackTransform → embed_memory: motion_pos_proj(256→768)+silu ‖ motion_emb → motion_encoder_static(1536→2048)
     → 与帧路 512 位按 mem_order 交错成 608 位 → context：拼进主序列

改动后（modulation 8×8 + demo 补帧；★ 为改数/改形的跳）
  h5 帧 ─→ 33 帧窗(★ demo 尾窗真 17–32 帧 + 第 es−1 帧重复) ─→ Wan VAE 同上 ─→ encoder 同上 ─→ motion 表 (★ 71,316 行)
  __getitem__: visible_motion_rows(entry, t, spec)  demo: ★ s+16 ≤ es−1 │ exec: 不变
     → rows (k,) ★ 每集 demo 多 1 行 → motion_emb (k,768) / motion_pos (k,256) 取法不变
     → _pad_motion 右填充到 ★ budget=160 → motion_mask (160,) → mem_order (512+160=672,)
  RepackTransform → embed_memory 同上，但 motion_encoder_static 出口 ★ 1536→1024（= memory_token_dim）
     → 交错成 ★ 672 位 mem_seq (b,672,1024) bf16 + mem_mask (b,672)
     → modulation：`compute_loss` / `sample_actions` 的 modulation 分支把 mem_seq / mem_mask 交给 `MemoryAttention` 做 cross-attention，不进主序列
```

每样本运动路字节量（budget 160）：`motion_emb` 160×768×4 = 491,520 B、`motion_pos` 160×256×4 = 163,840 B、`motion_mask` 160 B、`mem_order` 672×8 B；关闭态四键恒 `None`，基线 batch 内容不变。

**dataloader（`src/mme_vla_suite/training/framesamp_dataset.py`）：`__getitem__` 的 motion 分支一行不动，改的是契约与公式。**
- `motion_store.py`：`LAYOUT_SPECS` 规格表 + `MotionMeta.spec`；`seg_num_grid(L, min_real)`；`visible_motion_rows(entry, t, spec)` demo 条件改 `s + (spec.demo_min_real − 1) ≤ es − 1`；`parse_index` / `MotionMeta.load` 三方核对三新键。
- `FrameSampDataset.__init__`：`_req(budget == 96)` → `_req(budget % 16 == 0 and budget ≥ 16)`；新增 `_req(mcfg.demo_min_real_frames == spec.demo_min_real)`、`_req(mcfg.demo_tail_pad == spec.demo_tail_pad)`——YAML 与表不一致直接 raise。零截断预检（每集 `max_visible_count ≤ budget` 否则 raise）不动，这就是 budget 96 在新库必炸的那道闸（BinFill 最大 141 > 96）。
- `dataloader.py::_motion_gates` 不变：motion 表根取 `MMEVLA_MOTION_STORE` 或 YAML `motion.store_path`，`require_no_pack_lock` → `MotionMeta.load` → `require_verified` → `check_same_source`（motion 表记的清单 sha 必须 == framesamp 表的）。

**模型（`src/mme_vla_suite/models/integration/history_pi0.py`）：只放开一道闸。** `HistoryPi0.__init__` 现在显式 `raise` `motion_enabled and integration_type != "context"`，改成 `not in ("context", "modulation")`。其余零改动，理由是 modulation 的消费路径本来就通：`compute_loss` / `sample_actions` 的 modulation 分支已经写着 `mem_seq, mem_mask, _, _ = self.embed_memory(observation)`；`percep_mem.py` 的 `motion_encoder_static` 出口维取 `memory_token_dim`（modulation 下 1024，正好满足 `MemoryAttention` 的 `assert mem_width == x_width == 1024`）；`inputs_spec` 已从 `motion.budget` 推导 `mem_order` 长度。新增可训练参数两层：`motion_pos_proj` 256×768+768 = 197,376，`motion_encoder_static` 1536×1024+1024 = 1,573,888，合计 **1,771,264 ≈ 1.77 M**，参数树叶子 61 → **65**（`PARAM_TREE_EXACT … n_model=65`）。一条要写进正本的语义差异：modulation 的 `MemoryAttention` 用 `arange(mem_len)` 做 RoPE（padding 位占号、全在尾部），context 用 `cumsum(input_mask) − 1`；真 token 之间相对位置两者一致，动作到记忆的距离不同——这是 modulation 既有口径，不是本轮引入。

**配置两处。**
- 新 YAML `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml` = 基线 `perceptual-framesamp-modul-8frame-8x8.yaml`（sha256 `5b5ac2f8…`，一字不动）+ `motion` 节：`enabled: true`、`budget: <拍板值>`、`demo_min_real_frames: 17`、`demo_tail_pad: repeat_last`、`store_path: v1-store/datasets/4task-v2-1600ep-604f16da/motion`，其余（`dim 768` / `stride 16` / `window_frames 33` / `window_direction forward` / `grid_origin segment_start` / `pos_dim 256` / `frame_size 256` / `source_run` / `online_gpu`）照抄 `perceptual-framesamp-context-8frame-8x8-motion.yaml`。
- 新具名配置 `mme_vla_suite_b128_80k`（`src/mme_vla_suite/training/config.py::_CONFIGS`）= `mme_vla_suite_b128_60k` 逐字复制，只改 `num_train_steps 60_000 → 80_000`、`decay_steps 60_000 → 80_000`（peak == decay == 5e-5，余弦段是常数，改它只为自洽）；`save_interval` / `keep_period` 5_000 不动 → 16 个 checkpoint。**与基线的可比性**：warmup 5k 后 lr 恒 5e-5，80k run 的前 60k 步与基线的 lr 曲线逐步相同，因此 `60000` checkpoint 与基线终点 `59999` 是同步数、同 lr 历史的成对点；80k 终点另外多 20k 步。

### 四、推理怎么动

在线侧只有 `FrameSampMemory` 这一处知道 demo 段在哪里，补帧就在它里面做，**协议、sidecar、client 三件不改**：
- `src/mme_vla_suite/policies/framesamp_memory.py`：三处 demo 判据 `+ (W − 1) ≤ es − 1`（`_encode_ready_windows` 的 demo `while`、其后的原始帧清理条件、`visible_motion_frames` 的 demo `while`）统一改为 `+ (demo_min_real − 1) ≤ es − 1`；`_encode_window(f)` 对 demo 起点（`f < es`）取 `_raw_frames[f … min(f+32, es−1)]`，不足 33 帧用第 `es − 1` 帧 `np.repeat` 补齐后再 `np.stack`——与离线 `extract_wan.py` 的切法逐字节同源，所以 sidecar 收到的仍是 33 帧 uint8、6,488,064 B 的 payload（`motion_protocol.PAYLOAD_BYTES`），start_frame 仍是 `f`。首批 `add_buffer` 整段 demo 到货，新增的那个尾窗在首批就编完，exec 段增量逻辑不变。
- `src/mme_vla_suite/policies/policy.py::__init__` 的 `_motion_cfg` 键列表加 `demo_min_real_frames` / `demo_tail_pad` 两键，透传给 `FrameSampMemory`。
- `motion_protocol.py` / `motion_sidecar.py` / `motion_client.py` 不改；sidecar 权重仍是 `checkpoint_epoch_72.pt`（`bae96037…c15a`）+ VAE `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，握手 provenance 与新表 `store_meta.provenance` 逐键相等（客户端构造时已 raise 兜底）。
- 在线 vs 表的逐位对拍用既有 `scripts/training/g0/compare_online_motion.py --lib <新库> --yaml perceptual-framesamp-modul-8frame-8x8-motion.yaml --store-subdir framesamp-8x8`（按 eval 节奏驱动 `FrameSampMemory` + 真 sidecar），判定行 `ONLINE_ENC_BITEXACT=PASS` / `ONLINE_START_SET=PASS` / `ONLINE_POS=PASS` / `ONLINE_ORDER=PASS`——1,600 个补帧窗在线现算必须与离线表逐位相等，这是「训练看到的 = 推理看到的」的直接证据。
- 新任务的策略评估口径（仿真 `max_steps`、episode 集合、评估驱动）**本轮不定**，另起计划；本文只保证在线代码路径与训练同式。budget 表最后一列按旧 1300 步口径估算，见下。

### 五、budget：待拍板

按 ≥ 17 规则逐**样本**实算合法窗数（605,611 个，2026-09-17 复核）：均值 44.0，中位 39，P90 80，P95 93，P99 115，最大 **141**（BinFill ep367，nt 2304 / es 1152）。零窗样本 0%。

| budget | 截断样本 | 占比 | 截断 episode | 平均填充率 | 1300 步评估口径下末段 raise 的 episode |
|---:|---:|---:|---:|---:|---:|
| 96 | 24,366 | 4.02% | 112 | 45.3% | 951 |
| 112 | 7,817 | 1.29% | 43 | 39.2% | 271 |
| 128 | 488 | 0.08% | 6 | 34.4% | 112 |
| **144** | 0 | 0 | 0 | 30.5% | 6 |
| **160** | 0 | 0 | 0 | 27.5% | **0** |

最后一列：评估 `max_steps = 1300` 时 exec 网格恒 80 窗，`k_eval = demo_num_grid(es) + 80`，最大 151（es = 1152）；超 budget 时 `_prepare_motion` raise、评估驱动记 error。

- **144**：训练零截断、契约不改；评估侧 6 个最长 BinFill 集在 1300 步末段会 raise，要评估口径配合。
- **160**：训练与评估全零截断，代价是 cross-attention 多 16 个被 mask 的 K/V 位（modulation 下不进主 attention，只影响 `MemoryAttention` 的 K/V 长度 672 vs 656）。**推荐。**
- **保持 96 + 截断最早窗**：改零截断契约、动三处代码，4% 样本被截且丢的是 demo 起手段。不推荐。

拍板后改三处：新 YAML 的 `motion.budget`、`motion_gates_online.py` 的 `MOTION_CFG["budget"]`、本表上方「已拍板」行。**拍板必须早于第 2 步的 `commitV10.0`**（YAML 与测试在那个 commit 里）；未拍板前代码按 160 写，改一个数即可。

### 六、怎么证明没改坏

- **关闭态逐位不变**（基线不受影响）：dataset 交付对拍 `DS_EQUIV=PASS`；定点 batch `v1-store/fixtures/8x8` 喂改前 / 改后两版模型 `GRAD_EQ=PASS mismatches=0`；100 步守卫 `GUARD_GRAD_100=PASS`。任一不过不得宣称基线等价。
- **新表正确**：第二节第 5 步（D3 全量、D2 抽样、a7 / a9set / a10 / a11）。
- **开启态交付与消费**：500 样本 `visible_motion_rows` == 独立实现、交付行 == 表行、`mem_order` 合法置换；modulation 下 padding 位塞垃圾 loss 与梯度逐位不变（`motion_gates_model.py --gate m4`）；在线 vs 表逐位（第四节）。
- **训练收尾**：开启态 A/A 100 步逐位可复现；4 卡 b128 20 步 smoke `SMOKE20=PASS` + `PARAM_TREE_EXACT … n_model=65`；起跑 `PREFLIGHT=PASS n=25`。
- 全部判定行与工具见第二部分 F 节。

### 七、commit 与执行顺序：从现在到 80k 起跑

主副本上顺序执行，每个 commit 后立即 `git push`（`AGENTS.md` 第 11 条）；GPU 分配：建库 8 卡全开，对拍 / smoke 用 GPU 4–7，正式 run 用 GPU 4–7（与基线同卡数、同 `fsdp_devices=4`、同 per-device 32，保证成对可比）。

| 步 | 做什么 | GPU | commit / push | 判定 |
|---|---|---|---|---|
| 0 | 用户拍板 `motion.budget`（推荐 160）与 `run_name`（建议 `v2-1600ep-m8x8-modul-motion-b128-80k`） | — | — | 写回本文「已拍板」表 |
| 1 | 本文修订入库 | — | `docs: 重写 motion modulation 8×8 计划第一部分` → push | `git diff --check` 空 |
| 2 | 第二部分 A–E 全部代码改动（四处公式 / 抽取补帧 / oracle 抽样 / a11 / pack 三新键 / dataloader 契约 / 模型闸 / 在线侧 / 新 YAML / `mme_vla_suite_b128_80k` / 测试更新）+ CPU 单测（`test_pack_guards.py`、`motion_gates_online.py`）+ 关闭态对拍 V1 / V2 / V6 | 4–7 | `commitV10.0: demo 段补帧网格、modulation 接入 motion 与 80k 配置` → push；**之后 `git status --porcelain` 必须为空** | `DS_EQUIV=PASS` / `LEGACY_LAYOUT=PASS` / `GRAD_EQ=PASS` |
| 3 | 建库：`mv2-wan` → `mv2-encode`（**两者之间零 commit**）→ `mv2-pack` → `mv2-oracle` → 账目 a7 / a9set / a10 / a11 | 0–7 | 无 commit | 第二节第 2–5 步判定行全 PASS |
| 4 | 建库留档 `docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/{launch.md,result.md,records/}` + `docs/dataset-build-doc/README.md` 加行 | — | `docs: 1600 集新库 motion 表（demo 补帧 17）建库留档` → push | — |
| 5 | 开启态验证 V4 / V5 / V7 / V8 + 在线对拍（第四节）+ 4 卡 20 步 smoke（临时 run `smoke-m8x8-motion-<UTC>`，tmux `mv2-smoke`，跑完删） | 4–7（sidecar 用 GPU 3） | `docs: modulation motion 开启态守卫留档`（`docs/training-doc/mv2-guard-*/`）→ push | `A8_ROWS` / `A9_SET` / `MPOS` / `MEM_ORDER` / `M4_MASK` / `GUARD_GRAD_100` / `AA_100` / `ONLINE_*` / `SMOKE20` / `PARAM_TREE_EXACT n_model=65` 全 PASS |
| 6 | 起跑留档：`docs/training-doc/<run_name>/launch.md`（版本、命令、数据三件套 sha、判定行摘录、tmux 清单 `mv2-prod` / `mv2-prod-dense`）+ `docs/training-doc/README.md` 加行 | — | `docs: <run_name> 起跑留档` → push → **此刻 `TRAIN_HEAD=$(git rev-parse HEAD)`，工作区必须干净** | — |
| 7 | 生成 runner（= 基线 `records/prod-runner.sh`，换 `RUN` / 配置名 `mme_vla_suite_b128_80k` / `HC=perceptual-framesamp-modul-8frame-8x8-motion.yaml` / `--run-root …/mme_vla_suite_b128_80k/<run_name>`；`MMEVLA_MOTION_STORE` 不设，走 YAML `store_path`）→ `tmux new-session -d -s mv2-prod "bash <runner> '$TRAIN_HEAD' '$HC_SHA'"` → Monitor 盯 `PREFLIGHT=PASS n=25` / `Integration Type: modulation` / `motion memory 开启：store=… rows=71316` / `Step` / `Traceback` / `EXIT_CODE=` | 4–7 | 无 commit | `PREFLIGHT=PASS n=25` 后训练进入 Step 行；稳定 300 步后按基线口径记 ETA（基线 2.34 s/步 → 80k ≈ 52 h，motion 开启后须重测） |
| 8 | 训练期间主副本按基线做法锁只读（`chmod -R a-w src scripts packages`，记三份源码 sha）；如需并行开发，按 `AGENTS.md` 第 14 条再建 `-temp` 开发副本，训练结束后删 | — | — | — |

红线：不在 Wan → encode 之间 commit；不改 exec 规则、不改 stride、不做消融；不改关闭态 YAML 与既有 context YAML；不改 `mme_vla_suite` / `mme_vla_suite_b128` / `mme_vla_suite_b128_60k` 条目；tmux 只用 `tmux kill-session -t <全名>` 清理，删前删后各 `tmux ls`；正式 run 起跑前 `run_name` 必须经用户确认、run 根必须不存在。

---

## 第二部分（技术细节，供 agent 追踪）

### A. 契约与公式（四处同式）

**A1 `src/mme_vla_suite/datastore/motion_store.py`**
- 常量：`LAYOUT = "motion-768-grid16-demopad17-v1"`；新增 `DEMO_MIN_REAL_FRAMES = 17`、`EXEC_MIN_REAL_FRAMES = 33`、`DEMO_TAIL_PAD = "repeat_last"`。规格表 `LAYOUT_SPECS = {"motion-768-grid16-v1": (demo_min_real 33, demo_tail_pad "none"), "motion-768-grid16-demopad17-v1": (17, "repeat_last")}`；`MotionMeta` 增 `spec` 字段，`load` 按 `layout` 查表并核 `store_meta` / `motion_index` 的三新键与规格表三方同值（旧布局两文件缺三键时按规格表默认，不 raise）。
- 公式：`seg_num_chunks(L, min_real=33) = max(0, L − (min_real − 1))`；`seg_num_grid(L, min_real=33)`；`segment_grid_starts` 同参；`build_index_entries(manifest, spec)` demo 传 `spec.demo_min_real`。
- `visible_motion_rows(entry, t, spec)`：demo 条件 `s + (spec.demo_min_real − 1) ≤ es − 1`；exec 不变；`max_visible_count` 同参。
- `index_payload` 写三新键；`parse_index` 校验。

**A2 `scripts/dataset/wan/wan_common.py`**（子 venv 独立副本）：同名常量与 `seg_num_grid(L, min_real)`；`list_segments` demo 项 `min_real = 17`、exec 项 `33`，每项增 `min_real`。

**A3 `scripts/dataset/wan/extract_wan.py::process_segment`**
```python
off = wc.GRID_STRIDE * m
real = min(wc.WINDOW_FRAMES, item["seg_len"] - off)
if real < item["min_real"]:            # exec 段 min_real=33 → 与现行「窗口越段」raise 等价
    raise RuntimeError(...)
window = frames[off:off + real]
if real < wc.WINDOW_FRAMES:            # 只有 demo 段能进这里
    last = frames[item["seg_len"] - 1:item["seg_len"]]
    window = np.concatenate([window, np.repeat(last, wc.WINDOW_FRAMES - real, axis=0)])
window = np.ascontiguousarray(window)  # (33,256,256,3) uint8；后续 sha / encode_chunk 不变
```
`metadata.json` 每行增 `real_frames` / `pad_frames` / `pad_source_frame`（全域帧号 `es − 1`，exec 段为 `null`）；`input_frames_sha256` 仍对最终 33 帧算；`METADATA_SCHEMA` 升 3。`num_grid` 与公式核对改为 `wc.seg_num_grid(item["seg_len"], item["min_real"])`。

**A4 `scripts/dataset/wan/oracle_driver.py`**：`vae` 子命令独立重算起点 `range(0, max(0, L − (min_real − 1)), 16)` 并**独立实现**补帧（不 import `extract_wan` / `wan_common` 的切窗函数）；新增 `--sample-spec "padded:all,rest:0.10,seed:0"`，抽样集合（段 key + m）写入 oracle 输出 `sampled_windows.json` 供 `compare_wan.py latents` 对齐；`aggregate --kind vae` 合并各片抽样集合。`encoder` 子命令不抽样、全量。

**A5 `scripts/dataset/motion_checks.py`**：`a10` / `_independent_visible` 换公式（demo `min_real` 从 `store_meta.demo_min_real_frames` 读）；新增 `a11`：遍历 `wan-latents/*.metadata.json`，核 demo 段 `real_frames == min(33, L − 16m)`、`pad_frames + real_frames == 33`、`pad_source_frame == es − 1`、exec 段 `pad_frames` 全 0、补帧窗总数 == 集数 1,600；判定行 `A11_PAD=PASS padded=1600`。`a5` / `a6` 不适用新库，不跑。

**A6 `scripts/dataset/pack_motion_store.py`**：`cmd_pack` 写三新键与新 `layout`；`gather_provenance` 不动。`scripts/training/tests/test_pack_guards.py` 增规格表用例（旧布局可读、新布局三键缺一即 raise）。

### B. 建库命令序列（主副本，clean HEAD）

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA; source scripts/dataset/paths.sh; v1_prepare_dirs
LIB=$V1_STORE/datasets/4task-v2-1600ep-604f16da; RAW=$V1_STORE/raw-h5/4task-20260912-v2; MJ=/scratch/hongze/MotionJEPA
ls -ld $LIB $LIB/meta $RAW; BUILD_HEAD=$(git rev-parse HEAD); git status --porcelain   # 实体目录；porcelain 必须为空
# 第 2 步 Wan（tmux mv2-wan；日志 $LIB/logs/mv2-wan.log；Monitor 盯 STAGE_DONE stage=wan / STAGE_FAIL / Traceback / out of memory / EXIT_CODE=）
uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib $LIB --gpus 0,1,2,3,4,5,6,7 --raw-dir $RAW
# 第 3 步 encode（tmux mv2-encode）——与 wan 之间零 commit；--expected-ckpt-sha256 缺省取 ASSETS_LOCK.json 的 motionjepa_ckpt
uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib $LIB --gpus 0,1,2,3,4,5,6,7
# 第 4 步 pack / verify（tmux mv2-pack）
uv run --no-sync python scripts/dataset/pack_motion_store.py pack   --manifest $LIB/meta/episode_manifest.json --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion
uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store $LIB/motion --resume
# 第 5 步 oracle（MotionJEPA venv，--mj-repo $MJ；tmux mv2-oracle）：D3 全量、D2 抽样
CUDA_VISIBLE_DEVICES=7 … oracle_driver.py --mj-repo $MJ encoder --manifest $LIB/meta/episode_manifest.json --latents $LIB/wan-latents --out $LIB/oracle/wan-mj --expected-ckpt-sha256 bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a
for i in 0..7: CUDA_VISIBLE_DEVICES=$i … oracle_driver.py --mj-repo $MJ vae --manifest … --raw-dir $RAW --latents $LIB/wan-latents --out $LIB/oracle/wan-mj --shard-idx $i --num-shards 8 --sample-spec "padded:all,rest:0.10,seed:0"
… oracle_driver.py aggregate --manifest … --out $LIB/oracle/wan-mj --num-shards 8 --kind vae
uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents $LIB/wan-latents --oracle $LIB/oracle/wan-mj
uv run --no-sync python scripts/dataset/wan/compare_wan.py tokens  --store $LIB/motion --oracle $LIB/oracle/wan-mj
# 账目 a7 / a9set / a10 / a11（a10 的 --expect-* 取 motion_index.json totals 现算值）
```

Monitor 过滤管道（每级行缓冲）：`tail -n +1 -F <日志> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E 'STAGE_DONE|STAGE_FAIL|PACK_MOTION_DONE|VERIFY_MOTION=|ORACLE_.*=DONE|BITEXACT=|A1[01]_|A[79]_|Traceback|out of memory|EXIT_CODE='`。

判定行：`STAGE_DONE stage=wan workers=8 items=3200`、`STAGE_DONE stage=encode workers=8 items=3200`、`PACK_MOTION_DONE=1`、`VERIFY_MOTION=PASS scanned=71316 mismatches=0`、`ORACLE_ENCODER=DONE rows=71316`、`ENCODER_BITEXACT=PASS compared=71316 mismatches=0`、`ORACLE_VAE=DONE windows=<抽样数> frame_mismatches=0`、`WAN_BITEXACT=PASS compared=<抽样数> frame_mismatches=0 latent_mismatches=0`、`A7_BYTES=PASS`、`A9_INDEXSET=PASS samples=500 mismatches=0`、`A10_ROWS=PASS rows=71316 exec=35403 demo=35913`、`A11_PAD=PASS padded=1600`。留档 `docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/{launch.md,result.md,records/}`，`records/` 只放清洗后日志与判定行，不放 `.sh` / `.yaml`。

### C. dataloader / 配置

- `src/mme_vla_suite/training/framesamp_dataset.py::__init__`：`_req(int(mcfg.budget) % 16 == 0 and int(mcfg.budget) >= 16, …)`；`_req(int(mcfg.demo_min_real_frames) == self._motion_meta.spec.demo_min_real, …)`；`_req(str(mcfg.demo_tail_pad) == self._motion_meta.spec.demo_tail_pad, …)`；`__getitem__` 调 `ms.visible_motion_rows(entry, step, self._motion_meta.spec)`。三处「32 帧 / 96 / 608」注释改按变量描述。
- `src/mme_vla_suite/training/dataloader.py::_motion_gates`：不变。
- `src/mme_vla_suite/training/config.py`：`RepackTransform` 注释改 `(b, budget, …)` / `(b, 512 + budget)`；`_CONFIGS` 新增 `mme_vla_suite_b128_80k`（复制 `mme_vla_suite_b128_60k`，`num_train_steps=80_000`、`decay_steps=80_000`，注释写明与 60k 的唯一差异及「前 60k 步 lr 曲线逐步相同」）。
- 新 YAML `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml`（内容见第一部分第三节「配置两处」）。
- `scripts/training/compute_norm_stats.py::_NONE_KEYS` 不变；norm_stats 沿用新库 `856c75ea…`。

### D. 模型侧

- `src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0.__init__`：`if self.mem_encoder.motion_enabled and self.integration_type not in ("context", "modulation"): raise`。
- `history_observation.py`、`percep_mem.py` 只改注释（96 / 608 / 2048 字面 → 按配置描述）。
- `scripts/training/train.py::init_history_config`：核一遍 modulation 开启态也写全 `motion_provenance.json` 字段（预期无需改）。
- `scripts/training/legacy-eval/check_ckpt_param_tree.py`：预期 `n_model=65`，`MEM_PARAMS` 六条 modulation 路径 + 两条 motion 路径都在。

### E. 在线侧

- `src/mme_vla_suite/policies/framesamp_memory.py`：`motion_cfg` 增 `demo_min_real_frames` / `demo_tail_pad`；`_encode_ready_windows` demo `while` 与其后原始帧清理条件、`visible_motion_frames` demo `while` 三处同改 `+ (demo_min_real − 1) ≤ es − 1`；`_encode_window(f)` 对 `f < es` 取 `[f, min(f+32, es−1)]` 并 `np.repeat` 第 `es−1` 帧补齐；`_raw_frames` 清理语义不变（demo 编完后其帧可删）。
- `src/mme_vla_suite/policies/policy.py::__init__` 的 `_motion_cfg` 键列表增两键。
- `scripts/training/tests/motion_gates_online.py` 的 `MOTION_CFG` 增两键、`budget` 随拍板值；P2 / P4 用例加「es=114 首批编 7 窗、第 7 窗补 15 帧」断言。
- `scripts/training/g0/compare_online_motion.py`：读 YAML `motion` 节透传两键（脚本本身逻辑不变）。
- `motion_protocol.py`、`motion_sidecar.py`、`motion_client.py` **不改**。

### F. 验证与判定行

| 编号 | 内容 | 工具 | 判定行 | 顺序步 |
|---|---|---|---|---|
| V1 | 关闭态 dataset 交付逐位（modul-8x8，改前 vs 改后） | `dump_fixture_samples.py` + `compare_fixture_dumps.py` | `DS_EQUIV=PASS samples=≥3000 mismatches=0` | 2 |
| V2 | 旧表可读（规格表回归） | `MotionMeta.load(40ep/400ep motion)` + `motion_gates_online.py` 旧口径用例 | `LEGACY_LAYOUT=PASS` | 2 |
| V6 | 关闭态定点梯度（改前 vs 改后） | `single_step_grad_fixed.py` × 2 + `compare_fixed_grad.py`，fixture `v1-store/fixtures/8x8` | `GRAD_EQ=PASS mismatches=0` | 2 |
| V3 | 新表 D3 全量 / D2 抽样 / a7 a9set a10 a11 | B 节 | 见 B 节 | 3 |
| V4 | 开启态 dataloader 交付 | `motion_checks.py a9set` + 新增 `scripts/training/tests/dataloader_motion_check.py`（500 样本） | `A8_ROWS=PASS A9_SET=PASS MPOS=PASS MEM_ORDER=PASS` | 5 |
| V5 | 模型侧 mask 正确性（modulation） | `motion_gates_model.py --gate m4 --integration modulation`（新增 `--integration` 开关） | `M4_MASK=PASS` | 5 |
| V7 | 关闭态 100 步守卫 | `bench_train_steps.py` 2 卡 b8 确定性 flag | `GUARD_GRAD_100=PASS` | 5 |
| V8 | 开启态 A/A 100 步可复现 | 同上 × 2 | `AA_100=PASS hex_mismatch_steps=0` | 5 |
| V-online | 在线 vs 表逐位（含 1,600 补帧窗） | `compare_online_motion.py --lib $LIB --yaml <新 YAML> --store-subdir framesamp-8x8 --gpu 3` | `ONLINE_ENC_BITEXACT=PASS` / `ONLINE_START_SET=PASS` / `ONLINE_POS=PASS` / `ONLINE_ORDER=PASS` / `PROVENANCE=PASS` | 5 |
| V9 | 4 卡 b128 20 步 smoke（开启态，临时 run 跑完删） | 基线 `records/smoke-runner.sh` 同法，换 YAML / 配置名 | `SMOKE20=PASS`、`PARAM_TREE_EXACT … n_model=65 n_ckpt=65 missing=0 extra=0`、`motion memory 开启：… rows=71316` | 5 |
| V10 | 起跑 preflight | `preflight_train_launch.py`（runner 内，与 train 共用同一 `TRAIN_ARGS`） | `PREFLIGHT=PASS n=25` | 7 |

### G. commit 约束

- 第一部分第七节的表是唯一执行顺序；每个 commit 后立即 `git push`；只 `git add` 本轮明确文件。
- 步骤 3 的 Wan → encode 之间零 commit；步骤 2 结束后到步骤 3 起跑前 `git status --porcelain` 必须为空；步骤 6 的 `TRAIN_HEAD` 记在 docs commit 之后、runner 生成之前。
- 正式 run 的 runner 与 preflight 参数照抄 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/records/prod-runner.sh`，差异只有：`RUN`、配置名 `mme_vla_suite_b128_80k`、`HC=perceptual-framesamp-modul-8frame-8x8-motion.yaml` 及其 sha、`--run-root` 下的配置名子目录、tmux 名 `mv2-prod` / `mv2-prod-dense`；`MMEVLA_MOTION_STORE` 保持 unset。
