# 新库 motion 表构建 + modulation 8×8 接入 motion 计划（v2-motion）

> **状态：方案已固化，budget 已拍板 160，待 run_name 确认；未实施。** 2026-09-16 起草，2026-09-17 按用户要求重写第一部分。环境判定：**环境 B（AWS 单机 8×A100-SXM4-80GB）**，主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`。基线 `v2-1600ep-m8x8-modul-b128-60k` 已于 `2026-09-16T20:53:31Z` 以 `EXIT_CODE=0` 跑完，主副本已解锁、开发副本 `-temp` 已删除（`5432ba2`），**8 卡全空，一切工作回主副本进行**。无 turbo、无 GreatLakes。
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
| `motion.budget` | **160**（用户原话「budget按照训练和推理1300步定为160」：训练零截断，且按 1300 步评估口径 `k_eval = demo_num_grid(es) + 80` 最大 151 ≤ 160，评估亦零截断；见「budget」节） | 09-17 |
| `run_name` | **待确认**（本文建议 `v2-1600ep-m8x8-modul-motion-b128-80k`，起跑前按 `AGENTS.md` 第 6 条确认） | — |

---

## 第一部分（给人看）

### 这轮做什么

给 1600 集新库（BinFill / RouteStick / VideoRepick / VideoUnmaskSwap，605,611 个执行样本）建一张 motion token 表，然后用已跑完的 modulation 8×8 基线配置**只加一个 `motion` 节**，训练一条 80k 步的「+motion」模型，与基线成对可比。全文顺序：① demo 段补帧规则 → ② 建表五步 → ③ 训练链路哪些不动、哪些改 → ④ 推理侧哪些不动、哪些改 → ⑤ budget → ⑥ 验证 → ⑦ commit 与执行顺序。代码级细节全部在第二部分。

### 一、demo 段怎么补：数轴

一个 motion 窗口 = 连续 33 帧，喂给冻结的 MotionJEPA encoder 得一个 token。起点钉在段内网格 0, 16, 32, … 上，demo 段和 exec 段各自算网格、不跨段。

**现行规则的问题只在 demo 段尾部**：整窗放不下就不编，demo 最后 0–15 帧的运动没有任何窗覆盖。以 RouteStick 一集 `es = 100`（demo 帧 0–99）为例：

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

**新规则：起点只要还剩 ≥ 17 帧真实帧就编一窗，越过段尾的部分用 demo 最后一帧重复填满**：

```
demo 帧号   0        16       32       48       64       80       96  99 │100 exec →
            ├────────┼────────┼────────┼────────┼────────┼────────┼──┤
新规则      前 5 窗与现行完全相同（字节不变）
                                                         [80 ═══════ 99│99 99 … 99]   s=80  真 20 + 补 13（第 99 帧重复）
                                                                  ✗ s=96: 真实帧只剩 4 < 17，不编
```

补的是静止帧，encoder 看到的是「前 20 帧在动、后 13 帧定格」，语义就是「这段动作到此结束」。**每集恰好多 1 窗，且 demo 尾帧一定被盖住**：旧规则接受 `s ≤ es − 33`，新规则接受 `s ≤ es − 17`，多出的区间宽 16 = 一个网格步，里面有且只有一个网格点；这个新窗右端 ≥ es，必然覆盖到最后一帧。**exec 段一字不改**：窗口尾端不能超过当前帧，当前帧之后的帧根本不存在，没有段尾可补。

新库实算：demo 窗 34,313 → **35,913**，exec 35,403 不变，总行 **71,316**，补帧窗恰 **1,600**（= 集数）。同一个网格公式在四处代码里各写了一遍（训练侧契约、Wan 抽取器、oracle、在线 FrameSampMemory），**四处同改**。

### 二、建表：五步，哪步没做

建表链路照抄 400ep 库（Wan 抽取 → encode → pack/verify → oracle 对拍 → 账目检查），**链路本身不改，只换口径与规模**。现状：新库只有帧特征库，motion 相关的五样（latent、token、整表、oracle、留档）一样都没有；原始 H5、wan 子 venv、encoder 权重都在位。

| 步 | 模块 | 改不改 | 状态 | 预计耗时（8 卡） |
|---|---|---|---|---|
| 1 | **改代码** | 四处网格公式参数化；Wan 抽取器切窗时对 demo 尾窗补帧、元数据多记「真帧数 / 补帧数」；oracle 独立实现补帧并支持抽样；账目检查加一项「补帧账」；pack 写新布局名和三个新键 | **未做** | CPU 单测 < 5 min |
| 2 | **Wan 抽取** | 逻辑不改，只是切出来的 demo 尾窗多了补帧 | **未跑** | ≈ 3.6 h（≈ 39 GiB latent） |
| 3 | **encode** | 完全不改：输入是 latent，不知道也不需要知道补帧 | **未跑** | ≈ 19 min |
| 4 | **pack / verify** | 拼表逻辑不改，只多写三个契约键 | **未跑** | 分钟级 |
| 5 | **oracle 与账目** | MotionJEPA 仓库自己的 VAE / encoder 独立重算逐位比：encoder 全量，VAE 抽样（1,600 个补帧窗全部 + 其余 10%）；账目四项，其中「补帧账」是新的 | **未跑** | ≈ 1.2 h |

第 2 → 3 步之间**不能 commit**（provenance 要求两阶段 commit 唯一，400ep 曾因此重抽）。新表布局名换成带 `demopad17` 的新名，训练侧按布局名查规格表，**40ep / 400ep 旧表照旧可读**。全部命令与判定行见第二部分 B 节。

### 三、训练链路：哪些不动，哪些改

**不动的（及为什么）**
- **帧路**（SigLIP 8×8 帧特征库 → 512 位记忆）：与 motion 无关，基线 batch 内容逐字节不变。
- **dataloader 的 motion 取样逻辑**（查哪些行、右填充、与帧路交错）：公式在契约层，取样代码只是调用它。
- **模型的两层 motion 投影与 `embed_memory`**：context 版已经写好，modulation 复用；参数只在 `motion.enabled` 时创建（1.77 M），关闭态模型与基线完全相同。
- **modulation 的消费路径**：`compute_loss` / `sample_actions` 的 modulation 分支早已写着「取 memory 序列交给 MemoryAttention」，出口维正好是 1024，不用动。
- **基线 YAML、60k 配置、官方配置、Wan / encoder 权重**：一字不动，基线可复现。

**要改的（及加什么）**
- **motion 表契约**（`motion_store.py`）：加规格表，让「demo 最少真帧数 / 尾部补帧方式」成为布局的属性；新旧表共存。
- **dataloader 守卫**（`FrameSampDataset`）：把写死的 `budget == 96` 放开为 16 的倍数；新增两条核对——YAML 里写的补帧口径必须与表一致，不一致直接报错。
- **模型闸**（`HistoryPi0.__init__`）：现在 motion 只放行 context，改成也放行 modulation。就这一处。
- **新 YAML**：基线 YAML + `motion` 节（`budget: 160`、补帧口径两键、新表路径）。
- **新具名配置 `mme_vla_suite_b128_80k`**：复制 60k 条目，只把步数改 80k。lr 在 warmup 后恒定，所以前 60k 步与基线 lr 曲线逐步相同，`60000` checkpoint 与基线终点成对可比。

需要写进正本的一条语义差异：modulation 的 MemoryAttention 位置编码把 padding 位也占号（全在尾部），context 不占；真 token 之间相对位置一致。这是 modulation 既有口径，不是本轮引入。

### 四、推理侧：哪些不动，哪些改

**不动的**：sidecar 进程、通信协议（仍是 33 帧一包）、客户端、encoder 权重、exec 段增量编码逻辑、policy 的记忆装配。理由：补帧发生在「凑齐 33 帧」这一步之前，sidecar 收到的永远是 33 帧。

**要改的**：只有 `FrameSampMemory` 这一处知道 demo 段在哪里。它的 demo 判据从「整窗放得下」改成「还剩 ≥ 17 帧」，并在凑窗时用 demo 最后一帧补齐——与离线抽取器同一种切法，保证训练看到的 = 推理看到的。policy 层只是把 YAML 里的两个补帧口径键透传给它。

**怎么证明同源**：既有的在线对拍工具按 eval 节奏驱动 FrameSampMemory + 真 sidecar，逐窗与离线表比逐位，1,600 个补帧窗都要相等。

新任务的策略评估口径（仿真步数、episode 集合）本轮不定，另起计划。

### 五、budget：已拍板 160

按 ≥ 17 规则逐样本实算合法窗数（605,611 个）：均值 44.0，中位 39，P90 80，P95 93，P99 115，最大 **141**（BinFill ep367，es 1152）。

| budget | 截断样本 | 占比 | 截断 episode | 平均填充率 | 1300 步评估口径下末段 raise 的 episode |
|---:|---:|---:|---:|---:|---:|
| 96 | 24,366 | 4.02% | 112 | 45.3% | 951 |
| 112 | 7,817 | 1.29% | 43 | 39.2% | 271 |
| 128 | 488 | 0.08% | 6 | 34.4% | 112 |
| 144 | 0 | 0 | 0 | 30.5% | 6 |
| **160** | 0 | 0 | 0 | 27.5% | **0** |

最后一列：评估最多 1300 环境步、每 16 帧决策一次，最晚决策时刻 exec 段恒 80 窗，`k_eval = demo 窗数 + 80`，最大 151；超 budget 时在线装配直接 raise、评估驱动记 error。

**用户已拍板（2026-09-17）：`motion.budget = 160`**——训练零截断，1300 步评估口径下也零截断（151 ≤ 160）；代价只是 MemoryAttention 多 16 个被 mask 的 K/V 位。新任务评估若不取 1300 步，须按第二部分 C 节的式子重核此值。

### 六、怎么证明没改坏

- **关闭态逐位不变**：dataset 交付、定点 batch 梯度、100 步守卫，改前 vs 改后三项全逐位；任一不过不得宣称基线等价。
- **新表正确**：第二节第 5 步的 oracle 与账目。
- **开启态**：交付行 == 表行、交错次序合法、modulation 下 padding 位塞垃圾 loss 与梯度不变、在线 vs 表逐位、A/A 100 步可复现、4 卡 20 步 smoke + 参数树 65 叶精确匹配。
- 判定行与工具全在第二部分 F 节。

### 七、commit 与执行顺序：从现在到 80k 起跑

主副本上顺序执行，每个 commit 后立即 push。建库 8 卡全开；对拍、smoke、正式 run 用 GPU 4–7（与基线同卡数、同 per-device batch，保证成对可比）。

| 步 | 做什么 | commit |
|---|---|---|
| 0 | budget 已拍板 160；用户确认 `run_name`（建议 `v2-1600ep-m8x8-modul-motion-b128-80k`） | — |
| 1 | 本文入库 | `docs:` |
| 2 | 第二部分 A–E 全部代码 + CPU 单测 + 关闭态三项对拍 | `commitV10.0: demo 段补帧网格、modulation 接入 motion 与 80k 配置`；之后工作区必须干净 |
| 3 | 建库五步（tmux `mv2-wan` → `mv2-encode` → `mv2-pack` → `mv2-oracle`；Wan 与 encode 之间零 commit） | — |
| 4 | 建库留档 | `docs:` |
| 5 | 开启态验证 + 在线对拍 + 4 卡 20 步 smoke（临时 run 跑完删） | `docs:` 守卫留档 |
| 6 | 起跑留档 `docs/training-doc/<run_name>/launch.md` | `docs:`；此刻记 `TRAIN_HEAD` |
| 7 | 生成 runner（照抄基线 runner，换 run 名 / 80k 配置 / 新 YAML）→ tmux `mv2-prod` → preflight 25 项 PASS → 训练；Monitor 盯日志 | — |
| 8 | 训练期间主副本锁只读；需并行开发再建 `-temp` 副本，结束后删 | — |

基线 4 卡 2.34 s/步，80k 约 52 h；motion 开启后步时须重测。红线：不在 Wan → encode 之间 commit；不改 exec 规则、stride，不做消融；不改关闭态 YAML、既有 context YAML 与三个既有配置条目；tmux 只按全名逐个清理；正式 run 起跑前 `run_name` 必须经用户确认、run 根必须不存在。

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

- `src/mme_vla_suite/training/framesamp_dataset.py::__init__`：`_req(int(mcfg.budget) % 16 == 0 and int(mcfg.budget) >= 16, …)`（160 = 10 × 16 通过）；`_req(int(mcfg.demo_min_real_frames) == self._motion_meta.spec.demo_min_real, …)`；`_req(str(mcfg.demo_tail_pad) == self._motion_meta.spec.demo_tail_pad, …)`；`__getitem__` 调 `ms.visible_motion_rows(entry, step, self._motion_meta.spec)`。三处「32 帧 / 96 / 608」注释改按变量描述。
- `src/mme_vla_suite/training/dataloader.py::_motion_gates`：不变。
- `src/mme_vla_suite/training/config.py`：`RepackTransform` 注释改 `(b, budget, …)` / `(b, 512 + budget)`；`_CONFIGS` 新增 `mme_vla_suite_b128_80k`（复制 `mme_vla_suite_b128_60k`，`num_train_steps=80_000`、`decay_steps=80_000`，注释写明与 60k 的唯一差异及「前 60k 步 lr 曲线逐步相同」）。
- 新 YAML `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml` = 基线 `perceptual-framesamp-modul-8frame-8x8.yaml`（sha256 `5b5ac2f8…`，一字不动）+ `motion` 节：`enabled: true`、`budget: 160`、`demo_min_real_frames: 17`、`demo_tail_pad: repeat_last`、`store_path: v1-store/datasets/4task-v2-1600ep-604f16da/motion`，其余（`dim 768` / `stride 16` / `window_frames 33` / `window_direction forward` / `grid_origin segment_start` / `pos_dim 256` / `frame_size 256` / `source_run` / `online_gpu`）照抄 `perceptual-framesamp-context-8frame-8x8-motion.yaml`。
- budget 复核式（评估口径变更时用）：`k_eval = demo_num_grid(es) + len(range(0, max(0, τ_max − es − 32) + 1, 16))`，其中 `τ_max = es + 16·⌊(max_steps − 4)/16⌋`；`max_steps = 1300` 时 exec 项恒 80，新库 `k_eval` 最大 151 ≤ 160。
- `scripts/training/compute_norm_stats.py::_NONE_KEYS` 不变；norm_stats 沿用新库 `856c75ea…`。

### D. 模型侧

- `src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0.__init__`：`if self.mem_encoder.motion_enabled and self.integration_type not in ("context", "modulation"): raise`。
- `history_observation.py`、`percep_mem.py` 只改注释（96 / 608 / 2048 字面 → 按配置描述）。
- `scripts/training/train.py::init_history_config`：核一遍 modulation 开启态也写全 `motion_provenance.json` 字段（预期无需改）。
- `scripts/training/legacy-eval/check_ckpt_param_tree.py`：预期 `n_model=65`，`MEM_PARAMS` 六条 modulation 路径 + 两条 motion 路径都在。

### E. 在线侧

- `src/mme_vla_suite/policies/framesamp_memory.py`：`motion_cfg` 增 `demo_min_real_frames` / `demo_tail_pad`；`_encode_ready_windows` demo `while` 与其后原始帧清理条件、`visible_motion_frames` demo `while` 三处同改 `+ (demo_min_real − 1) ≤ es − 1`；`_encode_window(f)` 对 `f < es` 取 `[f, min(f+32, es−1)]` 并 `np.repeat` 第 `es−1` 帧补齐；`_raw_frames` 清理语义不变（demo 编完后其帧可删）。
- `src/mme_vla_suite/policies/policy.py::__init__` 的 `_motion_cfg` 键列表增两键。
- `scripts/training/tests/motion_gates_online.py` 的 `MOTION_CFG` 增两键、`budget` 改 160；P2 / P4 用例加「es=114 首批编 7 窗、第 7 窗补 15 帧」断言。
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
