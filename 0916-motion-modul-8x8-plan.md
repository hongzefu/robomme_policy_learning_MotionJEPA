# 新库 motion 表构建 + modulation 8×8 接入 motion 计划（v2-motion）

> **状态：方案已固化，待 budget 拍板；未实施。** 2026-09-16 起草。环境判定：**环境 B（AWS 单机 8×A100-SXM4-80GB）**，主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`（训练期锁只读），开发副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`，`v1-store` 为指向主副本的可写 symlink。无 turbo、无 GreatLakes。
>
> 本文按 `AGENTS.md` 第 2 条分两部分。正本 `docs/motion-memory.md` 写的是 context + 32 帧 4×4 + 旧四任务口径，本文只写**与其不同**的部分；实施后正本另立 `docs:` 更新。

**用户已拍板（2026-09-16）**

| 项 | 决定 |
|---|---|
| 接入目标 | **modulation 8×8**，与在跑基线 `v2-1600ep-m8x8-modul-b128-60k` 同配置、只多 motion，成对可比 |
| demo 段窗口 | **全覆盖 + 真实帧 ≥ 17**：起点 `s ∈ range(0, es, 16)` 且 `es − s ≥ 17`；窗口越过 `es − 1` 的部分用第 `es − 1` 帧重复填充凑满 33 帧 |
| exec 段窗口 | **不变**：尾端 ≤ 当前帧的完整 33 帧窗，不补帧 |
| 新库 motion 表 | **单独构建**（1600 集新库当前没有 motion 表） |
| `motion.budget` | **待用户讨论后决定**（见「budget：待拍板」；表的构建不依赖 budget，可先建） |

---

## 第一部分（给人看）

### 这轮做什么

给 1600 集新库（`v1-store/datasets/4task-v2-1600ep-604f16da`，BinFill / RouteStick / VideoRepick / VideoUnmaskSwap，605,611 个执行样本）建一张 motion token 表，让正在训练的 **modulation 8×8** 基线配置能吃 motion，训练出第一条与基线成对的「+motion」模型。建表链路照抄 `docs/dataset-build-doc/4task-motion-400ep/launch.md`，唯一口径变化是 demo 段尾部补帧；模型侧只放开一道闸。

### 现状（2026-09-16 实测）

- **新库没有 motion 表**：库下只有 `framesamp` / `framesamp-8x8` / `source` / `meta`。原始 H5 在 `v1-store/raw-h5/4task-20260912-v2/`（四个合并文件 793.8 GB），wan 子 venv `v1-store/venvs/wan` 完好（torch 2.9.0+cu128 / diffusers 0.39.0）。
- **motion 只接 context**：`history_pi0.py::HistoryPi0.__init__` 显式 `raise` `motion_enabled and integration_type != "context"`。但 modulation 的消费路径已经通——`compute_loss` / `sample_actions` 的 modulation 分支 `mem_seq, mem_mask, _, _ = self.embed_memory(observation)`，`percep_mem.py` 的 `motion_encoder_static` 输出维取 `memory_token_dim`（modulation 下 1024，正好满足 `MemoryAttention` 的 `assert mem_width == x_width == 1024`）。
- **budget 96 必然溢出**：现行公式下 BinFill 最大合法窗数 140（demo 70 + exec 70），99 个 episode 超 96，`FrameSampDataset.__init__` 零截断预检会 raise。
- **demo「不够」只发生在段尾**：新库 demo 段全部 ≥ 100 帧（RouteStick `es ∈ {100,…,500}`；BinFill / RouteStick `es = nt/2`；VideoRepick `es ≈ nt/2`；VideoUnmaskSwap `es ∈ {114,168,216,264,318}`）。现行公式下 demo 尾部未被任何窗口覆盖的帧数在 0–15 均匀分布。
- **四处同式**：`datastore/motion_store.py::visible_motion_rows`、`wan/wan_common.py::seg_num_grid`、`wan/oracle_driver.py`（独立重算）、`policies/framesamp_memory.py::_encode_ready_windows` 各自实现同一网格公式，demo 规则一改四处都改。

### demo 段怎么补：数轴

以 RouteStick 一集 `es = 100`（demo 段帧号 0–99，exec 段从 100 起）为例。网格起点钉在 0, 16, 32, …，每窗 33 帧 `[s, s+32]`：

```
demo 帧号   0        16       32       48       64       80       96  99 │100 exec →
            ├────────┼────────┼────────┼────────┼────────┼────────┼──┤
现行规则    [0 ══════════════ 32]                                        s=0   真 33
                     [16 ═════════════ 48]                               s=16  真 33
                              [32 ═════════════ 64]                      s=32  真 33
                                       [48 ═════════════ 80]             s=48  真 33
                                                [64 ═════════════ 96]    s=64  真 33
                                                         ✗ s=80: 80+32=112 > 99，不编 → 帧 97–99 没有任何窗覆盖

新规则      前 5 窗同上，再加：
                                                         [80 ══════ 99│99 99 … 99]   s=80  真 20 + 补 13（第 99 帧重复）
                                                                  ✗ s=96: 真 4 帧 < 17，不编
```

规则一句话：**demo 起点 `s` 只要满足 `es − s ≥ 17` 就编一窗；窗口超出 demo 段的部分全部用 demo 最后一帧（第 `es − 1` 帧）填充。** 因为网格步长 16、窗长 33，满足 `es − 32 ≤ s < es − 16` 的网格点恰有一个，所以**每集恰多 1 窗**，且尾部 16 帧的运动一定被某窗覆盖。exec 段一律不补（`u + 32 ≤ t − es` 不变）。

公式：`demo_num_grid(L) = len(range(0, max(0, L − 16), 16))`，exec 仍 `len(range(0, max(0, L − 32), 16))`；统一写成 `seg_num_grid(L, min_real) = len(range(0, max(0, L − (min_real − 1)), 16))`，demo `min_real = 17`、exec `33`。

新库实算：demo 窗 34,313 → **35,913**，exec 35,403 不变，总行 **71,316**；补帧窗恰 1,600 个。

### 要改什么

**建表**（链路不改，只换口径与规模）：`run_local.py --stage wan` → `--stage encode` → `pack_motion_store.py pack | verify`。规模：71,316 窗，`wan-latents` ≈ 39.2 GiB（71,316 × 589,824 B），token 表 ≈ 209 MiB；按 400ep 库实测速率（Wan 0.69 窗/s/worker、encode 7.7 窗/s/worker）估 Wan 8 卡 ≈ 3.6 h、encode ≈ 20 min。布局名换 **`motion-768-grid16-demopad17-v1`**，meta 新增 `demo_min_real_frames: 17` / `exec_min_real_frames: 33` / `demo_tail_pad: repeat_last`；旧布局用规格表保留，40ep / 400ep 旧表照旧可读。

**dataloader**：`FrameSampDataset.__getitem__` motion 分支一行不动。改的是契约与公式：`motion_store.py`（规格表、`seg_num_grid(L, min_real)`、`visible_motion_rows` demo 条件 `s + 16 ≤ es − 1`、meta 三新键校验）、`framesamp_dataset.py::__init__`（`_req(budget == 96)` → `budget % 16 == 0`，并核 YAML 的 `demo_min_real_frames` 与表一致）。

**模型**：`HistoryPi0.__init__` 的闸改成 `integration_type not in ("context", "modulation")`。`embed_memory` / `compute_loss` / `sample_actions` / `percep_mem.py` / `history_gemma.py` 零改动；`inputs_spec` 已从 `motion.budget` 推导 `mem_order` 长度。新增可训练参数 `motion_pos_proj`（256→768）+ `motion_encoder_static`（1536→1024），合计 **1,771,264 ≈ 1.77 M**，参数树 61 → 65 叶。一条要写进正本的语义差异：modulation 的 `MemoryAttention` 用 `arange(mem_len)` 做 RoPE（padding 占号但全在尾部），context 用 `cumsum(input_mask) − 1`；真 token 间相对位置一致，动作到记忆的距离不同——这是 modulation 既有口径。

**在线侧**：`framesamp_memory.py` 的 demo `while` 条件 `+ 32 ≤ es − 1` → `+ 16 ≤ es − 1`，`_encode_window` 对 demo 起点做同样补帧后再发 sidecar；协议 payload 仍是 33 帧 6,488,064 B，**协议、sidecar、client 不改**。

**配置**：新 YAML `perceptual-framesamp-modul-8frame-8x8-motion.yaml` = `perceptual-framesamp-modul-8frame-8x8.yaml` + `motion` 节（`enabled: true`、`budget: <待拍板>`、`demo_min_real_frames: 17`、`demo_tail_pad: repeat_last`、`store_path: v1-store/datasets/4task-v2-1600ep-604f16da/motion`，其余照抄 context-motion YAML）。关闭态 YAML 一字不动，基线不受影响。

### budget：待拍板

按 ≥17 规则逐**样本**实算合法窗数（605,611 个）：均值 43.9，中位 39，P90 80，P95 93，P99 115，最大 **141**（BinFill ep367，nt 2304 / es 1152）。零窗样本 0%。

| budget | 截断样本 | 占比 | 截断 episode | 平均填充率 | 1300 步评估口径下末段 raise 的 episode |
|---:|---:|---:|---:|---:|---:|
| 96 | 24,366 | 4.02% | 112 | 45.2% | 951 |
| 112 | 7,817 | 1.29% | 43 | 39.1% | 271 |
| 128 | 488 | 0.08% | 6 | 34.3% | 112 |
| **144** | 0 | 0 | 0 | 30.5% | 6 |
| **160** | 0 | 0 | 0 | 27.4% | **0** |

最后一列：评估 `max_steps = 1300` 时 exec 网格恒 80 窗，`k_eval = demo_num_grid(es) + 80`，最大 151（es = 1152）；超 budget 时 `_prepare_motion` raise、评估驱动记 error。新任务评估口径尚未定，此列按旧口径算。

- **144**：训练零截断、契约不改；评估侧 6 个最长 BinFill 集在 1300 步末段会 raise，要评估口径配合。
- **160**：训练与评估全零截断，代价是 cross-attention 多 16 个被 mask 的 K/V 位（不进主 attention）。**推荐。**
- **保持 96 + 截断最早窗**：改零截断契约、动三处代码，4% 样本被截且丢的是 demo 起手段。不推荐。

拍板后改三处：新 YAML 的 `motion.budget`、`framesamp_dataset._req` 说明、本节补「已拍板：N」。

### 怎么证明没改坏（`AGENTS.md` 第 18 条）

- **关闭态逐位不变**（基线不受影响）：dataset 交付对拍 `DS_EQUIV`；定点 batch `v1-store/fixtures/8x8` 喂改前 / 改后两版模型 `GRAD_EQ`；100 步守卫 `GUARD_GRAD_100`。任一不过不得宣称基线等价。
- **新表正确**：D3 encoder oracle 全量逐位；D2 VAE oracle **抽样**（1,600 个补帧窗全部 + 随机 10% 其余，oracle 独立实现补帧）；`motion_checks.py a5–a10` + 新增 `a11` 补帧账。
- **开启态交付与消费**：500 样本 `visible_motion_rows` == 独立实现、交付行 == 表行、`mem_order` 合法置换；modulation 下 padding 位塞垃圾 loss 与梯度逐位不变。
- **训练收尾**：开启态 A/A 100 步逐位可复现；4 卡 b128 20 步 smoke `SMOKE20` + `PARAM_TREE_EXACT n_model=65`。
- 正式 run 另起，run_name 起跑前按第 6 条确认，配置条目复用 `mme_vla_suite_b128_60k`。

### 顺序与红线

1. 本文入库 → 开发副本改代码（契约 / 抽取补帧 / oracle / dataloader / 模型闸 / 在线 / YAML / 测试）→ commit、push；关闭态对拍在开发副本 GPU 0–3 做（只读 fixture 与旧表）。
2. 基线 `m8-prod` 跑完（本轮核对 53.0k/60.0k，剩约 4 h）→ 主副本解锁、`git pull` → **从主副本 clean HEAD 起建库**，8 卡全开；**Wan → encode 全程零 commit**（`gather_provenance` 要求两阶段 `git_commit` 唯一）。
3. 建库留档 → 开启态验证 → smoke → budget 拍板 → 正式 run 计划另起。

红线：不动在跑的 `m8-prod` / `m8-prod-dense`（GPU 4–7）；不在开发副本执行带输出根参数的建库命令（第 14 条）；不改 exec 规则、不改 stride、不做消融；不改关闭态 YAML 与既有 context YAML；tmux 前缀 `mv2-`，清理只用 `tmux kill-session -t <全名>`，删前删后各 `tmux ls`。

---

## 第二部分（技术细节，供 agent 追踪）

### A. 契约与公式（四处同式）

**A1 `src/mme_vla_suite/datastore/motion_store.py`**
- 常量：`LAYOUT = "motion-768-grid16-demopad17-v1"`；新增 `DEMO_MIN_REAL_FRAMES = 17`、`EXEC_MIN_REAL_FRAMES = 33`、`DEMO_TAIL_PAD = "repeat_last"`。规格表 `LAYOUT_SPECS = {"motion-768-grid16-v1": (demo_min_real 33, demo_tail_pad "none"), "motion-768-grid16-demopad17-v1": (17, "repeat_last")}`；`MotionMeta` 增 `spec` 字段，`load` 按 `layout` 查表并核 `store_meta` / `motion_index` 的三新键与规格表三方同值。
- 公式：`seg_num_chunks(L, min_real=33) = max(0, L − (min_real − 1))`；`seg_num_grid(L, min_real=33)`；`segment_grid_starts` 同参；`build_index_entries(manifest, spec)` demo 传 `spec.demo_min_real`。
- `visible_motion_rows(entry, t, spec)`：demo 条件 `s + (spec.demo_min_real − 1) ≤ es − 1`；exec 不变；`max_visible_count` 同参。
- `index_payload` 写三新键；`parse_index` 校验。

**A2 `scripts/dataset/wan/wan_common.py`**（子 venv 独立副本）：同名常量与 `seg_num_grid(L, min_real)`；`list_segments` demo 项 `min_real = 17`，每项增 `min_real`。

**A3 `scripts/dataset/wan/extract_wan.py::process_segment`**
```python
off = wc.GRID_STRIDE * m
real = min(wc.WINDOW_FRAMES, item["seg_len"] - off)
if real < item["min_real"]:            # exec 段 min_real=33 → 与现行 raise 等价
    raise RuntimeError(...)
window = frames[off:off + real]
if real < wc.WINDOW_FRAMES:            # 只有 demo 段能进这里
    last = frames[item["seg_len"] - 1:item["seg_len"]]
    window = np.concatenate([window, np.repeat(last, wc.WINDOW_FRAMES - real, axis=0)])
window = np.ascontiguousarray(window)  # (33,256,256,3) uint8；后续 sha / encode_chunk 不变
```
`metadata.json` 每行增 `real_frames` / `pad_frames` / `pad_source_frame`（全域帧号 `es − 1`）；`input_frames_sha256` 仍对最终 33 帧算；`METADATA_SCHEMA` 升 3。

**A4 `scripts/dataset/wan/oracle_driver.py`**：`vae` 子命令独立重算起点 `range(0, max(0, L − (min_real − 1)), 16)` 并**独立实现**补帧；新增 `--sample-spec "padded:all,rest:0.10,seed:0"`，抽样集合写入 oracle 输出供 `compare_wan.py` 对齐。

**A5 `scripts/dataset/motion_checks.py`**：`a10` / `_independent_visible` 换公式；新增 `a11`：遍历 `wan-latents/*.metadata.json`，核 demo 段 `real_frames == min(33, L − 16m)`、`pad_frames + real_frames == 33`、exec 段 `pad_frames` 全 0、补帧窗总数 == 1,600。

**A6 `scripts/dataset/pack_motion_store.py`**：`cmd_pack` 写三新键与新 `layout`；`gather_provenance` 不动。

### B. 建库命令序列（主副本，训练结束解锁后，clean HEAD）

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA; source scripts/dataset/paths.sh; v1_prepare_dirs
LIB=$V1_STORE/datasets/4task-v2-1600ep-604f16da; RAW=$V1_STORE/raw-h5/4task-20260912-v2; MJ=/scratch/hongze/MotionJEPA
ls -ld $LIB $LIB/meta; BUILD_HEAD=$(git rev-parse HEAD); git status --porcelain   # 实体目录；porcelain 必须为空
# Wan（tmux mv2-wan；Monitor 盯 STAGE_DONE stage=wan / Traceback / out of memory）
uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib $LIB --gpus 0,1,2,3,4,5,6,7 --raw-dir $RAW
# encode（tmux mv2-encode）——与 wan 之间零 commit
uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib $LIB --gpus 0,1,2,3,4,5,6,7
# pack / verify（tmux mv2-pack）
uv run --no-sync python scripts/dataset/pack_motion_store.py pack   --manifest $LIB/meta/episode_manifest.json --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion
uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store $LIB/motion --resume
# oracle（MotionJEPA venv，MJ_REPO=$MJ；tmux mv2-oracle）：D3 全量、D2 抽样
CUDA_VISIBLE_DEVICES=7 … oracle_driver.py --mj-repo $MJ encoder --manifest $LIB/meta/episode_manifest.json --latents $LIB/wan-latents --out $LIB/oracle/wan-mj --expected-ckpt-sha256 bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a
for i in 0..7: CUDA_VISIBLE_DEVICES=$i … oracle_driver.py --mj-repo $MJ vae --manifest … --raw-dir $RAW --latents $LIB/wan-latents --out $LIB/oracle/wan-mj --shard-idx $i --num-shards 8 --sample-spec "padded:all,rest:0.10,seed:0"
… oracle_driver.py aggregate … --kind vae
uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents $LIB/wan-latents --oracle $LIB/oracle/wan-mj
uv run --no-sync python scripts/dataset/wan/compare_wan.py tokens  --store $LIB/motion --oracle $LIB/oracle/wan-mj
# 附加检查 a5–a11（a10 的 --expect-* 取 motion_index.json totals 现算值）
```

判定行：`STAGE_DONE stage=wan workers=8 items=3200`（demo 1600 段 + exec 1600 段）、`STAGE_DONE stage=encode …`、`PACK_MOTION_DONE=1`、`VERIFY_MOTION=PASS scanned=71316 mismatches=0`、`ORACLE_ENCODER=DONE rows=71316`、`WAN_BITEXACT=PASS compared=<抽样数> … mismatches=0`、`TOKENS_BITEXACT=PASS`、`A10_ROWS=PASS rows=71316 exec=35403 demo=35913`、`A11_PAD=PASS padded=1600`。留档 `docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/{launch.md,result.md,records/}`。

### C. dataloader / 配置

- `src/mme_vla_suite/training/framesamp_dataset.py::__init__`：`_req(int(mcfg.budget) % 16 == 0 and int(mcfg.budget) >= 16, …)`；`_req(int(mcfg.demo_min_real_frames) == self._motion_meta.spec.demo_min_real, …)`；`_req(str(mcfg.demo_tail_pad) == self._motion_meta.spec.demo_tail_pad, …)`；`__getitem__` 调 `ms.visible_motion_rows(entry, step, self._motion_meta.spec)`。三处「32 帧 / 96 / 608」注释改按变量描述。
- `src/mme_vla_suite/training/dataloader.py::_motion_gates`：不变。
- `src/mme_vla_suite/training/config.py`：`RepackTransform` 注释改 `(b, budget, …)` / `(b, 512 + budget)`。
- 新 YAML `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml`（内容见第一部分「配置」）。
- `scripts/training/compute_norm_stats.py::_NONE_KEYS` 不变。

### D. 模型侧

- `src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0.__init__`：`if self.mem_encoder.motion_enabled and self.integration_type not in ("context", "modulation"): raise`。
- `history_observation.py`、`percep_mem.py` 只改注释（96 / 608 / 2048 字面 → 按配置描述）。
- `scripts/training/train.py::init_history_config`：核一遍 modulation 开启态也写全 `motion_provenance.json` 字段（预期无需改）。

### E. 在线侧

- `src/mme_vla_suite/policies/framesamp_memory.py`：`motion_cfg` 增 `demo_min_real_frames` / `demo_tail_pad`；`_encode_ready_windows` demo 条件 `+ (demo_min_real − 1) ≤ es − 1`；`_encode_window(f)` 对 demo 起点（`f < es`）取 `[f, min(f+32, es−1)]` 并 `np.repeat` 第 `es−1` 帧补齐；`visible_motion_frames` demo 条件同改；`_raw_frames` 清理逻辑不变。
- `src/mme_vla_suite/policies/policy.py::__init__` 的 `_motion_cfg` 键列表增两键。
- `scripts/training/tests/motion_gates_online.py` 的 `MOTION_CFG` 与断言随 budget / 规则更新。
- `motion_protocol.py`、`motion_sidecar.py`、`motion_client.py` **不改**。

### F. 验证与判定行

| 编号 | 内容 | 工具 | 判定行 |
|---|---|---|---|
| V1 | 关闭态 dataset 交付逐位（modul-8x8，改前 vs 改后） | `dump_fixture_samples.py` + `compare_fixture_dumps.py` | `DS_EQUIV=PASS samples=≥3000 mismatches=0` |
| V2 | 旧表可读（规格表回归） | `MotionMeta.load(40ep/400ep motion)` + `motion_gates_online.py` 旧口径用例 | `LEGACY_LAYOUT=PASS` |
| V3 | 新表 D3 全量 / D2 抽样 / a5–a11 | B 节 | 见 B 节 |
| V4 | 开启态 dataloader 交付 | `motion_checks.py` + 新增 `dataloader_motion_check.py`（500 样本） | `A8_ROWS=PASS A9_SET=PASS MPOS=PASS MEM_ORDER=PASS` |
| V5 | 模型侧 mask 正确性（modulation） | `motion_gates_model.py --gate m4 --integration modulation` | `M4_MASK=PASS` |
| V6 | 关闭态定点梯度（改前 vs 改后） | `single_step_grad_fixed.py` × 2 + `compare_fixed_grad.py`，fixture `v1-store/fixtures/8x8` | `GRAD_EQ=PASS mismatches=0` |
| V7 | 关闭态 100 步守卫 | `bench_train_steps.py` 2 卡 b8 确定性 flag | `GUARD_GRAD_100=PASS` |
| V8 | 开启态 A/A 100 步可复现 | 同上 × 2 | `AA_100=PASS hex_mismatch_steps=0` |
| V9 | 4 卡 b128 20 步 smoke（开启态，临时 run 跑完删） | `0915-1600ep-m8x8-modul-training-plan.md` B 节同法 | `SMOKE20=PASS`、`PARAM_TREE_EXACT … n_model=65` |
| V10 | 起跑 preflight | `preflight_train_launch.py` | `PREFLIGHT=PASS` |

### G. 顺序、留档与 commit

1. 本文入库（`docs:`），push。
2. 开发副本：A–E 全部代码改动 + CPU 侧单测（`test_pack_guards.py`、`motion_gates_online.py` 更新）→ `commitV10.0: demo 段补帧网格与 modulation 接入 motion`，push。V1 / V2 / V6 在开发副本 GPU 0–3 做（不写 `v1-store` 数据区）。
3. 训练结束 → 主副本解锁、`git pull` → B 节建库（8 卡）→ 建库留档 `docs:`。
4. V3–V5、V7–V9 → 留档 `docs/training-doc/mv2-guard-*/`。
5. budget 拍板 → YAML 定值 → 正式 run 计划另起（run_name 待确认）。

commit 约束：步骤 3 的 Wan → encode 之间零 commit；每个 commit 后立即 push；只 `git add` 本轮明确文件。
