# BinFill 补 demo 前缀后重测三个 50k 模型（150 条 × 3）

本计划的前置背景是 `primary700-nomotion50k-gl`（26.00%）、`primary700-motion50k-gl`（40.71%）、
`primary700-modul32frame-50k-gl`（34.86%）三轮 700 条评测，均已跑完并留档。本计划处理其中暴露出来的
BinFill 异常，与前三轮的评测口径无重叠。**本文件只是计划，尚未实施。**

## 第一部分：给人看

### Context：为什么做这件事

用户指令原话：「参考 https://github.com/hongzefu/robomme_policy_learning_MotionJEPA/tree/v2-motionmem 训练集合的构成
binfill需要增加demo / 还有什么训练/推理不一致的 先对抗验证 / 然后重新测50k 512 512+motion 2048的binfill」。
随后对 motion 预算问题追加：「motion budget这个问题 仍然2000步 超出160给出继续运行方案」。
最后：「把计划固化在根目录 不执行」。

**要解决的问题**：三条 run 在 BinFill 上出现方向一致、且随感知记忆变宽而加剧的退化——
8 帧/512 档 29.3%、512+motion 12.0%、32 帧/2048 档 10.7%。两条机制完全不同的 run 撞进同一个坑，
`primary700-modul32frame-50k-gl/result.md` 当时只能记成「感知记忆一变强就变差的共性问题」，没有归因。

**结构性错位已由三路独立调查确证**（下面三条都是代码 + 实测事实）：BinFill 是四个任务里唯一
「训练有 demo 段、评测没有」的。**但「这个错位就是退化的原因」目前只是归因假设，正是本轮实验要验证的东西**
——不要在留档里把它写成已证结论。

- 训练侧：`third_party/robomme_benchmark/scripts/generate_dataset_newseed.py::_binfill_demo_deliverable`
  （用户 2026-09-11 决策的「路线 B」）在 episode 录完、h5/mp4 落盘后，把整条成功轨迹复制一遍接在自己前面。
  `_binfill_duplicate_h5` 用 `h5py.Group.copy` 整组搬运，只改 `info/is_video_demo=True` 与 `info/is_completed=False`
  两个标量——**h5 里 demo 段与 exec 段逐位相同**，红框只加在 mp4 上而 mp4 不进训练链路。
  实测 BinFill 400 集 **400/400** 满足 `num_timesteps == 2 × exec_start_idx`，`exec_start_idx` min 264 / mean 642 / max 1152。
- 评测侧：`BinFill.py::_initialize_episode` 的 `task_list` 共 2N+1 条，`"demonstration"` 三处全是 `False`，
  `DemonstrationWrapper.get_demonstration_trajectory()` 筛不到 demo 任务
  → `reset()` 的 `front_rgb_list` 只有 1 帧 → `eval.py::init_episode` 的 `exec_start_idx = len(image_buffer)-1 = 0`。
  实测 300 次 reset 全部为 0。
- 取帧两侧共用 `src/mme_vla_suite/shared/sampling.py::even_sampling_indices`，训练侧也只用到当前 t 为止，
  **算法本身一致**；差的纯粹是「喂进去的帧序列里有没有 demo 前缀」。训练时 8 帧档有 4–7 帧、32 帧档有
  16–28 帧落在 demo 段，评测时是 0；motion 侧更彻底——训练时 demo 窗 16–71 个常驻，评测时
  `while s+16 <= es-1 = -1` 一次都不转。**记忆通道越宽、被掏空的比例越大**，方向上与观测到的退化梯度
  （29.3% → 12.0% → 10.7%）一致——这是本轮要检验的假设，不是已证因果。
- 其它三个任务训练 es 区间与评测 es 区间完全对齐（RouteStick 100–500/100–500、VideoRepick 252–508/254–502、
  VideoUnmaskSwap 114–318/114–318）。

**一个必须先说清的前提**：即使补上 demo，BinFill 也不可能与训练完全一致——训练时 demo 段与 exec 段在 h5 里
逐位相同（模型能直接「抄」自己待会儿要走的轨迹），评测时 demo 只能是专家轨迹、exec 是策略自己的。
这是训练数据构造本身的性质，评测侧修不掉。本轮能消除的是「有没有前缀段」这个结构性错位。

### 结论先行：怎么做

**三块改动 + 一轮重测**：

1. **预生成 demo 前缀**：用 candidates 的同一 seed + 同一 spec **另建一个 env**，把 BinFill 的 `task_list` 里
   `demonstration` 临时翻成 True 让 `DemonstrationWrapper` 在 `reset()` 里把整条跑完并收帧，收完**丢弃该 env**；
   正式评测再新建一个干净 env（`demonstration` 仍为 False）交给策略。monkey-patch 只在主仓库的预生成脚本里做，
   **不动 benchmark 源码、不换 gitlink**。
2. **评测端注入**：在 `SpecEnvRunner.get_init_obs` 把 demo 帧拼到 `images`/`wrist_images`/`states` 前面，
   默认关闭、只对 BinFill 生效。
3. **motion 窗超预算降级**：`max_steps` 保持 2000，`_prepare_motion` 在 `k > budget` 时按
   「demo 段全保 + exec 段内等距降采样」继续跑，不再抛异常把整集判成 error。
4. **重测**：3 个模型 × BinFill 150 条（easy/medium/hard 各 50）。无 demo 对照直接用现有 700 条里的
   BinFill 结果，不重跑。

**为什么直接翻 benchmark 的 `"demonstration"` 标志不行**（已确证，不要再试）：方块入 bin 是
`is_any_obj_dropped_onto_delete` 里的 `obj.set_pose(sapien.Pose(p=[10,10,0]))` 物理删除，
`*_cubes_in_bin` 单调累加且无复位路径；demo 跑完 `sequential_task_check` 的任务指针已走到
`All tasks completed`，策略第一步就拿 success。BinFill 也没有 VideoRepick/RouteStick 那种
`"name": "NO RECORD"` + `solve_strong_reset` 尾巴（后者只归位机械臂 qpos，不碰物体），
spawn 余量也不够演两遍（easy spawn 5 / target 3）。**「另建一个 env 演完就扔」正是绕开这个死结的办法。**

### motion 降级口径：demo 全保 + exec 段内等距降采样

**超预算的唯一来源是 exec 段**。训练侧 1600 集实测 demo 窗上限 **71**、exec 窗上限 **70**、合计上限 **141**
（全集无一超 160，与 `framesamp_dataset.py` 的建库期预检一致）。评测侧 `max_steps=2000` 让 exec 窗恒定在
`len(range(0, 1969, 16)) = 124`，是训练 exec 上限的 1.77 倍；demo 侧则落在训练支撑集内。

所以降级让 demo 段**一窗不动**（与训练逐窗同形同内容），超额全部由 exec 段承担：

```
k_demo + k_exec ≤ 160  →  原样，与今天逐字节一致
k_demo + k_exec > 160  →  demo 段全保，exec 段在段内 np.linspace 等距取 (160 − k_demo) 个
```

实现上另有一条 `demo ≤ budget//2 = 80` 的保护，防 demo 变长时把 exec 挤成 0。
**这条保护会不会真的丢掉 demo 窗，必须靠长度闸保证，不能靠「与训练同分布」推断**——
训练观测到的 `k_demo ≤ 71` 是样本最大值，不是新生成 demo 的上界。落地办法见第二部分 A 节的
`D ≤ 1281` 硬闸（`D=1281 ⇒ k_demo=80`）与预生成后的 `DEMO_WINDOW_GUARD` 硬验。

**降级不需要重编 motion——编码与装配是分开的两层**：

| 层 | 函数 | 做什么 | 受降级影响吗 |
|---|---|---|---|
| 编码 | `_encode_ready_windows` → `_encode_window` | 每 16 帧到货就把新够格的窗送 Wan VAE + MotionJEPA 编一次，按起点帧号存进 `_history_feats_motion` 字典。**每个窗一辈子只编一次** | **否**，与 budget 无关 |
| 装配 | `_prepare_motion` | 每次 infer 从字典里取出当前合法的窗，填进固定形状的 `(160,768)` 数组 | 是，只改「挑哪 160 个」 |

被装配层丢掉的窗仍留在字典里，下一步（t+16）重新算等距取点时可能又被选中，**直接从字典取，不重编**。
算力账：编码最坏 195 次 × 1.68 s ≈ 328 s（这是补 demo 带来的，与降级无关），降级本身是一次
`np.linspace`，微秒级。

这也正是**不能连编码一起跳过**的原因：每步选中的集合都在变，没有哪个窗能永久跳过；且
`_next_grid_start_exec` 这个游标同时兼着原始帧淘汰线（`keep_from = es + _next_grid_start_exec`），
冻住它会让 `_raw_frames` 无限增长。

**触发时机**：第 j 次 infer 对应 `t−es = 16j`、`k_exec(j) = max(0, j−1)`，
所以首次触发在 `j = 162 − k_demo`，即 `t−es = 16(162 − k_demo)`；受影响的 infer 次数 = `max(0, k_demo − 36)`。

| D（= es） | k_demo | 首次触发于 exec 第几步 | 受影响 infer / 126 |
|---|---|---|---|
| 638（训练中位） | 39 | 1968 | 3（2.4%） |
| 800 | 49 | 1808 | 13（10%） |
| 1152（训练最大） | 71 | **1456** | 35（28%） |

注意 D=1152 时**第 1456 步就会触发，不需要跑满 2000 步**——集只要活过 1456 步就进触发组。

**开关不能放 YAML**：评测侧的 motion 配置来自 checkpoint 桶里的 `history_config.resolved.yaml`，其 sha256 被
`check_checkpoint.py::RUNS` 与 `motion_provenance.json` 三方钉死（`policy_config.py::_load_resolved_snapshot`）——
改仓库里那份根本到不了评测路径，改桶里那份直接炸 sha 闸。**只能走环境变量**
`MMEVLA_MOTION_OVERFLOW=raise|resample`，**默认 `raise`**（照抄既有 `MMEVLA_MOTION_PROV_RELAX` 的先例）。

**已有 550 条旧结果沿用是安全的**：降级代码完整包在 `if k > B:` 内，而旧 11 组实测 `max(es) = 502`
⇒ `k_max = 31 + 124 = 155 < 160`，分支不可达。起跑前须按验证节的两步法复核这个上界（判据 `max(es) ≤ 592`）。

### 已查清但本轮不处理的其它不一致

| 编号 | 内容 | 处置 |
|---|---|---|
| S2 | `max_steps=2000` 让 `t/es` 冲出训练分布（训练上限 2.00–4.13，评测 RouteStick easy 可达 21） | 一改就与已有三方结果不可比，单独立项 |
| S4 | motion sidecar 离线表烧在 A100（cc 8.0/108 SM），评测在 A40（8.6/84），靠 `MMEVLA_MOTION_PROV_RELAX` 放行，幅度从未测过 | 记待办 |
| S7 | 2 条评测 prompt 训练 60 种里没有（`put four red cubes and one blue cube …`、`put one red cube and four blue cubes …`），都在 BinFill | 记入留档，不改 |
| — | `scripts/dataset/scan_16task_memory.py` 的 `TASK_WITH_VIDEO_DEMO` 漏了 BinFill（路线 B 之后 h5 确实有 demo 段） | 既存、与本轮无关，单开任务 |

**确认一致、不必再查**：prompt 文本与大小写（两侧本就全小写，`.lower()` 是 no-op）、tokenizer 与 transforms 链、
图像预处理（无 BGR、无 ImageNet 归一）、views、8 维 state、norm_stats（三 run 同一 sha `856c75ea…`）、
action 空间与 delta/absolute 掩码、帧采样算法、`mem_order`、motion 窗公式、budget、任务与难度集合、
训练/评测 episode 零重叠（BinFill 训练 0–145 / 评测 153–204）。

### 执行顺序（每步带判定行）

1. **代码改动 + 单测**（全部在起跑前落地并 commit；此后到全部分片结束前不再 commit、不留未跟踪文件——
   `EXPECTED_GIT_HEAD` 闸与 `run_shard.sh` 的 porcelain 闸都会拒绝脏仓库）。
   判定：`bash -n`、`motion_gates_online.py`、`eval_rhythm_gates.py`、`verify_sources.py → SOURCES_PASS`、
   新增的 `motion_overflow_bitcmp.py → MOTION_DEFAULT_BITEXACT=PASS` 与 `MOTION_HEADROOM=PASS`。
2. **冒烟：一条 episode 的 demo 生成 + 注入**（≤5 分钟，占一张卡）。判定见「验证」一节。
   这一步回答两个只能实测的问题：**1153 帧过 SigLIP 会不会 OOM、单集耗时**。
   （「planner 重试是否有效」这一条**不在这里回答**——需要走过 RRT\* 分支的病例，见验证节。）
3. **全量预生成 150 条 demo**（8 个占位作业动态抢单，估 20–60 min，`--retries 1`）。
   判定：`PREGEN_OK episodes=150 ok=<n> failed=<m> used_rrt=<n>` + `DEMO_WINDOW_GUARD`。
   跑完后用 `used_rrt=True` 的集做重试有效性实测，再定最终剔除策略；失败率 >5% 回来找用户裁决。
4. **提交重测**：3 run × 10 片 = 30 行 manifest，复用现有 `gl_hold_pool.sh` 的 MANIFEST 模式。
   判定：30 个 `SHARD_PASS`，3 个 `MERGE_OK episodes=150`。
5. **留档 + commit + push**。

### 明确不做

- 不改 benchmark 源码、不动 gitlink、不建 benchmark 分支。
- 不改 `max_steps`（保持 2000）、不改 `motion.budget`（保持 160）、不改训练链路、不重跑非 BinFill 的 550 条。
- 不去掉 BinFill prompt 里的数量提示（那是另一个实验，本轮不做）。
- planner 失败的 episode **不换 seed、不静默无 demo 跑**（`DemoPrefixStore.get` 缺条目直接 raise）。
  重试次数默认 1，最终剔除策略等 RRT\* 病例实测出来再定；被剔除的集在对照时按同一子集重算 baseline。
- 不做多 seed、不补老基线、不处理上表 S2/S4/S7。

---

## 第二部分：技术细节（供 agent 追踪）

### A. 预生成（新增 3 个文件 + 改 1 个）

**`scripts/evaluation/pregen_binfill_demo.py`**（新增，核心）

monkey-patch 下手点 = wrap `DemonstrationWrapper.get_demonstration_trajectory`（**不是** patch `BinFill._initialize_episode`，
后者是 ManiSkill `BaseEnv.reset` 的内部回调、可能因 reconfigure 被调多次，且其后紧跟 `task4recovery` / `inject_fail_grasp`）：

```python
_ORIG = DemonstrationWrapper.get_demonstration_trajectory

def _patched(self):
    """只对 BinFill：在原实现扫描 task_list 之前把全部 2N+1 条的 demonstration 翻成 True。
    翻的是 self.unwrapped.task_list——原实现第一行经 gym.Wrapper.__getattr__ 取到同一个 list 对象。
    最后一条 press the button 也翻：训练侧 demo 是含按钮的整条成功轨迹，不按按钮的 demo 与训练分布不符。
    副作用（方块被删、任务指针走到 All tasks completed）无所谓——这个 env 演完就扔。"""
    if self.unwrapped.spec.id == "BinFill":
        tl = self.unwrapped.task_list
        if len(tl) % 2 != 1:
            raise RuntimeError(f"BinFill task_list 长度 {len(tl)} 不是 2N+1")
        for e in tl:
            e["demonstration"] = True
    return _ORIG(self)

DemonstrationWrapper.get_demonstration_trajectory = _patched
```

env 构造**必须复用 `examples/robomme/env_runner.py::SpecEnvRunner`**，不要直接调 `make_env_for_spec`——
这样 `RUNTIME["kwargs"]`、`_wrap_env` 的 11 个 `include_*`、`max_steps+2`、难度校验全部与评测侧逐字同源。
候选行走 `robomme.injection_candidates.load_candidates` / `candidate_key`，与 `eval.py::load_plan` 同一条。

取帧：`D = len(pre["images"]) - 1`，存 `pre["images"][:-1]`。丢掉的 `[-1]` 是 demo 跑完后的复位帧，
注入场景下这个角色由干净 env 自己的 reset 帧担任。三个 buffer 必须等长
（`pack_buffer` 对 `image_buffer`/`state_buffer` 各做 `np.stack`）。形状：front/wrist `(256,256,3) uint8`
（`BinFill.py` 的 `CameraConfig("base_camera", ..., 256, 256, ...)`，与 `motion.frame_size: 256` 及 `add_buffer` 的
raise 校验对上）、state `(8,) float32`（`env_runner.py::pack_state`，预生成与在线共用同一函数）。

四道正确性校验，全部记进 h5 attrs，任一不成立即判 planner 失败：

```python
ok = (info_flat.get("status") == "success"
      and all(in_bin[c] == n for c, n in row["spec"]["objects"]["target_count"].items())
      and sum(in_bin.values()) == row["spec"]["objects"]["put_in_total"]
      and int(base.timestep) == len(base.task_list)     # 任务指针走完 ⟹ 按钮曾被按下（见下）
      and demo_wrapper.episode_success)
```

**⚠ 按钮这一项绝不能写成 `is_button_pressed(base, obj=base.button)`**：
`subgoal_planner_func.py::solve_button` 按下按钮后会**抬手**，按钮弹簧随即回位，而成功状态早已在
`sequential_task_check` 里锁存。于是 `status=="success"` + 计数正确 + `episode_success=True` 与
`is_button_pressed()==False` **可以同时成立**——按结束瞬时状态判会把正常样本当成 planner 失败剔除。
正确的锁存量是任务指针：`task_list` 最后一条 `press the button` 的 `func` 就是 `is_button_pressed`，
`sequential_task_check` 里 `self.timestep` 单调递增不回退、走完才置 `num_tasks`，
所以 `base.timestep == len(base.task_list)` ⟺ 某个 evaluate 时刻按钮确实被按下过。
h5 attrs 里 `button_pressed_final`（结束瞬时值）仍然记录，但**只作参考、不进判据**。

`*_cubes_in_bin` 只在 `is_any_obj_dropped_onto_delete`（`subgoal_evaluate_func.py`）里 +1，是「真放进去了」
最直接的证据。`get_demonstration_trajectory` 会吞掉 screw→RRT* 双重失败并 `continue`，失败集会安静产出
「少放一个方块」的 demo，上面四道就是拦它的。

**两道硬闸**（取交集，任一不满足即标 `demo_status="too_long"` 剔除并计数）：

| 闸 | 判据 | 由来 |
|---|---|---|
| pos 表 | `D + MAX_STEPS + 1 ≤ 4096` ⟹ `D ≤ 2095` | `FrameSampMemory.__init__` 的 `max_steps=4096` 无人覆盖，`pos_emb` 只有 4096 行 |
| **demo 窗** | `k_demo(D) = ⌊(D−17)/16⌋+1 ≤ 80` ⟹ **`D ≤ 1281`** | 让「降级时 demo 一窗不丢」成为**可执行约束**而不是分布推断 |

生效的是更紧的 `D ≤ 1281`。训练 D 上限 1152 ⇒ `k_demo = 71`，留 9 窗余量。

**预生成全部跑完后再硬验一次实测上界**，判定行：

```
DEMO_WINDOW_GUARD max_D=<n> max_k_demo=<n> cap=80 train_observed_max=71 over_train=<n>
```

`max_k_demo > 80` 即停（理论上被硬闸挡住，真出现说明闸写错了）；
`max_k_demo > 71`（即超出训练见过的最大值）不停，但要在留档里点名有几条、分别是哪几条。

落点 `v1-store/demo-prefix/binfill-<identity_sha256[:12]>/`（`v1-store/` 已整目录 gitignore）：
`manifest.json` / `index.json`（键 `"BinFill/<difficulty>/<episode>"`）/ `BinFill/<diff>/ep<N>.h5` / `logs/`。
h5 编码：`front_png` vlen uint8 用 `cv2.imencode(".png")` **无损**（进 SigLIP 与 Wan VAE，绝不能有损）、
`wrist_jpg` q90（只进 mp4）、`state (D,8) float32`。实测 256×256×3 sim 渲染帧 PNG ≈ 104 KB（比 0.53），
150 条约 **11–12 GB**；**不要用 h5 gzip filter**（RGB 交错无预测器，实测只有 0.62–0.75）。
attrs 含 `front_raw_sha256` 供加载端逐位验证解码。

分片 10 片 × 15 集。`--max-new-envs 20` 达到即退出 + 外层 `while` 循环续跑，照抄 `run_shard.sh` 的分块思路与
`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536`（Vulkan 静态 TLS 红线 ≈27 次 `make_env`/进程，
见 `docs/training-doc/tic-vulkan-makeenv/result.md`）。

**`scripts/evaluation/run_demo_shard.sh`**（新增）：clean-HEAD 闸 + `verify_sources.py` + 分块循环 + 验收。
**`scripts/evaluation/gl_demo_pool.sh`**（新增）：逐字复制 `gl_hold_pool.sh` 的 `mkdir claims/<item>` 抢单原语，
只换 `QUEUE_DIR`、清单（`items.txt` 行为分片下标 0..9）与 body（`srun --jobid --overlap --exact` 调 `run_demo_shard.sh`，
带 `/usr/bin/env -u MANIFEST`——commit `e9c46c6` 踩过这个坑）。**不改 `gl_hold_pool.sh` / `gl_hold_queue.sh`**。
**`scripts/evaluation/make_shard_plan.py`**（改）：加 `task` 子命令产 BinFill-only 计划，只收 `demo_status=="ok"` 的集。

### B. 评测端注入（改 3 个文件）

**`examples/robomme/env_runner.py::SpecEnvRunner`**——注入点选这里，不选 `eval.py::init_episode`：
`eval.py` 下游的三个 buffer、recorder 循环、`exec_start_idx = len(image_buffer)-1`、
`is_video_demo` 表达式**五处全部自动正确**，`eval.py` 逻辑一行不用改。

```python
def __init__(self, ..., demo_prefix_store: str = ""):
    # 守卫①：其余三任务有原生 demo，再拼一段会双重叠加
    if demo_prefix_store and task != "BinFill":
        raise ValueError(f"demo 前缀注入只支持 BinFill，得到 {task}")
    self._demo_store = DemoPrefixStore(demo_prefix_store) if demo_prefix_store else None

def get_init_obs(self):
    obs = super().get_init_obs()
    if self._demo_store is None:
        return obs                      # 关闭时逐字等价
    prefix = self._demo_store.get(*self._demo_key)   # 缺条目直接 raise，绝不静默降级
    if prefix["task_goal"] != obs["task_goal"]:
        raise ValueError(...)
    # 形状/等长断言后：
    obs["images"] = list(prefix["images"]) + list(obs["images"])   # wrist/states 同
    self.demo_prefix_len = len(prefix["images"])
    return obs
```

新增 `examples/robomme/demo_prefix.py::DemoPrefixStore`（读 index、开 h5、`cv2.imdecode`、比 `front_raw_sha256`）。

**off-by-one 核对（结论：正确）**：注入后 `image_buffer = [demo 0..D−1] + [干净 env 的 reset 帧]`，长度 D+1，
`exec_start_idx = D`。训练侧语义 demo 段 `[0, es)`、exec 起点 `es`——一致。与原生 demo 任务同构
（VideoRepick 的 `reset()` 返回 `[D 帧 planner] + [1 帧复位步]`，`es = D`）。运动路尾窗补齐用 `[es−1]*(W−real)`
（`demo_tail_pad: repeat_last`），`es−1 = D−1` 是最后一帧 planner 帧（方块已入 bin、按钮已按过），
正是训练侧 demo 段末帧的对应物。

**`examples/robomme/utils.py::TASK_WITH_VIDEO_DEMO` 不加 `"BinFill"`**——那条常量表达「该任务恒有 video demo」，
对 baseline 700 条的 BinFill 是假命题。改 `eval.py::init_episode` 唯一一行走显式长度：

```python
demo_len = getattr(env_runner, "demo_prefix_len", 0)
is_video_demo=(i < demo_len) if demo_len else
              (env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1),
```

其余三份 `TASK_WITH_VIDEO_DEMO` 副本（`scripts/training/legacy-eval/robomme-{local,remote}/utils.py`、
`scripts/motion-variance/robomme/utils.py`）本轮走不到，不同步。

**开关**：`eval.py::Args` 加 `demo_prefix_store: str = ""`（默认关）；`evaluate()` 里守卫②
（计划里有非 `BinFill/*` 组即 raise）；`run_shard.sh` 按既有 `${EPISODE_WALL_S:+...}` 写法加
`${DEMO_PREFIX_STORE:+--args.demo_prefix_store="$DEMO_PREFIX_STORE"}` 并做绝对路径 + `index.json` 存在性校验；
`gl_hold_queue.sh` 的 `PASS` 透传白名单加 `DEMO_PREFIX_STORE`。

**`MMEVLA_ENC_CHUNK`（新增，默认 0 = 关闭）**：首批 `add_buffer` 最坏送 1153 帧进 SigLIP
（现有最长案例 VideoRepick 502 帧），`image_jnp` 一次性 `(1153,1,224,224,3) float32` ≈ 694 MB、
输出 bf16 ≈ 1.2 GB，OOM 风险未测。在 `framesamp_memory.py::add_buffer` 的编码段加固定尺寸分批：

```python
chunk = int(os.environ.get("MMEVLA_ENC_CHUNK", "0")) or t   # 0 → 一次循环，逐字等价
```

**只切编码循环，绝不把 demo 段拆成多次 `add_buffer`**——demo 的 while 判据
`_next_grid_start_demo + 16 <= es - 1` 不依赖 `last_frame`，拆批会去编尚未到货的窗口并抛
「起点 f 的 33 帧不齐」。`_encode_ready_windows` 仍在 `add_buffer` 末尾调用一次，运动路看到的仍是
「整段 demo 首批到货」。**口径说明**：开启时切 batch 维可能因 XLA 为不同 batch 尺寸选不同 kernel 而改变 bf16 累加序，
不保证逐位相同。**被切的不止 demo 段**——首批是 `D` 帧 demo **加 1 帧干净 env 的 reset 观测**，
后者是 exec 段第 0 帧、有历史口径，同样落进新的 batch。稳态 16 帧批恒 `t < chunk`、一次循环、不受影响。
因此必须先跑 `enc_chunk_cmp.py`（验证节）把这个差异**测出数来**再决定，不能凭「只切了新东西」放行。
本轮暂定 `MMEVLA_ENC_CHUNK=256`，实测超出 0.33% 量级则回来重议。

### C. motion 超预算降级（改 3 个文件）

**`framesamp_memory.py`**：`__init__` 读 `motion_cfg["overflow"]`（默认 `"raise"`）并初始化
`motion_downsample_steps` / `motion_downsample_max_k` / `motion_last_k`（`clear()` 里一并清零）；
新增两个模块级辅助 `_even_pick(n, keep)`（`keep == n` 时恒等返回，禁止走 linspace；用
`np.linspace(...).astype(np.int64)` **截断**而非 `round`，与帧路 `even_sampling_indices` 同一算子）
与 `_motion_quota(k_demo, k_exec, budget)`；`_prepare_motion` 改为：

```python
frames = self.visible_motion_frames(step_idx)
k = len(frames); B = self.motion_budget
self.motion_last_k = k
if k > B:
    if self.motion_overflow != "resample":
        raise RuntimeError(...)            # 默认口径，与今天一字不差
    es = self.exec_start_idx
    demo  = [f for f in frames if f <  es]
    exec_ = [f for f in frames if f >= es]
    keep_demo, keep_exec = _motion_quota(len(demo), len(exec_), B)
    frames = ([demo[i]  for i in _even_pick(len(demo),  keep_demo)] +
              [exec_[i] for i in _even_pick(len(exec_), keep_exec)])
    if len(frames) != B or any(b <= a for a, b in zip(frames, frames[1:])):
        raise RuntimeError(...)            # 显式核，不靠注释
    self.motion_downsample_steps += 1
    self.motion_downsample_max_k = max(self.motion_downsample_max_k, k)
    print(f"MOTION_DOWNSAMPLE step={step_idx} es={es} k={k} k_demo={len(demo)} k_exec={len(exec_)} "
          f"kept_demo={keep_demo} kept_exec={keep_exec} budget={B}", flush=True)
    k = len(frames)
# 以下（emb/pos 分配、for 循环、mask[:k]、pad_times）一字未动
```

**不要「去重后取」**：`k ∈ [161,400)`、`B=160` 时步长 `(k−1)/(B−1) > 1`，截断构造不会产生重复；
去重反而危险——一旦 `len(frames) < B`，`mask[:k]` 与 `pad_times` 会静默给出「非满」形态而无任何断言能发现。

**编码开销不变**（已核）：`_encode_ready_windows` 的两个 while 只看 `demo_min_real / W / last_frame / es`，
与 budget 无关，最坏 195 次编码。**不建议连编码一起跳**：降级保留集每 16 帧重算一次，被丢的窗在下一步可能
被选中，没有窗可以永久跳过；且 `_next_grid_start_exec` 是双重语义（既是「编到哪」又是
`keep_from = es + _next_grid_start_exec` 的原始帧淘汰线），冻住它会让 `_raw_frames` 无限增长。
单窗实测 1.68–1.77 s（greatlakes A40，从 `primary700-motion80k-gl/s*/server.log` 的 `TIMING add_buffer_ms` 算出）。

**`policy.py`**：`_motion_cfg` 组装处读环境变量（优先级 = 环境变量 > resolved 快照可选键 > `"raise"`），
非法值 raise，并 `print(f"MOTION_OVERFLOW={overflow} budget={...}")` 落 `server.log`；
`infer` 出口随 `infer_time_ms`（已有先例）回传 `motion_k` 与 `motion_downsample_steps`。

**`eval.py`**：`get_action_chunk` 留住整个响应（现在是 `client.infer(element)["actions"]`），
累计 `motion_k_max` / `motion_downsample_steps` / `motion_infers` 到 `EpisodeState` 新增的三个字段
（**不要**放进 `clear_buffers`，它每个 chunk 调一次）；与 `progress.json` 同一个 `finally` 块写
`motion_stats.json`（**不能塞进 `progress.json`**——`check_shard.py` 硬断言 `type(value) is bool`）。

**`check_shard.py`**：加 `MOTION_WINDOWS=PASS|FAIL|SKIP` 闸门。公式必须在本文件**本地重写**
（`CLIENT_UV` 的 `client/pyproject.toml` 没有 `mme_vla_suite`），体例照抄
`scripts/training/tests/eval_es_bound.py::predicted_windows`——两个独立实现互相复核，共用代码会让这道闸失效。
五条判据：覆盖（非 error 集都有记录）、公式（服务端枚举 == 客户端闭式）、
降级触发与预算严格互为充要、未开 resample 时降级次数必须为 0、budget 自洽。
`motion_stats.json` 不存在时按 `EXPECT_MOTION_STATS` 分叉：期望为真（本轮 motion run）→ **FAIL**；
否则 → `SKIP`（非 motion 模型、以及复核旧 run 的 `check.json` 不失败）。
`merge_shards.py` 合并各片 `motion_stats.json` 到 `merged/`。

### D. 重测编排

三个 run：

| run_name | CKPT | POLICY | 预期降级 |
|---|---|---|---|
| `binfilldemo-nomotion50k-gl` | `robomme-vla-modul-60k-v1/50000` | `perceptual-framesamp-modul-8frame-8x8` | 不涉及 |
| `binfilldemo-motion50k-gl` | `robomme-vla-modul-motion-80k-v1/50000` | `perceptual-framesamp-modul-8frame-8x8-motion` | 约 63.5% 的集会触发 |
| `binfilldemo-modul32frame-50k-gl` | `robomme-vla-modul-2048-80k-v1/50000` | `perceptual-framesamp-modul-32frame-8x8` | 不涉及 |

共用口径与前三轮逐字相同（seed 7、`max_steps=2000`、`EPISODE_WALL_S=2400`、`EVAL_TIMEOUT=14400`、
benchmark gitlink `b4e97f2`、1 卡 / 1 CPU / 24G），新增 `DEMO_PREFIX_STORE`、`MMEVLA_ENC_CHUNK=256`、
motion run 另加 `MMEVLA_MOTION_OVERFLOW=resample`。`CHUNK_EPISODES=15`（BinFill 150 条 / 10 片正好 15 集/片）。
30 行 manifest 走现有 `gl_hold_pool.sh` 的 MANIFEST 模式，**那个脚本一行不用改**。

**耗时估算**：无 motion 两条各约 4.2 h 串行、motion 一条约 6.5 h 串行（motion 单集增量最大——
`W_d(D) × 1.64 s`，D=642 时 +66 s、D=1152 时 +116 s），8 worker 动态抢单约 **2–3 h 墙钟**，
加预生成 20–60 min。

**host RAM 需留意**：`primary700-motion50k-gl` 实测 cgroup anon 峰 18.24 GiB / 24 G 上限，补 demo 后估
+1.5–2 GB → 约 20 GiB，余量变薄。起跑前用 `scontrol show job` 确认占位作业的实际 `--mem`；
若确为 24 G 且冒烟显示吃紧，motion 那条 run 另提一批 `--mem=32G` 的作业。

### 验证

**起跑前（≤5 分钟，纯 CPU）**

- `bash -n` 全部改动的 shell；`verify_sources.py → SOURCES_PASS`。
- `motion_gates_online.py`（budget=8/es=200 的阴性用例）与 `eval_rhythm_gates.py`
  **必须在不设 `MMEVLA_MOTION_OVERFLOW` 的情况下重跑并通过**——证明默认口径没变。
- 新增 `scripts/training/tests/motion_overflow_bitcmp.py`：注入确定性伪 token（`_prepare_motion` 的输出
  不依赖编码器），对 `git worktree add` 出来的改动前基线与新码各跑一遍全部 es × 126 步，
  逐叶 sha256 比对 → `MOTION_DEFAULT_BITEXACT=PASS`。**这条只验装配层，不验 SigLIP 数值**（见下一条）。
- `MOTION_HEADROOM=PASS es_max=<n> k_max=<n> budget=160`，判据 `es_max ≤ 592`。
  **es 上界不能用 `grep … | tail -1` 求**——那只是「已有日志里的最大值」，日志缺失就会给出偏小的假上界
  （50k run 的 s9 server.log 为空、s4 只有 2 条）。正确做法两步：
  (a) 从 `candidates.jsonl` + 各 run 的 plan 直接枚举 11 组全部 550 条的组×难度，按每个 `(task,difficulty)`
      的 demo 长度上界（benchmark 元数据里固定的示范视频长度）算理论 `es_max`；
  (b) 用日志实测值**交叉核对**，并先断言日志条数 == 期望集数，条数不足即判 `INCOMPLETE` 停下来。
  两条都过才认 `es_max`。
- **新增 SigLIP 分批真实对照** `scripts/training/tests/enc_chunk_cmp.py`：
  拿一条真实 demo 前缀的 `D+1` 帧（**必须含最后那一帧干净 env 的 reset 观测**），
  用同一个 checkpoint 的 `vision_enc_fn` 分别以 `MMEVLA_ENC_CHUNK=0`（一次性）与 `256`（分批）各编一遍，
  逐帧比相对差与余弦。**覆盖点必须包含 reset 帧与每个分批边界帧**（`b0=256/512/768/…` 及其前一帧）。
  判定行 `ENC_CHUNK_CMP rel_max=<..> cos_min=<..> reset_frame_rel=<..> boundary_rel_max=<..>`。
  阈值先不预设——**这一条是测量，不是闸门**：本轮第一次跑出来的数就是基线，写进留档；
  若 `rel_max` 超过既有 f32/bf16 编码器差异的量级（`docs/train-infer-consistency.md` 七节记的 0.33%），
  停下来交用户决定是否改用其它规避 OOM 的办法。

**冒烟（一张卡，约 5 分钟）**——挑 `BinFill/easy/ep156`（`put_in_total=1`，最短）

- `DEMO_OK task=BinFill diff=easy ep=156 D=<D> status=success in_bin={...} target={...} timestep_done=True`，
  `D` 落在 264–1152 的量级内；另落一个 `smoke.mp4` 肉眼确认真的演了一遍并按了按钮。
- **重试有效性：必须用确认走过 RRT\* 的病例测，不能用这条 easy 集**。
  `DemonstrationWrapper` 里只有 **screw 规划失败才回退 RRT\***，一条顺利的 easy 集大概率根本没进这个分支，
  三次结果相同只能说明「screw 路径是确定性的」，推不出「重试无效」。
  正确做法：预生成脚本记录每条集**是否进过 RRT\* 分支**（在 `_solve_task_without_hard_fail` 的回退路径上打点，
  或从 planner 日志里抓），先跑一遍拿到走过 RRT\* 的集，再在**那些集**上连跑 3 次比 `D` 与 checks。
  在拿到这个结论之前 **`--retries` 默认设 1、不设 0**，并把每条的 `attempts` 与 `used_rrt` 记进 h5 attrs；
  剔除策略等实测出来再定，不预先下全局结论。
- 注入后核对：`exec_start_idx == D`、首批 `len(images) == D+1`、`_history_feats` 条目 `D+1` 键 `0..D`、
  `len(_history_feats_motion) == W_d(D) = ⌊(D−17)/16⌋+1` 键 `0,16,32,…`、
  首次 infer 的 `len(visible_motion_frames(D)) == W_d(D)`、`_raw_frames` 首批后只剩 1 条（索引 D）、
  `mem_order` 长度 `512+160=672` 且置换自检不报错。
- `np.abs(front[0].astype(int) - reset_frame.astype(int)).mean()` 应接近 0——
  demo 段第 0 帧与 exec 段第 0 帧是同一场景初始态的两次渲染，这对应训练里「demo 段第 0 帧 == exec 段第 0 帧」的逐位恒等。
  **这是最值得先看一眼的一致性证据。**
- 再跑一条 D 偏大的 hard 集，确认降级真的生效（`MOTION_DOWNSAMPLE` 行出现且不再抛 `RuntimeError`）。

**与训练分布的对照判据**——`num_timesteps == 2·exec_start_idx` **在评测侧不应成立、不能当判据**
（那是训练侧整条复制造成的结构性恒等；评测 exec 段长度由策略决定，baseline 实测 BinFill 帧数 min 97 / max 2001）。
该对的是前半段：`DEMO_ES_DIST train=[264,642,1152] eval=[min,mean,max] overlap=<..>`。

**验收**：30 个 `SHARD_PASS`、3 个 `MERGE_OK episodes=150`（若有 planner 失败剔除则为对应数）、
`DEMO_WINDOW_GUARD` 通过、`MOTION_WINDOWS=PASS`。

两处验收口径要收紧：

- **本轮 motion run 缺 `motion_stats.json` 必须判 FAIL，不能走 SKIP**。`SKIP` 只对「非 motion 模型」
  或「复核旧 run 的 `check.json`」有效。实现：`check_shard.py` 读一个显式期望
  （`EXPECT_MOTION_STATS=1`，由 motion run 那 10 行 manifest 对应的 worker 传入），
  期望为真而文件缺失 → `MOTION_WINDOWS=FAIL reason=missing-motion-stats`。
- **降级分层统计只作描述，不作因果验收**。「预期两层成功率无显著差异」**不成立**：
  触发降级需要集活过 `16(162−k_demo)` 步，而早早成功的集天然进不了触发组——两组本来就不可比，
  这是选择偏倚，不是处理效应。留档里照常给出两层的 n 与成功率，但措辞限于描述；
  想真的判降级有没有害，只能在同一批集上跑 `raise` 与 `resample` 两次做配对比较，
  而 `raise` 那侧会大量 error——**本轮不做，记为待办**。

### 留档

`docs/training-doc/binfilldemo-{nomotion50k,motion50k,modul32frame-50k}-gl/{launch.md,result.md,records/}`，
主结果表放 `binfilldemo-nomotion50k-gl/result.md`，另两个回链。内容：指令原话、根因的四条证据、
demo store 的 manifest 与 sha、planner 失败清单与剔除后的共同子集口径、
**六列对照**（三个模型 × 有/无 demo）的分难度表、motion 降级统计、
以及必须写明的两条坦白——(1) 补 demo 同时改变了帧路（`even_sampling_indices` 现在要在 `[0, es+count]` 上取，
对三个模型都成立，量级远大于 motion 降级），BinFill 新旧结果是「两种输入条件」而非「同条件下的改进」；
(2) `t−es > 1136` 之后 exec 窗数就超过训练全集上限 70，这对现有 550 条旧结果同样成立，
降级只是把这个已存在的偏离又推远一点，别当成降级引入的新问题。
`docs/training-doc/README.md` 加三行索引。
