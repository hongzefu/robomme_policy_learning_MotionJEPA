# binfilldemo-* 起跑留档（BinFill 补 demo 前缀重测，三个 run 共用）

结果见 [`result.md`](result.md)。计划见根目录 [`0922-binfill-demo-prefix-plan.md`](../../../0922-binfill-demo-prefix-plan.md)。

## 用户指令原话（按时间）

1. 「参考 https://github.com/hongzefu/robomme_policy_learning_MotionJEPA/tree/v2-motionmem 训练集合的构成
   binfill需要增加demo / 还有什么训练/推理不一致的 先对抗验证 / 然后重新测50k 512 512+motion 2048的binfill」
2. AskUserQuestion 三项拍板：demo 来源 =「同 seed/spec 现跑专家轨迹」；重测范围 =「只补 demo，3 模型 × 150 条」；
   第三问用户改写为自定义要求：「motion budget这个问题 仍然2000步 超出160给出继续运行方案」
3. 「把计划固化在根目录 不执行」（计划落 `0922-binfill-demo-prefix-plan.md`，commit `3e9fc30`）
4. 「开始执行」
5. 中途追问「现在做到哪一步了 还有什么没做」
6. SigLIP 分批裁决：「不开分批，先验最大 D 的集（推荐）」
7. planner 失败集裁决：「跑 142 条，对照同子集重算（推荐）」
8. 澄清「我要的是根据142条planner能成功的算成功率结果」

## 为什么要补 demo

BinFill 是四个评测任务里唯一「训练有 demo 段、评测没有」的：

- 训练侧 `generate_dataset_newseed.py::_binfill_demo_deliverable`（2026-09-11 路线 B）把整条成功轨迹
  复制一遍接在自己前面，400/400 满足 `num_timesteps == 2 × exec_start_idx`；
- 评测侧 `BinFill.py::_initialize_episode` 的 `task_list` 里 `demonstration` 三处全 `False`，
  `reset()` 只返回 1 帧，`exec_start_idx = 0`。

本轮又加了一条实测佐证：三个已完成 run 的 `client.log` 各 691 行 `exec_start_idx` + 9 条 error = 700 条计划集数，
其中**恰有 150 行为 0、全部是 BinFill**。

## 为什么不能直接翻 benchmark 的 `demonstration` 标志

方块入 bin 是 `is_any_obj_dropped_onto_delete` 里的物理删除（`set_pose(p=[10,10,0])`），
`*_cubes_in_bin` 单调累加且无复位路径；demo 跑完后 `sequential_task_check` 的任务指针已走到
"All tasks completed"，策略第一步就白拿 success。BinFill 也没有 VideoRepick/RouteStick 那种
`"NO RECORD"` + `solve_strong_reset` 收尾任务，spawn 余量更不够演两遍（easy spawn 5 / target 3）。

**绕开的办法**：同一条候选（同 seed、同 spec）起**两个** env——预生成那个把 `task_list` 的
`demonstration` 全翻成 True 让 planner 演完整条、只取画面，演完即销毁；正式评测再起一个全新的
干净 env（`demonstration` 仍为 False）交给策略。monkey-patch 只在主仓库的预生成脚本里做，
**不动 benchmark 源码、不换 gitlink**（仍是 `b4e97f22fe007078e297205898c07c1acbc69165`）。

## 被评权重与口径

| run_name | CKPT | POLICY | POLICY_CONFIG |
|---|---|---|---|
| `binfilldemo-nomotion50k-gl` | `robomme-vla-modul-60k-v1/50000` | `perceptual-framesamp-modul-8frame-8x8` | `mme_vla_suite` |
| `binfilldemo-motion50k-gl` | `robomme-vla-modul-motion-80k-v1/50000` | `perceptual-framesamp-modul-8frame-8x8-motion` | `mme_vla_suite_b128_80k` |
| `binfilldemo-modul32frame-50k-gl` | `robomme-vla-modul-2048-80k-v1/50000` | `perceptual-framesamp-modul-32frame-8x8` | `mme_vla_suite_b128_80k` |

共用口径与前三轮 700 条**逐字相同**：seed 7、`max_steps=2000`、`EPISODE_WALL_S=2400`、
`EVAL_TIMEOUT=14400`、1 卡 / 1 CPU / 24G、benchmark gitlink `b4e97f2`、候选库 identity
`3b4de03a0b4639181f9c68086aea11e9b51ffcb65bbed701a44f573e74e0beb7`。

本轮新增：`DEMO_PREFIX_STORE`（前缀库根）、`CHUNK_EPISODES=15`；motion 那条另加
`MMEVLA_MOTION_OVERFLOW=resample` 与 `EXPECT_MOTION_STATS=1`（由 `gl_hold_pool.sh` 按 policy 名是否以
`-motion` 结尾自动分叉）。**`MMEVLA_ENC_CHUNK` 不设**（= 0 = 不分批，与改动前逐字等价）。

起跑 HEAD：`215279e83a89b4e5aae04a0f1005013ef144e8b7`（clean）。

## demo 前缀库

落点 `v1-store/demo-prefix/binfill-3b4de03a0b46/`（11.2 GB，不进 git）。
front 帧走 PNG **无损**（同时进 SigLIP 与 Wan VAE），wrist 走 JPEG q90（只进 mp4）；
h5 attrs 存 `front_raw_sha256`，读取端逐位复核。

四道正确性校验（任一不成立即判 planner 失败）：`status == "success"`、各色入 bin 数 == `target_count`、
入 bin 总数 == `put_in_total`、**`int(base.timestep) == len(base.task_list)`**、`episode_success`。

**按钮那一项绝不能写成瞬时 `is_button_pressed(base, obj=base.button)`**：`solve_button` 按下后会抬手、
弹簧回位，而成功状态早已在 `sequential_task_check` 里锁存。本轮实测复现了这个陷阱——
`easy/156` 的 `button_pressed_final=False` 与 `status=success` + 计数正确 + `episode_success=True`
**同时成立**，按瞬时状态判会把这条完全正常的样本当 planner 失败剔除。正确的锁存量是任务指针。
（这条是 Codex 静态审计 P1-1 指出的，本轮由实测坐实。）

两道硬闸取交集：pos 表闸 `D ≤ 2095`（`FrameSampMemory.__init__` 的 `max_steps=4096`）、
demo 窗闸 `k_demo(D) ≤ 80`（等价 `D ≤ 1296`）。
计划正文写的 `D ≤ 1281` 是个算错的保守值——`k_demo(D) = (D−17)//16 + 1 ≤ 80` 的充要界是 **1296**，
代码改用充要判据本身。实测 `max_D=1107 / max_k_demo=69`，两种写法都够不着。

## 作业与执行

8 个既有 gpu-hold 作业（`61728418`–`61728421`、`61730613`–`61730616`，1×A40 / 1 CPU / 24G）
动态抢单，不提交新 job。

1. **预生成**：`gl_demo_pool.sh` 抢 10 片 × 15 集。全部 10 片零失败退出，
   `PREGEN_OK episodes=150 ok=142 failed=8 used_rrt=6 D_min=261 D_mean=631.8 D_max=1107`。约 9 分钟。
2. **计划**：`make_shard_plan.py task --demo-store <库根> --shards 10` 只收 `demo_status == "ok"` 的集，
   `PLAN_OK task=BinFill shards=10 total=142 by_difficulty={'easy': 49, 'hard': 44, 'medium': 49}`。
3. **重测**：30 行 manifest（3 run × 10 片）走 `gl_hold_pool.sh` 的 MANIFEST 模式，8 worker 抢单。
   **30/30 单元 `rc=0`**、30 个 `SHARD_PASS`、3 个 `MERGE_OK episodes=142 errors=0`。约 1 小时 45 分钟墙钟。

## 起跑前闸（纯 CPU）

```
MOTION_DEFAULT_BITEXACT=PASS base=2cf4747 cases=22 comparisons=2772
                            breakdown={"digest": 2627, "RAISE:budget": 145}
MOTION_RESAMPLE_ACTIVE=PASS changed_infers=145
MOTION_HEADROOM=PASS        es_max=502 k_demo=31 k_exec=124 k_max=155 budget=160
ONLINE_GATES=PASS           {'p1': True, 'p2': True, 'p3': True, 'p4': True}
SOURCES_PASS                benchmark=b4e97f22fe007078e297205898c07c1acbc69165
```

`MOTION_DEFAULT_BITEXACT` 取基线 commit `2cf4747` 的 `framesamp_memory.py` 原文动态加载成第二个模块，
22 个 es 档 × 126 次推理逐叶 sha256 比对：2627 组逐位相同、145 组两边都因超预算 raise，零其它异常。
`MOTION_RESAMPLE_ACTIVE` 的 `changed_infers=145` 与之精确对上——**降级只动了原本会让整集判 error 的推理，
其余 2627 次一字未动**。

`MOTION_HEADROOM` 不用 `grep … | tail -1` 求上界（那只是「已有日志里的最大值」）。
改为覆盖核对 + 实测最大值：三个旧 run 各 691 行 `exec_start_idx` + 9 条 error = 700，
缺口恰好是三者完全相同的 9 条 reset 卡死集。非 BinFill 的 541 条实测 `max(es)=502`
⇒ `k_max = 31 + 124 = 155 < 160`，**降级分支在旧 550 条上不可达，沿用安全**。

`eval_rhythm_gates.py` 在本工作副本跑不了（依赖训练库 `v1-store/datasets/4task-motion-400ep`，
只在本机 `/data` 那份副本里），本轮未改动该文件，计划把它列为起跑前闸是疏漏。

## 冒烟与专项验证

| 判定行 | 结果 |
|---|---|
| `DEMO_OK easy/156` | D=270 k_demo=16 status=success **button_final=False** used_rrt=False 5.63 s |
| `SMOKE_RESET easy/156` | `frame0_vs_reset: mean_abs=0.0 max_abs=0` |
| `SMOKE_INJECT easy/156` | es=270 首批 271 帧 28.0 s，帧路键 0..270、demo 窗 16、`_raw_frames=[270]`、`mem_order=672` |
| `ENC_CHUNK_CMP` | 全局 rel_fro 0.090%、cos_min 0.9999917、逐帧最大 0.41%（基线 0.33% / 0.99998） |
| `SMOKE_RESET hard/202` | `max_abs=0`（最大 D 的集，第二次确认逐位相同） |
| `SMOKE_INJECT hard/202` | D=1107 首批 **1108 帧 179.9 s 不 OOM**；喂 95 批后降级 3 次、首次 t_rel=1488、`kept_demo=69` 全保 |

`hard/202` 那条同时回答了两件事：最坏首批不 OOM（支撑「不开分批」的决定），
以及**端到端降级在真实链路上触发且与闭式预测一次不差**。

## 意外与处置

1. **计划的 `D ≤ 1281` 算错**，充要界是 1296；改用充要判据 `k_demo(D) ≤ 80` 本身。
2. **计划的 `k_exec(j) = max(0, j−1)` 差 1**，应为 `max(0, j−2)`（第 j 次推理时 `t_rel = 16(j−1)`）。
   由它推出的触发步数（1456 / 1968）反而是对的，只有推理序号差一。代码与验证脚本统一用 `t_rel` 闭式。
3. **`h5py` / `opencv-python` 原先只是传递依赖**，预生成脚本与 `DemoPrefixStore` 要直接 import，
   按 uv 规则显式写进 `scripts/evaluation/client/pyproject.toml` 并 `uv lock`——lock 只增加 4 行，
   零版本变动、零新包，解析图不变。
4. **冒烟脚本四次返工**（链路本体一次没错）：单进程直验撞上 `robomme` 与 `mme_vla_suite` 分属互斥 venv →
   拆成 `export-reset`/`verify`；漏了 `finalize` 导致缺 `index.json` → 加 `--partial`；
   误判「失败在合成 exec 批」→ 实际 stub sidecar 靠 `stub_decode(window)` 从**像素**解帧编号、
   只认合成的带编号帧，**喂任何真实渲染帧都必报错**，改走真 sidecar。
5. **`enc_chunk_cmp` 指标两次选错**：先用逐元素相对差（理论上界就是 2，bf16 里大量接近 0 的元素让它恒等于 2），
   再用逐帧最大 rel_fro 去比全局 rel_fro 的基线。改为与 `docs/train-infer-consistency.md` 第七节 1
   同口径的全局 rel_fro + 逐帧余弦最小。
6. **预生成中途 commit 险些卡住 s8/s9**：`run_demo_shard.sh` 每次抢到新分片都会检查 clean HEAD，
   改 `make_shard_plan.py` 后工作区变脏，立即提交（`215279e`）才没影响后两片。
   **评测/预生成期间不得留脏工作区**，`gl_eval_shard.sbatch` 另有 `EXPECTED_GIT_HEAD` 比对，
   期间任何 commit 都会让后续单元拒绝起跑。
7. **`make_shard_plan task` 的整除断言过严**：剔除 8 条后 142 不是 10 的倍数，计划出不来；
   `rows[i::shards]` 本就保证各片最多差 1 条，断言改为只要求条数不少于分片数。
8. **失败率 5.3% 超 5% 闸**，按计划停下来交用户裁决，用户决定按 142 条跑、对照同子集重算。

## 产物

`records/` 下：`manifest.tsv`、`demo-prefix-{manifest,index}.json`、`enc_chunk_cmp.json`、
`merged/{log,progress}.json`；另两个 run 的 `merged/` 在各自目录，motion 那条另有
`merged/motion_stats.json` 与 `maxd-inject-hard202.json`。

编排脚本（不进 git）：`v1-store/demo-prefix/{pool-run,make-eval-plans,eval-pool-run,maxd-run,smoke-*}.sh`。
