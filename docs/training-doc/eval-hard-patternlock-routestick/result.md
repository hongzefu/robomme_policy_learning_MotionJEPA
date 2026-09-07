# eval-hard-patternlock-routestick —— 结果

## 结论

官方两个变体在 **PatternLock / RouteStick 的 hard 档**上（val + test 两 split，每组 48 集，单 seed 42）：

| 变体 | PatternLock-test | PatternLock-val | RouteStick-test | RouteStick-val | 合计 | 成功率 |
|---|---|---|---|---|---|---|
| `perceptual-framesamp-context` | 0/12 | 0/12 | 0/12 | 0/12 | **0/48** | **0.00%** |
| `perceptual-framesamp-modul` | 2/12 | 2/12 | 2/12 | 2/12 | **8/48** | **16.67%** |

**modul 显著优于 context**：Fisher 精确检验单尾 **p = 0.00285**（即若两者真实成功率相同，
context 组 48 集一个都不成功的概率仅 0.29%）。这不是噪声。

modul 的 16.67% 是既有四任务（Permanence suite）hard 档合计 4/48 = 8.33% 的 **2.0 倍**；
context 的 0/48 则低于该基率，其在 8.33% 基率下出现的概率为 1.54%。

一个值得注意的巧合：modul 四个 unit **全部恰好 2/12**。若真实率为 1/6，四个 unit 都恰好落在 2
的概率约 0.77%。val 与 test 是两批完全独立的环境实例（seed 零交集），却给出同一数字。

## 判据

两组各自独立汇总，四条判据全部 PASS。

context 组（`records/context/summary.txt`）：

```
HARD_EVAL=DONE units=4 episodes=48/48 PatternLock-test=0/12 PatternLock-val=0/12 RouteStick-test=0/12 RouteStick-val=0/12 mean_rate=0.0000
HARD_ONLY=PASS checked=48 non_hard=0 no_video=0
EP_SET=PASS units=4 expect=12集
SPLIT_SEED=PASS 日志自证 4/4 相符; 跨 split seed 比对 2 对/100 集全不同; 同名视频 24 个(观察项)
```

modul 组（`records/modul/summary.txt`）：

```
HARD_EVAL=DONE units=4 episodes=48/48 PatternLock-test=2/12 PatternLock-val=2/12 RouteStick-test=2/12 RouteStick-val=2/12 mean_rate=0.1667
HARD_ONLY=PASS checked=48 non_hard=0 no_video=0
EP_SET=PASS units=4 expect=12集
SPLIT_SEED=PASS 日志自证 4/4 相符; 跨 split seed 比对 2 对/100 集全不同; 同名视频 20 个(观察项)
```

modul 另有参数树核对（`records/modul/param_tree.json`）：

```
PARAM_TREE_EXACT=PASS config=mme_vla_suite history_config=perceptual-framesamp-modul.yaml
  yaml_sha256=823c3948e75a9335 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
```

modul 是 61 个参数、context 是 55 个（后者见 `eval-official-framesamp-context/records/param_tree.json`）
——两变体结构本就不同（`integration_type` 不同）。两者都走 `create_trained_policy` 的 legacy 分支
（`strict=False`、`remove_extra_params=True`，权重对不上会**静默裁剪**），故必须核参数树；
modul 核对显示零裁剪，结果可信。

### 环境 split 确实生效

`SPLIT_SEED` 的支撑数据（`records/*/per_episode.json` 实读）：

| 任务 | split | seed 范围 | 逐集与另一 split 相同的集数 |
|---|---|---|---|
| PatternLock | test | 650300–654700 | 0 |
| PatternLock | val | 1150300–1154700 | 0 |
| RouteStick | test | 660300–664700 | 0 |
| RouteStick | val | 1160300–1164700 | 0 |

seed 严格落在 `BASE + 100 × episode` 上（ep3 → 650000+300 = 650300）。集号恒为
`[3,7,11,…,47]`，难度回读值（`env.unwrapped.difficulty`，经视频名）48/48 全为 `hard`。

## 逐集明细

| 任务 | split | context 成功集 | modul 成功集 |
|---|---|---|---|
| PatternLock | test | 无 | ep 11, 43 |
| PatternLock | val | 无 | ep 11, 27 |
| RouteStick | test | 无 | ep 27, 31 |
| RouteStick | val | 无 | ep 19, 31 |

flag 分布：context `{fail: 48}`（零 success、零 timeout）；modul `{fail: 40, success: 8}`。
**modul 成功的 8 集，context 无一例外全部失败。**

窗数（`infer` 次数，反映 episode 在被判失败前撑了多久）：

| unit | context | modul |
|---|---|---|
| PatternLock-test | 31 | 62 |
| PatternLock-val | 39 | 55 |
| RouteStick-test | 73 | 102 |
| RouteStick-val | 76 | 115 |

modul 在每个 unit 上都撑了约 1.4–2.0 倍的窗数，与其更高的成功率方向一致。

## 解读与局限

1. **本轮回答了「模型是否学过这两个任务」**。此前 context 的 0/48 存在解读歧义——可能是官方
   ckpt 压根没在 PatternLock/RouteStick 上训练过（其 `norm_stats.json` 只有全局 8 维统计、不分任务，
   无法从 ckpt 内部证实覆盖）。modul 在同样 48 集上拿到 16.67%，**证明这两个任务在官方训练里是被
   覆盖的**，因此 context 的 0% 是该变体自身在这两个任务上的表现，不是"没见过"。

2. **不能推广到 hard 档以外**。本轮只评 hard 12 集/组。context 在 easy/medium 上的表现未测，
   两变体的总体优劣不能由本轮断言。既有 `eval-3seed-context-vs-motion` 显示 context 在四任务
   全难度混跑下是 24.5%，与本轮不可直接比较（任务不同、难度档不同）。

3. **单 seed**。policy 采样 seed 只跑了 42。既有留档实测 seed 噪声在 200 集基数下约 0.5–1.3pp；
   本轮基数仅 48 集，噪声更大。但 8/48 vs 0/48 的差距（p=0.00285）远大于该量级的噪声。

4. **两个变体是不同的 checkpoint，不只是不同配置**。都是官方发布的 79999，但各自独立训练，
   故差异不能单纯归因于 `integration_type` 这一处结构差别。

5. modul 是**上游脚本的默认变体**（`scripts/training/eval.sh` 的 `MODEL_TYPE`），context 是本仓库
   特意锁死的（`scripts/training/g0/README.md` 明载「官方脚本默认 modul → 本基准锁死 context」）。
   本轮结果说明：在这两个 Imitation suite 任务的 hard 档上，该锁定选择是不利的。

## 资源与耗时

| 项 | context 组 | modul 组 |
|---|---|---|
| 起跑 commit | `ff94539` | `bb49c2a` |
| 起跑时刻 | 14:14:08 | 14:27:38 |
| 墙钟 | 约 6 分钟（4 单元并行） | 约 8 分钟（4 单元并行） |
| GPU | 2 × RTX 6000 Ada，每卡 2 worker | 同左 |
| 显存 | GPU0 39614 / GPU1 38556 MiB | GPU0 39892 / GPU1 38569 MiB |
| `infer` 中位 | 70 ms | 70–71 ms |

modul 权重：HF 公开仓 `Yinpei/perceptual-framesamp-modul` @ `c0f565dd…`，`79999.zip`
11,878,950,895 B，sha256 `2bfde48a…faa62` 与 HF LFS 一致；下载 257 s、解压 40 s。

## 计划外的意外

### 一、modul 组首次起跑被 clean-HEAD 门禁拦下
context 跑完后写了 `launch.md` / `download_modul.sh` 与 records 却未提交，工作区转脏，
4 个 modul 单元全部被 `eval_shard.sh` 的 clean-HEAD 门禁挡下并干净退出（无残留会话、无半截产物）。
提交留档（`bb49c2a`）后重起即通过。门禁行为符合预期，未作修改。

### 二、预检窗数异常的排查
PatternLock 预检每集仅 2 次 `infer`（32 步即终止，而 `max_steps=1300`）。起 1 集 ButtonUnmask 对照：
ButtonUnmask `infer` 14 次且 `首批>16帧 n=0`（无 demo 前缀）；PatternLock 各 2 次且 `frames=199/93`
（有 demo 前缀）。demo 前缀的有无恰好对应 `utils.py::TASK_WITH_VIDEO_DEMO` 的成员关系，交叉印证链路正常。
2 窗是 PatternLock 的任务特性——描错一笔即判失败、环境主动终止（走 `stop_flag` 路径而非 `timeout` 路径）。
对照集 ButtonUnmask ep3 亦为 fail，与既有留档的「ButtonUnmask hard 0/12」一致。

### 三、`SPLIT_SEED` 第 3 层证据降级为观察项的决策得到验证
「同名视频」在 context 组为 24 个（= 2 任务 × 12 集，全部撞名）、modul 组为 20 个。原因是视频名格式
`<task>_ep<k>_<flag>_<goal>_<difficulty>.mp4`——`goal` 是固定模板文案，当两组同一集的 `flag` 也相同时
名字必然相撞。若把该层当硬判据，context 组会误报 FAIL。真正有力的第 2 层（跨 split 逐集 seed 比对）
已充分证明两组是不同环境实例。

### 四、上游 benchmark 的一处缺陷（本轮不受影响）
`RouteStick.py` 的难度 fallback 中，`else` 分支算完 `seed % 3` 的三分支后，紧跟一行**未注释**的
`self.difficulty = "easy"`，无条件覆盖结果。任何直接 `gym.make("RouteStick", seed=…)` 而不传
`difficulty` 的用法都会静默退化成 easy。（`PatternLock.py` 同位置那行是注释掉的。）本轮走
`BenchmarkEnvBuilder` 路径、元数据总会显式传 `difficulty`，且 `HARD_ONLY=PASS`（48/48 回读为 hard）
已实测覆盖该风险。

## 未决

- **是否补 easy/medium 档对照**：本轮只测 hard。若要判断两变体的总体优劣、以及 context 的 0% 是否
  在低难度档同样成立，需补跑 easy（`EP_START=0 EP_STRIDE=4`，每任务 13 集）与 medium
  （`EP_START=2 EP_STRIDE=4`，每任务 12 集）。已向用户提出，待决定。
- **是否补多 seed**：本轮单 seed 42。若需给出 ±std，可照 `eval-3seed-context-vs-motion` 体例补 7 / 2024。
