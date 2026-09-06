# eval-medium-patternlock-routestick —— 结果

## 结论

官方两个变体在 **PatternLock / RouteStick 的 medium 档**（val + test 两 split，每组 48 集，单 seed 42）：

| 变体 | PatternLock-test | PatternLock-val | RouteStick-test | RouteStick-val | 合计 | 成功率 |
|---|---|---|---|---|---|---|
| `perceptual-framesamp-context` | 0/12 | 0/12 | 0/12 | 2/12 | **2/48** | **4.17%** |
| `perceptual-framesamp-modul` | 7/12 | 6/12 | 6/12 | 4/12 | **23/48** | **47.92%** |

组间 Fisher 精确检验单尾 **p = 4.9 × 10⁻⁷**，差 43.75pp。

## 与 hard 轮合并：完整四格矩阵

配合 `../eval-hard-patternlock-routestick/`（同口径、同 ckpt、同 seed，仅难度档不同）：

| 变体 | hard | medium |
|---|---|---|
| `framesamp-context` | 0/48 = **0.00%** | 2/48 = **4.17%** |
| `framesamp-modul` | 8/48 = **16.67%** | 23/48 = **47.92%** |
| 组间绝对差 | 16.67 pp（p = 0.0028） | 43.75 pp（p = 4.9e-07） |

### 关键发现：难度效应把两种解释分开了

组间差异在两档都显著，但更有信息量的是**各变体自身随难度的变化**：

| 变体 | hard → medium | Fisher 单尾 p | 难度效应 |
|---|---|---|---|
| `framesamp-modul` | 16.67% → 47.92% | **0.00099** | **显著** |
| `framesamp-context` | 0.00% → 4.17% | 0.247 | **不显著** |

modul 在难度降低时明显做得更好——这是一个真正在解任务的模型应有的表现。context 则几乎没有
改善：把难度从 hard 降到 medium，它仍停留在 4.17%（且 2 集成功全部集中在 RouteStick-val 一个 unit，
另外三个 unit 依旧是 0/12）。

**因此结论不是「context 只在 hard 档塌陷」，而是 context 在这两个任务上基本不具备能力**——
连 medium 都做不了，降低难度也救不回来。hard 轮结束时提出的那个解读歧义（0% 是「没学过」还是
「太难」）至此完全消解：modul 证明任务在官方训练中被覆盖，而 context 的难度曲线是平的。

## 判据

两组各自独立汇总，四条判据全部 PASS。

context 组（`records/context/summary.txt`）：

```
HARD_EVAL=DONE units=4 episodes=48/48 PatternLock-test=0/12 PatternLock-val=0/12 RouteStick-test=0/12 RouteStick-val=2/12 mean_rate=0.0417
DIFFICULTY_ONLY=PASS expect=medium checked=48 mismatched=0 no_video=0
EP_SET=PASS units=4 expect=12集
SPLIT_SEED=PASS 日志自证 4/4 相符; 跨 split seed 比对 2 对/100 集全不同; 同名视频 22 个(观察项)
```

modul 组（`records/modul/summary.txt`）：

```
HARD_EVAL=DONE units=4 episodes=48/48 PatternLock-test=7/12 PatternLock-val=6/12 RouteStick-test=6/12 RouteStick-val=4/12 mean_rate=0.4792
DIFFICULTY_ONLY=PASS expect=medium checked=48 mismatched=0 no_video=0
EP_SET=PASS units=4 expect=12集
SPLIT_SEED=PASS 日志自证 4/4 相符; 跨 split seed 比对 2 对/100 集全不同; 同名视频 13 个(观察项)
```

`DIFFICULTY_ONLY=PASS expect=medium checked=48 mismatched=0` 表示 96 集的难度回读值
（`env.unwrapped.difficulty`，经视频名）全部为 `medium`，集号恒为 `[2,6,10,…,46]`。

「同名视频」观察项随成功率上升而下降（context 22 个 → modul 13 个），因为成功集的 flag 变为
`success`、不再与另一 split 撞名——再次印证把该层降为观察项而非硬判据是对的。

modul 权重的参数树核对见 `../eval-hard-patternlock-routestick/records/modul/param_tree.json`
（`PARAM_TREE_EXACT=PASS n_model=61 n_ckpt=61 missing=0 extra=0`），本轮用的是同一份权重。

## 资源与耗时

| 项 | context 组 | modul 组 |
|---|---|---|
| 起跑 commit | `643133b` | `4a1574e` |
| 起跑时刻 | 15:41:39 | 15:50:53 |
| 分片 | `MODE=stride WORKERS=4 ONLY=w2`，4 单元并行 | 同左 |
| 端口 | 9302 / 9312 / 9322 / 9332 | 9342 / 9352 / 9362 / 9372 |
| `infer` 次数 | PL 38/45、RS 66/88 | PL 77/65、RS 138/127 |
| `infer` 中位 | 70–72 ms | 70–71 ms |

两组必须串行：每个 policy 约占 18.6 GB，2 × 46 GB 的卡最多容纳 4 个 worker。
全部 8 个 tmux 会话自然退出，未执行任何 `kill-session`。

窗数上 modul 依然全面高于 context（如 RouteStick-test 138 vs 66），与其更高成功率方向一致。

## 局限

1. **单 seed 42**。48 集基数下噪声不小，但两档的组间差距（p = 0.0028 与 4.9e-07）都远大于该量级。
2. **easy 档未测**。若要完整的三档曲线，可补 `--expect-difficulty easy`（脚本本轮已支持）；
   easy 每任务 26 集，集号为 `episode % 4 ∈ {0,1}`，现有 stride 分片无法一次取全（`ONLY=w0`
   只得 ep 0,4,…,48 共 13 集，`ONLY=w1` 得另 13 集），需两片合并。
3. **两变体是各自独立训练的 checkpoint**，差异不能单纯归因于 `integration_type` 这一处结构差别。
4. 结论仅限 PatternLock / RouteStick 两个 Imitation suite 任务，不推广到其余 14 个任务。
   既有 `eval-3seed-context-vs-motion` 显示 context 在 Permanence 四任务全难度混跑下是 24.5%。

## 与仓库既有选择的关系

modul 是**上游脚本的默认变体**（`scripts/training/eval.sh` 的 `MODEL_TYPE`），context 是本仓库
特意锁死的（`scripts/training/g0/README.md` 明载「官方脚本默认 modul → 本基准锁死 context」）。
本轮与 hard 轮合并显示：在这两个 Imitation suite 任务上，无论 hard 还是 medium，该锁定选择都是
明显不利的——medium 档相差 11.5 倍。
