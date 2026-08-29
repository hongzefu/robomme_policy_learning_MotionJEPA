# v3 破坏性单一化重构计划：训练链只留 packed framesamp，建库域自包含隔离，scripts 目录统一

> **本文件是自包含的单一权威文档**（2026-08-29 定稿）：执行本计划的人和 agent 不需要再阅读其他计划 md——决策依据、commit 切片、逐文件改动、对拍闸门、G3 runbook、链路图要点、风险登记全部内联。历史沿革见 git 历史与会话记录。
>
> **锚点**：分支 `v1-dataloader-Restructure`，现 HEAD `732fae3b13e2ff5f485d7014473b99ed577de387`（初稿制定时为 `0ce75be`；其后五个提交 `f2eb900`/`d42b3cc`/`a898299`/`6d0d576`/`732fae3` **全是本文档自身的 docs 提交、代码零变更**，故初稿的全部源码事实描述在本锚点上仍然有效），工作区 clean。commit 编号沿用 V3.x 序列（V3.7 之后），计划占用 **V3.8–V3.14**。
>
> **本稿是 2026-08-29 对抗审计后的修订版**：经一轮 17-agent 对抗验证（8 维事实核对 + 8 维对抗证伪 + 综合裁决，锚定 `732fae3b`）与一轮 Codex 独立审计交叉核对，原稿查出 9 项会让执行当场失败的缺陷与十余项事实错误，已全部落进本稿。两轮审计一致认定：**原稿的 bitwise 红线本身没有缺陷**——所有 blocker 都是「跑不起来」，不是「跑出错数」。修订触及范围的七项决策由用户当场逐条拍板，见二节后半「第二轮拍板」。
>
> **唯一不可商量的红线**：收敛后训练侧交付给模型的字节与现在逐位一致，由正确性族 G0 对拍（G3 节点）终局裁决。

---

# 第一部分（给人看）

## 一、Context（为什么做这件事）

v4（IO 重构，commitV3.0–V3.7）交付 `FrameSampDataset` 后，训练主链已切到 packed 库；但仓库仍保留 legacy 数据链（`RoboMMEDataset` + 散 npy + `MemoryBuffer`）与 v1 范围外的 recurrent / symbolic / token_dropping 三类分支，另有七个历史专题脚本目录散落。本轮把训练链路收敛为「一条数据链、一种数据形制」，建库工具彻底搬出训练代码射程，脚本目录统一为训练 / 数据集预处理两个域。

## 二、用户已拍板的决策（2026-08-29 会话逐条确认）

1. **破坏性重构**：删除 legacy（非 packed）数据链，packed `FrameSampDataset` 成唯一训练数据路径。
2. 数据格式只保留 perceptual **frame_sampling**；模型侧删 recurrent / symbolic / tokendrop 分支，**保留 context / modulation / expert 三种集成**（challenge_interface 用 framesamp-modul；modul/expert 只保在线评估/部署，`FrameSampDataset` 的 `integration_type=="context"` 形制断言不放宽）。
3. **建库链产物与原先完整一致，并与训练代码彻底隔离**（建库域自包含副本，训练/在线侧不再 import `mem_buffer`）。
4. **对拍正确性族 G0**：收尾从 clean HEAD 跑 G3（packed 1000 步确定性档）离线对拍 G0 固化产物 bitwise。
5. **scripts/ 统一**为 `scripts/training/` + `scripts/dataset/` 两个顶层文件夹；`train.py` 与评估入口全部迁入；历史专题目录（bottleneck-bench、bottleneck-bench-v2、dtype-unify、compare_batches）退役删除（活量具先迁走）。

### 第二轮拍板（2026-08-29 对抗审计后，逐条确认）

审计暴露出七处原稿没有决定权限、必须用户拍板的岔路，逐条落定如下。每条后括注对应的审计发现。

6. **`pi05_baseline` 训练入口整删**。原稿第 1 条要求 packed 成唯一数据路径，但 `pi05_baseline` 是 `use_history=False, history_config=None`（`training/config.py` 的 `_CONFIGS` 首项），而 `FrameSampDataset.__init__` 的第一条形制断言就是 `_req(hc is not None, "history_config 不能为 None")` —— 数据链一单一化，这个入口必死。决定：**删**，而不是为它保留 legacy 旁路。连带删 `scripts/finetune_pi05_baseline.sh`、`history_pi0.inputs_spec` 的外层 `use_history` 二分、`policies/policy.py` 的 `config is None` 分支。（原稿全文未登记此冲突；两轮审计各自独立查出。）
7. **建库域四个冻结副本豁免「新增注释必须中文」**。`shared/{data_utils,posemb_3d,siglip_tokenizer,mem_buffer}.py` 四个文件**含中文行数实测为 0**（全英文 docstring/注释），逐字节复制进 `dataset_builder/` 即「仓库新增全英文注释文件」，与 AGENTS 1 冲突；而翻译成中文会让 COPY_DIFF 与 sha256 哨兵同时失效。决定：**在本文件声明豁免**——冻结副本视同 `git mv` 产物，不适用「新增注释用中文」，逐字节同一性优先。
8. **建库输出路径加 `--force` 闸**（V3.8 内落地）。`DatasetProcessor.__init__` 在任何校验之前执行 `if os.path.exists(self.dataset_path): shutil.rmtree(self.dataset_path)`，而 `--preprocessed_data_path` 来自 CLI 无任何约束：误传 `v1-store/datasets` 或 `v1-store` 会删掉 678 GB 不可恢复数据。决定：**默认拒绝已存在目标，要删必须显式 `--force`**。（canonical containment 白名单本轮不做，见第 11 条。）
9. **失败的 record 目录不再删除**（V3.9 内落地）。`run_2gpu_epoch_bench.sh` 在非零退出时 `rm -rf -- "${RECORD_DIR}"`，而本计划六节要求 G3「首次 state 分叉立刻停跑」——停跑即非零退出，刚抓到的分叉证据当场被删，只能盲目重跑 2.5 小时。决定：改为原子改名 `${RECORD_DIR}.failed-<n>`。
10. **`runs/` → `v1-store/` 路径收敛 + 裸 `python` 换 `uv run`**（V3.13 内落地，只做路径与执行器，不动评估逻辑）。`compute_results.py` 的 `runs/evaluation`、`eval.sh` 的 `runs/ckpts` 与裸 `python`、`unzip_ckpt.py` 的默认 `runs/ckpts` 违反 AGENTS 14（单一 `v1-store/`）与 AGENTS 3（禁裸 python）。这些文件本轮本就要改（删 symbolic 菜单），顺路收敛。
11. **根解析用「改对数字 + 自证断言」，不引入统一解析器**。全仓十余处 `parents[N]` / `_HERE.parent.parent` / `../..` 都写死层数，V3.14 搬深一层后全错；其中 `data-preprocess-GL/paths.sh` 的失败是**静默的**（见 6.3）。决定：逐文件把层数改对，**并在每个入口加一句自证断言**（算出的根必须存在 `pyproject.toml`，否则立刻 fail-loud 退出），把所有静默失败变成响亮失败。
12. **N4（ONLINE_MEM）改在 clean HEAD 上用 `git show` 提取 A 侧**，不再要求脏工作区共存窗口，也不拆 `V3.12a/V3.12b`。原稿 384 行的「兜底 A 侧」升为**主流程**，据此解掉与 AGENTS 17（>5 分钟须从 clean HEAD 起跑）的冲突。
13. **本轮不动 `compare_baseline.py`**。该比较器有三处 fail-open（缺标量静默 `continue` 但仍打 `keys=5`、`INDEX_SEQ` 只比最短公共前缀、canonical 与 index 结果不进最终 `verdict` 与退出码），加上已知的 raw 失配使总行 `DET_CHECK` 恒为 FAIL —— G3 结论目前只能人工拼。决定：**量具不改**，改为把判读纪律逐条写死进六节与 G3 的 `launch.md`，并把三处 fail-open 登记进八节盲区。
14. **`run_name` 的字符校验与 `rm -rf` containment 本轮不做**（明确不处置，登记为 R21）。`EXP_NAME`/`RUN_TAG` 未经校验即参与路径拼接，清理处的 `case` 模式与被匹配值由同一套变量展开、属词法自比，挡不住含 `../` 的取值。经评估风险概率低（run_name 由用户手输且每次经确认），本轮不加护栏。

### 两处对已确认退役清单的事实修正（探查后发现，随本计划一并生效）

- **`dump_index_seq.py` 不退役**：它不 import `RoboMMEDataset`，其 `--legacy-root` 实指源 pkl 库（本轮不删）。修正为保留，迁入 `scripts/training/tests/`，仅改帮助文本措辞。
- **dtype-unify 的活工具清单从 4 件扩为 6 件**：`dump_fixture_samples.py`（唯一的逐样本/逐 batch 逐键落盘量具）与 `compare_dtype_fix.py`（`COMPARE_GRAD=` 判定行出口）也保留迁走；目录本身仍整删。

## 三、方案总览

**七个 commit + G3，顺序不能换：**

| # | 提交 | 做什么 | 为什么排这个位置 |
|---|---|---|---|
| V3.8 | 建库域隔离 | 四个共用文件复制入建库域，建库脚本改 import 指向；**另加建库输出 `--force` 闸** | 训练侧零改动；先落地，后面删共用文件时建库链已安全。`--force` 闸同域顺路做，不碰任何计算路径 |
| V3.9 | 数据链单一化 | 删 legacy 读取器与 backend 三态；norm stats 脚本内联最小读取器；**删 `pi05_baseline` 训练入口**；**失败 record 改保留** | 数据入口先收口，后面改 transforms 才只有一个消费者。`pi05_baseline` 死于本刀（无 history 配置进不了 packed Dataset），故同刀删除，不留会炸的中间态；record 保留必须先于第一次烟测生效 |
| V3.10 | transforms 瘦身 | 删 recurrent/symbolic 的键与分支 | 必须在模型改动前：先让废字段断流，模型侧删除就成纯死代码清理 |
| V3.11 | 模型侧单一化 | 删 recurrent/symbolic 模型分支与 stats 返回链；**删 `inputs_spec` 的 `use_history` 二分** | 依赖 V3.10 已断流；二分的唯一依赖方 `pi05_baseline` 已在 V3.9 删除 |
| V3.12 | 在线评估改造 | 新写 `FrameSampMemory`，训练/在线脱离 mem_buffer；shared/ 重划 | 不影响训练数值，但动训练侧 import 路径，须在长跑前 |
| V3.13 | 脚本与示例清理 | 删 symbolic 启动菜单与 examples subgoal 子树；**`runs/` → `v1-store/` 收敛 + 裸 `python` 换 `uv run`** | 纯外围，不进训练链路；这些文件本刀本就要改，路径与执行器顺路收敛（不动评估逻辑） |
| V3.14 | 目录统一 | 幸存脚本搬入 `scripts/training/` + `scripts/dataset/`；**git mv + 逐文件根解析修正（不是「纯搬移」）** | 最后搬移，G3 在最终布局上验收整条链 |
| G3 | 正确性长跑 | 全部冻结后从 clean HEAD 跑 1000 步确定性档，对拍 G0 固化产物 bitwise | 一次覆盖七个提交 |

每个 commit 各有 ≤5 分钟的便宜验证（AGENTS 4），另设分段对拍闸门 N1–N5（第二部分五节）——真出问题不必在 1000 步长跑里回溯七个提交。**对拍 A 侧沿用 G0b 固化产物**（`docs/training-doc/v1-grad-baseline-g0b/records/r1/`），不在当前 HEAD 重录基线（G 链「跑一次固化」纪律；G2 已证当前链 bitwise ≡ G0b）。

**全局烟测口径（审计新增，四刀通用，漏一项即当场失败）**：凡本计划写「确定性档 `STEPS=5` 烟测」处，命令一律带齐

```
STEPS=5 SAVE_INTERVAL=5 BATCH_DIGESTS=1 WARMUP_STEPS=0 KEEP_JAX_CACHE=1 \
EXP_NAME=v1-restructure-smoke RUN_TAG=<每刀不同> \
DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp
```

三个必带项的由来：① `WARMUP_STEPS` 不给则取 `run_2gpu_epoch_bench.sh` 的默认 **50**，`STEPS=5` 时收尾统计走 `for s in range(warmup+1, steps)` 得空序列、`statistics.median([])` 抛 `StatisticsError`，脚本非零退出、`BENCH_PASS` 永远打不出——**训练 5 步全部跑完才崩在统计处**，症状与被改代码毫无关系，极易误判成「这一刀改坏了」；② `SAVE_INTERVAL=0` 会连锁置 `BATCH_DIGESTS=0` 与 `BENCH_CHECKSUM=0`，记录器不装、`index_sequence.json` 根本不产出，任何 `INDEX_SEQ` 判据只能打 `SKIP`；③ 驱动默认数据集是 legacy `GL_DATASET`，不显式给 `DATASET_PATH` 则 V3.9 之后的第一个烟测就用错库。

## 四、scripts/ 顶层前后对比（V3.14 完成后的目标形态）

修改前（现状 22 项）→ 修改后（2 个域文件夹），逐项去向：

| 现状条目 | 性质 | 去向 |
|---|---|---|
| `train.py` | 训练主入口 | → `training/train.py`（另修 `wandb.run.log_code` 的 `parent.parent`，见 7.2） |
| `finetune_mme_vla_suite.sh` | 训练启动脚本 | → `training/`（菜单在 V3.13 已清成三个 framesamp） |
| `finetune_pi05_baseline.sh` | 无 history baseline 启动脚本 | **V3.9 删除**（入口整删，二节第 6 条） |
| `eval.sh` / `serve_policy.py` / `compute_results.py` | 评估/部署入口 | → `training/`（symbolic 分支在 V3.13 已删；同刀做 `runs/` → `v1-store/` 收敛与裸 `python` → `uv run`） |
| `compute_norm_stats.py` | norm stats 生产（V3.9 已内联读取器） | → `training/` |
| `download_pi05_base.py` | 下载 pi05 初始权重 | → `training/` |
| `unzip_ckpt.py` | checkpoint 解压工具 | → `training/` |
| `__init__.py` | 包标记（一行注释） | → `training/__init__.py` 原样随迁 |
| `smoke-local/` | G 链量具（bench/preflight/compare 四件） | → `training/bench/`（另收 `compare_online_memory.py` 新量具） |
| `train-prod/` | 正式训练启动器 + 趋势分析 | → `training/prod/` |
| `dtype-unify/` | dtype 专题（历史验收） | 6 件活工具（`single_step_grad` / `dump_fixture_samples` / `compare_dtype_fix` / `_common` / `test_padding_dtype` / `run_dtype_grad.sh`）→ `training/tests/`，`analyze_util.py` → `training/bench/`；**目录整删** |
| `bottleneck-bench/` | 历史性能专题 | `gl-dataloader/` V3.9 已删；**余部整删** |
| `bottleneck-bench-v2/` | 历史性能专题 | `analyze_gpu_util.py`（活量具，prod sbatch 在用）→ `training/bench/`；**余部整删** |
| `data-pack-framesamp/` | 打包工具 + 训练侧守卫混装 | 打包件（`pack_framesamp_store.py` / `probe_layout.py` / `run_pack.sh` / README）→ `dataset/pack/`；守卫与量具（`test_pack_guards.py` / `spawn_matrix.py` / `dump_index_seq.py`）→ `training/tests/`；`compare_batches.py` V3.9 已删；**目录删** |
| `data-preprocess-GL/` | GL 建库域主体 | → `dataset/gl/` 整目录搬迁（含 `gl_submit.py`、sbatch、自己的 `paths.sh`、`legacy/`）。**「冻结随迁」的说法作废**：`paths.sh` 与 `build_shard.py` / `finalize_checks.py` / `compare_datasets.py` / `test_guards.py` 五处根解析必须随深度修正，`gl_build_dataset.sbatch` 与 `step1_submit.sh` 的自引用路径同改（详见 7.2） |
| `build_dataset.py` | 建库分派入口 | → `dataset/build_dataset.py`（V3.8 已加 `--force` 闸） |
| `tarxz_h5.py` | 原始 H5 压缩工具 | → `dataset/` |
| `unzip_data.py` | 数据解压工具 | → `dataset/` |
| `finetune_vlm_subgoal_predictor.sh` | symbolic 数据生产配套（vlm_subgoal builder 同域，冻结不改内容） | → `dataset/` |
| `__pycache__/` | 缓存 | 不进 git，忽略 |
| （新增） | 训练域自带 `paths.sh` | `training/paths.sh`（切断对建库域 paths.sh 的跨域 source） |

修改后顶层只剩两项：

```
scripts/
  training/   train.py、finetune_mme_vla_suite.sh、eval.sh、serve_policy.py、compute_results.py、
              compute_norm_stats.py、download_pi05_base.py、unzip_ckpt.py、__init__.py、paths.sh
              ├ prod/   ├ bench/   └ tests/
  dataset/    build_dataset.py、tarxz_h5.py、unzip_data.py、finetune_vlm_subgoal_predictor.sh
              ├ gl/     └ pack/
```

（`finetune_pi05_baseline.sh` 不在最终布局里——V3.9 已删。）

## 五、影响面结论

- 已建好的 packed 库与正式训练：零影响，不重建。
- challenge_interface（framesamp-modul）：零改动（三种集成保留，`policy_config`/serving 不动）。
- **`pi05_baseline` 训练入口：整删**（二节第 6 条）。该 TrainConfig、其启动脚本、`inputs_spec` 的 `use_history` 二分、`policy.py` 的 `config is None` 分支一并消失；此后仓库没有「无 history」的训练路径。想跑无 history 对照须从 git 历史取回配置并自备数据路由。
- 未来建库：建库域自包含冻结（**四个副本冻结，其余建库文件因根解析修正而有改动**），计算路径一字不变。
- 未来 norm stats：`compute_norm_stats.py` 内联最小 pkl 读取器，链路不断。**但本轮无新旧输出对拍**（G 链全程不执行该脚本、只消费既有 `norm_stats.json`），口径差要到下次重算才显形——已登记为八节盲区第 8 条。
- G 链量具（bench/preflight/compare）：判据不变；`run_2gpu_epoch_bench.sh` 有两处行为改动（V3.9 删 backend 两行 + 失败 record 改保留），`check_baseline_env.py`/`bench_train_steps.py` 等在 V3.14 有根解析修正；历史留档 `docs/training-doc/*/launch.md` 一律不改。
- 评估与 checkpoint 落点：`runs/evaluation`、`runs/ckpts` 在 V3.13 收敛到 `v1-store/evaluation/`、`v1-store/train-runs/`；`eval.sh` 的裸 `python` 换 `uv run`。评估逻辑本身不动。

## 六、两条核心保证的原理（给人看）

### 6.1 「建库链产物与原先完整一致」怎么保证

**第 0 层：已有产物根本不被触碰。** 本轮不重建任何库——678 GB 的 4task-gl、packed 库、norm_stats 全部原样。所以「产物一致」要保证的其实是：**将来再跑建库链时，代码行为和今天逐位相同**。

**第 1 层：源码同一性是机器判定的，不靠行为测试猜（COPY_DIFF）。** 四个文件从 `shared/` 复制进 `dataset_builder/` 时，验收判据是

```bash
COPY_BASE=732fae3b13e2ff5f485d7014473b99ed577de387   # V3.8 的父提交，钉死不随 HEAD 走
git show $COPY_BASE:src/mme_vla_suite/shared/$f.py | diff -u - src/mme_vla_suite/dataset_builder/$f.py
```

三个叶子文件必须**零差异**，`mem_buffer.py` 的差异必须**恰好 3 行且全是 import 语句**（`shared.` → `dataset_builder.` 的指向替换），函数体一个字符不许动。判定行 `COPY_DIFF=PASS files=4 nonimport_lines=0 base=732fae3b`。这一过，「计算逻辑没变」就是源码级证明，不是推测。

**A 侧锚点为什么必须钉死（审计修正）**：原稿写的是 `git show HEAD:shared/<f>.py`，而本计划 V3.12 会删掉 `shared/mem_buffer.py` 与 `shared/siglip_tokenizer.py`、瘦身 `shared/data_utils.py` —— 到 V3.14 tip 上 `git show HEAD:` 会直接 `fatal: path does not exist in 'HEAD'`，而 N4c 又标着「V3.8 起持续」、六节前置门还要求这条判定行 PASS 才准起跑 G3。用浮动 `HEAD` 等于给自己上了一道过不去的闸。钉死 `COPY_BASE` 后，这条判据在任何时点、任何 HEAD 上都可复算。V3.12 之后 `shared/` 侧源文件已不在工作区，**接力主判据是 R13 的 sha256 哨兵**（见下），COPY_DIFF 退为可复算的历史证明。

**注释语言豁免（二节第 7 条）**：四个源文件实测**含中文行数为 0**（全英文 docstring 与注释）。逐字节复制会在仓库里新增全英文注释文件，与 AGENTS 1 字面冲突；翻译则 COPY_DIFF 与 sha256 哨兵同时失效。本文件据此声明：**冻结副本视同 `git mv` 产物，不适用「新增注释必须中文」**，逐字节同一性优先。复制时**一个字符都不许改**，包括英文注释。

**第 2 层：依赖闭合，训练侧后续怎么删改都波及不到（IMPORT_ISOLATION）。** 这正是「彻底隔离」和「产物一致」互相成全的地方：V3.9–V3.12 会删 `shared/mem_buffer.py`、瘦身 `shared/data_utils.py`——如果建库脚本还绑在 `shared/` 上，产物一致当场就破。隔离后用双向判据钉死：静态 grep（建库域文件不得出现任何 `mme_vla_suite.shared/training/policies` 引用）+ 动态断言（全新解释器 import 建库入口后 `sys.modules` 里零训练侧模块泄漏——这一条抓静态 grep 抓不到的传递依赖）。判定行 `IMPORT_ISOLATION=PASS builder_leaks=0 train_leaks=0 online_leaks=0`。openpi 底座本轮整文件不碰，不在波及面里。

**第 3 层：同架构行为抽查兜住「万一 diff 看漏」（BUILDER_SPOT + 三方对拍）。**

- `finalize_checks.py` 里现成的 `spot_check` 就是为此设计的量具：从原始 H5 重算 `token_emb_{step}.npy` 与存盘产物比对。用改造后的建库域代码对本机真值库 `ref-crossarch`（47 episode，本机 RTX 6000 Ada 产出）抽 32 条跑，判定行 `BUILDER_SPOT=PASS n=32 checked=32 max_diff=0.000e+00`。
- **它不是 bitwise，措辞必须准确（审计修正）**：`spot_check` 的实现是两侧各 `np.asarray(...).astype(np.float64)` 后取 `max(|a-b|)` 是否为 0，属**float64 数值零差**，不是逐位比对。它抓不到三类变化：同值但 dtype 变了、`+0.0/-0.0` 的位型差、数值相同而字节布局不同。原稿「零容差逐位比对」的说法作废。真正的逐位证明由第 1 层（源码同一）与下面的三方对拍（`leaf_sha256`）承担，`max_diff` 只作诊断。
- **抽样池必须先收窄（审计修正，否则必 FAIL）**：`spot_check(manifest, out_dir, raw_dir, n, seed)` 的抽样源就是传入 manifest 的 `episodes` 全体、库内不做过滤，缺文件即计入 errs。全局 `v1-store/episode_manifest.json` 有 **1600** 条，而 `ref-crossarch/features` 只有 **47** 条 —— 直接传全局 manifest 时 32 抽的命中率仅约 3%，期望约 31 条落进「抽检缺文件」分支必 FAIL；更坑的是 `worst` 只在比对成功时更新，判定行会呈现「FAIL 但 `max_diff=0.000e+00`」的误导组合，把排障引向「环境漂移 vs 拷贝错」的错误二分。**做法**：先按 `ref-crossarch/meta/_shard0of1.json` 的 47 条 episodes 过滤出子集 manifest 再传入；`raw_dir` 写死取值；`OPENPI_DATA_HOME` 必须显式给（`siglip_tokenizer.py` 里该变量**不做 expanduser**，而 `spot_check` 内 `prepare_buffer=True` 会走自建 SigLipTokenizer 分支）；判定行区分 `n` 与 `checked`。
- 另外 ONLINE_MEM harness 顺带做**三方对拍**：同一个 jitted SigLIP、同一批真实帧，分别喂旧 `shared.mem_buffer`、新 `dataset_builder.mem_buffer`，`get_history_feats` 全键逐位（`leaf_sha256`，这才是真 bitwise）——直接证明副本与原件行为逐位相同，零额外 GPU 开销。**注意**：这条与 IMPORT_ISOLATION 的「量具不得 import `dataset_builder`」字面互斥，须走五节 c-2 的量具白名单，不得为过闸砍掉三方对拍——砍了本层就只剩非 bitwise 的 BUILDER_SPOT。
- 失败时先跑 **null 对照**（git 提取的旧代码跑同一组抽样）：null 也挂 → 是 2026-08-23 以来的环境漂移（驱动/jaxlib），与重构无关；null 过而副本挂 → import 重接绑错了兄弟模块。

**防退化哨兵（R13）**：`test_guards.py` 加用例断言建库域副本的 sha256 等于固化常量——防止将来有人好心把训练侧的改动「同步」回建库域副本，让冻结悄悄失效。**四个副本各钉一条，不是只钉 `mem_buffer.py`**（原稿只写了一条；V3.12 之后 `shared/` 侧三个源文件或删或改，这四条哨兵就是唯一在岗的冻结证明）。

**三处必须知道的边界（计划里已如实登记）**：
1. **禁跑 `finalize_checks` 的 check 子命令、禁对 4task-gl 抽查**——check 子命令无错时会**改写**目标库的 `meta/stats.json` 和 `meta/provenance.json`，而 `meta/stats.json` 在 G0 的数据集指纹里，跑一次基线就作废；且 4task-gl 是 GL A40 产出，本机 Ada 重算跨架构逐位不可得（仓库已有实证报告），对它跑必挂且不说明任何问题。所以只允许 `import spot_check` 函数、只对本机同架构库跑。（**顺带纠一处事实**：`provenance.json` 其实**没有**进 G0 的数据集指纹——finalize 写的是 `meta/provenance.json`，而 `check_baseline_env.py` 的 `_dataset_spot_digest` 读的是数据集根下的 `provenance.json`、且 `if p.exists()` 不存在就静默跳过，尽管 scheme 名仍写着含 provenance。本轮**不修这个路径**，见八节盲区第 9 条与 R22。）
2. **GL 侧不重跑**（八节盲区第 4 条）：这套保证证到的是「代码逐字节同一 + 本机同架构复算逐位一致」，不是「在 GL 上重建一遍 678 GB 逐位相同」。后者由「同一代码 + GL 环境未变」推得，诚实标注为推论而非实测。
3. **`--force` 闸只挡「误删」，不做路径白名单**（二节第 8、14 条）：V3.8 给建库输出加的是「默认拒绝已存在目标、要删必须显式 `--force`」。它挡住的是最现实的事故形态——重建时少写一层目录名、传成 `v1-store/datasets` 或 `v1-store`（都是已存在目录，无 `--force` 直接拒绝）。它**不**挡「显式加了 `--force` 又传错路径」，也**不**做 canonical containment、软链接祖先、挂载点检查。这层留待另开一轮，本轮如实登记为 R20。

### 6.2 「G3 对拍 G0 就能证明训练语义没变」怎么成立

**第 0 层：G0 是什么。** `docs/training-doc/v1-grad-baseline-g0b/records/r1/`——重构前训练语义在受控确定性档（`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`）下的一轮真实训练：本机 2×RTX 6000 Ada、b8、seed 42、1000 步，四类产物固化进 git——逐步五标量的 IEEE 浮点位（`scalars_hex.tsv`：loss / grad_norm / llm_grad_norm / mem_enc_norm / param_norm）、12 个摘要步的完整 TrainState（params + Adam 动量 + EMA）逐叶 sha（`param_checksums.jsonl`）、14 个记录步的输入 batch 逐键指纹（`batch_digests.jsonl`，`n_keys=12`）、全部样本抽取顺序（`index_sequence.json`）。

**第 1 层：复跑零抖动已被四份证据钉死。** 四档确定性实验里 D2 与 **D2-cold**（清空编译缓存、独立重编译两次）双 PASS——G3 代码变了、必然现场重编译，对拍依然成立；G0b 两轮千步自证 + G1（dtype 后）+ G2（packed IO 后）复现出同一个 `scalars_hex.tsv` sha256 = `c799a0b2…105757`，等于这条管线的复跑噪声被四份独立证据钉死为**严格零**。

**第 1.5 层：起跑前必须清干净三个能悄悄换数据源的环境变量（审计新增）。** `training/dataloader.py` 的 `_create_framesamp_dataset` 会读 `MMEVLA_FRAMESAMP_SOURCE`（换源库根）、`MMEVLA_FRAMESAMP_MANIFEST`（换 manifest）、`MMEVLA_FRAMESAMP_VERIFY`（改校验档）。原稿前置门只列了 `MMEVLA_FRAMESAMP_ALLOW_*`。环境里残留一个 `SOURCE` 就会从另一个源库读 pkl，**而 preflight 的 `dataset_spot` 锚在 `--dataset` 命令行参数上、根本不会失配**——指纹全绿、交付字节已变。前置门判据改为 `env | grep MMEVLA_FRAMESAMP` 输出为空。

**第 2 层：环境指纹 preflight 排除环境漂移。** 起跑前 `check_baseline_env.py check --baseline <G0b-r1> --dataset v1-store/datasets/4task-gl --steps 1000 --batch-size 8`（**必须带** `XLA_FLAGS`/`XLA_PYTHON_CLIENT_MEM_FRACTION`/`CUDA_VISIBLE_DEVICES` 三个环境变量——它们都在指纹里，裸跑必 FAIL 三项）：逐项比对 uv.lock 全量 sha、torch/jax/jaxlib/numpy/ml_dtypes 版本、GPU 型号与驱动、norm_stats/tokenizer/pi05 权重/数据集抽样指纹等数十项，任一不符 `BASELINE_ENV=FAIL`、基线作废。**指纹不含仓库代码 sha**——PASS 只证「还有资格引用 G0」，绝不能当「代码没改坏」的证据，后者只能由第 3 层回答。

**第 3 层：四维位级对比，任何一跳改一个字节都藏不住。** G3 与 G0 同 seed、同数据、同 b8 跑 1000 步后 `compare_baseline.py <G0b-r1> <G3> --tier g3-vs-g0b` 四分项全部要求逐位：① `SCALARS steps=1000 keys=5 hex_mismatch_steps=0`（每步五标量浮点位）；② `STATE_DIGEST rows=12 mismatch=0`（参数/优化器摘要）；③ `BATCH_DIGEST_CANONICAL rows=14 mismatch=0` + `CANON_CHECK=PASS`（喂进模型的 batch 内容）；④ `INDEX_SEQ=PASS n=8072`（前 8,000 条抽样顺序前缀）。数据装配、变换、模型前向、梯度、优化器更新任何一跳的任何字节差异都会在标量或摘要里位级暴露。收官一行：`sha256(records/scalars_hex.tsv) == c799a0b2…105757`。

**归因唯一化**：第 1 层排除「重跑噪声」、第 2 层排除「环境变了」，于是 **G3 逐位等于 G0 的唯一解释就是七个提交合起来没改训练语义；不等的唯一解释就是改了**——随后按便宜到贵的阶梯定位，禁止直接重跑 2.5 小时的 G3。阶梯按**是否在 G3 的因果路径上**分两组（审计修正）：

- **在链上**（真能定位 G3 失配）：`GRAD_FIXTURE` 15 分钟（单步）→ `SMOKE5` 35 分钟（前 5 步）→ **G3 过程中的增量 state 对拍**（每出一次摘要即与 G0b-r1 同步骤比，12 个摘要步天然是二分粒度）。注意前两者只覆盖到第 5 步，而「N1–N5 全过、G3 仍失配」的唯一情形恰恰是**步 5 之后才分叉**——那时只有增量对拍能定位，故它是正式一级，不是可选技巧。
- **不在链上**（只证建库/在线侧，G3 失配时先跑是浪费）：`COPY_DIFF` 10 秒 → `IMPORT_ISOLATION` 1 分钟 → `ONLINE_MEM` 10 分钟。

**已知不作判据的两项（预先声明防误读）**：raw 口径 `BATCH_DIGEST mismatch=4 first_bad_step=100 bad_keys=2 (static_image_emb/static_pos_emb)` 是 V2.4b dtype 统一的**预期失配**——与 G2 逐字吻合即正常，**不是** 4 步×2 键则升格为信号；总行 `DET_CHECK=FAIL` 是已拍板不修的工具聚合缺口，分项判读为准。

**三处必须人工补位的 fail-open（审计新增；二节第 13 条已定本轮不改量具，故纪律写死在这里与 G3 的 `launch.md`）**：

1. **缺标量会被静默跳过，但判定行仍固定打 `keys=5`。** `compare_scalars` 对缺失的键直接 `continue`。若某轮改动让 `train_step` 的 info 少了一个键（`mem_enc_norm` 是最可能的那个——它来自 memory encoder 支路），`SCALARS ... keys=5 hex_mismatch_steps=0` 照样打出来，而实际只比了 4 个键、memory 支路的梯度改动完全逃逸。**判读纪律**：拿到 `SCALARS` PASS 后，必须另行核对 `records/scalars_hex.tsv` 的表头恰为 `step\tloss.hex\tgrad_norm.hex\tllm_grad_norm.hex\tmem_enc_norm.hex\tparam_norm.hex` 六列，缺列即判 FAIL。
2. **`INDEX_SEQ` 只比两侧最短公共前缀。** `n = min(sa["n"], sb["n"])`，一侧被截短也可能打 PASS。**判读纪律**：同时核对 G3 侧 `index_sequence.json` 的 `n` 不小于 G0b 侧，且 `n ≥ steps×batch = 8000`。
3. **`canonical` 与 `INDEX_SEQ` 的结果不进最终 `verdict`、不进退出码。** 最终 verdict 只由 scalar / state / raw batch 决定，而 raw 又是已知 FAIL —— 所以**退出码与总行在 G3 里都不具判据资格**。**判读纪律**：四分项逐行人工核对，任一分项不达标即整体判 FAIL，不看退出码；四分项的期望值逐字写进 `launch.md`，判读时对照勾选。

### 6.3 V3.14 为什么**不是**「纯搬移」（审计推翻原稿定性）

原稿把 V3.14 写成「纯 git mv + 路径修正」、把 `data-preprocess-GL/` 写成「整目录冻结随迁」。这在当前源码上不成立，原因只有一个：**仓库里所有脚本都靠「我在第几层，往上数 N 层」来定位仓库根**，搬深一层后全部错位。

```
现在：<repo>/scripts/smoke-local/bench_train_steps.py
      _REPO_ROOT = Path(__file__).resolve().parents[2]   → <repo>          ✓
搬后：<repo>/scripts/training/bench/bench_train_steps.py   ← 深了一层
      parents[2]                                          → <repo>/scripts  ✗
```

后果分两类，第二类才是真正的理由：

- **响亮的（好办）**：`check_baseline_env.py` 的 `_REPO_ROOT / "uv.lock"` 会读到 `<repo>/scripts/uv.lock`，而 `_sha256_file` 是无保护的 `p.open("rb")` → 直接 `FileNotFoundError`，一眼看见就修。
- **静默的（必须专门堵）**：`data-preprocess-GL/paths.sh` 用 `${V1_SCRIPT_DIR}/../..` 求根，搬到 `dataset/gl/` 后得 `<repo>/scripts`；而它的 fail-loud 检查只做**前缀匹配**（`REPO_ROOT` 必须以 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/` 开头）——`<repo>/scripts` 当然也在这个前缀下，**校验照常通过**。于是 `V1_STORE=<repo>/scripts/v1-store`，`v1_prepare_dirs()` 一路 `mkdir -p` 造出整棵假目录树，建库产物全写进去，不报错、不告警。而根 `.gitignore` 只有 `/v1-store/`（前导 `/` 只锚仓库根），`scripts/v1-store/` **不在忽略范围**，会直接出现在 `git status` 里。这同时违反 AGENTS 14 的「派生产物一律收敛到单一根 `v1-store/`」。source 它的有 `step0_setup_turbo.sh`、`step1_submit.sh`、`step2_verify.sh`、`stage_models.sh`、`legacy/step_bench.sh`、`legacy/step_local_baseline.sh` 六个。

**据此本轮的做法（二节第 11 条）**：逐文件把层数改对，**并给每个入口加一句自证断言**——算出来的根必须存在 `pyproject.toml`，否则立刻 fail-loud 退出。这样层数数错也是响亮失败而非静默错位，同时不必引入统一的根解析器、改动仍是逐行小改。同类写法的完整清单见 7.2「根解析修正」组。

## 七、全部改动文件清单（新增 / 修改 / 删除 / 搬移）

> 按文件索引的总清单；每项标注所属 commit。与第二部分逐 commit 条目同源，冲突时以第二部分为准。
>
> **本节经 2026-08-29 审计补全**：原稿此表自称「全部改动文件清单」，实测漏登记十余项（`check_baseline_env.py`、`run_dtype_dump.sh`、`greatlakes.md`、`pyproject.toml` 的 ruff 键、`README-ZH.md`、`train.py` 的 `wandb.log_code`、GL sbatch 与 `step1_submit.sh`、`eval.sh` 的 `serve_policy.py` 自引用等），已全部补入。

### 7.1 新增文件（9 项）

| 文件 | 作用 | 改动内容 | commit |
|---|---|---|---|
| `src/mme_vla_suite/dataset_builder/data_utils.py` | 建库域副本：even_sampling / 左右 padding / pool 四函数 | 自 `shared/data_utils.py` **逐字节复制**，零改动 | V3.8 |
| `src/mme_vla_suite/dataset_builder/posemb_3d.py` | 建库域副本：`PosEmb3D` 3D 位置编码 | 逐字节复制，零改动 | V3.8 |
| `src/mme_vla_suite/dataset_builder/siglip_tokenizer.py` | 建库域副本：SigLIP 编码器加载 | 逐字节复制，零改动 | V3.8 |
| `src/mme_vla_suite/dataset_builder/mem_buffer.py` | 建库域副本：`MemoryBuffer` 全类（含 token_drop，建库仍用） | 复制后**恰改 3 行 import**（`shared.` → `dataset_builder.`），函数体一字不动 | V3.8 |
| `src/mme_vla_suite/shared/sampling.py` | 训练/在线共用的选帧函数新家 | `even_sampling_indices` 函数体原样搬入；只 import numpy，不拉 flax（解 worker 导入负担） | V3.12 |
| `src/mme_vla_suite/policies/framesamp_memory.py` | 在线评估 framesamp 专用记忆缓冲 `FrameSampMemory` | 新写：`add_buffer`（注入 `vision_enc_fn`、只算 4x4 池化）+ `prepare_frame_sampling`（与旧装配逐字同式，复用 `right_padding_token_emb`）+ `n_steps`/`clear` | V3.12 |
| `scripts/smoke-local/compare_online_memory.py` | ONLINE_MEM 三层 A/B 量具（POS_TABLE/ENC_LAYER/ASSEMBLY 判定行） | 新写；V3.14 随迁 `training/bench/` | V3.12 |
| `scripts/training/paths.sh` | 训练域自带路径定义 | 新建，切断 `run_2gpu_epoch_bench.sh` 对建库域 `paths.sh` 的跨域 source | V3.14 |
| `scripts/training/` + `scripts/dataset/` 目录结构 | 两域顶层 | 见第一部分四节对照表 | V3.14 |

### 7.2 修改文件（按域分组）

**建库域（V3.8，改 import 行 + 哨兵 + `--force` 闸）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `src/mme_vla_suite/dataset_builder/build_robomme_dataset.py` | 旧建库主流程（`DatasetProcessor`，被 build_shard 复用） | ①import 行改指 `dataset_builder.mem_buffer`；②**`__init__` 加 `--force` 闸**：现状是在任何校验之前无条件 `if os.path.exists(self.dataset_path): shutil.rmtree(self.dataset_path)`，改为新增 `force: bool = False` 形参，目标已存在且 `force` 为假时**显式 raise** 并提示加 `--force`；只在 `force` 为真时才 rmtree。**rmtree 之后的建库计算路径一行不动**（产物一致性不受影响） |
| `scripts/build_dataset.py` | 建库分派 CLI | 新增 `--force` 开关，透传给 `DatasetProcessor`；帮助文本写明「不加 `--force` 时拒绝已存在的输出目录」。（V3.14 搬 `dataset/build_dataset.py`） |
| `scripts/data-preprocess-GL/build_shard.py` | GL 8×GPU 分片建库主循环 | 同上一行 import。**注意**：它构造 `DatasetProcessor` 的调用点须显式传 `force=True`（分片重跑本就要覆盖自己的分片目录），否则 GL 建库当场被新闸拦住 |
| `scripts/data-preprocess-GL/finalize_checks.py` | 建库收尾校验（含 `spot_check` 复算） | 函数内局部 import 改指 |
| `scripts/data-preprocess-GL/compare_datasets.py` | 新旧库三层对拍（v4 验收资产） | 函数内局部 import 改指 |
| `scripts/data-preprocess-GL/test_guards.py` | 建库守卫 pytest | 加 **4 条** sha256 哨兵用例（`dataset_builder/{data_utils,posemb_3d,siglip_tokenizer,mem_buffer}.py` 各一条，断言等于固化常量，防发散）；另加一条 `--force` 闸的负向用例（目标已存在且未给 force 时必须 raise、且**目录仍在**） |

**训练数据链（V3.9–V3.10）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `src/mme_vla_suite/training/dataloader.py` | 训练 DataLoader 装配入口 | 删 `_resolve_backend` 三态与 RoboMMEDataset import，`create_data_loader` 压成无条件 packed；`DataLoaderImpl`/`_create_framesamp_dataset` 本体一字不动 |
| `src/mme_vla_suite/training/framesamp_dataset.py` | packed 唯一训练数据集 | V3.9 仅改三处注释措辞；V3.10 `_NONE_KEYS` 删 6 项（recur_*4+subgoal2）；V3.12 import 改指 `shared.sampling`；`integration_type=="context"` 断言不动 |
| `src/mme_vla_suite/training/config.py` | 训练配置 + transforms 工厂 + tokenizer | **V3.9**：⑥删 `_CONFIGS` 里的 `pi05_baseline` TrainConfig（二节第 6 条；它 `use_history=False, history_config=None`，packed 单一化后进不了 `FrameSampDataset`）。**V3.10**：①RepackTransform 删 6 键；②`TokenizePromptWithSymbolicMemory` 删 symbolic 分支与两个无默认 pop；③`ModelTransformFactory` 删 symbolic 块与 `max_token_len*=2`；④`PaligemmaTokenizer.tokenize` 删 subgoal 形参分支（先改 `_download.maybe_download` 再删 import，R2）；⑤删死类 `LeRobotMMEVLARealRobotDataConfig`、`MMEVLAWeightLoader` |
| `src/mme_vla_suite/shared/data_utils.py` | 训练/在线共享工具 | 删 `even_sampling_indices`（搬走）与 `left_padding_token_emb`（调用者归零）；保留 right_padding + pool |
| `scripts/compute_norm_stats.py` | norm stats 唯一生产链 | 删 RoboMMEDataset 依赖，内联 `_PklSampleDataset`（等价性八要点）；加 `--output-dir`，**验证/试跑时必填**（现状默认直接写生产 `norm_stats.json`，而该文件是 G0 指纹项；漏给参数即覆盖基线，事前 sha256 备份只能发现损坏、不能恢复） |

**模型侧（V3.11）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `src/mme_vla_suite/models/integration/history_pi0.py` | 模型主体（HistoryPi0） | `create`/`inputs_spec`/`__init__`/`embed_memory`/`embed_prefix`/`compute_loss`/`sample_actions` 七处删 recurrent/symbolic 分支与 stats 返回链；**`inputs_spec` 的外层 `use_history` 二分同刀删除**（原稿写「保留，pi05_baseline 依赖」，而该入口已在 V3.9 删除）；expert/modulation 构图与 lazy_init 参数一字不动 |
| `src/mme_vla_suite/models/integration/history_observation.py` | 训练/在线共用的观测数据类 | 删 **`recur_*`4**（`recur_image_emb`/`recur_pos_emb`/`recur_state_emb`/`recur_mask`，原稿写「8」是把四处镜像重复计数）+ `symbolic_*`2 字段及 from_dict/to_dict/from_base_obs/preprocess_observation 四处镜像；`static_*` 与基类字段全留 |
| `src/mme_vla_suite/models/representation/mem_encoder.py` | memory token 投影器 `FeatureEncoder` | 删 `encoder_recur`/`encode_recurrent_memory`/`ouput_dim_for_recur`/`ndim==5` 分支 |
| `src/mme_vla_suite/models/representation/percep_mem.py` | perceptual memory 编码模块 | 删 `ouput_dim_for_recur=None` 实参与无消费点的 `mem_type` 赋值 |
| `src/mme_vla_suite/models/representation/utils.py` | 初始化器等工具 | 保留 `kernel_init`+`kernel_init_out_proj`（history_gemma 依赖），删 ttt/rmt/rope 系函数 |
| `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context.yaml` | 唯一训练配置 | 值一字不动；可选加中文字段注释（R6 schema 补偿，须 G3 前完成） |
| `scripts/train.py` | 训练主入口 | **V3.11**：`train_step` 三返回改二返回（注解同 hunk）、删 `get_stats` 与 recurrent 统计打印、`ptrain_step` out_shardings 同步；四个 bench 指纹字符串一字不动；**`info` 必须仍含 `mem_enc_norm`**（R17，比较器缺键会静默跳过、判定行仍打 `keys=5`）。**V3.14**：`wandb.run.log_code(epath.Path(__file__).parent.parent)` 的层数改对——`train.py` 由 `scripts/` 搬到 `scripts/training/` 后 `parent.parent` 只归档 `scripts/`，不再归档仓库根 |

**在线评估侧（V3.12）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `src/mme_vla_suite/policies/policy.py` | 在线评估 policy（serving/challenge 共用） | 删 mem_buffer import 换 `FrameSampMemory`；`_prepare_mem_buffer` 压单支——**连带删掉 `config is None` 分支**（它服务的 `pi05_baseline` 已在 V3.9 删除；若 `pi05_baseline` 曾被保留，「压单支」会直接打死无 history 的在线路径，这是二节第 6 条必须先拍板的原因）；infer 断言改 `n_steps`+显式 raise；`_prepare_history` 只留 frame_sampling |
| `src/mme_vla_suite/policies/robomme_policy.py` | 输入 transforms（`RoboMMEInputs`） | 删 recur_*4 + subgoal2 共 6 个 `data.get` |

**量具与外围脚本（V3.9–V3.14）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `scripts/dtype-unify/single_step_grad.py` | 单步定点梯度对拍（GRAD_FIXTURE 的 B 侧） | V3.9 数据源改 `_create_framesamp_dataset`；V3.11 `loss_fn` 同步去 stats/has_aux（第二耦合点）；V3.14 迁 `training/tests/` |
| `scripts/dtype-unify/dump_fixture_samples.py` | 逐样本/逐 batch 逐键落盘量具 | 数据源改指 packed；迁 `training/tests/` |
| `scripts/dtype-unify/run_dtype_grad.sh` | GRAD_FIXTURE 驱动 | `--dataset-path` 改 `${DATASET_PATH:-${GL_DATASET}}`；迁 `training/tests/` |
| `scripts/data-pack-framesamp/test_pack_guards.py` | packed 库/数据集守卫 pytest | 删 backend 三态用例（含 `pytest.raises(match="MMEVLA_DATA_BACKEND")`）；import 改 `shared.sampling`；迁 `training/tests/` |
| `scripts/data-pack-framesamp/dump_index_seq.py` | index 序列交付对拍 | 仅改 `--legacy-root` help 文本（实指源 pkl 库）；迁 `training/tests/` |
| `scripts/data-pack-framesamp/spawn_matrix.py` | spawn/fd 泄漏守卫 | docstring 措辞更新（flax 导入链说明失效）；迁 `training/tests/` |
| `scripts/smoke-local/run_2gpu_epoch_bench.sh` | G 链正式起跑驱动 | **V3.9**：①删 env.json 的 `MMEVLA_DATA_BACKEND`/`backend_source` 两行；②**失败时不再删记录目录**（二节第 9 条）——现状 `RC != 0` 时执行 `rm -rf -- "${RECORD_DIR}"`，与六节「G3 首次 state 分叉立刻停跑」直接冲突（停跑即非零退出，刚抓到的分叉证据当场被删）。改为原子改名 `mv "${RECORD_DIR}" "${RECORD_DIR}.failed-<n>"`，`<n>` 取现存后缀最大值 +1，绝不覆盖已有失败记录。**V3.14**：自引用路径 + source 训练域自带 paths.sh + 根解析修正 |
| `scripts/smoke-local/bench_train_steps.py` | G 链 bench 入口（源码指纹护栏所在） | 仅 V3.14 的 sys.path 插入改指 `scripts/training`；护栏与判据零改动 |
| `scripts/train-prod/gl_train_prod.sbatch` | 正式训练 sbatch | 删 `MMEVLA_DATA_BACKEND` 四处（export/注释/env.json 键/echo，set -u 下必须同删）；**V3.14 要改的是四处路径不是一处**（原稿只登记了 analyze_gpu_util）：`"entry": "scripts/train-prod/prod_train_once.py"` 的 env.json 字段、`run_py "$REPO/scripts/train-prod/prod_train_once.py"`、`run_py "$REPO/scripts/bottleneck-bench-v2/analyze_gpu_util.py"`、`run_py "$REPO/scripts/train-prod/downsample_util_csv.py"` |
| `scripts/train-prod/prod_train_once.py` | 正式训练薄启动器 | V3.14 sys.path 插入 smoke-local→training/bench |
| `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | 历史 e2e sbatch | V3.9 同删 backend 四处（该文件 V3.14 随目录退役） |
| `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | util 稳态分析（prod 趋势在用的活量具） | 读历史 env.json 逻辑不改，加一行历史遗留注释；V3.14 迁 `training/bench/` |
| `scripts/eval.sh` | 在线评估启动 | ①删 symbolic/MemER 分派与菜单，默认 MODEL_TYPE 改 `perceptual-framesamp-modul`；②**`runs/ckpts` → `v1-store/train-runs/`**（AGENTS 14）；③**评估客户端的裸 `python` 换 `uv run`**（AGENTS 3）；④V3.14 同步 `serve_policy.py` 的自引用路径（现指 `scripts/serve_policy.py`） |
| `scripts/finetune_mme_vla_suite.sh` | 训练启动 | 注释菜单 14 变体 → 3 个 framesamp；V3.14 同步 `train.py` 自引用路径 |
| `scripts/compute_results.py` | 评估结果统计 | 删 `--symbolic_type` 与 symbolic 分支，默认 model_dir 改 framesamp-modul；**默认结果根 `runs/evaluation` → `v1-store/evaluation/`** |
| `scripts/unzip_ckpt.py` | checkpoint 解压工具 | 默认目标 `runs/ckpts` → `v1-store/train-runs/`（V3.13） |
| `scripts/download_pi05_base.py` | 下载 pi05 初始权重 | 权重落点显式指向 `v1-store/models`（不覆盖 `HOME`，只设 `OPENPI_DATA_HOME`；AGENTS 14）（V3.13） |
| `examples/robomme/eval.py` | 评估客户端 | 删 subgoal 全链（import/参数/注入/存档分支） |
| `examples/robomme/utils.py` | 客户端工具 | 删 `SUBGOAL_TYPES` 与 record 的 subgoal 形参/文字叠加 |
| `examples/robomme/env_runner.py` | 环境 runner | 删两个 subgoal oracle property |
| `pyproject.toml` | 项目配置 | V3.14。**原稿登记错了对象**：`testpaths = ["src", "scripts", "packages"]` 只列顶层、**无需改**；真正失配的是 `[tool.ruff.lint.per-file-ignores]` 的四条 glob——`"scripts/data-preprocess-GL/*.py"`、`"scripts/smoke-local/*.py"`、`"scripts/dtype-unify/*.py"`、`"scripts/data-preprocess-GL/gl_submit.py"` 全部指向即将消失的目录，不改则中文标识符的 RUF001/002/003 豁免失效、ruff 当场报一片 |
| `README.md` | 仓库说明 | V3.14 规范命令改新路径 |

**根解析修正（V3.14，二节第 11 条；每处「改对层数 + 加 `pyproject.toml` 自证断言」）**

> 这一组是原稿「纯搬移」定性作废的直接原因，见 6.3。逐个改完后按 7.5 的判定行统一验收。

| 文件 | 现状写法 | 搬后要点 |
|---|---|---|
| `scripts/data-preprocess-GL/paths.sh` | `REPO_ROOT="$(cd "${V1_SCRIPT_DIR}/../.." && pwd)"`，且 fail-loud 只做 turbo 前缀匹配 | **最危险的一处（静默）**：搬到 `dataset/gl/` 后 `../..` 得 `<repo>/scripts`，前缀校验照过，`V1_STORE` 变成 `<repo>/scripts/v1-store` 并被 `v1_prepare_dirs()` 真的建出来。改 `../../..`，并把前缀校验换成/补上 `if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then echo "错误: 仓库根解析失败 ${REPO_ROOT}" >&2; exit 1; fi` |
| `scripts/smoke-local/bench_train_steps.py` | `parents[2]` + `sys.path.insert(0, str(_REPO_ROOT / "scripts"))` 后 `import train` | 搬到 `training/bench/` 深一层。**注意原稿的改法本身是错的**：把插入目标写成 `.../"scripts"/"training"` 而层数不改，会得到 `<repo>/scripts/scripts/training`。正解是层数改 `parents[3]`、插入目标 `_REPO_ROOT / "scripts" / "training"` |
| `scripts/smoke-local/check_baseline_env.py` | `parents[2]`，且 `_sha256_file(_REPO_ROOT / "uv.lock")` 无异常保护 | **原稿全表未登记此文件**。搬后不改即 `FileNotFoundError: <repo>/scripts/uv.lock`，preflight 直接跑不起来 |
| `scripts/train-prod/prod_train_once.py` | `parents[2]` + sys.path 插 smoke-local | 层数 + 插入目标同改 `training/bench` |
| `scripts/dtype-unify/_common.py` | `parents[2]` | 搬 `training/tests/` |
| `scripts/dtype-unify/single_step_grad.py` | `_HERE.parent.parent` | 搬 `training/tests/` |
| `scripts/data-pack-framesamp/{pack_framesamp_store,spawn_matrix,probe_layout,dump_index_seq}.py` | `parents[2]` 一类 | 四件分属两域（pack → `dataset/pack/`，后三件 → `training/tests/`），层数各自算 |
| `scripts/data-pack-framesamp/run_pack.sh` | `../..` | → `dataset/pack/` |
| `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | `parents[2]` | → `training/bench/` |
| `scripts/data-preprocess-GL/{build_shard,finalize_checks,compare_datasets,test_guards}.py` | `_HERE.parents[1]` | → `dataset/gl/`，「冻结随迁」对这四个文件不成立 |
| `scripts/data-preprocess-GL/gl_build_dataset.sbatch`、`step1_submit.sh` | 内部调用旧路径的脚本名 | sbatch 内的入口路径与提交器提交的 sbatch 路径同步改 |
| 形如 `sys.path.insert(0, _REPO_ROOT / "src")` 的插入 | — | **这一类改错了无害**：`mme_vla_suite` 已由 `.pth` editable 装入，Python 对不存在的 sys.path 条目静默忽略。真正会炸的是指向 `"scripts"` 的插入，以及所有用 `_REPO_ROOT` 拼数据/资产路径的地方 |

**原稿漏登记的活文件（V3.14 一并处理）**

| 文件 | 为什么必须改 |
|---|---|
| `greatlakes.md` | AGENTS 8 指定的集群提交**唯一权威源**，正文有 5 处以上写死 `scripts/data-preprocess-GL/gl_submit.py`（含「提交器」定义行与三条可直接复制的 `uv run` 命令）。不改则权威文档指向不存在的路径 |
| `scripts/dtype-unify/run_dtype_dump.sh` | 原稿的 dtype-unify「6 件活工具」清单里没有它，但它是 `dump_fixture_samples.py` 的驱动。若随目录整删，留下的就是一个没有驱动的半截量具 → 迁 `training/tests/` |
| `README-ZH.md` | 与 `README.md` 同样含 `scripts/` 路径的规范命令 |
| `src/mme_vla_suite/datastore/README.md`、`scripts/smoke-local/README.md`、`scripts/dtype-unify/README.md`、`docs/manual_evaluation.md` | 均含指向旧路径的命令；`datastore/README.md` 另承担 R6 的 history_config schema 文档化 |
| `src/mme_vla_suite/training/framesamp_dataset.py` 的注释 | 有 8 处「旧路径」注释指向 V3.9 要删的 `training/dataset.py`（原稿只写「改三处」）；`dump_fixture_samples.py` 顶部「走裸 `RoboMMEDataset`」的说明在 V3.9 后既指向已删类、又与新实现相反（AGENTS 9） |

### 7.3 删除文件

| 文件/目录 | 作用（删除理由） | commit |
|---|---|---|
| `src/mme_vla_suite/training/dataset.py` | legacy 数据链（RoboMMEDataset/SampleDataset），packed 唯一化后无消费者 | V3.9 |
| `scripts/finetune_pi05_baseline.sh` | `pi05_baseline` 启动脚本（入口整删，二节第 6 条） | V3.9 |
| `training/config.py` 的 `pi05_baseline` TrainConfig（条目非整文件） | 无 history 训练入口，packed 单一化后 `FrameSampDataset` 必拒 | V3.9 |
| `scripts/data-pack-framesamp/compare_batches.py` | packed-vs-legacy 对拍工具，A 侧消失 | V3.9 |
| `scripts/bottleneck-bench/gl-dataloader/` 整目录 | 调私有 `_resolve_backend` 的历史基准 | V3.9 |
| `src/mme_vla_suite/models/representation/{recur_mem,rmt,ttt}.py` | recurrent 记忆模型（RMT/TTT），唯一消费链已删 | V3.11 |
| `models/config/robomme/` 11 个非 framesamp yaml + `models/config/base.yaml` | tokendrop/recurrent/symbolic 配置与无引用模板 | V3.11 |
| `src/mme_vla_suite/shared/mem_buffer.py` | 训练/在线均已脱钩，建库域有冻结副本 | V3.12 |
| `src/mme_vla_suite/shared/siglip_tokenizer.py` | 训练/在线零使用（在线注入模型编码器），建库域有副本 | V3.12 |
| `examples/robomme/subgoal_predictor.py` + `subgoal_prediction/` 子树 | symbolic 在线预测链 | V3.13 |
| `scripts/bottleneck-bench/` 余部、`bottleneck-bench-v2/`、`dtype-unify/`（活件迁走后） | 历史专题目录退役 | V3.14 |
| 旧目录空壳：`smoke-local/`、`train-prod/`、`data-preprocess-GL/`、`data-pack-framesamp/` | 内容已迁入两域 | V3.14 |

### 7.4 搬移（`git mv`；**「内容零改动」只对不含根解析与自引用路径的文件成立**）

- `data-preprocess-GL/` 整目录 → `dataset/gl/`；`pack_framesamp_store.py`/`probe_layout.py`/`run_pack.sh`/README → `dataset/pack/`；`build_dataset.py`/`tarxz_h5.py`/`unzip_data.py`/`finetune_vlm_subgoal_predictor.sh` → `dataset/`。
- `smoke-local/` 四件 + `compare_online_memory.py` → `training/bench/`；`train-prod/` → `training/prod/`；`analyze_util.py`/`_common.py`/`test_padding_dtype.py`/`run_dtype_dump.sh` → 各自去处见四节对照表；`train.py` 等顶层散件 → `training/`。
- **真正零改动的只有**：`finetune_vlm_subgoal_predictor.sh`、`tarxz_h5.py`、`unzip_data.py`、各 README 之外无路径引用的叶子文件、`__init__.py`。其余一律对照 7.2「根解析修正」组逐个核对。

### 7.5 V3.14 迁移验收（判定行）

原稿给 V3.14 的收官判据是「全仓 grep 旧目录名零残留」，这条**既不充分也不可达**：

- **不充分**：顶层散件（`train.py`、`compute_norm_stats.py`、`serve_policy.py`…）的引用里根本不出现旧**目录**名，`parents[N]` 错位更不出现任何路径字符串——这两类断链它一个都抓不到。
- **不可达**：仓库根的历史文档（`greatlakes.md`、`docs/` 留档、本文件自身）必然还写着旧目录名，而计划另一条又要求「历史留档一律不改」，grep 永远返回非零。

改为三条可机器判定的验收，逐条落 `records/`：

```
RELOCATION_REFS=PASS old_refs=0            # 排除 docs/、v1-store/、本文件后，全仓无旧路径引用
RELOCATION_ROOT=PASS entries=<N> mismatch=0 # 每个入口的根解析实测等于 <REPO_ROOT>
RELOCATION_COLLECT=PASS errors=0            # uv run pytest --collect-only -q src scripts packages
```

`RELOCATION_ROOT` 的做法：对 7.2「根解析修正」组的每个入口，用只读方式打印它算出的根（Python 入口 `-c "import runpy…"` 或直接读模块级常量；shell 入口 `source` 后 echo），逐条比对等于仓库根。**这是唯一能抓住 `paths.sh` 那种静默错位的判据**——它不报错、不缺文件，只是根变了。

---

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

- 统一前置：`cd <仓库根>`，`export UV_LINK_MODE=copy`；pytest 一律显式路径（pyproject testpaths 含 scripts，裸跑会全量收集）。
- **两个贯穿全篇的常量**：`COPY_BASE=732fae3b13e2ff5f485d7014473b99ed577de387`（COPY_DIFF 与 ONLINE_MEM 的 A 侧提取锚点，钉死不随 HEAD 走）；确定性档烟测的完整参数见第一部分三节「全局烟测口径」，**四刀验证一律照抄，不得只写 `STEPS=5`**。
- **保留路径逐字清单（G3 bitwise 前提，逐条 review 打勾）**：
  - 数据：`FrameSampDataset.__getitem__` 全体（身份互校 / actions 截断 / even_sampling_indices / `_pad` / reshape+repeat / `_normalize_state`）；`FrameSampStore` 与 `datastore/` 全体；`even_sampling_indices` 函数体（搬模块可以，改一个字符不行）。
  - loader：`DataLoaderImpl`、`_create_framesamp_dataset` 全体、`transform_dataset`/`TorchDataLoader` 构造参数逐字不变；**openpi `data_loader.py` 整文件不碰**。
  - transform：`RepackTransform` 保留 9 键名与顺序；`RoboMMEInputs` 的 state/image/image_mask/actions/prompt 构造；`DeltaActions(make_bool_mask(7,-1))`；`InjectDefaultPrompt`/`ResizeImages(224,224)`/`PadStatesAndActions` 位置参数；`PaligemmaTokenizer.tokenize` 的 else 支与 padding/截断。
  - 模型：非 expert 分支 `_gemma.Module(configs=[paligemma, action_expert], ...)` 与 `lazy_init(use_adarms=[False,True], mem_mods=[False,False])`；`action_in_proj`/`time_mlp_in`/`time_mlp_out`/`action_out_proj` 构造顺序（rng 消耗序）；`PerceptualMemory`/`FeatureEncoder` 的 pos_proj→encoder_static 构造顺序与 kernel_init；`embed_memory` perceptual 体、`embed_prefix` context 分支与 ar/na mask 累积规则、`embed_suffix` 全体、`compute_loss` 的 rng split 顺序与 `beta(1.5,1)*0.999+0.001`；三参版 `make_attn_mask`。
  - 训练：`train_step` 的 `fold_in(rng, state.step)`/`DiffState`/`tx.update`/EMA/info 键（**含 `mem_enc_norm`**）；`init_train_state` 全体；main 的 mesh/sharding/循环结构；**train.main 四个指纹字符串一字不动**：`wandb.log(reduced_info`、`_checkpoints.save_state(`、`init_train_state(`、`create_data_loader(`。
- **graphdef 声明**：本轮删除若干无消费点静态属性（`percep_mem.mem_type`、`encoder_recur`、`HistoryPi0.representation_type`）会改 `nnx.split` 的 static 侧但不改 params/rng/loss——对拍判据不得纳入 graphdef。
- **None 键实证**（第一块不做全量 dump 的依据）：G0b-r1 的 `batch_digests.jsonl` 首行 **`n_keys=12`** 且键集不含任何 `recur_*`/subgoal 键——`tree_flatten`/`_collate_fn` 把 `None` 当空子树剪掉，删恒 None 键在交付面是可证的恒等变换。subgoal 是 pkl 里的真字符串、靠 `TokenizePromptWithSymbolicMemory` 两个无默认 `pop` 拦下：删除必须成对（Repack 去键 + 去 pop），做漏即 `n_keys≠12`、烟测/G3 必挂。

## 一、commit 逐文件改动清单

### commitV3.8 — 建库域隔离 + 建库输出 `--force` 闸（训练侧零改动）

- 新增 `src/mme_vla_suite/dataset_builder/{data_utils,posemb_3d,siglip_tokenizer}.py`：自 `shared/` **逐字节复制**（三者无 mme 内部 import）。**含英文注释一并原样复制**——注释语言豁免见 6.1，禁止顺手翻译。
- 新增 `src/mme_vla_suite/dataset_builder/mem_buffer.py`：复制后**恰好改 3 行 import**（模块级 data_utils 星导入、`prepare_buffer` 分支内 `PosEmb3D` 与 `SigLipTokenizer` 两个懒 import → 全指 `dataset_builder.*`），函数体一字不动；**diff 差异行数==3 是验收判据**。
- `dataset_builder/build_robomme_dataset.py`、`scripts/data-preprocess-GL/{build_shard,finalize_checks,compare_datasets}.py`：import 行改指 `dataset_builder.mem_buffer`（注释保持冻结）。
- **`--force` 闸（二节第 8 条）**：`DatasetProcessor.__init__` 现状在任何校验前无条件
  ```python
  if os.path.exists(self.dataset_path):
      shutil.rmtree(self.dataset_path)
  ```
  而 `dataset_path` 直接来自 `scripts/build_dataset.py` 的 `--preprocessed_data_path`（默认 `data/robomme_preprocessed_data`，无任何约束）——误传 `v1-store/datasets` 或 `v1-store` 会删掉 678 GB 不可恢复数据。改法：新增 `force: bool = False` 形参；目标已存在且 `force` 为假时**显式 raise**（消息里给出「确认要覆盖请加 `--force`」），只有 `force` 为真才 rmtree。`build_dataset.py` 加 `--force` 开关透传。**`build_shard.py` 的构造点必须显式传 `force=True`**（分片重跑本就覆盖自己的分片目录），否则 GL 建库被新闸拦住。**rmtree 之后的建库计算路径一行不动**。
- 防发散哨兵：`scripts/data-preprocess-GL/test_guards.py` 加 **4 条** sha256 哨兵用例（四个副本各一条，断言 == 固化常量；建库域四件自 V3.8 起冻结，与 shared/ 不再同源），另加 1 条 `--force` 负向用例（目标已存在且未给 force 时 raise，**且断言目录仍在**）。
- `dataset_builder/` 无 `__init__.py`（隐式命名空间包，与全仓一致），不需建包文件。
- 验证（≤2min）：以 `COPY_BASE=732fae3b13e2ff5f485d7014473b99ed577de387` 为 A 侧跑 `git show $COPY_BASE:src/mme_vla_suite/shared/$f.py | diff -u - src/mme_vla_suite/dataset_builder/$f.py` ×4 —— 三文件零差异 + `mem_buffer` diff==3 行；建库域 import 闭环；`grep -rn "mme_vla_suite.shared" src/mme_vla_suite/dataset_builder scripts/data-preprocess-GL` 为空；`uv run pytest scripts/data-preprocess-GL/test_guards.py -q`；ruff。

### commitV3.9 — 数据链单一化

- **删** `training/dataset.py` 整文件（`SampleDataset`/`RoboMMEDataset`/`load_vector_file`）。
- **删 `pi05_baseline` 训练入口（二节第 6 条）**：`training/config.py` 的 `_CONFIGS` 首项 TrainConfig（`use_history=False, history_config=None`）+ `scripts/finetune_pi05_baseline.sh` 整文件。**必须与本刀同 commit**：`create_data_loader` 一压成无条件 packed，这个入口就必然撞上 `FrameSampDataset.__init__` 的第一条形制断言 `_req(hc is not None, "history_config 不能为 None")`——留着它就是留一个必炸的入口。其在 `history_pi0.inputs_spec`（V3.11）与 `policies/policy.py`（V3.12）的两处依赖随后各自清理。
- **失败 record 改保留（二节第 9 条）**：`smoke-local/run_2gpu_epoch_bench.sh` 现状在 `RC != 0` 时 `rm -rf -- "${RECORD_DIR}"`，改为原子改名 `${RECORD_DIR}.failed-<n>`（`<n>` 取现存最大值 +1，不覆盖历史失败）。**必须先于第一次烟测生效**，否则 N1 一失败，用于定位的 batch/state/env 记录当场消失；G3 的「首次分叉立刻停跑」更是直接踩这条。
- `training/dataloader.py`：删 `_resolve_backend` 与 RoboMMEDataset import；`create_data_loader` 压成无条件 `_create_framesamp_dataset(...)`；`import os` 保留（`_create_framesamp_dataset` 仍用）；`DataLoaderImpl` 与 `_create_framesamp_dataset` 本体一字不动（bench `_install_idx_probe` 依赖 loader 内部结构）。
- `framesamp_dataset.py`：仅改三处「与 RoboMMEDataset 逐字一致」注释措辞；代码零改动（`_NONE_KEYS` 在 V3.10 才动）。
- `scripts/compute_norm_stats.py`：删 RoboMMEDataset import，内联 `_PklSampleDataset`。**等价性八要点**：①`__len__` 读 `meta/stats.json` 的 `execution_samples` 优先否则 `total_samples`；②裸 `pickle.load(data/{idx}.pkl)` 无任何转换；③`actions[:action_horizon]` 截断；④不建 mem_buffer 不读 features（旧代码 `history_config=None` 即如此）；⑤state 原始值不归一化（`compute_norm_stats=True` 语义）；⑥random 分支全不触发、连 `random.seed` 都不需要；⑦`*_online` pop 可省略（RepackTransform 白名单丢弃未列键）；⑧尾部补 None 键集合必须与**当刀**的 RepackTransform 键集合一致。**原稿此条写错并与 V3.10 自相矛盾**：原文既说「== V3.10 后 RepackTransform 键集合」又在 V3.10 写「补空键集合同步收敛」——若 V3.9 就按 V3.10 口径补，V3.10 便无收敛可做；而事实是 V3.9 落地时 RepackTransform 仍是 15 键（含 `recur_*`4 与两个 subgoal），openpi 的 `RepackTransform` 是 `flat_item[k]` **硬索引**、缺键即 KeyError，源 pkl 键集里又没有任何 `static_*`/`recur_*`。且括注「`static_*`4 + `prompt`」（5 键）与「RepackTransform 键集合」（删 6 后 9 键）根本不是同一个集合。**正确写法**：V3.9 补 11 键（== 当刀 `framesamp_dataset._NONE_KEYS` 的全集），保持 `if key not in data:` 条件赋值；V3.10 删 6 键后同步收成 `static_*`4 + `prompt`。
  新增 `--output-dir`，**验证/试跑时必填**：现状默认直接写生产 `norm_stats.json`，而该文件是 G0 环境指纹项，漏给参数即覆盖基线；动手前照旧先 sha256 备份 `v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`（备份只能发现损坏、不能恢复，故以「必填」为主防线）。
- `scripts/dtype-unify/{single_step_grad,dump_fixture_samples}.py`：`RoboMMEDataset(...)` → `mme_vla_suite.training.dataloader._create_framesamp_dataset(...)`（参数同名直传，`dataset_path` 指 packed 库）；其源码指纹护栏四条本轮全部仍成立，不放宽。
- `scripts/dtype-unify/run_dtype_grad.sh`：`--dataset-path` 改 `${DATASET_PATH:-${GL_DATASET}}`（`GL_DATASET` 在 paths.sh 是 readonly，环境覆盖不了）——GRAD_FIXTURE 前置。
- **删** `scripts/data-pack-framesamp/compare_batches.py`（legacy A 侧消失，历史结论已固化在 docs/）。
- `test_pack_guards.py`：删 backend 三态用例（含 `pytest.raises(..., match="MMEVLA_DATA_BACKEND")` 断言），保留 packed 闸与 G1–G14 全部守卫。
- `dump_index_seq.py`：仅改 `--legacy-root` help 文本（「源 pkl 库根」）。
- **删** `scripts/bottleneck-bench/gl-dataloader/` 整目录（`dataloader_bench.py` 调私有 `_resolve_backend`；配套 sbatch/submit 同批）。
- `MMEVLA_DATA_BACKEND` 清理（**set -u 下 export 与 echo 必须同删**，漏删 echo 当场失败）：`train-prod/gl_train_prod.sbatch`（export/注释/env.json 键/echo 四处）、`bottleneck-bench-v2/gl_e2e_fix.sbatch`（同四处）、`run_2gpu_epoch_bench.sh`（env.json 的 `MMEVLA_DATA_BACKEND` 与 `backend_source` 两行）；`analyze_gpu_util.py` 读历史 env.json 的逻辑**不改**（加一行历史遗留注释）。
- 验证（≤5min）：死引用 grep 清零（白名单 analyze_gpu_util）；train/compute_norm_stats/dataloader import 闭环 + `assert not hasattr(dl,'_resolve_backend')`；`assert "pi05_baseline" not in {c.name for c in _CONFIGS}`；pytest test_pack_guards；确定性档烟测**按三节全局口径给全参数**：
  ```
  STEPS=5 SAVE_INTERVAL=5 BATCH_DIGESTS=1 WARMUP_STEPS=0 KEEP_JAX_CACHE=1 \
  EXP_NAME=v1-restructure-smoke RUN_TAG=v39-smoke \
  DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp
  ```
  （各 commit 复用同一 EXP_NAME 命中编译缓存，RUN_TAG 每次不同；首次冷编译超 5min 按 AGENTS 7 进 tmux。**原稿此处写的 `SAVE_INTERVAL=0` 会连锁关掉 batch digest 与 checksum，令下面的 `INDEX_SEQ` 判据无从产出**，故改 5。）
- **闸门 N1**：烟测记录 `compare_baseline.py --tier v39-smoke` vs G0b：`SCALARS steps=5 hex_mismatch_steps=0` + `INDEX_SEQ=PASS`，PASS 才进 V3.10。
  - **`n` 的口径修正**：原稿写死 `n=40`（=5×8）是错的。`INDEX_SEQ` 取两侧 `n` 的 `min`，而 runner 自身的守卫是 `if seq["n"] < steps * batch` 才报错——序列**恒长于** `steps×batch`（含 prefetch 余量，千步档实测 8072）。判据改为「`INDEX_SEQ=PASS` 且 G3/烟测侧 `n ≥ 40`」，具体数值以首轮实测钉死后写进 launch.md。
  - **`--tier` 只是标签**：`compare_baseline.py` 的 `--tier` 默认 `adhoc`、仅打印在 `DET_CHECK` 行里，**不切换任何判据严格度**。写什么 tier 都不改变比较行为，判读一律以四分项逐行为准。

### commitV3.10 — transforms 瘦身

- `training/config.py`：
  ① `RoboMMEDataConfig.create` 的 RepackTransform 删 6 键（`recur_image_emb`/`recur_pos_emb`/`recur_state_emb`/`recur_mask`/`simple_subgoal`/`grounded_subgoal`）；
  ② `TokenizePromptWithSymbolicMemory` 删 symbolic 分支与两个无默认 `pop`，收敛为 `tokenize(prompt, state)`（可更名 `TokenizePromptWithState`，纯改名无数值影响）；
  ③ `ModelTransformFactory` PI05 分支删 symbolic 块与 `max_token_len*=2`（确认 `get_history_config` import 无其它使用点后删）；
  ④ `PaligemmaTokenizer.tokenize` 删 `subgoal` 形参与分支——**禁止**换 openpi 的 `TokenizePrompt`（本地 discrete-state 分支是分号文本 `Task: {t}; State:`、openpi 是逗号，静默改 pi05 discrete 口径）；
  ⑤ 删死类 `LeRobotMMEVLARealRobotDataConfig`（引用未 import 的类名，调用即 NameError）；
  ⑥ 删 `MMEVLAWeightLoader`——**顺序陷阱**：先把 `PaligemmaTokenizer.__init__` 的 `download.maybe_download` 改为文件顶部已有的 `_download.maybe_download`（同一模块对象），**再**删 `from openpi.training.weight_loaders import ...` 行；ruff 不报此坑，只在运行时构造 tokenizer 才炸。
- `policies/robomme_policy.py`：`RoboMMEInputs.__call__` 删 6 个 `data.get`（recur_*4 + subgoal2）及分组注释。
- `framesamp_dataset.py`：`_NONE_KEYS` **只删 6 项**（recur_*4 + subgoal2），`static_*`4 + `prompt` 保留（RepackTransform 是 `flat_item[k]` 硬索引，兜底零成本）。
- `compute_norm_stats.py` 的补空键集合同步收敛。
- 验证（≤5min）：config import；`uv run python -c "...PaligemmaTokenizer(64)"`（⑥ 陷阱验证）；pytest；ruff。
- **闸门 N2（SMOKE5）**：确定性档（三节全局口径，含 `DATASET_PATH`）→ `compare_baseline.py --tier smoke5` vs G0b：SCALARS 5 步 0 失配、`BATCH_DIGEST rows=3 mismatch=0`（raw 步 0/1/2 必过——V2.4b 预期失配首现于步 100）、`CANON_CHECK=PASS steps=3`、`INDEX_SEQ=PASS`；总行 DET_CHECK 不作结论。SMOKE5 约 30–40min，属对拍闸门不受单 commit 5min 约束，进 tmux；按 AGENTS 17 独立留档（九节）。
  - **加一条白捡的判据（审计新增）**：`SAVE_INTERVAL=5` 会免费产出**步 0**（init 之后、第一次更新之前）的完整 TrainState 逐叶摘要，比较器会自动打 `STATE_DIGEST rows=1`，而 A 侧 G0b-r1 的 `param_checksums.jsonl` 第一行正是 step 0。**判据补 `STATE_DIGEST rows=1 mismatch=0`**——〇节自陈本轮最高风险是 rng 消耗序与构造顺序变化，而初始参数树逐叶摘要正是它最直接的证伪器。原稿只在 N3 检查过一次（V3.11 tip），V3.12–V3.14 之后再无任何闸门复核初始参数树。
  - **`n_keys=12` 的产出者说明**：`compare_baseline.py` **不输出** `n_keys` 字段，原稿三处闸门把它当判定行是错的。改为直接核对本轮 `records/batch_digests.jsonl` 首行的 `n_keys` 为 12 且键集不含 `recur_*`/subgoal；另外 `BATCH_DIGEST_CANONICAL` 的摘要域已含键名，③ 本身即是键集闸，两者互为兜底。

### commitV3.11 — 模型侧单一化 + stats 链整删

- `history_pi0.py`：`create()` 删 symbolic 块；`inputs_spec` 只留 perceptual，**外层 `use_history` 二分同刀删除**（原稿写「保留，pi05_baseline 依赖」——该入口已在 V3.9 整删，二分再无消费者）；`__init__` 删 recurrent/symbolic elif 与 `representation_type` 属性，`PerceptualMemory` 改无条件构造，expert/else 的 llm 构造与 `lazy_init` 参数一字不动；`embed_memory` 压单体去 stats；`embed_prefix` 删 symbolic 分支、语言分支收敛 `if obs.tokenized_prompt is not None`、返回 4 元组；`compute_loss` 解包改 4 元组、`!= "symbolic"` 条件简化为 `if self.use_history`、单返回；`sample_actions` 三处 5 元组解包改 4 元组（expert/modulation/else 分支逻辑不动）。
- `history_observation.py`：删 **`recur_*`4**（`recur_image_emb`/`recur_pos_emb`/`recur_state_emb`/`recur_mask`；**原稿写「8」是把四处镜像重复计数**，实测字段就是 4 个）+ `symbolic_*`2 字段，及 from_dict/to_dict/from_base_obs/preprocess_observation 四处镜像；`static_*` 与基类字段全保留（from_dict 按名 `data.get`，多余键静默忽略，删字段安全）。
- `scripts/train.py`：`train_step` 返回注解改二元组（`@at.typecheck` 校验，与 return 同 hunk）；`loss_fn` 去 stats、`nnx.value_and_grad` 去 `has_aux`；`return new_state, info`；删 `get_stats` 整函数（`import numpy as np` 被 wandb 图像块用，不删）；`ptrain_step` out_shardings 二元组；主循环解包同步；删 recurrent 统计打印块。**四个指纹字符串一字不动**。
- `scripts/dtype-unify/single_step_grad.py`：`_grad_only.loss_fn` 同步去 stats/has_aux（**stats 第二耦合点**，与 train.py 同 commit，漏改即解包炸）；建议给 `_guard_train_step_source` 新增一个锁新形态的 needle。
- **删** `representation/{recur_mem,rmt,ttt}.py`；`mem_encoder.py` 删 `encoder_recur`/`encode_recurrent_memory`/`ouput_dim_for_recur`/`ndim==5` 分支（percep_mem 显式传 None，删除不改参数树与 RNG 消耗）；`percep_mem.py` 删 `ouput_dim_for_recur=None` 实参与 `mem_type` 赋值；`representation/utils.py` 保留 `kernel_init` + `kernel_init_out_proj`（history_gemma 依赖），删 ttt/rmt/rope 系函数（删前逐个 grep 复核）。
- **删** 11 个非 framesamp yaml + `base.yaml`；三个 framesamp yaml **一字不动**（形制断言与 `_EXPECTED_HISTORY_CONFIG` 依赖字段与文件名）；schema 文档化补偿：给 context yaml 加中文注释或记入 `datastore/README.md`——若对拍口径含文件 sha 须在 G3 前 commit 完成。
- 验证（≤5min）：三个 integration 的 `HistoryPi0Config.inputs_spec` 构图冒烟；train.py 指纹断言 + `not hasattr(train,'get_stats')`；ruff；确定性档 STEPS=5 烟测。
- **闸门 N3（GRAD_FIXTURE）**：commit 后跑（`run_dtype_grad.sh` 有 clean-HEAD 硬闸：porcelain 非空即 `exit 1`）：

  ```bash
  DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp RUN_TAG=<本刀唯一标签> \
    bash scripts/dtype-unify/run_dtype_grad.sh
  ```

  判据 `COMPARE_GRAD=PASS kinds=3 mismatches=0`（`allfull` 阴性对照必过）。**A 侧统一引用已固化的 `docs/training-doc/v1-dtype-p5-grad/records/`**（三定点 batch × 32 梯度叶 sha + loss_hex，`same_origin=PASS`），无需现场采集——原稿在同一段里既写 `--grad-a v1-store/dtype-unify/...` 又写「直接引用 docs 固化 JSON」，两个 A 侧口径打架，以 docs 固化件为准。**`RUN_TAG` 是驱动的必填项，原稿 runbook 漏给。** 这是 `has_aux=True→False`（本轮唯一可能动 jaxpr 的改动）的直接证伪器。不进 V3.12。
  - **护栏自洽（审计新增）**：本刀同时修改 `single_step_grad.py` 的 `loss_fn`，而该文件自带 `_guard_train_step_source` 源码指纹护栏。改完必须同刀更新护栏 needle 到新形态，否则 N3 自己跑不起来；更新 needle **不影响 A 侧固化件的 `same_origin`**（A 侧是产物 JSON，不含被护栏保护的源码）。

### commitV3.12 — 在线评估侧 + shared/ 重划

- 新增 `policies/framesamp_memory.py`：`FrameSampMemory`——构造必填 `vision_enc_fn`；`n_steps` 属性替代旧代码摸 `_history_feats` 私有；`add_buffer` 逐跳与旧实现同式，仅三处合法差异（只算 `token_per_image` 一档池化、`jax.device_get` 提到循环外一次、不存 image_pixels/多余档位——四条不可观测性论证：pool 是纯函数各档独立、PosEmb3D 无 RNG 无参数、image_pixels 唯一消费者是 token_drop 打分与死码可视化、装配只读三键）；重复 step 显式 raise 不用 assert；`prepare_frame_sampling` 装配与旧 `_prepare_frame_sampling` 逐字同式，**必须复用 `right_padding_token_emb`、不得改写成 `_pad` 预分配版**。**理由改写（审计纠错）**：原稿给的理由「前者短样本分支 concatenate 会提升 f64」**是错的**——`right_padding_token_emb` 三处 `np.concatenate` 的 `np.zeros` 都已显式带 `dtype=<输入>.dtype`（V2.4b 修复，`test_padding_dtype.py` 是其回归守卫），不会提升；真正没有 dtype 的是本刀要删的 `left_padding_token_emb`。正确理由是「**在线侧本轮只换模块、不换数值路径**：复用同一函数才有逐位同一的机器保证，改写成预分配版就要重新证明等价，收益为零」。原稿这条错误理由与三节图 B 的 bf16/f32 标注本就自相矛盾。**禁把 encode+pool 包进新 jax.jit**（融合变则 bf16 累加序可能变位）。
- `policies/policy.py`：删 mem_buffer import；`_prepare_mem_buffer` 压单支；infer 断言改 `n_steps` + 显式 raise（禁 assert）；`_prepare_history` 只留 frame_sampling 四行赋值；`add_buffer`/`reset`/`_normalize_state` 不动。
- **闸门 N4（ONLINE_MEM，改在 clean HEAD 上跑；二节第 12 条）**：**原稿要求的「commit 前脏工作区共存窗口」作废**——ONLINE_MEM 实测 8–12 分钟，AGENTS 17 要求 >5 分钟的 run 从 clean HEAD 起跑并留档，两者正面冲突（仓库同类工具一贯执行该硬闸：`run_dtype_grad.sh` porcelain 非空即 `exit 1`）。改法：**先完成 V3.12 全部改动并 commit**（含删除 `shared/mem_buffer.py`），再在 clean HEAD 上用 `git show <BASE>:` 把 A 侧四个源文件提取到 scratch 目录跑 A/B——即原稿 384 行的「兜底 A 侧」升为主流程。**四个文件必须一起提取**（`mem_buffer` + `data_utils` + `posemb_3d` + `siglip_tokenizer`）并改包名，只提 `mem_buffer` 会让 A 侧绑到 B 侧的新实现、退化成自比。这样既不拆 `V3.12a/V3.12b`，也不需要脏工作区。明细见五节 a。
- **删** `shared/mem_buffer.py`、`shared/siglip_tokenizer.py`（训练/在线零使用：在线注入模型 `vision_enc_fn`，建库域有副本）。
- 新增 `shared/sampling.py`：原样搬 `even_sampling_indices`（只 import numpy，不拉 flax——解 dataloader worker 的 jax/flax 导入负担）；`shared/data_utils.py` 删 `even_sampling_indices` 与 `left_padding_token_emb`（唯一消费者已删），保留 `right_padding_token_emb` + `pool_tokens_to_size`；`framesamp_dataset.py` 与 `test_pack_guards.py` 的 import 改指 `shared.sampling`；`spawn_matrix.py` docstring 措辞更新。
- 验证（≤3min）：`grep mem_buffer|siglip_tokenizer` 于 training/policies/models/serving 为空；challenge_interface + policy import 闭环；`import shared.sampling 后 'flax' not in sys.modules`；pytest（test_pack_guards + test_padding_dtype）；STEPS=5 烟测。

### commitV3.13 — scripts/examples 清理

- `eval.sh`：删 symbolic/MemER 分派与菜单，默认 MODEL_TYPE 改 `perceptual-framesamp-modul`；`finetune_mme_vla_suite.sh` 菜单只留三个 framesamp（**`finetune_pi05_baseline.sh` 已在 V3.9 删除**）；`compute_results.py` 删 `--symbolic_type` 及 symbolic 分支、默认 model_dir 改 framesamp-modul。
- **`runs/` → `v1-store/` 收敛 + 裸 `python` 换 `uv run`（二节第 10 条，只动路径与执行器，不动评估逻辑）**：`eval.sh` 的 `runs/ckpts` → `v1-store/train-runs/`、评估客户端的裸 `python` → `uv run`；`compute_results.py` 的 `runs/evaluation` → `v1-store/evaluation/`；`unzip_ckpt.py` 默认 `runs/ckpts` → `v1-store/train-runs/`；`download_pi05_base.py` 显式 `OPENPI_DATA_HOME=v1-store/models`（**禁覆盖 `HOME`**，AGENTS 14）。这几处是 AGENTS 14「派生产物收敛到单一根」与 AGENTS 3「禁裸 python」的存量违规，本刀既然要改这些文件，顺路收敛。
- `examples/robomme/`：eval.py 删 subgoal 全链（import/参数/注入/存档分支）；删 `subgoal_predictor.py` 与 `subgoal_prediction/` 子树；utils.py 删 `SUBGOAL_TYPES` 与 record 的 subgoal 形参；env_runner.py 删两个 oracle property。
- 不动：`build_dataset.py`、`dataset_builder/vlm_subgoal_*`、`finetune_vlm_subgoal_predictor.sh`（建库域/数据生产，冻结）。
- 验证（≤2min）：bash -n；ruff；examples ast.parse；`pytest --collect-only -q src scripts packages` 零 error；`grep -rn "runs/ckpts\|runs/evaluation" scripts` 为空；`grep symbolic|subgoal` **按显式白名单判读**（不是「为空」）。
  - **判据修正（审计）**：原稿要求「`grep symbolic|subgoal`（排除建库域）为空」**不可达，且会诱使执行者去改冻结路径**。实测本刀之后仍会命中三类合法残留：①`framesamp_dataset.py` 的 `data.pop("simple_subgoal_online")` / `data.pop("grounded_subgoal_online")` —— 位于 `__getitem__` 内，属〇节「保留路径逐字清单」的冻结体，**一个字符都不许动**；②同文件顶部说明 `_NONE_KEYS` 由来的注释；③建库/数据生产侧的 `finetune_vlm_subgoal_predictor.sh` 与 `build_dataset.py`（明确冻结不动）。判据改为：命中集合**恰好等于**这三类白名单，出现白名单外的命中才算 FAIL。

### commitV3.14 — scripts/ 目录统一（git mv + 逐文件根解析修正；**不是「纯搬移」**，理由见 6.3）

目标形态：

```
scripts/
  training/                  ← 训练域（finetune_pi05_baseline.sh 已在 V3.9 删除）
    train.py、finetune_mme_vla_suite.sh、eval.sh、serve_policy.py、
    compute_results.py、compute_norm_stats.py、download_pi05_base.py、unzip_ckpt.py、__init__.py
    prod/     ← train-prod/（gl_train_prod.sbatch、prod_train_once.py、downsample_util_csv.py）
    bench/    ← smoke-local/ 四件 + compare_online_memory.py + analyze_gpu_util.py（自 bottleneck-bench-v2）
                 + analyze_util.py（自 dtype-unify）
    tests/    ← test_pack_guards.py、spawn_matrix.py、dump_index_seq.py、test_padding_dtype.py、
                 single_step_grad.py、dump_fixture_samples.py、compare_dtype_fix.py、_common.py、
                 run_dtype_grad.sh
    paths.sh  ← 训练域自带（切断对建库域 paths.sh 的跨域 source）
  dataset/                   ← 数据集预处理域（自包含隔离域）
    gl/       ← data-preprocess-GL/ 整目录（计算路径冻结；paths.sh 与四个 py 的根解析、
                 gl_build_dataset.sbatch/step1_submit.sh 的自引用路径必须随深度改，见 6.3）
    pack/     ← pack_framesamp_store.py、probe_layout.py、run_pack.sh、README
    build_dataset.py、tarxz_h5.py、unzip_data.py、finetune_vlm_subgoal_predictor.sh
```

（顶层散件逐项去向的完整对照表见第一部分四节。）

- 随后删除空壳/退役目录：`bottleneck-bench/` 余部、`bottleneck-bench-v2/`、`dtype-unify/`、`data-pack-framesamp/`、`smoke-local/`、`train-prod/`、`data-preprocess-GL/`。
- **路径修正点：以第一部分 7.2「根解析修正」组与「原稿漏登记的活文件」组为准**（原稿此处只列了六项，实测漏 `check_baseline_env.py`、`paths.sh` 的 `../..`、GL 的 sbatch 与提交器、`train.py` 的 `wandb.log_code`、`greatlakes.md`、`run_dtype_dump.sh`、`README-ZH.md` 等十余项；`pyproject.toml` 要改的也不是 `testpaths` 而是 ruff 的 `per-file-ignores` 四条 glob）。逐条改完按 7.5 三条判定行验收。**历史留档 docs/training-doc/*/launch.md 一律不改。**
- **每个入口加根自证断言**（二节第 11 条）：算出的 `_REPO_ROOT` / `REPO_ROOT` 必须满足 `(root/"pyproject.toml").exists()`，否则立刻 fail-loud。这一条是 `paths.sh` 那类**静默错位**的唯一防线（详见 6.3）。
- 验证（≤5min）：7.5 的 `RELOCATION_REFS` / `RELOCATION_ROOT` / `RELOCATION_COLLECT` 三条判定行；`STEPS=5` 烟测走新路径（三节全局口径）。**原稿的「全仓 grep 旧目录名零残留」作废**——它抓不到顶层散件路径与 `parents[N]` 错位，且因仓库根历史文档必然含旧目录名而永远返回非零（详见 7.5）。
- **预 G3 闸**：SMOKE5 在最终布局上重跑（同 N2 判据，含新增的 `STATE_DIGEST rows=1 mismatch=0`）；按 AGENTS 17 独立留档。

## 二、对拍闸门总表

| 闸门 | 位置 | 内容 | 判据 |
|---|---|---|---|
| N1 | V3.9 tip | 确定性档 STEPS=5 烟测（三节全局口径）→ compare vs G0b | `SCALARS steps=5 hex_mismatch_steps=0` + `INDEX_SEQ=PASS`（`n` 首轮实测钉死，**不是 40**） |
| N2 | V3.10 tip | SMOKE5（带摘要） | SCALARS / raw `BATCH_DIGEST rows=3 mismatch=0` / `CANON_CHECK=PASS steps=3` / **`STATE_DIGEST rows=1 mismatch=0`** / `INDEX_SEQ=PASS`；另单独核对 `batch_digests.jsonl` 首行 `n_keys=12`（比较器不产出该字段） |
| N3 | V3.11 tip | GRAD_FIXTURE vs 固化 A 侧 `docs/training-doc/v1-dtype-p5-grad/records/` | `COMPARE_GRAD=PASS kinds=3 mismatches=0`；驱动须给 `RUN_TAG` |
| N4 | **V3.12 commit 之后、clean HEAD** | ONLINE_MEM：`FrameSampMemory` vs `git show <BASE>:` 提取的旧 `MemoryBuffer` A/B | POS_TABLE / ENC_LAYER / ASSEMBLY 三层逐位 |
| N4c | V3.8 起持续（A 侧钉 `COPY_BASE`） | COPY_DIFF + IMPORT_ISOLATION | 见五节；V3.12 之后 COPY_DIFF 的在岗接力是 `test_guards.py` 的 4 条 sha256 哨兵 |
| 预 G3 | V3.14 tip | SMOKE5 重跑（新布局）+ 7.5 三条迁移判定行 | 同 N2 + `RELOCATION_REFS/ROOT/COLLECT` |
| N5 | V3.14 后 clean HEAD | **G3** 1000 步 vs G0b | 四分项逐行人工核对（六节）；**退出码与总行 `DET_CHECK` 均不具判据资格** |

**失败定位阶梯**（G3 失败不得直接重跑 G3）。原稿只按「便宜→贵」排序，审计后**按是否在 G3 因果路径上分两组**——G3 失配时先跑不在链上的三项是纯浪费：

- **在链上**：`GRAD_FIXTURE`(15min,2卡) → `SMOKE5`(35min,2卡) → **G3 过程中的增量 state 对拍**（每出一次摘要即与 G0b-r1 同步骤比，12 个摘要步天然二分）。前两者只覆盖到第 5 步，而「N1–N5 全过、G3 仍失配」的唯一情形正是**步 5 之后才分叉**——那时只有增量对拍能定位，故它是正式一级。
- **不在链上**（只证建库/在线侧）：`COPY_DIFF`(10s) → `IMPORT_ISOLATION`(1min) → `ONLINE_MEM`(10min,1卡)。

## 三、链路图（AGENTS 18，G3 docs commit 落档）

**交付形态修正（审计）**：AGENTS 18 的字面要求是**重构前、重构后各一张训练链路图**。原稿此处写「两张图」，给的却是训练/在线/建库三条链各一张「前后合一」表——用一张混合表替代前后两图不满足该条。落档时交付：**图 A-before（重构前训练链 25 跳）+ 图 A-after（重构后训练链）两张独立图**，另附**图 B（在线评估链）**与**图 C（建库链）**作为辅助表。下面是三条链的要点（完整表随 G3 留档落 `docs/training-doc/<run_name>/`）：

- **图 A 训练链（25 跳）**：前后基本同构，删除的只有旁路——A1 backend 分派、A2 legacy 数据集、A13 `_NONE_KEYS` 缩减（None 非叶子零贡献，n_keys=12 实证）、A14/A15 Repack/Inputs 去 6 键、A18 tokenizer 死分支、A23 Observation 删字段（叶子集不变）、A24 模型死分支与 stats 链（`stats≡None` 非叶子，论证 HLO 中性、G3 实证）。A3–A12、A16–A22、A25 全部逐字保留并标形状/dtype/字节量（packed 交付 per-batch ≈30.9 MiB；A12/A16 的 state 归一化输出恒 f64；A25 的 info 五标量含 `mem_enc_norm`——scalars_hex 第 5 列，丢了 compare_scalars 会静默跳过）。
- **图 B 在线评估链（15 跳）**：B0 pos 表 1.01GiB→192MiB（只算 4x4 一张，逐位不变仍设 POS_TABLE 判据）、B6 池化 ×3→×1、B8 存储 ≈780KB/步→≈112KB/步、B9 token_drop 整删、B14 客户端 subgoal 注入删除（无对拍，盲区）；B2–B5、B7、B10–B13 执行分支逐字保留；B12 四元组 `(img (512,2048) bf16, pos (512,768) f32, state (512,8) f32, mask (512,) bool)` 为 A/B 判据锚点。
- **图 C 建库链（7 跳）**：C1–C4 只改 import 边、C5 计算跳逐字节保留（COPY_DIFF 证明）、C6 产物 `token_emb_{step}.npy` 7 键 602,951 B 逐位不变、C7 新增反向 import 护栏。

## 四、（并入五节）

## 五、第一块：非训练轻量对拍明细

**c-1 COPY_DIFF（秒级，主判据）**：`git show $COPY_BASE:src/.../shared/$f.py | diff -u - src/.../dataset_builder/$f.py` ×4，`COPY_BASE=732fae3b13e2ff5f485d7014473b99ed577de387`；PASS = 差异只在 import 行且仅 `shared.→dataset_builder.` 替换。判定行 `COPY_DIFF=PASS files=4 nonimport_lines=0 base=732fae3b`。**A 侧禁用浮动 `HEAD`**：V3.12 之后 `shared/` 侧三个源文件或删或改，用 `HEAD` 会直接 `fatal: path does not exist`，把六节前置门锁死（详见 6.1 第 1 层）。V3.12 之后本判据退为「可复算的历史证明」，在岗的是 `test_guards.py` 的 4 条 sha256 哨兵。

**c-2 IMPORT_ISOLATION（秒级）**：静态 grep 双向（建库域不 import shared/training/policies/models/datastore；训练/在线/量具不 import dataset_builder）+ 动态新解释器 `sys.modules` 泄漏断言双向。判定行 `IMPORT_ISOLATION=PASS builder_leaks=0 train_leaks=0 online_leaks=0 whitelisted=1`。
- **量具白名单（审计新增，否则与三方对拍字面互斥）**：`compare_online_memory.py` **必须** import `dataset_builder.mem_buffer`——那正是 6.1 第 3 层「副本与原件行为逐位相同」的唯一 bitwise 证据来源。把它列进 c-2 的显式白名单（唯一一项，判定行给出 `whitelisted=1`），**不得**为了让判定行好看而砍掉三方对拍：砍了这一层就只剩非 bitwise 的 BUILDER_SPOT。白名单只对**测试/量具**开口，训练与在线侧一律 0 泄漏。

**a ONLINE_MEM（8–12min，1 卡 ~10GB，CUDA_VISIBLE_DEVICES=0）**：
- 新脚本 `compare_online_memory.py`（先放 smoke-local，V3.14 随迁 bench/），判定行体例同 compare_baseline。
- 消除编译非确定性：进程内只构造一次 `enc = jax.jit(SigLipTokenizer().__call__)`，同一 callable 注入 A/B 双方（需 `OPENPI_DATA_HOME=v1-store/models`）；不走 `prepare_buffer` 默认自建分支。选 SigLipTokenizer 而非 14GB pi05：更轻且与建库域同编码器，一套 harness 兼做**三方对拍**（旧 `shared.mem_buffer` / 新 `dataset_builder.mem_buffer` / 新 `framesamp_memory` 的 `get_history_feats` 逐位）——c-1 PASS 时 BUILDER_SPOT 降为可选。
- 输入：真实 h5 帧为主（`/data/hongzefu/robomme_data_h5_v2_4env400ep/record_dataset_ButtonUnmask.h5`，(256,256,3) u8——必须真实 256 帧才走 resize_with_pad 跳）+ 合成极值帧验不炸（非判据）。
- 步位网格（覆盖 even_sampling 全分支）：0/1/2、15/16、**30**（恰 1 行填充）、**31**（恰填满零填充）、**32**（首进 linspace 含重复索引）、33/34、100/291、585。喂帧须覆盖各步 `even_sampling_indices(step, 32)` 索引的**并集**——`default_history_feats_gather_fn` 直取 `self._history_feats[idx]`，缺项即裸 KeyError。
- **越界探针改 4096，不是 4095（审计纠错）**：`MemoryBuffer` 的 `max_steps` 默认 **4096**，而在线侧不传该参数 → 4x4 pos 表就是 4096 行，`[4095:4096]` 是**表内合法末行**，A/B 两侧都应正常通过（把它当「越界探针」是误设）。真正的越界是 **4096**，且 numpy 切片越界**不抛异常、静默返回 `(0,16,768)` 空数组**——所以这条判据必须写成「A/B 两侧都显式 raise」，**禁止依赖 numpy 切片行为**。新实现若把有效域缩短到 4096 以下，那是语义变化，不是实现细节。
- 三层判据（复用 `dtype-unify/_common.py` 的 `leaf_sha256`＝sha256(dtype‖shape‖tobytes)，bf16 位型安全；**禁 allclose/==**）：`POS_TABLE=PASS`（4x4 pos 表共同行全表）→ `ENC_LAYER=PASS steps=13 keys=3 mismatch=0`（image_emb_4x4/pos_emb_4x4/state_emb；A 侧多出的 8x8/2x2/image_pixels 为白名单删除不比）→ `ASSEMBLY=PASS steps=13 mismatch=0`（四元组）。分层免费定位编码跳 vs 装配跳。
- >5min 适用 AGENTS 17：产物落 `v1-store/bench/online-mem/<TAG>/`，留档并入 G3 run 目录 `records/block1/`。
- **A 侧取法（主流程，二节第 12 条）**：`git show $COPY_BASE:...` 把**四个**源文件提取到 scratch 目录 + sed 改包名。**必须连 `data_utils`/`posemb_3d`/`siglip_tokenizer` 一起提取**，只提 `mem_buffer` 会让 A 侧绑到 B 侧新实现、退化为自比。原稿把这条列为「兜底（回归重跑时）」、主流程走脏工作区共存——审计推翻：脏工作区跑 8–12 分钟违反 AGENTS 17，而本取法在 clean HEAD 上同样成立，故直接作为主流程，N4 移到 V3.12 commit **之后**执行。

**c-3 BUILDER_SPOT（可选确认档，5–8min，1 卡）**：**必须对 `v1-store/datasets/ref-crossarch`（本机 47 ep 同架构库）跑，禁对 4task-gl**（GL 是 A40 产物，跨架构逐位不可得，已有实证报告）。做法：直接 `import spot_check` 函数调用，n=32、seed=20260829。判定行 `BUILDER_SPOT=PASS n=32 checked=32 max_diff=0.000e+00`。失败第一步跑 null 对照（git 提取旧 mem_buffer 同 picks）分「环境漂移 vs 拷贝错」。**禁跑 `finalize_checks.py` 的 cmd_check 子命令**——无错时会改写目标库 `meta/stats.json` 与 `meta/provenance.json`，对 4task-gl 跑等于污染 G0 数据集指纹、基线作废。

**c-3 的三条执行前提（审计新增；缺任一条这个档位不可执行）**：
1. **抽样池必须先收窄成 47 条子集**。`spot_check(manifest, out_dir, raw_dir, n, seed)` 的抽样源是传入 manifest 的 `episodes` 全体、库内不做过滤，缺文件即计入 errs。全局 `v1-store/episode_manifest.json` 有 **1600** 条而 `ref-crossarch/features` 只有 **47** 条——直接传全局 manifest，32 抽的命中率约 3%，期望约 31 条落进「抽检缺文件」分支必 FAIL；且 `worst` 只在比对成功时更新，会打出「FAIL 但 `max_diff=0.000e+00`」的误导组合。做法：按 `ref-crossarch/meta/_shard0of1.json` 的 47 条 episodes 过滤出子集 manifest 再传入。
2. **`raw_dir` 与 `OPENPI_DATA_HOME` 都要显式给**：`spot_check` 内 `prepare_buffer=True` 会走自建 `SigLipTokenizer` 分支，而 `siglip_tokenizer.py` 读 `OPENPI_DATA_HOME` 时**不做 `expanduser`**，给 `~/...` 形式会静默找不到权重。
3. **判定行区分 `n` 与 `checked`**：`n` 是请求抽样数，`checked` 是真正完成比对的条数，两者不等即说明抽样池仍不匹配，而不是「代码有问题」。

**b-1 GRAD_FIXTURE / b-2 SMOKE5**：见一节 V3.11/V3.10 闸门。**执行顺序（审计修正）**：原稿写「a 与 c-3 须在 b-1 之前完成」，但 N4（=a）已按二节第 12 条移到 **V3.12 commit 之后**，而 b-1 在 V3.11 tip——顺序自然变为 `b-2(V3.10) → b-1(V3.11) → a(V3.12 后)`，各自绑在自己那一刀的 tip 上，不再有「谁必须先于谁」的额外约束。仍然成立的纪律：**b-1/b-2/a 与 G3 不得并行**（显存与 autotune 互扰）；c-1/c-2 是秒级静态检查，任何时候都可跑；c-3 可与 a 同机并行。第一块合计 ≈55–75 min。

## 六、第二块：G3 runbook

**前置门（全过才起跑）**：工作区 porcelain 空、从 clean HEAD 起跑；第一块五判定行全 PASS（COPY_DIFF/IMPORT_ISOLATION/ONLINE_MEM/GRAD_FIXTURE/SMOKE5）；**7.5 三条迁移判定行全 PASS**（`RELOCATION_REFS`/`RELOCATION_ROOT`/`RELOCATION_COLLECT`）；bench 四道源码护栏 + `TorchDataLoader.__iter__` 三 needle 在新 train.py 上仍成立；`train_step` info 仍含 `mem_enc_norm`**且 `scalars_hex.tsv` 表头为六列**（R17：比较器缺键只会静默跳过、判定行仍打 `keys=5`）；packed 库 `status=verified`；单 epoch 1000×8=8,000 < 395,289；**run_name 起跑前经用户确认**（AGENTS 6；建议 `v1-postclean-g3`，EXP_NAME=RUN_TAG 独立编译缓存）。

**前置门的两条审计新增**：

1. **环境里不得残留任何 `MMEVLA_FRAMESAMP_*`**，判据 `env | grep MMEVLA_FRAMESAMP` 输出为空。原稿只列 `MMEVLA_FRAMESAMP_ALLOW_*`，而 `_create_framesamp_dataset` 还读 `MMEVLA_FRAMESAMP_SOURCE`（换源库根）、`MMEVLA_FRAMESAMP_MANIFEST`（换 manifest）、`MMEVLA_FRAMESAMP_VERIFY`（改校验档）——残留一个 `SOURCE` 就从另一个源库读 pkl，**而 preflight 的 `dataset_spot` 锚在 `--dataset` 参数上、不会失配**：指纹全绿、交付字节已变。
2. **packed 库本体要单独钉一条**。preflight 的 `dataset_spot` 指纹读的是 legacy 源库 `4task-gl`（packed 库无 `data/`），**packed 库本体不在任何指纹里**：若它在 G2 与 G3 之间被重建（新 meta 与新 manifest 自洽、`status=verified` 照过），四个 `static_*` 键必变、G3 全线失配而前置门全绿。现成锚点已存在但原稿没用——`run_2gpu_epoch_bench.sh` 会把 `store_meta_sha256` 与 `manifest_sha256` 写进 env.json **顶层**（而 `cmd_check` 只比 `fingerprint` 子树）。起跑前人工比对本轮这两个值与 `docs/training-doc/v1-framesamp-g2/records/env.json` 一致。

**preflight（必须带环境变量——指纹采 XLA_FLAGS / XLA_PYTHON_CLIENT_MEM_FRACTION / CUDA_VISIBLE_DEVICES，裸跑必 FAIL 三项）**：

```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=0,1 \
UV_LINK_MODE=copy uv run scripts/training/bench/check_baseline_env.py check \
  --baseline docs/training-doc/v1-grad-baseline-g0b/records/r1 \
  --dataset v1-store/datasets/4task-gl --steps 1000 --batch-size 8
```

`--dataset` 必须指 legacy 源库 4task-gl（packed 库无 data/ 会炸；packed 训练仍从源库读 pkl，源库指纹才是锚）。判读纪律写进 launch.md：指纹不含仓库代码 sha，**preflight PASS 只证「引用 G0 的资格还在」，不能当「没改坏」的证据**。

**起跑（tmux，AGENTS 7；V3.14 后新路径）**：照 G2 模板（`docs/training-doc/v1-framesamp-g2/launch.md`）去掉 `MMEVLA_DATA_BACKEND` 行：

```bash
tmux new-session -d -s <run_name> "set -o pipefail; cd <REPO_ROOT>; \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 \
  EXP_NAME=<run_name> RUN_TAG=<run_name> \
  DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
  PYTHONUNBUFFERED=1 bash scripts/training/bench/run_2gpu_epoch_bench.sh \
    2>&1 | tee v1-store/logs/<run_name>-driver.log; \
  echo \"EXIT_CODE=\$?\" >> v1-store/logs/<run_name>-driver.log"
```

摘要步集与 G0 对齐（state 12 次：0/每 100/299/999；batch 14 行）。预计 2–2.5h。launch.md 注明 env.json 不再含 backend 字段（量具字段变更，preflight 不比对该字段）。

**tmux session 名用 `<run_name>`、不用固定的 `g3`**（审计新增）：起跑前先 `tmux has-session -t <run_name>` 确认不存在；需要第二轮时用 `<run_name>-2`。固定名在补跑第二轮时会与前一轮撞名，而 AGENTS 7 判死活正是靠 `has-session`。配套：V3.9 已把失败 record 从 `rm -rf` 改成 `.failed-<n>` 原子改名，**首轮 correctness FAIL 的记录永久保留**，第二轮只用于诊断、不得覆盖首轮结论。

**盯日志**：Monitor 挂 driver.log，`grep --line-buffered` 过滤 `^Step (0|[0-9]*00|299|999):|BENCH_|EXIT_CODE|Traceback|Error|OOM|digest`；中间过滤级一律行缓冲（stdbuf -oL tr / awk fflush / sed -u）。**过程中增量对拍**：每出一次 state 摘要即与 G0b-r1 同步骤比 `state_digest`，首次分叉立刻停跑，省 1–2h。

**对拍判读（四分项，G2 先例）**：

```bash
UV_LINK_MODE=copy uv run scripts/training/bench/compare_baseline.py \
  docs/training-doc/v1-grad-baseline-g0b/records/r1 \
  v1-store/bench/2gpu-epoch-bench/<run_name> --tier g3-vs-g0b \
  | tee docs/training-doc/<run_name>/records/compare_vs_g0_r1.txt
```

① `SCALARS steps=1000 keys=5 hex_mismatch_steps=0`；② `STATE_DIGEST rows=12 mismatch=0`；③ `BATCH_DIGEST_CANONICAL rows=14 mismatch=0` + `CANON_CHECK=PASS steps=14`；④ `INDEX_SEQ=PASS`（前 8,000 条前缀一致；`n` 与 G2 同为 8072 时逐字吻合）。**不作判据**：raw `BATCH_DIGEST mismatch=4 first_bad_step=100 bad_keys=2 (static_image_emb/static_pos_emb)`——V2.4b dtype 统一的已知预期失配，与 G2 逐字吻合即正常，**不是 4 步×2 键则升格为信号**；总行 `DET_CHECK=FAIL` 是已拍板不修的工具聚合缺口。

**四分项必须逐行人工核对，并补三处 fail-open（二节第 13 条已定不改量具，纪律照抄进 `launch.md`）**：

- **`--tier` 只是标签**：`compare_baseline.py` 的 `--tier` 默认 `adhoc`、仅打印在 `DET_CHECK` 行，**不切换任何判据严格度**。写 `g3-vs-g0b` 不会让比较更严格。
- **补位 1（`keys=5` 是常量）**：`compare_scalars` 对缺失的键直接 `continue` 却仍固定打 `keys=5`。拿到 ① PASS 后**另行核对** `records/scalars_hex.tsv` 表头恰为 `step\tloss.hex\tgrad_norm.hex\tllm_grad_norm.hex\tmem_enc_norm.hex\tparam_norm.hex` 六列，缺列即判 FAIL（最可能缺的是 `mem_enc_norm`，R17）。
- **补位 2（`INDEX_SEQ` 只比最短公共前缀）**：`n = min(sa["n"], sb["n"])`，一侧被截短也可能打 PASS。核对 G3 侧 `index_sequence.json` 的 `n` 不小于 G0b 侧且 `n ≥ 8000`。
- **补位 3（退出码不可信）**：最终 `verdict` 只由 scalar/state/**raw** batch 决定，canonical 与 index **不进** verdict 与退出码，而 raw 又是已知 FAIL —— 所以**总行与退出码在 G3 里都不具判据资格**，一律以四分项逐行为准。
- 额外必检 `n_keys=12`：比较器**不产出**该字段，直接核对本轮 `records/batch_digests.jsonl` 首行。

**收官**：`scalars_hex.tsv` 手工投影（表头 `step\tloss.hex\tgrad_norm.hex\tllm_grad_norm.hex\tmem_enc_norm.hex\tparam_norm.hex`，1001 行，末尾单换行）→ sha256 必须等于 `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`（G0b r1/r2、G1、G2 四份同值——一行收官）；`check_baseline_env.py manifest` 生成防腐清单；留档 `docs/training-doc/<run_name>/`（launch.md/result.md/records/ 按 G2 体例 + `records/block1/` 收第一块判定行与产物）；AGENTS 18 两张链路图 + 两块一致性讨论随 docs commit 落档。

**轮数**：一轮。G0b 两轮千步自证 + G1/G2 同 sha 四份独立证据已钉死本管线复跑抖动为严格零；任一分项 FAIL 才补跑同 commit 第二轮（分离「语义变了」vs「非确定性回归」）。util/步时不作性能结论；性能另跑 `v1-g3-speed`（speed 统一口径），不在本计划范围、届时另行确认。

## 七、风险登记

| # | 风险 | 规避 |
|---|---|---|
| R1 | 保留路径被「顺手简化」，G3 失败 | 〇节保留路径清单逐条 review 打勾 |
| R2 | 删 `MMEVLAWeightLoader` 连带删 import，`PaligemmaTokenizer` 的 `download.maybe_download` 运行时 NameError（ruff 不报） | 先改 `_download.maybe_download` 再删 import 行；验证构造 tokenizer |
| R3 | 换 openpi `TokenizePrompt` 静默改 pi05 discrete 口径（分号 vs 逗号文本） | 禁换，保留本地实现 |
| R4 | `train_step` 返回注解未同步（@at.typecheck 报错浪费一次冷编译） | 注解与 return 同 hunk |
| R5 | bench/prod/compute-only 源码指纹护栏被打断 | 四字符串一字不动；openpi data_loader.py 不碰 |
| R6 | 删 base.yaml 后 history_config schema 无处可查 | context yaml 加中文注释或记入 datastore/README；含 sha 口径须 G3 前完成 |
| R7 | 验证 compute_norm_stats 覆盖生产 norm_stats.json（G0 指纹项） | 加 `--output-dir`；动手前 sha256 备份 |
| R8 | 冷编译撑爆 5min 验证 | 各 commit 复用 `EXP_NAME=v1-restructure-smoke` + `KEEP_JAX_CACHE=1`；超时进 tmux |
| R9 | pytest 收集失败（testpaths 含 scripts） | 每 commit 末 `pytest --collect-only -q src scripts packages` 零 error |
| R10 | `_NONE_KEYS` 与 Repack 键集合脱钩 | 两处同 commit；烟测第一个 batch 即暴露 |
| R11 | `set -u` 下 sbatch 引用已删变量 | export/echo 同删 + `bash -n` + `grep -c` 为 0 |
| R12 | examples 删 subgoal 后遗留 import | `grep -rn subgoal examples/robomme` 为空 |
| R13 | 建库域副本与 shared/ 悄悄发散 | V3.8 起冻结声明 + test_guards sha256 哨兵用例 |
| R14 | graphdef 变更被误判为回归 | 对拍判据不纳入 `nnx.split` static 侧 |
| R15 | preflight 裸跑必 FAIL（指纹采三个环境变量） | 必带 XLA_FLAGS / MEM_FRACTION / CUDA_VISIBLE_DEVICES |
| R16 | `finalize_checks` cmd_check 改写 meta 污染 G0 数据集指纹 | 禁跑该子命令；只 import `spot_check()` 对 ref-crossarch 调用 |
| R17 | `mem_enc_norm` 丢失后 compare_scalars 静默跳过该键、判据变弱不报错 | 前置门显式检查 info 键 |
| R18 | ONLINE_MEM 的 B6 跳被包进新 jax.jit（融合变 → bf16 累加序变位） | 新模块保持 encode 与 pool 分离调用，同旧实现 |
| R19 | 烟测漏给 `WARMUP_STEPS=0`/`SAVE_INTERVAL=5`/`DATASET_PATH`，训练跑完后崩在收尾统计或判据无从产出，症状与改动无关、误导排障 | 三节全局烟测口径；四刀命令逐条内联，动手前对照 |
| R20 | `--force` 闸只挡「误删已存在目标」，不做 canonical containment；显式 `--force` + 错路径仍可删 `v1-store` | 二节第 8、14 条已拍板本轮只做 `--force`；containment 白名单、软链接祖先、挂载点检查另开一轮，本条为已知未处置项 |
| R21 | `EXP_NAME`/`RUN_TAG` 未校验即参与路径拼接，清理处 `case` 模式与被匹配值由同一套变量展开、属词法自比，挡不住含 `../` 的取值 | **本轮明确不处置**（二节第 14 条）：概率低且 run_name 每次经用户确认。已知未处置项 |
| R22 | preflight 的 `provenance` 实际没进指纹（finalize 写 `meta/provenance.json`，`_dataset_spot_digest` 读根下 `provenance.json` 且 `if p.exists()` 静默跳过），而 scheme 名仍写着含 provenance | **本轮禁止修这个路径**：一改指纹口径，G0b 留档记录的指纹当场失配、`BASELINE_ENV=FAIL`、G0 基线作废，而整个 G3 对拍正建立在能引用 G0 上。只登记进八节盲区，修复留到 G3 之后 |
| R23 | jax 编译缓存复用（`KEEP_JAX_CACHE=1` + 同 `EXP_NAME`）在代码变更后命中旧缓存，造成假 PASS | 各刀烟测共用 `EXP_NAME=v1-restructure-smoke` 是为省冷编译；**G3 用独立 `EXP_NAME=RUN_TAG=<run_name>`、不复用任何烟测缓存**（D2-cold 已证跨重编译 bitwise 成立，G3 现场重编译不影响对拍） |
| R24 | `git mv` 后 `__pycache__` 残留旧模块，import 到已删模块而不自知 | V3.14 搬移前清 `find scripts src -name __pycache__ -type d`（不进 git，删之无损）；`RELOCATION_COLLECT` 在清理后跑 |
| R25 | 七刀中途中断（机器重启/被打断），工作区停在半刀状态 | 每刀自身是可运行的完整状态（各刀末尾都有烟测闸）；中断后以 `git status --short` + 最近一次 commit 为准回到刀边界，不在半刀状态上继续 |

## 八、对拍盲区诚实清单（写入 G3 result.md）

1. modulation/expert 集成无基线（中）：G 链只锚 context；modul/expert 分支只能 `git diff` 源码级论证「逐字未动」，不得宣称已验证。
2. examples symbolic 删除无行为对拍（中）：只有语法/import 冒烟；复活需从 git 历史取回。
3. 在线整链（B13 之后到 sample_actions）无端到端 A 侧（中）：执行分支逐字保留 + 本轮不改 sample_actions，源码级论证 + 评审兜底。
4. 建库域 GL 侧不重跑（中）：BUILDER_SPOT 只证本机同架构复算一致；COPY_DIFF 把风险降为「源码同一 + 环境未变」。
5. ONLINE_MEM 用 SigLipTokenizer 桩而非 policy 真实注入路径（低）：注入点一行未改，源码级论证。
6. `has_aux` 改动无 HLO 级直接 diff（低-中）：GRAD_FIXTURE 是最强前置证伪；可选 HLO_DIFF（需重构前预存 A 侧）默认不做。
7. treedef 变化必然重编译（低）：D2-cold 已授权跨编译 bitwise 对拍。

**审计新增的六条（原稿未登记或登记不足）**：

8. **`compute_norm_stats.py` 的内联读取器无任何对拍**（中）：`norm_stats.json` 在 G 链里是**输入常量**（被 `check_baseline_env.py` 固化进指纹），全链一行都不执行 `compute_norm_stats.py`。内联读取器的口径差要到**下次重算 norm stats** 才显形，届时所有训练的输入归一化被改。本轮只有 V3.9「等价性八要点」的源码级论证 + R7 的覆盖防护，**没有新旧输出对拍**。
9. **preflight 的 provenance 实际未进指纹**（中）：见 R22。本轮不修，如实登记——「数据集指纹含 stats+provenance」这句在今天只有前半成立。
10. **比较器的三处 fail-open**（中）：缺标量静默跳过但仍打 `keys=5`、`INDEX_SEQ` 只比最短公共前缀、canonical 与 index 不进 verdict 与退出码。本轮不改量具（二节第 13 条），靠六节的人工补位纪律兜底——**这是一层人工防线，不是机器防线**。
11. **`--tier` 不切换严格度**（低）：它只是打印标签。任何「用 `--tier g3-vs-g0b` 就更严格」的读法都是错的。
12. **`run_name` 与建库输出路径的破坏性护栏不完整**（中）：见 R20、R21，均为本轮明确不处置项。
13. **G3 只覆盖被实际执行到的跳**（中）：1000 步 / b8 / seed 42 / context 集成 / 本机 2×RTX 6000 Ada。凡未被执行的改动——modul/expert 构图、在线侧、`examples/`、建库域、`compute_norm_stats`、`sample_actions` 三分支——bitwise 一律沉默。这不是缺陷而是边界，但**不得**被「G3 全绿」的结论吞掉：result.md 必须把这些路径显式标 `UNVERIFIED`。

## 九、留档与 commit 纪律

- 每 commit：`git status --short` 核对 → 逐文件 `git add` → 中文 subject（commitV3.8–V3.14）+ 详版 body（AGENTS 11 六要素）。
- G3 >5min：AGENTS 12/17 留档；run_name 起跑前确认（AGENTS 6）；tmux 模板 + Monitor 行缓冲过滤（AGENTS 7）。
- 烟测临时 run（v39-smoke 等）验证完按 AGENTS 6 清理。
- **凡 >5 分钟的闸门一律各自独立留档**（审计补全；原稿只安排了 G3 与 ONLINE_MEM）：
  | 闸门 | 时长 | 留档 |
  |---|---|---|
  | N2 SMOKE5（V3.10 tip） | 30–40min | `docs/training-doc/<tag>/` 的 launch.md / result.md / records/ |
  | N3 GRAD_FIXTURE（V3.11 tip） | ~15min | 同上 |
  | N4 ONLINE_MEM（V3.12 commit 后） | 8–12min | 产物落 `v1-store/bench/online-mem/<TAG>/`，留档并入 G3 run 目录 `records/block1/` |
  | 预 G3 SMOKE5（V3.14 tip） | 30–40min | 独立 `<tag>`，不覆盖 N2 那轮 |
  | G3（V3.14 后 clean HEAD） | 2–2.5h | `docs/training-doc/<run_name>/` 全套 + AGENTS 18 的四张链路图 + 两块一致性讨论 |
  这四项都在 commit tip 上跑、不触发 clean-HEAD 冲突（N4 已按二节第 12 条移到 commit 之后），只是留档条目原稿漏列。
- **失败记录不得清理**：V3.9 起失败 record 改为 `.failed-<n>` 原子改名保留（二节第 9 条），它是定位分叉的唯一证据，**不适用**「烟测临时 run 验证完清理」那条。
