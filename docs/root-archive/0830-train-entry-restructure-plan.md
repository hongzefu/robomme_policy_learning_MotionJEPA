# v5.0 训练入口重构计划：train.py 单跑收敛 + prod 直调 + bench 拆分 g0/util + 双重对拍

> **本文件是 v5.0 工作的唯一权威文档**（2026-08-30 第三版定稿）。版本沿革：第一版（`v5.0-unified-entry-plan.md`，见 git 历史）是「新增 `main.py` 统一入口」方案，同日被用户否决；第二版经 `git mv` 更名为本文件，改为本方案；**第三版（本版）是两轮对抗性审计后的修订定稿**——Claude 侧 24 agent workflow 审计（8 路取证 + 逐条反驳 + 综合）与 Codex 侧独立审计各出一份报告，用户逐项裁决后落档。commit 编号沿用 **commitV5.0**。
>
> **锚点**：分支 `v1-dataloader-Restructure`，第三版定稿时代码与 G3 收官 `0eb69c1` 零差异（第二版落档 commit `1dec4b7` 只动本文件），工作区 clean。上游锚点：`RoboMME/robomme_policy_learning` 的 main、fork（`hongzefu/robomme_policy_learning_MotionJEPA`）的 main、本地 main 三者同为 **`ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`**（ls-remote 实测同 sha），且恰为本分支分叉点（`git merge-base` 同值；分支落后 main 零 commit、领先 122 commit；本地已有 main 分支，无需 fetch）。
>
> **当前状态**：第三版计划待用户放行后实施；**C2（代码改动）、C2.5（记录器 smoke）、C3（G0 重锚 run）、C4（上游对拍 run）均未启动，待用户逐项另行指令**。
>
> **红线（第三版修订）**：`scripts/training/g0/`（原 bench/）量具族 **可执行逻辑与判据零改动，仅允许路径与文案 hunk**（第二版原文「内容零字节改动」与第七节「应改自述」清单自相矛盾，本版按实际可执行范围重述）；G0b 固化基线产物一个字节不碰；一切数值等价主张以位级判据机器裁决，判读不依赖人眼逐行核对（新增 `g0_gate.py` fail-closed 总闸承担）。

---

# 第一部分（给人看）

## 一、Context 与用户拍板

### 1.1 首轮七点（2026-08-30 原话）

用户否决 main.py 统一入口方案（「不要这个main了！」「不再需要main」），改为七点：

1. 六个脚本（`train.py`、`serve_policy.py`、`compute_norm_stats.py`、`compute_results.py`、`unzip_ckpt.py`、`download_pi05_base.py`）各自独立调用，不建统一分发器。
2. 「train py破坏性更改 改为对齐 scripts/training/prod 集群提交训练的模式」+「train py 需要尽可能简洁 不要再搞两次训练的调用」——`train.py` 本体改单跑正式训练语义（≡ 现 `prod_train_once.py`）。
3. 「prod 这里不要再提供训练用的python文件 直接调用」——删 `prod_train_once.py`，集群 sbatch 直调 `train.py`。
4. 「bench 拆分成两个文件夹 一个看util一个对比g0」。
5. 「对train py做最小规模的修改 要再次完整测试g0对比」。
6. 「另外还要实现和 https://github.com/RoboMME/robomme_policy_learning 的main branch的对比」（该仓库 main 与 fork/本地 main 同 sha，锚点唯一，无歧义）。
7. 「出详细方案」「直接改」→ 后收敛为「改回md 不直接启动」（先落档计划，实施另行指令）。

### 1.2 两轮对抗性审计后的修订拍板（2026-08-30，同日第二轮）

用户先后指令：「启动workflow对抗性验证」→「加上codex的结论 / 综合决策 / 哪些需要用户决策或偏离原方向 哪些可以自主改 / 说人话 先让用户浏览」→ 逐项裁决：

| 议题 | 用户裁决 |
|---|---|
| 新记录器全程零执行 | **补一次本机 smoke**（本版 4.4 / 第八节） |
| 删 prod 后丢失的护栏 | **选 (a)**：`train.py` 保留 fail-loud（`use_history` + `overwrite/resume`） |
| 官方示例改后仍跑不通 | 「无所谓 不用改官方」→ **收窄主张**，不动 `README.md` / `finetune_mme_vla_suite.sh` |
| sbatch 把 analyzer 崩溃伪装成成功 | **修最小版**（本版第十一节） |
| AGENTS 18 闭环 | 「d5做」→ **三件全做**：数据逐跳图（选 (a) 照字面画）、第一块写明适配解释、新增 `G0_EQ` fail-closed 总闸 |
| C4 判据加固 | **三个漏洞全修**（摘要域、resolved-config、来源污染） |
| C4 步数 | 「改为和g0一样跑1000步」→ **A/B 两侧各 1000 步** |
| 其余文档精度 9 项 | 「其他都同意」→ 自主改 |

**审计结论中被驳回、不再处理的方向（6 条）**：`--model.use_history` 下划线写法（tyro `_strings.swap_delimeters` 归一化，能跑）；「tentative 零影响缺实证」（依据是三条静态代码事实，且计划自述这就是待验证对象）；「G3 墙钟 54min 无出处」（实为同配方基线 G0b-r1 的 loop 墙钟 54.83 min，属定语漏写）；「wandb 代理被 step 0 的 `camera_views` 搞崩」；「sbatch 头部旧引用未列入清单」；「`batch%fsdp` 护栏丢失」（见三.7，有两道等价兜底）。

### 1.3 背景事实（探索与审计实测，支撑各设计决策）

- `train.py` 官方 `__main__` 双跑**照现状必崩**：第一次调用（tentative）已建 checkpoint 目录，第二次在默认 `overwrite=False/resume=False` 下走进 `src/openpi/training/checkpoints.py` 的 `initialize_checkpoint_dir` 的 `FileExistsError` 分支（该函数的 `checkpoint_dir.mkdir(parents=True, exist_ok=True)` 在 if 分支外无条件执行）；官方 shell 不带 `--overwrite`。`prod_train_once.py` 文件头登记为「正式入口的可运行性阻断」，v1 全部 e2e 走 bench、集群走 prod 单跑，从没撞上。
- tentative 预热对正式轨迹的影响**是 C4 的被测对象、不是本计划的前提**。支持「零影响」的是三条静态代码事实：`rng = jax.random.key(config.seed)` 每次从 config 纯函数重派生；`--overwrite` 触发 `checkpoint_dir.rmtree()` 后重建；tentative 的 `break` 位于 `save_state` 块之前。C4 A 侧 1000 步就是来实测这条从未跑过的路径。
- bench 与 prod 的量具护栏全是 `inspect.getsource(train.main)` 上的**子串探针**（`wandb.log(reduced_info` / `_checkpoints.save_state(` / `init_train_state(` / `create_data_loader(`，另一道锚在 openpi 的 `TorchDataLoader.__iter__`）——审计实测确认：全族无 `sha256|hashlib|sourcelines` 对源码做指纹，四个目标子串所在行与本计划删除集**逐行零重叠**，删签名参数不触发误判。
- `metrics.jsonl` 是 `analyze_gpu_util.py` 的唯一硬依赖（`load_metrics` 无兜底 `FileNotFoundError`），现网唯一生产者是 prod 的 `_WandbProxy`——且 prod 是 `from bench_train_steps import _WandbProxy` **复用同一份代码**，不是另起一份。
- 环境同一性：分叉至今 `uv.lock` 零 diff；`pyproject.toml` 仅 +9 行 ruff lint 豁免。
- 没有任何 python import `prod_train_once`；bench 内 6 个 py 彼此零 Python import；全部目录层深假设为 `scripts/training/<X>/`，同层拆分免改 `parents[3]`。
- **`train.main` 的真实调用点只有 2 个**（`bench_train_steps.py`、`prod_train_once.py`），不是三个：`tests/single_step_grad.py` 虽 `import train as _train`，但只用 `inspect.getsource(_train.train_step)` 与 `_train.init_train_state(...)`，从不调 `.main(`；`tests/` 其余文件不 import train。两个真实调用点均以单位置参数直调、无 `tentative_run=` 关键字。
- **官方示例文档在本计划改动后仍跑不通**（与双跑无关的两处独立阻断）：`finetune_mme_vla_suite.sh` 与 `README.md` 均写死 `MME_VLA_TYPE="perceptual-framesamp-modul"`，而 `src/mme_vla_suite/training/framesamp_dataset.py` 的形制断言硬要求 `integration_type == "context"`，modul 配置必 raise；两处 `--dataset-path` 又是 legacy 目录（`data/robomme_preprocessed_data[_sample]`），而 `create_data_loader` 已单一化为 packed `_create_framesamp_dataset`、要 `StoreMeta.load()` 读 `meta/store_meta.json`。用户裁决不修官方示例，故本计划的主张收窄为**仅消除双跑 `FileExistsError`**。

## 二、改动总览（目录前后）

```
【前】scripts/training/
├── train.py              # __main__ 双跑（tentative+正式），照现状必崩
├── bench/                # 8 文件混居：G0 量具 + util 分析 + N4 对拍
├── prod/                 # prod_train_once.py + gl_train_prod.sbatch + downsample_util_csv.py
└── tests/ 等

【后】scripts/training/
├── train.py              # ★唯一破坏性修改：单跑、删 tentative、两条 fail-loud、自带可选 metrics 记录器
├── g0/                   # ★bench/ 更名拆出：G0 对拍量具族（git mv，可执行逻辑零改动）
│   ├── bench_train_steps.py   run_2gpu_epoch_bench.sh   check_baseline_env.py
│   ├── compare_baseline.py    compare_online_memory.py  README.md
├── util/                 # ★新目录：GPU 利用率观测族（git mv）
│   ├── analyze_gpu_util.py    analyze_util.py   downsample_util_csv.py（原在 prod/）
├── prod/
│   └── gl_train_prod.sbatch   # ★唯一残留：直调 train.py + 退出码硬化；prod_train_once.py 删除
├── tests/                # +3 harness：g0_gate.py / entry_equiv.py / run_entry_equiv.sh
└── 其余顶层脚本不动
```

`compare_online_memory.py` 归 `g0/`：它是 N4 非训练对拍量具（不在 G 链上，但判定行体例同 `compare_baseline.py` 族，且是 bench 内唯一依赖 `tests/_common.py` 的文件）。

**新增 harness 一律放 `tests/`，不放 `g0/`**——`g0/` 的定位是内容冻结的量具族，往里加新文件会让红线边界变模糊；`tests/` 本就可演进（已有 `_common.py`、`test_padding_dtype.py`，且 `compare_online_memory.py` 已依赖它）。

## 三、train.py：最小破坏性修改（对齐 prod 单跑语义）

1. **删 tentative 机制三处**：`main()` 签名的 `tentative_run: bool = False` 参数、`tentative_run_step = 10` 定义块与 `tentative_run_step += config.resum_ckpt_id` 偏移行（保留 `start_step += config.resum_ckpt_id`）、循环内 `if tentative_run and step > tentative_run_step: … break` 块。数值无关论证：这些行在 `tentative_run=False` 下本就是死代码；两个真实消费者均以 `train.main(config)` 单位置参数直调，签名删默认参数不影响。
2. **删死 import**：`from openpi.training.optimizer import CosineDecaySchedule`（全文无引用）；`import time` 保留（新记录器 `wall_time` 用）。
3. **`__main__` 改单跑**：`main(_config.cli())` 一次，无 sleep、无二次调用——`FileExistsError` 阻断就地消失。**主张边界（第三版收窄）**：本改动**只**消除双跑造成的 `FileExistsError`；官方 shell 与 README 的示例因 `perceptual-framesamp-modul` 配置与 legacy `--dataset-path` 两处独立阻断**仍不可跑通**（见 1.3 末条），修复它们不在本计划范围（第十三节）。
4. **保留两条 fail-loud 护栏**（用户裁决 (a)，置于 `main()` 体最前、早于权重加载与 JIT）：

   ```python
   if config.overwrite or config.resume:
       raise ValueError("本入口禁用 --overwrite / --resume：续跑语义有损（checkpoint 只存 EMA，丢 AdamW 动量与 warmup 计数）")
   if not config.model.use_history:
       raise ValueError("正式训练必须启用 --model.use-history")
   ```

   `use_history` 这道是三道里最不可省的：`src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0.__init__` 的 `else` 分支专门写了 `# safe setting` 把 `history_config`/`integration_type` 置空——关掉它**一定跑得通、一定不报错**，会静默训出不含记忆分支的模型，而 `analyze_gpu_util.py` 通篇不读这些字段、判读链路零告警。
   ⚠ 该护栏与 C4 A 侧的 `--overwrite` 冲突，但 A 侧跑的是**上游代码**（无此护栏），B 侧不带 `--overwrite`，两不相犯；`bench_train_steps.py` 自带的同款护栏也不受影响。
5. **自带可选 metrics 记录器**（模块级新增，`train.main` 函数体不碰）：环境变量 `TRAIN_RECORD_DIR` 设置时，`__main__` 路径把模块全局 `wandb` 换成透明代理——逐 log 追加 `metrics.jsonl`（行格式 `{"step":…, "wall_time":…, "<key>":{"dec":…, "hex":float.hex()}}`，与 bench `_WandbProxy` 逐字段相同），finally 写精简 `run_meta.json` 并 `wandb.finish()`（异常吞掉）。未设时零行为差异。防覆盖护栏沿袭 prod 踩坑结论：**先 `mkdir(parents=True, exist_ok=True)`，再只按 `metrics.jsonl` 是否已存在判**，不按目录非空判（job 59092143 实测教训：sbatch 会先建目录、写 env.json、起五路采样器，那时目录里已有 6 个文件）。
6. **量具零扰动**：四探针所在行均不在删除集（审计逐行比对确认）；`bench_train_steps.py` 可执行逻辑零改动，speed 链与 G 链构造性不受扰。
7. **舍弃项（明示，第三版修订）**：
   - **ckpt 目录预检**不进 train.py——sbatch 侧 `[ -e "$CKPT_DIR" ]` 检查仍在，且经 `checkpoint_base_dir/name/exp_name` 推导确为同一路径。
   - **`batch % fsdp_devices` 护栏无需迁移**：`train.py::main` 体最前已有 `if config.batch_size % jax.device_count() != 0: raise ValueError(...)`，`src/openpi/training/sharding.py::make_mesh` 另有 `if jax.device_count() % num_fsdp_devices != 0: raise ValueError(...)`。由整除传递性（F|D 且 D|B ⟹ F|B），两道同时通过即保证 `batch_size % fsdp_devices == 0`，且这对任意拓扑成立、fail-loud 点比原护栏更早。
   - **`_CacheEventCounter`** 编译缓存计数不进 train.py（bench 量具自带的那份继续用；审计确认 `run_meta.json` 在四个下游脚本中零引用）。
   - prod 文件头四条坑登记（双跑必崩 / bench 不落权重 / 不能 patch `wandb.log` / record dir 不能走命令行）迁入 sbatch 头注释与 `g0/README.md`。

修改前后的逐步流程图、g0/util 两族量具全景对照、以及 AGENTS 18 要求的数据逐跳链路图，见文末**附录 A–D**。

## 四、验证设计（四道，执行顺序 4.3 → 4.4 → 4.1 → 4.2）

### 4.1 G0 完整重锚（「再次完整测试g0对比」）——照 G3 runbook 原样重跑

口径与 G3（`docs/training-doc/v1-postclean-g3/`）逐项相同：**1000 步**、b8、2 卡 fsdp2、workers 4、seed 42、确定性档 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、packed 库、`SAVE_INTERVAL=100`（驱动转译为 train 侧 `--save-interval 1` + 记录器侧 `BENCH_DIGEST_INTERVAL=100`）、`EXTRA_DIGEST_STEPS=299`；驱动换新路径 `scripts/training/g0/run_2gpu_epoch_bench.sh`；参考墙钟约 54 min（取同配方基线 G0b-r1 的 loop 实测 54.83 min；G3 留档本身未记耗时，性能另跑 `v1-g3-speed`），预算 1–2 h。

**定位（第三版修订，重要）**：本 run 通过 `bench_train_steps.py` 的 `import train as _train` → `_train.main(config)` 启动，被导入模块的 `__name__` 是 `"train"`，**`if __name__ == "__main__":` 块结构性不执行**。因此它证明的是「`train.main` 函数体删掉三处 tentative 死代码后位级等价」，**对新 `__main__` 与新记录器零覆盖**。AGENTS 18「第二块（本机训练梯度一致）」不再挂在本 run 上，改挂 4.2 的 B 侧（见 4.6）。

判据四分项（照 G3 判读纪律）：`SCALARS steps=1000 keys=5 hex_mismatch_steps=0`、`STATE_DIGEST rows=12 mismatch=0`、`BATCH_DIGEST_CANONICAL rows=14 mismatch=0` + `CANON_CHECK=PASS steps=14`、`INDEX_SEQ=PASS n=8072`。**不作判据**：raw `BATCH_DIGEST mismatch=4`（`static_image_emb`/`static_pos_emb`，dtype 统一已知预期失配，须与 G2/G3 逐字吻合）与 `DET_CHECK=FAIL` 总行/退出码。收官：`scalars_hex.tsv` 手工投影 sha256 必须 = `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`（G0b r1/r2、G1、G2、G3 五份同值，本轮为第六份）。

**判读改为机器裁决（第三版新增）**：第二版把 `compare_baseline.py` 的三处 fail-open（缺 scalar key 即 `continue`、index 只比最短公共前缀、canonical/INDEX_SEQ 不进总 verdict）交由「人工补位」。AGENTS 16 那条中位数假象的教训正是人眼判读吃的亏，故本版新增 `scripts/training/tests/g0_gate.py` **fail-closed 外层总闸**，输出唯一一行 `G0_EQ=PASS`：它不重新解析 records（避免两把尺子），只做①解析 `compare_baseline.py` stdout、四分项任一行缺失即 FAIL；②独立补检 `compare_baseline.py` 根本不看的六项（`scalars_hex.tsv` 表头恰六列 + 行数恰 1001 + sha256 命中锚点、`batch_digests.jsonl` 首行 `n_keys=12`、本侧 index `n ≥ 8072`、`BASELINE_ENV=PASS`、raw `BATCH_DIGEST mismatch` **恰为 4** 且失配键恰为那两个）。其中「表头六列 + 行数 1001」正是堵住 `continue` 那处 fail-open 的手段——五个 key 一个不少、每步一行都在，就跳不掉了。闸自身的反向自测见第九节。

过程中增量 state 对拍（每 100 步与 G0b-r1 同步骤比 digest，首次分叉立停）提供 fail-fast，故不单独跑 SMOKE5；失败按既有阶梯 GRAD_FIXTURE(15min) → SMOKE5(35min) → 增量二分定位，不得直接重跑。

本 run 裁决两个变量：`train.main` 体内删 tentative 行、目录拆分（记录器在 bench 路径根本不装，构造性零影响）。

### 4.2 上游 main 分支对拍（「和 RoboMME main branch 的对比」）——A/B 两侧各 1000 步

**可比性的机器实证**：G0b 起跑 commit `570287f` 的 `scripts/train.py` 与 main tip `ecf086c` **逐字节相同**（blob 同为 `a8be7a5`）；`ecf086c→570287f` 在 `src/openpi/`、`src/mme_vla_suite/{training,models,policies,shared}/`、`uv.lock` 上全部零 diff（仅建库脚本 +23 行与 ruff 配置 +8 行）——**G0b 的训练数值语义 = 上游 main 的训练数值语义**。分叉后 `train.py` 的两笔内容改动（V4.3 删 stats 返回链、V4.6 搬移+log_code 修正）已被 G0b→G1→G2→G3 位级链条覆盖。唯一从未实测的是「main 官方入口 `__main__` 双跑」这条执行路径本身——本对拍把它补上。

| | A 侧（上游原版） | B 侧（本分支） |
|---|---|---|
| 代码 | `git worktree` @ `ecf086c` + 独立 `uv sync`（`uv.lock` 零 diff，环境同一） | V5.0 tip |
| 入口 | `scripts/train.py` 官方 `__main__` 双跑（runpy 执行，**加 `--overwrite`** 跑通正式段） | 新 `scripts/training/train.py` 官方 `__main__` 单跑 |
| 数据 | legacy 库 `v1-store/datasets/4task-gl` | packed 库 `v1-store/datasets/4task-gl-framesamp` |
| 步数 | **1000**（第三版由 20 改，用户裁决） | **1000** |
| 口径 | b8 / seed 42 / workers 4 / fsdp 2 / log-interval 1 / save-interval 100 / context 变体 / 确定性档 | 同左 |

**为什么 1000 步比 20 步强一档**：1000 步 + 上表口径正是 G0b/G1/G2/G3 五份固化产物的口径，于是 A 侧与 B 侧可以**各自独立**投影出 `scalars_hex.tsv` 去对同一个锚点 `c799a0b2…`，不再只是 A/B 互比。20 步方案的结构性缺陷是：A/B 之间同时变了六样（入口文件、源码版本、legacy vs packed 库、有无 `--overwrite`、`exp_name`（连带 `~/.cache/jax_{exp_name}` 编译缓存目录）、checkpoint 根），互比 PASS 只能宣称「这一整套组合端到端数值相同」；而两侧各自对历史锚点，失败时可精确定位：

| A 侧 | B 侧 | 判读 |
|---|---|---|
| ✅ | ✅ | 第六、七份同值，入口改动等价，收工 |
| ❌ | ✅ | **tentative 预热确实污染了正式轨迹**——正是从未实测过的那个怀疑点 |
| ✅ | ❌ | 新 `train.py` 改坏了 |
| ❌ | ❌ | harness 自己的补丁在改数，先修工具 |

⚠ A 侧命中锚点是**待验**、不是预期必然：A 侧与 G0b 的差异恰恰包含被测的 tentative 段（以及 `exp_name` 不同导致的冷编译，跨编译 bitwise 由 D2-cold 实证背书；`save_interval` 不同不影响数值，因 harness 已把 `save_state` 换成不落权重的摘要器）。

argv 差异**五处**并逐项登记（第二版漏记 `--checkpoint-base-dir`）：入口文件、`--exp-name`、`--dataset-path`（链路差异载体，正是被测对象）、`--checkpoint-base-dir`、A 侧独有 `--overwrite`。其中 `exp_name` 除进 ckpt 路径与 wandb 名外，**还决定 `~/.cache/jax_{exp_name}` 编译缓存目录**（`train.py::main` 的 `jax.config.update("jax_compilation_cache_dir", ...)`）——这正是「两侧编译缓存天然独立」的成因，第二版把它写成了外部安排，实为 argv 的连带效应。

A 侧 tentative 段（预期 step 0..11 共 12 行，`tentative_run_step=10` 与总步数无关，1000 步下照旧）单独留档不判——tentative 段零 save（break 在 save 块之前，机器事实）。预计每侧约 1 h loop + 冷编译，两侧串行合计约半天 + 2 h。

### 4.3 秒级闸（C2 提交前）

`uv run pytest --collect-only -q src scripts packages` 零 error；`ruff check` 零新增违例；`grep -rn "training/bench\|prod_train_once" scripts/ src/ pyproject.toml` 零活引用残留（**注意：第二版清单不全，照原清单改完此闸必然失败**——审计实跑该 grep 命中 13 文件 20 行，其中 `scripts/dataset/gl/README.md` 两处未列入任何清单，见第七节）；`bash -n` 过 sbatch 与 `g0/run_2gpu_epoch_bench.sh`；train.py 语法自检 + 四探针子串自查。

### 4.4 记录器功能验收 smoke（第三版新增，C2 与 C3 之间，本机 ≤5 min）

**为什么必须有**：审计发现第二版三道验证对新记录器**全程零覆盖**——C3 走 bench 路径不触 `__main__`（4.1 已述）；C4 B 侧显式不设 `TRAIN_RECORD_DIR`（观测由 harness 代理承担）；C2 秒级闸全是静态检查；而第二版 R3 的「analyzer 冒烟」用的是**历史 record 目录**，其生产者是 prod 从 bench 原样 import 的旧 `_WandbProxy`，与 `train.py` 里独立重新定义的 `_MetricsProxy` 代码不共享。于是新记录器的首次真实执行 = 首次集群正式提交，而它是 `prod_train_once.py` 删除后 `metrics.jsonl` 的唯一生产者。

**四个用例**（合计 ≤5 min，`--num-train-steps 2`、`--no-wandb-enabled`、临时 exp-name，跑完清理）：

| # | 用例 | 判据 |
|---|---|---|
| S1 | 正常跑 2 步，设 `TRAIN_RECORD_DIR` | `metrics.jsonl` + `run_meta.json` 齐全；`util/analyze_gpu_util.py` 的 `load_metrics` 不抛异常、`RESULT` 六行可打印、解析步数 = 2 |
| S2 | 对**同一目录**再跑一次 | 必须 `FileExistsError`；且不得误伤——目录内预置 `env.json` 等 6 个文件（模拟 sbatch 先建目录）时首跑必须正常通过 |
| S3 | 训练中途抛异常 / `SIGTERM` | `finally` 已写 `run_meta.json`，且**不吞掉原异常**（进程退出码非零） |
| S4 | 不设 `TRAIN_RECORD_DIR` 跑 2 步 | 零文件产出，行为与改动前一致 |

**判据口径必须写死**：`analyze_gpu_util.py` 的 `E2E95_ACCEPT` 五项阈值与 `TREND_OK` 是按 600/1000 步标定的，喂 2 步数据大概率报不通过——那不是记录器的错。S1 判据**只取「`load_metrics` 不抛 + `RESULT` 六行可打 + 步数解析 = 2」，明确不看 accept/trend 结论**。不写死这条，实施时会被假 FAIL 卡住。

S3 覆盖的是集群长训被 Slurm 超时 kill 的真实路径（正常退出与被 kill 走的不是同一条），2 步正常跑测不出来。

### 4.5 判定行汇总

```
# C2 秒级闸
COLLECT/RUFF/GREP/BASH-N 全绿（无专用判定行，逐条命令零 error）
# C2.5：记录器 smoke（v5-recorder-smoke，临时 run 跑完即清）
RECORDER_SMOKE=PASS cases=4        # S1..S4 全过；S1 内含 analyzer 可读性
# C3：G0 完整重锚（v1-singlerun-g0）
BASELINE_ENV=PASS
SCALARS steps=1000 keys=5 hex_mismatch_steps=0
STATE_DIGEST rows=12 mismatch=0
BATCH_DIGEST_CANONICAL rows=14 mismatch=0 ；CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072
sha256(scalars_hex.tsv) == c799a0b2…
G0_EQ=PASS                          # ★ 上述全部 + 六项补检的机器总闸，一行定生死
# C4：上游对拍（v1-upstream-eq）
A_SIDE_SEGMENTS tentative_rows=12 main_rows=1000        # 留档非判据
ENTRY_SCALARS steps=1000 keys=5 hex_mismatch=0
ENTRY_STATE_DIGEST rows=<首轮实测钉死> mismatch=0
ENTRY_RESOLVED_CFG mismatch=0 whitelist=4
ENTRY_PROVENANCE=PASS
A_SCALARS_SHA256 == c799a0b2…       # 第六份同值
B_SCALARS_SHA256 == c799a0b2…       # 第七份同值
ENTRY_EQ=PASS                       # ★ 定义为上面五条子判据全过
```

`ENTRY_EQ=PASS` 第三版重定义为**五条子判据全过**：A 命中锚点 ∧ B 命中锚点 ∧ A-vs-B 逐步 hex 零失配 ∧ 状态摘要零失配 ∧ 两侧 provenance 各自归位。第二版曾担心该名字名不副实（组合变量太多），1000 步改法解决后名字站得住，无需改名。

### 4.6 AGENTS 18 两块一致性的适配说明（第三版新增）

AGENTS 18 要求三样：前后两张数据逐跳图、第一块（非训练轻量对拍）、第二块（本机训练梯度一致）。逐条落地与**适配解释**：

- **链路图** → 见附录 D。第二版附录 A 画的是 `__main__` 调用序列（控制流），不是「数据源到进入模型的逐跳图 + shape/dtype/字节量」。本版重画，并遵守两条纪律：①**每个数字标机器来源**——loader 输出那一跳的 shape/dtype 直接从 C3 run 的 `batch_digests.jsonl` 投影并注明来源文件与 step，进 loader 之前几跳（H5 → packed store → `FrameSampDataset.__getitem__` → transforms → collate）无机器记录，一律标注「源码推导，锚点 xxx」，两类不混装；②**分两次落笔**——C2 时先画源码推导部分，C3 出 records 后补实测数字。字节量口径定为「per-batch 张量占用 = `prod(shape) × itemsize`」，另单列「本跳是否发生拷贝 / dtype 转换」（第 18 条真正关心的是「这一跳有没有改数」）；NFS 实际读字节走 `nfs_read.csv` 另一通道，不混进此图。观测链路（记录器插入点）画作旁支，标明只读转发。
- **第一块** → **按字面在本次改动下无对应物，据实写明，不假装满足**：第 18 条第一块要比的是「新旧两条数据链路交付内容一致」，而本次改的是入口层，数据链路一行未动，硬造前后对拍必然 PASS 且零信息量。替代方案：以 C3 run 的 `index_sequence.json` 与 `batch_digests.jsonl` 对 G0b-r1 的逐键比对充当（确为 index 序列 + 逐 batch 内容 + dtype/shape 逐键，只是比较对象从「新链路 vs 旧链路」变为「本轮 run vs 固化基线」），判据即 4.1 的 `INDEX_SEQ` / `BATCH_DIGEST_CANONICAL` 两分项。⚠ 第二版把 4.4 的记录器 smoke 说成第一块是概念偷换（那比的是观测链路的文件格式，不是数据链路），本版撤回该说法，smoke 单独定位为记录器功能验收。
- **第二块** → 改挂 **4.2 的 B 侧**（唯一真正执行新 `__main__` 的 1000 步真实训练）。第二版挂 4.1 不成立（bench 路径结构性绕开被测代码）。第 18 条末句「引用既有基线而非同场次重跑对照侧，必须先过环境指纹 preflight 并在留档写明所引用基线的 run_name、commit 与指纹比对结论」照办：C3 引 G0b-r1（`BASELINE_ENV=PASS`），C4 引 G0b/G2 锚点 `c799a0b2…`，两处留档均写明。

## 五、commit 切片与 run_name

1. **C1 `docs:`**——第二版落档（已完成，`1dec4b7`）。**C1b `docs:`（本 commit）**——两轮对抗性审计后的第三版修订。
2. **C2 `commitV5.0:`**——全部代码改动（train.py 含 3 行护栏与记录器、删 prod_train_once.py、sbatch 直调 + 退出码硬化、g0/util 拆分、`tests/g0_gate.py` 与 `tests/entry_equiv.py` + `run_entry_equiv.sh`、活引用与自述同步），过 4.3 秒级闸后提交。`g0_gate.py` 的反向自测（第九节）在本阶段跑完，不等 C3。
3. **C2.5**——记录器 smoke（4.4 四用例，本机 ≤5 min，临时 run 跑完即清，不单独 commit；结果并入 C3 留档）。
4. **C3**——G0 完整重锚 run（clean HEAD、tmux + Monitor，run_name 拟 **`v1-singlerun-g0`**，起跑前按 AGENTS 6 确认）→ **起跑前先提交 `launch.md`**，跑完补 result.md/records（AGENTS 12/17）→ 留档 `docs/training-doc/v1-singlerun-g0/`（含附录 D 两图与 4.6 两块讨论）→ `docs:` commit。
5. **C4**——上游对拍 run（run_name 拟 **`v1-upstream-eq`**，A/B exp-name `entry-eq-a/b` 临时 run）→ 同样**起跑前先提交 `launch.md`** → 留档 `docs/training-doc/v1-upstream-eq/` → `docs:` commit；PASS 后清理 worktree、两侧临时 ckpt 目录与 `v1-store/entryeq/`（FAIL 记录按 `.failed-<n>` 惯例保留）。

sbatch 改动本机只能静态验证；首次集群提交按 `greatlakes.md` 走用户放行，不在本计划范围。

---

# 第二部分（技术细节，供 agent 追踪）

## 六、train.py 逐处改动（语句锚点；第三版删除全部硬编码行号，AGENTS 9）

1. 删 `from openpi.training.optimizer import CosineDecaySchedule`（全文无引用）；保留 `import time`；**补 `import os`、`import json`、`import pathlib`、`import sys`**（现文件均未 import；`sys` 供 `_finalize_record` 记 `sys.argv`——第二版清单漏列）。
2. 签名 → `def main(config: _config.TrainConfig):`。
3. 删 `tentative_run_step = 10` 定义块（含其两行注释）与 `tentative_run_step += config.resum_ckpt_id`；**保留** `start_step += config.resum_ckpt_id`（两者无耦合，审计确认 `tentative_run_step` 全文仅出现三处，删除集完全覆盖、不留 `NameError`）。
4. 删循环内 `if tentative_run and step > tentative_run_step: … break` 块。
5. 在 `main()` 体最前（`init_logging()` 之后、`batch_size % jax.device_count()` 既有检查附近）插入三.4 的两条 fail-loud。
6. 模块级新增（`main` 之后、`__main__` 之前）：
   - `_MetricsProxy` 类：`log()` 组行写 `metrics.jsonl` 后转发真 wandb，`__getattr__` 透传。
   - `_install_metrics_recorder(dir)`：**先 `dir.mkdir(parents=True, exist_ok=True)`**（第二版漏规定；照抄 prod `_record_dir()` 的既有做法），再 `metrics.jsonl` 已存在 → `FileExistsError`；最后 `globals()["wandb"] = _MetricsProxy(wandb, dir / "metrics.jsonl")`。
   - `_finalize_record(dir, config)`：写 `run_meta.json`（`sys.argv` / `entry="scripts/training/train.py"` / checkpoint_dir / num_train_steps / log_interval / save_interval）；`wandb.finish()` try/except 吞掉。
7. `__main__` 替换：

```python
if __name__ == "__main__":
    _record_dir = os.environ.get("TRAIN_RECORD_DIR")
    _cfg = _config.cli()
    if _record_dir:
        _install_metrics_recorder(pathlib.Path(_record_dir))
    try:
        main(_cfg)
    finally:
        if _record_dir:
            _finalize_record(pathlib.Path(_record_dir), _cfg)
```

8. 自查：`inspect.getsource(main)` 仍含四探针（4.3 秒级闸覆盖）。

## 七、目录拆分与引用同步执行清单

**git mv（可执行逻辑与判据零改动，仅允许路径/文案 hunk）**：`bench/{bench_train_steps.py, check_baseline_env.py, compare_baseline.py, compare_online_memory.py, README.md, run_2gpu_epoch_bench.sh} → g0/`；`bench/{analyze_gpu_util.py, analyze_util.py} → util/`；`prod/downsample_util_csv.py → util/`；`git rm prod/prod_train_once.py`。

**必改活引用（第三版重新穷举：5 个文件、10 处改动点；第二版标称「共 6 处」而表内仅 4 行，两个口径都对不上，本版改为按文件与改动点双计数）**：

| 文件 | 改动点数 | 改动 |
|---|---|---|
| `g0/run_2gpu_epoch_bench.sh` | 2 | 自引用 ×2：`scripts/training/bench/bench_train_steps.py` 与 `…/check_baseline_env.py` → `g0/` |
| `prod/gl_train_prod.sbatch` | 5 | 入口行 `prod/prod_train_once.py` → `train.py`，**其余 15 个 CLI 参数逐字符不变**；`PROD_RECORD_DIR` → `TRAIN_RECORD_DIR`；env.json `entry` 字符串 → `scripts/training/train.py`；收尾 `bench/analyze_gpu_util.py` → `util/`、`prod/downsample_util_csv.py` → `util/`；另加第十一节退出码硬化与头注释迁移说明 |
| `tests/test_padding_dtype.py` | 2 | 代码内绝对路径 `bench/bench_train_steps.py` → `g0/`；**docstring 里的同名路径**（第二版只列了代码那处） |
| `pyproject.toml` | 1 | per-file-ignores：`"scripts/training/bench/*.py"` → `"scripts/training/g0/*.py"` + `"scripts/training/util/*.py"` |
| **`scripts/dataset/gl/README.md`** | 2 | **第二版完全漏列，且这是 4.3 秒级闸 grep 必然失败的直接原因**。两处引用 `scripts/training/bench/run_gl_dataset_training_smoke.sh` 与 `scripts/training/bench/smoke_train_once.py`——**这两个文件早已不存在**（AGENTS 项目 scope 记载 `smoke_train_once.py` 判定不可靠已删弃）。属既有失效引用，非本轮拆分造成，但必须一并清理，否则 grep 闸过不去 |

**应改自述/文案（不崩但失真）**：`README-ZH.md`（「本地 bench/smoke 在 …」行）；`g0/README.md`（「五个文件」勘误为实际清单 + 全部用法路径 + 迁入 prod 坑登记）；`util/analyze_util.py`、`util/analyze_gpu_util.py`、**`util/downsample_util_csv.py`**（第二版漏列，与另两个 util 文件处境完全对称，且 `util/` 下无 README 可顺带覆盖）、`g0/compare_online_memory.py` 文件头用法行；`tests/_common.py`、`tests/compare_dtype_fix.py`、`src/mme_vla_suite/datastore/framesamp_store.py` 的路径注释；`scripts/training/paths.sh` 头注释里的 `bench/` 字样（不会命中 `training/bench` grep，但文案失真）；`scripts/dataset/gl/step2_verify.sh` echo 文案。

**第二版的一条空指令已删**：原清单要求改 `g0/bench_train_steps.py` 的「文件头用法行」——该文件全文对 `usage|用法|示例|uv run scripts` 零命中，其 docstring 讲的是设计动机、不含任何命令行范式，该项永远无法落地。

**明确不改**：`docs/training-doc/**` 全部历史留档（records 内 argv 已进 `BASELINE_MANIFEST.json` sha256，改一字节即 `BASELINE_ENV=FAIL`、基线作废）；v1/v2/v3 系列既往计划 md 的旧路径（既定惯例，且均在仓库根目录、不落在 4.3 grep 的 `scripts/ src/ pyproject.toml` 范围内）；`README.md` 与 `scripts/training/finetune_mme_vla_suite.sh` 的官方示例（用户裁决「无所谓 不用改官方」）。

## 八、记录器 smoke runbook（4.4 落地）

四用例共用底座（本机 1 卡即可，`--num-train-steps 2`、`--no-wandb-enabled`、临时目录）：

```bash
SM=v1-store/tmp/v5-recorder-smoke        # 跑完整目录删除
# S1 正常
TRAIN_RECORD_DIR=$SM/r1 UV_LINK_MODE=copy uv run scripts/training/train.py mme_vla_suite \
  --exp-name v5-recorder-smoke-1 --num-train-steps 2 --log-interval 1 --save-interval 10000 \
  --batch-size 8 --num-workers 2 --seed 42 --fsdp-devices 1 \
  --dataset-path <REPO>/v1-store/datasets/4task-gl-framesamp \
  --assets-base-dir <REPO>/v1-store/train-assets \
  --checkpoint-base-dir $SM/ckpt \
  --weight-loader.params-path <REPO>/v1-store/models/openpi-assets/checkpoints/pi05_base/params \
  --model.use-history --model.history-config perceptual-framesamp-context.yaml --no-wandb-enabled
UV_LINK_MODE=copy uv run scripts/training/util/analyze_gpu_util.py $SM/r1 --steps 2   # 只看能否解析
```

- **S2**：先 `mkdir -p $SM/r2 && touch $SM/r2/{env.json,gpu_util.csv,gpu_util_dense.csv,nfs_read.csv,meminfo.csv,compute_apps.csv}`（模拟 sbatch 先建目录写 6 文件）→ 首跑必须**正常通过**（不得被「目录非空」误伤）；再对 `$SM/r2` 跑第二次 → 必须 `FileExistsError`。
- **S3**：`--num-train-steps 2` 起跑后立即 `kill -TERM <pid>`（或临时在 `main` 内注入 `raise`）→ 检查 `$SM/r3/run_meta.json` 已写、进程退出码非零、原始异常/信号未被吞。
- **S4**：不设 `TRAIN_RECORD_DIR` 跑一次 → `find $SM/r4 2>/dev/null` 无输出，且 stdout 与改动前一致。

判定行 `RECORDER_SMOKE=PASS cases=4`；S1 的 analyzer 判据只看「不抛 + `RESULT` 六行 + 解析步数 = 2」，不看 `E2E95_ACCEPT` / `TREND_OK`。收尾 `rm -rf $SM`。

## 九、G0 完整重锚 runbook（照 G3 留档移植，仅路径、run_name 与总闸变）

前置门：porcelain 空、clean HEAD；`env | grep MMEVLA_FRAMESAMP` 为空；packed 库锚点人工比对（本轮 env.json 顶层 `store_meta_sha256` 与 G2/G3 的 `3990165c9cebffdadaceb01cc88470645a3d62f9af65eabccf4331d3fcd5b556` 一致）；`tmux has-session -t v1-singlerun-g0` 无撞名；**`launch.md` 已提交**。

preflight（必带三 env，缺任一必 FAIL 三项）：

```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=0,1 \
UV_LINK_MODE=copy uv run scripts/training/g0/check_baseline_env.py check \
  --baseline docs/training-doc/v1-grad-baseline-g0b/records/r1 \
  --dataset v1-store/datasets/4task-gl --steps 1000 --batch-size 8
# 判定 BASELINE_ENV=PASS；--dataset 必须指 legacy 源库（packed 库无 data/ 会炸）
# 判读纪律：指纹不含仓库代码 sha，PASS 只证「引用 G0 的资格还在」，不证「代码没改坏」
```

起跑（G3 模板，路径改 g0/，`<REPO>` = 仓库根绝对路径）：

```bash
tmux new-session -d -s v1-singlerun-g0 "set -o pipefail; cd <REPO>; \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 \
  EXP_NAME=v1-singlerun-g0 RUN_TAG=v1-singlerun-g0 \
  DATASET_PATH=<REPO>/v1-store/datasets/4task-gl-framesamp \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
  UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  bash scripts/training/g0/run_2gpu_epoch_bench.sh \
    2>&1 | tee v1-store/logs/v1-singlerun-g0-driver.log; \
  echo \"EXIT_CODE=\$?\" >> v1-store/logs/v1-singlerun-g0-driver.log"
```

Monitor 挂 driver.log（每级行缓冲）：`tail -n +1 -F <log> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "Step (0|[0-9]*00|299|999):|BENCH_|EXIT_CODE|Traceback|Error|OOM|digest"`。过程中每出一次 state 摘要即与 G0b-r1 同步骤比 digest，首次分叉立停。

判读（先出四分项，再过总闸）：

```bash
UV_LINK_MODE=copy uv run scripts/training/g0/compare_baseline.py \
  docs/training-doc/v1-grad-baseline-g0b/records/r1 \
  v1-store/bench/2gpu-epoch-bench/v1-singlerun-g0 --tier singlerun-vs-g0b \
  | tee docs/training-doc/v1-singlerun-g0/records/compare_vs_g0_r1.txt

UV_LINK_MODE=copy uv run scripts/training/tests/g0_gate.py \
  --compare-out docs/training-doc/v1-singlerun-g0/records/compare_vs_g0_r1.txt \
  --run-dir v1-store/bench/2gpu-epoch-bench/v1-singlerun-g0 \
  --scalars docs/training-doc/v1-singlerun-g0/records/scalars_hex.tsv \
  --expect-sha256 c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757 \
  --env-out <preflight 输出文件>
# 唯一判定行：G0_EQ=PASS
```

**`g0_gate.py` 规格**（`scripts/training/tests/g0_gate.py`，纯只读，不 import 训练代码）：

- **A 类（解析 `compare_baseline.py` stdout）**：四分项判定行逐行正则提取，**任一行缺失即 FAIL**（堵住「canonical/INDEX_SEQ 不进总 verdict」）；`SCALARS … hex_mismatch_steps=0`、`STATE_DIGEST rows=12 mismatch=0`、`BATCH_DIGEST_CANONICAL rows=14 mismatch=0`、`CANON_CHECK=PASS steps=14`、`INDEX_SEQ=PASS n>=8072` 逐项校验；raw `BATCH_DIGEST mismatch` **必须恰为 4**，且失配键集合恰为 `{static_image_emb, static_pos_emb}`（多一个少一个都 FAIL）。
- **B 类（独立补检，`compare_baseline.py` 根本不看）**：`scalars_hex.tsv` 表头恰六列（`step\tloss.hex\tgrad_norm.hex\tllm_grad_norm.hex\tmem_enc_norm.hex\tparam_norm.hex`）+ 行数恰 1001 + 末尾单换行 + sha256 命中 `--expect-sha256`（这三项合起来堵住「缺 scalar key 即 `continue`」那处 fail-open：五个 key 一个不少、每步一行都在，就跳不掉）；`batch_digests.jsonl` 首行 `n_keys=12`；preflight 输出含 `BASELINE_ENV=PASS`。
- **不做的事**：不重新解析 records 计算任何比对结果（避免与 `compare_baseline.py` 成为两把尺子），只做「解析 + 补检」。
- **反向自测（C2 阶段完成，不等 C3）**：正样本 = G3 固化 records（`docs/training-doc/v1-postclean-g3/records/`）+ 其 compare 输出，必须 `G0_EQ=PASS`；负样本 = 在 `/tmp` 拷贝上构造 4 个（删 `scalars_hex.tsv` 任一行 / 改其中一个 hex 字符 / 删 `index_sequence.json` / 把 `n_keys` 改成 11），每个必须 FAIL 且报出具体原因。**四个负样本不全 FAIL，闸不算可用。** 构造只在 `/tmp` 拷贝上做，原始留档一个字节不碰。

收官投影 `scalars_hex.tsv` → sha256 = `c799a0b2…`；`check_baseline_env.py manifest` 生成防腐清单；留档含附录 D 两图与 4.6 两块讨论。

## 十、上游 main 对拍 runbook（1000 步版）

A 侧准备（worktree 属临时产物，v1-store 内不进 git，PASS 后清理；AGENTS 19「快照内禁执行」只约束纯审计，此处为执行型 worktree，本计划授权）：

```bash
git worktree add --detach <REPO>/v1-store/entryeq/worktree-main ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b
cd <REPO>/v1-store/entryeq/worktree-main && UV_LINK_MODE=copy uv sync
```

preflight 三项（任一不过即停、交用户）：

- **P1**：A 侧 `--num-train-steps=2` 冒烟（main 代码读 `4task-gl` 出前两步 loss）。⚠ **必须走 harness 或指临时目录**：`(step % save_interval == 0 and step > start_step) or step == num_train_steps - 1` 会在 step 1 命中末步强制保存；且 2 步下 tentative 段（`step > 10` 才 break）根本不会 break，会跑完 2 步存一次，正式段再被 `--overwrite` 删掉重建、又存一次。走 harness 时 `save_state` 已被换成不落权重的摘要器，问题消失；否则 ckpt 目录指 `$TMP` 并在 P1 后 `rm -rf`。
- **P2**：`git show ecf086c:scripts/train.py` 与 config 源码静态核对 `--overwrite` / `--no-wandb-enabled` / `--num-train-steps` 等键位（审计已验：A 侧 argv 全部 flag 在 `ecf086c` 版 `TrainConfig`/`weight_loaders.py`/`history_pi0.py` 中均存在）。
- **P3**：两侧实际加载 norm_stats sha256 相同（均为 `<REPO>/v1-store/train-assets` 下 GL 已算好的那份，G0b 指纹 `norm_stats_sha256=709f22ff…` 为参照）。

A 侧 argv（数值口径照 G0b 固化 argv，加 `--overwrite`、1000 步、save-interval 100）：

```
scripts/train.py mme_vla_suite --exp-name entry-eq-a --overwrite
  --num-train-steps 1000 --log-interval 1 --save-interval 100
  --batch-size 8 --num-workers 4 --seed 42 --fsdp-devices 2
  --dataset-path <REPO>/v1-store/datasets/4task-gl
  --assets-base-dir <REPO>/v1-store/train-assets
  --checkpoint-base-dir <REPO>/v1-store/entryeq/ckpt-a
  --weight-loader.params-path <REPO>/v1-store/models/openpi-assets/checkpoints/pi05_base/params
  --model.use-history --model.history-config perceptual-framesamp-context.yaml
  --no-wandb-enabled
```

B 侧：入口 `scripts/training/train.py`、`--exp-name entry-eq-b`、无 `--overwrite`、`--dataset-path …/4task-gl-framesamp`、`--checkpoint-base-dir …/entryeq/ckpt-b`，其余逐字符同。env 两侧同：确定性档 `XLA_FLAGS`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`CUDA_VISIBLE_DEVICES=0,1`；编译缓存目录经 `~/.cache/jax_{exp_name}` 由 `--exp-name` 自动分离（跨编译 bitwise 由 D2-cold 实证授权）。B 侧不设 `TRAIN_RECORD_DIR`（内置记录器不装，观测全由 harness 代理承担）。

**`--save-interval` 由第二版的 10000 改为 100**：harness 已把 `save_state` 换成不落真权重的摘要器，"保存"只是记一次 TrainState 摘要，改成 100 可拿到约 10 行逐步状态对拍，而不是只有末步 1 行。首轮实测行数钉死进判定行。

**长任务规范（1000 步 × 2 侧，AGENTS 6/7/12/17）**：两侧各起一个 detached tmux session（`v1-upstream-eq-a` / `-b`）、各一份 `tee` 日志、结束写 `EXIT_CODE=`、**各挂一个 Monitor**（禁止一条 `tail -F` 挂两个日志），管道每级行缓冲；起跑前先提交 `launch.md`、确认全新 run_name。

**harness `tests/entry_equiv.py`**：`--entry <入口py> --record-dir <dir> --expect-root <目录> -- <argv尾巴>`。启动入口前装三处只读补丁 + 一道来源断言：

1. **wandb 代理**：`sys.modules["wandb"]` 预载，显式 `log` 方法逐步记五标量 `float.hex()` 落盘后转发**调用时查找**的真 `wandb.log`，其余属性 `__getattr__` 委托真模块（规避「`wandb.init()` 重赋值模块级 `log` 盖掉补丁」的 2026-08-24 登记坑）。
2. **`save_state` 摘要器（第三版加强）**：`openpi.training.checkpoints.save_state` 模块属性替换（`train.py` 的 `_checkpoints.save_state(...)` 属调用时属性查找，补丁必中），**不落真权重**。摘要域由第二版的裸 `sha256(device_get(leaf).tobytes())` 改为**双口径**：
   - `strict`：`sha256(keystr(path) + str(dtype) + str(shape) + tobytes())` —— 供 A/B 互比；裸字节口径下 dtype 不同、shape 转置（(2,3) 与 (3,2) 的 `tobytes()` 完全相同）、树中位置不同均会误通过。
   - `g0-compat`：照抄 `g0/bench_train_steps.py::_leaf_sha256` 的纯字节口径 —— 供与 G0b 固化 `param_checksums.jsonl` 交叉比（`tests/_common.py::leaf_sha256` 与之逐字节同结果，由 `test_padding_dtype.py::test_hash_kouging_matches_bench` 锁死，不得单方面改口径）。
   - 另记两个与叶子内容无关的字段：`treedef_sha = sha256(repr(tree_structure(state)))`、`keyset_sha = sha256(排序后全部 keystr)` —— 堵住「结构变了但每个叶子字节恰好对上」。
3. **resolved-config digest（第三版新增）**：`TrainState` 的 `tx` 与 `ema_decay` 标了 `struct.field(pytree_node=False)`（见 `src/openpi/training/utils.py`），**不是 pytree leaf**，逐叶遍历遍历不到，故「末步完整 TrainState 逐叶 sha」名不副实。补 `RESOLVED_CFG_SHA = sha256(json.dumps(dataclasses.asdict(config), sort_keys=True, default=repr))`，两侧比对排除白名单四项（`exp_name` / `dataset_path` / `checkpoint_base_dir` / `overwrite`）；`ema_decay` 与优化器、LR schedule 的 repr **单独列出对比、不埋进总哈希**，失配时可直接定位字段。
4. **来源污染防护（第三版新增，最阴的误通过）**：A 侧在 worktree venv 内运行而 harness 在主仓库，若继承的 `PYTHONPATH` 含主仓库路径，A 侧可能 import 到 B 侧的 `openpi` —— 两侧跑同一份代码，必然 PASS 且全程无异常。三道一起上：①起进程前 `env -u PYTHONPATH -u PYTHONHOME`；②harness 在 import 完成、调 main 之前断言每个项目模块（`openpi`、`mme_vla_suite`、入口模块）的 `pathlib.Path(m.__file__).resolve().is_relative_to(EXPECT_ROOT)`，不符即当场 fail-loud；③把每个模块的 `__file__` 与其文件 sha256、以及各侧 cwd 的 `git rev-parse HEAD` 写进 `provenance.json`，判读时交叉核对（A 侧全在 worktree 下、B 侧全在主仓库下）。判定行 `ENTRY_PROVENANCE=PASS`。

随后 `sys.path.insert(0, 入口目录)`、`sys.argv = [入口名, *尾巴]`、`runpy.run_path(入口, run_name="__main__")`。A 侧命令形如 `cd worktree && env -u PYTHONPATH uv run python <主仓库 harness 绝对路径> --expect-root <worktree> …`；harness 只 import 标准库 + wandb + checkpoints，不得 import 本分支训练代码。记录器按 step 回绕切分 A 侧 tentative / 正式两段；harness 启动后断言代理生效、行数与步数不符即 fail-loud。

**`scalars_hex.tsv` 投影**：两侧记录落盘后，用与 C3 收官**完全相同**的投影口径生成 `scalars_hex.tsv`（表头恰六列、同列序、1001 行、末尾单换行），再各自算 sha256 对锚点。口径不一致则 sha256 必然不同，此步是 A/B 各自对历史锚点的前提，投影脚本两处必须共用同一份。

`tests/run_entry_equiv.sh` 驱动：两侧串行（各自 tmux）→ 读 record jsonl 比对 → 输出 4.5 判定行；FAIL 记录 `.failed-<n>` 原子改名保留。`ENTRY_STATE_DIGEST rows` 首轮实测钉死（`--save-interval 100` + 末步强制保存，预期约 10–11 行）；A 侧 tentative 段零 save（break 位于 save 块之前，且 `--save-interval 100` 下 step 0..11 无整除步）。

## 十一、sbatch 退出码硬化（第三版新增）

**现状缺陷**（既有，非本轮引入，但删掉 prod wrapper 后风险被放大）：`gl_train_prod.sbatch` 用 `set -uo pipefail`（**无 `-e`**），末尾 `RC=$?` 只取训练退出码、`echo "EXIT_CODE=$RC"; exit "$RC"`。于是 `write_env`、`analyze_gpu_util.py`、`downsample_util_csv.py` 任一失败都不改变最终状态——**新记录器 schema 出错正好以「analyzer 崩溃」形式出现，而 Slurm 仍看到 `EXIT_CODE=0`**。

**最小版修法**（用户裁决）：

- `write_env` → `ENV_RC`，训练 → `TRAIN_RC`（保留现名），analyzer → `ANALYZE_RC`，downsample → `DOWNSAMPLE_RC`，四者各自 `echo` 一行留档。
- 最终 `RC` 取四者的「首个非零」（训练失败优先报训练码），`echo "EXIT_CODE=$RC"; exit "$RC"`。
- `trap` 保证 `stop_samplers` 在任何退出路径都执行，且 `EXIT_CODE=` 行必写（含被 Slurm `SIGTERM` 的情形）。
- 头注释登记：本节改动原因 + 从 `prod_train_once.py` 迁入的四条坑 + 「续跑有损（checkpoint 只存 EMA、丢 AdamW 动量与 warmup 计数）」。

`bash -n` + 逐行 diff 评审；本机无 slurm，动态行为只能首次集群提交时验证（按 `greatlakes.md` 用户放行）。

## 十二、风险登记

| # | 风险 | 规避 |
|---|---|---|
| R1 | train.py 新增记录器意外改数 | 全在 `train.main` 体外、只读转发；C3 千步位级 + C4 双侧锚点三重裁决 |
| R2 | 删 tentative 行触碰指纹探针 | 探针行不在删除集（审计逐行确认，且全族无源码哈希）；4.3 子串自查 + C3 实证 |
| R3 | metrics.jsonl 格式漂移致 analyzer 崩 | 逐字段照抄 bench `_WandbProxy` schema；**4.4 smoke 用新记录器产出的新目录跑 analyzer**（第二版用历史目录，与新代码零交集，已修） |
| R4 | 拆分漏改活引用 | 第三版重新穷举 5 文件 10 处（补 `scripts/dataset/gl/README.md` 等）；grep / pytest-collect / ruff 三闸兜底 |
| R5 | sbatch 改动本机无 slurm 只能静态验证 | `bash -n` + 逐行 diff 评审；首次集群提交按 greatlakes.md 用户放行 |
| R6 | A 侧 worktree 环境/数据不兼容 | `uv.lock` 零 diff 在案；三项 preflight fail-loud，失败即停交用户 |
| R7 | A/B argv 不同被质疑引入差异 | 差异**五处**逐项登记（第二版漏 `--checkpoint-base-dir`）；`exp_name` 进 ckpt 路径、wandb 名与 `~/.cache/jax_{exp_name}` 编译缓存目录三处，后者正是两侧缓存天然分离的成因 |
| R8 | wandb 补丁被 `wandb.init()` 盖掉致静默缺记 | `sys.modules` 代理预载 + harness 断言行数与步数一致 |
| R9 | C3 FAIL | 不直接重跑；按阶梯 GRAD_FIXTURE(15min) → SMOKE5(35min) → 增量 state 二分定位 |
| **R10** | **`use_history` 误配静默训出错模型** | 三.4 保留 fail-loud 护栏（`HistoryPi0.__init__` 的 `else` 分支保证关掉它一定跑得通、判读链路零告警，是本轮唯一「静默」失败模式） |
| **R11** | **`g0_gate.py` 自身写错且永远输出 PASS** | 第九节反向自测：G3 固化 records 正样本必须 PASS + `/tmp` 拷贝上 4 个负样本必须全 FAIL，C2 阶段完成；闸不重新解析 records，只做解析 + 补检，不与 `compare_baseline.py` 成为两把尺子 |
| **R12** | **C4 A 侧 import 到 B 侧代码致必然 PASS** | 第十节第 4 点三道防护（清 `PYTHONPATH` + `__file__` 断言 + `provenance.json` 交叉核对），判定行 `ENTRY_PROVENANCE=PASS` |
| **R13** | **叶子摘要域过窄致误通过** | 第十节第 2–3 点：strict 口径含 path/dtype/shape、另记 `treedef_sha`/`keyset_sha`、静态配置单算 `RESOLVED_CFG_SHA`（`tx`/`ema_decay` 非 leaf，逐叶遍历遍历不到） |
| **R14** | **sbatch 把 analyzer 崩溃伪装成成功** | 第十一节退出码硬化（四段 RC + trap） |

## 十三、明确不做（需另拍板）

`finetune_mme_vla_suite.sh` / `README.md` 的官方示例修复（用户裁决「无所谓 不用改官方」；两处仍因 `perceptual-framesamp-modul` 与 legacy `--dataset-path` 跑不通，与本计划的双跑修复无关，见 1.3 末条）；`compute_results.py` / `unzip_ckpt.py` 的 CLI 改写；既往计划 md 旧路径回改；`check_baseline_env.py` 指纹盲区 R22 修复（禁修——改口径即 G0b 留档指纹当场失配、基线作废）；C4 之外再加同侧 direct-vs-entry 对照 run（1000 步改法已由「A/B 各自对历史锚点」提供等效隔离，不再需要）。

---

# 附录（给人看）

## 附 A、train.py 修改前后的具体流程（控制流）

**修改前**（官方入口，双跑——实际不可运行）：

```
uv run scripts/training/train.py mme_vla_suite --exp-name=...
└─ __main__:
   ① main(_config.cli(), tentative_run=True)      # 第一次：tentative 预热
   │   ├─ 解析 argv → TrainConfig
   │   ├─ initialize_checkpoint_dir   ← 把 checkpoint 目录建了出来
   │   ├─ init_wandb / init_history_config
   │   ├─ 权重加载 + JIT 编译 + init_train_state
   │   └─ 训练循环 step 0..11（step>10 即 break，状态全丢、不落权重，纯 A40 预热）
   ② time.sleep(20)
   ③ main(_config.cli())                           # 第二次：本该是正式训练
       └─ initialize_checkpoint_dir → 目录已存在 + overwrite=False/resume=False
          → FileExistsError ✗ 必崩（官方 shell 不带 --overwrite）
```

所以现状没人走这个入口：本机对拍/测速绕道 bench，集群绕道 `prod_train_once.py`。

**修改后**（单跑，≡ prod 语义）：

```
uv run scripts/training/train.py mme_vla_suite --exp-name=...
└─ __main__:
   ① _record_dir = os.environ.get("TRAIN_RECORD_DIR")     # 可选记录器开关
   ② _cfg = _config.cli()                                  # 解析一次
   ③ [若设 TRAIN_RECORD_DIR] mkdir → 查 metrics.jsonl 不存在 → 换模块全局 wandb 为透明代理
   ④ main(_cfg)                                            # 只调一次，无预热、无 sleep
   │   ├─ init_logging → batch%device 检查 → ★两条 fail-loud 护栏（overwrite/resume、use_history）
   │   ├─ initialize_checkpoint_dir（全新目录，畅通）
   │   ├─ init_wandb / init_history_config
   │   ├─ 权重加载 + JIT 编译 + init_train_state
   │   └─ 训练循环 step 0..num_train_steps-1（无 tentative break）
   │        每 log_interval 步 wandb.log（设了记录器则顺手落 metrics.jsonl）
   │        每 save_interval 步 + 末步 save_state 真落 checkpoint
   ⑤ [finally，若设记录器] 写 run_meta.json + wandb.finish()
```

`train.main` 函数体内的变化只有两处：删三处 tentative 死代码（签名参数、`tentative_run_step` 定义与偏移、break 块）、加两条 fail-loud 护栏（体最前、早于任何计算）——**训练循环的计算一行不变**。代价是丢掉预热轮（第一步吃满 JIT 编译时间），sbatch 的 jax 编译缓存软链本来就是为此而设。

## 附 B、bench_train_steps.py（g0 量具族入口）相比 train.py 增加了什么

它是包在 `train.main` 外面的**位级轨迹记录仪 + 防误用护栏**，训练计算零增改：`import train` → 装观测补丁 → 调一次 `train.main(config)`。⚠ 正因为是 `import`，`train.py` 的 `__main__` 块结构性不执行——这正是 4.1 覆盖不到本轮 `__main__` 改动的原因。

| 增加项 | 替换/包装的对象 | 产出 |
|---|---|---|
| 逐步标量记录 | `train.wandb` → `_WandbProxy` | `metrics.jsonl`（五标量 dec + `float.hex()`，G0 位级对拍主数据源） |
| TrainState 摘要 | `_checkpoints.save_state` → 摘要器 | `param_checksums.jsonl`（params+Adam 动量+EMA 逐叶 sha256，纯字节口径）；**全程不落真权重** |
| 步 0 摘要 | `init_train_state` 包装 | 初始化后立即记一次 |
| 输入摘要 + index 序列 | `TorchDataLoader.__iter__` 包装 | `batch_digests.jsonl`（raw+canonical 双口径）、`index_sequence.json` |
| batch_sampler 旁证 | `create_data_loader` 包装（`BENCH_DUMP_IDX=1`） | `idx_seq.jsonl` |
| 编译缓存计数 | `jax.monitoring` 监听 | `run_meta.json`（判热缓存/冷编译） |
| 护栏 | —— | ≤1200 步、强制关 wandb、禁 overwrite/resume、锁 history_config、强制 `log_interval=1`、五道源码指纹（全为子串探针，无哈希） |

records 喂给 `compare_baseline.py` 出四分项判定行，再过 `tests/g0_gate.py` 出 `G0_EQ=PASS`；测速同样走它（`BENCH_PERF_MODE` / `SAVE_INTERVAL=0` 档关摘要、只留计时）。一句话：多了眼睛、绑了保险、不落权重。

## 附 C、GPU 利用率观测族（util/）调用什么接口、相比 train.py 增加了什么

关键结论：**它们不调训练的任何接口——不 import train、不进训练进程，是纯事后分析器**，输入全是文件。数据由两条独立通道产出：

```
训练进程（train.py，设 TRAIN_RECORD_DIR）──→ metrics.jsonl        # 每步 wall_time → 步时
                                                                      ↘
sbatch 并行起的五路后台采样器 ──────────→ gpu_util_dense.csv 等  ──→ 事后分析器读文件判读
  （nvidia-smi -lms 500 / 15s legacy /                               ↗
   /proc mountstats / meminfo / compute-apps）
```

| 文件 | 读什么 | 出什么 | 谁调它 |
|---|---|---|---|
| `analyze_gpu_util.py` | `metrics.jsonl`（**必需**）+ `env.json` + `gpu_util_dense.csv` + `gpu_util.csv` + `nfs_read.csv` | `RESULT` 六行 + `E2E95_ACCEPT` 五项 + `TREND_OK`（AGENTS 16 口径：util 均值、0% 采样占比、慢步分层） | `gl_train_prod.sbatch` 收尾自动调 |
| `analyze_util.py` | `metrics.jsonl` + `util-lms500.csv`（另一套 csv 列格式，与上者互不通用） | `UTIL_SUMMARY` 一行 + JSON | 无脚本调用，手工用 |
| `downsample_util_csv.py` | `gpu_util_dense.csv` | 抽稀后的归档 csv | sbatch 收尾自动调 |

**相比 train.py 增加了什么**：对训练进程本身**零增加**——不注入、不 patch、不影响任何数值；增加的全部在进程外（硬件侧采样 + 事后判读）。它们与 train.py 的唯一交叉点是 `metrics.jsonl` 这个文件约定：`analyze_gpu_util.py` 拿它算逐步步时，其生产者现为 prod 的 `_WandbProxy`——**这正是本计划要给 train.py 内置 `TRAIN_RECORD_DIR` 记录器的原因**：`prod_train_once.py` 删掉后 `metrics.jsonl` 须由 train.py 自产，否则集群长训收尾判读当场 `FileNotFoundError`。也正因为 analyzer 崩溃在现状 sbatch 下会被吞成 `EXIT_CODE=0`，第十一节的退出码硬化必须同批做。

## 附 D、AGENTS 18 数据逐跳链路图（占位 + 落笔规则）

**本图分两次落笔**：C2 阶段先画「源码推导」部分（下表结构与锚点先定），C3 出 records 后补「机器实测」数字并注明来源文件与 step。C3 留档 `docs/training-doc/v1-singlerun-g0/` 内交付最终两张图（改动前 / 改动后）。

**纪律**：每一跳的 shape/dtype/字节量必须标出处，两类严格分开——

- `[实测]`：来源 `records/batch_digests.jsonl`（canonical 口径含逐键 shape/dtype，首行 `n_keys=12`）与 `index_sequence.json`，注明 step。
- `[推导]`：读代码得出，注明稳定锚点（函数/类/配置键名），**不得冒充实测**。

字节量口径：per-batch 张量占用 = `prod(shape) × itemsize`；NFS 实际读字节不进本图（走 `nfs_read.csv` 另一通道）。每跳必须回答「这一跳有没有改数」——这是第 18 条的核心列。

链路骨架（前后两张图结构相同，因本次改动不在数据链路上；此为**待证结论、由 4.1 的 `BATCH_DIGEST_CANONICAL` 与 `INDEX_SEQ` 两分项机器裁决**，不是免证前提）：

| # | 跳 | 锚点 | shape/dtype/字节 | 改数? |
|---|---|---|---|---|
| 1 | 全局原始 H5 | `/data/hongzefu` 原件 → turbo 暂存副本 | [推导] | 否（逐文件 sha256 同源） |
| 2 | packed store 落盘 | `src/mme_vla_suite/datastore/framesamp_store.py` | [推导] | 建库时一次性，训练期只读 |
| 3 | `FrameSampDataset.__getitem__` | `src/mme_vla_suite/training/framesamp_dataset.py` | [推导] | 待填 |
| 4 | transforms | `transform_dataset(...)`（`dataloader.py`） | [推导] | 待填 |
| 5 | collate → `TorchDataLoader` 输出 | `openpi` `TorchDataLoader.__iter__` | **[实测]** `batch_digests.jsonl` 逐键 | 待填（raw 口径已知 2 键 dtype 统一差异） |
| 6 | sharding → 进模型 | `jax.sharding.NamedSharding(mesh, PartitionSpec(DATA_AXIS))` | [推导] | 否（只分片不改值） |
| — | 观测旁支 | `wandb` → `_MetricsProxy` / `_WandbProxy` | —— | **否**（只读转发，`float.hex()` 从 `reduced_info` 取值） |

改动前后两张图的唯一差异在观测旁支的实现来源（改动前：仅 bench/prod 的 `_WandbProxy`；改动后：新增 train.py 内置 `_MetricsProxy`，仅 `__main__` 路径装载），数据链路第 1–6 跳逐格相同。
