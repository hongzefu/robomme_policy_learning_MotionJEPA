# v3 破坏性单一化重构计划：训练链只留 packed framesamp，建库域自包含隔离，scripts 目录统一

> **本文件是自包含的单一权威文档**（2026-08-29 定稿）：执行本计划的人和 agent 不需要再阅读其他计划 md——决策依据、commit 切片、逐文件改动、对拍闸门、G3 runbook、链路图要点、风险登记全部内联。历史沿革见 git 历史与会话记录。
>
> **锚点**：分支 `v1-dataloader-Restructure`，制定时 HEAD `0ce75be`，工作区 clean。commit 编号沿用 V3.x 序列（V3.7 之后），计划占用 **V3.8–V3.14**。
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

### 两处对已确认退役清单的事实修正（探查后发现，随本计划一并生效）

- **`dump_index_seq.py` 不退役**：它不 import `RoboMMEDataset`，其 `--legacy-root` 实指源 pkl 库（本轮不删）。修正为保留，迁入 `scripts/training/tests/`，仅改帮助文本措辞。
- **dtype-unify 的活工具清单从 4 件扩为 6 件**：`dump_fixture_samples.py`（唯一的逐样本/逐 batch 逐键落盘量具）与 `compare_dtype_fix.py`（`COMPARE_GRAD=` 判定行出口）也保留迁走；目录本身仍整删。

## 三、方案总览

**七个 commit + G3，顺序不能换：**

| # | 提交 | 做什么 | 为什么排这个位置 |
|---|---|---|---|
| V3.8 | 建库域隔离 | 四个共用文件复制入建库域，建库脚本改 import 指向 | 只加文件、只改指向，训练侧零改动；先落地，后面删共用文件时建库链已安全 |
| V3.9 | 数据链单一化 | 删 legacy 读取器与 backend 三态；norm stats 脚本内联最小读取器 | 数据入口先收口，后面改 transforms 才只有一个消费者 |
| V3.10 | transforms 瘦身 | 删 recurrent/symbolic 的键与分支 | 必须在模型改动前：先让废字段断流，模型侧删除就成纯死代码清理 |
| V3.11 | 模型侧单一化 | 删 recurrent/symbolic 模型分支与 stats 返回链 | 依赖 V3.10 已断流 |
| V3.12 | 在线评估改造 | 新写 `FrameSampMemory`，训练/在线脱离 mem_buffer；shared/ 重划 | 不影响训练数值，但动训练侧 import 路径，须在长跑前 |
| V3.13 | 脚本与示例清理 | 删 symbolic 启动菜单与 examples subgoal 子树 | 纯外围，不进训练链路 |
| V3.14 | 目录统一 | 幸存脚本搬入 `scripts/training/` + `scripts/dataset/`，纯 git mv + 路径修正 | 最后搬移，G3 在最终布局上验收整条链 |
| G3 | 正确性长跑 | 全部冻结后从 clean HEAD 跑 1000 步确定性档，对拍 G0 固化产物 bitwise | 一次覆盖七个提交 |

每个 commit 各有 ≤5 分钟的便宜验证（AGENTS 4），另设分段对拍闸门 N1–N5（第二部分五节）——真出问题不必在 1000 步长跑里回溯七个提交。**对拍 A 侧沿用 G0b 固化产物**（`docs/training-doc/v1-grad-baseline-g0b/records/r1/`），不在当前 HEAD 重录基线（G 链「跑一次固化」纪律；G2 已证当前链 bitwise ≡ G0b）。

## 四、scripts/ 顶层前后对比（V3.14 完成后的目标形态）

修改前（现状 22 项）→ 修改后（2 个域文件夹），逐项去向：

| 现状条目 | 性质 | 去向 |
|---|---|---|
| `train.py` | 训练主入口 | → `training/train.py` |
| `finetune_mme_vla_suite.sh` / `finetune_pi05_baseline.sh` | 训练启动脚本 | → `training/`（菜单在 V3.13 已清成三个 framesamp） |
| `eval.sh` / `serve_policy.py` / `compute_results.py` | 评估/部署入口 | → `training/`（symbolic 分支在 V3.13 已删） |
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
| `data-preprocess-GL/` | GL 建库域主体 | → `dataset/gl/` 整目录冻结随迁（含 `gl_submit.py`、sbatch、自己的 `paths.sh`、`legacy/`） |
| `build_dataset.py` | 建库分派入口 | → `dataset/build_dataset.py` |
| `tarxz_h5.py` | 原始 H5 压缩工具 | → `dataset/` |
| `unzip_data.py` | 数据解压工具 | → `dataset/` |
| `finetune_vlm_subgoal_predictor.sh` | symbolic 数据生产配套（vlm_subgoal builder 同域，冻结不改内容） | → `dataset/` |
| `__pycache__/` | 缓存 | 不进 git，忽略 |
| （新增） | 训练域自带 `paths.sh` | `training/paths.sh`（切断对建库域 paths.sh 的跨域 source） |

修改后顶层只剩两项：

```
scripts/
  training/   train.py、finetune_*.sh、eval.sh、serve_policy.py、compute_results.py、
              compute_norm_stats.py、download_pi05_base.py、unzip_ckpt.py、__init__.py、paths.sh
              ├ prod/   ├ bench/   └ tests/
  dataset/    build_dataset.py、tarxz_h5.py、unzip_data.py、finetune_vlm_subgoal_predictor.sh
              ├ gl/     └ pack/
```

## 五、影响面结论

- 已建好的 packed 库与正式训练：零影响，不重建。
- challenge_interface（framesamp-modul）：零改动（三种集成保留，`policy_config`/serving 不动）。
- 未来建库：建库域自包含冻结，产物一字不变。
- 未来 norm stats：`compute_norm_stats.py` 内联最小 pkl 读取器，链路不断。
- G 链量具（bench/preflight/compare）：内容与判据不变，V3.14 搬新路径；历史留档 `docs/training-doc/*/launch.md` 一律不改。

## 六、两条核心保证的原理（给人看）

### 6.1 「建库链产物与原先完整一致」怎么保证

**第 0 层：已有产物根本不被触碰。** 本轮不重建任何库——678 GB 的 4task-gl、packed 库、norm_stats 全部原样。所以「产物一致」要保证的其实是：**将来再跑建库链时，代码行为和今天逐位相同**。

**第 1 层：源码同一性是机器判定的，不靠行为测试猜（COPY_DIFF）。** 四个文件从 `shared/` 复制进 `dataset_builder/` 时，验收判据是 `git show HEAD:shared/<f>.py | diff -u - dataset_builder/<f>.py`：三个叶子文件必须**零差异**，`mem_buffer.py` 的差异必须**恰好 3 行且全是 import 语句**（`shared.` → `dataset_builder.` 的指向替换），函数体一个字符不许动。判定行 `COPY_DIFF=PASS files=4 nonimport_lines=0`。这一过，「计算逻辑没变」就是源码级证明，不是推测。

**第 2 层：依赖闭合，训练侧后续怎么删改都波及不到（IMPORT_ISOLATION）。** 这正是「彻底隔离」和「产物一致」互相成全的地方：V3.9–V3.12 会删 `shared/mem_buffer.py`、瘦身 `shared/data_utils.py`——如果建库脚本还绑在 `shared/` 上，产物一致当场就破。隔离后用双向判据钉死：静态 grep（建库域文件不得出现任何 `mme_vla_suite.shared/training/policies` 引用）+ 动态断言（全新解释器 import 建库入口后 `sys.modules` 里零训练侧模块泄漏——这一条抓静态 grep 抓不到的传递依赖）。判定行 `IMPORT_ISOLATION=PASS builder_leaks=0 train_leaks=0 online_leaks=0`。openpi 底座本轮整文件不碰，不在波及面里。

**第 3 层：同架构行为抽查兜住「万一 diff 看漏」（BUILDER_SPOT + 三方对拍）。**

- `finalize_checks.py` 里现成的 `spot_check` 就是为此设计的量具：从原始 H5 重算 `token_emb_{step}.npy` 与存盘产物**零容差逐位比对**。用改造后的建库域代码对本机真值库 `ref-crossarch`（47 episode，本机 RTX 6000 Ada 产出）抽 32 条跑，判定行 `BUILDER_SPOT=PASS n=32 max_diff=0.000e+00`。
- 另外 ONLINE_MEM harness 顺带做**三方对拍**：同一个 jitted SigLIP、同一批真实帧，分别喂旧 `shared.mem_buffer`、新 `dataset_builder.mem_buffer`，`get_history_feats` 全键逐位——直接证明副本与原件行为逐位相同，零额外 GPU 开销。
- 失败时先跑 **null 对照**（git 提取的旧代码跑同一组抽样）：null 也挂 → 是 2026-08-23 以来的环境漂移（驱动/jaxlib），与重构无关；null 过而副本挂 → import 重接绑错了兄弟模块。

**防退化哨兵（R13）**：`test_guards.py` 加一条用例，断言 `dataset_builder/mem_buffer.py` 的 sha256 等于固化常量——防止将来有人好心把训练侧的改动「同步」回建库域副本，让冻结悄悄失效。

**两处必须知道的边界（计划里已如实登记）**：
1. **禁跑 `finalize_checks` 的 check 子命令、禁对 4task-gl 抽查**——check 子命令无错时会**改写**目标库的 `meta/stats.json` 和 `provenance.json`，而这两个文件在 G0 的数据集指纹里，跑一次基线就作废；且 4task-gl 是 GL A40 产出，本机 Ada 重算跨架构逐位不可得（仓库已有实证报告），对它跑必挂且不说明任何问题。所以只允许 `import spot_check` 函数、只对本机同架构库跑。
2. **GL 侧不重跑**（八节盲区第 4 条）：这套保证证到的是「代码逐字节同一 + 本机同架构复算逐位一致」，不是「在 GL 上重建一遍 678 GB 逐位相同」。后者由「同一代码 + GL 环境未变」推得，诚实标注为推论而非实测。

### 6.2 「G3 对拍 G0 就能证明训练语义没变」怎么成立

**第 0 层：G0 是什么。** `docs/training-doc/v1-grad-baseline-g0b/records/r1/`——重构前训练语义在受控确定性档（`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`）下的一轮真实训练：本机 2×RTX 6000 Ada、b8、seed 42、1000 步，四类产物固化进 git——逐步五标量的 IEEE 浮点位（`scalars_hex.tsv`：loss / grad_norm / llm_grad_norm / mem_enc_norm / param_norm）、12 个摘要步的完整 TrainState（params + Adam 动量 + EMA）逐叶 sha（`param_checksums.jsonl`）、14 个记录步的输入 batch 逐键指纹（`batch_digests.jsonl`，`n_keys=12`）、全部样本抽取顺序（`index_sequence.json`）。

**第 1 层：复跑零抖动已被四份证据钉死。** 四档确定性实验里 D2 与 **D2-cold**（清空编译缓存、独立重编译两次）双 PASS——G3 代码变了、必然现场重编译，对拍依然成立；G0b 两轮千步自证 + G1（dtype 后）+ G2（packed IO 后）复现出同一个 `scalars_hex.tsv` sha256 = `c799a0b2…105757`，等于这条管线的复跑噪声被四份独立证据钉死为**严格零**。

**第 2 层：环境指纹 preflight 排除环境漂移。** 起跑前 `check_baseline_env.py check --baseline <G0b-r1> --dataset v1-store/datasets/4task-gl --steps 1000 --batch-size 8`（**必须带** `XLA_FLAGS`/`XLA_PYTHON_CLIENT_MEM_FRACTION`/`CUDA_VISIBLE_DEVICES` 三个环境变量——它们都在指纹里，裸跑必 FAIL 三项）：逐项比对 uv.lock 全量 sha、torch/jax/jaxlib/numpy/ml_dtypes 版本、GPU 型号与驱动、norm_stats/tokenizer/pi05 权重/数据集抽样指纹等数十项，任一不符 `BASELINE_ENV=FAIL`、基线作废。**指纹不含仓库代码 sha**——PASS 只证「还有资格引用 G0」，绝不能当「代码没改坏」的证据，后者只能由第 3 层回答。

**第 3 层：四维位级对比，任何一跳改一个字节都藏不住。** G3 与 G0 同 seed、同数据、同 b8 跑 1000 步后 `compare_baseline.py <G0b-r1> <G3> --tier g3-vs-g0b` 四分项全部要求逐位：① `SCALARS steps=1000 keys=5 hex_mismatch_steps=0`（每步五标量浮点位）；② `STATE_DIGEST rows=12 mismatch=0`（参数/优化器摘要）；③ `BATCH_DIGEST_CANONICAL rows=14 mismatch=0` + `CANON_CHECK=PASS`（喂进模型的 batch 内容）；④ `INDEX_SEQ=PASS n=8072`（前 8,000 条抽样顺序前缀）。数据装配、变换、模型前向、梯度、优化器更新任何一跳的任何字节差异都会在标量或摘要里位级暴露。收官一行：`sha256(records/scalars_hex.tsv) == c799a0b2…105757`。

**归因唯一化**：第 1 层排除「重跑噪声」、第 2 层排除「环境变了」，于是 **G3 逐位等于 G0 的唯一解释就是七个提交合起来没改训练语义；不等的唯一解释就是改了**——随后按便宜到贵的阶梯定位（`COPY_DIFF` 10 秒 → `IMPORT_ISOLATION` 1 分钟 → `ONLINE_MEM` 10 分钟 → `GRAD_FIXTURE` 15 分钟 → `SMOKE5` 35 分钟），禁止直接重跑 2.5 小时的 G3。

**已知不作判据的两项（预先声明防误读）**：raw 口径 `BATCH_DIGEST mismatch=4 first_bad_step=100 bad_keys=2 (static_image_emb/static_pos_emb)` 是 V2.4b dtype 统一的**预期失配**——与 G2 逐字吻合即正常，**不是** 4 步×2 键则升格为信号；总行 `DET_CHECK=FAIL` 是已拍板不修的工具聚合缺口，分项判读为准。

## 七、全部改动文件清单（新增 / 修改 / 删除 / 纯搬移）

> 按文件索引的总清单；每项标注所属 commit。与第二部分逐 commit 条目同源，冲突时以第二部分为准。

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

**建库域（V3.8，只改 import 行 + 哨兵）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `src/mme_vla_suite/dataset_builder/build_robomme_dataset.py` | 旧建库主流程（`DatasetProcessor`，被 build_shard 复用） | import 行改指 `dataset_builder.mem_buffer` |
| `scripts/data-preprocess-GL/build_shard.py` | GL 8×GPU 分片建库主循环 | 同上一行 import |
| `scripts/data-preprocess-GL/finalize_checks.py` | 建库收尾校验（含 `spot_check` 零容差复算） | 函数内局部 import 改指 |
| `scripts/data-preprocess-GL/compare_datasets.py` | 新旧库三层对拍（v4 验收资产） | 函数内局部 import 改指 |
| `scripts/data-preprocess-GL/test_guards.py` | 建库守卫 pytest | 加 sha256 哨兵用例（断言副本 `mem_buffer.py` 哈希等于固化常量，防发散） |

**训练数据链（V3.9–V3.10）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `src/mme_vla_suite/training/dataloader.py` | 训练 DataLoader 装配入口 | 删 `_resolve_backend` 三态与 RoboMMEDataset import，`create_data_loader` 压成无条件 packed；`DataLoaderImpl`/`_create_framesamp_dataset` 本体一字不动 |
| `src/mme_vla_suite/training/framesamp_dataset.py` | packed 唯一训练数据集 | V3.9 仅改三处注释措辞；V3.10 `_NONE_KEYS` 删 6 项（recur_*4+subgoal2）；V3.12 import 改指 `shared.sampling`；`integration_type=="context"` 断言不动 |
| `src/mme_vla_suite/training/config.py` | 训练配置 + transforms 工厂 + tokenizer | ①RepackTransform 删 6 键；②`TokenizePromptWithSymbolicMemory` 删 symbolic 分支与两个无默认 pop；③`ModelTransformFactory` 删 symbolic 块与 `max_token_len*=2`；④`PaligemmaTokenizer.tokenize` 删 subgoal 形参分支（先改 `_download.maybe_download` 再删 import，R2）；⑤删死类 `LeRobotMMEVLARealRobotDataConfig`、`MMEVLAWeightLoader` |
| `src/mme_vla_suite/shared/data_utils.py` | 训练/在线共享工具 | 删 `even_sampling_indices`（搬走）与 `left_padding_token_emb`（调用者归零）；保留 right_padding + pool |
| `scripts/compute_norm_stats.py` | norm stats 唯一生产链 | 删 RoboMMEDataset 依赖，内联 `_PklSampleDataset`（等价性八要点）；加 `--output-dir` 防覆盖生产 norm_stats.json |

**模型侧（V3.11）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `src/mme_vla_suite/models/integration/history_pi0.py` | 模型主体（HistoryPi0） | `create`/`inputs_spec`/`__init__`/`embed_memory`/`embed_prefix`/`compute_loss`/`sample_actions` 七处删 recurrent/symbolic 分支与 stats 返回链；expert/modulation 构图与 lazy_init 参数一字不动 |
| `src/mme_vla_suite/models/integration/history_observation.py` | 训练/在线共用的观测数据类 | 删 `recur_*`8 + `symbolic_*`2 字段及四处镜像；`static_*` 与基类字段全留 |
| `src/mme_vla_suite/models/representation/mem_encoder.py` | memory token 投影器 `FeatureEncoder` | 删 `encoder_recur`/`encode_recurrent_memory`/`ouput_dim_for_recur`/`ndim==5` 分支 |
| `src/mme_vla_suite/models/representation/percep_mem.py` | perceptual memory 编码模块 | 删 `ouput_dim_for_recur=None` 实参与无消费点的 `mem_type` 赋值 |
| `src/mme_vla_suite/models/representation/utils.py` | 初始化器等工具 | 保留 `kernel_init`+`kernel_init_out_proj`（history_gemma 依赖），删 ttt/rmt/rope 系函数 |
| `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context.yaml` | 唯一训练配置 | 值一字不动；可选加中文字段注释（R6 schema 补偿，须 G3 前完成） |
| `scripts/train.py` | 训练主入口 | `train_step` 三返回改二返回（注解同 hunk）、删 `get_stats` 与 recurrent 统计打印、`ptrain_step` out_shardings 同步；四个 bench 指纹字符串一字不动 |

**在线评估侧（V3.12）**

| 文件 | 作用 | 改动内容 |
|---|---|---|
| `src/mme_vla_suite/policies/policy.py` | 在线评估 policy（serving/challenge 共用） | 删 mem_buffer import 换 `FrameSampMemory`；`_prepare_mem_buffer` 压单支；infer 断言改 `n_steps`+显式 raise；`_prepare_history` 只留 frame_sampling |
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
| `scripts/smoke-local/run_2gpu_epoch_bench.sh` | G 链正式起跑驱动 | V3.9 删 env.json 的 `MMEVLA_DATA_BACKEND`/`backend_source` 两行；V3.14 自引用路径 + source 训练域自带 paths.sh |
| `scripts/smoke-local/bench_train_steps.py` | G 链 bench 入口（源码指纹护栏所在） | 仅 V3.14 的 sys.path 插入改指 `scripts/training`；护栏与判据零改动 |
| `scripts/train-prod/gl_train_prod.sbatch` | 正式训练 sbatch | 删 `MMEVLA_DATA_BACKEND` 四处（export/注释/env.json 键/echo，set -u 下必须同删）；V3.14 analyze_gpu_util 路径更新 |
| `scripts/train-prod/prod_train_once.py` | 正式训练薄启动器 | V3.14 sys.path 插入 smoke-local→training/bench |
| `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | 历史 e2e sbatch | V3.9 同删 backend 四处（该文件 V3.14 随目录退役） |
| `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | util 稳态分析（prod 趋势在用的活量具） | 读历史 env.json 逻辑不改，加一行历史遗留注释；V3.14 迁 `training/bench/` |
| `scripts/eval.sh` | 在线评估启动 | 删 symbolic/MemER 分派与菜单，默认 MODEL_TYPE 改 `perceptual-framesamp-modul` |
| `scripts/finetune_mme_vla_suite.sh` | 训练启动 | 注释菜单 14 变体 → 3 个 framesamp |
| `scripts/compute_results.py` | 评估结果统计 | 删 `--symbolic_type` 与 symbolic 分支，默认 model_dir 改 framesamp-modul |
| `examples/robomme/eval.py` | 评估客户端 | 删 subgoal 全链（import/参数/注入/存档分支） |
| `examples/robomme/utils.py` | 客户端工具 | 删 `SUBGOAL_TYPES` 与 record 的 subgoal 形参/文字叠加 |
| `examples/robomme/env_runner.py` | 环境 runner | 删两个 subgoal oracle property |
| `pyproject.toml` | 项目配置 | V3.14 testpaths 指新目录 |
| `README.md` | 仓库说明 | V3.14 规范命令改新路径 |

### 7.3 删除文件

| 文件/目录 | 作用（删除理由） | commit |
|---|---|---|
| `src/mme_vla_suite/training/dataset.py` | legacy 数据链（RoboMMEDataset/SampleDataset），packed 唯一化后无消费者 | V3.9 |
| `scripts/data-pack-framesamp/compare_batches.py` | packed-vs-legacy 对拍工具，A 侧消失 | V3.9 |
| `scripts/bottleneck-bench/gl-dataloader/` 整目录 | 调私有 `_resolve_backend` 的历史基准 | V3.9 |
| `src/mme_vla_suite/models/representation/{recur_mem,rmt,ttt}.py` | recurrent 记忆模型（RMT/TTT），唯一消费链已删 | V3.11 |
| `models/config/robomme/` 11 个非 framesamp yaml + `models/config/base.yaml` | tokendrop/recurrent/symbolic 配置与无引用模板 | V3.11 |
| `src/mme_vla_suite/shared/mem_buffer.py` | 训练/在线均已脱钩，建库域有冻结副本 | V3.12 |
| `src/mme_vla_suite/shared/siglip_tokenizer.py` | 训练/在线零使用（在线注入模型编码器），建库域有副本 | V3.12 |
| `examples/robomme/subgoal_predictor.py` + `subgoal_prediction/` 子树 | symbolic 在线预测链 | V3.13 |
| `scripts/bottleneck-bench/` 余部、`bottleneck-bench-v2/`、`dtype-unify/`（活件迁走后） | 历史专题目录退役 | V3.14 |
| 旧目录空壳：`smoke-local/`、`train-prod/`、`data-preprocess-GL/`、`data-pack-framesamp/` | 内容已迁入两域 | V3.14 |

### 7.4 纯搬移（内容零改动，V3.14，git mv）

- `data-preprocess-GL/` 整目录 → `dataset/gl/`（冻结随迁）；`pack_framesamp_store.py`/`probe_layout.py`/`run_pack.sh`/README → `dataset/pack/`；`build_dataset.py`/`tarxz_h5.py`/`unzip_data.py`/`finetune_vlm_subgoal_predictor.sh` → `dataset/`。
- `smoke-local/` 四件 + `compare_online_memory.py` → `training/bench/`；`train-prod/` → `training/prod/`；`analyze_util.py`/`_common.py`/`test_padding_dtype.py` → 各自去处见四节对照表；`train.py` 等顶层散件 → `training/`。

---

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

- 统一前置：`cd <仓库根>`，`export UV_LINK_MODE=copy`；pytest 一律显式路径（pyproject testpaths 含 scripts，裸跑会全量收集）。
- **保留路径逐字清单（G3 bitwise 前提，逐条 review 打勾）**：
  - 数据：`FrameSampDataset.__getitem__` 全体（身份互校 / actions 截断 / even_sampling_indices / `_pad` / reshape+repeat / `_normalize_state`）；`FrameSampStore` 与 `datastore/` 全体；`even_sampling_indices` 函数体（搬模块可以，改一个字符不行）。
  - loader：`DataLoaderImpl`、`_create_framesamp_dataset` 全体、`transform_dataset`/`TorchDataLoader` 构造参数逐字不变；**openpi `data_loader.py` 整文件不碰**。
  - transform：`RepackTransform` 保留 9 键名与顺序；`RoboMMEInputs` 的 state/image/image_mask/actions/prompt 构造；`DeltaActions(make_bool_mask(7,-1))`；`InjectDefaultPrompt`/`ResizeImages(224,224)`/`PadStatesAndActions` 位置参数；`PaligemmaTokenizer.tokenize` 的 else 支与 padding/截断。
  - 模型：非 expert 分支 `_gemma.Module(configs=[paligemma, action_expert], ...)` 与 `lazy_init(use_adarms=[False,True], mem_mods=[False,False])`；`action_in_proj`/`time_mlp_in`/`time_mlp_out`/`action_out_proj` 构造顺序（rng 消耗序）；`PerceptualMemory`/`FeatureEncoder` 的 pos_proj→encoder_static 构造顺序与 kernel_init；`embed_memory` perceptual 体、`embed_prefix` context 分支与 ar/na mask 累积规则、`embed_suffix` 全体、`compute_loss` 的 rng split 顺序与 `beta(1.5,1)*0.999+0.001`；三参版 `make_attn_mask`。
  - 训练：`train_step` 的 `fold_in(rng, state.step)`/`DiffState`/`tx.update`/EMA/info 键（**含 `mem_enc_norm`**）；`init_train_state` 全体；main 的 mesh/sharding/循环结构；**train.main 四个指纹字符串一字不动**：`wandb.log(reduced_info`、`_checkpoints.save_state(`、`init_train_state(`、`create_data_loader(`。
- **graphdef 声明**：本轮删除若干无消费点静态属性（`percep_mem.mem_type`、`encoder_recur`、`HistoryPi0.representation_type`）会改 `nnx.split` 的 static 侧但不改 params/rng/loss——对拍判据不得纳入 graphdef。
- **None 键实证**（第一块不做全量 dump 的依据）：G0b-r1 的 `batch_digests.jsonl` 首行 **`n_keys=12`** 且键集不含任何 `recur_*`/subgoal 键——`tree_flatten`/`_collate_fn` 把 `None` 当空子树剪掉，删恒 None 键在交付面是可证的恒等变换。subgoal 是 pkl 里的真字符串、靠 `TokenizePromptWithSymbolicMemory` 两个无默认 `pop` 拦下：删除必须成对（Repack 去键 + 去 pop），做漏即 `n_keys≠12`、烟测/G3 必挂。

## 一、commit 逐文件改动清单

### commitV3.8 — 建库域隔离（训练侧零改动）

- 新增 `src/mme_vla_suite/dataset_builder/{data_utils,posemb_3d,siglip_tokenizer}.py`：自 `shared/` **逐字节复制**（三者无 mme 内部 import）。
- 新增 `src/mme_vla_suite/dataset_builder/mem_buffer.py`：复制后**恰好改 3 行 import**（模块级 data_utils 星导入、`prepare_buffer` 分支内 `PosEmb3D` 与 `SigLipTokenizer` 两个懒 import → 全指 `dataset_builder.*`），函数体一字不动；**diff 差异行数==3 是验收判据**。
- `dataset_builder/build_robomme_dataset.py`、`scripts/data-preprocess-GL/{build_shard,finalize_checks,compare_datasets}.py`：import 行改指 `dataset_builder.mem_buffer`（注释保持冻结）。
- 防发散哨兵：`scripts/data-preprocess-GL/test_guards.py` 加用例断言 `dataset_builder/mem_buffer.py` sha256 == 固化常量（建库域四件自 V3.8 起冻结，与 shared/ 不再同源）。
- `dataset_builder/` 无 `__init__.py`（隐式命名空间包，与全仓一致），不需建包文件。
- 验证（≤2min）：三文件 diff 零差异 + mem_buffer diff==3 行；建库域 import 闭环；`grep -rn "mme_vla_suite.shared" src/mme_vla_suite/dataset_builder scripts/data-preprocess-GL` 为空；`uv run pytest scripts/data-preprocess-GL/test_guards.py -q`；ruff。

### commitV3.9 — 数据链单一化

- **删** `training/dataset.py` 整文件（`SampleDataset`/`RoboMMEDataset`/`load_vector_file`）。
- `training/dataloader.py`：删 `_resolve_backend` 与 RoboMMEDataset import；`create_data_loader` 压成无条件 `_create_framesamp_dataset(...)`；`import os` 保留（`_create_framesamp_dataset` 仍用）；`DataLoaderImpl` 与 `_create_framesamp_dataset` 本体一字不动（bench `_install_idx_probe` 依赖 loader 内部结构）。
- `framesamp_dataset.py`：仅改三处「与 RoboMMEDataset 逐字一致」注释措辞；代码零改动（`_NONE_KEYS` 在 V3.10 才动）。
- `scripts/compute_norm_stats.py`：删 RoboMMEDataset import，内联 `_PklSampleDataset`。**等价性八要点**：①`__len__` 读 `meta/stats.json` 的 `execution_samples` 优先否则 `total_samples`；②裸 `pickle.load(data/{idx}.pkl)` 无任何转换；③`actions[:action_horizon]` 截断；④不建 mem_buffer 不读 features（旧代码 `history_config=None` 即如此）；⑤state 原始值不归一化（`compute_norm_stats=True` 语义）；⑥random 分支全不触发、连 `random.seed` 都不需要；⑦`*_online` pop 可省略（RepackTransform 白名单丢弃未列键）；⑧尾部补 None 键集合 == V3.10 后 RepackTransform 键集合（`static_*`4 + `prompt`）。新增可选 `--output-dir`（防验证覆盖生产 `norm_stats.json`；动手前先 sha256 备份 `v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`）。
- `scripts/dtype-unify/{single_step_grad,dump_fixture_samples}.py`：`RoboMMEDataset(...)` → `mme_vla_suite.training.dataloader._create_framesamp_dataset(...)`（参数同名直传，`dataset_path` 指 packed 库）；其源码指纹护栏四条本轮全部仍成立，不放宽。
- `scripts/dtype-unify/run_dtype_grad.sh`：`--dataset-path` 改 `${DATASET_PATH:-${GL_DATASET}}`（`GL_DATASET` 在 paths.sh 是 readonly，环境覆盖不了）——GRAD_FIXTURE 前置。
- **删** `scripts/data-pack-framesamp/compare_batches.py`（legacy A 侧消失，历史结论已固化在 docs/）。
- `test_pack_guards.py`：删 backend 三态用例（含 `pytest.raises(..., match="MMEVLA_DATA_BACKEND")` 断言），保留 packed 闸与 G1–G14 全部守卫。
- `dump_index_seq.py`：仅改 `--legacy-root` help 文本（「源 pkl 库根」）。
- **删** `scripts/bottleneck-bench/gl-dataloader/` 整目录（`dataloader_bench.py` 调私有 `_resolve_backend`；配套 sbatch/submit 同批）。
- `MMEVLA_DATA_BACKEND` 清理（**set -u 下 export 与 echo 必须同删**，漏删 echo 当场失败）：`train-prod/gl_train_prod.sbatch`（export/注释/env.json 键/echo 四处）、`bottleneck-bench-v2/gl_e2e_fix.sbatch`（同四处）、`run_2gpu_epoch_bench.sh`（env.json 的 `MMEVLA_DATA_BACKEND` 与 `backend_source` 两行）；`analyze_gpu_util.py` 读历史 env.json 的逻辑**不改**（加一行历史遗留注释）。
- 验证（≤5min）：死引用 grep 清零（白名单 analyze_gpu_util）；train/compute_norm_stats/dataloader import 闭环 + `assert not hasattr(dl,'_resolve_backend')`；pytest test_pack_guards；确定性档 `STEPS=5 SAVE_INTERVAL=0 EXP_NAME=v1-restructure-smoke RUN_TAG=v39-smoke KEEP_JAX_CACHE=1` 烟测（各 commit 复用同一 EXP_NAME 命中编译缓存，RUN_TAG 每次不同；首次冷编译超 5min 按 AGENTS 7 进 tmux）。
- **闸门 N1**：烟测记录 `compare_baseline.py --tier v39-smoke` vs G0b：`SCALARS steps=5 hex_mismatch_steps=0` + `INDEX_SEQ=PASS n=40`，PASS 才进 V3.10。

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
- **闸门 N2（SMOKE5）**：确定性档 `STEPS=5 SAVE_INTERVAL=5 BATCH_DIGESTS=1 WARMUP_STEPS=0` → `compare_baseline.py --tier smoke5` vs G0b：SCALARS 5 步 0 失配、`BATCH_DIGEST rows=3 mismatch=0`（raw 步 0/1/2 必过——V2.4b 预期失配首现于步 100）、`CANON_CHECK=PASS steps=3`、**`n_keys=12`**（键集回归闸）、INDEX_SEQ n=40；总行 DET_CHECK 不作结论。SMOKE5 约 30–40min，属对拍闸门不受单 commit 5min 约束，进 tmux。

### commitV3.11 — 模型侧单一化 + stats 链整删

- `history_pi0.py`：`create()` 删 symbolic 块；`inputs_spec` 只留 perceptual（外层 `use_history` 二分保留，pi05_baseline 依赖）；`__init__` 删 recurrent/symbolic elif 与 `representation_type` 属性，`PerceptualMemory` 改无条件构造，expert/else 的 llm 构造与 `lazy_init` 参数一字不动；`embed_memory` 压单体去 stats；`embed_prefix` 删 symbolic 分支、语言分支收敛 `if obs.tokenized_prompt is not None`、返回 4 元组；`compute_loss` 解包改 4 元组、`!= "symbolic"` 条件简化为 `if self.use_history`、单返回；`sample_actions` 三处 5 元组解包改 4 元组（expert/modulation/else 分支逻辑不动）。
- `history_observation.py`：删 `recur_*`8 + `symbolic_*`2 字段及 from_dict/to_dict/from_base_obs/preprocess_observation 四处镜像；`static_*` 与基类字段全保留（from_dict 按名 `data.get`，多余键静默忽略，删字段安全）。
- `scripts/train.py`：`train_step` 返回注解改二元组（`@at.typecheck` 校验，与 return 同 hunk）；`loss_fn` 去 stats、`nnx.value_and_grad` 去 `has_aux`；`return new_state, info`；删 `get_stats` 整函数（`import numpy as np` 被 wandb 图像块用，不删）；`ptrain_step` out_shardings 二元组；主循环解包同步；删 recurrent 统计打印块。**四个指纹字符串一字不动**。
- `scripts/dtype-unify/single_step_grad.py`：`_grad_only.loss_fn` 同步去 stats/has_aux（**stats 第二耦合点**，与 train.py 同 commit，漏改即解包炸）；建议给 `_guard_train_step_source` 新增一个锁新形态的 needle。
- **删** `representation/{recur_mem,rmt,ttt}.py`；`mem_encoder.py` 删 `encoder_recur`/`encode_recurrent_memory`/`ouput_dim_for_recur`/`ndim==5` 分支（percep_mem 显式传 None，删除不改参数树与 RNG 消耗）；`percep_mem.py` 删 `ouput_dim_for_recur=None` 实参与 `mem_type` 赋值；`representation/utils.py` 保留 `kernel_init` + `kernel_init_out_proj`（history_gemma 依赖），删 ttt/rmt/rope 系函数（删前逐个 grep 复核）。
- **删** 11 个非 framesamp yaml + `base.yaml`；三个 framesamp yaml **一字不动**（形制断言与 `_EXPECTED_HISTORY_CONFIG` 依赖字段与文件名）；schema 文档化补偿：给 context yaml 加中文注释或记入 `datastore/README.md`——若对拍口径含文件 sha 须在 G3 前 commit 完成。
- 验证（≤5min）：三个 integration 的 `HistoryPi0Config.inputs_spec` 构图冒烟；train.py 指纹断言 + `not hasattr(train,'get_stats')`；ruff；确定性档 STEPS=5 烟测。
- **闸门 N3（GRAD_FIXTURE）**：commit 后（`run_dtype_grad.sh` 有 clean-HEAD 硬闸）跑 `DATASET_PATH=.../4task-gl-framesamp bash scripts/dtype-unify/run_dtype_grad.sh` + `compare_dtype_fix.py --grad-a v1-store/dtype-unify/v1-dtype-p5-grad-grad --grad-b <新>`：`COMPARE_GRAD=PASS kinds=3 mismatches=0`（`allfull` 阴性对照必过）。**A 侧直接引用已固化的 `docs/training-doc/v1-dtype-p5-grad/records/grad_summary.json`**（三定点 batch × 32 梯度叶 sha + loss_hex，`same_origin=PASS`），无需现场采集。这是 `has_aux=True→False`（本轮唯一可能动 jaxpr 的改动）的直接证伪器。不过不进 V3.12。

### commitV3.12 — 在线评估侧 + shared/ 重划

- 新增 `policies/framesamp_memory.py`：`FrameSampMemory`——构造必填 `vision_enc_fn`；`n_steps` 属性替代旧代码摸 `_history_feats` 私有；`add_buffer` 逐跳与旧实现同式，仅三处合法差异（只算 `token_per_image` 一档池化、`jax.device_get` 提到循环外一次、不存 image_pixels/多余档位——四条不可观测性论证：pool 是纯函数各档独立、PosEmb3D 无 RNG 无参数、image_pixels 唯一消费者是 token_drop 打分与死码可视化、装配只读三键）；重复 step 显式 raise 不用 assert；`prepare_frame_sampling` 装配与旧 `_prepare_frame_sampling` 逐字同式，**必须复用 `right_padding_token_emb` 不得改写成 `_pad` 预分配版**（前者短样本分支 concatenate 会提升 f64，旧在线行为如此，本轮在线侧只换模块不换数值）。**禁把 encode+pool 包进新 jax.jit**（融合变则 bf16 累加序可能变位）。
- `policies/policy.py`：删 mem_buffer import；`_prepare_mem_buffer` 压单支；infer 断言改 `n_steps` + 显式 raise（禁 assert）；`_prepare_history` 只留 frame_sampling 四行赋值；`add_buffer`/`reset`/`_normalize_state` 不动。
- **闸门 N4（ONLINE_MEM，commit 前的工作区窗口内跑）**：新模块已落地、`shared/mem_buffer.py` 尚未删——同进程共存 import 做 A/B，PASS 后才删旧文件、才 commit。明细见五节。
- **删** `shared/mem_buffer.py`、`shared/siglip_tokenizer.py`（训练/在线零使用：在线注入模型 `vision_enc_fn`，建库域有副本）。
- 新增 `shared/sampling.py`：原样搬 `even_sampling_indices`（只 import numpy，不拉 flax——解 dataloader worker 的 jax/flax 导入负担）；`shared/data_utils.py` 删 `even_sampling_indices` 与 `left_padding_token_emb`（唯一消费者已删），保留 `right_padding_token_emb` + `pool_tokens_to_size`；`framesamp_dataset.py` 与 `test_pack_guards.py` 的 import 改指 `shared.sampling`；`spawn_matrix.py` docstring 措辞更新。
- 验证（≤3min）：`grep mem_buffer|siglip_tokenizer` 于 training/policies/models/serving 为空；challenge_interface + policy import 闭环；`import shared.sampling 后 'flax' not in sys.modules`；pytest（test_pack_guards + test_padding_dtype）；STEPS=5 烟测。

### commitV3.13 — scripts/examples 清理

- `eval.sh`：删 symbolic/MemER 分派与菜单，默认 MODEL_TYPE 改 `perceptual-framesamp-modul`；`finetune_mme_vla_suite.sh` 菜单只留三个 framesamp；`compute_results.py` 删 `--symbolic_type` 及 symbolic 分支、默认 model_dir 改 framesamp-modul。
- `examples/robomme/`：eval.py 删 subgoal 全链（import/参数/注入/存档分支）；删 `subgoal_predictor.py` 与 `subgoal_prediction/` 子树；utils.py 删 `SUBGOAL_TYPES` 与 record 的 subgoal 形参；env_runner.py 删两个 oracle property。
- 不动：`build_dataset.py`、`dataset_builder/vlm_subgoal_*`、`finetune_vlm_subgoal_predictor.sh`（建库域/数据生产，冻结）。
- 验证（≤2min）：bash -n；ruff；examples ast.parse；`pytest --collect-only -q src scripts packages` 零 error；`grep symbolic|subgoal`（排除建库域）为空。

### commitV3.14 — scripts/ 目录统一（纯搬移 + 路径修正）

目标形态：

```
scripts/
  training/                  ← 训练域
    train.py、finetune_mme_vla_suite.sh、finetune_pi05_baseline.sh、eval.sh、serve_policy.py、
    compute_results.py、compute_norm_stats.py、download_pi05_base.py、unzip_ckpt.py、__init__.py
    prod/     ← train-prod/（gl_train_prod.sbatch、prod_train_once.py、downsample_util_csv.py）
    bench/    ← smoke-local/ 四件 + compare_online_memory.py + analyze_gpu_util.py（自 bottleneck-bench-v2）
                 + analyze_util.py（自 dtype-unify）
    tests/    ← test_pack_guards.py、spawn_matrix.py、dump_index_seq.py、test_padding_dtype.py、
                 single_step_grad.py、dump_fixture_samples.py、compare_dtype_fix.py、_common.py、
                 run_dtype_grad.sh
    paths.sh  ← 训练域自带（切断对建库域 paths.sh 的跨域 source）
  dataset/                   ← 数据集预处理域（自包含隔离域）
    gl/       ← data-preprocess-GL/ 整目录（内容冻结，只随目录移动）
    pack/     ← pack_framesamp_store.py、probe_layout.py、run_pack.sh、README
    build_dataset.py、tarxz_h5.py、unzip_data.py、finetune_vlm_subgoal_predictor.sh
```

（顶层散件逐项去向的完整对照表见第一部分四节。）

- 随后删除空壳/退役目录：`bottleneck-bench/` 余部、`bottleneck-bench-v2/`、`dtype-unify/`、`data-pack-framesamp/`、`smoke-local/`、`train-prod/`、`data-preprocess-GL/`。
- 路径修正点：`prod_train_once.py` 的 sys.path 插入（smoke-local→training/bench）；`gl_train_prod.sbatch` 的 analyze_gpu_util 路径；`run_2gpu_epoch_bench.sh` 自引用路径与 paths.sh source；`pyproject.toml` testpaths；README 规范命令；`bench_train_steps.py`/`single_step_grad.py` 等 `sys.path.insert(0,'scripts')` 类插入改指 `scripts/training`。**历史留档 docs/training-doc/*/launch.md 一律不改。**
- 验证（≤5min）：全仓 grep 旧目录名（排除 docs/ 历史留档与 v1-store）零残留；pytest --collect-only 零 error；`STEPS=5` 烟测走新路径。
- **预 G3 闸**：SMOKE5 在最终布局上重跑（同 N2 判据）。

## 二、对拍闸门总表

| 闸门 | 位置 | 内容 | 判据 |
|---|---|---|---|
| N1 | V3.9 tip | 确定性档 STEPS=5 烟测 → compare vs G0b | SCALARS 5 步 0 失配、INDEX_SEQ n=40 |
| N2 | V3.10 tip | SMOKE5（带摘要） | SCALARS/raw BATCH_DIGEST 步 0-2/CANON/**n_keys=12**/INDEX_SEQ |
| N3 | V3.11 tip | GRAD_FIXTURE vs 固化 A 侧 `v1-dtype-p5-grad` | `COMPARE_GRAD=PASS kinds=3 mismatches=0` |
| N4 | V3.12 工作区（commit 前） | ONLINE_MEM：FrameSampMemory vs 旧 MemoryBuffer A/B | POS_TABLE / ENC_LAYER / ASSEMBLY 三层逐位 |
| N4c | V3.8 起持续 | COPY_DIFF + IMPORT_ISOLATION | 见五节 |
| 预 G3 | V3.14 tip | SMOKE5 重跑（新布局） | 同 N2 |
| N5 | V3.14 后 clean HEAD | **G3** 1000 步 vs G0b | 四分项 bitwise（六节） |

**失败定位阶梯（严格从便宜到贵，G3 失败不得直接重跑 G3）**：COPY_DIFF(10s) → IMPORT_ISOLATION(1min) → ONLINE_MEM(10min,1卡) → GRAD_FIXTURE(15min,2卡) → SMOKE5(35min,2卡) → G3(2.5h,2卡)。

## 三、链路图（AGENTS 18 两张图，G3 docs commit 落档）

三张逐跳表要点（完整表随 G3 留档落 `docs/training-doc/<run_name>/`）：

- **图 A 训练链（25 跳）**：前后基本同构，删除的只有旁路——A1 backend 分派、A2 legacy 数据集、A13 `_NONE_KEYS` 缩减（None 非叶子零贡献，n_keys=12 实证）、A14/A15 Repack/Inputs 去 6 键、A18 tokenizer 死分支、A23 Observation 删字段（叶子集不变）、A24 模型死分支与 stats 链（`stats≡None` 非叶子，论证 HLO 中性、G3 实证）。A3–A12、A16–A22、A25 全部逐字保留并标形状/dtype/字节量（packed 交付 per-batch ≈30.9 MiB；A12/A16 的 state 归一化输出恒 f64；A25 的 info 五标量含 `mem_enc_norm`——scalars_hex 第 5 列，丢了 compare_scalars 会静默跳过）。
- **图 B 在线评估链（15 跳）**：B0 pos 表 1.01GiB→192MiB（只算 4x4 一张，逐位不变仍设 POS_TABLE 判据）、B6 池化 ×3→×1、B8 存储 ≈780KB/步→≈112KB/步、B9 token_drop 整删、B14 客户端 subgoal 注入删除（无对拍，盲区）；B2–B5、B7、B10–B13 执行分支逐字保留；B12 四元组 `(img (512,2048) bf16, pos (512,768) f32, state (512,8) f32, mask (512,) bool)` 为 A/B 判据锚点。
- **图 C 建库链（7 跳）**：C1–C4 只改 import 边、C5 计算跳逐字节保留（COPY_DIFF 证明）、C6 产物 `token_emb_{step}.npy` 7 键 602,951 B 逐位不变、C7 新增反向 import 护栏。

## 四、（并入五节）

## 五、第一块：非训练轻量对拍明细

**c-1 COPY_DIFF（秒级，主判据）**：`git show HEAD:src/.../shared/$f.py | diff -u - src/.../dataset_builder/$f.py` ×4；PASS = 差异只在 import 行且仅 `shared.→dataset_builder.` 替换。判定行 `COPY_DIFF=PASS files=4 nonimport_lines=0`。

**c-2 IMPORT_ISOLATION（秒级）**：静态 grep 双向（建库域不 import shared/training/policies/models/datastore；训练/在线/量具不 import dataset_builder）+ 动态新解释器 `sys.modules` 泄漏断言双向。判定行 `IMPORT_ISOLATION=PASS builder_leaks=0 train_leaks=0 online_leaks=0`。

**a ONLINE_MEM（8–12min，1 卡 ~10GB，CUDA_VISIBLE_DEVICES=0）**：
- 新脚本 `compare_online_memory.py`（先放 smoke-local，V3.14 随迁 bench/），判定行体例同 compare_baseline。
- 消除编译非确定性：进程内只构造一次 `enc = jax.jit(SigLipTokenizer().__call__)`，同一 callable 注入 A/B 双方（需 `OPENPI_DATA_HOME=v1-store/models`）；不走 `prepare_buffer` 默认自建分支。选 SigLipTokenizer 而非 14GB pi05：更轻且与建库域同编码器，一套 harness 兼做**三方对拍**（旧 `shared.mem_buffer` / 新 `dataset_builder.mem_buffer` / 新 `framesamp_memory` 的 `get_history_feats` 逐位）——c-1 PASS 时 BUILDER_SPOT 降为可选。
- 输入：真实 h5 帧为主（`/data/hongzefu/robomme_data_h5_v2_4env400ep/record_dataset_ButtonUnmask.h5`，(256,256,3) u8——必须真实 256 帧才走 resize_with_pad 跳）+ 合成极值帧验不炸（非判据）。
- 步位网格（覆盖 even_sampling 全分支）：0/1/2、15/16、**30**（恰 1 行填充）、**31**（恰填满零填充）、**32**（首进 linspace 含重复索引）、33/34、100/291、585；越界探针 4095 只跑 A 侧，新模块表更短时必须 fail-loud。
- 三层判据（复用 `dtype-unify/_common.py` 的 `leaf_sha256`＝sha256(dtype‖shape‖tobytes)，bf16 位型安全；**禁 allclose/==**）：`POS_TABLE=PASS`（4x4 pos 表共同行全表）→ `ENC_LAYER=PASS steps=13 keys=3 mismatch=0`（image_emb_4x4/pos_emb_4x4/state_emb；A 侧多出的 8x8/2x2/image_pixels 为白名单删除不比）→ `ASSEMBLY=PASS steps=13 mismatch=0`（四元组）。分层免费定位编码跳 vs 装配跳。
- >5min 适用 AGENTS 17：产物落 `v1-store/bench/online-mem/<TAG>/`，留档并入 G3 run 目录 `records/block1/`。
- 兜底 A 侧（回归重跑时）：`git show <BASE>:...` 提取四文件到 scratch + sed 改包名——必须连 data_utils/posemb_3d 一起提取，否则 A 侧绑到 B 侧新实现、退化为自比。

**c-3 BUILDER_SPOT（可选确认档，5–8min，1 卡）**：**必须对 `v1-store/datasets/ref-crossarch`（本机 47 ep 同架构库）跑，禁对 4task-gl**（GL 是 A40 产物，跨架构逐位不可得，已有实证报告）。做法：直接 `import spot_check` 函数调用，n=32、seed=20260829。判定行 `BUILDER_SPOT=PASS n=32 max_diff=0.000e+00`。失败第一步跑 null 对照（git 提取旧 mem_buffer 同 picks）分「环境漂移 vs 拷贝错」。**禁跑 `finalize_checks.py` 的 cmd_check 子命令**——无错时会改写目标库 `meta/stats.json` 与 `provenance.json`，对 4task-gl 跑等于污染 G0 数据集指纹、基线作废。

**b-1 GRAD_FIXTURE / b-2 SMOKE5**：见一节 V3.11/V3.10 闸门。串行纪律：b-1/b-2 与 G3 不得并行（显存/autotune）；a 与 c-3 可同机并行，均须在 b-1 之前完成。第一块合计 ≈55–75 min。

## 六、第二块：G3 runbook

**前置门（全过才起跑）**：工作区 porcelain 空、从 clean HEAD 起跑；第一块五判定行全 PASS（COPY_DIFF/IMPORT_ISOLATION/ONLINE_MEM/GRAD_FIXTURE/SMOKE5）；bench 四道源码护栏 + `TorchDataLoader.__iter__` 三 needle 在新 train.py 上仍成立；`train_step` info 仍含 `mem_enc_norm`；packed 库 `status=verified` 且 `MMEVLA_FRAMESAMP_ALLOW_*` 未设；单 epoch 1000×8=8,000 < 395,289；**run_name 起跑前经用户确认**（AGENTS 6；建议 `v1-postclean-g3`，EXP_NAME=RUN_TAG 独立编译缓存）。

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
tmux new-session -d -s g3 "set -o pipefail; cd <REPO_ROOT>; \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 \
  EXP_NAME=<run_name> RUN_TAG=<run_name> \
  DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
  PYTHONUNBUFFERED=1 bash scripts/training/bench/run_2gpu_epoch_bench.sh \
    2>&1 | tee v1-store/logs/<run_name>-driver.log; \
  echo \"EXIT_CODE=\$?\" >> v1-store/logs/<run_name>-driver.log"
```

摘要步集与 G0 对齐（state 12 次：0/每 100/299/999；batch 14 行）。预计 2–2.5h。launch.md 注明 env.json 不再含 backend 字段（量具字段变更，preflight 不比对该字段）。

**盯日志**：Monitor 挂 driver.log，`grep --line-buffered` 过滤 `^Step (0|[0-9]*00|299|999):|BENCH_|EXIT_CODE|Traceback|Error|OOM|digest`；中间过滤级一律行缓冲（stdbuf -oL tr / awk fflush / sed -u）。**过程中增量对拍**：每出一次 state 摘要即与 G0b-r1 同步骤比 `state_digest`，首次分叉立刻停跑，省 1–2h。

**对拍判读（四分项，G2 先例）**：

```bash
UV_LINK_MODE=copy uv run scripts/training/bench/compare_baseline.py \
  docs/training-doc/v1-grad-baseline-g0b/records/r1 \
  v1-store/bench/2gpu-epoch-bench/<run_name> --tier g3-vs-g0b \
  | tee docs/training-doc/<run_name>/records/compare_vs_g0_r1.txt
```

① `SCALARS steps=1000 keys=5 hex_mismatch_steps=0`；② `STATE_DIGEST rows=12 mismatch=0`；③ `BATCH_DIGEST_CANONICAL rows=14 mismatch=0` + `CANON_CHECK=PASS steps=14`；④ `INDEX_SEQ=PASS n=8072`（前 8,000 前缀一致）。**不作判据**：raw `BATCH_DIGEST mismatch=4 first_bad_step=100 bad_keys=2 (static_image_emb/static_pos_emb)`——V2.4b dtype 统一的已知预期失配，与 G2 逐字吻合即正常，**不是 4 步×2 键则升格为信号**；总行 `DET_CHECK=FAIL` 是已拍板不修的工具聚合缺口。额外必检 `n_keys=12`。

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

## 八、对拍盲区诚实清单（写入 G3 result.md）

1. modulation/expert 集成无基线（中）：G 链只锚 context；modul/expert 分支只能 `git diff` 源码级论证「逐字未动」，不得宣称已验证。
2. examples symbolic 删除无行为对拍（中）：只有语法/import 冒烟；复活需从 git 历史取回。
3. 在线整链（B13 之后到 sample_actions）无端到端 A 侧（中）：执行分支逐字保留 + 本轮不改 sample_actions，源码级论证 + 评审兜底。
4. 建库域 GL 侧不重跑（中）：BUILDER_SPOT 只证本机同架构复算一致；COPY_DIFF 把风险降为「源码同一 + 环境未变」。
5. ONLINE_MEM 用 SigLipTokenizer 桩而非 policy 真实注入路径（低）：注入点一行未改，源码级论证。
6. `has_aux` 改动无 HLO 级直接 diff（低-中）：GRAD_FIXTURE 是最强前置证伪；可选 HLO_DIFF（需重构前预存 A 侧）默认不做。
7. treedef 变化必然重编译（低）：D2-cold 已授权跨编译 bitwise 对拍。

## 九、留档与 commit 纪律

- 每 commit：`git status --short` 核对 → 逐文件 `git add` → 中文 subject（commitV3.8–V3.14）+ 详版 body（AGENTS 11 六要素）。
- G3 >5min：AGENTS 12/17 留档；run_name 起跑前确认（AGENTS 6）；tmux 模板 + Monitor 行缓冲过滤（AGENTS 7）。
- 烟测临时 run（v39-smoke 等）验证完按 AGENTS 6 清理。
- ONLINE_MEM >5min 单独适用 AGENTS 17，产物并入 G3 run 目录 `records/block1/`。
