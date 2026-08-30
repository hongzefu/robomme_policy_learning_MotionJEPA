# v1-postclean-g3（G3）result — 四分项全绿，收官 sha256 与 G0b/G1/G2 同值

## 判定行（`records/compare_vs_g0_r1.txt` 逐行核对）

```
SCALARS steps=1000 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
STATE_DIGEST rows=12 mismatch=0
BATCH_DIGEST rows=14 mismatch=4 first_bad_step=100 bad_keys=2 首个: ["['static_image_emb']"]   ← 已知预期失配，不作判据
BATCH_DIGEST_CANONICAL rows=14 mismatch=0
CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072（共同前缀逐个一致, steps≈1000）
DET_CHECK=FAIL …batch_digest_diff=4   ← 总行不具判据资格（raw 聚合缺口，已拍板不修）
```

- **四分项**：① 1000 步五标量 IEEE 浮点位零失配；② 12 摘要步完整 TrainState
  （params+Adam 动量+EMA）逐叶 sha 零失配；③ 14 记录步 canonical batch 零失配；
  ④ 8,072 条抽取顺序前缀逐个一致（两侧 n 同为 8072，补位 2 过）。
- **raw 预期失配**与 G2 记录逐字吻合（mismatch=4 first_bad_step=100 bad_keys=2，
  static_image_emb/static_pos_emb——V2.4b dtype 统一的固有口径差）。
- **补位 1**：scalars_hex.tsv 表头恰六列、mem_enc_norm 在岗；**补位 3**：按四分项
  逐行判读，未采用退出码。额外必检 `batch_digests.jsonl` 首行 n_keys=12、键集无
  recur_*/subgoal。
- **过程中增量 state 对拍**：12/12 摘要步实时 MATCH（监视进程只报异常，全程静默）。
- **中途抽查**：步 0–200 五标量 hex 零失配（201 步）；步 0 五标量与 G0b 逐字相同。

## 收官一行

```
sha256(records/scalars_hex.tsv) = c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757
```

与 G0b r1、G0b r2、G1、G2 四份同值——本次为第五份。**归因唯一化成立**：第 1 层
（D2/D2-cold 四份证据钉死复跑噪声为严格零）排除重跑抖动、第 2 层（BASELINE_ENV=PASS）
排除环境漂移，故 G3 逐位等于 G0 的唯一解释是：**七个 commit（V4.0–V4.6）合起来
没有改变训练语义**。

## 轮数

一轮（计划六节：四份独立证据已钉死管线复跑抖动为零，全分项 PASS 无需第二轮）。
util/步时不作性能结论；性能另跑 `v1-g3-speed`（不在本计划范围，届时另行确认）。

## 对拍盲区诚实清单（计划八节，13 条如实登记；G3 全绿不覆盖以下路径，均标 UNVERIFIED）

1. modulation/expert 集成无基线（中）：G 链只锚 context；modul/expert 分支仅
   源码级论证「逐字未动」（V4.3 的 llm 构图与 lazy_init 参数一字不动），UNVERIFIED。
2. examples symbolic 删除无行为对拍（中）：只有语法/import 冒烟；复活需 git 历史。
3. 在线整链 B13 之后到 sample_actions 无端到端 A 侧（中）：执行分支逐字保留 +
   本轮不改 sample_actions（仅 5→4 元组解包），源码级论证，UNVERIFIED。
4. 建库域 GL 侧不重跑（中）：COPY_DIFF+哨兵证「源码逐字节同一」，本机同架构复算
   一致由 N4 三方对拍兜底（BUILDER_SPOT 未跑，c-1 PASS 时为可选档）；「GL 上重建
   逐位相同」是推论非实测。
5. ONLINE_MEM 用 SigLipTokenizer 桩而非 policy 真实注入路径（低）：注入点一行未改。
6. has_aux 改动无 HLO 级直接 diff（低-中）：N3 GRAD_FIXTURE 三定点 batch × 32
   梯度叶逐位为最强前置证伪；HLO_DIFF 默认不做。
7. treedef 变化必然重编译（低）：D2-cold 已授权跨编译 bitwise 对拍（本轮 G3 独立
   EXP_NAME 现场重编译，四分项照样全绿——第五份实证）。
8. compute_norm_stats 内联读取器无新旧输出对拍（中）：norm_stats.json 是 G 链输入
   常量、全链不执行该脚本；等价性只有八要点源码级论证 + --output-dir 必填防覆盖。
   口径差要到下次重算才显形，UNVERIFIED。
9. preflight 的 provenance 实际未进指纹（中）：R22，本轮禁修（改指纹口径即作废
   G0 基线）；「数据集指纹含 stats+provenance」今天只有前半成立。
10. 比较器三处 fail-open（中）：本轮不改量具，靠 launch.md 判读纪律人工补位——
    人工防线，不是机器防线。
11. --tier 不切换严格度（低）：只是打印标签。
12. run_name 与建库输出路径的破坏性护栏不完整（中）：R20（--force 不做 canonical
    containment）、R21（EXP_NAME/RUN_TAG 不校验 ../），均为本轮明确不处置项。
13. G3 只覆盖被实际执行到的跳（中）：1000 步 / b8 / seed 42 / context 集成 /
    本机 2×Ada。未执行的改动——modul/expert 构图、在线侧、examples/、建库域、
    compute_norm_stats、sample_actions 三分支——bitwise 一律沉默，以上各条
    UNVERIFIED 标注即此边界的显式化，不得被「G3 全绿」的结论吞掉。

## 两块一致性讨论（AGENTS 18）

- **第一块（非训练轻量化测试）**：五族判定行全 PASS——COPY_DIFF（源码逐字节同一，
  机器判定）、IMPORT_ISOLATION（双向 0 泄漏）、GRAD_FIXTURE（单步梯度 32 叶逐位）、
  ONLINE_MEM（三方 POS_TABLE/ENC_LAYER/ASSEMBLY 逐位 + OOB 探针）、SMOKE5×3 轮
  （N1/N2/预 G3，SCALARS/STATE_DIGEST/BATCH/CANON/INDEX 全零失配）。判据全部显式
  （逐位 sha 或 hex，无 allclose）。汇总见 `records/block1/`。
- **第二块（本机训练一致，终局检验）**：即本 run——新链 1000 步 vs 重构前固化基线
  G0b-r1，逐步 loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm 浮点位、12 步
  参数树摘要、14 步输入指纹、8072 条抽取顺序全部位级一致。引用基线经环境指纹
  preflight（BASELINE_ENV=PASS），基线 run_name=v1-grad-baseline-g0b、
  commit=G0b 留档所记、指纹比对结论 PASS。
- 前后链路图（图 A-before/A-after + 辅助图 B/C）见同目录 `chainmaps.md`。
