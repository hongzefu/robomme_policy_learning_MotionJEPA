# v1-upstream-eq 结果（C4：上游 main 对拍）——ENTRY_EQ=PASS，双侧命中锚点（第七、八份同值）

- **结论先行**：`ENTRY_EQ=PASS`，五条子判据全过；A/B 两侧 `scalars_hex.tsv` sha256 **均**
  命中锚点 `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`。四象限判读落
  第一格（A✅B✅）：**新 `train.py` 单跑入口与上游 main（`ecf086c`）官方 `__main__` 双跑
  位级等价**；「tentative 预热是否污染正式轨迹」这一从未实测过的怀疑点就此裁决为**零污染**
  （A 侧带 12 步预热 + `--overwrite` 仍逐位命中历史锚点）。
- **份数口径修正**：计划 4.5 注释按当时排序称 A/B 为「第六、七份」；实际时序上 C3
  （`v1-singlerun-g0`）已先落第六份，本轮 A 侧为**第七份**、B 侧为**第八份**同值。
- **起跑 commit**：`f641f40`（launch.md 先行提交 `8e42fbf`）。2026-08-30 两侧串行完成：
  A 侧 loop（正式段 step0→999）55.48 min、B 侧 54.62 min，均冷编译（P1 后已删
  `~/.cache/jax_entry-eq-a` 保证 A 侧正式跑仍为冷编译口径）；两侧 harness
  `ENTRY_RUN=OK`、`EXIT_CODE=0`。
- **AGENTS 18 第二块（本机训练梯度一致）挂本 run B 侧**（计划 4.6 适配）：B 侧是新
  `__main__` 的首次 1000 步真实训练，逐步 loss/梯度范数标量与状态摘要对 A 侧及历史锚点
  双重一致。基线引用：锚点 `c799a0b2…` 源自 G0b/G2 链（G0b commit `570287f`），环境同一性
  由 `uv.lock` 零 diff + P3 norm_stats 同源 sha256 `709f22ff…` 背书。

## 判定行实录（judge_out.txt 原文）

```
A_SIDE_SEGMENTS tentative_rows=12 main_rows=1000        # 留档非判据，与预期逐字吻合
ENTRY_SCALARS steps=1000 keys=5 hex_mismatch=0
ENTRY_STATE_DIGEST rows=10 mismatch=0                   # 首轮实测钉死 rows=10（步 100..900 九次 + 末步 999；step 0 不摘要）
CFG_SPOTLIGHT field=ema_decay SAME
CFG_SPOTLIGHT field=optimizer SAME
CFG_SPOTLIGHT field=lr_schedule SAME
ENTRY_RESOLVED_CFG mismatch=0 whitelist=4               # 白名单：exp_name/dataset_path/checkpoint_base_dir/overwrite
ENTRY_PROVENANCE=PASS                                   # A 全在 worktree、B 全在主仓库、两根不同
A_SCALARS_SHA256 c799a0b2… anchor_hit=True
B_SCALARS_SHA256 c799a0b2… anchor_hit=True
ENTRY_EQ=PASS
```

状态摘要为 strict 口径（key+dtype+shape+bytes 逐叶）+ `treedef_sha`/`keyset_sha` 三域互比，
177 叶/次，全程不落真权重。

## preflight 实录

- **P1**（A 侧 2 步 harness 冒烟）：PASS——`SEGMENTS tentative_rows=2 main_rows=2`、
  `ENTRY_RUN=OK`、`EXIT_CODE=0`；tentative 段与正式段 2 步标量逐位同值（rng 从 config
  纯函数重派生的首个直接实证）。临时目录与 jax 缓存跑后即删。
- **P2**（静态键位）：PASS——`ecf086c` 版 `config.py` 15 字段 +
  `weight_loaders.py::params_path` + `history_pi0.py::use_history/history_config` 逐一在位。
- **P3**（norm_stats 同源）：PASS——两侧同指 `v1-store/train-assets`，
  `mme_vla_suite/robomme/norm_stats.json` sha256 = `709f22ff…` 命中 G0b 指纹。

## 过程旁注

- A 侧 tentative 段（12 行）单独留档 `records/a/metrics_tentative.jsonl`，不判；其步 0/1
  标量与正式段步 0/1 逐位同值（`--overwrite` rmtree 重建 + rng 重派生的完整链路实证）。
- 两侧全程标量与 C3/G0b 逐字同值（Monitor 实时逐百步核对）。
- argv 差异五处照登记执行（`run_entry_equiv.sh::common_args` 机器保证其余逐字符同）。

## 产物清单与清理

`records/`：`a/`、`b/` 两目录逐字节拷贝（metrics_all/metrics/metrics_tentative(A)/
state_digests/resolved_config/provenance/harness_meta/scalars_hex.tsv）+ `judge_out.txt`。
日志留 `v1-store/logs/v1-upstream-eq-{a,b}.log` 与 `entryeq-p1.log`（不进 git）。

PASS 后清理（计划第五节）：worktree（`git worktree remove --force` + `prune`）、
`v1-store/entryeq/` 整目录（含 ckpt-a/ckpt-b/records 原件）、`~/.cache/jax_entry-eq-{a,b}`。

## v5.0 全链路收官状态

C2（代码）→ C2.5（RECORDER_SMOKE=PASS cases=4）→ C3（G0_EQ=PASS，第六份）→
C4（ENTRY_EQ=PASS，第七、八份）四步全绿，v5.0 训练入口重构的等价性主张全部由机器判定行
裁决成立。剩余：sbatch 动态行为待首次集群提交时验证（按 greatlakes.md 走用户放行，不在
本计划范围）。
