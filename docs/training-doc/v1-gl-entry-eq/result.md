# v1-gl-entry-eq 结果（D2：集群 A40 上游对拍）——ENTRY_EQ=PASS（豁免判，用户裁决）

- **结论先行**：新入口 `scripts/training/train.py` 与上游 main `ecf086c` 官方入口在
  **4×A40 + 确定性 XLA 档**下位级等价。A/B 两侧 1000 步五标量逐位一致
  （`hex_mismatch=0`）、状态摘要 10 行三域全同（`mismatch=0`）、resolved config 白名单外
  零差异；两侧 `scalars_hex.tsv` sha256 **同值**：
  **A40 锚点 `140658da6c0225f6cdae41da6955e381afde76cee2ba0828b2e3fd7ed68a314d`**
  （与本机锚点 `c799a0b2…` 不同属预期——硬件/编译路径不同，本锚点为 A40 确定性档首个）。
- **判定口径（豁免判）**：job 59357942 内 judge 因 `ENTRY_PROVENANCE` 的 B 侧 HEAD 断言
  FAIL（唯一失败项）判 `ENTRY_EQ=FAIL`、`EXIT_CODE=1`；经用户 2026-08-31 裁决**接受豁免
  判 PASS**。豁免依据双重证据：
  1. `git diff 43680ee..9c81bec` 实证漂移仅 `docs/training-doc/v1-gl-entry-eq/records/
     submit.md` 26 行插入，**零代码改动**（9c81bec 是本人在 job 过闸后提交的三提实录，
     B 段收尾 provenance 记下的 cwd HEAD 因此与提交时锚不符——硬化后的 judge 如实抓住
     带外提交，行为正确）；
  2. B 侧实际运行代码由 provenance 逐模块 `__file__`+sha256 锁定未变；本机以修正期望
     `--expect-head-b 9c81bec` 重跑 judge（judge 是 records 的纯函数，C4 同款做法）
     **八条判定行全绿、RC=0**，输出固化 `records/judge_out_waiver.txt`（job 内 FAIL 版
     原文见 `records/judge_out_injob.txt`）。
- **D3 放行闸**：视为通过（用户裁决）。`WANDB_PROBE=PASS`（计算节点出网，真 wandb 代码，
  probe run 即删）。

## 判定行（豁免复判版，records 固化于本目录）

```
A_SIDE_SEGMENTS tentative_rows=12 main_rows=1000
B_SIDE_SEGMENTS tentative_rows=0 main_rows=1000
ENTRY_SCALARS steps=1000 keys=5 hex_mismatch=0
ENTRY_STATE_DIGEST rows=10 mismatch=0
ENTRY_RESOLVED_CFG mismatch=0 whitelist=4
ENTRY_PROVENANCE=PASS
A_SCALARS_SHA256 140658da… anchor=SKIPPED   # A40 无本机锚点，两侧互比同值即判据
B_SCALARS_SHA256 140658da… anchor=SKIPPED
ENTRY_EQ=PASS
```

## 运行包络实测（job 59357942，第三提）

- A 段（legacy 库 + tentative 12 步）elapsed **4:19:55**、稳态 12–13 s/步；B 段（packed 库）
  elapsed **3:52:41**；全程 **8:25:24** —— 计划原 5h walltime 严重不足，12h 是对的。
- `--mem=240G` 下 MaxRSS=251657992K ≈ **240.0 GB 仍贴死申请上限**（NFS 页缓存填满额度的
  已知行为），但匿名内存裕度足够、**零 oom_kill**，A/B 共 20 个摘要点全过。
- 确定性 XLA 档（`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`）A40 首测
  成立：两侧逐位同值即为直接证据。

## 三次提交与两个真 FAIL 的过程（细节见 records/submit.md）

1. 59349729（128G/5h）：A/B 双侧均在第 100 步首次摘要点被 cgroup OOM 杀（Slurm
   `oom_kill` 记账 + MaxRSS 贴死 128G），证死因为摘要器逐叶 device_get+tobytes 尖峰叠加
   页缓存，与数据库无关；残档存 `v1-store/entryeq/records-oom-59349729/`（清理时删除）。
2. 59357885（240G/12h）：撞上一提残档，被 harness 防覆盖闸 `FileExistsError` 拦下（12 秒）。
3. 59357942（240G/12h，干净目录）：本 result 所据 run，科学判据全绿。

## 过程教训（已入档）

- **锚定 HEAD 的集群 job 从提交到结束期间冻结一切 commit**（不只到过闸为止）——
  provenance 是收尾时间点的快照；本次 FAIL 即由此触发。
- 对拍重提前必须清 `v1-store/entryeq/records/` 与 `ckpt-a/b` 残档。

## 清理（PASS 后执行）

worktree（`git worktree remove --force` + prune）、`v1-store/entryeq/` 全目录、
`$STORE/cache/jax/gl-entry-eq-{a,b}`、集群侧 `~/.cache/jax_gl-entry-eq-{a,b}` 软链。
