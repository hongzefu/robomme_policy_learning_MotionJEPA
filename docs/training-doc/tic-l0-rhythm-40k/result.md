# tic-l0-rhythm-40k —— 结果

> 起跑 commit `9b3b95f`（clean HEAD；工具版本 commitV7.1 `a8cfa17`）。2026-09-07 21:05:25 → 21:09:47（4 min 22 s），环境 B，CPU（`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`），tmux `tic-l0rhythm`（结束自行退出）。
> 原始日志 `records/l0-rhythm-m1.log`，明细 `records/l0.json`、`records/rhythm.json`。**三段全部 PASS，`EXIT_CODE=0`。**

## 一、判定行原文

### A1 第 0 关：配置与库同源（`scripts/training/g0/check_config_provenance.py`）

```
NORM_STATS_SAME=PASS sha=41ff90e4bfacc158 train_src=…/v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json infer_src=…/awsprod40k-b128-motion/39999/assets/robomme/norm_stats.json keys=[actions,state] arrays=8
LIB_PROVENANCE_MATCH=PASS framesamp_manifest_sha256=92fa17e97fba9434 motion_manifest_sha256=92fa17e97fba9434 framesamp_store_meta_sha256=dffdd47b09aad281 motion_index_sha256=74185921690cd26c motion_store_meta_sha256=e19fb5fdb45829d8 motion_table_sha256=6e70604da518c156 all_equal=1
MOTION_STORE_PATH=PASS snapshot=v1-store/datasets/4task-motion-40ep/motion provenance=…/4task-motion-400ep/motion effective=…/4task-motion-400ep/motion override=MMEVLA_MOTION_STORE guard=raise infer_reads_store=0 same_manifest=1 snapshot_default=raise
CKPT_PARAM_TREE=PASS missing=0 extra=0 leaves=59 model_leaves=59 ckpt_has_motion=1
CKPT_DTYPE_PROFILE f32_leaves=36 bf16_leaves=23 img_all_bf16=1 trainable_all_f32=1 by_dtype={'bfloat16': 23, 'float32': 36} img_dtypes=['bfloat16']
TIC_L0=PASS
```

### A2 评估真节奏 CPU 复刻（`scripts/training/tests/eval_rhythm_gates.py`）

```
RHYTHM_EQ=PASS episodes=5 points=120 tau_mismatches=0 es_mismatches=0 k_mismatches=0 order_sha_mismatches=0 driver=examples/robomme/utils.py:EpisodeState
EVAL_TERMINATION=PASS max_steps=1300 infer_calls=82 first_batch_included=1 tau_max=es+1296 last_partial_batch_dropped=1 residual_frames=4 termination=count_guard env_done_cases=5
TAU_LONG=PASS cases=2 es=0 tau=1296 k=80 raise=0 order_legal=1 | es=216 tau=1512 k=92 raise=0 order_legal=1 budget=96 headroom_min=4
ES_BOUNDARY=PASS scanned=[200,320) first_raise_es=289 predicted=289 k_at_288=96 k_at_289=97 real_es_values=[0,66,114,168,216] real_max_k=92 margin_windows=4
TIC_RHYTHM=PASS
```

### A3 A19 参数化后在 400 ep 库重跑 M1（`scripts/training/tests/motion_gates_model.py --gate m1`）

```
[m1 helper] checked=1863 mismatches=0
[m1 real] samples=101066 mismatches=0 有效数分布 {"n": 101066, "k_median": 9.0, "k_mean": 10.314339144717314, "k_max": 34, "k_p25": 5.0, "k_p75": 15.0, "k_p90": 20.0, "k_p95": 23.0, "k_p99": 27.0, "zero_frac": 0.06332495596936655, "fill_rate": 0.10744103275747202}
A19_VALID_DIST=PASS source=manifest samples=101066 median=9.0 mean=10.31 max=34 p25/p75/p90/p95/p99=5.0/15.0/20.0/23.0/27.0 zero_frac=0.0633 fill_rate=0.107 expect_mismatches=0 measured_from=dataset_motion_mask
MOTION_DELIVERY=PASS samples=101066 mismatches=0 helper_checked=1863
```

## 二、结论

1. **配置与库同源成立**：训练配置 `mme_vla_suite_b128` 实际加载的 norm stats 与 checkpoint 内 assets 8 个数组逐位相同；当前 400 ep 库六个指纹与生产 run `motion_provenance.json` 逐个相等——本轮全部对拍用的库就是训练时那张表；checkpoint 参数树 59 叶与模型定义零缺零多。
2. **快照记错的 motion 库路径不会静默用错库**：`history_config.resolved.yaml` 记 `store_path` 为 40 ep 路径，但不设 `MMEVLA_MOTION_STORE` 覆盖时 `check_same_source` 直接 raise（`snapshot_default=raise`），设覆盖时 effective 即 400 ep；`src/mme_vla_suite/policies/` 无任何 `MotionStore` 引用（推理不读表）。这是记录瑕疵，不是数据风险，本轮不修（计划八节 3）。
3. **评估节奏实测坐实计划推算**：5 集 120 个决策时刻三方一致（真实 `eval.py` 控制流 / `motion_gates_online._drive` / 脚本独立清单）；`max_steps=1300` 兜底下每集 82 次推理、最晚决策时刻 `es+1296`、末尾 4 帧永不上送；最长 demo 段 es=216 时 motion 窗 92、预算 96 余 4；**es ≥ 289 起在线必 raise**（k=97 > 96），评估会把该集静默记成 error（失败链见脚本 docstring 与 `docs/train-infer-consistency.md`）。
4. **`A19_VALID_DIST` FAIL 正式闭合**：期望改为按当前库 `meta/episode_manifest.json` 用 oracle 公式独立重算、实测改为 `FrameSampDataset[idx]["motion_mask"].sum()` 真实交付后，400 ep 库 101,066 个 exec 样本逐样本全等（`expect_mismatches=0`），分布中位 9 / 均值 10.31 / 最大 34 / 零起点 6.33%；`MOTION_DELIVERY` 四键逐位 `mismatches=0`。此前 `env-b-aws-replication.md` 记录的 `MOTION_DELIVERY=FAIL` 只由写死的 40 ep 期望数字触发（那次逐样本 `mismatches=0` 本就成立）。

## 三、异常处置

无。三段均一次通过，无重跑；tmux 会话 `tic-l0rhythm` 随命令结束自行退出，未执行任何 kill。

## 四、观察项

- `CKPT_DTYPE_PROFILE`：checkpoint 盘上 f32 36 叶 / bf16 23 叶，`img` 子树全 bf16、可训练叶全 f32；生产 `create_trained_policy` 以 bf16 加载全部叶，该差异的动作影响见 `tic-obs-model-40k` 的 `ACT_CKPT_DTYPE`。
- `_drive` 与 `eval.py` 在环境步数 C ≡ 0 (mod 16) 时会差一个决策点（本轮五集 C%16 = 10/12/11/4/1，未触发）；`compare_train_infer_obs.py` 已改用与 `eval.py` 同一公式生成时刻清单。
