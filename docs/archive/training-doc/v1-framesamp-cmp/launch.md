# v1-framesamp-cmp（S5 第一块·不训练对拍）launch 记录

对应 `v2-framesamp-restructure-plan.md` 阶段 3 S5（C.1 index 序列等价 + C.2 样本/batch
内容等价 + G6b）。run_name 2026-08-27 经用户确认。AGENTS 17：>5 min 诊断 run，
起跑前预提交本文件。

## 起跑环境

- **起跑 HEAD**：本 launch.md 的 docs commit 本身（S4 教训：预提交先于起跑，起跑
  HEAD 即含本文件的 commit；结果留档将回填全 sha）
- **机器**：本机；两工具均 `JAX_PLATFORMS=cpu`（不占 GPU）；tmux detached（AGENTS 7）

## 输入

- legacy 侧：`v1-store/datasets/4task-gl`（RoboMMEDataset，旧链路逐字未动）
- packed 侧：`v1-store/datasets/4task-gl-framesamp`（S4 交付，status=verified，
  `VERIFY_PACK=PASS scanned=483291 mismatches=0`）
- 清单 sha256 `20da0dfe…f37758a3`；变换栈与 `create_data_loader` 同构
  （`get_config("mme_vla_suite")` + `v1-store/train-assets` + `OPENPI_DATA_HOME=v1-store/models`）

## 命令

```bash
tmux new-session -d -s s5-cmp "set -o pipefail; cd <REPO_ROOT>; \
  PYTHONUNBUFFERED=1 UV_LINK_MODE=copy bash -c ' \
    uv run python scripts/data-pack-framesamp/dump_index_seq.py \
      --out v1-store/bench/framesamp-cmp/v1-framesamp-cmp/idx-dump && \
    uv run python scripts/data-pack-framesamp/compare_batches.py \
      --out v1-store/bench/framesamp-cmp/v1-framesamp-cmp/cmp \
      --idx-file v1-store/bench/framesamp-cmp/v1-framesamp-cmp/idx-dump/idx_seq_w4.json \
  ' 2>&1 | tee v1-store/logs/v1-framesamp-cmp.log; \
  echo \"EXIT_CODE=\$?\" >> v1-store/logs/v1-framesamp-cmp.log"
```

参数全为工具默认值：dump 三档 w0/w4/w8 × 200 步 × b8 × seed 42；compare 定点集
step∈{0,1,2,29,30,31,32,33} 各 200 + 每 episode 首样本 1,600 + 随机 5,000
（seed 20260827，去重保序）+ 真实序列前 200 batch 过 `_collate_fn`。

## 判据（三行全过才算 S5 PASS）

1. `INDEX_SEQ_EQ=PASS steps=200 batch=8 seed=42 workers=0,4,8`
2. `G6B=PASS len=395289 video_first_steps=ok`
3. `COMPARE_BATCH=PASS samples=… batches=200 mismatches=0`（全键 shape/dtype/位串零容差）

预计 25–45 min。记录目录 `v1-store/bench/framesamp-cmp/v1-framesamp-cmp/`
（idx 序列 json、compare_result.json、exemplar 位型容器）；结果留档 result.md 随
V3.3 收官提交。
