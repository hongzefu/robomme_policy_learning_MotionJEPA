# v1-grad-baseline-g0b 结果

## 自证判定（r1 vs r2，`compare_baseline.py`）

```
DET_CHECK=PASS tier=g0b-selfcheck steps=1000 scalar_hex_diff=0 state_digest_diff=0 batch_digest_diff=0
CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072（共同前缀逐个一致）
```

- 1000 步五标量 hex、12 次完整 TrainState 摘要（177 叶子）、14 次输入摘要（raw 与 canonical 双口径）、8072 个样本 index **全部逐位一致**。
- 两轮 `scalars_hex.tsv` sha256 相同：`c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`。
- 编译缓存：r1 冷编译（cache_misses=2）→ r2 全命中（cache_hits=2、零 miss），与 P2 的 D2/D2-cold 结论自洽。

## 新旧前缀对拍（r1 vs 旧 `v1-grad-baseline-g0` round1）——P1b 量具改造等价性实证

```
DET_CHECK=PASS tier=g0b-prefix-final steps=300 scalar_hex_diff=0 state_digest_diff=0 batch_digest_diff=0
```

- 前 300 步五标量 hex、共同摘要步 0/100/200/**299**（附加摘要步精确对齐旧末步）的 `state_digest`、共同步 raw `batch_digests` **全部逐位一致**；旧产物无 canonical/index（P1b 前）按预期 WARN/SKIP。
- 结论：**uv runner 收敛、canonical 双口径、`--save-interval 1` 步集机制均未改变计算**；据此按用户裁定删除旧 G0 records（本 commit 内执行），G0b 升任基线链头。

## 运行统计（仅留档参考，禁作性能结论——红线 B7）

| 轮 | 稳态步时中位（剔摘要步±1） | util 非摘要段均值 | 0% 采样占比 | 摘要段 util 均值 | 摘要单次 |
|---|---|---|---|---|---|
| r1 | 1.968 s/step (n=929, p10 1.865 / p90 2.011)，慢步 0 | 94.1% | 1.9% | 17.2%（含 3 次 45.4 GiB 数组落盘，单次最长 472 s） | 中位 88.9 s ×12 |
| r2 | 1.980 s/step (n=929, p10 1.876 / p90 2.049)，慢步 3（均值 3.16 s） | 92.3% | 3.0% | 51.1% | 中位 90.8 s ×12 |

- 采样：`nvidia-smi -lms 500`（原始 csv 在 `records/r{1,2}/util-lms500.csv`，15 s legacy 通道并存）；稳态窗 = 步 50–999。
- epoch 外推为确定性档数字、被摘要停顿污染，禁止引用；生产口径见 `v1-g0-speed-r2`。

## 产物与处置

- 固化产物：`records/r{1,2}/`（metrics / param_checksums / batch_digests / index_sequence / scalars_hex.tsv / env / run_meta / util×2 + `BASELINE_MANIFEST.json`；r1 另含 `state_dump.SHA256SUMS`）。引用前必过 `check_baseline_env.py check`（`BASELINE_ENV=PASS` 硬前置）。
- TrainState 数组（0/299/999 三步 ×45.4 GiB）存本机 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（用户裁定不留 NFS），迁移经逐文件 sha256 核对。
- 编译缓存目录已清理，sha256 清单留证：`jax-cache-sha256-manifest.txt`。
- 旧 `v1-grad-baseline-g0` 的 records 已删除（对拍通过后，用户裁定）；其 launch/result.md 保留并加注被取代。
