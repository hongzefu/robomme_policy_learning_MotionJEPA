# 4task-gl-framesamp 全量打包构建留档（S4 结果）

对应 `v2-framesamp-restructure-plan.md` 阶段 2 S4；launch 预提交见同目录
[`launch.md`](launch.md)（探针 30 抽样原始输出亦在该文件）。**判定：**

```
PACK_DONE=1
VERIFY_PACK=PASS scanned=483291 mismatches=0
EXIT_CODE=0
```

## 可复现口径

- **起跑 HEAD**：`3b52d211fc4f1aef7e82a92fd398c0153dd6ab6f`（clean HEAD；launch.md
  预提交 commit 本身。meta `packer.git_commit` 同值实证；launch.md 原文误写前一
  commit `6ee7494`，已在该文件内更正注记，两 commit 代码零差异）
- **命令**：`tmux new-session -d -s pack-framesamp "bash scripts/data-pack-framesamp/run_pack.sh"`
  （默认参数：`--reader decode`、`--procs 16`；pack → verify --resume 串联）
- **源库**：`v1-store/datasets/4task-gl/`（只读未动）；**清单** sha256
  `20da0dfe…f37758a3`（meta `manifest_sha256` 同值）
- **输出**：`v1-store/datasets/4task-gl-framesamp/`；`build_uuid=d684fc2f22294024a1159b7898dac2ff`，
  host `sled-vail`，python 3.11.14 / numpy 1.26.4 / ml_dtypes 0.4.1
- **日志**：`v1-store/logs/pack-framesamp.log`（v1-store 不进 git，关键行已内联本文件）

## 实测结果

| 项 | 实测 | 计划预估 |
|---|---|---|
| pack 总耗时（含 state 表全量 decode + 32 part 并行） | **2,941 s ≈ 49 min**（19:41:16→20:30:18） | 40–80 min ✓ |
| verify 总耗时（全量三键对拍 + row_digests） | **1,061 s ≈ 17.7 min**（20:30:20→20:48:01） | 20–40 min（略优）✓ |
| 单 part 耗时（n=32） | min 272 s / med 548 s / max 647 s | — |
| 小表 | pos 28,803,072 B（=586×49,152）、state 15,465,312 B（=483,291×32） | 28.8 / 15.5 MB ✓ |
| 库体积（表观） | **30 GiB = 31.67 GB 十进制**（`du --apparent-size`；裸 `du` 36 GiB 为 turbo 块分配开销） | 31.7 GB ✓ |
| part 数 / 残留 | 32 个（part_000..031），`.tmp` 残留 0 | 32 ✓ |
| 末 part | episodes[1573..1599]、9,471 行、620,691,456 B | 逐字一致 ✓ |

## 校验结论

- **写侧（100% 覆盖，pack 路径内）**：① 逐帧 pos memcmp（钉 t 不钉 g）② state 与
  小表同源自证 ③ episode slab read-after-write——三者任一失配即中断，本次零触发；
  part sha256 取读回字节计算后原子 `os.replace` 落盘。
- **全量 verify（g 级零遗漏唯一凭据）**：重新完整 decode 全部 483,291 个源 npy，
  三键各经真实读 API（`read_image_rows` / `pos_rows` / `state_rows`）逐行 memcmp，
  `scanned=483291 mismatches=0`；逐行 blake2b-128 摘要 `meta/row_digests.blake2b.bin`
  （7,732,656 B = 483,291×16，文件 sha256 `d61410fe…371cbb61`）。
- **meta**：`status="verified"`（阶段 2 原子回填），`manifest_scope="full"`，
  `num_rows=483291 / num_exec_samples=395289 / num_pos_rows=586`；pack.lock 已随
  回填释放，读侧分派可消费。

## 当前状态与下一步

打包库交付完成（阶段 2 收官）。下一步阶段 3 S5：不训练轻量对拍（约 8,200 定点
样本 + 200 真实 batch，判据 `COMPARE_BATCH=PASS` + G6b），随后 S6 G2 训练对拍。
