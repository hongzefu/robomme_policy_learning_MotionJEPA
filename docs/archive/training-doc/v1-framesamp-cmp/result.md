# v1-framesamp-cmp（S5 第一块·不训练对拍）result

launch 见同目录 [`launch.md`](launch.md)。**三行判据全过（第一块 PASS）**：

```
INDEX_SEQ_EQ=PASS steps=200 batch=8 seed=42 workers=0,4,8 sha256=98a9c4525e34f88d…
G6B=PASS len=395289 video_first_steps=ok
COMPARE_BATCH=PASS samples=7945 batches=200 mismatches=0
EXIT_CODE=0
```

## 口径与实测

- **起跑 HEAD**：`936a901d55bbc90fa4993959a78257bcc8983cb9`（launch 预提交 commit，
  clean 起跑；run 进行期间仓库有后续 commit——42c7650/cf64ddd/729de4e——均不影响
  已起跑进程的代码快照）
- **C.1 index 序列**：w0/w4/w8 三档各 200 步 × b8 = 1,600 index 逐条相同
  （sha256 `98a9c452…`）；前置复核 legacy `execution_samples` == packed
  `num_exec_samples` == 395,289
- **C.2 定点集**：名义 8,200（step∈{0,1,2,29,30,31,32,33} 各 200 + 每 episode
  首样本 1,600 + seed 20260827 均匀随机 5,000），去重后 **7,945** 个样本全对拍，
  transform 之后全键 shape/dtype/原始位串零容差、**零失配**；对拍层含嵌套
  image/image_mask dict 展平（逐相机键）
- **batch 级**：C.1 真实序列（w4 dump）前 **200 个 batch** 过 `_collate_fn` 对拍，
  零失配——覆盖到进 device_put 前的最后一层
- **耗时**：compare 段 733.9 s（≈12.2 min，单进程 CPU；含 legacy 侧全量散 npy 读）；
  与 S6 `v1-framesamp-g2` 并行执行（用户批准的豁免，见 S6 launch.md），本 run
  判据与负载无关
- **records/**：`judgment_lines.txt`（判定行原文）、`compare_result.json`、
  `idx_seq_w{0,4,8}.json`（全序列 + sha256）、`exemplar_sha256.txt`（位型容器逐键
  sha256 清单——容器本体 2.0 MB 留 `v1-store/bench/framesamp-cmp/v1-framesamp-cmp/
  cmp/exemplar/`，不进 git）

## 结论

**第一块通过**：packed 链路（FrameSampDataset + FrameSampStore + 4task-gl-framesamp
库）与 legacy 链路在 transform 之后、collate 之后交付内容逐位一致；index 序列在
w0/w4/w8 档位下构造性不变得到端到端实测。按放行规则，「IO 重构不改变训练语义」的
宣称仍需 S6 G2 bitwise 通过后方可成立（S6 并行进行中）。
