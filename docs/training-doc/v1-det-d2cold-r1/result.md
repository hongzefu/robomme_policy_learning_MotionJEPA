# v1-det-d2cold-r1 结果

- **训练**：100 步全部完成，loss 有限；100 步墙钟 5.1 min（含摘要停顿）
- **稳态步时（仅留档参考，禁作性能结论）**：中位 1.899 s/step（n=48，warmup 50 步与摘要步已剔除）
- **TrainState 摘要**：steps [0, 50, 99]，177 叶子/次，单次耗时中位 90.3 s
- **编译缓存事件**：{'tasks_using_cache': 1, 'compile_requests_use_cache': 15, 'cache_misses': 2}
- **本档两轮对拍判定（`compare_baseline.py`，权威判定行）**：

  ```
  DET_CHECK=PASS tier=d2cold steps=100 scalar_hex_diff=0 state_digest_diff=0 batch_digest_diff=0
  ```

- **结论**：两轮各自独立冷编译（各 cache_misses=2、零命中），100 步仍完全逐位一致——「编译两次得到同样行为」成立。交叉对拍 v1-det-d2-r1 vs v1-det-d2cold-r1（同 flags、不同缓存目录）亦 PASS，四轮确定性 run 彼此逐位一致。**D2-cold PASS ⇒ 三支处置走第一支：G0 固化后可跨期充当 bitwise 判据一侧。**
