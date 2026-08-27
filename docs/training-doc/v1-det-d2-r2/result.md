# v1-det-d2-r2 结果

- **训练**：100 步全部完成，loss 有限；100 步墙钟 4.8 min（含摘要停顿）
- **稳态步时（仅留档参考，禁作性能结论）**：中位 1.910 s/step（n=48，warmup 50 步与摘要步已剔除）
- **TrainState 摘要**：steps [0, 50, 99]，177 叶子/次，单次耗时中位 90.2 s
- **编译缓存事件**：{'tasks_using_cache': 1, 'compile_requests_use_cache': 15, 'cache_hits': 2}
- **本档两轮对拍判定（`compare_baseline.py`，权威判定行）**：

  ```
  DET_CHECK=PASS tier=d2 steps=100 scalar_hex_diff=0 state_digest_diff=0 batch_digest_diff=0
  ```

- **结论**：两轮 100 步完全逐位一致：逐步五标量 hex、完整 TrainState 摘要（177 叶子 ×3 次）、输入摘要全部零分歧。
