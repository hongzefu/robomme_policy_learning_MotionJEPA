# v1-det-d0-r2 结果

- **训练**：100 步全部完成，loss 有限；100 步墙钟 3.9 min（含摘要停顿）
- **稳态步时（仅留档参考，禁作性能结论）**：中位 1.039 s/step（n=48，warmup 50 步与摘要步已剔除）
- **TrainState 摘要**：steps [0, 50, 99]，177 叶子/次，单次耗时中位 89.9 s
- **编译缓存事件**：{'tasks_using_cache': 1, 'compile_requests_use_cache': 15, 'cache_misses': 2}
- **本档两轮对拍判定（`compare_baseline.py`，权威判定行）**：

  ```
  DET_CHECK=FAIL tier=d0 steps=100 scalar_hex_diff=100 state_digest_diff=2 batch_digest_diff=0
  ```

- **结论**：两轮从步 0 起逐步标量 hex 全部不同（100/100 步）；rel 噪声底：loss median 2.721e-03 / p95 1.149e-02 / max 4.624e-02，grad_norm median 2.439e-02 / max 5.399e-01，llm_grad_norm max 5.358e-01，mem_enc_norm max 5.538e-01，param_norm max 6.770e-08（EMA 平滑所致极小）。输入摘要 5/5 逐位一致（分歧全在计算侧：独立重编译 + 默认 autotune 的 kernel 选择差异）；步 0 初始 TrainState 一致，步 50/99 摘要分歧 124 叶子。
