# v1-det-d1-r1 结果

- **训练**：100 步全部完成，loss 有限；100 步墙钟 3.9 min（含摘要停顿）
- **稳态步时（仅留档参考，禁作性能结论）**：中位 1.020 s/step（n=48，warmup 50 步与摘要步已剔除）
- **TrainState 摘要**：steps [0, 50, 99]，177 叶子/次，单次耗时中位 89.8 s
- **编译缓存事件**：{'tasks_using_cache': 1, 'compile_requests_use_cache': 15, 'cache_misses': 2}
- **本档两轮对拍判定（`compare_baseline.py`，权威判定行）**：

  ```
  DET_CHECK=FAIL tier=d1 steps=100 scalar_hex_diff=2 state_digest_diff=2 batch_digest_diff=0
  ```

- **结论**：接近但未达 bitwise：100 步中仅 2 步（step 7 / 43）标量 hex 不同，且都只在 llm_grad_norm、差异为 ULP 级（如 118.03035736 vs 118.03034973）；loss 全程逐位一致。末段 TrainState 分歧恰好只有 embedder input_embedding 一族 4 叶子（params/mu/nu/ema）——定位为 embedding 反向 scatter-add 的 atomics 非确定累加，同一 executable 运行期仍存在，须 deterministic_ops 消除。
