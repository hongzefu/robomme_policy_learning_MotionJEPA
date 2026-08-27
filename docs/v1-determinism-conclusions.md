# v1 训练链路确定性核心结论（P2 四档实验定档，2026-08-26）

本文件是 `v1-gradient-baseline.md` P2（确定性预备实验）的**核心一致性结论独立留档**。
逐轮原始产物与判定行见 `docs/training-doc/v1-det-*/`；符号、档位定义与三支处置的权威
表述在 `v1-gradient-baseline.md`（P2 节），本文不重复其规约、只陈述实测结论。

## 一、实验口径

- commit `d9e509e41a1665c15faff6ef62f2fef6ac813813`（V2.1）clean HEAD 起跑；本机
  2×RTX 6000 Ada、batch 8、seed 42、`fsdp_devices=2`、`num_workers=4`、100 步、
  SAVE_INTERVAL=50（TrainState 摘要 @ 步 0/50/99，177 叶子含 params/opt_state/EMA/step）、
  输入摘要 @ 步 0/1/2/50/99；数据集 `v1-store/datasets/4task-gl`。
- 判定工具 `scripts/smoke-local/compare_baseline.py`；每档判据：两轮逐步五标量
  hex diff + 全部 `state_digest` diff + 全部 `batch_digest` diff 为空。

## 二、四档结果总表

| 档 | 设置 | 判定 | 分歧概貌 |
|---|---|---|---|
| D0 | 每轮删缓存、无 XLA flags（字面现状） | **FAIL**（预期内） | 步 0 起 100/100 步标量全分歧；步 50 起 TrainState 124 叶子分歧 |
| D1 | 两轮共用编译缓存、无 flags | **FAIL**（仅差 ULP） | 100 步中 2 步（step 7/43）`llm_grad_norm` 差一个最低位；TrainState 分歧只有 embedder `input_embedding` 一族 4 叶子（params/mu/nu/ema） |
| D2 | D1 + `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` | **PASS** | 逐位一致，零分歧 |
| D2-cold | 同 D2 flags，两轮各用全新空缓存（强制独立编译） | **PASS** | 各自冷编译（各 cache_misses=2、零命中）仍逐位一致；与 D2 轮交叉对拍亦 PASS |

全部八轮的输入摘要（`batch_digests`）**5/5 逐位一致**：同 seed 下 dataloader 交付内容
完全确定，一切分歧均来自计算侧。

## 三、机制定位

1. **D0 的分歧源是「独立重编译 + 默认 autotune 的 kernel 选择」**：删缓存重跑时 XLA
   per-fusion autotune 可选中不同 kernel/归约实现，从步 0 第一次前向就逐位不同。
   其重跑噪声底（rel = |a−b|/max(|a|,|b|,1e-8)，100 步）：

   | 标量 | median | p95 | max |
   |---|---|---|---|
   | loss | 2.721e-03 | 1.149e-02 | 4.624e-02 |
   | grad_norm | 2.439e-02 | 1.146e-01 | 5.399e-01 |
   | llm_grad_norm | 2.292e-02 | 1.107e-01 | 5.358e-01 |
   | mem_enc_norm | 2.746e-02 | 1.224e-01 | 5.538e-01 |
   | param_norm | 0 | 0 | 6.770e-08 |

   此表即量化判据（基线计划六节）的 **D0 null 上界**：任何跨 HLO 对拍若 rel 与它同
   量级，等价性不可判。也说明 100 步内混沌放大已把 kernel 级微差放大到梯度范数 54%。
2. **D1 的残余分歧源是 embedding 反向 scatter-add 的 atomics 非确定累加**：同一
   executable（r2 编译缓存全命中、零 miss）运行期仍在 `input_embedding` 梯度处产生
   ULP 级差异，并经 Adam 动量累积进 mu/nu/params/ema 四叶子。loss 与其余标量全程
   逐位一致——非确定点狭窄且定位精确。附带实证：完整 TrainState 摘要（含 opt_state）
   比标量灵敏——P1 验收的 3 步 run 中标量全一致而 Adam mu 已分歧。
3. **D2 flags 同时消除以上两源**：`deterministic_ops` 治 atomics，`autotune_level=0`
   治 kernel 选择漂移；D2-cold 进一步证明在此档位下**编译本身是确定的**——两次独立
   冷编译产出行为逐位相同的 executable。

## 四、结论与授权（三支处置走第一支）

1. **D2-cold PASS ⇒ G0 黄金基线可「一次跑定、跨期充当 bitwise 判据一侧」**：未来
   G1/G2/G3 计算图改变、必然现场重编译，也不损失 bitwise 可比性（前提：过
   `check_baseline_env.py` preflight，环境指纹逐项一致）。
2. **正确性族 run 的固定确定性档**：`XLA_FLAGS="--xla_gpu_deterministic_ops=true
   --xla_gpu_autotune_level=0"`——G0 及后续一切正确性 A/B 一律注入；G0 的编译缓存
   目录允许清理（留 sha256 清单作证据）。
3. **性能族（speed 链）不注入上述 flags**：确定性档的 kernel 选择不代表生产性能口径
   （基线计划符号总表）。
4. **D0 两轮产物固化留档**（`docs/training-doc/v1-det-d0-r{1,2}/`），标注「非判据
   基线，只作噪声底与口径对照」；三节表格为其权威摘录。
5. 同 seed 下 dataloader 输入交付逐位确定（八轮 40 条输入摘要零分歧），后续对拍中
   若见 `batch_digest` 分歧即直接指向数据侧改动而非计算噪声。
