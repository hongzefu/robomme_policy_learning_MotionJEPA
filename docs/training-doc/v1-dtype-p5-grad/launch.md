# v1-dtype-p5-grad（单步定点梯度 B 侧 + 逐元素对拍）

- **目的**：第二块（训练梯度一致）里最便宜的一档——固定初始 state + 三个固定 batch，
  只算一步前向反向，把梯度逐元素与 P3 的 A 侧对拍。它先于千步 run 跑，作用是「粗差
  先抓出来」，并能定位到具体哪个参数叶子先分叉。
- **起跑 commit**：`a0f76f8`（commitV2.4b，三行修复）。起跑前工作区 clean。
- **A 侧来源**：`v1-store/dtype-unify/v1-dtype-p3-dump-pre-grad/`（P3 于 `f2e7348` 产出）
  + 梯度数组 `/data/hongzefu/v1-baselines/dtype-p5-grad-pre/`。
- **三个定点 batch 的分工**（取自 `_common.build_fixture_batches` 的定点计划，两侧
  batch_id 与样本 index 完全相同）：
  - `mixed1`（1 短 + 7 满长，batch_id 0）——**主判据**，唯一存在 dtype 差异的典型场景；
  - `allshort`（全短样本，batch_id 50）——差异密度最大化；
  - `allfull`（全满长，batch_id 100）——**阴性对照**：两侧交付本就同为 bf16/f32，
    若它不逐位相同，说明改动越界（与 dtype 无关），必须立刻停下排查。
- **口径**：2×RTX 6000 Ada、b8、seed 42、`fsdp_devices=2`、
  `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`（D2 档）；
  梯度调用包在 `sharding.set_mesh(mesh)` 上下文里（模型内部的 activation sharding
  constraint 依赖活动 mesh，缺它 HLO 就与真实训练不同）。
- **命令**：
  ```
  RUN_TAG=v1-dtype-p5-grad \
    FIXTURE_DIR=v1-store/fixtures/dtype-unify-v1 \
    GRAD_ARRAYS_DIR=/data/hongzefu/v1-baselines/dtype-p5-grad-post \
    bash scripts/dtype-unify/run_dtype_grad.sh
  uv run scripts/dtype-unify/compare_dtype_fix.py \
    --grad-a v1-store/dtype-unify/v1-dtype-p3-dump-pre-grad \
    --grad-b v1-store/dtype-unify/v1-dtype-p5-grad-grad \
    --report v1-store/dtype-unify/p5-grad-report.json
  ```
- **判据**：三个 batch 的 32 个梯度叶子逐叶 sha256 全等（bitwise）+ 单步 loss hex 全等；
  输入侧先自证两侧 batch 的 canonical 摘要一致（数值同、只是 dtype 不同）。
  阴性对照失配单独标为越界告警。
