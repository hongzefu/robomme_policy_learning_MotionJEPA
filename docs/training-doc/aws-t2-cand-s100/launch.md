# aws-t2-cand-s100（T2 candidate：HEAD 关闭态，与 aws-t2-ref-s100 同机对拍）launch

> 环境 B（AWS 单机 8×A100-SXM4-80GB，仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，介质 AWS 本地 NVMe RAID `/dev/md0`），2026-09-04；库 `v1-store/datasets/4task-motion-40ep`（环境 B 复刻，见 `docs/dataset-build-doc/4task-motion-40ep-aws/`）。
> 用户要求测试类训练 ≤100 步；本轮全部 run 统一 **100 步 × b8 × 2 卡 / fsdp 2 / seed 42 / WORKERS 4 / 确定性档 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` / `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`**，
> 摘要步 {0,25,50,75,99}（`SAVE_INTERVAL=25 + EXTRA_DIGEST_STEPS=99`），输入摘要步 {0,1,2,25,50,75,99}（7 × 8 = 56 样本），800 样本 < 11,530 单 epoch。
> 8 卡当 4 组并行：T2 ref GPU0,1 / T2 cand GPU2,3 / T3 closed GPU4,5 / T3 open GPU6,7。本机数字与环境 A（Ada / turbo）不得混比，且确定性档 run 不作性能结论。

- **目的**：HEAD `8093ebda23ec566533067e319bab506baaf80de5`（commitV6.12）关闭态 YAML `perceptual-framesamp-context.yaml`，与 S2_BASE 旧码 reference（`aws-t2-ref-s100`）逐位对拍，
  证明 motion 接线（S2）+ 环境 B 适配（commitV6.12）在关闭态**不改任何数**。
- **argv**：与 ref **逐项相同**，只换 `--exp-name aws-t2-cand-s100`、`--checkpoint-base-dir …/train-runs/aws-t2-cand-s100`；在主树直跑（不经 PYTHONPATH），脚本 `t2-run.sh cand`。
- **GPU**：CUDA 2,3（与 ref 的 0,1 并行；见 result.md「gate 卡号指纹」）。
- **gate**：`g0_gate.py --profile t2 --reference-manifest <ref>/t2_reference_manifest.json --run-dir-b <cand records> --log-b v1-store/logs/aws-t2-cand-s100.log --steps 100 --batch-size 8 --env-out <ref check 输出>`，
  唯一成功行 `T2_EQ=PASS steps=100 batch=8 record_steps=[0,25,50,75,99] digest_steps=[0,1,2,25,50,75,99]`。
