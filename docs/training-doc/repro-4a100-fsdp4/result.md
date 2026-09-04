# repro-4a100-fsdp4 — 结果

## 结论

**官方 MME-VLA 训练命令在 4 × A100 + `--fsdp-devices=4` 下跑通；模拟 40 GB 显存 + 关闭 P2P 的一档同样跑通，`EXIT_CODE=0`。**
即：**4 × 40 GB 够用**，`--batch-size=64` 不需要下调。

| 档 | GPU | 显存上限 | P2P | 20 步 | `EXIT_CODE` | checkpoint |
|---|---|---|---|---|---|---|
| `repro-4a100-fsdp4-40gsim` | 0-3 | 38 G（`MEM_FRACTION=0.475`） | 关（`NCCL_P2P_DISABLE=1`） | 20/20 | **0** | step 19 ✓ |
| `repro-4a100-fsdp4-80g` | 4-7 | 76 G（`MEM_FRACTION=0.95`） | 开（NVLink） | 20/20 | **0** | step 19 ✓ |

两档 `Step 0` 数值一致（`param_norm=1815.5465` 完全相同，loss 0.0891 / 0.0892），说明 38 G 一档不是靠降级换来的通过。

## 最重要的发现：官方 train.py 设计上就跑两遍，第一遍是热身

`scripts/train.py` 结尾（commit `89efeaab`）：

```python
if __name__ == "__main__":
    main(_config.cli(), tentative_run=True)   # 第一遍：热身，step > tentative_run_step(=10) 即 break
    time.sleep(20)
    main(_config.cli())                       # 第二遍：正式训练
```

热身遍在主循环里的退出点：

```python
tentative_run_step = 10 # on our cluster, we need to run the tentative run for a few steps to warm up the machine.
# Otherwise it would be very slow. I guess it is because of JAX compilation cache.
...
if tentative_run and step > tentative_run_step:
    print("\n\n\n==========Tentative run completed==========\n\n\n")
    break
```

因此**日志里会出现两次完整启动**——两次 `Initialized data loader:`、两次 `Restoring checkpoint from .../pi05_base/params`、
两次 `Step 0` 且数值一模一样，中间隔 `time.sleep(20)`。本轮两档都精确停在 `11.0it/20.0it`（即 `step=11` 时 break），
与代码逻辑吻合。**这是预期行为，不是崩溃、不是重启、不是死循环**；容易被误读成「训练卡住又从头开始了」。

热身的收益在本轮实测里很明显（同一档热身遍 → 正式遍）：

| 档 | 热身遍步时 | 正式遍步时 |
|---|---|---|
| 40gsim | 17.2 s/it（冷编译期峰值） | **6.5 s/it** |
| 80g | 9.7 s/it（冷编译期峰值） | **3.2 s/it** |

## 显存判读（注意：不要把 nvidia-smi 数字当成真实需求）

`records/gpu_all8.csv`（`nvidia-smi -lms 500` 密集采样 7,736 行）各卡显存峰值：

| GPU | 档 | 显存峰值 |
|---|---|---|
| 0-3 | 40gsim | **38.4 G** |
| 4-7 | 80g | **77.6 G** |

这两个数字分别正是 `0.475 × 80 G` 与 `0.95 × 80 G`——**XLA 是预分配模式，nvidia-smi 看到的是预分配池的大小，
不是模型的真实峰值需求**。所以不能读成「训练真的用掉了 38.4 G」。

**「40 G 够」的有效证据是另一条**：在 38 G 池上限内跑完全部 20 步，全程无 `RESOURCE_EXHAUSTED` / OOM，
正常存 checkpoint 并 `EXIT_CODE=0`。真实需求的上界由此被夹在 38 G 以内。

## GPU util（本轮不作为吞吐结论）

按 AGENTS.md 第 16 条口径（均值 / 0% 占比 / 活跃期均值，不用中位数）：

| 档 | util 均值 | 0% 采样占比 | 活跃期 util 均值 |
|---|---|---|---|
| 40gsim（GPU 0-3） | 24.4–25.5% | 68.1–69.0% | 27.8–28.7% |
| 80g（GPU 4-7） | 7.6–8.7% | 84.1–84.6% | 10.4–10.9% |

**这些数字不可作为吞吐基准**，原因有三：只跑 20 步、其中相当比例是 XLA 编译期；两档同场次并行，
各自 `--num-workers=4` 互相争 CPU 与 NVMe IO；采样窗口覆盖了两遍启动之间的 `time.sleep(20)`。
本轮目的是「能否跑起来 / 38 G 够不够」，吞吐需另起单档基准 run 测。

同理，正式遍 6.5 s/it（40gsim）与 3.2 s/it（80g）之间约 2 倍的差距，**主要**指向 `NCCL_P2P_DISABLE=1`
带来的 FSDP 通信代价，但受并行干扰，**不能当作干净的 P2P 开销测量**；若要给对方一个可信的加速判断，
需两档串行各跑一次。

## 环境差异说明

数据集用本机 `4task-motion-400ep/source`（`ButtonUnmask` / `ButtonUnmaskSwap` / `VideoUnmask` / `VideoUnmaskSwap`），
对方用 `PickXtimes`。格式、schema、dataloader 代码路径、batch 形状全同，仅样本内容与任务语义不同，
对本轮两个判据无影响。batch 形状实测：

```
[0].images['base_0_rgb']: (64, 224, 224, 3)@float32
[0].images['left_wrist_0_rgb']: (64, 224, 224, 3)@float32
[0].state: (64, 32)@float32
[0].static_image_emb: (64, 512, 2048)@float32
[0].static_mask: (64, 512)@bool
[0].static_pos_emb: (64, 512, 768)@float32
[0].static_state_emb: (64, 512, 8)@float32
[1]: (64, 20, 32)@float32
====== Using History, Representation Type: perceptual , Integration Type: modulation ======
Perceptual Memory using frame_sampling type
Total Model Size: 3282.45 MB / Trainable 2886.86 MB
```

## 可转发给对方的命令

在 4 × A100-40GB 上验证等效通过（本机以 `MEM_FRACTION=0.475` 于 80 GB 卡上模拟 38 GB 可用显存复现）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_IB_DISABLE=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
OPENPI_DATA_HOME=<你的 openpi 资产根> \
uv run scripts/train.py mme_vla_suite \
  --exp-name=<your_exp_name> \
  --batch-size=64 \
  --num-train-steps=200 \
  --num-workers=4 \
  --fsdp-devices=4 \
  --dataset-path=data/robomme_preprocessed_PickXtimes \
  --model.use-history \
  --model.history-config=perceptual-framesamp-modul.yaml \
  --no-wandb-enabled
```

即**他原来那条命令不用改**。要点三条：

1. **看到日志跑两遍、`Step 0` 出现两次、中间停约 20 秒，是官方 `tentative_run` 热身机制，不是故障**——
   别在第一遍 break 处判定训练失败。
2. **40 G 显存够**，`--batch-size=64` + `--fsdp-devices=4` 无需下调。
3. 若嫌慢：无 P2P 时 FSDP 通信是主要开销之一（本轮同场次观察到约 2 倍差距，非干净测量）；
   `NCCL_IB_DISABLE=1` 保留即可，可另测 `NCCL_P2P_DISABLE` 是否已被驱动自动禁用。

## 产物

- 日志：`v1-store/logs/repro-4a100-fsdp4-40gsim.log`、`...-80g.log`（副本见 `records/`）
- GPU 采样：`records/gpu_all8.csv`（7,736 行）
- checkpoint：`v1-store/train-runs/mme_vla_suite/repro-4a100-fsdp4-{40gsim,80g}/19/`
- 官方代码 worktree：`v1-store/worktrees/official-89efeaab`（detached @ `89efeaab`，保留以便复跑）

## tmux 会话清理

本轮起过 `repro-40gsim`、`repro-80g` 两个会话，均已随脚本结束**自行退出**（`tmux ls` 已无 `repro-` 前缀会话），
无需手动 kill。用户自有会话 `0`、`1`、`claude-private` 全程未动。
