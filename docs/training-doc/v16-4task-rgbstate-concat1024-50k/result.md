# 1114 4task RGB+state VLA：modulation concat 1×1024 50k 结果（context 段被用户叫停）

## ① 一句话结论

modulation concat 1×1024（4task rgbstate encoder ep144，stride 32，预算 96）step 50000 本机 692 局 seed 7：**324/692 = 46.82%**，比同库同口径的 Wan 固定池化 768（stride 16、预算 160）43.06% 高 3.76 pp；比 GL 上旧 v7 encoder（1×768）modulation 的 54.62% 低 7.80 pp（跨环境，差距 46/54 局来自 RouteStick）。context 段 11-15 美东 20:48 按用户指令停止（约 step 100，无 ckpt）。

```
EVAL692=PASS episodes=692 success=324 rate=0.4682 preseeded_errors=9 new_errors=0 retried=0 expect_motion_stats=1
MOTION_MERGED episodes=683 k_max=93 downsampled_episodes=0
```

## ⑦ 训练过程

- TRAIN_HEAD b5bdccf（commitV15.27Beta），tmux v16-chain，11-15 美东 05:18 起跑、约 18:47 训满；实测 0.965 s/步（8×A100，B128，FSDP8）。
- loss：step 100 0.0564 → 10000 0.0043 → 30000 0.0024 → 50000 0.0019；motion_out_var 0.42 → 0.98；全程无 nan/报错。
- `TRAIN_FINAL_RECORD=PASS run=v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k steps=50001 tail_steps=0 ema_leaves=65`
- `CHAIN_RUN=PASS step=50000 ckpts=10`；`INC_END done=10`；`STAGE=PASS steps=10 objects=245 bytes=119142803734`
- `CKPT_DELIVERY_READY=PASS step=50000 root_complete=1 sha256_mismatch=0 strict_restore=1 online_infer=1`
- `HF_VERIFY=PASS bucket=HongzeFu/robomme-vla-4task-m8x8-modul-s2rgbstate-concat1x1024-50k-v1 scope=ckpt objects=245 bytes=119142803734 mismatches=0`；`CKPT_UPLOAD=PASS steps=10`
- `EVAL_HF_VERIFY=PASS ... scope=eval evals=1 objects=1275 bytes=1143421630 mismatches=0`
- stage 删除：S5 `stage-delete` 因交付判定行整行比较缺陷误判 SKIP（不会误删）；主会话逐条核三条件后按确切路径手工删除，`STAGE_DELETE=PASS manual=1`，/scratch 可用 772G→878G。缺陷待修（只读改为前缀匹配）。

## ⑧ 评估（分任务、分难度；对照两条同库 modulation）

| 组 | 旧 v7 enc 1×768（GL A40，54.62%） | Wan 池化 768 s16（本机，43.06%） | 本次 rgbstate 1×1024（本机，46.82%） |
|---|---|---|---|
| BinFill easy | 44/49 | 44/49 | 42/49 |
| BinFill medium | 44/49 | 30/49 | 41/49 |
| BinFill hard | 23/44 | 10/44 | 20/44 |
| RouteStick easy | 50/50 | 50/50 | 49/50 |
| RouteStick medium | 47/50 | 48/50 | 42/50 |
| RouteStick hard | 41/50 | 27/50 | 28/50 |
| RouteStick xhard | 29/50 | 13/50 | 2/50 |
| VideoRepick easy | 22/50 | 12/50 | 27/50 |
| VideoRepick medium | 19/50 | 16/50 | 18/50 |
| VideoRepick xhard | 23/50 | 10/50 | 13/50 |
| VideoUnmaskSwap easy | 8/50 | 14/50 | 12/50 |
| VideoUnmaskSwap medium | 17/50 | 15/50 | 15/50 |
| VideoUnmaskSwap hard | 7/50 | 4/50 | 5/50 |
| VideoUnmaskSwap xhard | 4/50 | 5/50 | 10/50 |
| **合计** | **378/692** | **298/692** | **324/692** |

按难度（本次 vs 旧 v7）：easy 65.3% vs 62.3%，medium 58.3% vs 63.8%，hard 36.8% vs 49.3%，xhard 16.7% vs 37.3%。旧 v7 一列出自 docs/training-doc/binfilldemo-nomotion50k-gl/result.md（b9d1908），GL A40＋历史前缀，跨环境，只作参考。

R 阶段 heldout 重建（只记录）：新 rgbstate ep144 PSNR 26.01 / latent 0.0105，旧 v7 ep72 PSNR 28.60 / latent 0.0021（docs 在 MJ 仓 v12-3decoder 8f1b4e2）。

## ⑨ 用户决策

- 11-15 美东：「Modulation 版本大约什么时候训完可以出结果？」「分任务、分难度说」。
- 11-15 美东 20:48：「停止你现在的 Context模式训练」→ kill v16-chain、v16-up-c；C 段约 step 100 停，无 ckpt；ctx bucket 仅根快照 19 文件、run 目录 51 MB 保留待定。
- 11-15：「我要的你就是比较这两个，我并不需要你重新评估」（与旧 v7 encoder 54.62% 对比，见 ⑧）。

## ⑩ 计划外事件

见 launch.md 第 1–8 条（FIX1/FIX2/FIX3、三次 Beta、短测调用修正）；另：11-15 美东约 22:04 用户让其他会话把主检出切到 v17 分支，v16-up-eval-m 当时在跑、不受影响（EVAL_HF_VERIFY=PASS）；本结果在独立 v16 worktree 提交。

## ⑪ 下一步

context 段是否重训待用户定；S5 stage-delete 前缀匹配修复待做；seed 42／0 两次评估「剩下两次之后再跑」。
