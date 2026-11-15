# 1114 4task RGB+state encoder 下游 VLA：modulation／context concat 1×1024 各 50k（launch）

> 计划：policy 仓根目录 `1114-4task-rgbstate-s2concat1024-vla-50k-plan.html`（v3.1，71b4c61）。开工令：用户 11-14 美东「开工」。11-14 晚追加：「尽可能在明天早上八点前不要来问我，尽可能自己做，除非有特别严重的问题。然后如果碰到问题，都交给 Fable 来裁决。」

## ② 版本与代码状态

- policy 仓分支 `v16-4task-rgbstate-s2c1024-50k`；Beta 提交即本文件所在提交（TRAIN_HEAD）。
- 合入链：S2 268dded → S1 3dc626a → S4 8c78b1a → S3 7edcaf1 → S5 dc16c7b（POST_MERGE_REVIEW=PASS）→ 回填 94d44f5（encoder sha、motion_rows=36484、train_max=71、两条 resolved sha）→ FIX1 c659a1b（sidecar 加载前登记 sys.modules）→ 首个 Beta ae2b809（作废）→ FIX2 合并（local-built 根快照）→ V15.26Beta 7a205b3（作废）→ FIX3 合并（链 delivery 环境）。
- stage2 运行项目 `scripts/dataset/wan_stage2_rgbstate`，MJ 依赖 26c68ca（memexport b7cb8a2），独立 venv `v1-store/venvs/wan-stage2-rgbstate`。
- MJ 侧 S6a／S6 与重建档案已合入 v12-3decoder（8f1b4e2，1113 done 597f79f 之后）。
- encoder：`v1-store/external/motionjepa/stage2-4task-rgbstate-144ep-1113-a/checkpoint_epoch_144.pt`（硬链接，sha256 ece3465b57f9d1ddea130a631aff1d1006b2c8ba5c68a0e5e92b8215ae80b55c，等于 1113 `CKPT_HF_VERIFY=PASS epoch=144 bytes=225219235`）。

## ③ 启动与配置还原

- 配置：`git show <Beta>:src/mme_vla_suite/training/config.py` 中 `mme_vla_suite_4task_b128_50k`（80k 条目只改 name 与 num_train_steps=50_001）；history yaml `perceptual-framesamp-{modul,ctx}-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml`（只差 integration_type 与 memory_token_dim 1024/2048）。
- 帧记忆 8 帧×8×8（512）＋ motion 96 窗×1×1024，608 槽；B128、FSDP8、lr 恒 5e-5；NUM_TRAIN_STEPS=50001，CKPT_STEPS=5000…50000 共 10 份；评估 stage/50000、seed 7、692 局、32 片每卡 4 worker。
- run：`v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k`、`v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k`；bucket `HongzeFu/robomme-vla-4task-m8x8-{modul,ctx}-s2rgbstate-concat1x1024-50k-v1`。
- 正式链：`TRAIN_HEAD=<Beta 全 sha> FINAL_ROOT=$R/v1-store/validation/4task-rgbstate-concat-50k bash scripts/training/run_4task_concat_modul_eval_ctx_eval_50k.sh`（tmux v16-chain）。

## 上游交接与 GPU 起点（I.1）

- 消息 1 `req=vla16-11142010-1`（11-14 20:10 美东发）：② 处理中；结果（11-15 03:4x）done handoff=597f79f、ep144 sha=ece3465b…；③ 已回。
- 消息 2 `req=vla16-11150335-2`（11-15 03:35 发）：② 收尾不再需要 GPU、ep144 sha 一致；③ 已回。
- G1：train attempt1 `EXIT_CODE=0`；本方只读 `TRAIN_TERMINAL=PASS arch=wan-latent-stage2-blockwise-pool4-rgb-state epoch=144 epoch_completed=true global_step=13680`（11-15 03:30）。G2：p5eval `STAGE2_EVAL=DONE`、`EXIT_CODE=0`、会话已退出。G3：8 卡无计算进程。`GPU_START=PASS` 11-15 03:34 美东。

## 闸门判定行（原文）

```
FRAMESAMP8X8=PASS num_rows=1192918
RAW_H5_INVENTORY=RECORD files=54 sha_ok=54 g4prep_found=36/36 extra=18
ENCODER_REGISTER=PASS sha256=ece3465b57f9d1ddea130a631aff1d1006b2c8ba5c68a0e5e92b8215ae80b55c reg=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/external/motionjepa/stage2-4task-rgbstate-144ep-1113-a
DATA_LINEAGE=PASS mode=member-pin encoder_ckpt=ece3465b member_pin=2cb9026fc874964941ddb8c1cf1c0425a3b8ac94d3bd8e2082779f4e95f5c8bd members=1600 h5_files=4 maps=4 vla_input_pin=655e64b4fd033aea1ff79a1e5ad47f886a148592496759ed2a667fe167c57d86 mismatches=0
STAGE2_G2=PASS kind=local-built-latents project=wan_stage2_rgbstate windows=64 demo_tail=9 worst_mse=0.0 worst_max=0.0
STAGE2_G3=PASS kind=local-built-latents project=wan_stage2_rgbstate windows=32 seed=0 max_abs=0 leaves=65 masks=rgb-state
MOTION_XPATH=RECORD common=868 sampled=64 excluded=0 compared=64 max_abs=0.03838753700256348 mean_abs=0.0011832204995840812 max_rel=23.6973029397382 member_map_mismatch=0 no_cache=0 real_pad_mismatch=0 frame_ids_mismatch=0
STAGE2_G0=PASS kind=local-built-latents strategy=task4x16-endpoints-pad8-seed0-v1 windows=64 tasks=4 demo_tail=9 sha_mismatches=0 members_reused=1 members_extracted=63
ONLINE_INPUT_IDENTITY=PASS windows=64 handshake=1 input_sha_mismatch=0 source=member_h5
ONLINE_OFFLINE_PARITY=PASS windows=64 handshake=1 response_bytes=4096 encoder_max_abs=0 e2e_max_abs=0.0 frame_ids_mismatch=0 state_machine_mismatch=0 stub_windows=6 demo_tail=1 exec_batches=6 first_packet=D+1 max_followup_frames=16 reset=1
RGB_RECON_VIDEO=PASS arch=stage2-rgbstate groups=50 videos=50 frames=14712 gap_frames=162 psnr_mean=26.0115 floor_psnr_mean=29.7347 latent_rgb=0.0105321 identity_rgb=0.184181 gpus=4 elapsed_s=905.6
RGB_RECON_VIDEO=PASS arch=v7 groups=50 videos=50 frames=14712 gap_frames=162 psnr_mean=28.5978 floor_psnr_mean=29.7347 latent_rgb=0.00205198 identity_rgb=0.184181 gpus=4 elapsed_s=857.1
RGB_RECON_COMPOSE=PASS groups=50 top=v7:4task v7 ep72 bottom=stage2-rgbstate:4task rgbstate ep144 rendered=50 gpus=1 elapsed_s=659.2
OLD_ENC_HOLDOUT_OVERLAP=RECORD unseen=28/28 seen_by_old_encoder=-
HELDOUT_H5=PASS n=28 bytes=12752404536 out=/scratch/hongze/source-4task-h5-604f16da/h5-heldout manifest=/scratch/hongze/source-4task-h5-604f16da/h5-heldout/heldout_manifest.json manifest_sha256=2c74b6f0e89fe103b66296144a42d39608c2f72b9d6b6b6e4d4703012c468f6b
STAGE2_EXPORT=PASS segments=3200 rows=36484
ENCODER_BITEXACT=PASS compared=36484 mismatches=0
VERIFY_MOTION=PASS scanned=36484 mismatches=0 nonfinite=0 segment_set_mismatch=0 real_plus_pad_ne_33=0 exec_padded=0 pad_not_last_demo=0 pad_source_mismatch=0 demo_tail_padded=821 demo_tail_dropped=779 legal_zero_segments=0
```

## 实施中调整与工程故障（事后汇报）

1. 开工前一致性审查 FAIL n=1（日志收据名）：统一 delivery-<m|c>.log、up-<m|c>-final.log、$LOG/<RUN_NAME>.log（71b4c61）。
2. rgbstate venv 私有仓库 HTTPS 拉取缺凭据：仅该命令环境变量接 gh 凭据助手，未改全局配置。
3. 导出短测 `--max-segments 1` 与 `--check-dropout 64` 冲突（4task stride32 每段 ≤36 窗）：短测改 4 段，正式导出参数不变。
4. L0b 首次 sidecar 真实模式 AttributeError（未登记 sys.modules）：FIX1 修复、审查合入后重跑 PASS。
5. 审查提示导出 geometry 写死全 1：核 1113 val aug0 200/200 common_valid 全真，等价，未改代码。
6. 首个 Beta ae2b809 上短测失败：训练侧根快照缺 raw_dir（S2/S3 接缝）。FIX2 顺链修 3 处并在真实 motion 表 CPU 跑通快照与下游校验；G4 在线恢复输入复制到 $LOG/validation/（sha 不变）。重打本 Beta（V15.26Beta），ae2b809 作废、不用于正式训练。
7. 短测 SAVE_LOAD 两次调用修正（我方运行脚本，非代码）：父进程须 CUDA_VISIBLE_DEVICES=0；须设 MMEVLA_V1_STORE／OPENPI_DATA_HOME，并按历史口径设 JAX_COMPILATION_CACHE_DIR／MMEVLA_JAX_CACHE_DIR 指向训练同一编译缓存（未设时 reference_output_max_abs=2^-9，设后 0；EMA 恢复两次均 ema_mismatch=0）。
8. 由上条发现正式链 delivery 同样缺 MMEVLA_V1_STORE／OPENPI_DATA_HOME：FIX3 修复、审查合入。V15.26Beta 7a205b3 上的两条短测（TRAIN_FINAL_RECORD、SAVE_LOAD 均 PASS）仍有效：FIX3 只改链 delivery 环境与测试期望，不触训练与评估代码。重打本 Beta（V15.27Beta），7a205b3 作废、不用于正式训练。

## tmux 会话清单（本轮）

v16-heldout-h5、v16-l0、v16-recon；后续 v16-smoke、v16-chain、v16-up-m、v16-up-c、v16-up-eval-m、v16-up-eval-c。
