#!/usr/bin/env bash
# 512+motion modulation（motion 开启）—— 复刻 run v2-1600ep-m8x8-modul-motion-b128-80k。
# 80,000 step × global batch 128，config 内置 fsdp_devices=8 / num_workers=16。
# 每个 dataloader worker 常驻 pos 表 432 MB + state 表 36 MB + motion 表 219 MB ≈ 690 MB，
# 16 worker 约 11 GB 内存，起跑前确认机器有余量。
# 用法: train-80k.sh <run_name> [额外的 train.py 参数...]
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-modul-8frame-8x8-motion.yaml
RUN_NAME="${1:?用法: train-80k.sh <run_name> [额外参数...]}"; shift || true
EXTRA_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_train_common.sh" "$RUN_NAME"
