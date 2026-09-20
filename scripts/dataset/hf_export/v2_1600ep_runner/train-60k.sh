#!/usr/bin/env bash
# 512 modulation（motion 关闭）—— 复刻 run v2-1600ep-m8x8-modul-b128-60k。
# 60,000 step × global batch 128，config 内置 fsdp_devices=4 / num_workers=8；
# 8 卡机器想按 80k 那条的并行度跑，追加 --fsdp-devices 8 --num-workers 16。
# 用法: train-60k.sh <run_name> [额外的 train.py 参数...]
CONFIG=mme_vla_suite_b128_60k
HC=perceptual-framesamp-modul-8frame-8x8.yaml
RUN_NAME="${1:?用法: train-60k.sh <run_name> [额外参数...]}"; shift || true
EXTRA_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_train_common.sh" "$RUN_NAME"
