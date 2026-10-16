#!/usr/bin/env bash
# 512 modulation（motion 关闭）—— 16 任务 mixed8-200 新库，run mixed8-200-m8x8-modul-b128-80k。
# 80,000 step × global batch 128（mme_vla_suite_b128_80k；用户裁决 512 也改 80k）。
# config 内置 fsdp_devices=8 / num_workers=16；4 卡跑时调用方设 FSDP_DEVICES=4 NUM_WORKERS=8。
# 用法: [FSDP_DEVICES=4 NUM_WORKERS=8] train-512-80k.sh <run_name> [额外的 train.py 参数...]
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-modul-8frame-8x8.yaml
RUN_NAME="${1:?用法: train-512-80k.sh <run_name> [额外参数...]}"; shift || true
EXTRA_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_train_common.sh" "$RUN_NAME"
