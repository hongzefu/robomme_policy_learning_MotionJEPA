#!/usr/bin/env bash
# modulation 2048（32 帧 × 8×8，motion 关闭，memory_token_dim 1024）—— 16 任务 mixed8-200 新库，run mixed8-200-m32x8x8-modul-b128-80k。
# 80,000 step × global batch 128（mme_vla_suite_b128_80k；1018-mixed8-2048-context-modul-plan.md）。
# config 内置 fsdp_devices=8 / num_workers=16；4 卡跑时调用方设 FSDP_DEVICES=4 NUM_WORKERS=8。
# 用法: [FSDP_DEVICES=4 NUM_WORKERS=8] train-modul2048-80k.sh <run_name> [额外的 train.py 参数...]
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-modul-32frame-8x8.yaml
RUN_NAME="${1:?用法: train-modul2048-80k.sh <run_name> [额外参数...]}"; shift || true
EXTRA_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_train_common.sh" "$RUN_NAME"
