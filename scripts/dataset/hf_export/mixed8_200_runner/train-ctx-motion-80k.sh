#!/usr/bin/env bash
# 512+motion context 同路（motion 开启）—— 16 任务 mixed8-200 新库，run mixed8-200-m8x8-context-motion-b128-80k。
# 帧 512 与 motion 160 都拼进 PaliGemma 主干前缀（integration_type context，memory_token_dim 2048，59 参数叶）；
# 80,000 step × global batch 128（mme_vla_suite_b128_80k）；motion 表与 encoder 来源见
# perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml 的 motion.store_path / motion.source_run。
# 计划：1019-mixed8-ctx512-motion-plan.md。
# config 内置 fsdp_devices=8 / num_workers=16；4 卡跑时调用方设 FSDP_DEVICES=4 NUM_WORKERS=8。
# 用法: [FSDP_DEVICES=4 NUM_WORKERS=8] train-ctx-motion-80k.sh <run_name> [额外的 train.py 参数...]
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml
RUN_NAME="${1:?用法: train-ctx-motion-80k.sh <run_name> [额外参数...]}"; shift || true
EXTRA_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_train_common.sh" "$RUN_NAME"
