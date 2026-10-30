#!/usr/bin/env bash
# driver：把 16 任务 mixed8-200 新库上的 2048 modulation（32 帧 × 8×8，motion 关闭）80k run
# mixed8-200-m32x8x8-modul-b128-80k 的全部 16 份 orbax checkpoint（5000…75000 共 15 份 + 79999）、run 根下的
# config 快照与 provenance、清洗后的训练日志上传到**公开** HF bucket
# HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1，做「上传前 / 上传后」两遍独立 sha256 逐行对照验收。
#
# 计划：docs/root-archive/1018-mixed8-2048-context-modul-plan.md 第二部分 §1 S2 表；蓝本 run_mixed8_200_vla_512_80k_ckpt_export.sh
# （只换 run 身份：RUN_NAME、BUCKET_ID、EXPORT_NAME、ALLOWLIST、README_SRC、DRIVER_NAME）。全部阶段在
# _mixed8_200_vla_ckpt_export_common.sh，与其余 mixed8-200 VLA 驱动共用；本文件只钉死这一条 run 的身份。
# 训练起跑 runner：scripts/dataset/hf_export/mixed8_200_runner/train-modul2048-80k.sh
# （HC=perceptual-framesamp-modul-32frame-8x8.yaml）。训练中的增量上传走不进 git 的
# v1-store/m8v-scripts/m8v-ckpt-inc.sh；本驱动在训练 EXIT_CODE=0 之后做全量回读验收。
#
# 用法（正式运行在主检出；训练 EXIT_CODE=0 之后）：
#   SMOKE=1 bash scripts/dataset/hf_export/run_mixed8_200_vla_modul2048_80k_ckpt_export.sh  # 预检 + stage + pre sha + 体检，不碰网络
#   bash scripts/dataset/hf_export/run_mixed8_200_vla_modul2048_80k_ckpt_export.sh          # 全量
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="mixed8-200-m32x8x8-modul-b128-80k"
BUCKET_ID="HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1"
EXPORT_NAME="hf-ckpt-mixed8-200-vla-modul2048-80k"
README_SRC="$HERE/mixed8_200_vla_modul2048_80k_bucket_README.md"
ALLOWLIST="$HERE/hygiene_allowlist_mixed8_200_vla_modul2048_80k.json"
EXPECT_MOTION=false
DRIVER_NAME="run_mixed8_200_vla_modul2048_80k_ckpt_export.sh"
source "$HERE/_mixed8_200_vla_ckpt_export_common.sh"
