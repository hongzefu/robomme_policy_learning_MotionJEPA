#!/usr/bin/env bash
# driver：把 16 任务 mixed8-200 新库上的 512+motion modulation（motion 开启）80k run
# mixed8-200-m8x8-modul-motion-b128-80k 的全部 16 份 orbax checkpoint（5000…75000 共 15 份 + 79999）、run 根下的
# config 快照与 provenance、清洗后的训练日志上传到**公开** HF bucket
# HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1，做「上传前 / 上传后」两遍独立 sha256 逐行对照验收。
#
# 计划：1016-mixed8-vla-plan.html 3.1 表 A1b 行、口径 29 / 30；蓝本 run_motion80k_ckpt_export.sh /
# motion80k_bucket_README.md（同为 80k 的 16 份）。全部阶段在 _mixed8_200_vla_ckpt_export_common.sh，
# 与 512 那条共用；本文件只钉死这一条 run 的身份。与蓝本的差别（公开、去私有开关、private==false
# 断言、公开前体检、回读前 df 断言、完成信号在最后）见共用体文件头。
#
# 用法（正式运行在主检出；训练 EXIT_CODE=0 之后）：
#   SMOKE=1 bash scripts/dataset/hf_export/run_mixed8_200_vla_motion_80k_ckpt_export.sh  # 预检 + stage + pre sha + 体检，不碰网络
#   bash scripts/dataset/hf_export/run_mixed8_200_vla_motion_80k_ckpt_export.sh          # 全量
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="mixed8-200-m8x8-modul-motion-b128-80k"
BUCKET_ID="HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1"
EXPORT_NAME="hf-ckpt-mixed8-200-vla-motion-80k"
README_SRC="$HERE/mixed8_200_vla_motion_80k_bucket_README.md"
ALLOWLIST="$HERE/hygiene_allowlist_mixed8_200_vla_motion_80k.json"
EXPECT_MOTION=true
DRIVER_NAME="run_mixed8_200_vla_motion_80k_ckpt_export.sh"
source "$HERE/_mixed8_200_vla_ckpt_export_common.sh"
