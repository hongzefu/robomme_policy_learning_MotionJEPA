#!/usr/bin/env bash
# 守在 detached tmux 里：等 80k 训练进程退出 → 宽限 5 分钟（让驱动写完 END_UTC/EXIT_CODE、
# 让 wandb 把本地 run 目录 sync 完）→ 自动跑趟 B。
#
# 趟 B 不需要人工输入：README 的「训练结果」一节由 motion80k_result_block.py 从训练日志现读生成。
# 训练没跑满 16 个 step 目录、或末尾不是 79999，导出脚本自己会断言失败并停下，不会传出半份东西。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
RUN_NAME="v2-1600ep-m8x8-modul-motion-b128-80k"
ELOG="$REPO/v1-store/exports/hf-ckpt-$RUN_NAME/logs/export.log"

echo "[watcher] $(date -u '+%F %T UTC') 开始等待训练退出"
while pgrep -f "train\.py mme_vla_suite_b128_80k" > /dev/null; do sleep 120; done
echo "[watcher] $(date -u '+%F %T UTC') 训练进程已退出，宽限 300s 后启动趟 B"
sleep 300
echo "[watcher] $(date -u '+%F %T UTC') 启动趟 B"
bash "$REPO/scripts/dataset/hf_export/run_motion80k_ckpt_export.sh" 2>&1 | tee -a "$ELOG"
echo "[watcher] $(date -u '+%F %T UTC') 趟 B 结束，退出码 ${PIPESTATUS[0]}"
