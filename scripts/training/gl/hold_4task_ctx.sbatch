#!/usr/bin/env bash
# GL 四任务×400 context 512/1024 b128 50k 训练（计划 1111-gl-4task400-ctx512-1024-b128-50k-plan.html）专用占位 job。
# 按正本 templates/hold_job.sbatch 实例化，底稿为 1101 GL 复现实际用过的 hold_o16.sbatch；
# 对应 greatlakes.md「资源约束」「算力使用规则」与 AGENTS.md 第 8 条。
#
# 占位 job 本身不跑工作负载，只 sleep infinity；工作负载一律经 srun 塞进来，运行器见同目录 run_in_hold_4task_ctx.sh：
#   srun --jobid=<占位 JobID> --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
#     /usr/bin/env RUN_NAME=<run> CONFIG=<config> HISTORY=<yaml 文件名> GRAD_ACCUM=<k> MAX_STEPS=<n> \
#         TRAIN_HEAD=<40 位 sha> EXPECT_LIB_PIN=<64 位 sha256> [FSDP=4] [WORKERS=8] [RESUME=0|1] \
#     bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/scripts/training/gl/run_in_hold_4task_ctx.sh
#
# 规格固定为 4×A40、16 CPU、128G（每卡 4 CPU + 32G）；--time 与 --job-name 不写在本文件里，
# 一律由提交命令行给（sbatch 命令行参数优先于脚本内 #SBATCH；本文件不设默认值，漏给 --time 会落到分区默认值，务必写上）。
#
# 提交示例（在登录节点、turbo 仓库根执行；T 按计划与 MaxTime 定）：
#   T=7-00:00:00
#   jid=$(sbatch --parsable --job-name=g4c-hold --time=$T scripts/training/gl/hold_4task_ctx.sbatch) \
#     && echo "$jid" >> v1-store/logs/hold-jobs-g4c.txt
# 收尾只按清单逐个取消，禁止 scancel -u、禁止按名字模式批量取消：
#   while read -r jid; do scancel "$jid"; done < v1-store/logs/hold-jobs-g4c.txt
# 注：不要加 --qos=interactive（chaijy2/spgpu 实测报 Invalid qos specification）；spgpu 强制至少 1 GPU。
# 注：--gpu_cmode=shared 让塞进来的多个进程（及 torch + 图形互操作）共用同一张卡。
#
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/g4c-hold-%j.log
set -euo pipefail

echo "占位 job 已就位: JOB_ID=${SLURM_JOB_ID:-未知} 节点=$(hostname) GPU=${SLURM_GPUS_ON_NODE:-未知} 起始=$(date -Is)"
exec sleep infinity
