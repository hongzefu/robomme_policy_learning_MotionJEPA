#!/usr/bin/env bash
# 1031 GL 复现（gl-orig100-ctx-reproduce）专用占位 job。按正本 templates/hold_job.sbatch 实例化，
# 对应 greatlakes.md「资源约束」「算力使用规则」与 AGENTS.md 第 8 条。
#
# 占位 job 本身不跑工作负载，只 sleep infinity；工作负载一律经 srun 塞进来，运行器见同目录 run_in_hold_o16.sh：
#   srun --jobid=<占位 JobID> --overlap --exact --ntasks=1 --cpus-per-task=$((4*N)) --gpu_cmode=shared \
#     env GPUS=4 GLOBAL_BATCH=64 HISTORY=<yaml> RUN_NAME=<run> ... \
#     bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/scripts/training/gl/run_in_hold_o16.sh
#
# 规格：每卡 4 CPU + 32G 内存；卡数 N 与 walltime T 不写死，由提交命令行参数覆盖下方默认值
# （#SBATCH 行不做 shell 展开，只能写字面值；sbatch 命令行参数优先于脚本内 #SBATCH）。
# 下方默认值是 N=4、T=48:00:00（第一阶段已放行的 1 份 4×A40 48 h 占位）。
#
# 提交示例（在登录节点、turbo 仓库根执行；N、T 按实际填）：
#   N=4; T=48:00:00
#   jid=$(sbatch --parsable --gres=gpu:$N --cpus-per-task=$((4*N)) --mem=$((32*N))G --time=$T \
#           scripts/training/gl/hold_o16.sbatch) \
#     && echo "$jid" >> v1-store/logs/hold-jobs-o16gl.txt
#   第二阶段超长占位只改 T（如 T=7-00:00:00，按冒烟实测与 MaxTime 定）。
# 收尾只按清单逐个取消，禁止 scancel -u、禁止按名字模式批量取消：
#   while read -r jid; do scancel "$jid"; done < v1-store/logs/hold-jobs-o16gl.txt
# 注：不要加 --qos=interactive（chaijy2/spgpu 实测报 Invalid qos specification）；spgpu 强制至少 1 GPU。
# 注：--gpu_cmode=shared 让塞进来的多个进程共用同一张卡。
#
#SBATCH --job-name=o16gl-hold
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/o16gl-hold-%j.log
set -euo pipefail

echo "占位 job 已就位: JOB_ID=${SLURM_JOB_ID:-未知} 节点=$(hostname) GPU=${SLURM_GPUS_ON_NODE:-未知} 起始=$(date -Is)"
exec sleep infinity
