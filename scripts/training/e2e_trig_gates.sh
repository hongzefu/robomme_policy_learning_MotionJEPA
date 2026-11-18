# shellcheck shell=bash
# e2e-trig 触发器的闸门表（1117-e2e-enc-train-plan.html 第二部分〇.1.1、三节 G1–G6、RB-C1 / RB-D1；S5）。
# 由 scripts/training/e2e_wait_nm.sh --gates-file <本文件> source；每行
#   gate_def <gate> <aNN> <tmux 会话> <日志> <前置检查命令> <启动命令> <判定命令>
# 三条命令以 bash -c 在仓库根目录执行（主检出 /scratch/hongze/robomme_policy_learning_MotionJEPA），可用 GATE GATE_ATTEMPT
# GATE_SESSION GATE_LOG E2E_REPO E2E_LOCK_DIR 与帮助函数 trig_verdict_all / trig_need_sub / trig_file_has。
# 次序（推荐次序，触发器每轮最多起一个、前一闸门判过且 8 卡 memory.used 之和为 0 才起下一个）：
#   G1（含 G1b/G1c，主会话已跑完：只判 v1-store/logs/e2e-g1.log，永不启动）→ G3（1 卡）→ G4（8 卡 --smoke20）
#   → G6-bench → G6-shmprobe → G6-ckptprobe（末段聚合打印 G6 各行）→ G5（1 卡）→ G5b（GPU 0 同卡 4 片）。
# 不在表内、由主会话在 CPU 上跑并写固定日志名的前置（触发器只读其判定行）：
#   G9 → v1-store/logs/e2e-g9.log（INIT_EQ=PASS、R3_OFF_EQ=PASS）；G0 → v1-store/logs/e2e-g0.log（DATA_LINEAGE=PASS）。
# 重做某闸门：主会话把旧锁改名为 ….launched.superseded-aNN，并把本表对应行的 aNN（会话名、日志名、run 名）一起递增。
# G5 依赖 BinFill:367（全库 c_env 最长局）的单 episode H5 在盘：
#   v1-store/datasets/4task-v2-1600ep-604f16da/raw-h5/record_dataset_BinFill_hard/hdf5_files/BinFill_ep111_seed15100.h5
#   （sha256 9d57e21e…30a2，1,529,131,008 B；本机原无，需主会话先按 map 的 tar_object 补下），缺则 G5 前置报 missing。

E2E_L="$LOCK_DIR"
E2E_V1="$REPO/v1-store"
E2E_DS="$E2E_V1/datasets/4task-v2-1600ep-604f16da"
E2E_RUNS="$E2E_V1/train-runs/mme_vla_suite_b128_80k"
E2E_YAML=src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-e2eenc.yaml
E2E_RUNNER=scripts/training/run_4task_e2eenc_50k.sh
E2E_G5_EPISODES=BinFill:367,BinFill:357,RouteStick:366,VideoRepick:191,VideoUnmaskSwap:345
E2E_G5_LONG_H5=$E2E_DS/raw-h5/record_dataset_BinFill_hard/hdf5_files/BinFill_ep111_seed15100.h5

# G1 / G1b / G1c：主会话 2026-11-18 美东 00:05–00:07 已在 GPU0 跑完（launch.md）；两行 E2E_PORT_PARITY（depth=1、depth=8）都须 PASS
gate_def G1 a01 e2e-g1 "$E2E_V1/logs/e2e-g1.log" \
  'echo done_by_main_session; exit 1' \
  'false' \
  'v=$(trig_verdict_all "$GATE_LOG" E2E_PORT_PARITY E2E_BF16_FLOOR E2E_BF16_REF); n=$(grep -c "^E2E_PORT_PARITY=PASS" "$GATE_LOG" 2>/dev/null); if [ "$v" = PASS ] && [ "${n:-0}" -lt 2 ]; then echo PENDING; else echo "$v"; fi'

# G3（1 卡，GPU 0）：S1–S3 合入、新表在盘、G9 通过
gate_def G3 a01 e2e-g3-a01 "$E2E_L/e2e-g3-a01.log" \
  "trig_need_sub S1 S2 S3 || exit 1; trig_file_has $E2E_V1/logs/e2e-g9.log '^INIT_EQ=PASS' && trig_file_has $E2E_V1/logs/e2e-g9.log '^R3_OFF_EQ=PASS' || { echo g9; exit 1; }; [ -f $E2E_DS/motion-wan-raw9-s16/meta/store_meta.json ] || { echo table16; exit 1; }" \
  "CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 JAX_DEFAULT_MATMUL_PRECISION=highest uv run --no-sync python scripts/training/g0/e2e_mem_vs_motionmem.py --ckpt $E2E_V1/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt --table768 $E2E_DS/motion --table16 $E2E_DS/motion-wan-raw9-s16 --latents $E2E_DS/wan-latents --windows scripts/training/g0/e2e_windows_v1.json --torch-device cuda" \
  'trig_verdict_all "$GATE_LOG" E2E_MEM_FP32 E2E_MEM_VS_MOTIONMEM'

# G4（8 卡）：runner --smoke20（run 名 e2e-smoke20，check_checkpoint RUNS 同名），需 S3、S5 合入与 G0 PASS；TRAIN_HEAD 取当前 HEAD（runner 核 clean）
gate_def G4 a01 e2e-g4-a01 "$E2E_L/e2e-g4-a01.log" \
  "trig_need_sub S3 S5 || exit 1; trig_file_has $E2E_V1/logs/e2e-g0.log '^DATA_LINEAGE=PASS' || { echo g0; exit 1; }" \
  "bash $E2E_RUNNER \"\$(git rev-parse HEAD)\" --smoke20" \
  'trig_verdict_all "$GATE_LOG" DATA_LINEAGE E2E_TRAINABLE SMOKE20'

# G6 ①②③（8 卡，依次）：run 名与会话名同带 -a01；末段 ckptprobe 由 runner 调 e2e_g6_judge.py aggregate 打印 G6 各行
gate_def G6-bench a01 e2e-bench-a01 "$E2E_L/e2e-bench-a01.log" \
  'trig_need_sub S3 S5' \
  "bash $E2E_RUNNER \"\$(git rev-parse HEAD)\" --bench300 --run-name e2e-bench-a01" \
  'trig_verdict_all "$GATE_LOG" G6_STAGE BENCH300'
gate_def G6-shmprobe a01 e2e-shmprobe-a01 "$E2E_L/e2e-shmprobe-a01.log" \
  'trig_need_sub S3 S5' \
  "bash $E2E_RUNNER \"\$(git rev-parse HEAD)\" --shmprobe --run-name e2e-shmprobe-a01" \
  'trig_verdict_all "$GATE_LOG" G6_STAGE SHM_PEAK'
gate_def G6-ckptprobe a01 e2e-ckptprobe-a01 "$E2E_L/e2e-ckptprobe-a01.log" \
  'trig_need_sub S3 S5' \
  "bash $E2E_RUNNER \"\$(git rev-parse HEAD)\" --ckptprobe --run-name e2e-ckptprobe-a01" \
  'trig_verdict_all "$GATE_LOG" G6_STAGE SMOKE_CKPT BENCH300 GPU_UTIL SHM_PEAK GPU_PEAK G6'

# G5（1 卡，GPU 0；sidecar 同卡）：S4 合入、G4 冒烟权重在盘、最长局 H5 在盘
gate_def G5 a01 e2e-g5-a01 "$E2E_L/e2e-g5-a01.log" \
  "trig_need_sub S4 || exit 1; [ -d $E2E_RUNS/e2e-smoke20/19/params ] || { echo g4_ckpt; exit 1; }; [ -f $E2E_G5_LONG_H5 ] || { echo h5_BinFill367; exit 1; }" \
  "CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run --no-sync python scripts/training/g0/compare_train_infer_obs.py --acceptance --episodes $E2E_G5_EPISODES --ckpt $E2E_RUNS/e2e-smoke20/19 --lib $E2E_DS --store-subdir framesamp-8x8 --train-config mme_vla_suite_b128_80k --norm-stats $E2E_V1/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json --h5-root $E2E_DS/raw-h5 --episode-map-dir $REPO/docs/dataset-build-doc/4task-v2-1600ep-604f16da/records --history-yaml $E2E_YAML --motion sidecar --motion-subdir motion-wan-raw9-s16 --motion-gpu 0 --out $E2E_V1/reports/tic/e2e-g5-a01.json" \
  'trig_verdict_all "$GATE_LOG" TIC_SIDECAR MOTION_S_VS_SIDECAR MEM_S_VS_TRAINSET OBS_T_VS_I'

# G5b（GPU 0 同卡 4 片，显存占比 0.2）：评估目录 v1-store/evaluation/e2e-g5b-a01
gate_def G5b a01 e2e-g5b-a01 "$E2E_L/e2e-g5b-a01.log" \
  "[ -d $E2E_RUNS/e2e-smoke20/19/params ] || { echo g4_ckpt; exit 1; }" \
  "RUN_NAME=e2e-g5b-a01 EVAL_PER_GPU=4 EVAL_MEM_FRACTION=0.2 bash scripts/evaluation/local_692_pool_e2e.sh --g5b $E2E_RUNS/e2e-smoke20/19 7" \
  'trig_verdict_all "$GATE_LOG" EVAL692_PLAN_IDENTITY E2E_ONLINE_PERF'
