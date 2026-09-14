# t8-m8-a2 启动记录

此记录为已授权计划的正式1000更新轨迹准备；只有400ep完整输入验收与对应100步排错通过后才启动。用户原话：“开始实现 有问题越早问用户越好 一口气全做完！”，以及“注意你只能用4个gpu”。本轨迹使用物理GPU 6,7，同profile三条始终同卡对，fsdp 2。run_name为用户已确认的 `t8-m8-a2`，启动前断言记录与checkpoint根均不存在，禁止overwrite/resume。

参考源码 `REF=99faacb1319adfc63c0cf9a15187e24c34e38fd1`；候选源码 `CAND=c08ec2060a544af1869c1e24f755e536150569ca`。本条为参考侧，实际源码HEAD必须等于REF，显式PYTHONPATH指REF/src。起跑HEAD、干净状态与四个源码导入绝对路径由run_meta及日志记录。

数据为400ep、101066个执行样本，直接读取源npy的验证专用Dataset，配置 `perceptual-framesamp-context-8frame-8x8-motion.yaml`，motion开启，显式绑定同库motion表。全部资产在本仓库v1-store，归一化明确使用 `train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json`，文件SHA `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`；按用户“采用文件 SHA＋解析后数组摘要双重检查”的决定，同时核实际loader收到的8个统计数组摘要。

batch 8、worker 4、seed 42、1000次参数更新、log/save interval 1均为启动覆盖，不改全局默认。XLA确定性档、独立scratch编译缓存、uv共享主树环境。硬件为AWS A100-SXM4-80GB，存储为 `/dev/md0` XFS本地NVMe RAID。

下列命令体放入detached tmux `t8-m8-a2`，外层 `set -o pipefail` + `tee v1-store/logs/t8-m8-a2.log`，结束记录UTC时间与 `EXIT_CODE=${PIPESTATUS[0]}`。同profile按A1→A2→B运行；A1/A2完整重复性先通过再起B。阶段5十二条轨迹期间不再提交代码或文档。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
unset JAX_PLATFORMS BENCH_STATE_DUMP_STEPS BENCH_STATE_DUMP_DIR BENCH_SAVE_FINAL_CKPT BENCH_FINAL_STEP BENCH_PERF_MODE BENCH_DUMP_IDX
export BENCH_RECORD_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-m8-a2
export BENCH_REF_COMMIT=99faacb1319adfc63c0cf9a15187e24c34e38fd1 BENCH_CAND_COMMIT=c08ec2060a544af1869c1e24f755e536150569ca
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/t8-m8-a2"
export BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000
export BENCH_EXTRA_DIGEST_STEPS=1,2,24,49,99,199,399,599,799
test ! -e "$BENCH_RECORD_DIR"
test ! -e "$V1_STORE/train-runs/t8-m8-a2"
test -z "$(git status --porcelain)"
export MMEVLA_MOTION_STORE="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/motion"
export PYTHONPATH="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/ref-8x8/src" UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv"
export BENCH_DATASET_IMPL=refnpy BENCH_REF_SOURCE="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source" BENCH_REF_MANIFEST="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json"
export BENCH_REF_MOTION="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/motion"

cd /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/ref-8x8
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\nRUN=t8-m8-a2\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir "$BENCH_RECORD_DIR" --v1-store "$V1_STORE" --source "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source" --manifest "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json" --dataset "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source" --norm-stats "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$BENCH_RECORD_DIR"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-m8-a1" --record-dir "$BENCH_RECORD_DIR" --steps 1000 --batch-size 8 --dataset "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source"
nvidia-smi --id=6,7 --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$BENCH_RECORD_DIR/gpu_util_dense.csv" &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite --exp-name t8-m8-a2 \
 --assets-base-dir "$V1_STORE/train-assets" --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
 --checkpoint-base-dir "$V1_STORE/train-runs/t8-m8-a2" --batch-size 8 --num-workers 4 --num-train-steps 1000 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 \
 --dataset-path "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source" --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
 --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8-motion.yaml --no-wandb-enabled
kill "$sampler_pid"
wait "$sampler_pid" 2>/dev/null || true
trap - EXIT
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/project_scalars.py "$BENCH_RECORD_DIR/metrics.jsonl" "$BENCH_RECORD_DIR/scalars_hex.tsv"
JAX_PLATFORMS=cpu uv run --no-sync python - <<'PY'
import bisect, datetime, json, math, os, pathlib, statistics
rec=pathlib.Path(os.environ["BENCH_RECORD_DIR"])
metrics=[json.loads(line) for line in (rec/"metrics.jsonl").read_text().splitlines()]
states=[json.loads(line) for line in (rec/"param_checksums.jsonl").read_text().splitlines()]
meta=json.loads((rec/"run_meta.json").read_text())
rows=[(r["step"],r["wall_time"]) for r in metrics]
digest={r["loop_step"] for r in states if r["phase"]=="post_update"}|{0}
intervals=[(s1,t0,t1) for (s0,t0),(s1,t1) in zip(rows,rows[1:]) if s1==s0+1 and s1>=50 and s1 not in digest and s0 not in digest]
dur=[b-a for _,a,b in intervals]
assert dur
cut=1.5*statistics.median(dur)
samples=[]
for line in (rec/"gpu_util_dense.csv").read_text().splitlines():
    p=[x.strip() for x in line.split(",")]
    if len(p)!=4: continue
    try: samples.append((datetime.datetime.strptime(p[0],"%Y/%m/%d %H:%M:%S.%f").timestamp(),int(p[1]),float(p[2]),float(p[3])))
    except ValueError: pass
assert samples and {g for _,g,_,_ in samples}==set(map(int,os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
starts=[a for _,a,_ in intervals]
group={"all":[],"slow":[],"other":[]}
for row in samples:
    ts,g,u,m=row
    i=bisect.bisect_right(starts,ts)-1
    if i<0 or ts>intervals[i][2]: continue
    _,a,b=intervals[i]
    group["all"].append(row)
    group["slow" if b-a>cut else "other"].append(row)
def util_summary(vals):
    if not vals: return {"n":0,"mean_pct":None,"zero_pct_share":None}
    return {"n":len(vals),"mean_pct":statistics.fmean(v[2] for v in vals),"zero_pct_share":100*sum(v[2]==0 for v in vals)/len(vals),"per_gpu":{str(g):{"mean_pct":statistics.fmean(v[2] for v in vals if v[1]==g),"zero_pct_share":100*sum(v[2]==0 for v in vals if v[1]==g)/sum(v[1]==g for v in vals)} for g in sorted({v[1] for v in vals})}}
out={"storage":"AWS 本地 NVMe RAID /dev/md0 XFS","batch_size":8,"num_workers":4,"warmup_steps":50,"steady_step_range":[50,len(metrics)-1],"excluded_digest_and_next":sorted(digest|{s+1 for s in digest}),"sampling_ms":500,"nvml_note":"按500ms采样，利用率本身为设备内部周期均值，相同读数不作为更高时间分辨率","steady_steps":len(dur),"step_mean_s":statistics.fmean(dur),"samples_per_second":8/statistics.fmean(dur),"slow_threshold_s":cut,"slow_definition":"超过稳态步时中位数的1.5倍；中位数只用于分层阈值","util":{k:util_summary(v) for k,v in group.items()},"peak_gpu_memory_mib":{str(g):max(m for _,gi,_,m in samples if gi==g) for g in sorted({v[1] for v in samples})},"checksum_seconds":sum(r.get("checksum_seconds",0) for r in states),"checksum_count":len(states),"observed_first_to_last_metric_s":rows[-1][1]-rows[0][1]}
assert out["util"]["all"]["n"]>0
(rec/"performance_stratified.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
env=json.loads((rec/"env.json").read_text())
env.update(batch_size=8,num_workers=4,log_interval=1,epoch_samples=meta["epoch_samples"],epoch_steps=meta["epoch_samples"]//8,storage=out["storage"],gpu_sampling_ms=500)
(rec/"env.json").write_text(json.dumps(env,ensure_ascii=False,indent=2)+"\n")
print("PERFORMANCE_DONE mean_step_s="+str(out["step_mean_s"])+" samples_s="+str(out["samples_per_second"])+" util_mean="+str(out["util"]["all"]["mean_pct"])+" zero_share="+str(out["util"]["all"]["zero_pct_share"]))
PY
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$BENCH_RECORD_DIR"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-m8-a1" --record-dir "$BENCH_RECORD_DIR" --steps 1000 --batch-size 8 --dataset "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source"
printf 'TRAJECTORY_DONE run=t8-m8-a2 steps=1000\n'

```

每次更新取五标量hex，完整TrainState取11个state_step `0,2,3,25,50,100,200,400,600,800,1000`，193叶；11份batch含None键与样本索引，主进程索引前8000项逐位比较且记录数至少8072。有限性、参数活性、真实norm_stats与导入来源同时通过，才接受对应profile的GATE_8X8。本条不额外保存大权重，完整参数、优化器和EMA由逐叶摘要取证。

稳态性能排除前50步及摘要步/下一步，500ms采样，报告步时均值、吞吐、GPU利用率均值、0%采样占比及慢步/其他步分层，单独报告全部摘要成本。显存为NVML记录的预分配占用，不等同于活跃张量内存。性能不作为放宽数值判据的理由。
