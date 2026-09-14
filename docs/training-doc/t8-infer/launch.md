# C8与M8 checkpoint推理及闭环启动口径

本记录承接[完整计划](../../../8frame-8x8-training-plan.md)，使用已经通过四组1000步逐位训练验收的两份候选EMA checkpoint。用户要求“一口气全做完”，限定四张GPU，并确认“让评估脚本读取 checkpoint 保存的配置，自动算出正确长度”；C8前缀应为1088，M8应为1184。新代码固定在CAND `c08ec2060a544af1869c1e24f755e536150569ca`，之后只有文档提交；各次运行的实际clean HEAD和UTC时间由外层日志记录。

## 输入、输出和判据

checkpoint为 `v1-store/train-runs/t8-{c8,m8}-b/mme_vla_suite/t8-{c8,m8}-b/999`，分别来自1000更新，保存EMA参数；原训练主树为 `38f0db46db19a3b645613e04fba3f6e635b47054`，源代码CAND。原始输入是400ep库，使用其已全量verify的 `framesamp-8x8/`。norm_stats明确取400ep原文件，其SHA为 `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`，并核训练/推理真实解析数组摘要。位置表与真实SigLIP三方池化已在[早期检查](../t8-infer-m8/result.md)通过，共用帧路径不重复跑。

按profile串行执行三个阶段，均从clean HEAD启动：`gates`为关0及关1–5，`probe`为一集仿真及关6，`batch`为48集完整闭环。关0只用CPU；关1–5及探针用物理GPU7；批量闭环用4、5、6、7四卡。全部位于AWS本地NVMe RAID `/dev/md0`，不与其他GPU任务混用资源。结果分别落 `v1-store/reports/t8-infer-<profile>/<phase>/`，日志 `v1-store/logs/t8-infer-<profile>-<phase>.log`。对应外层tmux会话同日志名去掉扩展名；批量另创建 `mv-t42-normal-t8c8-w0..3` 或 `mv-t42-normal-t8m8-w0..3`。

关1–5沿用默认5集、120决策点，M8真sidecar140窗，禁止max-points裁剪；bf16整段/缓存差保留为观察项，真f32/highest的15点重跑要求rel_fro≤1e-6且prefix KV逐位相同。探针为默认首任务ButtonUnmask的test episode0，seed42、max_steps1300、1集1任务；它验证在线不变量，不覆盖全部测试目标。旧400ep训练缺少ButtonUnmask test3、7、19、23的目标组合，本轮没有修改数据或放宽prompt门。48集保持驱动原始stride划分，包含3、7，不能为避开缺口删集。每种开关要求48/48、errors0、单集推理≤82、mem_order合法，M8真编码次数等于公式窗数；成功率记录但不作测试模型能力指标。

内存每0.5秒采样一次，仅采本轮进程树；闭环worker通过完整checkpoint路径和独有policy_name定位。记录RSS之和及逐进程峰值，共享页可能重复计数，因此不把RSS之和当作物理独占内存。nvidia-smi独立按500ms采显存和util。这里目的是内存证据，不据此下稳态吞吐结论；8×8在线4096行位置表理论大小768MiB，另报告实测RSS。仿真用已核实的micromamba环境，通过uv启动，GLIBC_TUNABLES沿用8192。

## 可复现命令

下方内联Python只负责按既有入口编排、记录日志及资源，没有修改模型或评估算法。每次令 `PROFILE=c8|m8`、`PHASE=gates|probe|batch`，在对应唯一tmux会话中执行；外层使用 `set -o pipefail`、`tee`，无论成功失败均写出 `EXIT_CODE=`。启动前校验Git状态为空、输出目录和日志不存在。探针server就绪后才启动客户端，结束后只终止该server自己创建的进程组；不触碰其他会话。批量驱动独立检查clean HEAD并拒绝续评。

```bash
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cuda
test -z "$(git status --porcelain)"
uv run --no-sync python - "$PROFILE" "$PHASE" <<'T8_INFER_PY'
import asyncio, contextlib, datetime, json, os, pathlib, signal, socket, sys, time
import psutil

REPO = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
V1 = REPO / "v1-store"
PROFILE, PHASE = sys.argv[1:3]
assert PROFILE in ("c8", "m8") and PHASE in ("gates", "probe", "batch")
MOTION = PROFILE == "m8"
CKPT = V1 / f"train-runs/t8-{PROFILE}-b/mme_vla_suite/t8-{PROFILE}-b/999"
LIB = V1 / "datasets/4task-motion-400ep"
NORM = V1 / "train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
OUT = V1 / f"reports/t8-infer-{PROFILE}" / PHASE
OUT.mkdir(parents=True, exist_ok=False)
assert (CKPT / "params").is_dir()
PORT = 9378
POLICY = f"t8-probe-{PROFILE}"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="7", JAX_PLATFORMS="cuda",
           XLA_PYTHON_CLIENT_MEM_FRACTION="0.55", PYTHONUNBUFFERED="1",
           MMEVLA_JAX_CACHE_DIR=str(V1 / f"cache/jax/t8-infer-{PROFILE}-{PHASE}"))
if MOTION:
    ENV["MMEVLA_MOTION_STORE"] = str(LIB / "motion")
else:
    ENV.pop("MMEVLA_MOTION_STORE", None)
PY = sys.executable
G0 = REPO / "scripts/training/g0"
active = {}
stop_sampling = asyncio.Event()
totals = {"sample_count": 0, "peak_sum_rss_bytes": 0, "peak_process_rss_bytes": {},
          "interval_s": 0.5, "rss_note": "本轮进程树RSS之和，共享页可能重复计数；不是物理独占内存"}

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

async def sample_host():
    with (OUT / "host-memory.jsonl").open("w") as fh:
        while not stop_sampling.is_set():
            targets = {}
            for role, p in list(active.items()):
                if p.returncode is None:
                    with contextlib.suppress(psutil.Error):
                        root = psutil.Process(p.pid)
                        for proc in [root, *root.children(recursive=True)]:
                            targets[proc.pid] = (role, proc)
            if PHASE == "batch":
                # tmux worker不是驱动的子进程，按本轮独有checkpoint和policy_name定位。
                for proc in psutil.process_iter(["pid", "cmdline"]):
                    with contextlib.suppress(psutil.Error):
                        args = proc.info["cmdline"] or []
                        server = any("serve_policy_mv.py" in a for a in args) and any(str(CKPT) in a for a in args)
                        client = any(a == "eval.py" or a.endswith("/eval.py") for a in args) and any(
                            f"mv-normal-s42-test-t8{PROFILE}-w" in a for a in args)
                        if server or client:
                            role = "policy" if server else "simulation"
                            for child in [proc, *proc.children(recursive=True)]:
                                targets[child.pid] = (role, child)
            rows = []
            for pid, (role, proc) in targets.items():
                with contextlib.suppress(psutil.Error):
                    rss = proc.memory_info().rss
                    rows.append({"pid": pid, "role": role, "rss_bytes": rss})
                    key = str(pid)
                    totals["peak_process_rss_bytes"][key] = max(rss, totals["peak_process_rss_bytes"].get(key, 0))
            total = sum(r["rss_bytes"] for r in rows)
            totals["peak_sum_rss_bytes"] = max(total, totals["peak_sum_rss_bytes"])
            totals["sample_count"] += 1
            fh.write(json.dumps({"utc": utc(), "sum_rss_bytes": total, "processes": rows}) + "\n")
            fh.flush()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_sampling.wait(), timeout=0.5)

async def pump(proc, log, ready=None):
    with log.open("w") as fh:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace")
            fh.write(line); fh.flush()
            print(line, end="", flush=True)
            if ready is not None and "server listening on" in line:
                ready.set()

async def start(role, cmd, env=None, cwd=REPO, ready=None):
    print(f"STAGE_START role={role} utc={utc()} argv={json.dumps([str(x) for x in cmd])}", flush=True)
    p = await asyncio.create_subprocess_exec(*map(str, cmd), cwd=str(cwd), env=env or ENV,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        start_new_session=True, limit=16*1024*1024)
    active[role] = p
    task = asyncio.create_task(pump(p, OUT / f"{role}.log", ready))
    return p, task

async def finish(role, pair):
    p, task = pair
    rc = await p.wait()
    await task
    print(f"STAGE_EXIT role={role} rc={rc} utc={utc()}", flush=True)
    if rc:
        raise RuntimeError(f"{role} 退出码 {rc}")

async def stop_owned(pair):
    p, task = pair
    if p.returncode is None:
        # start_new_session保证该进程组仅属于此处启动的server及其sidecar。
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(p.wait(), 30)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            await p.wait()
    await task
    print(f"SERVER_CONTROLLED_STOP pid={p.pid} rc={p.returncode}", flush=True)

async def main():
    gpu_ids = "4,5,6,7" if PHASE == "batch" else "7"
    gpu_file = (OUT / "gpu-500ms.csv").open("w")
    gpu = await asyncio.create_subprocess_exec("nvidia-smi", "-i", gpu_ids,
        "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits", "-lms", "500", stdout=gpu_file)
    host_task = asyncio.create_task(sample_host())
    try:
        if PHASE == "gates":
            common = ["--ckpt", CKPT, "--lib", LIB, "--store-subdir", "framesamp-8x8",
                      "--train-config", "mme_vla_suite", "--norm-stats", NORM]
            cpu_env = dict(ENV, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
            await finish("gate0", await start("gate0", [PY, G0 / "check_config_provenance.py", *common,
                "--neg-lib", V1 / "datasets/4task-motion-40ep", "--out", OUT / "provenance.json"], cpu_env))
            extra = ["--motion", "sidecar", "--motion-gpu", "7"] if MOTION else ["--motion", "store"]
            await finish("gates1-5", await start("gates1-5", [PY, G0 / "compare_train_infer_obs.py",
                *common, *extra, "--f32-diag-points-per-episode", "3", "--out", OUT / "tic.json"]))
        elif PHASE == "probe":
            assert not (V1 / "evaluation" / POLICY).exists()
            with socket.socket() as sock:
                assert sock.connect_ex(("127.0.0.1", PORT)) != 0, "探针端口已占用"
            ready = asyncio.Event()
            extra = ["--motion-gpu", "7"] if MOTION else []
            server = await start("server", [PY, G0 / "serve_policy_probe.py", "--ckpt", CKPT,
                "--lib", LIB, "--store-subdir", "framesamp-8x8", "--config", "mme_vla_suite",
                "--port", PORT, "--host", "127.0.0.1", "--seed", "42",
                "--probe-out", OUT / "probe.jsonl", *extra], ready=ready)
            try:
                waiter = asyncio.create_task(ready.wait())
                exited = asyncio.create_task(server[0].wait())
                done, pending = await asyncio.wait([waiter, exited], timeout=1800,
                    return_when=asyncio.FIRST_COMPLETED)
                for task in pending: task.cancel()
                assert ready.is_set() and server[0].returncode is None, "探针server未就绪"
                sim_env = dict(ENV, GLIBC_TUNABLES="glibc.rtld.optional_static_tls=8192")
                sim = ["uv", "run", "--no-project", "--no-sync",
                    "/scratch/hongze/micromamba/envs/robomme/bin/python", "eval.py",
                    f"--args.port={PORT}", "--args.host=127.0.0.1", "--args.model_seed=42",
                    f"--args.policy_name={POLICY}", "--args.model_ckpt_id=999",
                    "--args.only_tasks=ButtonUnmask", "--args.episode_start=0", "--args.max_episodes=1",
                    "--args.episode_stride=1", "--args.max_steps=1300", "--args.dataset=test",
                    f"--args.save_dir={V1 / 'evaluation'}"]
                await finish("simulation", await start("simulation", sim, sim_env,
                    REPO / "scripts/motion-variance/robomme"))
            finally:
                await stop_owned(server)
            progress = V1 / "evaluation" / POLICY / "ckpt999/seed42/progress.json"
            await finish("gate6", await start("gate6", [PY, G0 / "summarize_eval_probe.py",
                "--probe", OUT / "probe.jsonl", "--progress", progress,
                "--eval-log", OUT / "simulation.log", "--server-log", OUT / "server.log",
                "--lib", LIB, "--max-frames", "8", "--tokens-per-frame", "64",
                "--expect-episodes", "1", "--expect-tasks", "1", "--max-steps", "1300",
                "--out", OUT / "probe-summary.json"], dict(ENV, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")))
        else:
            for worker in range(4):
                assert not (V1 / f"evaluation/mv-normal-s42-test-t8{PROFILE}-w{worker}").exists()
                assert not (V1 / f"logs/mv-t42-normal-t8{PROFILE}-w{worker}.log").exists()
            batch_env = dict(ENV, COND="normal", SPLIT="test", SEED="42", WORKERS="4",
                GPU_LIST="4,5,6,7", CUDA_VISIBLE_DEVICES="4,5,6,7", EP_COUNT="10",
                RUN_SUFFIX=f"-t8{PROFILE}", CKPT_OVERRIDE=str(CKPT), MV_MOTION_OFF="0" if MOTION else "1",
                PORT_BASE="9370", MV_ALLOW_DIRTY="0")
            await finish("batch", await start("batch", ["bash", REPO / "scripts/motion-variance/run_batch_mv.sh"], batch_env))
    finally:
        stop_sampling.set()
        await host_task
        (OUT / "host-memory-summary.json").write_text(json.dumps(totals, indent=2) + "\n")
        if gpu.returncode is None:
            gpu.terminate()
        await gpu.wait()
        gpu_file.close()
    print(f"INFER_PHASE=PASS profile={PROFILE} phase={PHASE} utc={utc()}", flush=True)

asyncio.run(main())
T8_INFER_PY
```

完整执行顺序为C8的gates、probe与M8的gates、probe，然后M8及C8各一批48集；长任务按GPU空闲情况依次启动。全梯度任务完成并释放相应GPU后才使用这些设备。关0 CPU检查也保留在其所属可复现会话内。

## 当前状态

此文件为起跑前记录；实测结果分别回写[C8](../t8-infer-c8/result.md)与[M8](../t8-infer-m8/result.md)，不能把这里的目标数字当作已通过。
