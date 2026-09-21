"""包装真实训练入口进行轻量计时，并从完整记录生成八卡稳态与80k外推报告。"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]


class HostTiming:
    """保留主线程阶段，仅在99、899末尾和999保存前同步；没有 profiler。"""

    def __init__(self, root, steps):
        if steps != 1000:
            raise ValueError("独立测速固定1000步")
        self.file = (Path(root) / "step_timing.jsonl").open("x")
        self.row = None
        self.count = 0
        self.closed = False
        self.flush = None

    def synchronize(self, location):
        if self.flush is None:
            raise RuntimeError("缺少真实训练 flush 回调")
        start = time.perf_counter()
        self.flush()
        self.row["sync"] = {"location": location, "wall_end": time.time(),
                            "seconds": time.perf_counter() - start}

    @contextlib.contextmanager
    def step(self, step, *, flush=None):
        if step != self.count or self.closed or self.count >= 1000:
            raise ValueError("计时步骤重复、乱序或关闭后继续")
        self.flush = flush
        self.row = {"step": int(step), "wall_start": time.time(), "phases_s": {}}
        start = time.perf_counter()
        completed = False
        try:
            yield
            if step in (99, 899):
                self.synchronize("window_end")
            completed = True
        finally:
            self.row.update(completed=completed, host_step_s=time.perf_counter() - start, wall_end=time.time())
            self.file.write(json.dumps(self.row) + "\n")
            self.file.flush()
            self.row = None
            self.flush = None
            self.count += 1

    @contextlib.contextmanager
    def phase(self, name):
        if self.row is None:
            raise RuntimeError("阶段没有所属训练步")
        if name == "checkpoint":
            if self.row["step"] != 999:
                raise ValueError("测速不允许中间保存")
            self.synchronize("before_final_save")
            self.row["save_wall_start"] = time.time()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.row["phases_s"][name] = time.perf_counter() - start

    def close(self, *, flush=None):
        if not self.closed:
            self.closed = True
            self.file.close()
            print(f"HOST_TIMING_DONE steps={self.count} profiler=0", flush=True)


def host_sample(pid):
    """RSS分别记主进程和后代，不将共享页重复累计冒充物理独占内存。"""
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        if key in ("MemAvailable", "Cached"):
            info[key + "_bytes"] = int(rest.split()[0]) * 1024
    queue, rss = [pid], {}
    while queue:
        current = queue.pop()
        if str(current) in rss:
            continue
        try:
            statm = (Path("/proc") / str(current) / "statm").read_text().split()
            rss[str(current)] = int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
            tasks = Path("/proc") / str(current) / "task"
            children = set()
            for task in tasks.iterdir():
                children.update(int(x) for x in (task / "children").read_text().split())
            queue.extend(children)
        except (FileNotFoundError, ProcessLookupError):
            continue
    shm = os.statvfs("/dev/shm")
    info.update(wall_time=time.time(), main_pid=pid, rss_bytes=rss,
                shm_used=(shm.f_blocks-shm.f_bfree)*shm.f_frsize,
                shm_capacity=shm.f_blocks*shm.f_frsize,
                md0_read_sectors=int(Path("/sys/block/md0/stat").read_text().split()[2]))
    return info


def sampler(path, pid, stopped):
    with Path(path).open("x") as f:
        count=0
        while True:
            sample=host_sample(pid)
            if count%20==0:
                processes={}
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdecimal():continue
                    try:
                        io={k:int(v) for k,v in (line.split(":",1) for line in (entry/"io").read_text().splitlines())}
                        processes[entry.name]={"name":(entry/"comm").read_text().strip(),
                                              "read_bytes":io["read_bytes"],"write_bytes":io["write_bytes"]}
                    except (OSError,ValueError,KeyError):continue
                sample["system_io"]=processes
            f.write(json.dumps(sample) + "\n")
            f.flush()
            count+=1
            if stopped.wait(0.5):
                break


def run(args):
    root = ROOT
    records = Path(args.records).resolve()
    if not records.is_relative_to(root / "v1-store"):
        raise ValueError("记录必须落本仓库 v1-store")
    if os.environ.get("XLA_FLAGS"):
        raise ValueError("生产测速必须 unset 验证用 XLA_FLAGS")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True):
        raise ValueError("测速必须从 clean HEAD 起跑")
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    if not train_args:
        raise ValueError("缺真实训练 CLI 参数")
    sys.path.insert(0,str(root/"scripts/training"))
    from modul_launch_contract import validate_argv
    validate_argv(train_args,args.mode)
    records.mkdir(parents=True, exist_ok=False)
    wrapper_started=time.time()
    # 按Linux进程出生时刻计P，包含Python导入；不以模型初始化完成冒充起点。
    ticks=int(Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19])
    started=time.time()-time.clock_gettime(time.CLOCK_BOOTTIME)+ticks/os.sysconf("SC_CLK_TCK")
    metadata = {"start_wall": started,"wrapper_start_wall":wrapper_started,"start_clock":"proc_start_ticks+CLOCK_BOOTTIME", "pid": os.getpid(), "argv": train_args,"mode":args.mode,
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "profiler": False, "entry_calls": 0, "save_calls": []}
    expected_head=os.environ.get("MMEVLA_EXPECTED_TRAIN_HEAD")
    if expected_head and metadata["head"]!=expected_head:
        raise ValueError("源码在preflight后发生变化")
    (records / "speed_start.json").write_text(json.dumps(metadata, indent=2))
    os.environ["TRAIN_RECORD_DIR"] = str(records)
    os.environ["TRAIN_TIMING_STEPS"] = "1000" if args.mode == "perf" else "0"
    import step_timing
    import jax
    from openpi.training import checkpoints
    from mme_vla_suite.training import dataloader

    def forbidden_trace(*a, **kw):
        raise RuntimeError("测速禁止开启 JAX profiler")

    original_timer, original_trace = step_timing.StepTiming, jax.profiler.start_trace
    original_save = checkpoints.save_state
    original_loader = dataloader.create_data_loader
    step_timing.StepTiming = HostTiming
    jax.profiler.start_trace = forbidden_trace

    def observed_save(manager, state, loader, step):
        expected_step = 999 if args.mode == "perf" else 19
        if step != expected_step:
            raise ValueError(f"真实保存步骤不是{expected_step}")
        entry = {"step": int(step), "state_step": int(state.step), "start_wall": time.time(),
                 "checkpoint_root": str(manager.directory)}
        result = original_save(manager, state, loader, step)
        entry["return_wall"] = time.time()
        metadata["save_calls"].append(entry)
        return result

    checkpoints.save_state = observed_save
    def observed_loader(*a,**kw):
        result=original_loader(*a,**kw)
        loader=result._data_loader.torch_loader
        metadata["loader"]={"batch_size":loader.batch_size,"workers":loader.num_workers,
                            "prefetch_factor":loader.prefetch_factor,"persistent_workers":loader.persistent_workers,
                            "pin_memory":loader.pin_memory}
        return result
    dataloader.create_data_loader=observed_loader
    stopped = threading.Event()
    def sample_resources():
        try:
            sampler(records / "host_samples.jsonl",os.getpid(),stopped)
        except Exception as error:
            metadata["sampling_error"]=repr(error)
    thread = threading.Thread(target=sample_resources, daemon=True)
    thread.start()
    previous = sys.argv
    try:
        if step_timing.StepTiming is not HostTiming:
            raise RuntimeError("轻量计时替换未生效")
        sys.argv = [str(root / "scripts/training/train.py"), *train_args]
        metadata["entry_calls"] += 1
        runpy.run_path(sys.argv[0], run_name="__main__")
        metadata["success"] = True
    finally:
        metadata["end_wall"] = time.time()
        stopped.set()
        thread.join(timeout=5)
        metadata["sampler_stopped"] = not thread.is_alive()
        (records / "speed_run.json").write_text(json.dumps(metadata, indent=2))
        sys.argv = previous
        checkpoints.save_state = original_save
        dataloader.create_data_loader=original_loader
        step_timing.StepTiming, jax.profiler.start_trace = original_timer, original_trace


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def must(ok, why):
    if not ok:
        raise ValueError(why)


def gpu_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        must(len(parts) == 4, "GPU采样行不完整")
        stamp = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
        rows.append((stamp, int(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def report(args):
    root = Path(args.records)
    meta = json.loads((root / "speed_run.json").read_text())
    timing = read_rows(root / "step_timing.jsonl")
    metrics = read_rows(root / "metrics.jsonl")
    must(meta.get("success") and meta["entry_calls"] == 1 and meta["sampler_stopped"] and not meta.get("sampling_error"), "训练或采样未成功完成")
    must(not meta["profiler"] and not (root / "step_trace").exists(), "存在profiler或trace")
    must(meta["loader"]["batch_size"]==128 and meta["loader"]["workers"]==16 and meta["loader"]["prefetch_factor"]==2,"实际loader形制不同")
    must([r["step"] for r in timing] == list(range(1000)) and all(r["completed"] for r in timing), "1000步计时不完整")
    must([(r["step"], r["sync"]["location"]) for r in timing if "sync" in r] ==
         [(99, "window_end"), (899, "window_end"), (999, "before_final_save")], "三处同步位置不符")
    must([r["step"] for r in metrics] == list(range(0, 1000, 100)), "生产log100记录不完整")
    scalar_keys = {"loss", "grad_norm", "llm_grad_norm", "mem_enc_norm", "param_norm"}
    must(all(set(r)-{"step", "wall_time"} == scalar_keys and
             all(math.isfinite(r[k]["dec"]) for k in scalar_keys) for r in metrics), "记录标量缺失或非有限")
    log_lines=Path(args.log).read_text().splitlines()
    must(log_lines[-1] == "EXIT_CODE=0", "缺成功退出记录")
    launch_rows=[json.loads(line.removeprefix("LAUNCH_RESOLVED=")) for line in log_lines if line.startswith("LAUNCH_RESOLVED=")]
    environments=[json.loads(line.removeprefix("RUN_ENV=")) for line in log_lines if line.startswith("RUN_ENV=")]
    must(len(launch_rows)==len(environments)==1,"缺唯一的实际配置与环境记录")
    must("TRAIN_HEAD="+meta["head"] in log_lines,"起跑日志与测速源码锚点不同")
    actual=launch_rows[0]["actual"]
    must((actual["batch_size"],actual["num_train_steps"],actual["num_workers"],actual["fsdp_devices"])==(128,1000,16,8),"测速实际配置不同")
    must(actual["history_values"]["budget"]==2048,"测速未使用2048记忆")
    must(len(meta["save_calls"]) == 1, "真实保存调用次数错误")
    save = meta["save_calls"][0]
    run_meta=json.loads((root/"run_meta.json").read_text())
    must(run_meta["argv"][1:]==meta["argv"],"训练实际argv与包装入口不同")
    must(Path(actual["checkpoint_dir"]).resolve()==Path(save["checkpoint_root"]).resolve()==Path(run_meta["checkpoint_dir"]).resolve(),"训练实际输出根与preflight不同")
    ckpt = Path(save["checkpoint_root"]) / "999"
    must(save["state_step"] == 1000 and (ckpt / "params").is_dir() and (ckpt / "assets").is_dir(), "末步checkpoint不完整")
    commit=json.loads((ckpt/"_CHECKPOINT_METADATA").read_text())
    must(commit["commit_timestamp_nsecs"] > commit["init_timestamp_nsecs"] > 0 and
         (ckpt/"params/manifest.ocdbt").is_file() and (ckpt/"params/_METADATA").is_file(),"缺checkpoint正式完成元数据")
    start, end = timing[99]["sync"]["wall_end"], timing[899]["sync"]["wall_end"]
    mu = (end-start)/800
    P = start-meta["start_wall"]
    B = save["return_wall"]-save["start_wall"]
    C = meta["end_wall"]-save["start_wall"]
    must(mu > 0 and P > 0 and C >= B >= 0, "计时边界顺序错误")
    steady = timing[100:900]
    host_mean = statistics.fmean(r["host_step_s"] for r in steady)
    slow = {r["step"] for r in steady if r["host_step_s"] > 1.5*host_mean}
    starts = [r["wall_start"] for r in steady]
    samples = gpu_rows(args.gpu)
    window = [r for r in samples if start <= r[0] <= end]
    per_gpu, coverage_ok = {}, True
    for gpu in range(8):
        full = sorted(r for r in samples if r[1] == gpu)
        vals = [r for r in window if r[1] == gpu]
        must(len(vals) >= 2, f"GPU{gpu}缺主窗采样")
        intervals = [b[0]-a[0] for a,b in zip(vals, vals[1:])]
        coverage = (vals[-1][0]-vals[0][0])/(end-start)
        gaps = max(intervals)
        count_ratio=len(vals)/((end-start)/.5)
        coverage_ok &= coverage >= .99 and count_ratio >= .90 and gaps <= 2 and statistics.fmean(intervals)<=.6 and full[0][0] <= start and full[-1][0] >= end
        groups = {"slow": [], "other": []}
        for row in vals:
            i = bisect.bisect_right(starts, row[0])-1
            if i >= 0 and row[0] <= steady[i]["wall_end"]:
                groups["slow" if steady[i]["step"] in slow else "other"].append(row[2])
        per_gpu[str(gpu)] = {"mean_pct": statistics.fmean(r[2] for r in vals),
            "zero_pct": 100*sum(r[2] == 0 for r in vals)/len(vals), "samples": len(vals),
            "memory_peak_mib": max(r[3] for r in full), "coverage": coverage,
            "sample_count_ratio":count_ratio,
            "interval_mean_s": statistics.fmean(intervals), "max_gap_s": gaps,
            "repeated_adjacent": sum(a[2] == b[2] for a,b in zip(vals, vals[1:])),
            **{key + "_mean_pct": statistics.fmean(v) if v else None for key,v in groups.items()},
            **{key + "_samples": len(v) for key,v in groups.items()}}
    hosts = read_rows(root / "host_samples.jsonl")
    must(hosts[0]["wall_time"]<=start and hosts[-1]["wall_time"]>=end,"主机采样未覆盖稳态窗口")
    host_window=[r for r in hosts if start<=r["wall_time"]<=end]
    must(len(host_window)>1,"主窗主机采样不足")
    io_samples=[r for r in host_window if "system_io" in r]
    must(len(io_samples)>=2,"缺稳态并发IO观察")
    io_coverage=io_samples[0]["wall_time"]-start<=15 and end-io_samples[-1]["wall_time"]<=15
    own_pids={pid for r in hosts for pid in r["rss_bytes"]}
    external={}
    for row in io_samples:
        for pid,record in row["system_io"].items():
            if pid in own_pids:continue
            state=external.setdefault(pid,{"name":record["name"],"first":record,"last":record})
            state["last"]=record
    external_io=[{"pid":pid,"name":r["name"],"read_bytes":max(0,r["last"]["read_bytes"]-r["first"]["read_bytes"]),
                  "write_bytes":max(0,r["last"]["write_bytes"]-r["first"]["write_bytes"])} for pid,r in external.items()]
    external_io=[r for r in external_io if r["read_bytes"] or r["write_bytes"]]
    # 预先固定复核阈值：外部物理IO超过1 GiB，不直接签发稳态报告。
    external_busy=sum(r["read_bytes"]+r["write_bytes"] for r in external_io)>2**30
    shm_ratio = max(r["shm_used"]/r["shm_capacity"] for r in hosts)
    must(shm_ratio <= .70, "共享内存峰值超过70%")
    blocks = []
    for lo in range(100,900,100):
        block = timing[lo:lo+100]
        wall = block[-1]["wall_end"]-block[0]["wall_start"]
        blocks.append({"first": lo, "last": lo+99, "host_mean_s": wall/100,
                       "data_wait_fraction": sum(r["phases_s"]["data_next"] for r in block)/wall})
    trend = statistics.fmean(r["host_step_s"] for r in timing[900:999])
    tail_sync_mean=(timing[999]["sync"]["wall_end"]-end)/100
    center = (P+79900*mu+15*B+C)/3600
    conservative = (P+79900*mu+16*C)/3600
    lo = (P+79900*min(x["host_mean_s"] for x in blocks)+15*B+C)/3600
    hi = (P+79900*max(x["host_mean_s"] for x in blocks)+16*C)/3600
    # 排除额外同步造成的首尾块偏移；内部块比较趋势，尾段另外使用已记录的双端同步。
    # 超过20%只触发报告不完整，绝不自动调参或添加逐步同步。
    drift = statistics.fmean(x["host_mean_s"] for x in blocks[-3:-1])/statistics.fmean(x["host_mean_s"] for x in blocks[1:3])
    ready = coverage_ok and io_coverage and not external_busy and .8 <= drift <= 1.2 and .8 <= tail_sync_mean/mu <= 1.2
    util_mean = statistics.fmean(r[2] for r in window)
    zero_pct = 100*sum(r[2] == 0 for r in window)/len(window)
    util_strata={}
    for group in ("slow","other"):
        count=sum(g[group+"_samples"] for g in per_gpu.values())
        value=sum(g[group+"_samples"]*(g[group+"_mean_pct"] or 0) for g in per_gpu.values())/count if count else None
        util_strata[group]={"mean_pct":value,"samples":count}
    report_data = {"status": "READY" if ready else "INCOMPLETE", "storage": "AWS本地NVMe RAID /dev/md0 XFS",
        "perf_head":meta["head"],"wrapper_sha256":meta["wrapper_sha256"],"argv":meta["argv"],
        "launch":launch_rows[0],"environment":environments[0],"loader":meta["loader"],
        "batch":128, "workers":16, "prefetch_factor":2, "warmup":[0,99], "steady":[100,899],
        "step_mean_s":mu, "samples_per_second":128/mu, "P_s":P, "B_s":B, "C_upper_s":C,
        "eta_hours":center, "conservative_save_hours":conservative, "scenario_hours":[lo,hi],
        "per_gpu":per_gpu, "gpu_mean_pct":util_mean, "gpu_zero_pct":zero_pct,"gpu_strata":util_strata,
        "shm_peak_ratio":shm_ratio, "blocks":blocks, "tail_host_mean_s":trend,"tail_sync_mean_s":tail_sync_mean,
        "drift_ratio":drift,"drift_blocks":{"early":[200,399],"late":[600,799]},
        "logical_image_gib_s":128/mu/128,
        "md0_read_bytes":(hosts[-1]["md0_read_sectors"]-hosts[0]["md0_read_sectors"])*512,
        "md0_steady_read_bytes":(host_window[-1]["md0_read_sectors"]-host_window[0]["md0_read_sectors"])*512,
        "external_io":external_io,"external_io_review_threshold_bytes":2**30,"io_coverage":io_coverage,
        "host_window_wall":[host_window[0]["wall_time"],host_window[-1]["wall_time"]],
        "cached_bytes_range":[min(r["Cached_bytes"] for r in host_window),max(r["Cached_bytes"] for r in host_window)],
        "memavailable_min_bytes":min(r["MemAvailable_bytes"] for r in hosts),
        "rss_peak_by_pid":{pid:max(r["rss_bytes"].get(pid,0) for r in hosts) for pid in {p for r in hosts for p in r["rss_bytes"]}},
        "limits":["80k为外推，没有实际运行80k。", "C包含训练入口收尾，是保存完成时间上界。",
                  "分块仅为主线程情景，不是独立GPU速度或置信区间。", "未复现周期写盘争用、长期缓存演化、W&B与外部负载。",
                  "NVML重复读数不增加独立证据；显存读数含JAX预分配，不等于活跃张量峰值。",
                  "设备计数包含同设备其他进程；约10秒的进程IO观察可能漏掉短命进程。"]}
    with Path(args.out).open("x") as f:
        json.dump(report_data,f,ensure_ascii=False,indent=2)
    print("SPEED_WINDOW=PASS steps=1000 warmup=0..99 steady=100..899 steady_steps=800 profiler=0")
    print(f"GPU_COVERAGE={'PASS' if coverage_ok else 'FAIL'}")
    print(f"SHM_PEAK=PASS ratio={shm_ratio:.6f}")
    print(f"GPU_UTIL mean={util_mean:.4f} zero_pct={zero_pct:.4f} slow_mean={util_strata['slow']['mean_pct']} other_mean={util_strata['other']['mean_pct']}")
    print(f"ETA80K=ESTIMATE hours={center:.4f} range_hours={lo:.4f},{hi:.4f}")
    print(f"SPEED_REPORT={report_data['status']}")
    return 0 if ready else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--records", required=True)
    r.add_argument("--mode", choices=("perf","smoke"), default="perf")
    r.add_argument("train_args", nargs=argparse.REMAINDER)
    r = sub.add_parser("report")
    for key in ("records", "gpu", "log", "out"):
        r.add_argument("--" + key, required=True)
    args = p.parse_args()
    if args.command == "run":
        return run(args)
    try:
        return report(args)
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(f"SPEED_REPORT=INCOMPLETE reason={error}",flush=True)
        if not Path(args.out).exists():
            with Path(args.out).open("x") as f:
                json.dump({"status":"INCOMPLETE","reason":str(error),"records":args.records},f,ensure_ascii=False,indent=2)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
