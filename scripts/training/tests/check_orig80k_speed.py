"""原版四卡300步测速：进程内轻量计时、真实保存与独立报告。"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import hashlib
import itertools
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

from check_modul_speed import gpu_rows
from check_modul_speed import host_sample

ROOT = Path(__file__).resolve().parents[3]
STEPS = 300
WARMUP = 100
SCALARS = {"loss", "grad_norm", "llm_grad_norm", "mem_enc_norm", "param_norm"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class HostTiming:
    """只在预热末步与末次真实保存前同步，不开启profiler。"""

    def __init__(self, root, steps):
        require(steps in (20, STEPS), "计时只允许20步包装验证或300步正式测速")
        self.steps = steps
        self.file = (Path(root) / "step_timing.jsonl").open("x")
        self.count = 0
        self.row = None
        self.flush = None
        self.closed = False

    def synchronize(self, location):
        require(self.flush is not None, "缺少设备完成回调")
        start = time.perf_counter()
        self.flush()
        self.row["sync"] = {"location": location, "wall_end": time.time(),
                            "seconds": time.perf_counter() - start}

    @contextlib.contextmanager
    def step(self, step, *, flush=None):
        require(not self.closed and self.row is None and step == self.count < self.steps,
                "计时步骤重复、乱序或越界")
        self.flush = flush
        self.row = {"step": int(step), "wall_start": time.time(), "phases_s": {}}
        start = time.perf_counter()
        completed = False
        try:
            yield
            if step == WARMUP - 1:
                self.synchronize("warmup_end")
            if step == self.steps - 1:
                require(self.row.get("sync", {}).get("location") == "before_final_save",
                        "末步未经过真实保存计时")
            completed = True
        finally:
            self.row.update(completed=completed, host_step_s=time.perf_counter() - start,
                            wall_end=time.time())
            self.file.write(json.dumps(self.row, allow_nan=False) + "\n")
            self.file.flush()
            self.row = None
            self.flush = None
            self.count += 1

    @contextlib.contextmanager
    def phase(self, name):
        require(self.row is not None, "计时阶段没有所属步骤")
        require(name not in self.row["phases_s"], "同一步计时阶段重复")
        if name == "checkpoint":
            require(self.row["step"] == self.steps - 1, f"观测只能保存末步{self.steps - 1}")
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
            print(f"ORIG80K_TIMING_DONE steps={self.count} profiler=0", flush=True)


def sample_host(path, stopped, metadata):
    try:
        with Path(path).open("x") as stream:
            while True:
                stream.write(json.dumps(host_sample(os.getpid()), allow_nan=False) + "\n")
                stream.flush()
                if stopped.wait(0.5):
                    break
    except Exception as error:
        metadata["sampling_error"] = repr(error)


def run(args):
    root = ROOT.resolve()
    records = Path(args.records).resolve()
    require(records.is_relative_to(root / "v1-store"), "测速记录必须位于本仓库v1-store")
    require(not os.environ.get("XLA_FLAGS"), "测速必须清除验证用XLA_FLAGS")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            "测速必须从clean HEAD启动")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(os.environ.get("MMEVLA_EXPECTED_TRAIN_HEAD") == head, "测速HEAD与已通过preflight的锚点不同")
    argv = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    sys.path.insert(0, str(root / "scripts/training"))
    from orig80k_contract import validate_argv

    mode = args.mode
    require(mode in ("perf", "smoke"), "观测入口只允许perf或20步smoke验证")
    expected_steps = STEPS if mode == "perf" else 20
    validate_argv(argv, mode)
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(len(gpu_ids) == 4 and len(set(gpu_ids)) == 4 and all(x.isdecimal() for x in gpu_ids),
            "测速必须指定四个不同物理GPU编号")
    owned = ("speed_start.json", "speed_run.json", "step_timing.jsonl", "host_samples.jsonl")
    require(not any((records / name).exists() for name in owned), "拒绝覆盖既有测速记录")
    records.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": 1, "mode": mode, "steps": expected_steps, "head": head, "argv": argv,
                "gpu_ids": [int(x) for x in gpu_ids], "pid": os.getpid(),
                "start_wall": time.time(), "profiler": False, "entry_calls": 0, "save_calls": [],
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "timing_difference": "轻量阶段记录；若到达step99则在其完成时同步，最后一步保存前同步"}
    write_json(records / "speed_start.json", metadata)
    import jax
    import step_timing

    from mme_vla_suite.training import dataloader
    from openpi.training import checkpoints

    original_timer = step_timing.StepTiming
    original_trace = jax.profiler.start_trace
    original_save = checkpoints.save_state
    original_loader = dataloader.create_data_loader
    previous_argv = sys.argv
    previous_env = {key: os.environ.get(key) for key in ("TRAIN_RECORD_DIR", "TRAIN_TIMING_STEPS")}
    stopped = threading.Event()
    thread = threading.Thread(target=sample_host,
                              args=(records / "host_samples.jsonl", stopped, metadata), daemon=True)

    def forbidden_trace(*a, **kw):
        raise RuntimeError("原版测速禁止开启profiler")

    def observed_save(manager, state, loader, step):
        require(step == expected_steps - 1 and not metadata["save_calls"], "真实保存步骤或次数错误")
        record = {"step": int(step), "state_step": int(state.step), "start_wall": time.time(),
                  "checkpoint_root": str(manager.directory)}
        result = original_save(manager, state, loader, step)
        record["return_wall"] = time.time()
        metadata["save_calls"].append(record)
        return result

    def observed_loader(*a, **kw):
        result = original_loader(*a, **kw)
        loader = result._data_loader.torch_loader  # noqa: SLF001 -- 只读核对真实训练loader形制
        metadata["loader"] = {"batch_size": loader.batch_size, "workers": loader.num_workers,
                              "prefetch_factor": loader.prefetch_factor,
                              "persistent_workers": loader.persistent_workers}
        require(loader.batch_size == 64 and loader.num_workers == 4, "实际loader不是b64/workers4")
        return result

    try:
        step_timing.StepTiming = HostTiming
        jax.profiler.start_trace = forbidden_trace
        checkpoints.save_state = observed_save
        dataloader.create_data_loader = observed_loader
        os.environ["TRAIN_RECORD_DIR"] = str(records)
        # 复用已有阶段钩子，但在入口加载前替换计时实现，绝不调用原profiler。
        os.environ["TRAIN_TIMING_STEPS"] = str(expected_steps)
        thread.start()
        sys.argv = [str(root / "scripts/training/train.py"), *argv]
        metadata["entry_calls"] += 1
        runpy.run_path(sys.argv[0], run_name="__main__")
        metadata["success"] = True
    finally:
        metadata["end_wall"] = time.time()
        stopped.set()
        if thread.ident is not None:
            thread.join(timeout=5)
        metadata["sampler_stopped"] = not thread.is_alive()
        sys.argv = previous_argv
        step_timing.StepTiming = original_timer
        jax.profiler.start_trace = original_trace
        checkpoints.save_state = original_save
        dataloader.create_data_loader = original_loader
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        write_json(records / "speed_run.json", metadata)


def summarize(records, gpu_path):
    root = Path(records)
    meta = json.loads((root / "speed_run.json").read_text())
    require(meta.get("mode") == "perf" and meta.get("steps") == STEPS, "20步包装验证不能用于300步测速报告")
    require(meta.get("success") and meta.get("entry_calls") == 1 and meta.get("sampler_stopped")
            and not meta.get("sampling_error"), "训练或主机采样未正常完成")
    require(meta.get("profiler") is False and not (root / "step_trace").exists(), "存在profiler记录")
    require(meta["loader"]["batch_size"] == 64 and meta["loader"]["workers"] == 4, "实际loader档位错误")
    timing = rows(root / "step_timing.jsonl")
    require([row["step"] for row in timing] == list(range(STEPS))
            and all(row.get("completed") for row in timing), "300步计时缺失、重复或失败")
    require([(row["step"], row["sync"]["location"]) for row in timing if "sync" in row]
            == [(99, "warmup_end"), (299, "before_final_save")], "同步边界错误")
    metrics = rows(root / "metrics.jsonl")
    require([row["step"] for row in metrics] == [0, 100, 200], "log100步集合错误")
    for row in metrics:
        require(set(row) - {"step", "wall_time"} == SCALARS, "五标量字段不完整")
        require(all(math.isfinite(row[key]["dec"])
                    and float(row[key]["dec"]).hex() == row[key]["hex"] for key in SCALARS),
                "标量非有限或十进制/十六进制不一致")
    require(len(meta["save_calls"]) == 1, "末次保存调用次数错误")
    save = meta["save_calls"][0]
    require(save["step"] == 299 and save["state_step"] == 300, "末步状态计数错误")
    start = timing[99]["sync"]["wall_end"]
    end = timing[299]["sync"]["wall_end"]
    require(meta["start_wall"] < start < end <= save["start_wall"]
            <= save["return_wall"] <= meta["end_wall"], "测速时间边界顺序错误")
    steady = timing[100:300]
    durations = [row["host_step_s"] for row in steady]
    require(all(math.isfinite(x) and x > 0 for x in durations), "单步主机耗时非法")
    # 最后一步保存时间不属于稳态，分类使用保存前的主机区间。
    durations[-1] = end - steady[-1]["wall_start"]
    threshold = statistics.median(durations) * 2
    slow = {row["step"] for row, duration in zip(steady, durations, strict=True) if duration > threshold}
    starts = [row["wall_start"] for row in steady]
    samples = gpu_rows(gpu_path)
    expected_gpus = meta["gpu_ids"]
    require(len(expected_gpus) == len(set(expected_gpus)) == 4, "GPU身份记录错误")
    require({row[1] for row in samples} == set(expected_gpus), "GPU采样与本run设备不一致")
    per_gpu = {}
    for gpu in expected_gpus:
        all_values = [row for row in samples if row[1] == gpu]
        values = [row for row in all_values if start <= row[0] <= end]
        require(len(values) > 1 and all_values[0][0] <= start and all_values[-1][0] >= end,
                f"GPU{gpu}采样没有覆盖稳态窗口")
        intervals = [b[0] - a[0] for a, b in itertools.pairwise(values)]
        require(max(intervals) <= 2 and statistics.fmean(intervals) <= .6
                and len(values) >= .9 * (end - start) / .5, f"GPU{gpu}采样密度不足")
        groups = {"slow": [], "other": []}
        for stamp, _, util, _ in values:
            index = bisect.bisect_right(starts, stamp) - 1
            if index >= 0 and stamp <= min(steady[index]["wall_end"], end):
                groups["slow" if steady[index]["step"] in slow else "other"].append(util)
        per_gpu[str(gpu)] = {
            "mean_pct": statistics.fmean(row[2] for row in values),
            "zero_pct": 100 * sum(row[2] == 0 for row in values) / len(values),
            "samples": len(values), "mean_interval_s": statistics.fmean(intervals),
            "max_gap_s": max(intervals), "memory_peak_mib": max(row[3] for row in all_values),
            "same_adjacent_readings": sum(a[2] == b[2] for a, b in itertools.pairwise(values)),
            **{f"{key}_mean_pct": statistics.fmean(value) if value else None for key, value in groups.items()},
            **{f"{key}_samples": len(value) for key, value in groups.items()},
        }
    hosts = rows(root / "host_samples.jsonl")
    require(len(hosts) > 1 and hosts[0]["wall_time"] <= start and hosts[-1]["wall_time"] >= end,
            "主机采样未覆盖稳态窗口")
    return {"head": meta["head"], "gpu_ids": expected_gpus, "batch_size": 64, "workers": 4,
            "storage": "AWS本地NVMe RAID /dev/md0", "warmup_steps": [0, 99],
            "steady_steps": [100, 299], "steady_wall": [start, end], "steady_seconds": end - start,
            "samples_per_second": 200 * 64 / (end - start), "mean_step_s": (end - start) / 200,
            "initialization_and_warmup_s": start - meta["start_wall"],
            "save_dispatch_s": save["return_wall"] - save["start_wall"],
            "save_and_finish_s": meta["end_wall"] - save["start_wall"],
            "estimated_80k_eta_s": max(0, start - meta["start_wall"] - (end - start) / 2)
                + 80000 * (end - start) / 200 + 8 * (meta["end_wall"] - save["start_wall"]),
            "eta_note": "初始化余量+80000稳态步+8次保存；保存成本由一次末步实测外推",
            "slow_step_threshold_s": threshold, "slow_steps": sorted(slow), "gpu": per_gpu,
            "gpu_sampling_interval_s": .5,
            "host_rss_sum_peak_bytes": max(sum(row["rss_bytes"].values()) for row in hosts),
            "host_rss_note": "各进程RSS之和含共享页重复计数，不是物理独占内存",
            "shm_peak_bytes": max(row["shm_used"] for row in hosts),
            "mem_available_min_bytes": min(row["MemAvailable_bytes"] for row in hosts),
            "checkpoint_validation": "另由check_orig80k_completion.py执行真实恢复验收"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--records", required=True)
    run_parser.add_argument("--mode", choices=("perf", "smoke"), default="perf",
                            help="smoke仅供20步包装开关对照，不能用于测速结论")
    run_parser.add_argument("train_args", nargs=argparse.REMAINDER)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--records", required=True)
    report_parser.add_argument("--gpu", required=True)
    report_parser.add_argument("--peer-records")
    report_parser.add_argument("--peer-gpu")
    report_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
        return
    result = summarize(args.records, args.gpu)
    require(bool(args.peer_records) == bool(args.peer_gpu), "并跑报告的peer参数必须成对提供")
    if args.peer_records:
        peer = summarize(args.peer_records, args.peer_gpu)
        require(peer["head"] == result["head"] and not set(peer["gpu_ids"]) & set(result["gpu_ids"]),
                "并跑两侧版本不同或GPU重叠")
        start = max(result["steady_wall"][0], peer["steady_wall"][0])
        end = min(result["steady_wall"][1], peer["steady_wall"][1])
        require(end > start, "两侧稳态窗口没有实际重叠")
        span = max(result["steady_wall"][1], peer["steady_wall"][1]) - min(result["steady_wall"][0], peer["steady_wall"][0])
        result = {"full_or_first": result, "counting_or_second": peer,
                  "steady_overlap_seconds": end - start, "combined_steady_span_seconds": span,
                  "combined_samples_per_second_over_span": 2 * 200 * 64 / span}
    write_json(args.out, result)
    print(f"ORIG80K_SPEED=PASS out={args.out}")


if __name__ == "__main__":
    main()
