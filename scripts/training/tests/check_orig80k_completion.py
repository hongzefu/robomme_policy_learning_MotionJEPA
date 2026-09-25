"""核对原版三模式的真实末步、尾窗、保存完成和恢复数组，不启动策略评估。"""
# 独立脚本必须先把训练工具目录加入导入路径。
# ruff: noqa: E402

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/training"))
from config_record import file_record
from final_record import SCALARS
from final_record import summarize_tree
from orig80k_contract import CONFIG
from orig80k_contract import MODE_STEPS
from orig80k_contract import require
from orig80k_contract import validate_argv


def load(path):
    return json.loads(Path(path).read_text())


def expected_checkpoints(mode):
    steps = MODE_STEPS[mode]
    return sorted({*range(10000, steps, 10000), steps - 1})


def validate_scalar(value, *, finite_flag=False):
    require(isinstance(value, dict), "标量记录格式错误")
    if finite_flag:
        require(value.get("finite") is True, "尾窗缺有限值证据或存在非有限值")
    number = value.get("dec")
    require(type(number) in (int, float) and math.isfinite(number)
            and float(number).hex() == value.get("hex"), "标量非有限或十进制/十六进制内容不同")


def validate_records(start, final, wait, metrics, exits, names, *, mode, run, head):
    require(mode in MODE_STEPS and re.fullmatch(r"[0-9a-f]{40}", head), "模式或完整 HEAD 非法")
    steps = MODE_STEPS[mode]
    identity = {"run_name": run, "head": head, "budget": 512, "num_train_steps": steps,
                "log_interval": 100, "save_interval": 10000, "keep_period": 10000}
    require(all(start.get(key) == final.get(key) == value for key, value in identity.items()), "运行身份或配置不同")
    uuid.UUID(start["run_uuid"])
    require(start["run_uuid"] == final["run_uuid"] == wait["run_uuid"], "运行 UUID 不同")
    require(bool(start.get("wandb_run_id")) and start["wandb_run_id"] == final.get("wandb_run_id"), "W&B 运行身份错误")
    require(start["complete"] == final["complete"], "起跑与末步完整配置不同")
    fields = start["complete"]["train_config"]["fields"]
    require(all(fields.get(key) == value for key, value in {
        "name": CONFIG, "exp_name": run, "num_train_steps": steps, "batch_size": 64,
        "num_workers": 4, "fsdp_devices": 4, "seed": 42, "log_interval": 100,
        "save_interval": 10000, "keep_period": 10000, "ema_decay": .999, "wandb_enabled": True,
        "overwrite": False, "resume": False}.items()), "记录的真实配置不是本模式原版四卡配置")
    require(wait.get("head") == head and wait.get("wait_until_finished") is True, "缺少异步保存完成记录")
    require(exits == ["EXIT_CODE=0"], "必须有唯一成功终态")
    require(names == expected_checkpoints(mode), "checkpoint 集合有缺失或多余项")
    require(final.get("state_step") == steps
            and final.get("loop_step") == final.get("checkpoint_step") == steps - 1, "末步实际更新次数错误")
    require([row.get("step") for row in metrics] == list(range(0, steps, 100)), "普通日志缺步、重复或乱序")
    for row in metrics:
        require(set(row) - {"step", "wall_time"} == SCALARS, "普通日志缺完整五标量")
        for key in SCALARS:
            validate_scalar(row[key])
    first = (steps - 1) // 100 * 100 + 1
    tail = final["tail"]
    require([row.get("step") for row in tail] == list(range(first, steps)), "最终尾窗缺步、重复或乱序")
    require(set(final["tail_means"]) == SCALARS, "尾窗均值缺字段")
    for row in tail:
        require(set(row["scalars"]) == SCALARS, "尾窗缺完整五标量")
        for value in row["scalars"].values():
            validate_scalar(value, finite_flag=True)
    for key in SCALARS:
        expected = sum(row["scalars"][key]["dec"] for row in tail) / len(tail) if tail else None
        require(final["tail_means"][key] == expected, "尾窗均值与逐步记录不同")
    leaves = final["ema_leaves"]
    require(bool(leaves) and all(row.get("finite") is True and row.get("dtype") and isinstance(row.get("shape"), list)
            and type(row.get("bytes")) is int and row["bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) for row in leaves.values()),
            "末步 EMA 缺摘要、dtype/shape 或有限值证据")


def validate_arrays(final, restored):
    observed = summarize_tree(restored)
    require(bool(observed) and all(row["finite"] for row in observed.values()), "真实恢复权重含非有限值或叶集为空")
    require(observed == final["ema_leaves"], "真实恢复权重与本次末步 EMA 的 dtype/shape/字节摘要不同")
    return observed


def restore_arrays(final, params_path):
    """使用生产恢复器保留原始 dtype，不用强制转换掩盖保存差异。"""
    import numpy as np

    from openpi.models.model import restore_params

    restored = restore_params(params_path, restore_type=np.ndarray, dtype=None)
    return validate_arrays(final, restored)


def require_process_gone(pid):
    require(type(pid) is int and pid > 1, "记录的进程 PID 非法")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise ValueError(f"本轮进程仍存在，尚不能完成验收: {pid}")


def validate_gpu_sampling(records, lines, gpu_ids, start, wait):
    """采样必须连续覆盖运行现场；停止日志不能把提前退出重命名为主动停止。"""
    from check_modul_speed import gpu_rows

    records = Path(records)
    stops = [line for line in lines if line.startswith("GPU_SAMPLER_STOP ")]
    require(len(stops) == 1, "GPU 采样未唯一收尾")
    match = re.fullmatch(r"GPU_SAMPLER_STOP pid=(\d+) exit_code=(\d+) reason=([a-z_]+)", stops[0])
    require(match is not None and match[3] == "driver_stop" and int(match[2]) in (0, 143),
            "GPU 采样提前退出或停止状态异常")
    require([line for line in lines if line.startswith("GPU_SAMPLER_PID=")] == ["GPU_SAMPLER_PID=" + match[1]],
            "GPU 采样停止的 PID 与启动 PID 不同")
    require([line for line in lines if line.startswith("GPU_SAMPLER_FINAL_SAMPLE ")] ==
            ["GPU_SAMPLER_FINAL_SAMPLE exit_code=0"], "缺少成功的运行终点 GPU 采样")
    error_path = records / "gpu.csv.err"
    require(error_path.is_file() and not error_path.read_text().strip(), "GPU 采样错误文件缺失或含错误")
    samples = gpu_rows(records / "gpu.csv")
    require(len(gpu_ids) == len(set(gpu_ids)) == 4 and {row[1] for row in samples} == set(gpu_ids),
            "GPU 采样缺卡、多卡或为空")
    times = [datetime.datetime.fromisoformat(value) for value in (start["started_at"], wait["completed_at"])]
    require(all(value.tzinfo is not None for value in times), "运行窗口缺时区")
    begin, end = [value.timestamp() for value in times]
    require(begin < end, "运行窗口开始与完成时刻无效")
    summary = {}
    for gpu in gpu_ids:
        device = [row for row in samples if row[1] == gpu]
        require(device[0][0] <= begin and device[-1][0] >= end, f"GPU{gpu}采样未覆盖真实运行窗口")
        # 沿用测速器的 500ms 密度判据，同时纳入跨越窗口边界的相邻间隔。
        intervals = [right[0] - left[0] for left, right in itertools.pairwise(device)
                     if right[0] >= begin and left[0] <= end]
        window = [row for row in device if begin <= row[0] <= end]
        require(intervals and max(intervals) <= 2 and statistics.fmean(intervals) <= .6
                and len(window) >= .9 * (end - begin) / .5, f"GPU{gpu}采样中断或密度不足")
        summary[str(gpu)] = {"samples": len(window), "max_gap_s": max(intervals),
                             "mean_interval_s": statistics.fmean(intervals)}
    return {"training_window": [begin, end], "gpu": summary}


def check(args):
    records = Path(args.records)
    run_root = Path(args.run_root)
    require(records.is_absolute() and run_root.is_absolute(), "记录与 checkpoint 必须使用绝对路径")
    require(records == records.resolve() and run_root == run_root.resolve(), "记录或权重路径包含符号链接")
    require(records == ROOT / "v1-store/bench/orig80k" / args.run
            and run_root == ROOT / "v1-store/train-runs" / CONFIG / args.run, "目录与本 run 不符")
    launch = load(records / "launch.json")
    start, final, wait = [load(records / "final" / name)
                          for name in ("start.json", "final.json", "checkpoint_wait_done.json")]
    require((launch.get("mode"), launch.get("run_name"), launch.get("head")) == (args.mode, args.run, args.head),
            "启动契约与待验收运行不同")
    validate_argv(launch["argv"], args.mode)
    require(launch["actual"]["complete"] == start["complete"], "真实训练配置与启动契约不同")
    require(launch["actual"].get("jax_enable_x64") is False, "启动记录缺 x64 禁用证据")
    require(all(Path(item["checkpoint_dir"]) == run_root for item in (launch, start, final)), "末步或启动记录指向另一输出根")
    runtime = load(records / "runtime.json")
    require((runtime.get("device_count"), runtime.get("fsdp_devices"), runtime.get("batch_size"), runtime.get("num_workers")) ==
            (4, 4, 64, 4), "真实设备或 loader 形制不同")
    require(runtime.get("exp_name") == args.run and runtime.get("config_name") == CONFIG and runtime.get("seed") == 42,
            "真实 runtime 身份或种子不同")
    require(runtime.get("cuda_visible_devices") == launch["environment"]["CUDA_VISIBLE_DEVICES"], "真实 GPU 分组与启动记录不同")
    metrics_path = records / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    log = Path(args.log)
    require(log.resolve() == ROOT / "v1-store/logs" / f"{args.run}.driver.log", "日志不是本次 driver 日志")
    lines = log.read_text().splitlines()
    exits = [line for line in lines if line.startswith("EXIT_CODE=")]
    for prefix in ("TRAIN_PIPE_EXIT=", "TEE_EXIT="):
        require([line for line in lines if line.startswith(prefix)] == [prefix + "0"], "训练或 tee 未成功收尾")
    for prefix in ("GPU_SAMPLER_PID=", "TRAIN_WRAPPER_PID="):
        pids = [int(line.removeprefix(prefix)) for line in lines if line.startswith(prefix)]
        require(len(pids) == 1, "缺唯一的训练或 GPU 采样 PID")
        require_process_gone(pids[0])
    gpu_sampling = validate_gpu_sampling(records, lines,
        [int(value) for value in runtime["cuda_visible_devices"].split(",")], start, wait)
    require_process_gone(start["pid"])
    numeric = [path for path in run_root.iterdir() if path.name.isdecimal()]
    require(all(path.is_dir() and not path.is_symlink() and str(int(path.name)) == path.name for path in numeric),
            "checkpoint 数字路径不是规范实体目录")
    names = sorted(int(path.name) for path in numeric)
    require(not any("orbax-checkpoint-tmp" in path.name for path in run_root.iterdir()), "存在未完成的临时 checkpoint")
    validate_records(start, final, wait, metrics, exits, names, mode=args.mode, run=args.run, head=args.head)
    for step in names:
        checkpoint = run_root / str(step)
        metadata = load(checkpoint / "_CHECKPOINT_METADATA")
        require(metadata["commit_timestamp_nsecs"] > metadata["init_timestamp_nsecs"] > 0, "checkpoint 未提交完成")
        require(all((checkpoint / suffix).is_file() for suffix in
                    ("params/manifest.ocdbt", "params/_METADATA", "assets/robomme/norm_stats.json")), "checkpoint 文件不完整")
        saved_norm = load(checkpoint / "assets/robomme/norm_stats.json")
        original_norm = load(start["complete"]["assets"]["norm_stats"]["path"])
        require(saved_norm == original_norm, "保存的 norm_stats 内容与启动资产不同")
    checkpoint = run_root / str(MODE_STEPS[args.mode] - 1)
    observed = restore_arrays(final, checkpoint / "params")
    files = [records / name for name in ("launch.json", "runtime.json", "metrics.jsonl", "gpu.csv", "gpu.csv.err")]
    files += [records / "final" / name for name in ("start.json", "final.json", "checkpoint_wait_done.json")]
    files += [log, checkpoint / "_CHECKPOINT_METADATA", checkpoint / "params/_METADATA"]
    result = {"status": "PASS", "mode": args.mode, "run_name": args.run, "head": args.head,
              "run_uuid": start["run_uuid"], "state_step": MODE_STEPS[args.mode], "checkpoints": names,
              "final": MODE_STEPS[args.mode] - 1, "restored_leaves": observed, "gpu_sampling": gpu_sampling,
              "files": [file_record(path) for path in files]}
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"RUN_COMPLETED=PASS run={args.run} checkpoints={len(names)} final={result['final']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODE_STEPS)
    for name in ("records", "run-root", "log", "run", "head", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        check(args)
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"RUN_COMPLETED=FAIL reason={error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
