"""仅使用留档原始输入，复算训练结束时的数值、吞吐和 GPU 采样统计。"""

from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics


RECORDS = Path(__file__).resolve().parent
SCALARS = ("grad_norm", "llm_grad_norm", "loss", "mem_enc_norm", "param_norm")
EXPECTED_STEPS = list(range(0, 60000, 100))
EXPECTED_CHECKPOINTS = list(range(5000, 60000, 5000)) + [59999]
GPU_IDS = (4, 5, 6, 7)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def one_marker(text, name):
    values = re.findall(rf"^{name}=(.+)$", text, re.MULTILINE)
    require(len(values) == 1, f"{name} 必须恰好出现一次，实际为 {len(values)}")
    return values[0]


def gpu_summary(rows):
    require(bool(rows), "GPU 统计窗口没有采样")
    zeros = sum(row[2] == 0 for row in rows)
    return {
        "sample_count": len(rows),
        "util_mean_pct": statistics.fmean(row[2] for row in rows),
        "zero_sample_count": zeros,
        "zero_sample_share_pct": 100 * zeros / len(rows),
        "memory_used_max_MiB": max(row[3] for row in rows),
    }


def gpu_by_card(rows):
    return {
        str(gpu): gpu_summary([row for row in rows if row[1] == gpu])
        for gpu in GPU_IDS
    }


def main():
    # gzip 仅是无损存储容器；摘要同时记录压缩文件和解压后的日志字节。
    paths = [RECORDS / name for name in ("train.log.gz", "metrics.jsonl", "gpu_util_15s_full.csv")]
    with gzip.open(paths[0], "rb") as stream:
        log_bytes = stream.read()
    text = log_bytes.decode("utf-8").replace("\r", "\n")
    started = datetime.fromisoformat(one_marker(text, "START_UTC").replace("Z", "+00:00"))
    ended = datetime.fromisoformat(one_marker(text, "END_UTC").replace("Z", "+00:00"))
    exit_code = int(one_marker(text, "EXIT_CODE"))
    require(ended > started and exit_code == 0, "训练未正常结束或起止时刻顺序错误")

    metrics = [json.loads(line) for line in paths[1].read_text().splitlines() if line.strip()]
    require([row["step"] for row in metrics] == EXPECTED_STEPS, "指标必须完整覆盖 0、100、…、59900")
    for row in metrics:
        require(set(row) == {"step", "wall_time", *SCALARS}, f"step {row['step']} 指标键集合不符")
        require(math.isfinite(row["wall_time"]), "指标时间戳存在非有限值")
        for key in SCALARS:
            decimal = row[key]["dec"]
            require(math.isfinite(decimal), f"step {row['step']} 的 {key} 非有限")
            require(float.fromhex(row[key]["hex"]) == decimal, f"step {row['step']} 的 {key} 十进制与十六进制不一致")
    require(all(a["wall_time"] < b["wall_time"] for a, b in zip(metrics, metrics[1:])), "指标时间戳未严格递增")
    require(started.timestamp() <= metrics[0]["wall_time"] < metrics[-1]["wall_time"] <= ended.timestamp(), "指标时间戳超出训练起止范围")

    # step 0 是单步记录，其后每条是之前 100 步的聚合值；不解析 tqdm 的舍入步数。
    window = metrics[1:]
    t0, t1 = window[0]["wall_time"], window[-1]["wall_time"]
    updates = window[-1]["step"] - window[0]["step"]
    intervals = [(b["wall_time"] - a["wall_time"]) / 100 for a, b in zip(window, window[1:])]
    slow_threshold = 1.5 * statistics.median(intervals)
    last_ten = metrics[-10:]

    gpu_rows = []
    for line_number, fields in enumerate(csv.reader(io.StringIO(paths[2].read_text())), 1):
        require(len(fields) == 4, f"GPU CSV 第 {line_number} 行列数错误")
        stamp, gpu, util, memory = (field.strip() for field in fields)
        row = (datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp(),
               int(gpu), float(util), float(memory))
        require(row[1] in GPU_IDS and all(math.isfinite(value) for value in row), "GPU 采样卡号或数值错误")
        require(0 <= row[2] <= 100 and row[3] >= 0, "GPU 利用率或显存超出有效范围")
        gpu_rows.append(row)
    require(bool(gpu_rows), "GPU CSV 为空")
    require(all(a[0] <= b[0] for a, b in zip(gpu_rows, gpu_rows[1:])), "GPU 采样时间戳未递增")
    require(started.timestamp() <= gpu_rows[0][0] <= gpu_rows[-1][0] <= ended.timestamp(), "GPU 采样超出训练起止范围")
    counts = Counter(row[1] for row in gpu_rows)
    require(set(counts) == set(GPU_IDS) and len(set(counts.values())) == 1, "四卡全程采样数量不同或卡号缺失")
    window_gpu = [row for row in gpu_rows if t0 <= row[0] < t1]
    sample_intervals = {}
    for gpu in GPU_IDS:
        stamps = [row[0] for row in gpu_rows if row[1] == gpu]
        sample_intervals[str(gpu)] = statistics.median(b - a for a, b in zip(stamps, stamps[1:]))

    # 保存完成标志只能证明日志中的完成报告；权重完整性由独立的 CPU restore 验收。
    saved = [int(step) for step in re.findall(r"Finished asynchronous save [^\n]+/([0-9]+)\s+\(", text)]
    require(saved == EXPECTED_CHECKPOINTS, "日志中 12 个 checkpoint 异步保存完成标志不齐")
    errors = [line for line in text.splitlines()
              if re.search(r"\[E\]|\[W\]|Traceback|RESOURCE_EXHAUSTED|out of memory|(?<![A-Za-z_])(?:nan|inf|NaN|Infinity)(?![A-Za-z_])", line)]
    require(not errors, "日志包含错误、警告或非有限数值模式，请人工复核")
    clocks = [match[1] for line in text.splitlines() if (match := re.match(r"^(\d{2}:\d{2}:\d{2}\.\d+)", line))]
    clock_reversals = [{"before": a, "after": b} for a, b in zip(clocks, clocks[1:]) if b < a]
    milestone_steps = {0, 100, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 59000, 59900}
    result = {
        "run": "v2-1600ep-m8x8-modul-b128-60k",
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "decompressed_train_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "start_utc": started.isoformat(),
        "end_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "duration": str(ended - started),
        "exit_code": exit_code,
        "metrics": {
            "count": len(metrics),
            "steps": {"first": 0, "last": 59900, "interval": 100, "complete": True},
            "all_five_scalars_finite": True,
            "hex_matches_decimal": True,
            "timestamps_strictly_increasing": True,
            "first_log_utc": utc_iso(metrics[0]["wall_time"]),
            "last_log_utc": utc_iso(metrics[-1]["wall_time"]),
            "milestones": [{"step": row["step"], **{key: row[key]["dec"] for key in SCALARS}}
                           for row in metrics if row["step"] in milestone_steps],
            "last_ten_logs": {
                "steps": [row["step"] for row in last_ten],
                "loss_mean": statistics.fmean(row["loss"]["dec"] for row in last_ten),
                "covered_training_step_range_inclusive": [58901, 59900],
                "note": "十条日志各聚合 100 步，合计 1000 步；训练最后 59901–59999 共 99 步未触发日志，不能称为训练最后 1000 步的精确均值。",
            },
        },
        "performance_window": {
            "log_step_range": [100, 59900],
            "start_utc": utc_iso(t0),
            "end_utc": utc_iso(t1),
            "duration_seconds": t1 - t0,
            "updates": updates,
            "mean_seconds_per_update": (t1 - t0) / updates,
            "batch_size": 128,
            "samples_per_second": 128 * updates / (t1 - t0),
            "interval_count": len(intervals),
            "steps_per_interval": 100,
            "min_interval_seconds_per_update": min(intervals),
            "max_interval_seconds_per_update": max(intervals),
            "slow_interval_threshold_seconds_per_update": slow_threshold,
            "slow_interval_count": sum(value > slow_threshold for value in intervals),
            "note": "使用 metrics 的精确 step/wall_time，统计 598 个 100 步间隔；包含窗口内数据等待与 checkpoint 开销，不是单步延时分布。慢区间阈值仅按区间步时中位数的 1.5 倍定义，不作逐步慢步判定。",
        },
        "gpu_15s": {
            "first_sample_utc": utc_iso(gpu_rows[0][0]),
            "last_sample_utc": utc_iso(gpu_rows[-1][0]),
            "full_run": gpu_summary(gpu_rows),
            "full_run_per_gpu": gpu_by_card(gpu_rows),
            "metrics_window": gpu_summary(window_gpu),
            "metrics_window_per_gpu": gpu_by_card(window_gpu),
            "per_gpu_median_actual_sampling_interval_seconds": sample_intervals,
            "note": "全程 15 秒离散采样的等权均值与 0% 样本占比只能作为采样估计；采样间隔大于训练步时，不能据此逐步定位等待或证明 GPU 吃满。显存值为驱动报告的设备显存占用，包含 JAX 预分配，不能等同模型实际工作集。",
        },
        "checkpoint_log": {"completed_steps": saved, "count": len(saved), "final_step": saved[-1],
                           "note": "仅核对完整日志中异步保存完成标志；checkpoint 文件与 61/61 参数树的 CPU restore 另行验收。"},
        "log_checks": {"error_pattern_matches": errors, "clock_reversals": clock_reversals,
                       "note": "日志显示时钟不带日期，跨午夜会回绕；本脚本的耗时只使用 START_UTC/END_UTC 与 metrics epoch 时间戳，不解析 tqdm 的 k 单位舍入进度。"},
    }
    output = RECORDS / "final_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("FINAL_ANALYSIS=PASS metrics=600 scalars=3000 checkpoints=12")
    print(json.dumps({"duration": result["duration"], "last_ten_logs_loss_mean": result["metrics"]["last_ten_logs"]["loss_mean"],
                      "performance_window": result["performance_window"], "gpu_15s": result["gpu_15s"],
                      "log_checks": result["log_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
