"""保留完整主线程/NVML统计，并独立核对原始XPlane的真实设备覆盖。"""

import bisect
import collections
import csv
import datetime
import json
import mmap
import pathlib
import statistics

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

ROOT = pathlib.Path(__file__).resolve().parents[1] / "bench/v2-1600ep-m8x8-modul-motion-b128-80k"
UTC = datetime.timezone.utc


def fields(data, start=0, end=None):
    """只遍历protobuf顶层字段，跳过无须解析的大型主机事件。"""
    end = len(data) if end is None else end
    offset = start

    def number():
        nonlocal offset
        value = 0
        for shift in range(0, 70, 7):
            if offset >= end:
                raise ValueError("截断varint")
            byte = data[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("varint过长")

    while offset < end:
        tag = number()
        field, wire = tag >> 3, tag & 7
        if wire == 2:
            length = number()
            begin = offset
            offset += length
            if offset > end:
                raise ValueError("截断字段")
            yield field, begin, offset
        elif wire == 0:
            number()
        elif wire in (1, 5):
            offset += 8 if wire == 1 else 4
        else:
            raise ValueError(("未知wire", wire))


def raw_coverage():
    # 字段定义：https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto
    descriptor = descriptor_pb2.FileDescriptorProto(name="mv2_gpu_subset.proto", package="mv2", syntax="proto3")
    for name, definitions in (
        ("Event", [("offset_ps", 2, 3, False, None), ("duration_ps", 3, 3, False, None)]),
        ("Line", [("name", 2, 9, False, None), ("timestamp_ns", 3, 3, False, None), ("events", 4, 11, True, "Event")]),
        ("Plane", [("name", 2, 9, False, None), ("lines", 3, 11, True, "Line")]),
    ):
        message = descriptor.message_type.add(name=name)
        for key, number, kind, repeated, child in definitions:
            field = message.field.add(name=key, number=number, type=kind, label=3 if repeated else 1)
            if child:
                field.type_name = ".mv2." + child
    pool = descriptor_pool.DescriptorPool()
    pool.Add(descriptor)
    plane_type = message_factory.GetMessageClass(pool.FindMessageTypeByName("mv2.Plane"))
    diagnostic = json.loads((ROOT / "trace_coverage_diagnostic.json").read_text())
    lo = diagnostic["steps"]["100"]["start"]
    hi = diagnostic["steps"]["299"]["end"]
    early_lo = diagnostic["steps"]["5"]["start"]
    early_hi = diagnostic["steps"]["24"]["end"]
    source, = (ROOT / "step_trace").rglob("*.xplane.pb")
    output = {}
    with source.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for field, begin, end in fields(data):
            if field != 1:
                continue
            name = next(data[a:b].decode() for key, a, b in fields(data, begin, end) if key == 2)
            if not name.startswith("/device:GPU:"):
                continue
            plane = plane_type()
            plane.ParseFromString(data[begin:end])
            plane.DiscardUnknownFields()
            count = in_window = early_count = 0
            first, last = float("inf"), 0.0
            for line in plane.lines:
                if not line.name.startswith("Stream #"):
                    continue
                for event in line.events:
                    start = line.timestamp_ns / 1000 + event.offset_ps / 1e6
                    finish = start + event.duration_ps / 1e6
                    count += 1
                    first, last = min(first, start), max(last, finish)
                    in_window += finish > lo and start < hi
                    early_count += finish > early_lo and start < early_hi
            json_streams = [v for k, v in diagnostic["gpu_streams"].items() if k.startswith(name + " ")]
            assert count == sum(v["count"] for v in json_streams)
            assert abs(last - max(v["end"] for v in json_streams)) < 0.001
            output[name] = {"events": count, "start_relative_us": first, "end_relative_us": last,
                            "events_in_step100_299": in_window, "events_in_step5_24": early_count,
                            "covers_early_window_bounds": first <= early_lo and last >= early_hi,
                            "matches_json_count_and_endpoint": True}
    assert len(output) == 8
    return {"source": str(source), "gpu_planes": output,
            "all_gpu_physical_events": sum(x["events"] for x in output.values()),
            "step100_299_relative_us": [lo, hi], "step5_24_relative_us": [early_lo, early_hi],
            "note": "XPlane已归零时间戳，使用与JSON一致的相对微秒；不把它转换成1970年UTC。前次探索记录中的UTC列无效，本记录取代该列及其跨时钟比较。"}


def util(values):
    return {"samples": len(values), "mean": statistics.mean(values) if values else None,
            "zero_fraction": sum(x == 0 for x in values) / len(values) if values else None}


def host_stats():
    all_rows = [json.loads(s) for s in (ROOT / "step_timing.jsonl").read_text().splitlines()]
    rows = [r for r in all_rows if 100 <= r["step"] <= 299]
    assert [r["step"] for r in rows] == list(range(100, 300)) and all(r["completed"] for r in rows)
    start, end = rows[0]["wall_start"], rows[-1]["wall_end"]
    starts = [r["wall_start"] for r in rows]
    threshold = 1.5 * statistics.mean(r["host_step_s"] for r in rows)
    samples, slow, other = [], [], []
    per_gpu = collections.defaultdict(list)
    with (ROOT / "gpu_util_500ms.step300.snapshot.csv").open() as handle:
        for raw in csv.reader(handle):
            stamp = datetime.datetime.strptime(raw[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC).timestamp()
            if not start <= stamp <= end:
                continue
            gpu, value = int(raw[1]), float(raw[2])
            samples.append(value)
            per_gpu[gpu].append(value)
            index = max(0, bisect.bisect_right(starts, stamp) - 1)
            (slow if rows[index]["host_step_s"] > threshold else other).append(value)
    assert set(per_gpu) == set(range(8))
    mean = (end - start) / len(rows)
    return {"window": {"start_step": 100, "end_step": 299, "steps": 200,
                       "start_utc": datetime.datetime.fromtimestamp(start, UTC).isoformat(),
                       "end_utc": datetime.datetime.fromtimestamp(end, UTC).isoformat()},
            "runtime": json.loads((ROOT / "runtime.json").read_text()),
            "mean_step_s": mean, "samples_per_second": 128 / mean,
            "host_phases_mean_s": {key: statistics.mean(r["phases_s"].get(key, 0) for r in rows)
                                   for key in ("train_dispatch", "data_next", "logging", "checkpoint")},
            "host_data_wait_fraction": sum(r["phases_s"]["data_next"] for r in rows) / sum(r["host_step_s"] for r in rows),
            "slow_threshold_host_step_s": threshold,
            "slow_step_ids": [r["step"] for r in rows if r["host_step_s"] > threshold],
            "gpu_util": util(samples), "gpu_util_slow_steps": util(slow), "gpu_util_other_steps": util(other),
            "gpu_util_per_device": {str(k): util(v) for k, v in per_gpu.items()},
            "eta_remaining_at_step300_hours": (80000 - 300) * mean / 3600,
            "gpu_device_breakdown_status": "缺失：原始设备事件在第25步内截止，不能将短窗结果填入此窗口。",
            "note": "dispatch不是GPU计算耗时；数据等待与设备工作可重叠；未测得该稳态窗口的等待期间GPU空闲比例。"}


if __name__ == "__main__":
    for name, value in (("xplane_coverage_corrected.json", raw_coverage()), ("perf_step300_host_nvml.json", host_stats())):
        with (ROOT / name).open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        print(name, json.dumps(value, ensure_ascii=False), flush=True)
