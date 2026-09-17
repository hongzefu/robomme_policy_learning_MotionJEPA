"""汇总同一稳态窗口的主线程等待、设备 trace 与 NVML；异步阶段不作可相加解释。"""

import argparse
import bisect
import csv
import datetime
import gzip
import json
import math
import pathlib
import statistics


def merged(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end <= start: continue
        if result and start <= result[-1][1]: result[-1][1]=max(result[-1][1],end)
        else: result.append([start,end])
    return result


def duration(intervals):
    return sum(end-start for start,end in merged(intervals))


def intersection_duration(left,right):
    a,b=merged(left),merged(right)
    i=j=0; total=0.0
    while i<len(a) and j<len(b):
        total+=max(0.0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1] < b[j][1]: i+=1
        else: j+=1
    return total


def utilization(rows):
    if not rows: return {"samples":0,"mean":None,"zero_fraction":None}
    return {"samples":len(rows),"mean":statistics.mean(rows),"zero_fraction":sum(x==0 for x in rows)/len(rows)}


def summarize(records, gpu_csv, warmup=100, end_step=299):
    runtime=json.loads((records/"runtime.json").read_text())
    rows=[json.loads(s) for s in (records/"step_timing.jsonl").read_text().splitlines() if s]
    selected=[r for r in rows if warmup <= r["step"] <= end_step]
    if [r["step"] for r in selected] != list(range(warmup,end_step+1)) or not all(r["completed"] for r in selected):
        raise ValueError("计时没有覆盖要求的完整稳态窗口")
    if any(not math.isfinite(r["host_step_s"]) or r["host_step_s"] <= 0 for r in selected):
        raise ValueError("主线程计时不合法")
    trace_files=list((records/"step_trace").rglob("*.trace.json.gz"))
    if len(trace_files)!=1: raise ValueError("trace 文件缺失或混入多轮")
    with gzip.open(trace_files[0]) as f: trace=json.load(f)
    events=trace["traceEvents"]
    gpu_pids={e["pid"]:int(e["args"]["name"].rsplit(":",1)[1]) for e in events
              if e.get("name")=="process_name" and e.get("args",{}).get("name","").startswith("/device:GPU:")}
    if len(gpu_pids)!=runtime["device_count"]: raise ValueError("trace 没有覆盖全部 GPU")
    step_events={int(e["args"]["step_num"]):e for e in events if e.get("ph")=="X" and e.get("name")=="motionjepa_train"}
    if any(r["step"] not in step_events for r in selected): raise ValueError("trace 缺少逐步注解")
    first,last=step_events[warmup],step_events[end_step]
    lo,hi=float(first["ts"]),float(last["ts"])+float(last["dur"])
    if hi <= lo: raise ValueError("trace 时间窗非法")
    gpu_intervals={pid:[] for pid in gpu_pids}
    categories={"gemm":0.0,"communication":0.0,"transfer":0.0,"other":0.0}
    event_counts={pid:0 for pid in gpu_pids}
    for e in events:
        pid=e.get("pid")
        if pid not in gpu_pids or e.get("ph")!="X": continue
        begin,finish=max(lo,float(e["ts"])),min(hi,float(e["ts"])+float(e.get("dur",0)))
        if finish <= begin: continue
        gpu_intervals[pid].append((begin,finish)); event_counts[pid]+=1
        label=(str(e.get("name",""))+" "+str(e.get("args",{}).get("hlo_op",""))).lower()
        if any(x in label for x in ("nccl","all_reduce","all-reduce","all_gather","all-gather","reduce_scatter")): kind="communication"
        elif any(x in label for x in ("memcpy","memset")): kind="transfer"
        elif any(x in label for x in ("gemm","matmul","dot")): kind="gemm"
        else: kind="other"
        categories[kind]+=finish-begin
    if not all(event_counts.values()): raise ValueError("稳态窗口内有 GPU 缺少设备事件")
    waits=[]
    for e in events:
        if e.get("name")=="motionjepa_data_next" and e.get("ph")=="X" and warmup <= int(e["args"]["step_num"]) <= end_step:
            waits.append((float(e["ts"]),float(e["ts"])+float(e["dur"])))
    any_busy=merged([interval for intervals in gpu_intervals.values() for interval in intervals])
    wait_total=duration(waits)
    wait_idle=max(0.0,wait_total-intersection_duration(waits,any_busy))
    start_wall,end_wall=selected[0]["wall_start"],selected[-1]["wall_end"]
    wall_seconds=end_wall-start_wall
    mean_step=wall_seconds/len(selected)
    starts=[r["wall_start"] for r in selected]
    slow_limit=1.5*statistics.mean(r["host_step_s"] for r in selected)
    samples=[]; slow=[]; other=[]; per_gpu={}
    with gpu_csv.open() as f:
        for row in csv.reader(f):
            if len(row)!=4: raise ValueError("NVML CSV 不是四列")
            timestamp=datetime.datetime.strptime(row[0].strip(),"%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            if not start_wall <= timestamp <= end_wall: continue
            gpu,value=int(row[1]),float(row[2])
            samples.append(value); per_gpu.setdefault(gpu,[]).append(value)
            index=min(len(selected)-1,max(0,bisect.bisect_right(starts,timestamp)-1))
            (slow if selected[index]["host_step_s"] > slow_limit else other).append(value)
    physical={int(x) for x in runtime["cuda_visible_devices"].split(",")}
    if set(per_gpu)!=physical or not samples: raise ValueError("NVML 未覆盖窗口内全部可见 GPU")
    phases={name:statistics.mean(r["phases_s"].get(name,0) for r in selected)
            for name in ("train_dispatch","data_next","logging","checkpoint")}
    result={"window":{"start_step":warmup,"end_step":end_step,"steps":len(selected),"start_utc_epoch":start_wall,"end_utc_epoch":end_wall},
            "runtime":runtime,"mean_step_s":mean_step,"samples_per_second":runtime["batch_size"]/mean_step,
            "host_phases_mean_s":phases,"host_data_wait_fraction":sum(r["phases_s"]["data_next"] for r in selected)/sum(r["host_step_s"] for r in selected),
            "gpu_kernel_mean_ms_per_step":{k:v/1000/len(gpu_pids)/len(selected) for k,v in categories.items()},
            "gpu_busy_mean_ms_per_step":statistics.mean(duration(v) for v in gpu_intervals.values())/1000/len(selected),
            "host_data_wait_all_gpu_idle_s":wait_idle/1e6,"gpu_util":utilization(samples),
            "gpu_util_slow_steps":utilization(slow),"gpu_util_other_steps":utilization(other),
            "slow_threshold_host_step_s":slow_limit,"gpu_util_per_device":{str(k):utilization(v) for k,v in per_gpu.items()},
            "trace_file":str(trace_files[0]),"trace_event_counts":{str(gpu_pids[k]):v for k,v in event_counts.items()},
            "measurement_note":"设备事件按同一墙钟窗口统计；异步执行可重叠，kernel 累计时间与主线程各阶段不能相加。NVML 使用原始密集采样，不把相同读数视为新增独立证据。"}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",type=pathlib.Path,required=True)
    parser.add_argument("--gpu-csv",type=pathlib.Path,required=True)
    parser.add_argument("--warmup-steps",type=int,default=100)
    parser.add_argument("--end-step",type=int,default=299)
    parser.add_argument("--out",type=pathlib.Path,required=True)
    args=parser.parse_args()
    result=summarize(args.records,args.gpu_csv,args.warmup_steps,args.end_step)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(f"STEADY_PERF=PASS steps={result['window']['steps']} mean_step_s={result['mean_step_s']:.6f} samples_per_second={result['samples_per_second']:.3f}")
    print("HOST_PHASES "+json.dumps(result["host_phases_mean_s"]))
    print("DEVICE_PHASES_MS_PER_STEP "+json.dumps(result["gpu_kernel_mean_ms_per_step"]))
    print("GPU_UTIL "+json.dumps(result["gpu_util"]))
    print("GPU_UTIL_STRATA "+json.dumps({"slow":result["gpu_util_slow_steps"],"other":result["gpu_util_other_steps"]}))


if __name__ == "__main__":
    main()
