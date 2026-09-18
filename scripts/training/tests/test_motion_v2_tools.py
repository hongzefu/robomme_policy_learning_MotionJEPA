"""利用真实改前记录检验验收器拒绝缺项，并验证 preflight 与计时汇总的边界。"""

import copy
import datetime
import gzip
import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/training"))
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from finish_check import metric_rows,trajectory
from summarize_step_timing import duration,intersection_duration,merged,summarize
import preflight_train_launch as preflight
from eval_es_bound import TASKS,aggregate as aggregate_es,predicted_windows

BASE=ROOT/"docs/training-doc/mv2-v7-guard-base/records"
HEAD="2126b1b1c436166662ff89a629985ecd4524fc42"


def test_trajectory_reads_real_baseline():
    metrics,states,batches,indices=trajectory(BASE,HEAD)
    assert (len(metrics),len(states),len(batches),indices["n"])==(100,5,7,872)


@pytest.mark.parametrize("damage",["missing_step","duplicate_step","missing_scalar","nan","bad_hex"])
def test_metrics_reject_corruption(tmp_path,damage):
    rows=[json.loads(x) for x in (BASE/"metrics.jsonl").read_text().splitlines()]
    if damage=="missing_step": rows.pop(40)
    elif damage=="duplicate_step": rows[40]["step"]=39
    elif damage=="missing_scalar": del rows[40]["grad_norm"]
    elif damage=="nan": rows[40]["loss"]["dec"]=float("nan")
    else: rows[40]["loss"]["hex"]="0x0p+0"
    path=tmp_path/"metrics.jsonl"; path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises((ValueError,KeyError)): metric_rows(path,100)


def test_preflight_motion_five_checks(tmp_path,monkeypatch):
    root=tmp_path/"v1-store"; lib=root/"datasets/test"
    frames=lib/"framesamp-8x8"; frames.mkdir(parents=True)
    motion=lib/"motion"; (motion/"meta").mkdir(parents=True)
    metadata=motion/"meta/store_meta.json"
    metadata.write_text(json.dumps({"layout":"motion-768-grid16-demopad17-v1","num_rows":71316}))
    args=SimpleNamespace(motion_store=str(motion),motion_store_meta_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                         motion_layout="motion-768-grid16-demopad17-v1",motion_rows=71316)
    monkeypatch.delenv("MMEVLA_MOTION_STORE",raising=False)
    preflight._RESULTS.clear(); preflight.check_motion_store(args,frames,root)
    assert len(preflight._RESULTS)==5 and all(ok for _,ok in preflight._RESULTS)
    for field,value in (("motion_store_meta_sha256","0"*64),("motion_layout","motion-768-grid16-v1"),("motion_rows",71315)):
        bad=copy.copy(args); setattr(bad,field,value)
        preflight._RESULTS.clear(); preflight.check_motion_store(bad,frames,root)
        assert not all(ok for _,ok in preflight._RESULTS)
    monkeypatch.setenv("MMEVLA_MOTION_STORE","")
    preflight._RESULTS.clear(); preflight.check_motion_store(args,frames,root)
    assert dict(preflight._RESULTS)["MOTION_ENV_UNSET"] is False


def test_timing_overlap_is_not_added_twice():
    intervals=[(0,2),(1,3),(4,6)]
    assert merged(intervals)==[[0,3],[4,6]]
    assert duration(intervals)==5
    assert intersection_duration(intervals,[(2,5)])==2


def test_timing_summary_uses_one_wall_window(tmp_path):
    # 合成时轴只用于测试统计器；不作为任何硬件性能结论。
    runtime={"device_count":1,"batch_size":8,"cuda_visible_devices":"0"}
    (tmp_path/"runtime.json").write_text(json.dumps(runtime))
    origin=1700000000.0
    rows=[dict(step=i,completed=True,wall_start=origin+i,wall_end=origin+i+1,
               host_step_s=1.0,phases_s={"data_next":0.2,"train_dispatch":0.8}) for i in range(3)]
    (tmp_path/"step_timing.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    events=[dict(ph="M",pid=1,name="process_name",args={"name":"/device:GPU:0"}),
            dict(ph="M",pid=1,tid=9,name="thread_name",args={"name":"Stream #9(Compute)"}),
            dict(ph="M",pid=1,tid=10000,name="thread_name",args={"name":"XLA Ops"})]
    for i in range(3):
        events.extend([
            dict(ph="X",pid=701,name="motionjepa_train",ts=i*1e6,dur=1e6,args={"step_num":str(i)}),
            dict(ph="X",pid=701,name="motionjepa_data_next",ts=(i+0.75)*1e6,dur=0.2e6,args={"step_num":str(i)}),
            dict(ph="X",pid=1,tid=9,name="gemm",ts=(i+0.1)*1e6,dur=0.8e6),
            dict(ph="X",pid=1,tid=9,name="nccl_all_reduce",ts=(i+0.7)*1e6,dur=0.2e6),
            dict(ph="X",pid=1,tid=10000,name="nccl_all_reduce",ts=i*1e6,dur=1e6),
        ])
    trace=tmp_path/"step_trace/test.trace.json.gz"; trace.parent.mkdir()
    trace.write_bytes(gzip.compress(json.dumps({"traceEvents":events}).encode()))
    csv=tmp_path/"gpu.csv"
    csv.write_text("\n".join(datetime.datetime.fromtimestamp(origin+i+0.5,datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")+f",0,{value},10" for i,value in enumerate((90,0,100))))
    result=summarize(tmp_path,csv,0,2)
    assert result["mean_step_s"]==1 and result["samples_per_second"]==8
    assert result["gpu_busy_mean_ms_per_step"]==pytest.approx(800)
    assert result["gpu_kernel_mean_ms_per_step"]["communication"]==pytest.approx(200)
    assert result["host_data_wait_all_gpu_idle_s"]==pytest.approx(0.15)
    assert result["gpu_util"]["mean"]==pytest.approx(190/3)
    assert result["gpu_util"]["zero_fraction"]==pytest.approx(1/3)
    assert result["gpu_ignored_non_stream_events"]==3
    assert result["gpu_step_event_coverage"]=={"0":{"steps_with_events":3,"missing_steps":[]}}
    # 真实故障形态：设备采集提前截止，但主线程、元数据和派生XLA轨道仍然完整。
    original_trace=trace.read_bytes()
    for missing_step in (1,2):
        damaged=[e for e in events if not (e.get("ph")=="X" and e.get("pid")==1
                  and e.get("tid")==9 and missing_step*1e6 <= e["ts"] < (missing_step+1)*1e6)]
        trace.write_bytes(gzip.compress(json.dumps({"traceEvents":damaged}).encode()))
        with pytest.raises(ValueError,match="未覆盖每个步骤"):
            summarize(tmp_path,csv,0,2)
    # 异步kernel跨越主线程步骤边界是合法覆盖，不要求每步恰好有一次新launch。
    spanning=[e for e in events if not (e.get("ph")=="X" and e.get("pid")==1 and e.get("tid")==9)]
    spanning.append(dict(ph="X",pid=1,tid=9,name="gemm",ts=0.1e6,dur=2.8e6))
    trace.write_bytes(gzip.compress(json.dumps({"traceEvents":spanning}).encode()))
    assert summarize(tmp_path,csv,0,2)["gpu_step_event_coverage"]["0"]["steps_with_events"]==3
    trace.write_bytes(original_trace)
    (tmp_path/"step_trace_metadata.json").write_text(json.dumps({"trace_viewer_event_limit":15}))
    with pytest.raises(ValueError,match="事件上限"): summarize(tmp_path,csv,0,2)
    # 恢复文件必须绑定本轮原始数据及其自身内容，拒绝误选其他run的同名步骤。
    from reexport_step_trace import write_binding
    original=trace.parent/'source.xplane.pb'; original.write_bytes(b'fixture-original')
    recovered=tmp_path/'recovered'; recovered.mkdir()
    copied=recovered/'full.trace.json.gz'; copied.write_bytes(trace.read_bytes())
    binding=write_binding(original,recovered,100000000,'测试夹具')
    assert summarize(tmp_path,csv,0,2,recovered)['trace_reexport_binding']==binding
    original.write_bytes(b'fixture-other-run')
    with pytest.raises(ValueError,match='不同源'): summarize(tmp_path,csv,0,2,recovered)
    original.write_bytes(b'fixture-original'); copied.write_bytes(b'wrong-json')
    with pytest.raises(ValueError,match='JSON指纹'): summarize(tmp_path,csv,0,2,recovered)


def test_eval_bound_requires_all_episodes_and_recomputes_budget(tmp_path):
    assert predicted_windows(1296)==160 and predicted_windows(1297)==161
    paths=[]
    for task in TASKS:
        path=tmp_path/f"{task}.json"
        report=dict(task=task,complete=True,episodes=[dict(episode=i,es=100,k_eval=86) for i in range(50)],
                    python="测试解释器",packages={},env_runner_sha256="固定",robomme_commit="固定",
                    budget=160,max_steps=1300)
        path.write_text(json.dumps(report)); paths.append(path)
    args=SimpleNamespace(reports=paths,out=tmp_path/"pass.json",budget=160,max_steps=1300)
    aggregate_es(args)
    assert json.loads(args.out.read_text())["episodes"]==200
    report=json.loads(paths[0].read_text()); report["episodes"][0]["es"]=1297
    paths[0].write_text(json.dumps(report)); args.out=tmp_path/"over.json"
    with pytest.raises(SystemExit): aggregate_es(args)
    assert json.loads(args.out.read_text())["headroom"]==-1
    report["episodes"].pop(); paths[0].write_text(json.dumps(report)); args.out=tmp_path/"missing.json"
    with pytest.raises(ValueError,match="200"): aggregate_es(args)
