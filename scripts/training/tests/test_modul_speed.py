"""轻量包装的同步位置、入口透明性与失败路径自测，不启动训练。"""

import importlib.util
import json
import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_modul_speed as speed


def test_three_syncs_and_phase_order(tmp_path):
    timer=speed.HostTiming(tmp_path,1000)
    events=[]
    sentinel={"input":[1,2,3],"parameters":[4,5,6]}
    before=json.dumps(sentinel,sort_keys=True)
    for step in range(1000):
        with timer.step(step,flush=lambda step=step:events.append((step,"flush"))):
            with timer.phase("train_dispatch"):pass
            with timer.phase("data_next"):pass
            if step==999:
                with timer.phase("checkpoint"):events.append((step,"save"))
    timer.close();timer.close()
    assert events==[(99,"flush"),(899,"flush"),(999,"flush"),(999,"save")]
    assert json.dumps(sentinel,sort_keys=True)==before
    rows=speed.read_rows(tmp_path/"step_timing.jsonl")
    assert [r["step"] for r in rows]==list(range(1000))
    assert all(r["completed"] and r["host_step_s"]>=0 for r in rows)
    assert {r["step"] for r in rows if "sync" in r}=={99,899,999}
    assert rows[-1]["sync"]["wall_end"]<=rows[-1]["save_wall_start"]<=rows[-1]["wall_end"]
    assert not (tmp_path/"step_trace").exists()


def test_failed_step_is_not_complete(tmp_path):
    timer=speed.HostTiming(tmp_path,1000)
    with pytest.raises(RuntimeError,match="故障"):
        with timer.step(0,flush=lambda:None):raise RuntimeError("故障")
    timer.close()
    assert speed.read_rows(tmp_path/"step_timing.jsonl")[0]["completed"] is False


def test_missing_flush_rejected(tmp_path):
    timer=speed.HostTiming(tmp_path,1000)
    timer.count=99
    with pytest.raises(RuntimeError,match="flush"):
        with timer.step(99):pass
    timer.close()


def test_real_entry_called_once_argv_preserved(tmp_path,monkeypatch):
    # 只替换真实训练入口，包装本身、计时与恢复逻辑全部实跑。
    monkeypatch.syspath_prepend(str(speed.ROOT/"scripts/training"))
    monkeypatch.setattr(speed,"ROOT",tmp_path)
    directory=tmp_path/"v1-store/speed-wrapper"
    calls=[]
    original_args=list(sys.argv)
    monkeypatch.delenv("XLA_FLAGS",raising=False)
    monkeypatch.delenv("MMEVLA_EXPECTED_TRAIN_HEAD",raising=False)
    original_check=speed.subprocess.check_output
    monkeypatch.setattr(speed.subprocess,"check_output",lambda args,**kw:
                        ("" if "status" in args else "a"*40) if args[0]=="git" else original_check(args,**kw))
    def fake_entry(path,run_name):
        calls.append((path,run_name,list(sys.argv)))
        import step_timing,jax
        assert step_timing.StepTiming is speed.HostTiming
        with pytest.raises(RuntimeError,match="profiler"):
            jax.profiler.start_trace("禁止的trace")
        assert sys.argv[1:]==["mme_vla_suite_b128_80k","--exp-name","test-only","--num-train-steps","1000","--no-wandb-enabled"]
    monkeypatch.setattr(speed.runpy,"run_path",fake_entry)
    monkeypatch.setattr(speed,"host_sample",lambda pid:{"wall_time":0,"pid":pid})
    speed.run(SimpleNamespace(records=str(directory),mode="perf",
                             train_args=["--","mme_vla_suite_b128_80k","--exp-name","test-only","--num-train-steps","1000","--no-wandb-enabled"]))
    assert len(calls)==1 and sys.argv==original_args
    metadata=json.loads((directory/"speed_run.json").read_text())
    assert metadata["entry_calls"]==1 and metadata["success"] and metadata["sampler_stopped"]
    assert metadata["profiler"] is False


def test_verification_xla_flags_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(speed,"ROOT",tmp_path)
    monkeypatch.setenv("XLA_FLAGS","--xla_gpu_autotune_level=0")
    directory=tmp_path/"v1-store/forbidden"
    with pytest.raises(ValueError,match="XLA_FLAGS"):
        speed.run(SimpleNamespace(records=str(directory),mode="perf",train_args=["mme_vla_suite_b128_80k"]))
    assert not directory.exists()


def report_fixture(tmp_path):
    """合成记录只用于验证报告器；名称和路径均为pytest临时夹具。"""
    root=tmp_path/"records";root.mkdir()
    base=1_790_000_000.
    timing=[]
    for step in range(1000):
        row={"step":step,"wall_start":base+50+step,"wall_end":base+51+step,
             "host_step_s":1.,"completed":True,"phases_s":{"train_dispatch":.8,"data_next":.2}}
        if step in (99,899):row["sync"]={"location":"window_end","wall_end":row["wall_end"],"seconds":.1}
        if step==999:
            row["sync"]={"location":"before_final_save","wall_end":base+1050,"seconds":.1}
            row["save_wall_start"]=base+1050
            row["phases_s"]["checkpoint"]=1.
        timing.append(row)
    (root/"step_timing.jsonl").write_text("".join(json.dumps(r)+"\n" for r in timing))
    metrics=[{"step":s,"wall_time":base+50+s,**{k:{"dec":.5,"hex":(.5).hex()} for k in
             ("loss","grad_norm","llm_grad_norm","mem_enc_norm","param_norm")}} for s in range(0,1000,100)]
    (root/"metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in metrics))
    checkpoint=tmp_path/"checkpoints/999";(checkpoint/"params").mkdir(parents=True);(checkpoint/"assets").mkdir()
    (checkpoint/"_CHECKPOINT_METADATA").write_text(json.dumps({"init_timestamp_nsecs":1,"commit_timestamp_nsecs":2}))
    for name in ("manifest.ocdbt","_METADATA"):(checkpoint/"params"/name).write_text("测试夹具")
    metadata={"success":True,"entry_calls":1,"sampler_stopped":True,"profiler":False,"start_wall":base,"end_wall":base+1053,
              "head":"a"*40,"wrapper_sha256":"b"*64,"argv":["测试夹具"],
              "loader":{"batch_size":128,"workers":16,"prefetch_factor":2},
              "save_calls":[{"step":999,"state_step":1000,"start_wall":base+1050,"return_wall":base+1051,"checkpoint_root":str(checkpoint.parent)}]}
    (root/"speed_run.json").write_text(json.dumps(metadata))
    (root/"run_meta.json").write_text(json.dumps({"argv":["train.py",*metadata["argv"]],"checkpoint_dir":str(checkpoint.parent)}))
    hosts=[{"wall_time":base+.5*i,"rss_bytes":{"42":1024},"shm_used":100,"shm_capacity":1000,
            "MemAvailable_bytes":10000,"Cached_bytes":1000,"md0_read_sectors":2*i} for i in range(2107)]
    for i in range(0,len(hosts),20):
        hosts[i]["system_io"]={"42":{"name":"测试进程","read_bytes":i,"write_bytes":i}}
    (root/"host_samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in hosts))
    gpu=tmp_path/"gpu.csv"
    gpu.write_text("".join(datetime.datetime.fromtimestamp(base+.5*i,datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")+
                           f", {g}, 95, 70000\n" for i in range(2107) for g in range(8)))
    log=tmp_path/"run.log"
    log.write_text("TRAIN_HEAD="+"a"*40+"\nLAUNCH_RESOLVED="+json.dumps({"actual":{"batch_size":128,"num_train_steps":1000,
        "num_workers":16,"fsdp_devices":8,"history_values":{"budget":2048},"checkpoint_dir":str(checkpoint.parent)}})+"\nRUN_ENV="+json.dumps({"test_fixture":True})+"\nEXIT_CODE=0\n")
    return SimpleNamespace(records=str(root),gpu=str(gpu),log=str(log),out=str(tmp_path/"report.json"))


def test_report_formula_and_no_slow_samples(tmp_path):
    args=report_fixture(tmp_path)
    assert speed.report(args)==0
    report=json.loads(Path(args.out).read_text())
    assert report["step_mean_s"]==1 and report["samples_per_second"]==128
    assert report["eta_hours"]==(150+79900+15+3)/3600
    assert report["gpu_strata"]["slow"]=={"mean_pct":None,"samples":0}
    assert report["status"]=="READY"


def test_async_boundary_wait_is_not_misread_as_drift(tmp_path):
    args=report_fixture(tmp_path);root=Path(args.records)
    rows=speed.read_rows(root/"step_timing.jsonl")
    clock=rows[99]["wall_end"]
    for step in range(100,1000):
        row=rows[step];row["wall_start"]=clock
        # 主线程每步派发0.5秒，log100和已有边界同步排空积压；真实GPU均速仍为1秒。
        duration=.5
        if step in (200,300,400,500,600,700,800):duration+=50
        if step in (899,999):duration+=50
        row["host_step_s"]=duration;clock+=duration;row["wall_end"]=clock
        if "sync" in row:row["sync"]["wall_end"]=clock
    (root/"step_timing.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    assert speed.report(args)==0
    result=json.loads(Path(args.out).read_text())
    assert result["drift_ratio"]==1 and result["tail_sync_mean_s"]==1
    assert result["tail_host_mean_s"]==.5


@pytest.mark.parametrize("fault",["missing_step","missing_sync","missing_gpu","exit_code","checkpoint","nan","inf","duplicate","sparse"])
def test_incomplete_speed_evidence_rejected(tmp_path,fault):
    args=report_fixture(tmp_path);root=Path(args.records)
    if fault in ("missing_step","missing_sync"):
        path=root/"step_timing.jsonl";rows=speed.read_rows(path)
        if fault=="missing_step":rows.pop(400)
        else:rows[99].pop("sync")
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    elif fault=="missing_gpu":
        path=Path(args.gpu);path.write_text("\n".join(l for l in path.read_text().splitlines() if ", 7," not in l)+"\n")
    elif fault=="exit_code":Path(args.log).write_text("EXIT_CODE=1\n")
    elif fault in ("nan","inf","duplicate","sparse"):
        path=Path(args.gpu);lines=path.read_text().splitlines()
        if fault in ("nan","inf"):lines[3000]=lines[3000].replace(", 95,",f", {fault},")
        elif fault=="duplicate":lines.insert(3001,lines[3000])
        else:lines=[line for i,line in enumerate(lines) if (i//8)%8==0]
        path.write_text("\n".join(lines)+"\n")
        if fault=="sparse":
            assert speed.report(args)==1
            assert json.loads(Path(args.out).read_text())["status"]=="INCOMPLETE"
            return
    else:(tmp_path/"checkpoints/999/_CHECKPOINT_METADATA").unlink()
    with pytest.raises((ValueError,KeyError,OSError)):speed.report(args)
