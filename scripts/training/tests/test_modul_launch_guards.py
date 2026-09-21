"""启动参数绑定及零训练负例；不允许以打印失败后继续调用训练冒充拒绝。"""

import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/training"))
import modul_launch_contract as contract


def arguments(mode="prod"):
    run=contract.PROD_RUN if mode=="prod" else "test-m2048-"+mode
    v1=ROOT/"v1-store"
    asset=v1/"train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
    ds=v1/"datasets/4task-v2-1600ep-604f16da/framesamp-8x8"
    args=[contract.CONFIG,"--exp-name",run,"--assets-base-dir",str(v1/"train-assets"),
          "--data.assets.assets-dir",str(asset),"--data.assets.asset-id","robomme",
          "--checkpoint-base-dir",str(v1/"train-runs"),"--dataset-path",str(ds),"--model.history-config",contract.YAML]
    if mode!="prod":args += ["--num-train-steps","20" if mode=="smoke" else "1000","--no-wandb-enabled"]
    namespace=SimpleNamespace(repo=str(ROOT),launch_mode=mode,expected_run_name=run,history_config=contract.YAML,
        run_root=str(v1/"train-runs"/contract.CONFIG/run),assets_dir=str(asset),asset_id="robomme",dataset_path=str(ds),
        norm_stats_sha256=contract.NORM_SHA,train_head="a"*40,history_config_sha256="b"*64,
        approval_record=None,runner=None,report=None)
    return args,namespace


@pytest.fixture(scope="module")
def parsed():
    argv,_=arguments()
    return contract.parse_config(argv,ROOT)


@pytest.mark.parametrize("mode",["prod","smoke","perf"])
def test_real_cli_configuration(mode):
    argv,a=arguments(mode)
    contract.validate_argv(argv,mode)
    actual=contract.validate_config(contract.parse_config(argv,ROOT),a)
    assert actual["num_train_steps"]=={"prod":80000,"smoke":20,"perf":1000}[mode]


@pytest.mark.parametrize("extra",[
    ["--num-train-steps","20"],["--batch-size","128"],["--num-workers","4"],
    ["--fsdp-devices","2"],["--lr-schedule.peak-lr","0.0001"],
    ["--exp-name",contract.PROD_RUN],["--overwrite"],
])
def test_prod_rejects_coverage_and_duplicates(extra):
    argv,_=arguments()
    calls=[]
    with pytest.raises(ValueError):
        contract.validate_argv(argv+extra,"prod")
        calls.append("train")
    assert calls==[]


@pytest.mark.parametrize("field,value",[
    ("name","mme_vla_suite_b128_60k"),("batch_size",64),("num_workers",4),("fsdp_devices",2),
    ("num_train_steps",20),("checkpoint_dir","/scratch/hongze/wrong"),
])
def test_parsed_negative_never_trains(parsed,field,value):
    _,a=arguments();bad=copy.deepcopy(parsed);bad["actual"][field]=value
    calls=[]
    with pytest.raises(ValueError):
        contract.validate_config(bad,a)
        calls.append("train")
    assert calls==[]


def test_missing_approval_never_trains(parsed):
    _,a=arguments();calls=[]
    with pytest.raises(ValueError,match="同意"):
        contract.validate_approval(a,parsed["actual"],{})
        calls.append("train")
    assert calls==[]


def test_approval_binds_every_digest(parsed,tmp_path):
    _,a=arguments()
    runner=tmp_path/"runner.sh";runner.write_text("# 测试桩，无训练\n")
    report=tmp_path/"report.json";report.write_text(json.dumps({"status":"READY","perf_head":a.train_head,"environment":contract.runtime_environment(ROOT)}))
    a.runner=str(runner);a.report=str(report);a.approval_record=str(tmp_path/"approval.json")
    data={"manifest_file_sha256":contract.MANIFEST_SHA,"store_meta_sha256":"c"*64}
    approval={"train_head":a.train_head,"runner_sha256":contract.sha(runner),"report_sha256":contract.sha(report),
        "run_name":a.expected_run_name,"config_sha256":contract.json_sha(parsed["actual"]),
        "yaml_sha256":a.history_config_sha256,"norm_stats_sha256":a.norm_stats_sha256,
        **data,"run_root":a.run_root,"user_quote":"测试夹具中的同意，不是真实用户授权",
        "approved":True,"approved_at":"2026-09-20T00:00:00+00:00"}
    Path(a.approval_record).write_text(json.dumps(approval))
    contract.validate_approval(a,parsed["actual"],data)
    for key in ("train_head","runner_sha256","report_sha256","run_name","config_sha256","yaml_sha256","norm_stats_sha256","manifest_file_sha256","store_meta_sha256","run_root"):
        Path(a.approval_record).write_text(json.dumps({**approval,key:"wrong"}))
        with pytest.raises(ValueError):contract.validate_approval(a,parsed["actual"],data)


def test_busy_gpu_never_trains(monkeypatch):
    monkeypatch.setattr(contract.shutil,"disk_usage",lambda p:SimpleNamespace(free=500*10**9))
    monkeypatch.setattr(contract.subprocess,"run",lambda *a,**kw:SimpleNamespace(returncode=0,stdout="\n".join(f"{g}, {1 if g==4 else 0}" for g in range(8)),stderr=""))
    calls=[]
    with pytest.raises(ValueError,match="空闲"):
        contract.resources(ROOT,"prod")
        calls.append("train")
    assert calls==[]


@pytest.mark.parametrize("fault",[None,"head","yaml","missing_approval","duplicate","smoke_steps","wrong_config","run_root","busy_gpu"])
def test_full_preflight_with_train_stub(tmp_path,monkeypatch,fault,capsys):
    """真实preflight及CLI解析；仅隔离Git/硬件状态和训练调用，不初始化模型。"""
    import preflight_train_launch as preflight
    argv,a=arguments()
    monkeypatch.setenv("OPENPI_DATA_HOME",str(ROOT/"v1-store/models"))
    monkeypatch.setenv("MMEVLA_FRAMESAMP_SOURCE",str(Path(a.dataset_path).parent/"source"))
    monkeypatch.setenv("MMEVLA_FRAMESAMP_MANIFEST",str(Path(a.dataset_path).parent/"meta/episode_manifest.json"))
    monkeypatch.setenv("TRAIN_TIMING_STEPS","0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0,1,2,3,4,5,6,7")
    monkeypatch.delenv("XLA_FLAGS",raising=False)
    monkeypatch.delenv("PYTHONPATH",raising=False)
    actual=contract.parse_config(argv,ROOT)["actual"]
    a.history_config_sha256=contract.sha(ROOT/"src/mme_vla_suite/models/config/robomme"/contract.YAML)
    runner=tmp_path/"fixture-runner.sh";runner.write_text("# 只读测试桩\n")
    report=tmp_path/"report.json";report.write_text(json.dumps({"status":"READY","perf_head":a.train_head,"environment":contract.runtime_environment(ROOT)}))
    approval=tmp_path/"approval.json"
    data=contract.validate_data(a)
    approval.write_text(json.dumps({"train_head":a.train_head,"runner_sha256":contract.sha(runner),
        "report_sha256":contract.sha(report),"run_name":a.expected_run_name,"config_sha256":contract.json_sha(actual),
        "yaml_sha256":a.history_config_sha256,"norm_stats_sha256":a.norm_stats_sha256,
        "manifest_file_sha256":data["manifest_file_sha256"],"store_meta_sha256":data["store_meta_sha256"],
        "run_root":a.run_root,"approved":True,"user_quote":"测试夹具；禁止当作真实授权",
        "approved_at":"2026-09-20T00:00:00+00:00"}))
    monkeypatch.setattr(preflight,"git",lambda wt,*args:(0,"wrong" if fault=="head" else a.train_head) if args==("rev-parse","HEAD") else (0,""))
    original=contract.subprocess.run
    def run(command,*args,**kw):
        if command[0]=="nvidia-smi" and "--query-gpu=index,memory.used" in command:
            return SimpleNamespace(returncode=0,stdout="\n".join(f"{i}, {1 if fault=='busy_gpu' and i==0 else 0}" for i in range(8)),stderr="")
        if command[0]=="tmux":return SimpleNamespace(returncode=0,stdout="",stderr="")
        return original(command,*args,**kw)
    monkeypatch.setattr(contract.subprocess,"run",run)
    monkeypatch.setattr(contract.shutil,"disk_usage",lambda p:SimpleNamespace(free=500*10**9))
    if fault=="yaml":a.history_config_sha256="wrong"
    if fault=="missing_approval":approval.unlink()
    if fault=="run_root":a.run_root=str(tmp_path/"wrong-run-root")
    if fault=="duplicate":argv += ["--exp-name",a.expected_run_name]
    if fault=="smoke_steps":argv += ["--num-train-steps","20"]
    if fault=="wrong_config":argv[0]="mme_vla_suite_b128_60k"
    pre_args=["preflight_train_launch.py","--repo",str(ROOT),"--train-head",a.train_head,
        "--history-config",a.history_config,"--history-config-sha256",a.history_config_sha256,
        "--assets-dir",a.assets_dir,"--asset-id",a.asset_id,"--norm-stats-sha256",a.norm_stats_sha256,
        "--dataset-path",a.dataset_path,"--run-root",a.run_root,"--launch-mode","prod",
        "--expected-run-name",a.expected_run_name,"--approval-record",str(approval),
        "--runner",str(runner),"--report",str(report),"--",*argv]
    monkeypatch.setattr(sys,"argv",pre_args)
    calls=[]
    code=preflight.main()
    if code==0:calls.append("真实训练入口的桩")
    if fault is None:
        assert code==0 and calls==["真实训练入口的桩"]
        assert len(preflight._RESULTS)>=26 and all(ok for _,ok in preflight._RESULTS)
    else:
        assert code!=0 and calls==[]
    capsys.readouterr()


@pytest.mark.parametrize("mode",["prod","smoke","perf"])
@pytest.mark.parametrize("preflight_exit",[0,7])
def test_shell_runner_propagates_preflight_failure(tmp_path,mode,preflight_exit):
    """真实runner的内层set-e必须阻止失败后的训练调用；外部入口全换为计数桩。"""
    source=(ROOT/"scripts/training/prod/run_modul2048.sh").read_text()
    fixture=tmp_path/"repo";fixture.mkdir()
    main_line="MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA"
    assert source.count(main_line)==1
    runner=tmp_path/"runner.sh"
    runner.write_text(source.replace(main_line,"MAIN="+str(fixture)))
    paths=fixture/"scripts/training/paths.sh";paths.parent.mkdir(parents=True)
    paths.write_text('V1_STORE="'+str(fixture/'v1-store')+'"\n')
    for sub in ("logs","bench/m2048","secrets"):(fixture/"v1-store"/sub).mkdir(parents=True)
    (fixture/"v1-store/secrets/wandb.env").write_text("# 空白测试凭据，未读真实secret\n")
    bins=tmp_path/"bin";bins.mkdir()
    git=bins/"git";git.write_text('#!/bin/bash\nif [ "$1" = rev-parse ]; then printf "%s\\n" "'+"a"*40+'"; fi\n')
    uv=bins/"uv";uv.write_text('''#!/bin/bash
case "$*" in
  *preflight_train_launch.py*) printf 'preflight\n' >> "$TEST_CALLS"; exit "$TEST_PREFLIGHT_EXIT" ;;
  *scripts/training/train.py*|*check_modul_speed.py*) printf 'train\n' >> "$TEST_CALLS"; exit 0 ;;
  *) exit 0 ;;
esac
''')
    gpu=bins/"nvidia-smi";gpu.write_text("#!/bin/bash\nexit 0\n")
    for binary in (git,uv,gpu):binary.chmod(0o700)
    calls=tmp_path/"calls.txt"
    env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ["PATH"],TEST_CALLS=str(calls),TEST_PREFLIGHT_EXIT=str(preflight_exit))
    result=subprocess.run(["bash",str(runner),mode,"test-run","a"*40,"b"*64,"test-approval","test-report"],
                          env=env,text=True,capture_output=True,timeout=20)
    got=calls.read_text().splitlines()
    assert got==(["preflight","train"] if preflight_exit==0 else ["preflight"])
    assert result.returncode==preflight_exit
    assert f"EXIT_CODE={preflight_exit}" in result.stdout
