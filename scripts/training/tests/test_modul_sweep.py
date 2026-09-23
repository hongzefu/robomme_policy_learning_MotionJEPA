"""顺序调度和真实shell入口失败后不得调用下一档训练。"""

import os
import json
import gzip
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/training/prod"))
import modul_budget_workflow as workflow


@pytest.mark.parametrize("fault",[None,"baseline","verify4096","train4096","complete4096","handoff","verify1024","resources1024"])
def test_queue_order_and_failure_stop(monkeypatch,fault):
    instance = object.__new__(workflow.Workflow)
    instance.head = "a"*40
    events = []
    def fail(name):
        if fault == name:
            raise ValueError(name)
    def resources(*args):
        if "complete4096" in events:
            fail("resources1024")
    monkeypatch.setattr(workflow,"resources",resources)
    def baseline():
        events.append("baseline");fail("baseline")
    def verify(budget):
        events.append(f"verify{budget}");fail(f"verify{budget}");return "report"
    def execute(name, command, changes=None):
        if name == "handoff-4096":
            events.append("handoff");fail("handoff")
        else:
            budget = 4096 if name == workflow.PROD_RUNS[4096] else 1024
            events.append(f"train{budget}");fail(f"train{budget}")
    def complete(budget,run):
        events.append(f"complete{budget}");fail(f"complete{budget}");return "completion"
    instance.baseline = baseline
    instance.verify_budget = verify
    instance.execute = execute
    instance.complete = complete
    instance.approve = lambda *args:"approval"
    instance.production_runner = lambda *args:[]
    instance.event = lambda **kwargs:None
    instance.archive_all = lambda:None
    if fault:
        with pytest.raises(ValueError):instance.run()
        assert "train1024" not in events
    else:
        instance.run()
        assert events == ["baseline","verify4096","train4096","complete4096","handoff","verify1024","train1024","complete1024"]


@pytest.mark.parametrize("budget",[4096,1024])
@pytest.mark.parametrize("fault",[None,"preflight","handoff","train"])
def test_new_shell_entry_uses_real_failure_control(tmp_path,budget,fault):
    source = (ROOT/"scripts/training/prod/run_modul_budget.sh").read_text()
    root=tmp_path/"repo";root.mkdir()
    runner=tmp_path/"runner.sh"
    runner.write_text(source.replace("MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA","MAIN="+str(root)))
    paths=root/"scripts/training/paths.sh";paths.parent.mkdir(parents=True)
    paths.write_text('V1_STORE="'+str(root/"v1-store")+'"\n')
    for folder in ("logs","secrets"):(root/"v1-store"/folder).mkdir(parents=True)
    (root/"v1-store/secrets/wandb.env").write_text("# 测试夹具，无凭据\n")
    bins=tmp_path/"bin";bins.mkdir()
    scripts={"git":'#!/bin/bash\nif [ "$1" = rev-parse ]; then printf "%s\\n" "'+"a"*40+'"; fi\n',
             "nvidia-smi":"#!/bin/bash\nexit 0\n",
             "uv":'''#!/bin/bash
case "$*" in
  *check_modul_completion.py*) printf 'handoff\n' >> "$TEST_CALLS"; test "$TEST_FAULT" != handoff ;;
  *preflight_train_launch.py*) printf 'preflight\n' >> "$TEST_CALLS"; test "$TEST_FAULT" != preflight ;;
  *scripts/training/train.py*) printf 'train\n' >> "$TEST_CALLS"; test "$TEST_FAULT" != train ;;
  *) exit 0 ;;
esac
'''}
    for name,text in scripts.items():
        path=bins/name;path.write_text(text);path.chmod(0o700)
    calls=tmp_path/"calls"
    env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ["PATH"],TEST_CALLS=str(calls),TEST_FAULT=fault or "")
    result=subprocess.run(["bash",str(runner),"prod",str(budget),"test-run","a"*40,"b"*64,"approval","report","baseline","handoff"],env=env,text=True,capture_output=True,timeout=30)
    got=calls.read_text().splitlines()
    if fault == "preflight" or (fault == "handoff" and budget == 1024):
        assert "train" not in got and result.returncode != 0
    elif fault == "train":
        assert got[-1] == "train" and result.returncode != 0
    else:
        assert got == (["handoff"] if budget == 1024 else [])+["preflight","train"]
        assert result.returncode == 0


def test_finished_archive_preserves_records_and_excludes_weights(tmp_path,monkeypatch):
    """实写临时档案；Git仅用计数桩，不能提交或推送测试夹具。"""
    monkeypatch.setattr(workflow,"ROOT",tmp_path)
    monkeypatch.setattr(workflow,"resources",lambda *args:None)
    instance=object.__new__(workflow.Workflow)
    instance.head="a"*40;instance.before_head="b"*40;instance.batch="fixture"
    instance.store=tmp_path/"v1-store"
    instance.root=instance.store/"bench/modul-budget-sweep/fixture"
    instance.root.mkdir(parents=True)
    instance.clean=lambda:None
    docs=tmp_path/"docs/training-doc"
    names=[workflow.PROD_RUNS[b] for b in (4096,1024)]
    for name in [instance.batch,*names,"fixture-m4096-input","fixture-m1024-input"]:
        target=docs/name;target.mkdir(parents=True)
        (target/"launch.md").write_text("# 起跑前记录\n")
        (target/"result.md").write_text("尚未运行\n")
    (docs/"README.md").write_text("\n".join(f"| `{name}/` | 夹具 | 未开始 |" for name in [instance.batch,*names])+"\n")
    events=[]
    stages=names+[f"m{b}-{command}" for b in (4096,1024) for command in ("yaml","frames","guards","assembly","pad","online","collate","oracle","init")]
    for stage in stages:
        events.extend([{"stage":stage,"status":"START","time":"2026-09-23T00:00:00+00:00","command":["夹具"]},
                       {"stage":stage,"status":"PASS","time":"2026-09-23T01:00:00+00:00","exit_code":0}])
        (instance.root/(stage+".log")).write_text("训练阶段\n50%|中间进度\nEXIT_CODE=0\n")
        if stage.startswith("m"):(instance.root/(stage+".json")).write_text("{}\n")
    (instance.root/"events.jsonl").write_text("".join(json.dumps(row)+"\n" for row in events))
    for filename in ("start.json","baseline-2048.json"):(instance.root/filename).write_text("{}\n")
    for budget in (4096,1024):
        for suffix in ("speed","approval","provenance","reload","refnpy-packed"):
            (instance.root/f"m{budget}-{suffix}.json").write_text("{}\n")
    for name in names:
        record=instance.run_records(name);record.mkdir(parents=True)
        (record/"run_meta.json").write_text("{}\n")
        (record/"metrics.jsonl").write_text('{"step":0}\n')
        (record/"arrays").mkdir();(record/"arrays/weight_sentinel.bin").write_bytes(b"weight")
        Path(str(record)+".gpu.csv").write_text("原始采样字节\n")
        (instance.root/(name+".completed.json")).write_text(json.dumps({"checkpoints":[*range(5000,80000,5000),79999]}))
    staged=[];calls=[]
    def fake_run(command,**kwargs):
        calls.append(command)
        if command[:2]==["git","add"]:staged.extend(command[command.index("--")+1:])
        return subprocess.CompletedProcess(command,0)
    def fake_output(command,**kwargs):
        if command[:4]==["git","diff","--cached","--name-only"]:return "\n".join(staged)+"\n"
        if command[:3]==["git","show","-s"]:return "commitV11.5Beta: 夹具\n"
        if command[:2]==["git","rev-parse"]:return "c"*40+"\n"
        raise AssertionError(command)
    monkeypatch.setattr(workflow.subprocess,"run",fake_run)
    monkeypatch.setattr(workflow.subprocess,"check_output",fake_output)
    instance.archive_all()
    assert calls[-1]==["git","push"] and calls[-2][:2]==["git","commit"]
    assert not list(docs.rglob("*.bin")) and not list(docs.rglob("*.yaml")) and not list(docs.rglob("*.sh"))
    for name in names:
        assert gzip.open(docs/name/"records/gpu.csv.gz","rt").read()=="原始采样字节\n"
        assert (docs/name/"records/run.summary.log").read_text()=="训练阶段\nEXIT_CODE=0\n"
        assert "80000步" in (docs/name/"result.md").read_text()
