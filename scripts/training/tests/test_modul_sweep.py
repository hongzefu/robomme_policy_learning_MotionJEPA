"""顺序调度和真实shell入口失败后不得调用下一档训练。"""

import os
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
