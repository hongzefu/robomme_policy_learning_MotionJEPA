"""新预算、完整配置与最终保存的拒绝测试；测试授权仅为夹具。"""

import copy
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/training"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import config_record
import modul_launch_contract as contract
import check_modul_completion as completion
from test_modul_launch_guards import arguments, speed_report


@pytest.fixture(scope="module")
def baseline():
    argv,_ = arguments()
    return contract.parse_config(argv,ROOT)


@pytest.mark.parametrize("budget",[1024,4096])
def test_new_budget_real_configuration(baseline,tmp_path,budget):
    argv,args = arguments()
    argv[argv.index("--model.history-config")+1] = contract.BUDGET_YAMLS[budget]
    argv[argv.index("--exp-name")+1] = contract.PROD_RUNS[budget]
    args.expected_run_name = contract.PROD_RUNS[budget]
    args.history_config = contract.BUDGET_YAMLS[budget]
    args.run_root = str(ROOT/"v1-store/train-runs"/contract.CONFIG/args.expected_run_name)
    path = tmp_path/"baseline.json"
    path.write_text(json.dumps({"source_head":"55647ff33c8ddb9ec324fdbcee8bd1491456725b","complete":baseline["actual"]["complete"]}))
    args.config_baseline = str(path)
    parsed = contract.parse_config(argv,ROOT)
    actual = contract.validate_config(parsed,args)
    assert actual["history_values"]["budget"] == budget
    changes = config_record.compare_records(baseline["actual"]["complete"],actual["complete"],config_record.BUDGET_FIELDS)
    assert "history_values.budget" in changes
    assert "train_config.fields.model.fields.history_config" in changes


@pytest.mark.parametrize("fault",["unknown","missing","lr_type","resume","transform"])
def test_complete_configuration_rejects_unlisted_fields(baseline,fault):
    original = baseline["actual"]["complete"]
    candidate = copy.deepcopy(original)
    fields = candidate["train_config"]["fields"]
    if fault == "unknown": fields["unknown_future_option"] = True
    elif fault == "missing": fields.pop("policy_metadata")
    elif fault == "lr_type": fields["lr_schedule"]["type"] = "错误的调度器类型"
    elif fault == "resume": fields["resum_ckpt_id"] = 5000
    else: candidate["resolved_data"]["fields"]["use_quantile_norm"] = False
    calls = []
    with pytest.raises(ValueError):
        config_record.compare_records(original,candidate,config_record.BUDGET_FIELDS)
        calls.append("train")
    assert calls == []


def test_valid_other_budget_report_with_correct_hashes_is_rejected(baseline,tmp_path):
    _,args = arguments()
    actual = baseline["actual"]
    runner = tmp_path/"runner.sh"; runner.write_text("# 无训练的测试夹具\n")
    report = speed_report(actual,args)
    report["launch"]["actual"]["history_values"]["budget"] = 1024
    report["launch"]["actual"]["model"]["history_config"] = contract.BUDGET_YAMLS[1024]
    report["launch"]["actual"]["complete"]["history_values"]["budget"] = 1024
    report["launch"]["actual"]["complete"]["train_config"]["fields"]["model"]["fields"]["history_config"] = contract.BUDGET_YAMLS[1024]
    report_path = tmp_path/"report.json"; report_path.write_text(json.dumps(report))
    args.runner,args.report,args.approval_record = str(runner),str(report_path),str(tmp_path/"approval.json")
    data = {"manifest_file_sha256":contract.MANIFEST_SHA,"store_meta_sha256":"c"*64}
    approval = {"train_head":args.train_head,"runner_sha256":contract.sha(runner),"report_sha256":contract.sha(report_path),
        "run_name":args.expected_run_name,"config_sha256":contract.json_sha(actual),"yaml_sha256":args.history_config_sha256,
        "norm_stats_sha256":args.norm_stats_sha256,**data,"run_root":args.run_root,
        "approved":True,"user_quote":"测试夹具，不是真实授权","approved_at":"2026-09-20T00:00:00+00:00"}
    Path(args.approval_record).write_text(json.dumps(approval))
    calls = []
    with pytest.raises(ValueError,match="另一记忆预算"):
        contract.validate_approval(args,actual,data)
        calls.append("train")
    assert calls == []


def completion_fixture():
    arrays = {f"param_{i}":np.asarray([i+1],np.float32) for i in range(61)}
    start = {"run_name":contract.PROD_RUNS[4096],"head":"a"*40,"budget":4096,"num_train_steps":80000,
             "log_interval":100,"run_uuid":str(uuid.uuid4()),"wandb_run_id":"fixture","complete":{"fixture":True}}
    scalar = {"dec":1.,"hex":float(1).hex(),"finite":True}
    final = {**start,"state_step":80000,"loop_step":79999,"checkpoint_step":79999,
             "tail":[{"step":s,"scalars":{key:dict(scalar) for key in completion.SCALARS}} for s in range(79901,80000)],
             "tail_means":{key:1. for key in completion.SCALARS},"ema_leaves":completion.summarize_tree(arrays)}
    wait = {"run_uuid":start["run_uuid"],"head":start["head"],"wait_until_finished":True}
    metrics = [{"step":s,"wall_time":float(s),**{key:{"dec":1.,"hex":float(1).hex()} for key in completion.SCALARS}} for s in range(0,80000,100)]
    return arrays,start,final,wait,metrics,["EXIT_CODE=0"],[*range(5000,80000,5000),79999]


def check_fixture(values):
    arrays,start,final,wait,metrics,exits,names = values
    completion.validate_records(start,final,wait,metrics,exits,names,run=contract.PROD_RUNS[4096],head="a"*40,budget=4096,steps=80000,interval=100)
    completion.validate_arrays(final,arrays)


def test_completion_positive_fixture():
    check_fixture(completion_fixture())


def test_real_final_writer_and_reader_agree(tmp_path):
    from types import SimpleNamespace
    from final_record import finish, committed
    arrays,start,_,_,metrics,exits,names = completion_fixture()
    handle = (tmp_path,start)
    state = SimpleNamespace(ema_params=arrays,step=80000)
    finish(handle,state,[{key:np.asarray(1.,np.float32) for key in completion.SCALARS} for _ in range(99)],79999)
    committed(handle)
    final = json.loads((tmp_path/"final.json").read_text())
    wait = json.loads((tmp_path/"checkpoint_wait_done.json").read_text())
    completion.validate_records(start,final,wait,metrics,exits,names,run=contract.PROD_RUNS[4096],head="a"*40,budget=4096,steps=80000,interval=100)
    completion.validate_arrays(final,arrays)


@pytest.mark.parametrize("fault",["nonzero","killed","missing_final","only_last_log","old_run","swap_params","tail_nan","params_nan","state_step","head","budget","wait","duplicate_exit"])
def test_failed_completion_never_hands_off(fault):
    values = list(completion_fixture())
    arrays,start,final,wait,metrics,exits,names = values
    if fault == "nonzero": exits[:] = ["EXIT_CODE=1"]
    elif fault == "killed": exits.clear()
    elif fault == "missing_final": names.pop()
    elif fault == "only_last_log": metrics[:] = metrics[-1:]
    elif fault == "old_run": start["run_name"] = final["run_name"] = contract.PROD_RUNS[2048]
    elif fault == "swap_params": arrays["param_0"] += 1
    elif fault == "tail_nan": final["tail"][-1]["scalars"]["loss"]["dec"] = float("nan")
    elif fault == "params_nan": arrays["param_0"][:] = np.nan
    elif fault == "state_step": final["state_step"] = 79999
    elif fault == "head": final["head"] = "b"*40
    elif fault == "budget": final["budget"] = 1024
    elif fault == "wait": wait["wait_until_finished"] = False
    else: exits.append("EXIT_CODE=0")
    calls = []
    with pytest.raises(ValueError):
        check_fixture(values)
        calls.append("1024")
    assert calls == []


def test_loadable_nan_checkpoint_is_rejected(tmp_path):
    import orbax.checkpoint as ocp
    from openpi.models.model import restore_params
    values = list(completion_fixture())
    arrays = values[0]
    arrays["param_0"][:] = np.nan
    path = tmp_path/"params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path,{"params":arrays})
    loaded = restore_params(path,restore_type=np.ndarray,dtype=None)
    with pytest.raises(ValueError,match="NaN/Inf"):
        completion.validate_arrays(values[2],loaded)
