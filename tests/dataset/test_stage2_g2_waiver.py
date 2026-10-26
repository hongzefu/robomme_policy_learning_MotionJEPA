"""G2例外固定身份、原失败字节保真与真实消费链；不编码窗口、不伪造数值通过。"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
for relative in ("tests/datastore", "tests/policies", "tests/dataset"):
    sys.path.insert(0, str(ROOT / relative))
from test_motion_store_stage2 import stage2_identity
from test_pack_motion_stage2 import stage2_fx, pack, certify, verify_args
from test_stage2_export_contract import export, snapshot, refresh_ready, write
import test_motion_client_stage2 as client_tests
from test_stage2_restore import restore
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.policies.motion_client import validate_stage2_evidence

FAILURE = ('{"error":"G2 超阈值：ButtonUnmask_ep178_exec:10 mse=1.747542904391321e-07 max=0.013544513720267467",'
           '"gate":"G2","producer":"stage2-numeric-validator-v1","schema":"stage2-validation-gate-v1","status":"FAIL"}\n')


def waived(provenance=None, *, checkpoint=None, samples=None):
    prov = provenance or stage2_identity()
    return dict(schema="stage2-validation-gate-v1",gate="G2",status="WAIVED",producer="stage2-g2-user-exception-v1",
                checkpoint_sha256=checkpoint or ms.STAGE2_G2_WAIVER_CHECKPOINT_SHA256,
                samples_sha256=samples or ms.STAGE2_G2_WAIVER_SAMPLES_SHA256,
                source_pin=copy.deepcopy(prov["source_pin"]),numerics=copy.deepcopy(prov["numerics"]),
                online_vae=copy.deepcopy(prov["stage2_compat"]["online_vae"]),
                exception=dict(schema="stage2-g2-user-exception-v1",authorized_by="user",user_quote="可以忽略 继续做",
                    authorized_at="2026-10-26T19:21:45Z",scope="g2_cross_precision_numerics_only",numeric_validation_complete=False,
                    original_failure_sha256=ms.STAGE2_G2_WAIVER_FAILURE_SHA256,original_failure_json=FAILURE))


def expected(record):
    return {key:copy.deepcopy(record[key]) for key in ("checkpoint_sha256","samples_sha256","source_pin","numerics","online_vae")}


def test_exact_authorization_and_original_failure_without_numerics():
    assert ms.STAGE2_G2_WAIVER_CHECKPOINT_SHA256 == "27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035"
    assert ms.STAGE2_G2_WAIVER_SAMPLES_SHA256 == "4cb05b6b3097b22a8594576def1505957399a924d4189591980292e3bd320d30"
    assert ms.STAGE2_G2_WAIVER_FAILURE_SHA256 == "88cc88fb69827f00763b5499e3aa3d39bdb411db3e186660eed596716d6dce72"
    assert ms.sha256_bytes(FAILURE.encode()) == ms.STAGE2_G2_WAIVER_FAILURE_SHA256
    record=waived();before=copy.deepcopy(record)
    assert ms.validate_stage2_g2_evidence(record,**expected(record),demo_tail=16)=="WAIVED"
    assert record==before and "metrics" not in record
    record["metrics"]=None
    assert ms.validate_stage2_g2_evidence(record,**expected(record))=="WAIVED"


@pytest.mark.parametrize("fault",["checkpoint","samples","failure_sha","failure_bytes","failure_reformat","quote","scope",
    "authorized_by","authorized_at","complete","metrics","status_pass","producer","three_key_pin","missing_numerics",
    "missing_vae","changed_pin","changed_numerics","changed_vae","extra_claim","wrong_gate"])
def test_waiver_never_broadens_identity_or_claims_completion(fault):
    record=waived();args=expected(record);exc=record["exception"]
    if fault in ("checkpoint","samples"):
        key=f"{fault}_sha256";record[key]="0"*64;args[key]="0"*64
    elif fault=="failure_sha":exc["original_failure_sha256"]="0"*64
    elif fault=="failure_bytes":exc["original_failure_json"]=FAILURE.rstrip()
    elif fault=="failure_reformat":exc["original_failure_json"]=json.dumps(json.loads(FAILURE),ensure_ascii=False)
    elif fault=="quote":exc["user_quote"]="继续做"
    elif fault=="scope":exc["scope"]="all_gates"
    elif fault=="authorized_by":exc["authorized_by"]="assistant"
    elif fault=="authorized_at":exc["authorized_at"]="2026-10-27T19:21:45Z"
    elif fault=="complete":exc["numeric_validation_complete"]=True
    elif fault=="metrics":record["metrics"]=dict(windows=64,demo_tail=16,worst_mse=0,worst_max=0)
    elif fault=="status_pass":
        record["status"]="PASS";record["metrics"]=dict(windows=64,demo_tail=16,worst_mse=0,worst_max=0)
    elif fault=="producer":record["producer"]="other"
    elif fault=="three_key_pin":
        for group in (args["source_pin"],record["source_pin"]):
            del group["vae_adapter_sha256"];del group["dependency_lock_sha256"]
    elif fault=="missing_numerics":del record["numerics"]
    elif fault=="missing_vae":del record["online_vae"]
    elif fault=="changed_pin":record["source_pin"]["export_source_sha256"]="0"*64
    elif fault=="changed_numerics":record["numerics"]["torch"]="错误版本"
    elif fault=="changed_vae":record["online_vae"]["vae_state_sha256"]="0"*64
    elif fault=="extra_claim":record["windows_validated"]=64
    else:record["gate"]="G3"
    with pytest.raises(ValueError):ms.validate_stage2_g2_evidence(record,**args,demo_tail=16)


@pytest.mark.parametrize("fault",[None,"mse","max","windows","demo_tail","nan","boolean","missing"])
def test_pass_keeps_original_strict_numerical_gate(fault):
    record=waived();record.pop("exception");record.pop("producer");record["status"]="PASS"
    record["metrics"]=dict(windows=64,demo_tail=16,worst_mse=1e-6,worst_max=0.01)
    if fault is None:
        assert ms.validate_stage2_g2_evidence(record,**expected(record),demo_tail=16)=="PASS";return
    if fault=="mse":record["metrics"]["worst_mse"]=1.01e-6
    if fault=="max":record["metrics"]["worst_max"]=0.0101
    if fault=="windows":record["metrics"]["windows"]=1
    if fault=="demo_tail":record["metrics"]["demo_tail"]=17
    if fault=="nan":record["metrics"]["worst_mse"]=float("nan")
    if fault=="boolean":record["metrics"]["worst_mse"]=False
    if fault=="missing":del record["metrics"]
    with pytest.raises(ValueError):ms.validate_stage2_g2_evidence(record,**expected(record),demo_tail=16)


def use_fixture_anchors(monkeypatch,checkpoint,samples):
    # 合成消费链只替换预期对象的摘要；上面的测试单独锁死生产三锚，生产没有覆盖入口。
    monkeypatch.setattr(ms,"STAGE2_G2_WAIVER_CHECKPOINT_SHA256",checkpoint)
    monkeypatch.setattr(ms,"STAGE2_G2_WAIVER_SAMPLES_SHA256",samples)


def test_pack_selects_waiver_and_preserves_original_failure(stage2_fx,monkeypatch):
    s=stage2_fx;g2=json.loads((s.validation/"G2.json").read_text())
    prov=dict(source_pin=g2["source_pin"],numerics=g2["numerics"],stage2_compat=dict(online_vae=g2["online_vae"]))
    use_fixture_anchors(monkeypatch,s.fx.ckpt_sha,g2["samples_sha256"])
    record=waived(prov,checkpoint=s.fx.ckpt_sha,samples=g2["samples_sha256"])
    (s.validation/"G2.json").write_bytes(FAILURE.encode())
    path=s.validation/"G2-waiver.json";write(path,record)
    pack.cmd_pack(s.args);certify(s);pack.cmd_verify(verify_args(s))
    meta=ms.MotionMeta.load(s.args.out);ref=meta.provenance["stage2_compat"]["evidence"]["G2"]
    assert ref["path"]==str(path) and ref["sha256"]==ms.sha256_file(path)
    assert (s.validation/"G2.json").read_bytes()==FAILURE.encode()
    assert json.loads(path.read_text())["status"]=="WAIVED"


def test_invalid_waiver_cannot_fall_back_to_existing_pass(stage2_fx):
    s=stage2_fx;assert json.loads((s.validation/"G2.json").read_text())["status"]=="PASS"
    write(s.validation/"G2-waiver.json",waived())
    with pytest.raises(ValueError):pack.cmd_pack(s.args)
    assert not (Path(s.args.out)/ms.MOTION_TABLE_RELPATH).exists()


def closed_snapshot(snapshot,export,tmp_path,monkeypatch):
    directory=tmp_path/"sources";directory.mkdir()
    prov=client_tests.provenance.__wrapped__(directory)
    prov["encoder"]=copy.deepcopy(snapshot.prov["encoder"])
    prov.update(motion_enabled=True,resolved_sha256=snapshot.prov["resolved_sha256"])
    ckpt=prov["encoder"]["checkpoint_sha256"];evidence=prov["stage2_compat"]["evidence"]
    use_fixture_anchors(monkeypatch,ckpt,evidence["G0"]["sha256"])
    for gate,item in evidence.items():
        value=json.loads((directory/item["path"]).read_text())
        if gate=="G2":value=waived(prov,checkpoint=ckpt,samples=evidence["G0"]["sha256"])
        if gate=="G3":value["checkpoint_sha256"]=ckpt
        path=write(snapshot.run/f"motion_evidence/{gate}.json",value)
        if gate=="G0":
            # G0 文件保留原字节，其余门继续绑定同一摘要。
            path.write_bytes((directory/item["path"]).read_bytes())
        item.update(path=f"motion_evidence/{gate}.json",sha256=ms.sha256_file(path))
    prov["stage2_compat"]["restore_receipt_sha256"]=evidence["restore_receipt"]["sha256"]
    write(snapshot.run/"motion_provenance.json",prov)
    deps=json.loads((snapshot.run/"encoder_dependencies.json").read_text())
    for key in ("encoder","source_pin","vae"):deps[key]=copy.deepcopy(prov[key])
    write(snapshot.run/"encoder_dependencies.json",deps);refresh_ready(snapshot.run,export)
    return prov


def test_client_restore_export_consume_same_waived_bytes(snapshot,export,tmp_path,monkeypatch):
    prov=closed_snapshot(snapshot,export,tmp_path,monkeypatch)
    validate_stage2_evidence(prov,snapshot.run)
    assert restore.verify_snapshot(snapshot.run)==prov
    sources,restored,_=export.snapshot_sources(snapshot.run,snapshot.registered)
    assert restored==prov and len(json.loads((snapshot.run/export.READY).read_text())["files"])==12
    value=json.loads(sources["motion_evidence/G2.json"].read_text())
    assert value["status"]=="WAIVED" and value["exception"]["original_failure_json"]==FAILURE
    assert "metrics" not in value


@pytest.mark.parametrize("gate",["G0","G3"])
def test_other_failed_gates_remain_fatal(snapshot,export,tmp_path,monkeypatch,gate):
    prov=closed_snapshot(snapshot,export,tmp_path,monkeypatch)
    item=prov["stage2_compat"]["evidence"][gate];path=snapshot.run/item["path"]
    value=json.loads(path.read_text());value["status"]="WAIVED";write(path,value)
    item["sha256"]=ms.sha256_file(path);write(snapshot.run/"motion_provenance.json",prov);refresh_ready(snapshot.run,export)
    for invoke in (lambda:validate_stage2_evidence(prov,snapshot.run),lambda:restore.verify_snapshot(snapshot.run),
                   lambda:export.snapshot_sources(snapshot.run,snapshot.registered)):
        with pytest.raises(ValueError):invoke()
