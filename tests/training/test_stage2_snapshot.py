"""完整六组快照、原子发布失败边界及真实小模型内存EMA参考。"""
import copy
import dataclasses
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx,struct

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tests/datastore"))
sys.path.insert(0,str(ROOT/"tests/dataset"))
from test_motion_store_stage2 import stage2_identity
from test_pack_motion_stage2_cached import cached_fx
from mme_vla_suite.datastore import motion_store as ms

spec=importlib.util.spec_from_file_location("train_stage2_snapshot",ROOT/"scripts/training/train.py")
train=importlib.util.module_from_spec(spec);sys.modules[spec.name]=train;spec.loader.exec_module(train)


@pytest.fixture
def snapshot(tmp_path):
    root=tmp_path/"run";root.mkdir()
    (root/"history_config.txt").write_text("合成配置.yaml")
    (root/"history_config.resolved.yaml").write_text("motion: {enabled: true}\n")
    sha=ms.sha256_file(root/"history_config.resolved.yaml")
    (root/"history_config.resolved.sha256").write_text(sha+"\n")
    provenance=stage2_identity();provenance["resolved_sha256"]=sha
    for key,item in provenance["stage2_compat"]["evidence"].items():
        source=tmp_path/f"{key}.json";source.write_text(json.dumps({"验证":key}))
        item.update(path=str(source),sha256=ms.sha256_file(source))
    provenance["stage2_compat"]["restore_receipt_sha256"]=provenance["stage2_compat"]["evidence"]["restore_receipt"]["sha256"]
    arrays = {"images":np.zeros((82,256,256,3),np.uint8),"states":np.zeros((82,8),np.float32),
              "wrist_image":np.zeros((256,256,3),np.uint8)}
    np.savez_compressed(tmp_path/"online_restore_inputs.npz",**arrays)
    portable = dict(schema="stage2-online-restore-inputs-v1",exec_start_idx=17,exec_batch_lengths=[33,32],prompt="合成验证提示",
                    payload=dict(path="online_restore_inputs.npz",sha256=ms.sha256_file(tmp_path/"online_restore_inputs.npz")),
                    sources=[dict(path="仅诊断来源",sha256="a"*64)],
                    tokenizer=dict(uri="gs://big_vision/paligemma_tokenizer.model",sha256="b"*64),
                    arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=ms.sha256_bytes(value.tobytes()))
                            for key,value in arrays.items()})
    (tmp_path/"online_restore_inputs.json").write_text(json.dumps(portable))
    registered=tmp_path/"registered";run=registered/provenance["encoder"]["run_name"];run.mkdir(parents=True)
    for name,key in (("checkpoint_epoch_105.pt","checkpoint_sha256"),("config.yaml","config_sha256")):
        p=run/name;p.write_bytes(name.encode());provenance["encoder"][key]=ms.sha256_file(p)
    return root,registered,provenance


def test_snapshot_roundtrip_and_atomic_ready(snapshot):
    root,registered,provenance=snapshot
    ready=train.write_stage2_motion_snapshot(root,provenance,registered_root=registered)
    saved=json.loads((root/"motion_provenance.json").read_text())
    for key in ms.STAGE2_GROUPS:
        if key!="stage2_compat":assert saved[key]==provenance[key]
    for label,item in saved["stage2_compat"]["evidence"].items():
        assert item["path"]==f"motion_evidence/{label}.json"
        assert (root/item["path"]).read_bytes()==pathlib.Path(provenance["stage2_compat"]["evidence"][label]["path"]).read_bytes()
    assert "motion_snapshot.ready.json" not in ready["files"]
    assert len(ready["files"])==12
    assert all(ms.sha256_file(root/name)==sha for name,sha in ready["files"].items())
    for name in ("online_restore_inputs.json","online_restore_inputs.npz"):
        assert (root/name).read_bytes()==(root.parent/name).read_bytes()
    dependencies=json.loads((root/"encoder_dependencies.json").read_text())
    assert {f["role"] for f in dependencies["files"]}=={"encoder_checkpoint","encoder_config"}
    for f in dependencies["files"]:
        assert ms.sha256_file(f["source_path"])==f["sha256"]
        assert not pathlib.Path(f["relative_path"]).is_absolute()
    with pytest.raises(FileExistsError):train.write_stage2_motion_snapshot(root,provenance,registered_root=registered)


def test_failed_snapshot_never_publishes_ready(snapshot,monkeypatch):
    root,registered,provenance=snapshot
    original=train._atomic_stage2_file
    def fail(path,data):
        if path.name=="encoder_dependencies.json":raise OSError("故障注入")
        return original(path,data)
    monkeypatch.setattr(train,"_atomic_stage2_file",fail)
    with pytest.raises(OSError):train.write_stage2_motion_snapshot(root,provenance,registered_root=registered)
    assert not (root/"motion_snapshot.ready.json").exists()


def test_cached_snapshot_preserves_evidence_and_streaming_batches(snapshot, cached_fx):
    from test_pack_motion_stage2_cached import pack, load, dump, verify_args
    from mme_vla_suite.policies.motion_client import validate_stage2_evidence
    root, _, _ = snapshot
    s = cached_fx
    pack.cmd_pack(s.args)
    pack.cmd_verify(verify_args(s))
    provenance = ms.MotionMeta.load(s.args.out).provenance
    provenance["resolved_sha256"] = ms.sha256_file(root / "history_config.resolved.yaml")
    portable = load(root.parent / "online_restore_inputs.json")
    portable["exec_batch_lengths"] = [1, 16, 16, 16, 16]
    dump(s.validation / "online_restore_inputs.json", portable)
    (s.validation / "online_restore_inputs.npz").write_bytes((root.parent / "online_restore_inputs.npz").read_bytes())
    train.write_stage2_motion_snapshot(root, provenance, registered_root=s.fx.enc_dir.parent)
    saved = load(root / "motion_provenance.json")
    validate_stage2_evidence(saved, root)
    assert ms.stage2_uses_cache(saved)
    assert set(saved["stage2_compat"]["evidence"]) == set(ms.stage2_evidence_roles(saved))
    assert not (root / "motion_evidence/restore_receipt.json").exists()
    assert load(root / "online_restore_inputs.json")["exec_batch_lengths"] == [1, 16, 16, 16, 16]


@pytest.mark.parametrize("fault",["missing_group","changed_evidence","symlink_dependency","changed_dependency","portable_sha","array_sha"])
def test_snapshot_source_negatives(snapshot,fault):
    root,registered,provenance=snapshot
    if fault=="missing_group":del provenance["numerics"]
    elif fault=="changed_evidence":pathlib.Path(provenance["stage2_compat"]["evidence"]["G2"]["path"]).write_text("变动")
    elif fault in ("portable_sha","array_sha"):
        path=root.parent/"online_restore_inputs.json";value=json.loads(path.read_text())
        if fault=="portable_sha":value["payload"]["sha256"]="0"*64
        else:value["arrays"]["states"]["sha256"]="0"*64
        path.write_text(json.dumps(value))
    else:
        p=registered/provenance["encoder"]["run_name"]/"config.yaml"
        if fault=="changed_dependency":p.write_text("变动")
        else:
            target=p.with_suffix(".real");p.rename(target);p.symlink_to(target)
    with pytest.raises(ValueError):train.write_stage2_motion_snapshot(root,provenance,registered_root=registered)
    assert not (root/"motion_snapshot.ready.json").exists()


@struct.dataclass
class TinyObservation:
    state:object
    def to_dict(self):return {"state":self.state,"none":None}


class TinyModel(nnx.Module):
    def __init__(self,value):self.kernel=nnx.Param(value)
    def sample_actions(self,rng,observation,*,num_steps=10):
        return observation.state @ self.kernel.value + jax.random.normal(rng,(1,2))/num_steps


class TinyConfig:
    def load(self,params,*,remove_extra_params):
        assert remove_extra_params is False
        return TinyModel(params["kernel"])


def test_pre_save_reference_uses_live_memory_ema_and_exact_arrays(tmp_path):
    ema=nnx.state(TinyModel(jnp.arange(4,dtype=jnp.float32).reshape(2,2)))
    state=SimpleNamespace(ema_params=ema,step=20,params=nnx.state(TinyModel(jnp.zeros((2,2)))))
    config=SimpleNamespace(num_train_steps=20,model=TinyConfig())
    observation=TinyObservation(jnp.asarray([[2.,3.]]))
    record=train.write_stage2_pre_save_reference(tmp_path,config,state,observation,step=19)
    assert record["parameters"]["leaves"]["kernel"]["sha256"]==ms.sha256_bytes(np.arange(4,dtype=np.float32).tobytes())
    assert record["fp32_ema"]==record["parameters"]
    assert record["state_step"]==20 and record["inference"]["sharding"]["batch_size"]==1
    for key in ("inputs","outputs"):
        info=record[key];assert ms.sha256_file(tmp_path/info["path"])==info["sha256"]
        with np.load(tmp_path/info["path"],allow_pickle=False) as data:
            for name,meta in info["leaves"].items():assert ms.sha256_bytes(data[name].tobytes())==meta["sha256"]
    expected=TinyModel(jnp.arange(4,dtype=jnp.bfloat16).reshape(2,2)).sample_actions(
        jax.random.split(jax.random.key(42))[1],observation,num_steps=10)
    with np.load(tmp_path/record["outputs"]["path"]) as data:
        np.testing.assert_array_equal(data["leaf_0000"],np.asarray(expected))
    with pytest.raises(FileExistsError):train.write_stage2_pre_save_reference(tmp_path,config,state,observation,step=19)


def test_bfloat16_summary_keeps_original_bits():
    value=jnp.asarray([1.,-0.],dtype=jnp.bfloat16)
    summary,fp32=train.stage2_parameter_summary({"x":value})
    assert summary["leaves"]["x"]["dtype"]=="bfloat16" and not fp32["leaves"]
    assert summary["leaves"]["x"]["sha256"]==ms.sha256_bytes(np.asarray(value).tobytes())
