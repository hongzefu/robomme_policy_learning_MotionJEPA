"""实际配置、loader和mesh交叉核验；具名默认无需重复写进CLI。"""

import hashlib
import json
from pathlib import Path
import sys
import subprocess
from copy import deepcopy

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import check_modul_train_records as checker


def fixture(root):
    source = str(Path(__file__).resolve().parents[3])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    params = [f"['img']['p{i}']" for i in range(23)] + [f"['p{i}']" for i in range(38)]
    keys = [prefix+path for prefix in ("params","ema_params") for path in params]
    keys += [f"opt_state[1][0].{moment}['p{i}']" for moment in ("mu","nu") for i in range(38)]
    keys += ["step","opt_state[1][0].count","opt_state[1][2].count"]
    per_leaf = {key:"a"*64 for key in keys}
    digest = hashlib.sha256("".join(f"{key}:{per_leaf[key]}\n" for key in sorted(keys)).encode()).hexdigest()
    states = [{"state_step":s,"phase":"init" if s==0 else "post_update","loop_step":None if s==0 else s-1,
               "per_leaf":per_leaf,"per_leaf_finite":{key:True for key in keys},"n_leaves":201,"state_digest":digest} for s in (0,1)]
    norm = {str(i):"norm" for i in range(8)}
    actual = {"num_train_steps":1,"batch_size":128,"num_workers":16,"seed":42,"fsdp_devices":8}
    metadata = {"start_status":"","source_status":"","start_head":head,"source_head":head,"source_root":source,
        "import_origins":{key:source+"/src/"+key+".py" for key in ("train","framesamp_dataset","framesamp_store","mme_vla_suite")},
        "norm_stats_file_sha256":checker.NORM_SHA,"norm_stats_actual":norm,"norm_stats_expected":norm,
        "argv":["bench_train_steps.py","mme_vla_suite_b128_80k","--num-train-steps","1"],
        "actual_config":{"config_record_version":2,"train_config":{"fields":actual},"history_values":{"budget":2048}},
        "batch_size":128,"loader":{"batch_size":128,"workers":16,"prefetch_factor":2,"persistent_workers":True,"pin_memory":False},
        "environment":{"CUDA_VISIBLE_DEVICES":"0,1,2,3,4,5,6,7","XLA_FLAGS":"--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"},
        "bench_checksum_enabled":True,"bench_batch_digests_enabled":True,"bench_dump_idx_enabled":True,
        "checksum_workers":8,"digest_interval_effective":1,"extra_digest_steps":[],"state_dump_steps":[],
        "history_config_resolved_sha256":"b"*64,"history_config":"perceptual-framesamp-modul-32frame-8x8.yaml"}
    runtime = {**actual,"mesh":{"batch":1,"fsdp":8},"device_count":8,"cuda_visible_devices":"0,1,2,3,4,5,6,7",
               "devices":[f"CUDA_{i}" for i in range(8)]}
    indices = list(range(34*128))
    rows = [{"batch":i,"indices":indices[i*128:(i+1)*128]} for i in range(34)]
    batch = {"step":0,"sample_indices":indices[:128],"static_mask_counts":[2048]*128,
        "per_key":{**{f"['{key}']":None for key in ("motion_emb","motion_pos","motion_mask","mem_order")},
        "['static_mask']":hashlib.sha256(b"bool"+str((128,2048)).encode()+bytes([1])*128*2048).hexdigest()}}
    for name,value in (("run_meta.json",metadata),("runtime.json",runtime),
        ("index_sequence.json",{"n":len(indices),"indices":indices,"indices_sha256":hashlib.sha256(json.dumps(indices).encode()).hexdigest()})):
        (root/name).write_text(json.dumps(value))
    for name,values in (("param_checksums.jsonl",states),("idx_seq.jsonl",rows),("batch_digests.jsonl",[batch]),
        ("metrics.jsonl",[{"step":0,"wall_time":0.,**{key:{"dec":1.,"hex":float(1).hex()} for key in checker.SCALARS}}])):
        (root/name).write_text("".join(json.dumps(value)+"\n" for value in values))
    return metadata,runtime


def test_inherited_configuration_and_real_runtime_accepted(tmp_path):
    fixture(tmp_path)
    checker.validate(tmp_path,1,128,"0,1,2,3,4,5,6,7",16,8,2048)


@pytest.mark.parametrize("fault",["explicit_batch","workers","seed","fsdp","devices","budget","mask_counts"])
def test_wrong_actual_configuration_rejected(tmp_path,fault):
    meta,runtime = fixture(tmp_path)
    fields = meta["actual_config"]["train_config"]["fields"]
    if fault == "explicit_batch":
        meta["argv"] += ["--batch-size","64"]
        fields["batch_size"] = meta["batch_size"] = runtime["batch_size"] = meta["loader"]["batch_size"] = 64
    elif fault == "workers": meta["loader"]["workers"] = 4
    elif fault == "seed": fields["seed"] = 1
    elif fault == "fsdp": runtime["mesh"]["fsdp"] = 2
    elif fault == "devices": runtime["devices"][-1] = runtime["devices"][0]
    elif fault == "budget": meta["actual_config"]["history_values"]["budget"] = 4096
    else:
        path=tmp_path/"batch_digests.jsonl"; row=json.loads(path.read_text());row["static_mask_counts"][0]=0
        path.write_text(json.dumps(row)+"\n")
    (tmp_path/"run_meta.json").write_text(json.dumps(meta))
    (tmp_path/"runtime.json").write_text(json.dumps(runtime))
    with pytest.raises(ValueError):
        checker.validate(tmp_path,1,128,"0,1,2,3,4,5,6,7",16,8,2048)


@pytest.mark.parametrize("key,value", [("checksum_workers", 1), ("digest_interval_effective", 2),
    ("bench_checksum_enabled", False), ("bench_batch_digests_enabled", False),
    ("bench_dump_idx_enabled", False), ("extra_digest_steps", [1]), ("state_dump_steps", [0])])
def test_mixed_recorder_settings_rejected(tmp_path, key, value):
    meta, _ = fixture(tmp_path)
    changed = deepcopy(meta)
    changed[key] = value
    with pytest.raises(ValueError, match="取证源码或设置不同"):
        checker.same_recorder({"recorder": checker.recorder_identity(meta)},
                              {"recorder": checker.recorder_identity(changed)})


def test_recorder_source_changes_rejected(tmp_path, monkeypatch):
    meta, _ = fixture(tmp_path)
    first = checker.recorder_identity(meta)
    original = checker.subprocess.run
    def changed_source(command, **kwargs):
        result = original(command, **kwargs)
        if command[-1].endswith("state_checksum.py"):
            result.stdout += "\n# 测试注入不同取证器\n".encode()
        return result
    monkeypatch.setattr(checker.subprocess, "run", changed_source)
    with pytest.raises(ValueError, match="取证源码或设置不同"):
        checker.same_recorder({"recorder": first}, {"recorder": checker.recorder_identity(meta)})


def test_missing_recorder_identity_rejected(tmp_path):
    meta, _ = fixture(tmp_path)
    meta.pop("checksum_workers")
    with pytest.raises(ValueError, match="缺取证设置"):
        checker.recorder_identity(meta)


def test_training_revision_is_separate_from_recorder(tmp_path):
    meta, _ = fixture(tmp_path)
    other = deepcopy(meta)
    other["source_head"] = "b" * 40
    checker.same_recorder({"recorder": checker.recorder_identity(meta)},
                          {"recorder": checker.recorder_identity(other)})
