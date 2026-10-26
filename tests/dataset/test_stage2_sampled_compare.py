"""抽样G5的CPU合成合同；不执行GPU或改写正式数据。"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mme_vla_suite.datastore import motion_store as ms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dataset/wan_stage2"))
import compare_stage2 as C


def entry(task="Task", row_base=0, exec_length=65, global_id=0):
    spec = ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT]
    demo_rows = ms.seg_num_grid(49, 17, 32)
    exec_rows = ms.seg_num_grid(exec_length, 33, 32)
    return ms.IndexEntry(g=global_id, h5_file=f"record_dataset_{task}.h5", raw_ep_idx=0,
                         num_timesteps=49+exec_length, exec_start_idx=49, spec=spec,
                         demo=ms.SegmentInfo(row_base=row_base, num_grid=demo_rows, num_chunks=33, seg_len=49),
                         exec=ms.SegmentInfo(row_base=row_base+demo_rows, num_grid=exec_rows,
                                             num_chunks=exec_length-32, seg_len=exec_length))


@pytest.fixture
def dataset(tmp_path):
    latent_root, tokens_root = tmp_path / "latents", tmp_path / "tokens"
    latent_root.mkdir()
    tokens_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b'{"synthetic":true}\n')
    manifest_sha = C.V.sha_file(manifest_path)
    identity = dict(encoder=dict(checkpoint_sha256="1"*64), encoder_state_sha256=dict(sha256="2"*64),
                    numerics=dict(dtype="float32", batch=1), source_pin=dict(mj_repo_commit="3"*40,
                    export_source_sha256="4"*64, encoder_source_sha256="5"*64))
    current = entry()
    total, files, table, keys = {}, [], [], []
    for kind in ("demo", "exec"):
        part = getattr(current, kind)
        key = ms.segment_key(current.h5_file, 0, kind)
        keys.append(key)
        start = 0 if kind == "demo" else 49
        old_count = ms.seg_num_grid(part.seg_len, current.spec.min_real(kind), 16)
        rows = []
        for grid in range(old_count):
            real = min(33, part.seg_len-16*grid)
            rows.append(dict(m=grid, seg_offset=16*grid, start_global_frame=start+16*grid,
                             real_frames=real, pad_frames=33-real,
                             pad_source_frame=start+part.seg_len-1 if real<33 else None,
                             input_shape=[33,256,256,3], input_dtype="uint8", input_frames_sha256="6"*64))
        raw = np.stack([np.full(C.V.LATENT_SHAPE, part.row_base+grid, np.float32) for grid in range(old_count)])
        latent_path = latent_root / f"{key}.bin"
        latent_path.write_bytes(raw.tobytes())
        digest = C.V.sha_file(latent_path)
        Path(str(latent_path)+".sha256").write_text(f"{digest}  {latent_path.name}\n")
        lm = dict(segment=key, segment_kind=kind, seg_start_global=start, seg_len=part.seg_len,
                  g=current.g, h5_file=current.h5_file, raw_ep_idx=current.raw_ep_idx,
                  num_grid=old_count, grid_stride=16, window_frames=33, rows=rows,
                  bytes=raw.nbytes, sha256=digest)
        latent_meta = latent_root / f"{key}.metadata.json"
        C.V.atomic_json(latent_meta,lm)
        total[key] = copy.deepcopy(lm)
        token = np.stack([np.full(1024, part.row_base+i, np.float32) for i in range(part.num_grid)])
        token_path = tokens_root / f"{key}.f32.bin"
        token_path.write_bytes(token.tobytes())
        token_sha = C.V.sha_file(token_path)
        Path(str(token_path)+".sha256").write_text(f"{token_sha}  {token_path.name}\n")
        tm = dict(schema="stage2-memory-export-v1", layout=C.V.LAYOUT, segment=key, segment_kind=kind,
                  g=current.g, h5_file=current.h5_file, raw_ep_idx=current.raw_ep_idx,
                  seg_start_global=start, seg_len=part.seg_len, row_shape=[1,1024], dtype="float32",
                  token_dim=1024, token_bytes=4096, bytes=token.nbytes, num_grid=part.num_grid,
                  grid_stride=32, window_frames=33, source_grid=list(range(0,old_count,2)),
                  rows=rows[::2], sha256=token_sha, manifest_sha256=manifest_sha,
                  input_latent_sha256=digest, input_metadata_sha256=C.V.sha_file(latent_meta),
                  input_block_sha256=[C.V.sha_bytes(value.tobytes()) for value in raw[::2]], **identity)
        token_meta = tokens_root / f"{key}.metadata.json"
        C.V.atomic_json(token_meta,tm)
        files.append(dict(key=key, sha256=token_sha, num_grid=part.num_grid,
                          metadata_sha256=C.V.sha_file(token_meta)))
        table.extend(token)
    C.V.atomic_json(latent_root / "metadata.json",dict(segments=total))
    C.V.atomic_json(tokens_root / "export_record.json",dict(schema="stage2-memory-export-v1",scope="full",
                   segments=2,rows=len(table),segment_keys=keys,manifest_sha256=manifest_sha,files=files,**identity))
    prov = copy.deepcopy(identity)
    prov["source_pin"].update(vae_adapter_sha256="7"*64,dependency_lock_sha256="8"*64)
    prov.update(tokens_root=str(tokens_root),episode_manifest_file_sha256=manifest_sha,
                latents_metadata_sha256=C.V.sha_file(latent_root / "metadata.json"))
    meta = SimpleNamespace(entries=[current],num_rows=len(table),provenance=prov,manifest_path=str(manifest_path))
    return SimpleNamespace(meta=meta,table=np.stack(table),latent_root=latent_root,tokens_root=tokens_root,keys=keys)


def run_cpu(data):
    return C.cpu_full_integrity(meta=data.meta,table=data.table,latent_root=data.latent_root)


def test_cpu_really_covers_all_rows_and_source_files(dataset):
    result = run_cpu(dataset)
    assert result["packed_compared"] == 4 and result["packed_mismatches"] == 0
    assert result["source_grid"] == {key:[0,2] for key in dataset.keys}
    assert result["source_tokens_manifest_sha256"] == C.V.sha_file(dataset.tokens_root / "export_record.json")
    assert len(result["source_stamps"]) == 15


@pytest.mark.parametrize("row",range(4))
def test_any_cpu_row_mismatch_is_rejected(dataset,row):
    dataset.table.view(np.uint32)[row,0] ^= 1
    with pytest.raises(ValueError,match="原始字节"):
        run_cpu(dataset)


@pytest.mark.parametrize("damage",["extra_token","missing_token","token_sidecar","latent_sidecar","metadata_sha",
                                    "root_manifest","root_file_missing","wrong_identity","wrong_episode","tail_padding","odd_latent"])
def test_source_integrity_failures(dataset,damage):
    key=dataset.keys[0]
    token=dataset.tokens_root / f"{key}.f32.bin"
    token_meta=dataset.tokens_root / f"{key}.metadata.json"
    root=dataset.tokens_root / "export_record.json"
    if damage=="extra_token":
        (dataset.tokens_root / "unknown.f32.bin").write_bytes(b"x")
    elif damage=="missing_token":
        token.unlink()
    elif damage=="token_sidecar":
        Path(str(token)+".sha256").write_text("0"*64)
    elif damage=="latent_sidecar":
        (dataset.latent_root / f"{key}.bin.sha256").write_text("0"*64)
    elif damage=="odd_latent":
        with (dataset.latent_root / f"{key}.bin").open("r+b") as stream:
            stream.seek(C.V.CHUNK_BYTES)
            stream.write(np.float32(123).tobytes())
    elif damage in ("root_manifest","root_file_missing"):
        record=C.V.read_json(root)
        if damage=="root_manifest": record["manifest_sha256"]="0"*64
        else: record["files"].pop()
        C.V.atomic_json(root,record)
    else:
        record=C.V.read_json(token_meta)
        if damage=="metadata_sha": record["worker"]={"changed":True}
        elif damage=="wrong_identity": record["encoder"]["checkpoint_sha256"]="0"*64
        elif damage=="wrong_episode": record["g"]=3
        else: record["rows"][-1]["pad_source_frame"]=49
        C.V.atomic_json(token_meta,record)
        if damage=="wrong_episode":
            export=C.V.read_json(root)
            export["files"][0]["metadata_sha256"]=C.V.sha_file(token_meta)
            C.V.atomic_json(root,export)
    with pytest.raises((ValueError,FileNotFoundError)):
        run_cpu(dataset)


def test_source_change_after_full_hash_is_rejected(dataset):
    result=run_cpu(dataset)
    path=dataset.latent_root / f"{dataset.keys[0]}.bin"
    with path.open("r+b") as stream: stream.write(np.float32(999).tobytes())
    with pytest.raises(ValueError,match="期间"):
        C.check_stamps(result["source_stamps"])


def test_encoder_really_called_exactly_1024_times(tmp_path):
    rows=1040
    path=tmp_path / "Task_ep0_exec.bin"
    with path.open("wb") as stream: stream.truncate((2*rows-1)*C.V.CHUNK_BYTES)
    indices=list(range(0,1024))
    with path.open("r+b") as stream:
        for row in indices:
            stream.seek(2*row*C.V.CHUNK_BYTES)
            stream.write(np.float32(row).tobytes())
    table=np.repeat(np.arange(rows,dtype=np.float32)[:,None],1024,axis=1)
    plan=dict(seed=0,sample_indices=indices,sample_indices_sha256=C.V.sha_bytes(C.V.canonical(indices)))
    calls=[]

    def encode(latent):
        value=float(latent[0,0,0,0])
        calls.append(int(value))
        return np.full(1024,value,np.float32)

    assert C.compare_sampled_rows(table=table,segments=[("Task_ep0_exec",0,rows)],encode=encode,
                                  latent_root=tmp_path,sample_plan=plan)==1024
    assert calls==indices


@pytest.mark.parametrize("damage",["duplicate","out_of_range","short","sha","seed"])
def test_bad_sample_plan_never_calls_encoder(tmp_path,damage):
    indices=list(range(1024))
    plan=dict(seed=0,sample_indices=indices,sample_indices_sha256=C.V.sha_bytes(C.V.canonical(indices)))
    if damage=="duplicate": indices[-1]=0
    elif damage=="out_of_range": indices[-1]=1024
    elif damage=="short": indices.pop()
    elif damage=="sha": plan["sample_indices_sha256"]="0"*64
    else: plan["seed"]=1

    def forbidden(value):
        raise AssertionError("坏抽样计划不得执行前向")

    with pytest.raises(ValueError,match="共享抽样计划"):
        C.compare_sampled_rows(table=np.zeros((1024,1024),np.float32),segments=[],encode=forbidden,
                               latent_root=tmp_path,sample_plan=plan)


def test_shared_plan_has_fixed_coverage_when_integrated():
    if not hasattr(ms,"stage2_encoder_sample_plan"):
        pytest.skip("共享选窗函数由S2独立分支提供，合并后再验证")
    entries=[entry(task=f"Task{i:02d}",row_base=i*65,exec_length=2017,global_id=i) for i in range(16)]
    plan=ms.stage2_encoder_sample_plan(entries)
    assert len(plan["sample_indices"])==1024 and len(set(plan["sample_indices"]))==1024
    assert plan["task_counts"]=={f"Task{i:02d}":64 for i in range(16)}
    assert plan["padded_count"]==16 and plan["demo_count"]==32 and plan["exec_count"]==992
    assert plan==ms.stage2_encoder_sample_plan(entries)
    assert C.AUTHORIZATION_QUOTE==ms.STAGE2_SAMPLE_AUTHORIZATION


@pytest.mark.parametrize("schema",[C.SCHEMA,C.SAMPLED_SCHEMA])
def test_sampled_interrupt_revokes_own_record_and_preserves_pack_lock(tmp_path,monkeypatch,schema):
    (tmp_path / "meta").mkdir()
    record=tmp_path / "meta/encoder_bitexact.json"
    pack_lock=tmp_path / "meta/pack.lock"
    C.V.atomic_json(record,dict(schema=schema,status="PASS"))
    pack_lock.write_bytes(b"original-pack-lock")

    def interrupted(root):
        raise KeyboardInterrupt

    monkeypatch.setattr(ms.MotionMeta,"load",interrupted)
    with pytest.raises(KeyboardInterrupt):
        C.certify(SimpleNamespace(store=tmp_path,scope="sampled",batch=1))
    assert not record.exists() and not (tmp_path / "meta/encoder_bitexact.lock").exists()
    assert pack_lock.read_bytes()==b"original-pack-lock"


@pytest.mark.parametrize("damage",["one_bit","nan","short"])
def test_sampled_output_or_input_damage_fails(tmp_path,damage):
    path=tmp_path / "Task_ep0_exec.bin"
    path.write_bytes(b"short" if damage=="short" else np.zeros(C.V.LATENT_SHAPE,np.float32).tobytes())
    indices=list(range(1024))
    plan=dict(seed=0,sample_indices=indices,sample_indices_sha256=C.V.sha_bytes(C.V.canonical(indices)))
    calls=[]

    def encode(value):
        calls.append(1)
        output=np.zeros(1024,np.float32)
        if damage=="nan": output[0]=np.nan
        else: output.view(np.uint32)[0]=1
        return output

    with pytest.raises(ValueError):
        C.compare_sampled_rows(table=np.zeros((1024,1024),np.float32),segments=[("Task_ep0_exec",0,1024)],
                               encode=encode,latent_root=tmp_path,sample_plan=plan)
    assert len(calls)==(0 if damage=="short" else 1)
