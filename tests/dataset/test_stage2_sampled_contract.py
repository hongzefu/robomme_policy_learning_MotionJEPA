"""分层抽样、全表CPU覆盖与可携带认证合同的有界CPU正负例。"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
for relative in ("tests/datastore","scripts/dataset"):
    sys.path.insert(0,str(ROOT/relative))
from test_motion_store_stage2 import manifest_for,stage2_identity
import pack_motion_store as pack
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import manifest_sha256


def sample_entries():
    # 每任务8个短demo补帧段，加一份长demo和长exec；任务编号故意与词典序不同。
    specs=[(50,17)]*8+[(4098,2049)]
    manifest=manifest_for(specs*16)
    for i,ep in enumerate(manifest["episodes"]):
        ep["h5_file"]=f"record_dataset_Task{15-i//9}.h5"
        ep["raw_ep_idx"]=i%9
    manifest["canonical_order"]=list(dict.fromkeys(ep["h5_file"] for ep in manifest["episodes"]))
    manifest["sha256"]=manifest_sha256(manifest)
    return manifest,ms.build_index_entries(manifest,ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT])


def test_exact_deterministic_plan_and_strata():
    _,entries=sample_entries();plan=ms.stage2_encoder_sample_plan(entries)
    assert plan==ms.stage2_encoder_sample_plan(entries)
    assert len(plan["sample_indices"])==1024 and set(plan["task_counts"].values())=={64}
    assert plan["strategy"]=="task64-endpoints-pad8-seed0-v1" and plan["seed"]==0
    selected=set(plan["sample_indices"])
    for task in plan["task_counts"]:
        pools={"demo":[],"exec":[],"pad":[],"plain":[]}
        for e in entries:
            if ms.task_of_h5(e.h5_file)!=task:continue
            for kind in ms.SEGMENTS:
                seg=getattr(e,kind)
                for j in range(seg.num_grid):
                    row=seg.row_base+j;pools[kind].append(row)
                    if kind=="demo":pools["pad" if seg.seg_len-j*32<33 else "plain"].append(row)
        for kind in ms.SEGMENTS:
            assert {pools[kind][0],pools[kind][-1]}<=selected
        assert len(selected.intersection(pools["pad"]))>=min(8,len(pools["pad"]))
        assert len(selected.intersection(pools["exec"]))>=min(16,len(pools["exec"]))
        assert len(selected.intersection(pools["plain"]))>=min(16,len(pools["plain"]))
    assert plan["demo_count"]+plan["exec_count"]==1024
    assert plan["padded_count"]==128
    with pytest.raises(ValueError):ms.stage2_encoder_sample_plan(entries[:-9])
    _,too_short=sample_entries();too_short=[e for e in too_short if e.raw_ep_idx!=8]
    with pytest.raises(ValueError):ms.stage2_encoder_sample_plan(too_short)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(ms.canonical_json_bytes(value)+b"\n")


@pytest.fixture
def store(tmp_path):
    manifest,entries=sample_entries();spec=entries[0].spec;rows=ms.index_totals(entries)["rows"]
    root=tmp_path/"store";(root/"meta").mkdir(parents=True)
    latent=tmp_path/"latents";latent.mkdir();tokens=tmp_path/"tokens";tokens.mkdir()
    manifest_path=tmp_path/"episode_manifest.json";write(manifest_path,manifest)
    index=ms.index_payload(manifest,entries,spec=spec,layout=ms.STAGE2_LAYOUT,mj_repo_commit="a"*40)
    write(root/ms.INDEX_RELPATH,index)
    prov=stage2_identity();total={"segments":{}};mapping={}
    table=bytearray()
    for e,kind,key,base,count in pack.iter_rows_in_order(entries):
        seg=getattr(e,kind);start=0 if kind=="demo" else e.exec_start_idx
        old_count=ms.seg_num_grid(seg.seg_len,spec.min_real(kind));old_rows=[]
        for g in range(old_count):
            real=min(33,seg.seg_len-g*16);pad=33-real
            old_rows.append(dict(m=g,seg_offset=g*16,start_global_frame=start+g*16,real_frames=real,pad_frames=pad,
               pad_source_frame=start+seg.seg_len-1 if pad else None,input_shape=[33,256,256,3],input_dtype="uint8",input_frames_sha256="a"*64))
        lm=dict(segment=key,segment_kind=kind,seg_start_global=start,seg_len=seg.seg_len,num_grid=old_count,
                grid_stride=16,window_frames=33,rows=old_rows)
        write(latent/f"{key}.metadata.json",lm);total["segments"][key]={"rows":old_rows}
        grid=list(range(0,old_count,2));mapping[key]=grid
        data=np.full((count,1,1024),base,dtype=np.float32).tobytes();table.extend(data)
        (tokens/f"{key}.f32.bin").write_bytes(data)
        (tokens/f"{key}.f32.bin.sha256").write_text(ms.sha256_bytes(data)+"\n")
        write(tokens/f"{key}.metadata.json",dict(lm,schema="stage2-memory-export-v1",layout=ms.STAGE2_LAYOUT,
            num_grid=count,grid_stride=32,row_shape=[1,1024],dtype="float32",token_dim=1024,token_bytes=4096,
            bytes=len(data),sha256=ms.sha256_bytes(data),source_grid=grid,rows=[old_rows[g] for g in grid]))
    write(latent/"metadata.json",total);write(tokens/"export_record.json",{"fixture":"仅CPU合成"})
    (root/ms.MOTION_TABLE_RELPATH).write_bytes(table)
    for label,item in prov["stage2_compat"]["evidence"].items():
        p=tmp_path/f"{label}.json";write(p,{"fixture":label});item.update(path=str(p),sha256=ms.sha256_file(p))
    prov["stage2_compat"]["restore_receipt_sha256"]=prov["stage2_compat"]["evidence"]["restore_receipt"]["sha256"]
    prov.update(manifest_sha256=manifest["sha256"],motion_index_sha256=ms.sha256_file(root/ms.INDEX_RELPATH),
        latents_metadata_sha256=ms.sha256_file(latent/"metadata.json"),source_grid_sha256=ms.sha256_bytes(ms.canonical_json_bytes(mapping)),
        latents_root=str(latent),tokens_root=str(tokens))
    raw=dict(schema=1,layout=ms.STAGE2_LAYOUT,status="packed",manifest_scope="full",**spec.fields(),
        byte_order="little",array_order="C",grid_stride=32,window_frames=33,grid_origin="segment_start",window_direction="forward",
        truncation_policy="none",frame_size=256,num_rows=rows,totals=ms.index_totals(entries),manifest_sha256=manifest["sha256"],
        manifest_path=str(manifest_path),motion_index_sha256=prov["motion_index_sha256"],provenance=prov,
        tables={ms.MOTION_KEY:dict(row_shape=[1,1024],dtype="float32",row_bytes=4096,num_rows=rows,relpath=ms.MOTION_TABLE_RELPATH,
                                  byte_count=len(table),sha256=ms.sha256_bytes(table))})
    write(root/ms.META_RELPATH,raw)
    record=dict(schema="stage2-encoder-sampled-v1",scope="sampled",status="PASS",layout=ms.STAGE2_LAYOUT,
        compared=1024,total_rows=rows,mismatches=0,packed_compared=rows,packed_mismatches=0,
        **ms.stage2_encoder_sample_plan(entries),authorization_quote=ms.STAGE2_SAMPLE_AUTHORIZATION,
        store_table_sha256=ms.sha256_bytes(table),motion_index_sha256=prov["motion_index_sha256"],
        checkpoint_sha256=prov["encoder"]["checkpoint_sha256"],source_grid_sha256=prov["source_grid_sha256"],
        source_metadata_sha256=prov["latents_metadata_sha256"],source_tokens_manifest_sha256=ms.sha256_file(tokens/"export_record.json"),
        encoder_state_sha256=prov["encoder_state_sha256"],numerics=prov["numerics"],segments=len(mapping))
    write(root/pack.BITEXACT_RELPATH,record)
    return root,raw,record


def test_sampled_verify_is_still_full_cpu_and_carries_exact_text(store):
    from types import SimpleNamespace
    root,raw,record=store;original=(root/pack.BITEXACT_RELPATH).read_bytes()
    assert pack.check_bitexact_record(root,raw)==record
    pack.cmd_verify(SimpleNamespace(store=str(root),manifest=None,tokens=None,resume=True,force_break_lock=False,check_padding=True))
    meta=ms.MotionMeta.load(root);summary=meta.provenance["stage2_compat"]["encoder_validation"]
    assert meta.status=="verified" and meta.raw["verify"]["scanned"]==raw["num_rows"]
    assert summary["encoder_compared"]==1024 and summary["packed_compared"]==raw["num_rows"]
    assert summary["record_json"].encode()==original and summary["record_sha256"]==ms.sha256_bytes(original)
    portable={k:copy.deepcopy(meta.provenance[k]) for k in ms.STAGE2_GROUPS}
    portable["motion_table_sha256"]=record["store_table_sha256"]
    portable["motion_index_sha256"]=record["motion_index_sha256"]
    # 携带校验没有原机root/latents路径，仍可严格校范围与原文摘要。
    ms.validate_stage2_provenance(portable)
    assert pack.check_bitexact_record(root,meta.raw)==record


@pytest.mark.parametrize("fault",["partial","cpu_partial","cpu_mismatch","wrong_indices","seed","task_counts","quote",
                                   "metadata_sha","tokens_sha","state","numerics","packed_bytes","index_bytes"])
def test_sampled_certificate_rejects_wrong_coverage_or_identity(store,fault):
    root,raw,record=store
    if fault=="partial":record["compared"]=100
    if fault=="cpu_partial":record["packed_compared"]-=1
    if fault=="cpu_mismatch":record["packed_mismatches"]=1
    if fault=="wrong_indices":
        selected=set(record["sample_indices"]);replacement=next(i for i in range(raw["num_rows"]) if i not in selected)
        selected.remove(min(selected));selected.add(replacement)
        record["sample_indices"]=sorted(selected);record["sample_indices_sha256"]=ms.sha256_bytes(ms.canonical_json_bytes(sorted(selected)))
    if fault=="seed":record["seed"]=1
    if fault=="task_counts":record["task_counts"][next(iter(record["task_counts"]))]=63
    if fault=="quote":record["authorization_quote"]="同意"
    if fault=="metadata_sha":record["source_metadata_sha256"]="0"*64
    if fault=="tokens_sha":record["source_tokens_manifest_sha256"]="0"*64
    if fault=="state":record["encoder_state_sha256"]={}
    if fault=="numerics":record["numerics"]={}
    if fault=="packed_bytes":
        p=root/ms.MOTION_TABLE_RELPATH;data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
    if fault=="index_bytes":
        with (root/ms.INDEX_RELPATH).open("ab") as stream:stream.write(b" ")
    write(root/pack.BITEXACT_RELPATH,record)
    with pytest.raises((ValueError,RuntimeError)):pack.check_bitexact_record(root,raw)


@pytest.mark.parametrize("fault",["raw_sha","raw_text","total","encoder_count","packed_count","sample_sha","quote","full_claim"])
def test_portable_summary_rejects_misrepresentation(store,fault):
    root,raw,record=store
    summary=pack._encoder_validation_summary((root/pack.BITEXACT_RELPATH).read_bytes(),record,raw["num_rows"],raw["num_rows"])
    if fault=="raw_sha":summary["record_sha256"]="0"*64
    if fault=="raw_text":summary["record_json"]+=" "
    if fault=="total":summary["total_rows"]+=1
    if fault=="encoder_count":summary["encoder_compared"]=raw["num_rows"]
    if fault=="packed_count":summary["packed_compared"]=1024
    if fault=="sample_sha":summary["sample_indices_sha256"]="0"*64
    if fault=="quote":summary["authorization_quote"]="同意"
    if fault=="full_claim":summary["scope"]="full"
    with pytest.raises(ValueError):ms.validate_stage2_encoder_validation(summary,raw["provenance"])


def test_pad8_is_priority_quota_not_final_cap():
    manifest=manifest_for([(50,17)]*(16*40))
    for i,ep in enumerate(manifest["episodes"]):
        ep["h5_file"]=f"record_dataset_Task{i//40}.h5";ep["raw_ep_idx"]=i%40
    entries=ms.build_index_entries(manifest,ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT])
    plan=ms.stage2_encoder_sample_plan(entries)
    assert plan["padded_count"]>16*8 and len(plan["sample_indices"])==1024
    assert plan["demo_count"]+plan["exec_count"]==1024
