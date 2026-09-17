"""demo 补帧契约、独立公式与续跑的 CPU 回归；全部破坏性夹具仅写 tmp_path。"""

import copy
import hashlib
import importlib
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for relative in ("scripts/dataset", "scripts/dataset/wan", "scripts/training/tests", "scripts/training/g0"):
    sys.path.insert(0, str(ROOT / relative))

from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import manifest_sha256
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
import wan_common as wc
from encode_motion import token_outputs_complete
from motion_checks import _independent_visible, audit_padding
from motion_gates_model import _synthetic_entry, oracle_visible
from hand_calc_8frame import expected_motion
from summarize_eval_probe import motion_frames_formula
from eval_rhythm_gates import predict_k


BOUNDARIES = [(0, [], []), (16, [], []), (17, [0], []), (32, [0], []),
              (33, [0,16], [0]), (66, [0,16,32,48], [0,16,32]),
              (114, [0,16,32,48,64,80,96], [0,16,32,48,64,80])]
EXEC_STARTS = {0:[], 16:[], 31:[], 32:[0], 33:[0], 48:[0,16]}


def manifest_for(specs, raw_dir="/scratch/hongze/测试夹具"):
    episodes, offset, total_offset = [], 0, 0
    for g, (total, es) in enumerate(specs):
        episodes.append(dict(global_episode_idx=g,h5_file="record_dataset_T.h5",raw_ep_idx=g,
                             num_timesteps=total,exec_start_idx=es,exec_samples=total-es,
                             exec_sample_offset=offset,total_sample_offset=total_offset,shard_idx=0))
        offset += total-es; total_offset += total
    result = dict(version=1,raw_dir=str(raw_dir),canonical_order=["record_dataset_T.h5"],num_shards=1,
                  totals=dict(episodes=len(episodes),timesteps=total_offset,exec_samples=offset),
                  shard_load_timesteps=[total_offset],episodes=episodes)
    result["sha256"] = manifest_sha256(result)
    return result


@pytest.mark.parametrize("es,new,old", BOUNDARIES)
@pytest.mark.parametrize("minimum", [17,33])
def test_independent_window_boundaries(es, new, old, minimum):
    demo = new if minimum == 17 else old
    spec = ms.LayoutSpec(minimum,33,"repeat_last" if minimum == 17 else "none")
    manifest = manifest_for([(es+65,es)])
    entry = ms.build_index_entries(manifest,spec)[0]
    assert ms.seg_num_grid(es,minimum) == wc.seg_num_grid(es,minimum) == len(demo)
    assert ms.segment_grid_starts(es,minimum) == demo
    segments = wc.list_segments(manifest,demo_min_real=minimum,exec_min_real=33)
    assert sum(s["num_grid"] for s in segments if s["segment"] == "demo") == len(demo)
    independent, _ = _synthetic_entry(0,es+65,es,0,demo_min_real=minimum)
    mem = FrameSampMemory.__new__(FrameSampMemory)
    mem.exec_start_idx, mem.motion_stride, mem.motion_window, mem.demo_min_real = es,16,33,minimum
    rows = {"T_ep0_demo":{"rows":[{"m":m,"start_global_frame":s} for m,s in enumerate(demo)]},
            "T_ep0_exec":{"rows":[{"m":m,"start_global_frame":es+16*m} for m in range(3)]}}
    for delta, exec_offsets in EXEC_STARTS.items():
        want = demo + [es+x for x in exec_offsets]
        t = es+delta
        assert ms.visible_motion_rows(entry,t)[1].tolist() == want
        assert mem.visible_motion_frames(t) == want
        assert _independent_visible(manifest,{"segments":rows},0,t,{"demo_min_real_frames":minimum})[0] == want
        assert [f for _,f in oracle_visible(independent,t,demo_min_real=minimum)] == want
        assert expected_motion(manifest["episodes"][0],t,minimum) == want
        assert motion_frames_formula(es,t,demo_min_real=minimum) == want
        assert predict_k(es,t,demo_min_real=minimum) == len(want)


def test_oracle_boundary_in_isolated_process():
    # oracle 禁止与 mme_vla_suite 同进程导入，真实独立解释器避免削弱这道守卫。
    code = """import json,sys; sys.path.insert(0,'scripts/dataset/wan'); import oracle_driver as o
cases=json.loads(sys.stdin.read()); out=[]
for es in cases:
 for mr in (17,33):
  m={'episodes':[{'global_episode_idx':0,'h5_file':'record_dataset_T.h5','raw_ep_idx':0,'num_timesteps':es+65,'exec_start_idx':es}]}
  ss=o.expected_segments(m,demo_min_real=mr,exec_min_real=33)
  out.append([es,mr,next((s['starts'] for s in ss if s['seg']=='demo'),[])])
print(json.dumps(out))"""
    result = subprocess.run(["uv","run","--no-sync","python","-c",code],cwd=ROOT,
                            input=json.dumps([x[0] for x in BOUNDARIES]),text=True,capture_output=True,check=True)
    actual = json.loads(result.stdout)
    want = [[es,mr,new if mr==17 else old] for es,new,old in BOUNDARIES for mr in (17,33)]
    assert actual == want


def test_legacy_extra_checks_reject_new_layout(tmp_path):
    from extra_checks import load_index, visible
    (tmp_path/"meta").mkdir()
    (tmp_path/"meta/motion_index.json").write_text(json.dumps({"layout":ms.LAYOUT}))
    with pytest.raises(ValueError,match="不适用"):
        load_index(tmp_path)
    for es, _, old in BOUNDARIES:
        entry,_ = _synthetic_entry(0,es+65,es,0,demo_min_real=33)
        for delta, offsets in EXEC_STARTS.items():
            assert [x[3] for x in visible(entry,es+delta)] == old+[es+x for x in offsets]


@pytest.mark.parametrize("layout", list(ms.LAYOUT_SPECS))
def test_layout_fields_writer_parser_and_pickle(layout):
    import pickle
    spec=ms.LAYOUT_SPECS[layout]
    manifest=manifest_for([(150,66)])
    entries=ms.build_index_entries(manifest,spec)
    assert pickle.loads(pickle.dumps(entries)) == entries
    assert hash(entries[0]) == hash(pickle.loads(pickle.dumps(entries[0])))
    payload=ms.index_payload(manifest,entries,spec=spec,layout=layout,mj_repo_commit="测试")
    assert ms.parse_index(payload) == entries
    for key in spec.fields():
        broken=copy.deepcopy(payload); del broken[key]
        with pytest.raises(ValueError,match="缺字段"):
            ms.parse_index(broken)
    broken=copy.deepcopy(payload)
    for key in spec.fields(): del broken[key]
    if layout == "motion-768-grid16-v1":
        assert ms.parse_index(broken) == entries
    else:
        with pytest.raises(ValueError): ms.parse_index(broken)
    broken=copy.deepcopy(payload); broken["demo_min_real_frames"] = 33 if spec.demo_min_real == 17 else 17
    with pytest.raises(ValueError,match="不符"): ms.parse_index(broken)
    other="motion-768-grid16-v1" if layout == ms.LAYOUT else ms.LAYOUT
    with pytest.raises(ValueError): ms.index_payload(manifest,entries,spec=spec,layout=other,mj_repo_commit="测试")


def test_token_resume_two_segments_and_changed_input(tmp_path):
    for key in ("T_ep0_demo","T_ep0_exec"):
        latent=tmp_path/f"{key}.bin"; latent.write_bytes(bytes(wc.CHUNK_BYTES))
        token=tmp_path/f"{key}.f32.bin"; token.write_bytes(bytes(wc.TOKEN_BYTES))
        (tmp_path/f"{key}.f32.bin.sha256").write_text(hashlib.sha256(token.read_bytes()).hexdigest())
        metadata=tmp_path/f"{key}.metadata.json"
        metadata.write_text(json.dumps({"schema":1,"input_latent_sha256":hashlib.sha256(latent.read_bytes()).hexdigest()}))
        assert token_outputs_complete(tmp_path,key,1,latent)
        latent.write_bytes(b"x"+bytes(wc.CHUNK_BYTES-1))
        assert not token_outputs_complete(tmp_path,key,1,latent)
        latent.write_bytes(bytes(wc.CHUNK_BYTES))
        metadata.rename(tmp_path/f"{key}.f32.metadata.json")
        assert not token_outputs_complete(tmp_path,key,1,latent)


def test_extract_real_h5_padding_aggregate_and_a11(tmp_path):
    import h5py
    import torch
    from extract_wan import process_segment
    from run_local import aggregate_segments
    from mme_vla_suite.policies.motion_protocol import stub_frame, stub_decode
    raw=tmp_path/"raw"; raw.mkdir()
    lat=tmp_path/"wan-latents"; lat.mkdir()
    specs=[(es+65,es) for es in (17,32,33,66,114)]
    manifest=manifest_for(specs,raw)
    manifest_path=tmp_path/"episode_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with h5py.File(raw/"record_dataset_T.h5","w") as file:
        for g,(total,es) in enumerate(specs):
            for t in range(total):
                file.create_dataset(f"episode_{g}/timestep_{t}/obs/front_rgb",data=stub_frame(t))
    class Encoder:
        pin_numerics=staticmethod(lambda: None)
        @staticmethod
        def encode_chunk(vae,window,device):
            assert window.shape == (33,256,256,3) and window.dtype == np.uint8
            return torch.zeros(wc.LAT_SHAPE,dtype=torch.float32)
    worker=dict(hostname="测试",gpu_uuid="CPU",worker="cpu",pid=1)
    items=wc.list_segments(manifest)
    for item in items:
        meta=process_segment(Encoder,torch,None,None,item,raw,lat,worker,{})
        for row in meta["rows"]:
            offset=row["seg_offset"]; start=item["seg_start"]+offset
            real=min(33,item["seg_len"]-offset)
            frames=list(range(start,start+real))+[item["seg_start"]+item["seg_len"]-1]*(33-real)
            expected=np.stack([stub_frame(t) for t in frames])
            assert row["input_frames_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
            assert row["real_frames"] == real and row["pad_frames"] == 33-real
    assert aggregate_segments(lat,manifest_path,"wan") == len(items)
    result=audit_padding(manifest,lat)
    assert result["padded"] == result["eligible"] == 5
    combined=json.loads((lat/"metadata.json").read_text())
    assert combined["schema"] == 3 and combined["raw_dir"] == str(raw)
    target=lat/"T_ep4_demo.metadata.json"
    original=json.loads(target.read_text())
    assert original["rows"][-1]["real_frames"] == 18 and original["rows"][-1]["pad_frames"] == 15
    for key,bad in (("pad_source_frame",114),("real_frames",19),("pad_frames",14)):
        altered=copy.deepcopy(original); altered["rows"][-1][key]=bad
        target.write_text(json.dumps(altered))
        matching=copy.deepcopy(combined); matching["segments"]["T_ep4_demo"]["rows"]=altered["rows"]
        (lat/"metadata.json").write_text(json.dumps(matching))
        with pytest.raises(ValueError): audit_padding(manifest,lat)
    target.write_text(json.dumps(original))
    (lat/"metadata.json").write_text(json.dumps(combined))


def test_budget_and_byte_comparator():
    from compare_online_motion import bytes_equal, report_passed
    assert predict_k(1296,1296+1296,demo_min_real=17) == 160
    assert predict_k(1297,1297+1296,demo_min_real=17) == 161
    assert not bytes_equal(np.array([0],np.float32),np.array([-0.0],np.float32))
    assert not bytes_equal(np.array([1],np.float32),np.array([1],np.float64))
    report=dict(mode="stub",mismatch_total=1,start_ok=True,pos_ok=True,order_ok=True,rows_seen=[0],expected_rows=1)
    assert not report_passed(report)


def test_motion_meta_requires_matching_contract(tmp_path):
    manifest=manifest_for([(40,0)])
    spec=ms.LAYOUT_SPECS[ms.LAYOUT]
    entries=ms.build_index_entries(manifest,spec)
    index=ms.index_payload(manifest,entries,spec=spec,layout=ms.LAYOUT,mj_repo_commit="测试")
    (tmp_path/"meta").mkdir()
    index_path=tmp_path/ms.INDEX_RELPATH
    index_path.write_text(json.dumps(index))
    raw=dict(schema=1,layout=ms.LAYOUT,status="packed",byte_order="little",array_order="C",
             grid_stride=16,window_frames=33,grid_origin="segment_start",window_direction="forward",
             truncation_policy="none",frame_size=256,num_rows=1,manifest_sha256=manifest["sha256"],
             manifest_path="测试清单",motion_index_sha256=ms.sha256_file(index_path),provenance={},
             tables={"motion_token":dict(row_shape=[768],dtype="float32",row_bytes=3072,num_rows=1,
                                         relpath=ms.MOTION_TABLE_RELPATH,byte_count=3072)},**spec.fields())
    meta_path=tmp_path/ms.META_RELPATH
    meta_path.write_text(json.dumps(raw))
    assert ms.MotionMeta.load(tmp_path).spec == spec
    for key in spec.fields():
        bad=copy.deepcopy(raw); del bad[key]
        meta_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match="缺字段"): ms.MotionMeta.load(tmp_path)
    bad={**raw, "layout":"motion-768-grid16-v1", **ms.LAYOUT_SPECS["motion-768-grid16-v1"].fields()}
    meta_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="布局或窗口契约不同"): ms.MotionMeta.load(tmp_path)


def test_pack_three_raw_bindings_and_contract(tmp_path):
    from pack_motion_store import check_raw_sources
    raw_dir=tmp_path/"raw"; raw_dir.mkdir()
    lat=tmp_path/"wan-latents"; lat.mkdir()
    manifest=manifest_for([(70,33)],raw_dir)
    manifest_path=tmp_path/"episode_manifest.json"; manifest_path.write_text(json.dumps(manifest))
    inputs=tmp_path/"input_manifest.json"; inputs.write_text(json.dumps({"raw_dir":str(raw_dir)}))
    latent=dict(raw_dir=str(raw_dir),**ms.LAYOUT_SPECS[ms.LAYOUT].fields())
    meta=lat/"metadata.json"; meta.write_text(json.dumps(latent))
    report=dict(raw_dir=str(raw_dir),manifest_sha256=manifest["sha256"],tested_metadata_sha256=ms.sha256_file(meta),
                frame_mismatches=0,metadata_mismatches=0,demo_min_real_frames=17,exec_min_real_frames=33)
    oracle=tmp_path/"vae_report.json"; oracle.write_text(json.dumps(report))
    assert check_raw_sources(manifest_path,lat,raw_dir,oracle) == inputs
    for path in (inputs,meta,oracle):
        saved=path.read_bytes(); obj=json.loads(saved); obj["raw_dir"]=str(tmp_path/"wrong")
        path.write_text(json.dumps(obj))
        with pytest.raises(ValueError,match="三方目录"): check_raw_sources(manifest_path,lat,raw_dir,oracle)
        path.write_bytes(saved)
    for path in (meta,oracle):
        saved=path.read_bytes(); obj=json.loads(saved); obj["demo_min_real_frames"]=33
        path.write_text(json.dumps(obj))
        with pytest.raises(ValueError,match="窗口契约"): check_raw_sources(manifest_path,lat,raw_dir,oracle)
        path.write_bytes(saved)


@pytest.mark.parametrize("fraction", ["0.00","0.10"])
def test_sample_compare_rejects_missing_window(tmp_path,fraction):
    from compare_wan import expected_samples,cmd_latents,CHUNK_F32
    lat=tmp_path/"latents"; lat.mkdir()
    oracle=tmp_path/"oracle"; oracle.mkdir()
    meta=dict(demo_min_real_frames=17,segments={
        "T_ep0_demo":dict(segment_kind="demo",seg_len=33,num_grid=2),
        "T_ep0_exec":dict(segment_kind="exec",seg_len=65,num_grid=3)})
    (lat/"metadata.json").write_text(json.dumps(meta))
    sample_spec=f"padded:all,rest:{fraction},seed:0"
    selected,padded=expected_samples(meta,sample_spec)
    assert padded == {("T_ep0_demo",1)} and padded <= selected
    for key,segment in meta["segments"].items():
        for root in (lat,oracle): (root/f"{key}.bin").write_bytes(bytes(segment["num_grid"]*CHUNK_F32*4))
    report=dict(tested_metadata_sha256=ms.sha256_file(lat/"metadata.json"),segments=meta["segments"],
                frame_mismatches=0,metadata_mismatches=0,metadata_rows_checked=5,windows=len(selected),sample_spec=sample_spec)
    (oracle/"vae_report.json").write_text(json.dumps(report))
    sample=dict(sample_spec=sample_spec,windows=[dict(segment=k,m=m) for k,m in sorted(selected)])
    path=oracle/"sampled_windows.json"; path.write_text(json.dumps(sample))
    args=SimpleNamespace(latents=lat,oracle=oracle,sampled=path,sample_spec=sample_spec,expect_padded=1)
    cmd_latents(args)
    sample["windows"].pop(); path.write_text(json.dumps(sample))
    with pytest.raises(SystemExit,match="独立重算"): cmd_latents(args)
