"""真实打包、独立认证合同、verify及读取的 CPU 小表链路与来源负控。"""
import copy
import dataclasses
import functools
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dataset"))
sys.path.insert(0, str(ROOT / "tests/datastore"))
import pack_motion_store as pack
import test_build_mixed8_v10_motion as legacy
from test_motion_store_stage2 import stage2_identity
from mme_vla_suite.datastore import motion_store as ms


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def stage2_fx(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "SPECS", [(165,100),(83,50),(32,0)])
    fx = legacy.fx.__wrapped__(tmp_path)
    # 复用旧夹具的真实历史 receipt 核验；只改变合成缓存所需的固定身份值。
    total = load(fx.latents / "metadata.json")
    for key in fx.keys:
        lm = load(fx.latents / f"{key}.metadata.json")
        lm.update(grid_stride=16,window_frames=33)
        for row in lm["rows"]:
            row.update(input_shape=[33,256,256,3],input_dtype="uint8",input_frames_sha256="a"*64)
        dump(fx.latents / f"{key}.metadata.json",lm)
        total["segments"][key]["rows"] = lm["rows"]
    dump(fx.latents / "metadata.json",total)
    total_sha = ms.sha256_file(fx.latents / "metadata.json")
    files = fx.restore / "restore_files.sha256"
    files.write_text(f"{total_sha}  wan-latents/metadata.json\n")
    oracle = fx.restore / "oracle/wan-mj/vae_report.json"
    report = load(oracle);report["tested_metadata_sha256"] = total_sha;dump(oracle,report)
    receipt = load(fx.receipt)
    receipt["latents"].update(total_metadata_sha256=total_sha,files_list_sha256=ms.sha256_file(files))
    receipt["oracle"]["vae_report_sha256"] = ms.sha256_file(oracle)
    dump(fx.receipt,receipt)
    fx.pins = dataclasses.replace(fx.pins,total_metadata_sha256=total_sha)
    monkeypatch.setattr(pack,"load_vae_compat",functools.partial(pack.load_vae_compat,pins=fx.pins))
    identity = stage2_identity()
    enc = identity["encoder"]
    (fx.enc_dir / enc["checkpoint_name"]).write_bytes(b"synthetic-EMA")
    enc.update(run_name=fx.enc_dir.name,checkpoint_sha256=ms.sha256_file(fx.enc_dir/enc["checkpoint_name"]),
               config_sha256=ms.sha256_file(fx.enc_dir/"config.yaml"))
    fx.ckpt_sha = enc["checkpoint_sha256"]
    pin_dir = tmp_path / "stage2-pin";pin_dir.mkdir()
    (pin_dir / "uv.lock").write_text("合成锁\n")
    pin = identity["source_pin"]
    pin.update(vae_adapter_sha256=ms.sha256_file(ROOT / "scripts/dataset/wan/wan_motion_infer.py"),
               dependency_lock_sha256=ms.sha256_file(pin_dir/"uv.lock"))
    pin_path = pin_dir / "SOURCE_PIN.json";dump(pin_path,pin)
    groups = {k: copy.deepcopy(identity[k]) for k in ("encoder","encoder_state_sha256","source_pin","numerics")}
    groups["source_pin"] = {k:pin[k] for k in ("mj_repo_commit","export_source_sha256","encoder_source_sha256")}
    entries = ms.build_index_entries(fx.manifest,ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT])
    fx.tokens.mkdir()
    content = {}
    for e,seg,key,base,ng in pack.iter_rows_in_order(entries):
        lm = load(fx.latents / f"{key}.metadata.json")
        selected = list(range(0,lm["num_grid"],2))
        latent = (fx.latents / f"{key}.bin").read_bytes()
        blob = (np.arange(ng*1024,dtype=np.float32)+base*1024).reshape(ng,1,1024).tobytes()
        content[key] = blob
        (fx.tokens/f"{key}.f32.bin").write_bytes(blob)
        (fx.tokens/f"{key}.f32.bin.sha256").write_text(ms.sha256_bytes(blob)+f"  {key}.f32.bin\n")
        metadata = {**groups,"worker":{"pid":123,"device":"cpu","hostname":"fixture"},"schema":"stage2-memory-export-v1","layout":ms.STAGE2_LAYOUT,
                    "manifest_sha256":ms.sha256_file(fx.manifest_path),
                    "segment":key,"segment_kind":seg,"seg_start_global":lm["seg_start_global"],"seg_len":lm["seg_len"],
                    "grid_stride":32,"window_frames":33,"row_shape":[1,1024],"dtype":"float32",
                    "token_dim":1024,"token_bytes":4096,"num_grid":ng,"bytes":len(blob),"sha256":ms.sha256_bytes(blob),
                    "source_grid":selected,"rows":[lm["rows"][g] for g in selected],
                    "input_latent_sha256":ms.sha256_bytes(latent),
                    "input_block_sha256":[ms.sha256_bytes(latent[g*pack.CHUNK_BYTES:(g+1)*pack.CHUNK_BYTES]) for g in selected]}
        dump(fx.tokens/f"{key}.metadata.json",metadata)
    dump(fx.tokens/"export_record.json",{**groups,"schema":"stage2-memory-export-v1","scope":"full",
          "segments":len(content),"rows":ms.index_totals(entries)["rows"],"segment_keys":list(content),"manifest_sha256":ms.sha256_file(fx.manifest_path)})
    validation=tmp_path/"validation";validation.mkdir()
    windows=[]
    for i in range(64):
        task,grid=f"task{i//4}",2*(i%4)
        demo=i%4==0;segment="demo" if demo else "exec"
        real,pad=(17,16) if demo else (33,0)
        frames=list(range(grid*16,grid*16+real))+[grid*16+real-1]*pad
        sources=[dict(kind="demo_h5",path=f"合成/{task}.h5",sha256="a"*64)] if demo else [
                 dict(kind="exec_pkl",path=f"合成/{task}_{frame}.pkl",sha256="a"*64) for frame in frames]
        windows.append(dict(key=f"{task}_ep0_{segment}",task=task,episode=0,segment_kind=segment,source_grid=grid,
             latent_path=f"合成/{task}.bin",frame_ids=frames,real_frames=real,pad_frames=pad,pad_source_frame=frames[-1] if pad else None,
             sources=sources,source_frame_indices=[dict(source=0,index=frame) for frame in frames] if demo else [
             dict(source=j,index=0) for j in range(33)],input_frames_sha256="a"*64,latent_block_sha256="a"*64))
    dump(validation/"windows.json",dict(schema="stage2-validation-inputs-v1",gate="G0",status="PASS",seed=0,
          manifest_sha256=fx.manifest["sha256"],latent_metadata_sha256=total_sha,windows=windows,
          counts=dict(windows=64,tasks=16,demo_tail=16,sha_mismatches=0)))
    metrics={"G2":dict(windows=64,demo_tail=16,worst_mse=0,worst_max=0),
             "G3":dict(windows=32,seed=0,max_abs=0,geometry_all_true=1,leaves=65,state_key="model",exact=1,live_negative_rejected=1)}
    online_vae={k:legacy.vae_info()[k] for k in ("vae_id","vae_state_sha256","vae_dtype","latent_mode","batch","tf32","amp")}
    online_vae.update(adapter_sha256=pin["vae_adapter_sha256"],numerics=groups["numerics"],
                      preprocess=dict(frame_size=256,input_dtype="uint8",input_layout="THWC",window_frames=33))
    for gate in metrics:
        dump(validation/f"{gate}.json",dict(schema="stage2-validation-gate-v1",gate=gate,status="PASS",checkpoint_sha256=fx.ckpt_sha,
             samples_sha256=ms.sha256_file(validation/"windows.json"),metrics=metrics[gate],source_pin=pin,
             numerics=groups["numerics"],**({"online_vae":online_vae} if gate=="G2" else {})))
    args=SimpleNamespace(layout="stage2",latent_grid_subsample=2,stage2_validation=str(validation),stage2_source_pin=str(pin_path),
          manifest=str(fx.manifest_path),tokens=str(fx.tokens),latents=str(fx.latents),out=str(tmp_path/"store"),
          raw_dir=fx.manifest["raw_dir"],encoder_run_dir=str(fx.enc_dir),oracle_report=None,restore_receipt=str(fx.receipt),
          expected_ckpt_sha256=fx.ckpt_sha,resume=False,force_break_lock=False)
    return SimpleNamespace(fx=fx,args=args,entries=entries,content=content,validation=validation,identity=identity)


def certify(stage2):
    root=pathlib.Path(stage2.args.out);meta=ms.MotionMeta.load(root)
    record=dict(schema="stage2-encoder-bitexact-v1",scope="full",layout=ms.STAGE2_LAYOUT,status="PASS",
                compared=meta.num_rows,mismatches=0,store_table_sha256=meta.table_sha256,motion_index_sha256=meta.motion_index_sha256,
                checkpoint_sha256=meta.provenance["encoder"]["checkpoint_sha256"],source_grid_sha256=meta.provenance["source_grid_sha256"])
    dump(root/pack.BITEXACT_RELPATH,record)
    return record


def verify_args(stage2):
    return SimpleNamespace(store=stage2.args.out,manifest=None,tokens=None,resume=True,force_break_lock=False,check_padding=True)


def test_real_pack_verify_load(stage2_fx):
    s=stage2_fx;pack.cmd_pack(s.args)
    root=pathlib.Path(s.args.out)
    assert ms.MotionMeta.load(root).num_rows == 8
    assert (root/ms.MOTION_TABLE_RELPATH).read_bytes() == b"".join(s.content.values())
    assert ms.MotionMeta.load(root).entries[0].demo.num_grid == 3
    with pytest.raises(RuntimeError,match="比较记录"): pack.cmd_verify(verify_args(s))
    certify(s);pack.cmd_verify(verify_args(s))
    meta=ms.MotionMeta.load(root);assert meta.status=="verified"
    summary=meta.provenance["stage2_compat"]["encoder_validation"]
    assert summary["scope"]=="full" and summary["encoder_compared"]==summary["packed_compared"]==meta.num_rows
    assert "sample_indices_sha256" not in summary and "authorization_quote" not in summary
    assert summary["record_json"].encode()==(root/pack.BITEXACT_RELPATH).read_bytes()
    assert meta.provenance["episode_manifest_file_sha256"]==ms.sha256_file(s.fx.manifest_path)
    assert meta.manifest_sha256==s.fx.manifest["sha256"]
    assert meta.provenance["episode_manifest_file_sha256"]!=meta.manifest_sha256
    assert meta.raw["verify"]["demo_tail_padded"]==1
    assert meta.raw["verify"]["legal_zero_segments"]==1
    assert not (root/ms.LOCK_RELPATH).exists()
    store=ms.MotionStore(root,verify_level="full")
    assert store.rows([3]).tobytes() == b"".join(s.content.values())[3*4096:4*4096]
    assert store.rows([]).shape == (0,1,1024)


@pytest.mark.parametrize("fault",["missing_segment","extra_segment","padding","odd_latent","state","pin","G0","G2","G3","partial_export","historical_rows","gate_pin","gate_numerics","gate_vae","manifest_canonical","export_manifest_canonical"])
def test_bad_sources_rejected_before_table(stage2_fx,fault):
    s=stage2_fx;key=next(iter(s.content));path=s.fx.tokens/f"{key}.metadata.json"
    if fault=="missing_segment":path.unlink()
    elif fault=="extra_segment":dump(s.fx.tokens/"unexpected.metadata.json",{})
    elif fault=="odd_latent":
        path=s.fx.latents/f"{key}.bin";data=bytearray(path.read_bytes());data[pack.CHUNK_BYTES]^=1;path.write_bytes(data)
    elif fault=="G0":
        path=s.validation/"windows.json";value=load(path);del value["windows"][0]["sources"];dump(path,value)
    elif fault=="historical_rows":
        lm_path=s.fx.latents/f"{key}.metadata.json";lm=load(lm_path);tm=load(path)
        lm["rows"][0]["input_frames_sha256"]="0"*64;tm["rows"][0]["input_frames_sha256"]="0"*64
        dump(lm_path,lm);dump(path,tm)
    elif fault.startswith("gate_"):
        path=s.validation/"G2.json";value=load(path)
        if fault=="gate_pin":value["source_pin"]["vae_adapter_sha256"]="0"*64
        elif fault=="gate_numerics":value["numerics"]["torch"]="错误版本"
        else:value["online_vae"]["vae_state_sha256"]="0"*64
        dump(path,value)
    elif fault in ("G2","G3"):
        path=s.validation/f"{fault}.json";value=load(path);value["samples_sha256"]="0"*64;dump(path,value)
    elif fault=="partial_export":
        path=s.fx.tokens/"export_record.json";value=load(path);value["scope"]="smoke";dump(path,value)
    elif fault in ("manifest_canonical","export_manifest_canonical"):
        if fault=="export_manifest_canonical":path=s.fx.tokens/"export_record.json"
        value=load(path);value["manifest_sha256"]=s.fx.manifest["sha256"];dump(path,value)
    else:
        value=load(path)
        if fault=="padding":value["rows"][0]["pad_source_frame"]=99
        if fault=="state":value["encoder_state_sha256"]["leaves"]["state_00"]["sha256"]="0"*64
        if fault=="pin":value["source_pin"]["mj_repo_commit"]="0"*40
        dump(path,value)
    with pytest.raises((ValueError,FileNotFoundError)):pack.cmd_pack(s.args)
    assert not (pathlib.Path(s.args.out)/ms.MOTION_TABLE_RELPATH).exists()


@pytest.mark.parametrize("fault",["partial","stale_table","stale_index","source_grid","layout","live_lock","foreign_lock"])
def test_certification_and_lock_handoff(stage2_fx,fault):
    s=stage2_fx;pack.cmd_pack(s.args);record=certify(s);root=pathlib.Path(s.args.out)
    if fault=="partial":record["scope"]="sample"
    elif fault=="stale_table":record["store_table_sha256"]="0"*64
    elif fault=="stale_index":record["motion_index_sha256"]="0"*64
    elif fault=="source_grid":record["source_grid_sha256"]="0"*64
    elif fault=="layout":record["layout"]=ms.LAYOUT
    elif fault in ("live_lock","foreign_lock"):
        value=load(root/ms.LOCK_RELPATH)
        if fault=="live_lock":value["pid"]=1
        else:value["host"]="另一台机器"
        dump(root/ms.LOCK_RELPATH,value)
    dump(root/pack.BITEXACT_RELPATH,record)
    with pytest.raises(RuntimeError):pack.cmd_verify(verify_args(s))
    assert ms.MotionMeta.load(root).status=="packed"
