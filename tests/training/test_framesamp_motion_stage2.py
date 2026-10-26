"""三episode真实motion库的loader装配、身份负控和旧两种readout回归。"""
import copy
import pathlib
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tests/dataset"))
sys.path.insert(0,str(ROOT/"tests/datastore"))
from test_pack_motion_stage2 import stage2_fx,pack,certify,verify_args
from test_motion_store_stage2 import baseline_module
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.motion_adapter import motion_policy_spec,STAGE2_READOUT
from mme_vla_suite.training import framesamp_dataset as fd
from mme_vla_suite.training import dataloader


@pytest.fixture
def dataset(stage2_fx,monkeypatch,tmp_path):
    s=stage2_fx;pack.cmd_pack(s.args);certify(s);pack.cmd_verify(verify_args(s))
    cfg=OmegaConf.load(ROOT/"src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml")
    cfg.motion.update(source_shape=[1,1024],readout=STAGE2_READOUT,num_tokens=1,dim=1024,budget=96,stride=32,
                      checkpoint_sha256=s.fx.ckpt_sha,store_path=s.args.out,
                      source_run=f"{s.fx.enc_dir.name}/checkpoint_epoch_105.pt#model")
    source=tmp_path/"source";(source/"data").mkdir(parents=True)
    for ep in s.fx.manifest["episodes"]:
        for step in range(ep["exec_start_idx"],ep["num_timesteps"]):
            idx=ep["exec_sample_offset"]+step-ep["exec_start_idx"]
            sample=dict(epis_idx=np.array([ep["global_episode_idx"]]),step_idx=np.array([step]),
                        actions=np.zeros((20,8)),simple_subgoal_online=None,grounded_subgoal_online=None)
            (source/"data"/f"{idx}.pkl").write_bytes(pickle.dumps(sample))
    fm=SimpleNamespace(spec=SimpleNamespace(tokens_per_frame=64,layout="合成帧表"),
          manifest_path=str(s.fx.manifest_path),source_dataset_root=str(source),manifest_scope="full",
          num_exec_samples=s.fx.manifest["totals"]["exec_samples"],manifest_sha256=s.fx.manifest["sha256"])
    monkeypatch.setattr(fd.StoreMeta,"load",lambda root:fm)
    monkeypatch.setattr(fd,"run_fast_checks",lambda *a,**kw:None)
    data_config=SimpleNamespace(norm_stats={"state":SimpleNamespace(mean=np.zeros(8),std=np.ones(8))},use_quantile_norm=False)
    def create(history=cfg):
        ds=fd.FrameSampDataset(str(tmp_path/"frames"),data_config,history,20,motion_root=s.args.out)
        frames=SimpleNamespace(read_image_rows=lambda rows:np.zeros((len(rows),64,2048),np.float32),
             pos_rows=lambda rows:np.broadcast_to(np.asarray(rows,dtype=np.float32)[:,None,None],(len(rows),64,768)),
             state_rows=lambda rows:np.zeros((len(rows),8),np.float32))
        ds._ensure_store=lambda:frames
        return ds
    return SimpleNamespace(ds=create(),create=create,config=cfg,frame_meta=fm,fixture=s)


def test_three_episode_shape_mask_padding_and_byte_offsets(dataset):
    ds=dataset.ds
    for ep in dataset.fixture.fx.manifest["episodes"]:
        for step in (ep["exec_start_idx"],ep["num_timesteps"]-1):
            idx=ep["exec_sample_offset"]+step-ep["exec_start_idx"]
            sample=ds[idx];rows,frames=ms.visible_motion_rows(ds._motion_entries[ep["global_episode_idx"]],step);k=len(rows)
            assert sample["motion_emb"].shape==(96,1024)
            assert sample["motion_pos"].shape==(96,256)
            assert sample["motion_mask"].dtype==np.bool_
            np.testing.assert_array_equal(sample["motion_mask"],np.arange(96)<k)
            assert not np.any(sample["motion_emb"][k:]) and not np.any(sample["motion_pos"][k:])
            assert sample["motion_emb"][:k].tobytes()==ds._ensure_motion_store().rows(rows).tobytes()
            assert np.all(sample["motion_pos"][:k,0]==frames)
    assert dataloader._motion_gates(dataset.config,dataset.frame_meta)==dataset.fixture.args.out


@pytest.mark.parametrize("fault",["arch","sha","stride","budget","readout"])
def test_contract_errors_fail_before_consumption(dataset,fault):
    cfg=copy.deepcopy(dataset.config)
    if fault=="arch":
        meta=copy.deepcopy(dataset.ds._motion_meta);meta.provenance["encoder"]["arch"]="旧模型"
        with pytest.raises(ValueError):fd.check_motion_contract(cfg.motion,meta)
        return
    field,value={"sha":("checkpoint_sha256","0"*64),"stride":("stride",16),
                 "budget":("budget",1),"readout":("readout","adjacent_pair_v1")}[fault]
    cfg.motion[field]=value
    with pytest.raises(ValueError):dataset.create(cfg)


@pytest.mark.parametrize("layout",["motion-768-grid16-v1","motion-768-grid16-demopad17-v1",ms.LAYOUT])
def test_legacy_getitem_equal_baseline(dataset,layout):
    old=baseline_module("src/mme_vla_suite/training/framesamp_dataset.py","baseline_framesamp_dataset")
    ds=dataset.ds
    spec=ms.LAYOUT_SPECS[layout]
    entries=ms.build_index_entries(dataset.fixture.fx.manifest,spec)
    raw=np.random.default_rng(42).standard_normal((ms.index_totals(entries)["rows"],)+spec.row_shape).astype(np.float32)
    config=dict(source_shape=[8,256],readout="adjacent_pair_v1",num_tokens=4,dim=512,budget=640) if layout==ms.LAYOUT else dict(budget=160,dim=768)
    policy=motion_policy_spec(config)
    ds._motion_meta=SimpleNamespace(row_shape=spec.row_shape)
    ds._motion_entries=entries;ds._motion_budget=policy.budget;ds._motion_window_budget=policy.window_budget
    ds._motion_num_tokens=policy.num_tokens;ds._motion_readout=policy.readout
    ds._ensure_motion_store=lambda:SimpleNamespace(rows=lambda indices:raw[indices])
    before=old.FrameSampDataset.__new__(old.FrameSampDataset);before.__dict__.update(ds.__dict__)
    for ep in dataset.fixture.fx.manifest["episodes"]:
        idx=ep["exec_sample_offset"]+ep["exec_samples"]-1
        expected,actual=before[idx],ds[idx]
        assert expected.keys()==actual.keys()
        for key in expected:
            if isinstance(expected[key],np.ndarray):
                assert expected[key].dtype==actual[key].dtype and expected[key].shape==actual[key].shape
                assert expected[key].tobytes()==actual[key].tobytes(),key
            else:assert expected[key]==actual[key]
