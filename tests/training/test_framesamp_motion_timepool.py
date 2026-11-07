"""check_motion_contract 与 FrameSampDataset 对零参数池化库的严格核对、混配拒绝和读取形状。"""
import copy
import pathlib
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/dataset"))
sys.path.insert(0, str(ROOT / "tests/datastore"))
from test_export_wan_timepool import build_tree, run
from test_pack_motion_timepool import cli, pack_argv, verify_argv, write_records
from test_pack_motion_stage2 import certify, pack, stage2_fx, verify_args  # noqa: F401  stage2_fx 是夹具
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.motion_adapter import STAGE2_READOUT
from mme_vla_suite.shared import wan_timepool as tp
from mme_vla_suite.training import dataloader
from mme_vla_suite.training import framesamp_dataset as fd

BASE_YAML = ROOT / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml"


def timepool_motion(store, **overrides):
    """模拟 D 的新 YAML：只换来源身份，没有 source_run 与 checkpoint_sha256。"""
    cfg = OmegaConf.load(BASE_YAML)
    cfg.motion.update(source_shape=[1, 1024], readout=tp.TIMEPOOL_READOUT, encoder_kind=tp.TIMEPOOL_ENCODER_KIND,
                      num_tokens=1, dim=1024, budget=96, stride=32, store_path=str(store))
    for key in ("source_run", "checkpoint_sha256"):
        cfg.motion.pop(key, None)
    cfg.motion.update(overrides)
    return cfg


@pytest.fixture
def built(tmp_path, monkeypatch):
    t = build_tree(tmp_path)
    t.store = tmp_path / "store" / "motion-wan-timepool1024"
    t.records = tmp_path / "records"
    run(t)
    cli(monkeypatch, pack_argv(t))
    return t


@pytest.fixture
def verified(built, monkeypatch):
    write_records(built)
    cli(monkeypatch, verify_argv(built, records=True))
    return built


def test_contract_accepts_verified_timepool_store(verified):
    cfg = timepool_motion(verified.store)
    spec = fd.check_motion_contract(cfg.motion, ms.MotionMeta.load(verified.store))
    assert spec.readout == tp.TIMEPOOL_READOUT and spec.layouts == (tp.TIMEPOOL_LAYOUT,)


@pytest.mark.parametrize("overrides", [dict(source_run="r/checkpoint_epoch_1.pt#model"),
                                       dict(checkpoint_sha256="a" * 64), dict(encoder_kind="stage2_pool4_concat"),
                                       dict(store_path=None)])
def test_contract_rejects_learned_identity_on_timepool_yaml(verified, overrides):
    cfg = timepool_motion(verified.store, **overrides)
    with pytest.raises(ValueError):
        fd.check_motion_contract(cfg.motion, ms.MotionMeta.load(verified.store))


def test_contract_rejects_timepool_kind_with_stage2_readout(verified):
    cfg = timepool_motion(verified.store, readout=STAGE2_READOUT)
    with pytest.raises(ValueError, match="encoder_kind"):
        fd.check_motion_contract(cfg.motion, ms.MotionMeta.load(verified.store))


def test_stage2_yaml_cannot_read_timepool_store(verified):
    cfg = timepool_motion(verified.store, readout=STAGE2_READOUT, encoder_kind="stage2_pool4_concat",
                          checkpoint_sha256="a" * 64)
    with pytest.raises(ValueError, match="layout"):
        fd.check_motion_contract(cfg.motion, ms.MotionMeta.load(verified.store))


def test_timepool_yaml_cannot_read_stage2_store(stage2_fx):  # noqa: F811
    s = stage2_fx
    pack.cmd_pack(s.args)
    certify(s)
    pack.cmd_verify(verify_args(s))
    cfg = timepool_motion(s.args.out)
    with pytest.raises(ValueError, match="layout"):
        fd.check_motion_contract(cfg.motion, ms.MotionMeta.load(s.args.out))


def test_packed_store_refused_even_with_unverified_escape(built, monkeypatch):
    monkeypatch.setenv("MMEVLA_MOTION_ALLOW_UNVERIFIED", "1")
    cfg = timepool_motion(built.store)
    meta = ms.MotionMeta.load(built.store)
    ms.require_verified(meta)                   # 旧逃生阀放行
    with pytest.raises(ValueError, match="未 verified"):
        fd.check_motion_contract(cfg.motion, meta)


def test_motion_gates_accept_without_source_run_and_refuse_lock(verified, built):
    cfg = timepool_motion(verified.store)
    frame_meta = SimpleNamespace(manifest_path=str(verified.manifest_path),
                                 manifest_sha256=verified.manifest["sha256"])
    assert dataloader._motion_gates(cfg, frame_meta) == str(verified.store)
    (verified.store / ms.LOCK_RELPATH).write_text("{}")
    with pytest.raises(RuntimeError, match="pack.lock"):
        dataloader._motion_gates(cfg, frame_meta)


def test_motion_gates_refuse_packed_store(built):
    (built.store / ms.LOCK_RELPATH).unlink()
    frame_meta = SimpleNamespace(manifest_path=str(built.manifest_path), manifest_sha256=built.manifest["sha256"])
    with pytest.raises(RuntimeError, match="未通过 verify"):
        dataloader._motion_gates(timepool_motion(built.store), frame_meta)


@pytest.fixture
def dataset(verified, tmp_path, monkeypatch):
    cfg = timepool_motion(verified.store)
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    for ep in verified.manifest["episodes"]:
        for step in range(ep["exec_start_idx"], ep["num_timesteps"]):
            idx = ep["exec_sample_offset"] + step - ep["exec_start_idx"]
            sample = dict(epis_idx=np.array([ep["global_episode_idx"]]), step_idx=np.array([step]),
                          actions=np.zeros((20, 8)), simple_subgoal_online=None, grounded_subgoal_online=None)
            (source / "data" / f"{idx}.pkl").write_bytes(pickle.dumps(sample))
    fm = SimpleNamespace(spec=SimpleNamespace(tokens_per_frame=64, layout="合成帧表"),
                         manifest_path=str(verified.manifest_path), source_dataset_root=str(source),
                         manifest_scope="full", num_exec_samples=verified.manifest["totals"]["exec_samples"],
                         manifest_sha256=verified.manifest["sha256"])
    monkeypatch.setattr(fd.StoreMeta, "load", lambda root: fm)
    monkeypatch.setattr(fd, "run_fast_checks", lambda *a, **kw: None)
    data_config = SimpleNamespace(norm_stats={"state": SimpleNamespace(mean=np.zeros(8), std=np.ones(8))},
                                  use_quantile_norm=False)
    ds = fd.FrameSampDataset(str(tmp_path / "frames"), data_config, cfg, 20, motion_root=str(verified.store))
    frames = SimpleNamespace(
        read_image_rows=lambda rows: np.zeros((len(rows), cfg.token_per_image, 2048), np.float32),
        pos_rows=lambda rows: np.broadcast_to(np.asarray(rows, dtype=np.float32)[:, None, None],
                                              (len(rows), cfg.token_per_image, 768)),
        state_rows=lambda rows: np.zeros((len(rows), 8), np.float32))
    ds._ensure_store = lambda: frames
    return SimpleNamespace(ds=ds, fixture=verified)


def test_getitem_delivers_flat_1024_tokens_with_same_visible_windows(dataset):
    ds = dataset.ds
    for idx in range(len(ds)):
        sample = ds[idx]
        g, step = int(ds._epis_of[idx]), int(ds._step_of[idx])
        rows, _ = ms.visible_motion_rows(ds._motion_entries[g], step)
        k = len(rows)
        assert sample["motion_emb"].shape == (96, 1024) and sample["motion_emb"].dtype == np.float32
        np.testing.assert_array_equal(sample["motion_mask"], np.arange(96) < k)
        assert sample["motion_emb"][:k].tobytes() == ds._ensure_motion_store().rows(rows).tobytes()
        assert not np.any(sample["motion_emb"][k:])


def test_stage2_contract_unchanged_for_stage2_store(stage2_fx):  # noqa: F811
    """旧分支：缺 checkpoint_sha256 仍按 stage2 规则拒绝，新分支没有放宽它。"""
    s = stage2_fx
    pack.cmd_pack(s.args)
    certify(s)
    pack.cmd_verify(verify_args(s))
    cfg = timepool_motion(s.args.out, readout=STAGE2_READOUT, encoder_kind="stage2_pool4_concat")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        fd.check_motion_contract(cfg.motion, ms.MotionMeta.load(s.args.out))
    good = copy.deepcopy(cfg)
    good.motion.checkpoint_sha256 = s.fx.ckpt_sha
    assert fd.check_motion_contract(good.motion, ms.MotionMeta.load(s.args.out)).readout == STAGE2_READOUT
