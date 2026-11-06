"""context 1024 两档训练读取：形制白名单、MemoryBuffer 等价、pickle、关闭态与基线同字节（合成小库）。"""
import json
import pathlib
import pickle
import subprocess
from types import ModuleType, SimpleNamespace

import ml_dtypes
import numpy as np
from omegaconf import OmegaConf
import pytest
import yaml

from mme_vla_suite.dataset_builder.mem_buffer import MemoryBuffer
from mme_vla_suite.datastore.manifest import manifest_sha256
from mme_vla_suite.training import framesamp_dataset as fd

ROOT = pathlib.Path(__file__).resolve().parents[2]
YAMLS = "src/mme_vla_suite/models/config/robomme"
BASELINE = "d80d660"
UPSTREAM = "ecf086c"
# (num_timesteps, exec_start_idx)：长集超预算、带 demo 前缀集、短集
SPECS = [(90, 0), (50, 12), (5, 0)]
TOKENS_OF = {16: "4x4", 64: "8x8"}


def git_show(ref, relative):
    return subprocess.check_output(["git", "show", f"{ref}:{relative}"], cwd=ROOT, text=True)


def history(budget=1024, tokens=64, kind="token_dropping", fusion="context", dim=2048, motion=False):
    return OmegaConf.create({
        "budget": budget, "num_views": 1, "token_per_image": tokens, "representation_type": "perceptual",
        "perceptual_memory": {"type": kind}, "integration_type": fusion, "memory_token_dim": dim,
        "use_state_emb": False, "memory_feature": {"img": {"input_dim": 2048}, "pos": {"input_dim": 768}},
        **({"motion": {"enabled": True}} if motion else {}),
    })


class FakeStore:
    """按全局行号给可追溯内容的帧表；接口同 FrameSampStore 被 FrameSampDataset 用到的部分。"""

    def __init__(self, tokens, rows, seed=0):
        rng = np.random.default_rng(seed)
        self.image = rng.standard_normal((rows, tokens, 2048), np.float32).astype(ml_dtypes.bfloat16)
        self.pos = rng.standard_normal((max(s[0] for s in SPECS), tokens, 768), np.float32)
        self.state = rng.standard_normal((rows, 8), np.float32)
        self.owner_pid = None

    def read_image_rows(self, rows):
        return self.image[rows]

    def pos_rows(self, frames):
        return self.pos[frames]

    def state_rows(self, rows):
        return self.state[rows]

    def close(self):
        pass


@pytest.fixture
def lib(tmp_path, monkeypatch):
    episodes, exec_offset, total = [], 0, 0
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for g, (nt, es) in enumerate(SPECS):
        episodes.append(dict(global_episode_idx=g, h5_file=f"record_dataset_T{g}.h5", raw_ep_idx=g, num_timesteps=nt,
                             exec_start_idx=es, exec_samples=nt - es, exec_sample_offset=exec_offset,
                             total_sample_offset=total, shard_idx=0))
        for step in range(es, nt):
            sample = dict(epis_idx=np.array([g]), step_idx=np.array([step]), actions=np.zeros((50, 8)),
                          simple_subgoal_online=None, grounded_subgoal_online=None)
            (source / "data" / f"{exec_offset + step - es}.pkl").write_bytes(pickle.dumps(sample))
        # kept：JSON 序即分数升序；首帧 64 个在尾部（与真实堆的 1000.0 分一致）
        later = [[t, 0, p] for t in range(1, nt) for p in range(64)]
        rng.shuffle(later)
        kept = later[: 2048 - 64] + [[0, 0, p] for p in range(64)]
        (source / "features" / f"episode_{g}").mkdir(parents=True)
        (source / "features" / f"episode_{g}" / "kept_indices.json").write_text(json.dumps(kept))
        exec_offset += nt - es
        total += nt
    manifest = dict(version=1, totals=dict(episodes=len(SPECS), timesteps=total, exec_samples=exec_offset),
                    episodes=episodes)
    manifest["sha256"] = manifest_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    stores = {t: FakeStore(t, total) for t in TOKENS_OF}

    def meta_for(root):
        tokens = int(pathlib.Path(root).name)
        return SimpleNamespace(spec=SimpleNamespace(tokens_per_frame=tokens, layout=f"合成{tokens}"),
                               manifest_path=str(manifest_path), source_dataset_root=str(source),
                               manifest_scope="full", num_exec_samples=exec_offset, num_rows=total)

    def store_for(root, *, meta, manifest_path, source_root):
        store = stores[int(pathlib.Path(root).name)]
        store.owner_pid = fd.os.getpid()
        return store

    monkeypatch.setattr(fd.StoreMeta, "load", meta_for)
    monkeypatch.setattr(fd, "run_fast_checks", lambda *a, **kw: None)
    monkeypatch.setattr(fd, "FrameSampStore", store_for)
    stats = SimpleNamespace(q01=-np.ones(8), q99=np.ones(8) * 2, mean=np.zeros(8), std=np.ones(8))
    data_config = SimpleNamespace(norm_stats={"state": stats}, use_quantile_norm=True)

    def create(hc, horizon=20):
        return fd.FrameSampDataset(str(tmp_path / str(hc.token_per_image)), data_config, hc, horizon)

    return SimpleNamespace(create=create, episodes=episodes, source=source, stores=stores)


def gather(store, tokens, row_base):
    key = TOKENS_OF[tokens]

    def fn(indices):
        return {t: {f"image_emb_{key}": store.image[row_base + t][None], f"pos_emb_{key}": store.pos[t][None],
                    "state_emb": store.state[row_base + t]} for t in indices}
    return fn


def exec_index(ep, step):
    return ep["exec_sample_offset"] + step - ep["exec_start_idx"]


def assert_same(ds, sample, ref):
    img, pos, stt, mask = ref
    assert sample["static_image_emb"].dtype == ml_dtypes.bfloat16
    assert sample["static_image_emb"].shape == img.shape
    assert np.array_equal(sample["static_image_emb"].astype(np.float32), img.astype(np.float32))
    for got, want in ((sample["static_pos_emb"], pos), (sample["static_state_emb"], ds._normalize_state(stt)),
                      (sample["static_mask"], mask)):
        assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()


# ―― YAML ――
@pytest.mark.parametrize("stem", ["perceptual-framesamp-context", "perceptual-tokendrop-context"])
def test_yaml_only_changes_budget_from_upstream(stem):
    upstream = yaml.safe_load(git_show(UPSTREAM, f"{YAMLS}/{stem}.yaml"))
    ours = yaml.safe_load((ROOT / YAMLS / f"{stem}-1024.yaml").read_text())
    assert "motion" not in ours
    assert upstream["budget"] == 512 and ours["budget"] == 1024
    assert {k for k in upstream.keys() | ours.keys() if upstream.get(k) != ours.get(k)} == {"budget"}


@pytest.mark.parametrize("name,tokens,kind", [("framesamp", 16, "frame_sampling"), ("tokendrop", 64, "token_dropping")])
def test_real_yaml_admitted(lib, name, tokens, kind):
    hc = OmegaConf.load(ROOT / YAMLS / f"perceptual-{name}-context-1024.yaml")
    assert (hc.token_per_image, hc.perceptual_memory.type) == (tokens, kind)
    lib.create(hc)


# ―― 形制白名单 ――
@pytest.mark.parametrize("kwargs", [
    dict(tokens=16, kind="token_dropping"),                      # token_dropping 只允许 (1024,64,1)
    dict(budget=512, tokens=64, kind="token_dropping"),
    dict(budget=2048, tokens=64, kind="token_dropping"),
    dict(kind="token_dropping", fusion="modulation", dim=1024),
    dict(kind="token_dropping", motion=True),
    dict(tokens=64, kind="frame_sampling"),                      # (1024,64,1)+context 不放行 frame_sampling
    dict(tokens=16, kind="frame_sampling", fusion="modulation", dim=1024),
    dict(tokens=16, kind="frame_sampling", motion=True),
    dict(kind="expert"),
])
def test_shape_rejected(lib, kwargs):
    with pytest.raises(ValueError, match="形制断言失败"):
        lib.create(history(**kwargs))


# ―― token_dropping ――
def test_token_dropping_matches_memory_buffer(lib):
    ds = lib.create(history())
    buf, store = MemoryBuffer(), lib.stores[64]
    over_budget, padded = 0, 0
    for ep in lib.episodes:
        path = lib.source / "features" / f"episode_{ep['global_episode_idx']}" / "kept_indices.json"
        kept = json.loads(path.read_text())
        es, nt = ep["exec_start_idx"], ep["num_timesteps"]
        for step in sorted({es, es + 1, max(es, nt // 2), nt - 1}):
            ref = buf.prepare_token_dropping(step, 1024, gather(store, 64, ep["total_sample_offset"]), kept)
            sample = ds[exec_index(ep, step)]
            assert_same(ds, sample, ref)
            assert sample["static_image_emb"].shape == (1024, 2048)
            assert sample["static_pos_emb"].shape == (1024, 768)
            assert sample["static_state_emb"].shape == (1024, 8)
            assert sample["actions"].shape[0] == 20
            assert "simple_subgoal_online" not in sample
            assert sample["motion_emb"] is None
            count = sum(item[0] <= step for item in kept)
            over_budget += count > 1024
            padded += count < 1024
            k = min(count, 1024)
            np.testing.assert_array_equal(sample["static_mask"], np.arange(1024) < k)
            assert not np.any(sample["static_image_emb"][k:].astype(np.float32))
            assert not np.any(sample["static_pos_emb"][k:])
    assert over_budget
    assert padded


def test_token_dropping_keeps_json_tail_then_sorts(lib):
    """独立于 MemoryBuffer 的手算：超预算时留 JSON 序末尾（分数最高），再按 (frame,view,patch) 排。"""
    ep = lib.episodes[0]
    kept = json.loads((lib.source / "features" / "episode_0" / "kept_indices.json").read_text())
    ds = lib.create(history())
    assert ds._kept_indices[0].tolist() == kept            # JSON 原顺序未被改动
    step = ep["num_timesteps"] - 1
    want = sorted(tuple(item) for item in [item for item in kept if item[0] <= step][-1024:])
    assert len(want) == 1024
    store = lib.stores[64]
    sample = ds[exec_index(ep, step)]
    expected = np.stack([store.pos[t][p] for t, _, p in want])
    assert sample["static_pos_emb"].tobytes() == expected.tobytes()


def test_token_dropping_survives_pickle_without_rereading(lib):
    ds = lib.create(history())
    idx = exec_index(lib.episodes[1], 30)
    before = ds[idx]
    assert ds._store is not None
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._store is None
    assert clone._atexit_registered is False
    (lib.source / "features" / "episode_1" / "kept_indices.json").unlink()  # worker 不再读 JSON
    after = clone[idx]
    for key in ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"):
        assert before[key].tobytes() == after[key].tobytes()


@pytest.mark.parametrize("payload", ["[[0, 0, 64]]", "[[0, 1, 0]]", "[[9999, 0, 0]]", "[[-1, 0, 0]]", "[]", "[[0, 0]]"])
def test_bad_kept_indices_fail_loud(lib, payload):
    (lib.source / "features" / "episode_2" / "kept_indices.json").write_text(payload)
    with pytest.raises(ValueError, match="kept_indices.json"):
        lib.create(history())


def test_frame_sampling_does_not_load_kept_indices(lib):
    (lib.source / "features" / "episode_2" / "kept_indices.json").unlink()
    assert lib.create(history(tokens=16, kind="frame_sampling"))._kept_indices is None


# ―― frame_sampling 1024 ――
def test_frame_sampling_1024_matches_memory_buffer(lib):
    ds = lib.create(history(tokens=16, kind="frame_sampling"))
    buf, store = MemoryBuffer(), lib.stores[16]
    for ep in lib.episodes:
        for step in sorted({ep["exec_start_idx"], ep["num_timesteps"] - 1}):
            ref = buf.prepare_frame_sampling(step, 1024, 16, gather(store, 16, ep["total_sample_offset"]))
            sample = ds[exec_index(ep, step)]
            assert_same(ds, sample, ref)
            assert sample["static_image_emb"].shape == (1024, 2048)
            assert sample["static_mask"].sum() == min(step + 1, 64) * 16


# ―― 关闭态：与基线 sha 的实现同字节 ――
def baseline_module():
    module = ModuleType("baseline_framesamp_dataset_ctx1024")
    relative = "src/mme_vla_suite/training/framesamp_dataset.py"
    exec(compile(git_show(BASELINE, relative), str(ROOT / relative), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("kwargs", [
    dict(budget=512, tokens=16, kind="frame_sampling"),
    dict(budget=512, tokens=64, kind="frame_sampling"),
    dict(budget=2048, tokens=64, kind="frame_sampling"),
    dict(budget=512, tokens=16, kind="frame_sampling", fusion="modulation", dim=1024),
    dict(budget=1024, tokens=64, kind="frame_sampling", fusion="modulation", dim=1024),
])
def test_existing_shapes_byte_identical_to_baseline(lib, kwargs):
    old = baseline_module()
    ds = lib.create(history(**kwargs))
    ds._ensure_store()
    before = old.FrameSampDataset.__new__(old.FrameSampDataset)
    before.__dict__.update(ds.__dict__)
    for ep in lib.episodes:
        for step in (ep["exec_start_idx"], ep["num_timesteps"] - 1):
            idx = exec_index(ep, step)
            want, got = before[idx], ds[idx]
            assert want.keys() == got.keys()
            for key, value in want.items():
                if isinstance(value, np.ndarray):
                    assert (value.dtype, value.shape, value.tobytes()) == (got[key].dtype, got[key].shape, got[key].tobytes())
                else:
                    assert value == got[key]
