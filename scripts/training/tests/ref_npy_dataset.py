"""仅供验证的源 npy 参考链；生产 loader 不导入本模块。"""

from __future__ import annotations

import atexit
import os
import pathlib
import pickle

import numpy as np

from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.framesamp_store import load_manifest
from mme_vla_suite.shared.sampling import even_sampling_indices, memory_order, pad_times


class RefNpyFrameSampDataset:
    """绕开 packed 表与候选 Dataset，保留同一采样和 motion 协议。"""

    def __init__(self, source_root, manifest_path, data_config, history_config,
                 action_horizon, motion_root=None):
        hc = history_config
        if (int(hc.budget), int(hc.token_per_image), int(hc.num_views)) not in {(512, 16, 1), (512, 64, 1)}:
            raise ValueError("参考链只支持 32×16 与 8×64")
        if (hc.representation_type != "perceptual"
                or hc.integration_type not in ("context", "modulation")
                or hc.perceptual_memory.type != "frame_sampling"):
            raise ValueError("参考链形制不符")
        self._source_root = str(pathlib.Path(source_root).resolve())
        self._manifest_path = str(pathlib.Path(manifest_path).resolve())
        manifest = load_manifest(self._manifest_path)
        self._manifest = manifest
        n = int(manifest["totals"]["exec_samples"])
        self._epis_of = np.full(n, -1, dtype=np.int64)
        self._step_of = np.full(n, -1, dtype=np.int64)
        for ep in manifest["episodes"]:
            off, count = ep["exec_sample_offset"], ep["exec_samples"]
            if count != ep["num_timesteps"] - ep["exec_start_idx"] or np.any(self._epis_of[off:off + count] != -1):
                raise ValueError("清单执行样本区间重叠或长度不符")
            self._epis_of[off:off + count] = ep["global_episode_idx"]
            self._step_of[off:off + count] = np.arange(ep["exec_start_idx"], ep["num_timesteps"])
        if np.any(self._epis_of < 0):
            raise ValueError("清单执行样本区间有空洞")
        self._tokens_per_frame = int(hc.token_per_image)
        self._max_frames = 512 // self._tokens_per_frame
        g = 8 if self._tokens_per_frame == 64 else 4
        self._image_key, self._pos_key = f"image_emb_{g}x{g}", f"pos_emb_{g}x{g}"
        self.action_horizon = action_horizon
        self.state_norm_stats = data_config.norm_stats["state"]
        self.use_quantiles = data_config.use_quantile_norm
        mc = getattr(hc, "motion", None)
        self._motion_enabled = bool(mc is not None and mc.get("enabled", False))
        self._motion_root = pathlib.Path(motion_root) if motion_root else None
        self._motion_meta = None
        self._mstore = None
        self._atexit_registered = False
        if self._motion_enabled:
            if self._motion_root is None:
                raise ValueError("参考链开启 motion 时必须显式给出 motion_root")
            self._motion_meta = ms.MotionMeta.load(self._motion_root)
            ms.run_fast_checks(self._motion_meta, manifest_path=self._manifest_path)
            ms.check_same_source(manifest["sha256"], self._motion_meta, manifest)
            self._motion_budget, self._motion_pos_dim = int(mc.budget), int(mc.pos_dim)
            if (self._motion_budget, self._motion_pos_dim, int(mc.stride), int(mc.window_frames)) != (96, 256, 16, 33):
                raise ValueError("参考链 motion 协议不符")
            if any(ms.max_visible_count(e) > self._motion_budget for e in self._motion_meta.entries):
                raise ValueError("合法 motion 窗数超过预算")

    def __len__(self):
        return len(self._epis_of)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_mstore"] = None
        d["_atexit_registered"] = False
        return d

    def _ensure_motion_store(self):
        if self._mstore is not None and self._mstore.owner_pid != os.getpid():
            self.close()
        if self._mstore is None:
            self._mstore = ms.MotionStore(self._motion_root, meta=self._motion_meta,
                                         manifest_path=self._manifest_path)
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True
        return self._mstore

    def close(self):
        if self._mstore is not None:
            self._mstore.close()
            self._mstore = None

    def _load_frame(self, episode, frame):
        p = pathlib.Path(self._source_root) / "features" / f"episode_{episode}" / f"token_emb_{frame}.npy"
        return np.load(p, allow_pickle=True).item()

    def __getitem__(self, idx):
        # 延迟导入补零函数，避免主进程构造时初始化 JAX；原实现完整复用。
        from mme_vla_suite.shared.data_utils import right_padding_token_emb

        idx = int(idx)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        g, step = int(self._epis_of[idx]), int(self._step_of[idx])
        with (pathlib.Path(self._source_root) / "data" / f"{idx}.pkl").open("rb") as f:
            data = pickle.load(f)
        if (int(data["epis_idx"].item()), int(data["step_idx"].item())) != (g, step):
            raise RuntimeError(f"参考链身份互校失败: index={idx}")
        data["actions"] = data["actions"][:self.action_horizon]
        data.pop("simple_subgoal_online")
        data.pop("grounded_subgoal_online")
        frames = even_sampling_indices(step, self._max_frames)
        features = [self._load_frame(g, f) for f in frames]
        img = np.stack([f[self._image_key] for f in features])
        pos = np.stack([f[self._pos_key] for f in features])
        state = np.stack([f["state_emb"] for f in features])
        img, pos, state, mask = right_padding_token_emb(
            img, pos, state, np.ones(len(frames), dtype=np.bool_), self._max_frames)
        state = np.repeat(state, self._tokens_per_frame, axis=0)
        ns = self.state_norm_stats
        if self.use_quantiles:
            state = (state - ns.q01) / (ns.q99 - ns.q01 + 1e-6) * 2.0 - 1.0
        else:
            state = (state - ns.mean) / (ns.std + 1e-6)
        data.update(static_image_emb=img.reshape(-1, img.shape[-1]),
                    static_pos_emb=pos.reshape(-1, pos.shape[-1]),
                    static_state_emb=state, static_mask=np.repeat(mask, self._tokens_per_frame))
        if self._motion_enabled:
            rows, times = ms.visible_motion_rows(self._motion_meta.entries[g], step)
            k = len(rows)
            if k > self._motion_budget:
                raise ValueError("参考链禁止截断 motion")
            emb = np.zeros((self._motion_budget, 768), dtype=np.float32)
            mpos = np.zeros((self._motion_budget, self._motion_pos_dim), dtype=np.float32)
            mmask = np.zeros(self._motion_budget, dtype=np.bool_)
            if k:
                emb[:k] = self._ensure_motion_store().rows(rows)
                mpos[:k] = np.stack([self._load_frame(g, int(t))[self._pos_key][0, 0, :self._motion_pos_dim] for t in times])
                mmask[:k] = True
            data.update(motion_emb=emb, motion_pos=mpos, motion_mask=mmask,
                        mem_order=memory_order(pad_times(frames, self._max_frames), self._tokens_per_frame,
                                               pad_times(times, self._motion_budget)))
        for key in ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask", "prompt",
                    "motion_emb", "motion_pos", "motion_mask", "mem_order"):
            data.setdefault(key, None)
        return data


def fixture_manifest_path(dataset_path):
    """显式清单优先；packed 侧可从 meta 还原，源 npy 侧必须显式指定。"""
    import json
    value = os.environ.get("DTYPE_MANIFEST")
    if value:
        return pathlib.Path(value)
    return pathlib.Path(json.loads((pathlib.Path(dataset_path) / "meta/store_meta.json").read_text())["manifest_path"])


def create_fixture_dataset(dataset_path, data_config, history_config, action_horizon):
    """取证工具专用分派，不改变生产分派函数。"""
    impl = os.environ.get("DTYPE_DUMP_IMPL", "packed")
    if impl == "packed":
        from mme_vla_suite.training.dataloader import _create_framesamp_dataset
        return _create_framesamp_dataset(dataset_path, data_config, history_config, action_horizon)
    if impl != "refnpy":
        raise ValueError(f"未知 DTYPE_DUMP_IMPL={impl}")
    return RefNpyFrameSampDataset(dataset_path, fixture_manifest_path(dataset_path), data_config,
                                 history_config, action_horizon, os.environ.get("MMEVLA_MOTION_STORE"))
