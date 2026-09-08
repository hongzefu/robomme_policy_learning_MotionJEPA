#!/usr/bin/env python3
"""决策点驱动（阶段 0 / 阶段 1 / adapter 自检共用）：按真实评估节奏把 h5 原始帧喂进生产口径 policy，
在每个决策时刻产出 `HistAugObservation`（走生产 `_prepare_history` + `_input_transform`）与训练侧真值动作。

骨架逐字派生自 scripts/training/g0/compare_train_infer_obs.py（load_episode_full / expected_taus / build_points /
feed 节奏 / MotionStore 查表闭包），只保留 B 臂（在线 bf16 SigLIP）+ store 档 motion；
与之不同的两点：
  - `resolve_episodes` 不要求 es 取值互异（阶段 1 需要同时含 ButtonUnmask 与 ButtonUnmaskSwap，两者 es 都是 0），
    并支持 `<task>:longest` 取该任务 T 最长的一集；
  - policy 用 mv_model_adapter.build_policy 构造（不起 stub 子进程），motion 编码句柄是查表闭包。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mv_common as C  # noqa: E402
from mv_model_adapter import ZeroMotion, build_policy  # noqa: E402

DEFAULT_EPISODES_OPEN_LOOP = ("ButtonUnmask:longest,ButtonUnmaskSwap:3,VideoUnmask:3,"
                              "VideoUnmaskSwap:5,VideoUnmaskSwap:31,VideoUnmaskSwap:3")


def load_episode_full(raw_dir: pathlib.Path, h5_file: str, raw_ep_idx: int, T: int, n_read: int):
    """读 h5 一集的前 n_read 帧（front_rgb / wrist_rgb / joint+gripper）与 setup/task_goal 原文。"""
    import h5py
    frames, wrists, states = [], [], []
    with h5py.File(raw_dir / h5_file, "r") as f:
        g = f[f"episode_{raw_ep_idx}"]
        ts_ids = sorted(int(k.split("_")[-1]) for k in g.keys() if k.startswith("timestep_"))
        if len(ts_ids) != T or ts_ids != list(range(T)):
            raise SystemExit(f"错误: {h5_file} episode_{raw_ep_idx} timesteps {len(ts_ids)} != 清单 {T}")
        tg = g["setup"]["task_goal"][()]
        task_goal = (tg[0] if getattr(tg, "shape", ()) else tg)
        task_goal = task_goal.decode() if isinstance(task_goal, bytes) else str(task_goal)
        for t in ts_ids[:n_read]:
            ts = g[f"timestep_{t}"]
            img = ts["obs"]["front_rgb"][()]
            wr = ts["obs"]["wrist_rgb"][()]
            if img.shape != (256, 256, 3) or img.dtype != np.uint8 or wr.shape != (256, 256, 3) or wr.dtype != np.uint8:
                raise SystemExit(f"错误: 帧形制 front={img.shape} {img.dtype} wrist={wr.shape} {wr.dtype}")
            frames.append(img)
            wrists.append(wr)
            joint = ts["obs"]["joint_state"][()]
            grip = ts["obs"]["gripper_state"][()]
            states.append(np.concatenate([joint, grip[:1]], axis=0, dtype=np.float32))
    return np.stack(frames)[:, None], np.stack(wrists), np.stack(states), task_goal


def resolve_episodes(manifest: dict, spec_str: str) -> list[dict]:
    """解析 `<task>:<raw_ep_idx>|longest` 列表；缺失 / 重复即 SystemExit（不静默回退）。"""
    eps = manifest["episodes"]
    picked: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for spec in spec_str.split(","):
        if ":" not in spec:
            raise SystemExit(f"错误: --episodes 条目 {spec!r} 不是 <task>:<raw_ep_idx|longest>")
        task, raw = spec.rsplit(":", 1)
        h5 = f"record_dataset_{task}.h5"
        if raw == "longest":
            cand = [e for e in eps if e["h5_file"] == h5]
            if not cand:
                raise SystemExit(f"错误: 库 manifest 中没有任务 {task}")
            e = max(cand, key=lambda x: (int(x["num_timesteps"]), -int(x["raw_ep_idx"])))
        else:
            cand = [e for e in eps if e["h5_file"] == h5 and int(e["raw_ep_idx"]) == int(raw)]
            if len(cand) != 1:
                raise SystemExit(f"错误: 条目 {spec!r} 在库 manifest 中命中 {len(cand)} 条")
            e = cand[0]
        key = (h5, int(e["raw_ep_idx"]))
        if key in seen:
            raise SystemExit(f"错误: 条目 {spec!r} 重复")
        seen.add(key)
        picked.append(e)
    return picked


def build_points(picked: list[dict], max_points: int) -> list[dict]:
    out = []
    for e in picked:
        es = int(e["exec_start_idx"])
        T = int(e["num_timesteps"])
        taus = C.expected_taus(es, T)
        if max_points > 0:
            taus = taus[:max_points]
        out.append({"h5_file": e["h5_file"], "task": C.task_of_h5(e["h5_file"]), "raw_ep_idx": int(e["raw_ep_idx"]),
                    "g": int(e["global_episode_idx"]), "es": es, "T": T, "c_env": T - 1 - es,
                    "row_base": int(e["total_sample_offset"]), "taus": taus})
    return out


class PointSource:
    def __init__(self, *, lib: pathlib.Path, ckpt_dir: pathlib.Path, train_config_name: str,
                 episodes_spec: str, max_points: int = 0, seed: int = 42):
        os.chdir(C.REPO_ROOT)      # train_config.data.assets.assets_dir 是仓库相对路径
        self.lib = pathlib.Path(lib).resolve()
        self.ckpt_dir = pathlib.Path(ckpt_dir).resolve()
        self.run_root = self.ckpt_dir.parent
        motion_root = C.require_motion_store(self.lib)

        import jax
        import jax.numpy as jnp
        import omegaconf
        from openpi.training.data_loader import transform_dataset
        from mme_vla_suite.training import config as _config
        from mme_vla_suite.datastore import motion_store as ms
        from mme_vla_suite.datastore.framesamp_store import StoreMeta
        from mme_vla_suite.training.dataloader import _create_framesamp_dataset, _motion_gates
        from mme_vla_suite.models.integration.history_observation import HistAugObservation
        self.jax, self.jnp, self.ms = jax, jnp, ms
        self._HistAugObservation = HistAugObservation

        t0 = time.perf_counter()
        self.train_config = _config.get_config(train_config_name)
        self.policy = build_policy(self.train_config, self.ckpt_dir, seed=seed, motion_factory=lambda **kw: ZeroMotion())
        if not self.policy.motion_enabled:
            raise SystemExit("错误: 被评 checkpoint 不是 motion 开启态")
        self.model = self.policy._model
        print(f"[mv_points] policy 构造 {time.perf_counter() - t0:.1f}s ckpt={self.ckpt_dir}", flush=True)

        data_config = self.train_config.data.create(self.train_config.assets_dirs, self.train_config.model)
        hc = omegaconf.OmegaConf.load(self.run_root / "history_config.resolved.yaml")
        self.ds_raw = _create_framesamp_dataset(str(self.lib / "framesamp"), data_config, hc, int(self.model.action_horizon))
        self.ds_tf = transform_dataset(self.ds_raw, data_config)
        fmeta = StoreMeta.load(str(self.lib / "framesamp"))
        mroot = _motion_gates(hc, fmeta)
        if pathlib.Path(mroot).resolve() != motion_root.resolve():
            raise SystemExit(f"错误: dataloader 解析的 motion 库 {mroot} != {motion_root}")
        self.mmeta = ms.MotionMeta.load(mroot)
        self.mstore = ms.MotionStore(mroot, meta=self.mmeta)
        self.entries = self.ds_raw._motion_entries
        self.manifest = json.load(open(self.lib / "meta" / "episode_manifest.json", encoding="utf-8"))
        self.raw_dir = pathlib.Path(self.manifest["raw_dir"])
        self.ns_actions_std = float(np.mean(np.asarray(data_config.norm_stats["actions"].std)))
        self.picked = resolve_episodes(self.manifest, episodes_spec)
        self.eps = build_points(self.picked, max_points)
        for e in self.eps:
            for t in e["taus"]:
                hit = np.flatnonzero((self.ds_raw._epis_of == e["g"]) & (self.ds_raw._step_of == t))
                if len(hit) != 1:
                    raise SystemExit(f"错误: 决策时刻 (g={e['g']}, t={t}) 在 FrameSampDataset 中命中 {len(hit)} 条样本")
        self.n_points = sum(len(e["taus"]) for e in self.eps)
        print("POINTS episodes=" + str(len(self.eps)) + f" points={self.n_points} " + " ".join(
            f"{e['task']}:{e['raw_ep_idx']}(g={e['g']},es={e['es']},T={e['T']},pts={len(e['taus'])})" for e in self.eps), flush=True)

    def obs_of(self, inputs_np: dict):
        jax, jnp = self.jax, self.jnp
        return self._HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_np))

    def iter_points(self, max_total: int = 0):
        policy = self.policy
        n_yield = 0
        for ep in self.eps:
            g, es, T = ep["g"], ep["es"], ep["T"]
            taus = ep["taus"]
            if not taus:
                continue
            entry = self.entries[g]
            rows_all, f_all = self.ms.visible_motion_rows(entry, T - 1)
            f2row = {int(f): int(r) for r, f in zip(rows_all.tolist(), f_all.tolist())}
            mstore = self.mstore

            class _Lookup:
                def __call__(self_, window, start_frame):
                    return np.asarray(mstore.rows(np.asarray([f2row[int(start_frame)]], dtype=np.int64))[0], dtype=np.float32)

                def close(self_):
                    pass

            policy._motion_client = _Lookup()
            policy.reset()                       # 重建 FrameSampMemory，注入查表闭包
            n_read = min(T, taus[-1] + 1)
            frames, wrists, states, task_goal = load_episode_full(self.raw_dir, ep["h5_file"], ep["raw_ep_idx"], T, n_read)

            def feed(lo, hi):
                policy.add_buffer({"images": frames[lo:hi], "state": states[lo:hi],
                                   "exec_start_idx": es if lo == 0 else 0})

            def point(t: int, j: int):
                if policy.step_idx != t:
                    raise RuntimeError(f"policy.step_idx {policy.step_idx} != 决策时刻 {t}")
                element = {"observation/image": frames[t, 0], "observation/wrist_image": wrists[t],
                           "observation/state": states[t], "prompt": task_goal}
                assembled = policy._prepare_history(dict(element))
                infer_inputs = policy._input_transform(dict(assembled))
                hit = np.flatnonzero((self.ds_raw._epis_of == g) & (self.ds_raw._step_of == t))
                idx = int(hit[0])
                actions = np.asarray(self.ds_tf[idx]["actions"], np.float32)
                vis = policy.mem_buffer.visible_motion_frames(t)
                return {"g": g, "task": ep["task"], "raw_ep_idx": ep["raw_ep_idx"], "es": es, "T": T, "t": t,
                        "point_idx": j, "n_points": len(taus), "k": len(vis), "frames": [int(x) for x in vis],
                        "entry": entry, "f2row": f2row, "idx": idx, "element": element, "assembled": assembled,
                        "infer_inputs": infer_inputs, "obs": self.obs_of(infer_inputs), "actions": actions,
                        "mem_order": np.asarray(assembled["mem_order"]), "motion_mask": np.asarray(assembled["motion_mask"])}

            feed(0, es + 1)
            yield point(es, 0)
            n_yield += 1
            if max_total and n_yield >= max_total:
                return
            for j, tau in enumerate(taus[1:], start=1):
                feed(tau - 15, tau + 1)
                yield point(tau, j)
                n_yield += 1
                if max_total and n_yield >= max_total:
                    return

    def close(self):
        self.mstore.close()
        self.ds_raw.close()
