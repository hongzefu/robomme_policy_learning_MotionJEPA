#!/usr/bin/env python3
"""会落盘 VLA 主干 attention 的策略服务端（计划 1103-interpret-motion-attention-plan.html 第二部分 §1.2）。

命令行 = scripts/training/serve_policy.py::Args 全集 + --attn-out（必填）+ --save-top-frames（默认 3）。
构造与 metadata、websocket 服务一律复用 serve_policy：本模块只把 serve_policy.create_policy 换成
「原函数构造 → 换 _sample_actions 为 AttnAdapter 展开版 → 挂记录器」，再调 serve_policy.main，
因此 eval.py validate_metadata 核的 run_identity_sha256 / server_nonce / model_seed / model_kind 与主线逐字段相同。

src/ 零改动：所有遮蔽都在实例属性上（policy.reset / add_buffer / infer / _sample_actions、
mem_buffer._prepare_motion），websocket 层只把模块内的 RpcSession 名字换成一个记录 rpc 的子类，
用来拿到 reset 时的 episode_nonce（服务端调用 policy.reset() / infer() 时不传 rpc）。

每次 infer 返回之后写 <attn-out>/<episode_nonce>/infer_<seq:04d>.npz（键表见 NPZ_KEYS，与 §1.3 单一真源一致），
不压缩；每局目录另写 episode.json，top-N 帧原图写 frames/<frame_idx>.jpg（去重）。

用法（通常由 scripts/interpret/run_local.py 启动）：
  python -m scripts.interpret.serve_policy_attn --port 8011 --seed 42 --attn-out <dir> \
      policy:checkpoint --policy.config <cfg> --policy.dir <ckpt>
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
import time
import uuid

import numpy as np

# §1.3 npz 键表（单一真源在计划里；S3 只按这里的键消费）
NPZ_KEYS = ("P", "MEM", "step_idx", "exec_start_idx", "infer_seq", "mem_order", "mem_mask", "frame_times",
            "motion_times", "motion_mask", "txt_len", "actions", "state", "top_frames", "probs_action", "probs_text")
NUM_STEPS, NUM_LAYERS, NUM_HEADS, ACTION_HORIZON, TEXT_LEN = 10, 18, 8, 20, 64
ACTION_DIM, STATE_DIM = 8, 8

# websocket 层本次 RPC 的操作名与 eval_rpc（由 _TrackingSession.validate 写入；同步执行，无并发）
_RPC = {"op": None, "rpc": None}


def _pad_times(times, budget: int) -> np.ndarray:
    """与 policy.py 里 pad_times 同法右填充到 budget，只是 padding 记 −1（§1.3 契约）。"""
    t = np.asarray(times, dtype=np.int64).reshape(-1)
    if t.size > budget:
        raise ValueError(f"真实时刻数 {t.size} > 预算 {budget}")
    out = np.full(budget, -1, dtype=np.int32)
    out[:t.size] = t
    return out


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _save_jpg(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(image).save(path, quality=95)
    except ImportError:
        import cv2
        cv2.imwrite(str(path), image[..., ::-1])


def top_frame_indices(probs_action, mem_order, frame_times, tokens_per_frame: int, n: int) -> np.ndarray:
    """帧流前 n 名的全域帧号：probs_action 全平均（步 × 层 × 头 × query，f32）后，每帧 tokens_per_frame 个 token 求和。"""
    if n <= 0:
        return np.zeros(0, np.int32)
    mem = int(mem_order.shape[0])
    a = np.asarray(probs_action).mean(axis=(0, 1, 2, 3), dtype=np.float32)[:mem]
    n_frames = int(frame_times.shape[0])
    frame_budget = n_frames * tokens_per_frame
    cand = np.asarray(mem_order, np.int64)
    is_frame = cand < frame_budget
    mass = np.bincount(cand[is_frame] // tokens_per_frame, weights=a[is_frame], minlength=n_frames)[:n_frames]
    valid = np.flatnonzero(frame_times >= 0)
    order = valid[np.argsort(-mass[valid], kind="stable")][:n]
    return frame_times[order].astype(np.int32)


class AttnRecorder:
    """在 policy 实例属性上遮蔽 reset / add_buffer / infer / _sample_actions，每次推理落一份 npz。"""

    def __init__(self, policy, adapter, attn_out, *, save_top_frames: int = 3):
        self.policy, self.adapter = policy, adapter
        self.attn_out = Path(attn_out)
        self.save_top_frames = int(save_top_frames)
        cfg = policy.config
        self.budget = int(cfg.budget)
        self.token_per_image = int(cfg.token_per_image)
        self.num_views = int(cfg.num_views)
        self.tokens_per_frame = self.token_per_image * self.num_views
        self.max_frames = self.budget // self.tokens_per_frame
        self.motion_enabled = bool(policy.motion_enabled)
        self.motion_budget = int(cfg.motion.budget) if self.motion_enabled else 0
        self.ep_seq = 0
        self.ep_dir: Path | None = None
        self.episode: dict = {}
        self.infer_seq = 0
        self.frames: dict[int, np.ndarray] = {}
        self.saved_frames: set[int] = set()
        self.captured: dict = {}
        self.printed_env = False
        self._orig_reset, self._orig_add, self._orig_infer = policy.reset, policy.add_buffer, policy.infer

    # ―― 安装 ――
    def install(self) -> None:
        inner = self.adapter.make_sample_fn()

        def sample(rng, observation, **kwargs):
            # 只存引用，不在计时段内搬回主机；记录在 infer 返回之后做
            self.captured["static_mask"] = getattr(observation, "static_mask", None)
            self.captured["motion_mask_obs"] = getattr(observation, "motion_mask", None)
            self.captured["mem_order"] = getattr(observation, "mem_order", None)
            self.captured["prompt_mask"] = getattr(observation, "tokenized_prompt_mask", None)
            return inner(rng, observation, **kwargs)

        self.policy._sample_actions = sample
        self.policy.reset = self.reset
        self.policy.add_buffer = self.add_buffer
        self.policy.infer = self.infer
        self._hook_mem_buffer()

    def _hook_mem_buffer(self) -> None:
        # mem_buffer 每次 reset 重建，包装须在 reset 后重新挂；只捕获返回值，不重复调用（不重复编码 sidecar）
        mem_buffer = self.policy.mem_buffer
        if not self.motion_enabled or mem_buffer is None:
            return
        original = mem_buffer._prepare_motion

        def prepare_motion(step_idx):
            result = original(step_idx)
            self.captured["motion"] = result
            return result

        mem_buffer._prepare_motion = prepare_motion

    # ―― 遮蔽的 policy 方法 ――
    def reset(self) -> None:
        self._orig_reset()
        self._hook_mem_buffer()
        self.ep_seq += 1
        rpc = _RPC["rpc"] if _RPC["op"] == "reset" else None
        nonce = rpc.get("episode_nonce") if isinstance(rpc, dict) else None
        if not nonce:
            nonce = f"local-{uuid.uuid4().hex}"
        self.ep_dir = self.attn_out / str(nonce)
        self.ep_dir.mkdir(parents=True, exist_ok=False)
        self.infer_seq = 0
        self.frames.clear()
        self.saved_frames.clear()
        self.episode = {"episode_nonce": str(nonce), "ep_seq": self.ep_seq, "reset_rpc": rpc,
                        "server_pid": os.getpid(), "reset_unix_time": time.time(),
                        "model_kind": self.policy.metadata.get("model_kind"), "motion_enabled": self.motion_enabled,
                        "save_top_frames": self.save_top_frames, "infers": 0, "prompt": None}
        _write_json(self.ep_dir / "episode.json", self.episode)
        print(f"ATTN_EPISODE_BEGIN nonce={nonce} ep_seq={self.ep_seq} dir={self.ep_dir}", flush=True)

    def add_buffer(self, obs: dict) -> None:
        before = int(self.policy.step_idx)
        self._orig_add(obs)
        if self.save_top_frames > 0:
            images = np.asarray(obs["images"])
            for i, step in enumerate(range(before + 1, int(self.policy.step_idx) + 1)):
                self.frames[step] = images[i]

    def infer(self, obs: dict) -> dict:
        if self.ep_dir is None:
            raise RuntimeError("infer 之前必须先 reset（无 episode 目录）")
        self.captured = {}
        outputs = self._orig_infer(obs)
        # 写盘在 infer 返回之后（infer_time_ms 已在原 infer 计时段内算完）
        record = self.build_record(obs, outputs)
        path = self.ep_dir / f"infer_{self.infer_seq:04d}.npz"
        tmp = self.ep_dir / f".infer_{self.infer_seq:04d}.{uuid.uuid4().hex}.tmp.npz"
        np.savez(tmp, **record)
        os.replace(tmp, path)
        self._save_frames(record["top_frames"])
        if not self.printed_env:
            print(f"ATTN_SERVE_ENV P={int(record['P'])} motion={int(self.motion_enabled)} L={NUM_LAYERS} AH={ACTION_HORIZON} "
                  f"attn_out={self.attn_out}", flush=True)
            self.printed_env = True
        print(f"ATTN_INFER nonce={self.episode['episode_nonce']} infer_seq={self.infer_seq} step_idx={int(record['step_idx'])} "
              f"top_frames={record['top_frames'].tolist()} npz={path.name}", flush=True)
        self.infer_seq += 1
        self.episode.update(infers=self.infer_seq, prompt=self.episode.get("prompt") or _as_str(obs.get("prompt")))
        _write_json(self.ep_dir / "episode.json", self.episode)
        return outputs

    # ―― 记录 ――
    def build_record(self, obs: dict, outputs: dict) -> dict:
        last = getattr(self.adapter, "last_attn", None)
        if not isinstance(last, dict):
            raise RuntimeError("adapter.last_attn 缺失：采样函数没有走 AttnAdapter")
        P, MEM = int(last["P"]), int(last["MEM"])
        expected_mem = self.budget + self.motion_budget
        if MEM != expected_mem:
            raise RuntimeError(f"MEM={MEM} 与配置 budget+motion.budget={expected_mem} 不符")
        probs_action = np.asarray(last["probs_action"]).astype(np.float16, copy=False)
        probs_text = np.asarray(last["probs_text"]).astype(np.float16, copy=False)
        if probs_action.shape != (NUM_STEPS, NUM_LAYERS, NUM_HEADS, ACTION_HORIZON, P + ACTION_HORIZON):
            raise RuntimeError(f"probs_action 形状 {probs_action.shape} 不符")
        if probs_text.shape != (NUM_LAYERS, NUM_HEADS, TEXT_LEN, P):
            raise RuntimeError(f"probs_text 形状 {probs_text.shape} 不符")

        static_mask = np.asarray(self.captured["static_mask"]).reshape(-1).astype(bool)
        if self.motion_enabled:
            if self.captured.get("motion") is None or self.captured.get("mem_order") is None:
                raise RuntimeError("motion 开启但本次推理未捕获 _prepare_motion / mem_order")
            mem_order = np.asarray(self.captured["mem_order"]).reshape(-1).astype(np.int32)
            obs_motion_mask = np.asarray(self.captured["motion_mask_obs"]).reshape(-1).astype(bool)
            full_mask = np.concatenate([static_mask, obs_motion_mask])
            _, _, motion_mask, motion_times = self.captured["motion"]
            motion_mask = np.asarray(motion_mask).reshape(-1).astype(bool)
            motion_times = np.where(motion_mask, np.asarray(motion_times).reshape(-1), -1).astype(np.int32)
        else:
            mem_order = np.arange(MEM, dtype=np.int32)
            full_mask = static_mask
            motion_mask, motion_times = np.zeros(0, bool), np.zeros(0, np.int32)
        if mem_order.shape != (MEM,) or full_mask.shape != (MEM,):
            raise RuntimeError(f"mem_order {mem_order.shape} / 记忆区 mask {full_mask.shape} 与 MEM={MEM} 不符")
        mem_mask = full_mask[mem_order]

        step_idx = int(self.policy.step_idx)
        frames = self.policy.mem_buffer.get_frame_sampling_indices(step_idx, self.budget, self.token_per_image)
        frame_times = _pad_times(frames, self.max_frames)

        prompt_mask = self.captured.get("prompt_mask")
        txt_len = int(np.asarray(prompt_mask).sum()) if prompt_mask is not None else 0
        actions = np.asarray(outputs["actions"], dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError(f"actions 形状 {actions.shape} 不符 (20, 8)")
        state = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)
        if state.shape != (STATE_DIM,):
            raise RuntimeError(f"observation/state 形状 {state.shape} 不符 (8,)")
        top = top_frame_indices(probs_action, mem_order, frame_times, self.tokens_per_frame, self.save_top_frames)
        return {
            "P": np.int64(P), "MEM": np.int64(MEM), "step_idx": np.int64(step_idx),
            "exec_start_idx": np.int64(int(self.policy.exec_start_idx)), "infer_seq": np.int64(self.infer_seq),
            "mem_order": mem_order, "mem_mask": mem_mask, "frame_times": frame_times,
            "motion_times": motion_times, "motion_mask": motion_mask, "txt_len": np.int64(txt_len),
            "actions": actions, "state": state, "top_frames": top,
            "probs_action": probs_action, "probs_text": probs_text,
        }

    def _save_frames(self, top) -> None:
        if self.save_top_frames <= 0 or len(top) == 0:
            return
        directory = self.ep_dir / "frames"
        directory.mkdir(exist_ok=True)
        for frame in (int(x) for x in top):
            if frame in self.saved_frames:
                continue
            image = self.frames.get(frame)
            if image is None:
                print(f"ATTN_FRAME_MISSING frame={frame}", flush=True)
                continue
            # (views, h, w, 3) 横向拼接各视角
            image = np.asarray(image)
            if image.ndim == 4:
                image = np.concatenate(list(image), axis=1)
            _save_jpg(directory / f"{frame}.jpg", np.ascontiguousarray(image.astype(np.uint8)))
            self.saved_frames.add(frame)


def _as_str(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(np.asarray(value).item()) if isinstance(value, np.ndarray) else str(value)


def _install_rpc_tracking() -> None:
    from mme_vla_suite.serving import websocket_policy_server
    from scripts.evaluation.runtime_contract import RpcSession

    class _TrackingSession(RpcSession):
        def validate(self, obs):
            operation = super().validate(obs)
            _RPC.update(op=operation, rpc=obs.get("eval_rpc"))
            return operation

    websocket_policy_server.RpcSession = _TrackingSession


def _make_args_class():
    from scripts.training import serve_policy

    @dataclasses.dataclass(kw_only=True)
    class Args(serve_policy.Args):
        # 原始 attention 落盘根目录（必填）；每局一个 <episode_nonce>/ 子目录
        attn_out: Path
        # 每次推理另存帧流前 N 名的原图 jpg；0 关闭
        save_top_frames: int = 3

    return Args


def main(args) -> None:
    from scripts.interpret.attn_adapter import AttnAdapter
    from scripts.training import serve_policy

    if args.save_top_frames < 0:
        raise SystemExit("--save-top-frames 不得为负")
    attn_out = Path(args.attn_out).resolve()
    attn_out.mkdir(parents=True, exist_ok=True)
    original_create = serve_policy.create_policy

    def create_policy(policy_args):
        policy = original_create(policy_args)
        recorder = AttnRecorder(policy, AttnAdapter(policy), attn_out, save_top_frames=args.save_top_frames)
        recorder.install()
        P = getattr(recorder.adapter, "P", None)
        if P is not None:
            print(f"ATTN_SERVE_ENV P={int(P)} motion={int(recorder.motion_enabled)} L={NUM_LAYERS} AH={ACTION_HORIZON} "
                  f"attn_out={attn_out}", flush=True)
            recorder.printed_env = True
        return policy

    serve_policy.create_policy = create_policy
    _install_rpc_tracking()
    # metadata、server 构造与 sidecar 收尾全部走主线 main，字段与 serve_policy.py 完全一致
    serve_policy.main(args)


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(_make_args_class()))
