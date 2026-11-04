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
--save-prefix（默认关）开启时，adapter 以 save_prefix=True 构造，每次 infer npz 写完后另写
<attn-out>/<episode_nonce>/prefix_<seq:04d>.npz（键见 PREFIX_NPZ_KEYS：probs_prefix f16 (18, 8, P, P)、P、MEM、
infer_seq、step_idx），不压缩、先写临时文件再 os.replace。
带 motion token 的 VLA 的 infer_<seq>.npz 另存实际输入 motion_emb、motion_pos，去掉 batch 轴并保留原 dtype；
motion_mask 使用实际输入有效位，motion_times 与该 token 轴对齐。两键不写入不带 motion token 的 VLA 的记录。

--ablate <名>（S7 因果遮挡，默认 none）：
  mask_motion            遮所有 motion token（记忆位 mem_order[s] ≥ budget）；基线（无 motion）报错退出
  mask_vis               遮全域帧号 ∈ [0, 32) 的帧 token（帧 f = mem_order[s] // tokens_per_frame，帧号 = frame_times[f]）
  heads:L3H0,L16H0       头遮挡（层、头从 0 起）：该层该头 attention 输出置零
  组合用 + 连接，如 mask_vis+heads:L14H6。key 遮挡只作用于当次有效的记忆 token（与 mem_mask 取与），
  每次推理按观测重算并动态传入 adapter；只改 attention mask 的 key 列，不改 input_mask / positions / mem_order。
  每份 infer npz 另附两键 ABLATE_NPZ_KEYS：ablate（名，str）与 n_masked_keys（本次被遮 key 数，int64；none 时 0）；
  episode.json 记 ablate 与逐次 n_masked_keys；启动打印 ATTN_SERVE_ENV ... ablate=<名>。

motion 内容替换（S8；只换 motion 编码内容，motion token 个数、位置、motion_mask、mem_order 一概不动）：
  motion_zero            每个 motion 窗口的编码换成 np.zeros(dim, f32)
  motion_swap:<DIR>      换成供体局同起点窗口的编码；供体 = DIR/<task>_ep<id>.npz（--record-motion 录下），
                         供体局由 --swap-donor task:ep 指定（run_local 按 spec.swap_map 逐局成块并传入）；
                         供体缺该起点时取起点 ≤ 它的最大窗口，再没有就取供体第一个窗口（donor_lookup）。
  两种替换都先照常调用原 motion client（sidecar 侧调用序列与计时与主线一致），再丢弃其输出换成替换值；
  可与 key 遮挡 / 头遮挡用 + 组合。替换时每份 infer npz 另附 n_motion_replaced（本局截至本次推理累计替换窗数，int64）；
  episode.json 的 motion_content 记 mode、donor、n_windows、n_replaced 与逐窗 replace_log [start, donor_start]。
--record-motion <DIR>：原样返回原编码，同时把本局每个窗口 (start_frame, emb) 写 DIR/<episode_nonce>.npz
  （start_frames int32 (n,)、embs f32 (n, dim)、counts int32 (n,)、mode；swap 时另有 donor_start_frames、donor）；
  另存 call_start_frames int32 (c,)、call_embs f32 (c, dim)、call_seq int64 (c,)，按调用顺序保留重复起点的全部原编码；
  每次 add_buffer / infer 后有新窗口即重写（先写临时文件再 os.replace），进程被杀也不丢已完成窗口；
  记录的永远是原 client 的编码（替换前）。run_local 收尾时按 terminal 记录的局身份把 <nonce>.npz 改名为 <task>_ep<id>.npz。

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
# --save-prefix 另写的 prefix_<seq>.npz 键表
PREFIX_NPZ_KEYS = ("probs_prefix", "P", "MEM", "infer_seq", "step_idx")
# S7 附加键（每份 infer npz 都写；§1.3 原 16 键不变）
ABLATE_NPZ_KEYS = ("ablate", "n_masked_keys")
# S8 motion 内容替换时另附的键（只在 motion_zero / motion_swap 时写）
MOTION_REPLACE_NPZ_KEYS = ("n_motion_replaced",)
# 带 motion token 的 VLA 每次推理的实际输入；不带 motion token 的 VLA 不写这两键
MOTION_INPUT_NPZ_KEYS = ("motion_emb", "motion_pos")
MASK_KINDS = ("mask_motion", "mask_vis")
VIS_WINDOW = (0, 32)          # mask_vis 遮的全域帧号区间 [lo, hi)
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


def parse_ablate(name) -> dict:
    """--ablate 字符串 → {"name", "masks": frozenset(MASK_KINDS 子集), "heads": ((layer, head), ...),
    "motion": None | {"kind": "zero"} | {"kind": "swap", "dir": <供体目录>}}；非法即 ValueError。"""
    import re
    text = (name or "none").strip()
    masks, heads, motion = set(), set(), None
    if text != "none":
        for part in text.split("+"):
            part = part.strip()
            if part in MASK_KINDS:
                masks.add(part)
            elif part == "motion_zero" or part.startswith("motion_swap:"):
                if motion is not None:
                    raise ValueError(f"--ablate {text!r}: motion_zero / motion_swap 只能出现一次")
                if part == "motion_zero":
                    motion = {"kind": "zero"}
                else:
                    donor_dir = part[len("motion_swap:"):].strip()
                    if not donor_dir:
                        raise ValueError(f"--ablate {text!r}: motion_swap: 后须给供体目录")
                    motion = {"kind": "swap", "dir": donor_dir}
            elif part.startswith("heads:"):
                items = [x.strip() for x in part[len("heads:"):].split(",") if x.strip()]
                if not items:
                    raise ValueError(f"--ablate {text!r}: heads: 后为空")
                for item in items:
                    m = re.fullmatch(r"L(\d+)H(\d+)", item)
                    if m is None:
                        raise ValueError(f"--ablate {text!r}: 头写法应为 L<层>H<头>，得到 {item!r}")
                    layer, head = int(m.group(1)), int(m.group(2))
                    if not (0 <= layer < NUM_LAYERS and 0 <= head < NUM_HEADS):
                        raise ValueError(f"--ablate {text!r}: L{layer}H{head} 越界（层 0..{NUM_LAYERS - 1}，头 0..{NUM_HEADS - 1}）")
                    heads.add((layer, head))
            else:
                raise ValueError(f"--ablate {text!r}: 未知项 {part!r}（可选 none / {' / '.join(MASK_KINDS)} / "
                                 f"heads:L<i>H<j>,... / motion_zero / motion_swap:<DIR>）")
    return {"name": text, "masks": frozenset(masks), "heads": tuple(sorted(heads)), "motion": motion}


def parse_episode(text) -> tuple[str, int]:
    """"task:ep" → (task, int(ep))；非法即 ValueError。"""
    task, sep, ep = str(text).strip().rpartition(":")
    if not sep or not task or not ep.strip().isdigit():
        raise ValueError(f"局写法应为 task:ep，得到 {text!r}")
    return task, int(ep)


def record_name(task: str, episode_id: int) -> str:
    """--record-motion 改名后的文件名，也是 motion_swap 供体文件名。"""
    return f"{task}_ep{int(episode_id)}.npz"


def donor_lookup(donor_starts, start_frame: int) -> int:
    """供体窗口下标：同起点优先；否则起点 ≤ start_frame 的最大窗口；再没有就第 0 个窗口。"""
    starts = np.asarray(donor_starts, np.int64).reshape(-1)
    if starts.size == 0:
        raise ValueError("供体没有任何窗口")
    exact = np.flatnonzero(starts == start_frame)
    if exact.size:
        return int(exact[0])
    lower = np.flatnonzero(starts <= start_frame)
    if lower.size:
        return int(lower[np.argmax(starts[lower])])
    return 0


def _accepts_start(fn) -> bool:
    """与 framesamp_memory._accepts_start 同判据：接受第二个位置参数才传起点帧号。"""
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(params) >= 2


class MotionContentClient:
    """替换 policy._motion_client 的包装句柄，签名 (frames, start_frame) -> (dim,) f32。

    每次先照常调用原 client（sidecar 侧行为与计时不变），再按 mode 处理返回值：none 原样、zero 置零、
    swap 换成供体同起点窗口（donor_lookup）。record_dir 给出时同时记录原编码的完整调用轨迹；
    旧字段同起点只留首次，counts 计次，以保持供体替换接口兼容。
    其余属性（provenance、close 等）透传原 client。"""

    def __init__(self, inner, *, mode: str = "none", donor: dict | None = None, record_dir=None):
        if mode not in ("none", "zero", "swap"):
            raise ValueError(f"未知 motion 替换模式 {mode!r}")
        if mode == "swap" and donor is None:
            raise ValueError("swap 模式须给供体")
        self.inner, self.mode, self.donor = inner, mode, donor
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self._inner_accepts_start = _accepts_start(inner)
        self.episode_nonce = None
        self._reset_state()

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _reset_state(self):
        self.starts, self.embs, self.counts, self.donor_starts = [], [], [], []
        self.call_starts, self.call_embs = [], []
        self.index: dict[int, int] = {}
        self.replace_log: list[list[int]] = []   # 每次替换 [start_frame, donor_start_frame]；zero 时 donor 记 -1
        self.replaced = 0
        self.dirty = False

    def __call__(self, frames, start_frame=None):
        emb = self.inner(frames, start_frame) if self._inner_accepts_start else self.inner(frames)
        emb = np.asarray(emb, dtype=np.float32)
        donor_start = -1
        if self.mode == "none":
            out = emb
        elif self.mode == "zero":
            out = np.zeros_like(emb)
        else:
            if start_frame is None:
                raise RuntimeError("motion_swap 需要起点帧号，但 motion_enc_fn 调用未传 start_frame")
            i = donor_lookup(self.donor["start_frames"], int(start_frame))
            out = np.asarray(self.donor["embs"][i], dtype=np.float32).copy()
            if out.shape != emb.shape:
                raise RuntimeError(f"供体编码形状 {out.shape} 与原编码 {emb.shape} 不符")
            donor_start = int(self.donor["start_frames"][i])
        if self.mode != "none":
            self.replaced += 1
            self.replace_log.append([int(start_frame) if start_frame is not None else -1, donor_start])
        key = int(start_frame) if start_frame is not None else len(self.starts)
        if self.record_dir is not None:
            self.call_starts.append(key)
            self.call_embs.append(emb.copy())
        if key in self.index:
            self.counts[self.index[key]] += 1
        else:
            self.index[key] = len(self.starts)
            self.starts.append(key)
            self.embs.append(emb.copy())
            self.counts.append(1)
            self.donor_starts.append(donor_start)
        self.dirty = True
        return out

    def begin_episode(self, nonce: str) -> None:
        self.flush()
        self.episode_nonce = str(nonce)
        self._reset_state()

    def flush(self) -> Path | None:
        """有新窗口时重写 record_dir/<nonce>.npz；不记录或无变化时什么也不做。"""
        if self.record_dir is None or self.episode_nonce is None or not self.dirty:
            return None
        self.record_dir.mkdir(parents=True, exist_ok=True)
        path = self.record_dir / f"{self.episode_nonce}.npz"
        tmp = self.record_dir / f".{self.episode_nonce}.{uuid.uuid4().hex}.tmp.npz"
        record = {"start_frames": np.asarray(self.starts, np.int32), "embs": np.stack(self.embs).astype(np.float32),
                  "counts": np.asarray(self.counts, np.int32), "mode": np.str_(self.mode),
                  "episode_nonce": np.str_(self.episode_nonce),
                  "call_start_frames": np.asarray(self.call_starts, np.int32),
                  "call_embs": np.stack(self.call_embs),
                  "call_seq": np.arange(len(self.call_starts), dtype=np.int64)}
        if self.mode == "swap":
            record["donor_start_frames"] = np.asarray(self.donor_starts, np.int32)
            record["donor"] = np.str_(self.donor.get("name", ""))
        np.savez(tmp, **record)
        os.replace(tmp, path)
        self.dirty = False
        return path

    def summary(self) -> dict:
        return {"mode": self.mode, "donor": (self.donor or {}).get("name"), "n_windows": len(self.starts),
                "n_replaced": self.replaced, "replace_log": [list(x) for x in self.replace_log],
                "record_file": (str(self.record_dir / f"{self.episode_nonce}.npz")
                                if self.record_dir is not None and self.episode_nonce else None)}


def load_donor(donor_dir, donor_episode) -> dict:
    """读 DIR/<task>_ep<id>.npz → {"name", "path", "start_frames", "embs"}；缺文件 / 无窗口即 SystemExit。"""
    task, ep = parse_episode(donor_episode)
    path = Path(donor_dir) / record_name(task, ep)
    if not path.is_file():
        raise SystemExit(f"错误: motion_swap 供体文件不存在: {path}")
    with np.load(path) as z:
        starts = np.asarray(z["start_frames"], np.int32)
        embs = np.asarray(z["embs"], np.float32)
    if starts.size == 0 or embs.ndim != 2 or embs.shape[0] != starts.size:
        raise SystemExit(f"错误: 供体文件 {path} 无窗口或 start_frames / embs 不对齐")
    return {"name": f"{task}:{ep}", "path": str(path), "start_frames": starts, "embs": embs}


def memory_key_mask(masks, mem_order, frame_times, *, budget: int, tokens_per_frame: int, valid=None,
                    vis_window=VIS_WINDOW) -> np.ndarray:
    """记忆区 key 遮挡 bool[MEM]（True = 遮）：mem_order[s] ≥ budget 为 motion token，< budget 为帧 token
    （帧 f = mem_order[s] // tokens_per_frame，全域帧号 frame_times[f]，−1 为 padding 帧）。valid 给出时只遮有效位。"""
    order = np.asarray(mem_order, dtype=np.int64).reshape(-1)
    times = np.asarray(frame_times, dtype=np.int64).reshape(-1)
    out = np.zeros(order.shape, bool)
    if "mask_motion" in masks:
        out |= order >= budget
    if "mask_vis" in masks:
        is_frame = order < budget
        f = np.where(is_frame, order // tokens_per_frame, 0)
        inside = is_frame & (f < times.size)
        t = np.where(inside, times[np.minimum(f, max(times.size - 1, 0))] if times.size else -1, -1)
        out |= inside & (t >= vis_window[0]) & (t < vis_window[1])
    if valid is not None:
        out &= np.asarray(valid, bool).reshape(-1)
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

    def __init__(self, policy, adapter, attn_out, *, save_top_frames: int = 3, save_prefix: bool = False,
                 ablate: str = "none", record_motion=None, swap_donor: str | None = None):
        self.policy, self.adapter = policy, adapter
        self.attn_out = Path(attn_out)
        self.save_top_frames = int(save_top_frames)
        self.save_prefix = bool(save_prefix)
        cfg = policy.config
        self.budget = int(cfg.budget)
        self.token_per_image = int(cfg.token_per_image)
        self.num_views = int(cfg.num_views)
        self.tokens_per_frame = self.token_per_image * self.num_views
        self.max_frames = self.budget // self.tokens_per_frame
        self.motion_enabled = bool(policy.motion_enabled)
        self.motion_budget = int(cfg.motion.budget) if self.motion_enabled else 0
        self.ablate = parse_ablate(ablate)
        if "mask_motion" in self.ablate["masks"] and not self.motion_enabled:
            raise SystemExit(f"错误: --ablate {self.ablate['name']} 含 mask_motion，但本模型无 motion token（基线）")
        motion = self.ablate["motion"]
        if (motion is not None or record_motion is not None) and not self.motion_enabled:
            raise SystemExit("错误: motion_zero / motion_swap / --record-motion 需要 motion 模型，本模型无 motion token（基线）")
        if (motion is not None and motion["kind"] == "swap") != (swap_donor is not None):
            raise SystemExit("错误: --ablate motion_swap:<DIR> 与 --swap-donor task:ep 必须同时给出")
        self.record_motion = Path(record_motion) if record_motion is not None else None
        self.donor = load_donor(motion["dir"], swap_donor) if swap_donor is not None else None
        self.motion_client: MotionContentClient | None = None
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
            if self.motion_enabled:
                for key in MOTION_INPUT_NPZ_KEYS:
                    self.captured[key] = getattr(observation, key)
            self.captured["mem_order"] = getattr(observation, "mem_order", None)
            self.captured["prompt_mask"] = getattr(observation, "tokenized_prompt_mask", None)
            self.captured["n_masked_keys"] = 0
            if self.ablate["masks"]:
                mask = self.key_mask(observation)
                self.captured["n_masked_keys"] = int(mask.sum())
                return inner(rng, observation, mask_keys=mask, **kwargs)
            return inner(rng, observation, **kwargs)

        self.policy._sample_actions = sample
        self.policy.reset = self.reset
        self.policy.add_buffer = self.add_buffer
        self.policy.infer = self.infer
        self._install_motion_client()
        self._hook_mem_buffer()

    def _install_motion_client(self) -> None:
        """motion 内容替换 / 记录：policy._motion_client 换成包装句柄（policy.reset 用它重建 mem_buffer），
        已建好的 mem_buffer.motion_enc 也一并换掉（首局 reset 之前那个实例）。"""
        motion = self.ablate["motion"]
        if motion is None and self.record_motion is None:
            return
        mode = "none" if motion is None else motion["kind"]
        self.motion_client = MotionContentClient(self.policy._motion_client, mode=mode, donor=self.donor,
                                                 record_dir=self.record_motion)
        self.policy._motion_client = self.motion_client
        mem_buffer = getattr(self.policy, "mem_buffer", None)
        if mem_buffer is not None and hasattr(mem_buffer, "motion_enc"):
            mem_buffer.motion_enc = self.motion_client
        import atexit
        atexit.register(self.motion_client.flush)
        print(f"ATTN_MOTION_CONTENT mode={mode} donor={(self.donor or {}).get('path')} "
              f"record_motion={self.record_motion}", flush=True)

    def _frame_times(self) -> np.ndarray:
        frames = self.policy.mem_buffer.get_frame_sampling_indices(int(self.policy.step_idx), self.budget,
                                                                  self.token_per_image)
        return _pad_times(frames, self.max_frames)

    def _memory_layout(self, observation) -> tuple[np.ndarray, np.ndarray]:
        """(mem_order, 记忆区有效位 mem_mask)，与 build_record 同法；基线无 mem_order 时恒等。"""
        static_mask = np.asarray(getattr(observation, "static_mask")).reshape(-1).astype(bool)
        if self.motion_enabled:
            mem_order = np.asarray(getattr(observation, "mem_order")).reshape(-1).astype(np.int64)
            full = np.concatenate([static_mask, np.asarray(getattr(observation, "motion_mask")).reshape(-1).astype(bool)])
        else:
            mem_order = np.arange(self.budget, dtype=np.int64)
            full = static_mask
        if mem_order.shape != full.shape:
            raise RuntimeError(f"mem_order {mem_order.shape} 与记忆区 mask {full.shape} 不符")
        return mem_order, full[mem_order]

    def key_mask(self, observation) -> np.ndarray:
        """本次推理的记忆区 key 遮挡 bool[MEM]（adapter 右补 False 到 P）。"""
        mem_order, valid = self._memory_layout(observation)
        return memory_key_mask(self.ablate["masks"], mem_order, self._frame_times(), budget=self.budget,
                               tokens_per_frame=self.tokens_per_frame, valid=valid)

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
        if self.motion_client is not None:
            self.motion_client.begin_episode(str(nonce))
        self.episode = {"episode_nonce": str(nonce), "ep_seq": self.ep_seq, "reset_rpc": rpc,
                        "server_pid": os.getpid(), "reset_unix_time": time.time(),
                        "model_kind": self.policy.metadata.get("model_kind"), "motion_enabled": self.motion_enabled,
                        "save_top_frames": self.save_top_frames, "infers": 0, "prompt": None,
                        "ablate": self.ablate["name"], "n_masked_keys": []}
        if self.motion_client is not None:
            self.episode["motion_content"] = self.motion_client.summary()
        _write_json(self.ep_dir / "episode.json", self.episode)
        print(f"ATTN_EPISODE_BEGIN nonce={nonce} ep_seq={self.ep_seq} dir={self.ep_dir}", flush=True)

    def add_buffer(self, obs: dict) -> None:
        before = int(self.policy.step_idx)
        self._orig_add(obs)
        if self.motion_client is not None:
            self.motion_client.flush()
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
        if self.save_prefix:
            self._write_prefix(record)
        self._save_frames(record["top_frames"])
        if not self.printed_env:
            print(f"ATTN_SERVE_ENV P={int(record['P'])} motion={int(self.motion_enabled)} L={NUM_LAYERS} AH={ACTION_HORIZON} "
                  f"attn_out={self.attn_out} ablate={self.ablate['name']}", flush=True)
            self.printed_env = True
        print(f"ATTN_INFER nonce={self.episode['episode_nonce']} infer_seq={self.infer_seq} step_idx={int(record['step_idx'])} "
              f"top_frames={record['top_frames'].tolist()} npz={path.name} ablate={self.ablate['name']} "
              f"n_masked_keys={int(record['n_masked_keys'])}"
              + (f" n_motion_replaced={int(record['n_motion_replaced'])}" if "n_motion_replaced" in record else ""),
              flush=True)
        self.infer_seq += 1
        self.episode.setdefault("n_masked_keys", []).append(int(record["n_masked_keys"]))
        if self.motion_client is not None:
            self.motion_client.flush()
            self.episode["motion_content"] = self.motion_client.summary()
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
            _, _, _, motion_times = self.captured["motion"]
            # 与策略组包时的拆分一致，只展开时刻，不重新编码或改动实际输入。
            if getattr(self.policy.config.motion, "readout", None) == "stage2_pool4_split_v1":
                motion_times = np.repeat(motion_times, 4)
            motion_mask = obs_motion_mask
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
        record = {
            "P": np.int64(P), "MEM": np.int64(MEM), "step_idx": np.int64(step_idx),
            "exec_start_idx": np.int64(int(self.policy.exec_start_idx)), "infer_seq": np.int64(self.infer_seq),
            "mem_order": mem_order, "mem_mask": mem_mask, "frame_times": frame_times,
            "motion_times": motion_times, "motion_mask": motion_mask, "txt_len": np.int64(txt_len),
            "actions": actions, "state": state, "top_frames": top,
            "probs_action": probs_action, "probs_text": probs_text,
            "ablate": np.str_(self.ablate["name"]), "n_masked_keys": np.int64(int(self.captured.get("n_masked_keys", 0))),
        }
        if self.motion_enabled:
            for key in MOTION_INPUT_NPZ_KEYS:
                record[key] = np.asarray(self.captured[key])[0]
        if self.ablate["motion"] is not None and self.motion_client is not None:
            record["n_motion_replaced"] = np.int64(self.motion_client.replaced)
        return record

    def _write_prefix(self, record: dict) -> None:
        """prefix_<seq>.npz：prefill 每层完整 prefix 自注意力 (18, 8, P, P) f16。"""
        P = int(record["P"])
        probs_prefix = np.asarray(self.adapter.last_attn["probs_prefix"]).astype(np.float16, copy=False)
        if probs_prefix.shape != (NUM_LAYERS, NUM_HEADS, P, P):
            raise RuntimeError(f"probs_prefix 形状 {probs_prefix.shape} 不符 {(NUM_LAYERS, NUM_HEADS, P, P)}")
        path = self.ep_dir / f"prefix_{self.infer_seq:04d}.npz"
        tmp = self.ep_dir / f".prefix_{self.infer_seq:04d}.{uuid.uuid4().hex}.tmp.npz"
        np.savez(tmp, probs_prefix=probs_prefix, P=record["P"], MEM=record["MEM"],
                 infer_seq=record["infer_seq"], step_idx=record["step_idx"])
        os.replace(tmp, path)

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
        # 另写 prefix_<seq>.npz：prefill 每层完整 prefix 自注意力 (18, 8, P, P) f16；默认关
        save_prefix: bool = False
        # 因果遮挡 / motion 内容替换：none | mask_motion | mask_vis | heads:L3H0,L16H0 | motion_zero |
        # motion_swap:<供体目录>，可用 + 组合；默认 none
        ablate: str = "none"
        # 记录每局每个 motion 窗口的原编码到 <目录>/<episode_nonce>.npz（run_local 收尾改名 <task>_ep<id>.npz）
        record_motion: Path | None = None
        # motion_swap 的供体局 task:ep（供体文件 <供体目录>/<task>_ep<id>.npz）
        swap_donor: str | None = None

    return Args


def main(args) -> None:
    from scripts.interpret.attn_adapter import AttnAdapter
    from scripts.training import serve_policy

    if args.save_top_frames < 0:
        raise SystemExit("--save-top-frames 不得为负")
    ablate_name = str(getattr(args, "ablate", "none") or "none")
    try:
        ablate = parse_ablate(ablate_name)
    except ValueError as exc:
        raise SystemExit(f"错误: {exc}") from exc
    attn_out = Path(args.attn_out).resolve()
    attn_out.mkdir(parents=True, exist_ok=True)
    original_create = serve_policy.create_policy

    def create_policy(policy_args):
        policy = original_create(policy_args)
        save_prefix = bool(getattr(args, "save_prefix", False))
        kw ={"save_prefix": True} if save_prefix else {}
        if ablate["heads"]:
            kw["ablate_heads"] = ablate["heads"]
        adapter = AttnAdapter(policy, **kw)
        record_motion = getattr(args, "record_motion", None)
        recorder = AttnRecorder(policy, adapter, attn_out, save_top_frames=args.save_top_frames,
                                save_prefix=save_prefix, ablate=ablate_name,
                                record_motion=Path(record_motion).resolve() if record_motion else None,
                                swap_donor=getattr(args, "swap_donor", None))
        recorder.install()
        P = getattr(recorder.adapter, "P", None)
        if P is not None:
            print(f"ATTN_SERVE_ENV P={int(P)} motion={int(recorder.motion_enabled)} L={NUM_LAYERS} AH={ACTION_HORIZON} "
                  f"attn_out={attn_out} ablate={ablate['name']}", flush=True)
            recorder.printed_env = True
        return policy

    serve_policy.create_policy = create_policy
    print(f"ATTN_ABLATE name={ablate['name']} masks={sorted(ablate['masks'])} heads={list(ablate['heads'])} "
          f"motion={ablate['motion']} swap_donor={getattr(args, 'swap_donor', None)} "
          f"record_motion={getattr(args, 'record_motion', None)}", flush=True)
    _install_rpc_tracking()
    # metadata、server 构造与 sidecar 收尾全部走主线 main，字段与 serve_policy.py 完全一致
    serve_policy.main(args)


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(_make_args_class()))
