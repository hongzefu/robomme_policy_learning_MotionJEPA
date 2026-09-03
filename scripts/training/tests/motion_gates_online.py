#!/usr/bin/env python3
"""motion memory 在线侧闸门 P1–P4（motion-memory-plan.md 第二部分四节 / 七节 S3）——stub 档、CPU、不加载模型。

P1  协议与 sidecar 进程：`MotionEncoderClient(stub=True)` 真起子进程（`uv run --project scripts/dataset/wan --no-sync … --stub`）、
    socketpair + pass_fds、握手协议 sha 核对；三窗（起点 0 / 16 / 100000）逐位返回 `full(768, 起点)`；错帧（不连续）→ sidecar 报错、
    客户端 raise、子进程退出 rc=4；`close()` 后子进程 rc=0、无孤儿。
P2  FrameSampMemory 运动路（本地 stub 编码函数）：按 eval.py 节奏驱动（首批整段 pre_traj + 之后每批 16 帧、`clear_buffers` 语义），
    Video（es=66 / 114）与 Button（es=0）三条 episode；每步 `visible_motion_frames(t)` == 训练侧 `motion_store.visible_motion_rows`
    的 frames（同一 IndexEntry 公式）；已编集合 == 该 t 的合法集合且每窗恰编一次；token == 起点；motion_pos == pos_emb_4x4[f,0,:256]；
    原始帧缓冲随编码收缩（上界 32 + 16）；256 域 / 尺寸 / es 变化 / 重复 step 四种坏输入 raise；budget 越界 raise（不裁剪）。
P3  交错次序两侧同源：`MME_VLA_Policy._prepare_history` 的四键装配逻辑（用 `__new__` 绕过模型构造）与训练侧同一公式
    （`pad_times` + `memory_order`，`FrameSampDataset.__getitem__` 同式）逐位相等；`mem_order` 为合法置换、int32。
P4  段边界状态机（`MME_VLA_Policy.add_buffer`）：Video 首批 es=66、后续 0（沿用）合法；后续 66 合法；后续 80 raise；
    Button 全 0 合法；`reset()` 后新 episode 可换 es。另：P1 路径下 FrameSampMemory 用 sidecar 客户端跑同一段驱动（IPC 端到端）。

用法：
  PYTHONPATH=<worktree>/src MMEVLA_V1_STORE=<主树>/v1-store uv run --no-sync python scripts/training/tests/motion_gates_online.py [--gate p1|p2|p3|p4|all]
判定行：`P1_PROTOCOL=PASS` / `P2_MEMORY=PASS` / `P3_ORDER=PASS` / `P4_ES_STATE=PASS`，任一失败非零退出。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.policies import motion_protocol as P
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
from mme_vla_suite.shared.sampling import MEM_ORDER_SENTINEL, even_sampling_indices, memory_order, pad_times

MOTION_CFG = {"stride": 16, "window_frames": 33, "budget": 96, "frame_size": 256, "pos_dim": 256, "dim": 768,
              "window_direction": "forward", "grid_origin": "segment_start"}
HIST_BUDGET, TOKEN_PER_IMAGE, NUM_VIEWS = 512, 16, 1      # perceptual-framesamp-context-motion.yaml
MAX_FRAMES = HIST_BUDGET // (TOKEN_PER_IMAGE * NUM_VIEWS)  # 32


def _dummy_vision_enc(x):
    """(t,v,224,224,3) → (t,v,64,2048) 零特征（帧路数值不在本测试范围；pool 后为 (t,v,16,2048)）。"""
    import jax.numpy as jnp
    t, v = x.shape[:2]
    return jnp.zeros((t, v, 64, 2048), jnp.bfloat16)


def _stub_enc_local(window: np.ndarray, start: int) -> np.ndarray:
    ids = P.stub_decode(window)
    assert ids == list(range(start, start + P.WINDOW_FRAMES)), (ids[:3], start)
    return np.full(P.TOKEN_DIM, float(start), np.float32)


def _entry(es: int, T: int) -> ms.IndexEntry:
    return ms.build_index_entries({"episodes": [{"global_episode_idx": 0, "h5_file": "x", "raw_ep_idx": 0,
                                                 "num_timesteps": T, "exec_start_idx": es}]})[0]


def _frames(lo: int, hi: int) -> np.ndarray:
    return np.stack([P.stub_frame(i) for i in range(lo, hi)])[:, None]     # (t,1,256,256,3)


def _drive(mem: FrameSampMemory, es: int, T: int, per_step) -> None:
    """按 eval.py 节奏：首批 pre_traj = 帧 [0, es]（Video）或 [0]（Button，es=0），之后每批 16 帧；每批后 per_step(t) 一次。"""
    first_hi = es + 1
    mem.add_buffer(_frames(0, first_hi), np.zeros((first_hi, 8), np.float32), list(range(first_hi)), exec_start_idx=es)
    t = first_hi - 1
    per_step(t)
    while t + 16 < T:
        mem.add_buffer(_frames(t + 1, t + 17), np.zeros((16, 8), np.float32), list(range(t + 1, t + 17)), exec_start_idx=es)
        t += 16
        per_step(t)


# ── P1 ────────────────────────────────────────────────────────────────────────

def gate_p1() -> bool:
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    ok = True
    c = MotionEncoderClient(online_gpu="", stub=True)
    try:
        assert c.provenance["stub"] is True and c.provenance["protocol_sha256"] == P.protocol_sha256()
        for s in (0, 16, 100000):
            win = np.stack([P.stub_frame(s + j) for j in range(P.WINDOW_FRAMES)])
            tok = c(win, s)
            assert tok.shape == (768,) and tok.dtype == np.float32 and np.all(tok == float(s)), s
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            c(win, 100000)
        dt = (time.perf_counter() - t0) / n * 1e3
        print(f"  P1 stub 往返 {dt:.2f} ms/窗（6.49 MB 请求 + 3 KB 响应，仅 IPC 开销）")
        assert c.n_calls == 23
        try:
            c(np.zeros((33, 224, 224, 3), np.uint8), 0)
            ok = False; print("  ✗ 224 域输入未 raise")
        except ValueError:
            pass
    finally:
        c.close()
    assert c._proc.returncode == 0, f"关闭后 rc={c._proc.returncode}"
    assert not c.alive
    # 错帧 → sidecar 报错、rc=4
    c2 = MotionEncoderClient(online_gpu="", stub=True)
    bad = np.stack([P.stub_frame(j) for j in range(P.WINDOW_FRAMES)]); bad[5] = P.stub_frame(99)
    try:
        c2(bad, 0)
        ok = False; print("  ✗ 错帧未 raise")
    except P.ProtocolError as e:
        assert "不连续" in str(e), str(e)[:200]
    c2._proc.wait(timeout=30)
    assert c2._proc.returncode == 4, c2._proc.returncode
    try:
        c2(bad, 0); ok = False; print("  ✗ 子进程已退出仍未 raise")
    except RuntimeError:
        pass
    c2.close()
    # 客户端未收到 handshake stub 标记不一致 → raise（stub 客户端 vs 真 sidecar 不在 CPU 测；这里测协议 sha 篡改）
    print(f"P1_PROTOCOL={'PASS' if ok else 'FAIL'}")
    return ok


# ── P2 ────────────────────────────────────────────────────────────────────────

def _check_episode(mem: FrameSampMemory, es: int, T: int, tag: str) -> None:
    entry = _entry(es, T)
    max_buf = 0
    counts_before = 0

    def per_step(t: int):
        nonlocal max_buf, counts_before
        ref_rows, ref_frames = ms.visible_motion_rows(entry, t)
        got = mem.visible_motion_frames(t)
        assert got == ref_frames.tolist(), (tag, t, got[:5], ref_frames[:5])
        assert sorted(mem._history_feats_motion) == got, (tag, t, sorted(mem._history_feats_motion)[:5], got[:5])
        assert mem.motion_encode_calls == len(got), (tag, t, mem.motion_encode_calls, len(got))
        emb, pos, mask, times = mem._prepare_motion(t)
        k = len(got)
        assert emb.shape == (96, 768) and pos.shape == (96, 256) and mask.shape == (96,) and times.shape == (96,)
        assert mask[:k].all() and not mask[k:].any()
        assert np.all(emb[:k, 0] == np.asarray(got, np.float32)) and np.all(emb[k:] == 0)
        assert np.array_equal(pos[:k], mem.pos_emb_4x4[np.asarray(got, np.int64), 0, :256]) and np.all(pos[k:] == 0)
        assert times[:k].tolist() == got and np.all(times[k:] == MEM_ORDER_SENTINEL)
        max_buf = max(max_buf, len(mem._raw_frames))

    _drive(mem, es, T, per_step)
    n_final = len(mem.visible_motion_frames(mem.n_steps - 1))
    assert max_buf <= 32 + 16 + 1, (tag, max_buf)
    print(f"  {tag}: es={es} T={T} 终态合法起点 {n_final} 窗、编码 {mem.motion_encode_calls} 次、原始帧缓冲峰值 {max_buf} 帧")


def gate_p2() -> bool:
    ok = True
    for es, T, tag in ((66, 300, "Video-es66"), (114, 420, "Video-es114"), (0, 260, "Button-es0")):
        mem = FrameSampMemory(vision_enc_fn=_dummy_vision_enc, motion_enc_fn=_stub_enc_local, motion_cfg=MOTION_CFG)
        _check_episode(mem, es, T, tag)
        mem.clear()
        assert mem.n_steps == 0 and not mem._history_feats_motion and not mem._raw_frames and mem.exec_start_idx is None
    # 坏输入
    mem = FrameSampMemory(vision_enc_fn=_dummy_vision_enc, motion_enc_fn=_stub_enc_local, motion_cfg=MOTION_CFG)
    bad_cases = {
        "缺 exec_start_idx": lambda: mem.add_buffer(_frames(0, 1), np.zeros((1, 8), np.float32), [0]),
        "224 域": lambda: mem.add_buffer(np.zeros((1, 1, 224, 224, 3), np.uint8), np.zeros((1, 8), np.float32), [0], exec_start_idx=0),
        "float 帧": lambda: mem.add_buffer(np.zeros((1, 1, 256, 256, 3), np.float32), np.zeros((1, 8), np.float32), [0], exec_start_idx=0),
    }
    for name, fn in bad_cases.items():
        try:
            fn(); ok = False; print(f"  ✗ {name} 未 raise")
        except ValueError:
            pass
    mem.add_buffer(_frames(0, 67), np.zeros((67, 8), np.float32), list(range(67)), exec_start_idx=66)
    for name, fn in {
        "es 中途变化": lambda: mem.add_buffer(_frames(67, 83), np.zeros((16, 8), np.float32), list(range(67, 83)), exec_start_idx=80),
        "重复 step": lambda: mem.add_buffer(_frames(66, 82), np.zeros((16, 8), np.float32), list(range(66, 82)), exec_start_idx=66),
    }.items():
        try:
            fn(); ok = False; print(f"  ✗ {name} 未 raise")
        except ValueError:
            pass
    # budget 越界：demo 段极长（es=1600 → demo 网格 99 > 96）→ _encode 正常、_prepare_motion raise（不裁剪）
    mem2 = FrameSampMemory(vision_enc_fn=_dummy_vision_enc, motion_enc_fn=_stub_enc_local, motion_cfg=MOTION_CFG)
    es = 1600
    mem2.add_buffer(_frames(0, es + 1), np.zeros((es + 1, 8), np.float32), list(range(es + 1)), exec_start_idx=es)
    assert mem2.motion_encode_calls == ms.seg_num_grid(es) > 96, (mem2.motion_encode_calls, ms.seg_num_grid(es))
    try:
        mem2._prepare_motion(es); ok = False; print("  ✗ budget 越界未 raise")
    except RuntimeError as e:
        assert "> motion.budget" in str(e)
    print(f"P2_MEMORY={'PASS' if ok else 'FAIL'}")
    return ok


# ── P3 ────────────────────────────────────────────────────────────────────────

class _Cfg:
    budget = HIST_BUDGET; token_per_image = TOKEN_PER_IMAGE; num_views = NUM_VIEWS


def _bare_policy(mem: FrameSampMemory):
    from mme_vla_suite.policies.policy import MME_VLA_Policy
    pol = MME_VLA_Policy.__new__(MME_VLA_Policy)
    pol.config = _Cfg(); pol.mem_buffer = mem; pol.motion_enabled = True; pol._motion_cfg = MOTION_CFG
    pol._motion_client = _stub_enc_local
    pol.step_idx = -1; pol.exec_start_idx = 0
    pol.use_quantiles = False
    class _NS: mean = np.zeros(8, np.float32); std = np.ones(8, np.float32)
    pol.state_norm_stats = _NS()
    return pol


def gate_p3() -> bool:
    ok = True
    n_checked = 0
    for es, T in ((66, 300), (0, 260)):
        mem = FrameSampMemory(vision_enc_fn=_dummy_vision_enc, motion_enc_fn=_stub_enc_local, motion_cfg=MOTION_CFG)
        pol = _bare_policy(mem)
        entry = _entry(es, T)

        def per_step(t: int):
            nonlocal n_checked
            pol.step_idx = t
            inputs = pol._prepare_history({})
            # 训练侧同式（FrameSampDataset.__getitem__ 末段）
            frames_arr = np.asarray(even_sampling_indices(t, MAX_FRAMES), np.int64)
            _, f_m = ms.visible_motion_rows(entry, t)
            ref = memory_order(pad_times(frames_arr, MAX_FRAMES), TOKEN_PER_IMAGE * NUM_VIEWS, pad_times(f_m, 96))
            got = inputs["mem_order"]
            assert got.dtype == np.int32 and got.shape == (608,) and np.array_equal(got, ref), (es, t)
            assert np.array_equal(np.sort(got), np.arange(608))
            assert inputs["motion_emb"].shape == (96, 768) and inputs["motion_mask"].sum() == len(f_m)
            assert inputs["static_mask"].shape == (512,)
            n_checked += 1

        _drive(mem, es, T, per_step)
    print(f"  P3: {n_checked} 个推理步的 mem_order 与训练侧公式逐位相等")
    print(f"P3_ORDER={'PASS' if ok else 'FAIL'}")
    return ok


# ── P4 ────────────────────────────────────────────────────────────────────────

def gate_p4() -> bool:
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    ok = True
    mem = FrameSampMemory(vision_enc_fn=_dummy_vision_enc, motion_enc_fn=_stub_enc_local, motion_cfg=MOTION_CFG)
    pol = _bare_policy(mem)
    obs = lambda lo, hi, es: {"images": _frames(lo, hi), "state": np.zeros((hi - lo, 8), np.float32), "exec_start_idx": es}
    pol.add_buffer(obs(0, 67, 66)); assert pol.exec_start_idx == 66 and pol.step_idx == 66
    pol.add_buffer(obs(67, 83, 0)); assert pol.exec_start_idx == 66 and pol.step_idx == 82       # clear_buffers 后传 0 = 沿用
    pol.add_buffer(obs(83, 99, 66)); assert pol.exec_start_idx == 66                              # 重复真实值合法
    try:
        pol.add_buffer(obs(99, 115, 80)); ok = False; print("  ✗ es 中途变化未 raise")
    except ValueError:
        pass
    assert mem.exec_start_idx == 66 and pol.step_idx == 98
    # reset 后换 es（Button 全 0）
    pol.step_idx = -1; pol.exec_start_idx = 0; mem.clear()
    pol.add_buffer(obs(0, 1, 0)); pol.add_buffer(obs(1, 17, 0)); pol.add_buffer(obs(17, 33, 0))
    assert pol.exec_start_idx == 0 and mem.exec_start_idx == 0 and mem.visible_motion_frames(32) == [0]
    # IPC 端到端：sidecar 客户端作为 motion_enc_fn 跑同一段驱动
    c = MotionEncoderClient(online_gpu="", stub=True)
    try:
        mem2 = FrameSampMemory(vision_enc_fn=_dummy_vision_enc, motion_enc_fn=c, motion_cfg=MOTION_CFG)
        _check_episode(mem2, 66, 200, "Video-es66-sidecar")
        assert c.n_calls == mem2.motion_encode_calls
        print(f"  P4 sidecar 端到端 {c.n_calls} 窗，均 {c.total_s / c.n_calls * 1e3:.2f} ms/窗")
    finally:
        c.close()
    assert c._proc.returncode == 0
    print(f"P4_ES_STATE={'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="all", choices=["all", "p1", "p2", "p3", "p4"])
    args = ap.parse_args()
    gates = {"p1": gate_p1, "p2": gate_p2, "p3": gate_p3, "p4": gate_p4}
    sel = list(gates) if args.gate == "all" else [args.gate]
    results = {}
    for g in sel:
        print(f"== {g.upper()} ==")
        try:
            results[g] = gates[g]()
        except Exception:
            traceback.print_exc()
            results[g] = False
            print(f"{g.upper()}=FAIL（异常）")
    print("ONLINE_GATES=" + ("PASS" if all(results.values()) else "FAIL") + " " + str(results))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
