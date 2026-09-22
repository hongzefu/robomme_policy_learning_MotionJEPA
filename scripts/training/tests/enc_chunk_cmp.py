"""MMEVLA_ENC_CHUNK 的真实编码对照（0922-binfill-demo-prefix-plan.md 验证节）。

**这一条是测量，不是闸门。** 第一次跑出来的数就是基线，写进留档；只有当 ``rel_max`` 超过
既有 f32/bf16 编码器差异的量级（``docs/train-infer-consistency.md`` 七节记的 0.33%）时才停下来，
交用户决定是否改用其它规避 OOM 的办法。

**为什么必须实测**：BinFill 补 demo 后首批 ``add_buffer`` 最多送 1282 帧进 SigLIP
（现有最长案例 VideoRepick 才 502 帧），``image_jnp`` 一次性 ``(1282,1,224,224,3) float32``
约 771 MiB、bf16 输出约 1.3 GiB，OOM 风险未测。分批能压住峰值，代价是切 batch 维可能让 XLA
为不同 batch 尺寸选不同 kernel，从而改变 bf16 累加序——不保证逐位相同。

**被切的不止 demo 段**：首批是 ``D`` 帧 demo **加 1 帧干净 env 的 reset 观测**，后者是 exec 段
第 0 帧、有历史口径，同样落进新的 batch。所以覆盖点必须包含 reset 帧与每个分批边界帧。

走的是生产路径本身（``FrameSampMemory.add_buffer``），不是这里重抄一遍编码链——重抄会漂移。

用法（server 环境，需 GPU）：
    uv run scripts/training/tests/enc_chunk_cmp.py \
        --ckpt <绝对路径>/50000 --config mme_vla_suite \
        --prefix-h5 <store>/BinFill/easy/ep155.h5 --chunk 256
判定行：
    ENC_CHUNK_CMP rel_max=<..> cos_min=<..> reset_frame_rel=<..> boundary_rel_max=<..> ...
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/robomme"))

#: 既有「训练 f32 逐帧编 vs 推理 bf16 整批编」的差异量级（docs/train-infer-consistency.md 第七节 1，
#: 2409 帧记忆 token 实测）。这是参照系不是阈值：超过它就停下来交用户裁决，不自行放行。
#: 该节原话：「差异只来自精度和批形状，两者各贡献约 0.3%」——批形状本就是既有差异来源之一，
#: 本脚本测的分批属同一类，理应落在这个量级之内。
REFERENCE_REL = 0.0033      # rel_fro
REFERENCE_COS = 0.99998     # 余弦最小值


def load_frames(args) -> tuple[np.ndarray, np.ndarray, int]:
    """返回 (images (t,1,256,256,3) uint8, states (t,8) f32, D)。

    帧序与在线首批完全一致：``D`` 帧 demo 前缀 + 1 帧干净 env 的 reset 观测。
    ``--reset-frame-npy`` 不给时用 demo 第 0 帧顶替那一位——两者是同一场景初始态的两次渲染，
    像素上应当极接近，对「分批是否改变数值」这个问题没有影响（它只关心 batch 边界在哪）。
    """
    from demo_prefix import DemoPrefixStore, frames_sha256   # noqa: F401  (确保模块可导入)
    import cv2
    import h5py

    with h5py.File(args.prefix_h5, "r") as h5:
        d = int(h5.attrs["D"])
        front = [cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
                 for buf in h5["front_png"]]
        states = np.asarray(h5["state"][...], dtype=np.float32)
    front = [np.ascontiguousarray(f, dtype=np.uint8) for f in front]
    if len(front) != d:
        raise SystemExit(f"h5 里 front 帧数 {len(front)} != D={d}")

    if args.reset_frame_npy:
        reset_frame = np.ascontiguousarray(np.load(args.reset_frame_npy), dtype=np.uint8)
        if reset_frame.shape != (256, 256, 3):
            raise SystemExit(f"reset 帧形状 {reset_frame.shape} != (256,256,3)")
    else:
        reset_frame = front[0].copy()
    frames = front + [reset_frame]
    images = np.stack(frames)[:, None]                       # (t,1,256,256,3)
    state_all = np.concatenate([states, states[-1:]], axis=0)
    return np.ascontiguousarray(images), np.ascontiguousarray(state_all), d


def encode_with_chunk(policy, images, states, chunk: int) -> np.ndarray:
    """用生产路径 FrameSampMemory.add_buffer 编一遍，返回 (t, v, token_per_image, dim) float32。"""
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory

    config = policy.config
    os.environ["MMEVLA_ENC_CHUNK"] = str(chunk)
    mem = FrameSampMemory(
        num_views=config.num_views,
        img_emb_dim=config.memory_feature.img.input_dim,
        pos_emb_dim=config.memory_feature.pos.input_dim,
        state_emb_dim=config.memory_feature.state.input_dim,
        vision_enc_fn=policy._vision_encode,
        token_per_image=int(config.token_per_image),
    )   # 不开 motion：chunk 只影响帧路 SigLIP 编码，运动路另有自己的编码器
    t0 = time.perf_counter()
    mem.add_buffer(images, states, list(range(images.shape[0])))
    seconds = time.perf_counter() - t0
    key = mem.image_key
    out = np.stack([np.asarray(mem._history_feats[i][key], dtype=np.float32)
                    for i in range(images.shape[0])])
    print(f"[enc] chunk={chunk} frames={images.shape[0]} shape={out.shape} "
          f"seconds={seconds:.1f}", flush=True)
    del mem
    return out


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """逐帧比相对 Frobenius 范数差与余弦——与 ``docs/train-infer-consistency.md`` 第七节 1 同口径。

    那一节记的基线是「2409 帧记忆 token **相对差 0.33%、余弦最小 0.99998**」，
    指的就是 rel_fro 与 cos，所以这里必须用同一对指标，否则没法比。

    **不要用逐元素相对差**：``2|a-b|/(|a|+|b|+eps)`` 的理论上界是 2，只要某个元素符号相反或
    一方为 0 就取到 2；bf16 特征里大量元素接近 0，那个指标在那里没有信息量。
    本函数仍然把它算出来记进报告，但只作参考、不进判据。
    """
    eps = np.float64(1e-12)
    rel_fro, cos, elem_max = [], [], []
    for i in range(a.shape[0]):
        x = a[i].reshape(-1).astype(np.float64)
        y = b[i].reshape(-1).astype(np.float64)
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        rel_fro.append(float(np.linalg.norm(x - y) / (nx + eps)))
        cos.append(1.0 if nx == 0 and ny == 0 else float(np.dot(x, y) / (nx * ny + eps)))
        elem_max.append(float(np.max(2.0 * np.abs(x - y) / (np.abs(x) + np.abs(y) + eps))))
    # 全局 rel_fro：把所有帧当成一个大矩阵，与文档「2409 帧」那个数同口径
    fa, fb = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    global_rel_fro = float(np.linalg.norm(fa - fb) / (np.linalg.norm(fa) + eps))
    return {"rel_fro": rel_fro, "cos": cos, "elem_max": elem_max,
            "global_rel_fro": global_rel_fro}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--prefix-h5", required=True)
    parser.add_argument("--reset-frame-npy", default="")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.chunk < 1:
        raise SystemExit("--chunk 必须为正整数（0 = 关闭，没有对照的意义）")

    import jax
    from unittest import mock
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.policies import motion_client as mc

    if jax.default_backend() != "gpu":
        raise SystemExit(f"错误: jax 后端 {jax.default_backend()} != gpu，分批的 kernel 选择只在 GPU 上有意义")

    images, states, d = load_frames(args)
    t = images.shape[0]
    print(f"[enc] prefix={args.prefix_h5} D={d} total_frames={t} chunk={args.chunk}", flush=True)

    original = mc.MotionEncoderClient
    with mock.patch.object(mc, "MotionEncoderClient",
                           lambda **kw: original(**(kw | {"online_gpu": ""}))):
        policy = _policy_config.create_trained_policy(
            _config.get_config(args.config), pathlib.Path(args.ckpt), motion_stub=True)
    if getattr(policy, "_motion_client", None) is not None:
        try:
            policy._motion_client.close()
        except Exception:
            pass

    full = encode_with_chunk(policy, images, states, 0)
    chunked = encode_with_chunk(policy, images, states, args.chunk)
    stats = compare(full, chunked)
    rel, cos = stats["rel_fro"], stats["cos"]

    # 覆盖点：reset 帧（最后一帧，exec 段第 0 帧）与每个分批边界帧及其前一帧
    boundaries = sorted({b for b0 in range(args.chunk, t, args.chunk) for b in (b0 - 1, b0)})
    boundary_rel = {int(i): rel[i] for i in boundaries}
    summary = {
        "prefix_h5": str(args.prefix_h5), "D": d, "frames": t, "chunk": args.chunk,
        "ckpt": str(args.ckpt), "config": args.config,
        "rel_fro_max": max(rel), "rel_fro_mean": float(np.mean(rel)),
        "global_rel_fro": stats["global_rel_fro"],
        "cos_min": min(cos), "cos_mean": float(np.mean(cos)),
        "reset_frame_index": t - 1, "reset_frame_rel_fro": rel[-1], "reset_frame_cos": cos[-1],
        "boundary_indices": boundaries,
        "boundary_rel_fro_max": max(boundary_rel.values()) if boundary_rel else 0.0,
        "boundary_rel_fro": boundary_rel,
        "bitexact": bool(max(rel) == 0.0),
        # 逐元素相对差只作参考：理论上界是 2，bf16 里大量接近 0 的元素会让它恒等于 2，没有信息量
        "elem_rel_max_reference_only": max(stats["elem_max"]),
        "reference_rel_fro": REFERENCE_REL, "reference_cos_min": REFERENCE_COS,
    }
    print("ENC_CHUNK_CMP " + json.dumps(
        {k: summary[k] for k in ("rel_fro_max", "global_rel_fro", "cos_min",
                                 "reset_frame_rel_fro", "boundary_rel_fro_max",
                                 "bitexact", "D", "frames", "chunk")}, ensure_ascii=False), flush=True)
    print(f"ENC_CHUNK_BASELINE 既有「训练 f32 逐帧编 vs 推理 bf16 整批编」差异："
          f"rel_fro={REFERENCE_REL} cos_min={REFERENCE_COS}"
          f"（docs/train-infer-consistency.md 第七节 1，2409 帧实测；"
          f"该节明确「差异只来自精度和批形状，两者各贡献约 0.3%」——批形状本就是既有来源之一）",
          flush=True)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    if summary["rel_fro_max"] > REFERENCE_REL or summary["cos_min"] < REFERENCE_COS:
        print(f"[warn] rel_fro_max={summary['rel_fro_max']:.6g} / cos_min={summary['cos_min']:.8f} "
              f"超过既有编码器差异量级（{REFERENCE_REL} / {REFERENCE_COS}），"
              f"按计划停下来交用户裁决（不要自行放行）", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
