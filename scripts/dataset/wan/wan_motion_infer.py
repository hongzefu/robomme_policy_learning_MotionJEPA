#!/usr/bin/env python
"""口径 A 的最小自包含前向：Wan VAE 帧→latent（VAE 段）+ WanLatentMotionEncoder latent→motion token（encoder 段）。

这是 `scripts/evaluateLabelData-v4/swapclip_common.py` 里 `load_vae / encode_chunk / load_encoder / motion_token`
四个函数的逐字复刻，去掉了 benchmark 数据源 pin 校验与 dataset 依赖，只留数值路径；供 policy 仓库
（robomme_policy_learning_MotionJEPA）的 `extract_wan.py` / `encode_motion.py` 照抄，目标是**同硬件逐位一致**
（跨硬件允许不一致——A40 vs Ada 差 ~1e-5 级，隔离在 VAE `conv_out`，见 docs/dataset-build-doc/slurm-wan-extract-v1）。

两类标记：
  🔒 数值语句——改一个字就会改位，照抄。
  🛡 环境保险——本轮新增，在本机全是空操作（原 A/B 就是在这些默认值下实测的），但换到别的项目 / 进程里能
     挡住五类静默漂移：调用方外层 autocast、输入内存布局、全局数值开关、环境变量、软件版本。

实测依据（2026-09-02，RTX 6000 Ada，torch 2.9.0+cu128 / cudnn 91002）：
  * encoder 段：本口径 vs `scripts/evaluate/common/runtime.py::Runtime.encode` 在 B=1 下 24/24 逐位相同；
    bf16 autocast 下 TF32 两开关与 seed 都不改位；batch 1 vs 8 改最后一位（cuBLAS 按 GEMM 形状换 kernel，
    首处差异在 `input_proj`）——所以 **B=1 是硬约束**。
  * VAE 段：本口径 vs 抽取器 `scripts/dataset-build/extract_wan_chunk_latents_all.py::encode_window` 8/8 逐位；
    `.mode()` 不消耗随机数；输入张量 stride 不同 cudnn 会换 kernel（差 2e-6）——所以入口统一成连续布局。
  * checkpoint 主键 `encoder` 是 EMA 权重，`latents_mean/std` 与 RoPE cache 这类常数 buffer 也被 EMA 凸组合过
    （与 VAE config 真值差 ~1e-5）——所以**必须整份 strict 加载**，禁止重建模型后再调 `load_wan_latent_stats`。

用法（CLI，两条入口二选一）：
    uv run python scripts/inference-example/wan_motion_infer.py \
        --run wan-v8-filter10-72ep-a --checkpoint checkpoint_epoch_72.pt \
        --video_h5 /data/hongzefu/motionjepa-v7/data-raw/ButtonUnmask_ep0/video_exec.h5 --anchor 61
    uv run python scripts/inference-example/wan_motion_infer.py \
        --run wan-v8-filter10-72ep-a --checkpoint checkpoint_epoch_72.pt \
        --latent_bin /data/hongzefu/dataset-4env-v8/dataset-token/wan_chunk_latents/ButtonUnmask_ep0_exec.bin --chunk 61

作为模块使用时，调用方必须在起手显式调 `check_env()` + `pin_numerics()`（本模块 import 零副作用，不调就不保证）。
对拍守卫见同目录 `crosscheck.py`。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import warnings

import numpy as np
import torch
import yaml

# ---------------- 常量（照抄 swapclip_common / 抽取器，一字不动） ----------------

ARCH_TAG = "wan-latent-v7"
K = 8                                   # 未来组数（= 训练 data.max_horizon）
WINDOW = 4 * K + 1                      # 33 帧窗口 [t, t+32]
FRAME_SIZE = 256
LAT_G, LAT_C, LAT_H, LAT_W = 1 + K, 16, 32, 32
CHUNK_BYTES = LAT_G * LAT_C * LAT_H * LAT_W * 4        # 589,824：latent bin 里第 i 个 chunk 的偏移 = i × CHUNK_BYTES

# ---------------- 🛡 版本与环境钉死 ----------------

# 与 MotionJEPA uv.lock 一致；policy 仓库的子 venv 必须钉同一组（计划 D7）
PINNED = {"torch": "2.9.0+cu128", "cudnn": 91002, "diffusers": "0.39.0"}
# 这些环境变量会绕过代码里的开关：NVIDIA_TF32_OVERRIDE 强开 TF32；CUBLAS_WORKSPACE_CONFIG 改 cuBLAS 算法选择
FORBIDDEN_ENV = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")
# Wan2.1 VAE 权重 state_dict 指纹（sorted keys + tensor bytes，算法同 gl_probe_wan_consistency.vae_state_fingerprint；
# 记录于 docs/dataset-build-doc/slurm-wan-extract-v1/probe/V0_local_selfcheck.md）
VAE_STATE_SHA256_EXPECTED = "9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0"

# pin_numerics 之后各开关应有的读回值（crosscheck 逐项核）
EXPECTED_FLAGS = {
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": False,
    "deterministic_algorithms": False,
    "float32_matmul_precision": "highest",
    "bf16_reduced_precision_reduction": True,
    "fp16_reduced_precision_reduction": True,
    "sdp_flash": True,
    "sdp_mem_efficient": True,
    "sdp_math": True,
    "sdp_cudnn": True,
}


def numerics_snapshot():
    """当前全局数值开关的读回值（provenance 与 crosscheck 共用）。"""
    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "bf16_reduced_precision_reduction":
            bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "fp16_reduced_precision_reduction":
            bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "sdp_flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdp_mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdp_math": bool(torch.backends.cuda.math_sdp_enabled()),
        "sdp_cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
    }


def pin_numerics():
    """🛡 把 A/B 实测时的全部数值开关显式钉成同值（原 `disable_tf32()` 的超集）。

    前三行是原 A 的 `disable_tf32()`（🔒）；其余是 PyTorch 默认值，显式写出是为了防调用方进程里
    有人改过。**不强制 SDPA 后端**——强制某一后端可能与 A/B 的默认启发式分叉，这里只断言四个后端
    都还开着（默认态）。返回读回快照。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # torch 2.9 对旧 TF32 API 打 deprecation warning，不影响生效
        torch.set_float32_matmul_precision("highest")                              # 🛡 默认值
        torch.backends.cuda.matmul.allow_tf32 = False                              # 🔒
        torch.backends.cudnn.allow_tf32 = False                                    # 🔒 VAE 是 conv3d 网络，实质项
        torch.backends.cudnn.benchmark = False                                     # 🔒 True 会换 conv 算法
        torch.backends.cudnn.deterministic = False                                 # 🛡 默认值；True 会换算法
        torch.use_deterministic_algorithms(False)                                  # 🛡 默认值
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True   # 🛡 默认值；管 encoder 段 bf16 GEMM 累加
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True   # 🛡 默认值
    snap = numerics_snapshot()
    sdp = {k: v for k, v in snap.items() if k.startswith("sdp_")}
    assert all(sdp.values()), \
        f"SDPA 后端被关闭 {sdp}——A/B 在四后端全开的默认启发式下实测，请勿 disable 任何后端"
    assert snap == EXPECTED_FLAGS, f"数值开关读回值与预期不符：{snap}"
    return snap


def check_env(strict=True):
    """🛡 断言没有会绕过代码开关的环境变量。返回快照 {name: value|None}。"""
    snap = {k: os.environ.get(k) for k in FORBIDDEN_ENV}
    bad = [k for k, v in snap.items() if v]
    if bad and strict:
        raise SystemExit(f"环境变量 {bad} 已设置（{ {k: snap[k] for k in bad} }），会改变数值路径；"
                         f"请 unset 后再跑（或 strict=False 只记录不拦）")
    return snap


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def check_versions(strict=True, with_diffusers=False):
    """🛡 硬断言 torch / cudnn（/ diffusers）与 PINNED 一致。strict=False 只打印不拦。"""
    actual = {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version()}
    if with_diffusers:
        actual["diffusers"] = _pkg_version("diffusers")
    mismatch = {k: (actual[k], PINNED[k]) for k in actual if actual[k] != PINNED[k]}
    if mismatch:
        msg = f"版本与 PINNED 不符（实际, 期望）：{mismatch}——跨项目逐位一致的前提不成立"
        if strict:
            raise SystemExit(msg)
        print("[warn] " + msg, flush=True)
    return actual


def sha256_file(path, chunk_bytes=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def vae_state_sha256(vae):
    """VAE 全量权重指纹：按 key 排序逐 tensor 哈希（逐字照抄 gl_probe_wan_consistency.vae_state_fingerprint）。"""
    h = hashlib.sha256()
    sd = vae.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def provenance(device=None):
    """🛡 身份快照：软件版本、GPU、driver、本模块与 encoder 源码的 sha256、数值开关、环境变量。

    policy 侧把它原样写进 `store_meta.provenance`；跨项目对拍先比这张表再比数字。
    """
    import motion_jepa.models.wan_latent_encoder as _enc_src
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_pkg": _pkg_version("nvidia-cublas-cu12"),
        "cudnn_pkg": _pkg_version("nvidia-cudnn-cu12"),
        "diffusers": _pkg_version("diffusers"),
        "hostname": os.uname().nodename,
        "module_sha256": sha256_file(os.path.abspath(__file__)),
        "encoder_src_sha256": sha256_file(os.path.abspath(_enc_src.__file__)),
        "flags": numerics_snapshot(),
        "env": {k: os.environ.get(k) for k in FORBIDDEN_ENV},
    }
    if device is not None and torch.device(device).type == "cuda":
        props = torch.cuda.get_device_properties(torch.device(device))
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            driver = "unknown"
        info.update({"gpu_name": props.name, "compute_cap": f"{props.major}.{props.minor}",
                     "sm_count": props.multi_processor_count, "driver": driver})
    return info


# ---------------- VAE 段：33 帧 uint8 → 组优先 latent (9,16,32,32) fp32 ----------------

def load_vae(vae_id, device, expected_state_sha256=None, strict_versions=True):
    """加载 Wan2.1 VAE（fp32、eval、冻结、tiling/slicing 关）。返回 (vae, info)。"""
    check_versions(strict=strict_versions, with_diffusers=True)
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(vae_id, subfolder="vae",
                                           torch_dtype=torch.float32)          # 🔒
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    # tiling/slicing 会改变前向的空间拼接与 batch 组成，破坏对拍前提
    assert not vae.use_tiling and not vae.use_slicing, "vae.use_tiling / use_slicing 必须保持关闭"   # 🔒
    sha = vae_state_sha256(vae)
    if expected_state_sha256 is not None:
        assert sha == expected_state_sha256, \
            f"VAE 权重指纹 {sha[:16]}… ≠ 期望 {expected_state_sha256[:16]}…（HF 缓存里不是同一份权重）"
    info = {"vae_id": vae_id, "vae_state_sha256": sha, "vae_dtype": "float32",
            "latent_mode": "mode", "batch": 1, "tf32": "off", "amp": "off",
            **provenance(device)}
    return vae, info


@torch.no_grad()
def encode_chunk(vae, frames_window, device):
    """(33,256,256,3) uint8 → 组优先 latent (9,16,32,32) fp32（B=1、fp32、关 TF32），留在 device。

    调用前须已 `pin_numerics()`。本函数内部屏蔽调用方的外层 autocast（VAE 恒 fp32）。
    """
    frames_window = np.ascontiguousarray(frames_window)      # 🛡 统一内存布局：stride 不同 cudnn 会换 kernel（实测差 2e-6）
    assert frames_window.shape == (WINDOW, FRAME_SIZE, FRAME_SIZE, 3), \
        f"期望 (33,256,256,3)，得到 {frames_window.shape}"
    assert frames_window.dtype == np.uint8, f"期望 uint8，得到 {frames_window.dtype}"
    device = torch.device(device)
    with torch.autocast(device_type=device.type, enabled=False):                     # 🛡 屏蔽外层 autocast
        f = torch.from_numpy(frames_window).permute(0, 3, 1, 2).float().to(device)   # 🔒 (33,3,256,256) 0..255
        x = (f / 127.5 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)                        # 🔒 (1,3,33,256,256) [-1,1]
        z = vae.encode(x).latent_dist.mode()                                          # 🔒 mode（均值），不是 sample
    # 形状断言同时挡住 diffusers 对 T≠4k+1 的静默截断
    assert z.shape == (1, LAT_C, LAT_G, LAT_H, LAT_W), f"VAE 返回形状异常 {tuple(z.shape)}"
    lat = z.permute(0, 2, 1, 3, 4).contiguous()[0].float()                            # 🔒 通道优先 → 组优先
    assert torch.isfinite(lat).all()
    return lat


# ---------------- encoder 段：latent (9,16,32,32) → motion token (768,) ----------------

def load_encoder(run_dir, checkpoint_name, device, expected_sha256=None, strict_versions=True):
    """按 run 冻结 config 重建 WanLatentMotionEncoder 并整份 strict 加载主键 `encoder`（EMA）。

    返回 (encoder, info, use_amp)。affine 常数构造时传 None（NaN 占位），由 strict load 从 checkpoint 的
    persistent buffer 带入，**不读 VAE config**（EMA 已把 buffer 凸组合过，手填真值反而不逐位）。
    """
    check_versions(strict=strict_versions, with_diffusers=False)
    from motion_jepa.models import WanLatentMotionEncoder
    device = torch.device(device)

    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("wan"), f"{run_dir} 冻结 config 无 wan: 节——非 v7 run"
    ckpt_path = os.path.join(run_dir, checkpoint_name)
    ckpt_sha = sha256_file(ckpt_path)
    if expected_sha256 is not None:
        assert ckpt_sha == expected_sha256, \
            f"checkpoint sha256 {ckpt_sha[:16]}… ≠ 期望 {expected_sha256[:16]}…"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert ckpt.get("arch") == ARCH_TAG, f"{ckpt_path} arch={ckpt.get('arch')!r} ≠ {ARCH_TAG!r}"

    # 原 A 硬用常量 K 建 seq_len；这里多一句核对（loss_common.load_models 同款），数值无关
    K_cfg = int(cfg["data"].get("max_horizon", 8))
    assert K_cfg == K, f"run 的 data.max_horizon={K_cfg} ≠ 本模块窗口口径 K={K}"

    M = int((cfg.get("motion") or {}).get("num_tokens", 1))
    D = int(cfg["motion"]["dim"])
    enc_cfg = cfg["motion"].get("encoder") or {}
    encoder = WanLatentMotionEncoder(
        latent_channels=int(cfg["wan"]["latent_channels"]),
        latent_size=int(cfg["wan"]["latent_size"]),
        hidden_dim=int(enc_cfg.get("hidden_dim", 768)), motion_dim=D, num_tokens=M,
        seq_len=K + 1, depth=int(enc_cfg.get("depth", 8)),
        heads=int(enc_cfg.get("heads", 12)), dim_head=int(enc_cfg.get("dim_head", 64)),
        mlp_ratio=int(enc_cfg.get("mlp_ratio", 4)), dropout=0.0,
    ).to(device)

    state = {}
    for key, value in ckpt["encoder"].items():                 # 🔒 主键 encoder（EMA），不读 encoder_live
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        state[key] = value
    encoder.load_state_dict(state, strict=True)                # 🔒 整份 strict，含 EMA 过的 affine buffer 与 RoPE cache
    assert torch.isfinite(encoder.latents_std).all(), \
        "encoder affine buffer 仍是 NaN 占位（checkpoint 缺归一化常数）"
    encoder.eval()
    use_amp = cfg["training"].get("precision") == "bf16" and device.type == "cuda"   # 🔒 同原 A / B / train.py
    info = {
        "run_dir": os.path.abspath(run_dir),
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "arch": ARCH_TAG,
        "state_key": "encoder",
        "precision": cfg["training"].get("precision"),
        "amp": "bf16" if use_amp else "off",
        "tf32": "off",
        "batch": 1,
        "vae_id": cfg["wan"]["vae_id"],
        "motion_dims": [M, D],
        **provenance(device),
    }
    return encoder, info, use_amp


@torch.no_grad()
def motion_token(encoder, latent_block, use_amp, device):
    """(9,16,32,32) 未归一化 latent → motion token (M*D,)=(768,) fp32 numpy。B=1 硬约束。

    与 runtime.encode ≡ train.py 的 encoder(cat([cur, fut])) 同算法；bf16 run 挂 autocast（训练态读数）。
    调用前须已 `pin_numerics()`（实测 bf16 autocast 下 TF32 与 seed 都不改位，钉住只为卫生）。
    """
    device = torch.device(device)
    assert latent_block.shape == (LAT_G, LAT_C, LAT_H, LAT_W), f"期望 (9,16,32,32)，得到 {tuple(latent_block.shape)}"
    assert latent_block.dtype == torch.float32, f"期望 float32，得到 {latent_block.dtype}"
    latent_block = latent_block.contiguous()                   # 🛡 统一内存布局（bin 直读与 encode_chunk 输出本就连续）
    x = latent_block.unsqueeze(0).to(device)                   # 🔒 (1,9,16,32,32)，B=1
    torch.clear_autocast_cache()                               # 🛡 清掉调用方外层 autocast 可能留下的别种 dtype 权重缓存
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):   # 🔒 内层显式 autocast
        m = encoder(x)                                         # (1, M, D)，内部做入口 affine 归一化
    out = m.reshape(-1).float().cpu().numpy()
    assert np.isfinite(out).all()
    return out


# ---------------- 读输入的辅助（CLI / crosscheck 用） ----------------

def read_frames_window(video_h5, anchor):
    """v7 raw h5（`frames` dataset，(T,256,256,3) uint8 RGB）→ 帧 [anchor, anchor+33) 的连续 numpy。"""
    from motion_jepa.video_io import read_video_frames
    frames = read_video_frames(video_h5)                       # torch (T,H,W,3) uint8
    T = int(frames.shape[0])
    assert 0 <= anchor and anchor + WINDOW <= T, f"锚点 {anchor} 越界（T={T}，窗口 {WINDOW}）"
    return np.ascontiguousarray(frames[anchor:anchor + WINDOW].numpy())


def read_latent_block(bin_path, chunk):
    """按偏移 chunk × 589,824 B seek 直读一个组优先 (9,16,32,32) fp32 block（与 dataset_wan.read_block 同法）。"""
    with open(bin_path, "rb") as f:
        f.seek(chunk * CHUNK_BYTES)
        buf = f.read(CHUNK_BYTES)
    assert len(buf) == CHUNK_BYTES, f"{bin_path} chunk {chunk} 读不满一块（{len(buf)} B）"
    return torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(LAT_G, LAT_C, LAT_H, LAT_W)


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="口径 A 最小前向：33 帧 → latent → motion token")
    ap.add_argument("--run", default="wan-v8-filter10-72ep-a")
    ap.add_argument("--checkpoint", default="checkpoint_epoch_72.pt")
    ap.add_argument("--runs_root", default="./runs")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_h5", help="v7 raw 视频 h5，与 --anchor 配合走 VAE 段")
    src.add_argument("--latent_bin", help="离线 latent bin，与 --chunk 配合跳过 VAE 段")
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="实测无影响，钉住只为卫生")
    ap.add_argument("--out", help="把 token 存成 .npy")
    ap.add_argument("--provenance_out", help="把 info（含 provenance）存成 json")
    ap.add_argument("--expected_ckpt_sha256")
    ap.add_argument("--expected_vae_sha256", default=VAE_STATE_SHA256_EXPECTED,
                    help="传空串可跳过 VAE 指纹断言")
    ap.add_argument("--no_strict_versions", action="store_true", help="版本不符只警告不拦")
    args = ap.parse_args()

    check_env()
    flags = pin_numerics()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("口径 A 的逐位结论只在 CUDA 上实测过；CPU 路径不在保证范围")
    strict = not args.no_strict_versions

    run_dir = os.path.join(args.runs_root, args.run)
    encoder, info, use_amp = load_encoder(run_dir, args.checkpoint, device,
                                          expected_sha256=args.expected_ckpt_sha256,
                                          strict_versions=strict)
    print(f"[encoder] {info['run_dir']}/{info['checkpoint']} ep{info['checkpoint_epoch']} "
          f"sha256={info['checkpoint_sha256'][:16]}… amp={info['amp']} tf32={info['tf32']} batch={info['batch']}",
          flush=True)

    if args.video_h5:
        vae, vinfo = load_vae(info["vae_id"], device,
                              expected_state_sha256=(args.expected_vae_sha256 or None),
                              strict_versions=strict)
        print(f"[vae] {vinfo['vae_id']} state_sha256={vinfo['vae_state_sha256'][:16]}…", flush=True)
        frames = read_frames_window(args.video_h5, args.anchor)
        lat = encode_chunk(vae, frames, device)
        info["input"] = {"video_h5": os.path.abspath(args.video_h5), "anchor": args.anchor,
                         "vae_state_sha256": vinfo["vae_state_sha256"]}
        print(f"[vae] 帧 [{args.anchor}, {args.anchor + WINDOW}) → latent {tuple(lat.shape)} "
              f"rms={float(lat.pow(2).mean().sqrt()):.6f}", flush=True)
    else:
        lat = read_latent_block(args.latent_bin, args.chunk)
        info["input"] = {"latent_bin": os.path.abspath(args.latent_bin), "chunk": args.chunk}
        print(f"[bin] {args.latent_bin} chunk {args.chunk} → latent {tuple(lat.shape)} "
              f"rms={float(lat.pow(2).mean().sqrt()):.6f}", flush=True)

    tok = motion_token(encoder, lat, use_amp, device)
    print(f"[token] shape={tok.shape} rms={float(np.sqrt(np.mean(tok ** 2))):.6f} "
          f"前 8 维={np.array2string(tok[:8], precision=5)}", flush=True)
    print(f"[flags] {json.dumps(flags, ensure_ascii=False)}", flush=True)
    if args.out:
        np.save(args.out, tok.astype(np.float32))
        print(f"→ {args.out}", flush=True)
    if args.provenance_out:
        with open(args.provenance_out, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"→ {args.provenance_out}", flush=True)


if __name__ == "__main__":
    main()
