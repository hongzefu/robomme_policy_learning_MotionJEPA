#!/usr/bin/env python3
"""S0 探针：Wan VAE + MotionJEPA encoder 在本机 Ada 上的耗时 / 显存 / 精度漂移 / 跨卡 / 双 venv 逐位。

对应 motion-memory-plan.md 第二部分四节表二的 A2（延迟与漂移，只记录）、A3（跨卡探针）、A4（双 venv 探针）。
本脚本**不复写任何数值语句**：所有前向都经 `wan_motion_infer` 的 `load_vae / encode_chunk / load_encoder / motion_token`
调用，`--module copy` 导入本目录的复制件，`--module orig` 导入 MotionJEPA 树内原版（只在 MotionJEPA 的 uv 环境下有意义）。

子命令：
  bench    N 窗计时（每窗 cuda 同步）+ `max_memory_allocated` + A2 漂移（TF32 / bf16 两档 vs fp32 关 TF32，余弦与 max|diff|，只记录）
  encode   N 窗 → npz（起点、33 帧 uint8 sha256、latent (N,9,16,32,32) f32、token (N,768) f32、两段 provenance）
  compare  两个 npz → 起点与帧 sha 必须相同；latent / token 逐位计数与 max|diff|；provenance 白名单逐键比对

窗口取自官方 h5 `episode_<i>/timestep_<t>/obs/front_rgb`（(256,256,3) uint8），起点按 `--start 0 --stride S` 的等差网格取 N 个。

用法（被测侧，wan 子 venv）：
  UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan CUDA_VISIBLE_DEVICES=1 \\
    uv run --project scripts/dataset/wan --no-sync python scripts/dataset/wan/probe_wan.py --module copy bench ...
用法（oracle 侧，MotionJEPA uv 环境）：
  PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy CUDA_VISIBLE_DEVICES=1 \\
    uv run --project /nfs/.../MotionJEPA --no-sync python scripts/dataset/wan/probe_wan.py --module orig encode ...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

import h5py
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "assets"))
import assets_lock as al  # noqa: E402

MJ_REPO_DEFAULT = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA"
VAE_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
ENCODER_RUN_DIR_DEFAULT = str(_REPO_ROOT / "v1-store" / "external" / "motionjepa" / "wan-v8-filter10-72ep-a")
CKPT_NAME = "checkpoint_epoch_72.pt"


def _expected_ckpt(args):
    """ckpt 期望值：默认取 ASSETS_LOCK.json 钉死的那份，显式传 SKIP 才跳过。

    本文件是**探针 / 对拍工具**，探一个未入 lock 的 run_dir 是它的本职，所以保留 SKIP 出口；
    生产路径（encode_motion.py 的 required=True、run_local.py、motion_sidecar.py）不提供 SKIP。
    """
    if args.expected_ckpt_sha256 == "SKIP":
        return None
    return args.expected_ckpt_sha256 or al.expected_sha256("motionjepa_ckpt")


# A4 provenance 白名单：两侧必须逐键相等；刻意排除 hostname / python 补丁号 / 路径类键（计划 A4）
PROV_SAME_KEYS = (
    "torch", "cuda", "cudnn", "cublas_pkg", "cudnn_pkg", "diffusers",
    "gpu_name", "compute_cap", "sm_count", "driver", "flags", "env",
    "module_sha256", "encoder_src_sha256",
)
VAE_SAME_KEYS = ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp")
ENC_SAME_KEYS = ("checkpoint", "checkpoint_sha256", "checkpoint_epoch", "arch", "state_key",
                 "precision", "amp", "tf32", "batch", "vae_id", "motion_dims")


def load_module(which: str, mj_repo: str):
    """按名字加载 wan_motion_infer：copy = 本目录复制件；orig = MotionJEPA 树内原版。"""
    if which == "copy":
        path = _HERE / "wan_motion_infer.py"
    else:
        path = pathlib.Path(mj_repo) / "scripts" / "inference-example" / "wan_motion_infer.py"
    if not path.exists():
        raise SystemExit(f"缺文件：{path}")
    spec = importlib.util.spec_from_file_location("wan_motion_infer", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def sha256_bytes(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def read_windows(h5_path: str, episode: int, starts: list[int], window: int) -> list[np.ndarray]:
    out = []
    with h5py.File(h5_path, "r") as f:
        ep = f[f"episode_{episode}"]
        T = sum(1 for k in ep.keys() if k.startswith("timestep_"))
        for s in starts:
            if s + window > T:
                raise SystemExit(f"起点 {s} 越界：episode_{episode} 只有 {T} 帧，窗口 {window}")
            frames = np.stack([ep[f"timestep_{t}/obs/front_rgb"][()] for t in range(s, s + window)])
            if frames.shape != (window, 256, 256, 3) or frames.dtype != np.uint8:
                raise SystemExit(f"帧形制异常 {frames.shape} {frames.dtype}")
            out.append(np.ascontiguousarray(frames))
    return out


def grid_starts(start: int, stride: int, n: int) -> list[int]:
    return [start + stride * i for i in range(n)]


def setup(args):
    """公共起手：导入模块、check_env + pin_numerics + check_versions、加载 VAE 与 encoder。"""
    import torch
    W, mod_path = load_module(args.module, args.mj_repo)
    W.check_env()
    flags = W.pin_numerics()
    W.check_versions(strict=True, with_diffusers=True)
    torch.manual_seed(0)
    device = torch.device("cuda")
    vae, vinfo = W.load_vae(VAE_ID, device, expected_state_sha256=W.VAE_STATE_SHA256_EXPECTED)
    encoder, einfo, use_amp = W.load_encoder(args.encoder_run_dir, CKPT_NAME, device,
                                             expected_sha256=_expected_ckpt(args))
    print(f"[setup] module={args.module} path={mod_path} sha256={W.sha256_file(str(mod_path))[:16]}…", flush=True)
    print(f"[setup] torch={torch.__version__} cudnn={torch.backends.cudnn.version()} "
          f"gpu={torch.cuda.get_device_name(0)} use_amp={use_amp}", flush=True)
    print(f"[setup] vae_state_sha256={vinfo['vae_state_sha256'][:16]}… ckpt_sha256={einfo['checkpoint_sha256'][:16]}… "
          f"epoch={einfo['checkpoint_epoch']}", flush=True)
    return W, torch, device, vae, vinfo, encoder, einfo, use_amp, flags


def run_pair(W, torch, device, vae, encoder, use_amp, frames):
    """一窗：encode_chunk → motion_token，返回 (latent cpu f32 numpy, token f32 numpy)。"""
    lat = W.encode_chunk(vae, frames, device)
    tok = W.motion_token(encoder, lat, use_amp, device)
    return lat.cpu().numpy().astype(np.float32, copy=False), tok.astype(np.float32, copy=False)


def cos_and_maxdiff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    cos = float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))
    d = float(np.abs(a64 - b64).max())
    rel = d / float(np.sqrt(np.mean(a64 ** 2)))
    return cos, d, rel


# ---------------- bench ----------------

def cmd_bench(args):
    W, torch, device, vae, vinfo, encoder, einfo, use_amp, flags = setup(args)
    starts = grid_starts(args.start, args.stride, args.n)
    windows = read_windows(args.h5, args.episode, starts, W.WINDOW)
    print(f"[bench] {len(windows)} 窗，起点 {starts}", flush=True)

    # 预热一窗（吃掉 CUDA 初始化、cudnn 首次开销），结果丢弃
    torch.cuda.reset_peak_memory_stats(device)
    run_pair(W, torch, device, vae, encoder, use_amp, windows[0])
    torch.cuda.synchronize(device)

    t_vae, t_enc, base = [], [], []
    torch.cuda.reset_peak_memory_stats(device)
    for fw in windows:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        lat = W.encode_chunk(vae, fw, device)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        tok = W.motion_token(encoder, lat, use_amp, device)
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        t_vae.append(t1 - t0)
        t_enc.append(t2 - t1)
        base.append((lat.cpu().numpy().astype(np.float32), tok.astype(np.float32)))
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
    tv, te = np.array(t_vae), np.array(t_enc)
    print(f"[bench] VAE 段 ms/窗: mean={tv.mean()*1e3:.1f} median={np.median(tv)*1e3:.1f} "
          f"min={tv.min()*1e3:.1f} max={tv.max()*1e3:.1f}", flush=True)
    print(f"[bench] encoder 段 ms/窗: mean={te.mean()*1e3:.1f} median={np.median(te)*1e3:.1f}", flush=True)
    print(f"[bench] 合计 ms/窗: mean={(tv+te).mean()*1e3:.1f}  max_memory_allocated={peak_mib:.0f} MiB", flush=True)

    # 同设置重跑逐位（确定性自证）
    rerun_bits = 0
    for fw, (lat0, tok0) in zip(windows, base):
        W.pin_numerics()
        lat1, tok1 = run_pair(W, torch, device, vae, encoder, use_amp, fw)
        rerun_bits += int(np.array_equal(lat0, lat1) and np.array_equal(tok0, tok1))
    print(f"[bench] 同设置重跑逐位相同 {rerun_bits}/{len(windows)}", flush=True)

    # ---- A2 漂移探针（只记录，不设通过线；生产/在线均不启用这两档）----
    import warnings

    def drift(tag, run_fn):
        lat_cos, lat_d, lat_rel, tok_cos, tok_d, tok_rel, bits = [], [], [], [], [], [], 0
        for fw, (lat0, tok0) in zip(windows, base):
            lat1, tok1 = run_fn(fw)
            c, d, r = cos_and_maxdiff(lat0, lat1); lat_cos.append(c); lat_d.append(d); lat_rel.append(r)
            c, d, r = cos_and_maxdiff(tok0, tok1); tok_cos.append(c); tok_d.append(d); tok_rel.append(r)
            bits += int(np.array_equal(tok0, tok1))
        rec = {"latent_min_cos": min(lat_cos), "latent_max_abs_diff": max(lat_d), "latent_max_rel": max(lat_rel),
               "token_min_cos": min(tok_cos), "token_max_abs_diff": max(tok_d), "token_max_rel": max(tok_rel),
               "token_bitwise_same": bits, "n": len(windows)}
        print(f"[A2 {tag}] latent min_cos={rec['latent_min_cos']:.8f} max|Δ|={rec['latent_max_abs_diff']:.3e} "
              f"rel={rec['latent_max_rel']:.3e} | token min_cos={rec['token_min_cos']:.8f} "
              f"max|Δ|={rec['token_max_abs_diff']:.3e} rel={rec['token_max_rel']:.3e} 逐位 {bits}/{len(windows)}", flush=True)
        return rec

    def run_tf32(fw):
        W.pin_numerics()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            return run_pair(W, torch, device, vae, encoder, use_amp, fw)
        finally:
            W.pin_numerics()

    def run_bf16_vae(fw):
        # 探针专用变体：VAE 段改在 bf16 autocast 下跑（encode_chunk 内部屏蔽外层 autocast，
        # 所以这里另写一份同步骤的前向，只用于记录漂移量，绝不进生产 / 在线路径）
        W.pin_numerics()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = torch.from_numpy(np.ascontiguousarray(fw)).permute(0, 3, 1, 2).float().to(device)
            x = (f / 127.5 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
            z = vae.encode(x).latent_dist.mode()
        lat = z.permute(0, 2, 1, 3, 4).contiguous()[0].float()
        tok = W.motion_token(encoder, lat, use_amp, device)
        return lat.cpu().numpy().astype(np.float32), tok.astype(np.float32)

    # 计时两档的速度（各跑一遍，取均值）
    def timed(run_fn):
        ts = []
        for fw in windows:
            torch.cuda.synchronize(device); t0 = time.perf_counter()
            run_fn(fw)
            torch.cuda.synchronize(device); ts.append(time.perf_counter() - t0)
        return float(np.mean(ts) * 1e3)

    a2 = {"tf32": drift("TF32 开（matmul+cudnn）", run_tf32), "bf16_vae": drift("VAE bf16 autocast", run_bf16_vae)}
    a2["tf32"]["mean_ms_per_window"] = timed(run_tf32)
    a2["bf16_vae"]["mean_ms_per_window"] = timed(run_bf16_vae)
    W.pin_numerics()
    print(f"[A2] 速度 ms/窗：fp32关TF32={(tv+te).mean()*1e3:.1f} TF32={a2['tf32']['mean_ms_per_window']:.1f} "
          f"VAE-bf16={a2['bf16_vae']['mean_ms_per_window']:.1f}", flush=True)

    result = {
        "module": args.module, "h5": os.path.abspath(args.h5), "episode": args.episode, "starts": starts,
        "n_windows": len(windows), "warmup_windows": 1,
        "vae_ms": {"mean": tv.mean() * 1e3, "median": float(np.median(tv)) * 1e3, "min": tv.min() * 1e3, "max": tv.max() * 1e3},
        "encoder_ms": {"mean": te.mean() * 1e3, "median": float(np.median(te)) * 1e3},
        "total_ms_mean": (tv + te).mean() * 1e3,
        "max_memory_allocated_mib": peak_mib,
        "rerun_bitwise_same": rerun_bits,
        "a2_drift": a2,
        "flags": flags, "vae_provenance": vinfo, "encoder_provenance": einfo,
    }
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"→ {args.out}", flush=True)
    ok = rerun_bits == len(windows)
    print(f"PROBE_BENCH={'PASS' if ok else 'FAIL'} windows={len(windows)} ms_per_window={(tv+te).mean()*1e3:.1f} "
          f"peak_mib={peak_mib:.0f} rerun_bitwise={rerun_bits}/{len(windows)}", flush=True)
    if not ok:
        raise SystemExit(1)


# ---------------- encode ----------------

def cmd_encode(args):
    W, torch, device, vae, vinfo, encoder, einfo, use_amp, flags = setup(args)
    starts = grid_starts(args.start, args.stride, args.n)
    windows = read_windows(args.h5, args.episode, starts, W.WINDOW)
    print(f"[encode] {len(windows)} 窗，起点 {starts[0]}..{starts[-1]} stride={args.stride}", flush=True)
    lats, toks, shas = [], [], []
    t0 = time.perf_counter()
    for fw in windows:
        W.pin_numerics()
        lat, tok = run_pair(W, torch, device, vae, encoder, use_amp, fw)
        lats.append(lat); toks.append(tok); shas.append(sha256_bytes(fw))
    torch.cuda.synchronize(device)
    el = time.perf_counter() - t0
    prov = {"module": args.module, "flags": flags, "vae": vinfo, "encoder": einfo,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_uuid": _gpu_uuid()}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, starts=np.array(starts, np.int64), frame_sha256=np.array(shas),
             latents=np.stack(lats).astype(np.float32), tokens=np.stack(toks).astype(np.float32),
             h5=np.array(os.path.abspath(args.h5)), episode=np.array(args.episode),
             provenance=np.array(json.dumps(prov, ensure_ascii=False)))
    print(f"→ {args.out}  ({el:.1f}s, {el/len(windows)*1e3:.0f} ms/窗含落盘)", flush=True)
    print(f"PROBE_ENCODE=DONE windows={len(windows)} module={args.module}", flush=True)


def _gpu_uuid():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis is not None and vis.strip():
            idx = int(vis.split(",")[0])
            for line in out:
                i, u = [s.strip() for s in line.split(",")]
                if int(i) == idx:
                    return u
        return out[0].split(",")[1].strip() if out else "unknown"
    except Exception:
        return "unknown"


# ---------------- compare ----------------

def cmd_compare(args):
    a = np.load(args.a, allow_pickle=False)
    b = np.load(args.b, allow_pickle=False)
    fails = []
    if not np.array_equal(a["starts"], b["starts"]):
        fails.append("starts 不同")
    if not np.array_equal(a["frame_sha256"], b["frame_sha256"]):
        fails.append("输入帧 sha256 不同（两侧读到的 33 帧不是同一份字节）")
    n = int(a["starts"].shape[0])
    lat_bits = sum(int(np.array_equal(a["latents"][i], b["latents"][i])) for i in range(n))
    tok_bits = sum(int(np.array_equal(a["tokens"][i], b["tokens"][i])) for i in range(n))
    lat_d = float(np.abs(a["latents"].astype(np.float64) - b["latents"].astype(np.float64)).max())
    tok_d = float(np.abs(a["tokens"].astype(np.float64) - b["tokens"].astype(np.float64)).max())
    print(f"[compare] latent 逐位 {lat_bits}/{n} max|Δ|={lat_d:.3e} | token 逐位 {tok_bits}/{n} max|Δ|={tok_d:.3e}", flush=True)

    pa = json.loads(str(a["provenance"])); pb = json.loads(str(b["provenance"]))
    prov_diff = []
    for section, keys in (("vae", PROV_SAME_KEYS + VAE_SAME_KEYS), ("encoder", PROV_SAME_KEYS + ENC_SAME_KEYS)):
        for k in keys:
            va, vb = pa[section].get(k), pb[section].get(k)
            if k in args.prov_ignore:
                continue
            if va != vb:
                prov_diff.append(f"{section}.{k}: {va!r} != {vb!r}")
    if pa["flags"] != pb["flags"]:
        prov_diff.append(f"flags: {pa['flags']} != {pb['flags']}")
    for d in prov_diff:
        print(f"[compare] provenance 不等 {d}", flush=True)
    print(f"[compare] gpu_uuid a={pa.get('gpu_uuid')} b={pb.get('gpu_uuid')}  module a={pa['module']} b={pb['module']}", flush=True)

    if args.require_bitwise and (lat_bits != n or tok_bits != n):
        fails.append(f"逐位失败 latent {lat_bits}/{n} token {tok_bits}/{n}")
    if prov_diff and not args.allow_prov_diff:
        fails.append(f"provenance 白名单键不等 {len(prov_diff)} 项")
    tag = args.tag or "PROBE_COMPARE"
    if fails:
        print(f"{tag}=FAIL compared={n} latent_bitwise={lat_bits} token_bitwise={tok_bits} reasons={fails}", flush=True)
        raise SystemExit(1)
    print(f"{tag}=PASS compared={n} latent_bitwise={lat_bits} token_bitwise={tok_bits} max_abs_diff={max(lat_d, tok_d):.3e}",
          flush=True)


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", choices=["copy", "orig"], default="copy")
    ap.add_argument("--mj-repo", default=MJ_REPO_DEFAULT)
    ap.add_argument("--encoder-run-dir", default=ENCODER_RUN_DIR_DEFAULT)
    ap.add_argument("--expected-ckpt-sha256", default="",
                    help="默认按 ASSETS_LOCK.json 校；探未入 lock 的 run_dir 时显式传 SKIP")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--h5", required=True)
    common.add_argument("--episode", type=int, default=0)
    common.add_argument("--start", type=int, default=0)
    common.add_argument("--stride", type=int, required=True)
    common.add_argument("--n", type=int, required=True)

    p = sub.add_parser("bench", parents=[common]); p.add_argument("--out", default=""); p.set_defaults(func=cmd_bench)
    p = sub.add_parser("encode", parents=[common]); p.add_argument("--out", required=True); p.set_defaults(func=cmd_encode)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--require-bitwise", action="store_true")
    p.add_argument("--allow-prov-diff", action="store_true")
    p.add_argument("--prov-ignore", nargs="*", default=[], help="额外忽略的 provenance 键")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
