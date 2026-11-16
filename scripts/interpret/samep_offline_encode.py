#!/usr/bin/env python3
"""同一局 steering 供体的离线批量编码（steer-speedup-1116 F1 后半，计划第二部分 1 节 S-A、7 节契约 C1 / C2）。

输入（契约 C1，由 samep_rs_variants.py --save-window-frames 写）：
    <variants>/vframes/<variant_key>.npz
        start_frames  int64 [W]                 与在线窗口起点同号
        frames        uint8 [W,33,256,256,3]    逐字节等于在线编码器收到的窗口输入
        meta          json 字符串（0 维）        key（须等于文件名主干）/ seed / difficulty / j / group / scope=all|core|demo_full …
    文件名主干 variant_key 由 samep_rs_variants.variant_key() 生成；W=0 的空文件计 empty，不编码也不写 C2。
输出（契约 C2，与上一轮 motion_swap_exact 供体 npz 同名同 dtype，serve_policy_attn.load_donor 可直接读）：
    <offline>/<variant_key>.npz
        start_frames  int32 [W]
        embs          float32 [W,D]             MotionMem D=768（VAE + encoder）；Frozen Wan D=768（VAE + 零参数池化）
        counts        int32 [W]（全 1）
    <offline>/OFFLINE_ENC.txt                   判定行 OFFLINE_ENC=PASS|FAIL model=… windows=… files=…
    <offline>/OFFLINE_ENC.shard<i>of<n>.json    每片清单（模型身份、快速标记、逐文件窗口数与输出 sha256、判定行）

数值：与在线 sidecar 快速模式同一套代码（`scripts/dataset/wan/wan_motion_fast.py`：VAE bf16 autocast、TF32 开、
cudnn.benchmark；encoder bf16 autocast；池化 `src/mme_vla_suite/shared/wan_pool768.py`），只是按批（默认 16）喂。
模型加载复用 `wan_motion_infer.load_vae / load_encoder`（SOURCE_PIN 同 sidecar 一样先核）。
`MMEVLA_MOTION_FAST_VAE_FP32=1` 或 `--vae-fp32` 为 ENC_TOL 失败的回退档（VAE fp32、TF32 关，保留批量化）。
批量一致性：每片取前 `--check-single` 个窗口逐窗（B=1）再编一次，与批量结果比 cos / maxabs（阈值同 ENC_TOL）。

多 GPU：`--shard i/n` 按排序后的文件名取第 i、i+n、… 个文件；各片写自己的 shard 清单，最后
`--merge-shards n` 汇总成 OFFLINE_ENC.txt（n=1 时直接写）。

运行环境：wan venv（需 torch / diffusers / motion_jepa），例如
    MMEVLA_V1_STORE=<v1-store> uv run --project scripts/dataset/wan --no-sync python scripts/interpret/samep_offline_encode.py \\
        --model motion --vframes <variants>/vframes --out <offline> --shard 0/1
本文件顶层只依赖标准库与 NumPy（torch 在 build_encoder 里才导入），单测可注入假编码函数。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import uuid

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
WAN_DIR = REPO / "scripts" / "dataset" / "wan"
POOL768_PATH = REPO / "src" / "mme_vla_suite" / "shared" / "wan_pool768.py"
WINDOW_SHAPE = (33, 256, 256, 3)
VAE_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_ENCODER_RUN = "wan-full1600-filter2-b176x4-72ep-a"
MODELS = ("motion", "wanpool")
MIN_COS, MAX_ABS = 0.999, 0.1


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_shard(text: str) -> tuple[int, int]:
    try:
        i, n = (int(x) for x in str(text).split("/"))
    except ValueError as exc:
        raise ValueError(f"--shard 须为 i/n，得到 {text!r}") from exc
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"--shard 越界: {text!r}")
    return i, n


def list_inputs(vframes_dir, shard=(0, 1)) -> list[pathlib.Path]:
    """按文件名排序后取第 i 片；目录里的临时文件（. 开头）不算。"""
    files = sorted(p for p in pathlib.Path(vframes_dir).glob("*.npz") if not p.name.startswith("."))
    i, n = shard
    return files[i::n]


def read_vframes(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """读并核契约 C1；任何不符 raise ValueError。frames 以 mmap 之外的普通数组返回（npz 不支持 mmap）。"""
    with np.load(path, allow_pickle=False) as z:
        missing = {"start_frames", "frames"} - set(z.files)
        if missing:
            raise ValueError(f"{path} 缺键 {sorted(missing)}")
        starts = np.asarray(z["start_frames"])
        frames = np.asarray(z["frames"])
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if meta.get("key") is not None and meta["key"] != pathlib.Path(path).stem:
        raise ValueError(f"{path} meta.key={meta['key']!r} 与文件名主干不符")
    if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError(f"{path} start_frames 须为一维整数")
    if frames.dtype != np.uint8 or frames.shape != (starts.size, *WINDOW_SHAPE):
        raise ValueError(f"{path} frames 须为 uint8 {(starts.size, *WINDOW_SHAPE)}，得到 {frames.dtype} {frames.shape}")
    if np.unique(starts).size != starts.size:
        raise ValueError(f"{path} start_frames 有重复")
    if (starts < 0).any() or (starts > np.iinfo(np.int32).max).any():
        raise ValueError(f"{path} start_frames 越出 int32")
    return starts.astype(np.int64), frames, meta


def encode_windows(frames: np.ndarray, encode_batch, batch: int) -> np.ndarray:
    """按批调用 encode_batch((b,33,256,256,3) uint8) -> (b,D) float32，拼回 (W,D)。"""
    if batch < 1:
        raise ValueError("--batch 须 ≥ 1")
    out = []
    for s in range(0, frames.shape[0], batch):
        chunk = np.ascontiguousarray(frames[s:s + batch])
        embs = np.asarray(encode_batch(chunk), dtype=np.float32)
        if embs.ndim != 2 or embs.shape[0] != chunk.shape[0]:
            raise ValueError(f"编码函数返回形状 {embs.shape}，期望 ({chunk.shape[0]}, D)")
        out.append(embs)
    embs = np.concatenate(out, axis=0)
    if not np.isfinite(embs).all():
        raise ValueError("编码结果含非有限值")
    return embs


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """逐行 cos 的最小值与全体最大绝对差（float64 计算；双零向量 cos 记 1）。"""
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    dot = (a * b).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where((na == 0) & (nb == 0), 1.0, dot / (na * nb))
    cos = np.nan_to_num(cos, nan=0.0)
    return float(cos.min()), float(np.abs(a - b).max())


def write_offline(out_dir, key: str, starts: np.ndarray, embs: np.ndarray) -> str:
    """原子写契约 C2 文件并逐位读回；返回输出 sha256。"""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}.npz"
    tmp = out_dir / f".{key}.{uuid.uuid4().hex}.tmp.npz"
    record = {"start_frames": np.asarray(starts, np.int32), "embs": np.asarray(embs, np.float32),
              "counts": np.ones(len(starts), np.int32)}
    np.savez(tmp, **record)
    os.replace(tmp, path)
    with np.load(path, allow_pickle=False) as back:
        ok = (sorted(back.files) == sorted(record)
              and all(back[k].dtype == record[k].dtype and np.array_equal(back[k], record[k]) for k in record))
    if not ok:
        raise ValueError(f"{path} 读回与写入不一致")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def existing_ok(out_dir, key: str, starts: np.ndarray) -> bool:
    path = pathlib.Path(out_dir) / f"{key}.npz"
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return (z["embs"].dtype == np.float32 and z["embs"].shape[0] == len(starts)
                    and np.array_equal(np.asarray(z["start_frames"], np.int64), starts))
    except Exception:
        return False


def judgement(model: str, s: dict) -> str:
    ok = s["files"] > 0 and s["bad"] == 0 and s["single_ok"]
    return (f"OFFLINE_ENC={'PASS' if ok else 'FAIL'} model={model} windows={s['windows']} files={s['files']} "
            f"bad={s['bad']} reused={s['reused']} empty={s['empty']} batch={s['batch']} single_check={s['single_n']} "
            f"single_min_cos={s['single_min_cos']:.6f} single_maxabs={s['single_maxabs']:.3g} "
            f"shard={s['shard'][0]}/{s['shard'][1]}")


def run(inputs, out_dir, encode_batch, *, model: str, batch: int = 16, check_single: int = 4,
        min_cos: float = MIN_COS, max_abs: float = MAX_ABS, shard=(0, 1), skip_existing: bool = False,
        identity: dict | None = None, log=print) -> dict:
    """编码一片输入文件并写清单；返回 summary（含 line）。"""
    out_dir = pathlib.Path(out_dir)
    s = {"files": 0, "windows": 0, "bad": 0, "reused": 0, "empty": 0, "batch": int(batch), "single_n": 0,
         "single_min_cos": 1.0, "single_maxabs": 0.0, "single_ok": True, "shard": list(shard),
         "entries": [], "errors": []}
    for path in inputs:
        key = pathlib.Path(path).stem
        try:
            starts, frames, meta = read_vframes(path)
            if starts.size == 0:                                  # S-B 对无相交窗的变体写空文件：不编码、不出 C2
                s["empty"] += 1
                s["entries"].append({"key": key, "windows": 0, "empty": True, "meta": meta})
                continue
            if skip_existing and existing_ok(out_dir, key, starts):
                s["reused"] += 1
                s["files"] += 1
                s["windows"] += len(starts)
                s["entries"].append({"key": key, "windows": int(len(starts)), "reused": True, "meta": meta})
                continue
            embs = encode_windows(frames, encode_batch, batch)
            want = check_single - s["single_n"]
            if want > 0:
                k = min(want, len(starts))
                single = np.concatenate([np.asarray(encode_batch(frames[i:i + 1]), np.float32) for i in range(k)])
                c, m = compare(embs[:k], single)
                s["single_n"] += k
                s["single_min_cos"] = min(s["single_min_cos"], c)
                s["single_maxabs"] = max(s["single_maxabs"], m)
            sha = write_offline(out_dir, key, starts, embs)
        except Exception as exc:                                  # 单文件失败计 bad，不中断整片
            s["bad"] += 1
            s["errors"].append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            log(f"[offline_enc] 失败 {path}: {exc}")
            continue
        s["files"] += 1
        s["windows"] += len(starts)
        s["entries"].append({"key": key, "windows": int(len(starts)), "dim": int(embs.shape[1]), "sha256": sha,
                             "meta": meta})
    s["single_ok"] = s["single_min_cos"] >= min_cos and s["single_maxabs"] <= max_abs
    s["line"] = judgement(model, s)
    i, n = shard
    manifest = {"schema": "samep-offline-encode-v1", "model": model, "identity": identity or {},
                "thresholds": {"min_cos": min_cos, "max_abs": max_abs}, **s}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"OFFLINE_ENC.shard{i}of{n}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    if n == 1:
        (out_dir / "OFFLINE_ENC.txt").write_text(s["line"] + "\n")
    return s


def merge_shards(out_dir, n: int) -> str:
    """汇总 n 片清单写 OFFLINE_ENC.txt；缺片、模型不一或任一片 FAIL 即 FAIL。"""
    out_dir = pathlib.Path(out_dir)
    tot = {"files": 0, "windows": 0, "bad": 0, "reused": 0, "empty": 0, "single_n": 0}
    models, fails, missing, min_c, max_a = set(), 0, 0, 1.0, 0.0
    for i in range(n):
        path = out_dir / f"OFFLINE_ENC.shard{i}of{n}.json"
        if not path.is_file():
            missing += 1
            continue
        m = json.loads(path.read_text())
        models.add(m["model"])
        for k in tot:
            tot[k] += int(m.get(k, 0))
        min_c, max_a = min(min_c, m["single_min_cos"]), max(max_a, m["single_maxabs"])
        fails += int(not m["line"].startswith("OFFLINE_ENC=PASS"))
    ok = missing == 0 and fails == 0 and len(models) == 1 and tot["files"] > 0
    line = (f"OFFLINE_ENC={'PASS' if ok else 'FAIL'} model={','.join(sorted(models)) or 'none'} "
            f"windows={tot['windows']} files={tot['files']} bad={tot['bad']} reused={tot['reused']} empty={tot['empty']} "
            f"shards={n} missing_shards={missing} failed_shards={fails} single_check={tot['single_n']} "
            f"single_min_cos={min_c:.6f} single_maxabs={max_a:.3g}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "OFFLINE_ENC.txt").write_text(line + "\n")
    return line


def build_encoder(args):
    """加载与在线 sidecar 同一套 VAE + encoder / 池化；返回 (encode_batch, identity)。"""
    v1 = pathlib.Path(args.v1_store or os.environ.get("MMEVLA_V1_STORE") or (REPO / "v1-store"))
    os.environ.setdefault("HF_HOME", str(v1 / "cache" / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    W = _load(WAN_DIR / "wan_motion_infer.py", "wan_motion_infer")
    pin = json.loads((WAN_DIR / "SOURCE_PIN.json").read_text(encoding="utf-8"))
    if W.sha256_file(str(WAN_DIR / "wan_motion_infer.py")) != pin["source_sha256"]:
        raise SystemExit("复制件 wan_motion_infer.py 与 SOURCE_PIN 不符")
    F = _load(WAN_DIR / "wan_motion_fast.py", "wan_motion_fast")
    F._W = W
    fp32 = bool(args.vae_fp32) or F.vae_fp32()
    W.check_env()
    flags = F.pin_numerics_fast(fp32)
    W.check_versions(strict=True, with_diffusers=True)
    torch.manual_seed(0)
    device = torch.device(args.device)
    vae, vinfo = W.load_vae(VAE_ID, device, expected_state_sha256=(args.expected_vae_sha256 or W.VAE_STATE_SHA256_EXPECTED))
    identity = {"model": args.model, "vae_state_sha256": vinfo["vae_state_sha256"], "vae_id": vinfo["vae_id"],
                "module_sha256": vinfo["module_sha256"], "fast": F.fast_info(fp32, flags, batch=args.batch),
                "gpu_name": vinfo.get("gpu_name"), "hostname": vinfo.get("hostname")}
    if args.model == "motion":
        lock = _load(REPO / "scripts" / "assets" / "assets_lock.py", "assets_lock")
        expected = args.expected_ckpt_sha256 or lock.expected_sha256("motionjepa_ckpt")
        run_dir = args.encoder_run_dir or str(v1 / "external" / "motionjepa" / DEFAULT_ENCODER_RUN)
        encoder, einfo, use_amp = W.load_encoder(run_dir, args.checkpoint, device, expected_sha256=expected)
        identity.update(checkpoint_sha256=einfo["checkpoint_sha256"], checkpoint=einfo["checkpoint"],
                        checkpoint_epoch=einfo["checkpoint_epoch"], arch=einfo["arch"], motion_dims=einfo["motion_dims"],
                        encoder_src_sha256=einfo["encoder_src_sha256"], encoder_amp=einfo["amp"])

        def encode_batch(windows):
            F.pin_numerics_fast(fp32)
            return F.motion_tokens(encoder, F.encode_chunks(vae, windows, device, fp32), use_amp, device)
    else:
        PL = _load(POOL768_PATH, "wan_pool768")
        pool_sha = PL.pool_source_sha256()
        if args.expected_pool_sha256 and args.expected_pool_sha256 != pool_sha:
            raise SystemExit(f"池化源码 sha {pool_sha} ≠ 期望 {args.expected_pool_sha256}")
        identity.update(encoder_kind=PL.POOL768_ENCODER_KIND, readout=PL.POOL768_READOUT, pool_source_sha256=pool_sha)

        def encode_batch(windows):
            F.pin_numerics_fast(fp32)
            z = F.encode_chunks(vae, windows, device, fp32).detach().float().cpu().numpy()
            return np.stack([PL.pool_raw_wan_latent_768(np.ascontiguousarray(z[i])).reshape(768) for i in range(len(z))])
    return encode_batch, identity


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="同一局供体离线批量编码（契约 C1 → C2）")
    ap.add_argument("--model", choices=MODELS, help="motion = MotionMem（VAE + encoder）；wanpool = Frozen Wan pooling")
    ap.add_argument("--vframes", help="契约 C1 目录 <variants>/vframes")
    ap.add_argument("--out", required=True, help="契约 C2 输出目录 <offline>")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge-shards", type=int, default=0, help="只汇总 n 片清单写 OFFLINE_ENC.txt，不编码")
    ap.add_argument("--check-single", type=int, default=4, help="每片逐窗复编的窗口数（批量一致性）")
    ap.add_argument("--min-cos", type=float, default=MIN_COS)
    ap.add_argument("--max-abs", type=float, default=MAX_ABS)
    ap.add_argument("--skip-existing", action="store_true", help="输出已存在且起点一致的文件不重编")
    ap.add_argument("--vae-fp32", action="store_true", help="回退档：VAE fp32、TF32 关（同 MMEVLA_MOTION_FAST_VAE_FP32=1）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--v1-store", default=None)
    ap.add_argument("--encoder-run-dir", default=None)
    ap.add_argument("--checkpoint", default="checkpoint_epoch_72.pt")
    ap.add_argument("--expected-ckpt-sha256", default="")
    ap.add_argument("--expected-vae-sha256", default="")
    ap.add_argument("--expected-pool-sha256", default="")
    args = ap.parse_args(argv)
    if args.merge_shards:
        line = merge_shards(args.out, args.merge_shards)
        print(line, flush=True)
        return 0 if line.startswith("OFFLINE_ENC=PASS") else 1
    if not args.model or not args.vframes:
        ap.error("编码时须给 --model 与 --vframes")
    shard = parse_shard(args.shard)
    inputs = list_inputs(args.vframes, shard)
    encode_batch, identity = build_encoder(args)
    s = run(inputs, args.out, encode_batch, model=args.model, batch=args.batch, check_single=args.check_single,
            min_cos=args.min_cos, max_abs=args.max_abs, shard=shard, skip_existing=args.skip_existing,
            identity=identity, log=lambda m: print(m, file=sys.stderr, flush=True))
    print(s["line"], flush=True)
    return 0 if s["line"].startswith("OFFLINE_ENC=PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
