#!/usr/bin/env python3
"""本机多 GPU 调度器：每 GPU 一个常驻 worker 进程、动态领任务，三阶段共用（motion-memory-plan.md 第二部分 1.3）。

  --stage siglip   主 venv：``build_shard.py --worker-mode``，工作项 = episode（按 num_timesteps LPT 降序），
                   产 ``<lib>/source/{features,data,meta}``；随后另跑 finalize_checks.py / pack_framesamp_store.py
  --stage wan      子 venv：``wan/extract_wan.py``，工作项 = 段（按网格窗数 LPT），产 ``<lib>/wan-latents/``；
                   收尾把逐段 metadata 汇总成 ``wan-latents/metadata.json``（schema 2，唯一窗口清单）
  --stage encode   子 venv：``wan/encode_motion.py``，工作项 = 段，产 ``<lib>/motion-tokens/`` + ``metadata.json``

领任务用 ``<out>/_claims/_claim_<key>``（O_CREAT|O_EXCL），完成即 unlink；收尾断言零残留 claim、工作项全覆盖。
jax 与 torch 不同进程；SigLIP 阶段与 Wan 阶段不并发（显存）。``--gpus 0,1`` 决定起几个 worker，每个只暴露一张卡
（``CUDA_VISIBLE_DEVICES``），崩溃隔离；``--require-free-mib`` 起跑预检每张卡的空闲显存。
日志：每 worker 一份 ``<lib>/logs/<stage>-<worker>.log``，本进程同时把各 worker 的行带前缀转发到 stdout；
结束打 ``STAGE_DONE stage=… workers=… items=… elapsed=…``（tmux 内由外层 tee 落总日志）。

环境变量与 paths.sh 同口径（本脚本自行显式设置，不覆盖 HOME）：OPENPI_DATA_HOME / HF_HOME / XDG_CACHE_HOME /
JAX_COMPILATION_CACHE_DIR / UV_LINK_MODE=copy；子 venv 另设 UV_PROJECT_ENVIRONMENT=$V1_STORE/venvs/wan、HF_HUB_OFFLINE=1。

用法：
  UV_LINK_MODE=copy uv run python scripts/dataset/run_local.py --stage wan --lib v1-store/datasets/4task-motion-40ep --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "assets"))
sys.path.insert(0, str(_HERE / "wan"))

import assets_lock as al  # noqa: E402

V1_STORE = _REPO_ROOT / "v1-store"
# 每个 stage 起手要校的资产（ASSETS_LOCK.json 里的名字）
STAGE_ASSETS = {"siglip": ["siglip_params"], "wan": ["wan_vae"],
                "encode": ["wan_vae", "motionjepa_ckpt", "motionjepa_config"]}
# 环境 A 默认本机 16 任务全集；环境 B 由 paths.sh 导出的 RAW_H5_DIR 覆盖（/scratch/hongze/robomme_data_h5）
RAW_H5_DEFAULT = os.environ.get("RAW_H5_DIR", "/data/hongzefu/robomme_data_h5")
ENCODER_RUN_DIR_DEFAULT = V1_STORE / "external" / "motionjepa" / "wan-v8-filter10-72ep-a"
CKPT_DEFAULT = "checkpoint_epoch_72.pt"


def base_env() -> dict:
    env = dict(os.environ)
    env.update({
        "OPENPI_DATA_HOME": str(V1_STORE / "models"),
        "XDG_CACHE_HOME": str(V1_STORE / "cache" / "xdg"),
        "HF_HOME": str(V1_STORE / "cache" / "hf"),
        "JAX_COMPILATION_CACHE_DIR": str(V1_STORE / "cache" / "jax"),
        "UV_LINK_MODE": "copy",
        "PYTHONUNBUFFERED": "1",
    })
    for k in ("XDG_CACHE_HOME", "HF_HOME", "JAX_COMPILATION_CACHE_DIR"):
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def gpu_free_mib() -> dict[int, int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, check=True).stdout
    res = {}
    for line in out.strip().splitlines():
        i, m = [s.strip() for s in line.split(",")]
        res[int(i)] = int(m)
    return res


def worker_cmd(stage: str, gpu: int, k: int, n: int, args, env: dict) -> tuple[list[str], dict, pathlib.Path]:
    lib = pathlib.Path(args.lib).resolve()
    label = f"gpu{gpu}"
    env = dict(env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    manifest = str(lib / "meta" / "episode_manifest.json")
    if stage == "siglip":
        out = lib / "source"
        # --no-sync：8 个 worker 同时起会争 uv 的同步锁（40 ep 留档结论：后续命令一律 --no-sync）
        cmd = ["uv", "run", "--no-sync", "python", str(_HERE / "build_shard.py"), "--manifest", manifest,
               "--raw_dir", args.raw_dir, "--out", str(out), "--worker-mode", "--worker-label", label,
               "--worker-idx", str(k), "--num-workers", str(n), "--resume", "--report_every", "2000"]
    else:
        env["UV_PROJECT_ENVIRONMENT"] = str(V1_STORE / "venvs" / "wan")
        env["HF_HUB_OFFLINE"] = "1"
        uvrun = ["uv", "run", "--project", str(_HERE / "wan"), "--no-sync", "python"]
        if stage == "wan":
            out = lib / "wan-latents"
            cmd = [*uvrun, str(_HERE / "wan" / "extract_wan.py"), "--manifest", manifest, "--raw-dir", args.raw_dir,
                   "--out", str(out), "--worker", label]
        else:
            out = lib / "motion-tokens"
            cmd = [*uvrun, str(_HERE / "wan" / "encode_motion.py"), "--manifest", manifest,
                   "--latents", str(lib / "wan-latents"), "--out", str(out), "--worker", label,
                   "--encoder-run-dir", str(args.encoder_run_dir), "--checkpoint", args.checkpoint,
                   "--expected-ckpt-sha256", args.expected_ckpt_sha256]
    return cmd, env, out


def pump(prefix: str, proc: subprocess.Popen, logf) -> None:
    for line in proc.stdout:
        logf.write(line)
        logf.flush()
        sys.stdout.write(f"[{prefix}] {line}")
        sys.stdout.flush()


def aggregate_segments(out: pathlib.Path, manifest_path: pathlib.Path, stage: str) -> int:
    """把逐段 metadata 汇总成 <out>/metadata.json（schema 2），断言集合 == 清单重算的段集合、零残留 claim。"""
    import wan_common as wc
    manifest = wc.load_manifest(manifest_path)
    expect = {it["key"]: it for it in wc.list_segments(manifest)}
    segs = {}
    workers = {}
    for p in sorted(out.glob("*.metadata.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        key = m["segment"]
        if key not in expect:
            raise SystemExit(f"{p} 的段 {key} 不在清单重算集合内")
        w = m["worker"]
        workers[f"{w['hostname']}:{w['gpu_uuid']}:{w['worker']}:{w['pid']}"] = w
        if stage == "wan":
            segs[key] = {"num_grid": m["num_grid"], "seg_len": m["seg_len"], "seg_start_global": m["seg_start_global"],
                         "segment_kind": m["segment_kind"], "g": m["g"], "rows": m["rows"], "sha256": m["sha256"],
                         "bytes": m["bytes"], "worker": w["worker"]}
        else:
            segs[key] = {"num_grid": m["num_grid"], "segment_kind": m["segment_kind"], "g": m["g"],
                         "sha256": m["sha256"], "bytes": m["bytes"], "input_latent_sha256": m["input_latent_sha256"],
                         "worker": w["worker"]}
        if int(m["num_grid"]) != expect[key]["num_grid"]:
            raise SystemExit(f"{key} num_grid {m['num_grid']} != 清单重算 {expect[key]['num_grid']}")
    missing = sorted(set(expect) - set(segs))
    if missing:
        raise SystemExit(f"缺 {len(missing)} 段产物: {missing[:8]}")
    left = sorted((out / "_claims").glob("_claim_*")) if (out / "_claims").is_dir() else []
    if left:
        raise SystemExit(f"残留 claim {len(left)}: {[p.name for p in left[:8]]}")
    tmps = sorted(out.glob("*.tmp.*"))
    if tmps:
        raise SystemExit(f"残留 tmp {len(tmps)}: {[p.name for p in tmps[:8]]}")
    payload = {"schema": 2, "stage": stage, "grid_stride": wc.GRID_STRIDE, "window_frames": wc.WINDOW_FRAMES,
               "grid_origin": wc.GRID_ORIGIN, "window_direction": wc.WINDOW_DIRECTION,
               "truncation_policy": wc.TRUNCATION_POLICY, "frame_size": wc.FRAME_SIZE,
               "manifest_sha256": manifest["sha256"], "manifest_path": str(manifest_path.resolve()),
               "segments": segs, "workers": list(workers.values()),
               "totals": {"segments": len(segs), "windows": sum(int(s["num_grid"]) for s in segs.values()),
                          "bytes": sum(int(s["bytes"]) for s in segs.values())}}
    wc.write_json(out / "metadata.json", payload)
    return len(segs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["siglip", "wan", "encode"], required=True)
    ap.add_argument("--lib", required=True, help="数据集库根 <lib>（含 meta/episode_manifest.json）")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--raw-dir", default=os.environ.get("RAW_H5_DIR", RAW_H5_DEFAULT))
    ap.add_argument("--require-free-mib", type=int, default=20000)
    ap.add_argument("--encoder-run-dir", default=str(ENCODER_RUN_DIR_DEFAULT))
    ap.add_argument("--checkpoint", default=CKPT_DEFAULT)
    ap.add_argument("--expected-ckpt-sha256", default="")
    args = ap.parse_args()

    # 建库域真正的资产前置：本文件是 Python、不 source paths.sh，那两个 v1_require_* 在本域零调用
    al.require(STAGE_ASSETS[args.stage], level="cheap")

    lib = pathlib.Path(args.lib).resolve()
    if lib.is_symlink():
        raise SystemExit(f"库根是符号链接，拒绝写入: {lib}")
    manifest_path = lib / "meta" / "episode_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"缺清单: {manifest_path}（先跑 scan_manifest.py build）")
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        raise SystemExit("--gpus 为空")
    free = gpu_free_mib()
    for g in gpus:
        if g not in free:
            raise SystemExit(f"GPU {g} 不存在（nvidia-smi 列出 {sorted(free)}）")
        if free[g] < args.require_free_mib:
            raise SystemExit(f"GPU {g} 空闲显存 {free[g]} MiB < 要求 {args.require_free_mib} MiB")
    if args.stage == "encode" and not args.expected_ckpt_sha256:
        # 期望值取自仓库钉死的 ASSETS_LOCK.json。此处**刻意不**现场哈希「即将被使用的那份 ckpt」——
        # 那是自证循环，只能证明 N 个 worker 用同一份字节，挡不住「这份文件本身就是错的」。
        args.expected_ckpt_sha256 = al.expected_sha256("motionjepa_ckpt")
        print(f"[run_local] ckpt sha256={args.expected_ckpt_sha256}（来源 ASSETS_LOCK.json）", flush=True)
    if shutil.which("uv") is None:
        raise SystemExit("缺 uv")

    env = base_env()
    (lib / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    procs = []
    outs = set()
    for k, g in enumerate(gpus):
        cmd, wenv, out = worker_cmd(args.stage, g, k, len(gpus), args, env)
        outs.add(out)
        logp = lib / "logs" / f"{args.stage}-gpu{g}.log"
        logf = open(logp, "a", encoding="utf-8")
        logf.write(f"### {time.strftime('%Y-%m-%dT%H:%M:%S')} cmd={' '.join(cmd)}\n")
        print(f"[run_local] 起 worker gpu{g}: {' '.join(cmd)}", flush=True)
        p = subprocess.Popen(cmd, env=wenv, cwd=str(_REPO_ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        th = threading.Thread(target=pump, args=(f"gpu{g}", p, logf), daemon=True)
        th.start()
        procs.append((g, p, th, logf))
    rc = {}
    for g, p, th, logf in procs:
        p.wait()
        th.join()
        logf.close()
        rc[g] = p.returncode
    elapsed = time.perf_counter() - t0
    bad = {g: c for g, c in rc.items() if c != 0}
    if bad:
        print(f"STAGE_FAIL stage={args.stage} workers={len(gpus)} exit_codes={rc} elapsed={elapsed:.0f}s", flush=True)
        raise SystemExit(1)
    out = next(iter(outs))
    if args.stage in ("wan", "encode"):
        items = aggregate_segments(out, manifest_path, args.stage)
    else:
        left = sorted((out / "_claims").glob("_claim_*")) if (out / "_claims").is_dir() else []
        if left:
            print(f"STAGE_FAIL stage=siglip 残留 claim {len(left)}", flush=True)
            raise SystemExit(1)
        items = len(json.loads(manifest_path.read_text(encoding="utf-8"))["episodes"])
    print(f"STAGE_DONE stage={args.stage} workers={len(gpus)} items={items} elapsed={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
