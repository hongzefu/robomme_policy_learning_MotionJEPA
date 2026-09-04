#!/usr/bin/env python3
"""S1 附加检查（wan 子 venv、需 GPU；motion-memory-plan.md 第二部分四节表二 A8 / A9（编码半） / A12）。

  a8    抽表逐位：随机 128 个 (段, 网格序号)，在线用复制件 ``motion_token``（B=1）对 ``wan-latents/<段>.bin`` 的该块编码，
        与 motion 表（按 motion_index 行序）对应行 ``np.array_equal``。判定行 ``A8_TABLE_BITEXACT=PASS sampled=128 mismatches=0``。
  a9enc 索引映射（编码半）：随机 500 个 (g, t) 的可见集合里每行，``row_base + m`` 读出的表行 == 直读该窗 latent 过 encoder。
        判定行 ``A9_ROWENC=PASS rows=<n> mismatches=0``。
  a12   v7 latent 旁证（非阻断）：与 ``/data/hongzefu/dataset-4env-v7/dataset-token/wan_chunk_latents/<段>.bin`` 同窗逐位；
        v7 是 stride 1 dense、chunk 索引 = 段内帧偏移，我方第 m 块对应 v7 第 16m 块；只比 ``16m < v7 num_chunks`` 的窗。
        判定行 ``V7_CROSSREF=PASS compared=<n> skipped=<n> mismatches=<n>``（FAIL 只作提示）。

不 import mme_vla_suite（子 venv 装不下），motion_index.json 直接读 JSON。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import random
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "assets"))
import assets_lock as al  # noqa: E402
import wan_common as wc  # noqa: E402

ENCODER_RUN_DIR_DEFAULT = str(_REPO_ROOT / "v1-store" / "external" / "motionjepa" / "wan-v8-filter10-72ep-a")

def _expected_ckpt(args):
    """ckpt 期望值：默认取 ASSETS_LOCK.json 钉死的那份，显式传 SKIP 才跳过。

    本文件是**探针 / 对拍工具**，探一个未入 lock 的 run_dir 是它的本职，所以保留 SKIP 出口；
    生产路径（encode_motion.py 的 required=True、run_local.py、motion_sidecar.py）不提供 SKIP。
    """
    if args.expected_ckpt_sha256 == "SKIP":
        return None
    return args.expected_ckpt_sha256 or al.expected_sha256("motionjepa_ckpt")



def load_infer_module():
    spec = importlib.util.spec_from_file_location("wan_motion_infer", str(_HERE / "wan_motion_infer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_index(motion_root: pathlib.Path) -> tuple[dict, np.ndarray]:
    index = json.loads((motion_root / "meta" / "motion_index.json").read_text(encoding="utf-8"))
    table = np.fromfile(motion_root / "motion_token.f32.bin", dtype=np.float32)
    table.shape = (int(index["totals"]["rows"]), wc.TOKEN_DIM)
    return index, table


def seg_key(e: dict, seg: str) -> str:
    return f"{wc.task_of_h5(e['h5_file'])}_ep{e['raw_ep_idx']}_{seg}"


def all_rows(index: dict) -> list[tuple[int, str, int]]:
    out = []
    for e in index["entries"]:
        for seg in ("demo", "exec"):
            s = e[seg]
            for m in range(int(s["num_grid"])):
                out.append((int(s["row_base"]) + m, seg_key(e, seg), m))
    return out


def setup_encoder(args):
    import torch
    W = load_infer_module()
    wc.load_source_pin(_HERE)
    W.check_env()
    W.pin_numerics()
    W.check_versions(strict=True, with_diffusers=False)
    torch.manual_seed(0)
    device = torch.device("cuda")
    encoder, einfo, use_amp = W.load_encoder(args.encoder_run_dir, args.checkpoint, device,
                                             expected_sha256=_expected_ckpt(args))
    return W, torch, device, encoder, use_amp, einfo


def encode_row(W, torch, device, encoder, use_amp, lat_root: pathlib.Path, key: str, m: int) -> np.ndarray:
    blk = W.read_latent_block(str(lat_root / f"{key}.bin"), m)
    W.pin_numerics()
    return W.motion_token(encoder, blk, use_amp, device).astype(np.float32)


def cmd_a8(args):
    W, torch, device, encoder, use_amp, einfo = setup_encoder(args)
    index, table = load_index(pathlib.Path(args.motion))
    rows = all_rows(index)
    rng = random.Random(args.seed)
    picks = rng.sample(rows, min(args.n, len(rows)))
    bad = 0
    for row, key, m in picks:
        tok = encode_row(W, torch, device, encoder, use_amp, pathlib.Path(args.latents), key, m)
        if not np.array_equal(tok.view(np.uint32), table[row].view(np.uint32)):
            bad += 1
            print(f"  ✗ row={row} {key} m={m} max|Δ|={float(np.abs(tok.astype(np.float64) - table[row]).max())}")
    ok = bad == 0
    print(f"A8_TABLE_BITEXACT={'PASS' if ok else 'FAIL'} sampled={len(picks)} mismatches={bad} rows_total={len(rows)} "
          f"ckpt={einfo['checkpoint_sha256'][:16]}…")
    if not ok:
        raise SystemExit(1)


def visible(e: dict, t: int) -> list[tuple[int, str, int, int]]:
    """(row, key, m, f) 按 f 升序——与 motion_store.visible_motion_rows 同式（独立书写）。"""
    es = int(e["exec_start_idx"])
    out = []
    d, x = e["demo"], e["exec"]
    for m in range(int(d["num_grid"])):
        s = 16 * m
        if s + 32 <= es - 1:
            out.append((int(d["row_base"]) + m, seg_key(e, "demo"), m, s))
    for m in range(int(x["num_grid"])):
        u = 16 * m
        if u + 32 <= t - es:
            out.append((int(x["row_base"]) + m, seg_key(e, "exec"), m, es + u))
    return sorted(out, key=lambda r: r[3])


def cmd_a9enc(args):
    W, torch, device, encoder, use_amp, _ = setup_encoder(args)
    index, table = load_index(pathlib.Path(args.motion))
    rng = random.Random(args.seed)
    cache: dict[tuple[str, int], np.ndarray] = {}
    checked = bad = samples = 0
    for _ in range(args.n):
        e = rng.choice(index["entries"])
        t = rng.randrange(int(e["exec_start_idx"]), int(e["num_timesteps"]))
        samples += 1
        for row, key, m, _f in visible(e, t):
            if (key, m) not in cache:
                cache[(key, m)] = encode_row(W, torch, device, encoder, use_amp, pathlib.Path(args.latents), key, m)
            checked += 1
            if not np.array_equal(cache[(key, m)].view(np.uint32), table[row].view(np.uint32)):
                bad += 1
    ok = bad == 0
    print(f"A9_ROWENC={'PASS' if ok else 'FAIL'} samples={samples} rows={checked} unique_windows={len(cache)} mismatches={bad}")
    if not ok:
        raise SystemExit(1)


def cmd_a12(args):
    index, _ = load_index(pathlib.Path(args.motion))
    v7 = pathlib.Path(args.v7_latents)
    v7meta = json.loads((v7 / "metadata.json").read_text(encoding="utf-8"))
    compared = skipped = bad = missing = 0
    for e in index["entries"]:
        for seg in ("demo", "exec"):
            s = e[seg]
            ng = int(s["num_grid"])
            if ng == 0:
                continue
            key = seg_key(e, seg)
            ours = np.fromfile(pathlib.Path(args.latents) / f"{key}.bin", dtype=np.float32).reshape(ng, -1)
            vm = v7meta.get(f"{key}.bin") or v7meta.get(key)
            if vm is None or not (v7 / f"{key}.bin").is_file():
                missing += 1
                continue
            n_v7 = int(vm["num_chunks"])
            theirs = np.fromfile(v7 / f"{key}.bin", dtype=np.float32).reshape(n_v7, -1)
            for m in range(ng):
                if 16 * m >= n_v7:
                    skipped += 1
                    continue
                compared += 1
                if not np.array_equal(ours[m].view(np.uint32), theirs[16 * m].view(np.uint32)):
                    bad += 1
    ok = bad == 0
    print(f"V7_CROSSREF={'PASS' if ok else 'FAIL'} compared={compared} skipped={skipped} mismatches={bad} "
          f"missing_segments={missing}（非阻断旁证；v7 无 provenance）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder-run-dir", default=ENCODER_RUN_DIR_DEFAULT)
    ap.add_argument("--checkpoint", default="checkpoint_epoch_72.pt")
    ap.add_argument("--expected-ckpt-sha256", default="",
                    help="默认按 ASSETS_LOCK.json 校；探未入 lock 的 run_dir 时显式传 SKIP")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("a8", cmd_a8), ("a9enc", cmd_a9enc)):
        p = sub.add_parser(name)
        p.add_argument("--motion", required=True)
        p.add_argument("--latents", required=True)
        p.add_argument("--n", type=int, default=128 if name == "a8" else 500)
        p.add_argument("--seed", type=int, default=20260903)
        p.set_defaults(func=fn)
    p = sub.add_parser("a12")
    p.add_argument("--motion", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--v7-latents", default="/data/hongzefu/dataset-4env-v7/dataset-token/wan_chunk_latents")
    p.set_defaults(func=cmd_a12)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
