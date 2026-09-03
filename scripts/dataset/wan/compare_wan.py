#!/usr/bin/env python3
"""D2 / D3 逐位比对：被测 latent / motion 表 vs oracle_driver.py 产出的原版真值。

  latents   被测 ``<lib>/wan-latents/<段>.bin`` vs ``<oracle>/<段>.bin`` 逐窗 f32 原始字节 ``np.array_equal``
            （全覆盖，含每段 exec 尾窗）；frame_mismatches 取自 oracle 的 ``vae_report.json``。
            判定行：``WAN_BITEXACT=PASS compared=<n> frame_mismatches=0 latent_mismatches=0``
  tokens    被测 ``<lib>/motion/motion_token.f32.bin``（经 motion_index 行序）vs ``<oracle>/motion_token.f32.bin``
            逐行 ``np.array_equal``；另比两侧 77 张量 sha256 清单、provenance 白名单键、affine finite、
            ``grep load_wan_latent_stats(`` 零命中（被测脚本）。
            判定行：``ENCODER_BITEXACT=PASS compared=<n> mismatches=0``

只依赖 numpy + stdlib（主 venv 或 wan 子 venv 皆可跑）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
CHUNK_F32 = 9 * 16 * 32 * 32
TOKEN_DIM = 768
PROV_SAME = ("torch", "cuda", "cudnn", "diffusers", "gpu_name", "compute_cap", "driver", "module_sha256",
             "encoder_src_sha256", "env", "flags")


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def cmd_latents(args):
    lat = pathlib.Path(args.latents)
    orc = pathlib.Path(args.oracle)
    meta = json.loads((lat / "metadata.json").read_text(encoding="utf-8"))
    rep = json.loads((orc / "vae_report.json").read_text(encoding="utf-8"))
    if rep.get("tested_metadata_sha256") != sha_file(lat / "metadata.json"):
        print("✗ oracle 核对的 metadata.json 与当前不是同一份字节")
        raise SystemExit(1)
    segs = sorted(meta["segments"])
    if set(segs) != set(rep["segments"]):
        print(f"✗ 段集合不符: {len(segs)} vs {len(rep['segments'])}")
        raise SystemExit(1)
    compared, bad = 0, []
    for key in segs:
        ng = int(meta["segments"][key]["num_grid"])
        a = np.fromfile(lat / f"{key}.bin", dtype=np.float32)
        b = np.fromfile(orc / f"{key}.bin", dtype=np.float32)
        if a.size != ng * CHUNK_F32 or b.size != ng * CHUNK_F32:
            bad.append((key, -1, "size"))
            continue
        a = a.reshape(ng, CHUNK_F32)
        b = b.reshape(ng, CHUNK_F32)
        for m in range(ng):
            compared += 1
            if not np.array_equal(a[m].view(np.uint32), b[m].view(np.uint32)):
                bad.append((key, m, float(np.abs(a[m].astype(np.float64) - b[m]).max())))
    fm = int(rep["frame_mismatches"])
    mm = int(rep.get("metadata_mismatches", 0))
    for key, m, d in bad[:20]:
        print(f"  ✗ {key} m={m} max|Δ|={d}")
    ok = (not bad) and fm == 0 and mm == 0 and compared == int(rep["windows"])
    print(f"WAN_BITEXACT={'PASS' if ok else 'FAIL'} compared={compared} frame_mismatches={fm} "
          f"latent_mismatches={len(bad)} metadata_mismatches={mm} oracle_windows={rep['windows']}")
    if not ok:
        raise SystemExit(1)


def cmd_tokens(args):
    store = pathlib.Path(args.store)
    orc = pathlib.Path(args.oracle)
    smeta = json.loads((store / "meta" / "store_meta.json").read_text(encoding="utf-8"))
    index = json.loads((store / "meta" / "motion_index.json").read_text(encoding="utf-8"))
    rep = json.loads((orc / "encoder_report.json").read_text(encoding="utf-8"))
    n = int(smeta["num_rows"])
    a = np.fromfile(store / "motion_token.f32.bin", dtype=np.float32)
    b = np.fromfile(orc / "motion_token.f32.bin", dtype=np.float32)
    if a.size != n * TOKEN_DIM or b.size != n * TOKEN_DIM or int(rep["rows"]) != n:
        print(f"✗ 行数不符: store {a.size // TOKEN_DIM} oracle {b.size // TOKEN_DIM} report {rep['rows']} meta {n}")
        raise SystemExit(1)
    a = a.reshape(n, TOKEN_DIM)
    b = b.reshape(n, TOKEN_DIM)
    # oracle row_map 的 (segment, m) 序必须与被测 index 行序一致
    exp = []
    for e in index["entries"]:
        task = e["h5_file"][len("record_dataset_"):-3]
        for seg in ("demo", "exec"):
            for m in range(int(e[seg]["num_grid"])):
                exp.append((f"{task}_ep{e['raw_ep_idx']}_{seg}", m))
    got = [(r["segment"], int(r["m"])) for r in rep["row_map"]]
    order_ok = exp == got
    if not order_ok:
        print("✗ oracle 行序与 motion_index 行序不同")
    bad = [i for i in range(n) if not np.array_equal(a[i].view(np.uint32), b[i].view(np.uint32))]
    for i in bad[:20]:
        print(f"  ✗ row={i} {exp[i] if i < len(exp) else '?'} max|Δ|={float(np.abs(a[i].astype(np.float64) - b[i]).max())}")
    # 附加：77 张量清单、provenance 白名单、affine finite、禁调 load_wan_latent_stats
    prov = smeta["provenance"]
    sha_ok = prov["encoder_state_sha256"] == rep["encoder_state_sha256"]
    if not sha_ok:
        print(f"✗ encoder_state_sha256 清单不同（{len(prov['encoder_state_sha256'])} vs {len(rep['encoder_state_sha256'])} 项）")
    enc_a, enc_b = prov["encoder"], rep["encoder_provenance"]
    prov_diff = [k for k in PROV_SAME + ("checkpoint_sha256", "precision", "amp", "tf32", "batch", "arch")
                 if enc_a.get(k) != enc_b.get(k) and not (k == "flags" and enc_a.get("flags") is None)]
    for k in prov_diff:
        print(f"✗ provenance 不等 {k}: {enc_a.get(k)!r} != {enc_b.get(k)!r}")
    ckpt_ok = enc_a.get("checkpoint_sha256") == rep["encoder_provenance"]["checkpoint_sha256"]
    grep = subprocess.run(["grep", "-rn", "load_wan_latent_stats(", str(_HERE / "encode_motion.py"),
                           str(_HERE / "extract_wan.py"), str(_HERE.parent / "pack_motion_store.py")],
                          capture_output=True, text=True)
    grep_ok = grep.returncode != 0
    if not grep_ok:
        print(f"✗ 被测脚本调用了 load_wan_latent_stats:\n{grep.stdout}")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    ok = (not bad) and order_ok and sha_ok and not prov_diff and ckpt_ok and grep_ok and finite
    print(f"ENCODER_BITEXACT={'PASS' if ok else 'FAIL'} compared={n} mismatches={len(bad)} order_ok={int(order_ok)} "
          f"state_sha_ok={int(sha_ok)} prov_ok={int(not prov_diff)} ckpt_ok={int(ckpt_ok)} "
          f"no_latent_stats_call={int(grep_ok)} finite={int(finite)}")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("latents")
    p.add_argument("--latents", required=True)
    p.add_argument("--oracle", required=True)
    p.set_defaults(func=cmd_latents)
    p = sub.add_parser("tokens")
    p.add_argument("--store", required=True)
    p.add_argument("--oracle", required=True)
    p.set_defaults(func=cmd_tokens)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
