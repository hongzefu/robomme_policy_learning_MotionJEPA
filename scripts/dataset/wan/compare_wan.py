#!/usr/bin/env python3
"""D2 / D3 逐位比对：被测 latent / motion 表 vs oracle_driver.py 产出的原版真值。

  latents   被测 ``<lib>/wan-latents/<段>.bin`` vs ``<oracle>/<段>.bin`` 逐窗 f32 原始字节 ``np.array_equal``
            （全覆盖，含每段 exec 尾窗）；frame_mismatches 取自 oracle 的 ``vae_report.json``。
            判定行：``WAN_BITEXACT=PASS compared=<n> frame_mismatches=0 latent_mismatches=0``
  tokens    被测 ``<lib>/motion/motion_token.f32.bin``（经 motion_index 行序）vs ``<oracle>/motion_token.f32.bin``
            逐行 ``np.array_equal``；另比两侧 77 张量 sha256 清单、provenance 白名单键、affine finite、
            ``grep load_wan_latent_stats(`` 零命中（被测脚本）。
            判定行：``ENCODER_BITEXACT=PASS compared=<n> mismatches=0``
            v10 合同：每行 raw8×256 = 2048 个 f32、8,192 B；store 须已 pack（status=packed/verified），
            适配件与上游原件的全文 SHA 各自对 SOURCE_PIN 核，不互相比较；packed 时写
            ``<store>/meta/encoder_bitexact.json``（兼容模式 verify 必需）。
  segtokens 冒烟：逐段 ``motion-tokens`` 文件 vs oracle 表（可带段过滤），判定行 ``ENCODER_SEG_BITEXACT=PASS``。

只依赖 numpy + stdlib（主 venv 或 wan 子 venv 皆可跑）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from decimal import Decimal
import subprocess

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
CHUNK_F32 = 9 * 16 * 32 * 32
# v10 原始八槽合同：每行 8×256 = 2048 个 f32、8,192 B（与 oracle_driver.TOKEN_CONTRACT 同值，独立书写）
TOKEN_SHAPE = (8, 256)
TOKEN_DIM = 8 * 256
TOKEN_BYTES = TOKEN_DIM * 4
TOKEN_CONTRACT = {"upstream_shape": [TOKEN_DIM], "row_shape": list(TOKEN_SHAPE), "row_elems": TOKEN_DIM,
                  "row_bytes": TOKEN_BYTES, "dtype": "float32", "order": "row_major"}
# module_sha256 不在两侧相等清单里：被测侧是 policy 适配件、oracle 侧是上游原件，二者分别钉 SOURCE_PIN（D6）。
PROV_SAME = ("torch", "cuda", "cudnn", "diffusers", "gpu_name", "compute_cap", "driver",
             "encoder_src_sha256", "env", "flags")
BITEXACT_RELPATH = "meta/encoder_bitexact.json"


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def expected_samples(meta: dict, sample_spec: str) -> tuple[set, set]:
    """比较器独立重算抽样与补帧集合；不导入 oracle 的抽样函数。"""
    match = re.fullmatch(r"padded:all,rest:(0(?:\.\d{1,4})?|1(?:\.0{1,4})?),seed:(\d+)", sample_spec)
    if match is None:
        raise SystemExit(f"比较器抽样规格不合法: {sample_spec!r}")
    limit = int(Decimal(match.group(1)) * 10000)
    seed = int(match.group(2))
    selected, padded = set(), set()
    for key, segment in meta["segments"].items():
        is_demo = segment["segment_kind"] == "demo"
        length = int(segment["seg_len"])
        minimum = int(meta.get("demo_min_real_frames", 33)) if is_demo else 33
        starts = list(range(0, max(0, length - minimum + 1), 16))
        if len(starts) != segment["num_grid"]:
            raise SystemExit(f"比较器独立网格计数不符: {key}")
        for m, start in enumerate(starts):
            row = (key, m)
            pad = is_demo and start + 33 > length
            if pad:
                padded.add(row)
            digest = hashlib.sha256(f"{seed}:{key}:{m}".encode()).digest()
            if pad or int.from_bytes(digest[:8], byteorder="big") % 10000 < limit:
                selected.add(row)
    return selected, padded


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
    if bool(args.sampled) != bool(args.sample_spec):
        raise SystemExit("--sampled 与 --sample-spec 必须一起提供")
    padded = set()
    if args.sampled:
        chosen, padded = expected_samples(meta, args.sample_spec)
        sample = json.loads(pathlib.Path(args.sampled).read_text())
        actual = {(r["segment"], int(r["m"])) for r in sample["windows"]}
        if (sample.get("sample_spec") != args.sample_spec or rep.get("sample_spec") != args.sample_spec
                or actual != chosen or len(actual) != len(sample["windows"])):
            raise SystemExit("抽样表与比较器独立重算集合不同，或抽样规格不一致")
        if len(padded) != args.expect_padded or not padded <= chosen:
            raise SystemExit(f"补帧集合覆盖不符: {len(padded)} != {args.expect_padded}")
        if rep.get("metadata_rows_checked") != sum(int(meta["segments"][k]["num_grid"]) for k in segs):
            raise SystemExit("oracle 元数据检查未覆盖全部窗口")
    else:
        chosen = {(key, m) for key in segs for m in range(int(meta["segments"][key]["num_grid"]))}
    covered_padded = set()
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
            if (key, m) not in chosen:
                continue
            compared += 1
            if (key, m) in padded:
                covered_padded.add((key, m))
            if not np.array_equal(a[m].view(np.uint32), b[m].view(np.uint32)):
                bad.append((key, m, float(np.abs(a[m].astype(np.float64) - b[m]).max())))
    fm = int(rep["frame_mismatches"])
    mm = int(rep.get("metadata_mismatches", 0))
    for key, m, d in bad[:20]:
        print(f"  ✗ {key} m={m} max|Δ|={d}")
    ok = (not bad) and fm == 0 and mm == 0 and compared == int(rep["windows"]) == len(chosen) and covered_padded == padded
    print(f"WAN_BITEXACT={'PASS' if ok else 'FAIL'} compared={compared} padded_covered={len(covered_padded)} frame_mismatches={fm} "
          f"latent_mismatches={len(bad)} metadata_mismatches={mm} oracle_windows={rep['windows']}")
    if not ok:
        raise SystemExit(1)


def source_checks(store_module_sha: str | None, rep: dict) -> list[str]:
    """被测适配件 / oracle 上游原件各自钉 SOURCE_PIN，不互相比较全文 SHA。返回失败项。"""
    pin = json.loads((_HERE / "SOURCE_PIN.json").read_text(encoding="utf-8"))
    bad = []
    if store_module_sha != pin["source_sha256"]:
        bad.append("store.module_sha256!=pin.source_sha256")
    if rep.get("orig_module_sha256") != pin.get("upstream_source_sha256"):
        bad.append("oracle.orig_module_sha256!=pin.upstream_source_sha256")
    if (rep.get("encoder_provenance") or {}).get("module_sha256") != pin.get("upstream_source_sha256"):
        bad.append("oracle.provenance.module_sha256!=pin.upstream_source_sha256")
    ident = rep.get("source_identity") or {}
    if (ident.get("upstream_module_sha256"), ident.get("adapter_module_sha256"), ident.get("mj_install_commit")) != (
            pin.get("upstream_source_sha256"), pin["source_sha256"], pin["mj_repo_commit"]):
        bad.append("oracle.source_identity!=pin")
    return bad


def cmd_tokens(args):
    store = pathlib.Path(args.store)
    orc = pathlib.Path(args.oracle)
    meta_path = store / "meta" / "store_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"✗ 被测 store 尚未 pack（缺 {meta_path}）；compare tokens 必须在 pack 之后运行")
    smeta = json.loads(meta_path.read_text(encoding="utf-8"))
    if smeta.get("status") not in ("packed", "verified"):
        raise SystemExit(f"✗ store status={smeta.get('status')!r}，compare tokens 只接受 packed / verified")
    table = smeta["tables"]["motion_token"]
    if (table.get("row_shape"), table.get("row_bytes")) != (list(TOKEN_SHAPE), TOKEN_BYTES):
        raise SystemExit(f"✗ store 行合同 {table.get('row_shape')} / {table.get('row_bytes')} B 不是 raw8×256 / {TOKEN_BYTES} B")
    index = json.loads((store / "meta" / "motion_index.json").read_text(encoding="utf-8"))
    rep = json.loads((orc / "encoder_report.json").read_text(encoding="utf-8"))
    if rep.get("token_contract") != TOKEN_CONTRACT:
        raise SystemExit(f"✗ oracle token 合同 {rep.get('token_contract')} 不是 {TOKEN_CONTRACT}")
    if rep.get("segments_filter"):
        raise SystemExit("✗ oracle 报告带段过滤（冒烟），不能与全表比较")
    n = int(smeta["num_rows"])
    if sha_file(store / "motion_token.f32.bin") != table["sha256"]:
        raise SystemExit("✗ store 表文件 SHA 与 store_meta 记录不符")
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
    prov_diff = [k for k in PROV_SAME + ("checkpoint_sha256", "checkpoint_epoch", "precision", "amp", "tf32", "batch",
                                         "arch", "state_key", "motion_dims")
                 if enc_a.get(k) != enc_b.get(k) and not (k == "flags" and enc_a.get("flags") is None)]
    prov_diff += source_checks(enc_a.get("module_sha256"), rep)
    for k in prov_diff:
        print(f"✗ provenance 不等 {k}: {enc_a.get(k, '')!r} != {enc_b.get(k, '')!r}")
    ckpt_ok = enc_a.get("checkpoint_sha256") == rep["encoder_provenance"]["checkpoint_sha256"]
    grep = subprocess.run(["grep", "-rn", "load_wan_latent_stats(", str(_HERE / "encode_motion.py"),
                           str(_HERE / "extract_wan.py"), str(_HERE.parent / "pack_motion_store.py")],
                          capture_output=True, text=True)
    grep_ok = grep.returncode != 0
    if not grep_ok:
        print(f"✗ 被测脚本调用了 load_wan_latent_stats:\n{grep.stdout}")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    ok = (not bad) and order_ok and sha_ok and not prov_diff and ckpt_ok and grep_ok and finite
    line = (f"ENCODER_BITEXACT={'PASS' if ok else 'FAIL'} compared={n} mismatches={len(bad)} order_ok={int(order_ok)} "
            f"state_sha_ok={int(sha_ok)} prov_ok={int(not prov_diff)} ckpt_ok={int(ckpt_ok)} "
            f"no_latent_stats_call={int(grep_ok)} finite={int(finite)} row_bytes={TOKEN_BYTES}")
    print(line)
    if not ok:
        raise SystemExit(1)
    if smeta["status"] == "packed":
        # 只在 packed 阶段写记录，供兼容模式 verify 核对（verified 库不改写）。
        rec = {"status": "PASS", "compared": n, "mismatches": 0, "store_table_sha256": table["sha256"],
               "motion_index_sha256": smeta["motion_index_sha256"], "oracle_table_sha256": rep["table_sha256"],
               "oracle_report_sha256": sha_file(orc / "encoder_report.json"),
               "checkpoint_sha256": enc_a.get("checkpoint_sha256"), "token_contract": TOKEN_CONTRACT,
               "line": line}
        tmp = store / (BITEXACT_RELPATH + ".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(store / BITEXACT_RELPATH)


def cmd_segtokens(args):
    """冒烟：逐段 motion-tokens 文件 vs oracle（可带段过滤）逐行逐位；不经 pack，不产出可发布记录。"""
    tok = pathlib.Path(args.tokens)
    orc = pathlib.Path(args.oracle)
    rep = json.loads((orc / "encoder_report.json").read_text(encoding="utf-8"))
    if rep.get("token_contract") != TOKEN_CONTRACT:
        raise SystemExit(f"✗ oracle token 合同 {rep.get('token_contract')} 不是 {TOKEN_CONTRACT}")
    b = np.fromfile(orc / "motion_token.f32.bin", dtype=np.float32)
    n = int(rep["rows"])
    if b.size != n * TOKEN_DIM or len(rep["row_map"]) != n or sha_file(orc / "motion_token.f32.bin") != rep["table_sha256"]:
        raise SystemExit("✗ oracle 表行数/SHA 与报告不符")
    b = b.reshape(n, TOKEN_DIM)
    by_seg: dict[str, list] = {}
    for r in rep["row_map"]:
        by_seg.setdefault(r["segment"], []).append(int(r["m"]))
    bad, compared, prov_bad = [], 0, []
    row = 0
    for key, ms_ in by_seg.items():
        if ms_ != list(range(len(ms_))):
            raise SystemExit(f"✗ oracle 段 {key} 行序不是 0..num_grid-1")
        meta = json.loads((tok / f"{key}.metadata.json").read_text(encoding="utf-8"))
        data = (tok / f"{key}.f32.bin").read_bytes()
        side = (tok / f"{key}.f32.bin.sha256").read_text().split()[0]
        if (meta.get("row_shape") != list(TOKEN_SHAPE) or meta.get("token_bytes") != TOKEN_BYTES
                or len(data) != len(ms_) * TOKEN_BYTES or hashlib.sha256(data).hexdigest() != side):
            raise SystemExit(f"✗ {key} token 文件合同/字节/SHA 不符")
        a = np.frombuffer(data, dtype=np.float32).reshape(len(ms_), TOKEN_DIM)
        for m in range(len(ms_)):
            compared += 1
            if not np.array_equal(a[m].view(np.uint32), b[row].view(np.uint32)):
                bad.append((key, m))
            row += 1
        enc_a, enc_b = meta["encoder"], rep["encoder_provenance"]
        prov_bad += [f"{key}:{k}" for k in ("checkpoint_sha256", "checkpoint_epoch", "arch", "state_key",
                                            "motion_dims", "amp", "tf32", "batch", "encoder_src_sha256")
                     if enc_a.get(k) != enc_b.get(k)]
        prov_bad += [f"{key}:{x}" for x in source_checks(enc_a.get("module_sha256"), rep)]
        if meta.get("encoder_state_sha256") != rep["encoder_state_sha256"]:
            prov_bad.append(f"{key}:encoder_state_sha256")
    for item in (bad[:20] + prov_bad[:20]):
        print(f"  ✗ {item}")
    ok = not bad and not prov_bad and compared == n
    print(f"ENCODER_SEG_BITEXACT={'PASS' if ok else 'FAIL'} segments={len(by_seg)} compared={compared} "
          f"mismatches={len(bad)} prov_ok={int(not prov_bad)} row_bytes={TOKEN_BYTES}")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("latents")
    p.add_argument("--latents", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--sampled", default=None)
    p.add_argument("--sample-spec", default=None)
    p.add_argument("--expect-padded", type=int, default=1600)
    p.set_defaults(func=cmd_latents)
    p = sub.add_parser("tokens")
    p.add_argument("--store", required=True)
    p.add_argument("--oracle", required=True)
    p.set_defaults(func=cmd_tokens)
    p = sub.add_parser("segtokens", help="冒烟：逐段 token 文件 vs oracle（不经 pack）")
    p.add_argument("--tokens", required=True)
    p.add_argument("--oracle", required=True)
    p.set_defaults(func=cmd_segtokens)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
