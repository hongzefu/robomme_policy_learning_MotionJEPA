#!/usr/bin/env python3
"""把恢复出的 grid16 raw Wan latent 导出为 grid32 零参数池化 token（1106 计划第 7、11.3 节）。

输入全部只读：原缓存计划、原导入收据、恢复收据（``wan-cache-import.restored.json``）、恢复 latent 根。
逐段顺序读取 ``<key>.bin`` 的全部窗口并按收据核 SHA，只取偶数 m，新索引 m//2，调用共享
``pool_raw_wan_latent`` 得到 1×1024。保留全部 demo 补尾窗与 exec fill 窗，不按 completed 裁切。

输出根 ``--out``（事先存在即拒绝；写在同级 ``.partial-<pid>`` 目录，全部成功后才改名）：

  <key>.f32.bin             该段 (num_grid, 1, 1024) float32 小端连续字节
  <key>.f32.bin.sha256      sidecar
  <key>.metadata.json       段身份与行合同（pack_motion_store.read_segment_tokens 严格核对）
  window_map.json           schema=wan-timepool-window-map-v1，行序等于 motion 表行序；pack 按字节写入 store
  export_record.json        来源记录与全量摘要；status=PASS 只在全部完成后出现

本脚本不加载 Torch/JAX，也不用 GPU；不写入任何输入文件。
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import pack_motion_store as pack  # noqa: E402
from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.shared import wan_timepool as tp  # noqa: E402

SOURCE_STRIDE = 16
TARGET_STRIDE = ms.TIMEPOOL_LAYOUT_SPEC.grid_stride
GRID_SUBSAMPLE = TARGET_STRIDE // SOURCE_STRIDE
# 真实 orig16 全库的固定计数（1106 计划第 5、10 节）；CLI 默认按它校验，小夹具须显式传入自己的值。
REAL_COUNTS = dict(source_windows=45228, windows=23253, copy=21814, fill=1439, demo_tail=386, exec_fill=1053)
LATENT_ROW_KEYS = ("input_frames_sha256", "latent_sha256")


def _load_json(path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _window_table(receipt: dict, label: str) -> dict:
    """走唯一收据合同；重复键、缺字段、错计数都在这里失败。"""
    try:
        return ms.validate_wan_cache_import_receipt(receipt)
    except ValueError as exc:
        raise ValueError(f"{label} 不符合缓存收据合同: {exc}") from exc


def check_inputs(plan: dict, original: dict, restored: dict, latents_root: pathlib.Path) -> dict:
    """计划、原收据、恢复收据三方按窗口主键核对；任何不符立刻失败。返回恢复侧逐窗表。"""
    body = {k: v for k, v in plan.items() if k != "sha256"}
    if plan.get("sha256") != ms.sha256_bytes(ms.canonical_json_bytes(body)):
        raise ValueError("缓存计划 canonical SHA 与文件内容不符")
    old_table = _window_table(original, "原收据")
    new_table = _window_table(restored, "恢复收据")
    for label, receipt in (("原收据", original), ("恢复收据", restored)):
        if receipt.get("plan_sha256") != plan["sha256"]:
            raise ValueError(f"{label}绑定的缓存计划不同")
    for key in ("manifest_sha256", "manifest_file_sha256", "input_manifest_sha256", "source_h5_files", "vae",
                "preprocess"):
        if ms.canonical_json_bytes(original.get(key)) != ms.canonical_json_bytes(restored.get(key)):
            raise ValueError(f"恢复收据 {key} 与原收据不同")
        if key in plan and ms.canonical_json_bytes(plan[key]) != ms.canonical_json_bytes(restored[key]):
            raise ValueError(f"恢复收据 {key} 与缓存计划不同")
    for key in ("repo", "revision"):
        if not plan.get(key) or not plan[key] == original["published"].get(key) == restored["published"].get(key):
            raise ValueError(f"发布 {key} 在计划与收据之间不同")
    if pathlib.Path(restored["latents"]["root"]).resolve() != latents_root:
        raise ValueError("恢复收据指向的 latent 根与 --latents 不同")
    if ms.sha256_file(latents_root / "metadata.json") != restored["latents"]["metadata_sha256"]:
        raise ValueError("恢复 latent 总 metadata 与恢复收据不符")

    planned = {(row["key"], row["m"]): row for row in plan["windows"]}
    if len(planned) != len(plan["windows"]) or set(planned) != set(old_table) or set(planned) != set(new_table):
        raise ValueError("计划、原收据、恢复收据的窗口主键集合不同")
    for ident, row in planned.items():
        old, new = old_table[ident], new_table[ident]
        for name, want in row.items():
            if old.get(name) != want or new.get(name) != want:
                raise ValueError(f"窗口 {ident} 字段 {name} 与缓存计划不同")
        for name in LATENT_ROW_KEYS:
            if old[name] != new[name]:
                raise ValueError(f"窗口 {ident} {name} 不符: 原收据 {old[name]} != 恢复收据 {new[name]}")
    return new_table


def select_windows(plan: dict, table: dict) -> tuple[list[dict], dict]:
    """按 motion 表行序（episode 序、先 demo 后 exec）排段，核对偶数窗恰好等于 grid32 网格。"""
    segments = sorted(plan["segments"], key=lambda s: (s["g"], ms.SEGMENTS.index(s["segment"])))
    counts = collections.Counter(source_windows=len(table))
    keys = set()
    for item in segments:
        key, ng = item["key"], item["num_grid"]
        if key in keys:
            raise ValueError(f"缓存计划段重复: {key}")
        keys.add(key)
        want_min = 17 if item["segment"] == "demo" else 33
        if (item["min_real"] != want_min or ng != ms.seg_num_grid(item["seg_len"], want_min, SOURCE_STRIDE)
                or key != ms.segment_key(item["h5_file"], item["raw_ep_idx"], item["segment"])):
            raise ValueError(f"{key} 段身份或 grid16 网格数不符")
        target = ms.seg_num_grid(item["seg_len"], want_min, TARGET_STRIDE)
        if target != (ng + 1) // 2:
            raise ValueError(f"{key} 偶数窗数 {(ng + 1) // 2} != grid32 网格数 {target}")
        for m in range(ng):
            row = table.get((key, m))
            if row is None:
                raise ValueError(f"收据缺窗口 {key}:{m}")
            real, pad = row["real_frames"], row["pad_frames"]
            fixed = dict(segment_kind=item["segment"], seg_offset=SOURCE_STRIDE * m,
                         start_global_frame=item["seg_start"] + SOURCE_STRIDE * m,
                         h5_file=item["h5_file"], raw_ep_idx=item["raw_ep_idx"])
            if (any(row.get(k) != v for k, v in fixed.items())
                    or real != min(33, item["seg_len"] - SOURCE_STRIDE * m) or real + pad != 33):
                raise ValueError(f"窗口 {key}:{m} 起点、段类型或补帧不符")
            if m % 2 == 0:
                counts["windows"] += 1
                counts[row["route"]] += 1
                counts["demo_tail"] += int(pad > 0)
                counts["exec_fill"] += int(item["segment"] == "exec" and row["route"] == "fill")
    if sum(item["num_grid"] for item in segments) != len(table):
        raise ValueError("缓存计划段网格数之和与窗口数不同")
    return segments, {k: int(counts[k]) for k in REAL_COUNTS}


def export_segment(item: dict, table: dict, restored_file: dict, latents_root: pathlib.Path, staging: pathlib.Path,
                   first_row: int, table_hash) -> tuple[list[dict], dict]:
    """单遍读整段：奇偶窗都核 SHA，偶数窗池化；返回逐窗映射行与 token 文件记录。"""
    key, ng = item["key"], item["num_grid"]
    path = latents_root / f"{key}.bin"
    if path.stat().st_size != ng * ms.WAN_CHUNK_BYTES or restored_file["bytes"] != ng * ms.WAN_CHUNK_BYTES:
        raise ValueError(f"{key} 恢复 latent 字节数不符")
    whole, token_hash = hashlib.sha256(), hashlib.sha256()
    windows, token_path = [], staging / f"{key}.f32.bin"
    with path.open("rb") as stream, token_path.open("wb") as tokens:
        for m in range(ng):
            blob = stream.read(ms.WAN_CHUNK_BYTES)
            row = table[key, m]
            got = ms.sha256_bytes(blob)
            if got != row["latent_sha256"]:
                raise ValueError(f"窗口 {key}:{m} latent 字节不符: 现场 {got} != 收据 {row['latent_sha256']}")
            whole.update(blob)
            if m % 2:
                continue
            raw = np.frombuffer(blob, dtype="<f4").reshape(tp.TIMEPOOL_INPUT_SHAPE)
            pooled = tp.pool_raw_wan_latent(raw).tobytes()
            tokens.write(pooled)
            token_hash.update(pooled)
            table_hash.update(pooled)
            windows.append(dict(row=first_row + len(windows), key=key, segment=item["segment"], m=m // 2, source_m=m,
                                route=row["route"], start_frame=row["start_global_frame"],
                                real_frames=row["real_frames"], pad_frames=row["pad_frames"],
                                input_frames_sha256=row["input_frames_sha256"], raw_latent_sha256=row["latent_sha256"],
                                pooled_sha256=ms.sha256_bytes(pooled)))
        tokens.flush()
        os.fsync(tokens.fileno())
    if whole.hexdigest() != restored_file["sha256"]:
        raise ValueError(f"{key} 整段 latent SHA 与恢复收据不符")
    digest, count = token_hash.hexdigest(), len(windows)
    (staging / f"{key}.f32.bin.sha256").write_text(f"{digest}  {token_path.name}\n", encoding="utf-8")
    meta = dict(schema=pack.TIMEPOOL_EXPORT_SCHEMA, layout=tp.TIMEPOOL_LAYOUT, readout=tp.TIMEPOOL_READOUT,
                encoder_kind=tp.TIMEPOOL_ENCODER_KIND, segment=key, segment_kind=item["segment"], g=item["g"],
                h5_file=item["h5_file"], raw_ep_idx=item["raw_ep_idx"], seg_start_global=item["seg_start"],
                seg_len=item["seg_len"], row_shape=list(tp.TIMEPOOL_SHAPE), dtype="float32",
                token_dim=1024, token_bytes=tp.TIMEPOOL_OUTPUT_BYTES, bytes=count * tp.TIMEPOOL_OUTPUT_BYTES,
                num_grid=count, grid_stride=TARGET_STRIDE, window_frames=ms.WINDOW_FRAMES,
                source_grid_stride=SOURCE_STRIDE, source_grid=list(range(0, ng, 2)),
                source_latent_sha256=whole.hexdigest(), sha256=digest)
    meta_bytes = ms.canonical_json_bytes(meta)
    (staging / f"{key}.metadata.json").write_bytes(meta_bytes)
    return windows, dict(key=key, num_grid=count, sha256=digest, metadata_sha256=ms.sha256_bytes(meta_bytes))


def run_export(plan_path, receipt_path, restored_receipt_path, latents_root, out, *, expected: dict = REAL_COUNTS) -> dict:
    """执行导出并返回 export_record。``expected`` 必须给全 REAL_COUNTS 的键，任何不符都不产生输出。"""
    plan_path, receipt_path = pathlib.Path(plan_path).resolve(), pathlib.Path(receipt_path).resolve()
    restored_receipt_path = pathlib.Path(restored_receipt_path).resolve()
    latents_root = pathlib.Path(latents_root).resolve()
    out = pathlib.Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {out}")
    if set(expected) != set(REAL_COUNTS):
        raise ValueError(f"expected 必须恰含 {sorted(REAL_COUNTS)}")

    plan, original, restored = _load_json(plan_path), _load_json(receipt_path), _load_json(restored_receipt_path)
    table = check_inputs(plan, original, restored, latents_root)
    segments, counts = select_windows(plan, table)
    if counts != dict(expected):
        raise ValueError(f"窗口计数 {counts} != 期望 {dict(expected)}")
    if plan.get("counts", {}).get("grid32", counts["windows"]) != counts["windows"]:
        raise ValueError("缓存计划 grid32 计数与偶数窗数不同")
    files = {item["path"]: item for item in restored["latents"]["files"]}
    if {p.name for p in latents_root.glob("*.bin")} != {f"{s['key']}.bin" for s in segments}:
        raise ValueError("恢复 latent 根的段文件集合与缓存计划不同")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f"{out.name}.partial-{os.getpid()}")
    staging.mkdir()
    try:
        windows, token_files, table_hash = [], [], hashlib.sha256()
        for item in segments:
            rows, record = export_segment(item, table, files[f"{item['key']}.bin"], latents_root, staging,
                                          len(windows), table_hash)
            windows += rows
            token_files.append(record)
        route_counts = {k: counts[k] for k in ("windows", "copy", "fill")}
        window_map = dict(schema=ms.TIMEPOOL_WINDOW_MAP_SCHEMA, layout=tp.TIMEPOOL_LAYOUT,
                          source_grid_stride=SOURCE_STRIDE, target_grid_stride=TARGET_STRIDE,
                          counts=route_counts, windows=windows)
        map_bytes = ms.canonical_json_bytes(window_map)
        (staging / pack.TIMEPOOL_EXPORT_MAP).write_bytes(map_bytes)
        record = dict(
            schema=pack.TIMEPOOL_EXPORT_SCHEMA, status="PASS", layout=tp.TIMEPOOL_LAYOUT, readout=tp.TIMEPOOL_READOUT,
            encoder_kind=tp.TIMEPOOL_ENCODER_KIND, encoder_ckpt=None, parameter_count=tp.TIMEPOOL_PARAMETER_COUNT,
            output_shape=list(tp.TIMEPOOL_SHAPE), grid_subsample=GRID_SUBSAMPLE, source_grid_stride=SOURCE_STRIDE,
            target_grid_stride=TARGET_STRIDE,
            pool=dict(version=tp.POOL_VERSION, source_sha256=tp.pool_source_sha256(), contract=dict(tp.POOL_CONTRACT)),
            counts={**counts, "segments": len(segments), "bytes": len(windows) * tp.TIMEPOOL_OUTPUT_BYTES},
            inputs=dict(plan=dict(path=str(plan_path), file_sha256=ms.sha256_file(plan_path),
                                  canonical_sha256=plan["sha256"]),
                        import_receipt=dict(path=str(receipt_path), file_sha256=ms.sha256_file(receipt_path)),
                        restored_receipt=dict(path=str(restored_receipt_path),
                                              file_sha256=ms.sha256_file(restored_receipt_path)),
                        restored_latents=dict(root=str(latents_root),
                                              metadata_sha256=restored["latents"]["metadata_sha256"])),
            manifests={k: restored[k] for k in ("manifest_sha256", "manifest_file_sha256", "input_manifest_sha256")},
            source_h5=restored["source_h5_files"],
            published={k: restored["published"][k] for k in ("repo", "revision")},
            numeric_modes=restored["numeric_modes"], vae=restored["vae"], preprocess=restored["preprocess"],
            window_map=dict(relpath=pack.TIMEPOOL_EXPORT_MAP, sha256=ms.sha256_bytes(map_bytes), rows=len(windows)),
            table_sha256=table_hash.hexdigest(), token_files=token_files,
            created_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        (staging / pack.TIMEPOOL_EXPORT_RECORD).write_bytes(ms.canonical_json_bytes(record))
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"输出目录在导出期间出现，拒绝覆盖: {out}")
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return record


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="原 wan-cache-plan.json（只读）")
    ap.add_argument("--receipt", required=True, help="原 wan-cache-import.json（只读）")
    ap.add_argument("--restored-receipt", required=True, help="恢复阶段的 wan-cache-import.restored.json（只读）")
    ap.add_argument("--latents", required=True, help="恢复 latent 根（<key>.bin + metadata.json，只读）")
    ap.add_argument("--out", required=True, help="新输出根，事先存在即拒绝")
    ap.add_argument("--grid-subsample", type=int, required=True, choices=(GRID_SUBSAMPLE,),
                    help="只接受 2：grid16 偶数窗对应 grid32")
    for name, value in REAL_COUNTS.items():
        ap.add_argument(f"--expected-{name.replace('_', '-')}", dest=f"expected_{name}", type=int, default=value,
                        help=f"默认真实全库计数 {value}；只有小夹具才改")
    args = ap.parse_args(argv)
    expected = {name: getattr(args, f"expected_{name}") for name in REAL_COUNTS}
    try:
        record = run_export(args.plan, args.receipt, args.restored_receipt, args.latents, args.out, expected=expected)
    except Exception as exc:
        print(f"EXPORT_WAN_TIMEPOOL=FAIL error={exc}", flush=True)
        raise
    c = record["counts"]
    print(f"EXPORT_WAN_TIMEPOOL=PASS source_windows={c['source_windows']} windows={c['windows']} copy={c['copy']} "
          f"fill={c['fill']} demo_tail={c['demo_tail']} exec_fill={c['exec_fill']} segments={c['segments']} "
          f"bytes={c['bytes']} table_sha256={record['table_sha256']} out={pathlib.Path(args.out).absolute()}", flush=True)


if __name__ == "__main__":
    main()
