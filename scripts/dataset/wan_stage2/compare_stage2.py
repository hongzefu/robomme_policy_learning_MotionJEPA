#!/usr/bin/env python3
"""G5a：从恢复 latent 独立重算整表，全部逐位一致才原子签发认证。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_stage2 as V

SCHEMA = "stage2-encoder-bitexact-v1"


def compare_rows(*, table, segments, encode, latent_root, root_metadata):
    """遍历全部索引行；encode 接收一个原始 latent，而非已抽取 token 文件。"""
    compared, mapping, file_hashes = 0, {}, {}
    for key, row_base, count in segments:
        if row_base != compared or count <= 0:
            raise ValueError("全表段索引必须连续覆盖且非空")
        path = Path(latent_root) / f"{key}.bin"
        meta_path = path.with_suffix(".metadata.json")
        metadata = V.read_json(meta_path)
        old_count = metadata.get("num_grid")
        if (type(old_count) is not int or old_count <= 0 or (old_count + 1) // 2 != count
                or metadata.get("bytes") != old_count * V.CHUNK_BYTES
                or path.stat().st_size != old_count * V.CHUNK_BYTES):
            raise ValueError(f"{key} 源网格/字节数不符合偶数格选择合同")
        digest = V.sha_file(path)
        if digest != metadata.get("sha256") or digest != root_metadata["segments"][key]["sha256"]:
            raise ValueError(f"{key} 源 latent 全文件 SHA 不符")
        file_hashes[str(path)] = digest
        file_hashes[str(meta_path)] = V.sha_file(meta_path)
        source_grid = list(range(0, old_count, 2))
        mapping[key] = source_grid
        with path.open("rb") as stream:
            for index, grid in enumerate(source_grid):
                stream.seek(grid * V.CHUNK_BYTES)
                block = stream.read(V.CHUNK_BYTES)
                if len(block) != V.CHUNK_BYTES:
                    raise ValueError("源 latent 块被截断")
                latent = np.frombuffer(block, dtype="<f4").reshape(V.LATENT_SHAPE).copy()
                if not np.isfinite(latent).all():
                    raise ValueError("源 latent 含非有限数")
                output = np.asarray(encode(latent))
                stored = np.asarray(table[row_base + index])
                if (output.dtype != np.float32 or output.shape != (1024,) or stored.shape != (1024,)
                        or not np.isfinite(output).all() or not np.isfinite(stored).all()
                        or output.astype("<f4", copy=False).tobytes() != stored.astype("<f4", copy=False).tobytes()):
                    raise ValueError(f"encoder 逐位不同：{key} 源格{grid} 表行{row_base + index}")
                compared += 1
                if compared % 1024 == 0:
                    print(f"STAGE2_COMPARE_PROGRESS compared={compared} total={len(table)}", flush=True)
    if compared != len(table):
        raise ValueError("比较范围未覆盖整个 packed 表")
    if set(root_metadata["segments"]) != set(mapping):
        raise ValueError("源 latent 段集合与 packed 索引不同")
    # 防止比较期间源文件变化，不能仅在开头核验一次身份。
    for path, digest in file_hashes.items():
        if V.sha_file(path) != digest:
            raise ValueError(f"比较期间源文件改变：{path}")
    return compared, mapping


def certify(args):
    from mme_vla_suite.datastore import motion_store as ms
    root = Path(args.store)
    record_path = root / "meta/encoder_bitexact.json"
    lock_path = root / "meta/encoder_bitexact.lock"
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        V.revoke_report(record_path, schema=SCHEMA)
        if args.scope != "full" or args.batch != 1:
            raise ValueError("全表认证强制 scope=full、batch=1")
        meta = ms.MotionMeta.load(root)
        if meta.raw["layout"] != V.LAYOUT or meta.num_rows != 65419:
            raise ValueError("正式 G5a 必须覆盖 stage2 的65,419行")
        ms.run_fast_checks(meta)
        ms.run_full_checks(meta)
        bound_paths = [root / ms.META_RELPATH, root / ms.INDEX_RELPATH,
                       root / ms.MOTION_TABLE_RELPATH, Path(args.ckpt), Path(args.latents) / "metadata.json"]
        hashes = {str(path): V.sha_file(path) for path in bound_paths}
        if hashes[str(Path(args.ckpt))] != args.expected_sha256:
            raise ValueError("checkpoint SHA 不符")
        if hashes[str(Path(args.latents) / "metadata.json")] != meta.provenance["latents_metadata_sha256"]:
            raise ValueError("源 latent 总 metadata SHA 与 store 不同")
        torch, memory, _, encoder, info = V.setup_runtime(args)
        V.assert_encoder_identity(info, meta.provenance)
        segments = []
        for entry in meta.entries:
            for kind in ("demo", "exec"):
                part = getattr(entry, kind)
                if part.num_grid:
                    segments.append((ms.segment_key(entry.h5_file, entry.raw_ep_idx, kind), part.row_base, part.num_grid))
        table = np.memmap(root / ms.MOTION_TABLE_RELPATH, mode="r", dtype="<f4", shape=(meta.num_rows, 1024))

        def encode(latent):
            value = torch.from_numpy(latent).unsqueeze(0).to(args.device)
            return memory.encode_windows(encoder, value)[0].cpu().numpy()

        compared, mapping = compare_rows(table=table, segments=segments, encode=encode, latent_root=args.latents,
                                         root_metadata=V.read_json(Path(args.latents) / "metadata.json"))
        source_sha = V.sha_bytes(V.canonical(mapping))
        if source_sha != meta.provenance["source_grid_sha256"]:
            raise ValueError("完整偶数格映射与 store 来源不同")
        for path, digest in hashes.items():
            if V.sha_file(path) != digest:
                raise ValueError(f"比较期间绑定文件改变：{path}")
        record = dict(schema=SCHEMA, status="PASS", producer=V.PRODUCER, scope="full", layout=V.LAYOUT,
                      compared=compared, mismatches=0, source_grid_sha256=source_sha,
                      store_table_sha256=hashes[str(root / ms.MOTION_TABLE_RELPATH)],
                      motion_index_sha256=hashes[str(root / ms.INDEX_RELPATH)],
                      checkpoint_sha256=args.expected_sha256, segments=len(mapping),
                      source_metadata_sha256=hashes[str(Path(args.latents) / "metadata.json")],
                      encoder_state_sha256=info["encoder_state_sha256"], numerics=info["numerics"])
        V.atomic_json(record_path, record)
        print(f"ENCODER_BITEXACT=PASS compared={compared} mismatches=0", flush=True)
        return record
    except BaseException:
        # 本次认证失败/中断后不能残留旧 PASS，亦不把部分比较伪装为 scope=full。
        if record_path.exists() and V.read_json(record_path).get("schema") == SCHEMA:
            record_path.unlink()
        raise
    finally:
        lock_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "latents", "ckpt", "expected-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scope", required=True, choices=("full",))
    parser.add_argument("--batch", type=int, default=1, choices=(1,))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    certify(args)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(V.REPO / "src"))
    raise SystemExit(main())
