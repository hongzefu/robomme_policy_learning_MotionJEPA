#!/usr/bin/env python3
"""G5：保留全表encoder验证，或CPU全表完整性加固定1024窗encoder验证。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_stage2 as V

SCHEMA = "stage2-encoder-bitexact-v1"
SAMPLED_SCHEMA = "stage2-encoder-sampled-v1"
INTEGRITY_SCHEMA = "stage2-packed-integrity-v1"
AUTHORIZATION_QUOTE = "同意按照你说过的更省时方案 做\nencoder只抽机小部分"


def file_stamp(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def check_stamps(stamps):
    """全文件SHA之后检查文件身份，避免再读一遍全部latent；变化即失败。"""
    for path, expected in stamps.items():
        if file_stamp(path) != expected:
            raise ValueError(f"验证期间源文件改变：{path}")


def check_sidecar(path, digest):
    fields = Path(str(path) + ".sha256").read_text(encoding="utf-8").split()
    if (len(fields) not in (1, 2) or fields[0] != digest
            or (len(fields) == 2 and fields[1] not in (Path(path).name, "*" + Path(path).name))):
        raise ValueError(f"源文件SHA sidecar不符：{path}")


def hash_latent(job):
    """单次完整读取含奇数格的latent，并逐偶数块核S1记录；最多四线程调用。"""
    path, old_count, expected_sha, expected_blocks, stamp = job
    digest, blocks = hashlib.sha256(), []
    with Path(path).open("rb") as stream:
        for grid in range(old_count):
            block = stream.read(V.CHUNK_BYTES)
            if len(block) != V.CHUNK_BYTES:
                raise ValueError(f"源latent块被截断：{path}")
            if not np.isfinite(np.frombuffer(block, dtype="<f4")).all():
                raise ValueError(f"源latent含非有限数：{path}")
            digest.update(block)
            if grid % 2 == 0:
                blocks.append(V.sha_bytes(block))
        if stream.read(1):
            raise ValueError(f"源latent含多余尾字节：{path}")
    if digest.hexdigest() != expected_sha or blocks != expected_blocks or file_stamp(path) != stamp:
        raise ValueError(f"源latent完整SHA、偶数块SHA或读取期间身份不符：{path}")
    check_sidecar(path, digest.hexdigest())


def compare_sampled_rows(*, table, segments, encode, latent_root, sample_plan):
    """只执行共享计划给出的1024次前向；不预热、不扩大或替换样本。"""
    indices = sample_plan["sample_indices"]
    if (sample_plan.get("seed") != 0 or not isinstance(indices, list) or len(indices) != 1024
            or any(type(index) is not int or not 0 <= index < len(table) for index in indices)
            or indices != sorted(set(indices))
            or sample_plan.get("sample_indices_sha256") != V.sha_bytes(V.canonical(indices))):
        raise ValueError("共享抽样计划必须含seed0、1024唯一升序合法行及精确摘要")
    segment_index, compared = 0, 0
    for row in indices:
        while segment_index < len(segments) and row >= segments[segment_index][1] + segments[segment_index][2]:
            segment_index += 1
        if segment_index == len(segments) or row < segments[segment_index][1]:
            raise ValueError("抽样行未被段索引覆盖")
        key, base, _ = segments[segment_index]
        with (Path(latent_root) / f"{key}.bin").open("rb") as stream:
            stream.seek(2 * (row - base) * V.CHUNK_BYTES)
            block = stream.read(V.CHUNK_BYTES)
        if len(block) != V.CHUNK_BYTES:
            raise ValueError("抽样latent块被截断")
        latent = np.frombuffer(block, dtype="<f4").reshape(V.LATENT_SHAPE).copy()
        if not np.isfinite(latent).all():
            raise ValueError("抽样latent含非有限数")
        output = np.asarray(encode(latent))
        if (output.shape != (1024,) or output.dtype != np.float32 or not np.isfinite(output).all()
                or output.astype("<f4", copy=False).tobytes() != table[row].astype("<f4", copy=False).tobytes()):
            raise ValueError(f"抽样encoder逐位不同：表行{row}，{key}")
        compared += 1
    return compared


def cpu_full_integrity(*, meta, table, latent_root, hash_workers=4):
    """packed全表真实memcmp全部S1段；根清单、元数据、sidecar及完整latent均闭合。"""
    latent_root = Path(latent_root)
    tokens_root = Path(meta.provenance["tokens_root"])
    if type(hash_workers) is not int or not 1 <= hash_workers <= 4:
        raise ValueError("CPU散列线程必须在1至4之间")
    pack = V.load_module(V.REPO / "scripts/dataset/pack_motion_store.py", "_stage2_compare_pack_contract")
    segments = list(pack.iter_rows_in_order(meta.entries))
    keys = [item[2] for item in segments]
    expected = set(keys)
    if not keys or len(keys) != len(expected):
        raise ValueError("完整表段身份为空或重复")
    for suffix in (".bin", ".bin.sha256", ".metadata.json"):
        found = {path.name.removesuffix(suffix) for path in latent_root.glob(f"*{suffix}")}
        if found != expected:
            raise ValueError(f"完整源latent段集合不符：{suffix}")
    allowed = {f"{key}{suffix}" for key in keys for suffix in (".f32.bin", ".f32.bin.sha256", ".metadata.json")}
    allowed.add("export_record.json")
    if {path.name for path in tokens_root.iterdir()} != allowed:
        raise ValueError("S1输出目录与完整段清单不闭合")
    watched = [latent_root / "metadata.json", tokens_root / "export_record.json", Path(meta.manifest_path)]
    watched += [latent_root / f"{key}{suffix}" for key in keys for suffix in (".bin", ".bin.sha256", ".metadata.json")]
    watched += [tokens_root / f"{key}{suffix}" for key in keys for suffix in (".f32.bin", ".f32.bin.sha256", ".metadata.json")]
    if any(path.is_symlink() or not path.is_file() for path in watched):
        raise ValueError("源文件必须是存在的普通实体文件")
    stamps = {str(path): file_stamp(path) for path in watched}
    manifest_sha = V.sha_file(meta.manifest_path)
    if (manifest_sha != meta.provenance["episode_manifest_file_sha256"]
            or V.sha_file(latent_root / "metadata.json") != meta.provenance["latents_metadata_sha256"]):
        raise ValueError("清单或源latent总metadata与packed来源不符")
    # 只复用既有只读的偶数格、段集合、逐行起点及尾窗合同，不调用pack/verify写入口。
    pack.validate_stage2_segments(latent_root, tokens_root, meta.entries)
    root_metadata = V.read_json(latent_root / "metadata.json")
    export_path = tokens_root / "export_record.json"
    export = V.read_json(export_path)
    if (export.get("schema") != "stage2-memory-export-v1" or export.get("scope") != "full"
            or export.get("segments") != len(keys) or export.get("rows") != len(table)
            or export.get("segment_keys") != sorted(keys) or export.get("manifest_sha256") != manifest_sha):
        raise ValueError("S1根记录未精确覆盖当前完整表与清单")
    identity = {name: meta.provenance[name] for name in ("encoder", "encoder_state_sha256", "numerics")}
    identity["encoder"] = {k: v for k, v in identity["encoder"].items() if k != "run_dir"}
    identity["source_pin"] = {key: meta.provenance["source_pin"][key]
                              for key in ("mj_repo_commit", "export_source_sha256", "encoder_source_sha256")}

    def check_identity(record):
        for group, wanted in identity.items():
            actual = record.get(group)
            if group == "encoder" and isinstance(actual, dict):
                actual = {key: value for key, value in actual.items() if key != "run_dir"}
            if V.canonical(actual) != V.canonical(wanted):
                raise ValueError(f"S1的{group}与packed来源不符")

    check_identity(export)
    pack.validate_stage2_export_files(export, tokens_root, meta.entries)
    cursor, mapping, jobs = 0, {}, []
    # S1按段键排序，packed按episode/demo/exec排序；只通过完整键映射连接两套顺序。
    for entry, _, key, row_base, count in segments:
        if row_base != cursor:
            raise ValueError("packed行覆盖不连续")
        token_path = tokens_root / f"{key}.f32.bin"
        data, tm = pack.read_segment_tokens(tokens_root, key, count, spec=entry.spec)
        check_sidecar(token_path, tm["sha256"])
        check_identity(tm)
        if tm.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"{key} token元数据与S1根记录不符")
        if table[row_base:row_base + count].astype("<f4", copy=False).tobytes() != data:
            raise ValueError(f"packed表与S1 token原始字节不同：{key}")
        latent_path = latent_root / f"{key}.bin"
        latent_meta_path = latent_root / f"{key}.metadata.json"
        lm = V.read_json(latent_meta_path)
        for field, expected_value in (("g", entry.g), ("h5_file", entry.h5_file), ("raw_ep_idx", entry.raw_ep_idx)):
            if lm.get(field) != expected_value or tm.get(field) != expected_value:
                raise ValueError(f"{key} 元数据episode身份与当前索引不同：{field}")
        old_count = lm["num_grid"]
        digest = V.require_sha(lm.get("sha256"), f"{key} latent SHA")
        if (old_count <= 0 or (old_count + 1) // 2 != count
                or latent_path.stat().st_size != old_count * V.CHUNK_BYTES or lm.get("bytes") != old_count * V.CHUNK_BYTES
                or tm.get("input_metadata_sha256") != V.sha_file(latent_meta_path)
                or tm.get("input_latent_sha256") != digest or root_metadata["segments"][key].get("sha256") != digest
                or not isinstance(tm.get("input_block_sha256"), list) or len(tm["input_block_sha256"]) != count):
            raise ValueError(f"{key} 完整源latent与S1身份不闭合")
        mapping[key] = list(range(0, old_count, 2))
        jobs.append((str(latent_path), old_count, digest, tm["input_block_sha256"], stamps[str(latent_path)]))
        cursor += count
    if cursor != len(table) or cursor != meta.num_rows:
        raise ValueError("CPU字节比较未覆盖完整packed表")
    executor = ThreadPoolExecutor(max_workers=hash_workers)
    try:
        for _ in executor.map(hash_latent, jobs):
            pass
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    check_stamps(stamps)
    return dict(packed_compared=cursor, packed_mismatches=0, source_grid=mapping,
                source_tokens_manifest_sha256=V.sha_file(export_path), source_stamps=stamps,
                segments=[(key, base, count) for _, _, key, base, count in segments])


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
        if record_path.exists():
            previous = V.read_json(record_path)
            if previous.get("schema") not in (SCHEMA, SAMPLED_SCHEMA, INTEGRITY_SCHEMA):
                raise ValueError("认证路径已用于不同合同")
            record_path.unlink()
        if args.scope not in ("full", "sampled") or args.batch != 1:
            raise ValueError("认证强制 scope=full或sampled、batch=1")
        meta = ms.MotionMeta.load(root)
        if (meta.raw["layout"] != V.LAYOUT or type(getattr(args, "expect_rows", None)) is not int
                or args.expect_rows <= 0 or meta.num_rows != args.expect_rows):
            raise ValueError("正式 G5a 行数必须与显式 --expect-rows 相同")
        ms.run_fast_checks(meta)
        ms.run_full_checks(meta)
        bound_paths = [root / ms.META_RELPATH, root / ms.INDEX_RELPATH,
                       root / ms.MOTION_TABLE_RELPATH, Path(args.ckpt), Path(args.latents) / "metadata.json"]
        hashes = {str(path): V.sha_file(path) for path in bound_paths}
        if hashes[str(Path(args.ckpt))] != args.expected_sha256:
            raise ValueError("checkpoint SHA 不符")
        if hashes[str(Path(args.latents) / "metadata.json")] != meta.provenance["latents_metadata_sha256"]:
            raise ValueError("源 latent 总 metadata SHA 与 store 不同")
        segments = []
        for entry in meta.entries:
            for kind in ("demo", "exec"):
                part = getattr(entry, kind)
                if part.num_grid:
                    segments.append((ms.segment_key(entry.h5_file, entry.raw_ep_idx, kind), part.row_base, part.num_grid))
        table = np.memmap(root / ms.MOTION_TABLE_RELPATH, mode="r", dtype="<f4", shape=(meta.num_rows, 1024))
        if ms.stage2_uses_cache(meta.provenance):
            integrity = cpu_full_integrity(meta=meta, table=table, latent_root=args.latents)
            if not np.isfinite(table).all():
                raise ValueError("packed表含非有限数")
            mapping = integrity["source_grid"]
            source_sha = V.sha_bytes(V.canonical(mapping))
            if source_sha != meta.provenance["source_grid_sha256"]:
                raise ValueError("缓存完整偶数格映射与 store 来源不同")
            for path, digest in hashes.items():
                if V.sha_file(path) != digest:
                    raise ValueError(f"完整性检查期间绑定文件改变：{path}")
            record = dict(schema=INTEGRITY_SCHEMA, status="PASS", scope="full", layout=V.LAYOUT,
                          compared=meta.num_rows, packed_compared=meta.num_rows, mismatches=0, nonfinite=0,
                          source_grid_sha256=source_sha, output_numerics=dict(ms.SAME_INPUT_NOT_REQUIRED),
                          store_table_sha256=meta.table_sha256, motion_index_sha256=meta.motion_index_sha256,
                          checkpoint_sha256=args.expected_sha256,
                          source_metadata_sha256=meta.provenance["latents_metadata_sha256"],
                          cache_import_receipt_sha256=meta.provenance["stage2_compat"]["cache_import_receipt_sha256"])
            V.atomic_json(record_path, record)
            print(f"PACKED_TOKEN_BYTES=PASS compared={meta.num_rows} mismatches=0 nonfinite=0", flush=True)
            print("ENCODER_BITEXACT=NOT_REQUIRED reason=user_decision_same_input", flush=True)
            return record
        integrity, sample_plan = None, None
        if args.scope == "sampled":
            sample_plan = ms.stage2_encoder_sample_plan(meta.entries)
            integrity = cpu_full_integrity(meta=meta, table=table, latent_root=args.latents)
            print(f"PACKED_TOKEN_BYTES=PASS compared={integrity['packed_compared']} mismatches=0", flush=True)
        torch, memory, _, encoder, info = V.setup_runtime(args)
        V.assert_encoder_identity(info, meta.provenance)

        def encode(latent):
            value = torch.from_numpy(latent).unsqueeze(0).to(args.device)
            return memory.encode_windows(encoder, value)[0].cpu().numpy()

        if args.scope == "full":
            compared, mapping = compare_rows(table=table, segments=segments, encode=encode, latent_root=args.latents,
                                             root_metadata=V.read_json(Path(args.latents) / "metadata.json"))
        else:
            compared = compare_sampled_rows(table=table, segments=integrity["segments"], encode=encode,
                                             latent_root=args.latents, sample_plan=sample_plan)
            mapping = integrity["source_grid"]
            check_stamps(integrity["source_stamps"])
        source_sha = V.sha_bytes(V.canonical(mapping))
        if source_sha != meta.provenance["source_grid_sha256"]:
            raise ValueError("完整偶数格映射与 store 来源不同")
        for path, digest in hashes.items():
            if V.sha_file(path) != digest:
                raise ValueError(f"比较期间绑定文件改变：{path}")
        record = dict(schema=SCHEMA, status="PASS", producer=V.PRODUCER, scope=args.scope, layout=V.LAYOUT,
                      compared=compared, mismatches=0, source_grid_sha256=source_sha,
                      store_table_sha256=hashes[str(root / ms.MOTION_TABLE_RELPATH)],
                      motion_index_sha256=hashes[str(root / ms.INDEX_RELPATH)],
                      checkpoint_sha256=args.expected_sha256, segments=len(mapping),
                      source_metadata_sha256=hashes[str(Path(args.latents) / "metadata.json")],
                      encoder_state_sha256=info["encoder_state_sha256"], numerics=info["numerics"])
        if args.scope == "sampled":
            record.update(schema=SAMPLED_SCHEMA, total_rows=meta.num_rows,
                          packed_compared=integrity["packed_compared"], packed_mismatches=0,
                          source_tokens_manifest_sha256=integrity["source_tokens_manifest_sha256"],
                          authorization_quote=AUTHORIZATION_QUOTE, **sample_plan)
        V.atomic_json(record_path, record)
        if args.scope == "full":
            print(f"ENCODER_BITEXACT=PASS compared={compared} mismatches=0", flush=True)
        else:
            print(f"ENCODER_SAMPLE_BITEXACT=PASS compared={compared} total={meta.num_rows} mismatches=0 seed=0", flush=True)
        return record
    except BaseException:
        # 本次认证失败/中断后不能残留旧 PASS，亦不把部分比较伪装为 scope=full。
        if record_path.exists() and V.read_json(record_path).get("schema") in (SCHEMA, SAMPLED_SCHEMA, INTEGRITY_SCHEMA):
            record_path.unlink()
        raise
    finally:
        lock_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "latents", "ckpt", "expected-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scope", required=True, choices=("full", "sampled"))
    parser.add_argument("--batch", type=int, default=1, choices=(1,))
    parser.add_argument("--expect-rows", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    certify(args)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(V.REPO / "src"))
    raise SystemExit(main())
