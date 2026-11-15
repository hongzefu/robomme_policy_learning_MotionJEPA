#!/usr/bin/env python3
"""MOTION_XPATH（1114 计划 §3、B check_motion_xpath 行，C04/N01）：只记录、不作门槛。

比较同一窗口的两条 latent 路径经同一 EMA encoder 后的 1024 维输出差异：
  A：1113 encoder 训练用的 fp16 latent（1113 数据根 rgbstate-4task 缓存，val/aug0）；
  B：旧库 fp32 latent（``<task>_ep<episode>_<kind>.bin`` 的第 start/16 块）。

共同窗口：``$DATA/index/windows.parquet`` 中 split=='val' 且 row_count==1 且 start%16==0（无 aug_id 列，
parity_aug_id 不参与过滤）。按 (task, kind) 分层、固定种子 0，每格 9 窗（7×9=63），再在排序序第 1 格补 1 窗，共 64。
对齐键 (task, 统一 episode, kind, start) → 旧库段第 start/16 块；另要求 source_frame_ids、real_frames、
pad_frames 与旧库段 metadata 逐项相等。成员与编号映射走契约②（1113 episodes.parquet 的 raw_episode/member
须与 episode_map 一致）。不等或无对应缓存的窗排除并分原因计数，排除后不补抽。

判定行：``MOTION_XPATH=RECORD common=<n> sampled=64 excluded=<n> ...``；max_abs>0.055 另打一行提醒汇报用户。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
PER_CELL = 9
TOTAL = 64
NOTE_MAX_ABS = 0.055
CHUNK_BYTES = 589824
LATENT_SHAPE = (9, 16, 32, 32)
REASONS = ("member_map_mismatch", "no_cache", "real_pad_mismatch", "frame_ids_mismatch")


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def project_module():
    return _load("_xpath_stage2_project", HERE / "stage2_project.py")


def member_module():
    return _load("_xpath_stage2_member_h5", HERE / "stage2_member_h5.py")


def common_windows(rows):
    """共同窗口：split=='val'、row_count==1、start%16==0；按 (task, kind, window_key) 排序（window_key 唯一）。"""
    keep = [dict(r) for r in rows
            if r["split"] == "val" and int(r["row_count"]) == 1 and int(r["start"]) % 16 == 0]
    return sorted(keep, key=lambda r: (r["task"], r["kind"], r["window_key"]))


def stratified_sample(common, *, seed=0, per_cell=PER_CELL, total=TOTAL):
    """按 (task, kind) 分层、固定种子：每格 per_cell 窗，余量依排序序从第 1 格起逐格各补 1 窗。"""
    cells = {}
    for row in common:
        cells.setdefault((row["task"], row["kind"]), []).append(row)
    order = sorted(cells)
    if len(order) * per_cell > total:
        raise ValueError(f"(task, kind) 格数 {len(order)}×{per_cell} 超过 {total}")
    rng = random.Random(seed)
    picked = {}
    for cell in order:
        if len(cells[cell]) < per_cell:
            raise ValueError(f"格 {cell} 可用窗 {len(cells[cell])} < {per_cell}")
        picked[cell] = rng.sample(cells[cell], per_cell)
    extra = total - len(order) * per_cell
    for cell in order[:extra]:
        rest = [r for r in cells[cell] if r not in picked[cell]]
        picked[cell] += rng.sample(rest, 1)
    sample = [row for cell in order for row in picked[cell]]
    if len(sample) != total or len({r["window_key"] for r in sample}) != total:
        raise ValueError("分层抽样未得到 64 个不同窗口")
    return sample


def align_window(window, *, episode_row, record, latents_root, metadata_cache=None):
    """返回 (原因或 None, 段 key, 块号)。原因取 REASONS 之一；None 表示可比较。"""
    task, episode, kind, start = window["task"], int(window["episode"]), window["kind"], int(window["start"])
    if (record is None or episode_row is None or record.episode != episode
            or int(episode_row["raw_episode"]) != record.raw_episode or episode_row["member"] != record.member):
        return "member_map_mismatch", None, None
    key = f"{task}_ep{episode}_{kind}"
    cache = {} if metadata_cache is None else metadata_cache
    meta_path = pathlib.Path(latents_root) / f"{key}.metadata.json"
    if key not in cache:
        cache[key] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    meta = cache[key]
    block = start // 16
    if meta is None or block >= int(meta["num_grid"]) or not (pathlib.Path(latents_root) / f"{key}.bin").is_file():
        return "no_cache", key, block
    row = meta["rows"][block]
    if int(window["real_frames"]) != row["real_frames"] or int(window["pad_frames"]) != row["pad_frames"]:
        return "real_pad_mismatch", key, block
    frames = list(range(row["start_global_frame"], row["start_global_frame"] + row["real_frames"]))
    frames += [row["pad_source_frame"]] * row["pad_frames"]
    if [int(f) for f in window["source_frame_ids"]] != frames:
        return "frame_ids_mismatch", key, block
    return None, key, block


def read_old_block(latents_root, key, block, expected_sha=None):
    with open(pathlib.Path(latents_root) / f"{key}.bin", "rb") as stream:
        stream.seek(block * CHUNK_BYTES)
        data = stream.read(CHUNK_BYTES)
    if len(data) != CHUNK_BYTES:
        raise ValueError(f"旧库块不完整：{key}:{block}")
    values = np.frombuffer(data, dtype="<f4").reshape(LATENT_SHAPE).copy()
    if not np.isfinite(values).all():
        raise ValueError(f"旧库块含非有限值：{key}:{block}")
    return values, hashlib.sha256(data).hexdigest()


def compare_vectors(a, b, eps=1e-6):
    """1024 维逐元素绝对差与相对差（相对差分母 |B|+eps）。"""
    a, b = np.asarray(a, np.float64).reshape(-1), np.asarray(b, np.float64).reshape(-1)
    if a.shape != (1024,) or b.shape != (1024,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("对拍向量必须是有限 1024 维")
    diff = np.abs(a - b)
    rel = diff / (np.abs(b) + eps)
    return dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()), max_rel=float(rel.max()),
                mean_rel=float(rel.mean()), rel_l2=float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps)))


def plan_alignment(window_rows, episode_rows, records, latents_root, *, seed=0):
    """CPU 部分：共同窗口 → 分层抽样 → 逐窗对齐与排除计数。"""
    common = common_windows(window_rows)
    sample = stratified_sample(common, seed=seed)
    episodes = {(r["task"], int(r["episode"])): r for r in episode_rows}
    excluded = {reason: 0 for reason in REASONS}
    aligned, cache = [], {}
    for window in sample:
        identity = (window["task"], int(window["episode"]))
        reason, key, block = align_window(window, episode_row=episodes.get(identity), record=records.get(identity),
                                          latents_root=latents_root, metadata_cache=cache)
        if reason is not None:
            excluded[reason] += 1
            continue
        aligned.append(dict(window_key=window["window_key"], task=window["task"], episode=int(window["episode"]),
                            kind=window["kind"], start=int(window["start"]), old_key=key, old_block=block,
                            is_tail=bool(window.get("is_tail", False))))
    cells = {}
    for window in sample:
        cells[f"{window['task']}/{window['kind']}"] = cells.get(f"{window['task']}/{window['kind']}", 0) + 1
    return dict(common=len(common), sampled=len(sample), excluded=sum(excluded.values()),
                excluded_by_reason=excluded, cells=cells, aligned=aligned,
                sample_window_keys=[w["window_key"] for w in sample])


def _atomic_new_json(path, value):
    path = pathlib.Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1, sort_keys=True)
            stream.write("\n")
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def run(args):
    import pyarrow.parquet as pq
    sys.path.insert(0, str(HERE / "wan_stage2"))
    import validate_stage2 as V
    data_root = pathlib.Path(args.data_root)
    window_rows = pq.read_table(data_root / "index/windows.parquet").to_pylist()
    episode_rows = pq.read_table(data_root / "index/episodes.parquet").to_pylist()
    members = member_module()
    records = members.load_member_records(args.episode_maps,
                                          expected_map_sha256=members.expected_maps_from_merge_done(args.merge_done))
    plan = plan_alignment(window_rows, episode_rows, records, args.latents, seed=args.seed)
    # 数值部分：两条路径过同一 EMA encoder（公开导出入口 encode_windows，fp32、TF32 关）。
    runtime = V.setup_runtime(args)
    torch, memory, _, encoder, info = runtime
    if info["encoder"]["arch"] != project_module().get_project(args.stage2_project).arch:
        raise ValueError("ckpt arch 与 --stage2-project 不符")
    reader_module = V.load_pinned_mj_source(args.mj_root, "scripts/dataset-build/rgb_gtxyz_reader.py",
                                           "_xpath_reader", info["source_pin"])
    reader = reader_module.PairedWanCacheReader(data_root, profile=reader_module.PROFILE_RGBSTATE)
    rows = []
    with torch.no_grad():
        for item in plan["aligned"]:
            pair = reader.read_pair(item["window_key"], 0)
            fp16 = np.asarray(pair["rgb_raw"])
            if fp16.shape != LATENT_SHAPE or fp16.dtype != np.float16:
                raise ValueError(f"1113 缓存 rgb_raw 形制不符：{item['window_key']}")
            old, old_sha = read_old_block(args.latents, item["old_key"], item["old_block"])
            a = memory.encode_windows(encoder, torch.from_numpy(fp16.astype(np.float32)).unsqueeze(0).to(args.device))
            b = memory.encode_windows(encoder, torch.from_numpy(old).unsqueeze(0).to(args.device))
            rows.append({**item, "old_block_sha256": old_sha,
                         **compare_vectors(a[0].cpu().numpy(), b[0].cpu().numpy())})
    summary = dict(compared=len(rows),
                   max_abs=max((r["max_abs"] for r in rows), default=None),
                   mean_abs=float(np.mean([r["mean_abs"] for r in rows])) if rows else None,
                   max_rel=max((r["max_rel"] for r in rows), default=None),
                   tail_compared=sum(r["is_tail"] for r in rows))
    record = dict(schema="motion-xpath-record-v1", verdict="RECORD", seed=args.seed,
                  stage2_project=args.stage2_project, checkpoint_sha256=args.expected_sha256,
                  windows_parquet_sha256=V.sha_file(data_root / "index/windows.parquet"),
                  latents_metadata_sha256=V.sha_file(pathlib.Path(args.latents) / "metadata.json"),
                  **{k: v for k, v in plan.items() if k != "aligned"}, summary=summary, windows=rows,
                  encoder=info["encoder"], numerics=info["numerics"])
    _atomic_new_json(args.out, record)
    reasons = " ".join(f"{k}={v}" for k, v in plan["excluded_by_reason"].items())
    print(f"MOTION_XPATH=RECORD common={plan['common']} sampled={plan['sampled']} excluded={plan['excluded']} "
          f"compared={summary['compared']} max_abs={summary['max_abs']} mean_abs={summary['mean_abs']} "
          f"max_rel={summary['max_rel']} {reasons}")
    if summary["max_abs"] is not None and summary["max_abs"] > NOTE_MAX_ABS:
        print(f"MOTION_XPATH_NOTE max_abs={summary['max_abs']} > {NOTE_MAX_ABS}：须在当轮汇报里告诉用户（只记录，不中断）")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    project_module().add_argument(parser, required=False)
    for name in ("data-root", "latents", "episode-maps", "merge-done", "ckpt", "expected-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mj-root", default=os.environ.get("MOTIONJEPA_SOURCE_ROOT", "/scratch/hongze/MotionJEPA"))
    args = parser.parse_args(argv)
    if args.stage2_project is None:
        # 契约①：本入口既无 store 也无 ckpt 推断来源之外的显式口径，必须显式给出。
        parser.error("check_motion_xpath 必须显式给出 --stage2-project")
    return run(args)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()
    sys.exit(0)
