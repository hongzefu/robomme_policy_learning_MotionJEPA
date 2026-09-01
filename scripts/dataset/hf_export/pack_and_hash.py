#!/usr/bin/env python3
"""把两个数据库打包+哈希，落到本机 /data 暂存目录（HF bucket 上传源）。

单遍读 NFS：每个源文件只 read 一次，同一份字节流同时喂给 sha256 与 tar/复制目标。
产物布局（stage/ 即 bucket 最终布局）：
  packed/4task-gl-framesamp/...            packed 库原样复制
  source/4task-gl/data_tars/data-*.tar     data/ 每 4096 个 pkl 一片
  source/4task-gl/features_tars/...        features/ 按 episode 分组 ~2GiB 一片
  source/4task-gl/meta/                    源库 meta 原样
  checksums/sha256-{shards,packed,source-files}.txt
  manifest/upload_manifest.json
断点续跑：进度记录在 <stage-root>/logs/pack_progress.jsonl，已完成分片/文件跳过。
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC_PACKED = REPO / "v1-store/datasets/4task-gl-framesamp"
SRC_SOURCE = REPO / "v1-store/datasets/4task-gl"
DEFAULT_STAGE_ROOT = Path("/data/hongzefu/hf-export-robomme-vla-motionjepa-v1")

DATA_PER_SHARD = 4096
FEAT_SHARD_TARGET = 2 * 1024**3  # 2 GiB
CHUNK = 8 * 1024 * 1024

_progress_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_and_hash(src: Path, dst: Path) -> tuple[str, int]:
    """复制文件（写 tmp 后原子 rename），同时算 sha256。返回 (hex, 字节数)。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            h.update(b)
            fo.write(b)
            n += len(b)
    os.replace(tmp, dst)
    return h.hexdigest(), n


def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


class Progress:
    """append-only jsonl 进度账本，支持断点续跑。"""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                self.entries[e["rel"]] = e
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a")

    def done(self, rel: str, stage_root: Path) -> dict | None:
        """已有进度且落盘文件尺寸吻合则返回该条目，否则 None。"""
        e = self.entries.get(rel)
        if e is None:
            return None
        p = stage_root / rel
        if p.exists() and p.stat().st_size == e["bytes"]:
            return e
        return None

    def record(self, e: dict) -> None:
        with _progress_lock:
            self.entries[e["rel"]] = e
            self._fh.write(json.dumps(e, ensure_ascii=False) + "\n")
            self._fh.flush()


def pack_tar_shard(
    rel: str,
    members: list[tuple[Path, str]],
    stage_root: Path,
    parts_dir: Path,
    progress: Progress,
    kind: str,
) -> dict:
    """members: [(源绝对路径, tar 内成员名)]。单遍读：成员读入内存→哈希→写 tar。"""
    out = stage_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    part_lines = []
    n_bytes = 0
    with tarfile.open(tmp, "w") as tf:
        for src, arcname in members:
            with open(src, "rb") as f:
                data = f.read()
            part_lines.append(f"{hashlib.sha256(data).hexdigest()}  {arcname}")
            ti = tarfile.TarInfo(name=arcname)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            tf.addfile(ti, io.BytesIO(data))
            n_bytes += len(data)
    os.replace(tmp, out)
    part_file = parts_dir / (out.name + ".sha256")
    part_file.write_text("\n".join(part_lines) + "\n")
    e = {
        "rel": rel,
        "kind": kind,
        "sha256": sha256_file(out),
        "bytes": out.stat().st_size,
        "members": len(members),
        "member_bytes": n_bytes,
    }
    progress.record(e)
    log(f"分片完成 {rel}（{len(members)} 个成员，{n_bytes/1e9:.2f} GB）")
    return e


def build_features_plan(plan_path: Path) -> list[dict]:
    """按 episode 目录累计字节贪心分组到 ~2GiB/片；计划持久化保证断点续跑分组不漂移。"""
    if plan_path.exists():
        return json.loads(plan_path.read_text())["shards"]
    log("扫描 features/ 逐 episode 统计大小（一次性 stat 全量文件）……")
    feat_root = SRC_SOURCE / "features"
    episodes = sorted(
        (d.name for d in os.scandir(feat_root) if d.is_dir()), key=natural_key
    )
    shards, cur_eps, cur_bytes = [], [], 0
    for ep in episodes:
        sz = sum(f.stat().st_size for f in os.scandir(feat_root / ep) if f.is_file())
        if cur_eps and cur_bytes + sz > FEAT_SHARD_TARGET:
            shards.append({"episodes": cur_eps, "bytes": cur_bytes})
            cur_eps, cur_bytes = [], 0
        cur_eps.append(ep)
        cur_bytes += sz
    if cur_eps:
        shards.append({"episodes": cur_eps, "bytes": cur_bytes})
    for i, s in enumerate(shards):
        s["rel"] = f"source/4task-gl/features_tars/features-{i:05d}.tar"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"shards": shards}, ensure_ascii=False, indent=1))
    total = sum(s["bytes"] for s in shards)
    log(f"features 计划：{len(episodes)} episode → {len(shards)} 片，共 {total/1e9:.1f} GB")
    return shards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit-shards",
        type=int,
        default=0,
        help="smoke 模式：只打前 N 个 data 分片 + 1 个 features 分片，跳过 packed 大 bin",
    )
    args = ap.parse_args()
    t0 = time.time()
    smoke = args.limit_shards > 0

    stage = args.stage_root / "stage"
    logs = args.stage_root / "logs"
    parts_dir = logs / "parts"
    for d in (stage, logs, parts_dir):
        d.mkdir(parents=True, exist_ok=True)
    progress = Progress(logs / "pack_progress.jsonl")

    store_meta = json.loads((SRC_PACKED / "meta/store_meta.json").read_text())
    expected_small = {
        "packed/4task-gl-framesamp/pos_emb_4x4.f32.bin": store_meta["tables"]["pos_emb_4x4"]["sha256"],
        "packed/4task-gl-framesamp/state_emb.f32.bin": store_meta["tables"]["state_emb"]["sha256"],
    }

    # ---- 任务清单 ----
    # 1) packed 库 + 源库 meta：原样复制文件
    copy_jobs: list[tuple[Path, str]] = []  # (src, stage 相对路径)
    for name in ("pos_emb_4x4.f32.bin", "state_emb.f32.bin"):
        copy_jobs.append((SRC_PACKED / name, f"packed/4task-gl-framesamp/{name}"))
    for name in sorted(os.listdir(SRC_PACKED / "meta")):
        copy_jobs.append((SRC_PACKED / "meta" / name, f"packed/4task-gl-framesamp/meta/{name}"))
    if not smoke:
        for f in sorted(os.scandir(SRC_PACKED / "image_emb_4x4"), key=lambda e: e.name):
            copy_jobs.append((Path(f.path), f"packed/4task-gl-framesamp/image_emb_4x4/{f.name}"))
    for name in sorted(os.listdir(SRC_SOURCE / "meta")):
        copy_jobs.append((SRC_SOURCE / "meta" / name, f"source/4task-gl/meta/{name}"))

    # 2) data/ 分片：命名 {idx}.pkl，idx ∈ [0, num_exec_samples)，先核对目录条目集合
    num = store_meta["num_exec_samples"]
    log(f"核对 data/ 目录条目（期望 {num} 个 pkl）……")
    actual = {e.name for e in os.scandir(SRC_SOURCE / "data")}
    expected_names = {f"{i}.pkl" for i in range(num)}
    if actual != expected_names:
        missing = sorted(expected_names - actual)[:5]
        extra = sorted(actual - expected_names)[:5]
        log(f"错误：data/ 条目与期望不符。缺失示例 {missing}，多出示例 {extra}")
        return 1
    n_data_shards = (num + DATA_PER_SHARD - 1) // DATA_PER_SHARD
    data_shard_ids = list(range(n_data_shards))

    # 3) features 分片计划（持久化）
    feat_shards = build_features_plan(logs / "features_plan.json")

    if smoke:
        data_shard_ids = data_shard_ids[: args.limit_shards]
        feat_shards = feat_shards[:1]
        log(f"smoke 模式：data 前 {len(data_shard_ids)} 片 + features 前 1 片，跳过 image_emb 大 bin")

    # ---- 执行 ----
    errors: list[str] = []

    def run_copy(src: Path, rel: str) -> None:
        if progress.done(rel, stage):
            log(f"跳过（已完成）{rel}")
            return
        digest, n = copy_and_hash(src, stage / rel)
        if rel in expected_small and digest != expected_small[rel]:
            errors.append(f"{rel} sha256={digest} 与 store_meta 记录 {expected_small[rel]} 不符")
            return
        progress.record({"rel": rel, "kind": "file", "sha256": digest, "bytes": n})
        log(f"复制完成 {rel}（{n/1e9:.2f} GB）")

    def run_data_shard(i: int) -> None:
        rel = f"source/4task-gl/data_tars/data-{i:05d}.tar"
        if progress.done(rel, stage):
            log(f"跳过（已完成）{rel}")
            return
        idxs = range(i * DATA_PER_SHARD, min((i + 1) * DATA_PER_SHARD, num))
        members = [(SRC_SOURCE / "data" / f"{k}.pkl", f"data/{k}.pkl") for k in idxs]
        pack_tar_shard(rel, members, stage, parts_dir, progress, "data_shard")

    def run_feat_shard(s: dict) -> None:
        rel = s["rel"]
        if progress.done(rel, stage):
            log(f"跳过（已完成）{rel}")
            return
        members = []
        for ep in s["episodes"]:
            ep_dir = SRC_SOURCE / "features" / ep
            for name in sorted(os.listdir(ep_dir), key=natural_key):
                members.append((ep_dir / name, f"features/{ep}/{name}"))
        pack_tar_shard(rel, members, stage, parts_dir, progress, "features_shard")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = []
        for src, rel in copy_jobs:
            futs.append(pool.submit(run_copy, src, rel))
        for i in data_shard_ids:
            futs.append(pool.submit(run_data_shard, i))
        for s in feat_shards:
            futs.append(pool.submit(run_feat_shard, s))
        for f in futs:
            f.result()  # 抛出 worker 异常

    if errors:
        for e in errors:
            log(f"错误：{e}")
        return 1

    # ---- 汇总清单（smoke 模式跳过，避免半成品清单误用）----
    if smoke:
        log(f"smoke 打包完成，耗时 {time.time()-t0:.0f}s（清单汇总留待全量跑）")
        return 0

    cks = stage / "checksums"
    cks.mkdir(exist_ok=True)
    shard_entries = sorted(
        (e for e in progress.entries.values() if e["kind"].endswith("_shard")),
        key=lambda e: e["rel"],
    )
    file_entries = sorted(
        (e for e in progress.entries.values() if e["kind"] == "file"),
        key=lambda e: e["rel"],
    )
    (cks / "sha256-shards.txt").write_text(
        "".join(f"{e['sha256']}  {e['rel']}\n" for e in shard_entries)
    )
    (cks / "sha256-packed.txt").write_text(
        "".join(f"{e['sha256']}  {e['rel']}\n" for e in file_entries)
    )
    with open(cks / "sha256-source-files.txt", "w") as fo:
        for pf in sorted(parts_dir.iterdir(), key=lambda p: p.name):
            fo.write(pf.read_text())

    n_shards = len(shard_entries)
    total_bytes = sum(e["bytes"] for e in shard_entries) + sum(e["bytes"] for e in file_entries)
    total_members = sum(e["members"] for e in shard_entries)
    (stage / "manifest").mkdir(exist_ok=True)
    (stage / "manifest/upload_manifest.json").write_text(
        json.dumps(
            {
                "bucket": "hf://buckets/HongzeFu/robomme-vla-motionjepa-v1",
                "created_from": {
                    "packed": str(SRC_PACKED),
                    "source": str(SRC_SOURCE),
                },
                "data_per_shard": DATA_PER_SHARD,
                "num_exec_samples": num,
                "shards": [
                    {k: e[k] for k in ("rel", "sha256", "bytes", "members")}
                    for e in shard_entries
                ],
                "plain_files": [
                    {k: e[k] for k in ("rel", "sha256", "bytes")} for e in file_entries
                ],
                "total_bytes": total_bytes,
                "total_tar_members": total_members,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    log(
        f"全部完成：{n_shards} 个 tar 分片 + {len(file_entries)} 个平文件，"
        f"共 {total_bytes/1e9:.1f} GB / tar 内成员 {total_members} 个，"
        f"耗时 {(time.time()-t0)/60:.1f} 分钟"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
