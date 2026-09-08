#!/usr/bin/env python3
"""把一个数据集库打包 + 哈希，落到暂存目录（HF bucket 上传源）。

单遍读：每个源文件只 read 一次，同一份字节流同时喂给 sha256 与 tar/复制目标。

两条 layout，各自的产物布局见 build_jobs()：
  gl-v1          环境 A 的 4task-gl + 4task-gl-framesamp（480 GB / 88 万小文件）
  motion400ep-v1 环境 B 的 4task-motion-400ep（130 GB / 22.9 万文件）

库路径、暂存根、目标 bucket 全部由 CLI 必填参数给出，**模块里不留任何硬编码路径**
（AGENTS 第 13 条：/data/hongzefu 与 /nfs/turbo 不得写进新脚本的默认值；两条链路各自
在自己的 driver 里显式声明用哪个库、落到哪）。

断点续跑：进度记录在 <stage-root>/logs/pack_progress.jsonl，已完成分片/文件跳过；
分片分组计划持久化在 <stage-root>/logs/*_plan.json，保证重跑时分组不漂移。
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

DATA_PER_SHARD = 4096
FEAT_SHARD_TARGET = 2 * 1024**3  # 2 GiB
FLAT_SHARD_TARGET = 2 * 1024**3  # 2 GiB，扁平小文件目录（wan-latents / oracle / motion-tokens）
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


def link_and_hash(src: Path, dst: Path) -> tuple[str, int]:
    """硬链接 + 算 sha256（不写第二份数据）。跨设备等 link 失败时回退到复制。

    直传件在 motion400ep-v1 下有 8.12 GB，走硬链接省掉一整遍写。代价是收尾阶段必须
    重算源侧 sha256 断言原件未被改坏（driver 的收尾阶段做这件事）。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        return copy_and_hash(src, dst)
    return sha256_file(dst), dst.stat().st_size


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
    """members: [(源绝对路径, tar 内成员名)]。单遍读：成员读入内存→哈希→写 tar。

    tar 成员的 mtime/uid/gid/uname/gname 全部归零、成员定序写入 → 分片逐位可复现。
    """
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


def build_features_plan(
    plan_path: Path, feat_root: Path, rel_prefix: str, target_bytes: int = FEAT_SHARD_TARGET
) -> list[dict]:
    """按 episode 目录累计字节贪心分组到 ~target/片；计划持久化保证断点续跑分组不漂移。"""
    if plan_path.exists():
        return json.loads(plan_path.read_text())["shards"]
    log("扫描 features/ 逐 episode 统计大小（一次性 stat 全量文件）……")
    episodes = sorted(
        (d.name for d in os.scandir(feat_root) if d.is_dir()), key=natural_key
    )
    shards, cur_eps, cur_bytes = [], [], 0
    for ep in episodes:
        sz = sum(f.stat().st_size for f in os.scandir(feat_root / ep) if f.is_file())
        if cur_eps and cur_bytes + sz > target_bytes:
            shards.append({"episodes": cur_eps, "bytes": cur_bytes})
            cur_eps, cur_bytes = [], 0
        cur_eps.append(ep)
        cur_bytes += sz
    if cur_eps:
        shards.append({"episodes": cur_eps, "bytes": cur_bytes})
    for i, s in enumerate(shards):
        s["rel"] = f"{rel_prefix}/features-{i:05d}.tar"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"shards": shards}, ensure_ascii=False, indent=1))
    total = sum(s["bytes"] for s in shards)
    log(f"features 计划：{len(episodes)} episode → {len(shards)} 片，共 {total/1e9:.1f} GB")
    return shards


def build_flat_plan(
    plan_path: Path,
    src_dir: Path,
    rel_prefix: str,
    arc_prefix: str,
    name: str,
    target_bytes: int = FLAT_SHARD_TARGET,
) -> list[dict]:
    """扁平（或浅层）小文件目录按累计字节贪心切片。

    wan-latents / oracle / motion-tokens 三个目录里有大量 65 字节的 .sha256 边车文件，
    直传等于逐文件 HTTP 往返，性价比极低，必须打 tar。递归收集以防目录下有子层。
    """
    if plan_path.exists():
        return json.loads(plan_path.read_text())["shards"]
    files: list[tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames.sort()
        for fn in sorted(filenames, key=natural_key):
            p = Path(dirpath) / fn
            files.append((str(p.relative_to(src_dir)), p.stat().st_size))
    files.sort(key=lambda t: natural_key(t[0]))
    shards, cur, cur_bytes = [], [], 0
    for relname, sz in files:
        if cur and cur_bytes + sz > target_bytes:
            shards.append({"names": cur, "bytes": cur_bytes})
            cur, cur_bytes = [], 0
        cur.append(relname)
        cur_bytes += sz
    if cur:
        shards.append({"names": cur, "bytes": cur_bytes})
    for i, s in enumerate(shards):
        s["rel"] = f"{rel_prefix}/{name}-{i:05d}.tar"
        s["src_dir"] = str(src_dir)
        s["arc_prefix"] = arc_prefix
        s["kind"] = "flat_shard"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"shards": shards}, ensure_ascii=False, indent=1))
    total = sum(s["bytes"] for s in shards)
    log(f"{name} 计划：{len(files)} 个文件 → {len(shards)} 片，共 {total/1e9:.2f} GB")
    return shards


def shard_members(g: dict) -> list[tuple[Path, str]]:
    """按分片描述现算成员表（延迟生成，避免 23 万个元组常驻内存）。"""
    kind = g["kind"]
    if kind == "data_shard":
        src = Path(g["src_dir"])
        return [(src / f"{k}.pkl", f'{g["arc_prefix"]}/{k}.pkl') for k in range(g["lo"], g["hi"])]
    if kind == "features_shard":
        src = Path(g["src_dir"])
        out = []
        for ep in g["episodes"]:
            ep_dir = src / ep
            for nm in sorted(os.listdir(ep_dir), key=natural_key):
                out.append((ep_dir / nm, f'{g["arc_prefix"]}/{ep}/{nm}'))
        return out
    if kind == "flat_shard":
        src = Path(g["src_dir"])
        return [(src / n, f'{g["arc_prefix"]}/{n}') for n in g["names"]]
    raise ValueError(f"未知分片类型 {kind}")


def _scan_dir_files(root: Path, rel_prefix: str, mode: str) -> list[tuple[Path, str, str]]:
    """递归收集目录下全部文件为 copy_jobs 条目 (src, stage 相对路径, mode)。"""
    jobs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames, key=natural_key):
            p = Path(dirpath) / fn
            jobs.append((p, f"{rel_prefix}/{p.relative_to(root)}", mode))
    return jobs


def build_jobs(args, logs: Path, smoke: bool):
    """按 layout 产出 (copy_jobs, tar_groups, expected_small, extra_manifest)。

    copy_jobs: [(src, stage 相对路径, "link"|"copy")]
    tar_groups: [分片描述 dict]，交给 shard_members() 现算成员
    expected_small: {stage 相对路径: 期望 sha256}，与 store_meta 的交叉锚点
    """
    lib = args.lib
    copy_jobs: list[tuple[Path, str, str]] = []
    tar_groups: list[dict] = []

    if args.layout == "gl-v1":
        # 环境 A：packed 库原样复制 + 源库 data/features 打 tar
        packed = args.packed_lib
        if packed is None:
            raise SystemExit("错误: layout=gl-v1 必须给 --packed-lib")
        store_meta = json.loads((packed / "meta/store_meta.json").read_text())
        expected_small = {
            "packed/4task-gl-framesamp/pos_emb_4x4.f32.bin": store_meta["tables"]["pos_emb_4x4"]["sha256"],
            "packed/4task-gl-framesamp/state_emb.f32.bin": store_meta["tables"]["state_emb"]["sha256"],
        }
        for name in ("pos_emb_4x4.f32.bin", "state_emb.f32.bin"):
            copy_jobs.append((packed / name, f"packed/4task-gl-framesamp/{name}", "copy"))
        for name in sorted(os.listdir(packed / "meta")):
            copy_jobs.append((packed / "meta" / name, f"packed/4task-gl-framesamp/meta/{name}", "copy"))
        if not smoke:
            for f in sorted(os.scandir(packed / "image_emb_4x4"), key=lambda e: e.name):
                copy_jobs.append((Path(f.path), f"packed/4task-gl-framesamp/image_emb_4x4/{f.name}", "copy"))
        for name in sorted(os.listdir(lib / "meta")):
            copy_jobs.append((lib / "meta" / name, f"source/4task-gl/meta/{name}", "copy"))
        data_prefix, feat_prefix = "source/4task-gl/data_tars", "source/4task-gl/features_tars"
        data_dir, feat_dir = lib / "data", lib / "features"
        num = store_meta["num_exec_samples"]

    elif args.layout == "motion400ep-v1":
        # 环境 B：4task-motion-400ep 全库。framesamp / motion / meta 原样直传（硬链接），
        # source/data、source/features 与三个扁平小文件目录打 tar。logs/ 显式不上传。
        fs_meta = json.loads((lib / "framesamp/meta/store_meta.json").read_text())
        mo_meta = json.loads((lib / "motion/meta/store_meta.json").read_text())
        # 锚点必须从**本库自己的** store_meta 现读：本库 state_emb 是 80ecd422…，
        # 与 480 GB 那次的 8445ff9c… 不同，抄留档常量会得到恒失败或恒通过的假验证。
        expected_small = {
            "framesamp/pos_emb_4x4.f32.bin": fs_meta["tables"]["pos_emb_4x4"]["sha256"],
            "framesamp/state_emb.f32.bin": fs_meta["tables"]["state_emb"]["sha256"],
            "motion/motion_token.f32.bin": mo_meta["tables"]["motion_token"]["sha256"],
        }
        for name in sorted(os.listdir(lib / "meta")):
            copy_jobs.append((lib / "meta" / name, f"meta/{name}", "link"))
        for name in sorted(os.listdir(lib / "source/meta")):
            copy_jobs.append((lib / "source/meta" / name, f"source/meta/{name}", "link"))
        copy_jobs += _scan_dir_files(lib / "motion", "motion", "link")
        if smoke:
            # smoke 跳过 framesamp 的 31 个 part 大 bin（8 GB），只带两个小 bin + meta 验锚点
            for name in ("pos_emb_4x4.f32.bin", "state_emb.f32.bin"):
                copy_jobs.append((lib / "framesamp" / name, f"framesamp/{name}", "link"))
            for name in sorted(os.listdir(lib / "framesamp/meta")):
                copy_jobs.append((lib / "framesamp/meta" / name, f"framesamp/meta/{name}", "link"))
        else:
            copy_jobs += _scan_dir_files(lib / "framesamp", "framesamp", "link")
        if args.norm_stats is not None:
            # 本库唯一的训练交付件；没有它这个数据集训不起来。镜像 openpi 的 assets/<repo_id>/ 布局
            copy_jobs.append((args.norm_stats, "assets/robomme/norm_stats.json", "link"))
        # 三个扁平小文件目录 → tar
        if not smoke:
            for name, sub, arc in (
                ("wan-latents", "wan-latents", "wan-latents"),
                ("oracle-wan-mj", "oracle", "oracle"),
                ("motion-tokens", "motion-tokens", "motion-tokens"),
            ):
                tar_groups += build_flat_plan(
                    logs / f"{name}_plan.json", lib / sub, f"{sub}_tars", arc, name
                )
        data_prefix, feat_prefix = "source/data_tars", "source/features_tars"
        data_dir, feat_dir = lib / "source/data", lib / "source/features"
        num = fs_meta["num_exec_samples"]
    else:
        raise SystemExit(f"错误: 未知 layout {args.layout}")

    # data/ 分片：命名 {idx}.pkl，idx ∈ [0, num_exec_samples)，先核对目录条目集合
    log(f"核对 data/ 目录条目（期望 {num} 个 pkl）……")
    actual = {e.name for e in os.scandir(data_dir)}
    expected_names = {f"{i}.pkl" for i in range(num)}
    if actual != expected_names:
        missing = sorted(expected_names - actual)[:5]
        extra = sorted(actual - expected_names)[:5]
        raise SystemExit(f"错误：data/ 条目与期望不符。缺失示例 {missing}，多出示例 {extra}")
    n_data_shards = (num + DATA_PER_SHARD - 1) // DATA_PER_SHARD
    data_groups = [
        {
            "kind": "data_shard",
            "rel": f"{data_prefix}/data-{i:05d}.tar",
            "src_dir": str(data_dir),
            "arc_prefix": "data",
            "lo": i * DATA_PER_SHARD,
            "hi": min((i + 1) * DATA_PER_SHARD, num),
        }
        for i in range(n_data_shards)
    ]
    feat_shards = build_features_plan(logs / "features_plan.json", feat_dir, feat_prefix)
    for s in feat_shards:
        s.update({"kind": "features_shard", "src_dir": str(feat_dir), "arc_prefix": "features"})

    if smoke:
        data_groups = data_groups[: args.limit_shards]
        feat_shards = feat_shards[:1]
        log(f"smoke 模式：data 前 {len(data_groups)} 片 + features 前 1 片，跳过大 bin 与扁平目录")

    tar_groups = data_groups + feat_shards + tar_groups
    return copy_jobs, tar_groups, expected_small, num


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, required=True, help="数据集库根")
    ap.add_argument("--packed-lib", type=Path, default=None, help="仅 gl-v1：packed 库根")
    ap.add_argument("--stage-root", type=Path, required=True, help="暂存根（其下自动建 stage/ logs/）")
    ap.add_argument("--bucket", required=True, help="目标 bucket URI，写进 upload_manifest.json")
    ap.add_argument("--layout", required=True, choices=("gl-v1", "motion400ep-v1"))
    ap.add_argument("--norm-stats", type=Path, default=None, help="额外带上的 norm_stats.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit-shards",
        type=int,
        default=0,
        help="smoke 模式：只打前 N 个 data 分片 + 1 个 features 分片，跳过大 bin",
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

    copy_jobs, tar_groups, expected_small, num = build_jobs(args, logs, smoke)

    # ---- 执行 ----
    errors: list[str] = []
    anchors_ok: list[str] = []

    def run_copy(src: Path, rel: str, mode: str) -> None:
        if progress.done(rel, stage):
            log(f"跳过（已完成）{rel}")
            if rel in expected_small:
                anchors_ok.append(rel)
            return
        digest, n = (link_and_hash if mode == "link" else copy_and_hash)(src, stage / rel)
        if rel in expected_small:
            if digest != expected_small[rel]:
                errors.append(f"{rel} sha256={digest} 与 store_meta 记录 {expected_small[rel]} 不符")
                return
            anchors_ok.append(rel)
        progress.record({"rel": rel, "kind": "file", "sha256": digest, "bytes": n})
        log(f"{'链接' if mode == 'link' else '复制'}完成 {rel}（{n/1e9:.2f} GB）")

    def run_shard(g: dict) -> None:
        if progress.done(g["rel"], stage):
            log(f"跳过（已完成）{g['rel']}")
            return
        pack_tar_shard(g["rel"], shard_members(g), stage, parts_dir, progress, g["kind"])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_copy, s, r, m) for s, r, m in copy_jobs]
        futs += [pool.submit(run_shard, g) for g in tar_groups]
        for f in futs:
            f.result()  # 抛出 worker 异常

    if errors:
        for e in errors:
            log(f"错误：{e}")
        return 1

    # ---- 汇总清单（smoke 模式跳过，避免半成品清单误用）----
    if smoke:
        log(f"ANCHOR_OK={len(set(anchors_ok))}")
        log(f"smoke 打包完成，耗时 {time.time()-t0:.0f}s（清单汇总留待全量跑）")
        return 0

    if len(set(anchors_ok)) != len(expected_small):
        log(f"错误：交叉锚点只命中 {len(set(anchors_ok))}/{len(expected_small)} 条")
        return 1

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
    # gl-v1 沿用历史文件名 sha256-packed.txt（verify_download.py 的默认值认它）；
    # motion400ep-v1 用 sha256-plain.txt —— 本库里 "packed" 特指 framesamp packed store，同名会歧义
    plain_name = "sha256-packed.txt" if args.layout == "gl-v1" else "sha256-plain.txt"
    (cks / plain_name).write_text(
        "".join(f"{e['sha256']}  {e['rel']}\n" for e in file_entries)
    )
    with open(cks / "sha256-source-files.txt", "w") as fo:
        for pf in sorted(parts_dir.iterdir(), key=lambda p: p.name):
            fo.write(pf.read_text())

    n_shards = len(shard_entries)
    total_bytes = sum(e["bytes"] for e in shard_entries) + sum(e["bytes"] for e in file_entries)
    total_members = sum(e["members"] for e in shard_entries)
    (stage / "manifest").mkdir(exist_ok=True)
    # created_from 只写仓库相对路径：str() 一个绝对 Path 会把构建机绝对路径写进要公开的清单
    (stage / "manifest/upload_manifest.json").write_text(
        json.dumps(
            {
                "bucket": args.bucket,
                "created_from": {
                    "lib": str(args.lib.resolve().relative_to(REPO)),
                    "layout": args.layout,
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
    log(f"ANCHOR_OK={len(set(anchors_ok))}")
    log(
        f"PACK_DONE shards={n_shards} plain={len(file_entries)} "
        f"members={total_members} bytes={total_bytes}"
    )
    log(
        f"全部完成：{n_shards} 个 tar 分片 + {len(file_entries)} 个平文件，"
        f"共 {total_bytes/1e9:.1f} GB / tar 内成员 {total_members} 个，"
        f"耗时 {(time.time()-t0)/60:.1f} 分钟"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
