#!/usr/bin/env python3
"""单集 H5 契约（1114 计划 B 契约②）：标准库＋h5py，按路径加载，G0、原帧读取、G4、XPATH 共用。

一个 episode 由 {task, episode, raw_episode, member, h5_sha256, h5_bytes, map_sha256} 描述，
来源是 git 内 4 份 ``record_dataset_<T>_episode_map.json``：

  new_idx                → 统一编号 episode（窗口键、输出命名、旧库 ``<task>_ep<episode>_<kind>`` 都用它）
  src_group=episode_<r>  → raw_episode（单集 H5 内的组名）
  member                 → 完整成员路径（含难度目录；跨难度可能同 basename，一律按完整路径解析）
  h5_sha256_local/h5_bytes → 摘要

打开文件用完整 member 路径；读组用 raw_episode；打开前核成员 sha、字节与 map sha。
例：BinFill new_idx=100 → ``BinFill_ep102_seed14200.h5`` 的组 ``episode_102``。

缺的成员按需从 ``<tar-root>/<难度目录>.h5.tar.xz`` 解到新子目录（同一 tar 只流式解一遍，多 tar 并行）。
子命令 ``inventory``：库内 raw-h5/ 全量盘点（N05）。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

MAP_PATTERN = re.compile(r"record_dataset_([A-Za-z0-9]+)_episode_map\.json")
GROUP_PATTERN = re.compile(r"episode_([0-9]+)")
FRAME_SHAPE = (256, 256, 3)


@dataclasses.dataclass(frozen=True)
class MemberRecord:
    task: str
    episode: int          # 统一编号（new_idx）
    raw_episode: int      # 单集 H5 组名 episode_<raw_episode>
    member: str           # 完整成员相对路径
    h5_sha256: str
    h5_bytes: int
    map_sha256: str

    @property
    def group(self) -> str:
        return f"episode_{self.raw_episode}"

    @property
    def tar_name(self) -> str:
        return f"{pathlib.PurePosixPath(self.member).parts[0]}.h5.tar.xz"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def sha256_file(path, chunk=8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} 必须为 64 位小写 SHA256")
    return value


def _safe_member(member):
    path = pathlib.PurePosixPath(member) if isinstance(member, str) else None
    if (path is None or not member or path.is_absolute() or ".." in path.parts or len(path.parts) < 2
            or not member.endswith(".h5")):
        raise ValueError(f"member 必须为相对 .h5 成员路径：{member!r}")
    return member


def load_member_records(maps_dir, *, expected_map_sha256=None) -> dict[tuple[str, int], MemberRecord]:
    """读 git 内 episode_map，按 (task, 统一 episode) 建索引；map sha 与期望值（如 MERGE_DONE）不等即拒。"""
    maps_dir = pathlib.Path(maps_dir)
    paths = sorted(p for p in maps_dir.iterdir() if MAP_PATTERN.fullmatch(p.name))
    if not paths:
        raise ValueError(f"目录内没有 record_dataset_<T>_episode_map.json：{maps_dir}")
    records, members = {}, set()
    seen_tasks = set()
    for path in paths:
        task = MAP_PATTERN.fullmatch(path.name)[1]
        map_sha = sha256_file(path)
        if expected_map_sha256 is not None and expected_map_sha256.get(task) != map_sha:
            raise ValueError(f"episode_map sha 与期望不符：{path.name}")
        seen_tasks.add(task)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"episode_map 必须为非空列表：{path.name}")
        for row in rows:
            match = GROUP_PATTERN.fullmatch(str(row.get("src_group")))
            episode, size = row.get("new_idx"), row.get("h5_bytes")
            if (match is None or type(episode) is not int or episode < 0 or type(size) is not int or size <= 0
                    or row.get("task") != task):
                raise ValueError(f"episode_map 行不合法：{path.name} {row.get('member')!r}")
            member = _safe_member(row.get("member"))
            if (task, episode) in records or member in members:
                raise ValueError(f"episode_map 统一编号或成员重复：{task} {episode} {member}")
            members.add(member)
            records[task, episode] = MemberRecord(
                task=task, episode=episode, raw_episode=int(match[1]), member=member,
                h5_sha256=_require_sha(row.get("h5_sha256_local"), f"{member}.h5_sha256_local"),
                h5_bytes=size, map_sha256=map_sha)
    if expected_map_sha256 is not None and seen_tasks != set(expected_map_sha256):
        raise ValueError("episode_map 任务集合与期望不符")
    return records


def expected_maps_from_merge_done(merge_done) -> dict:
    data = json.loads(pathlib.Path(merge_done).read_text(encoding="utf-8"))
    maps = data.get("maps_sha256")
    if not isinstance(maps, dict) or not maps:
        raise ValueError("MERGE_DONE 缺 maps_sha256")
    return {task: _require_sha(value, f"maps_sha256.{task}") for task, value in maps.items()}


def verify_member_file(record: MemberRecord, path, sha_cache=None) -> pathlib.Path:
    """按完整成员路径核字节与 sha；sha_cache 以 (路径, 字节, mtime) 记已核结果。"""
    path = pathlib.Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"单集 H5 不存在或是符号链接：{path}")
    stat = path.stat()
    if stat.st_size != record.h5_bytes:
        raise ValueError(f"单集 H5 字节数不符：{record.member} {stat.st_size} != {record.h5_bytes}")
    stamp = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    cache = {} if sha_cache is None else sha_cache
    if stamp not in cache:
        cache[stamp] = sha256_file(path)
    if cache[stamp] != record.h5_sha256:
        raise ValueError(f"单集 H5 sha256 不符：{record.member}")
    return path


def find_member(record: MemberRecord, roots) -> pathlib.Path | None:
    """只按 <root>/<完整 member> 找文件，不按 basename 搜索。"""
    for root in roots:
        candidate = pathlib.Path(root) / record.member
        if candidate.is_file():
            return candidate
    return None


def extract_members(records, tar_root, dest_root, *, jobs=6, xz_threads=0) -> dict[str, pathlib.Path]:
    """缺的成员从 tar 解到 dest_root/<member>（新文件，不覆盖已有）；同一 tar 一遍流式解出全部所需成员。"""
    tar_root, dest_root = pathlib.Path(tar_root), pathlib.Path(dest_root)
    by_tar = {}
    for record in records:
        target = dest_root / record.member
        if target.exists():
            raise FileExistsError(f"解包目标已存在，拒绝覆盖：{target}")
        by_tar.setdefault(record.tar_name, []).append(record)
    dest_root.mkdir(parents=True, exist_ok=True)

    def one(tar_name):
        tar_path = tar_root / tar_name
        if not tar_path.is_file():
            raise FileNotFoundError(f"缺源 tar：{tar_path}")
        group = by_tar[tar_name]
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=dest_root) as tmp:
            xz = subprocess.Popen(["xz", f"-T{xz_threads}", "-dc", str(tar_path)], stdout=subprocess.PIPE)
            try:
                subprocess.run(["tar", "-x", "-f", "-", "-C", tmp, "--no-same-owner",
                                *[r.member for r in group]], stdin=xz.stdout, check=True)
            finally:
                xz.stdout.close()
                if xz.wait() != 0:
                    raise RuntimeError(f"xz 解压失败：{tar_path}")
            out = {}
            for record in group:
                staged = pathlib.Path(tmp) / record.member
                verify_member_file(record, staged)
                target = dest_root / record.member
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(staged, target)          # 不覆盖：目标存在即 FileExistsError
                out[record.member] = target
            return out

    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(jobs, len(by_tar) or 1))) as pool:
        for part in pool.map(one, sorted(by_tar)):
            result.update(part)
    return result


def ensure_members(records, *, roots, tar_root=None, extract_root=None, sha_cache=None, jobs=6):
    """选窗后：已有的按完整路径核摘要；缺的从 tar 解到 extract_root（R2：只新建子目录）。返回 member→路径。"""
    paths, missing = {}, []
    for record in records:
        found = find_member(record, roots)
        if found is None:
            missing.append(record)
        else:
            paths[record.member] = verify_member_file(record, found, sha_cache)
    if missing:
        if tar_root is None or extract_root is None:
            raise FileNotFoundError("单集 H5 缺失且未给 tar 根与解包目录：" + ", ".join(r.member for r in missing[:5]))
        paths.update(extract_members(missing, tar_root, extract_root, jobs=jobs))
    return paths


def read_frames(record: MemberRecord, path, frame_ids):
    """按组名 episode_<raw_episode> 读 front_rgb；frame_ids 是 episode 内时步号。"""
    import h5py
    import numpy as np
    frames = []
    with h5py.File(path, "r") as handle:
        if list(handle.keys()) != [record.group]:
            raise ValueError(f"{record.member} 的组集合不是唯一的 {record.group}：{list(handle.keys())[:3]}")
        group = handle[record.group]
        for frame in frame_ids:
            data = group[f"timestep_{int(frame)}/obs/front_rgb"][()]
            if not isinstance(data, np.ndarray) or data.dtype != np.uint8 or data.shape != FRAME_SHAPE:
                raise ValueError(f"{record.member}:{record.group}/timestep_{frame} front_rgb 形状或 dtype 不符")
            frames.append(data)
    return np.ascontiguousarray(np.stack(frames))


def source_entry(record: MemberRecord, path) -> dict:
    """G0 窗口的 sources 项（kind=member_h5）。"""
    return {"kind": "member_h5", "path": str(pathlib.Path(path).resolve()), "sha256": record.h5_sha256,
            "bytes": record.h5_bytes, "member": record.member, "episode": record.episode,
            "raw_episode": record.raw_episode, "map_sha256": record.map_sha256}


def record_from_source(source: dict) -> MemberRecord:
    """从 G0 sources 项还原记录（消费端用，task 从 member 推出）。"""
    member = _safe_member(source.get("member"))
    match = re.fullmatch(r"([A-Za-z0-9]+)_ep[0-9]+_seed[0-9]+\.h5", pathlib.PurePosixPath(member).name)
    if match is None:
        raise ValueError(f"member 文件名不合法：{member}")
    for key in ("episode", "raw_episode", "bytes"):
        if type(source.get(key)) is not int or source[key] < 0:
            raise ValueError(f"member_h5 来源缺 {key}")
    return MemberRecord(task=match[1], episode=source["episode"], raw_episode=source["raw_episode"],
                        member=member, h5_sha256=_require_sha(source.get("sha256"), "member_h5.sha256"),
                        h5_bytes=source["bytes"], map_sha256=_require_sha(source.get("map_sha256"), "map_sha256"))


# ── N05：raw-h5/ 全量盘点 ─────────────────────────────────────────────────────

def inventory(raw_h5_root, records, g4prep_members, *, jobs=8) -> dict:
    """逐个按完整 member 路径核 map 里的 sha 与字节；g4prep 之外的单独列出并给出「核过可复用／只读不用」。"""
    root = pathlib.Path(raw_h5_root)
    by_member = {r.member: r for r in records.values()}
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*.h5") if p.is_file())

    def check(member):
        record = by_member.get(member)
        path = root / member
        size = path.stat().st_size
        if record is None:
            return dict(member=member, bytes=size, in_map=False, sha_ok=False, verdict="只读不用",
                        reason="不在 episode_map")
        sha = sha256_file(path)
        ok = size == record.h5_bytes and sha == record.h5_sha256
        return dict(member=member, task=record.task, episode=record.episode, raw_episode=record.raw_episode,
                    bytes=size, sha256=sha, in_map=True, sha_ok=ok,
                    verdict="核过可复用" if ok else "只读不用", reason=None if ok else "sha 或字节与 map 不符")

    # hashlib 大块 update 释放 GIL，线程池即可多核并行哈希。
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        rows = list(pool.map(check, files))
    for row in rows:
        row["in_g4prep"] = row["member"] in g4prep_members
    extra = [row for row in rows if not row["in_g4prep"]]
    return {"schema": "raw-h5-inventory-v1", "root": str(root.resolve()), "files": len(rows),
            "g4prep_listed": len(g4prep_members), "g4prep_found": sum(r["in_g4prep"] for r in rows),
            "g4prep_missing": sorted(set(g4prep_members) - set(files)),
            "sha_ok": sum(r["sha_ok"] for r in rows), "extra_count": len(extra),
            "extra": [{k: row.get(k) for k in ("member", "task", "episode", "sha_ok", "verdict", "reason")}
                      for row in extra],
            "entries": rows}


def g4prep_members_from_jsonl(path) -> set:
    members = set()
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            members.add(_safe_member(json.loads(line)["member"]))
    return members


def _atomic_new_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1, sort_keys=True)
            stream.write("\n")
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory", help="库内 raw-h5/ 全量盘点（N05）")
    inv.add_argument("--raw-h5-root", required=True)
    inv.add_argument("--episode-maps", required=True)
    inv.add_argument("--merge-done", required=True)
    inv.add_argument("--g4prep-episodes", required=True, help="_g4prep_work/episodes.jsonl（36 个已核成员）")
    inv.add_argument("--out", required=True)
    inv.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    records = load_member_records(args.episode_maps,
                                  expected_map_sha256=expected_maps_from_merge_done(args.merge_done))
    result = inventory(args.raw_h5_root, records, g4prep_members_from_jsonl(args.g4prep_episodes), jobs=args.jobs)
    if os.path.lexists(args.out):
        raise FileExistsError(f"盘点文件已存在，拒绝覆盖：{args.out}")
    _atomic_new_json(args.out, result)
    print(f"RAW_H5_INVENTORY=RECORD files={result['files']} sha_ok={result['sha_ok']} "
          f"g4prep_found={result['g4prep_found']}/{result['g4prep_listed']} extra={result['extra_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
