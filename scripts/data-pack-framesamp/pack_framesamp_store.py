#!/usr/bin/env python3
"""framesamp packed 特征库打包工具（v2-framesamp-restructure-plan.md A.1/A.2）。

把 4task-gl 的 483,291 个每帧 npy（602,951 B pickle dict，无法部分读取）压成
只含 framesample 真正要的三张表的连续大文件库：

  image_emb_4x4/part_000..031.bf16.bin   (rows,16,2048) bf16 裸字节，按 episode 边界切分
  pos_emb_4x4.f32.bin                    (num_pos_rows,16,768) f32——pos 是 t 的纯函数，只存一份
  state_emb.f32.bin                      (num_rows,8) f32
  meta/store_meta.json                   唯一契约（两阶段写：pack→packed、verify→verified）

子命令：
  plan    只算贪心切分并打印 part 表（不写盘）
  pack    构建三张表（写侧逐帧校验 + read-after-write + part sha256 原子落盘）
  verify  全量 483,291 帧写×读对拍（g 级零遗漏唯一凭据）+ 逐行 row_digests + meta 回填
  report  打印 meta 摘要

事务协议（A.2）：meta/pack.lock 排他锁（O_CREAT|O_EXCL；残锁 --resume 接管、异 host
--force-break-lock）；小表先行、主进程独写；image part 并行（每 part 唯一属主，天然
无锁）；progress 单写（父进程追加 pack_progress.jsonl）；崩溃后 --resume 按
「存在+大小+sha256」跳过完好 part、清 .tmp 重做。

格式常量、行号公式、StoreMeta、读 API 一律 import 自 mme_vla_suite.datastore，
绝不复制（B.0）。本模块顶层不 import jax（Pool 用 fork，jax 不允许进 fork 前进程）。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import multiprocessing
import os
import pathlib
import shutil
import socket
import sys
import time
import uuid

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import framesamp_store as fs  # noqa: E402
from mme_vla_suite.datastore.manifest import load_manifest, manifest_sha256  # noqa: E402

_MP = multiprocessing.get_context("fork")   # worker 继承小表/配置（COW），顶层无 jax


# ══ 帧读取（--reader decode|slice）════════════════════════════════════════════

# fork 前由主进程钉死的全局（worker 继承只读副本）
_G: dict = {}


def _src_npy(source_root: str, g: int, t: int) -> str:
    return os.path.join(source_root, "features", f"episode_{g}", f"token_emb_{t}.npy")


def read_frame_decode(path: str) -> tuple[bytes, bytes, bytes]:
    """全量反序列化（首跑默认，零布局假设），返回三键原始字节。"""
    with open(path, "rb") as f:
        d = np.load(f, allow_pickle=True).item()
    img, pos, stt = d[fs.IMAGE_KEY], d[fs.POS_KEY], d[fs.STATE_KEY]
    if img.shape != (1,) + fs.IMAGE_ROW_SHAPE or img.dtype != fs.IMAGE_DTYPE:
        raise ValueError(f"源帧 {path} image 形制不符: {img.shape} {img.dtype}")
    if pos.shape != (1,) + fs.POS_ROW_SHAPE or pos.dtype != np.float32:
        raise ValueError(f"源帧 {path} pos 形制不符: {pos.shape} {pos.dtype}")
    if stt.shape != fs.STATE_ROW_SHAPE or stt.dtype != np.float32:
        raise ValueError(f"源帧 {path} state 形制不符: {stt.shape} {stt.dtype}")
    return img.tobytes(), pos.tobytes(), stt.tobytes()


def read_frame_slice(path: str) -> tuple[bytes, bytes, bytes]:
    """按已实测偏移常量 pread 三个窗口（重跑加速档）。

    三重守卫（A.2）：st_size==602,951；文件前 64 B 与参考前缀逐字节相同（参考前缀
    由主进程用 decode 档自证后钉死）；逐帧 pos 窗口 100% memcmp 由写侧校验①承担。
    """
    prefix = _G["slice_prefix"]
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        if st.st_size != fs.SOURCE_NPY_SIZE:
            raise ValueError(f"源帧 {path} st_size={st.st_size} != {fs.SOURCE_NPY_SIZE}，"
                             f"slice 档前提被破坏，请改用 --reader decode")
        head = os.pread(fd, 64, 0)
        if head != prefix:
            raise ValueError(f"源帧 {path} 前 64 B 与参考前缀不符，slice 档前提被破坏")
        img = os.pread(fd, fs.IMAGE_ROW_BYTES, fs.SOURCE_IMAGE_OFFSET)
        pos = os.pread(fd, fs.POS_ROW_BYTES, fs.SOURCE_POS_OFFSET)
        stt = os.pread(fd, fs.STATE_ROW_BYTES, fs.SOURCE_STATE_OFFSET)
        if len(img) != fs.IMAGE_ROW_BYTES or len(pos) != fs.POS_ROW_BYTES \
                or len(stt) != fs.STATE_ROW_BYTES:
            raise ValueError(f"源帧 {path} slice 窗口短读")
        return img, pos, stt
    finally:
        os.close(fd)


def _read_frame(g: int, t: int) -> tuple[bytes, bytes, bytes]:
    path = _src_npy(_G["source_root"], g, t)
    if _G["reader"] == "slice":
        return read_frame_slice(path)
    return read_frame_decode(path)


def _pin_slice_prefix(source_root: str, ep0: dict) -> bytes:
    """slice 档参考前缀：取首 episode 首帧，decode 与 slice 窗口互证后钉死前 64 B。"""
    path = _src_npy(source_root, ep0["global_episode_idx"], 0)
    with open(path, "rb") as f:
        head = f.read(64)
    img_d, pos_d, stt_d = read_frame_decode(path)
    raw = pathlib.Path(path).read_bytes()
    if (raw[fs.SOURCE_IMAGE_OFFSET:fs.SOURCE_IMAGE_OFFSET + fs.IMAGE_ROW_BYTES] != img_d
            or raw[fs.SOURCE_POS_OFFSET:fs.SOURCE_POS_OFFSET + fs.POS_ROW_BYTES] != pos_d
            or raw[fs.SOURCE_STATE_OFFSET:fs.SOURCE_STATE_OFFSET + fs.STATE_ROW_BYTES] != stt_d):
        raise ValueError("slice 偏移常量与首帧 decode 结果不符——数据格式已变，"
                         "禁用 --reader slice 并复核 probe_layout.py")
    return head


# ══ pos 表旁证生成（G7 闸：CPU 后端一律拒绝）══════════════════════════════════


def generate_pos_table_posemb3d(num_pos_rows: int) -> np.ndarray:
    """PosEmb3D 现生成旁证路径——必须 GPU 后端（A.2 定论）。

    已实测 CPU 后端生成与库中值不逐位一致（max|diff| ≈ 7e-7），GPU 后端一致；
    故 CPU 后端直接 raise（守卫 G7 钉死）。主方案是源库抽取拼装（pack 即用），
    本旁证路径未实装生成体——需要时按 build_shard.py 的 PosEmb3D 切片补。
    """
    import jax  # 懒 import：本模块顶层禁 jax（fork 安全）
    platform = jax.devices()[0].platform
    if platform != "gpu":
        raise RuntimeError(
            f"pos 表 PosEmb3D 生成旁证要求 GPU 后端（CPU 生成与库中值不逐位一致，"
            f"max|diff|≈7e-7）: 当前 platform={platform!r}")
    raise NotImplementedError(
        "旁证生成体未实装（主方案为源库抽取拼装，A.2 定论）；需要旁证时按 "
        "build_shard.py 的 PosEmb3D 切片实装并逐位对拍抽取表")


# ══ 锁协议（A.2）══════════════════════════════════════════════════════════════


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(store_root: pathlib.Path, *, resume: bool, force_break: bool,
                 phase: str) -> dict:
    lock = store_root / fs.LOCK_RELPATH
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = {"build_uuid": uuid.uuid4().hex, "host": socket.gethostname(),
               "pid": os.getpid(), "phase": phase,
               "started_at": datetime.datetime.now().isoformat(timespec="seconds")}
    blob = (json.dumps(payload, ensure_ascii=False, indent=1) + "\n").encode()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, blob)
            os.fsync(fd)
        finally:
            os.close(fd)
        return payload
    except FileExistsError:
        pass
    old_text = lock.read_text(encoding="utf-8", errors="replace")
    try:
        old = json.loads(old_text)
    except json.JSONDecodeError:
        old = {}
    same_host = old.get("host") == payload["host"]
    if same_host and old.get("pid") == os.getpid():
        # 自有锁（同一驱动进程 pack→verify 接续）：直接接管换 phase
        tmp = lock.with_suffix(".lock.tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, lock)
        return payload
    if same_host and isinstance(old.get("pid"), int) and _pid_alive(old["pid"]):
        raise RuntimeError(f"pack.lock 属同 host 存活进程 pid={old['pid']}，拒跑: {lock}")
    if same_host:
        if not resume:
            raise RuntimeError(
                f"pack.lock 残锁（同 host、pid 不存活）: {lock}\n{old_text}"
                f"确认无并发后用 --resume 显式接管")
    else:
        if not force_break:
            raise RuntimeError(
                f"pack.lock 属异 host（跨 host 无法判活），一律拒跑: {lock}\n{old_text}"
                f"确认异 host 无进程后用 --force-break-lock 破锁")
        print(f"⚠ 异 host 锁全文:\n{old_text}", flush=True)
        ans = input("确认破异 host 锁并接管？输入 yes 继续: ").strip()
        if ans != "yes":
            raise RuntimeError("未确认破锁，退出")
    tmp = lock.with_suffix(".lock.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, lock)
    return payload


def release_lock(store_root: pathlib.Path) -> None:
    (store_root / fs.LOCK_RELPATH).unlink(missing_ok=True)


# ══ 通用小件 ══════════════════════════════════════════════════════════════════


def atomic_write_bytes(path: pathlib.Path, data) -> None:
    """tmp + fsync + replace + 目录 fsync（NFS 重命名可见性）。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _pick_episodes(manifest: dict, subset_prefix: int | None) -> list[dict]:
    episodes = manifest["episodes"]
    if subset_prefix is None:
        return episodes
    if not 0 <= subset_prefix < len(episodes):
        raise ValueError(f"--subset-prefix {subset_prefix} 越界（0..{len(episodes) - 1}）")
    chosen = episodes[:subset_prefix + 1]
    if not any(ep["num_timesteps"] >= 33 for ep in chosen):
        raise ValueError(
            "迷你库规格硬约束（A.1）：前缀内必须含至少一个 num_timesteps ≥ 33 的 "
            "episode（step=30 短样本与 step=31 满长样本都要存在）")
    return chosen


def plan_parts(episodes: list[dict]) -> list[list[dict]]:
    """贪心切分：按 global_episode_idx 升序累积 num_timesteps，累计 ≥ ceil(rows/32)
    即切；切点必在 episode 边界（一个样本的 32 帧必在同一 part）。"""
    total = sum(ep["num_timesteps"] for ep in episodes)
    thr = math.ceil(total / fs.TARGET_PARTS)
    groups: list[list[dict]] = []
    cur: list[dict] = []
    rows = 0
    for ep in episodes:
        cur.append(ep)
        rows += ep["num_timesteps"]
        if rows >= thr:
            groups.append(cur)
            cur, rows = [], 0
    if cur:
        groups.append(cur)
    return groups


def _mini_manifest_sha256(chosen: list[dict]) -> str:
    blob = json.dumps(chosen, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _spot_entries(source_root: pathlib.Path, episodes: list[dict],
                  num_exec: int) -> dict:
    """源库抽样指纹：8 个等距 (g,t) 的 npy + 8 个等距 exec id 的 pkl，
    各记 {relpath, size, blake2b-128 headtail}（读侧 fast 档抽 1 条复验）。"""
    row_starts = np.array([ep["total_sample_offset"] for ep in episodes], np.int64)
    num_rows = int(row_starts[-1] + episodes[-1]["num_timesteps"] - row_starts[0]) \
        if episodes else 0
    rels: list[str] = []
    k = min(8, max(1, num_rows))
    for i in range(k):
        row = row_starts[0] + i * (num_rows - 1) // max(1, k - 1)
        g_i = int(np.searchsorted(row_starts, row, side="right") - 1)
        ep = episodes[g_i]
        t = int(row - ep["total_sample_offset"])
        rels.append(f"features/episode_{ep['global_episode_idx']}/token_emb_{t}.npy")
    k = min(8, max(1, num_exec))
    for i in range(k):
        idx = i * (num_exec - 1) // max(1, k - 1)
        rels.append(f"data/{idx}.pkl")
    entries = []
    for rel in sorted(set(rels)):
        p = source_root / rel
        size, d, _ = fs.headtail_digest(p)
        entries.append({"relpath": rel, "size": size, "digest": d})
    return {"scheme": "8-equidistant-npy+8-equidistant-pkl-headtail1MiB-blake2b128",
            "entries": entries}


# ══ pack：小表 ════════════════════════════════════════════════════════════════


def _state_chunk_worker(chunk: list[tuple[int, int, int]]) -> list[tuple[int, bytes]]:
    """chunk 项 = (g, num_timesteps, rel_row0)；返回 (rel_row0, 该 episode 全帧 state 字节)。"""
    out = []
    for g, nt, rel_row0 in chunk:
        blob = bytearray(nt * fs.STATE_ROW_BYTES)
        for t in range(nt):
            _, _, stt = _read_frame(g, t)
            blob[t * fs.STATE_ROW_BYTES:(t + 1) * fs.STATE_ROW_BYTES] = stt
        out.append((rel_row0, bytes(blob)))
    return out


def build_small_tables(episodes: list[dict], procs: int) -> tuple[bytes, bytes, int]:
    """两张小表（主进程独写前的内存构建）：返回 (pos_table, state_table, num_pos_rows)。

    pos 表：取前缀内 num_timesteps 最大的 episode，其 t=0..max-1 帧逐位抽取拼装
    （主 pass 的 100% pos memcmp 即证明「只依赖 t」）。state 表：并行 decode 全部
    源帧的 state 键，主进程按行拼装（谁写全局表归属唯一）。
    """
    num_pos_rows = max(ep["num_timesteps"] for ep in episodes)
    donor = max(episodes, key=lambda e: e["num_timesteps"])
    g = donor["global_episode_idx"]
    print(f"[pack] pos 表：episode_{g}（{num_pos_rows} 帧）逐位抽取拼装", flush=True)
    pos = bytearray(num_pos_rows * fs.POS_ROW_BYTES)
    for t in range(num_pos_rows):
        _, p, _ = _read_frame(g, t)
        pos[t * fs.POS_ROW_BYTES:(t + 1) * fs.POS_ROW_BYTES] = p

    row0 = episodes[0]["total_sample_offset"]
    num_rows = sum(ep["num_timesteps"] for ep in episodes)
    print(f"[pack] state 表：{procs} 进程 decode {len(episodes)} episode / {num_rows} 帧",
          flush=True)
    items = [(ep["global_episode_idx"], ep["num_timesteps"],
              ep["total_sample_offset"] - row0) for ep in episodes]
    n_chunks = max(1, min(len(items), procs * 4))
    chunks = [items[i::n_chunks] for i in range(n_chunks)]
    state = bytearray(num_rows * fs.STATE_ROW_BYTES)
    t0 = time.perf_counter()
    with _MP.Pool(procs) as pool:
        done_eps = 0
        for res in pool.imap_unordered(_state_chunk_worker, chunks):
            for rel_row0, blob in res:
                state[rel_row0 * fs.STATE_ROW_BYTES:
                      rel_row0 * fs.STATE_ROW_BYTES + len(blob)] = blob
                done_eps += 1
            print(f"[pack] state 表进度 {done_eps}/{len(episodes)} episode "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return bytes(pos), bytes(state), num_pos_rows


# ══ pack：image part ═════════════════════════════════════════════════════════


def _pack_part_worker(task: dict) -> dict:
    """一个 part 的唯一属主：逐 episode slab 写 .tmp + 写侧逐帧校验 + read-after-write
    + sha256 + os.replace。返回 progress 记录。"""
    t0 = time.perf_counter()
    part_idx = task["index"]
    eps = task["episodes"]
    store_root = pathlib.Path(_G["store_root"])
    final = store_root / fs.IMAGE_PART_DIR / f"part_{part_idx:03d}.bf16.bin"
    tmp = final.with_name(final.name + ".tmp")
    pos_table: bytes = _G["pos_table"]
    state_table: bytes = _G["state_table"]
    row0_global = _G["row0_global"]
    sha = hashlib.sha256()
    offset = 0
    with open(tmp, "w+b") as f:   # 读写打开：③ read-after-write 需要在同一 fd 上 pread
        fd = f.fileno()
        for ep in eps:
            g, nt = ep["global_episode_idx"], ep["num_timesteps"]
            blob = bytearray(nt * fs.IMAGE_ROW_BYTES)
            for t in range(nt):
                img, pos, stt = _read_frame(g, t)
                # ① pos memcmp 钉死 t 与「pos 只依赖 t」（不钉 g——数学上分不出同 t 调包，
                #    g 级身份唯一凭据是 verify 全量对拍，F.1）
                if pos != pos_table[t * fs.POS_ROW_BYTES:(t + 1) * fs.POS_ROW_BYTES]:
                    raise RuntimeError(f"写侧校验①失败: episode_{g} t={t} pos ≠ pos_table[t]")
                # ② state memcmp 同源自证（防行内错乱）
                row_rel = ep["total_sample_offset"] - row0_global + t
                if stt != state_table[row_rel * fs.STATE_ROW_BYTES:
                                      (row_rel + 1) * fs.STATE_ROW_BYTES]:
                    raise RuntimeError(f"写侧校验②失败: episode_{g} t={t} state ≠ state 表同行")
                blob[t * fs.IMAGE_ROW_BYTES:(t + 1) * fs.IMAGE_ROW_BYTES] = img
            f.write(blob)
            f.flush()
            # ③ read-after-write：slab 落盘后 pread 读回 memcmp
            back = os.pread(fd, len(blob), offset)
            if back != bytes(blob):
                raise RuntimeError(f"写侧校验③失败: part {part_idx} episode_{g} 读回不符")
            sha.update(back)   # sha 取读回字节 = 盘上内容的摘要
            offset += len(blob)
        os.fsync(fd)
    size, ht, full_cov = fs.headtail_digest(tmp)
    if size != offset:
        raise RuntimeError(f"part {part_idx} 尺寸帐不符: {size} != {offset}")
    os.replace(tmp, final)
    dfd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return {"index": part_idx,
            "path": f"{fs.IMAGE_PART_DIR}/{final.name}",
            "start_row": eps[0]["total_sample_offset"] - row0_global,
            "num_rows": offset // fs.IMAGE_ROW_BYTES,
            "bytes": offset,
            "sha256": sha.hexdigest(),
            "head_tail_digest": ht,
            "full_covered": full_cov,
            "episodes": [eps[0]["global_episode_idx"], eps[-1]["global_episode_idx"]],
            "elapsed": round(time.perf_counter() - t0, 1)}


def _truncate_half_line(path: pathlib.Path) -> None:
    """--resume 前处理 progress 尾部半行：seek 到最后一个换行符并 ftruncate。"""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    with open(path, "r+b") as f:
        f.truncate(cut)


def _load_progress(path: pathlib.Path) -> dict[int, dict]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue   # 尾部半行：读侧直接丢弃
            done[rec["index"]] = rec
    return done


# ══ verify ════════════════════════════════════════════════════════════════════


def _verify_init(store_root: str, manifest_path: str, source_root: str) -> None:
    """verify worker initializer：各 worker 自建 FrameSampStore（真实读 API）。"""
    _G["vstore"] = fs.FrameSampStore(
        store_root, manifest_path=manifest_path, source_root=source_root)


def _verify_part_worker(task: dict) -> dict:
    """对一个 part 的全部 (g,t)：重新完整 decode 源 npy，三键各经真实读 API 对拍，
    逐行产出 blake2b-128 摘要（image‖pos‖state，store 侧字节）。"""
    store: fs.FrameSampStore = _G["vstore"]
    row0_global = _G["row0_global"]
    mismatches: list[tuple[int, str]] = []
    digests = bytearray()
    scanned = 0
    for ep in task["episodes"]:
        g, nt = ep["global_episode_idx"], ep["num_timesteps"]
        rows = np.arange(nt, dtype=np.int64) + (ep["total_sample_offset"] - row0_global)
        img_rows = store.read_image_rows(rows)          # 真实读 API①
        pos_rows = store.pos_rows(np.arange(nt))        # 真实读 API②
        stt_rows = store.state_rows(rows)               # 真实读 API③
        for t in range(nt):
            src_img, src_pos, src_stt = read_frame_decode(
                _src_npy(_G["source_root"], g, t))
            st_img = img_rows[t].tobytes()
            st_pos = pos_rows[t].tobytes()
            st_stt = stt_rows[t].tobytes()
            row = int(rows[t])
            if st_img != src_img:
                mismatches.append((row, "image"))
            if st_pos != src_pos:
                mismatches.append((row, "pos"))
            if st_stt != src_stt:
                mismatches.append((row, "state"))
            h = hashlib.blake2b(digest_size=fs.ROW_DIGEST_BYTES)
            h.update(st_img)
            h.update(st_pos)
            h.update(st_stt)
            digests += h.digest()
            scanned += 1
    return {"index": task["index"], "start_row": task["start_row"],
            "scanned": scanned, "digests": bytes(digests), "mismatches": mismatches}


def _verify_sample_worker(args: tuple) -> tuple[int, list[tuple[int, str]]]:
    """抽样档：单行三键对拍（开发期快检，不得用于交付判定）。"""
    row, g, t = args
    store: fs.FrameSampStore = _G["vstore"]
    row0_global = _G["row0_global"]
    src_img, src_pos, src_stt = read_frame_decode(_src_npy(_G["source_root"], g, t))
    bad = []
    if store.read_image_rows([row - row0_global])[0].tobytes() != src_img:
        bad.append((row, "image"))
    if store.pos_rows([t])[0].tobytes() != src_pos:
        bad.append((row, "pos"))
    if store.state_rows([row - row0_global])[0].tobytes() != src_stt:
        bad.append((row, "state"))
    return 1, bad


# ══ 子命令 ════════════════════════════════════════════════════════════════════


def cmd_plan(args) -> None:
    manifest = load_manifest(args.manifest)
    episodes = _pick_episodes(manifest, args.subset_prefix)
    groups = plan_parts(episodes)
    total = sum(ep["num_timesteps"] for ep in episodes)
    print(f"[plan] episodes={len(episodes)} rows={total} "
          f"阈值={math.ceil(total / fs.TARGET_PARTS)} parts={len(groups)}")
    for i, g in enumerate(groups):
        rows = sum(ep["num_timesteps"] for ep in g)
        print(f"  part_{i:03d}: episodes[{g[0]['global_episode_idx']}.."
              f"{g[-1]['global_episode_idx']}] rows={rows} "
              f"bytes={rows * fs.IMAGE_ROW_BYTES:,}")


def cmd_pack(args) -> None:
    t_all = time.perf_counter()
    manifest = load_manifest(args.manifest)
    episodes = _pick_episodes(manifest, args.subset_prefix)
    source_root = pathlib.Path(args.source).resolve()
    store_root = pathlib.Path(args.out).resolve()
    row0_global = episodes[0]["total_sample_offset"]   # 前缀子集恒 0（A.1）
    if row0_global != 0:
        raise RuntimeError(f"前缀子集 total_sample_offset[0] 必须为 0，实为 {row0_global}")
    num_rows = sum(ep["num_timesteps"] for ep in episodes)
    num_exec = sum(ep["exec_samples"] for ep in episodes)
    groups = plan_parts(episodes)
    procs = args.procs

    store_root.mkdir(parents=True, exist_ok=True)
    (store_root / fs.IMAGE_PART_DIR).mkdir(exist_ok=True)
    meta_path = store_root / fs.META_RELPATH

    lock = acquire_lock(store_root, resume=args.resume,
                        force_break=args.force_break_lock, phase="pack")
    try:
        if meta_path.exists() and not args.resume:
            raise RuntimeError(f"store_meta.json 已存在: {meta_path}；重打包须显式 --resume")

        # df 预检（A.2：全量 ≥ 40 GB；子集按估算的 2 倍）
        need = num_rows * fs.IMAGE_ROW_BYTES + num_rows * fs.STATE_ROW_BYTES \
            + max(ep["num_timesteps"] for ep in episodes) * fs.POS_ROW_BYTES
        floor = 40 * 10**9 if args.subset_prefix is None else 2 * need + 256 * 10**6
        free = shutil.disk_usage(store_root).free
        if free < floor:
            raise RuntimeError(f"磁盘余量不足: free={free / 1e9:.1f} GB < 需 {floor / 1e9:.1f} GB")

        _G.update(source_root=str(source_root), reader=args.reader,
                  store_root=str(store_root), row0_global=row0_global)
        if args.reader == "slice":
            _G["slice_prefix"] = _pin_slice_prefix(str(source_root), episodes[0])

        # ―― 小表先行、主进程独写（A.2）――
        pos_blob, state_blob, num_pos_rows = build_small_tables(episodes, procs)
        _G["pos_table"], _G["state_table"] = pos_blob, state_blob
        atomic_write_bytes(store_root / fs.POS_TABLE_RELPATH, pos_blob)
        atomic_write_bytes(store_root / fs.STATE_TABLE_RELPATH, state_blob)
        pos_sha = hashlib.sha256(pos_blob).hexdigest()
        state_sha = hashlib.sha256(state_blob).hexdigest()
        print(f"[pack] 小表落盘: pos {len(pos_blob):,} B / state {len(state_blob):,} B",
              flush=True)

        # ―― image part 并行（每 part 唯一属主）――
        progress_path = store_root / fs.PROGRESS_RELPATH
        if args.resume:
            _truncate_half_line(progress_path)
        done = _load_progress(progress_path) if args.resume else {}
        for rec in list(done.values()):   # resume：存在+大小+sha256 全过才跳过
            f = store_root / rec["path"]
            ok = f.is_file() and f.stat().st_size == rec["bytes"] \
                and fs.sha256_file(f) == rec["sha256"]
            if not ok:
                done.pop(rec["index"])
        for tmpf in (store_root / fs.IMAGE_PART_DIR).glob("*.tmp"):
            tmpf.unlink()   # .tmp 残留一律清除重做
        tasks = []
        for i, g in enumerate(groups):
            if i in done:
                continue
            tasks.append({"index": i, "episodes": g,
                          "start_row": g[0]["total_sample_offset"] - row0_global})
        print(f"[pack] image part: 共 {len(groups)} 个, 跳过已完好 {len(done)} 个, "
              f"本轮 {len(tasks)} 个 × {procs} 进程", flush=True)
        with _MP.Pool(procs) as pool:
            with open(progress_path, "a", encoding="utf-8") as pf:
                for rec in pool.imap_unordered(_pack_part_worker, tasks):
                    pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pf.flush()   # progress 单写（只有父进程追加）
                    done[rec["index"]] = rec
                    print(f"[pack] part_{rec['index']:03d} 完成 rows={rec['num_rows']} "
                          f"({rec['elapsed']}s), 进度 {len(done)}/{len(groups)}", flush=True)
        if len(done) != len(groups):
            raise RuntimeError(f"part 完成数 {len(done)} != 计划 {len(groups)}")

        # ―― meta 阶段 1（status=packed, verify=null）――
        parts_meta = []
        for i in range(len(groups)):
            rec = dict(done[i])
            rec.pop("elapsed", None)
            parts_meta.append(rec)
        import importlib.metadata as md
        import subprocess
        git_head = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
        prov_p = source_root / "meta" / "provenance.json"
        meta = {
            "schema": fs.META_SCHEMA,
            "layout": fs.LAYOUT,
            "status": "packed",
            "byte_order": fs.BYTE_ORDER,
            "array_order": fs.ARRAY_ORDER,
            "bf16_encoding": fs.BF16_ENCODING,
            "tables": {
                fs.IMAGE_KEY: {"row_shape": list(fs.IMAGE_ROW_SHAPE), "dtype": "bfloat16",
                               "row_bytes": fs.IMAGE_ROW_BYTES, "num_rows": num_rows,
                               "part_dir": fs.IMAGE_PART_DIR},
                fs.POS_KEY: {"row_shape": list(fs.POS_ROW_SHAPE), "dtype": "float32",
                             "row_bytes": fs.POS_ROW_BYTES, "num_rows": num_pos_rows,
                             "relpath": fs.POS_TABLE_RELPATH,
                             "byte_count": len(pos_blob), "sha256": pos_sha},
                fs.STATE_KEY: {"row_shape": list(fs.STATE_ROW_SHAPE), "dtype": "float32",
                               "row_bytes": fs.STATE_ROW_BYTES, "num_rows": num_rows,
                               "relpath": fs.STATE_TABLE_RELPATH,
                               "byte_count": len(state_blob), "sha256": state_sha},
            },
            "num_rows": num_rows,
            "num_exec_samples": num_exec,
            "num_pos_rows": num_pos_rows,
            "manifest_sha256": manifest["sha256"],
            "manifest_path": str(pathlib.Path(args.manifest).resolve()),
            "source_dataset_root": str(source_root),
            "source_provenance_sha256": fs.sha256_file(prov_p) if prov_p.is_file() else None,
            "source_spot_sha256": _spot_entries(source_root, episodes, num_exec),
            "manifest_scope": "full" if args.subset_prefix is None else "subset",
            "parts": parts_meta,
            "packer": {"build_uuid": lock["build_uuid"], "host": lock["host"],
                       "git_commit": git_head, "python": sys.version.split()[0],
                       "numpy": md.version("numpy"), "ml_dtypes": md.version("ml_dtypes"),
                       "reader": args.reader, "procs": procs,
                       "started_at": lock["started_at"], "finished_at": _now()},
            "verify": None,
            "row_digests": None,
        }
        if args.subset_prefix is not None:
            meta["subset_episodes"] = [ep["global_episode_idx"] for ep in episodes]
            meta["mini_manifest_sha256"] = _mini_manifest_sha256(episodes)
        atomic_write_bytes(meta_path, json.dumps(meta, ensure_ascii=False, indent=1).encode())
        fs.StoreMeta.load(store_root)   # 自检：契约能读回
        print(f"[pack] meta 阶段1 落盘（status=packed）; 总耗时 "
              f"{time.perf_counter() - t_all:.0f}s", flush=True)
        print("PACK_DONE=1 （pack.lock 保留，verify 回填后才释放——A.2）", flush=True)
    except BaseException:
        print(f"[pack] 失败：pack.lock 与半成品保留，供 --resume 续跑", flush=True)
        raise


def cmd_verify(args) -> None:
    t_all = time.perf_counter()
    store_root = pathlib.Path(args.store).resolve()
    meta = fs.StoreMeta.load(store_root)
    manifest_path = args.manifest or meta.manifest_path
    source_root = str(pathlib.Path(args.source).resolve()) if args.source \
        else meta.source_dataset_root
    manifest = load_manifest(manifest_path)
    if manifest["sha256"] != meta.manifest_sha256:
        raise RuntimeError("verify: 清单指纹与 meta 不符")
    episodes = manifest["episodes"]
    if meta.manifest_scope == "subset":
        episodes = episodes[:len(meta.subset_episodes)]
    row0_global = episodes[0]["total_sample_offset"]
    groups = plan_parts(episodes)
    if len(groups) != len(meta.parts):
        raise RuntimeError(f"verify: 现场切分 {len(groups)} part != meta {len(meta.parts)}")

    started_at = _now()
    lock_path = store_root / fs.LOCK_RELPATH
    had_lock = lock_path.exists()
    if had_lock:
        acquire_lock(store_root, resume=args.resume,
                     force_break=args.force_break_lock, phase="verify")
    else:
        acquire_lock(store_root, resume=False, force_break=False, phase="verify")

    _G.update(source_root=source_root, reader="decode",
              store_root=str(store_root), row0_global=row0_global)
    init = (str(store_root), manifest_path, source_root)

    if args.sample:
        rng = np.random.default_rng(args.seed)
        num_rows = meta.num_rows
        rows = np.sort(rng.choice(num_rows, size=min(args.sample, num_rows),
                                  replace=False))
        starts = np.array([ep["total_sample_offset"] for ep in episodes], np.int64)
        tasks = []
        for row in rows:
            gi = int(np.searchsorted(starts, row + row0_global, side="right") - 1)
            ep = episodes[gi]
            tasks.append((int(row) + row0_global,
                          ep["global_episode_idx"],
                          int(row) + row0_global - ep["total_sample_offset"]))
        scanned, bad = 0, []
        with _MP.Pool(args.procs, initializer=_verify_init, initargs=init) as pool:
            for n, b in pool.imap_unordered(_verify_sample_worker, tasks):
                scanned += n
                bad.extend(b)
        verdict = "PASS" if not bad else "FAIL"
        print(f"VERIFY_SAMPLE={verdict} scanned={scanned} mismatches={len(bad)} "
              f"seed={args.seed} ⚠ 抽样档仅供开发期快检，不得用于交付判定（A.2）")
        if not had_lock:
            release_lock(store_root)   # 接管来的锁保持原状——抽样不构成 verify 完成
        if bad:
            for row, key in bad[:10]:
                print(f"  失配 row={row} key={key}")
            raise SystemExit(1)
        return

    tasks = [{"index": i, "episodes": g,
              "start_row": g[0]["total_sample_offset"] - row0_global}
             for i, g in enumerate(groups)]
    scanned = 0
    mismatches: list[tuple[int, str]] = []
    results: dict[int, dict] = {}
    with _MP.Pool(args.procs, initializer=_verify_init, initargs=init) as pool:
        for res in pool.imap_unordered(_verify_part_worker, tasks):
            results[res["index"]] = res
            scanned += res["scanned"]
            mismatches.extend(res["mismatches"])
            print(f"[verify] part_{res['index']:03d} 扫 {res['scanned']} 行, "
                  f"失配 {len(res['mismatches'])}, 进度 {len(results)}/{len(groups)} "
                  f"({time.perf_counter() - t_all:.0f}s)", flush=True)

    if scanned != meta.num_rows:
        raise RuntimeError(f"verify 扫描行数 {scanned} != num_rows {meta.num_rows}（有遗漏）")

    if mismatches:
        print(f"VERIFY_PACK=FAIL scanned={scanned} mismatches={len(mismatches)}")
        for row, key in mismatches[:20]:
            print(f"  失配 row={row} key={key}")
        print("⚠ FAIL：不回填 meta、保留 pack.lock（阻断读侧消费）；"
              "定位修复后重打包再 verify")
        raise SystemExit(1)

    # 逐行摘要按 part 序拼接后单写（父进程）
    digests = bytearray()
    for i in range(len(groups)):
        digests += results[i]["digests"]
    if len(digests) != meta.num_rows * fs.ROW_DIGEST_BYTES:
        raise RuntimeError("row_digests 长度帐不符")
    atomic_write_bytes(store_root / fs.ROW_DIGESTS_RELPATH, digests)

    raw = dict(meta.raw)
    raw["status"] = "verified"
    raw["verify"] = {"scanned": scanned, "mismatches": 0, "sample": None,
                     "seed": None, "procs": args.procs,
                     "reader": "decode", "started_at": started_at,
                     "finished_at": _now(),
                     "conclusion": f"VERIFY_PACK=PASS scanned={scanned} mismatches=0"}
    raw["row_digests"] = {"relpath": fs.ROW_DIGESTS_RELPATH, "algo": "blake2b-128",
                          "covered_rows": scanned,
                          "coverage": fs.ROW_DIGEST_COVERAGE,
                          "byte_count": len(digests),
                          "sha256": hashlib.sha256(bytes(digests)).hexdigest()}
    atomic_write_bytes(store_root / fs.META_RELPATH,
                       json.dumps(raw, ensure_ascii=False, indent=1).encode())
    fs.StoreMeta.load(store_root)
    release_lock(store_root)   # 回填完成后才删锁（A.2）
    print(f"[verify] meta 回填（status=verified）+ row_digests 落盘, "
          f"总耗时 {time.perf_counter() - t_all:.0f}s", flush=True)
    print(f"VERIFY_PACK=PASS scanned={scanned} mismatches=0")


def cmd_report(args) -> None:
    meta = fs.StoreMeta.load(pathlib.Path(args.store))
    r = meta.raw
    print(f"layout={r['layout']} status={r['status']} scope={r['manifest_scope']}")
    print(f"num_rows={meta.num_rows} num_exec_samples={meta.num_exec_samples} "
          f"num_pos_rows={meta.num_pos_rows} parts={len(meta.parts)}")
    print(f"manifest={meta.manifest_path}\n  sha256={meta.manifest_sha256}")
    print(f"source={meta.source_dataset_root}")
    print(f"packer={r.get('packer')}")
    print(f"verify={r.get('verify')}")
    print(f"row_digests={r.get('row_digests')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="只算贪心切分并打印 part 表")
    p.add_argument("--manifest", required=True)
    p.add_argument("--subset-prefix", type=int, default=None,
                   help="迷你库：只打包 global_episode_idx 连续前缀 [0..K]")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("pack", help="构建三张表（写侧逐帧校验 + 原子落盘）")
    p.add_argument("--source", required=True, help="4task-gl 源库根")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="打包库根")
    p.add_argument("--reader", choices=["decode", "slice"], default="decode")
    p.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--subset-prefix", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-break-lock", action="store_true")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("verify", help="全量写×读对拍（零遗漏）+ row_digests + meta 回填")
    p.add_argument("--store", required=True)
    p.add_argument("--source", default=None, help="缺省取 meta.source_dataset_root")
    p.add_argument("--manifest", default=None, help="缺省取 meta.manifest_path")
    p.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--sample", type=int, default=None,
                   help="抽样档（开发期快检，不得用于交付判定）")
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-break-lock", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("report", help="打印 meta 摘要")
    p.add_argument("--store", required=True)
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
