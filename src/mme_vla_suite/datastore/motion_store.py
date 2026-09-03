"""motion 离线表：格式常量、motion_index 契约、MotionMeta、只读 MotionStore、起点查表公式。

对应 motion-memory-plan.md 第二部分一节 1.1（离线 motion 表格式契约）。体例照 ``framesamp_store.py``：
本文件是格式层唯一实现——打包工具（scripts/dataset/pack_motion_store.py）、FrameSampDataset（S2）、
在线侧（S3）与对拍工具一律从这里 import，绝不复制；``framesamp_store.py`` 一字不动（两套索引公式不同，
帧路按 ``row_of()`` 逐帧、运动路按段内网格，混放会互相污染）。

布局 ``motion-768-grid16-v1``：

    <motion_root>/
    ├── meta/store_meta.json          唯一契约，两阶段写：pack→"packed"、verify→"verified"
    ├── meta/motion_index.json        段基址表（唯一身份来源），store_meta 记其 sha256、加载时现场重算
    ├── meta/row_digests.blake2b.bin  逐行 blake2b-128（verify 产出）
    ├── meta/pack_progress.jsonl      断点续跑记录
    └── motion_token.f32.bin          (rows, 768) f32 裸字节；行序 = 清单 canonical_order 逐 episode，
                                      每 episode 先 demo 段后 exec 段，段内按网格序 0,16,32,… 升序

窗口口径（红线 14 / 15 冻结）：起点钉在段内绝对网格 ``0, 16, 32, …``（``GRID_ORIGIN = segment_start``，
demo / exec 各自起算、窗口不跨段），前视 33 帧 ``[起点, 起点+32]``，exec 段不截尾
``num_chunks = max(0, 段帧数 − 32)``、``num_grid = len(range(0, num_chunks, 16))``。

读取实现：表最大也只有几十 MiB（4env400ep 全量 78.45 MiB），每进程整表 ``np.fromfile`` 读入即可，
仍照抄 FrameSampStore 的三条纪律——记录 ``owner_pid``、``__reduce__`` 直接 raise 禁 pickle、跨进程懒构造
（懒构造由 FrameSampDataset 负责）。本模块不 import 任何 training/model 模块（单向依赖）。

沿用 framesamp 的禁 ``.npy`` 容器定论：一律裸 ``.bin`` + meta 声明 dtype。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import pathlib

import numpy as np

from mme_vla_suite.datastore.manifest import load_manifest

logger = logging.getLogger(__name__)

# ── 布局常量 ──────────────────────────────────────────────────────────────────
LAYOUT = "motion-768-grid16-v1"
META_SCHEMA = 1
INDEX_SCHEMA = 1

MOTION_KEY = "motion_token"
MOTION_ROW_SHAPE = (768,)
MOTION_DTYPE = np.float32
MOTION_ROW_BYTES = 768 * 4              # 3,072

WINDOW_FRAMES = 33                      # 与 MotionJEPA 的 WINDOW = 4*K+1 同值，verify 时核对
GRID_STRIDE = 16                        # 段内绝对网格步长；加载时须 == yaml motion.stride
GRID_ORIGIN = "segment_start"           # 网格锚点：每段各自从段起点起算，两段互不延续
WINDOW_DIRECTION = "forward"            # 前视：窗口 = [起点, 起点+32]
TRUNCATION_POLICY = "none"              # exec 段不截尾：num_chunks = max(0, 段帧数 − 32)
FRAME_SIZE = 256                        # 原始帧边长 h == w；离线抽取输入与在线 add_buffer 入库校验同用
LAYOUT_GRID_SUFFIX = f"grid{GRID_STRIDE}"

BYTE_ORDER = "little"
ARRAY_ORDER = "C"

META_RELPATH = "meta/store_meta.json"
INDEX_RELPATH = "meta/motion_index.json"
LOCK_RELPATH = "meta/pack.lock"
PROGRESS_RELPATH = "meta/pack_progress.jsonl"
ROW_DIGESTS_RELPATH = "meta/row_digests.blake2b.bin"
MOTION_TABLE_RELPATH = "motion_token.f32.bin"

ROW_DIGEST_BYTES = 16                   # blake2b-128
ROW_DIGEST_COVERAGE = "motion_token 行原始位串（store 侧字节）"

SEGMENTS = ("demo", "exec")             # 行序内 episode 内的段顺序

if not LAYOUT.endswith(f"-{LAYOUT_GRID_SUFFIX}-v1"):
    raise RuntimeError(f"LAYOUT {LAYOUT!r} 的 grid 后缀与 GRID_STRIDE={GRID_STRIDE} 不符")


# ── 网格公式（写读共用；训练侧 / 在线侧 / oracle 三方同式）───────────────────────


def seg_num_chunks(seg_len: int) -> int:
    """段内可作起点的帧数：exec 段不截尾，``max(0, 段帧数 − (WINDOW_FRAMES − 1))``。"""
    return max(0, int(seg_len) - (WINDOW_FRAMES - 1))


def seg_num_grid(seg_len: int) -> int:
    """段内网格起点数 = ``len(range(0, num_chunks, GRID_STRIDE))``。"""
    return len(range(0, seg_num_chunks(seg_len), GRID_STRIDE))


def segment_lengths(num_timesteps: int, exec_start_idx: int) -> dict[str, int]:
    """demo 段帧数 = exec_start_idx；exec 段帧数 = num_timesteps − exec_start_idx。"""
    es = int(exec_start_idx)
    nt = int(num_timesteps)
    if not 0 <= es <= nt:
        raise ValueError(f"exec_start_idx={es} 越界（num_timesteps={nt}）")
    return {"demo": es, "exec": nt - es}


def segment_grid_starts(seg_len: int) -> list[int]:
    """段内网格起点偏移列表 ``[0, 16, 32, …]``（长度 = seg_num_grid）。"""
    return list(range(0, seg_num_chunks(seg_len), GRID_STRIDE))


def visible_motion_rows(entry: "IndexEntry", t: int) -> tuple[np.ndarray, np.ndarray]:
    """给定 episode 的 index 条目与当前样本全域帧号 t，返回 (rows, frames)：

    - demo 段：``s = 16m``、合法条件 ``s + 32 ≤ es − 1``（整段已见、与 t 无关），全域起点 ``f = s``；
    - exec 段：``u = 16m``、合法条件 ``u + 32 ≤ t − es``，全域起点 ``f = es + u``；
    合并后按 f 升序。rows 是 motion 表全局行号（int64），frames 是全域起点帧号（int64）。
    预算（motion.budget）上限检查不在这里做——由调用方按配置 raise，本函数只负责集合本身。
    """
    es = int(entry.exec_start_idx)
    t = int(t)
    if not es <= t < entry.num_timesteps:
        raise ValueError(
            f"t={t} 不在 exec 段 [{es}, {entry.num_timesteps}) 内（g={entry.g}）")
    rows: list[int] = []
    frames: list[int] = []
    if entry.demo.row_base is not None:
        for m in range(entry.demo.num_grid):
            s = GRID_STRIDE * m
            if s + (WINDOW_FRAMES - 1) <= es - 1:
                rows.append(entry.demo.row_base + m)
                frames.append(s)
    if entry.exec.row_base is not None:
        for m in range(entry.exec.num_grid):
            u = GRID_STRIDE * m
            if u + (WINDOW_FRAMES - 1) <= t - es:
                rows.append(entry.exec.row_base + m)
                frames.append(es + u)
    r = np.asarray(rows, dtype=np.int64)
    f = np.asarray(frames, dtype=np.int64)
    if len(f) > 1 and not np.all(np.diff(f) > 0):
        order = np.argsort(f, kind="stable")
        r, f = r[order], f[order]
    return r, f


def max_visible_count(entry: "IndexEntry") -> int:
    """该 episode 任一 exec 样本的合法起点数上界（t = 最后一帧时取到）。"""
    r, _ = visible_motion_rows(entry, entry.num_timesteps - 1)
    return int(len(r))


# ── 通用小件 ──────────────────────────────────────────────────────────────────


def sha256_file(p: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def task_of_h5(h5_file: str) -> str:
    """``record_dataset_<Task>.h5`` → ``<Task>``。"""
    name = os.path.basename(h5_file)
    if not (name.startswith("record_dataset_") and name.endswith(".h5")):
        raise ValueError(f"h5 文件名不符合 record_dataset_<Task>.h5: {h5_file}")
    return name[len("record_dataset_"):-len(".h5")]


def segment_key(h5_file: str, raw_ep_idx: int, segment: str) -> str:
    """段工作项键：``<Task>_ep<j>_<exec|demo>``（与 MotionJEPA 抽取器同构）。"""
    if segment not in SEGMENTS:
        raise ValueError(f"segment 非法: {segment!r}")
    return f"{task_of_h5(h5_file)}_ep{int(raw_ep_idx)}_{segment}"


# ── motion_index.json ─────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SegmentInfo:
    row_base: int | None        # 该段首行在 motion 表中的全局行号；num_grid == 0 时为 None
    num_grid: int
    num_chunks: int
    seg_len: int


@dataclasses.dataclass(frozen=True)
class IndexEntry:
    g: int
    h5_file: str
    raw_ep_idx: int
    num_timesteps: int
    exec_start_idx: int
    demo: SegmentInfo
    exec: SegmentInfo


def build_index_entries(manifest: dict) -> list[IndexEntry]:
    """按清单 canonical 序（episodes 列表序 = global_episode_idx 序）算出每 episode 的段基址表。

    行序契约：逐 episode，每 episode 先 demo 后 exec，段内网格升序。这是 motion 表唯一的行序定义，
    打包器写、MotionMeta 校验、oracle 重算三方都从这里派生。
    """
    entries: list[IndexEntry] = []
    cursor = 0
    for g, ep in enumerate(manifest["episodes"]):
        if int(ep["global_episode_idx"]) != g:
            raise ValueError(f"清单 episodes 序错乱: 第 {g} 项 global_episode_idx={ep['global_episode_idx']}")
        lens = segment_lengths(ep["num_timesteps"], ep["exec_start_idx"])
        segs: dict[str, SegmentInfo] = {}
        for seg in SEGMENTS:
            L = lens[seg]
            ng = seg_num_grid(L)
            segs[seg] = SegmentInfo(row_base=(cursor if ng > 0 else None), num_grid=ng,
                                    num_chunks=seg_num_chunks(L), seg_len=L)
            cursor += ng
        entries.append(IndexEntry(
            g=g, h5_file=str(ep["h5_file"]), raw_ep_idx=int(ep["raw_ep_idx"]),
            num_timesteps=int(ep["num_timesteps"]), exec_start_idx=int(ep["exec_start_idx"]),
            demo=segs["demo"], exec=segs["exec"]))
    return entries


def index_totals(entries: list[IndexEntry]) -> dict[str, int]:
    exec_rows = sum(e.exec.num_grid for e in entries)
    demo_rows = sum(e.demo.num_grid for e in entries)
    return {"rows": exec_rows + demo_rows, "exec_rows": exec_rows, "demo_rows": demo_rows}


def index_payload(manifest: dict, entries: list[IndexEntry], *, mj_repo_commit: str) -> dict:
    """motion_index.json 的完整内容（不含任何运行时字段，可重算比对）。"""
    def seg(s: SegmentInfo) -> dict:
        return {"row_base": s.row_base, "num_grid": s.num_grid, "num_chunks": s.num_chunks,
                "seg_len": s.seg_len}
    return {
        "schema": INDEX_SCHEMA,
        "layout": LAYOUT,
        "grid_stride": GRID_STRIDE,
        "window_frames": WINDOW_FRAMES,
        "grid_origin": GRID_ORIGIN,
        "window_direction": WINDOW_DIRECTION,
        "truncation_policy": TRUNCATION_POLICY,
        "row_order": "canonical_order episodes; per episode demo then exec; grid ascending",
        "entries": [
            {"g": e.g, "h5_file": e.h5_file, "raw_ep_idx": e.raw_ep_idx,
             "num_timesteps": e.num_timesteps, "exec_start_idx": e.exec_start_idx,
             "demo": seg(e.demo), "exec": seg(e.exec)}
            for e in entries],
        "totals": index_totals(entries),
        "manifest_sha256": manifest["sha256"],
        "mj_repo_commit": mj_repo_commit,
    }


def parse_index(raw: dict, ctx: str = "motion_index.json") -> list[IndexEntry]:
    """解析并逐项校验 motion_index.json：常量三方同值、row_base 连续、num_grid 符合公式、totals 自洽。"""
    def need(k: str):
        if k not in raw:
            raise ValueError(f"{ctx} 缺字段 {k!r}")
        return raw[k]
    if int(need("schema")) != INDEX_SCHEMA:
        raise ValueError(f"{ctx} schema={raw['schema']} != {INDEX_SCHEMA}")
    if need("layout") != LAYOUT:
        raise ValueError(f"{ctx} layout={raw['layout']!r} != {LAYOUT!r}")
    for key, want in (("grid_stride", GRID_STRIDE), ("window_frames", WINDOW_FRAMES),
                      ("grid_origin", GRID_ORIGIN), ("window_direction", WINDOW_DIRECTION),
                      ("truncation_policy", TRUNCATION_POLICY)):
        if need(key) != want:
            raise ValueError(f"{ctx} {key}={raw[key]!r} != 格式常量 {want!r}")
    entries: list[IndexEntry] = []
    cursor = 0
    for g, e in enumerate(need("entries")):
        if int(e["g"]) != g:
            raise ValueError(f"{ctx} entries 序错乱: 第 {g} 项 g={e['g']}")
        lens = segment_lengths(e["num_timesteps"], e["exec_start_idx"])
        segs: dict[str, SegmentInfo] = {}
        for seg in SEGMENTS:
            s = e[seg]
            L = lens[seg]
            if int(s["seg_len"]) != L:
                raise ValueError(f"{ctx} g={g} {seg}.seg_len={s['seg_len']} != {L}")
            if int(s["num_chunks"]) != seg_num_chunks(L) or int(s["num_grid"]) != seg_num_grid(L):
                raise ValueError(
                    f"{ctx} g={g} {seg} num_chunks/num_grid={s['num_chunks']}/{s['num_grid']} "
                    f"!= 公式 {seg_num_chunks(L)}/{seg_num_grid(L)}")
            ng = int(s["num_grid"])
            if ng == 0:
                if s["row_base"] is not None:
                    raise ValueError(f"{ctx} g={g} {seg} num_grid=0 但 row_base={s['row_base']}")
            else:
                if int(s["row_base"]) != cursor:
                    raise ValueError(
                        f"{ctx} g={g} {seg} row_base={s['row_base']} != 累计 {cursor}（不连续）")
            segs[seg] = SegmentInfo(row_base=(cursor if ng > 0 else None), num_grid=ng,
                                    num_chunks=int(s["num_chunks"]), seg_len=L)
            cursor += ng
        entries.append(IndexEntry(
            g=g, h5_file=str(e["h5_file"]), raw_ep_idx=int(e["raw_ep_idx"]),
            num_timesteps=int(e["num_timesteps"]), exec_start_idx=int(e["exec_start_idx"]),
            demo=segs["demo"], exec=segs["exec"]))
    totals = need("totals")
    calc = index_totals(entries)
    if {k: int(totals[k]) for k in calc} != calc:
        raise ValueError(f"{ctx} totals={totals} != 现算 {calc}")
    if cursor != calc["rows"]:
        raise ValueError(f"{ctx} 行游标 {cursor} != totals.rows {calc['rows']}")
    return entries


def check_index_against_manifest(entries: list[IndexEntry], manifest: dict) -> None:
    """逐 episode 身份互校：g / h5_file / raw_ep_idx / num_timesteps / exec_start_idx 五字段。"""
    eps = manifest["episodes"]
    if len(eps) != len(entries):
        raise ValueError(f"motion index 条目数 {len(entries)} != 清单 episode 数 {len(eps)}")
    for e, ep in zip(entries, eps):
        ident = (int(ep["global_episode_idx"]), str(ep["h5_file"]), int(ep["raw_ep_idx"]),
                 int(ep["num_timesteps"]), int(ep["exec_start_idx"]))
        mine = (e.g, e.h5_file, e.raw_ep_idx, e.num_timesteps, e.exec_start_idx)
        if ident != mine:
            raise ValueError(f"motion index 与清单身份不符 g={e.g}: index={mine} manifest={ident}")


# ── MotionMeta ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class MotionMeta:
    """store_meta.json 的结构化视图 + 逐项 fail-loud 校验（含 motion_index.json 现场 sha256）。"""

    root: pathlib.Path
    raw: dict
    status: str
    num_rows: int
    manifest_sha256: str
    manifest_path: str
    motion_index_sha256: str
    table_sha256: str | None
    entries: tuple[IndexEntry, ...]
    provenance: dict

    @classmethod
    def load(cls, store_root: str | pathlib.Path) -> "MotionMeta":
        root = pathlib.Path(store_root)
        meta_path = root / META_RELPATH
        if not meta_path.is_file():
            raise FileNotFoundError(f"motion 库 meta 缺失: {meta_path}")
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"store_meta.json 损坏不可解析: {meta_path}: {e}") from e

        def need(key: str):
            if key not in raw:
                raise ValueError(f"store_meta.json 缺字段 {key!r}: {meta_path}")
            return raw[key]

        if int(need("schema")) != META_SCHEMA:
            raise ValueError(f"schema={raw['schema']} != {META_SCHEMA}")
        if need("layout") != LAYOUT:
            raise ValueError(f"layout 不符: {raw['layout']!r} != {LAYOUT!r}")
        if need("byte_order") != BYTE_ORDER or need("array_order") != ARRAY_ORDER:
            raise ValueError("byte_order/array_order 与格式常量不符")
        for key, want in (("grid_stride", GRID_STRIDE), ("window_frames", WINDOW_FRAMES),
                          ("grid_origin", GRID_ORIGIN), ("window_direction", WINDOW_DIRECTION),
                          ("truncation_policy", TRUNCATION_POLICY), ("frame_size", FRAME_SIZE)):
            if need(key) != want:
                raise ValueError(f"store_meta {key}={raw[key]!r} != 格式常量 {want!r}")
        status = need("status")
        if status not in ("packed", "verified"):
            raise ValueError(f"status 非法: {status!r}")
        t = need("tables").get(MOTION_KEY)
        if t is None:
            raise ValueError(f"tables 缺 {MOTION_KEY}")
        if tuple(t["row_shape"]) != MOTION_ROW_SHAPE or t["dtype"] != "float32" \
                or int(t["row_bytes"]) != MOTION_ROW_BYTES or t["relpath"] != MOTION_TABLE_RELPATH:
            raise ValueError(f"tables[{MOTION_KEY}] 形制与格式常量不符: {t}")
        num_rows = int(need("num_rows"))
        if int(t["num_rows"]) != num_rows or int(t["byte_count"]) != num_rows * MOTION_ROW_BYTES:
            raise ValueError("tables.num_rows / byte_count 与 num_rows 不符")

        # motion_index.json：现场重算 sha256，不等立即拒绝（R23 / 红线：唯一身份来源）
        idx_path = root / INDEX_RELPATH
        if not idx_path.is_file():
            raise FileNotFoundError(f"motion_index.json 缺失: {idx_path}")
        idx_bytes = idx_path.read_bytes()
        got = sha256_bytes(idx_bytes)
        if got != need("motion_index_sha256"):
            raise ValueError(
                f"motion_index.json sha256 不符（被改动？）: 现算 {got[:16]}… != "
                f"store_meta 记 {raw['motion_index_sha256'][:16]}…")
        entries = parse_index(json.loads(idx_bytes.decode("utf-8")), ctx=str(idx_path))
        totals = index_totals(entries)
        if totals["rows"] != num_rows:
            raise ValueError(f"motion_index totals.rows={totals['rows']} != num_rows {num_rows}")
        if json.loads(idx_bytes.decode("utf-8")).get("manifest_sha256") != need("manifest_sha256"):
            raise ValueError("motion_index.json 的 manifest_sha256 与 store_meta 不符")
        provenance = need("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("provenance 须为对象")
        return cls(root=root, raw=raw, status=status, num_rows=num_rows,
                   manifest_sha256=need("manifest_sha256"), manifest_path=need("manifest_path"),
                   motion_index_sha256=raw["motion_index_sha256"], table_sha256=t.get("sha256"),
                   entries=tuple(entries), provenance=provenance)


def run_fast_checks(meta: MotionMeta, *, manifest_path: str | None = None) -> None:
    """fast 档：清单 sha256 现场重算 + index 与清单逐 episode 身份互校 + 表文件大小。"""
    manifest = load_manifest(manifest_path or meta.manifest_path)
    if manifest["sha256"] != meta.manifest_sha256:
        raise ValueError(
            f"清单指纹不符: motion store_meta 记 {meta.manifest_sha256[:16]}… != "
            f"现场 {manifest['sha256'][:16]}…")
    check_index_against_manifest(list(meta.entries), manifest)
    table = meta.root / MOTION_TABLE_RELPATH
    if not table.is_file():
        raise FileNotFoundError(f"motion 表缺失: {table}")
    st = table.stat().st_size
    if st != meta.num_rows * MOTION_ROW_BYTES:
        raise ValueError(f"motion 表大小不符: {st} != {meta.num_rows} × {MOTION_ROW_BYTES}")


def run_full_checks(meta: MotionMeta) -> None:
    """full 档：整表 sha256 对 meta 比对。"""
    if meta.table_sha256 is None:
        raise ValueError("store_meta.tables.motion_token 缺 sha256")
    got = sha256_file(meta.root / MOTION_TABLE_RELPATH)
    if got != meta.table_sha256:
        raise ValueError(f"motion 表 sha256 不符（full 档）: {got[:16]}…")


def require_verified(meta: MotionMeta) -> None:
    """status != "verified" 即 raise；显式设 MMEVLA_MOTION_ALLOW_UNVERIFIED=1 可放行但必打 WARNING。"""
    if meta.status == "verified":
        return
    if os.environ.get("MMEVLA_MOTION_ALLOW_UNVERIFIED") == "1":
        logger.warning("motion 库 status=%r 未通过 verify，因 MMEVLA_MOTION_ALLOW_UNVERIFIED=1 放行"
                       "——仅限迷你库/开发期: %s", meta.status, meta.root)
        return
    raise RuntimeError(
        f"motion 库未通过 verify（status={meta.status!r}）: {meta.root}；"
        f"先跑 pack_motion_store.py verify，或（仅开发期）设 MMEVLA_MOTION_ALLOW_UNVERIFIED=1 放行")


def require_no_pack_lock(store_root: str | pathlib.Path) -> None:
    lock = pathlib.Path(store_root) / LOCK_RELPATH
    if lock.exists():
        raise RuntimeError(f"motion 库存在 pack.lock（打包或 verify 进行中/异常残留）: {lock}")


def check_same_source(frame_manifest_sha256: str, motion_meta: MotionMeta,
                      manifest: dict) -> None:
    """双 store 同源硬闸：framesamp 与 motion 两库必须绑同一份清单，index 逐 episode 身份相同。"""
    if frame_manifest_sha256 != motion_meta.manifest_sha256:
        raise ValueError(
            f"framesamp 库与 motion 库绑定的清单不同: {frame_manifest_sha256[:16]}… != "
            f"{motion_meta.manifest_sha256[:16]}…（两个各自 verified 的库被串配）")
    if manifest["sha256"] != motion_meta.manifest_sha256:
        raise ValueError("现场清单与 motion 库 manifest_sha256 不符")
    check_index_against_manifest(list(motion_meta.entries), manifest)


# ── MotionStore（只读，整表进程内）───────────────────────────────────────────


class MotionStore:
    """整表常驻进程内的只读访问层：构造即 ``np.fromfile`` 整表读入并记录 ``_owner_pid``；
    禁止 pickle 跨进程携带（由 FrameSampDataset 在 worker 内按 pid 懒构造）。"""

    def __init__(self, store_root: str | pathlib.Path, *, meta: MotionMeta | None = None,
                 manifest_path: str | None = None, verify_level: str = "fast"):
        if verify_level not in ("fast", "full"):
            raise ValueError(f"verify_level 非法: {verify_level!r}（∈ fast|full）")
        self._root = pathlib.Path(store_root)
        self._meta = meta if meta is not None else MotionMeta.load(self._root)
        run_fast_checks(self._meta, manifest_path=manifest_path)
        if verify_level == "full":
            run_full_checks(self._meta)
        table = np.fromfile(self._root / MOTION_TABLE_RELPATH, dtype=MOTION_DTYPE)
        if table.size != self._meta.num_rows * MOTION_ROW_SHAPE[0]:
            raise ValueError("motion 表元素数与 num_rows 不符")
        table.shape = (self._meta.num_rows,) + MOTION_ROW_SHAPE
        self._table = table
        self._owner_pid = os.getpid()

    @property
    def meta(self) -> MotionMeta:
        return self._meta

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    @property
    def num_rows(self) -> int:
        return self._meta.num_rows

    @property
    def table(self) -> np.ndarray:
        return self._table

    @property
    def entries(self) -> tuple[IndexEntry, ...]:
        return self._meta.entries

    def __reduce__(self):
        raise TypeError(
            "MotionStore 持有进程内整表，禁止 pickle 跨进程携带；"
            "由 FrameSampDataset 在 worker 内按 pid 懒构造")

    def rows(self, rows) -> np.ndarray:
        """按全局行号取 (n, 768) f32（进程内副本切片）。"""
        r = np.asarray(rows, dtype=np.int64)
        if len(r) and (int(r.min()) < 0 or int(r.max()) >= self._meta.num_rows):
            raise IndexError(f"motion 行号越界: [{r.min()}, {r.max()}] ∉ [0, {self._meta.num_rows})")
        return self._table[r]

    def close(self) -> None:
        self._table = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
