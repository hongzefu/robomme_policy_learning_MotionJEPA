"""framesamp packed 特征库：格式常量、StoreMeta 契约、只读 FrameSampStore。

对应 v2-framesamp-restructure-plan.md 的 A.1 / B.2（本文件是格式层唯一实现，
打包工具、FrameSampDataset、对拍工具一律从这里 import，绝不复制）：

- 布局 ``framesamp-4x4-v1``：三张表只存 framesample 真正用到的键——
  image_emb_4x4 按行铺进 32 个连续 part（(rows,16,2048) bf16 裸字节，按 episode
  边界切分，一个样本的 32 帧必落同一 part）；pos_emb_4x4 / state_emb 两张小表
  各一个 .bin（训练期整表读入进程内存，无 NFS mmap）。
- 行号公式 ``row_of()`` 写读共用，物理上不可分叉；t 是全 timestep 域帧号
  （含 demo 前缀）。
- 读侧 fail-loud：packed 模式任何校验不过直接 raise，绝不回退散 npy。
- 本模块不 import 任何 training/model 模块（单向依赖，B.1）。

fd/mmap 生命周期契约（B.2）：FrameSampStore 构造即打开全部 part fd 并整表读入
两张小表、记录 _owner_pid；跨进程携带（pickle）不被支持——懒加载与 pid 校验由
FrameSampDataset 负责（S3），Store 只提供 close() 与 owner_pid。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import pathlib

import ml_dtypes
import numpy as np

from mme_vla_suite.datastore.manifest import load_manifest

logger = logging.getLogger(__name__)

# ── 布局常量（数据格式常量；源 npy 布局的 SOURCE_* 偏移由 2026-08 探针实测固化，见 docs/dataset-build-doc/4task-gl-framesamp）──
LAYOUT = "framesamp-4x4-v1"
META_SCHEMA = 1

IMAGE_KEY = "image_emb_4x4"
POS_KEY = "pos_emb_4x4"
STATE_KEY = "state_emb"

IMAGE_ROW_SHAPE = (16, 2048)          # 每帧（=每行），bf16
IMAGE_DTYPE = ml_dtypes.bfloat16
IMAGE_ROW_BYTES = 16 * 2048 * 2       # 65,536

POS_ROW_SHAPE = (16, 768)             # 每 t 一行，f32（pos 是 t 的纯函数）
POS_DTYPE = np.float32
POS_ROW_BYTES = 16 * 768 * 4          # 49,152

STATE_ROW_SHAPE = (8,)                # 每全局行一行，f32
STATE_DTYPE = np.float32
STATE_ROW_BYTES = 8 * 4               # 32

BYTE_ORDER = "little"
ARRAY_ORDER = "C"
BF16_ENCODING = "ml_dtypes.bfloat16 (1s+8e+7m)"

TARGET_PARTS = 32                     # 贪心切分目标 part 数（阈值 ceil(rows/32)）

META_RELPATH = "meta/store_meta.json"
LOCK_RELPATH = "meta/pack.lock"
PROGRESS_RELPATH = "meta/pack_progress.jsonl"
ROW_DIGESTS_RELPATH = "meta/row_digests.blake2b.bin"
POS_TABLE_RELPATH = "pos_emb_4x4.f32.bin"
STATE_TABLE_RELPATH = "state_emb.f32.bin"
IMAGE_PART_DIR = "image_emb_4x4"

ROW_DIGEST_BYTES = 16                 # blake2b-128
ROW_DIGEST_COVERAGE = "image‖pos‖state 三键原始位串拼接（store 侧字节）"

# 源 npy 布局常量（4task-gl features/episode_{g}/token_emb_{t}.npy，
# np.save 的 pickle dict；120/120 文件大小一致、60/60 memcmp、独立复现 9/9 实测）
SOURCE_NPY_SIZE = 602951
SOURCE_IMAGE_OFFSET = 262595          # image_emb_4x4 (1,16,2048) bf16 数据段起点
SOURCE_POS_OFFSET = 541352            # pos_emb_4x4  (1,16,768)  f32  数据段起点
SOURCE_STATE_OFFSET = 602906          # state_emb    (8,)        f32  数据段起点

_HEADTAIL = 1 << 20                   # head_tail_digest 覆盖首尾各 1 MiB


def row_of(total_sample_offset: int, t: int) -> int:
    """全局行号公式（写读共用；t 为全 timestep 域帧号，含 demo 前缀）。"""
    return int(total_sample_offset) + int(t)


def build_exec_lookup(manifest: dict, *, num_episodes: int | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从清单派生执行样本查表数组（O(1) 换算，禁止目录序）。

    返回 (_epis_of, _step_of, _row_base)：
      _epis_of[idx] = g；_step_of[idx] = exec_start_idx + k（⚠ 必须带
      exec_start_idx——Video* 任务漏掉即错 66–216 帧）；_row_base[g] =
      total_sample_offset。身份校验一律显式 raise（R6，禁 assert）。
    num_episodes：只取 episodes 连续前缀 [0..num_episodes)（subset 迷你库——
    前缀保证全局行号即物理行号、偏移原值可用，A.1）；None = 全量。
    """
    episodes = manifest["episodes"]
    if num_episodes is not None:
        episodes = episodes[:num_episodes]
        n = sum(int(ep["exec_samples"]) for ep in episodes)
    else:
        n = int(manifest["totals"]["exec_samples"])
    epis_of = np.empty(n, np.int32)
    step_of = np.empty(n, np.int32)
    row_base = np.empty(len(episodes), np.int64)
    cursor = 0
    for g, ep in enumerate(episodes):
        if ep["global_episode_idx"] != g:
            raise ValueError(
                f"清单 episodes 序错乱: 第 {g} 项 global_episode_idx={ep['global_episode_idx']}")
        if ep["exec_sample_offset"] != cursor:
            raise ValueError(
                f"清单 exec_sample_offset 不连续: episode {g} 记 "
                f"{ep['exec_sample_offset']} != 累计 {cursor}")
        k = int(ep["exec_samples"])
        epis_of[cursor:cursor + k] = g
        step_of[cursor:cursor + k] = int(ep["exec_start_idx"]) + np.arange(k, dtype=np.int32)
        row_base[g] = int(ep["total_sample_offset"])
        cursor += k
    if cursor != n:
        raise ValueError(f"清单 exec_samples 总和 {cursor} != totals.exec_samples {n}")
    return epis_of, step_of, row_base


def mean_sampled_frames(manifest: dict, max_frames: int = 32) -> float:
    """执行样本均值选帧数 Σ min(step+1, max_frames) / N（S7.5：读盘字节帐现场推导，
    替代硬编码；step 含 exec_start_idx 偏移——Video* 任务首样本即满长）。

    真实清单实测 = 30.996（max_frames=32）；每样本读盘均值由调用侧按 backend 折算：
    legacy = pkl + mean_frames×602,951（整包 npy）；packed = pkl + mean_frames×65,536
    （image 行；pos/state 走进程内小表不走盘）。
    """
    total = 0
    n = 0
    for ep in manifest["episodes"]:
        k = int(ep["exec_samples"])
        s0 = int(ep["exec_start_idx"])
        # steps s0..s0+k-1；min(step+1, m) 分段闭式求和，避免 40 万次逐样本循环
        m = max_frames
        if s0 + 1 >= m:
            total += k * m
        elif s0 + k <= m:
            total += (s0 + 1 + s0 + k) * k // 2
        else:
            j = m - 1 - s0                    # 前 j 个样本 step+1 < m
            total += (s0 + 1 + m - 1) * j // 2 + (k - j) * m
        n += k
    if n == 0:
        raise ValueError("清单无执行样本")
    return total / n


SOURCE_PKL_BYTES_FLOOR = 395_440   # data/{idx}.pkl 下界（内嵌变长字符串，1.2 节实测）


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def headtail_digest(p: pathlib.Path) -> tuple[int, str, bool]:
    """(字节数, 首尾各 1 MiB 的 blake2b-128, 是否覆盖全文件)。

    与 scripts/training/g0/check_baseline_env.py 的 _headtail_digest 同口径；
    文件 ≤ 2 MiB 时覆盖全文件（full_covered=True）。
    """
    size = p.stat().st_size
    h = hashlib.blake2b(digest_size=16)
    full = size <= 2 * _HEADTAIL
    with p.open("rb") as f:
        if full:
            h.update(f.read())
        else:
            h.update(f.read(_HEADTAIL))
            f.seek(size - _HEADTAIL)
            h.update(f.read(_HEADTAIL))
    return size, h.hexdigest(), full


# ── StoreMeta ─────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class PartInfo:
    index: int
    path: str                 # 相对 store 根
    start_row: int
    num_rows: int
    bytes: int
    sha256: str
    head_tail_digest: str
    full_covered: bool


@dataclasses.dataclass(frozen=True)
class StoreMeta:
    """store_meta.json 的结构化视图 + 逐项 fail-loud 结构校验。"""

    root: pathlib.Path
    raw: dict
    status: str
    num_rows: int
    num_exec_samples: int
    num_pos_rows: int
    manifest_sha256: str
    manifest_path: str
    source_dataset_root: str
    manifest_scope: str                    # "full" | "subset"
    subset_episodes: list[int] | None
    parts: tuple[PartInfo, ...]

    @classmethod
    def load(cls, store_root: str | pathlib.Path) -> "StoreMeta":
        root = pathlib.Path(store_root)
        meta_path = root / META_RELPATH
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"packed 库 meta 缺失: {meta_path}（packed 模式绝不回退散 npy）")
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"store_meta.json 损坏不可解析: {meta_path}: {e}") from e

        def need(key: str):
            if key not in raw:
                raise ValueError(f"store_meta.json 缺字段 {key!r}: {meta_path}")
            return raw[key]

        if need("layout") != LAYOUT:
            raise ValueError(f"layout 不符: {raw['layout']!r} != {LAYOUT!r}")
        if need("byte_order") != BYTE_ORDER or need("array_order") != ARRAY_ORDER:
            raise ValueError("byte_order/array_order 与格式常量不符")
        if need("bf16_encoding") != BF16_ENCODING:
            raise ValueError(f"bf16_encoding 不符: {raw['bf16_encoding']!r}")
        status = need("status")
        if status not in ("packed", "verified"):
            raise ValueError(f"status 非法: {status!r}")
        scope = need("manifest_scope")
        if scope not in ("full", "subset"):
            raise ValueError(f"manifest_scope 非法: {scope!r}")
        subset = raw.get("subset_episodes")
        if scope == "subset":
            if not subset or "mini_manifest_sha256" not in raw:
                raise ValueError("subset 库必须带 subset_episodes 与 mini_manifest_sha256")
            if subset != list(range(len(subset))):
                raise ValueError(
                    f"subset_episodes 必须是 global_episode_idx 连续前缀 [0..k]: {subset[:8]}…")
        tables = need("tables")
        for key, row_shape, dtype_name, row_bytes in (
                (IMAGE_KEY, IMAGE_ROW_SHAPE, "bfloat16", IMAGE_ROW_BYTES),
                (POS_KEY, POS_ROW_SHAPE, "float32", POS_ROW_BYTES),
                (STATE_KEY, STATE_ROW_SHAPE, "float32", STATE_ROW_BYTES)):
            t = tables.get(key)
            if t is None:
                raise ValueError(f"tables 缺 {key}")
            if tuple(t["row_shape"]) != row_shape or t["dtype"] != dtype_name \
                    or t["row_bytes"] != row_bytes:
                raise ValueError(f"tables[{key}] 形制与格式常量不符: {t}")
        num_rows = int(need("num_rows"))
        num_pos_rows = int(need("num_pos_rows"))
        parts_raw = need("parts")
        parts = []
        cursor = 0
        for i, p in enumerate(parts_raw):
            if p["index"] != i:
                raise ValueError(f"parts[{i}].index={p['index']} 序错乱")
            if p["start_row"] != cursor:
                raise ValueError(
                    f"parts 行区间不连续: part {i} start_row={p['start_row']} != 累计 {cursor}")
            if p["bytes"] != p["num_rows"] * IMAGE_ROW_BYTES:
                raise ValueError(f"part {i} bytes 与 num_rows×row_bytes 不符")
            parts.append(PartInfo(
                index=i, path=p["path"], start_row=int(p["start_row"]),
                num_rows=int(p["num_rows"]), bytes=int(p["bytes"]),
                sha256=p["sha256"], head_tail_digest=p["head_tail_digest"],
                full_covered=bool(p["full_covered"])))
            cursor += int(p["num_rows"])
        if cursor != num_rows:
            raise ValueError(f"parts 覆盖行数 {cursor} != num_rows {num_rows}"
                             "（未连续覆盖声明行区间）")
        return cls(
            root=root, raw=raw, status=status, num_rows=num_rows,
            num_exec_samples=int(need("num_exec_samples")),
            num_pos_rows=num_pos_rows,
            manifest_sha256=need("manifest_sha256"),
            manifest_path=need("manifest_path"),
            source_dataset_root=need("source_dataset_root"),
            manifest_scope=scope,
            subset_episodes=list(subset) if subset else None,
            parts=tuple(parts),
        )


# ── 校验档位（B.2）───────────────────────────────────────────────────────────


def run_fast_checks(meta: StoreMeta, *, manifest_path: str | None = None,
                    source_root: str | None = None,
                    sample_indices: tuple[int, int] | None = None) -> None:
    """fast 档（默认，每进程懒构造时执行）；任何不过直接 raise。

    覆盖：layout（StoreMeta.load 已钉）/ manifest_sha256 现场重算比对 /
    parts 连续覆盖（StoreMeta.load 已钉）+ 每 part 存在且 st_size == meta.bytes /
    抽 1 个 part 首尾 1 MiB 与 head_tail_digest 复验 / 抽 1 条 source_spot 复验源库未动。
    manifest_path / source_root 未给时用 meta 内记录值（双根契约的 env 覆盖在
    分派层解析后传入）。sample_indices=(part_idx, spot_idx) 供测试钉死抽样点，
    默认按 pid 轮转。
    """
    manifest = load_manifest(manifest_path or meta.manifest_path)
    if manifest["sha256"] != meta.manifest_sha256:
        raise ValueError(
            f"清单指纹不符: store_meta 记 {meta.manifest_sha256[:16]}… != "
            f"现场 {manifest['sha256'][:16]}…（清单或库已换代，packed 模式拒绝混用）")
    for p in meta.parts:
        f = meta.root / p.path
        if not f.is_file():
            raise FileNotFoundError(f"part 缺失: {f}")
        st = f.stat().st_size
        if st != p.bytes:
            raise ValueError(f"part 大小不符: {f}: st_size={st} != meta {p.bytes}")
    for rel, shape, dtype_bytes in (
            (POS_TABLE_RELPATH, (meta.num_pos_rows,) + POS_ROW_SHAPE, 4),
            (STATE_TABLE_RELPATH, (meta.num_rows,) + STATE_ROW_SHAPE, 4)):
        f = meta.root / rel
        expect = int(np.prod(shape)) * dtype_bytes
        if not f.is_file():
            raise FileNotFoundError(f"小表缺失: {f}")
        if f.stat().st_size != expect:
            raise ValueError(f"小表大小不符: {f}: {f.stat().st_size} != {expect}")
    pi, si = sample_indices if sample_indices is not None \
        else (os.getpid() % len(meta.parts), os.getpid())
    part = meta.parts[pi]
    _, d, _ = headtail_digest(meta.root / part.path)
    if d != part.head_tail_digest:
        raise ValueError(f"part {part.index} head_tail_digest 不符（文件被改动）: {part.path}")
    spot = meta.raw.get("source_spot_sha256", {}).get("entries") or []
    if spot:
        entry = spot[si % len(spot)]
        src = pathlib.Path(source_root or meta.source_dataset_root) / entry["relpath"]
        if not src.is_file():
            raise FileNotFoundError(f"源库抽样文件缺失（源库被移动/删除？）: {src}")
        size, d, _ = headtail_digest(src)
        if size != entry["size"] or d != entry["digest"]:
            raise ValueError(f"源库抽样指纹不符（源库被改动）: {src}")


def run_full_checks(meta: StoreMeta) -> None:
    """full 档：全部 part + 两张小表完整 sha256 对 meta 比对（≈整库一读）。

    能抓「同尺寸中部翻转」。⚠ 禁止在性能 allocation 内执行——这一读会把整个
    store 预热进 page cache（B.2）；只允许独立 preflight 或 S4 verify 同场次。
    """
    for p in meta.parts:
        got = sha256_file(meta.root / p.path)
        if got != p.sha256:
            raise ValueError(f"part sha256 不符（full 档）: {p.path}: {got[:16]}…")
    for key, rel in ((POS_KEY, POS_TABLE_RELPATH), (STATE_KEY, STATE_TABLE_RELPATH)):
        want = meta.raw["tables"][key].get("sha256")
        got = sha256_file(meta.root / rel)
        if got != want:
            raise ValueError(f"小表 sha256 不符（full 档）: {rel}: {got[:16]}…")


# ── 分派层守卫助手（G14；接线在 S3 的 create_data_loader）──────────────────────


def require_verified(meta: StoreMeta) -> None:
    """status != "verified" 即 raise；显式设 MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1
    可放行但必打 WARNING（仅迷你库/开发期可用，R17）。"""
    if meta.status == "verified":
        return
    if os.environ.get("MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED") == "1":
        logger.warning(
            "packed 库 status=%r 未通过全量 verify，因 MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1 "
            "放行——仅限迷你库/开发期，S5 及以上禁止（R17）: %s", meta.status, meta.root)
        return
    raise RuntimeError(
        f"packed 库未通过全量 verify（status={meta.status!r}）: {meta.root}；"
        f"先跑 pack_framesamp_store.py verify，或（仅开发期）设 "
        f"MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1 放行")


def require_no_pack_lock(store_root: str | pathlib.Path) -> None:
    """meta/pack.lock 存在即 raise（打包/verify 进行中，读侧不得消费）。"""
    lock = pathlib.Path(store_root) / LOCK_RELPATH
    if lock.exists():
        raise RuntimeError(
            f"packed 库存在 pack.lock（打包或 verify 进行中/异常残留）: {lock}；"
            f"等其完成，或按打包工具锁协议处置后再消费")


# ── FrameSampStore（只读）────────────────────────────────────────────────────


class FrameSampStore:
    """常驻 fd + 进程内小表的只读访问层。

    - 构造即 os.open 全部 part fd（O_RDONLY|O_CLOEXEC）+ 两张小表 np.fromfile
      整表读入（进程内副本，非映射），记录 _owner_pid；
    - read_image_rows：先对全部行发 posix_fadvise(WILLNEED) 触发并发预读，再按
      连续行游程合并 os.preadv 直读进预分配 bf16 数组（短样本 32 行天然连续 →
      1 次调用）；短读循环补齐，EOF/越界立即 raise（B.2）；
    - 不看 meta.status / pack.lock——那是分派层的闸（verify 子命令持锁期间仍需
      经本类读库，A.1）。
    """

    def __init__(self, store_root: str | pathlib.Path, *,
                 meta: StoreMeta | None = None,
                 manifest_path: str | None = None,
                 source_root: str | None = None,
                 verify_level: str = "fast",
                 fast_sample: tuple[int, int] | None = None):
        if verify_level not in ("fast", "full"):
            raise ValueError(f"verify_level 非法: {verify_level!r}（∈ fast|full）")
        self._root = pathlib.Path(store_root)
        self._meta = meta if meta is not None else StoreMeta.load(self._root)
        run_fast_checks(self._meta, manifest_path=manifest_path,
                        source_root=source_root, sample_indices=fast_sample)
        if verify_level == "full":
            run_full_checks(self._meta)

        self._part_start = np.array([p.start_row for p in self._meta.parts], np.int64)
        self._part_rows = np.array([p.num_rows for p in self._meta.parts], np.int64)
        self._fds: list[int] = []
        try:
            for p in self._meta.parts:
                self._fds.append(os.open(self._root / p.path, os.O_RDONLY | os.O_CLOEXEC))
        except OSError:
            self.close()
            raise
        # 就地设 shape（不走 reshape 视图）：保证 .base is None——小表是进程内
        # 拥有内存的副本、非映射非视图（B.2/G10 契约）
        self._pos_table = np.fromfile(self._root / POS_TABLE_RELPATH, dtype=POS_DTYPE)
        self._pos_table.shape = (self._meta.num_pos_rows,) + POS_ROW_SHAPE
        self._state_table = np.fromfile(self._root / STATE_TABLE_RELPATH, dtype=STATE_DTYPE)
        self._state_table.shape = (self._meta.num_rows,) + STATE_ROW_SHAPE
        self._owner_pid = os.getpid()
        self._fadvise_ok = hasattr(os, "posix_fadvise")

    # ―― 属性 ――
    @property
    def meta(self) -> StoreMeta:
        return self._meta

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    @property
    def num_rows(self) -> int:
        return self._meta.num_rows

    @property
    def pos_table(self) -> np.ndarray:
        return self._pos_table

    @property
    def state_table(self) -> np.ndarray:
        return self._state_table

    def __reduce__(self):
        raise TypeError(
            "FrameSampStore 持有 fd 与进程内小表，禁止 pickle 跨进程携带；"
            "由 FrameSampDataset 在 worker 内按 pid 懒构造（B.2）")

    # ―― 读 API ――
    def _runs_of(self, rows: np.ndarray):
        """把行号序列切成 (输出起点, 行号起点, 行数) 的连续游程，再按 part 边界细分。"""
        runs = []
        i = 0
        n = len(rows)
        while i < n:
            j = i + 1
            while j < n and rows[j] == rows[j - 1] + 1:
                j += 1
            runs.append((i, int(rows[i]), j - i))
            i = j
        out = []
        for out_i, row0, cnt in runs:
            while cnt > 0:
                pi = int(np.searchsorted(self._part_start, row0, side="right") - 1)
                avail = int(self._part_start[pi] + self._part_rows[pi] - row0)
                take = min(cnt, avail)
                out.append((out_i, row0, take, pi))
                out_i += take
                row0 += take
                cnt -= take
        return out

    def _pread_exact(self, fd: int, mv: memoryview, offset: int, ctx: str) -> None:
        """preadv 短读循环补齐；读到 0 字节（EOF，文件短于声明）立即 raise。

        阻塞式常规文件读除 EOF 外不会返回 0，「零进展」与 EOF 在此等价——
        本仓库 NFS4.2 hard 挂载实测 320 次 2 MB 单调用零短读，本循环是稳健性兜底。
        """
        total = len(mv)
        filled = 0
        while filled < total:
            nread = os.preadv(fd, [mv[filled:]], offset + filled)
            if nread == 0:
                raise RuntimeError(
                    f"part 文件读到 EOF（短于 meta 声明，疑被截断）: {ctx} "
                    f"offset={offset + filled} 还差 {total - filled} B")
            filled += nread

    def read_image_rows(self, rows, out: np.ndarray | None = None) -> np.ndarray:
        """按全局行号读 image_emb_4x4，返回 (n,16,2048) bf16（0 open、0 pickle）。"""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError(f"rows 须一维: shape={rows.shape}")
        n = len(rows)
        if n and (int(rows.min()) < 0 or int(rows.max()) >= self._meta.num_rows):
            raise IndexError(
                f"行号越界: [{rows.min()}, {rows.max()}] ∉ [0, {self._meta.num_rows})")
        if out is None:
            raw = np.empty(n * IMAGE_ROW_BYTES, np.uint8)
        else:
            if out.dtype != np.uint8 or out.nbytes != n * IMAGE_ROW_BYTES:
                raise ValueError("out 须是 n×row_bytes 的 uint8 一维缓冲")
            raw = out
        segs = self._runs_of(rows)
        if self._fadvise_ok:
            for _, row0, cnt, pi in segs:
                off = (row0 - int(self._part_start[pi])) * IMAGE_ROW_BYTES
                try:
                    os.posix_fadvise(self._fds[pi], off, cnt * IMAGE_ROW_BYTES,
                                     os.POSIX_FADV_WILLNEED)
                except OSError as e:
                    logger.warning("posix_fadvise 不可用（%s），本进程永久跳过预读提示", e)
                    self._fadvise_ok = False
                    break
        mv = memoryview(raw)
        for out_i, row0, cnt, pi in segs:
            off = (row0 - int(self._part_start[pi])) * IMAGE_ROW_BYTES
            b0 = out_i * IMAGE_ROW_BYTES
            self._pread_exact(self._fds[pi], mv[b0:b0 + cnt * IMAGE_ROW_BYTES], off,
                              ctx=self._meta.parts[pi].path)
        return raw.view(IMAGE_DTYPE).reshape((n,) + IMAGE_ROW_SHAPE)

    def pos_rows(self, frames) -> np.ndarray:
        """按帧号 t 查 pos 小表，返回 (n,16,768) f32（进程内副本，无 NFS 缺页）。"""
        f = np.asarray(frames, dtype=np.int64)
        if len(f) and (int(f.min()) < 0 or int(f.max()) >= self._meta.num_pos_rows):
            raise IndexError(f"pos 帧号越界: [{f.min()}, {f.max()}] ∉ [0, {self._meta.num_pos_rows})")
        return self._pos_table[f]

    def state_rows(self, rows) -> np.ndarray:
        """按全局行号查 state 小表，返回 (n,8) f32。"""
        r = np.asarray(rows, dtype=np.int64)
        if len(r) and (int(r.min()) < 0 or int(r.max()) >= self._meta.num_rows):
            raise IndexError(f"state 行号越界: [{r.min()}, {r.max()}] ∉ [0, {self._meta.num_rows})")
        return self._state_table[r]

    def close(self) -> None:
        """显式关闭全部 part fd（32 个 fd 属内核资源，不允许靠 GC 兜底——B.2）。"""
        fds, self._fds = getattr(self, "_fds", []), []
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
