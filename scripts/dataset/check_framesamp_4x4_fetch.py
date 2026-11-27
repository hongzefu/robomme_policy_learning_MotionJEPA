#!/usr/bin/env python3
"""原生 4×4 framesamp 表拉回后的 G1/G2 判定（1127-4task-modul-fs4x4-512-1024-60k-plan.md）。

五项检查，依次输出判定行，最后一行汇总：

  fetch      逐文件字节数 + sha256 对上传清单 upload_manifest.json（plain_files 中
             rel 以 ``framesamp/`` 开头的项，本地映射到 store 下同相对路径），进程池并行
  rowdigest  逐行重算 blake2b-128（image‖pos‖state 的 store 侧原始字节，与
             pack_framesamp_store._verify_part_worker 同算法：image/state 按全局行号，
             pos 按帧号 t = 行号 − 该 episode 的 total_sample_offset），对 meta/row_digests
  xgrid      每行 4×4 image 与同行 8×8 image 的 2×2 平均池化（行主序 8×8 网格）做逐 token
             余弦（fp32），全部 ≥ 阈值（默认 0.9999）；打印最差 10 行
  probe      源探针 npy（features/episode_<g>/token_emb_<t>.npy）的 image/pos/state 与表中
             全局行 row_of(total_sample_offset[g], t) / 帧号 t 逐位相等
  meta       layout、status、num_rows、manifest_sha256，及与 8×8 表、episode 清单同源

rowdigest 与 xgrid 共用一次按 part 并行的扫描（mmap 按块读，内存有界）。
probe 与 rowdigest 按 store 自身的 layout 取键，因此也可对 8×8 表运行（自检用）。

用法（uv 管理，禁止裸 python）：
  uv run python scripts/dataset/check_framesamp_4x4_fetch.py --store <4x4> --ref8 <8x8> \
      --probes <source/features> --manifest-json <bucket-top/upload_manifest.json> \
      --episode-manifest <meta/episode_manifest.json> [--procs N] [--only fetch,rowdigest,...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import pathlib
import re
import sys
import time
import traceback

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import framesamp_store as fs  # noqa: E402
from mme_vla_suite.datastore.manifest import load_manifest  # noqa: E402

_MP = multiprocessing.get_context("fork")

CHECKS = ("fetch", "rowdigest", "xgrid", "probe", "meta")
EXPECT_ROWS = 1192918
EXPECT_MANIFEST_SHA = "4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918"
EXPECT_LAYOUT = "framesamp-4x4-v1"
REF_LAYOUT = "framesamp-8x8-v1"
CHUNK_ROWS = 256

# fork 前由主进程填好，worker 继承只读副本
_W: dict = {}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _default_procs() -> int:
    return max(1, min(32, (os.cpu_count() or 9) - 8))


# ══ fetch ═════════════════════════════════════════════════════════════════════


def _sha_worker(item: tuple[str, str]) -> tuple[str, int, str]:
    rel, path = item
    p = pathlib.Path(path)
    if p.is_symlink():
        return rel, -2, ""
    if not p.is_file():
        return rel, -1, ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return rel, p.stat().st_size, h.hexdigest()


def check_fetch(args) -> tuple[bool, str]:
    m = json.loads(pathlib.Path(args.manifest_json).read_text(encoding="utf-8"))
    prefix = args.manifest_prefix
    entries = [e for e in m["plain_files"] if e["rel"].startswith(prefix)]
    store = pathlib.Path(args.store)
    want = {e["rel"][len(prefix):]: e for e in entries}
    total = sum(int(e["bytes"]) for e in entries)
    mismatches = 0
    if len(entries) != args.expect_files:
        _log(f"[fetch] 清单 {prefix}* 项数 {len(entries)} != 期望 {args.expect_files}")
        mismatches += 1
    for rel in want:
        q = store
        for comp in pathlib.PurePosixPath(rel).parts[:-1]:
            q = q / comp
            if q.is_symlink():
                _log(f"[fetch] 拒绝符号链接目录 {q.relative_to(store)}（{rel}）")
                mismatches += 1
                break
    items = sorted(((rel, str(store / rel)) for rel in want),
                   key=lambda x: -int(want[x[0]]["bytes"]))
    with _MP.Pool(min(args.procs, max(1, len(items)))) as pool:
        for rel, size, sha in pool.imap_unordered(_sha_worker, items):
            e = want[rel]
            if size == -2:
                _log(f"[fetch] 拒绝符号链接 {rel}")
                mismatches += 1
            elif size < 0:
                _log(f"[fetch] 缺失 {rel}")
                mismatches += 1
            elif size != int(e["bytes"]) or sha != e["sha256"]:
                _log(f"[fetch] 不符 {rel}: bytes {size} vs {e['bytes']}, "
                     f"sha256 {sha[:16]}… vs {e['sha256'][:16]}…")
                mismatches += 1
            else:
                _log(f"[fetch] ok {rel} ({size} B)")
    # 清单外的多余文件（以 . 开头的目录/文件视为工具元数据，只提示不计）
    for p in sorted(store.rglob("*")):
        if p.is_symlink() and p.relative_to(store).as_posix() not in want:
            _log(f"[fetch] 清单外符号链接 {p.relative_to(store).as_posix()}")
            mismatches += 1
            continue
        if not p.is_file():
            continue
        rel = p.relative_to(store).as_posix()
        if rel in want:
            continue
        if any(part.startswith(".") for part in pathlib.PurePosixPath(rel).parts):
            _log(f"[fetch] 提示：忽略工具元数据 {rel}")
            continue
        _log(f"[fetch] 清单外多余文件 {rel}")
        mismatches += 1
    ok = mismatches == 0
    return ok, (f"FS4X4_FETCH={'PASS' if ok else 'FAIL'} files={len(entries)} "
                f"bytes={total} mismatches={mismatches}")


# ══ 共用：清单派生的行→t 表 ══════════════════════════════════════════════════


def _row_to_t(manifest: dict) -> np.ndarray:
    n = int(manifest["totals"]["timesteps"])
    t_of = np.full(n, -1, np.int64)
    cursor = 0
    for g, ep in enumerate(manifest["episodes"]):
        if ep["global_episode_idx"] != g:
            raise ValueError(f"清单 episodes 序错乱: 第 {g} 项")
        off, nt = int(ep["total_sample_offset"]), int(ep["num_timesteps"])
        if off != cursor:
            raise ValueError(f"清单 total_sample_offset 不连续: episode {g} {off} != {cursor}")
        t_of[off:off + nt] = np.arange(nt)
        cursor += nt
    if cursor != n:
        raise ValueError(f"清单 timesteps 总和 {cursor} != totals {n}")
    return t_of


class _RowReader:
    """按全局行号区间读某库的 image（uint16 视图），跨 part 边界拼接；mmap 惰性打开。"""

    def __init__(self, meta: fs.StoreMeta):
        self.meta = meta
        self.starts = np.array([p.start_row for p in meta.parts], np.int64)
        self.width = meta.spec.tokens_per_frame * 2048
        self._mm: dict[int, np.memmap] = {}

    def _part(self, i: int) -> np.memmap:
        if i not in self._mm:
            p = self.meta.parts[i]
            self._mm[i] = np.memmap(self.meta.root / p.path, dtype=np.uint16, mode="r",
                                    shape=(p.num_rows, self.width))
        return self._mm[i]

    def read(self, a: int, b: int) -> np.ndarray:
        out = []
        while a < b:
            i = int(np.searchsorted(self.starts, a, side="right") - 1)
            p = self.meta.parts[i]
            end = min(b, p.start_row + p.num_rows)
            out.append(np.asarray(self._part(i)[a - p.start_row:end - p.start_row]))
            a = end
        return out[0] if len(out) == 1 else np.concatenate(out)


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ══ rowdigest + xgrid（按 part 并行的单次扫描）══════════════════════════════


def _scan_part_worker(pi: int) -> dict:
    meta: fs.StoreMeta = _W["meta"]
    spec = meta.spec
    part = meta.parts[pi]
    t0 = time.time()
    img_mm = np.memmap(meta.root / part.path, dtype=np.uint16, mode="r",
                       shape=(part.num_rows, spec.tokens_per_frame * 2048))
    res = {"index": pi, "rows": part.num_rows, "dig_bad": 0, "dig_bad_rows": [],
           "x_bad": 0, "x_min": 1.0, "x_nonfinite": 0, "x_worst": []}
    do_dig, do_x = _W["do_dig"], _W["do_x"]
    if do_dig:
        t_of = _W["t_of"]
        pos_mm = np.memmap(meta.root / spec.pos_table_relpath, dtype=np.uint8, mode="r",
                           shape=(meta.num_pos_rows, spec.pos_row_bytes))
        st_mm = np.memmap(meta.root / fs.STATE_TABLE_RELPATH, dtype=np.uint8, mode="r",
                          shape=(meta.num_rows, fs.STATE_ROW_BYTES))
        ref_mm = np.memmap(meta.root / fs.ROW_DIGESTS_RELPATH, dtype=np.uint8, mode="r",
                           shape=(meta.num_rows, fs.ROW_DIGEST_BYTES))
    if do_x:
        r8 = _RowReader(_W["meta8"])
        thr = _W["xgrid_thr"]
        worst_c: list[np.ndarray] = []
        worst_r: list[np.ndarray] = []
    for a in range(0, part.num_rows, CHUNK_ROWS):
        b = min(part.num_rows, a + CHUNK_ROWS)
        g0 = part.start_row + a
        img = np.ascontiguousarray(img_mm[a:b])
        if do_dig:
            ts = t_of[g0:g0 + (b - a)]
            stt = np.asarray(st_mm[g0:g0 + (b - a)])
            ref = np.asarray(ref_mm[g0:g0 + (b - a)])
            for i in range(b - a):
                h = hashlib.blake2b(digest_size=fs.ROW_DIGEST_BYTES)
                h.update(img[i])
                h.update(pos_mm[int(ts[i])])
                h.update(stt[i])
                if h.digest() != ref[i].tobytes():
                    res["dig_bad"] += 1
                    if len(res["dig_bad_rows"]) < 20:
                        res["dig_bad_rows"].append(g0 + i)
        if do_x:
            n = b - a
            a4 = _bf16_to_f32(img).reshape(n, 16, 2048)
            p8 = _bf16_to_f32(r8.read(g0, g0 + n)).reshape(n, 4, 2, 4, 2, 2048)
            p8 = p8.sum(axis=(2, 4), dtype=np.float32).reshape(n, 16, 2048)   # 余弦与缩放无关
            dot = np.einsum("ntc,ntc->nt", a4, p8)
            na = np.sqrt(np.einsum("ntc,ntc->nt", a4, a4))
            nb = np.sqrt(np.einsum("ntc,ntc->nt", p8, p8))
            den = na * nb
            both_zero = (na == 0) & (nb == 0)
            cos = np.where(den > 0, dot / np.where(den > 0, den, 1), np.where(both_zero, 1.0, 0.0))
            fin = np.isfinite(cos).all(axis=1)
            row_min = cos.min(axis=1)
            res["x_nonfinite"] += int((~fin).sum())
            bad = ~(row_min >= thr) | ~fin          # NaN/inf 一律计入 bad
            res["x_bad"] += int(bad.sum())
            row_min = np.where(fin, row_min, -np.inf)   # 非有限行在排序与最小值里排最差
            res["x_min"] = min(res["x_min"], float(row_min.min()))
            k = min(10, n)
            idx = np.argpartition(row_min, k - 1)[:k]
            worst_c.append(row_min[idx])
            worst_r.append(idx + g0)
    if do_x and worst_c:
        c = np.concatenate(worst_c)
        r = np.concatenate(worst_r)
        o = np.argsort(c)[:10]
        res["x_worst"] = [(int(r[j]), float(c[j])) for j in o]
    res["secs"] = time.time() - t0
    return res


def scan_parts(args, meta: fs.StoreMeta, manifest: dict, do_dig: bool, do_x: bool,
               meta8: fs.StoreMeta | None) -> dict:
    _W.clear()
    _W.update(meta=meta, meta8=meta8, do_dig=do_dig, do_x=do_x,
              t_of=_row_to_t(manifest) if do_dig else None, xgrid_thr=args.xgrid_threshold)
    if do_dig:
        if len(_W["t_of"]) != meta.num_rows:
            raise ValueError(f"清单行数 {len(_W['t_of'])} != 库 num_rows {meta.num_rows}")
        rd = meta.root / fs.ROW_DIGESTS_RELPATH
        if not rd.is_file() or rd.stat().st_size != meta.num_rows * fs.ROW_DIGEST_BYTES:
            raise ValueError(f"row_digests 缺失或长度不符: {rd}")
        if int(_W["t_of"].max()) >= meta.num_pos_rows:
            raise ValueError("清单最大帧号超出 num_pos_rows")
    if do_x:
        if meta.spec.layout != EXPECT_LAYOUT or meta8.spec.layout != REF_LAYOUT:
            raise ValueError(f"xgrid 需要 4×4 库 + 8×8 参照: {meta.spec.layout} / {meta8.spec.layout}")
        if meta8.num_rows != meta.num_rows or meta8.manifest_sha256 != meta.manifest_sha256:
            raise ValueError("4×4 与 8×8 两表行序不同源（num_rows 或 manifest_sha256 不同）")
    parts = list(range(len(meta.parts)))
    if args.parts:
        parts = [int(x) for x in args.parts.split(",")]
    tot = {"rows": 0, "dig_bad": 0, "dig_bad_rows": [], "x_bad": 0, "x_min": 1.0,
           "x_nonfinite": 0, "x_worst": []}
    order = sorted(parts, key=lambda i: -meta.parts[i].num_rows)
    with _MP.Pool(min(args.procs, len(order))) as pool:
        for k, r in enumerate(pool.imap_unordered(_scan_part_worker, order), 1):
            msg = f"[scan] part {r['index']:03d} rows={r['rows']} {r['secs']:.1f}s ({k}/{len(order)})"
            if do_dig:
                msg += f" digest_bad={r['dig_bad']}"
            if do_x:
                msg += f" xgrid_min={r['x_min']:.7f} xgrid_bad={r['x_bad']}"
            _log(msg)
            tot["rows"] += r["rows"]
            tot["dig_bad"] += r["dig_bad"]
            tot["dig_bad_rows"] += r["dig_bad_rows"]
            tot["x_bad"] += r["x_bad"]
            tot["x_min"] = min(tot["x_min"], r["x_min"])
            tot["x_nonfinite"] += r["x_nonfinite"]
            tot["x_worst"] += r["x_worst"]
    tot["x_worst"] = sorted(tot["x_worst"], key=lambda x: x[1])[:10]
    tot["full"] = (len(parts) == len(set(parts)) and set(parts) == set(range(len(meta.parts)))
                   and tot["rows"] == meta.num_rows)
    return tot


# ══ probe ═════════════════════════════════════════════════════════════════════

_PROBE_RE = re.compile(r"episode_(\d+)/token_emb_(\d+)\.npy$")


def check_probe(args, meta: fs.StoreMeta, manifest: dict) -> tuple[bool, str]:
    spec = meta.spec
    files = sorted(pathlib.Path(args.probes).glob("episode_*/token_emb_*.npy"))
    reader = _RowReader(meta)
    pos_mm = np.memmap(meta.root / spec.pos_table_relpath, dtype=np.float32, mode="r",
                       shape=(meta.num_pos_rows,) + spec.pos_row_shape)
    st_mm = np.memmap(meta.root / fs.STATE_TABLE_RELPATH, dtype=np.float32, mode="r",
                      shape=(meta.num_rows,) + fs.STATE_ROW_SHAPE)
    mismatches = 0
    for f in files:
        m = _PROBE_RE.search(f.as_posix())
        g, t = int(m.group(1)), int(m.group(2))
        ep = manifest["episodes"][g]
        if ep["global_episode_idx"] != g or not 0 <= t < int(ep["num_timesteps"]):
            _log(f"[probe] 探针越界 episode={g} t={t}")
            mismatches += 1
            continue
        row = fs.row_of(ep["total_sample_offset"], t)
        d = np.load(f, allow_pickle=True).item()
        img, pos, stt = d[spec.image_key], d[spec.pos_key], d[fs.STATE_KEY]
        bad = []
        if img.shape != (1,) + spec.image_row_shape or str(img.dtype) != "bfloat16" \
                or reader.read(row, row + 1).tobytes() != img.tobytes():
            bad.append("image")
        if pos.shape != (1,) + spec.pos_row_shape or pos.dtype != np.float32 \
                or np.asarray(pos_mm[t]).tobytes() != pos.tobytes():
            bad.append("pos")
        if np.asarray(st_mm[row]).tobytes() != stt.tobytes():
            bad.append("state")
        _log(f"[probe] episode={g} t={t} row={row} {spec.image_key}/{spec.pos_key}/state "
             f"{'ok' if not bad else '不符: ' + ','.join(bad)}")
        mismatches += bool(bad)
    if len(files) != args.expect_probes:
        _log(f"[probe] 探针数 {len(files)} != 期望 {args.expect_probes}")
        mismatches += 1
    ok = mismatches == 0
    return ok, f"FS4X4_PROBE={'PASS' if ok else 'FAIL'} frames={len(files)} mismatches={mismatches}"


# ══ meta ══════════════════════════════════════════════════════════════════════


def check_meta(args, meta: fs.StoreMeta, manifest: dict, meta8: fs.StoreMeta | None) -> tuple[bool, str]:
    errs = []
    if meta.spec.layout != EXPECT_LAYOUT:
        errs.append(f"layout={meta.spec.layout}")
    if meta.status != "verified":
        errs.append(f"status={meta.status}")
    if meta.num_rows != args.expect_rows:
        errs.append(f"num_rows={meta.num_rows}")
    if meta.manifest_sha256 != args.expect_manifest_sha:
        errs.append(f"manifest_sha256={meta.manifest_sha256}")
    if manifest["sha256"] != args.expect_manifest_sha:
        errs.append(f"episode_manifest sha256={manifest['sha256']}")
    if int(manifest["totals"]["timesteps"]) != meta.num_rows:
        errs.append("episode_manifest timesteps != num_rows")
    if meta.num_pos_rows != max(int(e["num_timesteps"]) for e in manifest["episodes"]):
        errs.append(f"num_pos_rows={meta.num_pos_rows} != 最长 episode")
    rd = meta.raw.get("row_digests") or {}
    rd_path = meta.root / fs.ROW_DIGESTS_RELPATH
    if rd.get("covered_rows") != meta.num_rows or rd.get("algo") != "blake2b-128" \
            or rd.get("coverage") != fs.ROW_DIGEST_COVERAGE:
        errs.append(f"row_digests 元数据不符: {rd}")
    elif not rd_path.is_file() or fs.sha256_file(rd_path) != rd.get("sha256"):
        errs.append("row_digests 文件 sha256 与 meta 不符")
    if meta8 is None:
        errs.append("缺 8×8 参照 meta")
    else:
        if meta8.spec.layout != REF_LAYOUT:
            errs.append(f"参照 layout={meta8.spec.layout}")
        if meta8.manifest_sha256 != meta.manifest_sha256 or meta8.num_rows != meta.num_rows:
            errs.append("与 8×8 表 manifest_sha256/num_rows 不同")
    for e in errs:
        _log(f"[meta] {e}")
    ok = not errs
    return ok, f"FS4X4_META={'PASS' if ok else 'FAIL'}"


# ══ main ══════════════════════════════════════════════════════════════════════


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--ref8", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--manifest-json", required=True)
    ap.add_argument("--episode-manifest", required=True)
    ap.add_argument("--procs", type=int, default=_default_procs())
    ap.add_argument("--only", default=",".join(CHECKS))
    ap.add_argument("--parts", default="", help="只扫这些 part（逗号分隔；调试/自检用，非全量不得 PASS 汇总以外的结论）")
    ap.add_argument("--manifest-prefix", default="framesamp/")
    ap.add_argument("--expect-files", type=int, default=37)
    ap.add_argument("--expect-rows", type=int, default=EXPECT_ROWS)
    ap.add_argument("--expect-manifest-sha", default=EXPECT_MANIFEST_SHA)
    ap.add_argument("--expect-probes", type=int, default=8)
    ap.add_argument("--xgrid-threshold", type=float, default=0.9999)
    args = ap.parse_args(argv)
    only = [c for c in args.only.split(",") if c]
    unknown = set(only) - set(CHECKS)
    if unknown:
        ap.error(f"--only 含未知项 {sorted(unknown)}")
    t0 = time.time()
    results: dict[str, bool] = {}
    lines: dict[str, str] = {}
    label = {c: f"FS4X4_{c.upper()}" for c in CHECKS}

    def fail(c: str, e: BaseException) -> None:
        traceback.print_exc()
        results[c] = False
        lines[c] = f"{label[c]}=FAIL error={type(e).__name__}: {e}"

    manifest = meta = meta8 = None
    need_meta = set(only) & {"rowdigest", "xgrid", "probe", "meta"}
    if need_meta:
        try:
            manifest = load_manifest(args.episode_manifest)
            meta = fs.StoreMeta.load(args.store)
        except Exception as e:   # noqa: BLE001
            for c in need_meta:
                fail(c, e)
            need_meta = set()
        if need_meta & {"xgrid", "meta"}:
            try:
                meta8 = fs.StoreMeta.load(args.ref8)
            except Exception as e:   # noqa: BLE001
                for c in need_meta & {"xgrid", "meta"}:
                    fail(c, e)
                need_meta -= {"xgrid", "meta"}

    for c in CHECKS:
        if c not in only:
            continue
        if c == "fetch":
            try:
                ok, line = check_fetch(args)
                results[c], lines[c] = ok, line
            except Exception as e:   # noqa: BLE001
                fail(c, e)
        elif c == "rowdigest" and c in need_meta or c == "xgrid" and c in need_meta and "rowdigest" not in need_meta:
            do_dig = "rowdigest" in need_meta
            do_x = "xgrid" in need_meta
            try:
                tot = scan_parts(args, meta, manifest, do_dig, do_x, meta8)
                partial = "" if tot["full"] else " partial=1"
                if do_dig:
                    ok = tot["dig_bad"] == 0 and tot["full"]
                    if tot["dig_bad_rows"]:
                        _log(f"[rowdigest] 不符行（前 20）: {sorted(tot['dig_bad_rows'])[:20]}")
                    results["rowdigest"] = ok
                    lines["rowdigest"] = (f"FS4X4_ROWDIGEST={'PASS' if ok else 'FAIL'} rows={tot['rows']} "
                                          f"mismatches={tot['dig_bad']}{partial}")
                if do_x:
                    ok = tot["x_bad"] == 0 and tot["x_nonfinite"] == 0 and tot["full"] \
                        and bool(np.isfinite(tot["x_min"])) and tot["x_min"] >= args.xgrid_threshold
                    _log("[xgrid] 最差 10 行 (row, min_token_cos): "
                         + ", ".join(f"({r},{c_:.7f})" for r, c_ in tot["x_worst"]))
                    results["xgrid"] = ok
                    lines["xgrid"] = (f"FS4X4_XGRID={'PASS' if ok else 'FAIL'} rows={tot['rows']} "
                                      f"min_cos={tot['x_min']:.7f} bad_rows={tot['x_bad']} "
                                      f"nonfinite_rows={tot['x_nonfinite']}{partial}")
            except Exception as e:   # noqa: BLE001
                for cc in ("rowdigest", "xgrid"):
                    if cc in need_meta:
                        fail(cc, e)
        elif c == "probe" and c in need_meta:
            try:
                results[c], lines[c] = check_probe(args, meta, manifest)
            except Exception as e:   # noqa: BLE001
                fail(c, e)
        elif c == "meta" and c in need_meta:
            try:
                results[c], lines[c] = check_meta(args, meta, manifest, meta8)
            except Exception as e:   # noqa: BLE001
                fail(c, e)

    for c in CHECKS:
        if c in lines:
            _log(lines[c])
    failed = [c for c in CHECKS if c in results and not results[c]]
    ran = [c for c in CHECKS if c in results]
    _log(f"[check] 用时 {time.time() - t0:.1f}s")
    if failed:
        _log(f"FS4X4_CHECK=FAIL failed={','.join(failed)}")
        return 1
    if set(results) != set(CHECKS) or args.parts:
        # 只跑了部分检查或部分 part：不得给出 PASS 汇总
        _log(f"FS4X4_CHECK=PARTIAL ran={','.join(ran) or 'none'}"
             f"{' parts=' + args.parts if args.parts else ''}")
        return 2
    _log("FS4X4_CHECK=PASS failed=none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
