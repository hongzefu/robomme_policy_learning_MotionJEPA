#!/usr/bin/env python3
"""G4：在线 sidecar（--encoder-kind wan_raw9）与离线 motion-wan-raw9-s32 表逐字节对拍。

依据：根目录 1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.1 第 23 行、1.2 本文件行、runbook RB-C6。
由 compare_online_pool768.py 复制，差异只有三处：
1. 窗枚举改为 stride-32 表行：表第 ``row_base + m'`` 行是段内起点 ``32m'`` 的窗（``meta.spec.grid_stride`` 须为 32）；
2. 输入帧摘要取源 latent 元数据 ``<lib>/wan-latents/metadata.json`` 中 stride-16 网格的第 ``m = 2m'`` 行
   ``input_frames_sha256``（新表行 = 源 latent 第 2m' 窗的原字节）；
3. sidecar 返回 (9,16,32,32) float32 原 latent（589,824 B），客户端展平为 (9,16384)，与表行 ``tobytes()`` 逐字节相等才计 bitexact。

窗口抽样（与 pool768 版同一确定性哈希）：对每个窗 (段键, m') 取 ``int.from_bytes(SHA256(f"{seed}:{key}:{m'}")[:8], "big")``；
补帧窗（demo 段且 32m'+33 > 段长）按该值升序取前 ``--padded``（默认 8）个，其余窗取前 ``--windows − --padded`` 个。

两类帧来源（同 pool768 版）：exec 窗读 ``<source-root>/data/<exec_sample_offset + 32m' + i>.pkl`` 的 ``image`` 并核
``epis_idx`` / ``step_idx``；demo 窗（含全部补帧窗）读原始单 episode H5，经
``<episode-map-dir>/record_dataset_<task>_episode_map.json`` 定位行并核 ``h5_bytes`` / ``h5_sha256``。
map 找不到、行缺失、任一摘要不符即 ``ONLINE_RAW9=FAIL``（停）。

``--dry-run``：只读 --table（与同级 meta/episode_manifest.json），打印抽中窗、帧来源与所需 H5 episode，不读帧、不起 sidecar。

用法（主树，runbook RB-C6）：
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/g0/compare_online_raw9.py \\
      --table $LIB/motion-wan-raw9-s32 --source-root $LIB/source --h5-root $LIB/raw-h5 \\
      --episode-map-dir docs/dataset-build-doc/4task-v2-1600ep-604f16da/records --windows 64 --padded 8
判定行：ONLINE_RAW9=PASS windows=64 padded=8 bitexact=64
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import pickle
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.shared import wan_raw9 as R  # noqa: E402

WINDOW = 33
STRIDE = 32                 # 新表窗步长（= R.RAW9_STRIDE）
SRC_STRIDE = 16             # 源 wan-latents 网格步长：新表第 m' 行 = 源第 2m' 行
RAW9_KIND = "wan_raw9"
ROW_KEYS = ("task", "difficulty", "episode", "seed", "role", "member", "h5_sha256", "h5_bytes", "timestep_count")


class G4Fail(RuntimeError):
    """任何身份、摘要、映射或逐字节不符。"""


@dataclasses.dataclass(frozen=True)
class Window:
    row: int
    key: str
    m: int
    kind: str              # demo / exec
    g: int
    h5_file: str
    raw_ep_idx: int
    seg_start: int         # 段起点全域帧号（demo=0，exec=exec_start_idx）
    seg_len: int
    start: int             # 窗起点全域帧号
    padded: bool
    value: int

    @property
    def src_m(self) -> int:
        """源 latent（stride-16 网格）里同一窗的行号。"""
        return self.m * (STRIDE // SRC_STRIDE)

    @property
    def source(self) -> str:
        return "h5" if self.kind == "demo" else "pkl"


def sample_value(seed: int, key: str, m: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}:{m}".encode()).digest()[:8], "big")


def load_table_meta(table: pathlib.Path):
    """读 raw9 表 meta；布局步长须为 32（S1 的 LayoutSpec.grid_stride）。"""
    meta = ms.MotionMeta.load(table)
    if int(meta.spec.grid_stride) != STRIDE:
        raise G4Fail(f"目标表网格步长 {meta.spec.grid_stride} != {STRIDE}")
    return meta


def enumerate_windows(meta, seed: int) -> list[Window]:
    out = []
    for e in meta.entries:
        for kind in ms.SEGMENTS:
            seg = getattr(e, kind)
            if not seg.num_grid:
                continue
            key = ms.segment_key(e.h5_file, e.raw_ep_idx, kind)
            seg_start = 0 if kind == "demo" else e.exec_start_idx
            for m in range(seg.num_grid):
                off = STRIDE * m
                padded = kind == "demo" and off + WINDOW > seg.seg_len
                if kind == "exec" and off + WINDOW > seg.seg_len:
                    raise G4Fail(f"exec 窗越界 {key} m={m}")
                out.append(Window(row=seg.row_base + m, key=key, m=m, kind=kind, g=e.g, h5_file=e.h5_file,
                                  raw_ep_idx=e.raw_ep_idx, seg_start=seg_start, seg_len=seg.seg_len,
                                  start=seg_start + off, padded=padded, value=sample_value(seed, key, m)))
    return out


def select_windows(windows: list[Window], total: int, padded: int) -> list[Window]:
    if not 0 <= padded <= total:
        raise ValueError("--padded 须在 [0, --windows]")
    order = lambda w: (w.value, w.key, w.m)  # noqa: E731
    pad = sorted((w for w in windows if w.padded), key=order)[:padded]
    rest = sorted((w for w in windows if not w.padded), key=order)[:total - padded]
    if len(pad) != padded or len(rest) != total - padded:
        raise G4Fail(f"可抽窗口不足: padded {len(pad)}/{padded} rest {len(rest)}/{total - padded}")
    return sorted(pad + rest, key=lambda w: w.row)


def selection_sha256(chosen: list[Window]) -> str:
    return hashlib.sha256(json.dumps([[w.key, w.m, w.row] for w in chosen]).encode()).hexdigest()


def load_map_rows(map_dir: pathlib.Path | None, task: str) -> list[dict]:
    if map_dir is None:
        raise G4Fail("未给 --episode-map-dir")
    path = map_dir / f"record_dataset_{task}_episode_map.json"      # 与 merge_v2_h5.py::map_path 同名
    if not path.is_file():
        raise G4Fail(f"找不到 episode map: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise G4Fail(f"episode map 不是行列表: {path}")
    return rows


def map_row(maps: dict, map_dir, task: str, raw_ep_idx: int) -> dict:
    if task not in maps:
        maps[task] = load_map_rows(map_dir, task)
    hits = [r for r in maps[task] if int(r.get("new_idx", -1)) == int(raw_ep_idx)]
    if len(hits) != 1:
        raise G4Fail(f"episode map 缺行或重复: task={task} new_idx={raw_ep_idx} hits={len(hits)}")
    row = hits[0]
    missing = [k for k in ROW_KEYS + ("src_group",) if k not in row]
    if missing:
        raise G4Fail(f"episode map 行缺键 {missing}: task={task} new_idx={raw_ep_idx}")
    if row["role"] != "primary" or row["task"] != task:
        raise G4Fail(f"episode map 行 role/task 不符: {row['role']} {row['task']}")
    return row


def tar_object(row: dict) -> str:
    """h5 bucket 里 primary episode 所在的 tar.xz 对象（按 task × difficulty 打包）。"""
    return f"record_dataset_{row['task']}_{row['difficulty']}.h5.tar.xz"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assemble(frames: np.ndarray, off: int, seg_len: int, kind: str) -> np.ndarray:
    """段内帧 → 33 帧窗口：与 extract_wan.py::process_segment 同式（demo 不足 33 帧补段末帧，exec 禁止补帧）。"""
    real = min(WINDOW, seg_len - off)
    window = frames[off:off + real]
    if real < WINDOW:
        if kind != "demo":
            raise G4Fail("exec 窗不得补帧")
        window = np.concatenate([window, np.repeat(frames[seg_len - 1:seg_len], WINDOW - real, axis=0)])
    window = np.ascontiguousarray(window)
    if window.shape != (WINDOW, 256, 256, 3) or window.dtype != np.uint8:
        raise G4Fail(f"窗口形制异常 {window.shape} {window.dtype}")
    return window


def demo_window(w: Window, row: dict, h5_root: pathlib.Path, checked: dict, num_timesteps: int) -> np.ndarray:
    import h5py
    path = h5_root / row["member"]
    if path not in checked:
        if not path.is_file():
            raise G4Fail(f"缺单 episode H5: {path}")
        if path.stat().st_size != int(row["h5_bytes"]):
            raise G4Fail(f"H5 字节数不符: {path} {path.stat().st_size} != {row['h5_bytes']}")
        got = sha256_file(path)
        if got != row["h5_sha256"]:
            raise G4Fail(f"H5 sha256 不符: {path} {got} != {row['h5_sha256']}")
        checked[path] = got
    if int(row["timestep_count"]) != num_timesteps:
        raise G4Fail(f"map timestep_count {row['timestep_count']} != 清单 {num_timesteps}")
    off = w.start - w.seg_start
    last = min(off + WINDOW, w.seg_len)          # 补帧窗 last == seg_len，最后一个真实帧即段末帧
    steps = range(w.seg_start + off, w.seg_start + last)
    with h5py.File(path, "r") as f:
        if list(f) != [row["src_group"]]:
            raise G4Fail(f"H5 顶层组不符: {list(f)} != {row['src_group']}")
        g = f[row["src_group"]]
        if not all(bool(g[f"timestep_{t}"]["info"]["is_video_demo"][()]) for t in steps):
            raise G4Fail(f"demo 窗含非 demo 帧: {w.key} m={w.m}")
        real = np.stack([g[f"timestep_{t}"]["obs"]["front_rgb"][()] for t in steps])
    return assemble(real, 0, len(real), "demo")


def exec_window(w: Window, ep: dict, source_root: pathlib.Path) -> np.ndarray:
    off = w.start - w.seg_start
    imgs = []
    for i in range(WINDOW):
        step = w.start + i
        idx = int(ep["exec_sample_offset"]) + off + i
        p = source_root / "data" / f"{idx}.pkl"
        if not p.is_file():
            raise G4Fail(f"缺 source pkl: {p}")
        with open(p, "rb") as f:
            d = pickle.load(f)
        if int(np.asarray(d["epis_idx"]).reshape(-1)[0]) != w.g or int(np.asarray(d["step_idx"]).reshape(-1)[0]) != step:
            raise G4Fail(f"pkl 身份不符: {p} epis/step != {w.g}/{step}")
        if bool(np.asarray(d["is_demo"]).reshape(-1)[0]):
            raise G4Fail(f"exec 窗读到 demo 帧: {p}")
        imgs.append(np.asarray(d["image"]))
    frames = np.stack(imgs)
    return assemble(frames, 0, WINDOW, "exec")


def make_client(meta, gpu: str):
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    prov = meta.provenance
    if prov.get("raw9_source_sha256") != R.source_sha256():
        raise G4Fail("表 provenance.raw9_source_sha256 与当前 wan_raw9.py 不符")
    return MotionEncoderClient(online_gpu=gpu, encoder_kind=R.RAW9_ENCODER_KIND, readout=R.RAW9_READOUT,
                               expected_raw9_sha256=prov["raw9_source_sha256"], store_provenance={"vae": prov.get("vae")})


def run(args, client_factory=None) -> int:
    table = pathlib.Path(args.table)
    lib = table.parent
    meta = load_table_meta(table)
    if meta.provenance.get("encoder_kind") != RAW9_KIND:
        raise G4Fail(f"目标表不是 raw9 表: encoder_kind={meta.provenance.get('encoder_kind')!r}")
    manifest = ms.load_manifest(lib / "meta/episode_manifest.json")
    if manifest["sha256"] != meta.manifest_sha256:
        raise G4Fail("表绑定的 manifest 与同级 meta/episode_manifest.json 不符")
    eps = {int(e["global_episode_idx"]): e for e in manifest["episodes"]}
    windows = enumerate_windows(meta, args.seed)
    chosen = select_windows(windows, args.windows, args.padded)
    n_pad = sum(w.padded for w in chosen)
    map_dir = pathlib.Path(args.episode_map_dir) if args.episode_map_dir else None
    maps: dict = {}

    if args.dry_run:
        need = {}
        for w in chosen:
            rec = {"row": w.row, "key": w.key, "m": w.m, "kind": w.kind, "start": w.start, "padded": w.padded,
                   "source": w.source, "g": w.g, "h5_file": w.h5_file, "raw_ep_idx": w.raw_ep_idx}
            print("WINDOW " + json.dumps(rec, ensure_ascii=False))
            if w.source == "h5":
                need.setdefault((ms.task_of_h5(w.h5_file), w.raw_ep_idx), []).append(w.row)
        for (task, idx), rows in sorted(need.items()):
            rec = {"task": task, "raw_ep_idx": idx, "rows": rows}
            if map_dir is not None and (map_dir / f"record_dataset_{task}_episode_map.json").is_file():
                r = map_row(maps, map_dir, task, idx)
                rec.update({k: r[k] for k in ("difficulty", "episode", "member", "h5_sha256", "h5_bytes")}, tar_object=tar_object(r))
            print("H5_EPISODE " + json.dumps(rec, ensure_ascii=False))
        print(f"DRY_RUN windows={len(chosen)} padded={n_pad} demo={sum(w.kind == 'demo' for w in chosen)} "
              f"exec={sum(w.kind == 'exec' for w in chosen)} h5_episodes={len(need)} selection_sha256={selection_sha256(chosen)}")
        return 0

    lat_meta = json.loads((lib / "wan-latents" / "metadata.json").read_text(encoding="utf-8"))
    if lat_meta.get("manifest_sha256") != meta.manifest_sha256:
        raise G4Fail("wan-latents/metadata.json 的 manifest_sha256 与表不符")
    h5_root, source_root = pathlib.Path(args.h5_root), pathlib.Path(args.source_root)
    checked: dict = {}
    inputs = []
    for w in chosen:
        seg = lat_meta["segments"].get(w.key)
        rows = {int(r["m"]): r for r in (seg or {}).get("rows", [])}
        if w.src_m not in rows:
            raise G4Fail(f"latent 元数据缺窗口 {w.key} m={w.src_m}（新表 m'={w.m}）")
        lrow = rows[w.src_m]
        if (int(lrow.get("pad_frames", 0)) > 0) != w.padded:
            raise G4Fail(f"补帧标记与 latent 元数据不符 {w.key} m={w.src_m}")
        ep = eps[w.g]
        if w.source == "h5":
            r = map_row(maps, map_dir, ms.task_of_h5(w.h5_file), w.raw_ep_idx)
            window = demo_window(w, r, h5_root, checked, int(ep["num_timesteps"]))
        else:
            window = exec_window(w, ep, source_root)
        got = hashlib.sha256(window.tobytes()).hexdigest()
        if got != lrow.get("input_frames_sha256"):
            raise G4Fail(f"窗输入摘要不符 {w.key} m={w.src_m}（新表 m'={w.m}）: {got} != {lrow.get('input_frames_sha256')}")
        inputs.append((w, window))
    print(f"INPUTS_OK windows={len(inputs)} h5_files={len(checked)}", flush=True)

    table_mm = np.memmap(table / ms.MOTION_TABLE_RELPATH, dtype="<f4", mode="r",
                         shape=(meta.num_rows, R.RAW9_ROW_BYTES // 4))
    client = (client_factory or make_client)(meta, args.gpu)
    bitexact, bad = 0, []
    try:
        for w, window in inputs:
            token = np.asarray(client(window, w.start))
            off = np.array(table_mm[w.row])
            want_shape = (R.RAW9_TOKENS_PER_WINDOW, R.RAW9_TOKEN_DIM)
            if token.dtype == np.float32 and token.shape == want_shape and token.tobytes() == off.tobytes():
                bitexact += 1
            else:
                bad.append({"row": w.row, "key": w.key, "m": w.m})
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    ok = bitexact == len(chosen) and n_pad == args.padded
    if bad:
        print("MISMATCH " + json.dumps(bad[:20], ensure_ascii=False))
    print(f"ONLINE_RAW9={'PASS' if ok else 'FAIL'} windows={len(chosen)} padded={n_pad} bitexact={bitexact}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="motion-wan-raw9-s32 表根；同级需有 meta/episode_manifest.json 与 wan-latents/")
    ap.add_argument("--source-root", help="source 库根（exec 窗读 data/<idx>.pkl）")
    ap.add_argument("--h5-root", help="单 episode H5 解包根（map 行 member 相对它）")
    ap.add_argument("--episode-map-dir", help="merge_v2_h5.py 写出的 record_dataset_<task>_episode_map.json 所在目录")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--padded", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="与建库 oracle sample_spec 的 seed:0 相同")
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.dry_run and not (args.source_root and args.h5_root):
        ap.error("非 --dry-run 必须给出 --source-root 与 --h5-root")
    try:
        return run(args)
    except (G4Fail, FileNotFoundError, KeyError, ValueError) as e:
        print(f"ONLINE_RAW9=FAIL reason={type(e).__name__}: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
