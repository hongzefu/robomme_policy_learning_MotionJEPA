"""E2E-Enc 固定窗口清单 ``e2e_windows_v1.json``：生成、读入校验、取源 latent（计划 1117 一节 ``e2e_windows_v1.json`` 行）。

G1、G1b、G1c、G3 与训练诊断共用同一批窗口；各脚本经 ``load_manifest`` 读入时校验文件 sha256 必须等于
``E2E_WINDOWS_V1_SHA256``（提交记录里的值），不等即 raise。

清单从 v1-store **只读**生成（不写 v1-store）：
- 行序与表行号取自 768 维表 ``motion/meta/motion_index.json``（layout ``motion-768-grid16-demopad17-v1``；
  s16 bf16 新表按同一行序重排，计划 G2 ``INDEX_VS_MOTION768``），``table_row = entry[kind].row_base + m``；
- 每窗的窗起点、真实帧 / 补帧数取自 ``wan-latents/<段>.metadata.json``（``rows[m]``）；
- 源 latent 是 ``wan-latents/<段>.bin`` 里第 m 个 589,824 B 的 float32 (9,16,32,32) 块，记其 sha256。

选样（固定种子 20261118，四任务分层，每任务 512 窗，共 2048 窗）：
  每任务 64 个 demopad 尾窗（demo 段 ``pad_frames > 0`` 的窗，repeat_last 补尾）+ 192 个普通 demo 窗 +
  256 个 exec 窗；BinFill 的 demo 窗（评估时注入 demo 前缀）即其中 256 个 demo 窗。

用法（生成，只读 v1-store，写出到仓库）::

    uv run python scripts/training/g0/e2e_windows_v1.py generate \
        --dataset-root /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-v2-1600ep-604f16da \
        --out scripts/training/g0/e2e_windows_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

import numpy as np

E2E_WINDOWS_V1_SHA256 = "32e9aecc20d21bb68948a25f51657e1db2b3f50501dcbe2f0247ae28eae7b759"
DEFAULT_DATASET_ROOT = pathlib.Path(
    "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-v2-1600ep-604f16da")
DEFAULT_MANIFEST = pathlib.Path(__file__).resolve().parent / "e2e_windows_v1.json"
CHUNK_BYTES = 9 * 16 * 32 * 32 * 4  # 589,824
TASKS = ("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap")
SEED = 20261118
PER_TASK = {"demopad": 64, "demo": 192, "exec": 256}
LAYOUT_768 = "motion-768-grid16-demopad17-v1"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _task_of(h5_file: str) -> str:
    assert h5_file.startswith("record_dataset_") and h5_file.endswith(".h5"), h5_file
    return h5_file[len("record_dataset_"):-len(".h5")]


def generate(dataset_root: pathlib.Path) -> dict:
    meta = json.loads((dataset_root / "motion/meta/store_meta.json").read_text())
    idx_path = dataset_root / "motion/meta/motion_index.json"
    idx = json.loads(idx_path.read_text())
    if idx["layout"] != LAYOUT_768 or meta["layout"] != LAYOUT_768:
        raise RuntimeError(f"768 表 layout 不是 {LAYOUT_768}")
    idx_sha = file_sha256(idx_path)
    if idx_sha != meta["motion_index_sha256"]:
        raise RuntimeError("motion_index.json sha256 与 store_meta 不符")
    lat = dataset_root / "wan-latents"
    pools: dict[tuple[str, str], list[dict]] = {(t, k): [] for t in TASKS for k in PER_TASK}
    for e in idx["entries"]:
        task = _task_of(e["h5_file"])
        for kind in ("demo", "exec"):
            seg = f"{task}_ep{e['raw_ep_idx']}_{kind}"
            md = json.loads((lat / f"{seg}.metadata.json").read_text())
            if (md["g"], md["raw_ep_idx"], md["segment_kind"], md["num_grid"]) != (
                    e["g"], e["raw_ep_idx"], kind, e[kind]["num_grid"]):
                raise RuntimeError(f"{seg} metadata 与 motion_index 不一致")
            for r in md["rows"]:
                w = {
                    "task": task, "h5_file": e["h5_file"], "g": e["g"], "episode": e["raw_ep_idx"],
                    "exec_start_idx": e["exec_start_idx"], "segment": seg, "kind": kind, "m": r["m"],
                    "window_start": r["start_global_frame"], "seg_offset": r["seg_offset"],
                    "table_row": e[kind]["row_base"] + r["m"], "real_frames": r["real_frames"],
                    "pad_frames": r["pad_frames"],
                    "demopad": bool(kind == "demo" and r["pad_frames"] > 0),
                }
                pool = "exec" if kind == "exec" else ("demopad" if w["demopad"] else "demo")
                pools[(task, pool)].append(w)
    rng = np.random.default_rng(SEED)
    chosen = []
    for t in TASKS:
        for k, n in PER_TASK.items():
            pool = pools[(t, k)]
            if len(pool) < n:
                raise RuntimeError(f"{t}/{k} 只有 {len(pool)} 窗 < {n}")
            sel = rng.choice(len(pool), size=n, replace=False)
            chosen.extend(pool[i] for i in sorted(sel))
    chosen.sort(key=lambda w: w["table_row"])
    for i, w in enumerate(chosen):
        binp = lat / f"{w['segment']}.bin"
        nb = binp.stat().st_size
        ng = nb // CHUNK_BYTES
        if nb % CHUNK_BYTES or w["m"] >= ng:
            raise RuntimeError(f"{binp} 大小 {nb} 与 chunk 布局不符")
        with open(binp, "rb") as f:
            f.seek(w["m"] * CHUNK_BYTES)
            buf = f.read(CHUNK_BYTES)
        w["byte_offset"] = w["m"] * CHUNK_BYTES
        w["chunk_sha256"] = hashlib.sha256(buf).hexdigest()
        w["idx"] = i
    counts = Counter(f"{w['task']}/{'demopad' if w['demopad'] else w['kind']}" for w in chosen)
    return {
        "schema": 1,
        "name": "e2e_windows_v1",
        "plan": "1117-e2e-enc-train-plan.html（一节 e2e_windows_v1.json 行；G1/G1b/G1c/G3 共用）",
        "generator": "scripts/training/g0/e2e_windows_v1.py generate",
        "seed": SEED,
        "per_task": PER_TASK,
        "dataset_root": str(dataset_root),
        "latents_relpath": "wan-latents",
        "chunk_bytes": CHUNK_BYTES,
        "chunk_layout": "float32 (9,16,32,32) C 序；展平 1024c+32h+w",
        "table_layout_768": LAYOUT_768,
        "table_rows_768": meta["num_rows"],
        "motion_index_sha256": idx_sha,
        "table768_sha256": meta["tables"]["motion_token"]["sha256"],
        "num_windows": len(chosen),
        "counts": dict(sorted(counts.items())),
        "fields": ["idx", "task", "h5_file", "g", "episode", "exec_start_idx", "segment", "kind", "m",
                   "window_start", "seg_offset", "table_row", "real_frames", "pad_frames", "demopad",
                   "byte_offset", "chunk_sha256"],
        "windows": chosen,
    }


def dump(manifest: dict, out: pathlib.Path) -> str:
    head = {k: v for k, v in manifest.items() if k != "windows"}
    lines = [json.dumps(w, ensure_ascii=False, separators=(",", ":")) for w in manifest["windows"]]
    head_txt = json.dumps(head, ensure_ascii=False, indent=1)
    text = head_txt[:-2] + ',\n "windows": [\n' + ",\n".join(lines) + "\n ]\n}\n"
    if json.loads(text) != manifest:
        raise RuntimeError("清单序列化往返不一致")
    out.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_manifest(path=DEFAULT_MANIFEST, expect_sha: str | None = None) -> tuple[dict, str]:
    """读清单并校验 sha256（缺省对 ``E2E_WINDOWS_V1_SHA256``）；返回 (manifest, sha)。"""
    path = pathlib.Path(path)
    sha = file_sha256(path)
    want = E2E_WINDOWS_V1_SHA256 if expect_sha is None else expect_sha
    if sha != want:
        raise RuntimeError(f"窗口清单 sha256 {sha} != 提交记录 {want}（{path}）")
    return json.loads(path.read_text()), sha


def select_windows(manifest: dict, max_windows: int | None = None) -> list[dict]:
    """取前 n 窗（``max_windows`` 仅供 CPU 冒烟；按任务轮转取，保持四任务都在）。"""
    ws = manifest["windows"]
    if max_windows is None or max_windows >= len(ws):
        return list(ws)
    by_task: dict[str, list[dict]] = {}
    for w in ws:
        by_task.setdefault(w["task"], []).append(w)
    out, i = [], 0
    while len(out) < max_windows:
        for t in sorted(by_task):
            if i < len(by_task[t]) and len(out) < max_windows:
                out.append(by_task[t][i])
        i += 1
    return sorted(out, key=lambda w: w["idx"])


def read_window_latents(latents_root, windows, *, verify: bool = True) -> np.ndarray:
    """按清单从 ``<latents_root>/<段>.bin`` 只读取源 fp32 latent → (n, 9, 16384) float32。"""
    root = pathlib.Path(latents_root)
    out = np.empty((len(windows), 9, 16384), np.float32)
    for i, w in enumerate(windows):
        with open(root / f"{w['segment']}.bin", "rb") as f:
            f.seek(int(w["byte_offset"]))
            buf = f.read(CHUNK_BYTES)
        if len(buf) != CHUNK_BYTES:
            raise RuntimeError(f"{w['segment']} m={w['m']} 读不足 {CHUNK_BYTES} B")
        if verify and hashlib.sha256(buf).hexdigest() != w["chunk_sha256"]:
            raise RuntimeError(f"{w['segment']} m={w['m']} chunk sha256 不符")
        out[i] = np.frombuffer(buf, dtype="<f4").reshape(9, 16384)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--dataset-root", type=pathlib.Path, default=DEFAULT_DATASET_ROOT)
    g.add_argument("--out", type=pathlib.Path, default=DEFAULT_MANIFEST)
    c = sub.add_parser("check")
    c.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        m = generate(a.dataset_root)
        sha = dump(m, a.out)
        print(f"E2E_WINDOWS_V1 windows={m['num_windows']} counts={m['counts']} sha256={sha} out={a.out}")
        return 0
    m, sha = load_manifest(a.manifest)
    print(f"E2E_WINDOWS_V1=PASS windows={m['num_windows']} manifest_sha={sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
