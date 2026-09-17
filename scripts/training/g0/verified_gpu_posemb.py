"""验证专用 GPU 位置表：计算原生产组件输出，与整张离线表逐位核对后供 CPU 装配。"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import pathlib
import subprocess

import numpy as np

from mme_vla_suite.datastore.framesamp_store import StoreMeta
from mme_vla_suite.datastore.motion_store import sha256_file

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/mme_vla_suite/shared/posemb_3d.py"


def _store_info(store):
    store = pathlib.Path(store).resolve()
    meta = StoreMeta.load(store)
    path = store / meta.spec.pos_table_relpath
    shape = (int(meta.raw["num_pos_rows"]),) + meta.spec.pos_row_shape
    return meta, path, shape


def _check_prefix(table, path, shape):
    offline = np.memmap(path, mode="r", dtype=np.float32, shape=shape)
    if table.dtype != np.float32 or table.shape[1:] != shape[1:] or table.shape[0] < shape[0]:
        raise ValueError("GPU 位置表的 shape/dtype 无法覆盖离线表")
    for start in range(0, shape[0], 64):
        a = table[start:start+64][:shape[0]-start]
        b = offline[start:start+64]
        if a.tobytes() != b.tobytes():
            raise ValueError(f"GPU 位置表与离线表不逐位一致: 起始行 {start}")


def export_table(store, out, max_steps=4096):
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.shared.posemb_3d import PosEmb3D
    if jax.default_backend() != "gpu":
        raise ValueError("必须在 GPU 生成位置表，CPU 的三角函数结果不满足逐位口径")
    out = pathlib.Path(out)
    binary = out.with_suffix(".f32.bin")
    if out.exists() or binary.exists():
        raise FileExistsError("拒绝覆盖已生成的位置表或证明")
    meta, offline_path, offline_shape = _store_info(store)
    grid = math.isqrt(meta.spec.tokens_per_frame)
    table = np.asarray(PosEmb3D(768)(jnp.arange(max_steps), grid))
    _check_prefix(table, offline_path, offline_shape)
    out.parent.mkdir(parents=True, exist_ok=True)
    with binary.open("xb") as f:
        table.tofile(f)
    report = dict(schema=1, passed=True, source_sha256=sha256_file(SOURCE),
                  exporter_sha256=sha256_file(pathlib.Path(__file__)),
                  store=str(pathlib.Path(store).resolve()),
                  store_meta_sha256=sha256_file(pathlib.Path(store)/"meta/store_meta.json"),
                  offline_sha256=sha256_file(offline_path), offline_shape=list(offline_shape),
                  binary=binary.name, binary_sha256=sha256_file(binary),
                  shape=list(table.shape), dtype=str(table.dtype), grid=grid,
                  temporal_base=10000, spatial_base=1000,
                  devices=[str(d) for d in jax.devices()],
                  versions={k:importlib.metadata.version(k) for k in ("jax","jaxlib","numpy")},
                  source_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  source_status=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True))
    with out.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"GPU_POSEMB=PASS offline_rows={offline_shape[0]} cached_rows={max_steps} grid={grid} mismatches=0", flush=True)
    return report


def load_verified_table(report_path, store, tokens_per_frame):
    report_path = pathlib.Path(report_path)
    report = json.loads(report_path.read_text())
    meta, offline_path, offline_shape = _store_info(store)
    expected = {
        "schema": 1, "passed": True, "source_sha256": sha256_file(SOURCE),
        "exporter_sha256": sha256_file(pathlib.Path(__file__)),
        "store": str(pathlib.Path(store).resolve()),
        "store_meta_sha256": sha256_file(pathlib.Path(store)/"meta/store_meta.json"),
        "offline_sha256": sha256_file(offline_path), "offline_shape": list(offline_shape),
        "shape": [4096, tokens_per_frame, 768], "dtype": "float32",
        "grid": math.isqrt(tokens_per_frame), "temporal_base": 10000, "spatial_base": 1000,
    }
    if meta.spec.tokens_per_frame != tokens_per_frame or any(report.get(k) != v for k,v in expected.items()):
        raise ValueError("GPU 位置表的生产源码、配置或数据来源不一致")
    binary = report_path.parent / report["binary"]
    if binary.parent.resolve() != report_path.parent.resolve() or binary.is_symlink():
        raise ValueError("GPU 位置表路径越界或为符号链接")
    shape = tuple(report["shape"])
    if binary.stat().st_size != math.prod(shape)*4 or sha256_file(binary) != report["binary_sha256"]:
        raise ValueError("GPU 位置表字节长度或 SHA 不符")
    table = np.memmap(binary, mode="r", dtype=np.float32, shape=shape)
    _check_prefix(table, offline_path, offline_shape)
    return table, {"report_sha256": sha256_file(report_path), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    export_table(args.store, args.out)


if __name__ == "__main__":
    main()
