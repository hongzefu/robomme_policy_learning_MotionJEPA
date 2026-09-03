#!/usr/bin/env python3
"""基线环境指纹：dump（起跑留档）/ check（引用基线产物前的强制 preflight）/ manifest。

对应 v1-gradient-baseline.md 四节与 T5。三个子命令：

- `dump --record-dir DIR [--dataset PATH]`：采集当前环境指纹，合并进
  DIR/env.json 的 `fingerprint` 键（无 env.json 时单独写 fingerprint.json）。
  刻意跑在 `JAX_PLATFORMS=cpu` 下：GPU 身份由 nvidia-smi 提供，不为取指纹初始化
  CUDA（避免与随后的训练进程抢显存）。
- `check --baseline RECORDS_DIR [--steps N --batch-size N] [--dataset PATH]`：
  重新采集当前指纹，与基线 env.json 的 `fingerprint` 逐项比对；校验基线
  BASELINE_MANIFEST.json 全部条目 sha256；断言对拍 run 的单 epoch 约束
  `steps × batch_size < 395289`。输出单行 `BASELINE_ENV=PASS|FAIL`，FAIL 非零退出
  并列出逐项差异。触发任一失效条件 → 基线作废，必须重跑（登记簿记新版本）。
- `manifest RECORDS_DIR`：为固化产物生成 BASELINE_MANIFEST.json
  （逐产物 sha256 / 字节数 / 行数 / schema 版本，防产物腐烂与工具漂移）。

指纹 schema `baseline-env-fingerprint-v1`（大文件取「逐文件 (相对路径, 字节数,
首尾 1 MiB blake2b)」的聚合 sha256，方式在此定死、写进 schema 名——T5 第 3 条）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
def _epoch_samples(dataset: pathlib.Path) -> int:
    """epoch 样本数真值源（motion-memory-plan.md 2.8 / R14）：packed 根读 store_meta.num_exec_samples，
    旧 source 根读 stats.execution_samples；两者同时存在却不等、或都读不到即 raise。"""
    vals = {}
    sm = dataset / "meta" / "store_meta.json"
    st = dataset / "meta" / "stats.json"
    if sm.is_file():
        vals["store_meta"] = int(json.load(open(sm))["num_exec_samples"])
    if st.is_file():
        vals["stats"] = int(json.load(open(st))["execution_samples"])
    if not vals:
        raise ValueError(f"epoch 样本数无法从 {dataset}/meta/{{store_meta,stats}}.json 读出")
    if len(set(vals.values())) != 1:
        raise ValueError(f"epoch 样本数两处不等: {vals}")
    return next(iter(vals.values()))
_SCHEMA = "baseline-env-fingerprint-v1"
_MANIFEST_SCHEMA = "baseline-manifest-v1"
_SPOT_N = 16                     # 数据集抽样文件数（IO 重构计划 source_spot_sha256 口径）
_HEADTAIL = 1024 * 1024          # 大文件首尾各 1 MiB


def _sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _headtail_digest(p: pathlib.Path) -> tuple[int, str]:
    """(字节数, 首尾各 1 MiB 的 blake2b-128)；小于 2 MiB 覆盖全文件。"""
    size = p.stat().st_size
    h = hashlib.blake2b(digest_size=16)
    with p.open("rb") as f:
        if size <= 2 * _HEADTAIL:
            h.update(f.read())
        else:
            h.update(f.read(_HEADTAIL))
            f.seek(size - _HEADTAIL)
            h.update(f.read(_HEADTAIL))
    return size, h.hexdigest()


def _tree_spot_digest(root: pathlib.Path) -> dict:
    """目录树指纹：全部文件的 (相对路径, 字节数, 首尾 1 MiB blake2b) 聚合 sha256。"""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    g = hashlib.sha256()
    for p in files:
        size, d = _headtail_digest(p)
        g.update(f"{p.relative_to(root)}:{size}:{d}\n".encode())
    return {"scheme": "tree-relpath-size-headtail1MiB", "n_files": len(files),
            "digest": g.hexdigest()}


def _dataset_spot_digest(dataset: pathlib.Path) -> dict:
    """数据集抽样指纹：meta/stats.json + provenance.json 全量 sha256，
    加 data/ 排序名单中等距抽 16 个文件的 (名, 字节数, 首尾 1 MiB) 聚合。"""
    g = hashlib.sha256()
    for rel in ("meta/stats.json", "provenance.json"):
        p = dataset / rel
        if p.exists():
            g.update(f"{rel}:{_sha256_file(p)}\n".encode())
    data_dir = dataset / "data"
    names = sorted(e.name for e in os.scandir(data_dir) if e.is_file())
    idxs = [i * (len(names) - 1) // (_SPOT_N - 1) for i in range(_SPOT_N)] \
        if len(names) >= _SPOT_N else list(range(len(names)))
    for i in sorted(set(idxs)):
        p = data_dir / names[i]
        size, d = _headtail_digest(p)
        g.update(f"data/{names[i]}:{size}:{d}\n".encode())
    return {"scheme": f"stats+provenance+sorted-scandir-{_SPOT_N}-headtail1MiB",
            "n_data_files": len(names), "digest": g.hexdigest()}


def collect_fingerprint(dataset: pathlib.Path) -> dict:
    import importlib.metadata as md
    v1_store = _REPO_ROOT / "v1-store"
    models = v1_store / "models"
    fp: dict = {"schema": _SCHEMA}
    fp["uv_lock_sha256"] = _sha256_file(_REPO_ROOT / "uv.lock")
    fp["packages"] = {name: md.version(name)
                     for name in ("torch", "jax", "jaxlib", "numpy", "ml_dtypes")}
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True)
    fp["gpu"] = {"nvidia_smi": smi.stdout.strip().splitlines(),
                 "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    fp["assets"] = {
        "norm_stats_sha256": _sha256_file(
            v1_store / "train-assets/mme_vla_suite/robomme/norm_stats.json"),
        "tokenizer_sha256": _sha256_file(
            models / "big_vision/paligemma_tokenizer.model"),
        "pi05_base": _tree_spot_digest(
            models / "openpi-assets/checkpoints/pi05_base/params"),
        "episode_manifest_sha256_field": json.load(
            open(v1_store / "episode_manifest.json"))["sha256"],
        "dataset_spot": _dataset_spot_digest(dataset),
    }
    fp["xla"] = {
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "XLA_PYTHON_CLIENT_MEM_FRACTION":
            os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
    }
    import jax  # JAX_PLATFORMS=cpu 下 import，软件配置项与 GPU 无关
    fp["jax_config"] = {
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
    }
    return fp


def _diff(a, b, prefix="fingerprint"):
    """逐项深比较，返回差异行列表。"""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{prefix}.{k}")
    elif a != b:
        out.append(f"  {prefix}: 基线={a!r} 当前={b!r}")
    return out


def cmd_dump(args) -> int:
    fp = collect_fingerprint(pathlib.Path(args.dataset))
    d = pathlib.Path(args.record_dir)
    env_path = d / "env.json"
    if env_path.exists():
        env = json.load(open(env_path))
        env["fingerprint"] = fp
        json.dump(env, open(env_path, "w"), indent=2, ensure_ascii=False)
        print(f"OK 指纹并入 {env_path}")
    else:
        d.mkdir(parents=True, exist_ok=True)
        json.dump(fp, open(d / "fingerprint.json", "w"), indent=2, ensure_ascii=False)
        print(f"OK 指纹写入 {d / 'fingerprint.json'}")
    return 0


def cmd_manifest(args) -> int:
    d = pathlib.Path(args.records_dir)
    entries = {}
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.name == "BASELINE_MANIFEST.json":
            continue
        rel = str(p.relative_to(d))
        row = {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
        if p.suffix in (".jsonl", ".tsv"):
            row["lines"] = sum(1 for _ in p.open())
        entries[rel] = row
    json.dump({"schema": _MANIFEST_SCHEMA, "files": entries},
              open(d / "BASELINE_MANIFEST.json", "w"), indent=2, ensure_ascii=False)
    print(f"OK BASELINE_MANIFEST.json: {len(entries)} 个产物")
    return 0


def cmd_check(args) -> int:
    base = pathlib.Path(args.baseline)
    fails: list[str] = []

    env_path = base / "env.json"
    baseline_fp = (json.load(open(env_path)).get("fingerprint")
                   if env_path.exists() else None)
    if baseline_fp is None:
        fails.append(f"  基线 {env_path} 缺 fingerprint 键（基线本身留档不完整）")
    else:
        cur = collect_fingerprint(pathlib.Path(args.dataset))
        fails += _diff(baseline_fp, cur)

    man_path = base / "BASELINE_MANIFEST.json"
    if not man_path.exists():
        fails.append(f"  缺 {man_path}")
    else:
        man = json.load(open(man_path))
        for rel, row in man["files"].items():
            p = base / rel
            if not p.exists():
                fails.append(f"  产物缺失: {rel}")
            elif _sha256_file(p) != row["sha256"]:
                fails.append(f"  产物 sha256 不符（产物腐烂）: {rel}")

    if args.steps and args.batch_size:
        n = args.steps * args.batch_size
        epoch_samples = _epoch_samples(pathlib.Path(args.dataset))
        if n >= epoch_samples:
            fails.append(f"  单 epoch 约束违反: steps×batch = {n} ≥ {epoch_samples}"
                         "（跨 epoch 后 index 序列与 num_workers 相关，对拍失去意义）")

    if fails:
        print("BASELINE_ENV=FAIL 逐项差异:")
        print("\n".join(fails))
        return 1
    print("BASELINE_ENV=PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_dataset = str(_REPO_ROOT / "v1-store/datasets/4task-gl")

    p = sub.add_parser("dump")
    p.add_argument("--record-dir", required=True)
    p.add_argument("--dataset", default=default_dataset)

    p = sub.add_parser("check")
    p.add_argument("--baseline", required=True)
    p.add_argument("--dataset", default=default_dataset)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)

    p = sub.add_parser("manifest")
    p.add_argument("records_dir")

    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    return {"dump": cmd_dump, "check": cmd_check, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
