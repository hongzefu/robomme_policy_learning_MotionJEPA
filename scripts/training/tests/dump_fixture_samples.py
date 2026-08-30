#!/usr/bin/env python3
"""定点样本 / 定点 batch dump（dtype 统一修复第一块验证的取证工具）。

**它解决什么问题**：dtype 修复没有运行时开关，改动即行为——「修复前」与「修复后」
的交付内容只能分别在两个 clean HEAD 上各取一次证，再离线对拍。本脚本就是取证的
那一次跑，两侧用完全相同的定点样本集（由 `episode_manifest.json` 精确算出，不靠
shuffle 撞边界），因此两份产物逐键可比。

**两层取证，对应两层判据**：
1. 逐样本层（约 2,600 个）走裸数据集（commitV4.1 起为 packed `FrameSampDataset`，
   经 `_create_framesamp_dataset` 构造）——直接看 padding 产物，不被后续
   transforms 掩盖；
2. batch 层（200 个）走完整 `transform_dataset` + `_collate_fn`——与 P6 `batch_digests`
   的记录点（collate 后、device_put 前）逐字对齐，专验「batch 内含短样本时
   `np.stack` 把整批抬成 f64」这一行为的消失。

**落什么、不落什么**：全部键都记 raw + canonical 双口径 sha256 与 dtype/shape；
memory 四键额外落数组本体（位型容器），供失配时给出元素级 hex 定位。其余键与本
修复无关，位相同由 raw sha256 判定即足够。batch 层只记摘要——batch 由样本 stack
而成，元素级定位回到样本层即可，而 200 个 batch 的数组本体是十几 GB 的无谓开销。

用法（cwd 必须是仓库根——`get_history_config` 按相对路径加载 yaml）：

    DTYPE_DUMP_DIR=v1-store/dtype-unify/<run_name> JAX_PLATFORMS=cpu \\
    UV_LINK_MODE=copy uv run scripts/training/tests/dump_fixture_samples.py mme_vla_suite \\
        --assets-base-dir v1-store/train-assets \\
        --dataset-path v1-store/datasets/4task-gl \\
        --model.use-history --model.history-config perceptual-framesamp-context.yaml \\
        --no-wandb-enabled

可调环境变量：`DTYPE_DUMP_MODE`（samples|batches|both，默认 both）、
`DTYPE_DUMP_ARRAYS`（1|0，默认 1，是否落 memory 四键数组本体）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _common as C  # noqa: E402
import numpy as np  # noqa: E402

from mme_vla_suite.models.config.utils import get_history_config  # noqa: E402
import mme_vla_suite.training.config as _config  # noqa: E402
from mme_vla_suite.training.dataloader import _create_framesamp_dataset  # noqa: E402
from openpi.training.data_loader import _collate_fn  # noqa: E402
from openpi.training.data_loader import transform_dataset  # noqa: E402

_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"


def _out_dir() -> pathlib.Path:
    raw = os.environ.get("DTYPE_DUMP_DIR")
    if not raw:
        raise SystemExit("必须设置 DTYPE_DUMP_DIR 指定输出目录")
    d = pathlib.Path(raw)
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖既有取证产物: {d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_manifest_same_source(manifest_path: pathlib.Path, dataset_path: pathlib.Path) -> dict:
    """核对 manifest 与数据集同源：数据集 provenance 里记的 manifest sha 必须与实物相符。"""
    manifest = C.load_manifest(manifest_path)
    prov_p = dataset_path / "meta" / "provenance.json"
    if prov_p.exists():
        prov = json.loads(prov_p.read_text(encoding="utf-8"))
        want = prov.get("manifest_sha256")
        got = manifest.get("sha256")
        if want and got and want != got:
            raise SystemExit(
                f"manifest 与数据集不同源: provenance.manifest_sha256={want} 实物 sha256={got}"
            )
    return manifest


def _dump_samples(ds, manifest: dict, groups: dict, out: pathlib.Path, with_arrays: bool) -> int:
    sdir = out / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    rows = sdir / "summary.jsonl"
    n = 0
    t0 = time.time()
    with rows.open("w") as f:
        for gname, idxs in groups.items():
            for idx in idxs:
                item = ds[idx]
                epis, step = C.resolve_index(manifest, idx)
                # 同源自校验：manifest 算出的 (epis, step) 必须与 pkl 里记的一致
                got_epis = int(np.asarray(item["epis_idx"]).item())
                got_step = int(np.asarray(item["step_idx"]).item())
                if (got_epis, got_step) != (epis, step):
                    raise SystemExit(
                        f"index {idx} 同源校验失败: manifest 算出 ({epis},{step})，"
                        f"pkl 记的是 ({got_epis},{got_step})"
                    )
                keys = C.describe_tree(item)
                if with_arrays:
                    adir = sdir / "arrays" / str(idx)
                    for k, d in keys.items():
                        if d["kind"] == "array" and C.base_name(k) in C.MEMORY_KEYS:
                            C.save_array(adir, k, np.asarray(item[C.base_name(k)]))
                f.write(json.dumps({
                    "index": idx, "group": gname, "epis_idx": epis, "step_idx": step,
                    "is_short": step <= 30, "keys": keys,
                }, ensure_ascii=False) + "\n")
                n += 1
                if n % 100 == 0:
                    print(f"[samples] {n} 个已 dump，用时 {time.time() - t0:.1f}s", flush=True)
    print(f"[samples] 完成 {n} 个，用时 {time.time() - t0:.1f}s", flush=True)
    return n


def _dump_batches(tds, plan: list[dict], out: pathlib.Path) -> int:
    bdir = out / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    rows = bdir / "summary.jsonl"
    n = 0
    t0 = time.time()
    with rows.open("w") as f:
        for spec in plan:
            items = [tds[i] for i in spec["indices"]]
            batch = _collate_fn(items)
            keys = C.describe_tree(batch)
            f.write(json.dumps({
                "batch_id": spec["batch_id"], "kind": spec["kind"],
                "indices": spec["indices"], "keys": keys,
            }, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"[batches] {n} 个已 dump，用时 {time.time() - t0:.1f}s", flush=True)
    print(f"[batches] 完成 {n} 个，用时 {time.time() - t0:.1f}s", flush=True)
    return n


def main() -> None:
    config = _config.cli()
    if config.model.history_config != _EXPECTED_HISTORY_CONFIG:
        raise SystemExit(f"本工具只支持 {_EXPECTED_HISTORY_CONFIG}（当前 {config.model.history_config}）")

    out = _out_dir()
    mode = os.environ.get("DTYPE_DUMP_MODE", "both")
    with_arrays = os.environ.get("DTYPE_DUMP_ARRAYS", "1") != "0"

    dataset_path = pathlib.Path(config.dataset_path)
    manifest_path = C.REPO_ROOT / "v1-store" / "episode_manifest.json"
    manifest = _check_manifest_same_source(manifest_path, dataset_path)

    groups = C.build_fixture_indices(manifest)
    plan = C.build_fixture_batches(groups)
    # 冒烟开关：每组只取前 N 个样本、每种组成只取前 N 个 batch。正式取证不设它。
    # 两侧只要 limit 相同，裁剪后的定点集就仍然逐项一致，对拍的 plan 断言照常成立。
    limit = int(os.environ.get("DTYPE_DUMP_LIMIT", "0") or 0)
    if limit > 0:
        groups = {k: v[:limit] for k, v in groups.items()}
        keep, seen = [], {}
        for spec in plan:
            seen[spec["kind"]] = seen.get(spec["kind"], 0) + 1
            if seen[spec["kind"]] <= limit:
                keep.append(spec)
        plan = keep
    (out / "fixture_plan.json").write_text(json.dumps({
        "seed": C.FIXTURE_SEED, "limit": limit, "groups": groups, "batches": plan,
        "manifest_sha256": manifest.get("sha256"),
    }, ensure_ascii=False), encoding="utf-8")

    history_config = get_history_config(config.model.history_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    ds = _create_framesamp_dataset(
        dataset_path=str(dataset_path),
        data_config=data_config,
        history_config=history_config,
        action_horizon=config.model.action_horizon,
    )

    n_samples = n_batches = 0
    if mode in ("samples", "both"):
        n_samples = _dump_samples(ds, manifest, groups, out, with_arrays)
    if mode in ("batches", "both"):
        tds = transform_dataset(ds, data_config)
        n_batches = _dump_batches(tds, plan, out)

    C.write_manifest(out, extra={
        "git_head": os.environ.get("DTYPE_DUMP_GIT_HEAD", ""),
        "n_samples": n_samples, "n_batches": n_batches,
        "with_arrays": with_arrays, "mode": mode,
        "dataset_path": str(dataset_path),
    })
    print(f"DUMP_DONE samples={n_samples} batches={n_batches} out={out}", flush=True)


if __name__ == "__main__":
    main()
