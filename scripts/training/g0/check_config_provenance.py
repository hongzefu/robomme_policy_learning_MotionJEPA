#!/usr/bin/env python3
"""TIC 第 0 关（L0）：配置与库同源核对——归一化统计量 / 库指纹 / motion 库路径 / checkpoint 参数树与 dtype。

对应 TIC 计划第二部分 2.2。骨架仿 `scripts/training/g0/check_baseline_env.py`（单文件、逐条判定行、
唯一成功行、FAIL 非零退出）。全程 CPU（`JAX_PLATFORMS=cpu` + `CUDA_VISIBLE_DEVICES=""`），
不初始化 CUDA、不起 sidecar、不读 GPU。

五条判定行（前四条阻断，第五条观察）：

1. `NORM_STATS_SAME` —— 训练侧走生产口径 `mme_vla_suite.training.config.get_config(<train-config>)`
   的 `data.create(assets_dirs, model)` **实际加载出的 `data_config.norm_stats` 对象**（不是自己读文件），
   推理侧走 `openpi.training.checkpoints.load_norm_stats(<ckpt>/assets, <asset_id>)`；两侧每个 key 的
   mean / std / q01 / q99 用 `scripts/training/tests/_common.py::leaf_sha256` 逐位比（dtype 入哈希域）。
   两边尺子不同 ⇒ 训练与推理的归一化不同源，后面一切对拍无意义。

2. `LIB_PROVENANCE_MATCH` —— 从当前库现算六个指纹，与训练 run 的 `motion_provenance.json` 同名字段逐个相等：
   `meta/episode_manifest.json`（两字段 framesamp_manifest_sha256 / motion_manifest_sha256 共用同一文件，
   口径是 `datastore.manifest.manifest_sha256` 的规范化摘要，**不是**文件全文 sha256）、
   `framesamp/meta/store_meta.json`、`motion/meta/motion_index.json`、`motion/meta/store_meta.json`、
   `motion/motion_token.f32.bin`（后四个是文件全文 sha256）。证明本轮对拍用的就是训练时那张表。

3. `MOTION_STORE_PATH` —— 三件事：(a) provenance 里 motion 库与 framesamp 库绑的清单摘要相同；
   (b) 负向：把 `MMEVLA_MOTION_STORE` 指向另一个（40ep）库的 motion 目录后调
   `mme_vla_suite.training.dataloader._motion_gates`，必须被 `motion_store.check_same_source` raise；
   同一路子顺带回答「快照里记错的库路径会不会静默用错库」——本 run 的
   `history_config.resolved.yaml` 把 `motion.store_path` 记成了 40ep 库（训练时靠 `MMEVLA_MOTION_STORE`
   覆盖到 400ep），不设覆盖时同源闸同样 raise，故记错的快照路径不会静默生效（`snapshot_default=raise`）；
   (c) 静态：`src/mme_vla_suite/policies/` 下没有任何 `MotionStore` / `MMEVLA_MOTION_STORE` 引用
   （在线侧不读离线表，故快照里记错的库路径不会静默影响推理）。

4. `CKPT_PARAM_TREE` —— 按生产口径（run 内 `history_config.resolved.yaml` 快照 + `use_history=True`）
   建模型后调 `mme_vla_suite.policies.policy_config._assert_param_tree_exact`，missing / extra 均须为空。
   `leaves=` 是 checkpoint 参数叶数实测（预期 59；193 是 TrainState 叶数，两者不混用）。

5. `CKPT_DTYPE_PROFILE` —— `restore_params(dtype=None)` 逐叶 dtype 画像（观察项，不阻断）。

用法：
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= UV_LINK_MODE=copy uv run --no-sync python \
    scripts/training/g0/check_config_provenance.py [--out <records>/l0.json]
唯一成功行：`TIC_L0=PASS`；任一阻断项 FAIL 即打 `TIC_L0=FAIL` 并以非零码退出。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import traceback

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "training" / "tests"))

# 生产口径默认值（TIC 计划第二部分〇节）
DEF_CKPT = "v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999"
DEF_LIB = "v1-store/datasets/4task-motion-400ep"
DEF_TRAIN_CONFIG = "mme_vla_suite_b128"
NEG_LIB = "v1-store/datasets/4task-motion-40ep"   # 负向臂：另一份 verified 但清单不同的库

# provenance 字段 → (库内相对路径, 摘要口径)
PROV_FIELDS = (
    ("framesamp_manifest_sha256", "meta/episode_manifest.json", "manifest"),
    ("motion_manifest_sha256", "meta/episode_manifest.json", "manifest"),
    ("framesamp_store_meta_sha256", "framesamp/meta/store_meta.json", "file"),
    ("motion_index_sha256", "motion/meta/motion_index.json", "file"),
    ("motion_store_meta_sha256", "motion/meta/store_meta.json", "file"),
    ("motion_table_sha256", "motion/motion_token.f32.bin", "file"),
)

NORM_ARRAY_FIELDS = ("mean", "std", "q01", "q99")


def _abs(p: str) -> pathlib.Path:
    q = pathlib.Path(p)
    return q if q.is_absolute() else (_REPO_ROOT / q)


def _sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 判定 1：归一化统计量同源 ───────────────────────────────────────────────────

def gate_norm_stats(args, rec: dict) -> tuple[str, bool]:
    import _common as C
    import numpy as np
    from openpi.training import checkpoints as _checkpoints
    import mme_vla_suite.training.config as _mconfig

    train_config = _mconfig.get_config(args.train_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    train_ns = data_config.norm_stats
    asset_id = data_config.asset_id
    if asset_id is None:
        raise ValueError(f"train config {args.train_config} 的 data_config.asset_id 为 None，无法定位推理侧 norm_stats")
    if train_ns is None:
        raise ValueError(f"train config {args.train_config} 的 data_config.norm_stats 为 None（训练侧尺子没加载出来）")

    assets_dir = getattr(getattr(train_config.data, "assets", None), "assets_dir", None)
    train_src = str(_abs(assets_dir) / asset_id / "norm_stats.json") if assets_dir \
        else str(train_config.assets_dirs / asset_id / "norm_stats.json")
    ckpt = _abs(args.ckpt)
    infer_src = str(ckpt / "assets" / asset_id / "norm_stats.json")
    infer_ns = _checkpoints.load_norm_stats(ckpt / "assets", asset_id)
    if infer_ns is None:
        raise ValueError(f"推理侧 norm_stats 加载为 None: {infer_src}")

    diffs: list[str] = []
    if set(train_ns) != set(infer_ns):
        diffs.append(f"key 集合不同: 训练 {sorted(train_ns)} vs 推理 {sorted(infer_ns)}")
    n_arrays = 0
    agg = hashlib.sha256()
    for key in sorted(set(train_ns) & set(infer_ns)):
        for fld in NORM_ARRAY_FIELDS:
            a = getattr(train_ns[key], fld, None)
            b = getattr(infer_ns[key], fld, None)
            if a is None and b is None:
                continue
            if a is None or b is None:
                diffs.append(f"{key}.{fld}: 一侧为 None（训练 {a is None} / 推理 {b is None}）")
                continue
            sa = C.leaf_sha256(np.asarray(a))
            sb = C.leaf_sha256(np.asarray(b))
            n_arrays += 1
            agg.update(f"{key}.{fld}:{sa}\n".encode())
            if sa != sb:
                diffs.append(f"{key}.{fld}: leaf_sha256 {sa[:16]}… != {sb[:16]}…")
    ok = not diffs
    rec["norm_stats"] = {"train_src": train_src, "infer_src": infer_src, "asset_id": asset_id,
                         "keys": sorted(set(train_ns) | set(infer_ns)), "arrays": n_arrays,
                         "agg_sha256": agg.hexdigest(), "diffs": diffs}
    line = (f"NORM_STATS_SAME={'PASS' if ok else 'FAIL'} sha={agg.hexdigest()[:16]} "
            f"train_src={train_src} infer_src={infer_src} "
            f"keys=[{','.join(sorted(set(train_ns) | set(infer_ns)))}] arrays={n_arrays}")
    if diffs:
        line += "\n" + "\n".join(f"  {d}" for d in diffs)
    return line, ok


# ── 判定 2：库指纹与训练 run provenance 逐字段相等 ─────────────────────────────

def gate_lib_provenance(args, rec: dict) -> tuple[str, bool]:
    from mme_vla_suite.datastore.manifest import manifest_sha256

    lib = _abs(args.lib)
    prov = json.loads(_abs(args.provenance).read_text(encoding="utf-8"))
    fields: dict[str, dict] = {}
    diffs: list[str] = []
    for field, rel, scheme in PROV_FIELDS:
        p = lib / rel
        if not p.is_file():
            diffs.append(f"{field}: 库内文件缺失 {p}")
            fields[field] = {"path": str(p), "scheme": scheme, "current": None,
                             "provenance": prov.get(field), "equal": False}
            continue
        cur = (manifest_sha256(json.loads(p.read_text(encoding="utf-8")))
               if scheme == "manifest" else _sha256_file(p))
        want = prov.get(field)
        eq = (cur == want)
        fields[field] = {"path": str(p), "scheme": scheme, "current": cur, "provenance": want, "equal": eq}
        if not eq:
            diffs.append(f"{field}: 现算 {cur[:16]}… != provenance {str(want)[:16]}…（{rel}）")
    ok = not diffs
    rec["lib_provenance"] = {"lib": str(lib), "provenance": str(_abs(args.provenance)), "fields": fields}
    line = ("LIB_PROVENANCE_MATCH=" + ("PASS" if ok else "FAIL") + " "
            + " ".join(f"{f}={(fields[f]['current'] or 'MISSING')[:16]}" for f, _, _ in PROV_FIELDS)
            + f" all_equal={int(ok)}")
    if diffs:
        line += "\n" + "\n".join(f"  {d}" for d in diffs)
    return line, ok


# ── 判定 3：motion 库路径（正向 + 负向 + 静态） ───────────────────────────────

def gate_motion_store_path(args, rec: dict) -> tuple[str, bool]:
    from mme_vla_suite.datastore import StoreMeta
    from mme_vla_suite.policies.policy_config import _load_resolved_snapshot
    from mme_vla_suite.training.dataloader import _motion_gates

    lib = _abs(args.lib)
    neg_lib = _abs(args.neg_lib)
    run_root = _abs(args.ckpt).parent
    prov = json.loads(_abs(args.provenance).read_text(encoding="utf-8"))
    fails: list[str] = []

    # (a) provenance 内两库绑定的清单摘要相同
    same_manifest = prov.get("framesamp_manifest_sha256") == prov.get("motion_manifest_sha256")
    if not same_manifest:
        fails.append("provenance 的 framesamp_manifest_sha256 != motion_manifest_sha256（两库不同源）")

    history_config, snapshot_enabled = _load_resolved_snapshot(run_root)
    if not snapshot_enabled:
        fails.append(f"run 快照 motion.enabled=false，与本关前提（motion 开启态 run）不符: {run_root}")
    snapshot_store = str(getattr(history_config, "motion", {}).get("store_path", ""))
    frame_meta = StoreMeta.load(lib / "framesamp")
    saved = os.environ.get("MMEVLA_MOTION_STORE")

    def _gates_with(root: str | None):
        """在指定 MMEVLA_MOTION_STORE 覆盖（None = 不设，走快照 store_path）下调一次闸，返回 (结果, 异常)。"""
        prev = os.environ.pop("MMEVLA_MOTION_STORE", None)
        if root is not None:
            os.environ["MMEVLA_MOTION_STORE"] = root
        try:
            return _motion_gates(history_config, frame_meta), None
        except ValueError as e:
            return None, e
        finally:
            os.environ.pop("MMEVLA_MOTION_STORE", None)
            if prev is not None:
                os.environ["MMEVLA_MOTION_STORE"] = prev

    # 正向：训练与对拍的真实生效方式——env 覆盖指向本轮 --lib 的 motion 目录
    effective, eff_err = _gates_with(str(lib / "motion"))
    if eff_err is not None:
        fails.append(f"正向（MMEVLA_MOTION_STORE={lib / 'motion'}）意外 raise: {eff_err}")
    elif effective is None:
        fails.append("正向 _motion_gates 返回 None（motion 闸未开）")
    elif pathlib.Path(effective).resolve() != (lib / "motion").resolve():
        fails.append(f"正向生效 motion 根 {effective} != 期望 {lib / 'motion'}")

    # 快照默认值：本 run 的 history_config.resolved.yaml 把 store_path 记成了 40ep 库（训练时靠 env 覆盖到 400ep）。
    # 「快照里记错的库路径会不会静默用错库」的答案由此给出：不设覆盖时同源闸直接 raise，不会静默。
    snap_eff, snap_err = _gates_with(None)
    snapshot_default = "raise" if snap_err is not None else str(snap_eff)
    snapshot_matches_lib = pathlib.Path(snapshot_store or ".").resolve() == (lib / "motion").resolve() \
        if snapshot_store else False
    if not snapshot_matches_lib and snap_err is None:
        fails.append(f"快照 store_path={snapshot_store} 与 --lib 不同却未被同源闸拦下（静默用错库）: 返回 {snap_eff}")

    # (b) 负向：MMEVLA_MOTION_STORE 指另一库的 motion 目录，必须 raise
    got, err = _gates_with(str(neg_lib / "motion"))
    guard = "raise" if err is not None else "no-raise"
    guard_msg = str(err) if err is not None else ""
    if err is None:
        fails.append(f"负向未 raise：MMEVLA_MOTION_STORE={neg_lib / 'motion'} 竟返回 {got}")
    elif "绑定的清单不同" not in guard_msg:
        fails.append(f"负向 raise 了但不是同源闸的报错: {guard_msg[:160]}")
    if saved is not None and os.environ.get("MMEVLA_MOTION_STORE") != saved:
        fails.append("环境变量 MMEVLA_MOTION_STORE 未复原（脚本自身副作用）")

    # (c) 静态：在线侧不 import 离线表 / 不读 env 覆盖
    hits = subprocess.run(
        ["grep", "-rn", "-e", "MotionStore", "-e", "MMEVLA_MOTION_STORE",
         str(_REPO_ROOT / "src" / "mme_vla_suite" / "policies")],
        capture_output=True, text=True).stdout.strip()
    infer_reads_store = 0 if not hits else len(hits.splitlines())
    if infer_reads_store:
        fails.append(f"src/mme_vla_suite/policies/ 出现离线表引用 {infer_reads_store} 处:\n{hits}")

    ok = not fails
    rec["motion_store_path"] = {
        "snapshot_store_path": snapshot_store, "provenance_motion_root": prov.get("motion_root"),
        "effective": effective, "snapshot_default": snapshot_default,
        "snapshot_default_msg": (str(snap_err)[:300] if snap_err is not None else ""),
        "neg_lib": str(neg_lib / "motion"), "guard": guard,
        "guard_msg": guard_msg, "infer_reads_store": infer_reads_store,
        "same_manifest": same_manifest, "fails": fails}
    line = (f"MOTION_STORE_PATH={'PASS' if ok else 'FAIL'} snapshot={snapshot_store} "
            f"provenance={prov.get('motion_root')} effective={effective} "
            f"override=MMEVLA_MOTION_STORE guard={guard} infer_reads_store={infer_reads_store} "
            f"same_manifest={int(bool(same_manifest))} snapshot_default={snapshot_default}")
    if fails:
        line += "\n" + "\n".join(f"  {f}" for f in fails)
    return line, ok


# ── 判定 4 / 5：checkpoint 参数树与 dtype 画像 ────────────────────────────────

def _paths_of(tree) -> set[str]:
    """与 `policy_config._assert_param_tree_exact.paths_of` 逐字同口径。"""
    import jax
    return {"/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in kp)
            for kp, _ in jax.tree_util.tree_flatten_with_path(tree)[0]}


def gate_ckpt_param_tree(args, rec: dict):
    """返回 (判定 4 行, ok4, 判定 5 行, model, train_config)——两条判定共用同一次模型构造。"""
    import dataclasses

    import jax
    import jax.numpy as jnp
    from flax import nnx

    import openpi.models.model as _model
    import mme_vla_suite.training.config as _mconfig
    from mme_vla_suite.policies.policy_config import _assert_param_tree_exact, _load_resolved_snapshot, _params_have_motion

    ckpt = _abs(args.ckpt)
    run_root = ckpt.parent
    train_config = _mconfig.get_config(args.train_config)
    history_config, snapshot_enabled = _load_resolved_snapshot(run_root)

    # 生产口径：create_trained_policy 的 strict 分支（bf16 恢复 + 快照 history_config + use_history=True）
    params = _model.restore_params(ckpt / "params", dtype=jnp.bfloat16)
    ckpt_has_motion = _params_have_motion(params)
    if ckpt_has_motion != snapshot_enabled:
        raise ValueError(f"快照 motion.enabled={snapshot_enabled} 与 checkpoint 含 motion 参数={ckpt_has_motion} 不符")
    tc = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, history_config=history_config, use_history=True))
    model = tc.model.load(params, remove_extra_params=False)

    model_paths = _paths_of(nnx.state(model, nnx.Param).to_pure_dict())
    ckpt_paths = _paths_of(params)
    missing = sorted(model_paths - ckpt_paths)
    extra = sorted(ckpt_paths - model_paths)
    err = ""
    try:
        _assert_param_tree_exact(model, params)
    except ValueError as e:            # 与手算集合互为交叉印证
        err = str(e)
    leaves = len(jax.tree_util.tree_leaves(params))
    ok4 = not missing and not extra and not err
    rec["ckpt_param_tree"] = {"missing": missing[:20], "n_missing": len(missing),
                              "extra": extra[:20], "n_extra": len(extra),
                              "leaves": leaves, "model_leaves": len(model_paths),
                              "ckpt_has_motion": ckpt_has_motion, "raise_msg": err}
    line4 = (f"CKPT_PARAM_TREE={'PASS' if ok4 else 'FAIL'} missing={len(missing)} extra={len(extra)} "
             f"leaves={leaves} model_leaves={len(model_paths)} ckpt_has_motion={int(ckpt_has_motion)}")
    if err:
        line4 += f"\n  _assert_param_tree_exact raise: {err[:300]}"

    del params

    # 判定 5（观察）：原始 dtype 画像
    raw = _model.restore_params(ckpt / "params", dtype=None)
    flat = jax.tree_util.tree_flatten_with_path(raw)[0]
    by_dtype: dict[str, int] = {}
    img_dtypes: set[str] = set()
    train_dtypes: set[str] = set()
    try:
        frozen_paths = _paths_of(nnx.state(model, tc.freeze_filter).to_pure_dict())
    except Exception:                  # freeze_filter 解析失败不阻断（本条为观察项）
        frozen_paths = None
    for kp, leaf in flat:
        path = "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in kp)
        dt = str(jnp.asarray(leaf).dtype) if not hasattr(leaf, "dtype") else str(leaf.dtype)
        by_dtype[dt] = by_dtype.get(dt, 0) + 1
        if "img" in path.split("/"):
            img_dtypes.add(dt)
        if frozen_paths is not None and path not in frozen_paths:
            train_dtypes.add(dt)
    f32 = by_dtype.get("float32", 0)
    bf16 = by_dtype.get("bfloat16", 0)
    img_all_bf16 = int(bool(img_dtypes) and img_dtypes == {"bfloat16"})
    trainable_all_f32 = (int(bool(train_dtypes) and train_dtypes == {"float32"})
                         if frozen_paths is not None else "NA")
    rec["ckpt_dtype_profile"] = {"by_dtype": by_dtype, "img_dtypes": sorted(img_dtypes),
                                 "trainable_dtypes": sorted(train_dtypes),
                                 "n_frozen_paths": (len(frozen_paths) if frozen_paths is not None else None)}
    line5 = (f"CKPT_DTYPE_PROFILE f32_leaves={f32} bf16_leaves={bf16} img_all_bf16={img_all_bf16} "
             f"trainable_all_f32={trainable_all_f32} by_dtype={by_dtype} img_dtypes={sorted(img_dtypes)}")
    del raw
    return line4, ok4, line5


def main() -> int:
    ap = argparse.ArgumentParser(description="TIC L0：配置与库同源核对（CPU）")
    ap.add_argument("--ckpt", default=DEF_CKPT, help="checkpoint 步目录（默认生产口径 39999）")
    ap.add_argument("--lib", default=DEF_LIB, help="对拍所用数据库根")
    ap.add_argument("--train-config", default=DEF_TRAIN_CONFIG, help="训练配置名")
    ap.add_argument("--provenance", default="", help="训练 run 的 motion_provenance.json（默认 <run>/motion_provenance.json）")
    ap.add_argument("--neg-lib", default=NEG_LIB, help="负向臂用的另一份库（清单必须与 --lib 不同）")
    ap.add_argument("--out", default="", help="判定明细 JSON 落点（可选）")
    args = ap.parse_args()

    # CPU 硬闸：本关不碰 GPU，必须在 import jax 之前设好
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OPENPI_DATA_HOME", str(_REPO_ROOT / "v1-store" / "models"))
    if not args.provenance:
        args.provenance = str(_abs(args.ckpt).parent / "motion_provenance.json")

    rec: dict = {"schema": "tic-l0-v1", "args": vars(args)}
    results: list[bool] = []
    lines: list[str] = []

    def run(name, fn):
        try:
            line, ok = fn(args, rec)
        except Exception:
            traceback.print_exc()
            line, ok = f"{name}=FAIL（异常，见上方 traceback）", False
        print(line, flush=True)
        lines.append(line)
        results.append(ok)

    run("NORM_STATS_SAME", gate_norm_stats)
    run("LIB_PROVENANCE_MATCH", gate_lib_provenance)
    run("MOTION_STORE_PATH", gate_motion_store_path)
    try:
        line4, ok4, line5 = gate_ckpt_param_tree(args, rec)
    except Exception:
        traceback.print_exc()
        line4, ok4, line5 = "CKPT_PARAM_TREE=FAIL（异常，见上方 traceback）", False, "CKPT_DTYPE_PROFILE 未采集"
    print(line4, flush=True)
    print(line5, flush=True)
    lines += [line4, line5]
    results.append(ok4)

    ok = all(results)
    rec["lines"] = lines
    rec["pass"] = ok
    if args.out:
        out = _abs(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"明细写入 {out}")
    print(f"TIC_L0={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
