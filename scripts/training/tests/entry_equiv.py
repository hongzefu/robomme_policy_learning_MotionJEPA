#!/usr/bin/env python3
"""C4 上游 main 对拍 harness（v5.0，0830-train-entry-restructure-plan.md 第十节）。

两个子命令：

**run**：在目标侧环境里启动训练入口并装三处只读补丁 + 一道来源断言。
    uv run scripts/training/tests/entry_equiv.py run \\
      --entry <入口py绝对路径> --record-dir <dir> --expect-root <目录> \\
      --expect-steps 1000 -- <入口 argv 尾巴...>

  1. wandb 代理：`sys.modules["wandb"]` 预载，显式 `log` 逐步记五标量 `float.hex()`
     落 `metrics_all.jsonl` 后转发**调用时查找**的真 `wandb.log`，其余属性 `__getattr__`
     委托真模块（规避「`wandb.init()` 重赋值模块级 `log` 盖掉补丁」的 2026-08-24 坑）。
  2. `save_state` 摘要器：`openpi.training.checkpoints.save_state` 模块属性替换
     （train.py 的 `_checkpoints.save_state(...)` 属调用时属性查找，补丁必中），
     **不落真权重**。双口径：strict = sha256(key+dtype+shape+bytes)（A/B 互比；裸字节
     口径下 dtype 不同、shape 转置、树中位置不同均会误通过）；g0-compat = 照抄
     `g0/bench_train_steps.py::_leaf_sha256` 的 dtype+shape+bytes 口径（供与 G0b 固化
     `param_checksums.jsonl` 交叉比，口径由 `test_padding_dtype.py` 锁死不得单方面改）。
     另记 `treedef_sha` / `keyset_sha`（堵「结构变了但每个叶子字节恰好对上」）。
  3. resolved-config digest：`tx`/`ema_decay` 标了 `pytree_node=False` 不是 leaf，
     逐叶遍历遍历不到，故收尾用与入口相同的 argv 重调 `cli()`（config 是 argv 的纯
     函数）取 resolved config，逐顶层字段 repr 落 `resolved_config.json`。
  4. 来源污染防护：起跑前断言 `PYTHONPATH` 未设置 + `find_spec` 的 origin 在
     `--expect-root` 下；`initialize_checkpoint_dir` 包装里（main 已开始、任何计算
     之前）再对全部已加载项目模块断言 `__file__` 归属；收尾把模块 `__file__`+sha256
     与 cwd git HEAD 写 `provenance.json`。三道防住「A 侧 import 到 B 侧代码 →
     两侧跑同一份代码必然 PASS」这一最阴的误通过。

  harness 只 import 标准库 + wandb + numpy/jax + openpi.training.checkpoints，
  不 import 本仓库其他训练代码（resolved-config 经 sys.modules 取入口自己加载的
  config 模块，harness 不直接 import 它）。

**judge**：读两侧 record 目录，输出 4.5 的判定行（纯标准库 + 同目录 project_scalars）。
    uv run scripts/training/tests/entry_equiv.py judge \\
      --a-dir <A 侧 record> --b-dir <B 侧 record> [--expect-sha256 <锚点>] \\
      [--expect-steps N] [--expect-tentative-a 12] \\
      [--expect-state-steps 100,200,...,999] [--expect-head-a <sha> --expect-head-b <sha>]

  v5.1 硬化（0830-prod-60k-wandb-plan.md D1，堵对抗审计列出的假阳性孔洞）：
  步集合必须恰为 0..N-1、A 段 tentative 行数恰 12/N 且 B 段恰 0（进退出码）、
  状态摘要步集合恰为期望集且原始行数无重复、provenance HEAD 断言 + porcelain 必须空；
  --expect-sha256 改可选（A40 无本机锚点时打 anchor=SKIPPED、不进判据）。
  run 侧新增可重复 --forbid-root：断言项目模块不落在该目录下（B 侧传 A worktree
  路径——worktree 在仓库 v1-store/ 子目录下，仅 expect_root 归属检查拦不住）。

  0925：judge 必须提供 tentative/state/head 期望值，默认 upstream 保持异根；
  --mode same-entry 改判同根同源码，且要求 A=solo、B=concurrent 和
  --concurrent-peer-dir，step 10..99 交集覆盖双方各至少 50 个完整主机完成间隔。
  两种模式都拒绝非有限值和缺失起止证据。run 的 --jax-cache-dir 只改缓存路径，
  不改变计算配置。P1 两步只验 run 成功、分段、有限值和来源，不调用轨迹 judge。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import runpy
import subprocess
import sys
import time

_PROJECT_TOPLEVELS = ("openpi", "mme_vla_suite")
_SCALAR_KEYS = ["loss", "grad_norm", "llm_grad_norm", "mem_enc_norm", "param_norm"]
_CFG_WHITELIST = {"exp_name", "dataset_path", "checkpoint_base_dir", "overwrite"}
_SAME_ENTRY_WHITELIST = {"exp_name", "checkpoint_base_dir"}
_CFG_SPOTLIGHT = ["ema_decay", "optimizer", "lr_schedule"]  # 单独列出、不埋进总比对
_FINGERPRINT_VERSION = 1


def _file_record(path: pathlib.Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _runtime_fingerprint(config, root: pathlib.Path) -> dict:
    """绑定实际输入来源；大权重完整校验仍由启动前资产锁负责。"""
    import jax

    dataset = pathlib.Path(config.dataset_path).resolve(strict=True)
    store_path = dataset / "meta/store_meta.json"
    store = json.loads(store_path.read_text()) if store_path.is_file() else {}
    source = pathlib.Path(os.environ.get("MMEVLA_FRAMESAMP_SOURCE")
                          or store.get("source_dataset_root") or dataset).resolve(strict=True)
    manifest = pathlib.Path(os.environ.get("MMEVLA_FRAMESAMP_MANIFEST")
                            or store.get("manifest_path")
                            or source.parent / "meta/episode_manifest.json")
    source_records = {"episode_manifest": _file_record(manifest),
                      "stats": _file_record(source / "meta/stats.json")}
    for name in ("input_manifest.json",):
        path = source.parent / "meta" / name
        if path.is_file():
            source_records[name] = _file_record(path)
    asset_id = config.data.assets.asset_id or config.data.repo_id
    norm = pathlib.Path(config.data.assets.assets_dir or config.assets_dirs) / asset_id / "norm_stats.json"
    params = pathlib.Path(config.weight_loader.params_path).resolve(strict=True)
    initialization = {name: _file_record(params / name)
                      for name in ("_METADATA", "manifest.ocdbt", "commit_success.txt")
                      if (params / name).is_file()}
    if not initialization:
        raise ValueError(f"初始化权重缺少原生恢复索引: {params}")
    tokenizer_root = pathlib.Path(os.environ["OPENPI_DATA_HOME"]).resolve(strict=True)
    history = root / "src/mme_vla_suite/models/config/robomme" / config.model.history_config
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "").lower().replace("_", "-")
        if name:
            if name in packages and packages[name] != dist.version:
                raise ValueError(f"依赖存在多版本: {name}")
            packages[name] = dist.version
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    devices = [dict(zip(("index", "uuid", "name", "driver"),
                        (part.strip() for part in line.split(",")), strict=True))
               for line in smi.stdout.splitlines() if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    selected = [next(d for d in devices if token in (d["index"], d["uuid"]))
                for token in visible if token]
    if not selected or len({d["uuid"] for d in selected}) != len(selected):
        raise ValueError("CUDA_VISIBLE_DEVICES 必须明确指定互异的实际 GPU")
    env_names = {"XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "CUDA_VISIBLE_DEVICES",
                 "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "JAX_PLATFORMS",
                 "JAX_PLATFORM_NAME", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NVIDIA_TF32_OVERRIDE", "PYTHONHASHSEED"}
    return {"version": _FINGERPRINT_VERSION,
            "python": {"version": sys.version, "implementation": platform.python_implementation()},
            "dependencies": packages, "uv_lock_sha256": _file_record(root / "uv.lock")["sha256"],
            "devices": selected, "environment": {key: os.environ.get(key) for key in sorted(env_names)},
            "jax": {"enable_x64": bool(jax.config.jax_enable_x64),
                    "matmul_precision": str(jax.config.jax_default_matmul_precision)},
            "history": _file_record(history),
            "assets": {"norm_stats": _file_record(norm), "initialization": initialization,
                       "tokenizer": _file_record(tokenizer_root / "big_vision/paligemma_tokenizer.model")},
            "source": {"path": str(source), "records": source_records},
            "dataset": {"path": str(dataset), "store_meta": _file_record(store_path) if store else None}}


def _install_jax_cache_redirect(config, target: pathlib.Path):
    """只替换上游写死的缓存路径；返回恢复函数和实际转接调用记录。"""
    original = config.update
    calls = []

    def update(name, value, *args, **kwargs):
        if name == "jax_compilation_cache_dir":
            calls.append({"requested": str(value), "actual": str(target)})
            value = str(target)
        return original(name, value, *args, **kwargs)

    config.update = update
    return original, calls


class _CheckpointOwnership:
    """首轮拒绝复用输出，只允许上游正式段重建本次 tentative 自己创建的目录。"""

    def __init__(self, record_dir: pathlib.Path, *, upstream: bool):
        self.record_path = record_dir / "init_ownership.json"
        self.upstream = upstream
        self.events = []
        self.path = None
        self.inode = None

    def initialize(self, original, config, *args, **kwargs):
        # config.checkpoint_dir 已 resolve，必须从未解析的三个配置字段还原后检查链接。
        path = pathlib.Path(config.checkpoint_base_dir) / config.name / config.exp_name
        if not path.is_absolute() or path != path.resolve():
            raise ValueError(f"checkpoint 原始路径不是绝对实体路径或含符号链接: {path}")
        supplied = args[0] if args else kwargs.get("checkpoint_dir")
        if supplied is None or pathlib.Path(supplied).resolve() != path:
            raise ValueError("checkpoint 初始化路径与实际配置不同")
        overwrite = kwargs.get("overwrite", False)
        if kwargs.get("resume", False) or (overwrite and not self.upstream):
            raise ValueError("对拍禁止 resume；B 侧禁止 overwrite")
        limit = 2 if self.upstream else 1
        if len(self.events) >= limit:
            raise ValueError("checkpoint 初始化次数超过预期")
        if not self.events:
            if os.path.lexists(path):
                raise FileExistsError(f"拒绝清空或复用已有 checkpoint 输出: {path}")
        else:
            if path != self.path or not overwrite:
                raise ValueError("上游第二次初始化须 overwrite 本次 tentative 的同一路径")
            self._check_owned()
        effective = dict(kwargs)
        # 即使检查后出现并发创建，首轮底层也不得以 overwrite=True 清理它。
        if not self.events:
            effective["overwrite"] = False
        result = original(*args, **effective)
        if not path.is_dir() or path.is_symlink() or path != path.resolve():
            raise ValueError("初始化没有产生预期的实体 checkpoint 目录")
        stat = path.stat()
        self.path, self.inode = path, (stat.st_dev, stat.st_ino)
        self.events.append({"call": len(self.events) + 1, "path": str(path),
                            "requested_overwrite": bool(overwrite), "effective_overwrite": effective["overwrite"],
                            "device": stat.st_dev, "inode": stat.st_ino})
        self.record_path.write_text(json.dumps({"upstream": self.upstream, "events": self.events}, indent=2))
        return result

    def _check_owned(self):
        if self.path is None or not self.path.is_dir() or self.path.is_symlink() or self.path != self.path.resolve():
            raise ValueError("本次 checkpoint 目录已消失或被链接替换")
        stat = self.path.stat()
        if (stat.st_dev, stat.st_ino) != self.inode:
            raise ValueError("本次 checkpoint 目录归属已改变，禁止覆盖")

    def check_complete(self):
        if len(self.events) != (2 if self.upstream else 1):
            raise ValueError("checkpoint 初始化次数与官方入口段数不符")
        self._check_owned()


def _finite_metrics(rows: list[dict]) -> bool:
    if not rows:
        return False
    for row in rows:
        for key in _SCALAR_KEYS:
            value = row.get(key, {})
            try:
                number = float.fromhex(value["hex"])
                if (value.get("finite") is not True or not math.isfinite(number)
                        or float(value["dec"]).hex() != number.hex()):
                    return False
            except (KeyError, TypeError, ValueError, OverflowError):
                return False
    return True


def _finite_states(rows: list[dict]) -> bool:
    return bool(rows) and all(
        row.get("finite") is True and type(row.get("n_leaves")) is int
        and row["n_leaves"] > 0 and row.get("finite_leaves") == row["n_leaves"]
        and row.get("nonfinite_keys") == [] for row in rows)


# ────────────────────────── run 侧 ──────────────────────────

class _WandbRecorderProxy:
    def __init__(self, real, path: pathlib.Path):
        self._real = real
        self._path = path
        self.rows = 0

    def log(self, data, step=None, **kwargs):
        row: dict = {"step": int(step) if step is not None else None,
                     "wall_time": time.time()}
        n_scalar = 0
        for k, v in data.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            row[k] = {"dec": fv if math.isfinite(fv) else None,
                      "hex": fv.hex(), "finite": math.isfinite(fv)}
            n_scalar += 1
        if n_scalar:
            with self._path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.rows += 1
        # 属性在调用时才查找真 wandb.log（wandb.init() 重赋值也拦不住转发）
        return self._real.log(data, step=step, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _assert_provenance(expect_root: pathlib.Path,
                       forbid_roots: list[pathlib.Path]) -> None:
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] not in _PROJECT_TOPLEVELS:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = pathlib.Path(f).resolve()
        if not p.is_relative_to(expect_root):
            raise SystemExit(f"ENTRY_PROVENANCE 违例: 模块 {name} 来自 {p}，"
                             f"不在 --expect-root {expect_root} 下（来源污染，立即停）")
        # v5.1：forbid-root 补拦「禁根是 expect_root 子目录」的情形——B 侧 expect_root
        # 是仓库根、A worktree 在 v1-store/ 下，仅 is_relative_to(expect_root) 拦不住
        for fr in forbid_roots:
            if p.is_relative_to(fr):
                raise SystemExit(f"ENTRY_PROVENANCE 违例: 模块 {name} 来自 {p}，"
                                 f"落在 --forbid-root {fr} 下（来源污染，立即停）")


def _module_provenance(expect_root: pathlib.Path, entry: pathlib.Path) -> dict:
    mods = {}
    for name, mod in sorted(sys.modules.items()):
        if name.split(".")[0] not in _PROJECT_TOPLEVELS:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = pathlib.Path(f).resolve()
        mods[name] = {"file": str(p),
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
                          capture_output=True, text=True, check=True).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=os.getcwd(),
                               capture_output=True, text=True, check=True).stdout
    git_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=os.getcwd(),
                              capture_output=True, text=True, check=True).stdout.strip()
    return {"expect_root": str(expect_root),
            "entry": {"file": str(entry),
                      "sha256": hashlib.sha256(entry.read_bytes()).hexdigest()},
            "cwd": os.getcwd(), "git_head_of_cwd": head,
            "git_porcelain_of_cwd": porcelain, "git_root": git_root, "modules": mods}


def _assert_clean_snapshot(snapshot: dict, expected_head: str | None) -> None:
    if snapshot["git_porcelain_of_cwd"].strip():
        raise ValueError("对拍必须从 clean HEAD 启动且运行期间保持 clean")
    if pathlib.Path(snapshot["git_root"]).resolve() != pathlib.Path(snapshot["expect_root"]).resolve():
        raise ValueError("cwd 的 Git 根与 expect_root 不同")
    if expected_head is not None and snapshot["git_head_of_cwd"] != expected_head:
        raise ValueError("实际 HEAD 与期望完整提交不同")


def _split_segments(all_path: pathlib.Path, record_dir: pathlib.Path) -> tuple[int, int]:
    """按 step 回绕切分 tentative / 正式两段；>2 段即 fail-loud。"""
    segments: list[list[str]] = [[]]
    prev = None
    with all_path.open() as f:
        for line in f:
            step = json.loads(line).get("step")
            if prev is not None and step is not None and step < prev:
                segments.append([])
            if step is not None:
                prev = step
            segments[-1].append(line)
    if len(segments) > 2:
        raise SystemExit(f"metrics 段数 {len(segments)} > 2（step 回绕多于一次），无法切分")
    main_seg = segments[-1]
    tent_seg = segments[0] if len(segments) == 2 else []
    (record_dir / "metrics.jsonl").write_text("".join(main_seg))
    if tent_seg:
        (record_dir / "metrics_tentative.jsonl").write_text("".join(tent_seg))
    return len(tent_seg), len(main_seg)


def _dump_resolved_config(record_dir: pathlib.Path) -> None:
    cfgmod = sys.modules.get("mme_vla_suite.training.config")
    if cfgmod is None:
        (record_dir / "resolved_config.json").write_text(
            json.dumps({"error": "config 模块未加载（入口未跑到 import）"}))
        return
    cfg = cfgmod.cli()   # config 是 argv 的纯函数；sys.argv 仍为入口 argv
    fields = {f.name: repr(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    body = {k: v for k, v in fields.items() if k not in _CFG_WHITELIST}
    sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    (record_dir / "resolved_config.json").write_text(json.dumps(
        {"fields": fields, "whitelist": sorted(_CFG_WHITELIST),
         "resolved_cfg_sha_ex_whitelist": sha}, indent=2, ensure_ascii=False))


def _run(args, tail: list[str]) -> int:
    entry = pathlib.Path(args.entry).resolve()
    record_dir = pathlib.Path(args.record_dir).resolve()
    expect_root = pathlib.Path(args.expect_root).resolve()
    forbid_roots = [pathlib.Path(p).resolve() for p in (args.forbid_root or [])]
    if pathlib.Path(sys.prefix).resolve() != (expect_root / ".venv").resolve():
        raise ValueError("必须使用目标仓库自身的独立 uv .venv")
    cache_target = None
    if args.jax_cache_dir:
        candidate = pathlib.Path(args.jax_cache_dir)
        storage = pathlib.Path(__file__).resolve().parents[3] / "v1-store"
        if not candidate.is_absolute() or not candidate.resolve().is_relative_to(storage.resolve()):
            raise ValueError("JAX 缓存必须是本仓库 v1-store 下的绝对路径")
        cache_target = candidate.resolve()

    if "PYTHONPATH" in os.environ or "PYTHONHOME" in os.environ:
        raise SystemExit("来源污染防护: 必须以 env -u PYTHONPATH -u PYTHONHOME 启动")
    if not entry.is_relative_to(expect_root):
        raise SystemExit(f"入口 {entry} 不在 --expect-root {expect_root} 下")
    for fr in forbid_roots:
        if entry.is_relative_to(fr):
            raise SystemExit(f"入口 {entry} 落在 --forbid-root {fr} 下")
    import importlib.util
    for top in _PROJECT_TOPLEVELS:
        spec = importlib.util.find_spec(top)
        if spec is None or not spec.origin:
            raise SystemExit(f"find_spec({top!r}) 失败，环境不完整")
        origin = pathlib.Path(spec.origin).resolve()
        if not origin.is_relative_to(expect_root):
            raise SystemExit(f"来源污染: {top} 解析到 {spec.origin}，"
                             f"不在 {expect_root} 下")
        for fr in forbid_roots:
            if origin.is_relative_to(fr):
                raise SystemExit(f"来源污染: {top} 解析到 {spec.origin}，"
                                 f"落在 --forbid-root {fr} 下")

    if record_dir.exists() and any(record_dir.iterdir()):
        raise FileExistsError(f"记录目录非空，拒绝混用旧记录: {record_dir}")
    snapshot_start = _module_provenance(expect_root, entry)
    _assert_clean_snapshot(snapshot_start, args.expect_head)
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "provenance_start.json").write_text(json.dumps(snapshot_start, indent=2))
    all_path = record_dir / "metrics_all.jsonl"
    if all_path.exists():
        raise FileExistsError(f"记录已存在，拒绝覆盖: {all_path}")

    # 补丁 1：wandb 代理（必须在入口 import wandb 之前进 sys.modules）
    import wandb as _real_wandb
    proxy = _WandbRecorderProxy(_real_wandb, all_path)
    sys.modules["wandb"] = proxy   # type: ignore[assignment]

    # 补丁 2 + 断言点：save_state 摘要器与 initialize_checkpoint_dir 包装
    import jax  # noqa: I001 —— 延迟到断言通过后才 import 项目环境
    import numpy as np
    import openpi.training.checkpoints as _ckpt
    cache_calls = []
    original_update = None
    if cache_target is not None:
        original_update, cache_calls = _install_jax_cache_redirect(jax.config, cache_target)
    state_path = record_dir / "state_digests.jsonl"
    _orig_init_dir = _ckpt.initialize_checkpoint_dir
    original_save_state = _ckpt.save_state
    fingerprint_start = None
    upstream = entry == expect_root / "scripts/train.py" and bool(args.expect_tentative)
    ownership = _CheckpointOwnership(record_dir, upstream=upstream)

    def _wrapped_init_dir(*a, **kw):
        nonlocal fingerprint_start
        _assert_provenance(expect_root, forbid_roots)   # main 已开始、任何计算之前
        cfg = sys.modules["mme_vla_suite.training.config"].cli()
        current = _runtime_fingerprint(cfg, expect_root)
        if fingerprint_start is None:
            fingerprint_start = current
            (record_dir / "runtime_fingerprint.json").write_text(json.dumps(current, indent=2))
        elif current != fingerprint_start:
            raise ValueError("tentative 与正式段的输入或环境指纹变化")
        return ownership.initialize(_orig_init_dir, cfg, *a, **kw)

    def _summarize_state(checkpoint_manager, state, data_loader, step):
        trees = {"params": state.params, "opt_state": state.opt_state,
                 "step": state.step}
        if state.ema_params is not None:
            trees["ema_params"] = state.ema_params
        per_strict: dict[str, str] = {}
        per_g0: dict[str, str] = {}
        nonfinite_keys = []
        for tree_name, tree in trees.items():
            flat, _ = jax.tree_util.tree_flatten_with_path(tree)
            for path, leaf in flat:
                if leaf is None:
                    continue
                key = tree_name + jax.tree_util.keystr(path)
                arr = np.asarray(jax.device_get(leaf))
                if not bool(np.isfinite(arr).all()):
                    nonfinite_keys.append(key)
                hs = hashlib.sha256()
                hs.update(key.encode())
                hs.update(str(arr.dtype).encode())
                hs.update(str(arr.shape).encode())
                hs.update(arr.tobytes())
                per_strict[key] = hs.hexdigest()
                hg = hashlib.sha256()   # 照抄 g0 _leaf_sha256 口径
                hg.update(str(arr.dtype).encode())
                hg.update(str(arr.shape).encode())
                hg.update(arr.tobytes())
                per_g0[key] = hg.hexdigest()
        gs, gg = hashlib.sha256(), hashlib.sha256()
        for key in sorted(per_strict):
            gs.update(f"{key}:{per_strict[key]}\n".encode())
            gg.update(f"{key}:{per_g0[key]}\n".encode())
        row = {"step": int(step),
               "strict_global": gs.hexdigest(),
               "g0_global": gg.hexdigest(),
               "treedef_sha": hashlib.sha256(
                   repr(jax.tree_util.tree_structure(trees)).encode()).hexdigest(),
               "keyset_sha": hashlib.sha256(
                   "\n".join(sorted(per_strict)).encode()).hexdigest(),
               "n_leaves": len(per_strict),
               "finite": not nonfinite_keys,
               "finite_leaves": len(per_strict) - len(nonfinite_keys),
               "nonfinite_keys": nonfinite_keys,
               "per_leaf_strict": per_strict, "per_leaf_g0": per_g0}
        with state_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[entry_equiv] step {step}: TrainState 摘要已记（{len(per_strict)} 叶，"
              f"不落真权重）", flush=True)

    _ckpt.save_state = _summarize_state
    _ckpt.initialize_checkpoint_dir = _wrapped_init_dir

    original_argv, original_path = sys.argv, list(sys.path)
    sys.path.insert(0, str(entry.parent))
    sys.argv = [entry.name, *tail]
    (record_dir / "harness_meta.json").write_text(json.dumps(
        {"entry": str(entry), "argv_tail": tail, "expect_root": str(expect_root),
         "forbid_roots": [str(p) for p in forbid_roots],
         "expect_steps": args.expect_steps, "execution_role": args.execution_role,
         "harness_sha256": _file_record(pathlib.Path(__file__))["sha256"],
         "pid": os.getpid()}, indent=2, ensure_ascii=False))

    rc = 0
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0)
    except BaseException as e:  # 落档后原样上报
        print(f"[entry_equiv] 入口异常: {type(e).__name__}: {e}", flush=True)
        rc = 1
    finally:
        try:
            _assert_provenance(expect_root, forbid_roots)
            snapshot_end = _module_provenance(expect_root, entry)
            (record_dir / "provenance.json").write_text(json.dumps(snapshot_end, indent=2, ensure_ascii=False))
            _assert_clean_snapshot(snapshot_end, snapshot_start["git_head_of_cwd"])
            if snapshot_end["entry"] != snapshot_start["entry"]:
                raise ValueError("入口在运行期间变化")
            for name, record in snapshot_start["modules"].items():
                if snapshot_end["modules"].get(name) != record:
                    raise ValueError(f"模块在运行期间变化: {name}")
            _dump_resolved_config(record_dir)
            cfgmod = sys.modules.get("mme_vla_suite.training.config")
            fingerprint_end = _runtime_fingerprint(cfgmod.cli(), expect_root) if cfgmod else None
            (record_dir / "runtime_fingerprint_end.json").write_text(json.dumps(fingerprint_end, indent=2))
            if fingerprint_start is None or fingerprint_end != fingerprint_start:
                raise ValueError("运行前后指纹不同或起跑取证缺失")
            if rc == 0:
                ownership.check_complete()
            if cache_target is not None:
                actual_cache = str(jax.config.jax_compilation_cache_dir)
                if not cache_calls or actual_cache != str(cache_target):
                    raise ValueError("JAX 缓存适配未生效")
                (record_dir / "jax_cache.json").write_text(json.dumps(
                    {"calls": cache_calls, "actual": actual_cache}, indent=2))
        except BaseException as e:
            print(f"[entry_equiv] 收尾取证失败: {e}", flush=True)
            rc = 1
        finally:
            if original_update is not None:
                jax.config.update = original_update
            _ckpt.save_state = original_save_state
            _ckpt.initialize_checkpoint_dir = _orig_init_dir
            sys.modules["wandb"] = _real_wandb
            sys.argv = original_argv
            sys.path[:] = original_path

    if rc == 0:
        try:
            if proxy.rows == 0:
                raise ValueError("wandb 代理零记录——补丁未生效或训练没跑")
            tent, main_rows = _split_segments(all_path, record_dir)
            print(f"SEGMENTS tentative_rows={tent} main_rows={main_rows}")
            main_metrics = _load_jsonl(record_dir / "metrics.jsonl")
            if (main_rows != args.expect_steps
                    or [row.get("step") for row in main_metrics] != list(range(args.expect_steps))):
                raise ValueError("正式段步集合不完整或重复")
            if args.expect_tentative is not None and tent != args.expect_tentative:
                raise ValueError(f"tentative 行数 {tent} != {args.expect_tentative}")
            # P1 保留两段 step=1 摘要：这里检有限值，不把重复状态步当成轨迹对拍。
            if not _finite_metrics(_load_jsonl(all_path)) or not _finite_states(_load_jsonl(state_path)):
                raise ValueError("标量或状态存在非有限值或缺少完整记录")
            print("ENTRY_FINITE=PASS")
            print("ENTRY_RUN=OK")
        except (ValueError, OSError) as e:
            print(f"[entry_equiv] 运行验收失败: {e}", flush=True)
            rc = 1
    (record_dir / "run_status.json").write_text(json.dumps({"exit_code": rc, "entry_run_ok": rc == 0}))
    return rc


# ────────────────────────── judge 侧 ──────────────────────────

def _load_metrics(path: pathlib.Path) -> dict[int, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("step") is not None:
                out[int(r["step"])] = r
    return out


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.open() if x.strip()]


def _read_run_evidence(directory: pathlib.Path, expected_head: str | None) -> tuple[dict, dict, dict]:
    start = json.loads((directory / "provenance_start.json").read_text())
    end = json.loads((directory / "provenance.json").read_text())
    meta = json.loads((directory / "harness_meta.json").read_text())
    status = json.loads((directory / "run_status.json").read_text())
    if status != {"exit_code": 0, "entry_run_ok": True}:
        raise ValueError("入口未成功退出")
    for snapshot in (start, end):
        _assert_clean_snapshot(snapshot, expected_head)
        root = pathlib.Path(snapshot["expect_root"])
        if root != pathlib.Path(meta["expect_root"]) or not root.is_absolute():
            raise ValueError("provenance 与启动根不一致")
        paths = [snapshot["entry"]["file"], *(m["file"] for m in snapshot["modules"].values())]
        for value in paths:
            path = pathlib.Path(value)
            if not path.is_relative_to(root) or any(
                    path.is_relative_to(pathlib.Path(fr)) for fr in meta["forbid_roots"]):
                raise ValueError(f"记录的来源越界: {path}")
    if (start["git_head_of_cwd"] != end["git_head_of_cwd"] or start["entry"] != end["entry"]
            or end["entry"]["file"] != meta["entry"]):
        raise ValueError("起止 HEAD 或入口发生变化")
    if not {"mme_vla_suite.training.config", "openpi.training.checkpoints"} <= set(end["modules"]):
        raise ValueError("缺少实际训练模块来源")
    if any(end["modules"].get(key) != value for key, value in start["modules"].items()):
        raise ValueError("起止模块内容发生变化")
    fingerprint = json.loads((directory / "runtime_fingerprint.json").read_text())
    fingerprint_end = json.loads((directory / "runtime_fingerprint_end.json").read_text())
    fields = {"python", "dependencies", "uv_lock_sha256", "devices", "environment",
              "jax", "history", "assets", "source", "dataset"}
    if (fingerprint.get("version") != _FINGERPRINT_VERSION
            or any(not fingerprint.get(key) for key in fields) or fingerprint != fingerprint_end):
        raise ValueError("缺少完整指纹或起止指纹变化")
    if not meta.get("harness_sha256"):
        raise ValueError("缺少量具摘要")
    return end, fingerprint, meta


def _concurrent_overlap(current: pathlib.Path, peer: pathlib.Path, expected_head: str) -> dict:
    """按 step 10..99 的主机完成间隔判定，不把日志间隔称为 GPU 计算时间。"""
    if current.resolve() == peer.resolve():
        raise ValueError("并跑 peer 不能是本 run")
    evidence = [_read_run_evidence(path, expected_head) for path in (current, peer)]
    if any(meta.get("execution_role") != "concurrent" for _, _, meta in evidence):
        raise ValueError("并跑双方必须标记 concurrent")
    if evidence[0][0]["expect_root"] != evidence[1][0]["expect_root"]:
        raise ValueError("并跑双方必须使用同仓库根")
    uuids = [{gpu["uuid"] for gpu in fingerprint["devices"]} for _, fingerprint, _ in evidence]
    if any(len(group) != 4 for group in uuids) or uuids[0] & uuids[1]:
        raise ValueError("并跑必须为互不重叠的四卡加四卡")
    intervals = []
    for directory in (current, peer):
        rows = _load_jsonl(directory / "metrics.jsonl")
        if [r.get("step") for r in rows] != list(range(100)) or not _finite_metrics(rows):
            raise ValueError("并跑必须记录完整且有限的 100 步")
        times = [r["wall_time"] for r in rows]
        if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("主机完成时间必须严格递增且有限")
        intervals.append([(times[step - 1], times[step]) for step in range(10, 100)])
    begin = max(items[0][0] for items in intervals)
    end = min(items[-1][1] for items in intervals)
    complete = [sum(a >= begin and b <= end for a, b in items) for items in intervals]
    if end <= begin or min(complete) < 50:
        raise ValueError(f"并跑重叠不足：各侧完整步骤={complete}，均须至少 50")
    return {"window_steps": [10, 99], "begin": begin, "end": end,
            "seconds": end - begin, "complete_steps": complete}


def _judge(args) -> int:
    import project_scalars  # 同目录；与 C3 收官共用同一份投影实现
    a_dir, b_dir = pathlib.Path(args.a_dir), pathlib.Path(args.b_dir)
    fails: list[str] = []

    # 两侧分段行数（v5.1：--expect-tentative-a 传入时进退出码，堵「A 忘挂 --overwrite
    # 没跑 tentative 段 / B 多出 tentative 段」的假阳性）
    a_metrics = _load_metrics(a_dir / "metrics.jsonl")
    b_metrics = _load_metrics(b_dir / "metrics.jsonl")
    seg = {}
    for side, d, m in (("A", a_dir, a_metrics), ("B", b_dir, b_metrics)):
        tent = d / "metrics_tentative.jsonl"
        tent_rows = sum(1 for _ in tent.open()) if tent.exists() else 0
        raw_rows = sum(1 for x in (d / "metrics.jsonl").open() if x.strip())
        seg[side] = (tent_rows, raw_rows)
        print(f"{side}_SIDE_SEGMENTS tentative_rows={tent_rows} main_rows={raw_rows}")
        if raw_rows != len(m):
            fails.append(f"{side} 侧 metrics.jsonl 有重复/无 step 行: "
                         f"raw={raw_rows} unique_steps={len(m)}")
    if args.expect_tentative_a is not None:
        if seg["A"][0] != args.expect_tentative_a:
            fails.append(f"A 侧 tentative 行数 {seg['A'][0]} != 期望 {args.expect_tentative_a}")
        if seg["B"][0] != 0:
            fails.append(f"B 侧 tentative 行数 {seg['B'][0]} != 0")

    # ① ENTRY_SCALARS：逐步五键 hex 互比；步集合必须恰为 0..expect_steps-1（两侧同）
    expect_set = set(range(args.expect_steps))
    common = sorted(set(a_metrics) & set(b_metrics))
    mism = 0
    for s in common:
        for k in _SCALAR_KEYS:
            va, vb = a_metrics[s].get(k), b_metrics[s].get(k)
            if va is None or vb is None or va["hex"] != vb["hex"]:
                mism += 1
                break
    for side, m in (("A", a_metrics), ("B", b_metrics)):
        if set(m) != expect_set:
            missing = sorted(expect_set - set(m))[:5]
            extra = sorted(set(m) - expect_set)[:5]
            fails.append(f"{side} 侧步集合非恰 0..{args.expect_steps - 1}: "
                         f"缺{missing} 多{extra} (n={len(m)})")
    print(f"ENTRY_SCALARS steps={len(common)} keys={len(_SCALAR_KEYS)} hex_mismatch={mism}")
    if mism or len(common) != args.expect_steps:
        fails.append(f"ENTRY_SCALARS 失配: steps={len(common)} mismatch={mism}")

    # ② ENTRY_STATE_DIGEST：strict/treedef/keyset 三域互比；--expect-state-steps 传入时
    # 步集合必须恰为该集、原始行数 == 集合大小（无重复），两侧同
    a_st_rows = _load_jsonl(a_dir / "state_digests.jsonl")
    b_st_rows = _load_jsonl(b_dir / "state_digests.jsonl")
    finite_ok = True
    for directory, states in ((a_dir, a_st_rows), (b_dir, b_st_rows)):
        rows = _load_jsonl(directory / "metrics.jsonl")
        tentative = directory / "metrics_tentative.jsonl"
        if tentative.exists():
            rows.extend(_load_jsonl(tentative))
        finite_ok = finite_ok and _finite_metrics(rows) and _finite_states(states)
    print(f"ENTRY_FINITE={'PASS' if finite_ok else 'FAIL'}")
    if not finite_ok:
        fails.append("标量或状态非有限或缺少 finite 证据")
    a_st = {r["step"]: r for r in a_st_rows}
    b_st = {r["step"]: r for r in b_st_rows}
    st_common = sorted(set(a_st) & set(b_st))
    st_mism = sum(1 for s in st_common if any(
        a_st[s][k] != b_st[s][k] for k in ("strict_global", "treedef_sha", "keyset_sha")))
    if set(a_st) != set(b_st):
        fails.append(f"状态摘要步集合不同: A={sorted(a_st)} B={sorted(b_st)}")
    if args.expect_state_steps is not None:
        want = {int(x) for x in args.expect_state_steps.split(",")}
        for side, rows, st in (("A", a_st_rows, a_st), ("B", b_st_rows, b_st)):
            if set(st) != want:
                fails.append(f"{side} 侧状态摘要步集合 {sorted(st)} != 期望 {sorted(want)}")
            if len(rows) != len(want):
                fails.append(f"{side} 侧状态摘要原始行数 {len(rows)} != {len(want)}（有重复或缺行）")
    print(f"ENTRY_STATE_DIGEST rows={len(st_common)} mismatch={st_mism}")
    if st_mism or not st_common:
        fails.append(f"ENTRY_STATE_DIGEST 失配: rows={len(st_common)} mismatch={st_mism}")

    # ③ ENTRY_RESOLVED_CFG：逐顶层字段 repr，排除白名单四项
    a_cfg = json.loads((a_dir / "resolved_config.json").read_text())
    b_cfg = json.loads((b_dir / "resolved_config.json").read_text())
    if "fields" not in a_cfg or "fields" not in b_cfg:
        fails.append("resolved_config 缺 fields（某侧入口未跑到 import config）")
        cfg_mism = -1
    else:
        fa, fb = a_cfg["fields"], b_cfg["fields"]
        whitelist = _SAME_ENTRY_WHITELIST if args.mode == "same-entry" else _CFG_WHITELIST
        keys = (set(fa) | set(fb)) - whitelist
        bad = sorted(k for k in keys if fa.get(k) != fb.get(k))
        cfg_mism = len(bad)
        for k in bad:
            print(f"CFG_DIFF field={k} a={fa.get(k)!r} b={fb.get(k)!r}")
        for k in _CFG_SPOTLIGHT:   # 单独列出对比、不埋进总哈希
            tag = "SAME" if fa.get(k) == fb.get(k) else "DIFF"
            print(f"CFG_SPOTLIGHT field={k} {tag}")
    print(f"ENTRY_RESOLVED_CFG mismatch={cfg_mism} mode={args.mode}")
    if cfg_mism != 0:
        fails.append(f"ENTRY_RESOLVED_CFG mismatch={cfg_mism}")

    # ④ 起止证据与两种比较模式。same-entry 的放行不能削弱 upstream 异根守卫。
    prov_ok = True
    evidence = []
    expect_heads = {"A": args.expect_head_a, "B": args.expect_head_b}
    for side, d in (("A", a_dir), ("B", b_dir)):
        try:
            evidence.append(_read_run_evidence(d, expect_heads[side]))
        except (ValueError, KeyError, OSError, TypeError) as e:
            print(f"PROVENANCE_BAD side={side} reason={e}")
            prov_ok = False
    try:
        if len(evidence) != 2:
            raise ValueError("双侧完整来源证据缺失")
        (pa, fpa, ma), (pb, fpb, mb) = evidence
        if ma["harness_sha256"] != mb["harness_sha256"]:
            raise ValueError("两侧量具内容不同")
        if args.mode == "same-entry":
            for key in ("expect_root", "entry", "git_head_of_cwd", "modules"):
                if pa[key] != pb[key]:
                    raise ValueError(f"same-entry 来源字段不同: {key}")
            if fpa != fpb:
                raise ValueError("same-entry 的依赖/设备/数据/资产/有效环境指纹不同")
            if ma.get("execution_role") != "solo" or mb.get("execution_role") != "concurrent":
                raise ValueError("same-entry 必须比较 A=solo 与 B=concurrent")
            if not args.concurrent_peer_dir:
                raise ValueError("same-entry 缺少 S2 另一侧 concurrent-peer-dir")
            overlap = _concurrent_overlap(b_dir, pathlib.Path(args.concurrent_peer_dir), args.expect_head_b)
            print("ENTRY_CONCURRENT=PASS " + json.dumps(overlap, sort_keys=True))
        else:
            if pa["expect_root"] == pb["expect_root"]:
                raise ValueError("upstream 两侧 expect_root 相同")
            for key in ("python", "dependencies", "uv_lock_sha256", "devices", "environment", "jax", "assets", "source"):
                if fpa[key] != fpb[key]:
                    raise ValueError(f"upstream 共同环境/输入不同: {key}")
            for key in ("sha256", "bytes"):
                if fpa["history"][key] != fpb["history"][key]:
                    raise ValueError("upstream history 内容不同")
    except (ValueError, KeyError, OSError, TypeError) as e:
        print(f"PROVENANCE_BAD reason={e}")
        prov_ok = False
    print(f"ENTRY_PROVENANCE={'PASS' if prov_ok else 'FAIL'}")
    if not prov_ok:
        fails.append("ENTRY_PROVENANCE=FAIL")

    # ⑤ 两侧各自投影对锚点；v5.1 起锚点可选（A40 无本机锚点时打 SKIPPED，不进判据）
    shas = {}
    for side, d in (("A", a_dir), ("B", b_dir)):
        try:
            text = project_scalars.project(d / "metrics.jsonl")
        except (SystemExit, ValueError, KeyError, OSError) as e:
            fails.append(f"{side} 侧标量投影失败: {e}")
            continue
        (d / "scalars_hex.tsv").write_text(text)
        shas[side] = hashlib.sha256(text.encode()).hexdigest()
        if args.expect_sha256 is None:
            print(f"{side}_SCALARS_SHA256 {shas[side]} anchor=SKIPPED")
            continue
        hit = shas[side] == args.expect_sha256
        print(f"{side}_SCALARS_SHA256 {shas[side]} anchor_hit={hit}")
        if not hit:
            fails.append(f"{side} 侧 scalars sha256 未命中锚点")

    verdict = "PASS" if not fails else "FAIL"
    for r in fails:
        print(f"ENTRY_EQ_FAIL reason={r}")
    print(f"ENTRY_EQ={verdict}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--entry", required=True)
    p_run.add_argument("--record-dir", required=True)
    p_run.add_argument("--expect-root", required=True)
    p_run.add_argument("--forbid-root", action="append", default=[],
                       help="可重复；断言项目模块 __file__ 不落在该目录下")
    p_run.add_argument("--expect-steps", type=int, default=1000)
    p_run.add_argument("--expect-tentative", type=int, default=None,
                       help="P1 可要求 A=2、B=0；不对重复 tentative 状态 step 做轨迹判定")
    p_run.add_argument("--expect-head", help="起跑与收尾必须命中的完整提交")
    p_run.add_argument("--execution-role", choices=("upstream", "solo", "concurrent"), default="upstream")
    p_run.add_argument("--jax-cache-dir", help="只转接 JAX 缓存路径，本仓库 v1-store 下的绝对目录")
    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--a-dir", required=True)
    p_judge.add_argument("--b-dir", required=True)
    p_judge.add_argument("--expect-sha256", default=None,
                        help="v5.1 起可选；缺省打 anchor=SKIPPED、不进判据")
    p_judge.add_argument("--expect-steps", type=int, default=1000,
                        help="scalar 步集合必须恰为 0..N-1（两侧同）")
    p_judge.add_argument("--expect-tentative-a", type=int, required=True,
                        help="A 段 tentative 行数必须恰为此值且 B 段恰 0，进退出码")
    p_judge.add_argument("--expect-state-steps", required=True,
                        help="逗号分隔；状态摘要步集合必须恰为此集且无重复（两侧同）")
    p_judge.add_argument("--expect-head-a", required=True,
                        help="A 侧 provenance git HEAD 断言 + porcelain 必须空")
    p_judge.add_argument("--expect-head-b", required=True,
                        help="B 侧 provenance git HEAD 断言 + porcelain 必须空")
    p_judge.add_argument("--mode", choices=("upstream", "same-entry"), default="upstream")
    p_judge.add_argument("--concurrent-peer-dir", help="same-entry 时 S2 另一库的完整记录目录")

    argv = sys.argv[1:]
    tail: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    for name in ("expect_head", "expect_head_a", "expect_head_b"):
        value = getattr(args, name, None)
        if value is not None and (len(value) != 40 or any(c not in "0123456789abcdef" for c in value)):
            ap.error(f"--{name.replace('_', '-')} 必须为完整 40 位 SHA")
    if args.expect_steps < 2:
        ap.error("对拍至少两步，单步无法可靠识别上游 tentative 回绕")
    if args.cmd == "run":
        return _run(args, tail)
    return _judge(args)


if __name__ == "__main__":
    sys.exit(main())
