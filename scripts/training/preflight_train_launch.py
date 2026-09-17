#!/usr/bin/env python
"""正式训练起跑前的统一自检（只读、零副作用）。

背景（2026-09-15 用户拍板的隔离方案）：长训练期间**训练留在主副本**
`/scratch/hongze/robomme_policy_learning_MotionJEPA`，起跑后对 src / scripts / packages
执行 `chmod -R a-w` 锁死只读；一切开发工作转到开发副本
`/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`（自带 .venv，v1-store 是指向
主副本的 symlink）。本脚本把「训练确实从主副本、用主副本的环境、拿对数据起跑」全部
变成起跑前的硬断言，非零退出即中止（由 runner 的 body() 内 set -e 兜底）。

最要紧的一类误操作是**在开发副本里误起正式训练**——两边源码高度相似，跑起来不会有任何
症状，但训练读的是随时在改的开发代码。CHECK_CWD / CHECK_PKG_* / CHECK_SYS_PREFIX /
CHECK_V1_STORE_REAL / CHECK_NOT_DEV_COPY 五条从不同角度堵这一种。

约束：只用 stdlib。importlib.util.find_spec 只查顶层包名——顶层查询不执行被查包的
__init__.py（只走 sys.path finder 定位），只有查 "a.b" 这种带点的子模块才会先 import 父包 a。
本脚本因此绝不会触发 JAX 初始化，也不 mkdir、不写任何文件。

用法（trailing argv 必须与随后交给 train.py 的参数数组逐字相同，用同一个 bash 数组）：
  python scripts/training/preflight_train_launch.py \
      --repo "$MAIN" --train-head "$TRAIN_HEAD" \
      --history-config perceptual-framesamp-modul-8frame-8x8.yaml \
      --history-config-sha256 <起跑 commit 上实测> \
      --assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
      --asset-id robomme --norm-stats-sha256 <sha> \
      --dataset-path "$DS/framesamp-8x8" --run-root "$V1_STORE/train-runs/<cfg>/<RUN>" \
      -- "${TRAIN_ARGS[@]}"
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

_RESULTS: list[tuple[str, bool]] = []


def check(name: str, item: str, ok, expect, actual) -> bool:
    """打印一行判定并记账。item 形如 '第1件cd'——FAIL 时一眼看出是四件事里的哪一件。"""
    ok = bool(ok)
    _RESULTS.append((name, ok))
    print(f"CHECK_{name}={'PASS' if ok else 'FAIL'} 件={item} 期望={expect} 实际={actual}",
          flush=True)
    return ok


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def under(root: pathlib.Path, p: pathlib.Path) -> bool:
    return p == root or root in p.parents


def git(wt: str, *args: str) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", wt, *args], capture_output=True, text=True, check=False)
    return r.returncode, (r.stdout + r.stderr).strip()


def cli_value(argv: list[str], flag: str):
    """从 trailing argv 里取 --flag VALUE 或 --flag=VALUE 的值；缺失返回 None。"""
    for i, a in enumerate(argv):
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) else ""
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return None


def check_motion_store(a, dataset_path: pathlib.Path, v1: pathlib.Path) -> None:
    """motion 期望值来自 runner；路径同时与帧库的同级目录交叉核对。"""
    root = pathlib.Path(a.motion_store)
    expected_root = dataset_path.resolve().parent / "motion"
    path_ok = (root.is_absolute() and root.is_dir() and not root.is_symlink()
               and root.resolve() == expected_root and under(v1.resolve(), root.resolve()))
    check("MOTION_STORE_PATH", "motion 数据", path_ok, expected_root, root)
    meta_path = root / "meta/store_meta.json"
    meta = {}
    try:
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict):
            meta = {}
    except (OSError, ValueError):
        pass
    got_sha = sha256_file(meta_path) if meta_path.is_file() else "<缺文件>"
    check("MOTION_STORE_META_SHA256", "motion 数据", got_sha == a.motion_store_meta_sha256,
          a.motion_store_meta_sha256, got_sha)
    check("MOTION_LAYOUT", "motion 数据", meta.get("layout") == a.motion_layout,
          a.motion_layout, meta.get("layout"))
    check("MOTION_ROWS", "motion 数据", type(meta.get("num_rows")) is int and meta["num_rows"] == a.motion_rows,
          a.motion_rows, meta.get("num_rows"))
    check("MOTION_ENV_UNSET", "motion 数据", "MMEVLA_MOTION_STORE" not in os.environ,
          "未设置", os.environ.get("MMEVLA_MOTION_STORE", "<未设置>"))


def main() -> int:
    _RESULTS.clear()
    raw = sys.argv[1:]
    train_argv: list[str] = []
    if "--" in raw:
        i = raw.index("--")
        raw, train_argv = raw[:i], raw[i + 1:]

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--repo", required=True, help="主副本仓库根；训练必须从这里起跑")
    ap.add_argument("--bench-copy", action="store_true", help="显式允许隔离的基准 worktree")
    ap.add_argument("--v1-store-realpath", help="基准副本共享的主副本实体存储目录")
    ap.add_argument("--train-head", required=True, help="40 位 sha 字面量，不得写 $(git rev-parse HEAD)")
    ap.add_argument("--history-config", required=True)
    ap.add_argument("--history-config-sha256", required=True)
    ap.add_argument("--assets-dir", required=True, help="norm_stats 所在目录的绝对路径")
    ap.add_argument("--asset-id", default="robomme")
    ap.add_argument("--norm-stats-sha256", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--run-root", default=None, help="checkpoint run 根；存在即 FAIL")
    ap.add_argument("--motion-store")
    ap.add_argument("--motion-store-meta-sha256")
    ap.add_argument("--motion-layout")
    ap.add_argument("--motion-rows", type=int)
    ap.add_argument("--train-script", default="scripts/training/train.py",
                    help="按 cwd 相对解析，与 runner 里交给 python 的那条路径逐字相同")
    a = ap.parse_args(raw)
    motion_args = (a.motion_store, a.motion_store_meta_sha256, a.motion_layout, a.motion_rows)
    if any(x is not None for x in motion_args) and not all(x is not None for x in motion_args):
        ap.error("四个 --motion-* 期望值参数必须一起提供")
    if a.bench_copy != (a.v1_store_realpath is not None):
        ap.error("--bench-copy 与 --v1-store-realpath 必须同时提供")

    wt = pathlib.Path(a.repo).resolve()
    venv = wt / ".venv"

    # ── 起跑位置：cwd 必须是主副本 ──────────────────────────────────────────
    cwd = pathlib.Path.cwd().resolve()
    check("CWD", "起跑位置", cwd == wt, wt, cwd)

    # train.py 按 cwd 相对解析：cwd 错时它会落到主树，是第1件的第二道独立证据
    tp = pathlib.Path(a.train_script)
    tp_abs = tp.resolve()
    check("TRAIN_PY", "起跑位置",
          tp.is_file() and under(wt, tp_abs), f"存在且在 {wt} 下", f"{tp_abs} is_file={tp.is_file()}")

    # YAML 同样按 cwd 相对解析（models/config/utils.py::get_history_config 的 os.path.join）
    yml = pathlib.Path("src/mme_vla_suite/models/config/robomme") / a.history_config
    yml_abs = yml.resolve()
    check("YAML_PATH", "起跑位置", yml.is_file() and under(wt, yml_abs),
          f"存在且在 {wt} 下", f"{yml_abs} is_file={yml.is_file()}")
    yml_sha = sha256_file(yml_abs) if yml.is_file() else "<缺文件>"
    check("YAML_SHA256", "起跑位置", yml_sha == a.history_config_sha256,
          a.history_config_sha256, yml_sha)

    # ── 起跑位置：三个包的 origin 必须都在主副本下 ──────────────────────────
    # 新方案靠主副本 .venv 的 editable 安装加载，不需要 PYTHONPATH；若设了，每一项都必须在主副本下
    # （指向 -temp 的 PYTHONPATH 会让训练读开发副本的代码，且完全无症状）
    got_pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    bad_pp = [p for p in got_pp if not under(wt, pathlib.Path(p).resolve())]
    check("PYTHONPATH", "起跑位置", not bad_pp,
          f"未设置，或每一项都在 {wt} 下", os.environ.get("PYTHONPATH", "<未设置>"))

    for mod in ("mme_vla_suite", "openpi", "openpi_client"):
        # 只查顶层名：find_spec 对顶层不执行该包的 __init__.py（带点的子模块才会 import 父包）
        try:
            spec = importlib.util.find_spec(mod)
        except Exception as e:  # noqa: BLE001
            spec = None
            origin = f"<find_spec 异常 {type(e).__name__}: {e}>"
        else:
            origin = spec.origin if spec is not None else "<未找到>"
        ok = spec is not None and spec.origin is not None and \
            under(wt, pathlib.Path(spec.origin).resolve())
        check(f"PKG_{mod.upper()}", "起跑位置", ok, f"origin 在 {wt} 下", origin)

    # ── 数据参数：norm_stats 绝对路径（起跑前就查，不等 dataloader 抛 TypeError）──
    assets_dir = pathlib.Path(a.assets_dir)
    check("ASSETS_DIR_ABS", "数据参数", assets_dir.is_absolute(), "绝对路径", str(assets_dir))
    ns = assets_dir / a.asset_id / "norm_stats.json"
    check("NORM_STATS_FILE", "数据参数", ns.is_file(), f"存在 {ns}",
          "存在" if ns.is_file() else "不存在")
    ns_sha = sha256_file(ns) if ns.is_file() else "<缺文件>"
    check("NORM_STATS_SHA256", "数据参数", ns_sha == a.norm_stats_sha256,
          a.norm_stats_sha256, ns_sha)

    # CLI 侧：train.py 真的收到了同一个绝对路径（config.py 取值 `assets_dir or assets_dirs` 短路，
    # 不显式覆盖就永远走条目 → cd $WT 后落空 → _load_norm_stats 只 logging.info 后返回 None）
    cli_ad = cli_value(train_argv, "--data.assets.assets-dir")
    check("CLI_ASSETS_DIR", "数据参数", cli_ad == str(assets_dir),
          str(assets_dir), cli_ad if cli_ad is not None else "<未出现在 train.py argv>")
    cli_id = cli_value(train_argv, "--data.assets.asset-id")
    check("CLI_ASSET_ID", "数据参数", cli_id == a.asset_id,
          a.asset_id, cli_id if cli_id is not None else "<未出现在 train.py argv>")
    cli_hc = cli_value(train_argv, "--model.history-config")
    check("CLI_HISTORY_CONFIG", "起跑位置", cli_hc == a.history_config,
          a.history_config, cli_hc if cli_hc is not None else "<未出现在 train.py argv>")
    cli_dp = cli_value(train_argv, "--dataset-path")
    check("CLI_DATASET_PATH", "数据", cli_dp == str(pathlib.Path(a.dataset_path)),
          str(pathlib.Path(a.dataset_path)),
          cli_dp if cli_dp is not None else "<未出现在 train.py argv>")

    # ── 起跑位置：解释器确实是主副本 .venv ──────────────────────────────────
    prefix = pathlib.Path(sys.prefix).resolve()
    check("SYS_PREFIX", "起跑位置", prefix == venv, venv, prefix)
    # 主副本身份两条：开发副本的 v1-store 是 symlink、目录名带 -temp，两者都能把「在开发副本
    # 里误起正式训练」当场挡下（AGENTS 第 14 条环境 B 段的例外条款）
    v1 = wt / "v1-store"
    if a.bench_copy:
        target = pathlib.Path(a.v1_store_realpath)
        check("V1_STORE_LINK_TARGET", "起跑位置",
              target.is_absolute() and target.is_dir() and not target.is_symlink()
              and target == target.resolve() and v1.is_symlink() and v1.resolve() == target,
              f"{v1} 链向实体目录 {target}", str(v1.resolve()))
    else:
        check("V1_STORE_REAL", "起跑位置", v1.is_dir() and not v1.is_symlink(),
              f"{v1} 是实体目录（开发副本那份是 symlink）",
              "symlink" if v1.is_symlink() else ("实体目录" if v1.is_dir() else "不存在"))
    if a.bench_copy:
        rc, worktrees = git(str(target.parent), "worktree", "list", "--porcelain")
        registered = {line.removeprefix("worktree ") for line in worktrees.splitlines()
                      if line.startswith("worktree ")}
        check("BENCH_COPY_IS_WORKTREE", "起跑位置", rc == 0 and str(wt) in registered,
              f"{wt} 注册在 {target.parent} 的 worktree 清单", sorted(registered))
    else:
        check("NOT_DEV_COPY", "起跑位置", not wt.name.endswith("-temp"),
              "仓库根不是 -temp 开发副本", wt.name)

    # 关键第三方包必须来自该 venv（空 venv 里 jax 根本不存在，这条把它挡在起跑前）
    try:
        jspec = importlib.util.find_spec("jax")
    except Exception as e:  # noqa: BLE001
        jspec = None
        jorigin = f"<find_spec 异常 {type(e).__name__}: {e}>"
    else:
        jorigin = jspec.origin if jspec is not None else "<未找到>"
    check("PKG_JAX_IN_VENV", "起跑位置",
          jspec is not None and jspec.origin is not None and
          under(venv, pathlib.Path(jspec.origin).resolve()),
          f"origin 在 {venv} 下", jorigin)

    # ── 仓库状态与数据 ──────────────────────────────────────────────────────
    rc, head = git(str(wt), "rev-parse", "HEAD")
    check("REPO_HEAD", "仓库状态", rc == 0 and head == a.train_head, a.train_head, head or f"<git rc={rc}>")
    rc, porc = git(str(wt), "status", "--porcelain")
    check("REPO_CLEAN", "仓库状态", rc == 0 and porc == "", "git status --porcelain 为空",
          porc.replace("\n", " | ") if porc else "空")

    ds = pathlib.Path(a.dataset_path)
    check("DATASET_STORE_META", "数据", (ds / "meta" / "store_meta.json").is_file(),
          f"存在 {ds}/meta/store_meta.json",
          "存在" if (ds / "meta" / "store_meta.json").is_file() else "不存在")
    src_root = os.environ.get("MMEVLA_FRAMESAMP_SOURCE", "")
    check("FRAMESAMP_SOURCE", "数据", bool(src_root) and pathlib.Path(src_root).is_dir(),
          "MMEVLA_FRAMESAMP_SOURCE 指向已存在目录", src_root or "<未设置>")
    mani = os.environ.get("MMEVLA_FRAMESAMP_MANIFEST", "")
    check("FRAMESAMP_MANIFEST", "数据", bool(mani) and pathlib.Path(mani).is_file(),
          "MMEVLA_FRAMESAMP_MANIFEST 指向已存在文件", mani or "<未设置>")

    if a.run_root:
        rr = pathlib.Path(a.run_root)
        check("RUN_ROOT_ABSENT", "留档", not rr.exists(), f"{rr} 不存在",
              "存在" if rr.exists() else "不存在")

    if a.motion_store is not None:
        check_motion_store(a, ds, v1)

    failed = [n for n, ok in _RESULTS if not ok]
    if failed:
        print(f"PREFLIGHT=FAIL failed={','.join(failed)}", flush=True)
        return 1
    print(f"PREFLIGHT=PASS n={len(_RESULTS)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
