#!/usr/bin/env python3
"""按 ``ASSETS_LOCK.json`` 取回并校验五个外部模型资产。异地无 NFS 机器复刻链路的唯一入口。

子命令：
  plan    不联网，列出每条资产的落点 / PRESENT-MISSING / 字节数 / 来源；末行 ``ASSETS_PLAN total=… missing=…``
  fetch   缺什么取什么（``--force`` 强制重取），取回后一律按 ``--level full`` 复校
  verify  只校不取，末行 ``ASSETS=PASS`` / ``ASSETS=FAIL``（默认 ``--level full``）
  show    打印单条 lock 记录

刻意**不** ``uv add huggingface-hub``：根 ``uv.lock`` 一动，``scripts/training/g0/check_baseline_env.py``
的 ``uv_lock_sha256`` 就变，G0b 黄金基线全部 FAIL。``huggingface_hub`` 由 ``transformers==4.53.2``
传递提供，根 lock 已把它钉在 0.32.x，下面有版本漂移断言。

也刻意**不**手写 urllib 下载器：``huggingface.co/<repo>/resolve/<rev>/<path>`` 会 302 到
``us.aws.cdn.hf.co`` 的预签名 URL，naive urllib 读到的是最后一跳的 ``etag``（xet merkle root），
而规范值是第一跳的 ``x-linked-etag``（LFS sha256），两者不等、拿它校验必然误判；且 Python 的
``HTTPRedirectHandler`` 只剔 content-length/content-type，会把 ``Authorization`` 原样发给 CDN。

用法：
  uv run python scripts/assets/fetch_assets.py plan
  uv run python scripts/assets/fetch_assets.py fetch            # 私有资产需先 export HF_TOKEN=hf_…
  uv run python scripts/assets/fetch_assets.py verify --level full
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import assets_lock as al  # noqa: E402

REPO_ROOT = al.REPO_ROOT
if not (REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {REPO_ROOT}（缺 pyproject.toml）")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return str(n)


def _entry_bytes(entry: dict) -> int:
    if entry["kind"] == "file":
        return entry["bytes"]
    if entry["kind"] == "dir":
        return entry["total_bytes"]
    return sum(v["bytes"] for v in entry["files"].values())


def _source_str(entry: dict) -> str:
    s = entry["source"]
    if s["type"].startswith("hf"):
        tail = f" {s['filename']}" if "filename" in s else f" {s.get('allow_patterns')}"
        priv = "私有，需 HF_TOKEN" if s.get("private") else "公开"
        return f"hf://{s['repo_id']}@{s['revision'][:12]}…{tail}（{priv}）"
    return f"{s['uri']}（anon）"


def _present(entry: dict, root: pathlib.Path) -> bool:
    p = root / entry["dest"]
    return p.is_dir() if entry["kind"] in ("dir", "hf_snapshot_subdir") else p.is_file()


def _guard_symlink(path: pathlib.Path) -> None:
    """禁止穿透 symlink 向 turbo 只读归档写入（AGENTS.md 第 13/14 条）。"""
    for p in [path, *path.parents]:
        if p == REPO_ROOT:
            break
        if p.is_symlink():
            raise SystemExit(
                f"错误: 落点路径 {p} 是符号链接（→ {p.resolve()}），拒绝穿透写入。\n"
                f"  本机上 v1-store/models/* 是指向 turbo 只读归档的引用；确需重取请先移除该 symlink "
                f"并改为本机实体目录。")


# ── 子命令 ────────────────────────────────────────────────────────────────────


def cmd_plan(args: argparse.Namespace, lock: dict, names: list[str]) -> int:
    root = pathlib.Path(args.root)
    total = missing = 0
    for name in names:
        entry = al.asset(name, lock)
        n = _entry_bytes(entry)
        total += n
        ok = _present(entry, root)
        missing += 0 if ok else 1
        print(f"  {'PRESENT' if ok else 'MISSING'}  {name:22s} {_fmt_bytes(n):>9s}  "
              f"{entry['dest']}\n            ← {_source_str(entry)}")
    print(f"ASSETS_PLAN total={_fmt_bytes(total)} assets={len(names)} missing={missing}")
    return 0


def cmd_verify(args: argparse.Namespace, lock: dict, names: list[str]) -> int:
    root = pathlib.Path(args.root)
    fails = []
    for name in names:
        t0 = time.perf_counter()
        f = al.verify(lock, root, level=args.level, names=[name])
        fails += f
        mark = "✓" if not f else "✗"
        print(f"  {mark} {name:22s} level={args.level} {time.perf_counter() - t0:6.2f}s")
        for line in f:
            print(f"      {line}")
    if fails:
        print(f"  修法: uv run python scripts/assets/fetch_assets.py fetch --force --assets {','.join(names)}")
    print(f"ASSETS={'PASS' if not fails else 'FAIL'} assets={len(names)} mismatches={len(fails)}")
    return 0 if not fails else 1


def _fetch_one(name: str, entry: dict, root: pathlib.Path, token: str | None) -> None:
    src = entry["source"]
    if src["type"] in ("gs_file", "gs_dir"):
        os.environ.setdefault("OPENPI_DATA_HOME", str(root / "v1-store" / "models"))
        from openpi.shared import download  # 延迟 import：只有 gs 分支才需要 fsspec/gcsfs

        got = download.maybe_download(src["uri"], gs={"token": "anon"} if src.get("anon") else {})
        print(f"    gs 落点 {got}")
        return

    import huggingface_hub

    if not huggingface_hub.__version__.startswith("0.32."):
        raise SystemExit(f"错误: huggingface_hub 版本漂移到 {huggingface_hub.__version__}，"
                         f"本脚本按根 uv.lock 钉的 0.32.x 写")
    if src["type"] == "hf_snapshot":
        got = huggingface_hub.snapshot_download(
            repo_id=src["repo_id"], repo_type=src.get("repo_type", "model"),
            revision=src["revision"], allow_patterns=src.get("allow_patterns"),
            cache_dir=str(root / "v1-store" / "cache" / "hf" / "hub"), token=token)
        print(f"    snapshot 落点 {got}")
        # revision 给的是 commit sha 时 huggingface_hub 不写 refs/main，而离线加载（HF_HUB_OFFLINE=1 按 repo_id 解析
        # main）与 verify 的 hf_snapshot_subdir 分支都要它 == 钉死的 revision。环境 B 从零复刻时首次踩中（2026-09-04）。
        # 缺则补写为钉死 revision；已存在且不同则响亮失败、不覆盖（那是另一份 main，不属本 lock）。
        ref = root / entry["dest"] / "refs" / "main"
        if not ref.is_file():
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(src["revision"])
            print(f"    补写 refs/main = {src['revision'][:12]}…（钉 sha 的 snapshot_download 不写 ref）")
        elif ref.read_text().strip() != src["revision"]:
            raise SystemExit(f"错误: {ref} 现为 {ref.read_text().strip()[:12]}… != lock revision "
                             f"{src['revision'][:12]}…，拒绝覆盖；请人工裁决这份 HF 缓存")
        return

    dest = root / entry["dest"]
    _guard_symlink(dest)
    cached = huggingface_hub.hf_hub_download(
        repo_id=src["repo_id"], repo_type=src.get("repo_type", "model"),
        revision=src["revision"], filename=src["filename"], token=token)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    shutil.copyfile(cached, tmp)          # HF 缓存里是 symlink→blob，copyfile 解引用取字节
    os.replace(tmp, dest)                 # 原子落盘
    print(f"    落盘 {dest}")


def cmd_fetch(args: argparse.Namespace, lock: dict, names: list[str]) -> int:
    root = pathlib.Path(args.root)
    # HF_HUB_OFFLINE 由 scripts/dataset/paths.sh 无条件 export；取资产这一步必须解除（删变量而非设 "0"）
    if os.environ.pop("HF_HUB_OFFLINE", None) is not None:
        print("  [fetch] 已在本进程内临时解除 HF_HUB_OFFLINE（不影响父进程与其它子进程）")
    os.environ.setdefault("HF_HOME", str(root / "v1-store" / "cache" / "hf"))
    os.environ.setdefault("HF_TOKEN_PATH", str(pathlib.Path.home() / ".cache/huggingface/token"))

    token = os.environ.get("HF_TOKEN")
    todo = [n for n in names if args.force or not _present(al.asset(n, lock), root)]
    # needs_token 的资产**开工前**先探一次可达性，401 立刻停，不要下到一半
    need_token = [n for n in todo if al.asset(n, lock).get("needs_token")]
    if need_token:
        import huggingface_hub

        api = huggingface_hub.HfApi()
        for n in need_token:
            src = al.asset(n, lock)["source"]
            try:
                api.repo_info(src["repo_id"], repo_type=src.get("repo_type", "model"), token=token)
            except Exception as exc:
                raise SystemExit(
                    f"错误: 私有资产 {n} 的 repo {src['repo_id']} 不可达：{exc}\n"
                    f"  修法: export HF_TOKEN=hf_…（需对该 repo 有 read 权限）\n"
                    f"  注意 HF_HOME 若被 paths.sh 指到 v1-store/cache/hf，`hf auth login` 存的 token "
                    f"不会被自动读到；本脚本已把 HF_TOKEN_PATH 指回 $HOME/.cache/huggingface/token") from exc

    if not todo:
        print("  全部已在位，无需下载")
    for name in todo:
        entry = al.asset(name, lock)
        print(f"  取 {name}（{_fmt_bytes(_entry_bytes(entry))}）← {_source_str(entry)}")
        t0 = time.perf_counter()
        _fetch_one(name, entry, root, token)
        print(f"    完成 {time.perf_counter() - t0:.1f}s")

    args.level = "full"                   # 新取回的一律全量复校，不给 cheap 档留侥幸
    return cmd_verify(args, lock, names)


def cmd_show(args: argparse.Namespace, lock: dict, names: list[str]) -> int:
    for name in names:
        print(json.dumps({name: al.asset(name, lock)}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="按 ASSETS_LOCK.json 取回并校验外部模型资产")
    ap.add_argument("cmd", choices=["plan", "fetch", "verify", "show"])
    ap.add_argument("--assets", default="", help="逗号分隔的资产名子集，默认全部")
    ap.add_argument("--level", choices=list(al.LEVELS), default="full")
    ap.add_argument("--lock", default=str(al.LOCK_PATH))
    ap.add_argument("--root", default=str(REPO_ROOT), help="仓库根（负向测试可注入假根）")
    ap.add_argument("--force", action="store_true", help="fetch：已在位也强制重取")
    args = ap.parse_args()

    lock = al.load_lock(args.lock)
    names = [n.strip() for n in args.assets.split(",") if n.strip()] or list(lock["assets"])
    for n in names:
        al.asset(n, lock)                 # 未知名当场 fail-loud
    return {"plan": cmd_plan, "fetch": cmd_fetch, "verify": cmd_verify, "show": cmd_show}[args.cmd](
        args, lock, names)


if __name__ == "__main__":
    sys.exit(main())
