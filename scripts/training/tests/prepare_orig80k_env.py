"""为固定上游提交创建独立uv环境；只临时移除缺失workspace成员。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
OLD_MEMBER = 'members = ["packages/*", "sandbox2/flash_attn_jax"]'
NEW_MEMBER = 'members = ["packages/*"]'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(argv, cwd, *, env=None, capture=False, input_text=None):
    return subprocess.run(argv, cwd=cwd, env=env, input=input_text, text=True,
                          check=True, capture_output=capture)


def git(repo, *argv):
    return command(["git", *argv], repo, capture=True).stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workspace_patch(original):
    """补丁的完整上下文来自固定提交，拒绝含糊或多次出现的成员声明。"""
    import difflib

    require(original.count(OLD_MEMBER) == 1, "上游workspace声明不符合计划，停止准备")
    modified = original.replace(OLD_MEMBER, NEW_MEMBER)
    return "".join(difflib.unified_diff(original.splitlines(keepends=True),
                                      modified.splitlines(keepends=True),
                                      fromfile="a/pyproject.toml", tofile="b/pyproject.toml"))


PROBE = r"""
import hashlib, importlib.metadata, importlib.util, json, pathlib, sys
result = {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
          "packages": {}, "modules": {}}
for name in ("numpy", "jax", "jaxlib", "torch", "flax", "optax", "orbax-checkpoint", "ml-dtypes"):
    result["packages"][name] = importlib.metadata.version(name)
for name in ("mme_vla_suite", "openpi", "openpi_client"):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise RuntimeError("缺项目模块: " + name)
    path = pathlib.Path(spec.origin).resolve()
    result["modules"][name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps(result, sort_keys=True))
"""


def validate_probes(upstream, current, worktree):
    require(upstream["python"] == current["python"], "两侧Python版本/构建信息不一致")
    require(upstream["packages"] == current["packages"], "两侧关键依赖版本不一致")
    require(Path(upstream["prefix"]).resolve() == (worktree / ".venv").resolve(), "A未使用独立venv")
    for name, module in upstream["modules"].items():
        require(Path(module["path"]).is_relative_to(worktree), f"A模块来源污染: {name}")
    for name, module in current["modules"].items():
        path = Path(module["path"])
        require(path.is_relative_to(ROOT) and not path.is_relative_to(worktree), f"B模块来源污染: {name}")


def prepare(args):
    root = ROOT.resolve()
    require(Path.cwd().resolve() == root, "环境准备必须从主仓库运行")
    require(not git(root, "status", "--porcelain"), "环境P0必须从clean HEAD启动")
    require(git(root, "rev-parse", "HEAD") == args.head, "实施HEAD与准备锚点不符")
    worktree = Path(args.worktree).absolute()
    records = Path(args.records).absolute()
    for path in (worktree, records):
        require(path.resolve().is_relative_to(root / "v1-store"), "P0产物必须落本仓库v1-store")
        require(not path.exists() and not path.is_symlink(), f"拒绝复用已有输出: {path}")
    require(not (root / "v1-store").is_symlink(), "主副本v1-store必须为实体目录")
    records.mkdir(parents=True)
    metadata = {"schema": 1, "head": args.head, "upstream": UPSTREAM, "worktree": str(worktree),
                "started_at": time.time(), "success": False, "commands": []}
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(root / "v1-store/cache/uv"), UV_LINK_MODE="copy",
               UV_PYTHON_INSTALL_DIR=str(root / "v1-store/cache/uv-python"),
               UV_PROJECT_ENVIRONMENT=str(worktree / ".venv"),
               XDG_CACHE_HOME=str(root / "v1-store/cache/xdg"), UV_PYTHON_DOWNLOADS="never",
               PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    patch_applied = False
    patch_path = None
    try:
        command(["git", "worktree", "add", "--detach", str(worktree), UPSTREAM], root)
        require(git(worktree, "rev-parse", "HEAD") == UPSTREAM, "worktree版本不符")
        require(not (worktree / "sandbox2/flash_attn_jax").exists(), "缺失成员的前提改变，停止准备")
        project = worktree / "pyproject.toml"
        lock = worktree / "uv.lock"
        originals = {"pyproject.toml": sha(project), "uv.lock": sha(lock)}
        require(sha(lock) == sha(root / "uv.lock"), "两侧uv.lock已不同，不能沿用同依赖假设")
        patch = workspace_patch(project.read_text())
        metadata["original_sha256"] = originals
        metadata["patch_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
        # 临时补丁只在本轮/tmp载体中存放，两次操作消费相同字节。
        with tempfile.TemporaryDirectory(prefix="orig80k-env-") as temporary:
            patch_path = Path(temporary) / "workspace.patch"
            patch_path.write_text(patch)
            command(["git", "apply", "--check", str(patch_path)], worktree)
            command(["git", "apply", str(patch_path)], worktree)
            patch_applied = True
            try:
                argv = ["uv", "sync", "--frozen", "--python", str(Path(sys.executable).resolve())]
                metadata["commands"].append(argv)
                command(argv, worktree, env=env)
            finally:
                command(["git", "apply", "--reverse", "--check", str(patch_path)], worktree)
                command(["git", "apply", "--reverse", str(patch_path)], worktree)
                patch_applied = False
        require({name: sha(worktree / name) for name in originals} == originals,
                "环境准备后源码或锁文件摘要变化")
        require(not git(worktree, "status", "--porcelain"), "A环境准备后工作区不clean")
        require(not git(root, "status", "--porcelain") and git(root, "rev-parse", "HEAD") == args.head,
                "环境准备期间主仓库改变")
        # 两个解释器均为uv管理环境，A无需重新解析已恢复的workspace声明。
        upstream = json.loads(command([str(worktree / ".venv/bin/python"), "-c", PROBE],
                                      worktree, env=env, capture=True).stdout)
        current = json.loads(command([str(root / ".venv/bin/python"), "-c", PROBE],
                                     root, env=env, capture=True).stdout)
        metadata.update(upstream_environment=upstream, current_environment=current)
        validate_probes(upstream, current, worktree)
        metadata["success"] = True
        print("ORIG80K_ENV=PASS", flush=True)
    finally:
        metadata.update(ended_at=time.time(), temporary_patch_applied=patch_applied)
        with (records / "environment.json").open("x") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        if not metadata["success"]:
            print("ORIG80K_ENV=FAIL 保留本轮worktree及记录，禁止继续起跑", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--records", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
