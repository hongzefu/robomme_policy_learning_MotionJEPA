"""外部模型资产锁：读 ``ASSETS_LOCK.json`` 并校验落盘资产的身份。

为什么要有这个文件：五个外部权重（SigLIP / PaliGemma tokenizer / pi05_base / Wan2.1 VAE /
MotionJEPA encoder+decoder ckpt）此前的「真锚点」散落在三处 markdown 与一个无人读取的 txt 里，
代码侧只有 ``[[ -f ]]`` 存在性检查；``run_local.py`` 甚至现场哈希「即将被使用的那份 ckpt」再把结果
当期望值（自证循环）。本模块把这些锚点收敛成一张进 git 的表，并让链路真正读它。

**stdlib only**：主 venv（jax 0.5.3 / torch 2.7.1）与 wan 子 venv（torch 2.9.0 / diffusers 0.39.0）
都要 import 它，所以不得依赖 numpy / h5py / torch 等任何第三方包。

口径复用（三处必须同值，由 ``scripts/assets/test_assets_lock.py`` 断言）：
  - ``manifest_sha256`` 与 ``scripts/dataset/wan/wan_common.py::manifest_sha256``、
    ``src/mme_vla_suite/datastore/manifest.py::manifest_sha256`` 逐字同口径；
  - ``headtail_digest`` 与 ``src/mme_vla_suite/datastore/framesamp_store.py::headtail_digest``、
    ``scripts/training/g0/check_baseline_env.py::_headtail_digest`` 同口径。

两个校验档：
  - ``cheap``：逐文件字节数 + 首尾各 1 MiB 的 blake2b-128。O(ms)，用于每个 worker/每次 source 的前置。
    **挡不住保持长度的中段字节篡改**——这是显式契约，由测试 ``test_flip_midfile_*`` 钉住。
  - ``full``：逐文件全量 sha256。五个资产合计实测 10.4 s（其中 14 GB 的 pi05_base 占 8.6 s），
    用于取回后复校与显式的 ``fetch_assets.py verify``。
"""

from __future__ import annotations

import hashlib
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
LOCK_PATH = _HERE / "ASSETS_LOCK.json"

_HEADTAIL = 1 << 20        # 首尾各 1 MiB，与 framesamp_store._HEADTAIL 同值
_CHUNK = 8 << 20
LEVELS = ("cheap", "full")


# ── 摘要原语 ──────────────────────────────────────────────────────────────────


def manifest_sha256(payload: dict) -> str:
    """剔除 ``sha256`` 键后的 canonical JSON 的 sha256（lock 自防篡改用）。"""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(_CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def headtail_digest(p: pathlib.Path) -> tuple[int, str, bool]:
    """(字节数, 首尾各 1 MiB 的 blake2b-128, 是否覆盖全文件)。≤ 2 MiB 时覆盖全文件。"""
    size = p.stat().st_size
    h = hashlib.blake2b(digest_size=16)
    full = size <= 2 * _HEADTAIL
    with p.open("rb") as f:
        if full:
            h.update(f.read())
        else:
            h.update(f.read(_HEADTAIL))
            f.seek(size - _HEADTAIL)
            h.update(f.read(_HEADTAIL))
    return size, h.hexdigest(), full


# ── lock 读取 ─────────────────────────────────────────────────────────────────


def load_lock(path: str | pathlib.Path = LOCK_PATH) -> dict:
    """读 lock 并现场重算顶层 sha256；不符即 fail-loud（lock 被改动过）。"""
    p = pathlib.Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    expect = manifest_sha256(payload)
    if payload.get("sha256") != expect:
        raise SystemExit(
            f"错误: {p} 自哈希不符：记录 {payload.get('sha256')} != 现算 {expect}\n"
            f"  说明 lock 内容被改过；改 lock 必须同时更新顶层 sha256 字段。")
    return payload


def asset(name: str, lock: dict | None = None) -> dict:
    """取一条资产记录。未知名 fail-loud——绝不返回 None，防 `or None` 的静默跳过老路复活。"""
    lock = lock if lock is not None else load_lock()
    entry = lock["assets"].get(name)
    if entry is None:
        raise SystemExit(f"错误: ASSETS_LOCK.json 里没有资产 {name!r}；已知资产: "
                         f"{sorted(lock['assets'])}")
    return entry


def expected_sha256(name: str, lock: dict | None = None) -> str:
    """给 `--expected-ckpt-sha256` 之类的调用方提供仓库钉死的期望值。"""
    entry = asset(name, lock)
    sha = entry.get("sha256")
    if not sha:
        raise SystemExit(f"错误: 资产 {name!r} 不是单文件型（kind={entry.get('kind')!r}），无单一 sha256")
    return sha


# ── 校验 ──────────────────────────────────────────────────────────────────────


def _check_one(path: pathlib.Path, spec: dict, level: str, label: str) -> list[str]:
    if not path.is_file():
        return [f"{label}: 文件缺失 {path}"]
    size = path.stat().st_size
    if size != spec["bytes"]:
        return [f"{label}: 字节数 现 {size} != 期望 {spec['bytes']}（{path}）"]
    if level == "full":
        got = sha256_file(path)
        if got != spec["sha256"]:
            return [f"{label}: sha256 现 {got} != 期望 {spec['sha256']}（{path}）"]
    else:
        _, got, _ = headtail_digest(path)
        if got != spec["headtail"]:
            return [f"{label}: headtail 现 {got} != 期望 {spec['headtail']}（{path}）"]
    return []


def verify(lock: dict, root: str | pathlib.Path, *, level: str = "full",
           names: list[str] | None = None) -> list[str]:
    """校验落盘资产，返回失败原因列表（空 = 通过）。

    ``root`` 可注入（负向测试在 tmp_path 里造假资产树），**不得把仓库根写死在函数体内**。
    """
    if level not in LEVELS:
        raise SystemExit(f"错误: level 非法 {level!r}（∈ {LEVELS}）")
    root = pathlib.Path(root)
    fails: list[str] = []
    for name in (names if names is not None else list(lock["assets"])):
        entry = asset(name, lock)
        kind = entry["kind"]
        if kind == "file":
            fails += _check_one(root / entry["dest"], entry, level, name)
        elif kind == "dir":
            base = root / entry["dest"]
            if not base.is_dir():
                fails.append(f"{name}: 目录缺失 {base}")
                continue
            found = sum(1 for p in base.rglob("*") if p.is_file())
            if found != entry["n_files"]:
                fails.append(f"{name}: 文件数 现 {found} != 期望 {entry['n_files']}（{base}）")
            for rel, spec in entry["files"].items():
                fails += _check_one(base / rel, spec, level, f"{name}/{rel}")
        elif kind == "hf_snapshot_subdir":
            base = root / entry["dest"]
            ref = base / "refs" / "main"
            rev = entry["source"]["revision"]
            if not ref.is_file():
                fails.append(f"{name}: 缺 refs/main（{ref}）")
            elif ref.read_text().strip() != rev:
                fails.append(f"{name}: refs/main 现 {ref.read_text().strip()} != 期望 revision {rev}")
            snap = base / entry["snapshot_rel"]
            for rel, spec in entry["files"].items():
                link = snap / rel
                if not link.exists():
                    fails.append(f"{name}/{rel}: 缺失 {link}")
                    continue
                target = link.resolve()
                if spec.get("blob_name_is_sha256") and target.name != spec["sha256"]:
                    fails.append(f"{name}/{rel}: HF blob 文件名 {target.name} != sha256 {spec['sha256']}")
                fails += _check_one(target, spec, level, f"{name}/{rel}")
        else:
            fails.append(f"{name}: 未知 kind {kind!r}")
    return fails


def require(names: list[str], *, level: str = "cheap", root: str | pathlib.Path = REPO_ROOT,
            lock: dict | None = None) -> None:
    """链路前置：不过即 fail-loud，并给出修法。"""
    lock = lock if lock is not None else load_lock()
    fails = verify(lock, root, level=level, names=names)
    if fails:
        lines = "\n".join(f"  {f}" for f in fails)
        raise SystemExit(
            f"错误: 资产校验失败（level={level}）\n{lines}\n"
            f"  修法: uv run python scripts/assets/fetch_assets.py fetch --assets {','.join(names)} --force\n"
            f"  私有资产（HongzeFu/MotionJEPA）需先 export HF_TOKEN=hf_…")
