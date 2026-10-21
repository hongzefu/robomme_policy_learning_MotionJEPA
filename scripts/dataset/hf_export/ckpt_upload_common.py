"""检查点增量上传的共享工具（U1 编写并唯一维护；U2 的 VLA 上传器可复用）。

本模块只依赖标准库，不导入 torch/JAX，不初始化 CUDA。对外接口如下，全部为稳定名称：

时间与日志
    utc_now() -> str
        当前 UTC 时间，格式 ``YYYY-MM-DDTHH:MM:SSZ``。
    log(event, **fields) -> None
        向 stdout 打一行 ``<UTC> <EVENT> k=v ...`` 并立即 flush；值里的空白替换成 ``_``。

文件身份
    sha256_file(path) -> (sha256_hex, bytes)
        流式计算整文件 SHA256 与字节数。
    stat_identity(path) -> dict(size=int, mtime_ns=int)
        只读 stat，用于判断源文件在两次检查之间是否变化。
    copy_with_hash(src, dst) -> (sha256_hex, bytes)
        独立复制（逐块读写并 fsync，绝不建硬链接），同时返回复制内容的 SHA256。
    atomic_write_json(path, obj) / atomic_write_text(path, text)
        同目录临时文件写完 fsync 后 os.replace，读者只会看到旧版或新版。
    read_json(path) -> object | None
        文件不存在返回 None；存在但不是合法 JSON 抛 ValueError。

进程与凭据
    SingleInstanceLock(path)
        上下文管理器；fcntl 非阻塞独占锁，已被占用时抛 UploaderStop("UPLOAD_LOCK_BUSY")。
    load_hf_token(env_file) -> str
        按 ``KEY=VALUE`` 逐行解析（不执行 shell），只取 HF_TOKEN；缺失抛 UploaderStop。
        token 只进子进程环境变量，不打印、不写日志或清单。

远端（HF Storage Bucket，经 hf CLI 子进程）
    HFBucket(bucket_id, token, *, cache_root, cli=HF_CLI, io_prefix=IO_PREFIX)
        .namespace / .uri(path)
        .whoami() -> str
        .list_namespace() -> list[dict]          # hf buckets list <ns> --format json
        .entry() -> dict | None                  # 本 bucket 的 {id, private, total_files, ...}
        .create_private() -> None                # hf buckets create <id> --private
        .list_files() -> dict[str, int]          # 路径 -> 字节数（-R --format json）
        .upload(local, remote_path) -> RemoteResult
        .download(remote_path, local) -> RemoteResult
      每次调用都是独立子进程：前缀 ``ionice -c3 nice -n19``，``CUDA_VISIBLE_DEVICES=''``，
      只在该子进程里删除 ``HF_HUB_OFFLINE``，并注入 HF_TOKEN（鉴权读写，私有 bucket 可用）。
      失败不抛异常而是返回 ``RemoteResult(ok=False, detail=...)``；detail 已抹去 token。
      ``whoami/list_namespace/list_files/create_private`` 失败抛 RemoteError。

账本与归属
    ObjectLedger(root, *, schema, run_name, bucket_id, kind)
        每个对象一份 JSON marker：``<root>/markers/<对象名>.json``；归属文件 ``<root>/owner.json``。
        .owner_expected() -> dict
        .check_all() -> None        # 扫全部已有 marker/owner，身份不符抛 UploaderStop("UPLOAD_MARKER_MISMATCH")
        .load(name) -> dict | None  # 读并核身份
        .save(name, marker) -> dict # 补齐身份字段后原子写
        .names() -> list[str]
    ensure_private_bucket(bucket, ledger) -> None
        bucket 不存在则创建为私有；已存在必须 private=true；已存在且非空而本地无本任务 owner.json 则拒绝。
        通过后写 owner.json。拒绝时抛 UploaderStop（BUCKET_NOT_PRIVATE / BUCKET_FOREIGN / ...）。

异常
    UploaderStop(event, detail)：必须停掉上传器的情况（身份冲突、归属不符等），由入口转成非零退出码。
    RemoteError(detail)：远端查询类命令失败（不是逐对象上传失败）。
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HF_CLI = ("uvx", "--python", "3.11", "--from", "huggingface_hub[cli]==1.30.0", "hf")
IO_PREFIX = ("ionice", "-c", "3", "nice", "-n", "19")
CHUNK = 8 << 20
UV_CACHE_DEFAULT = "/scratch/hongze/.cache/uv"
UV_PYTHON_DEFAULT = "/scratch/hongze/.cache/uv-python"


class UploaderStop(Exception):
    """必须停止上传器的事件；event 是日志事件名。"""

    def __init__(self, event: str, detail: str = ""):
        super().__init__(f"{event}: {detail}")
        self.event = event
        self.detail = detail


class RemoteError(Exception):
    """远端查询命令失败（网络、鉴权、CLI 错误）。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return re.sub(r"\s+", "_", str(value))


def log(event: str, **fields) -> None:
    parts = [utc_now(), event] + [f"{k}={_fmt(v)}" for k, v in fields.items()]
    print(" ".join(parts), flush=True)


def sha256_file(path) -> tuple[str, int]:
    digest, total = hashlib.sha256(), 0
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def stat_identity(path) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def copy_with_hash(src, dst) -> tuple[str, int]:
    """独立复制 src → dst（新 inode，不是硬链接），返回复制内容的 SHA256 与字节数。"""
    digest, total = hashlib.sha256(), 0
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    with open(src, "rb") as reader, open(dst, "xb") as writer:
        for block in iter(lambda: reader.read(CHUNK), b""):
            digest.update(block)
            writer.write(block)
            total += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    return digest.hexdigest(), total


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n")


def read_json(path):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return None
    return json.loads(path.read_text())


class SingleInstanceLock:
    """同一导出目录同一时刻只允许一个上传器进程（watch 与 finalize 互斥）。"""

    def __init__(self, path):
        self.path = Path(path)
        self._fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._fd)
            self._fd = None
            raise UploaderStop("UPLOAD_LOCK_BUSY", str(self.path))
        os.ftruncate(self._fd, 0)
        os.write(self._fd, f"{os.getpid()}\n".encode())
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        return False


def load_hf_token(env_file) -> str:
    token = None
    for line in Path(env_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == "HF_TOKEN":
            token = value.strip().strip('"').strip("'")
    if not token:
        raise UploaderStop("HF_TOKEN_MISSING", f"{env_file} 未提供 HF_TOKEN")
    return token


@dataclass
class RemoteResult:
    ok: bool
    rc: int
    detail: str


class HFBucket:
    """HF Storage Bucket 的最小命令封装；每个操作一个低优先级、无 GPU 的子进程。"""

    def __init__(self, bucket_id: str, token: str, *, cache_root, cli=HF_CLI, io_prefix=IO_PREFIX,
                 timeout_query=600, timeout_transfer=6 * 3600):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", bucket_id):
            raise UploaderStop("BUCKET_ID_INVALID", bucket_id)
        self.bucket_id = bucket_id
        self.namespace = bucket_id.split("/")[0]
        self._token = token
        self._cache_root = Path(cache_root)
        self._cli = tuple(cli)
        self._io_prefix = tuple(io_prefix)
        self._timeout_query = timeout_query
        self._timeout_transfer = timeout_transfer

    def uri(self, remote_path: str) -> str:
        return f"hf://buckets/{self.bucket_id}/{remote_path.lstrip('/')}"

    def _env(self) -> dict:
        env = dict(os.environ)
        for key in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(key, None)
        env.update({
            "HF_TOKEN": self._token,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(self._cache_root / "hf-home"),
            "HF_XET_CACHE": str(self._cache_root / "hf-xet"),
            "UV_LINK_MODE": "copy",
            "PYTHONUNBUFFERED": "1",
        })
        env.setdefault("UV_CACHE_DIR", UV_CACHE_DEFAULT)
        env.setdefault("UV_PYTHON_INSTALL_DIR", UV_PYTHON_DEFAULT)
        return env

    def _clean(self, text: str) -> str:
        text = (text or "").replace(self._token, "<redacted>") if self._token else (text or "")
        return " | ".join(text.strip().splitlines()[-5:])[:800]

    def _run(self, args, timeout) -> tuple[int, str, str]:
        argv = list(self._io_prefix) + list(self._cli) + list(args)
        try:
            proc = subprocess.run(argv, env=self._env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"
        except OSError as exc:
            return 127, "", f"{type(exc).__name__}: {exc}"
        return proc.returncode, proc.stdout, proc.stderr

    def _query_json(self, args):
        rc, out, err = self._run(list(args) + ["--format", "json"], self._timeout_query)
        if rc != 0:
            raise RemoteError(f"hf {' '.join(args[:2])} rc={rc} {self._clean(err)}")
        try:
            return json.loads(out)
        except ValueError as exc:
            raise RemoteError(f"hf {' '.join(args[:2])} 输出不是 JSON: {exc}")

    def whoami(self) -> str:
        rc, out, err = self._run(["auth", "whoami", "--format", "agent"], self._timeout_query)
        if rc != 0:
            raise RemoteError(f"hf auth whoami rc={rc} {self._clean(err)}")
        return out

    def list_namespace(self) -> list:
        data = self._query_json(["buckets", "list", self.namespace])
        if not isinstance(data, list):
            raise RemoteError("bucket 列表不是数组")
        return data

    def entry(self):
        for item in self.list_namespace():
            if isinstance(item, dict) and item.get("id") == self.bucket_id:
                return item
        return None

    def create_private(self) -> None:
        rc, _, err = self._run(["buckets", "create", self.bucket_id, "--private"], self._timeout_query)
        if rc != 0:
            raise RemoteError(f"hf buckets create rc={rc} {self._clean(err)}")

    def list_files(self) -> dict:
        data = self._query_json(["buckets", "list", self.bucket_id, "-R"])
        if not isinstance(data, list):
            raise RemoteError("bucket 文件列表不是数组")
        files = {}
        for item in data:
            if isinstance(item, dict) and item.get("type", "file") == "file":
                files[str(item["path"])] = int(item["size"])
        return files

    def upload(self, local, remote_path: str) -> RemoteResult:
        rc, _, err = self._run(["buckets", "cp", str(local), self.uri(remote_path)], self._timeout_transfer)
        return RemoteResult(rc == 0, rc, self._clean(err))

    def download(self, remote_path: str, local) -> RemoteResult:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        rc, _, err = self._run(["buckets", "cp", self.uri(remote_path), str(local)], self._timeout_transfer)
        return RemoteResult(rc == 0, rc, self._clean(err))


class ObjectLedger:
    """逐对象身份 marker；状态只允许 snapshot / uploaded / verified。"""

    STATES = ("snapshot", "uploaded", "verified")

    def __init__(self, root, *, schema: str, run_name: str, bucket_id: str, kind: str):
        self.root = Path(root)
        self.marker_dir = self.root / "markers"
        self.owner_path = self.root / "owner.json"
        self.schema, self.run_name, self.bucket_id, self.kind = schema, run_name, bucket_id, kind

    def owner_expected(self) -> dict:
        return {"schema": self.schema, "run_name": self.run_name, "bucket_id": self.bucket_id,
                "kind": self.kind}

    def _identity_ok(self, data, name=None) -> bool:
        if not isinstance(data, dict):
            return False
        if any(data.get(k) != v for k, v in self.owner_expected().items()):
            return False
        if name is not None:
            return data.get("object") == name and data.get("status") in self.STATES
        return True

    def read_owner(self):
        try:
            owner = read_json(self.owner_path)
        except ValueError:
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", "owner.json 不是合法 JSON")
        if owner is not None and not self._identity_ok(owner):
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", f"owner.json 属于其他 run/bucket: {owner}")
        return owner

    def write_owner(self) -> None:
        atomic_write_json(self.owner_path, dict(self.owner_expected(), created_at=utc_now()))

    def names(self) -> list:
        if not self.marker_dir.is_dir():
            return []
        return sorted(p.name[:-len(".json")] for p in self.marker_dir.glob("*.json"))

    def check_all(self) -> None:
        self.read_owner()
        for name in self.names():
            self.load(name)

    def load(self, name: str):
        path = self.marker_dir / f"{name}.json"
        if path.is_symlink():
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", f"{path} 是符号链接")
        try:
            data = read_json(path)
        except ValueError:
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", f"{path} 不是合法 JSON")
        if data is None:
            return None
        if not self._identity_ok(data, name):
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", f"{path} 身份不符")
        return data

    def save(self, name: str, marker: dict) -> dict:
        data = dict(marker)
        data.update(self.owner_expected())
        data["object"] = name
        if data.get("status") not in self.STATES:
            raise ValueError(f"非法状态 {data.get('status')}")
        data["updated_at"] = utc_now()
        atomic_write_json(self.marker_dir / f"{name}.json", data)
        return data


def ensure_private_bucket(bucket: HFBucket, ledger: ObjectLedger) -> None:
    """核 HF 身份与 bucket 可见性/归属；只会新建私有 bucket，绝不修改已有 bucket 的可见性。"""
    who = bucket.whoami()
    if bucket.namespace not in who:
        raise UploaderStop("HF_WHOAMI_MISMATCH", "当前凭据不属于 bucket 命名空间")
    owner = ledger.read_owner()
    entry = bucket.entry()
    if entry is None:
        if owner is not None:
            raise UploaderStop("BUCKET_MISSING", "本地有归属记录但远端 bucket 不存在")
        bucket.create_private()
        entry = bucket.entry()
        if entry is None:
            raise UploaderStop("BUCKET_CREATE_FAIL", "创建后仍查不到 bucket")
        log("BUCKET_CREATED", bucket=bucket.bucket_id, private=entry.get("private"))
    if entry.get("private") is not True:
        raise UploaderStop("BUCKET_NOT_PRIVATE", f"{bucket.bucket_id} private={entry.get('private')}")
    total = entry.get("total_files", entry.get("totalFiles"))
    if owner is None:
        if not isinstance(total, int) or total != 0:
            raise UploaderStop("BUCKET_FOREIGN", f"{bucket.bucket_id} 非空(total_files={total})且无本任务归属记录")
        ledger.write_owner()
    log("BUCKET_READY", bucket=bucket.bucket_id, private=True, total_files=total)
