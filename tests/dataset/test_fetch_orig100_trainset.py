"""fetch_orig100_trainset.py 的离线测试：本地 HTTP 服务（支持 Range）模拟 bucket，假 manifest + 假 tar。"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dataset/hf_export/fetch_orig100_trainset.py"
spec = importlib.util.spec_from_file_location("fetch_orig100_trainset", SCRIPT)
fot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fot)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_tar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


SOURCE = {
    "data/0.pkl": b"pkl-zero" * 100,
    "data/1.pkl": b"pkl-one" * 50,
    "features/episode_0/kept_indices.json": b"[0, 1, 2]",
    "features/episode_5/kept_indices.json": b"[3, 4]",
}
DATA_TAR = {k: v for k, v in SOURCE.items() if k.startswith("data/")}
FEAT_TAR = {k: v for k, v in SOURCE.items() if k.startswith("features/")}
LATENTS = {
    "wan-latents/BinFill_ep0_exec.bin": b"latent-zero" * 100,
    "wan-latents/BinFill_ep0_exec.bin.sha256": b"latent-sidecar",
    "wan-latents/BinFill_ep0_exec.metadata.json": b'{"shape": [1, 16]}',
}
LATENTS_INCLUDE = ("wan-latents_tars/", "checksums/", "manifest/")


def build_bucket(root: Path, source_sums: dict[str, bytes] | None = None, data_tar=None, latent_tar=None) -> dict:
    """在 root 下造一个假 bucket，返回 {rel: bytes}。"""
    data_tar = DATA_TAR if data_tar is None else data_tar
    latent_tar = LATENTS if latent_tar is None else latent_tar
    sums_src = {**SOURCE, **LATENTS} if source_sums is None else source_sums
    tars = {
        "source/data_tars/data-00000.tar": make_tar(data_tar),
        "source/features_tars/features-00000.tar": make_tar(FEAT_TAR),
        "wan-latents_tars/wan-00000.tar": make_tar(latent_tar),
    }
    objs = {
        "meta/episode_manifest.json": b'{"episodes": 2}',
        "source/meta/info.json": b"{}",
        "framesamp-4x4/image_emb_4x4/part_000.bf16.bin": bytes(range(256)) * 4000,  # ~1 MB，测续传
        "framesamp-8x8/part_000.bf16.bin": b"not-included",
        "assets/robomme/norm_stats.json": b'{"norm": 1}',
        "checksums/sha256-source-files.txt": "".join(
            "%s  %s\n" % (sha(v), k) for k, v in sums_src.items()).encode(),
        "manifest/upload_manifest.json": json.dumps({"shards": [
            {"rel": "source/data_tars/data-00000.tar", "members": len(data_tar)},
            {"rel": "source/features_tars/features-00000.tar", "members": len(FEAT_TAR)},
            {"rel": "wan-latents_tars/wan-00000.tar", "members": len(LATENTS)},
        ]}).encode(),
        **tars,
    }
    manifest = {"bucket": "fake", "dataset": "fake-ds", "schema": "stage2-delivery-manifest-v1",
                "files": {k: {"bytes": len(v), "sha256": sha(v)} for k, v in objs.items()}}
    objs["delivery_manifest.json"] = json.dumps(manifest).encode()
    for rel, data in objs.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return objs


class RangeHandler(SimpleHTTPRequestHandler):
    """支持 `Range: bytes=N-`；可注入一次性/永久损坏。"""

    state: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        rel = self.path.lstrip("/")
        self.state.setdefault("hits", []).append((rel, self.headers.get("Range")))
        p = Path(self.directory) / rel
        if not p.is_file():
            self.send_error(404)
            return
        data = p.read_bytes()
        corrupt = self.state.get("corrupt", {})
        if corrupt.get(rel, 0) != 0:
            if corrupt[rel] > 0:
                corrupt[rel] -= 1
            data = b"X" + data[1:]
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "bucket"
    objs = build_bucket(root)
    RangeHandler.state = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(root)))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = "http://127.0.0.1:%d/" % server.server_address[1]
    yield {"root": root, "objs": objs, "url": url, "state": RangeHandler.state}
    server.shutdown()


def run(argv, capsys):
    code = fot.main([str(a) for a in argv])
    out = capsys.readouterr().out
    return code, out


def do_plan(bucket, tmp_path, capsys, include=None):
    plan = tmp_path / "plan.json"
    argv = ["plan", "--manifest-url", bucket["url"] + "delivery_manifest.json",
            "--base-url", bucket["url"], "--out", plan]
    if include is not None:
        argv += ["--include", *include]
    code, out = run(argv, capsys)
    assert code == 0
    return plan, out


def test_plan_filters_default_include(bucket, tmp_path, capsys):
    plan, out = do_plan(bucket, tmp_path, capsys)
    data = json.loads(plan.read_text())
    paths = {o["path"] for o in data["objects"]}
    assert "framesamp-8x8/part_000.bf16.bin" not in paths
    assert "wan-latents_tars/wan-00000.tar" not in paths
    assert "delivery_manifest.json" not in paths
    want = {k for k in bucket["objs"] if k.startswith(fot.DEFAULT_INCLUDE)}
    assert paths == want
    total = sum(len(bucket["objs"][k]) for k in want)
    assert "PLAN objects=%d bytes=%d" % (len(want), total) in out


def test_fetch_unpack_verify_end_to_end(bucket, tmp_path, capsys):
    plan, _ = do_plan(bucket, tmp_path, capsys)
    lib = tmp_path / "lib"
    code, out = run(["fetch", "--plan", plan, "--out", lib, "--workers", 3], capsys)
    assert code == 0, out
    n = len(json.loads(plan.read_text())["objects"])
    assert "TRAINSET_FETCH=PASS objects=%d" % n in out
    assert "sha_mismatch=0" in out
    assert (lib / "delivery_manifest.json").read_bytes() == bucket["objs"]["delivery_manifest.json"]
    assert not list(lib.rglob("*.part"))

    # 幂等：重跑不再发任何请求
    bucket["state"]["hits"].clear()
    code, out = run(["fetch", "--plan", plan, "--out", lib], capsys)
    assert code == 0 and "skipped=%d" % n in out
    assert bucket["state"]["hits"] == []

    code, out = run(["unpack", "--lib", lib, "--workers", 2], capsys)
    assert code == 0, out
    assert "TRAINSET_UNPACK=PASS members=%d mismatch=0" % len(SOURCE) in out
    for rel, data in SOURCE.items():
        assert (lib / "source" / rel).read_bytes() == data
    assert not (lib / "source/data_tars/data-00000.tar").exists()
    assert not (lib / "source/features_tars/features-00000.tar").exists()

    # 解包后重跑 fetch 不重下 tar；重跑 unpack 走标记
    bucket["state"]["hits"].clear()
    code, out = run(["fetch", "--plan", plan, "--out", lib], capsys)
    assert code == 0 and bucket["state"]["hits"] == []
    code, out = run(["unpack", "--lib", lib], capsys)
    assert code == 0 and "TRAINSET_UNPACK=PASS members=%d" % len(SOURCE) in out

    code, out = run(["verify", "--lib", lib, "--deep"], capsys)
    assert code == 0, out
    assert "TRAINSET_VERIFY=PASS" in out and "source_files=4 expected=4" in out

    # 平文件被改坏 → verify FAIL
    (lib / "assets/robomme/norm_stats.json").write_bytes(b'{"norm": 2}')
    code, out = run(["verify", "--lib", lib], capsys)
    assert code == 1 and "plain_mismatch=1" in out


def test_fetch_resumes_from_part(bucket, tmp_path, capsys):
    plan, _ = do_plan(bucket, tmp_path, capsys)
    lib = tmp_path / "lib"
    rel = "framesamp-4x4/image_emb_4x4/part_000.bf16.bin"
    data = bucket["objs"][rel]
    part = lib / (rel + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(data[: len(data) // 3])
    code, out = run(["fetch", "--plan", plan, "--out", lib], capsys)
    assert code == 0, out
    assert (rel, "bytes=%d-" % (len(data) // 3)) in bucket["state"]["hits"]
    assert (lib / rel).read_bytes() == data


@pytest.mark.parametrize("include", [LATENTS_INCLUDE, (*fot.DEFAULT_INCLUDE, "wan-latents_tars/")])
def test_latents_http_unpack_verify(bucket, tmp_path, capsys, include):
    """真实 HTTP 拉取和 tar 解包；只选 latent 与 source 混选均落在正确根目录。"""
    plan, _ = do_plan(bucket, tmp_path, capsys, include=include)
    lib = tmp_path / "lib"
    code, out = run(["fetch", "--plan", plan, "--out", lib], capsys)
    assert code == 0 and "TRAINSET_FETCH=PASS" in out
    code, out = run(["unpack", "--lib", lib], capsys)
    assert code == 0 and "TRAINSET_UNPACK=PASS" in out, out
    for rel, data in LATENTS.items():
        assert (lib / rel).read_bytes() == data
    assert not (lib / "source/wan-latents").exists()
    tar_rel = "wan-latents_tars/wan-00000.tar"
    assert not (lib / tar_rel).exists()
    assert fot.unpack_marker(lib, tar_rel).is_file()
    expected = len(LATENTS) + (len(SOURCE) if "source/data_tars/" in include else 0)
    code, out = run(["verify", "--lib", lib, "--deep"], capsys)
    assert code == 0 and "TRAINSET_VERIFY=PASS" in out, out
    assert f"unpacked_files={expected} expected={expected}" in out and "deep_mismatch=0" in out

    # 成功删除 tar 后，重跑 fetch/unpack 沿用既有标记，不再下载。
    bucket["state"]["hits"].clear()
    code, out = run(["fetch", "--plan", plan, "--out", lib], capsys)
    assert code == 0 and bucket["state"]["hits"] == []
    code, out = run(["unpack", "--lib", lib], capsys)
    assert code == 0 and "TRAINSET_UNPACK=PASS" in out

    # 等长篡改不能靠计数检出，必须由逐成员 SHA 拒绝。
    rel = next(iter(LATENTS))
    (lib / rel).write_bytes(b"X" + LATENTS[rel][1:])
    code, out = run(["verify", "--lib", lib, "--deep"], capsys)
    assert code == 1 and "TRAINSET_VERIFY=FAIL" in out and "deep_mismatch=1" in out
    (lib / rel).write_bytes(LATENTS[rel])

    # 多出文件时，原有成员 SHA 仍匹配，根目录文件数必须拒绝。
    (lib / "wan-latents/extra.bin").write_bytes(b"unexpected")
    code, out = run(["verify", "--lib", lib, "--deep"], capsys)
    assert code == 1 and "TRAINSET_VERIFY=FAIL" in out and "deep_mismatch=0" in out
    assert f"unpacked_files={expected + 1} expected={expected}" in out


@pytest.mark.parametrize("damage", ["sha", "members"])
def test_latents_unpack_rejects_member_damage(bucket, tmp_path, capsys, damage):
    bad = dict(LATENTS)
    if damage == "sha":
        rel = next(iter(bad))
        bad[rel] = b"X" + bad[rel][1:]
    else:
        bad.pop("wan-latents/BinFill_ep0_exec.metadata.json")
    build_bucket(bucket["root"], latent_tar=bad)
    plan, _ = do_plan(bucket, tmp_path, capsys, include=LATENTS_INCLUDE)
    lib = tmp_path / "lib"
    code, out = run(["fetch", "--plan", plan, "--out", lib], capsys)
    assert code == 0, out
    code, out = run(["unpack", "--lib", lib], capsys)
    assert code == 1 and "TRAINSET_UNPACK=FAIL" in out and "mismatch=1" in out, out
    assert (lib / "wan-latents_tars/wan-00000.tar").is_file()
    assert not fot.unpack_marker(lib, "wan-latents_tars/wan-00000.tar").exists()
    assert not (lib / "source/wan-latents").exists()


def test_fetch_mismatch_retry_then_fail(bucket, tmp_path, capsys):
    plan, _ = do_plan(bucket, tmp_path, capsys)
    rel = "assets/robomme/norm_stats.json"
    # 损坏一次：重试后通过
    bucket["state"]["corrupt"] = {rel: 1}
    code, out = run(["fetch", "--plan", plan, "--out", tmp_path / "lib1", "--retries", 2], capsys)
    assert code == 0 and "FETCH_MISMATCH " + rel in out and "TRAINSET_FETCH=PASS" in out
    # 永久损坏：有限次后 FAIL，且不留 .part 与目标
    bucket["state"]["corrupt"] = {rel: -1}
    lib2 = tmp_path / "lib2"
    code, out = run(["fetch", "--plan", plan, "--out", lib2, "--retries", 1], capsys)
    assert code == 1 and "TRAINSET_FETCH=FAIL" in out and "sha_mismatch=1" in out
    assert not (lib2 / rel).exists() and not (lib2 / (rel + ".part")).exists()


def _fetch_with_bucket(tmp_path, capsys, **kw):
    root = tmp_path / "b2"
    build_bucket(root, **kw)
    lib = tmp_path / "lib"
    plan = tmp_path / "plan.json"
    run(["plan", "--manifest", root / "delivery_manifest.json", "--base-url", root.as_uri() + "/",
         "--out", plan], capsys)
    code, out = run(["fetch", "--plan", plan, "--out", lib], capsys)  # file:// 也可用（不测续传）
    assert code == 0, out
    return lib


def test_unpack_member_sha_mismatch_keeps_tar(tmp_path, capsys):
    bad = dict(SOURCE)
    bad["data/1.pkl"] = b"different"
    lib = _fetch_with_bucket(tmp_path, capsys, source_sums=bad)
    code, out = run(["unpack", "--lib", lib], capsys)
    assert code == 1 and "TRAINSET_UNPACK=FAIL" in out and "mismatch=1" in out
    assert (lib / "source/data_tars/data-00000.tar").exists()
    assert not (lib / "source/data/1.pkl").exists()
    code, out = run(["verify", "--lib", lib], capsys)
    assert code == 1 and "tars_unpacked=1/2" in out


def test_unpack_rejects_path_traversal(tmp_path, capsys):
    evil = dict(DATA_TAR)
    evil["../evil.txt"] = b"pwned"
    lib = _fetch_with_bucket(tmp_path, capsys, data_tar=evil)
    code, out = run(["unpack", "--lib", lib], capsys)
    assert code == 1 and "unsafe:../evil.txt" in out
    assert not (lib / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()
    assert (lib / "source/data_tars/data-00000.tar").exists()


def test_safe_rel():
    for bad in ["/abs", "a/../b", "..", "./a", "a\\b", ""]:
        with pytest.raises(ValueError):
            fot.safe_rel(bad)
    assert str(fot.safe_rel("data/0.pkl")) == "data/0.pkl"


def test_zz_banner():
    print("FETCH_TOOL_TESTS=PASS")
