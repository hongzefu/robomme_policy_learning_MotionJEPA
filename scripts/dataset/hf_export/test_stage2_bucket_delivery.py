"""本地HTTP真实流式请求覆盖匿名校验和上传职责边界。"""
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

import pytest

import stage2_bucket_delivery as delivery


@pytest.fixture
def sample(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    payloads = {"motion_snapshot.ready.json": b'{"ready":true}',
                "5000/checkpoint": b"x" * (delivery.CHUNK_BYTES + 13)}
    for name, body in payloads.items():
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    bucket = next(iter(delivery.RUNS))
    manifest = stage / "delivery_manifest.json"
    manifest.write_text(json.dumps({"schema": "stage2-delivery-manifest-v1", "bucket": bucket,
        "run_name": delivery.RUNS[bucket],
        "files": {n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for n, b in payloads.items()}}))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _, files = delivery.load_manifest(manifest, sha, bucket)
    state = tmp_path / "state"
    state.mkdir()
    seen = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            super().do_GET()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(stage)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(delivery, "resolve_url", lambda bucket, name: f"http://127.0.0.1:{server.server_port}/{name}")
    cli = tmp_path / "hf"

    def listing(objects):
        # 真实子进程同时断言匿名环境和全新的空HF_HOME。
        cli.write_text(f"#!{sys.executable}\nimport os,json,pathlib,sys\n"
            "assert sys.argv[1:]==['buckets','list'," + repr(bucket) + ",'-R','--format','json']\n"
            "assert all(k not in os.environ for k in " + repr(delivery.TOKEN_KEYS) + ")\n"
            "assert os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']=='1'\n"
            "assert not list(pathlib.Path(os.environ['HF_HOME']).iterdir())\n"
            "print(" + repr(json.dumps([{"type": "file", "path": n, "size": v["bytes"]} for n, v in objects.items()])) + ")\n")
        cli.chmod(0o700)

    listing(files)
    for key in delivery.TOKEN_KEYS:
        monkeypatch.setenv(key, "不可泄漏的测试凭据")
    yield manifest, sha, bucket, state, cli, files, seen, listing
    server.shutdown()
    server.server_close()
    thread.join()


def test_streaming_complete_and_anonymous(sample):
    manifest, sha, bucket, state, cli, files, seen, _ = sample
    delivery.verify(manifest, sha, bucket, state, cli)
    report = json.loads((state / "verify_report.json").read_text())
    assert report["status"] == "pass"
    assert report["sha_match"] == report["size_equal"] == len(files)
    assert "/delivery_manifest.json" in {path for path, _ in seen}
    assert all(auth is None for _, auth in seen)
    assert sorted(p.name for p in state.iterdir()) == ["verify_report.json"]


@pytest.mark.parametrize("damage", ["sha", "missing", "extra", "size"])
def test_remote_failures(sample, damage):
    manifest, sha, bucket, state, cli, files, _, listing = sample
    if damage == "sha":
        (manifest.parent / "motion_snapshot.ready.json").write_bytes(b'{"ready":fals}')
    elif damage == "missing":
        files.pop("delivery_manifest.json")
    elif damage == "extra":
        files["unknown"] = {"bytes": 1}
    else:
        files["delivery_manifest.json"]["bytes"] += 1
    listing(files)
    assert delivery.main(["verify", "--manifest", str(manifest), "--expected-manifest-sha256", sha,
        "--bucket", bucket, "--state-dir", str(state), "--hf-cli", str(cli)]) == 1
    assert json.loads((state / "verify_report.json").read_text())["status"] == "fail"


def test_smoke_only_one_real_object(sample):
    manifest, sha, bucket, state, cli, _, seen, _ = sample
    delivery.verify(manifest, sha, bucket, state, cli, smoke=True)
    assert seen == [("/motion_snapshot.ready.json", None)]
    assert not (state / "verify_report.json").exists()


def test_manifest_sha_and_run_identity(sample):
    manifest, sha, bucket, *_ = sample
    with pytest.raises(ValueError, match="清单SHA"):
        delivery.load_manifest(manifest, "0" * 64, bucket)
    data = json.loads(manifest.read_text())
    data["run_name"] = "其他运行"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="身份"):
        delivery.load_manifest(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), bucket)


def test_verify_invalid_manifest_leaves_failure_report(sample):
    manifest, _, bucket, state, cli, *_ = sample
    with pytest.raises(ValueError, match="清单SHA"):
        delivery.verify(manifest, "0" * 64, bucket, state, cli)
    report = json.loads((state / "verify_report.json").read_text())
    assert report["status"] == "fail" and report["objects"] == 0


def test_upload_ownership_and_no_readback(sample, monkeypatch):
    manifest, sha, bucket, state, cli, files, _, _ = sample
    remote = {"entry": None}
    sent = []

    class Bucket:
        def __init__(self, *args, **kwargs):
            assert args[1] == "不可泄漏的测试凭据"
            assert kwargs["cli"] == (str(cli),)

        def entry(self):
            return remote["entry"]

        def _run(self, args, timeout):
            assert args == ["buckets", "create", bucket]
            remote["entry"] = {"private": False}
            return 0, "", ""

        def upload(self, local, name):
            sent.append(name)
            return type("Result", (), {"ok": True})()

    monkeypatch.setattr(delivery.io, "HFBucket", Bucket)
    delivery.upload(manifest, sha, bucket, state, cli)
    assert set(sent) == set(files)
    assert json.loads((state / "upload_receipt.json").read_text())["status"] == "uploaded"
    sent.clear()
    delivery.upload(manifest, sha, bucket, state, cli)
    assert set(sent) == set(files)  # 同尺寸远端不作为SHA依据，续跑直接重传。
    remote["entry"] = {"private": True}
    with pytest.raises(ValueError, match="必须公开"):
        delivery.upload(manifest, sha, bucket, state, cli)
    remote["entry"] = {"private": False}
    owner = json.loads((state / "owner.json").read_text())
    (state / "owner.json").write_text(json.dumps({**owner, "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="归属冲突"):
        delivery.upload(manifest, sha, bucket, state, cli)
    (state / "owner.json").unlink()
    with pytest.raises(ValueError, match="拒绝接管"):
        delivery.upload(manifest, sha, bucket, state, cli)


def test_upload_rejects_stage_drift_before_network(sample, monkeypatch):
    manifest, sha, bucket, state, cli, *_ = sample
    (manifest.parent / "motion_snapshot.ready.json").write_bytes("损坏".encode())
    monkeypatch.setattr(delivery.io, "HFBucket", lambda *a, **k: pytest.fail("不应联网"))
    with pytest.raises(ValueError, match="stage指纹"):
        delivery.upload(manifest, sha, bucket, state, cli)
