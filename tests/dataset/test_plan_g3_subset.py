"""G3 子集的离线成功、断点续传、摘要拒绝及原闸门抽样对拍。"""

import ast
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dataset/wan_stage2/plan_g3_subset.py"
SPEC = importlib.util.spec_from_file_location("plan_g3_subset", SCRIPT)
g3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g3)


def meta(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class Handler(SimpleHTTPRequestHandler):
    """记录对象请求并支持断点；测试不访问外网。"""
    hits = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.hits.append((self.path, self.headers.get("Range")))
        data = (Path(self.directory) / self.path.lstrip("/")).read_bytes()
        start = int(self.headers["Range"].split("=")[1].split("-")[0]) if "Range" in self.headers else 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
        self.end_headers()
        self.wfile.write(data[start:])


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "bucket"
    windows = [dict(window_key=f"Task/ep{i:03d}/exec/s0000/wp2", task="Task", episode=i,
                    split="val" if i % 5 else "train", row_start=i * 127) for i in range(75)]
    objs = {}
    for name in g3.INDEX:
        if name.endswith(".parquet"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(windows), target)
            objs[name] = target.read_bytes()
        else:
            episodes = []
            for w in windows:
                name_ep = f"masks/source_valid/Task/ep{w['episode']:03d}.bits"
                raw = f"掩码{w['episode']}".encode()
                objs[name_ep] = raw
                episodes.append(dict(task="Task", episode=w["episode"], file=name_ep, **meta(raw)))
            objs[name] = json.dumps(dict(episodes=episodes)).encode()
    for sid in range(3):
        files = {}
        for filename in g3.SHARD_FILES:
            raw = (f"{sid}/{filename}/" * 12).encode()
            objs[f"shards/{sid:06d}/{filename}"] = raw
            files[filename] = meta(raw)
        objs[f"shards/{sid:06d}/DONE.json"] = json.dumps(dict(files=files)).encode()
    objs[g3.SPEC] = json.dumps(dict(schema_version=2, rows_per_shard=4096,
                                   index_files={name: meta(objs[name]) for name in g3.INDEX})).encode()
    objs[g3.SUMS] = "".join(f"{meta(raw)['sha256']}  {name}\n" for name, raw in sorted(objs.items())).encode()
    for name, raw in objs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    Handler.hits = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(root=root, objects=objs, windows=windows,
                          url=f"http://127.0.0.1:{server.server_port}/")
    server.shutdown()
    thread.join()


def plan(bucket, dest):
    assert g3.main(["plan", "--root", str(dest), "--base-url", bucket.url, "--retries", "0"]) == 0
    return json.loads((dest / "g3-subset-plan.json").read_bytes())


def test_selection_matches_gate_g3():
    windows = {f"Task/{i:04d}": dict(split="val" if i % 3 else "train") for i in reversed(range(200))}
    tree = ast.parse((ROOT / "scripts/dataset/wan_stage2/validate_stage2.py").read_text())
    gate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "gate_g3")
    assignments = [node for node in gate.body if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("candidates", "selected")]
    scope = dict(reader=SimpleNamespace(windows=windows), np=np)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "gate_g3", "exec"), scope)
    assert g3.select_windows(windows) == scope["selected"]
    with pytest.raises(ValueError, match="不足32"):
        g3.select_windows({"only": dict(split="val")})


def test_plan_fetch_success_and_resume(bucket, tmp_path, capsys):
    dest = tmp_path / "subset"
    result = plan(bucket, dest)
    by_key = {w["window_key"]: w for w in bucket.windows}
    assert result["windows"] == g3.select_windows(by_key)
    shards = {by_key[key]["row_start"] // 4096 for key in result["windows"]}
    want = {g3.SPEC, g3.SUMS, *g3.INDEX}
    want.update(f"shards/{sid:06d}/{filename}" for sid in shards for filename in (*g3.SHARD_FILES, "DONE.json"))
    want.update(f"masks/source_valid/Task/ep{by_key[key]['episode']:03d}.bits" for key in result["windows"])
    assert {obj["path"] for obj in result["objects"]} == want
    assert result["total_bytes"] == sum(len(bucket.objects[name]) for name in want)
    assert all(path in {"/" + g3.SPEC, "/" + g3.SUMS, *("/" + name for name in g3.INDEX)}
               or path.endswith("/DONE.json") for path, _ in Handler.hits)
    obj = next(obj for obj in result["objects"] if obj["path"].endswith("rgb.raw.f16.bin"))
    path = dest / obj["path"]
    path.with_name(path.name + ".part").write_bytes(bucket.objects[obj["path"]][:7])
    assert g3.main(["fetch", "--root", str(dest), "--retries", "0"]) == 0
    assert ("/" + obj["path"], "bytes=7-") in Handler.hits
    for name in want:
        assert (dest / name).read_bytes() == bucket.objects[name]
    Handler.hits.clear()
    assert g3.main(["fetch", "--root", str(dest), "--retries", "0"]) == 0
    assert Handler.hits == []
    out = capsys.readouterr().out
    assert "G3_SUBSET_PLAN=PASS windows=32" in out
    assert f"G3_SUBSET=PASS shards={len(shards)} bytes={result['total_bytes']} sha_mismatch=0" in out


def test_reject_index_sha(bucket, tmp_path, capsys):
    (bucket.root / "index/windows.parquet").write_bytes("受控损坏".encode())
    dest = tmp_path / "subset"
    assert g3.main(["plan", "--root", str(dest), "--base-url", bucket.url, "--retries", "0"]) == 1
    assert "G3_SUBSET_PLAN=FAIL" in capsys.readouterr().err
    assert not (dest / "g3-subset-plan.json").exists()


def test_reject_shard_sha(bucket, tmp_path, capsys):
    dest = tmp_path / "subset"
    result = plan(bucket, dest)
    obj = next(obj for obj in result["objects"] if obj["path"].endswith("rgb.raw.f16.bin"))
    (bucket.root / obj["path"]).write_bytes(b"X" * obj["bytes"])
    assert g3.main(["fetch", "--root", str(dest), "--retries", "0"]) == 1
    assert "G3_SUBSET=FAIL" in capsys.readouterr().err
    assert not (dest / obj["path"]).exists()
    assert not (dest / obj["path"]).with_name("rgb.raw.f16.bin.part").exists()
