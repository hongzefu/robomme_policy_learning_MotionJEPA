#!/usr/bin/env python3
"""mixed8_pair4_ckpt_upload.py 的 CPU 离线验收（假 hf/tmux 命令，不访问真实 HF、不建真实 bucket）。

伪造 Orbax 目录树（``_CHECKPOINT_METADATA``、params/assets 若干文件、``*.orbax-checkpoint-tmp-*`` 临时目录）
与 final-record 三件套；用小 ``--steps/--save-interval`` 参数化预期 step 集合。

运行：pytest（若可用）或 ``python test_mixed8_pair4_ckpt_upload.py``（内置最小 runner，末行打印判定）。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "mixed8_pair4_ckpt_upload.py"
PY = sys.executable
BUCKET = "HongzeFu/test-pair4-vla-bucket"
TOKEN = "fake-token-for-offline-test"
UUID = "11111111-2222-3333-4444-555555555555"
HEAD = "a" * 40
STEPS, INTERVAL = 13, 3          # 预期 step：3,6,9,12（12 = steps-1）
EXPECTED = [3, 6, 9, 12]

FAKE_UVX = r'''
import json, os, pathlib, shutil, sys
root = pathlib.Path(os.environ["FAKE_ROOT"])
sp = root / "state.json"
st = json.loads(sp.read_text())
def ev(kind, **kw):
    with (root / "events.jsonl").open("a") as f:
        f.write(json.dumps(dict(kind=kind, **kw)) + "\n")
def save():
    sp.write_text(json.dumps(st))
def take(key):
    if st.get(key, 0) > 0:
        st[key] -= 1; save(); return True
    return False
args = sys.argv[1:]
args = args[args.index("hf") + 1:]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "上传子进程必须屏蔽 GPU"
assert "HF_HUB_OFFLINE" not in os.environ, "上传子进程应解除 HF_HUB_OFFLINE"
assert os.environ.get("HF_TOKEN") == "fake-token-for-offline-test", "必须鉴权"
assert not any("fake-token" in a for a in sys.argv), "token 不得进 argv"
remote = root / "remote"
def bdir(bid): return remote / bid.replace("/", "__")
def local(v):
    pre = "hf://buckets/"
    if v.startswith(pre):
        rest = v[len(pre):]
        ns, name, path = rest.split("/", 2)
        return bdir(ns + "/" + name) / path, path
    return pathlib.Path(v), None
if args[:2] == ["auth", "whoami"]:
    print("user=HongzeFu")
elif args[:2] == ["buckets", "list"]:
    target = args[2]
    if take("list_fail"):
        ev("list_fail"); sys.exit(1)
    if "/" not in target:
        ev("list_buckets")
        out = []
        for bid, meta in st["buckets"].items():
            files = [p for p in bdir(bid).rglob("*") if p.is_file()] if bdir(bid).exists() else []
            out.append(dict(id=bid, private=meta["private"], total_files=len(files),
                            size=sum(p.stat().st_size for p in files), created_at="x"))
        print(json.dumps(out))
    else:
        ev("list_files")
        d = bdir(target)
        print(json.dumps([dict(type="file", path=str(p.relative_to(d)), size=p.stat().st_size, xet_hash="x")
                          for p in sorted(d.rglob("*")) if p.is_file()]))
elif args[:2] == ["buckets", "create"]:
    assert "--private" in args, "必须创建私有 bucket"
    ev("create", bucket=args[2])
    st["buckets"][args[2]] = dict(private=True); save(); bdir(args[2]).mkdir(parents=True, exist_ok=True)
elif args[:2] == ["buckets", "cp"]:
    src, src_rel = local(args[2]); dst, dst_rel = local(args[3])
    if src_rel is not None:
        ev("download", name=src_rel)
        if take("download_fail"): sys.exit(1)
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
        if take("download_corrupt"): dst.write_bytes(b"corrupt")
    else:
        ev("upload", name=dst_rel)
        if take("upload_fail"): sys.exit(1)
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
else:
    raise AssertionError(args)
'''

FAKE_TMUX = r'''
import json, os, pathlib, sys
st = json.loads((pathlib.Path(os.environ["FAKE_ROOT"]) / "state.json").read_text())
assert sys.argv[1:3] == ["has-session", "-t"] and sys.argv[3].startswith("=")
sys.exit(0 if st.get("alive") else 1)
'''


class Env:
    def __init__(self, tmp, run="pair4-test-run-a"):
        self.tmp = Path(tmp)
        self.run = run
        self.run_dir = self.tmp / "train-runs" / "mme_vla_suite_b128_80k" / run
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "config_record.json").write_text('{"cfg": 1}\n')
        (self.run_dir / "history_config.resolved.yaml").write_text("budget: 4\n")
        self.fr = self.tmp / "final-record"
        self.export = self.tmp / "export"
        self.fake = self.tmp / "fake"
        self.fake.mkdir()
        bindir = self.tmp / "bin"
        bindir.mkdir()
        for name, body in (("uvx", FAKE_UVX), ("tmux", FAKE_TMUX)):
            f = bindir / name
            f.write_text(f"#!{PY}\n{body}")
            f.chmod(0o755)
        self.secrets = self.tmp / "hf.env"
        self.secrets.write_text(f"HF_TOKEN={TOKEN}\n")
        self.meta_in = self.tmp / "meta-in"
        self.meta_in.mkdir()
        (self.meta_in / "history.yaml").write_text("pairs: 4\n")
        (self.meta_in / "norm_stats.json").write_text('{"norm": [1, 2]}\n')
        (self.meta_in / "motion_meta.json").write_text('{"dim": 2048, "rows": 10}\n')
        self.state({"buckets": {}, "alive": True})
        self.env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", FAKE_ROOT=str(self.fake))
        self.env.pop("HF_TOKEN", None)
        self.env["HF_HUB_OFFLINE"] = "1"

    def state(self, update=None):
        p = self.fake / "state.json"
        st = json.loads(p.read_text()) if p.exists() else {}
        if update:
            st.update(update)
            p.write_text(json.dumps(st))
        return st

    def events(self, kind=None):
        p = self.fake / "events.jsonl"
        evs = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        return [e for e in evs if kind is None or e["kind"] == kind]

    def start_json(self, **over):
        self.fr.mkdir(exist_ok=True)
        body = {"record_version": 1, "run_uuid": UUID, "run_name": self.run, "head": HEAD,
                "checkpoint_dir": str(self.run_dir), "num_train_steps": STEPS, "save_interval": INTERVAL}
        body.update(over)
        (self.fr / "start.json").write_text(json.dumps(body))

    def wait_done(self, **over):
        body = {"run_uuid": UUID, "head": HEAD, "wait_until_finished": True, "completed_at": "x"}
        body.update(over)
        (self.fr / "checkpoint_wait_done.json").write_text(json.dumps(body))

    def final_json(self, **over):
        body = {"run_uuid": UUID, "run_name": self.run, "head": HEAD, "checkpoint_dir": str(self.run_dir),
                "state_step": STEPS, "loop_step": STEPS - 1, "checkpoint_step": STEPS - 1}
        body.update(over)
        (self.fr / "final.json").write_text(json.dumps(body))

    def step(self, n, *, commit=True, salt=b""):
        d = self.run_dir / str(n)
        (d / "params" / "ocdbt.process_0" / "d").mkdir(parents=True, exist_ok=True)
        (d / "assets" / "robomme").mkdir(parents=True, exist_ok=True)
        (d / "params" / "_METADATA").write_text(f'{{"step": {n}}}')
        (d / "params" / "manifest.ocdbt").write_bytes(b"m" * 64 + salt)
        (d / "params" / "ocdbt.process_0" / "d" / f"blob{n}").write_bytes(os.urandom(4096) + salt)
        (d / "assets" / "robomme" / "norm_stats.json").write_text('{"norm": [1, 2]}\n')
        meta = {"item_handlers": {}, "init_timestamp_nsecs": 1000 + n}
        if commit:
            meta["commit_timestamp_nsecs"] = 2000 + n
        (d / "_CHECKPOINT_METADATA").write_text(json.dumps(meta))
        return d

    def tmp_dir(self, n):
        d = self.run_dir / f"{n}.orbax-checkpoint-tmp-1792549501"
        (d / "params").mkdir(parents=True)
        (d / "params" / "partial").write_bytes(b"p" * 100)
        return d

    def run_up(self, mode, *, bucket=BUCKET, extra=()):
        argv = [PY, str(SCRIPT), mode, "--run-dir", str(self.run_dir), "--final-record-dir", str(self.fr),
                "--bucket", bucket, "--export-root", str(self.export), "--train-tmux", "train-test",
                "--steps", str(STEPS), "--save-interval", str(INTERVAL), "--hf-env", str(self.secrets),
                "--retry-sleep", "0"]
        if mode == "watch":
            argv += ["--poll-seconds", "30", "--max-rounds", "1"]
        else:
            argv += ["--history-yaml", str(self.meta_in / "history.yaml"),
                     "--norm-stats", str(self.meta_in / "norm_stats.json"),
                     "--motion-store-meta", str(self.meta_in / "motion_meta.json"),
                     "--encoder-sha256", "b" * 64]
        proc = subprocess.run(argv + list(extra), env=self.env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=240)
        return proc.returncode, proc.stdout

    def marker(self, step):
        p = self.export / "markers" / f"step-{step}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def remote_dir(self):
        return self.fake / "remote" / BUCKET.replace("/", "__")

    def remote_paths(self):
        d = self.remote_dir()
        return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.exists() else []


def uploads(env, prefix):
    return [e for e in env.events("upload") if e["name"].startswith(prefix)]


def assert_no_token(env, out):
    assert TOKEN not in out
    for p in env.export.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".txt", "") and p.stat().st_size < 1 << 20:
            assert TOKEN not in p.read_text(errors="ignore"), p


# ------------------------------------------------------------------------------------------------

def test_full_cycle_expected_steps_and_finalize(tmp_path):
    """预期 4 个 step 不漏、tmp 目录不传、首份训练中 verified、finalize PASS 且远端对象集完整。"""
    e = Env(tmp_path)
    e.start_json()
    e.step(3)
    e.tmp_dir(6)
    rc, out = e.run_up("watch")
    assert rc == 0, out
    assert "VLA_SKIP_DIR" in out and "orbax_tmp" in out, out
    m = e.marker(3)
    assert m["status"] == "verified" and m["verified_during_training"] is True and m["run_uuid"] == UUID, out
    assert m["file_count"] == 5 and len(m["digest"]) == 64
    assert not [x for x in e.events("upload") if "orbax-checkpoint-tmp" in x["name"]]
    assert e.state()["buckets"][BUCKET]["private"] is True
    assert "3/params/ocdbt.process_0/d/blob3" in e.remote_paths()
    assert "_upload_manifests/3.json" in e.remote_paths()
    for n in (6, 9, 12):
        e.step(n)
    e.wait_done()
    e.final_json()
    e.state({"alive": False})
    rc, out = e.run_up("watch")
    assert rc == 0 and "VLA_WATCH_END verified=4 pending=0" in out.replace(" verified=4", " verified=4"), out
    for n in EXPECTED:
        assert e.marker(n)["status"] == "verified", (n, out)
        assert len(uploads(e, f"{n}/")) == 5, (n, uploads(e, f"{n}/"))
    rc, out = e.run_up("finalize")
    assert rc == 0, out
    last = out.strip().splitlines()[-1]
    assert last.startswith("VLA_UPLOAD=PASS objects=4 verified=4 during_training=true first_verified_at="), last
    assert "train_end_at=none" not in last
    remote = e.remote_paths()
    for want in ("SHA256SUMS", "_run-meta/final-record/start.json", "_run-meta/final-record/final.json",
                 "_run-meta/final-record/checkpoint_wait_done.json", "_run-meta/norm_stats.json",
                 "_run-meta/history/history.yaml", "_run-meta/motion_store/motion_meta.json",
                 "_run-meta/source_summary.json", "_run-meta/run-root/config_record.json"):
        assert want in remote, (want, remote)
    sums = (e.export / "meta" / "SHA256SUMS").read_text().splitlines()
    covered = {line.split("  ", 1)[1] for line in sums}
    assert covered == set(remote) - {"SHA256SUMS"}, sorted(set(remote) ^ covered)
    summary = json.loads((e.remote_dir() / "_run-meta" / "source_summary.json").read_text())
    assert summary["run_uuid"] == UUID and summary["encoder_ckpt_sha256"] == "b" * 64
    assert summary["expected_steps"] == EXPECTED
    assert_no_token(e, out)
    # 重跑 finalize 幂等：不重复上传
    n_up = len(e.events("upload"))
    rc, out = e.run_up("finalize")
    assert rc == 0 and len(e.events("upload")) == n_up, out


def test_expected_set_production_parameters():
    sys.path.insert(0, str(HERE))
    import mixed8_pair4_ckpt_upload as mod
    steps = mod.expected_steps(80000, 5000)
    assert steps == list(range(5000, 80000, 5000)) + [79999] and len(steps) == 16, steps
    assert mod.expected_steps(13, 3) == EXPECTED


def test_missing_commit_timestamp_not_uploaded(tmp_path):
    """缺/无效 commit_timestamp_nsecs：不传、不留 marker；补齐后下一轮上传。"""
    e = Env(tmp_path)
    e.start_json()
    d = e.step(3, commit=False)
    rc, out = e.run_up("watch")
    assert rc == 0 and "commit_timestamp_invalid" in out and e.marker(3) is None and not e.events("upload"), out
    (d / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": True}))
    rc, out = e.run_up("watch")
    assert rc == 0 and e.marker(3) is None and not e.events("upload"), out
    (d / "_CHECKPOINT_METADATA").write_text(json.dumps({"init_timestamp_nsecs": 1, "commit_timestamp_nsecs": 5}))
    rc, out = e.run_up("watch")
    assert rc == 0 and e.marker(3)["status"] == "verified", out


def test_start_json_required_and_identity(tmp_path):
    """无 start.json 不传；start.json 的 checkpoint_dir 不是本 run → 身份错误停。"""
    e = Env(tmp_path / "a")
    e.step(3)
    rc, out = e.run_up("watch")
    assert rc == 0 and "start_json_missing" in out and not e.events("upload"), out
    e = Env(tmp_path / "b")
    e.start_json(checkpoint_dir="/elsewhere/run")
    e.step(3)
    rc, out = e.run_up("watch")
    assert rc == 2 and "VLA_UPLOAD_IDENTITY" in out and not e.events("upload"), out


def test_last_step_needs_wait_done(tmp_path):
    """最后一份（steps-1）缺 wait_done / wait_until_finished=false 不 verified；补齐后才传；错 uuid 停。"""
    e = Env(tmp_path / "a")
    e.start_json()
    e.step(12)
    rc, out = e.run_up("watch")
    assert rc == 0 and "wait_done_missing" in out and e.marker(12) is None and not e.events("upload"), out
    e.wait_done(wait_until_finished=False)
    rc, out = e.run_up("watch")
    assert rc == 0 and "wait_done_not_finished" in out and e.marker(12) is None, out
    e.wait_done()
    rc, out = e.run_up("watch")
    assert rc == 0 and e.marker(12)["status"] == "verified", out
    e = Env(tmp_path / "b")
    e.start_json()
    e.step(12)
    e.wait_done(run_uuid="other-uuid")
    rc, out = e.run_up("watch")
    assert rc == 2 and "VLA_UPLOAD_IDENTITY" in out and not e.events("upload"), out


def test_unstable_snapshot_detected_in_process(tmp_path):
    """两遍枚举之间有新成员写入：判未稳定（不计错误）；稳定后才形成快照。"""
    sys.path.insert(0, str(HERE))
    import mixed8_pair4_ckpt_upload as mod
    e = Env(tmp_path)
    d = e.step(3)
    real = mod.enumerate_members
    calls = {"n": 0}

    def racing(step_dir, with_hash):
        out = real(step_dir, with_hash)
        calls["n"] += 1
        if calls["n"] == 1:
            (Path(step_dir) / "params" / "late_member").write_bytes(b"x")
        return out
    mod.enumerate_members = racing
    try:
        snap, reason = mod.snapshot_step(d)
    finally:
        mod.enumerate_members = real
    assert snap is None and reason == "member_set_changed", reason
    snap, reason = mod.snapshot_step(d)
    assert snap is not None and "params/late_member" in snap["members"]


def test_conflict_after_snapshot_and_after_verified(tmp_path):
    """快照后（上传失败未 verified）成员变化、已 verified 后字节变化 → VLA_UPLOAD_CONFLICT 非零退出。"""
    e = Env(tmp_path / "a")
    e.start_json()
    d = e.step(3)
    e.state({"upload_fail": 99})
    rc, out = e.run_up("watch")
    assert rc == 0 and e.marker(3)["status"] == "snapshot", out
    (d / "params" / "extra_member").write_bytes(b"x")
    rc, out = e.run_up("watch")
    assert rc == 2 and "VLA_UPLOAD_CONFLICT" in out, out

    e = Env(tmp_path / "b")
    e.start_json()
    d = e.step(3)
    rc, out = e.run_up("watch")
    assert rc == 0 and e.marker(3)["status"] == "verified", out
    blob = d / "params" / "ocdbt.process_0" / "d" / "blob3"
    data = bytearray(blob.read_bytes())
    data[0] ^= 0xFF
    blob.write_bytes(bytes(data))
    rc, out = e.run_up("watch")
    assert rc == 2 and "VLA_UPLOAD_CONFLICT" in out, out


def test_wrong_uuid_and_bucket_markers_refused(tmp_path):
    """错 run_uuid / 错 bucket 的 marker、非空无归属 bucket、公开 bucket 均拒绝，且不上传。"""
    e = Env(tmp_path / "a")
    e.start_json()
    e.step(3)
    rc, out = e.run_up("watch")
    assert rc == 0, out
    e.start_json(run_uuid="99999999-0000-0000-0000-000000000000")
    e.step(6)
    n_up = len(e.events("upload"))
    rc, out = e.run_up("watch")
    assert rc == 2 and "UPLOAD_MARKER_MISMATCH" in out and len(e.events("upload")) == n_up, out

    e = Env(tmp_path / "b")
    e.start_json()
    e.step(3)
    (e.export / "markers").mkdir(parents=True)
    (e.export / "markers" / "step-3.json").write_text(json.dumps({
        "schema": "mixed8-pair4-vla-ckpt-upload/1", "run_name": e.run, "bucket_id": "HongzeFu/other-bucket",
        "kind": "vla", "object": "step-3", "status": "verified", "run_uuid": UUID}))
    rc, out = e.run_up("watch")
    assert rc == 2 and "UPLOAD_MARKER_MISMATCH" in out and not e.events("upload"), out

    e = Env(tmp_path / "c")
    e.start_json()
    e.step(3)
    e.state({"buckets": {BUCKET: {"private": True}}})
    e.remote_dir().mkdir(parents=True)
    (e.remote_dir() / "someone-else.bin").write_bytes(b"x")
    rc, out = e.run_up("watch")
    assert rc == 2 and "BUCKET_FOREIGN" in out and not e.events("upload"), out

    e = Env(tmp_path / "d")
    e.start_json()
    e.step(3)
    e.state({"buckets": {BUCKET: {"private": False}}})
    rc, out = e.run_up("watch")
    assert rc == 2 and "BUCKET_NOT_PRIVATE" in out and not e.events("upload"), out


def test_readback_failure_not_verified(tmp_path):
    """上传成功但回读失败/字节不符：不记 verified；finalize 新一轮重试后成功，during_training=false。"""
    e = Env(tmp_path)
    e.start_json()
    e.step(3)
    e.state({"download_fail": 2, "download_corrupt": 1})  # 两次断线＋一次字节不符，每次尝试各失败一次
    rc, out = e.run_up("watch")
    assert rc == 0, out
    m = e.marker(3)
    assert m["status"] != "verified" and m["attempts"]["training"] == 3, (m, out)
    assert out.count("VLA_READBACK_FAIL") == 3, out
    for n in (6, 9, 12):
        e.step(n)
    e.wait_done()
    e.final_json()
    e.state({"alive": False})
    rc, out = e.run_up("finalize")
    assert rc == 0, out
    assert e.marker(3)["status"] == "verified"
    last = out.strip().splitlines()[-1]
    assert last.startswith("VLA_UPLOAD=PASS objects=4 verified=4 during_training=false"), last
    assert "训练中上传未达成" in out


def test_retry_cap_training_and_finalize(tmp_path):
    """断线：训练期每 step 恰好 3 次尝试，之后轮次不再试；finalize 再给 3 次，仍失败则 FAIL。"""
    e = Env(tmp_path)
    e.start_json()
    e.step(3)
    e.state({"upload_fail": 9999})
    rc, out = e.run_up("watch")
    assert rc == 0 and len(uploads(e, "3/")) == 3 and "VLA_UPLOAD_GIVEUP" in out, out
    rc, out = e.run_up("watch")
    assert rc == 0 and len(uploads(e, "3/")) == 3, out
    for n in (6, 9, 12):
        e.step(n)
    e.wait_done()
    e.final_json()
    e.state({"alive": False})
    rc, out = e.run_up("watch")
    assert rc == 3 and "VLA_WATCH_END" in out, out
    assert len(uploads(e, "3/")) == 3 and len(uploads(e, "12/")) == 3
    rc, out = e.run_up("finalize")
    assert rc == 1, out
    assert len(uploads(e, "3/")) == 6 and len(uploads(e, "12/")) == 6
    assert out.strip().splitlines()[-1].startswith("VLA_UPLOAD=FAIL objects=4 verified=0"), out


def test_restart_resumes_without_duplicates(tmp_path):
    """重启：已 verified 的 step 核源与远端但不重复上传；部分成员已传的续传不重传；远端丢失重排队。"""
    e = Env(tmp_path)
    e.start_json()
    e.step(3)
    rc, out = e.run_up("watch")
    assert rc == 0, out
    e.step(6)
    e.state({"upload_fail": 0})
    lists_before = len(e.events("list_files"))
    rc, out = e.run_up("watch")
    assert rc == 0, out
    assert len(uploads(e, "3/")) == 5 and len(uploads(e, "6/")) == 5
    assert len(e.events("list_files")) > lists_before, "重启须核远端"
    (e.remote_dir() / "3" / "params" / "manifest.ocdbt").unlink()
    rc, out = e.run_up("watch")
    assert rc == 0 and "VLA_REMOTE_MISSING" in out, out
    assert len(uploads(e, "3/params/manifest.ocdbt")) == 2 and len(uploads(e, "3/")) == 6, out
    assert e.marker(3)["status"] == "verified"


def test_lock_and_token_and_usage(tmp_path):
    e = Env(tmp_path)
    proc = subprocess.run([PY, str(SCRIPT), "watch", "--run-dir", str(e.run_dir), "--final-record-dir", "rel/fr",
                           "--bucket", BUCKET, "--export-root", "rel/export", "--train-tmux", "t",
                           "--poll-seconds", "5"], env=e.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    assert proc.returncode == 64 and "poll-seconds" in proc.stdout and "export-root" in proc.stdout, proc.stdout
    e.start_json()
    e.step(3)
    rc, out = e.run_up("watch")
    assert rc == 0
    assert_no_token(e, out)
    for p in (e.export / "state.json", e.export / "owner.json", e.export / "markers" / "step-3.json"):
        assert TOKEN not in p.read_text()


def test_finalize_refuses_while_training(tmp_path):
    e = Env(tmp_path)
    e.start_json()
    rc, out = e.run_up("finalize")
    assert rc == 2 and "TRAIN_STILL_RUNNING" in out, out


def test_scan_mode_read_only(tmp_path):
    e = Env(tmp_path)
    e.step(3)
    e.step(6, commit=False)
    e.tmp_dir(9)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    proc = subprocess.run([PY, str(SCRIPT), "scan", "--run-dir", str(e.run_dir), "--steps", str(STEPS),
                           "--save-interval", str(INTERVAL)], env=e.env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    assert proc.returncode == 0, out
    assert "VLA_SCAN_STEP" in out and "step=3 committed=true stable=true" in out, out
    assert "step=6 committed=false" in out and "not_final_step_dir" in out, out
    assert out.count("VLA_SCAN_MEMBER step=3") == 5, out
    assert out.strip().splitlines()[-1] == "VLA_SCAN=DONE steps=1 missing=3", out
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before, "scan 不得写文件"
    assert not e.events()


# ------------------------------------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = Path(tempfile.mkdtemp(prefix=f"pair4up-{name}-"))
        try:
            if fn.__code__.co_argcount:
                fn(tmp)
            else:
                fn()
            print(f"PASS {name}", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            failed.append(name)
            print(f"FAIL {name} (保留 {tmp})\n{traceback.format_exc()}", flush=True)
    verdict = "PASS" if not failed else "FAIL"
    print(f"VLA_UPLOAD_DRYRUN={verdict} cases={len(tests)} failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_main())
