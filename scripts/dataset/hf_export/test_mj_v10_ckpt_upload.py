#!/usr/bin/env python3
"""mj_v10_ckpt_upload.py 的 CPU 离线验收（假 hf/tmux 命令，不访问真实 HF、不建真实 bucket）。

取舍：检查点用真实 MotionJEPA factory 构造（encoder/decoder 深度缩成 1 层、统计量 buffer 用固定值），
上传器的 CPU 校验子进程因此走真实的 strict load，而不是 monkeypatch 掉模型构造；
代价是每次校验约数秒，故 epochs 取 2～3、steps_per_epoch 取 7 做参数化，覆盖奇数 epoch 与 final。

运行：pytest（若可用）或 ``python test_mj_v10_ckpt_upload.py``（内置最小 runner，末行打印判定）。
环境变量 MJ_SRC 指定 MotionJEPA src（默认 /scratch/hongze/MotionJEPA/src，只读导入）。
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
SCRIPT = HERE / "mj_v10_ckpt_upload.py"
MJ_SRC = os.environ.get("MJ_SRC", "/scratch/hongze/MotionJEPA/src")
PY = sys.executable
BUCKET = "HongzeFu/test-v10-encoder-bucket"
STEPS = 7
TOKEN = "fake-token-for-offline-test"

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

_BASE = {}


def base_payload():
    """真实 factory 构造的小模型权重（只构造一次）。"""
    if _BASE:
        return _BASE
    sys.path.insert(0, MJ_SRC)
    import torch
    from omegaconf import OmegaConf
    from motion_jepa.models import factory
    cfg = OmegaConf.load(Path(MJ_SRC).parent / "configs" / "default.yaml")
    cfg.motion.encoder.depth = 1
    cfg.dit.depth = 1
    cfg = OmegaConf.to_container(cfg, resolve=False)
    original = factory.load_wan_latent_stats
    factory.load_wan_latent_stats = lambda vae_id: (torch.linspace(-1, 1, 16), torch.linspace(0.5, 2, 16))
    try:
        torch.manual_seed(0)
        enc = factory.build_encoder_from_cfg(cfg, activation_checkpoint=False)
        dec = factory.build_decoder_from_cfg(cfg)
    finally:
        factory.load_wan_latent_stats = original
    _BASE.update(cfg=cfg, enc=enc.state_dict(), dec=dec.state_dict(), torch=torch)
    return _BASE


def write_ckpt(path, run_name, *, kind, epoch, epochs, salt=0.0, override=None):
    b = base_payload()
    torch = b["torch"]
    cfg = dict(b["cfg"], run_name=run_name)

    def bump(sd):
        out = {k: v.clone() for k, v in sd.items()}
        key = next(k for k, v in out.items() if v.is_floating_point() and "latents" not in k)
        out[key] = out[key] + salt
        return out
    gs = STEPS * epoch
    common = {"epoch": epoch, "epoch_completed": True, "global_step": gs, "samples_seen_global": 1000 * epoch,
              "arch": "wan-latent-v10-st", "config": cfg, "ema_only": False, "has_live_weights": True,
              "encoder": bump(b["enc"]), "wan_decoder": bump(b["dec"]), "encoder_live": bump(b["enc"]),
              "wan_decoder_live": bump(b["dec"]),
              "optimizer": {"state": {0: {"exp_avg": torch.zeros(3)}}, "param_groups": [{"lr": 6e-4, "params": [0]}]}}
    if kind == "epoch":
        common.update(val_loss=None, val_sigreg_loss=None, val_latent_loss=None)
    else:
        common.update(checkpoint_id=f"{run_name}:step:{gs}", has_ema_weights=True, stop_reason="schedule_complete",
                      final_eval_complete={"ema": False, "live": False}, val_loss=None, exact_resume=False,
                      last_complete_evaluations={}, last_complete_eval_step=None)
    common.update(override or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(common, path)


class Env:
    def __init__(self, tmp, epochs=3, run="v10-test-run-a"):
        self.tmp = Path(tmp)
        self.epochs = epochs
        self.run = run
        self.run_dir = self.tmp / "runs" / run
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "config.yaml").write_text(f"run_name: {run}\n")
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

    def ckpt(self, n, **kw):
        write_ckpt(self.run_dir / f"checkpoint_epoch_{n}.pt", self.run, kind="epoch", epoch=n,
                   epochs=self.epochs, **kw)

    def final(self, **kw):
        write_ckpt(self.run_dir / "checkpoint_final.pt", self.run, kind="final", epoch=self.epochs,
                   epochs=self.epochs, **kw)

    def run_up(self, mode, *, bucket=BUCKET, extra=()):
        argv = [PY, str(SCRIPT), mode, "--run-dir", str(self.run_dir), "--bucket", bucket,
                "--export-root", str(self.export), "--train-tmux", "train-test", "--mj-src", MJ_SRC,
                "--epochs", str(self.epochs), "--steps-per-epoch", str(STEPS), "--hf-env", str(self.secrets),
                "--retry-sleep", "0"]
        if mode == "watch":
            argv += ["--poll-seconds", "30", "--max-rounds", "1"]
        proc = subprocess.run(argv + list(extra), env=self.env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=240)
        return proc.returncode, proc.stdout

    def marker(self, name):
        p = self.export / "markers" / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def remote_names(self):
        d = self.fake / "remote" / BUCKET.replace("/", "__")
        return sorted(p.name for p in d.iterdir()) if d.exists() else []


def uploads(env, name):
    return [e for e in env.events("upload") if e["name"] == name]


# ------------------------------------------------------------------------------------------------

def test_odd_epochs_final_and_finalize_pass(tmp_path):
    """奇数 epoch 与 final 不漏；首份训练中 verified → during_training=true；finalize PASS。"""
    e = Env(tmp_path, epochs=3)
    e.ckpt(1)
    rc, out = e.run_up("watch")
    assert rc == 0, out
    assert "motion_jepa_file=" in out and str(Path(MJ_SRC).resolve()) in out, out
    assert e.marker("checkpoint_epoch_1.pt")["status"] == "verified", out
    assert e.marker("checkpoint_epoch_1.pt")["verified_during_training"] is True
    assert e.events("create") and e.state()["buckets"][BUCKET]["private"] is True
    e.ckpt(2); e.ckpt(3); e.final()
    e.state({"alive": False})
    rc, out = e.run_up("watch")
    assert rc == 0 and "ENC_WATCH_END" in out, out
    for n in ("checkpoint_epoch_1.pt", "checkpoint_epoch_2.pt", "checkpoint_epoch_3.pt", "checkpoint_final.pt"):
        assert e.marker(n)["status"] == "verified", (n, out)
        assert len(uploads(e, n)) == 1, (n, e.events())
    assert not list((e.export / "snapshots").glob("*.pt")), "verified 后快照应删除"
    rc, out = e.run_up("finalize")
    assert rc == 0, out
    last = out.strip().splitlines()[-1]
    assert last.startswith("ENC_UPLOAD=PASS objects=4 verified=4 during_training=true first_verified_at="), last
    assert "train_end_at=" in last and "train_end_at=none" not in last
    sums = (e.export / "meta" / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 4 and sums[-1].endswith("  checkpoint_final.pt")
    assert e.remote_names() == sorted(["SHA256SUMS", "config.yaml", "checkpoint_epoch_1.pt", "checkpoint_epoch_2.pt",
                                       "checkpoint_epoch_3.pt", "checkpoint_final.pt"])
    assert TOKEN not in out


def test_half_written_not_uploaded(tmp_path):
    """半文件 torch.load 失败：不计错误、不传、不留 marker；写完后下一轮才上传。"""
    e = Env(tmp_path, epochs=2)
    e.ckpt(1)
    full = (e.run_dir / "checkpoint_epoch_1.pt").read_bytes()
    (e.run_dir / "checkpoint_epoch_1.pt").write_bytes(full[: len(full) // 2])
    rc, out = e.run_up("watch")
    assert rc == 0, out
    assert "ENC_WAIT" in out and "load_failed" in out, out
    assert not e.events("upload") and e.marker("checkpoint_epoch_1.pt") is None
    (e.run_dir / "checkpoint_epoch_1.pt").write_bytes(full)
    rc, out = e.run_up("watch")
    assert rc == 0 and e.marker("checkpoint_epoch_1.pt")["status"] == "verified", out


def test_unstable_copy_detected_in_process(tmp_path):
    """复制期间源被追加写：快照判为未稳定（返回 None），临时文件被清掉。"""
    sys.path.insert(0, str(HERE))
    import mj_v10_ckpt_upload as mod
    import ckpt_upload_common as common
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = run_dir / "checkpoint_epoch_1.pt"
    src.write_bytes(b"x" * 4096)

    class A:
        pass
    a = A()
    a.run_dir, a.export_root, a.run_name, a.bucket = str(run_dir), str(tmp_path / "exp"), "r", BUCKET
    up = mod.Uploader(a, "watch")
    up.snap_dir.mkdir(parents=True)
    real = common.copy_with_hash

    def racing_copy(s, d):
        result = real(s, d)
        with open(s, "ab") as f:
            f.write(b"more")
        return result
    common.copy_with_hash = racing_copy
    try:
        assert up.take_snapshot("checkpoint_epoch_1.pt") is None
    finally:
        common.copy_with_hash = real
    assert not list(up.snap_dir.iterdir())
    assert up.take_snapshot("checkpoint_epoch_1.pt") is not None
    snap = up.snap_dir / "checkpoint_epoch_1.pt"
    assert os.stat(snap).st_ino != os.stat(src).st_ino, "快照必须是独立复制"


def test_conflict_after_snapshot_and_after_verified(tmp_path):
    """快照后（上传失败未 verified）或已 verified 后源字节变化 → ENC_UPLOAD_CONFLICT 非零退出。"""
    e = Env(tmp_path, epochs=2)
    e.ckpt(1)
    e.state({"upload_fail": 99})
    rc, out = e.run_up("watch")
    assert rc == 0 and e.marker("checkpoint_epoch_1.pt")["status"] == "snapshot", out
    e.ckpt(1, salt=0.5)
    rc, out = e.run_up("watch")
    assert rc == 2 and "ENC_UPLOAD_CONFLICT" in out, out

    e2 = Env(tmp_path / "b", epochs=2)
    e2.ckpt(1)
    rc, out = e2.run_up("watch")
    assert rc == 0 and e2.marker("checkpoint_epoch_1.pt")["status"] == "verified", out
    e2.ckpt(1, salt=0.25)
    rc, out = e2.run_up("watch")
    assert rc == 2 and "ENC_UPLOAD_CONFLICT" in out, out


def test_wrong_identity_stops(tmp_path):
    """文件名 epoch 2 而内容 epoch 1（错身份）→ ENC_UPLOAD_IDENTITY 停止，不上传。"""
    e = Env(tmp_path, epochs=2)
    write_ckpt(e.run_dir / "checkpoint_epoch_2.pt", e.run, kind="epoch", epoch=1, epochs=2)
    rc, out = e.run_up("watch")
    assert rc == 2 and "ENC_UPLOAD_IDENTITY" in out, out
    assert not e.events("upload")


def test_marker_and_bucket_ownership_refused(tmp_path):
    """错 run/bucket 的 marker、非空无归属 bucket、公开 bucket 均拒绝，且不上传。"""
    e = Env(tmp_path / "a", epochs=2)
    e.ckpt(1)
    e.export.mkdir(parents=True)
    (e.export / "owner.json").write_text(json.dumps({"schema": "mj-v10-encoder-ckpt-upload/1", "run_name": "other",
                                                     "bucket_id": BUCKET, "kind": "encoder"}))
    rc, out = e.run_up("watch")
    assert rc == 2 and "UPLOAD_MARKER_MISMATCH" in out and not e.events("upload"), out

    e = Env(tmp_path / "b", epochs=2)
    e.ckpt(1)
    (e.export / "markers").mkdir(parents=True)
    (e.export / "markers" / "checkpoint_epoch_1.pt.json").write_text(json.dumps({
        "schema": "mj-v10-encoder-ckpt-upload/1", "run_name": e.run, "bucket_id": "HongzeFu/other-bucket",
        "kind": "encoder", "object": "checkpoint_epoch_1.pt", "status": "verified"}))
    rc, out = e.run_up("watch")
    assert rc == 2 and "UPLOAD_MARKER_MISMATCH" in out and not e.events("upload"), out

    e = Env(tmp_path / "c", epochs=2)
    e.ckpt(1)
    e.state({"buckets": {BUCKET: {"private": True}}})
    foreign = e.fake / "remote" / BUCKET.replace("/", "__")
    foreign.mkdir(parents=True)
    (foreign / "someone-else.bin").write_bytes(b"x")
    rc, out = e.run_up("watch")
    assert rc == 2 and "BUCKET_FOREIGN" in out and not e.events("upload"), out

    e = Env(tmp_path / "d", epochs=2)
    e.ckpt(1)
    e.state({"buckets": {BUCKET: {"private": False}}})
    rc, out = e.run_up("watch")
    assert rc == 2 and "BUCKET_NOT_PRIVATE" in out and not e.events("upload"), out


def test_readback_failure_not_verified(tmp_path):
    """上传成功但回读失败/字节不符：不记 verified；finalize 新一轮重试后成功。"""
    e = Env(tmp_path, epochs=1)
    e.ckpt(1)
    e.state({"download_corrupt": 2, "download_fail": 1})
    rc, out = e.run_up("watch")
    assert rc == 0, out
    m = e.marker("checkpoint_epoch_1.pt")
    assert m["status"] != "verified" and m["attempts"]["training"] == 3, (m, out)
    assert out.count("ENC_READBACK_FAIL") == 3, out
    e.final()
    e.state({"alive": False})
    rc, out = e.run_up("finalize")
    assert rc == 0, out
    assert e.marker("checkpoint_epoch_1.pt")["status"] == "verified"
    last = out.strip().splitlines()[-1]
    assert last.startswith("ENC_UPLOAD=PASS objects=2 verified=2 during_training=false"), last
    assert "训练中上传未达成" in out


def test_retry_cap_training_and_finalize(tmp_path):
    """断线：训练期每对象恰好 3 次，之后轮次不再尝试；finalize 再给 3 次，仍失败则 FAIL。"""
    e = Env(tmp_path, epochs=1)
    e.ckpt(1)
    e.state({"upload_fail": 999})
    rc, out = e.run_up("watch")
    assert rc == 0 and len(uploads(e, "checkpoint_epoch_1.pt")) == 3, out
    assert "ENC_UPLOAD_GIVEUP" in out
    rc, out = e.run_up("watch")
    assert rc == 0 and len(uploads(e, "checkpoint_epoch_1.pt")) == 3, out
    e.final()
    e.state({"alive": False})
    rc, out = e.run_up("watch")
    assert rc == 3 and "ENC_WATCH_END" in out, out
    assert len(uploads(e, "checkpoint_epoch_1.pt")) == 3
    assert len(uploads(e, "checkpoint_final.pt")) == 3
    rc, out = e.run_up("finalize")
    assert rc == 1, out
    assert len(uploads(e, "checkpoint_epoch_1.pt")) == 6 and len(uploads(e, "checkpoint_final.pt")) == 6
    assert out.strip().splitlines()[-1].startswith("ENC_UPLOAD=FAIL objects=2 verified=0"), out


def test_restart_resumes_without_duplicates(tmp_path):
    """重启：已 verified 的对象重新核源与远端但不重复上传；新对象正常上传。"""
    e = Env(tmp_path, epochs=2)
    e.ckpt(1)
    rc, out = e.run_up("watch")
    assert rc == 0, out
    e.ckpt(2)
    lists_before = len(e.events("list_files"))
    rc, out = e.run_up("watch")
    assert rc == 0, out
    assert len(uploads(e, "checkpoint_epoch_1.pt")) == 1
    assert len(uploads(e, "checkpoint_epoch_2.pt")) == 1
    assert len(e.events("list_files")) > lists_before, "重启须核远端"
    # 远端对象丢失 → 重新排队上传
    (e.fake / "remote" / BUCKET.replace("/", "__") / "checkpoint_epoch_1.pt").unlink()
    rc, out = e.run_up("watch")
    assert rc == 0 and "ENC_REMOTE_MISSING" in out and len(uploads(e, "checkpoint_epoch_1.pt")) == 2, out


def test_usage_rejects_fast_poll_and_relative_paths(tmp_path):
    e = Env(tmp_path, epochs=1)
    proc = subprocess.run([PY, str(SCRIPT), "watch", "--run-dir", str(e.run_dir), "--bucket", BUCKET,
                           "--export-root", "rel/export", "--train-tmux", "t", "--mj-src", MJ_SRC,
                           "--poll-seconds", "5"], env=e.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    assert proc.returncode == 64 and "poll-seconds" in proc.stdout and "export-root" in proc.stdout, proc.stdout


# ------------------------------------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = Path(tempfile.mkdtemp(prefix=f"mjv10up-{name}-"))
        try:
            fn(tmp)
            print(f"PASS {name}", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            failed.append(name)
            print(f"FAIL {name} (保留 {tmp})\n{traceback.format_exc()}", flush=True)
    verdict = "PASS" if not failed else "FAIL"
    print(f"ENC_UPLOAD_DRYRUN={verdict} cases={len(tests)} failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_main())
