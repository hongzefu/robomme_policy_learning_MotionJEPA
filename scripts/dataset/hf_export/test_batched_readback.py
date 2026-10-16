"""batched_readback.py 的单测：合成小 stage + 本地拷贝代替 hf，不触网。

覆盖计划 A1b 验收行：每批比对后回读目录被删空、汇总输出 SHA256_PRE_POST_DIFF=0；篡改一个字节时
非 0 退出；READBACK_PEAK_BYTES=<n> 且 n ≤ 上限；人为让一批超上限时非 0 退出；上限小于最大单对象
时在计划阶段就报错；中断后续跑跳过已完成批。

运行：uv run --no-sync python -m pytest scripts/dataset/hf_export/test_batched_readback.py -q
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batched_readback as br  # noqa: E402

PLAN_PY = HERE / "plan_upload_batches.py"
BR_PY = HERE / "batched_readback.py"

MAX = 60_000
SLACK = 20_000  # 目录项开销余量（ext4 每目录 4096 B）


def _hash_tree(root: Path, out: Path) -> None:
    """与驱动 hash_tree 完全同一条管道生成上传前清单。"""
    subprocess.run(
        ["bash", "-c",
         f"cd '{root}' && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\\n' | sort "
         f"| xargs -n 16 sha256sum | sort -k 2 > '{out}'"],
        check=True,
    )


def _make_stage(tmp: Path) -> Path:
    stage = tmp / "export" / "stage"
    (stage / "framesamp-8x8" / "meta").mkdir(parents=True)
    (stage / "source" / "data_tars").mkdir(parents=True)
    (stage / "checksums").mkdir(parents=True)
    (stage / "meta").mkdir(parents=True)
    # 8 个 9000 B 的大件 + 若干小件；两个 tar 带成员清单
    for i in range(8):
        (stage / "framesamp-8x8" / f"part-{i:03d}.bin").write_bytes(bytes([i]) * 9000)
    (stage / "framesamp-8x8" / "meta" / "store_meta.json").write_text('{"a": 1}\n')
    (stage / "meta" / "episode_manifest.json").write_text('{"episodes": []}\n')
    member_lines = []
    for t in range(2):
        buf = tmp / f"tar{t}"
        with tarfile.open(stage / "source" / "data_tars" / f"data-{t:05d}.tar", "w") as tf:
            for k in range(3):
                data = f"pkl-{t}-{k}".encode() * 50
                ti = tarfile.TarInfo(f"data/{t * 3 + k}.pkl")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
                import hashlib
                member_lines.append(f"{hashlib.sha256(data).hexdigest()}  data/{t * 3 + k}.pkl")
        del buf
    (stage / "checksums" / "sha256-source-files.txt").write_text("\n".join(member_lines) + "\n")
    (stage / "README.md").write_text("# synthetic\n")
    _hash_tree(stage, tmp / "pre.txt")
    shutil.move(str(tmp / "pre.txt"), str(stage / "SHA256SUMS.pre.txt"))
    return stage


def _plans(tmp: Path, stage: Path, max_bytes: int = MAX, slack: int = SLACK) -> tuple[Path, Path, int]:
    up = tmp / "export" / "logs" / "upload_plan.json"
    up.parent.mkdir(parents=True, exist_ok=True)
    # 上传装箱上限 20000 B：每批最多 2 个大件，回读侧再把相邻批合并到预算 40000 B 以内
    subprocess.run([sys.executable, str(PLAN_PY), "--stage", str(stage), "--out", str(up),
                    "--max-bytes", "20000"], check=True, capture_output=True)
    rbp = tmp / "export" / "logs" / "readback_plan.json"
    r = subprocess.run([sys.executable, str(BR_PY), "plan", "--upload-plan", str(up),
                        "--stage", str(stage), "--out", str(rbp),
                        "--max-bytes", str(max_bytes), "--slack", str(slack)],
                       capture_output=True, text=True)
    return up, rbp, r.returncode


def _run_cli(tmp: Path, stage: Path, rbp: Path, bucket: Path, max_bytes: int = MAX):
    exp = tmp / "export"
    cmd = [sys.executable, str(BR_PY), "run", "--readback-plan", str(rbp),
           "--pre", str(stage / "SHA256SUMS.pre.txt"), "--verify", str(exp / "verify"),
           "--state", str(exp / "logs" / "readback"), "--post-out", str(exp / "logs" / "post.txt"),
           "--max-bytes", str(max_bytes),
           "--members-checksums", str(stage / "checksums" / "sha256-source-files.txt"),
           "--sample-tars-per-batch", "2", "--nproc", "4",
           "--source", "local", "--local-root", str(bucket), "--retries", "3", "--retry-sleep", "0"]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture()
def env(tmp_path: Path):
    stage = _make_stage(tmp_path)
    _, rbp, rc = _plans(tmp_path, stage)
    assert rc == 0
    bucket = tmp_path / "fake-bucket"
    shutil.copytree(stage, bucket)  # 「上传」= 原样拷一份，模拟 bucket 内容
    return tmp_path, stage, rbp, bucket


def test_plan_merges_adjacent_batches_within_budget(env):
    tmp, stage, rbp, _ = env
    rp = json.loads(rbp.read_text())
    up = json.loads((tmp / "export" / "logs" / "upload_plan.json").read_text())
    assert len(rp["batches"]) > 1
    assert len(rp["batches"]) < len(up["batches"]) + len(up["root_files"])  # 确实合并了相邻批
    assert all(b["bytes"] <= MAX - SLACK for b in rp["batches"])
    assert rp["total_files"] == up["total_files"]


def test_happy_path_each_batch_cleared_and_diff_zero(env):
    tmp, stage, rbp, bucket = env
    r = _run_cli(tmp, stage, rbp, bucket)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    nb = len(json.loads(rbp.read_text())["batches"])
    ok_lines = [l for l in out.splitlines() if l.startswith("READBACK_BATCH_OK ")]
    assert len(ok_lines) == nb and nb > 1
    assert all(l.endswith("left_files=0") for l in ok_lines)
    assert sum(1 for l in out.splitlines() if l == "SHA256_PRE_POST_DIFF=0") == 1
    m = re.search(r"^READBACK_PEAK_BYTES=(\d+)$", out, re.M)
    assert m and 0 < int(m.group(1)) <= MAX
    assert list((tmp / "export" / "verify").iterdir()) == []
    post = (tmp / "export" / "logs" / "post.txt").read_text()
    assert post == (stage / "SHA256SUMS.pre.txt").read_text()
    assert "MEMBERS_SPOT=FAIL" not in out


def test_verify_dir_empty_before_every_batch(env):
    """进程内调用：钩子在每批下载前观察回读目录，必须为空（即上一批比对后已删空）。"""
    tmp, stage, rbp, bucket = env
    exp = tmp / "export"
    seen = []

    def hook(batch, verify):
        seen.append((batch["id"], br.list_files(verify)))

    args = Namespace(readback_plan=rbp, pre=stage / "SHA256SUMS.pre.txt", verify=exp / "verify",
                     state=exp / "logs" / "readback", post_out=exp / "logs" / "post.txt",
                     max_bytes=MAX, members_checksums=stage / "checksums" / "sha256-source-files.txt",
                     sample_tars_per_batch=1, seed=1, nproc=2, source="local", bucket=None,
                     local_root=bucket, hf_cmd="hf", retries=1, retry_sleep=0)
    assert br.run_batches(args, before_batch=hook) == 0
    assert len(seen) == len(json.loads(rbp.read_text())["batches"])
    assert all(files == [] for _, files in seen)


def test_tampered_byte_fails(env):
    tmp, stage, rbp, bucket = env
    victim = bucket / "framesamp-8x8" / "part-005.bin"
    data = bytearray(victim.read_bytes())
    data[100] ^= 0x01  # 同尺寸翻一个比特
    victim.write_bytes(bytes(data))
    r = _run_cli(tmp, stage, rbp, bucket)
    assert r.returncode != 0
    assert "SHA256_PRE_POST_DIFF=0" not in r.stdout
    assert "part-005.bin" in r.stdout


def test_tampered_tar_member_list_fails(env):
    tmp, stage, rbp, bucket = env
    cks = stage / "checksums" / "sha256-source-files.txt"
    lines = cks.read_text().splitlines()
    lines[0] = "0" * 64 + lines[0][64:]
    # 只改成员清单（本地 stage 侧这一份），tar 本体不动：sha256 层看不见，成员层必须拦住
    alt = tmp / "members-bad.txt"
    alt.write_text("\n".join(lines) + "\n")
    exp = tmp / "export"
    args = Namespace(readback_plan=rbp, pre=stage / "SHA256SUMS.pre.txt", verify=exp / "verify",
                     state=exp / "logs" / "readback", post_out=exp / "logs" / "post.txt",
                     max_bytes=MAX, members_checksums=alt, sample_tars_per_batch=5, seed=1, nproc=2,
                     source="local", bucket=None, local_root=bucket, hf_cmd="hf", retries=1,
                     retry_sleep=0)
    assert br.run_batches(args) == 1


def test_batch_over_limit_fails(env):
    """人为让一批超上限：bucket 里某个文件比计划大得多 → du 超上限 → 非 0 退出，且该批已删。"""
    tmp, stage, rbp, bucket = env
    (bucket / "framesamp-8x8" / "part-000.bin").write_bytes(b"x" * (MAX + 1))
    r = _run_cli(tmp, stage, rbp, bucket)
    assert r.returncode != 0
    m = re.search(r"^READBACK_PEAK_BYTES=(\d+)$", r.stdout, re.M)
    assert m and int(m.group(1)) > MAX
    assert "SHA256_PRE_POST_DIFF=0" not in r.stdout
    assert list((tmp / "export" / "verify").iterdir()) == []


def test_limit_below_largest_object_rejected_at_plan(tmp_path):
    stage = _make_stage(tmp_path)
    _, _, rc = _plans(tmp_path, stage, max_bytes=8000, slack=0)  # 最大单对象 9000 B
    assert rc != 0


def test_resume_skips_completed_batches(env):
    tmp, stage, rbp, bucket = env
    rp = json.loads(rbp.read_text())
    last_batch_file = rp["batches"][-1]["files"][0]
    good = (bucket / last_batch_file).read_bytes()
    (bucket / last_batch_file).write_bytes(b"\x00" + good[1:] if good[:1] != b"\x00" else b"\x01" + good[1:])
    r1 = _run_cli(tmp, stage, rbp, bucket)
    assert r1.returncode != 0
    done_before = sorted(p.name for p in (tmp / "export" / "logs" / "readback").glob("*.done.json"))
    assert len(done_before) == len(rp["batches"]) - 1
    (bucket / last_batch_file).write_bytes(good)
    r2 = _run_cli(tmp, stage, rbp, bucket)
    assert r2.returncode == 0, r2.stdout
    assert r2.stdout.count("已完成") >= len(rp["batches"]) - 1
    assert sum(1 for l in r2.stdout.splitlines() if l == "SHA256_PRE_POST_DIFF=0") == 1


def test_guard_refuses_non_verify_dir(tmp_path):
    with pytest.raises(SystemExit):
        br.guard_verify_dir(tmp_path / "stage")


def test_downloader_retries_first_plus_three(tmp_path):
    """--retries 3 = 首试 + 重试 3 次（计划口径 27）；仍失败时点名对象并 SystemExit。"""
    args = Namespace(source="local", bucket=None, local_root=tmp_path, hf_cmd="hf", retries=3, retry_sleep=0)
    dl = br.Downloader(args)
    n = []

    def boom():
        n.append(1)
        raise OSError("模拟失败")

    with pytest.raises(SystemExit) as ei:
        dl._retry("对象X", boom)
    assert len(n) == 4 and "重试 3 次仍失败" in str(ei.value)
