"""四个 mixed8-200 上传驱动的 SMOKE=1 离线 dry-run：合成小 fixture + M8X_* 路径覆盖，不触网、不读凭据。

覆盖计划 1016-mixed8-vla-plan.html 3.1 表 A1b 验收行 1–3：
  1. 四个驱动 SMOKE=1 dry-run 通过（trainset 走到「打 2 片 + 成员逐字节对源 + README 渲染」；encoder 与两条 VLA
     走到「预检 + stage + 上传前 sha256 + 公开前体检」），末尾「全部完成」；
  2. grep：没有 --private，没有对 exports/*/verify 的通配删除，没有冷启动阶段；
  3. 有 private==false 断言与 df 余量断言（trainset 在打 tar 前断言 tar 实体 + 单批上限 + 0.4 TB）。
另测几条负例：抽样 features 不是 8 个、VLA motion 开关与驱动不符、step 目录缺一个、encoder run 根下有白名单外条目。
分批回读本身（验收行 4、5）在 test_batched_readback.py。

运行：uv run --no-sync python -m pytest scripts/dataset/hf_export/test_mixed8_200_export_drivers.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TRAINSET = HERE / "run_mixed8_200_trainset_export.sh"
ENCODER = HERE / "run_mixed8_200_encoder_ckpt_export.sh"
VLA512 = HERE / "run_mixed8_200_vla_512_80k_ckpt_export.sh"
VLAMOT = HERE / "run_mixed8_200_vla_motion_80k_ckpt_export.sh"
COMMON = HERE / "_mixed8_200_vla_ckpt_export_common.sh"
DRIVERS = [TRAINSET, ENCODER, VLA512, VLAMOT]
MJ = Path("/scratch/hongze/MotionJEPA")
TASKS = ["BinFill", "ButtonUnmask", "ButtonUnmaskSwap", "InsertPeg", "MoveCube", "PatternLock", "PickHighlight",
         "PickXtimes", "RouteStick", "StopCube", "SwingXtimes", "VideoPlaceButton", "VideoPlaceOrder",
         "VideoRepick", "VideoUnmask", "VideoUnmaskSwap"]
STEPS = list(range(5000, 80000, 5000)) + [79999]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write(p: Path, data) -> bytes:
    p.parent.mkdir(parents=True, exist_ok=True)
    b = data.encode() if isinstance(data, str) else data
    p.write_bytes(b)
    return b


def run(script: Path, env_over: dict, timeout: int = 240) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("M8X_")}
    env.update(env_over)
    env["SMOKE"] = "1"
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------------ fixtures
def make_trainset_lib(root: Path, n_spot: int = 8) -> dict:
    lib = root / "lib"
    num = 5000                                   # > DATA_PER_SHARD(4096)：smoke 打 2 片
    for i in range(num):
        write(lib / "source/data" / f"{i}.pkl", f"pkl-{i}".encode() * 3)
    spot = []
    for k in range(n_spot):
        rel = f"features/episode_{k}/token_emb_0.npy"
        write(lib / "source" / rel, bytes([k]) * 64)
        spot.append({"relpath": rel, "sha256": "0" * 64})
    spot.append({"relpath": "data/0.pkl", "sha256": "0" * 64})
    write(lib / "source/meta/provenance.json", '{"ok": 1}\n')
    tabs = {}
    for sub, pos in (("framesamp-8x8", "pos_emb_8x8"), ("framesamp", "pos_emb_4x4")):
        pb = write(lib / sub / f"{pos}.f32.bin", os.urandom(256))
        sb = write(lib / sub / "state_emb.f32.bin", os.urandom(128))
        write(lib / sub / ("image_emb_8x8" if pos.endswith("8x8") else "image_emb_4x4") / "part_000.bf16.bin",
              os.urandom(512))
        meta = {"tables": {pos: {"sha256": sha(pb)}, "state_emb": {"sha256": sha(sb)}},
                "num_exec_samples": num, "status": "verified"}
        if sub == "framesamp-8x8":
            meta["source_spot_sha256"] = {"entries": spot}
        write(lib / sub / "meta/store_meta.json", json.dumps(meta))
        tabs[sub] = meta
    mb = write(lib / "motion/motion_token.f32.bin", os.urandom(300))
    write(lib / "motion/meta/store_meta.json", json.dumps({"tables": {"motion_token": {"sha256": sha(mb)}},
                                                           "status": "verified"}))
    write(lib / "motion/meta/motion_index.json", '{"rows": 1}\n')
    eps = [{"num_timesteps": 10 + i, "exec_samples": 5} for i in range(32)]
    write(lib / "meta/episode_manifest.json", json.dumps({"canonical_order": TASKS, "episodes": eps,
                                                          "sha256": "a" * 64}))
    write(lib / "meta/input_manifest.json", '{"inputs": []}\n')
    for sub in ("wan-latents", "oracle", "motion-tokens"):
        for j in range(3):
            write(lib / sub / f"seg_{j}.bin", os.urandom(64))
    write(lib / "logs/build.log", "建库日志\n")
    norm = root / "assets/norm_stats.json"
    write(norm, '{"norm_stats": {}}\n')
    runner = root / "runner"
    for f in ("_train_common.sh", "train-512-80k.sh", "train-motion-80k.sh"):
        write(runner / f, "#!/usr/bin/env bash\necho runner\n")
    return {"M8X_LIB": str(lib), "M8X_NORM_STATS": str(norm), "M8X_RUNNER_SRC": str(runner),
            "M8X_EXPORT_ROOT": str(root / "export")}


def make_encoder_fixture(root: Path) -> dict:
    src = root / "runs/enc"
    src.mkdir(parents=True)
    py = MJ / ".venv/bin/python"
    subprocess.run([str(py), "-I", "-c", (
        "import sys, torch\n"
        "for e in range(2, 74, 2):\n"
        "    torch.save({'epoch': e, 'ema_only': True, 'has_live_weights': False,\n"
        "                'encoder': {'w': torch.zeros(2) + e}, 'wan_decoder': {}},\n"
        "               f'{sys.argv[1]}/checkpoint_epoch_{e}.pt')\n"), str(src)],
        check=True, capture_output=True, env={"PATH": os.environ["PATH"]}, cwd="/")
    cfg = "seed: 42\n" + "".join(f"  k{i}: /scratch/hongze/dataset-x/{i}\n" for i in range(5))
    write(src / "config.yaml", cfg)
    rows = [{"epoch": e, "loss_total": 0.1 / e, "loss_raw/latent": 0.09 / e, "loss_raw/sigreg": 9.0 / e,
             "motion/std_batch": 0.9, "motion/cos_cross_chunk": 0.001} for e in range(1, 73)]
    write(src / "train_metrics_epoch.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    write(src / "sig_loss_epoch1.csv", "step,sig\n0,1.0\n")
    write(src / "gpu_samples.csv", "t,util\n0,99\n")
    write(src / "wandb/run-x/files/wandb-metadata.json", '{"remote": "git@github.com:someone/private.git"}\n')
    log = "启动横幅\n" + "".join(
        f"Epoch {e}/72:   0%|          | 0/9 [00:00<?, ?it/s]\rEpoch {e}/72:  50%|#####     | 4/9\n"
        f"Epoch  {e}/72 | lr=1e-4 | train: latent=0.001\n" for e in range(1, 73)) + "EXIT_CODE=0\n"
    tlog = root / "logs/enc.train.log"
    write(tlog, log)
    dl = root / "delivery"
    for i in range(200):
        write(dl / "output/videos" / f"{TASKS[i % 16]}_ep{i}.mp4", b"\x00\x00\x00\x18ftypmp42" + bytes([i % 256]))
    write(dl / "output/figures/x.png", b"\x89PNG")
    write(dl / "reports/README.md", "# 报告\n")
    write(dl / "reports/selected_checkpoint.json", '{"epoch": 72}\n')
    viz = root / "logs/m8v-viz.log"
    write(viz, "...\nDELIVERY=PASS episodes=200 cells=100 per_cell=2 ckpt=epoch_72\nEXIT_CODE=0\n")
    enc_copy = root / "external/checkpoint_epoch_72.pt"
    write(enc_copy, (src / "checkpoint_epoch_72.pt").read_bytes())
    return {"M8X_SRC": str(src), "M8X_TRAIN_LOG": str(tlog), "M8X_MJ": str(MJ), "M8X_DELIVERY": str(dl),
            "M8X_ENC_COPY": str(enc_copy), "M8X_VIZ_LOG": str(viz), "M8X_EXPORT_ROOT": str(root / "export")}


def make_vla_fixture(root: Path, run_name: str, motion: bool) -> dict:
    lib = root / "lib"
    fs8 = write(lib / "framesamp-8x8/meta/store_meta.json", '{"status": "verified"}\n')
    mo = write(lib / "motion/meta/store_meta.json", '{"status": "verified", "m": 1}\n')
    enc = write(root / "external/checkpoint_epoch_72.pt", b"encoder-bytes")
    src = root / "train-runs" / run_name
    for s in STEPS:
        write(src / str(s) / "_CHECKPOINT_METADATA", json.dumps({"commit_timestamp_nsecs": 1700000000 + s}))
        write(src / str(s) / "params/d/blob", os.urandom(128))
    resolved = write(src / "history_config.resolved.yaml", "motion:\n  enabled: %s\n" % ("true" if motion else "false"))
    write(src / "history_config.resolved.sha256", sha(resolved) + "\n")
    write(src / "history_config.txt", "perceptual-framesamp-modul-8frame-8x8.yaml")
    write(src / "wandb_id.txt", "abc123")
    prov = {"motion_enabled": motion, "resolved_sha256": sha(resolved), "framesamp_store_meta_sha256": sha(fs8),
            "framesamp_root": "/scratch/hongze/x/framesamp-8x8",
            "motion_store_meta_sha256": sha(mo) if motion else None,
            "encoder": {"checkpoint_sha256": sha(enc)} if motion else None}
    write(src / "motion_provenance.json", json.dumps(prov, indent=1))
    tlog = root / "logs" / f"{run_name}.log"
    write(tlog, f"RUN={run_name}\nHC=x.yaml\nGIT_HEAD={'f' * 40}\nSTART_UTC=2026-10-17T00:00:00Z\n"
                "step 100 loss=0.1\n 10%|#  | 100/1000\rEND_UTC=2026-10-18T00:00:00Z\nEXIT_CODE=0\n")
    return {"M8X_SRC": str(src), "M8X_TRAIN_LOG": str(tlog), "M8X_LIB": str(lib),
            "M8X_ENC_COPY": str(root / "external/checkpoint_epoch_72.pt"), "M8X_EXPORT_ROOT": str(root / "export")}


def assert_smoke_ok(r: subprocess.CompletedProcess) -> str:
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    lines = r.stdout.strip().splitlines()
    assert lines[-1] == "全部完成", lines[-3:]
    assert "SMOKE=PASS" in r.stdout
    assert re.search(r"^OVERRIDE M8X_EXPORT_ROOT=", r.stdout, re.M)   # 覆盖必须留痕
    return r.stdout


# ------------------------------------------------------------------ 验收 1：四个驱动 SMOKE=1 dry-run
def test_trainset_smoke(tmp_path):
    env = make_trainset_lib(tmp_path)
    out = assert_smoke_ok(run(TRAINSET, env))
    assert "HYGIENE=PASS" in out
    assert re.search(r"^DF_CHECK avail=\d+ need=\d+ tar_est=\d+ tar_have=0 tar_need=\d+ "
                     r"readback_batch_max=107374182400 margin=400000000000$", out, re.M)
    assert re.search(r"SMOKE_BYTEWISE checked=20 mismatches=0", out)
    assert "README_RENDERED tasks=16 episodes=32 exec=5000 spot_features=8" in out
    readme = (tmp_path / "export/tmp/README.smoke.md").read_text()
    assert "@@" not in readme and "BinFill / ButtonUnmask" in readme
    tars = sorted(p.name for p in (tmp_path / "export/stage/source/data_tars").iterdir())
    assert tars == ["data-00000.tar", "data-00001.tar"]
    # smoke 不碰网络：没有任何 hf / curl 调用的痕迹
    assert "whoami" not in out and "BUCKET_CREATED" not in out


def test_trainset_smoke_rejects_spot_count(tmp_path):
    env = make_trainset_lib(tmp_path, n_spot=7)
    r = run(TRAINSET, env)
    assert r.returncode != 0
    assert "source_spot_sha256 features/ 条目应为 8，实为 7" in (r.stdout + r.stderr)


@pytest.mark.skipif(not (MJ / ".venv/bin/python").exists(), reason="需要 MotionJEPA .venv 的 torch")
def test_encoder_smoke(tmp_path):
    env = make_encoder_fixture(tmp_path)
    out = assert_smoke_ok(run(ENCODER, env))
    assert "ENCODER_MATCHES_VLA_COPY=OK" in out
    assert "CKPT_EMA_ONLY=PASS checked=36 bad=0" in out
    assert "README_RENDERED ckpts=36 epochs=72 mp4=200 reports=2" in out
    assert "HYGIENE=PASS" in out
    stage = tmp_path / "export/stage"
    assert not (stage / "wandb").exists()                         # wandb/ 刻意不传
    assert not list(stage.rglob("*.png"))                         # figures/ 不在口径 29 范围
    tl = (stage / "train.log").read_text()
    assert tl.count("\nEpoch ") + tl.startswith("Epoch ") == 72 and "%|" not in tl
    assert tl.rstrip().endswith("EXIT_CODE=0")
    n = sum(1 for _ in stage.rglob("*") if _.is_file())
    assert n == 36 + 5 + 200 + 2 + 2                               # ckpt + 根白名单 + mp4 + reports + README/SHA256SUMS
    rep = json.loads((tmp_path / "export/logs/hygiene-stage.json").read_text())
    got = {(e["pattern"], e["path_glob"]): e["got"] for e in rep["accept_entries"]}
    assert got[("P9", "config.yaml")] == 5


@pytest.mark.skipif(not (MJ / ".venv/bin/python").exists(), reason="需要 MotionJEPA .venv 的 torch")
def test_encoder_rejects_unexpected_root_entry(tmp_path):
    env = make_encoder_fixture(tmp_path)
    write(Path(env["M8X_SRC"]) / "stray.txt", "x")
    r = run(ENCODER, env)
    assert r.returncode != 0 and "白名单外条目" in r.stdout and "stray.txt" in r.stdout


@pytest.mark.parametrize("script,run_name,motion", [
    (VLA512, "mixed8-200-m8x8-modul-b128-80k", False),
    (VLAMOT, "mixed8-200-m8x8-modul-motion-b128-80k", True),
])
def test_vla_smoke(tmp_path, script, run_name, motion):
    env = make_vla_fixture(tmp_path, run_name, motion)
    out = assert_smoke_ok(run(script, env))
    assert f"PROVENANCE_MATCH=PASS motion={motion}" in out
    assert "README_RENDERED steps=16" in out
    assert "HYGIENE=PASS" in out
    stage = tmp_path / "export/stage"
    assert sorted(int(p.name) for p in stage.iterdir() if p.is_dir() and p.name.isdigit()) == STEPS
    assert (stage / "_run-meta/train.log").read_text().rstrip().endswith("EXIT_CODE=0")
    readme = (stage / "README.md").read_text()
    assert "@@" not in readme and "f" * 40 in readme


@pytest.mark.skipif(not (MJ / ".venv/bin/python").exists(), reason="需要 MotionJEPA .venv 的 torch")
def test_encoder_rejects_missing_delivery_pass(tmp_path):
    env = make_encoder_fixture(tmp_path)
    write(Path(env["M8X_VIZ_LOG"]), "DELIVERY=FAIL reason=x\nEXIT_CODE=1\n")
    r = run(ENCODER, env)
    assert r.returncode != 0 and "DELIVERY=PASS" in r.stdout and "SMOKE=PASS" not in r.stdout


def test_vla_rejects_motion_mismatch(tmp_path):
    env = make_vla_fixture(tmp_path, "mixed8-200-m8x8-modul-b128-80k", motion=True)   # 512 驱动遇到 motion 开启的 run
    r = run(VLA512, env)
    assert r.returncode != 0 and "PROVENANCE_MATCH=FAIL" in r.stdout


def test_vla_rejects_missing_step(tmp_path):
    env = make_vla_fixture(tmp_path, "mixed8-200-m8x8-modul-motion-b128-80k", motion=True)
    import shutil
    shutil.rmtree(Path(env["M8X_SRC"]) / "40000")
    r = run(VLAMOT, env)
    assert r.returncode != 0 and "子目录不符" in r.stdout


def test_full_mode_refuses_export_root_outside_store(tmp_path):
    env = make_vla_fixture(tmp_path, "mixed8-200-m8x8-modul-b128-80k", motion=False)
    env["SMOKE"] = "0"
    e = {k: v for k, v in os.environ.items() if not k.startswith("M8X_")}
    e.update(env)
    r = subprocess.run(["bash", str(VLA512)], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "EXPORT_ROOT 必须在" in r.stdout


# ------------------------------------------------------------------ 验收 2、3：grep
def _code(p: Path) -> str:
    """去掉整行注释后的代码（注释里解释『为什么不做』的文字不算违例）。"""
    return "\n".join(l for l in p.read_text().splitlines() if not l.lstrip().startswith("#"))


@pytest.mark.parametrize("p", DRIVERS + [COMMON], ids=lambda p: p.name)
def test_no_private_no_wildcard_verify_delete_no_coldstart(p):
    raw, code = p.read_text(), _code(p)
    assert "--private" not in raw                                   # 连注释里也没有，裸 grep 即可复核
    assert not re.search(r"exports/\*", raw)
    assert "rm -rf" not in code or all("$STAGE" in l or "$TMPD/anon" in l
                                       for l in code.splitlines() if "rm -rf" in l)
    for bad in ("rehearse", "COLD_START", "train.py\" \"$cfg\"", "--num-train-steps", "CUDA_VISIBLE_DEVICES"):
        assert bad not in code


@pytest.mark.parametrize("p", [TRAINSET, ENCODER, COMMON], ids=lambda p: p.name)
def test_has_public_and_df_assertions(p):
    code = _code(p)
    assert re.search(r'False\|false\) : ;;', code)                 # assert_public 只认 private==false
    assert code.count("assert_public ") >= 2                        # 建后一次 + 收尾一次
    assert "hf buckets settings \"$BUCKET_ID\" --public" in code
    assert "DF_MARGIN_BYTES=400000000000" in code and "DF_CHECK" in code


def test_trainset_df_before_tar():
    code = _code(TRAINSET)
    df = code.index('DF_NEED=$((TAR_NEED + READBACK_BATCH_MAX_BYTES + DF_MARGIN_BYTES))')
    pack_full = code.index("--bucket \"$BUCKET\" --norm-stats \"$NORM_STATS\" --workers 8\n")
    assert df < pack_full
    assert code.index("batched_readback.py plan") < code.index("batched_readback.py run")
    assert "hf sync \"$BUCKET\" \"$VERIFY\"" not in code             # 整桶回读已改为分批
    assert code.rstrip().endswith('echo "全部完成"')
