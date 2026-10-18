"""客户端改动单测（1017-test-xhard5-12per 4.3 第 4、7、8、10、11 项）。

* RolloutRecorder 流式编码 + 逐步动作日志：合成帧跑 3 局（第 2 局故意让文字条行数变化），核 mp4 解码帧数、
  jsonl 行数、临时名均已改名、被 discard 的局没留下任何文件；再把结果喂给 check_shard.py 走 EVAL_PARAMS /
  MANIFEST_DIFFICULTY / ACTION_LOG / SHARD_PASS；
* SpecEnvRunner 按难度分流（用桩 BenchmarkEnvBuilder，不起仿真）；eval.py::load_plan 读清单；
* reset_clean.py、es_bound.py（合成 h5）、summarize.py。
临时环境：``uv run --no-project --with pytest --with numpy --with opencv-python-headless --with imageio
--with imageio-ffmpeg --with h5py python -m pytest scripts/evaluation/test_client_tools.py -q``
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parents[1] / "examples" / "robomme"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLIENT))
import utils as client_utils  # noqa: E402


def _frames(n, h=64, w=48):
    rng = np.random.default_rng(n)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _record_episode(save_dir, steps_dir, task, ep, n_demo, n_exec, flag, long_text=False):
    rec = client_utils.RolloutRecorder(save_dir / "videos", f"goal {task}", fps=30,
                                       steps_path=steps_dir / f"{task}_ep{ep}.jsonl")
    imgs = _frames(n_demo + n_exec)
    for i, img in enumerate(imgs):
        state = np.arange(8, dtype=np.float32) * (1000.0 if (long_text and i % 2) else 1.0)
        action = None if i < n_demo else np.full(8, 0.5 if not (long_text and i % 3) else 123456.789, np.float32)
        rec.record(image=img, wrist_image=img, state=state, action=action,
                   is_video_demo=i < n_demo, motion_k=None if i < n_demo else i)
    if flag is None:
        rec.discard()
    else:
        rec.save_video(f"{task}_ep{ep}_{flag}_goal {task}_{'easy' if ep < 50 else 'xhard14'}.mp4")
    return n_demo + n_exec


def test_recorder_streams_and_check_shard(tmp_path):
    import imageio.v2 as imageio
    run_dir = tmp_path / "s0"
    root = run_dir / "pol" / "ckpt60000" / "seed7"
    steps = root / "steps"
    manifest = tmp_path / "m.jsonl"
    rows = [{"task": "PickXtimes", "episode": 0, "seed": 510000, "difficulty": "easy"},
            {"task": "PickXtimes", "episode": 50, "seed": 515000, "difficulty": "xhard14"},
            {"task": "StopCube", "episode": 1, "seed": 530100, "difficulty": "easy"}]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    n0 = _record_episode(root, steps, "PickXtimes", 0, 3, 7, "success")
    n1 = _record_episode(root, steps, "PickXtimes", 50, 2, 9, "timeout", long_text=True)
    _record_episode(root, steps, "StopCube", 1, 0, 5, None)            # error 局：discard
    parts = list((root / "videos").glob("*.part"))
    assert len(parts) == 2                                            # 已登记、尚未回收
    finalized, failed = client_utils.finalize_all()
    assert (finalized, failed) == (2, 0)
    assert not list((root / "videos").glob("*.part"))
    videos = sorted((root / "videos").glob("*.mp4"))
    assert [v.name.split("_")[1] for v in videos] == ["ep0", "ep50"]
    for v, n in zip(videos, (n0, n1)):
        with imageio.get_reader(v) as reader:
            frames = [f for f in reader]
        assert len(frames) == n and frames[0].shape[1] == 48         # 半分辨率：拼接宽 96 → 48
    lines = (steps / "PickXtimes_ep50.jsonl").read_text().splitlines()
    assert len(lines) == n1 and json.loads(lines[0])["action"] is None and json.loads(lines[-1])["motion_k"] == n1 - 1
    assert len((steps / "StopCube_ep1.jsonl").read_text().splitlines()) == 5   # error 局的动作日志照留

    plan = {"shard_tag": "s0", "manifest": str(manifest), "manifest_sha256": sha,
            "groups": {"PickXtimes/easy": [0], "PickXtimes/xhard14": [50], "StopCube/easy": [1]}}
    plan_path = tmp_path / "shard0.json"
    plan_path.write_text(json.dumps(plan))
    (root / "plan.json").write_text(json.dumps({**plan, "max_steps": 2000}))
    (root / "progress.json").write_text(json.dumps(
        {"PickXtimes/easy": {"0": True}, "PickXtimes/xhard14": {"50": False}, "StopCube/easy": {"1": "error"}}))
    env = {**os.environ, "EXPECT_MOTION_STATS": "0", "EXPECT_DEMO_INJECT": "0"}
    proc = subprocess.run([sys.executable, str(HERE / "check_shard.py"), str(run_dir), "pol", "60000", "7", str(plan_path)],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for line in ("EVAL_PARAMS=PASS max_steps=2000", "MANIFEST_DIFFICULTY=PASS checked=3",
                 "ACTION_LOG=PASS episodes=2 rows_eq_steps=yes", "SHARD_PASS"):
        assert line in proc.stdout
    (root / "plan.json").write_text(json.dumps({**plan, "max_steps": 1300}))
    proc = subprocess.run([sys.executable, str(HERE / "check_shard.py"), str(run_dir), "pol", "60000", "7", str(plan_path)],
                          capture_output=True, text=True, env=env)
    assert proc.returncode != 0 and "EVAL_PARAMS=FAIL" in proc.stderr


def test_recorder_broken_pipe_marks_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(client_utils, "_ffmpeg_exe", lambda: "/bin/false")
    rec = client_utils.RolloutRecorder(tmp_path, "g", fps=30)
    img = _frames(1, 512, 512)[0]                                     # 一帧 > 64 KiB 管道，进程已退出必断
    for _ in range(3):
        rec.record(image=img, wrist_image=img, state=np.zeros(8))
    rec.save_video("x_ep0_fail_g_easy.mp4")
    assert rec.failed
    assert client_utils.finalize_all() == (1, 1)
    assert not list(tmp_path.glob("*.mp4")) and not list(tmp_path.glob("*.part"))


# ── SpecEnvRunner 分流（桩 robomme）────────────────────────────────────────────────
class _Env:
    def __init__(self, difficulty):
        self.unwrapped = types.SimpleNamespace(difficulty=difficulty)

    def close(self):
        pass


def _install_stub_robomme(monkeypatch, made):
    class Builder:
        def __init__(self, env_id, dataset="test", action_space="joint_angle", gui_render=False,
                     override_metadata_path=None, max_steps=10000):
            self.env_id, self.override = env_id, override_metadata_path
            made.append(self)

        def resolve_episode(self, ep):
            return (515000, "xhard14") if self.env_id.endswith("XHard") else (510000 + ep * 100, "easy")

        def make_env_for_episode(self, ep):
            return _Env("xhard14" if self.env_id.endswith("XHard") else "easy")

    pkg = types.ModuleType("robomme")
    env_mod = types.ModuleType("robomme.robomme_env")
    env_mod.__all__ = []
    wrap = types.ModuleType("robomme.env_record_wrapper")
    wrap.BenchmarkEnvBuilder = Builder
    for name, mod in (("robomme", pkg), ("robomme.robomme_env", env_mod), ("robomme.env_record_wrapper", wrap),
                      ("robomme_xhard", types.ModuleType("robomme_xhard"))):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "env_runner", raising=False)
    import env_runner
    return env_runner


def test_spec_env_runner_routes_by_difficulty(tmp_path, monkeypatch, capsys):
    made = []
    er = _install_stub_robomme(monkeypatch, made)
    monkeypatch.setenv("XHARD_METADATA_DIR", str(tmp_path))
    x = er.SpecEnvRunner("PickXtimes", "xhard14", str(tmp_path))
    x.make_env({"task": "PickXtimes", "episode": 50, "seed": 515000, "difficulty": "xhard14"})
    assert made[-1].env_id == "PickXtimesXHard" and Path(made[-1].override) == tmp_path
    assert x.env_id == "PickXtimes" and x.difficulty == "xhard14"
    b = er.SpecEnvRunner("MoveCube", "easy", str(tmp_path))
    b.make_env({"task": "MoveCube", "episode": 1, "seed": 510100, "difficulty": "easy"})
    assert made[-1].env_id == "MoveCube" and made[-1].override is None
    out = capsys.readouterr().out
    assert "ENV_ROUTE route=xhard env=PickXtimesXHard" in out and "ENV_ROUTE route=base env=MoveCube" in out
    with pytest.raises(ValueError):                                   # base 档 episode 必须 <50
        b.make_env({"task": "MoveCube", "episode": 50, "seed": 515000, "difficulty": "easy"})
    with pytest.raises(ValueError):                                   # metadata seed 与清单不符
        b.make_env({"task": "MoveCube", "episode": 2, "seed": 1, "difficulty": "easy"})
    monkeypatch.delenv("XHARD_METADATA_DIR")
    with pytest.raises(ValueError):                                   # xhard 组必须给 XHARD_METADATA_DIR
        er.SpecEnvRunner("PickXtimes", "xhard14", str(tmp_path))


def test_load_plan_reads_manifest(tmp_path):
    source = (CLIENT / "eval.py").read_text()
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if getattr(n, "name", None) in {"Args", "load_plan"}]
    import dataclasses
    ns = dict(dataclasses=dataclasses, json=json, Path=Path, hashlib=hashlib)
    exec(compile(tree, "eval.py", "exec"), ns)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"task": "BinFill", "episode": 50, "seed": 500000 + 5000, "difficulty": "xhard12"}) + "\n")
    plan = {"manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "groups": {"BinFill/xhard12": [50]}}
    (tmp_path / "p.json").write_text(json.dumps(plan))
    _, header, groups = ns["load_plan"](ns["Args"](episode_plan=str(tmp_path / "p.json")))
    assert groups["BinFill/xhard12"][0]["seed"] == 505000 and header["manifest_sha256"] == plan["manifest_sha256"]
    plan["manifest_sha256"] = "0" * 64
    (tmp_path / "p.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        ns["load_plan"](ns["Args"](episode_plan=str(tmp_path / "p.json")))


# ── reset_clean / es_bound / summarize ─────────────────────────────────────────────
def test_reset_clean(tmp_path):
    import reset_clean
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    rows = [{"step": i, "action": None if i < 2 else [0.1 * i] * 8, "state": [float(i)] * 8} for i in range(6)]
    (a / "BinFill_ep0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (b / "BinFill_ep0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert reset_clean.compare(a, b)["verdict"] == "PASS"
    rows[4]["action"] = [9.0] * 8
    (b / "BinFill_ep0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = reset_clean.compare(a, b)
    assert res["verdict"] == "FAIL" and res["first_diff_step"] == "BinFill_ep0:4"


def test_es_bound(tmp_path, capsys):
    import h5py
    import es_bound
    h5 = tmp_path / "g" / "record_dataset_PickXtimes.h5"
    h5.parent.mkdir()
    with h5py.File(h5, "w") as f:
        for name, seed, diff, demo in (("episode_50", 515000, "xhard12", 1247), ("episode_51", 515100, "xhard14", 3)):
            g = f.create_group(name)
            g["setup/seed"] = seed
            g["setup/difficulty"] = diff
            for i in range(demo + 2):
                g[f"timestep_{i}/info/is_video_demo"] = i < demo
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in (
        {"task": "PickXtimes", "episode": 50, "seed": 515000, "difficulty": "xhard12"},
        {"task": "PickXtimes", "episode": 51, "seed": 515100, "difficulty": "xhard14"},
        {"task": "PickXtimes", "episode": 0, "seed": 510000, "difficulty": "easy"})))
    sys.argv = ["es_bound.py", str(manifest), str(tmp_path / "g"), "--out", str(tmp_path / "es.json")]
    es_bound.main()
    out = capsys.readouterr().out
    assert "EVAL_ES_BOUND=DONE scope=xhard episodes=2 max_windows=201 over=1 max_step_idx=3247 over4096=0" in out
    assert es_bound.predicted_windows(1500, 2000) == 93 + 124


def test_summarize(tmp_path, capsys):
    import summarize
    rows = [{"task": "BinFill", "episode": e, "seed": 500000 + e * 100, "difficulty": d}
            for e, d in ((0, "easy"), (1, "easy"), (50, "xhard12"), (51, "xhard14"))]
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    specs = []
    for label, values in (("A", {"0": True, "1": False}), ("B", {"0": "error", "1": True})):
        run = tmp_path / label
        for k, (group, eps) in enumerate((("BinFill/easy", values), ("BinFill/xhard12", {"50": False}))):
            root = run / f"s{k}" / "pol" / "ckpt1" / "seed7"
            (root / "videos").mkdir(parents=True)
            prog = {group: eps}
            if k == 1:
                prog["BinFill/xhard14"] = {"51": True}
            (root / "progress.json").write_text(json.dumps(prog))
            (root / "plan.json").write_text(json.dumps({"manifest_sha256": sha}))
            for g, entries in prog.items():
                for e, v in entries.items():
                    if v == "error":
                        continue
                    flag = "success" if v else ("timeout" if e == "50" else "fail")
                    (root / "videos" / f"BinFill_ep{e}_{flag}_goal_{g.split('/')[1]}.mp4").write_bytes(b"")
        specs += ["--model", f"{label}={run}:pol:1"]
    sys.argv = ["summarize.py", str(manifest), *specs, "--out-dir", str(tmp_path / "sum")]
    summarize.main()
    out = capsys.readouterr().out
    for line in ("SHARDS_UNION=PASS model=A total=4", "SEED_MATCH=PASS model=B", "EVAL_VIDEO=PASS model=B videos=3",
                 "EVAL_A=DONE episodes=4 success=2 fail=1 timeout=1 error=0", "MANIFEST_SHARED=PASS models=2",
                 "EVAL_TABLE=DONE models=2 cells=3"):
        assert line in out, out
    table = (tmp_path / "sum" / "summary.md").read_text()
    assert "| BinFill | easy | 2 | 1/2 | 1/2（err 1） |" in table
