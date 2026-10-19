"""清单导出与分片单测（1017-test-xhard5-12per 4.3 第 3 项）：用合成 test metadata 跑 make_manifest / make_shard_plan。

合成口径：16 个任务各 50 条官方 base（ep0–49，难度按 e,e,m,h 循环 → easy 26 / medium 12 / hard 12），
再按计划 3.1 五组表追加 xhard（ep50 起，难度 = 有效循环[ep % 档数]，每档 12 条）。各任务有哪些 xhard 档以
benchmark 仓库 ``src/robomme_xhard/configs.py::TIER_CONFIGS`` 为准（2026-10-18 只读核对，抄在 XHARD_TIERS）；
MoveCube、InsertPeg 没有 xhard。只依赖标准库与 pytest：
``uv run --no-project --with pytest python -m pytest scripts/evaluation/test_manifest.py -q``
"""
import collections
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_manifest  # noqa: E402
import make_shard_plan  # noqa: E402

#: 与 configs.py::TIER_CONFIGS 一致（52 格）
XHARD_TIERS = {
    "BinFill": (12, 14, 16, 18, 20), "PickXtimes": (12, 14, 16, 18, 20), "SwingXtimes": (12, 14, 16, 18, 20),
    "StopCube": (14, 16, 18, 20), "VideoUnmask": (14,), "ButtonUnmask": (14,),
    "VideoUnmaskSwap": (12, 14, 16, 18, 20), "ButtonUnmaskSwap": (12, 14, 16, 18, 20),
    "PickHighlight": (12, 16, 18), "VideoRepick": (12, 14, 16, 18, 20), "VideoPlaceButton": (12, 14),
    "VideoPlaceOrder": (12,), "PatternLock": (12, 14, 16, 18, 20), "RouteStick": (12, 14, 16, 18, 20),
}
ENV_CODE = {t: i for i, t in enumerate(make_manifest.TASK_ORDER)}


def build_metadata(root: Path) -> Path:
    meta = root / "env_metadata" / "test"
    meta.mkdir(parents=True)
    assert sum(len(v) for v in XHARD_TIERS.values()) == 52
    for task in make_manifest.TASK_ORDER:
        records = []
        for ep in range(50):
            diff = ("easy", "easy", "medium", "hard")[ep % 4]
            records.append({"task": task, "episode": ep, "seed": 500000 + ENV_CODE[task] * 10000 + ep * 100,
                            "difficulty": diff})
        tiers = XHARD_TIERS.get(task, ())
        for ep in range(50, 50 + 12 * len(tiers)):
            records.append({"task": task, "episode": ep,
                            "seed": 500000 + ENV_CODE[task] * 10000 + ep * 100 + (ep % 3),  # attempt 换 seed 的样子
                            "difficulty": f"xhard{tiers[ep % len(tiers)]}"})
        payload = {"env_id": task, "record_count": len(records), "records": records}
        (meta / f"record_dataset_{task}_metadata.json").write_text(json.dumps(payload, indent=2))
    return meta


def run_export(meta: Path, out: Path):
    proc = subprocess.run([sys.executable, str(HERE / "make_manifest.py"), "export", str(out),
                           "--metadata-dir", str(meta)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return proc.stdout


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture()
def exported(tmp_path):
    meta = build_metadata(tmp_path)
    out = tmp_path / "manifest"
    stdout = run_export(meta, out)
    return tmp_path, meta, out, stdout


def test_export_counts_and_groups(exported):
    _, _, out, stdout = exported
    assert "MANIFEST=PASS rows=1200 cells=100" in stdout
    assert "SMOKE_SUBSET=PASS rows=2 subset_of_full=yes" in stdout
    rows = read_jsonl(out / make_manifest.FULL_NAME)
    cells = collections.Counter((r["task"], r["difficulty"]) for r in rows)
    assert len(rows) == 1200 and len(cells) == 100 and set(cells.values()) == {12}
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["difficulty"])].append(r["episode"])
    # base 档取官方 ep0–49 升序前 12
    assert by_cell[("MoveCube", "easy")] == [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21]
    assert by_cell[("MoveCube", "hard")] == list(range(3, 48, 4))
    assert not any(t in ("MoveCube", "InsertPeg") and d.startswith("xhard") for t, d in cells)
    # G4 StopCube：循环 [14,16,18,20]，ep50 落 xhard18
    assert by_cell[("StopCube", "xhard18")] == list(range(50, 98, 4))
    assert by_cell[("StopCube", "xhard14")] == list(range(52, 98, 4))
    # G3 PickHighlight：循环 [12,16,18]，ep50 落 xhard18
    assert by_cell[("PickHighlight", "xhard18")] == list(range(50, 86, 3))
    assert by_cell[("PickHighlight", "xhard12")] == list(range(51, 86, 3))
    # G5 / G2 / G1
    assert by_cell[("BinFill", "xhard12")] == list(range(50, 110, 5))
    assert by_cell[("VideoPlaceButton", "xhard14")] == list(range(51, 74, 2))
    assert by_cell[("VideoPlaceOrder", "xhard12")] == list(range(50, 62))
    # 排序口径 (task, difficulty, episode)
    assert [make_manifest.row_key(r) for r in rows] == sorted(make_manifest.row_key(r) for r in rows)


def test_smoke_subset_and_xhard_metadata(exported):
    _, meta, out, _ = exported
    full = {make_manifest.row_key(r) for r in read_jsonl(out / make_manifest.FULL_NAME)}
    smoke = read_jsonl(out / make_manifest.SMOKE_NAME)
    assert len(smoke) == 2 and {make_manifest.row_key(r) for r in smoke} <= full
    # 两次冒烟共用：一局 base、一局 xhard（两条分流路径各一局）
    assert [(r["task"], r["difficulty"], r["episode"]) for r in smoke] == [("BinFill", "easy", 0), ("BinFill", "xhard12", 50)]
    copies = sorted((out / "metadata").glob("record_dataset_*XHard_metadata.json"))
    assert len(copies) == 14
    sc = json.loads((out / "metadata" / "record_dataset_StopCubeXHard_metadata.json").read_text())
    assert sc["env_id"] == "StopCubeXHard" and sc["record_count"] == 48
    assert {r["task"] for r in sc["records"]} == {"StopCubeXHard"}
    src = {r["episode"]: r for r in json.loads((meta / "record_dataset_StopCube_metadata.json").read_text())["records"]}
    assert all(r["seed"] == src[r["episode"]]["seed"] and r["difficulty"] == src[r["episode"]]["difficulty"]
               for r in sc["records"])
    with pytest.raises(AssertionError):           # 拒绝覆盖已有产物
        run_export(meta, out)


def test_shards_ten_by_120(exported, capsys):
    tmp, _, out, _ = exported
    summary = make_shard_plan.make_shards(out / make_manifest.FULL_NAME, tmp / "run512" / "plans", 10,
                                          expect_per_shard=120)
    assert "SHARD_PLAN=PASS shards=10 per_shard=120" in capsys.readouterr().out
    assert summary["sizes"] == [120] * 10
    per_cell = collections.defaultdict(list)
    union = set()
    for i in range(10):
        plan = json.loads((tmp / "run512" / "plans" / f"shard{i}.json").read_text())
        assert plan["manifest_sha256"] == summary["manifest_sha256"] == plan["identity_sha256"]
        n = 0
        for group, eps in plan["groups"].items():
            per_cell[group].append(len(eps))
            union |= {(group, e) for e in eps}
            n += len(eps)
        assert n == 120
    assert len(union) == 1200
    assert all(max(v) - min(v) <= 1 for v in per_cell.values())
    with pytest.raises(SystemExit):               # 拒绝覆盖已有分片计划
        make_shard_plan.make_shards(out / make_manifest.FULL_NAME, tmp / "run512" / "plans", 10)


def test_smoke_plans_and_units(exported):
    tmp, _, out, _ = exported
    root = tmp / "evaluation"
    a = make_shard_plan.make_shards(out / make_manifest.SMOKE_NAME, root / "testx-official-smoke" / "plans", 1)
    b = make_shard_plan.make_shards(out / make_manifest.SMOKE_NAME, root / "testx-512m-10k-smoke" / "plans", 1)
    assert a["sizes"] == b["sizes"] == [2] and a["manifest_sha256"] == b["manifest_sha256"]
    for run in ("testx-512-60k", "testx-512m-60k", "testx-official-79999"):
        make_shard_plan.make_shards(out / make_manifest.FULL_NAME, root / run / "plans", 10, expect_per_shard=120)
    tsv = root / "testx" / "manifest.tsv"
    script = [sys.executable, str(HERE / "make_manifest.py"), "units", str(tsv), "--eval-root", str(root)]
    off = "/m/official-mme-vla/perceptual-framesamp-modul/79999"
    smoke = script + ["--stage", "smoke",
                      "--run", f"testx-official-smoke|{off}|official-framesamp-modul|mme_vla_suite|79999|1|0",
                      "--run", "testx-512m-10k-smoke|/m/mixed8-motion/10000|mixed8-512-motion|mme_vla_suite_b128_80k|10000|1|1"]
    proc = subprocess.run(smoke, capture_output=True, text=True)
    assert proc.returncode == 0 and "UNITS=PASS stage=smoke added=2 total=2" in proc.stdout, proc.stderr
    formal = script + ["--stage", "formal",
                       "--run", "testx-512-60k|/m/mixed8/60000|mixed8-512|mme_vla_suite_b128_80k|60000|10|0",
                       "--run", "testx-512m-60k|/m/mixed8-motion/60000|mixed8-512-motion|mme_vla_suite_b128_80k|60000|10|1",
                       "--run", f"testx-official-79999|{off}|official-framesamp-modul|mme_vla_suite|79999|10|0"]
    proc = subprocess.run(formal, capture_output=True, text=True)
    assert proc.returncode == 0 and "UNITS=PASS stage=formal added=30 total=32" in proc.stdout, proc.stderr
    lines = [l.split("\t") for l in tsv.read_text().splitlines()]
    assert len(lines) == 32 and all(len(l) == 6 for l in lines)
    order = [l[0] for l in lines]
    assert order[:2] == ["testx-official-smoke", "testx-512m-10k-smoke"]
    assert order[2:12] == ["testx-512-60k"] * 10 and order[12:22] == ["testx-512m-60k"] * 10
    assert order[22:] == ["testx-official-79999"] * 10
    for run, _, policy, *_ in lines:
        assert policy.endswith("-motion") == ("512m" in run)
    # 重复写入、motion 列与 policy 后缀不一致、formal 顺序不对，都拒绝
    assert subprocess.run(formal, capture_output=True, text=True).returncode != 0
    bad = script + ["--stage", "any", "--run", "x|/m/a/1|mixed8-512|c|1|1|1"]
    assert subprocess.run(bad, capture_output=True, text=True).returncode != 0
    base = [sys.executable, str(HERE / "make_manifest.py"), "units", str(tmp / "other.tsv"), "--stage", "formal"]
    swapped = base + ["--run", "a|/m/b/60000|mixed8-512-motion|c|60000|10|1",
                      "--run", "b|/m/a/60000|mixed8-512|c|60000|10|0",
                      "--run", "c|/m/o/79999|official|c|79999|10|0"]
    assert subprocess.run(swapped, capture_output=True, text=True).returncode != 0


def run_official(meta: Path, out: Path):
    return subprocess.run([sys.executable, str(HERE / "make_manifest.py"), "export-official", str(out),
                           "--metadata-dir", str(meta)], capture_output=True, text=True)


def test_export_official_excludes_xhard(tmp_path, capsys):
    """1019 官方 800 局口径：每任务只取 ep0–49，ep50 起追加的 xhard 一律排除；可直接切 10×80 片。"""
    meta = build_metadata(tmp_path)
    out = tmp_path / "official800"
    proc = run_official(meta, out)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "MANIFEST_OFFICIAL=PASS rows=800 tasks=16" in proc.stdout
    rows = read_jsonl(out / make_manifest.OFFICIAL_NAME)
    assert len(rows) == 800 and set(rows[0]) == {"task", "episode", "seed", "difficulty"}
    assert {r["difficulty"] for r in rows} <= set(make_manifest.BASE_DIFFICULTIES)
    per_task = collections.defaultdict(list)
    for r in rows:
        per_task[r["task"]].append(r["episode"])
    assert set(per_task) == set(make_manifest.TASK_ORDER)
    assert all(sorted(v) == list(range(50)) for v in per_task.values())
    assert [make_manifest.row_key(r) for r in rows] == sorted(make_manifest.row_key(r) for r in rows)
    meta_json = json.loads((out / "manifest_meta.json").read_text())
    assert meta_json["rows"] == 800 and meta_json["sources"]["BinFill"]["picked"] == 50
    assert run_official(meta, out).returncode != 0          # 拒绝覆盖已有产物
    summary = make_shard_plan.make_shards(out / make_manifest.OFFICIAL_NAME, tmp_path / "off" / "plans", 10,
                                          expect_per_shard=80)
    assert summary["sizes"] == [80] * 10
    assert "SHARD_PLAN=PASS shards=10 per_shard=80" in capsys.readouterr().out


def test_export_official_missing_episode_fails(tmp_path):
    meta = build_metadata(tmp_path)
    path = meta / "record_dataset_RouteStick_metadata.json"
    payload = json.loads(path.read_text())
    payload["records"] = [r for r in payload["records"] if r["episode"] != 17]
    payload["record_count"] = len(payload["records"])
    path.write_text(json.dumps(payload))
    proc = run_official(meta, tmp_path / "official800")
    assert proc.returncode != 0 and "MANIFEST_OFFICIAL=FAIL RouteStick" in proc.stderr


def test_eval_scripts_bash_syntax():
    for name in ("gl_hold_pool.sh", "gl_hold_queue.sh", "run_shard.sh", "gl_eval_shard.sbatch"):
        proc = subprocess.run(["bash", "-n", str(HERE / name)], capture_output=True, text=True)
        assert proc.returncode == 0, name + proc.stderr
