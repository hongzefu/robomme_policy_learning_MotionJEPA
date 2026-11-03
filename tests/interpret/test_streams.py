"""S3 单测：合成 npz 核聚合、分流求和、反推帧号／窗号、pad 质量、「揭开」候选与 KEY_STEP、出图。

合成两份 npz：P=1184（motion，MEM=608，含已知 mem_order 置换与部分 padding）与 P=1088（基线，MEM=512，
无 mem_order 键走恒等映射）。probs 每行在合法 key 上 softmax 归一，padding 列为 0（与 big_neg 后 softmax 一致）。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "interpret"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agg = _load("aggregate_attn")

STEPS, LAYERS, HEADS, AQ, TQ = 10, 18, 8, 20, 64
TXT_LEN = 40


def _softmax_rows(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.where(valid, logits, -np.inf)
    x = x - x.max(-1, keepdims=True)
    e = np.where(valid, np.exp(x), 0.0)
    return e / e.sum(-1, keepdims=True)


def make_npz(path: Path, *, motion: bool, seed: int, infer_seq: int = 0, step_idx: int = 300,
             boost: dict | None = None, actions: np.ndarray | None = None) -> dict:
    """造一份符合 §1.3 键契约的 npz，返回构造时的真值（各帧／窗质量的设计值）。"""
    rng = np.random.default_rng(seed)
    MEM = 608 if motion else 512
    P = MEM + 512 + 64
    frame_times = np.full(32, -1, np.int32)
    n_valid_frames = 30  # 最后 2 帧 padding
    frame_times[:n_valid_frames] = np.linspace(0, step_idx, n_valid_frames).astype(np.int32)
    raw_mask = np.zeros(MEM, bool)  # 原始序号空间的 mask
    raw_mask[: n_valid_frames * 16] = True
    if motion:
        motion_times = np.arange(96, dtype=np.int32) * 16
        motion_mask = np.zeros(96, bool)
        motion_mask[:10] = True  # 前 10 窗有效
        raw_mask[512:512 + 10] = True
        # 已知置换：有效 token 随机打乱、padding 排尾部（与 HistoryPi0.embed_memory 口径一致）
        valid_ids = np.flatnonzero(raw_mask)
        pad_ids = np.flatnonzero(~raw_mask)
        mem_order = np.concatenate([rng.permutation(valid_ids), pad_ids]).astype(np.int32)
    else:
        motion_times = np.zeros(0, np.int32)
        motion_mask = np.zeros(0, bool)
        mem_order = None
    order = mem_order if mem_order is not None else np.arange(MEM)
    mem_mask = raw_mask[order]

    # action query 可见的 key：记忆区有效位 + 当前图像 + 有效文本 + 20 个动作
    valid_a = np.zeros(P + AQ, bool)
    valid_a[:MEM] = mem_mask
    valid_a[MEM:MEM + 512] = True
    valid_a[MEM + 512:MEM + 512 + TXT_LEN] = True
    valid_a[P:] = True
    logits = rng.normal(size=(STEPS, LAYERS, HEADS, AQ, P + AQ))
    if boost:  # 给指定原始序号段加 logit，造出可预期的高注意力
        for raw_lo, raw_hi, val in boost.get("raw", []):
            cols = np.flatnonzero((order >= raw_lo) & (order < raw_hi))
            logits[..., cols] += val
    probs_action = _softmax_rows(logits, valid_a).astype(np.float16)
    valid_t = valid_a[:P].copy()
    probs_text = _softmax_rows(rng.normal(size=(LAYERS, HEADS, TQ, P)), valid_t)
    probs_text[:, :, TXT_LEN:] = 0.0  # padding 行全 0
    probs_text = probs_text.astype(np.float16)
    if actions is None:
        actions = np.tile(np.array([0, 0.4, 0, -2, 0, 2.4, 0.8, 1.0], np.float32), (20, 1))
    rec = dict(probs_action=probs_action, probs_text=probs_text, P=P, MEM=MEM, step_idx=step_idx,
               exec_start_idx=0, infer_seq=infer_seq, mem_mask=mem_mask, frame_times=frame_times,
               motion_times=motion_times, motion_mask=motion_mask, txt_len=TXT_LEN,
               actions=actions.astype(np.float32), state=actions[0].astype(np.float32),
               top_frames=np.array([0, 1, 2], np.int32))
    if mem_order is not None:
        rec["mem_order"] = mem_order
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **rec)
    return rec


def _reference(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """不经 split_streams 的独立参照：在原始序号空间里求帧／窗质量。"""
    a = rec["probs_action"].astype(np.float64).mean((0, 1, 2, 3))
    MEM = rec["MEM"]
    order = rec.get("mem_order", np.arange(MEM))
    raw = np.zeros(MEM)
    raw[order] = a[:MEM]  # 逆置换回原始序号
    frames = raw[:512].reshape(32, 16).sum(1)
    motion = raw[512:]
    return frames, motion


@pytest.mark.parametrize("motion", [True, False])
def test_split_streams_sum_and_mapping(tmp_path, motion):
    rec = make_npz(tmp_path / "x.npz", motion=motion, seed=1)
    with np.load(tmp_path / "x.npz") as z:
        st = agg.split_streams(z)
    m = st["mass"]
    assert st["pad"] <= 1e-6
    assert abs(sum(m.values()) - 1.0) <= 1e-4
    ref_f, ref_m = _reference(rec)
    # 帧号反推：f 对应 frame_times[f]，padding 帧不出现
    assert [x["t"] for x in st["frame"]] == rec["frame_times"][:30].tolist()
    np.testing.assert_allclose([x["mass"] for x in st["frame"]], ref_f[:30], atol=1e-9)
    if motion:
        assert [x["w"] for x in st["motion"]] == list(range(10))
        assert [x["t"] for x in st["motion"]] == (np.arange(10) * 16).tolist()
        np.testing.assert_allclose([x["mass"] for x in st["motion"]], ref_m[:10], atol=1e-9)
        assert st["has_motion"] and st["MEM"] == 608 and st["P"] == 1184
    else:
        assert st["motion"] == [] and m["motion"] == 0.0 and st["P"] == 1088
    assert m["action"] > 0 and m["image_base"] > 0 and m["image_wrist"] > 0 and m["text"] > 0


def test_split_streams_text_query_and_axes(tmp_path):
    make_npz(tmp_path / "x.npz", motion=True, seed=2)
    with np.load(tmp_path / "x.npz") as z:
        st = agg.split_streams(z, query="text")
        assert st["mass"]["action"] == 0.0 and abs(sum(st["mass"].values()) - 1) <= 1e-4
        st2 = agg.split_streams(z, steps="last", layers="0-5", heads="0,3")
        assert abs(sum(st2["mass"].values()) - 1) <= 1e-4


def test_pad_mass_assert(tmp_path):
    make_npz(tmp_path / "x.npz", motion=True, seed=3)
    with np.load(tmp_path / "x.npz") as z:
        bad = {k: z[k] for k in z.files}
    # 人为把一部分质量挪到 padding 列：应触发 pad 断言
    pa = bad["probs_action"].astype(np.float32)
    pad_cols = np.flatnonzero(~bad["mem_mask"])
    pa[..., pad_cols[0]] = 0.01
    bad["probs_action"] = pa.astype(np.float16)
    with pytest.raises(AssertionError):
        agg.split_streams(bad)


def _gripper_lift_actions():
    acts = np.tile(np.array([0, 0.4, 0, -2, 0, 2.4, 0.8, 1.0], np.float32), (20, 1))
    acts[8:, 7] = -1.0  # 第 8 行由开到闭
    return acts


def _write_episode(root: Path, name: str, motion: bool, key_boost: bool) -> Path:
    d = root / name
    boost = None
    if key_boost:
        # 关键推理：帧流放大裸露窗口内的帧（原始帧 0、1 → t=0、10），motion 流放大起点 64、80 的窗（与 [64,114) 相交）
        boost = {"raw": [(0, 32, 3.0)] + ([(512 + 4, 512 + 6, 3.0)] if motion else [])}
    make_npz(d / "infer_0000.npz", motion=motion, seed=10, infer_seq=0, step_idx=200)
    make_npz(d / "infer_0001.npz", motion=motion, seed=11, infer_seq=1, step_idx=300, boost=boost,
             actions=_gripper_lift_actions())
    (d / "episode.json").write_text(json.dumps({"task_name": "ButtonUnmaskSwap", "episode_id": 4,
                                                "model_kind": "stage2_pool4_concat" if motion else "frame_sampling"}))
    # 合成 sim_eef_pose.json：infer 1 在闭合后 z 抬升 5 cm；infer 0 不动
    flat = [0.20] * 20
    lift = [0.20] * 8 + [0.10] + [0.10 + 0.005 * i for i in range(1, 12)]
    (d / "sim_eef_pose.json").write_text(json.dumps({
        "task_name": "ButtonUnmaskSwap", "episode_id": 4, "swap_times": 1,
        "infers": {"0": {"step_idx": 200, "state_z": 0.2, "action_z": flat},
                   "1": {"step_idx": 300, "state_z": 0.2, "action_z": lift}}}))
    return d


def test_candidates_key_step_and_plot(tmp_path, capsys):
    runs = tmp_path / "runs"
    _write_episode(runs, "nonce-base", motion=False, key_boost=True)
    _write_episode(runs, "nonce-mot", motion=True, key_boost=True)
    rec_dir = tmp_path / "records"
    rc = agg.main(["--runs", str(runs), "--out", str(rec_dir), "--candidates", "--key-infer", "1"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "KEY_CANDIDATES task=ButtonUnmaskSwap ep=4 model=frame_sampling infers=[1]" in out
    assert "KEY_CANDIDATES task=ButtonUnmaskSwap ep=4 model=stage2_pool4_concat infers=[1]" in out
    assert out.strip().splitlines()[-1].startswith("ATTN_AGG=PASS episodes=2")
    kl = [ln for ln in out.splitlines() if ln.startswith("KEY_STEP=")]
    assert len(kl) == 2
    base_line = next(ln for ln in kl if "frame_sampling" in ln)
    assert "swap_share=na swap_uniform=na" in base_line

    summary = json.loads((rec_dir / "summary.json").read_text())
    assert (rec_dir / "nonce-mot" / "streams.json").is_file()
    ks = {k["has_motion"]: k for k in summary["key_steps"]}
    # 均匀期望：30 个有效帧里 t ∈ [0,32) 的个数（linspace(0,300,30) → t=0,10,20,31 共 4 个）
    ft = np.linspace(0, 300, 30).astype(int)
    assert ks[False]["vis_uniform"] == pytest.approx(np.sum(ft < 32) / 30)
    assert ks[False]["vis_share"] > ks[False]["vis_uniform"]  # 放大过裸露帧
    # motion：10 个有效窗起点 0,16,…,144；与 [64,114) 相交者 t<114 且 t+33>64 → 32,48,64,80,96,112 共 6 个
    assert ks[True]["swap_uniform"] == pytest.approx(6 / 10)
    assert ks[True]["swap_share"] > ks[True]["swap_uniform"]

    plot = _load("plot_attn")
    figs = tmp_path / "figs"
    rc = plot.main(["--records", str(rec_dir), "--out", str(figs), "--only", "key_step"])
    pout = capsys.readouterr().out
    assert rc == 0, pout
    assert pout.strip().splitlines()[-1] == "ATTN_PLOT=PASS figs=1"
    png = figs / "fig_key_step.png"
    assert png.is_file() and png.stat().st_size > 10_000


def test_candidates_need_lift(tmp_path):
    rec0 = make_npz(tmp_path / "e" / "infer_0000.npz", motion=False, seed=5, actions=_gripper_lift_actions())
    npzs = {0: rec0}
    flat = {"infers": {"0": {"step_idx": 300, "state_z": 0.2, "action_z": [0.2] * 8 + [0.1] + [0.11] * 11}}}
    assert agg.find_candidates(npzs, flat) == []  # 只抬 1 cm，不够 3 cm
