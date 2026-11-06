"""第一次取 bin 窗口的离线注意力（计划 1105-swap-first-bin-attention-plan.html，第二部分 1.4）。

只处理 windows.json 列出的 rollout_<seq>.npz，逐条：
  1. 原路径重放，超 1e-6 打 ROLLOUT_REPLAY=FAIL 并非零退出；
  2. 用记录的 sample_rng、noise 调 AttnAdapter.run(..., out_f16=True, save_prefix=True)；
  3. 写 <output>/<nonce>/prefix_<seq>.npz，再写 infer_<seq>.npz（后者同时是完成标记）。

键表沿用 serve_policy_attn.NPZ_KEYS、PREFIX_NPZ_KEYS，infer 另存 same_path_maxabs、attn_path_maxabs、input_sha256，
motion 模型再存 motion_emb、motion_pos。RolloutRecorder 没记的键按 AttnRecorder.build_record 同算法重建：
state 是物理状态，取轨迹 row_in_chunk==0 行的 state_before，不是观测里归一化后的 state；exec_start_idx 取同一行。
注意力由展开路径算出，动作来自记录的原路径；两者差值写进 attn_path_maxabs，如实报告，不算逐位等价。

行和与 padding 判据同上一轮 validate_capture：有效行和偏离 1 不超过 5e-4，padding key 与 padding query 行质量为 0；
不合格立刻 ATTN_WINDOW=FAIL 并非零退出。续跑按 input_sha256 跳过，指纹不一致报错不覆盖。

用法（JAX 确定性环境变量由外部脚本给；两卡各一个进程）：
  python -m scripts.interpret.first_bin_offline --windows <O>/<模型>/windows/windows.json \
    --run-root <O>/<模型>/g0/run --run-root <O>/<模型>/g1/run \
    --attn-root <O>/<模型>/g0/attn --attn-root <O>/<模型>/g1/attn \
    --output <O>/<模型>/offline --ckpt <dir> --config <配置名> --seed 42 --shard g0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

import numpy as np

from scripts.interpret.analysis.offline_mechanism import fingerprint
from scripts.interpret.offline_query import replay_query, unpack_observation
from scripts.interpret.rollout_offline import load_accepted, load_trace
from scripts.interpret.serve_policy_attn import (ACTION_HORIZON, MOTION_INPUT_NPZ_KEYS, NUM_HEADS, NUM_LAYERS, NUM_STEPS,
                                                 STATE_DIM, _pad_times, top_frame_indices)

WINDOW_BEFORE = 3         # 窗口 [max(0, G-3), G]
WINDOW_STATUSES = ("complete", "incomplete")      # 有窗口；incomplete 是 G<3、窗口不足 4 次
NO_WINDOW_STATUSES = ("no_pick", "no_close_before_pick")
TOP_FRAMES = 3            # 同 serve_policy_attn --save-top-frames 默认值
ROWSUM_TOL = 5e-4         # 同上一轮 validate_capture
EXTRA_NPZ_KEYS = ("same_path_maxabs", "attn_path_maxabs", "input_sha256")


def load_windows(path):
    """windows.json 顶层固定为 {"schema_version": 1, "episodes": [...]}。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("episodes"), list):
        raise SystemExit(f"{path} 的结构不是 schema_version=1 的 episodes 列表")
    return value["episodes"]


def check_window(entry):
    """window 必须正好是 [max(0,G-3)…G] 的逐项列表；不符就报错，不猜。"""
    G = int(entry["G"])
    expected = list(range(max(0, G - WINDOW_BEFORE), G + 1))
    if list(entry["window"]) != expected:
        raise SystemExit(f"{entry['episode_key']}：window={entry['window']} 与 G={G} 应得的 {expected} 不符")
    return expected


def locate_episodes(run_roots):
    """返回 {nonce: [(episode_key, run_root), ...]}，来自各分片 accepted。"""
    found = {}
    for run_root in run_roots:
        for nonce, episode_key in load_accepted(run_root).items():
            found.setdefault(nonce, []).append((episode_key, run_root))
    return found


def locate_record(attn_roots, nonce, seq):
    paths = [Path(root) / nonce / f"rollout_{seq:04d}.npz" for root in attn_roots]
    paths = [path for path in paths if path.is_file()]
    if len(paths) != 1:
        raise SystemExit(f"nonce={nonce} seq={seq} 的 rollout 记录应恰有 1 份，实际 {len(paths)} 份：{paths}")
    return paths[0]


def build_record(policy, record, row, layout, probs_action, probs_text, observation):
    """与 AttnRecorder.build_record 同算法，输入换成 rollout 记录；probs 已去 batch 轴。"""
    config = policy.config
    budget = int(config.budget)
    tokens_per_frame = int(config.token_per_image) * int(config.num_views)
    motion_enabled = bool(policy.motion_enabled)
    P, MEM = int(layout["P"]), int(layout["MEM"])
    expected_mem = budget + (int(config.motion.budget) if motion_enabled else 0)
    if MEM != expected_mem:
        raise RuntimeError(f"MEM={MEM} 与配置 budget+motion.budget={expected_mem} 不符")
    probs_action = np.asarray(probs_action).astype(np.float16, copy=False)
    probs_text = np.asarray(probs_text).astype(np.float16, copy=False)
    if probs_action.shape != (NUM_STEPS, NUM_LAYERS, NUM_HEADS, ACTION_HORIZON, P + ACTION_HORIZON):
        raise RuntimeError(f"probs_action 形状 {probs_action.shape} 不符")
    if probs_text.shape != (NUM_LAYERS, NUM_HEADS, int(layout["TXT"]), P):
        raise RuntimeError(f"probs_text 形状 {probs_text.shape} 不符")

    static_mask = np.asarray(observation.static_mask).reshape(-1).astype(bool)
    if motion_enabled:
        mem_order = np.asarray(observation.mem_order).reshape(-1).astype(np.int32)
        motion_mask = np.asarray(observation.motion_mask).reshape(-1).astype(bool)
        full_mask = np.concatenate([static_mask, motion_mask])
    else:
        mem_order = np.arange(MEM, dtype=np.int32)
        motion_mask = np.zeros(0, bool)
        full_mask = static_mask
    if mem_order.shape != (MEM,) or full_mask.shape != (MEM,):
        raise RuntimeError(f"mem_order {mem_order.shape} / 记忆区 mask {full_mask.shape} 与 MEM={MEM} 不符")

    frame_times = _pad_times(record["frame_times"], budget // tokens_per_frame)
    actions = np.asarray(record["actions"], dtype=np.float32)[0]
    state = np.asarray(row["state_before"], dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,):
        raise RuntimeError(f"state_before 形状 {state.shape} 不是 ({STATE_DIM},)")
    result = {
        "P": np.int64(P), "MEM": np.int64(MEM), "step_idx": np.int64(record["step_idx"]),
        "exec_start_idx": np.int64(row["exec_start_idx"]), "infer_seq": np.int64(record["infer_seq"]),
        "mem_order": mem_order, "mem_mask": full_mask[mem_order], "frame_times": frame_times,
        "motion_times": np.asarray(record["motion_times"], dtype=np.int32), "motion_mask": motion_mask,
        "txt_len": np.int64(int(np.asarray(observation.tokenized_prompt_mask).sum())),
        "actions": actions, "state": state,
        "top_frames": top_frame_indices(probs_action, mem_order, frame_times, tokens_per_frame, TOP_FRAMES),
        "probs_action": probs_action, "probs_text": probs_text,
    }
    if motion_enabled:
        for key in MOTION_INPUT_NPZ_KEYS:
            result[key] = np.asarray(getattr(observation, key))[0]
    return result


def attention_problem(record, probs_prefix):
    """返回 (行和最大误差, 问题描述或 None)；判据同 validate_capture，另按同法查文字 query。"""
    probs_action, probs_text = record["probs_action"], record["probs_text"]
    for name, value in (("probs_action", probs_action), ("probs_text", probs_text), ("probs_prefix", probs_prefix)):
        if not np.isfinite(value).all():
            return float("inf"), f"{name}_non_finite"
    P, MEM, TXT = int(record["P"]), int(record["MEM"]), probs_text.shape[2]
    text_valid = np.arange(TXT) < int(record["txt_len"])
    prefix_valid = np.r_[np.asarray(record["mem_mask"], bool), np.ones(P - MEM - TXT, bool), text_valid]
    action_keys = np.r_[prefix_valid, np.ones(ACTION_HORIZON, bool)]

    action_sums = probs_action.sum(-1, dtype=np.float32)
    text_sums = probs_text.sum(-1, dtype=np.float32)
    prefix_sums = probs_prefix.sum(-1, dtype=np.float32)
    error = max(float(np.abs(action_sums - 1).max()),
                float(np.abs(text_sums[..., text_valid] - 1).max(initial=0)),
                float(np.abs(prefix_sums[..., prefix_valid] - 1).max(initial=0)))
    if error > ROWSUM_TOL:
        return error, "rowsum"
    if float(probs_action[..., ~action_keys].sum(-1, dtype=np.float32).max(initial=0)) != 0:
        return error, "action_padding_key"
    if np.any(text_sums[..., ~text_valid] != 0):
        return error, "text_padding_query"
    if np.any(prefix_sums[..., ~prefix_valid] != 0):
        return error, "prefix_padding_query"
    return error, None


def stop_attention(path, error, reason, attn_path=None):
    print(f"ATTN_WINDOW=FAIL records=1 rowsum_err={error:.9g} attn_path_maxabs={attn_path} reason={reason} path={path}", flush=True)
    raise SystemExit(1)


def load_finished(infer_path, prefix_path, input_sha256):
    """已完成就重新查一遍注意力并返回读数；没完成返回 None；指纹不一致报错，不覆盖。"""
    if not infer_path.exists():
        return None
    with np.load(infer_path, allow_pickle=False) as archive:
        if str(archive["input_sha256"]) != input_sha256:
            raise SystemExit(f"{infer_path} 的 input_sha256 与当前输入不一致，不覆盖")
        if not prefix_path.exists():
            return None
        record = {key: archive[key] for key in archive.files}
    with np.load(prefix_path, allow_pickle=False) as archive:
        probs_prefix = archive["probs_prefix"]
    error, problem = attention_problem(record, probs_prefix)
    if problem is not None:
        stop_attention(infer_path, error, problem)
    return {"same_path_maxabs": float(record["same_path_maxabs"]), "attn_path_maxabs": float(record["attn_path_maxabs"]),
            "rowsum_err": error}


def write_npz(path, values):
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
    np.savez(temporary, **values)
    os.replace(temporary, path)


def process_record(policy, adapter, path, infer_path, prefix_path, row, input_sha256):
    """重放通过后算注意力并写出；返回读数。"""
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    with np.load(path, allow_pickle=False) as archive:
        record = {key: archive[key] for key in archive.files}
    replay = replay_query(policy, path)
    same_path = float(replay["same_path_maxabs"])
    if replay["status"] != "PASS":
        print(f"ROLLOUT_REPLAY=FAIL points=1 same_path_maxabs={same_path:.9g} tol=1e-6 path={path}", flush=True)
        raise SystemExit(1)

    observation = jax.tree.map(jnp.asarray, unpack_observation(record, HistAugObservation))
    kwargs = json.loads(str(record["sample_kwargs"]))
    if set(kwargs) - {"num_steps"}:
        raise ValueError(f"适配器不支持原采样参数：{sorted(kwargs)}")
    num_steps = int(kwargs.get("num_steps", NUM_STEPS))
    x_0, probs_action, probs_text, n_done, layout, probs_prefix = adapter.run(
        jnp.asarray(record["sample_rng"], dtype=jnp.uint32), observation, noise=jnp.asarray(record["noise"]),
        num_steps=num_steps, out_f16=True, save_prefix=True)
    x_0, probs_action, probs_text, n_done, probs_prefix = jax.device_get((x_0, probs_action, probs_text, n_done, probs_prefix))
    if int(n_done) != num_steps:
        raise RuntimeError(f"去噪循环实际步数 {int(n_done)} != num_steps {num_steps}")

    # 去 batch 轴，同 AttnAdapter.make_sample_fn
    result = build_record(policy, record, row, layout, np.asarray(probs_action)[:, :, 0], np.asarray(probs_text)[:, 0], observation)
    P = int(result["P"])
    probs_prefix = np.asarray(probs_prefix)[:, 0].astype(np.float16, copy=False)
    if probs_prefix.shape != (NUM_LAYERS, NUM_HEADS, P, P):
        raise RuntimeError(f"probs_prefix 形状 {probs_prefix.shape} 不符 {(NUM_LAYERS, NUM_HEADS, P, P)}")

    attn_path = float(np.max(np.abs(np.asarray(x_0, dtype=np.float32) - record["actions_normalized"])))
    if not np.isfinite(attn_path):
        stop_attention(path, float("inf"), "x_0_non_finite", attn_path)
    error, problem = attention_problem(result, probs_prefix)
    if problem is not None:
        stop_attention(path, error, problem, attn_path)

    result.update(same_path_maxabs=np.float64(same_path), attn_path_maxabs=np.float64(attn_path),
                  input_sha256=np.str_(input_sha256))
    infer_path.parent.mkdir(parents=True, exist_ok=True)
    write_npz(prefix_path, {"probs_prefix": probs_prefix, "P": result["P"], "MEM": result["MEM"],
                            "infer_seq": result["infer_seq"], "step_idx": result["step_idx"]})
    write_npz(infer_path, result)
    return {"same_path_maxabs": same_path, "attn_path_maxabs": attn_path, "rowsum_err": error}


def run(policy, adapter, args):
    entries = [entry for entry in load_windows(args.windows) if entry["shard"] == args.shard]
    episodes = locate_episodes(args.run_root)
    keys = [entry["episode_key"] for entry in entries]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"windows.json 的分片 {args.shard} 有重复的局")
    written = skipped = no_window = 0
    same_path_max = attn_path_max = rowsum_max = 0.0
    for entry in entries:
        episode_key = entry["episode_key"]
        status = entry["window_status"]
        if status in NO_WINDOW_STATUSES:     # 这些局没有窗口，windows.json 已计入判定行
            print(f"OFFLINE_SKIP episode={episode_key} reason={status}", flush=True)
            no_window += 1
            continue
        if status not in WINDOW_STATUSES:
            raise SystemExit(f"{episode_key}：未知 window_status={status}")
        window = check_window(entry)
        nonce = entry["episode_nonce"]
        located = episodes.get(nonce, [])
        if len(located) != 1 or located[0][0] != episode_key:
            raise SystemExit(f"{episode_key} 的 nonce={nonce} 在 accepted 里应恰有 1 条且局号一致，实际 {located}")
        run_root = located[0][1]
        trace = load_trace(run_root, nonce)
        for seq in window:
            row = trace.get(seq)
            if row is None:
                raise SystemExit(f"{episode_key}：轨迹里没有 infer_seq={seq} 的 row_in_chunk==0 行")
            if row["episode_key"] != episode_key:
                raise SystemExit(f"{episode_key}：轨迹行写的是 {row['episode_key']}")
            if row.get("exec_start_idx") is None:
                raise SystemExit(f"{episode_key}：infer_seq={seq} 的轨迹行缺 exec_start_idx")
            path = locate_record(args.attn_root, nonce, seq)
            infer_path = Path(args.output) / nonce / f"infer_{seq:04d}.npz"
            prefix_path = Path(args.output) / nonce / f"prefix_{seq:04d}.npz"
            input_sha256 = fingerprint(path)["sha256"]
            item = load_finished(infer_path, prefix_path, input_sha256)
            if item is not None:
                skipped += 1
            else:
                item = process_record(policy, adapter, path, infer_path, prefix_path, row, input_sha256)
                written += 1
            same_path_max = max(same_path_max, item["same_path_maxabs"])
            attn_path_max = max(attn_path_max, item["attn_path_maxabs"])
            rowsum_max = max(rowsum_max, item["rowsum_err"])
            print(f"ROLLOUT_REPLAY_RECORD episode={episode_key} nonce={nonce} infer_seq={seq} "
                  f"same_path_maxabs={item['same_path_maxabs']:.9g} attn_path_maxabs={item['attn_path_maxabs']:.9g} "
                  f"rowsum_err={item['rowsum_err']:.9g}", flush=True)
    records = written + skipped
    if not records:
        raise SystemExit(f"分片 {args.shard} 没有可处理的记录：episodes={len(entries)} no_window={no_window}")
    print(f"OFFLINE_RUN shard={args.shard} episodes={len(entries)} no_window={no_window} written={written} skipped={skipped}", flush=True)
    print(f"ROLLOUT_REPLAY=PASS points={records} same_path_maxabs={same_path_max:.9g} tol=1e-6", flush=True)
    print(f"ATTN_WINDOW=PASS records={records} rowsum_err={rowsum_max:.9g} attn_path_maxabs={attn_path_max:.9g}", flush=True)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--attn-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", choices=("g0", "g1"), required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # 同 rollout_offline.main：离线没有编码请求，不启动 motion sidecar。
    from scripts.interpret.attn_adapter import AttnAdapter, _ZeroMotion, _MV_DIR
    import sys
    sys.path.insert(0, str(_MV_DIR))
    from mv_model_adapter import build_policy
    from mme_vla_suite.training import config

    policy = build_policy(config.get_config(args.config), args.ckpt.resolve(), seed=args.seed,
                          motion_factory=lambda **kwargs: _ZeroMotion())
    return run(policy, AttnAdapter(policy, save_prefix=True), args)


if __name__ == "__main__":
    raise SystemExit(main())
