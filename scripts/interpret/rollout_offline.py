"""真实 rollout 记录的离线计算（计划 1105-stopcube-real-rollout-plan.html，K4、K6、K8、K9）。

对 <attn-root>/<nonce>/rollout_<seq>.npz 逐份执行：
  0. 由 nonce 找 <run-root>/attempts/*/client/traces/<nonce>/step_trace.jsonl，按 infer_seq 取 row_in_chunk==0 那一行；
  1. 原路径重放（ROLLOUT_REPLAY，容差 1e-6，不放宽）；
  2. 10 份独立噪声走原路径得到 P 用动作（只存不执行，高度换算不在本模块做）；
  3. 用记录里的实际噪声走展开路径，取注意力组质量和隐状态组均值；
  4. 同输入同噪声比较隐状态捕获开和关（HIDDEN_PARITY，必须逐位相同）。
任何一步不通过都立刻停止并非零退出。不替换、不遮挡任何输入。JAX 确定性环境变量由外部脚本给。
只处理 <run-root>/accepted/*.terminal.json 登记过的 nonce；重试或中断留下的旧 nonce 打印 OFFLINE_SKIP 并计数，不处理。

用法：
  python -m scripts.interpret.rollout_offline --attn-root <O>/<模型>/attn --run-root <O>/<模型>/run \
    --output <O>/<模型>/offline --ckpt <dir> --config <配置名> --seed 42 \
    [--parity all|first] [--query-batch-size 2] [--episodes StopCube:16 ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

from scripts.interpret.analysis.offline_mechanism import aggregate_attention, fingerprint, token_groups
from scripts.interpret.offline_query import query_noise, replay_query, sample_predictions, unpack_observation

P_PREDICTIONS = 10
HIDDEN_GROUPS = ("历史视觉", "motion", "当前视觉", "文字")     # hidden_* 第二维的固定顺序


def load_accepted(run_root):
    """返回 {episode_nonce: episode_key}，来自 run_local.collect 汇到 accepted/ 的 terminal。"""
    accepted = {}
    for path in sorted((Path(run_root) / "accepted").glob("*.terminal.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        accepted[record["episode_nonce"]] = record["episode_key"]
    return accepted


def load_trace(run_root, nonce):
    """返回 {infer_seq: row_in_chunk==0 的 step 行}；轨迹找不到或不唯一就报错，不补造。"""
    paths = sorted(Path(run_root).glob(f"attempts/*/client/traces/{nonce}/step_trace.jsonl"))
    if len(paths) != 1:
        raise SystemExit(f"nonce={nonce} 的 step_trace.jsonl 应恰有 1 份，实际 {len(paths)} 份：{paths}")
    rows = {}
    with paths[0].open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "step" or row["row_in_chunk"] != 0:
                continue
            if row["infer_seq"] in rows:
                raise SystemExit(f"{paths[0]} 中 infer_seq={row['infer_seq']} 有多条 row_in_chunk==0 的行")
            rows[row["infer_seq"]] = row
    if not rows:
        raise SystemExit(f"{paths[0]} 没有 step 行")
    return rows


def motion_window_groups(observation, groups, motion_times, mem):
    """motion窗:<start>：按 mem_order 把记忆位映射回 motion token，同一 motion_times 值归一组；基线不产生。"""
    positions = np.flatnonzero(groups["motion"][:mem])
    if positions.size == 0:
        return {}
    n_static = np.asarray(observation.static_mask).reshape(-1).size
    order = np.asarray(observation.mem_order).reshape(-1).astype(int)
    times = np.asarray(motion_times).reshape(-1)
    tokens = order[positions] - n_static
    if tokens.max() >= times.size:
        raise ValueError(f"有效 motion token {tokens.max()} 超出 motion_times 长度 {times.size}")
    token_times = times[tokens]
    if (token_times < 0).any():
        raise ValueError("有效 motion token 对应的 motion_times 是无效位")
    result = {}
    for start in sorted(set(token_times.tolist())):
        mask = np.zeros(len(groups["motion"]), bool)
        mask[positions[token_times == start]] = True
        result[f"motion窗:{start}"] = mask
    return result


def build_groups(observation, layout, record, tokens_per_frame):
    groups = token_groups(observation, layout, record["frame_times"], tokens_per_frame)
    groups.update(motion_window_groups(observation, groups, record["motion_times"], layout["MEM"]))
    hidden_masks = np.stack([groups[name][:layout["P"]] for name in HIDDEN_GROUPS])
    return groups, hidden_masks


def existing_result(path, input_sha256):
    """输出已存在时返回其读数；指纹与当前输入不一致就报错，不覆盖。"""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["input_sha256"]) != input_sha256:
            raise SystemExit(f"{path} 的 input_sha256 与当前输入不一致，不覆盖")
        return {"same_path_maxabs": float(archive["same_path_maxabs"]),
                "attn_path_maxabs": float(archive["attn_path_maxabs"])}


def process_record(policy, adapter, path, output_path, trace, input_sha256, *, seed, query_batch_size, check_parity):
    """算一份记录并写出；返回读数。parity_maxabs 为 None 表示本份不比较。"""
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    with np.load(path, allow_pickle=False) as archive:
        record = {key: archive[key] for key in archive.files}
    infer_seq = int(record["infer_seq"])
    row = trace.get(infer_seq)
    if row is None:
        raise SystemExit(f"{path}：轨迹里没有 infer_seq={infer_seq} 的 row_in_chunk==0 行")
    raw_state = np.asarray(row["state_before"], dtype=np.float32)
    if raw_state.shape != (8,):
        raise SystemExit(f"{path}：state_before 形状 {raw_state.shape} 不是 (8,)")

    replay = replay_query(policy, path)
    same_path = float(replay["same_path_maxabs"])
    if replay["status"] != "PASS":
        print(f"ROLLOUT_REPLAY=FAIL points=1 same_path_maxabs={same_path:.9g} tol=1e-6 path={path}", flush=True)
        raise SystemExit(1)

    observation = jax.tree.map(jnp.asarray, unpack_observation(record, HistAugObservation))
    kwargs = json.loads(str(record["sample_kwargs"]))
    if set(kwargs) - {"num_steps"}:
        raise ValueError(f"适配器不支持原采样参数：{sorted(kwargs)}")
    num_steps = int(kwargs.get("num_steps", 10))

    noise = query_noise(seed, row["episode_key"], infer_seq, P_PREDICTIONS,
                        int(policy._model.action_horizon), int(policy._model.action_dim))
    _, p_actions = sample_predictions(policy, observation, noise, sample_rng=jax.random.PRNGKey(seed),
                                      batch_size=query_batch_size, sample_kwargs=kwargs)

    layout = adapter.layout(observation)
    tokens_per_frame = int(policy.config.token_per_image) * int(policy.config.num_views)
    groups, hidden_masks = build_groups(observation, layout, record, tokens_per_frame)
    rng = jnp.asarray(record["sample_rng"], dtype=jnp.uint32)
    actual_noise = jnp.asarray(record["noise"])
    output = adapter.run(rng, observation, noise=actual_noise, num_steps=num_steps, out_f16=False,
                         capture_hidden=True, group_masks=hidden_masks)
    x_0, probs_action, probs_text = jax.device_get(output[:3])
    hidden = jax.device_get(output[-1])
    attn_path = float(np.max(np.abs(np.asarray(x_0) - record["actions_normalized"])))
    if not np.isfinite(attn_path):
        raise ValueError(f"{path}：展开路径动作含非有限值")

    parity_maxabs = None
    if check_parity:
        plain = jax.device_get(adapter.run(rng, observation, noise=actual_noise, num_steps=num_steps, out_f16=False)[:3])
        pairs = list(zip((x_0, probs_action, probs_text), plain))
        if not all(np.array_equal(np.asarray(on), np.asarray(off)) for on, off in pairs):
            parity_maxabs = max(float(np.max(np.abs(np.asarray(on, dtype=np.float64) - np.asarray(off, dtype=np.float64))))
                                for on, off in pairs)
            print(f"HIDDEN_PARITY=FAIL points=1 maxabs={parity_maxabs:.9g} path={path}", flush=True)
            raise SystemExit(1)
        parity_maxabs = 0.0

    attention = aggregate_attention(probs_action, groups)
    names = list(attention)
    result = dict(
        group_names=np.asarray(names),
        group_mass=np.asarray([attention[name]["mass"] for name in names], dtype=np.float32),
        group_tokens_all=np.asarray([attention[name]["tokens"] for name in names], dtype=np.int64),
        hidden_prefix=np.asarray(hidden["hidden_prefix"]),
        hidden_action=np.asarray(hidden["hidden_action"]),
        hidden_group_tokens=np.asarray(hidden["group_tokens"]),
        p_actions=p_actions,
        raw_state=raw_state,
        episode_key=np.str_(row["episode_key"]),
        infer_seq=np.int64(infer_seq),
        step_idx=np.int64(record["step_idx"]),
        chunk_start_env_step=np.int64(row["chunk_start_env_step"]),
        same_path_maxabs=np.float64(same_path),
        attn_path_maxabs=np.float64(attn_path),
        input_sha256=np.str_(input_sha256),
        model_id=np.str_(str(record["model_id"])),
        frame_times=record["frame_times"],
        motion_times=record["motion_times"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.stem}.{uuid.uuid4().hex}.tmp.npz"
    np.savez(temporary, **result)
    os.replace(temporary, output_path)
    return {"same_path_maxabs": same_path, "attn_path_maxabs": attn_path, "parity_maxabs": parity_maxabs}


def run(policy, adapter, args):
    started = time.perf_counter()
    wanted = set(args.episodes) if args.episodes else None
    accepted = load_accepted(args.run_root)
    seen = set()
    written = skipped = skipped_nonce = parity_points = 0
    same_path_max = attn_path_max = 0.0
    for directory in sorted(path for path in Path(args.attn_root).iterdir() if path.is_dir()):
        paths = sorted(directory.glob("rollout_*.npz"))
        if not paths:
            continue
        nonce = directory.name
        episode_key = accepted.get(nonce)
        if episode_key is None:
            print(f"OFFLINE_SKIP nonce={nonce} reason=no_terminal", flush=True)
            skipped_nonce += 1
            continue
        if wanted is not None and episode_key not in wanted:
            continue
        trace = load_trace(args.run_root, nonce)
        seen.add(episode_key)
        episode_started = time.perf_counter()
        episode_written = episode_skipped = 0
        for index, path in enumerate(paths):
            seq = re.fullmatch(r"rollout_(\d+)\.npz", path.name).group(1)
            output_path = Path(args.output) / nonce / f"offline_{seq}.npz"
            input_sha256 = fingerprint(path)["sha256"]
            item = existing_result(output_path, input_sha256)
            if item is not None:
                episode_skipped += 1
            else:
                item = process_record(policy, adapter, path, output_path, trace, input_sha256, seed=args.seed,
                                      query_batch_size=args.query_batch_size,
                                      check_parity=args.parity == "all" or index == 0)
                episode_written += 1
                parity_points += item["parity_maxabs"] is not None
            same_path_max = max(same_path_max, item["same_path_maxabs"])
            attn_path_max = max(attn_path_max, item["attn_path_maxabs"])
        written += episode_written
        skipped += episode_skipped
        print(f"OFFLINE_EPISODE episode={episode_key} nonce={nonce} records={episode_written} skipped={episode_skipped} "
              f"seconds={time.perf_counter() - episode_started:.1f}", flush=True)
    missing = sorted(wanted - seen) if wanted is not None else []
    if missing or not written + skipped:
        raise SystemExit(f"没有可处理的记录或缺少指定的局：records={written + skipped} missing={missing}")
    print(f"ROLLOUT_REPLAY=PASS points={written + skipped} same_path_maxabs={same_path_max:.9g} tol=1e-6", flush=True)
    print(f"HIDDEN_PARITY=PASS points={parity_points} maxabs=0" if parity_points else "HIDDEN_PARITY=SKIPPED points=0",
          flush=True)
    print(f"OFFLINE_RUN=PASS records={written} skipped={skipped} skipped_nonce={skipped_nonce} "
          f"attn_path_maxabs={attn_path_max:.9g} "
          f"seconds={time.perf_counter() - started:.1f}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attn-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parity", choices=("all", "first"), default="first")
    parser.add_argument("--query-batch-size", type=int, default=2)
    parser.add_argument("--episodes", nargs="+")
    args = parser.parse_args(argv)
    # 同 offline_query.main：离线没有编码请求，不启动 motion sidecar。
    from scripts.interpret.attn_adapter import AttnAdapter, _ZeroMotion, _MV_DIR
    import sys
    sys.path.insert(0, str(_MV_DIR))
    from mv_model_adapter import build_policy
    from mme_vla_suite.training import config

    policy = build_policy(config.get_config(args.config), args.ckpt.resolve(), seed=args.seed,
                          motion_factory=lambda **kwargs: _ZeroMotion())
    return run(policy, AttnAdapter(policy), args)


if __name__ == "__main__":
    raise SystemExit(main())
