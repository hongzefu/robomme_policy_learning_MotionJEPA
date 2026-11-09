"""StopCube 悬停期探针的示教回放与只做前缀的隐状态捕获（计划 1105-stopcube-hover-probe-plan.html，A3、A5、A8、B4 W2）。

bf16 的 policy 逐帧生成记忆（motion 模型走真实编码器），float32 模型只做前缀前向，不采样、不去噪。

用法（公共参数相同；check 不写文件，--output 不使用）：
  python -m scripts.interpret.hover_replay run --points <hover_points.json> --h5 <H5> --ckpt <dir> --config <配置名> \
    --label ctx|mot --output <根目录> --seed 42 [--stage2-python <解释器>] [--eval-target-contract <路径>] \
    [--action-samples K] [--episodes 0 1 ...]
  python -m scripts.interpret.hover_replay check --points ... --h5 ... --ckpt ... --config ... --label ctx|mot \
    --output <根目录> --seed 42 [--stage2-python <解释器>] [--eval-target-contract <路径>] [--action-samples K] \
    [--prefix-reference <旧库根>] --episode 0 --rollout-attn <某局 attn 目录>

--eval-target-contract 是运行契约（同 scripts/training/serve_policy.py），motion 模型用；基线不传。

--action-samples K（缺省 0）：K>0 时每个取点再跑 K 次全路径 adapter.run(capture_hidden=True)，把 hidden_action
(18,W_a) 在 K 份噪声上取平均，npz 多写 hidden_action 与 action_samples，指纹加入 action_samples 与 seed；
第 j 份噪声的 key 为 fold_in(fold_in(fold_in(key(seed), episode), t), j)，再拆成 noise key 与采样 rng。
K=0 时输出键与指纹和加这个参数之前逐位相同。check 在 K>0 时只对 16 的倍数的取点算动作，
打印 ACTION_PARITY 与 HIDDEN_PRECISION group=action。

--prefix-reference <库根>（只 check 用）：把本局回放的 hidden_prefix 与 <库根>/<label>/episode_<k>.npz 里同 t 的点比较，
打印 PREFIX_MATCH。

JAX 确定性环境变量由外部脚本给。
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import h5py
import numpy as np

from mme_vla_suite.datastore.motion_adapter import STAGE2_SPLIT_READOUT
from mme_vla_suite.shared.sampling import MEM_ORDER_SENTINEL, pad_times
from scripts.interpret.offline_query import pack_observation, unpack_observation
from scripts.interpret.rollout_offline import HIDDEN_GROUPS, build_groups

REPLAY_BATCH = 8                # 回放节奏：首批 1 帧，之后每批 8 帧
EVAL_BATCH = 16                 # 评测节奏：首批 1 帧，之后每批 16 帧
HIDDEN_FORMAT_VERSION = 1       # 写进指纹；hidden 数组的含义变化时递增
LATEST_MOTION_GROUP = "最近motion"
REPLAY_GROUPS = (*HIDDEN_GROUPS, LATEST_MOTION_GROUP)   # hidden_prefix 第三维的固定顺序
WINDOW_PREFIX = "motion窗:"
PARITY_TOL = 1e-3
PRECISION_MIN_RATIO = 10
PRECISION_NUM_STEPS = 10


def feed_batches(last_t, batch_size):
    """帧区间 [start, stop)：第 0 帧单独一批，之后每批 batch_size 帧，喂到 last_t 为止。"""
    batches = [(0, 1)]
    start = 1
    while start <= last_t:
        stop = min(start + batch_size, last_t + 1)
        batches.append((start, stop))
        start = stop
    return batches


def pack_state(joint_state, gripper_state):
    """同 examples/robomme/env_runner.py::pack_state。"""
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


def read_steps(h5_episode, start, stop):
    """读 [start, stop) 的前视帧 (n,1,256,256,3) uint8 和状态 (n,8) float32。"""
    frames, states = [], []
    for step in range(start, stop):
        obs = h5_episode[f"timestep_{step}/obs"]
        frames.append(obs["front_rgb"][()][None])
        states.append(pack_state(obs["joint_state"][()], obs["gripper_state"][()]))
    return np.stack(frames), np.stack(states)


def build_observation(policy, element):
    """同 policy.infer 的输入路径，不采样。"""
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    inputs = policy._input_transform(policy._prepare_history(dict(element)))
    return HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))


def memory_times(policy, t):
    """frame_times 与 token 轴对齐的 motion_times，和 _prepare_history、_prepare_motion 同值；基线的 motion_times 为空。"""
    config = policy.config
    budget, token_per_image = int(config.budget), int(config.token_per_image)
    max_frames = budget // (token_per_image * int(config.num_views))
    frame_times = pad_times(policy.mem_buffer.get_frame_sampling_indices(t, budget, token_per_image), max_frames)
    if not policy.motion_enabled:
        return {"frame_times": frame_times, "motion_times": np.zeros(0, np.int64)}
    motion_times = pad_times(policy.mem_buffer.visible_motion_frames(t), policy.mem_buffer.motion_budget)
    if config.motion.get("readout") == STAGE2_SPLIT_READOUT:
        motion_times = np.repeat(motion_times, 4)
    return {"frame_times": frame_times, "motion_times": motion_times}


def replay_group_masks(observation, layout, record, tokens_per_frame):
    """bool[5,P]：HIDDEN_GROUPS 四行加最近 motion；基线没有 motion 窗口，第 5 行全 False。"""
    groups, hidden_masks = build_groups(observation, layout, record, tokens_per_frame)
    latest = np.zeros(layout["P"], bool)
    windows = [name for name in groups if name.startswith(WINDOW_PREFIX)]
    if windows:
        latest = groups[max(windows, key=lambda name: int(name[len(WINDOW_PREFIX):]))][:layout["P"]]
    return np.concatenate([hidden_masks, latest[None]])


def action_noise(seed, episode, t, j, shape):
    """第 j 份噪声：(采样 rng, noise)。key 为 fold_in(fold_in(fold_in(key(seed), episode), t), j)，再拆成两份。"""
    import jax
    import jax.numpy as jnp

    base = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.key(int(seed)), int(episode)), int(t)), int(j))
    noise_key, rng = jax.random.split(base)
    return rng, jax.random.normal(noise_key, shape, jnp.float32)


def action_hidden(adapter, observation, masks, *, seed, episode, t, samples):
    """在 highest 精度下跑 samples 次全路径，返回 hidden_action (18,W_a) 在各份噪声上的平均，float32。"""
    import jax

    rows = []
    with jax.default_matmul_precision("highest"):
        for j in range(samples):
            rng, noise = action_noise(seed, episode, t, j, (1, adapter.AH, adapter.AD))
            output = adapter.run(rng, observation, noise=noise, num_steps=PRECISION_NUM_STEPS, out_f16=False,
                                 capture_hidden=True, group_masks=masks)
            rows.append(np.asarray(jax.device_get(output[-1])["hidden_action"], dtype=np.float64))
    return np.mean(np.stack(rows), axis=0).astype(np.float32)


def replay_episode(policy, adapter, h5_episode, entry, batch_size, *, ts=None, on_point=None,
                   action_samples=0, seed=0, episode=0):
    """回放一局，在每个取点算一次前缀隐状态。

    ts 缺省取 entry 的全部取点。on_point(t, observation, masks, record, hidden) 在每个取点算完后调用。
    action_samples>0 时每个取点再按 action_hidden 算动作隐状态，hidden 与返回值多出 hidden_action；
    seed、episode 只用于噪声 key。
    """
    import jax

    ts = [point["t"] for point in entry["points"]] if ts is None else ts
    if any(t % batch_size for t in ts):
        raise ValueError(f"取点必须落在批边界：batch_size={batch_size} ts={ts}")
    wanted = set(ts)
    tokens_per_frame = int(policy.config.token_per_image) * int(policy.config.num_views)
    vision_seconds = prefix_seconds = action_seconds = 0.0
    done, hidden_rows, token_rows, action_rows = [], [], [], []
    policy.reset()
    for start, stop in feed_batches(max(ts), batch_size):
        frames, states = read_steps(h5_episode, start, stop)
        started = time.perf_counter()
        policy.add_buffer({"images": frames, "state": states, "exec_start_idx": 0})
        vision_seconds += time.perf_counter() - started
        t = stop - 1
        if t not in wanted:
            continue
        observation = build_observation(policy, {
            "observation/image": frames[-1, 0],
            "observation/wrist_image": h5_episode[f"timestep_{t}/obs/wrist_rgb"][()],
            "observation/state": states[-1],
            "prompt": entry["prompt"],
        })
        record = memory_times(policy, t)
        masks = replay_group_masks(observation, adapter.layout(observation), record, tokens_per_frame)
        started = time.perf_counter()
        with jax.default_matmul_precision("highest"):
            hidden = jax.device_get(adapter.run_prefix(observation, masks))
        prefix_seconds += time.perf_counter() - started
        hidden = {"hidden_prefix": np.asarray(hidden["hidden_prefix"], dtype=np.float32),
                  "group_tokens": np.asarray(hidden["group_tokens"], dtype=np.int32)}
        if action_samples > 0:
            started = time.perf_counter()
            hidden["hidden_action"] = action_hidden(adapter, observation, masks, seed=seed, episode=episode, t=t,
                                                    samples=action_samples)
            action_seconds += time.perf_counter() - started
            action_rows.append(hidden["hidden_action"])
        done.append(t)
        hidden_rows.append(hidden["hidden_prefix"])
        token_rows.append(hidden["group_tokens"])
        if on_point is not None:
            on_point(t, observation, masks, record, hidden)
    result = {"t": np.asarray(done, dtype=np.int64), "hidden_prefix": np.stack(hidden_rows),
              "group_tokens": np.stack(token_rows), "vision_seconds": vision_seconds, "prefix_seconds": prefix_seconds,
              "action_seconds": action_seconds}
    if action_samples > 0:
        result["hidden_action"] = np.stack(action_rows)
    return result


def episode_fingerprint(entry, ckpt, config, *, action_samples=0, seed=None):
    """K=0 时 payload 与加动作捕获之前逐字相同；K>0 时再加 action_samples 与 seed。"""
    payload = {"entry": entry, "ckpt": str(ckpt), "config": config, "hidden_format": HIDDEN_FORMAT_VERSION}
    if action_samples > 0:
        payload.update(action_samples=int(action_samples), seed=int(seed))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def existing_points(path, fingerprint):
    """文件已存在且指纹相同时返回其点数；不存在返回 None；指纹不同报错，不覆盖。"""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["fingerprint"]) != fingerprint:
            raise SystemExit(f"{path} 的指纹与当前输入不一致，不覆盖")
        return len(archive["t"])


def write_episode(path, result, episode, label, fingerprint, action_samples=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
    extra = {}
    if action_samples > 0:     # K=0 时不写这两个键，键集合与旧版相同
        extra = {"hidden_action": np.asarray(result["hidden_action"], dtype=np.float32),
                 "action_samples": np.int64(action_samples)}
    np.savez(temporary, t=result["t"], hidden_prefix=result["hidden_prefix"], group_tokens=result["group_tokens"],
             episode=np.int64(episode), label=np.str_(label), fingerprint=np.str_(fingerprint), **extra)
    os.replace(temporary, path)


def run_episodes(policy, adapter, h5_file, points, *, output, label, ckpt, config, episodes=None, batch_size=REPLAY_BATCH,
                 action_samples=0, seed=42):
    """逐局写 <output>/<label>/episode_<k>.npz，返回退出码。没有点的局不写。"""
    all_episodes = points["episodes"]
    ids = sorted(int(k) for k in all_episodes) if episodes is None else sorted(episodes)
    expected = sum(len(all_episodes[str(k)]["points"]) for k in ids)
    total = 0
    for k in ids:
        entry = all_episodes[str(k)]
        if not entry["points"]:
            continue
        path = Path(output) / label / f"episode_{k}.npz"
        fingerprint = episode_fingerprint(entry, ckpt, config, action_samples=action_samples, seed=seed)
        count = existing_points(path, fingerprint)
        if count is not None:
            total += count
            continue
        started = time.perf_counter()
        result = replay_episode(policy, adapter, h5_file[f"episode_{k}"], entry, batch_size,
                                action_samples=action_samples, seed=seed, episode=k)
        write_episode(path, result, k, label, fingerprint, action_samples)
        total += len(result["t"])
        print(f"REPLAY_EPISODE episode={k} points={len(result['t'])} seconds={time.perf_counter() - started:.1f} "
              f"vision_seconds={result['vision_seconds']:.1f} prefix_seconds={result['prefix_seconds']:.1f} "
              f"action_seconds={result['action_seconds']:.1f}", flush=True)
    passed = total == expected
    print(f"REPLAY_RUN={'PASS' if passed else 'FAIL'} model={label} points={total} expected={expected}", flush=True)
    return 0 if passed else 1


def rollout_steps(rollout_attn):
    """{step_idx: rollout_*.npz 路径}。"""
    steps = {}
    for path in sorted(Path(rollout_attn).glob("rollout_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            steps[int(archive["step_idx"])] = path
    return steps


def layout_differences(observation, record, path):
    """回放观测与 rollout 记录不相同的项；记录的 frame_times 不补位，motion_times 无效位记 -1，先换成同一写法。"""
    with np.load(path, allow_pickle=False) as archive:
        recorded = {key: archive[key] for key in archive.files}
    replayed = pack_observation(observation)
    packed = pack_observation(unpack_observation(recorded))
    differences = []
    if json.loads(str(replayed["obs_schema"])) != json.loads(str(packed["obs_schema"])):
        differences.append("obs_schema")
    frame_times = record["frame_times"][record["frame_times"] != MEM_ORDER_SENTINEL]
    if not np.array_equal(frame_times, recorded["frame_times"]):
        differences.append("frame_times")
    motion_times = record["motion_times"]
    if motion_times.size:
        motion_times = np.where(np.asarray(observation.motion_mask).reshape(-1), motion_times, -1)
    if not np.array_equal(motion_times, recorded["motion_times"]):
        differences.append("motion_times")
    if not np.array_equal(replayed.get("obs/mem_order"), packed.get("obs/mem_order")):
        differences.append("mem_order")
    return differences


def numeric_maxabs(first, second):
    """pack_observation 结果里全部数值键的最大绝对差。"""
    return max(float(np.max(np.abs(first[key].astype(np.float64) - second[key].astype(np.float64))))
               for key in first if key != "obs_schema")


def precision_readings(prefix, full, group_tokens, names=REPLAY_GROUPS):
    """每个非空组的 (组名, 比值, 路径误差)。误差跨所有点和层取最大；距离是相邻点同层的逐元素最大差，跨层和相邻对取中位数。"""
    prefix = np.asarray(prefix, dtype=np.float64)
    full = np.asarray(full, dtype=np.float64)
    readings = []
    for g, name in enumerate(names):
        if not group_tokens[:, g].any():
            continue
        error = float(np.abs(prefix[:, :, g] - full[:, :, g]).max())
        distance = np.median(np.abs(prefix[1:, :, g] - prefix[:-1, :, g]).max(axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = float(np.float64(distance) / np.float64(error))
        readings.append((name, ratio, error))
    return readings


def action_distance(actions):
    """动作隐状态 (n,18,W_a) 相邻取点同层逐元素最大差，跨层与相邻对取中位数；不足两个点为 NaN。"""
    actions = np.asarray(actions, dtype=np.float64)
    if len(actions) < 2:
        return float("nan")
    return float(np.median(np.abs(actions[1:] - actions[:-1]).max(axis=-1)))


def prefix_match(replay, reference_root, label, episode):
    """(可比点数, 最大绝对差)：回放的 hidden_prefix 对 <库根>/<label>/episode_<k>.npz 中 t 相同的点；无可比点为 NaN。"""
    path = Path(reference_root) / label / f"episode_{episode}.npz"
    differences = []
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            reference = dict(zip(archive["t"].tolist(), archive["hidden_prefix"]))
        for t, hidden in zip(replay["t"].tolist(), replay["hidden_prefix"]):
            if t in reference:
                differences.append(float(np.max(np.abs(hidden.astype(np.float64) - reference[t].astype(np.float64)))))
    return len(differences), max(differences, default=float("nan"))


def check_episode(policy, adapter, h5_episode, entry, rollout_attn, *, seed, action_samples=0, episode=0,
                  prefix_reference=None, label=None):
    """REPLAY_PARITY 与 HIDDEN_PRECISION；K>0 时加 ACTION_PARITY 与动作组精度，给了参考库时加 PREFIX_MATCH；返回退出码。"""
    import jax
    import jax.numpy as jnp

    rng = jax.random.key(seed)
    noise = jax.random.normal(rng, (1, adapter.AH, adapter.AD), jnp.float32)
    rollouts = rollout_steps(rollout_attn)
    kept, full_rows = {}, []
    replay_actions, action_differences = {}, []
    compared = layout_mismatch = 0

    def on_replay(t, observation, masks, record, hidden):
        nonlocal compared, layout_mismatch
        with jax.default_matmul_precision("highest"):
            output = adapter.run(rng, observation, noise=noise, num_steps=PRECISION_NUM_STEPS, out_f16=False,
                                 capture_hidden=True, group_masks=masks)
        full_rows.append(np.asarray(jax.device_get(output[-1])["hidden_prefix"], dtype=np.float32))
        if t % EVAL_BATCH:
            return
        kept[t] = (pack_observation(observation), hidden["hidden_prefix"])
        if action_samples > 0:     # 只对批 16 能比较的点算动作
            replay_actions[t] = action_hidden(adapter, observation, masks, seed=seed, episode=episode, t=t,
                                              samples=action_samples)
        if t in rollouts:
            compared += 1
            differences = layout_differences(observation, record, rollouts[t])
            if differences:
                layout_mismatch += 1
                print(f"REPLAY_LAYOUT_DIFF step={t} fields={','.join(differences)}", flush=True)

    replay = replay_episode(policy, adapter, h5_episode, entry, REPLAY_BATCH, on_point=on_replay)
    differences = []

    def on_eval(t, observation, masks, record, hidden):
        packed, hidden_prefix = kept[t]
        differences.append(max(numeric_maxabs(pack_observation(observation), packed),
                               float(np.max(np.abs(hidden["hidden_prefix"].astype(np.float64)
                                                   - hidden_prefix.astype(np.float64))))))
        if action_samples > 0:
            action = action_hidden(adapter, observation, masks, seed=seed, episode=episode, t=t, samples=action_samples)
            action_differences.append(float(np.max(np.abs(action.astype(np.float64)
                                                          - replay_actions[t].astype(np.float64)))))

    if kept:
        replay_episode(policy, adapter, h5_episode, entry, EVAL_BATCH, ts=sorted(kept), on_point=on_eval)
    maxabs = max(differences, default=float("nan"))     # 没有可比的点时为 NaN，判 FAIL
    parity = layout_mismatch == 0 and maxabs <= PARITY_TOL
    print(f"REPLAY_PARITY_LAYOUT compared={compared}", flush=True)
    print(f"REPLAY_PARITY={'PASS' if parity else 'FAIL'} layout_mismatch={layout_mismatch} points={len(differences)} "
          f"maxabs={maxabs:.9g}", flush=True)

    readings = precision_readings(replay["hidden_prefix"], np.stack(full_rows), replay["group_tokens"])
    precision = True
    for name, ratio, error in readings:
        passed = ratio >= PRECISION_MIN_RATIO
        precision &= passed
        print(f"HIDDEN_PRECISION={'PASS' if passed else 'FAIL'} group={name} min_dist_over_err={ratio:.6g} "
              f"path_maxabs={error:.9g}", flush=True)

    action_ok = True
    if action_samples > 0:
        action_error = max(action_differences, default=float("nan"))     # 没有可比的点时为 NaN，判 FAIL
        action_parity = action_error <= PARITY_TOL
        print(f"ACTION_PARITY={'PASS' if action_parity else 'FAIL'} points={len(action_differences)} "
              f"maxabs={action_error:.9g}", flush=True)
        distance = action_distance([replay_actions[t] for t in sorted(replay_actions)])
        if action_error == 0 and not np.isnan(distance):
            ratio = float("inf")       # 两条路径逐位相同
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = float(np.float64(distance) / np.float64(action_error))
        action_precision = ratio >= PRECISION_MIN_RATIO    # NaN 判 FAIL
        print(f"HIDDEN_PRECISION={'PASS' if action_precision else 'FAIL'} group=action min_dist_over_err={ratio:.6g} "
              f"path_maxabs={action_error:.9g}", flush=True)
        action_ok = action_parity and action_precision

    prefix_ok = True
    if prefix_reference is not None:
        matched, prefix_error = prefix_match(replay, prefix_reference, label, episode)
        prefix_ok = prefix_error <= PARITY_TOL       # 没有可比的点时为 NaN，判 FAIL
        print(f"PREFIX_MATCH={'PASS' if prefix_ok else 'FAIL'} points={matched} maxabs={prefix_error:.9g}", flush=True)
    return 0 if parity and precision and action_ok and prefix_ok else 1


def build_models(args):
    """bf16 的 policy 用真实 motion 编码器；float32 模型与适配器照 diag_hidden.py。"""
    import jax.numpy as jnp
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config as _config
    from openpi.models import model as _model
    from scripts.interpret.attn_adapter import AttnAdapter, _ModelShim

    ckpt = args.ckpt.resolve()
    policy = policy_config.create_trained_policy(_config.get_config(args.config), ckpt, seed=args.seed, motion_stub=False,
                                                 stage2_python=args.stage2_python,
                                                 eval_target_contract=args.eval_target_contract)
    params = _model.restore_params(ckpt / "params", dtype=jnp.float32)
    strict = (ckpt.parent / "history_config.resolved.yaml").is_file()
    model32 = dataclasses.replace(policy._model.config, dtype="float32").load(params, remove_extra_params=not strict)
    del params
    gc.collect()
    return policy, AttnAdapter(_ModelShim(model32))


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--points", type=Path, required=True)
    common.add_argument("--h5", type=Path, required=True)
    common.add_argument("--ckpt", type=Path, required=True)
    common.add_argument("--config", required=True)
    common.add_argument("--label", choices=("ctx", "mot"), required=True)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--stage2-python", default=None)
    common.add_argument("--eval-target-contract", type=Path, default=None)
    common.add_argument("--action-samples", type=int, default=0)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", parents=[common])
    run_parser.add_argument("--episodes", type=int, nargs="+")
    check_parser = commands.add_parser("check", parents=[common])
    check_parser.add_argument("--episode", type=int, required=True)
    check_parser.add_argument("--rollout-attn", type=Path, required=True)
    check_parser.add_argument("--prefix-reference", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.action_samples < 0:
        parser.error("--action-samples 不能为负")

    points = json.loads(args.points.read_text(encoding="utf-8"))
    policy, adapter = build_models(args)
    client = getattr(policy, "_motion_client", None)
    try:
        with h5py.File(args.h5, "r") as h5_file:
            if args.command == "run":
                return run_episodes(policy, adapter, h5_file, points, output=args.output, label=args.label,
                                    ckpt=args.ckpt.resolve(), config=args.config, episodes=args.episodes,
                                    action_samples=args.action_samples, seed=args.seed)
            return check_episode(policy, adapter, h5_file[f"episode_{args.episode}"], points["episodes"][str(args.episode)],
                                 args.rollout_attn, seed=args.seed, action_samples=args.action_samples,
                                 episode=args.episode, prefix_reference=args.prefix_reference, label=args.label)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
