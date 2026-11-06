#!/usr/bin/env python3
"""A2：用 H5 原始帧重放在线 tokendrop，与建库 kept_indices.json 及训练侧取样口径对拍。

每任务取最长的 1 集（最可能覆盖容量淘汰）。走真实 FrameSampMemory.add_buffer，帧与 state 的读取
与 dataset_builder/build_robomme_dataset.py::_process_episode 相同，分包节奏与在线相同（首包 D+1 帧，
之后每包 16 帧）；SigLIP 换成全零替身，所以这里只验证打分与取样，不验证编码数值。

判据：
- 每集最终堆的 pop 序列与 kept_indices.json 完全相同（淘汰过的集也算）；
- 未淘汰的集，在 3 个 t 上在线 filter 结果 == 训练侧（kept 过滤到 t 后取末 BUDGET 个再排序）。
  kept 长度达 2048 的集，此前的堆内容可能含后来被淘汰的 token，t 上的差异只单列，不计 mismatch。

末行：ONLINE_TD_REPLAY=PASS episodes=16 mismatch=0，否则 ONLINE_TD_REPLAY=FAIL。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

import h5py
import numpy as np

from mme_vla_suite.policies.framesamp_memory import FrameSampMemory

DEFAULT_DATASET = "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-orig100-1600ep-9e109419"
BUDGET = 1024
KEPT_CAPACITY = 2048
STRIDE = 16 // 2  # streaming_obs_horizon // 2，与 build_robomme_dataset 的 execution_horizon // 2 同值
PACKET = 16


def stub_encoder(images):
    """形状同池化后的 8×8 token；broadcast 不占内存。"""
    return np.broadcast_to(np.zeros((), np.dtype("bfloat16")), (*images.shape[:2], 64, 2048))


def read_episode(group, num_timesteps):
    images, states = [], []
    for step_idx in range(num_timesteps):
        obs = group[f"timestep_{step_idx}"]["obs"]
        joint_state = obs["joint_state"][()]
        gripper_state = obs["gripper_state"][()]
        states.append(np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32))
        images.append(obs["front_rgb"][()][None, ...])
    return np.stack(images), np.stack(states)


def packets(num_timesteps, exec_start_idx):
    """首包 D+1 帧，之后每包 16 帧；返回每包的帧号列表。"""
    start = exec_start_idx + 1
    result = [list(range(start))]
    while start < num_timesteps:
        result.append(list(range(start, min(start + PACKET, num_timesteps))))
        start += PACKET
    return result


def query_steps(schedule):
    """取 1/3、2/3 与末尾三处的包尾帧号：在线只在包尾 infer。"""
    ends = [ids[-1] for ids in schedule]
    picked = [ends[(len(ends) - 1) // 3], ends[2 * (len(ends) - 1) // 3], ends[-1]]
    return list(dict.fromkeys(picked))


def training_side(kept, step_idx):
    return sorted([item for item in kept if item[0] <= step_idx][-BUDGET:])


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"index={index} online={a} expected={b}"
    return f"len online={len(left)} expected={len(right)}"


def replay(memory, group, entry, kept):
    """返回 (final_ok, final_detail, [(t, ok, detail)])。"""
    num_timesteps, exec_start_idx = entry["num_timesteps"], entry["exec_start_idx"]
    found = sum(1 for key in group.keys() if key.startswith("timestep_"))
    if found != num_timesteps:
        raise RuntimeError(f"H5 帧数 {found} != 清单 {num_timesteps}")
    images, states = read_episode(group, num_timesteps)
    schedule = packets(num_timesteps, exec_start_idx)
    queries = set(query_steps(schedule))
    query_results = []
    memory.clear()
    for ids in schedule:
        memory.add_buffer(images[ids], states[ids], ids)
        step_idx = ids[-1]
        if step_idx not in queries:
            continue
        online = memory.filter_token_dropping_indices(memory.get_token_dropping_indices(), step_idx, BUDGET)
        expected = training_side(kept, step_idx)
        image, position, state, mask = memory.prepare_token_dropping(step_idx, BUDGET)
        shapes_ok = (image.shape == (BUDGET, 2048) and position.shape == (BUDGET, 768)
                     and state.shape == (BUDGET, 8) and int(mask.sum()) == len(online))
        same = online == expected and shapes_ok
        query_results.append((step_idx, same, "" if same else first_difference(online, expected)))
    final = memory.get_token_dropping_indices()
    final_ok = final == kept
    return final_ok, "" if final_ok else first_difference(final, kept), query_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "meta/episode_manifest.json").read_text())
    chosen = {}
    for entry in manifest["episodes"]:
        best = chosen.get(entry["h5_file"])
        if best is None or entry["num_timesteps"] > best["num_timesteps"]:
            chosen[entry["h5_file"]] = entry

    memory = FrameSampMemory(vision_enc_fn=stub_encoder, token_per_image=64, token_drop_stride=STRIDE,
                             token_drop_keptsize=KEPT_CAPACITY)

    episodes = mismatch = evicted_episodes = queries_checked = evicted_query_diffs = 0
    for h5_file in manifest["canonical_order"]:
        entry = chosen[h5_file]
        episode_dir = f"episode_{entry['global_episode_idx']}"
        kept_path = dataset / "source" / "features" / episode_dir / "kept_indices.json"
        kept = [tuple(item) for item in json.loads(kept_path.read_text())]
        evicted = len(kept) >= KEPT_CAPACITY
        with h5py.File(Path(manifest["raw_dir"]) / h5_file, "r") as data:
            final_ok, final_detail, query_results = replay(memory, data[f"episode_{entry['raw_ep_idx']}"], entry, kept)
        episodes += 1
        evicted_episodes += evicted
        mismatch += not final_ok
        for step_idx, same, detail in query_results:
            if evicted:
                evicted_query_diffs += not same
                if not same:
                    print(f"  淘汰集差异（不计 mismatch） {episode_dir} t={step_idx}: {detail}", flush=True)
            else:
                queries_checked += 1
                mismatch += not same
                if not same:
                    print(f"  MISMATCH {episode_dir} t={step_idx}: {detail}", flush=True)
        if not final_ok:
            print(f"  MISMATCH {episode_dir} 最终堆: {final_detail}", flush=True)
        print(f"{h5_file} {episode_dir} T={entry['num_timesteps']} es={entry['exec_start_idx']} kept={len(kept)} "
              f"evicted={evicted} final_ok={final_ok} t={[item[0] for item in query_results]} "
              f"t_ok={[item[1] for item in query_results]}", flush=True)

    print(f"明细 evicted_episodes={evicted_episodes} queries_checked={queries_checked} "
          f"evicted_query_diffs={evicted_query_diffs}", flush=True)
    passed = mismatch == 0 and episodes == 16
    print(f"ONLINE_TD_REPLAY={'PASS' if passed else 'FAIL'} episodes={episodes} mismatch={mismatch}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
