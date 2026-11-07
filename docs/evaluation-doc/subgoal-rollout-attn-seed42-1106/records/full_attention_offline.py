"""把 PatternLock、RouteStick 每局全部推理的完整注意力落盘：infer_<seq>.npz（含 probs_action、probs_text）与 prefix_<seq>.npz（probs_prefix）。

用户原话（美东 2026-11-06）：「这一轮也要存下来！」「不要省略这些内容！」
复用 scripts.interpret.first_bin_offline 的 process_record、load_finished，只把「4 次窗口」换成「全部推理」。
原路径重放超 1e-6、行和超 5e-4 都非零退出；续跑按 input_sha256 跳过。

用法：full_attention_offline.py --run-root <run> --attn-root <attn> --output <out> --ckpt <dir> --config <名> --seed 42
"""
import argparse
from pathlib import Path
import sys

from scripts.interpret.analysis.offline_mechanism import fingerprint
from scripts.interpret.first_bin_offline import load_finished, process_record
from scripts.interpret.rollout_offline import load_accepted, load_trace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--attn-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ckpt", type=Path, required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

from scripts.interpret.attn_adapter import AttnAdapter, _ZeroMotion, _MV_DIR
sys.path.insert(0, str(_MV_DIR))
from mv_model_adapter import build_policy
from mme_vla_suite.training import config

policy = build_policy(config.get_config(args.config), args.ckpt.resolve(), seed=args.seed, motion_factory=lambda **kwargs: _ZeroMotion())
adapter = AttnAdapter(policy, save_prefix=True)
written = skipped = 0
same_max = attn_max = row_max = 0.0
for nonce, episode_key in sorted(load_accepted(args.run_root).items(), key=lambda item: item[1]):
    trace = load_trace(args.run_root, nonce)
    files = sorted((args.attn_root / nonce).glob("rollout_*.npz"))
    seqs = sorted(int(path.stem.split("_")[1]) for path in files)
    if seqs != sorted(trace):
        raise SystemExit(f"{episode_key}：rollout 记录 {seqs} 与轨迹动作块 {sorted(trace)} 不一致")
    for seq, path in zip(seqs, files):
        row = trace[seq]
        infer_path, prefix_path = args.output / nonce / f"infer_{seq:04d}.npz", args.output / nonce / f"prefix_{seq:04d}.npz"
        sha = fingerprint(path)["sha256"]
        item = load_finished(infer_path, prefix_path, sha)
        if item is None:
            item = process_record(policy, adapter, path, infer_path, prefix_path, row, sha)
            written += 1
        else:
            skipped += 1
        same_max, attn_max, row_max = max(same_max, item["same_path_maxabs"]), max(attn_max, item["attn_path_maxabs"]), max(row_max, item["rowsum_err"])
    print(f"FULL_EPISODE episode={episode_key} nonce={nonce} records={len(seqs)}", flush=True)
total = written + skipped
if not total:
    raise SystemExit("没有可处理的记录")
print(f"ROLLOUT_REPLAY=PASS points={total} same_path_maxabs={same_max:.9g} tol=1e-6", flush=True)
print(f"FULL_ATTN=PASS records={total} written={written} skipped={skipped} rowsum_err={row_max:.9g} attn_path_maxabs={attn_max:.9g}", flush=True)
