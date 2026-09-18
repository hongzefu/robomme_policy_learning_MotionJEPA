"""分片结果验收：计划全覆盖、取值合法、视频与非 error 集一一对应且可完整解码。

与单回合的 check_result.py 分开，不动那份——它承载着 vail-eval-gl-20260918T0400Z 已绿留档的
可复现口径。本验收的三点不同：
1. 组键是「任务/难度」而不是任务名（注入候选的唯一键是 (task, difficulty, episode)）；
2. episode 号来自计划、不是 range(N)；
3. **视频数 == 非 error 集数**，不是 == 总集数——eval.py 在 success_flag=="unknown" 时提前
   return 不落视频，抛异常那条路同样没有视频，这两种都记 "error"。
"""
import json
from pathlib import Path
import sys

import imageio.v2 as imageio


def check_shard(run_dir, policy, ckpt, seed, plan_path):
    root = Path(run_dir) / policy / f"ckpt{ckpt}" / f"seed{seed}"
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))

    planned = {(g, int(e)) for g, eps in plan["groups"].items() for e in eps}
    done = {(g, int(e)) for g, entries in progress.items() for e in entries}
    assert done == planned, (
        f"计划与结果不符：缺 {sorted(planned - done)[:5]}，多 {sorted(done - planned)[:5]}"
    )

    errors, successes, videos_expected = [], 0, []
    for group, entries in progress.items():
        task, difficulty = group.split("/")
        for episode, value in entries.items():
            if value == "error":
                errors.append({"group": group, "episode": int(episode)})
                continue
            assert type(value) is bool, f"非法取值 {group}/{episode}={value!r}"
            successes += value
            videos_expected.append((task, int(episode), difficulty))

    videos = sorted((root / "videos").glob("*.mp4"))
    assert len(videos) == len(videos_expected), (
        f"视频数 {len(videos)} != 非 error 集数 {len(videos_expected)}"
    )

    records = []
    for task, episode, difficulty in videos_expected:
        prefix, suffix = f"{task}_ep{episode}_", f"_{difficulty}.mp4"
        matched = [v for v in videos if v.name.startswith(prefix) and v.name.endswith(suffix)]
        assert len(matched) == 1, f"{task}/{difficulty} ep{episode} 匹配到 {len(matched)} 个视频"
        video = matched[0]
        count = 0
        with imageio.get_reader(video) as reader:
            for frame in reader:
                assert frame.ndim == 3 and frame.shape[-1] == 3
                count += 1
        assert count > 1, video
        records.append({"group": f"{task}/{difficulty}", "episode": episode,
                        "path": str(video), "frames": count, "bytes": video.stat().st_size})

    total = len(planned)
    summary = {
        "pipeline_pass": True,
        "shard_tag": plan.get("shard_tag", ""),
        "identity_sha256": plan.get("identity_sha256", ""),
        "episodes": total,
        "evaluated": total - len(errors),
        "successes": successes,
        "errors": errors,
        "groups": {g: {"episodes": len(entries),
                       "successes": sum(v is True for v in entries.values()),
                       "errors": sum(v == "error" for v in entries.values())}
                   for g, entries in sorted(progress.items())},
        "videos": records,
    }
    (root / "check.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SHARD_PASS", json.dumps({k: v for k, v in summary.items() if k != "videos"},
                                   ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run_dir, policy, ckpt, seed, plan_path = sys.argv[1:]
    check_shard(run_dir, policy, ckpt, seed, plan_path)
