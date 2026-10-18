"""160 窗预算核算（1017-test-xhard5-12per 4.3 第 7 项）：清单 xhard 行按本机生成产物的 demo 帧数算 motion 窗数上界。

口径：``es`` = 该局 h5 里 ``timestep_*/info/is_video_demo`` 为真的帧数（评估客户端 ``exec_start_idx`` 同一定义：
reset 返回 demo 帧 + 1 帧复位观测，es = demo 帧数）。窗数与 ``check_shard.py`` / ``eval_es_bound.py::predicted_windows``
同式：demo 段 ``len(range(0, max(0, es-16), 16))``，执行段在 max_steps 步时 ``(16*(max_steps//16) - 32)//16 + 1``；
最大全域帧号 = es + max_steps（与 ``add_buffer`` 最后一次调用同定义，给上界），须 < 4096（pos 表行数）。
超 160 的局不剔除，评估时走 ``MMEVLA_MOTION_OVERFLOW=resample``，这里只记录。

用法：``es_bound.py <清单 jsonl> <h5 目录>... [--out <json>] [--max-steps 2000] [--budget 160]``；
h5 目录递归找 ``*.h5``，按 ``episode_*/setup/seed`` 对上清单行（seed 在 test 布局内逐任务唯一），再核难度一致。
判定行：``EVAL_ES_BOUND=DONE scope=xhard episodes=<n> max_windows=<k> over=<j> max_step_idx=<m> over4096=<i>``。
"""
import argparse
import json
from pathlib import Path

import h5py

POS_ROWS = 4096


def predicted_windows(es: int, max_steps: int) -> int:
    demo = len(range(0, max(0, int(es) - 16), 16))
    delta = 16 * (int(max_steps) // 16)
    execution = 0 if delta < 32 else (delta - 32) // 16 + 1
    return demo + execution


def _str(value) -> str:
    value = value[()] if hasattr(value, "shape") else value
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def index_h5(dirs):
    """seed -> {file, episode, difficulty, demo}。"""
    index = {}
    for d in dirs:
        for path in sorted(Path(d).rglob("*.h5")):
            with h5py.File(path, "r") as f:
                for name in f:
                    if not name.startswith("episode_"):
                        continue
                    group = f[name]
                    seed = int(group["setup/seed"][()])
                    difficulty = _str(group["setup/difficulty"])
                    demo = 0
                    for key in group:
                        if key.startswith("timestep_"):
                            info = group[key].get("info")
                            if info is not None and "is_video_demo" in info and bool(info["is_video_demo"][()]):
                                demo += 1
                    if seed in index:
                        raise SystemExit(f"seed {seed} 在多个 h5 里出现：{index[seed]['file']} / {path}")
                    index[seed] = {"file": str(path), "episode": name, "difficulty": difficulty, "demo": demo}
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("h5_dirs", nargs="+")
    parser.add_argument("--out", default="")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--budget", type=int, default=160)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    xhard = [r for r in rows if str(r["difficulty"]).startswith("xhard")]
    index = index_h5(args.h5_dirs)
    results = []
    for r in xhard:
        hit = index.get(int(r["seed"]))
        if hit is None:
            raise SystemExit(f"EVAL_ES_BOUND=FAIL 找不到 {r['task']}/{r['difficulty']} ep{r['episode']} seed={r['seed']} 的 h5")
        if hit["difficulty"] != r["difficulty"]:
            raise SystemExit(f"EVAL_ES_BOUND=FAIL seed={r['seed']} h5 难度 {hit['difficulty']} != 清单 {r['difficulty']}")
        es = hit["demo"]
        k = predicted_windows(es, args.max_steps)
        results.append({**r, "es": es, "windows": k, "max_step_idx": es + args.max_steps,
                        "over_budget": k > args.budget, "h5": hit["file"], "h5_episode": hit["episode"]})
    max_windows = max((x["windows"] for x in results), default=0)
    over = sum(x["over_budget"] for x in results)
    max_idx = max((x["max_step_idx"] for x in results), default=0)
    over4096 = sum(x["max_step_idx"] >= POS_ROWS for x in results)
    summary = {"scope": "xhard", "episodes": len(results), "max_steps": args.max_steps, "budget": args.budget,
               "max_windows": max_windows, "over": over, "max_step_idx": max_idx, "over4096": over4096,
               "rows": results}
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"拒绝覆盖 {out}")
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"EVAL_ES_BOUND=DONE scope=xhard episodes={len(results)} max_windows={max_windows} over={over} "
          f"max_step_idx={max_idx} over4096={over4096}", flush=True)


if __name__ == "__main__":
    main()
