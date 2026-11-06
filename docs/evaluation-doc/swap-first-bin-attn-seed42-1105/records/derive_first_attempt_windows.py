"""把窗口表的锚点从“抬起前最后一次闭合”改到“第一次闭合”，生成首次闭合对照用的窗口文件。

只处理有 earlier_closes 的局（抓空后重试）；其余局原样保留。label 仍按第一次抬起的 bin，不改。
第一次闭合的信息全部取自该局 step_trace.jsonl 里 env_step_after 相同的那一行。

用法：python derive_first_attempt_windows.py <windows.json> <run-root> [<run-root> ...] --out <输出文件>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def trace_rows(run_roots, nonce):
    paths = [path for root in run_roots for path in Path(root).glob(f"attempts/*/client/traces/{nonce}/step_trace.jsonl")]
    if len(paths) != 1:
        raise SystemExit(f"nonce={nonce} 的轨迹应恰有 1 份，实际 {len(paths)}")
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("event") == "step"]


def derive(entry, rows):
    out = dict(entry, first_attempt_bin=entry.get("nearest_bin_at_C"), lifted_differs=False, retry_closes=0)
    closes = entry.get("earlier_closes") or []
    if not closes:
        return out
    first = closes[0]
    matches = [row for row in rows if row["env_step_after"] == first["step"]]
    if len(matches) != 1:
        raise SystemExit(f"{entry['episode_key']}：env_step_after={first['step']} 应恰有 1 行，实际 {len(matches)}")
    row = matches[0]
    g = int(row["infer_seq"])
    out.update(G=g, window=list(range(max(0, g - 3), g + 1)), C=first["step"],
               C_frame=int(row["exec_start_idx"]) + int(row["action_index"]) + 1,
               close_row_in_chunk=int(row["row_in_chunk"]), nearest_bin_at_C=first["nearest_bin"],
               first_attempt_bin=first["nearest_bin"], lifted_differs=first["nearest_bin"] != entry.get("picked_bin"),
               retry_closes=len(closes), earlier_closes=[],
               window_status="complete" if g >= 3 else "incomplete")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("windows", type=Path)
    parser.add_argument("run_roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.windows.read_text(encoding="utf-8"))
    episodes = []
    for entry in data["episodes"]:
        closes = entry.get("earlier_closes") or []
        episodes.append(derive(entry, trace_rows(args.run_roots, entry["episode_nonce"]) if closes else []))
    args.out.write_text(json.dumps(dict(data, episodes=episodes), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    moved = [e for e in episodes if e["retry_closes"]]
    differ = [e["episode_key"] for e in episodes if e["lifted_differs"]]
    print(f"FIRST_ATTEMPT_WINDOWS=PASS episodes={len(episodes)} moved={len(moved)} lifted_differs={len(differ)} "
          f"incomplete={sum(e['window_status'] == 'incomplete' for e in moved)}")
    print("LIFTED_DIFFERS_EPISODES=" + (",".join(differ) if differ else "none"))


if __name__ == "__main__":
    main()
