"""双消融三 seed 汇总（在 GreatLakes 登录节点上运行，读 turbo 上 6 个 run 根）。

口径同 10-31 三 seed 汇总：每个模型每个 seed 800 局，总体与逐任务成功率取三 seed 均值 ± 样本标准差（ddof=1）。
输出 JSON 到 stdout，并打印 THREE_SEED / SEED_ECHO / COMPUTE_DONE 判定行。
"""
import glob
import json
import os
import statistics
import sys

O = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/o16abl-80k"
MODELS = {"N": "noxyz", "S": "nostate"}
SEEDS = (42, 7, 0)


def ms(values):
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def main():
    out = {"runs": {}, "models": {}}
    lines = []
    for m in MODELS:
        per_seed = {}
        for seed in SEEDS:
            root = f"{O}/o16abl-{m}-seed{seed}/run800"
            summary = json.load(open(f"{root}/queue/summary.json"))
            spec = json.load(open(f"{root}/spec.json"))
            rows = [json.load(open(p)) for p in glob.glob(f"{root}/accepted/*.json")]
            mism = sum(r["run_identity_sha256"] != spec["run_identity_sha256"] for r in rows)
            by_task, by_diff = {}, {}
            for r in rows:
                ok = r["outcome"] == "success"
                by_task.setdefault(r["task_name"], [0, 0]); by_task[r["task_name"]][0] += 1; by_task[r["task_name"]][1] += ok
                by_diff.setdefault(r["difficulty"], [0, 0]); by_diff[r["difficulty"]][0] += 1; by_diff[r["difficulty"]][1] += ok
            succ = sum(r["outcome"] == "success" for r in rows)
            per_seed[seed] = {"n": len(rows), "success": succ, "rate": succ / len(rows) * 100,
                              "timeout": sum(r["outcome"] == "timeout" for r in rows),
                              "by_task": {k: v[1] / v[0] * 100 for k, v in by_task.items()},
                              "by_diff": {k: f"{v[1]}/{v[0]}" for k, v in by_diff.items()},
                              "summary": summary}
            lines.append(f"COMPUTE_DONE run={m}-s{seed} completed={summary['completed']} missing={len(summary['missing'])} "
                         f"errors={len(summary['errors'])} uncertain={len(summary['uncertain'])} retries={json.dumps(summary['retries'])}")
            lines.append(f"SEED_ECHO={'PASS' if mism == 0 and spec['model_seed'] == seed and len(rows) == 800 else 'FAIL'} "
                         f"run={m}-s{seed} seed={seed} records={len(rows)} identity_mismatch={mism} spec_seed={spec['model_seed']}")
        out["runs"][m] = per_seed
        mean, std = ms([per_seed[s]["rate"] for s in SEEDS])
        tasks = sorted(per_seed[SEEDS[0]]["by_task"])
        out["models"][m] = {"overall_mean": mean, "overall_std": std,
                            "by_task": {t: ms([per_seed[s]["by_task"][t] for s in SEEDS]) for t in tasks},
                            "timeouts": {s: per_seed[s]["timeout"] for s in SEEDS}}
        ok = all(per_seed[s]["n"] == 800 for s in SEEDS)
        lines.append(f"THREE_SEED={'PASS' if ok else 'FAIL'} model={m}({MODELS[m]}) seeds=7,42,0 runs=3 "
                     f"overall={mean:.2f}±{std:.2f} per_seed=" + ",".join(f"s{s}:{per_seed[s]['success']}/800" for s in SEEDS))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n".join(lines), file=sys.stderr)


if __name__ == "__main__":
    main()
