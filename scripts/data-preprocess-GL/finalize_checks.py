#!/usr/bin/env python3
"""集群产物的收尾守卫：完整性 / stats / provenance / 同架构零容差抽检。

这些检查全部在**集群侧**跑（afterok 串在 array 之后），回答的是「集群这边自己有没有
把活干完、干得可不可复现」，与跨架构一致性是两件事：

  完整性   —— 每个 episode 的 token_emb_{0..T-1}.npy 与 kept_indices.json 齐全，
              data/{0..N-1}.pkl 连续无空洞、无多余。分片跑丢一片、某 episode 半途
              被杀，都会在这里现形。
  stats    —— 按清单前缀和写 meta/stats.json，并断言与实际落盘数一致。形制与串行
              builder 完全一样，下游 SampleDataset 读它决定 __len__。
  provenance —— 分两层，别搞混：**每个分片**在自己的 meta/_shard*.json 里记
              hostname / GPU 型号 / jax·jaxlib 版本 / git commit / 资源档位；
              finalize 把八片的指纹汇总进 meta/provenance.json 的 shard_fingerprints，
              并断言 gpu_device_kind / jax / jaxlib / git_commit **跨片唯一**
              （host、slurm_job、资源档位只记录不断言——array task 本就在不同节点）。
              provenance.json 的**顶层**字段描述的则是跑 finalize 的那个节点。
              跨架构逐位一致做不到（greatlakes.md 已实证），所以交付按「换合同」口径：
              集群产物自成一份数据集，**机制上杜绝**与本地字节混用。
              ⚠ 该断言 2026-08-24 才补上：更早产出的库（如 4task-gl）sidecar 里没有
              指纹字段，重跑 check 会 fail-loud 报「产出于指纹字段引入之前」。
              日常验收 step2_verify.sh 不受影响，它只读 finalize 当时的日志判定行。
  抽检     —— 随机抽 N 条在**同一节点**复算并要求 max|diff| == 0。同架构，所以可以
              零容差。它排除的是线程调度、cudnn 算法选择这类非确定性——如果这条不过，
              后面任何跨架构比对都失去意义。
              可以逐条独立复算，是因为 token_emb_{step} 只依赖该步的图像、step_idx
              与 state，不依赖历史（历史只影响 kept_indices）。

子命令：
  hash-inputs  给 H5 目录算 sha256 清单（本地原件与 turbo 副本各算一次，用于证明同源）
  check        完整性 + stats + provenance + 抽检
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import socket
import subprocess
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from scan_manifest import load_manifest  # noqa: E402

CHUNK = 1 << 22  # 4 MiB


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def cmd_hash_inputs(args: argparse.Namespace) -> None:
    entries = {}
    t0 = time.perf_counter()
    for name in sorted(os.listdir(args.raw_dir)):
        if not name.endswith(".h5"):
            continue
        p = os.path.join(args.raw_dir, name)
        digest = sha256_file(p)
        entries[name] = {"size": os.path.getsize(p), "sha256": digest}
        print(f"  {name} size={entries[name]['size']} sha256={digest} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"raw_dir": os.path.abspath(args.raw_dir), "count": len(entries),
                    "files": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[hash-inputs] {len(entries)} 个 H5 -> {out}")


def check_inputs(manifest: dict, raw_dir: str, input_manifest: str,
                 level: str = "size") -> list[str]:
    """输入同源核验：turbo 这份 H5 必须与清单记录的一致。

    两档：
      size   —— 只比字节数（默认）。够用是因为「本机原件 == turbo 副本」的 sha256 等价
                已经在 step0 一次性证过，而这个目录此后没有任何写入方；size 档抓的是
                rsync 缺漏/截断这类实际会发生的故障，而 321 GB 的 NFS 读要 40 分钟，
                放进每次 finalize 里性价比太低。
      sha256 —— 全量摘要，显式要复核时才用。
    """
    errs: list[str] = []
    ref = json.loads(pathlib.Path(input_manifest).read_text())["files"]
    for name in manifest["canonical_order"]:
        p = os.path.join(raw_dir, name)
        if not os.path.isfile(p):
            errs.append(f"缺输入 H5: {p}")
            continue
        if name not in ref:
            errs.append(f"输入清单里没有 {name}")
            continue
        size = os.path.getsize(p)
        if size != ref[name]["size"]:
            errs.append(f"{name} 字节数不符: {size} != {ref[name]['size']}")
            continue
        if level != "sha256":
            print(f"  ✓ {name} 字节数一致（level=size）", flush=True)
            continue
        got = sha256_file(p)
        if got != ref[name]["sha256"]:
            errs.append(f"{name} sha256 不符: {got} != {ref[name]['sha256']}")
        else:
            print(f"  ✓ {name} sha256 同源", flush=True)
    return errs


def check_completeness(manifest: dict, out_dir: str) -> tuple[list[str], dict]:
    errs: list[str] = []
    feat_root = os.path.join(out_dir, "features")
    data_root = os.path.join(out_dir, "data")
    t0 = time.perf_counter()

    missing_feat = 0
    for i, ep in enumerate(manifest["episodes"]):
        g = ep["global_episode_idx"]
        d = os.path.join(feat_root, f"episode_{g}")
        if not os.path.isdir(d):
            errs.append(f"缺 feature 目录: episode_{g} ({ep['h5_file']}#{ep['raw_ep_idx']})")
            missing_feat += 1
            continue
        names = set(os.listdir(d))
        if "kept_indices.json" not in names:
            errs.append(f"episode_{g} 缺 kept_indices.json")
        else:
            # 只判存在性会放过「空壳/半截 JSON」——它是被杀的进程留下的，
            # 而此时 token_emb 早已写全、下面的计数判据必然通过。必须解析内容。
            try:
                json.loads((pathlib.Path(d) / "kept_indices.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errs.append(f"episode_{g} 的 kept_indices.json 不是合法 JSON（半截落盘？）: {exc}")
        tmp_left = sorted(f for f in names if f.endswith(".tmp"))
        if tmp_left:
            errs.append(f"episode_{g} 残留原子写临时文件 {tmp_left}（说明写到一半被杀）")
        n = sum(1 for f in names if f.startswith("token_emb_") and f.endswith(".npy"))
        if n != ep["num_timesteps"]:
            errs.append(f"episode_{g} token_emb 数 {n} != 期望 {ep['num_timesteps']}")
        if (i + 1) % 200 == 0:
            print(f"  ...已核 {i + 1}/{len(manifest['episodes'])} 个 episode "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if len(errs) > 50:
            errs.append("错误过多，提前中止 feature 核验")
            break

    expected_exec = manifest["totals"]["exec_samples"]
    got = set()
    bad = 0
    with os.scandir(data_root) as it:
        for e in it:
            if not e.name.endswith(".pkl"):
                continue
            try:
                got.add(int(e.name[:-4]))
            except ValueError:
                bad += 1
    if bad:
        errs.append(f"data/ 下有 {bad} 个文件名不是纯数字 .pkl")
    holes = sorted(set(range(expected_exec)) - got)
    extra = sorted(i for i in got if i >= expected_exec)
    if holes:
        errs.append(f"data/ 缺 {len(holes)} 个 pkl，首个空洞 id={holes[0]}")
    if extra:
        errs.append(f"data/ 多出 {len(extra)} 个 pkl，首个 id={extra[0]}")

    stats = {"execution_samples": expected_exec, "total_samples": manifest["totals"]["timesteps"]}
    print(f"  feature 目录缺失={missing_feat}  pkl 实得={len(got)} 期望={expected_exec}", flush=True)
    return errs, stats


def spot_check(manifest: dict, out_dir: str, raw_dir: str, n: int, seed: int) -> list[str]:
    """同一节点复算 n 条 token_emb，零容差比对。

    用 compute_token_drop_score=False 的干净 buffer 逐条独立复算：token drop 评分只影响
    kept_indices，不进 token_emb；且开着它会在 step_idx==7 时去引用尚不存在的
    _history_feats[0]。
    """
    import h5py

    from mme_vla_suite.dataset_builder.mem_buffer import MemoryBuffer

    rng = random.Random(seed)
    picks = []
    for _ in range(n):
        ep = rng.choice(manifest["episodes"])
        picks.append((ep, rng.randrange(ep["num_timesteps"])))

    buf = MemoryBuffer(num_views=1, compute_token_drop_score=False, prepare_buffer=True)
    errs: list[str] = []
    by_file: dict[str, list] = {}
    for ep, step in picks:
        by_file.setdefault(ep["h5_file"], []).append((ep, step))

    worst = 0.0
    checked = 0
    for h5_name in sorted(by_file):
        with h5py.File(os.path.join(raw_dir, h5_name), "r") as data:
            for ep, step in by_file[h5_name]:
                g = ep["global_episode_idx"]
                saved_p = os.path.join(out_dir, "features", f"episode_{g}", f"token_emb_{step}.npy")
                if not os.path.isfile(saved_p):
                    errs.append(f"抽检缺文件: {saved_p}")
                    continue
                ts = data[f"episode_{ep['raw_ep_idx']}"][f"timestep_{step}"]
                image = ts["obs"]["front_rgb"][()]
                state = np.concatenate(
                    [ts["obs"]["joint_state"][()], ts["obs"]["gripper_state"][()][:1]],
                    axis=0, dtype=np.float32,
                )
                buf.clear()
                buf.add_buffer(image[None, None, ...], state[None, ...], [step])
                fresh = buf.get_history_feats(step)
                saved = np.load(saved_p, allow_pickle=True).item()
                if set(fresh) != set(saved):
                    errs.append(f"episode_{g}/step{step} key 集合不同: {set(fresh)} vs {set(saved)}")
                    continue
                for k in fresh:
                    a = np.asarray(fresh[k]).astype(np.float64)
                    b = np.asarray(saved[k]).astype(np.float64)
                    if a.shape != b.shape:
                        errs.append(f"episode_{g}/step{step} {k} 形状不同 {a.shape} vs {b.shape}")
                        continue
                    d = float(np.max(np.abs(a - b))) if a.size else 0.0
                    worst = max(worst, d)
                    if d != 0.0:
                        errs.append(f"episode_{g}/step{step} {k} max|diff|={d:.3e} 非零")
                checked += 1
    print(f"  抽检 {checked}/{n} 条，全体 max|diff|={worst:.3e}", flush=True)
    if not errs and checked:
        print(f"  ✓ 同架构复算全部逐位一致 PASS（{checked} 条）", flush=True)
    return errs


def git_commit(repo: pathlib.Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except Exception:
        return "unknown"


# 必须跨全部分片唯一的指纹项。host / slurm_job / resource_tier 刻意不在此列：
# 8 个 array task 本来就落在不同节点上，单片重提时也允许换档位。
FINGERPRINT_SAME_KEYS = ("gpu_device_kind", "jax", "jaxlib", "git_commit")


def aggregate_shard_fingerprints(docs: list[tuple[str, dict]]) -> tuple[dict, list[str]]:
    """跨分片指纹汇总 + 同源断言（纯函数，便于单测）。

    这是「换合同」交付口径的机制保证：集群产物自成一份数据集，其可信度建立在
    「八片是同一份代码、同一套运行时、同一种卡跑出来的」之上。此前 provenance 里的
    指纹全部取自 finalize 自己那个节点，从未与分片比对过，这条承诺是空的。
    """
    errs: list[str] = []
    vals: dict[str, set[str]] = {k: set() for k in FINGERPRINT_SAME_KEYS}
    hosts, jobs, tiers = [], [], []
    for name, d in docs:
        fp = d.get("fingerprint")
        if not isinstance(fp, dict):
            errs.append(f"{name} 缺分片指纹（schema_version={d.get('schema_version')}）"
                        f"——该库产出于指纹字段引入之前，需重跑分片才能通过 finalize")
            continue
        hosts.append(fp.get("host", ""))
        jobs.append(fp.get("slurm_job", ""))
        tiers.append(fp.get("resource_tier", {}))
        for k in FINGERPRINT_SAME_KEYS:
            v = str(fp.get(k, ""))
            if v.startswith("unavailable:"):
                # 采集失败等于没有指纹，同源断言无从谈起 —— fail-loud，不放行
                errs.append(f"{name} 的 {k} 采集失败（{v}），无法断言同源")
            vals[k].add(v)
    for k in FINGERPRINT_SAME_KEYS:
        if len(vals[k]) > 1:
            errs.append(f"分片间 {k} 不一致：{sorted(vals[k])}"
                        f"——八片不是同源产出，产物不可作为一份数据集交付")
    agg = {k: (next(iter(vals[k])) if len(vals[k]) == 1 else sorted(vals[k]))
           for k in FINGERPRINT_SAME_KEYS}
    agg.update({"hosts": hosts, "slurm_jobs": jobs, "resource_tiers": tiers,
                "shards_with_fingerprint": len(hosts)})
    return agg, errs


def cmd_check(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    out_dir = args.out
    all_errs: list[str] = []

    print(f"=== [1/5] 输入 H5 同源核验（level={args.input_level}）===", flush=True)
    if args.input_manifest:
        all_errs += check_inputs(manifest, args.raw_dir, args.input_manifest,
                                 level=args.input_level)
    else:
        print("  （未给 --input_manifest，跳过）")

    print("=== [2/5] 产物完整性核验 ===", flush=True)
    errs, stats = check_completeness(manifest, out_dir)
    all_errs += errs

    print("=== [3/5] 分片 sidecar 汇总 ===", flush=True)
    meta = pathlib.Path(out_dir, "meta")
    sidecars = sorted(meta.glob("_shard*of*.json"))
    covered: set[int] = set()
    shard_info = []
    sidecar_docs: list[tuple[str, dict]] = []
    for s in sidecars:
        try:
            d = json.loads(s.read_text())
        except ValueError as exc:
            all_errs.append(f"{s.name} 不是合法 JSON（半截落盘？）: {exc}")
            continue
        sidecar_docs.append((s.name, d))
        # ⚠ 末尾的 `if k in d` 会静默吞掉白名单外的字段：往 sidecar 加字段必须同步改这里。
        # 指纹刻意做成嵌套子对象，就是为了不必再动这份白名单。
        shard_info.append({k: d[k] for k in ("schema_version", "shard_idx", "num_shards",
                                             "episodes_done", "episodes_skipped", "steps",
                                             "elapsed_s", "rate_step_per_s",
                                             "rate_steady_step_per_s") if k in d})
        shard_info[-1]["fingerprint"] = d.get("fingerprint")
        covered |= set(d.get("episodes", []))
        if d.get("manifest_sha256") != manifest["sha256"]:
            all_errs.append(f"{s.name} 的 manifest_sha256 与当前清单不符（混用了不同清单？）")
    fp_agg, fp_errs = aggregate_shard_fingerprints(sidecar_docs)
    all_errs += fp_errs
    want = {e["global_episode_idx"] for e in manifest["episodes"]}
    if covered != want:
        all_errs.append(f"分片 sidecar 覆盖的 episode 集合不等于清单全集: "
                        f"缺 {len(want - covered)} 多 {len(covered - want)}")
    leftover = list(pathlib.Path(out_dir, "_claims").glob("_claim_*")) if \
        pathlib.Path(out_dir, "_claims").is_dir() else []
    if leftover:
        all_errs.append(f"残留 claim {len(leftover)} 个：{[p.name for p in leftover]}"
                        f"（说明对应分片没跑完）")
    print(f"  sidecar={len(sidecars)} 覆盖 episode={len(covered)} 残留 claim={len(leftover)}")

    print(f"=== [4/5] 同架构零容差抽检（{args.spot_check} 条）===", flush=True)
    if args.spot_check > 0 and not all_errs:
        all_errs += spot_check(manifest, out_dir, args.raw_dir, args.spot_check, args.seed)
    elif all_errs:
        print("  前序检查已失败，跳过抽检")

    print("=== [5/5] 写 stats.json 与 provenance.json ===", flush=True)
    if not all_errs:
        pathlib.Path(out_dir, "meta").mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_dir, "meta", "stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
        try:
            import jax
            import jaxlib
            jax_ver, jaxlib_ver = jax.__version__, jaxlib.__version__
            gpu = str(jax.devices()[0].device_kind)
        except Exception as exc:
            jax_ver = jaxlib_ver = gpu = f"unavailable: {exc}"
        prov = {
            "produced_by": "scripts/data-preprocess-GL/build_shard.py",
            # ⚠ 语义分两层，别再搞混（2026-08-24 审查坐实过一次事实错误）：
            #   · 下面这组顶层字段描述的是**跑 finalize 的那个节点**，不是产出数据的
            #     8 个分片节点。历史上 step2 把这里的 resource_tier 当构建档位打印，
            #     实际打出来的是 finalize job 的 4 CPU / 32 G，而分片是 2 CPU / 24 G。
            #   · 构建节点的真实指纹在 shard_fingerprints（以及每片 meta/_shard*.json
            #     的 fingerprint），跨片同源断言也是在那上面做的。
            #   顶层键名保持不变：step2_verify.sh 的 jq 与已交付库都依赖这个形状。
            "finalize_host": socket.gethostname(),
            "host": socket.gethostname(),
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
            "gpu_device_kind": gpu,
            "jax": jax_ver,
            "jaxlib": jaxlib_ver,
            "git_commit": git_commit(_REPO_ROOT),
            "manifest_sha256": manifest["sha256"],
            "raw_dir": os.path.abspath(args.raw_dir),
            "resource_tier": {
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
                "mem_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE", ""),
            },
            "shard_fingerprints": fp_agg,
            "shards": shard_info,
            "stats": stats,
        }
        pathlib.Path(out_dir, "meta", "provenance.json").write_text(
            json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  stats={stats}")
        print(f"  provenance(finalize 节点): host={prov['host']} "
              f"gpu={prov['gpu_device_kind']} jax={prov['jax']}")
        print(f"  provenance(构建分片): gpu={fp_agg['gpu_device_kind']} jax={fp_agg['jax']} "
              f"commit={str(fp_agg['git_commit'])[:12]} "
              f"节点={fp_agg['shards_with_fingerprint']} 个 全体同源 ✓")

    if all_errs:
        print("\n=== 失败明细 ===")
        for e in all_errs[:60]:
            print(f"  ✗ {e}")
        print(f"FINALIZE_EXIT_CODE=1  错误 {len(all_errs)} 条")
        raise SystemExit(1)
    print("全部检查通过 PASS")
    print("FINALIZE_EXIT_CODE=0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hash-inputs", help="给 H5 目录算 sha256 清单")
    h.add_argument("--raw_dir", required=True)
    h.add_argument("--out", required=True)
    h.set_defaults(func=cmd_hash_inputs)

    c = sub.add_parser("check", help="完整性 + stats + provenance + 抽检")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--raw_dir", required=True)
    c.add_argument("--input_manifest", default="")
    c.add_argument("--input_level", choices=["size", "sha256"], default="size")
    c.add_argument("--spot_check", type=int, default=256)
    c.add_argument("--seed", type=int, default=20260823)
    c.set_defaults(func=cmd_check)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
