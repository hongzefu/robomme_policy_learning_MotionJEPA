#!/usr/bin/env python3
"""分片 worker：按 episode_manifest.json 处理指定分片的 episode。

**核心设计：子类化 `DatasetProcessor`，绝不复制它的逻辑。**
只覆盖两个方法——
  `__init__`：跳过原实现里的 `shutil.rmtree(dataset_path)`（那会把其他分片已写好的产物
              一并删掉），只 makedirs；
  `run`    ：改为按清单遍历本分片的 episode，并用清单预先算死的偏移量喂
              `_process_episode` 的三个计数器。
`_process_episode` 的**计算逻辑原样继承**——语义与串行 builder 同构是由构造方式保证的，
而不是靠事后对拍碰运气。原实现里现成的 `assert not os.path.exists(pkl_path)`
顺带就是跨分片撞号的兜底断言。
（唯一的例外是落盘方式：`kept_indices.json` 已从 `json.dump` 换成 `atomic_write_json`
的原子写，写出的**字节与改前逐字节相同**，第一层 bitexact 已重跑验证。改动动因见
`episode_is_complete` 的 docstring。）

用法（集群）：
  python build_shard.py --manifest <清单> --raw_dir <H5目录> --out <输出库> \
      --shard_idx $SLURM_ARRAY_TASK_ID --num_shards 8
用法（本地对照）：
  python build_shard.py --manifest ... --subset <sample 产物> --out <ref库> \
      --shard_idx 0 --num_shards 4
用法（本机多 GPU 动态领任务，v2-motionmem；由 scripts/dataset/run_local.py --stage siglip 起，每卡一进程）：
  CUDA_VISIBLE_DEVICES=<k> python build_shard.py --manifest ... --raw_dir ... --out <lib>/source \
      --worker-mode --worker-label gpu<k> --worker-idx <i> --num-workers <n> --resume
  工作项 = episode（按 num_timesteps LPT 降序），以 <out>/_claims/_claim_ep<g>（O_CREAT|O_EXCL）领一项、
  完成即 unlink；sidecar 仍写 meta/_shard<i>of<n>.json（finalize 的 glob 与跨 worker 指纹断言不变）。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import socket
import sys
import time

import h5py

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from finalize_checks import git_commit  # noqa: E402
from scan_manifest import assign_shards_lpt  # noqa: E402
from scan_manifest import load_manifest  # noqa: E402

from mme_vla_suite.dataset_builder.build_robomme_dataset import DatasetProcessor  # noqa: E402
from mme_vla_suite.dataset_builder.mem_buffer import MemoryBuffer  # noqa: E402


class ShardProcessor(DatasetProcessor):
    """只改「跑哪些 episode、从哪个 ID 起算」，不改「一个 episode 怎么处理」。"""

    def __init__(self, raw_data_path: str, out_dir: str, execution_horizon: int = 16) -> None:
        # 刻意不调用 super().__init__：它会 shutil.rmtree(out_dir)，在 8 分片并发下
        # 等于互删产物。这里逐字段复现它除 rmtree 外的全部初始化。
        self.raw_data_path = raw_data_path
        self.dataset_path = out_dir
        self.execution_horizon = execution_horizon
        self.visualize = False
        self.max_episodes = None

        self.feature_path = os.path.join(self.dataset_path, "features")
        self.data_path = os.path.join(self.dataset_path, "data")
        self.meta_path = os.path.join(self.dataset_path, "meta")
        for p in (self.dataset_path, self.feature_path, self.data_path, self.meta_path):
            os.makedirs(p, exist_ok=True)

        # data/ 下 pkl id 的一次性快照，仅 --resume 的完整性判据用（见 _exec_ids_present）
        self._exec_ids_cache: set[int] | None = None

    def _exec_ids_present(self) -> set[int]:
        """data/ 下已落盘的 pkl id 快照，整个分片只扫一次。

        为什么可以缓存：各分片的 exec id 区间由清单的前缀和算死、彼此不相交，
        我们只查自己名下尚未处理的 episode，别的分片并发写入不会让快照失效。
        为什么不逐文件 os.path.isfile：一个 episode 平均约 247 个 pkl，
        全量 1600 个 episode 是 39.5 万次 NFS LOOKUP，一次 scandir 是唯一可接受的写法。
        """
        if self._exec_ids_cache is None:
            ids: set[int] = set()
            with os.scandir(self.data_path) as it:
                for e in it:
                    if e.name.endswith(".pkl"):
                        try:
                            ids.add(int(e.name[:-4]))
                        except ValueError:
                            pass
            self._exec_ids_cache = ids
        return self._exec_ids_cache

    def episode_is_complete(self, ep: dict) -> bool:
        """完整性判据三段：kept_indices.json 内容合法、feature 数对得上、pkl 区间齐全。

        ⚠ 只判「kept_indices.json 存在」是不够的：它虽然是 `_process_episode` 的最后一步，
        但 open(path, "w") 一调用文件就已经「存在」了。进程若在写它的过程中被杀
        （walltime 耗尽被 SLURM 强杀正是续跑的触发场景），会留下空壳/半截 JSON，
        而此时该 episode 的 token_emb 早已写全、计数判据必然满足——旧判据会把这个
        坏 episode 当成完整、永久跳过。写入侧已改原子写（见 build_robomme_dataset.py 的
        atomic_write_json），这里再做一次内容解析兜底旧产物与非原子写的历史残留。
        """
        d = os.path.join(self.feature_path, f"episode_{ep['global_episode_idx']}")
        kept = os.path.join(d, "kept_indices.json")
        if not os.path.isfile(kept):
            return False
        try:
            json.loads(pathlib.Path(kept).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        n = sum(1 for f in os.listdir(d) if f.startswith("token_emb_") and f.endswith(".npy"))
        if n != ep["num_timesteps"]:
            return False
        # pkl 区间与 purge_episode 的删除范围必须逐字一致，否则会出现「判不完整但清不干净」
        start = ep["exec_sample_offset"]
        want = set(range(start, start + ep["exec_samples"]))
        return want <= self._exec_ids_present()

    def purge_episode(self, ep: dict) -> None:
        """清掉一个残缺 episode 的全部产物。必须做——`_process_episode` 里的
        `assert not os.path.exists(pkl_path)` 会在续跑撞上半截 pkl 时直接炸。
        删除范围由清单的偏移量精确框定，不会碰到别的 episode。"""
        shutil.rmtree(
            os.path.join(self.feature_path, f"episode_{ep['global_episode_idx']}"),
            ignore_errors=True,
        )
        start = ep["exec_sample_offset"]
        for i in range(start, start + ep["exec_samples"]):
            p = os.path.join(self.data_path, f"{i}.pkl")
            if os.path.exists(p):
                os.remove(p)

    def run_shard(self, episodes: list[dict], *, resume: bool, report_every: int) -> dict:
        # 与 DatasetProcessor.run() 完全相同的构造参数；一个 buffer 复用整个分片，
        # 每个 episode 结束时 `_process_episode` 内部会 clear()，无跨 episode 残留。
        mem_buffer = MemoryBuffer(
            num_views=1,
            compute_token_drop_score=True,
            token_drop_stride=self.execution_horizon // 2,
            prepare_buffer=True,
        )

        done_steps = 0
        skipped = 0
        # 两个时间基准：t_start 含首个 episode 的 SigLIP 加载与 XLA 编译（实测固定开销
        # ≈36 s），t_steady 从第二个 episode 起算。短跑时前者几乎全是启动开销——1370 步
        # 的实测里 rate 只有 7–12 step/s，而线性拟合出的边际速率是 ~67 step/s。
        # 档位实测必须看 rate_steady，否则档位之间的真实差异会被启动开销淹没。
        t_start = time.perf_counter()
        t_steady = None
        steady_steps = 0
        # 按 h5 文件分组，每个文件只 open 一次（NFS 上反复 open 很贵）
        by_file: dict[str, list[dict]] = {}
        for ep in episodes:
            by_file.setdefault(ep["h5_file"], []).append(ep)

        for h5_name in sorted(by_file):
            path = os.path.join(self.raw_data_path, h5_name)
            with h5py.File(path, "r") as data:
                for ep in sorted(by_file[h5_name], key=lambda e: e["raw_ep_idx"]):
                    if resume and self.episode_is_complete(ep):
                        skipped += 1
                        continue
                    self.purge_episode(ep)
                    t0 = time.perf_counter()
                    self._process_episode(
                        data,
                        ep["raw_ep_idx"],
                        ep["global_episode_idx"],
                        mem_buffer,
                        ep["exec_sample_offset"],
                        ep["total_sample_offset"],
                    )
                    dt = time.perf_counter() - t0
                    done_steps += ep["num_timesteps"]
                    if t_steady is None:
                        t_steady = time.perf_counter()      # 首个 episode 跑完才开始计稳态
                    else:
                        steady_steps += ep["num_timesteps"]
                    elapsed = time.perf_counter() - t_start
                    if report_every and (done_steps // max(1, report_every)) != (
                        (done_steps - ep["num_timesteps"]) // max(1, report_every)
                    ):
                        print(
                            f"PROGRESS steps={done_steps} elapsed={elapsed:.1f}s "
                            f"rate={done_steps / elapsed:.3f} step/s",
                            flush=True,
                        )
                    print(
                        f"EPISODE g={ep['global_episode_idx']} {h5_name}#{ep['raw_ep_idx']} "
                        f"steps={ep['num_timesteps']} took={dt:.1f}s "
                        f"rate={ep['num_timesteps'] / dt:.3f} step/s",
                        flush=True,
                    )

        now = time.perf_counter()
        elapsed = max(1e-9, now - t_start)
        steady_elapsed = max(1e-9, now - t_steady) if t_steady is not None else 0.0
        return {
            "episodes_done": len(episodes) - skipped,
            "episodes_skipped": skipped,
            "steps": done_steps,
            "elapsed_s": elapsed,
            "rate_step_per_s": done_steps / elapsed,
            # 注意语义：这是「启动开销 + 首个 episode」的合计，不是纯启动开销——
            # 稳态计时从首个 episode 跑完才开始。用它反推固定开销时要减掉首个 episode 的步数。
            "startup_plus_first_ep_s": round(elapsed - steady_elapsed, 3) if t_steady else None,
            "steady_steps": steady_steps,
            "steady_elapsed_s": round(steady_elapsed, 3),
            "rate_steady_step_per_s": (steady_steps / steady_elapsed) if steady_steps else None,
        }


def _claim_path(out_dir: str, g: int) -> pathlib.Path:
    return pathlib.Path(out_dir, "_claims", f"_claim_ep{g}")


def try_claim_episode(out_dir: str, g: int, label: str) -> bool:
    """os.open(O_CREAT|O_EXCL) 领一个 episode；已被领走返回 False。"""
    p = _claim_path(out_dir, g)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{label} pid={os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
    finally:
        os.close(fd)
    return True


def release_claim_episode(out_dir: str, g: int) -> None:
    _claim_path(out_dir, g).unlink(missing_ok=True)


class WorkerProcessor(ShardProcessor):
    """动态领任务的本机 worker：与 ShardProcessor 只差「跑哪些 episode」——按 LPT 序逐个 claim。

    `_process_episode` 与 run_shard 的计数口径原样继承；h5 句柄按文件懒开、整个 worker 复用。
    """

    def run_worker(self, episodes: list[dict], *, label: str, resume: bool, report_every: int) -> dict:
        mem_buffer = MemoryBuffer(
            num_views=1,
            compute_token_drop_score=True,
            token_drop_stride=self.execution_horizon // 2,
            prepare_buffer=True,
        )
        done_steps = 0
        skipped = 0
        processed: list[int] = []
        seen_complete: list[int] = []
        t_start = time.perf_counter()
        t_steady = None
        steady_steps = 0
        handles: dict[str, h5py.File] = {}
        try:
            for ep in sorted(episodes, key=lambda e: (-e["num_timesteps"], e["global_episode_idx"])):
                g = ep["global_episode_idx"]
                if resume and self.episode_is_complete(ep):
                    skipped += 1
                    seen_complete.append(g)
                    continue
                if not try_claim_episode(self.dataset_path, g, label):
                    continue
                try:
                    if resume and self.episode_is_complete(ep):   # 领到后再核一次（另一 worker 刚完成）
                        skipped += 1
                        seen_complete.append(g)
                        continue
                    self.purge_episode(ep)
                    h5_name = ep["h5_file"]
                    if h5_name not in handles:
                        handles[h5_name] = h5py.File(os.path.join(self.raw_data_path, h5_name), "r")
                    t0 = time.perf_counter()
                    self._process_episode(
                        handles[h5_name], ep["raw_ep_idx"], g, mem_buffer,
                        ep["exec_sample_offset"], ep["total_sample_offset"],
                    )
                    dt = time.perf_counter() - t0
                    processed.append(g)
                    done_steps += ep["num_timesteps"]
                    if t_steady is None:
                        t_steady = time.perf_counter()
                    else:
                        steady_steps += ep["num_timesteps"]
                    elapsed = time.perf_counter() - t_start
                    if report_every and (done_steps // max(1, report_every)) != (
                        (done_steps - ep["num_timesteps"]) // max(1, report_every)
                    ):
                        print(f"PROGRESS steps={done_steps} elapsed={elapsed:.1f}s "
                              f"rate={done_steps / elapsed:.3f} step/s", flush=True)
                    print(f"EPISODE g={g} {h5_name}#{ep['raw_ep_idx']} steps={ep['num_timesteps']} "
                          f"took={dt:.1f}s rate={ep['num_timesteps'] / dt:.3f} step/s worker={label}", flush=True)
                finally:
                    release_claim_episode(self.dataset_path, g)
        finally:
            for h in handles.values():
                h.close()
        now = time.perf_counter()
        elapsed = max(1e-9, now - t_start)
        steady_elapsed = max(1e-9, now - t_steady) if t_steady is not None else 0.0
        return {
            "episodes_done": len(processed),
            "episodes_skipped": skipped,
            "episodes_processed": sorted(processed),
            "episodes_seen_complete": sorted(seen_complete),
            "steps": done_steps,
            "elapsed_s": elapsed,
            "rate_step_per_s": done_steps / elapsed,
            "startup_plus_first_ep_s": round(elapsed - steady_elapsed, 3) if t_steady else None,
            "steady_steps": steady_steps,
            "steady_elapsed_s": round(steady_elapsed, 3),
            "rate_steady_step_per_s": (steady_steps / steady_elapsed) if steady_steps else None,
        }


def select_episodes(manifest: dict, args: argparse.Namespace) -> list[dict]:
    """先按 subset 过滤，再按 LPT 分片。

    分片一律用与 scan_manifest 相同的 `assign_shards_lpt` **重算**，而不是直接读清单里的
    shard_idx——这样 subset 场景（本地对照集）与全量场景走同一条代码路径。全量且
    num_shards 与清单一致时，重算结果必须与清单逐个相同，这里做硬断言兜底。
    """
    episodes = [dict(e) for e in manifest["episodes"]]

    if args.subset:
        keep = set(json.loads(pathlib.Path(args.subset).read_text())["global_episode_idx"])
        episodes = [e for e in episodes if e["global_episode_idx"] in keep]
        if not episodes:
            raise SystemExit(f"subset 过滤后为空: {args.subset}")

    recomputed = [dict(e) for e in episodes]
    assign_shards_lpt(recomputed, args.num_shards)
    if not args.subset and args.num_shards == manifest["num_shards"]:
        ordered = sorted(episodes, key=lambda e: e["global_episode_idx"])
        for a, b in zip(recomputed, ordered, strict=True):
            if a["shard_idx"] != b["shard_idx"]:
                raise SystemExit(
                    f"分片重算与清单不符 g={a['global_episode_idx']}: "
                    f"{a['shard_idx']} != {b['shard_idx']}"
                )

    mine = [e for e in recomputed if e["shard_idx"] == args.shard_idx]
    mine.sort(key=lambda e: e["global_episode_idx"])
    return mine


def _gpu_uuid() -> str:
    """当前进程可见的第一张卡的 uuid（按 CUDA_VISIBLE_DEVICES 解析；查不到记 unknown）。"""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=False).stdout.strip().splitlines()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis.strip():
            first = vis.split(",")[0].strip()
            for line in out:
                i, u = [s.strip() for s in line.split(",")]
                if first in (i, u):
                    return u
        return out[0].split(",")[1].strip() if out else "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--subset", default="", help="scan_manifest.py sample 的产物；给了就只跑这批")
    ap.add_argument("--resume", action="store_true", help="跳过已完整落盘的 episode")
    ap.add_argument("--require_empty_output", action="store_true", help="首次提交用：输出库必须为空")
    ap.add_argument("--report_every", type=int, default=2000, help="每多少 step 打一行 PROGRESS")
    ap.add_argument("--worker-mode", action="store_true", help="本机动态领任务模式（run_local.py 起）")
    ap.add_argument("--worker-label", default="", help="worker 标签（如 gpu0）")
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()
    if args.worker_mode:
        if not args.worker_label or args.num_workers < 1 or not 0 <= args.worker_idx < args.num_workers:
            raise SystemExit("--worker-mode 需要 --worker-label 与合法的 --worker-idx/--num-workers")
        args.shard_idx, args.num_shards = args.worker_idx, args.num_workers

    manifest = load_manifest(args.manifest)
    if manifest["totals"]["episodes"] == 0:
        raise SystemExit("清单为空")

    if args.require_empty_output:
        feat = os.path.join(args.out, "features")
        if os.path.isdir(feat) and any(os.scandir(feat)):
            raise SystemExit(f"要求空输出库但已有内容: {feat}")

    if args.worker_mode:
        mine = [dict(e) for e in manifest["episodes"]]
        if args.subset:
            keep = set(json.loads(pathlib.Path(args.subset).read_text())["global_episode_idx"])
            mine = [e for e in mine if e["global_episode_idx"] in keep]
    else:
        mine = select_episodes(manifest, args)
    steps = sum(e["num_timesteps"] for e in mine)
    print(
        f"[shard {args.shard_idx}/{args.num_shards}] episode={len(mine)} timestep={steps} "
        f"raw_dir={args.raw_dir} out={args.out}" + (f" worker={args.worker_label}" if args.worker_mode else ""),
        flush=True,
    )

    if args.worker_mode:
        proc = WorkerProcessor(args.raw_dir, args.out)
        stats = proc.run_worker(mine, label=args.worker_label, resume=args.resume,
                                report_every=args.report_every)
        covered = sorted(set(stats.pop("episodes_processed")) | set(stats.pop("episodes_seen_complete")))
    else:
        proc = ShardProcessor(args.raw_dir, args.out)
        stats = proc.run_shard(mine, resume=args.resume, report_every=args.report_every)
        covered = [e["global_episode_idx"] for e in mine]

    # 硬件/软件指纹：交付口径承诺「metadata 逐条带硬件/软件 provenance，finalize 断言
    # 全体同源」，但此前 sidecar 里一个指纹字段都没有，finalize 只能记它自己那个节点的
    # 信息（等于出生证明上写护士的名字）。这里由**产出数据的分片本人**记录。
    # jax 在此刻早已被 MemoryBuffer(prepare_buffer=True) 初始化完，取它零额外开销；
    # 放模块顶层反而会让 --help 也去加载 jax。
    try:
        import jax
        import jaxlib
        _gpu = str(jax.devices()[0].device_kind)
        _jax, _jaxlib = jax.__version__, jaxlib.__version__
    except Exception as exc:  # 采集失败必须留痕，不能静默成空串——finalize 会据此判死
        _gpu = _jax = _jaxlib = f"unavailable: {exc}"
    # v2-motionmem 起本机运行、不再有 SLURM 字段；指纹改记 worker 标签与所见 GPU（uuid 由 nvidia-smi 查）
    fingerprint = {
        "host": socket.gethostname(),
        "worker": args.worker_label or f"shard{args.shard_idx}",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": _gpu_uuid(),
        "gpu_device_kind": _gpu,
        "jax": _jax,
        "jaxlib": _jaxlib,
        "git_commit": git_commit(_REPO_ROOT),
        "pid": os.getpid(),
    }

    pathlib.Path(args.out, "meta").mkdir(parents=True, exist_ok=True)
    side = pathlib.Path(args.out, "meta", f"_shard{args.shard_idx}of{args.num_shards}.json")
    # 指纹用嵌套子对象而不是平铺：finalize 汇总时有一份字段白名单，平铺的话每加一个
    # 指纹字段都得同步改白名单（漏改会被 `if k in d` 静默吞掉）。schema_version 让
    # finalize 能对「产出于指纹引入之前的旧库」给出精确报错而不是含糊的 KeyError。
    side.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "shard_idx": args.shard_idx,
                "num_shards": args.num_shards,
                "manifest_sha256": manifest["sha256"],
                "subset": args.subset or None,
                "worker_mode": bool(args.worker_mode),
                "episodes": covered,
                "fingerprint": fingerprint,
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    steady = stats["rate_steady_step_per_s"]
    print(
        f"SHARD_DONE shard={args.shard_idx} episodes={stats['episodes_done']} "
        f"skipped={stats['episodes_skipped']} steps={stats['steps']} "
        f"elapsed={stats['elapsed_s']:.1f}s rate={stats['rate_step_per_s']:.3f} step/s "
        f"steady_steps={stats['steady_steps']} "
        f"rate_steady={steady:.3f} step/s" if steady else
        f"SHARD_DONE shard={args.shard_idx} episodes={stats['episodes_done']} "
        f"skipped={stats['episodes_skipped']} steps={stats['steps']} "
        f"elapsed={stats['elapsed_s']:.1f}s rate={stats['rate_step_per_s']:.3f} step/s "
        f"steady_steps=0 rate_steady=NA（只有一个 episode，全是启动开销）",
        flush=True,
    )


if __name__ == "__main__":
    main()
