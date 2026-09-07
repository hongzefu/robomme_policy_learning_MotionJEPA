#!/usr/bin/env python
"""
probe_vulkan_makeenv.py —— 纯排查脚本：找出 examples/robomme/eval.py 单进程内
第 28 次 EnvRunner.make_env 必抛 `vk::createInstanceUnique: ErrorIncompatibleDriver` 的根因。

不起 policy、不连 websocket、不做推理。只做一件事：循环
    make_env(episode) → （可选）reset 一次 → close_env()
共 --rounds 次，每轮打印一行资源快照，异常时打印 `CRASH at round=i err=...`。

用法（micromamba robomme 环境，只用 GPU 7）：
    CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
      /scratch/hongze/micromamba/envs/robomme/bin/python \
      scripts/training/legacy-eval/probe_vulkan_makeenv.py --rounds 35

开关见 build_parser()。所有输出落 v1-store/reports/tic-vulkan/。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO / "v1-store" / "reports" / "tic-vulkan"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAPIEN/Vulkan make_env 泄漏排查")
    p.add_argument("--rounds", type=int, default=35, help="循环轮数")
    p.add_argument("--task", type=str, default="ButtonUnmask", help="任务名（TASK_NAME_LIST 之一）")
    p.add_argument("--episode", type=int, default=0, help="固定用哪一集（默认 0，轮内不变，排除集间差异）")
    p.add_argument("--reset", dest="reset", action="store_true", default=True,
                   help="每轮 reset 一次（默认开）")
    p.add_argument("--no-reset", dest="reset", action="store_false",
                   help="只 make_env / close_env，不 reset（快很多，用于快速定位）")
    p.add_argument("--gc", action="store_true", help="每轮显式 gc.collect()")
    p.add_argument("--renderer-release", action="store_true",
                   help="每轮尝试显式释放 SAPIEN 渲染资源（clear_cache + 丢弃 RenderSystem 引用）")
    p.add_argument("--reuse-env", action="store_true", help="只建一次 env，每轮只 reset")
    p.add_argument("--subprocess", action="store_true", help="每轮在独立子进程里建/关")
    p.add_argument("--pin-renderer", action="store_true",
                   help="候选修法：进程启动时建一个 sapien RenderSystem 并全局持有，"
                        "让 svulkan2 全局 Context 的 shared_ptr 永不过期，后续 env 复用同一个 VkInstance")
    p.add_argument("--vk-loader-debug", action="store_true",
                   help="设 VK_LOADER_DEBUG=all，把 loader 日志重定向到文件并统计 vkCreateInstance/vkDestroyInstance")
    p.add_argument("--tag", type=str, default="baseline", help="本次实验标签，决定输出文件名")
    p.add_argument("--child-round", type=int, default=-1, help="内部用：子进程模式下的轮号")
    return p


# ── 早期环境变量设置：必须在 import sapien 之前完成 ──────────────────────────
_ARGS = build_parser().parse_args()
_VK_LOG_PATH = REPORT_DIR / f"vkloader-{_ARGS.tag}.log"
if _ARGS.vk_loader_debug:
    os.environ["VK_LOADER_DEBUG"] = "all"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if _ARGS.child_round < 0:
        # 主进程负责建/清空日志；C 层 stderr 也要落进去，所以 dup2 掉 fd 2
        _VK_LOG_PATH.write_text("")
    _vk_fh = open(_VK_LOG_PATH, "a", buffering=1)
    os.dup2(_vk_fh.fileno(), 2)

sys.path.insert(0, str(REPO / "examples" / "robomme"))

import gc  # noqa: E402


# ── 资源快照 ────────────────────────────────────────────────────────────────
def fd_stats() -> tuple[int, int]:
    """返回 (fd 总数, 指向 /dev/nvidia* 的 fd 数)。"""
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return -1, -1
    nvidia = 0
    for n in names:
        try:
            target = os.readlink(f"/proc/self/fd/{n}")
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            nvidia += 1
    return len(names), nvidia


def maps_stats() -> tuple[int, int, int, int]:
    """返回 (libvulkan+libnvidia* 映射行数, 这些行涉及的不同 so 文件数, /dev/nvidia* 映射行数, maps 总行数)。"""
    hit_lines = 0
    total = 0
    devnv = 0
    files = set()
    try:
        with open("/proc/self/maps", "r") as fh:
            for line in fh:
                total += 1
                parts = line.split(None, 5)
                path = parts[5].strip() if len(parts) >= 6 else ""
                base = os.path.basename(path)
                if path.startswith("/dev/nvidia"):
                    devnv += 1
                if base.startswith("libvulkan") or base.startswith("libnvidia") or base.startswith("libGLX_nvidia"):
                    hit_lines += 1
                    files.add(path)
    except OSError:
        return -1, -1, -1, -1
    return hit_lines, len(files), devnv, total


def proc_status(field: str) -> float:
    """读 /proc/self/status 的某个数值字段（kB 的转 MB，纯计数原样返回）。"""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    parts = line.split()
                    val = float(parts[1])
                    return val / 1024.0 if len(parts) > 2 and parts[2] == "kB" else val
    except OSError:
        pass
    return -1.0


def rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def gpu_mem_mb() -> float:
    """本进程在渲染卡上的显存（MiB）。CUDA_VISIBLE_DEVICES 的首个值即物理卡号。"""
    dev = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "0").split(",")[0].strip()
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", dev,
             "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return -1.0
    mine = os.getpid()
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == mine:
            try:
                return float(parts[1])
            except ValueError:
                return -1.0
    return 0.0


def sapien_obj_stats() -> tuple[int, Counter]:
    """gc 中 sapien / svulkan2 相关对象计数与按类型细分。"""
    counter: Counter = Counter()
    for obj in gc.get_objects():
        tp = type(obj)
        mod = getattr(tp, "__module__", "")
        if not isinstance(mod, str):
            continue
        if mod.startswith("sapien") or mod.startswith("svulkan2"):
            counter[f"{mod}.{tp.__name__}"] += 1
    return sum(counter.values()), counter


def vk_instance_counts() -> tuple[int, int]:
    """从 VK_LOADER_DEBUG 日志里数 vkCreateInstance / vkDestroyInstance 出现次数。"""
    if not _ARGS.vk_loader_debug or not _VK_LOG_PATH.exists():
        return -1, -1
    create = destroy = 0
    with open(_VK_LOG_PATH, "r", errors="replace") as fh:
        for line in fh:
            if "vkCreateInstance" in line:
                create += 1
            if "vkDestroyInstance" in line:
                destroy += 1
    return create, destroy


def snapshot() -> dict:
    fd_total, fd_nv = fd_stats()
    vk_lines, vk_files, devnv_maps, maps_total = maps_stats()
    sap_n, sap_detail = sapien_obj_stats()
    vk_create, vk_destroy = vk_instance_counts()
    return {
        "fd": fd_total,
        "fd_nvidia": fd_nv,
        "vk_maps": vk_lines,
        "vk_so_files": vk_files,
        "devnv_maps": devnv_maps,
        "maps_total": maps_total,
        "threads": int(proc_status("Threads")),
        "vmsize_mb": round(proc_status("VmSize"), 1),
        "rss_mb": round(rss_mb(), 1),
        "gpu_mem_mb": round(gpu_mem_mb(), 1),
        "sapien_objs": sap_n,
        "vk_create": vk_create,
        "vk_destroy": vk_destroy,
        "sapien_detail": dict(sap_detail.most_common(8)),
    }


def fmt(round_idx: int, snap: dict, dt: float) -> str:
    return (
        f"ROUND {round_idx} fd={snap['fd']} fd_nvidia={snap['fd_nvidia']} "
        f"vk_maps={snap['vk_maps']} vk_so_files={snap['vk_so_files']} devnv_maps={snap['devnv_maps']} "
        f"maps_total={snap['maps_total']} threads={snap['threads']} vmsize_mb={snap['vmsize_mb']} "
        f"rss_mb={snap['rss_mb']} gpu_mem_mb={snap['gpu_mem_mb']} sapien_objs={snap['sapien_objs']} "
        f"vk_create={snap['vk_create']} vk_destroy={snap['vk_destroy']} dt={dt:.2f}s"
    )


# ── 渲染器显式释放 ──────────────────────────────────────────────────────────
def try_renderer_release() -> str:
    """尽力显式释放 SAPIEN 渲染侧资源，返回做了哪些动作的描述。"""
    done = []
    try:
        import sapien
        import sapien.render as sr
    except ImportError as exc:
        return f"import-failed:{exc}"
    try:
        sr.clear_cache(models=True, images=True, shaders=True)
        done.append("clear_cache(shaders=True)")
    except Exception as exc:  # noqa: BLE001 - 排查脚本，任何失败都只记录
        done.append(f"clear_cache-failed:{exc}")
    # 丢弃仍被 python 侧持有的 RenderSystem / SapienRenderer / Scene 引用
    dropped = 0
    for obj in list(gc.get_objects()):
        tp = type(obj)
        mod = getattr(tp, "__module__", "")
        if not isinstance(mod, str):
            continue
        if mod.startswith("sapien") and tp.__name__ in ("RenderSystem", "SapienRenderer", "Scene"):
            dropped += 1
    done.append(f"still_alive_render_objs={dropped}")
    gc.collect()
    gc.collect()
    done.append("gc.collect x2")
    return " ".join(done)


_PINNED = []


def pin_renderer() -> str:
    """候选修法：建一个 sapien RenderSystem 并放进模块级列表长期持有。

    svulkan2 的全局渲染 Context 由 shared_ptr 管；ManiSkill 每次 close 会把 scene 与
    RenderSystem 全部释放，Context 引用计数归零 → 下次建 env 重新 vkCreateInstance。
    这里全程持有一个 RenderSystem，让 Context 引用计数永不归零。
    """
    try:
        import sapien
        import sapien.render as sr
    except ImportError as exc:
        return f"import-failed:{exc}"
    try:
        dev = sapien.Device("cuda")
        _PINNED.append(sr.RenderSystem(dev))
        return f"pinned RenderSystem on {dev}"
    except Exception as exc:  # noqa: BLE001
        return f"pin-failed:{type(exc).__name__}: {exc}"


# ── 单轮工作 ────────────────────────────────────────────────────────────────
def one_round(args, runner_holder: dict) -> None:
    from env_runner import EnvRunner

    if args.reuse_env:
        runner = runner_holder.get("runner")
        if runner is None:
            runner = EnvRunner(args.task, str(REPORT_DIR / "videos"), max_steps=1300)
            runner.make_env(args.episode)
            runner_holder["runner"] = runner
        runner.env.reset()
        return

    runner = EnvRunner(args.task, str(REPORT_DIR / "videos"), max_steps=1300)
    runner.make_env(args.episode)
    if args.reset:
        runner.get_init_obs()
    runner.close_env()
    del runner


def run_child_round(args) -> int:
    """子进程模式：只跑一轮，把快照以 JSON 打到 stdout 最后一行。"""
    err = ""
    try:
        one_round(args, {})
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    snap = snapshot()
    snap["err"] = err
    print("CHILD_SNAPSHOT " + json.dumps(snap))
    return 1 if err else 0


def main() -> int:
    args = _ARGS
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.child_round >= 0:
        return run_child_round(args)

    print(f"=== PROBE tag={args.tag} rounds={args.rounds} task={args.task} episode={args.episode} "
          f"reset={args.reset} gc={args.gc} renderer_release={args.renderer_release} "
          f"reuse_env={args.reuse_env} subprocess={args.subprocess} vk_loader_debug={args.vk_loader_debug} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} pid={os.getpid()} ===")
    print("=== PRE " + json.dumps(snapshot()) + " ===")

    if args.pin_renderer:
        print(f"  [pin-renderer] {pin_renderer()}")
        print("=== POST_PIN " + json.dumps(snapshot()) + " ===")

    records = []
    crash_round = -1
    holder: dict = {}

    for i in range(1, args.rounds + 1):
        t0 = time.time()
        err = ""
        if args.subprocess:
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--child-round", str(i), "--rounds", "1", "--task", args.task,
                   "--episode", str(args.episode), "--tag", args.tag]
            cmd.append("--reset" if args.reset else "--no-reset")
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
            snap = {}
            for line in proc.stdout.splitlines():
                if line.startswith("CHILD_SNAPSHOT "):
                    snap = json.loads(line[len("CHILD_SNAPSHOT "):])
            err = snap.pop("err", "") if snap else f"child-no-snapshot rc={proc.returncode}"
            if not snap:
                snap = snapshot()
            snap["child_rc"] = proc.returncode
        else:
            try:
                one_round(args, holder)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            if args.renderer_release:
                rel = try_renderer_release()
                if i == 1:
                    print(f"  [renderer-release] {rel}")
            if args.gc:
                gc.collect()
            snap = snapshot()
        dt = time.time() - t0
        print(fmt(i, snap, dt))
        rec = dict(snap)
        rec["round"] = i
        rec["dt_s"] = round(dt, 2)
        rec["err"] = err
        records.append(rec)
        if err:
            print(f"CRASH at round={i} err={err}")
            crash_round = i
            break

    out = REPORT_DIR / f"probe-{args.tag}.json"
    with open(out, "w") as fh:
        json.dump({
            "tag": args.tag,
            "argv": sys.argv[1:],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "rounds_requested": args.rounds,
            "rounds_done": len(records),
            "crash_round": crash_round,
            "records": records,
        }, fh, indent=2)
    print(f"=== PROBE_RESULT tag={args.tag} rounds_done={len(records)} crash_round={crash_round} out={out} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
