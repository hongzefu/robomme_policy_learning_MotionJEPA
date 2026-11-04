#!/usr/bin/env python3
"""本机 attention 评估驱动（计划第二部分 §1.4）：不经 Slurm 队列，逐 chunk「起 serve_policy_attn → 跑一个 eval.py chunk → 收尾」。

为什么不用 resident_worker：它把 worker 绑死在 Slurm step 内（worker必须在指定JobID的GPU srun step内执行），本机跑不了；
这里通过 import 复用它的 server_command / client_command / worker_environment / stop_process_tree，只改三处：
模块换成 scripts.interpret.serve_policy_attn、在 policy:checkpoint 之前插 --attn-out、motion 关闭的模型去掉 --motion-online-gpu。

spec（JSON）键：repo, policy_python, sim_python, stage2_python, checkpoint, policy_config,
model_kind（history | stage2_pool4_concat）, model_seed=42, purpose=validation, cuda, attn_out, run_root, manifest,
xla_mem_fraction；可选 port（起始端口，缺省 0=临时端口）、startup_timeout_seconds（缺省 600）、save_top_frames、
rpc_timeout_seconds、save_prefix（缺省 false；为真时服务端加 --save-prefix，另写 prefix_<seq>.npz）、
ablate（缺省 "none"；其他值原样透传 --ablate，语义见 serve_policy_attn 文档，如 mask_motion / mask_vis / heads:L3H0）。
run_identity_sha256 由本脚本自算（见 run_identity；ablate 非 none 时并入身份，none 时身份与 S7 之前相同）。

逐局打印 EPISODE_RESULT task=<t> ep=<i> success=<0|1> npz=<n>，末行
RUN_LOCAL=PASS|FAIL model=<model_kind> episodes=<n> success=<k> npz=<n>。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
import uuid

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.evaluation import manifest as M  # noqa: E402
from scripts.evaluation import resident_worker as RW  # noqa: E402

SERVER_MODULE, ATTN_MODULE = "scripts.training.serve_policy", "scripts.interpret.serve_policy_attn"
# model_kind 合法值取自 src/mme_vla_suite/policies/policy_config.py 生成 metadata 处
MODEL_KINDS = ("history", "stage2_pool4_concat")
MOTION_MODEL_KIND = "stage2_pool4_concat"
XLA_FLAGS = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
REQUIRED_KEYS = ("repo", "policy_python", "sim_python", "stage2_python", "checkpoint", "policy_config", "model_kind",
                 "cuda", "attn_out", "run_root", "manifest", "xla_mem_fraction")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_spec(path):
    spec = read_json(path)
    missing = [key for key in REQUIRED_KEYS if key not in spec]
    if missing:
        raise SystemExit(f"spec 缺键: {missing}")
    if spec["model_kind"] not in MODEL_KINDS:
        raise SystemExit(f"model_kind 只能是 {MODEL_KINDS}，得到 {spec['model_kind']!r}")
    spec.setdefault("model_seed", 42)
    spec.setdefault("purpose", "validation")
    spec["cuda"] = str(spec["cuda"])
    return spec


def run_identity(spec, manifest):
    """本地评估身份：清单封签 + checkpoint 绝对路径 + 配置名 + model_kind + seed（bind_model 只接受 20k/60k 步，不能用）。"""
    ident = {"manifest_seal": manifest["sha256"], "checkpoint": str(Path(spec["checkpoint"]).resolve()),
             "policy_config": spec["policy_config"], "model_kind": spec["model_kind"],
             "model_seed": int(spec["model_seed"])}
    if ablate_of(spec) != "none":
        ident["ablate"] = ablate_of(spec)
    return M.digest(ident)


def ablate_of(spec):
    value = spec.get("ablate")
    return "none" if value is None else str(value).strip() or "none"


def select_chunks(manifest, episodes_arg=None, limit=None):
    """返回 [(chunk_id, [episode, ...])]；给 --episodes 时只跑这些局，每局单独一块。"""
    by_key = {episode["episode_key"]: episode for episode in manifest["episodes"]}
    if episodes_arg:
        keys = [item.strip() for item in episodes_arg.split(",") if item.strip()]
        unknown = [key for key in keys if key not in by_key]
        if unknown:
            raise SystemExit(f"--episodes 不在清单里: {unknown}")
        if len(set(keys)) != len(keys):
            raise SystemExit("--episodes 有重复局")
        chunks = [(M.digest([key]), [by_key[key]]) for key in keys]
    else:
        chunks = [(chunk["chunk_id"], [by_key[key] for key in chunk["episode_keys"]]) for chunk in manifest["chunks"]]
    return chunks[:limit] if limit else chunks


def port_in_use(port):
    """与 (exec 3<>/dev/tcp/127.0.0.1/$PORT) 同效：能连上即已被占用。"""
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def pick_port(start=0, tries=64):
    for offset in range(tries):
        if start:
            port = start + offset
        else:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        if not port_in_use(port):
            return port
    raise RuntimeError(f"从 {start} 起 {tries} 个端口均被占用")


def build_server_command(spec, nonce, port):
    """resident_worker.server_command 原形 → 换模块、插 --attn-out、motion 关闭去掉 --motion-online-gpu。"""
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = spec["cuda"]   # server_command 从环境读卡号
    try:
        command = list(RW.server_command(spec, nonce, port))
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    if command.count(SERVER_MODULE) != 1 or command.count("policy:checkpoint") != 1:
        raise RuntimeError(f"server_command 形状变化: {command}")
    command[command.index(SERVER_MODULE)] = ATTN_MODULE
    if spec["model_kind"] != MOTION_MODEL_KIND:
        index = command.index("--motion-online-gpu")
        del command[index:index + 2]
    extra = ["--attn-out", str(spec["attn_out"])]
    if spec.get("save_top_frames") is not None:
        extra += ["--save-top-frames", str(int(spec["save_top_frames"]))]
    if spec.get("save_prefix"):
        extra += ["--save-prefix"]
    if ablate_of(spec) != "none":
        extra += ["--ablate", ablate_of(spec)]
    index = command.index("policy:checkpoint")
    return command[:index] + extra + command[index:]


def build_environment(spec):
    environment = RW.worker_environment(spec)
    environment.update(CUDA_VISIBLE_DEVICES=spec["cuda"], XLA_FLAGS=XLA_FLAGS,
                       XLA_PYTHON_CLIENT_MEM_FRACTION=str(spec["xla_mem_fraction"]))
    return environment


def tail(path, lines=40):
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "<无日志>"


def wait_ready(server, port, log_path, timeout):
    end = time.monotonic() + timeout
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"策略服务启动期间退出 code={server.returncode}\n{tail(log_path)}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() >= end:
            raise TimeoutError(f"策略服务 {timeout}s 内未就绪\n{tail(log_path)}")
        threading.Event().wait(0.5)


def run_chunk(spec, index, chunk_id, episodes, popen=subprocess.Popen):
    run_root = Path(spec["run_root"])
    nonce = uuid.uuid4().hex
    attempt = run_root / "attempts" / f"{index:03d}-{chunk_id[:12]}-{nonce[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    port = pick_port(int(spec.get("port", 0)))
    environment = build_environment(spec)
    grant_path, output, drain_file = attempt / "grant.json", attempt / "client", attempt / "drain"
    write_json(grant_path, {"chunk_id": chunk_id, "run_identity_sha256": spec["run_identity_sha256"],
                            "episodes": episodes})
    server_cmd = build_server_command(spec, nonce, port)
    client_cmd = RW.client_command(spec, grant_path, output, nonce, port, drain_file)
    write_json(attempt / "commands.json", {"server": server_cmd, "client": client_cmd, "port": port, "nonce": nonce,
                                           "chunk_id": chunk_id, "episode_keys": [e["episode_key"] for e in episodes]})
    print(f"CHUNK_BEGIN index={index} chunk_id={chunk_id} port={port} nonce={nonce} "
          f"episodes={','.join(e['episode_key'] for e in episodes)} attempt={attempt}", flush=True)
    server = client = None
    code = None
    try:
        with (attempt / "server.log").open("ab", buffering=0) as log:
            server = popen(server_cmd, cwd=spec["repo"], env=environment, stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
        wait_ready(server, port, attempt / "server.log", float(spec.get("startup_timeout_seconds", 600)))
        with (attempt / "sim.log").open("ab", buffering=0) as log:
            client = popen(client_cmd, cwd=spec["repo"], env=environment, stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
        while client.poll() is None:
            if server.poll() is not None:
                raise RuntimeError(f"策略服务在仿真期间退出 code={server.returncode}\n{tail(attempt / 'server.log')}")
            threading.Event().wait(0.5)
        code = client.returncode
    except Exception as exc:   # 记失败、照样收尾并汇总；SystemExit / KeyboardInterrupt 原样上抛
        print(f"CHUNK_ERROR index={index} error={type(exc).__name__}: {exc}", flush=True)
        code = -1
    finally:
        RW.stop_process_tree(client)
        RW.stop_process_tree(server)
    print(f"CHUNK_END index={index} client_exit={code}", flush=True)
    return attempt, output, code


def collect(spec, attempt, output, episodes):
    """把 terminal 汇到 accepted/，并把局身份并入 attn_out/<nonce>/episode.json。返回逐局结果。"""
    accepted = Path(spec["run_root"]) / "accepted"
    accepted.mkdir(parents=True, exist_ok=True)
    attn_out = Path(spec["attn_out"])
    records = {}
    for kind in ("started", "terminal"):
        for path in sorted(output.glob(f"*.{kind}.json")):
            record = read_json(path)
            records.setdefault(record["episode_key"], {})[kind] = record
            if kind == "terminal":
                shutil.copy2(path, accepted / f"{record['episode_key']}.{record['episode_nonce'][:8]}.terminal.json")
    results = []
    for episode in episodes:
        key = episode["episode_key"]
        found = records.get(key, {})
        terminal, started = found.get("terminal"), found.get("started")
        nonce = (terminal or started or {}).get("episode_nonce")
        npz = len(list((attn_out / nonce).glob("infer_*.npz"))) if nonce else 0
        if nonce and (attn_out / nonce / "episode.json").exists():
            info = read_json(attn_out / nonce / "episode.json")
            info.update({k: (terminal or started)[k] for k in ("task_name", "episode_id", "env_seed", "difficulty",
                                                               "episode_key", "chunk_id", "run_identity_sha256")})
            info.update(outcome=terminal["outcome"] if terminal else None, attempt=str(attempt))
            write_json(attn_out / nonce / "episode.json", info)
        # terminal 的 rpc_seq = 1 + batch_seq + replan_seq，官方循环里每次 infer 前恰好一次 add_buffer
        infers = (int(terminal["rpc_seq"]) - 1) // 2 if terminal else None
        success = int(bool(terminal) and terminal["outcome"] == "success")
        results.append({"task": episode["task_name"], "ep": episode["episode_id"], "success": success, "npz": npz,
                        "infers": infers, "terminal": terminal is not None, "nonce": nonce})
        print(f"EPISODE_RESULT task={episode['task_name']} ep={episode['episode_id']} success={success} npz={npz}",
              flush=True)
    return results


def main(argv=None, popen=subprocess.Popen):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--episodes", default=None, help="task:id,...：只跑清单里这些局，每局单独一块")
    parser.add_argument("--chunks", type=int, default=None, help="只跑前 N 块（冒烟用）")
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    manifest = M.validate_episodes(read_json(spec["manifest"]), official=False)
    spec["run_identity_sha256"] = run_identity(spec, manifest)
    Path(spec["run_root"]).mkdir(parents=True, exist_ok=True)
    Path(spec["attn_out"]).mkdir(parents=True, exist_ok=True)
    chunks = select_chunks(manifest, args.episodes, args.chunks)
    print(f"RUN_LOCAL_BEGIN model={spec['model_kind']} run_identity={spec['run_identity_sha256']} "
          f"manifest_seal={manifest['sha256']} chunks={len(chunks)} cuda={spec['cuda']} ablate={ablate_of(spec)}",
          flush=True)

    def stop(signum, _frame):
        raise SystemExit(128 + signum)   # 走 finally 收掉服务端与客户端进程组

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    results, codes = [], []
    for index, (chunk_id, episodes) in enumerate(chunks):
        attempt, output, code = run_chunk(spec, index, chunk_id, episodes, popen=popen)
        codes.append(code)
        results.extend(collect(spec, attempt, output, episodes))
        if code == -1:
            break   # 服务端或驱动故障：不再起后续 chunk
    ok = (all(code == 0 for code in codes) and all(r["terminal"] for r in results)
          and all(r["npz"] > 0 and r["npz"] == r["infers"] for r in results))
    total_npz, success = sum(r["npz"] for r in results), sum(r["success"] for r in results)
    write_json(Path(spec["run_root"]) / f"run_local-{uuid.uuid4().hex[:8]}.json",
               {"run_identity_sha256": spec["run_identity_sha256"], "client_exit_codes": codes, "results": results})
    print(f"RUN_LOCAL={'PASS' if ok else 'FAIL'} model={spec['model_kind']} episodes={len(results)} "
          f"success={success} npz={total_npz}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
