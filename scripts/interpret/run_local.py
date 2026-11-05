#!/usr/bin/env python3
"""本机 attention 评估驱动（计划第二部分 §1.4）：不经 Slurm 队列，逐 chunk「起 serve_policy_attn → 跑一个 eval.py chunk → 收尾」。

为什么不用 resident_worker：它把 worker 绑死在 Slurm step 内（worker必须在指定JobID的GPU srun step内执行），本机跑不了；
这里通过 import 复用它的 server_command / client_command / worker_environment / stop_process_tree，只改三处：
模块换成 scripts.interpret.serve_policy_attn、在 policy:checkpoint 之前插 --attn-out、motion 关闭的模型去掉 --motion-online-gpu。

spec（JSON）键：repo, policy_python, sim_python, stage2_python, checkpoint, policy_config,
model_kind（history | stage2_pool4_concat）, model_seed=42, purpose=validation, cuda, attn_out, run_root, manifest,
xla_mem_fraction；可选 port（起始端口，缺省 0=临时端口）、startup_timeout_seconds（缺省 600）、save_top_frames、
rpc_timeout_seconds、save_prefix（缺省 false；为真时服务端加 --save-prefix，另写 prefix_<seq>.npz）、
ablate（缺省 "none"；其他值原样透传 --ablate，语义见 serve_policy_attn 文档，如 mask_motion / mask_vis / heads:L3H0 /
motion_zero / motion_swap:<供体目录>）。
StopCube 可选键：stopcube_mode（normal、hold_extended、oracle_query、oracle_calibrate）、stopcube_config（固定 JSON 表路径）；
同时提供时改用 interpret 记录入口，模式与配置文件字节 sha256 并入身份，其他任务不受影响。
Oracle 另用 query_interval=4（节奏诊断可16）、query_record=10；校准另需 calibration_plan、calibration_kind，
多模型选点文件由 model_id（默认 model_kind）选择。校准选点文件内容也并入身份。
S8 motion 内容替换的可选键：
  record_motion   目录；服务端加 --record-motion，每局写 <目录>/<nonce>.npz，本脚本收尾按 terminal 局身份改名为
                  <目录>/<task>_ep<id>.npz（与 motion_swap 供体文件同名规则）；局的 episode.json 记 motion_record。
  swap_map        motion_swap 的供体映射，{"ButtonUnmaskSwap:4": "ButtonUnmaskSwap:8", ...} 或字符串
                  "ButtonUnmaskSwap:4=ButtonUnmaskSwap:8,..."；ablate 含 motion_swap 时必填、否则不得给出。
                  给出时每局单独成块（chunk_id = digest([episode_key])，与 --episodes 同法），服务端按块加
                  --swap-donor <供体局>：服务端在 reset 时只知道 nonce，单局成块后「本服务进程服务的局」即确定。
                  选中的每局都必须在 swap_map 里，否则启动前报错。
run_identity_sha256 由本脚本自算（见 run_identity；ablate 非 none 时并入身份，none 时身份与 S7 之前相同；
swap_map 给出时也并入身份；record_motion 只记录不改行为，不并入）。

逐局打印 EPISODE_RESULT task=<t> ep=<i> success=<0|1> npz=<n>，末行
RUN_LOCAL=PASS|FAIL model=<model_kind> episodes=<n> success=<k> npz=<n>。
"""
from __future__ import annotations

import argparse
import hashlib
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
    has_swap = any(part.strip().startswith("motion_swap:") for part in ablate_of(spec).split("+"))
    if spec.get("swap_map") is not None:
        spec["swap_map"] = parse_swap_map(spec["swap_map"])
    if has_swap != bool(spec.get("swap_map")):
        raise SystemExit("spec: ablate 含 motion_swap 时必须给非空 swap_map，反之 swap_map 只能配 motion_swap")
    uses_motion = has_swap or spec.get("record_motion") or any(
        part.strip() == "motion_zero" for part in ablate_of(spec).split("+"))
    if uses_motion and spec["model_kind"] != MOTION_MODEL_KIND:
        raise SystemExit(f"spec: motion_zero / motion_swap / record_motion 只适用于 {MOTION_MODEL_KIND}")
    if spec.get("stopcube_mode") is not None:
        if spec["stopcube_mode"] not in ("normal", "hold_extended", "oracle_query", "oracle_calibrate") or not spec.get("stopcube_config"):
            raise SystemExit("spec: stopcube_mode 非法或未提供 stopcube_config")
        spec["stopcube_config"] = str(Path(spec["stopcube_config"]).resolve())
        if spec["stopcube_mode"].startswith("oracle_"):
            spec.setdefault("query_interval", 4)
            spec.setdefault("query_record", 10)
            if spec["query_interval"] not in (4, 16) or spec["query_record"] != 10 or ablate_of(spec) != "none":
                raise SystemExit("Oracle 固定10预测、间隔4或诊断16，且禁止 ablate")
        if spec["stopcube_mode"] == "oracle_calibrate":
            from scripts.interpret.stopcube_runner import CALIBRATION_KINDS
            if not spec.get("calibration_plan") or spec.get("calibration_kind") not in CALIBRATION_KINDS:
                raise SystemExit("校准须提供 calibration_plan 和固定的 calibration_kind")
            spec["calibration_plan"] = str(Path(spec["calibration_plan"]).resolve())
    elif spec.get("stopcube_config") is not None:
        raise SystemExit("spec: stopcube_config 需要 stopcube_mode")
    return spec


def _episode_key(text):
    task, sep, ep = str(text).strip().rpartition(":")
    if not sep or not task or not ep.strip().isdigit():
        raise SystemExit(f"swap_map 局写法应为 task:ep，得到 {text!r}")
    return f"{task}:{int(ep)}"


def parse_swap_map(value):
    """dict 或 "a:1=b:2,..." → {"task:ep": "task:ep"}（局号规范成整数写法）。"""
    if isinstance(value, str):
        pairs = []
        for item in (x.strip() for x in value.split(",") if x.strip()):
            target, sep, donor = item.partition("=")
            if not sep:
                raise SystemExit(f"swap_map 项应为 task:ep=task:ep，得到 {item!r}")
            pairs.append((target, donor))
    elif isinstance(value, dict):
        pairs = list(value.items())
    else:
        raise SystemExit(f"swap_map 须为 dict 或字符串，得到 {type(value).__name__}")
    out = {}
    for target, donor in pairs:
        key = _episode_key(target)
        if key in out:
            raise SystemExit(f"swap_map 目标局重复: {key}")
        out[key] = _episode_key(donor)
    return out


def run_identity(spec, manifest):
    """本地评估身份：清单封签 + checkpoint 绝对路径 + 配置名 + model_kind + seed（bind_model 只接受 20k/60k 步，不能用）。"""
    ident = {"manifest_seal": manifest["sha256"], "checkpoint": str(Path(spec["checkpoint"]).resolve()),
             "policy_config": spec["policy_config"], "model_kind": spec["model_kind"],
             "model_seed": int(spec["model_seed"])}
    if ablate_of(spec) != "none":
        ident["ablate"] = ablate_of(spec)
    if spec.get("swap_map"):
        ident["swap_map"] = dict(sorted(parse_swap_map(spec["swap_map"]).items()))
    if spec.get("stopcube_mode") is not None:
        ident["stopcube_mode"] = spec["stopcube_mode"]
        ident["stopcube_config_sha256"] = hashlib.sha256(Path(spec["stopcube_config"]).read_bytes()).hexdigest()
        if spec["stopcube_mode"].startswith("oracle_"):
            ident.update(query_interval=spec.get("query_interval", 4), query_record=spec.get("query_record", 10))
        if spec["stopcube_mode"] == "oracle_calibrate":
            ident.update(calibration_plan_sha256=hashlib.sha256(Path(spec["calibration_plan"]).read_bytes()).hexdigest(),
                         calibration_kind=spec["calibration_kind"], model_id=spec.get("model_id", spec["model_kind"]))
    return M.digest(ident)


def ablate_of(spec):
    value = spec.get("ablate")
    return "none" if value is None else str(value).strip() or "none"


def select_chunks(manifest, episodes_arg=None, limit=None, single=False):
    """返回 [(chunk_id, [episode, ...])]；给 --episodes 时只跑这些局，每局单独一块；single=True（swap_map）时
    清单分块也拆成每局一块（按清单分块顺序）。"""
    by_key = {episode["episode_key"]: episode for episode in manifest["episodes"]}
    if episodes_arg:
        keys = [item.strip() for item in episodes_arg.split(",") if item.strip()]
        unknown = [key for key in keys if key not in by_key]
        if unknown:
            raise SystemExit(f"--episodes 不在清单里: {unknown}")
        if len(set(keys)) != len(keys):
            raise SystemExit("--episodes 有重复局")
        chunks = [(M.digest([key]), [by_key[key]]) for key in keys]
    elif single:
        chunks = [(M.digest([key]), [by_key[key]]) for chunk in manifest["chunks"] for key in chunk["episode_keys"]]
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


def build_server_command(spec, nonce, port, episodes=None):
    """resident_worker.server_command 原形 → 换模块、插 --attn-out、motion 关闭去掉 --motion-online-gpu。
    swap_map 给出时 episodes 必须恰为 1 局，按映射加 --swap-donor。"""
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
    if spec.get("stopcube_mode", "").startswith("oracle_"):
        extra += ["--query-record", str(spec.get("query_record", 10)), "--oracle-mode", spec["stopcube_mode"]]
    if spec.get("save_top_frames") is not None:
        extra += ["--save-top-frames", str(int(spec["save_top_frames"]))]
    if spec.get("save_prefix"):
        extra += ["--save-prefix"]
    if ablate_of(spec) != "none":
        extra += ["--ablate", ablate_of(spec)]
    if spec.get("record_motion"):
        extra += ["--record-motion", str(spec["record_motion"])]
    if spec.get("swap_map"):
        if not episodes or len(episodes) != 1:
            raise ValueError("swap_map 要求每块恰 1 局")
        key = episodes[0]["episode_key"]
        donor = parse_swap_map(spec["swap_map"]).get(_episode_key(key))
        if donor is None:
            raise ValueError(f"swap_map 没有局 {key}")
        extra += ["--swap-donor", donor]
    index = command.index("policy:checkpoint")
    return command[:index] + extra + command[index:]


def build_environment(spec):
    environment = RW.worker_environment(spec)
    environment.update(CUDA_VISIBLE_DEVICES=spec["cuda"], XLA_FLAGS=XLA_FLAGS,
                       XLA_PYTHON_CLIENT_MEM_FRACTION=str(spec["xla_mem_fraction"]))
    return environment


def build_client_command(spec, grant_path, output, nonce, port, drain_file):
    """默认命令逐字保留；仅显式 StopCube 模式替换 interpret 客户端。"""
    command = RW.client_command(spec, grant_path, output, nonce, port, drain_file)
    if spec.get("stopcube_mode") is not None:
        command[1] = str(Path(spec["repo"]) / "scripts/interpret/stopcube_runner.py")
        command += ["--stopcube-mode", spec["stopcube_mode"], "--stopcube-config", str(spec["stopcube_config"])]
        if spec["stopcube_mode"].startswith("oracle_"):
            command += ["--query-interval", str(spec.get("query_interval", 4)),
                        "--query-record", str(spec.get("query_record", 10))]
        if spec["stopcube_mode"] == "oracle_calibrate":
            command += ["--calibration-plan", spec["calibration_plan"], "--calibration-kind", spec["calibration_kind"],
                        "--calibration-model-id", spec.get("model_id", spec["model_kind"])]
    return command


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
    server_cmd = build_server_command(spec, nonce, port, episodes)
    client_cmd = build_client_command(spec, grant_path, output, nonce, port, drain_file)
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


def run_oracle_chunks(spec, chunks, popen=subprocess.Popen):
    """Oracle 清单共用一个服务；每个客户端至多5局、独立授权与输出。"""
    if not chunks:
        return
    nonce = uuid.uuid4().hex
    root = Path(spec["run_root"]) / "attempts" / f"oracle-service-{nonce[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    port = pick_port(int(spec.get("port", 0)))
    environment = build_environment(spec)
    server_cmd = build_server_command(spec, nonce, port)
    write_json(root / "server-command.json", {"server": server_cmd, "nonce": nonce, "port": port})
    server = client = None
    try:
        with (root / "server.log").open("ab", buffering=0) as log:
            server = popen(server_cmd, cwd=spec["repo"], env=environment, stdout=log,
                           stderr=subprocess.STDOUT, start_new_session=True)
        wait_ready(server, port, root / "server.log", float(spec.get("startup_timeout_seconds", 600)))
        batches = []
        for chunk_id, episodes in chunks:
            for start in range(0, len(episodes), 5):
                group = episodes[start:start + 5]
                batches.append((chunk_id if len(episodes) <= 5 else M.digest([e["episode_key"] for e in group]), group))
        for index, (chunk_id, episodes) in enumerate(batches):
            attempt = root / f"{index:03d}-{chunk_id[:12]}"
            attempt.mkdir(exist_ok=False)
            grant, output, drain = attempt / "grant.json", attempt / "client", attempt / "drain"
            write_json(grant, {"chunk_id": chunk_id, "run_identity_sha256": spec["run_identity_sha256"], "episodes": episodes})
            command = build_client_command(spec, grant, output, nonce, port, drain)
            write_json(attempt / "commands.json", {"server": server_cmd, "client": command, "port": port,
                       "nonce": nonce, "chunk_id": chunk_id, "episode_keys": [e["episode_key"] for e in episodes]})
            print(f"CHUNK_BEGIN index={index} chunk_id={chunk_id} port={port} nonce={nonce} attempt={attempt}", flush=True)
            code = -1
            try:
                if server.poll() is not None:
                    raise RuntimeError("常驻 Oracle 服务已退出")
                with (attempt / "sim.log").open("ab", buffering=0) as log:
                    client = popen(command, cwd=spec["repo"], env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
                while client.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError(f"策略服务在仿真期间退出\n{tail(root / 'server.log')}")
                    threading.Event().wait(0.5)
                code = client.returncode
            except Exception as exc:
                print(f"CHUNK_ERROR index={index} error={type(exc).__name__}: {exc}", flush=True)
            finally:
                RW.stop_process_tree(client)
                client = None
            print(f"CHUNK_END index={index} client_exit={code}", flush=True)
            yield attempt, output, code, episodes
            if code != 0:
                return
    finally:
        RW.stop_process_tree(client)
        RW.stop_process_tree(server)


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
        mode = spec.get("stopcube_mode")
        oracle_mode = mode in ("oracle_query", "oracle_calibrate")
        query_npz = len(list((attn_out / nonce).glob("query_*.npz"))) if nonce else 0
        receipt_path = output / "traces" / str(nonce) / "meta" / "hold_receipt.json"
        receipt = read_json(receipt_path) if oracle_mode and receipt_path.exists() else None
        if nonce and (attn_out / nonce / "episode.json").exists():
            info = read_json(attn_out / nonce / "episode.json")
            info.update({k: (terminal or started)[k] for k in ("task_name", "episode_id", "env_seed", "difficulty",
                                                               "episode_key", "chunk_id", "run_identity_sha256")})
            info.update(outcome=terminal["outcome"] if terminal else None, attempt=str(attempt))
            motion_record = rename_motion_record(spec, nonce, episode)
            if motion_record is not None:
                info["motion_record"] = motion_record
            if spec.get("swap_map"):
                info["swap_donor"] = parse_swap_map(spec["swap_map"]).get(_episode_key(key))
            if spec.get("stopcube_mode") is not None:
                receipt_path = output / "traces" / nonce / "meta" / "hold_receipt.json"
                receipt = read_json(receipt_path)
                info.update(trajectory_kind=spec["stopcube_mode"], hold_receipt=str(receipt_path.resolve()),
                            trace_path=receipt["trace_path"], video_path=receipt["video_path"])
            write_json(attn_out / nonce / "episode.json", info)
        # terminal 的 rpc_seq = 1 + batch_seq + replan_seq，官方循环里每次 infer 前恰好一次 add_buffer
        infers = (int(terminal["rpc_seq"]) - 1) // 2 if terminal else None
        history_batches = query_points = execution_infers = None
        cadence_ok = None
        if oracle_mode:
            history_batches = (receipt or {}).get("history_batches", 0)
            query_points = (receipt or {}).get("query_points", 0)
            execution_infers = (receipt or {}).get("execution_infers", 0)
            infers = execution_infers
            cadence_ok = bool(terminal and receipt and int(terminal["rpc_seq"]) == 1 + history_batches + query_points + execution_infers)
            if mode == "oracle_query":
                table = read_json(spec["stopcube_config"])["episodes"][key]
                expected_steps = list(range(table["first_env_step"], table["move_interval"] * table["target_pass"] + 1,
                                            spec.get("query_interval", 4)))
                cadence_ok = bool(cadence_ok and receipt["query_steps"] == expected_steps
                                  and query_npz == query_points == len(expected_steps) and execution_infers == 0)
            else:
                cadence_ok = bool(cadence_ok and query_npz == query_points == 0 and execution_infers > 0
                                  and receipt.get("actual_release_step") == receipt.get("release_step"))
        success = int(bool(terminal) and terminal["outcome"] == "success")
        row = {"task": episode["task_name"], "ep": episode["episode_id"], "success": success, "npz": npz,
               "infers": infers, "terminal": terminal is not None, "nonce": nonce}
        if oracle_mode:
            row.update(mode=mode, history_batches=history_batches, query_points=query_points,
                       execution_infers=execution_infers, query_npz=query_npz, cadence_ok=cadence_ok,
                       receipt_path=str(receipt_path.resolve()))
        results.append(row)
        print(f"EPISODE_RESULT task={episode['task_name']} ep={episode['episode_id']} success={success} npz={npz}",
              flush=True)
    return results


def rename_motion_record(spec, nonce, episode):
    """record_motion 目录下 <nonce>.npz → <task>_ep<id>.npz（同名已存在则覆盖并打印提示）；返回新路径或 None。"""
    if not spec.get("record_motion") or not nonce:
        return None
    directory = Path(spec["record_motion"])
    source = directory / f"{nonce}.npz"
    target = directory / f"{episode['task_name']}_ep{int(episode['episode_id'])}.npz"
    if not source.exists():
        return str(target) if target.exists() else None
    if target.exists():
        print(f"RECORD_MOTION_OVERWRITE {target}", flush=True)
    os.replace(source, target)
    return str(target)


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
    if spec.get("record_motion"):
        Path(spec["record_motion"]).mkdir(parents=True, exist_ok=True)
    chunks = select_chunks(manifest, args.episodes, args.chunks, single=bool(spec.get("swap_map")))
    if spec.get("stopcube_mode") is not None and any(
            e["task_name"] != "StopCube" for _, episodes in chunks for e in episodes):
        raise SystemExit("stopcube_mode 只能选择 StopCube 局")
    skipped = []
    if spec.get("stopcube_mode") == "oracle_calibrate":
        from scripts.interpret.stopcube_runner import load_calibration
        selected = load_calibration(spec["calibration_plan"], spec.get("model_id", spec["model_kind"]))
        filtered = []
        for chunk_id, episodes in chunks:
            kept = []
            for episode in episodes:
                entry = selected.get(episode["episode_key"], {}).get(spec["calibration_kind"])
                if entry is None:
                    raise SystemExit(f"校准文件缺选点 {episode['episode_key']} {spec['calibration_kind']}")
                if entry.get("status") == "skipped":
                    skipped.append(dict(episode_key=episode["episode_key"], kind=spec["calibration_kind"], reason=entry.get("reason")))
                    print(f"CALIBRATION_SKIP episode={episode['episode_key']} kind={spec['calibration_kind']} reason={entry.get('reason')}", flush=True)
                else:
                    kept.append(episode)
            if kept:
                filtered.append((chunk_id if len(kept) == len(episodes) else M.digest([e["episode_key"] for e in kept]), kept))
        chunks = filtered
    if spec.get("swap_map"):
        missing = [e["episode_key"] for _, eps in chunks for e in eps if _episode_key(e["episode_key"]) not in spec["swap_map"]]
        if missing:
            raise SystemExit(f"swap_map 缺这些局的供体: {missing}")
    print(f"RUN_LOCAL_BEGIN model={spec['model_kind']} run_identity={spec['run_identity_sha256']} "
          f"manifest_seal={manifest['sha256']} chunks={len(chunks)} cuda={spec['cuda']} ablate={ablate_of(spec)}",
          flush=True)

    def stop(signum, _frame):
        raise SystemExit(128 + signum)   # 走 finally 收掉服务端与客户端进程组

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    results, codes = [], []
    oracle_mode = spec.get("stopcube_mode") in ("oracle_query", "oracle_calibrate")
    if oracle_mode:
        attempts = run_oracle_chunks(spec, chunks, popen=popen)
    else:
        attempts = ((*run_chunk(spec, index, chunk_id, episodes, popen=popen), episodes)
                    for index, (chunk_id, episodes) in enumerate(chunks))
    try:
        for attempt, output, code, episodes in attempts:
            codes.append(code)
            results.extend(collect(spec, attempt, output, episodes))
            if code == -1:
                break   # 服务端或驱动故障：不再起后续 chunk
    finally:
        attempts.close()   # 信号或收集失败同样触发常驻服务的精确进程组收尾。
    ok = (all(code == 0 for code in codes) and all(r["terminal"] for r in results)
          and all(r["cadence_ok"] if "cadence_ok" in r else r["npz"] > 0 and r["npz"] == r["infers"] for r in results))
    total_npz, success = sum(r["npz"] for r in results), sum(r["success"] for r in results)
    write_json(Path(spec["run_root"]) / f"run_local-{uuid.uuid4().hex[:8]}.json",
               {"run_identity_sha256": spec["run_identity_sha256"], "client_exit_codes": codes, "results": results,
                **({"skipped": skipped} if spec.get("stopcube_mode") == "oracle_calibrate" else {})})
    print(f"RUN_LOCAL={'PASS' if ok else 'FAIL'} model={spec['model_kind']} episodes={len(results)} "
          f"success={success} npz={total_npz}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
