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
motion_zero / motion_swap:<供体目录> / motion_swap_exact:<供体目录> / motion_shift）。
StopCube 可选键：stopcube_mode（normal、hold_extended、oracle_query、oracle_calibrate）、stopcube_config（固定 JSON 表路径）；
同时提供时改用 interpret 记录入口，模式与配置文件字节 sha256 并入身份，其他任务不受影响。
Oracle 另用 query_interval=4（节奏诊断可16）、query_record=10；校准另需 calibration_plan、calibration_kind，
多模型选点文件由 model_id（默认 model_kind）选择。校准选点文件内容也并入身份。
query_batch_size 可显式指定1、2、5、10，缺省实际批量10；该批量进入身份，不做自动回退。
rollout_record（bool，只允许与 stopcube_mode=normal、bin_trace=true 或 subgoal_trace=true 同用）：开启时服务端加 --rollout-record，
走原采样路径，每次推理写 rollout_<seq>.npz；开启时并入身份，逐局 npz 计数改数 rollout_*.npz；缺键时身份与命令不变。
bin_trace（bool，与 stopcube_mode 互斥，清单只能含 ButtonUnmaskSwap、VideoUnmaskSwap）：开启时改用 bin_trace_runner 客户端写逐步轨迹，
并入身份；缺键或为假时身份与命令不变。
subgoal_trace（bool，与 stopcube_mode、bin_trace 互斥，清单只能含 PatternLock、RouteStick）：开启时改用 subgoal_trace_runner 客户端
写逐步轨迹，并入身份；缺键或为假时身份与命令不变。
movecube_force（bool，与 stopcube_mode、bin_trace、subgoal_trace、rollout_record 互斥，清单只能含 MoveCube）：配 force_table（动作表路径）
与 force_k（0、16、32、48、64、80）。走官方 scripts.training.serve_policy，不录注意力、不创建 attn_out（该键可省略），一个服务配
客户端每批至多 5 局；动作表内容与 force_k 并入身份。单局判定：有终局且 terminal.rpc_seq 推出的 infers>0。缺键或为假时身份与命令不变。
S8 motion 内容替换的可选键：
  record_motion   目录；服务端加 --record-motion，每局写 <目录>/<nonce>.npz，本脚本收尾按 terminal 局身份改名为
                  <目录>/<task>_ep<id>.npz（与 motion_swap 供体文件同名规则）；局的 episode.json 记 motion_record。
  swap_map        motion_swap / motion_swap_exact 的供体映射，{"ButtonUnmaskSwap:4": "ButtonUnmaskSwap:8", ...} 或字符串
                  "ButtonUnmaskSwap:4=ButtonUnmaskSwap:8,..."；ablate 含这两者之一时必填、否则不得给出（motion_shift 不需要）。
                  给出时每局单独成块（chunk_id = digest([episode_key])，与 --episodes 同法），服务端按块加
                  --swap-donor <供体局>：服务端在 reset 时只知道 nonce，单局成块后「本服务进程服务的局」即确定。
                  选中的每局都必须在 swap_map 里，否则启动前报错。
plfull 口径（PatternLock 全因子交换，计划 1109 第二部分〇′）：
  profile         "official"（缺省，旧行为逐字不变）或 "plfull"；plfull 时 plfull_metadata_dir 必填（自建 metadata 目录，
                  含 record_dataset_<task>_metadata.json），model_seed 只能 42、1、2；客户端加
                  --profile plfull --plfull-metadata-dir <目录>，局号允许 0–79；身份并入 profile 与 metadata 文件 sha256。
                  plfull 下 swap_map 不再拆单局块：清单分块原样（每块 1–5 局），服务端加 --swap-map <整份映射>，
                  客户端 reset 的 eval_rpc 带 episode_key，服务端 reset 时按它确定本局供体（--swap-donor 不再使用）。
Wan 时间池化模型（model_kind=wan_fixedpool1024）：policy_config 与 motion 模型同为 mme_vla_suite_orig16_b128_60k，
  checkpoint 目录的父目录须是 timepool 根快照（history_config.txt、motion_provenance.json 等）；sidecar 用 stage2_python
  （mme-vla-wan-stage2 venv）起 scripts/dataset/wan_timepool/timepool_sidecar.py，保留 --motion-online-gpu；
  不接受 target_contract（MotionEncoderClient 只允许 stage2 encoder 带评估目标合同），spec 里写 null / "" / "none" 均视为不传；
  sidecar 从 MMEVLA_V1_STORE 找 cache/hf（Wan2.1 VAE），可由 spec 键 v1_store 给出，否则须已在环境里。
save_attn（bool，缺省按 ablate 判定）：ablate 含 motion_swap: / motion_swap_exact: / motion_shift 时一律不写 infer_<seq>.npz，
无需配置，显式给 true 报错；其他 ablate 默认写，显式 false 时服务端加 --no-save-attn 不写。不写时与 save_prefix、rollout_record、
Oracle 模式互斥。
run_identity_sha256 由本脚本自算（见 run_identity；ablate 非 none 时并入身份，none 时身份与 S7 之前相同；
swap_map 给出时也并入身份；record_motion 只记录不改行为，不并入；save_attn=false 时并入，缺键或 true 时身份不变）。

逐局打印 EPISODE_RESULT task=<t> ep=<i> success=<0|1> npz=<n>，末行
RUN_LOCAL=PASS|FAIL model=<model_kind> episodes=<n> success=<k> npz=<n>。
不写注意力时判定改为：终局存在、episode.json 的 infers > 0，且 infer_*.npz 数为 0。
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
MODEL_KINDS = ("history", "stage2_pool4_concat", "wan_fixedpool1024")
MOTION_MODEL_KIND = "stage2_pool4_concat"
# Wan 时间池化 1024（与 mme_vla_suite.shared.wan_timepool.TIMEPOOL_ENCODER_KIND 同值，测试盯住）
WAN_MODEL_KIND = "wan_fixedpool1024"
# 带 motion token 的模型：保留 --motion-online-gpu，可做 motion 内容替换与录制
MOTION_KINDS = (MOTION_MODEL_KIND, WAN_MODEL_KIND)
# testx 由 resident_worker.client_command 原样处理（先导轮 pl_swap_prep 用），此处只放行，不改其行为
PROFILES = ("official", "testx", "plfull")
# 与 examples/robomme/eval.py::PLFULL_SEEDS 同值（测试盯住）
PLFULL_SEEDS = (42, 1, 2)
XLA_FLAGS = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
REQUIRED_KEYS = ("repo", "policy_python", "sim_python", "stage2_python", "checkpoint", "policy_config", "model_kind",
                 "cuda", "attn_out", "run_root", "manifest", "xla_mem_fraction")
# 与 bin_trace_runner.TASKS 同值（测试盯住）；驱动端不导入客户端模块，它依赖仿真侧环境。
BIN_TRACE_TASKS = ("ButtonUnmaskSwap", "VideoUnmaskSwap")
# 与 subgoal_trace_runner.TASKS 同值（测试盯住），不导入的理由同上。
SUBGOAL_TRACE_TASKS = ("PatternLock", "RouteStick")
# 与 movecube_force_runner.FORCE_KS 同值（测试盯住），不导入的理由同上。
FORCE_KS = (0, 16, 32, 48, 64, 80)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_spec(path):
    spec = read_json(path)
    missing = [key for key in REQUIRED_KEYS if key not in spec and not (key == "attn_out" and spec.get("movecube_force") is True)]
    if missing:
        raise SystemExit(f"spec 缺键: {missing}")
    if spec["model_kind"] not in MODEL_KINDS:
        raise SystemExit(f"model_kind 只能是 {MODEL_KINDS}，得到 {spec['model_kind']!r}")
    spec.setdefault("model_seed", 42)
    spec.setdefault("purpose", "validation")
    spec["cuda"] = str(spec["cuda"])
    profile = spec.get("profile", "official")
    if profile not in PROFILES:
        raise SystemExit(f"spec: profile 只能是 {PROFILES}，得到 {profile!r}")
    if profile == "plfull":
        if not spec.get("plfull_metadata_dir") or not Path(spec["plfull_metadata_dir"]).is_dir():
            raise SystemExit(f"spec: plfull 需要 plfull_metadata_dir 指向现存目录，得到 {spec.get('plfull_metadata_dir')!r}")
        spec["plfull_metadata_dir"] = str(Path(spec["plfull_metadata_dir"]).resolve())
        if type(spec["model_seed"]) is not int or spec["model_seed"] not in PLFULL_SEEDS:
            raise SystemExit(f"spec: plfull 的 model_seed 只能是 {PLFULL_SEEDS}")
        if spec.get("stopcube_mode") is not None or spec.get("movecube_force"):
            raise SystemExit("spec: plfull 不接受 stopcube_mode、movecube_force")
    elif spec.get("plfull_metadata_dir"):
        raise SystemExit("spec: plfull_metadata_dir 只用于 profile=plfull")
    if spec["model_kind"] == WAN_MODEL_KIND:
        if spec.get("target_contract") not in (None, "", "none"):
            raise SystemExit("spec: Wan 时间池化模型不接受 target_contract（只适用于 stage2 encoder），写 null 或 \"none\"")
        spec.pop("target_contract", None)
    has_swap = any(part.strip().startswith(("motion_swap:", "motion_swap_exact:")) for part in ablate_of(spec).split("+"))
    if spec.get("swap_map") is not None:
        spec["swap_map"] = parse_swap_map(spec["swap_map"])
    if has_swap != bool(spec.get("swap_map")):
        raise SystemExit("spec: ablate 含 motion_swap / motion_swap_exact 时必须给非空 swap_map，反之 swap_map 只能配这两者")
    uses_motion = has_swap or spec.get("record_motion") or any(
        part.strip() in ("motion_zero", "motion_shift") for part in ablate_of(spec).split("+"))
    if uses_motion and spec["model_kind"] not in MOTION_KINDS:
        raise SystemExit(f"spec: motion_zero / motion_swap / motion_swap_exact / motion_shift / record_motion 只适用于 {MOTION_KINDS}")
    if spec.get("stopcube_mode") is not None:
        if spec["stopcube_mode"] not in ("normal", "hold_extended", "oracle_query", "oracle_calibrate") or not spec.get("stopcube_config"):
            raise SystemExit("spec: stopcube_mode 非法或未提供 stopcube_config")
        spec["stopcube_config"] = str(Path(spec["stopcube_config"]).resolve())
        if spec["stopcube_mode"].startswith("oracle_"):
            spec.setdefault("query_interval", 4)
            spec.setdefault("query_record", 10)
            if spec["query_interval"] not in (4, 16) or spec["query_record"] != 10 or ablate_of(spec) != "none":
                raise SystemExit("Oracle 固定10预测、间隔4或诊断16，且禁止 ablate")
            if spec.get("query_batch_size") is not None and (
                    type(spec["query_batch_size"]) is not int or spec["query_batch_size"] not in (1, 2, 5, 10)):
                raise SystemExit("query_batch_size 只能显式选1、2、5、10，不做自动回退")
        if spec["stopcube_mode"] == "oracle_calibrate":
            from scripts.interpret.stopcube_runner import CALIBRATION_KINDS
            if not spec.get("calibration_plan") or spec.get("calibration_kind") not in CALIBRATION_KINDS:
                raise SystemExit("校准须提供 calibration_plan 和固定的 calibration_kind")
            spec["calibration_plan"] = str(Path(spec["calibration_plan"]).resolve())
    elif spec.get("stopcube_config") is not None:
        raise SystemExit("spec: stopcube_config 需要 stopcube_mode")
    if "bin_trace" in spec:
        if type(spec["bin_trace"]) is not bool:
            raise SystemExit("spec: bin_trace 必须为 bool")
        if spec["bin_trace"]:
            if spec.get("stopcube_mode") is not None:
                raise SystemExit("spec: bin_trace 与 stopcube_mode 互斥")
            others = sorted({e["task_name"] for e in read_json(spec["manifest"])["episodes"]} - set(BIN_TRACE_TASKS))
            if others:
                raise SystemExit(f"spec: bin_trace 的清单只能含 {BIN_TRACE_TASKS}，得到 {others}")
    if "subgoal_trace" in spec:
        if type(spec["subgoal_trace"]) is not bool:
            raise SystemExit("spec: subgoal_trace 必须为 bool")
        if spec["subgoal_trace"]:
            if spec.get("stopcube_mode") is not None or spec.get("bin_trace"):
                raise SystemExit("spec: subgoal_trace 与 stopcube_mode、bin_trace 互斥")
            others = sorted({e["task_name"] for e in read_json(spec["manifest"])["episodes"]} - set(SUBGOAL_TRACE_TASKS))
            if others:
                raise SystemExit(f"spec: subgoal_trace 的清单只能含 {SUBGOAL_TRACE_TASKS}，得到 {others}")
    if "movecube_force" in spec:
        if type(spec["movecube_force"]) is not bool:
            raise SystemExit("spec: movecube_force 必须为 bool")
        if spec["movecube_force"]:
            load_force_spec(spec)
    if "rollout_record" in spec:
        if type(spec["rollout_record"]) is not bool:
            raise SystemExit("spec: rollout_record 必须为 bool")
        if spec["rollout_record"] and spec.get("stopcube_mode") != "normal" and not spec.get("bin_trace") and not spec.get("subgoal_trace"):
            raise SystemExit("spec: rollout_record 只允许与 stopcube_mode=normal、bin_trace=true 或 subgoal_trace=true 同用")
    if "save_attn" in spec:
        if type(spec["save_attn"]) is not bool:
            raise SystemExit("spec: save_attn 必须为 bool")
        if spec["save_attn"] and replaces_motion(spec):
            raise SystemExit("spec: motion 替换实验不存注意力，不能给 save_attn=true")
    if not saves_attn(spec):
        if spec.get("save_prefix"):
            raise SystemExit("spec: save_prefix 需要存注意力，与不存注意力（save_attn=false 或 motion 替换实验）互斥")
        if spec.get("rollout_record") or str(spec.get("stopcube_mode")).startswith("oracle_"):
            raise SystemExit("spec: 不存注意力只作用于注意力记录路径，与 rollout_record、Oracle 模式互斥")
    return spec


def load_force_spec(spec):
    """movecube_force 模式的键检查：与其他记录模式互斥，也不接受官方服务不支持的服务端选项。"""
    if (spec.get("stopcube_mode") is not None or spec.get("bin_trace") or spec.get("subgoal_trace") or spec.get("rollout_record")):
        raise SystemExit("spec: movecube_force 与 stopcube_mode、bin_trace、subgoal_trace、rollout_record 互斥")
    if (ablate_of(spec) != "none" or spec.get("record_motion") or spec.get("save_prefix") or spec.get("save_attn")
            or spec.get("save_top_frames") is not None):
        raise SystemExit("spec: movecube_force 走官方服务，不接受 ablate、record_motion、save_prefix、save_attn、save_top_frames")
    if not spec.get("force_table") or type(spec.get("force_k")) is not int or spec["force_k"] not in FORCE_KS:
        raise SystemExit(f"spec: movecube_force 要求 force_table，且 force_k 只能取 {FORCE_KS}")
    spec["force_table"] = str(Path(spec["force_table"]).resolve())
    others = sorted({e["task_name"] for e in read_json(spec["manifest"])["episodes"]} - {"MoveCube"})
    if others:
        raise SystemExit(f"spec: movecube_force 的清单只能含 MoveCube，得到 {others}")


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
            ident.update(query_interval=spec.get("query_interval", 4), query_record=spec.get("query_record", 10),
                         query_batch_size=spec.get("query_batch_size") or spec.get("query_record", 10))
        if spec["stopcube_mode"] == "oracle_calibrate":
            ident.update(calibration_plan_sha256=hashlib.sha256(Path(spec["calibration_plan"]).read_bytes()).hexdigest(),
                         calibration_kind=spec["calibration_kind"], model_id=spec.get("model_id", spec["model_kind"]))
    if spec.get("rollout_record"):
        ident["rollout_record"] = True
    if spec.get("bin_trace"):
        ident["bin_trace"] = True
    if spec.get("subgoal_trace"):
        ident["subgoal_trace"] = True
    if spec.get("save_attn") is False:
        ident["save_attn"] = False
    if spec.get("movecube_force"):
        ident.update(movecube_force=True, force_k=spec["force_k"],
                     force_table_sha256=hashlib.sha256(Path(spec["force_table"]).read_bytes()).hexdigest())
    if spec.get("profile", "official") == "plfull":
        # 只在 plfull 时并入；官方口径身份与改动前相同。
        directory = Path(spec["plfull_metadata_dir"])
        ident.update(profile="plfull", plfull_metadata_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.json"))})
    return M.digest(ident)


def ablate_of(spec):
    value = spec.get("ablate")
    return "none" if value is None else str(value).strip() or "none"


def replaces_motion(spec):
    """与 serve_policy_attn.replaces_motion 同判据（测试盯住）；驱动端不导入服务端模块。"""
    return any(part.strip() == "motion_shift" or part.strip().startswith(("motion_swap:", "motion_swap_exact:"))
               for part in ablate_of(spec).split("+"))


def is_plfull(spec):
    return spec.get("profile", "official") == "plfull"


def saves_attn(spec):
    return spec.get("save_attn") is not False and not replaces_motion(spec) and not spec.get("movecube_force")


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
    swap_map 给出时：旧口径 episodes 必须恰为 1 局，按映射加 --swap-donor；plfull 加 --swap-map <整份映射>。
    movecube_force 不换模块、不插 --attn-out，只按模型去掉 --motion-online-gpu。"""
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
    if not spec.get("movecube_force"):
        command[command.index(SERVER_MODULE)] = ATTN_MODULE
    if spec["model_kind"] not in MOTION_KINDS:
        index = command.index("--motion-online-gpu")
        del command[index:index + 2]
    if spec.get("movecube_force"):
        return command
    extra = ["--attn-out", str(spec["attn_out"])]
    if spec.get("stopcube_mode", "").startswith("oracle_"):
        extra += ["--query-record", str(spec.get("query_record", 10)), "--oracle-mode", spec["stopcube_mode"]]
        if spec.get("query_batch_size") is not None:
            extra += ["--query-batch-size", str(spec["query_batch_size"])]
    if spec.get("rollout_record"):
        extra += ["--rollout-record"]
    if spec.get("save_top_frames") is not None:
        extra += ["--save-top-frames", str(int(spec["save_top_frames"]))]
    if spec.get("save_prefix"):
        extra += ["--save-prefix"]
    if spec.get("save_attn") is False:
        extra += ["--no-save-attn"]
    if ablate_of(spec) != "none":
        extra += ["--ablate", ablate_of(spec)]
    if spec.get("record_motion"):
        extra += ["--record-motion", str(spec["record_motion"])]
    if spec.get("swap_map") and is_plfull(spec):
        # 多局块：整份映射交给服务端，按客户端 reset 带来的 episode_key 逐局取供体。
        mapping = parse_swap_map(spec["swap_map"])
        missing = [e["episode_key"] for e in episodes or () if _episode_key(e["episode_key"]) not in mapping]
        if not episodes or missing:
            raise ValueError(f"swap_map 没有局 {missing}" if missing else "plfull swap_map 需要本块局清单")
        extra += ["--swap-map", ",".join(f"{k}={v}" for k, v in sorted(mapping.items()))]
    elif spec.get("swap_map"):
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
    if spec["model_kind"] == WAN_MODEL_KIND:
        # timepool sidecar 从 MMEVLA_V1_STORE/cache/hf 离线读 Wan VAE；运行工作树下没有 v1-store 时必须显式给出。
        if spec.get("v1_store"):
            environment["MMEVLA_V1_STORE"] = str(spec["v1_store"])
        if not environment.get("MMEVLA_V1_STORE"):
            raise ValueError("Wan 模型需要 MMEVLA_V1_STORE（spec 键 v1_store 或运行环境变量）")
    return environment


def build_client_command(spec, grant_path, output, nonce, port, drain_file):
    """默认命令逐字保留；仅显式 StopCube 模式、bin_trace、subgoal_trace 或 movecube_force 替换 interpret 客户端。"""
    command = RW.client_command(spec, grant_path, output, nonce, port, drain_file)
    if is_plfull(spec):
        command += ["--profile", "plfull", "--plfull-metadata-dir", str(spec["plfull_metadata_dir"])]
    if spec.get("bin_trace"):
        command[1] = str(Path(spec["repo"]) / "scripts/interpret/bin_trace_runner.py")
    if spec.get("subgoal_trace"):
        command[1] = str(Path(spec["repo"]) / "scripts/interpret/subgoal_trace_runner.py")
    if spec.get("movecube_force"):
        command[1] = str(Path(spec["repo"]) / "scripts/interpret/movecube_force_runner.py")
        command += ["--force-table", str(spec["force_table"]), "--force-k", str(spec["force_k"])]
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
    """Oracle 与 movecube_force 清单共用一个服务；每个客户端至多5局、独立授权与输出。"""
    if not chunks:
        return
    nonce = uuid.uuid4().hex
    prefix = "force-service" if spec.get("movecube_force") else "oracle-service"
    root = Path(spec["run_root"]) / "attempts" / f"{prefix}-{nonce[:8]}"
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


def collect_force(spec, output, episodes):
    """movecube_force 收尾：不读 attn_out 与 episode.json（官方服务不写），推理数由 terminal.rpc_seq 推出。"""
    accepted = Path(spec["run_root"]) / "accepted"
    accepted.mkdir(parents=True, exist_ok=True)
    records = {}
    for kind in ("started", "terminal"):
        for path in sorted(output.glob(f"*.{kind}.json")):
            record = read_json(path)
            records.setdefault(record["episode_key"], {})[kind] = record
            if kind == "terminal":
                shutil.copy2(path, accepted / f"{record['episode_key']}.{record['episode_nonce'][:8]}.terminal.json")
    results = []
    for episode in episodes:
        found = records.get(episode["episode_key"], {})
        terminal = found.get("terminal")
        nonce = (terminal or found.get("started") or {}).get("episode_nonce")
        receipt_path = output / "traces" / str(nonce) / "meta" / "force_receipt.json"
        receipt = read_json(receipt_path) if receipt_path.exists() else {}
        success = int(bool(terminal) and terminal["outcome"] == "success")
        results.append({"task": episode["task_name"], "ep": episode["episode_id"], "success": success, "npz": 0,
                        "infers": (int(terminal["rpc_seq"]) - 1) // 2 if terminal else None,
                        "terminal": terminal is not None, "nonce": nonce, "attn_saved": False, "force_k": spec["force_k"],
                        "grasp_success": receipt.get("grasp_success"), "env_status": receipt.get("env_status"),
                        "reached_release": receipt.get("reached_release"), "receipt_path": str(receipt_path.resolve())})
        print(f"EPISODE_RESULT task={episode['task_name']} ep={episode['episode_id']} success={success} npz=0", flush=True)
    return results


def collect(spec, attempt, output, episodes):
    """把 terminal 汇到 accepted/，并把局身份并入 attn_out/<nonce>/episode.json。返回逐局结果。"""
    if spec.get("movecube_force"):
        return collect_force(spec, output, episodes)
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
        # rollout 记录模式不写 infer_*.npz；RUN_LOCAL 判定与逐局结果改数 rollout_*.npz
        npz = len(list((attn_out / nonce).glob("rollout_*.npz" if spec.get("rollout_record") else "infer_*.npz"))) if nonce else 0
        mode = spec.get("stopcube_mode")
        oracle_mode = mode in ("oracle_query", "oracle_calibrate")
        query_npz = len(list((attn_out / nonce).glob("query_*.npz"))) if nonce else 0
        receipt_path = output / "traces" / str(nonce) / "meta" / "hold_receipt.json"
        receipt = read_json(receipt_path) if oracle_mode and receipt_path.exists() else None
        episode_infers = 0
        if nonce and (attn_out / nonce / "episode.json").exists():
            info = read_json(attn_out / nonce / "episode.json")
            episode_infers = int(info.get("infers", 0))
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
        if not saves_attn(spec):
            row.update(attn_saved=False, episode_infers=episode_infers)
        if oracle_mode:
            row.update(mode=mode, history_batches=history_batches, query_points=query_points,
                       execution_infers=execution_infers, query_npz=query_npz, cadence_ok=cadence_ok,
                       receipt_path=str(receipt_path.resolve()))
        results.append(row)
        print(f"EPISODE_RESULT task={episode['task_name']} ep={episode['episode_id']} success={success} npz={npz}",
              flush=True)
    return results


def row_ok(row):
    """单局判定：Oracle 看节奏；强推看终局与 rpc_seq 推出的 infers；不存注意力看终局与 episode.json 的 infers；
    其余 npz 数须等于推理数。"""
    if "cadence_ok" in row:
        return row["cadence_ok"]
    if "force_k" in row:
        return row["terminal"] and row["infers"] > 0
    if row.get("attn_saved") is False:
        return row["terminal"] and row["episode_infers"] > 0 and row["npz"] == 0
    return row["npz"] > 0 and row["npz"] == row["infers"]


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
    if not spec.get("movecube_force"):
        Path(spec["attn_out"]).mkdir(parents=True, exist_ok=True)
    if spec.get("record_motion"):
        Path(spec["record_motion"]).mkdir(parents=True, exist_ok=True)
    # 旧口径 swap_map 每局单独成块；plfull 由服务端逐局取供体，清单分块原样。
    chunks = select_chunks(manifest, args.episodes, args.chunks, single=bool(spec.get("swap_map")) and not is_plfull(spec))
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
    if oracle_mode or spec.get("movecube_force"):
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
          and all(row_ok(r) for r in results))
    total_npz, success = sum(r["npz"] for r in results), sum(r["success"] for r in results)
    write_json(Path(spec["run_root"]) / f"run_local-{uuid.uuid4().hex[:8]}.json",
               {"run_identity_sha256": spec["run_identity_sha256"], "client_exit_codes": codes, "results": results,
                **({"skipped": skipped} if spec.get("stopcube_mode") == "oracle_calibrate" else {})})
    print(f"RUN_LOCAL={'PASS' if ok else 'FAIL'} model={spec['model_kind']} episodes={len(results)} "
          f"success={success} npz={total_npz}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
