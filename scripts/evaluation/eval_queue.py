"""本机8×A100通用官方评测队列：按队列文件写死的顺序逐项评 800 局，空出的 GPU 立刻交给下一项。

四种模式（互斥）：
  --dry-run  CPU only：打印全部计划项、ckpt 是否存在、model_kind/配置/快照能否解析，不起任何 worker。
             加 --check-tree 再对每个现存 ckpt 做抽象建模（jax.eval_shape）并与 ckpt 参数树逐路径比对。
  --smoke    每个不同 ckpt 各跑 1 局（队列文件 smoke_episode），每个 ckpt 占 1 张 GPU 并行，输出 SMOKE=PASS|FAIL。
  （缺省）    正式评测：每卡 per_gpu 个常驻 worker；一项的 800 局全部领完后，空闲 worker 被 drain，
             该卡上 worker 全部退出即交给下一项（同一 GPU 同时只服务一项）。可断点续跑。
  --stop     只对 cwd 等于本执行副本根目录的 python 进程发 SIGTERM（调度器、控制器、worker、策略服务、
             仿真客户端与 sidecar），不碰其他目录的进程，也不 kill 任何 tmux 会话。

判定行：EVAL_START / EVAL_DONE / ITEM_BLOCKED / QUEUE_DONE / QUEUE_FATAL / SMOKE=… / PLAN …。
口径与 1107、o16-3seed-1031 相同：官方 profile、1300 步上限、正常失败不重试、基础设施错误每局最多重试 1 次。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
# 主检出 .venv 的 editable 安装指向主检出 src；dry-run 解析配置必须 import 执行副本自己的源码，故放到最前。
for _path in (REPO / "packages/openpi-client/src", REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.evaluation.manifest import build_official_manifest, episode_key, make_manifest  # noqa: E402
from scripts.evaluation.queue_store import QueueStore, publish, read_json, replace_json  # noqa: E402
from scripts.evaluation.runtime_contract import identity_sha256, sha256_file  # noqa: E402

# 产物根：与 motion_client 同一约定，MMEVLA_V1_STORE 优先（只读检查时可从 agent worktree 指到主检出），否则执行副本内 v1-store。
V1 = Path(os.environ.get("MMEVLA_V1_STORE") or REPO / "v1-store")
DEFAULT_QUEUE = REPO / "scripts/evaluation/eval3x3_1110_queue.json"
QUEUE_PY, WORKER_PY = REPO / "scripts/evaluation/official_queue.py", REPO / "scripts/evaluation/resident_worker.py"
YAML_DIR = REPO / "src/mme_vla_suite/models/config/robomme"
KINDS = ("history", "stage2_pool4_concat", "wan_fixedpool1024")
TICK = 5
TOTAL = 800


def say(line):
    print(line, flush=True)


# ── 队列文件与计划 ─────────────────────────────────────────────────────────────

def load_queue(path):
    queue = json.loads(Path(path).read_text(encoding="utf-8"))
    if queue.get("schema") != "eval3x3-queue-v1":
        raise ValueError("队列文件 schema 不符")
    return queue


def item_name(entry):
    return f"{entry['model']}-{entry['step']}-seed{entry['seed']}"


def plan_items(queue):
    """返回按执行顺序的计划项；optional_tail 只在开关打开时追加。"""
    entries = list(queue["items"])
    if queue.get("optional_tail_enabled"):
        entries += list(queue["optional_tail"])
    names = [item_name(e) for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("队列含重复项")
    for e in entries:
        if e["model"] not in queue["models"] or e["seed"] not in (0, 7, 42):
            raise ValueError(f"非法队列项: {e}")
    return [dict(e, name=item_name(e)) for e in entries]


def distinct_ckpts(items):
    seen = []
    for it in items:
        if (it["model"], it["step"]) not in seen:
            seen.append((it["model"], it["step"]))
    return seen


def repo_path(value):
    """队列里的相对路径：v1-store/ 开头的落到 V1，其余相对执行副本根。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return V1.joinpath(*path.parts[1:]) if path.parts[0] == "v1-store" else REPO / path


def train_root(queue, model):
    return repo_path(queue["models"][model]["train_root"])


def ckpt_dir(queue, model, step):
    return train_root(queue, model) / str(step)


def ckpt_present(path):
    return (path / "_CHECKPOINT_METADATA").is_file() and (path / "params").is_dir() and (path / "assets").is_dir()


def out_root(queue):
    return repo_path(queue["out_root"])


def smoke_record_path(queue, model, step):
    return out_root(queue) / "smoke" / f"{model}-{step}" / "SMOKE.json"


def code_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def require_execution_copy():
    """正式/冒烟/停止只能在 git worktree 执行副本里跑（.git 为文件）；主检出 .git 为目录，拒绝。"""
    if not (REPO / ".git").is_file():
        say(f"QUEUE_FATAL reason=not_execution_copy repo={REPO}")
        raise SystemExit(2)


# ── 模型侧检查（dry-run 与起跑前共用，纯 CPU） ───────────────────────────────────

def expected_kind(history_cfg):
    """与 policy_config.create_trained_policy 写进 metadata 的 model_kind 同一判定。"""
    motion = history_cfg.get("motion", None) if history_cfg is not None else None
    if motion is not None and motion.get("enabled", False):
        kind = str(motion.get("encoder_kind", "v10"))
        return kind if kind in ("stage2_pool4_concat", "wan_fixedpool1024") else "history"
    return "history"


def inspect_model(queue, model):
    """返回 {config, yaml, snapshot, kind, motion_spec, encoder} 各项检查结果字符串与细节。"""
    from omegaconf import OmegaConf
    spec = queue["models"][model]
    result = {}
    try:
        from mme_vla_suite.training import config as _config
        _config.get_config(spec["policy_config"])
        result["config"] = "ok"
    except Exception as exc:  # noqa: BLE001 —— dry-run 汇报每项结论，不中断
        result["config"] = f"FAIL({type(exc).__name__}:{exc})"
    yaml_path = YAML_DIR / spec["history_yaml"]
    cfg = None
    try:
        cfg = OmegaConf.load(yaml_path)
        result["yaml"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["yaml"] = f"FAIL({type(exc).__name__})"
    root = train_root(queue, model)
    snapshot = None
    if (root / "history_config.resolved.yaml").is_file():
        try:
            from mme_vla_suite.policies.policy_config import _load_resolved_snapshot
            snapshot, _ = _load_resolved_snapshot(root)
            result["snapshot"] = "ok"
        except Exception as exc:  # noqa: BLE001
            result["snapshot"] = f"FAIL({exc})"
    else:
        result["snapshot"] = "missing"
    if (root / "history_config.txt").is_file() and (root / "history_config.txt").read_text().strip() != spec["history_yaml"]:
        result["yaml"] = "FAIL(history_config.txt≠队列)"
    source = snapshot if snapshot is not None else cfg
    kind = expected_kind(source) if source is not None else None
    if spec["model_kind"] not in KINDS:
        result["kind"] = f"FAIL(unknown:{spec['model_kind']})"
    elif kind != spec["model_kind"]:
        result["kind"] = f"FAIL(derived={kind})"
    else:
        result["kind"] = "ok"
    motion = source.get("motion", None) if source is not None else None
    if motion is not None and motion.get("enabled", False):
        try:
            from mme_vla_suite.datastore.motion_adapter import motion_policy_spec
            ms = motion_policy_spec(OmegaConf.to_container(motion, resolve=True))
            result["readout"] = f"{ms.readout}:{ms.num_tokens}x{ms.dim}"
        except Exception as exc:  # noqa: BLE001
            result["readout"] = f"FAIL({exc})"
        if spec["model_kind"] == "stage2_pool4_concat":
            result["encoder"] = encoder_state(queue, model)[0]
    else:
        result["readout"] = "none"
    return result


def encoder_state(queue, model):
    """stage2 encoder 定位与 motion_client._init_stage2 相同：run 根交付实体优先，其次 v1-store 注册目录。"""
    spec = queue["models"][model]
    root = train_root(queue, model)
    if (root / "encoder_dependencies").is_dir():
        return "delivered", root / "encoder_dependencies"
    registered = V1 / "external/motionjepa" / spec["encoder_run"]
    if registered.exists():
        return "registered", registered
    candidate = spec.get("encoder_run_dir")
    if candidate and (Path(candidate) / "config.yaml").is_file() and any(Path(candidate).glob("checkpoint_epoch_*.pt")):
        return "linkable", Path(candidate)
    return f"missing({candidate})", None


def ensure_encoder(queue, model):
    """注册目录缺失但队列文件给的 encoder_run_dir 齐全时，建一条 v1-store 注册目录 symlink；返回是否可用。"""
    state, path = encoder_state(queue, model)
    if state in ("delivered", "registered"):
        return True
    if state == "linkable":
        link = V1 / "external/motionjepa" / queue["models"][model]["encoder_run"]
        link.symlink_to(path, target_is_directory=True)
        say(f"ENCODER_LINK model={model} link={link} -> {path}")
        return True
    return False


def ensure_contract(queue, model):
    """GL 训练的 split 模型：记录 runtime 合同（剔硬件字段、软件与精度仍严格），同 o16abl-3seed-1108。"""
    if not queue["models"][model].get("target_contract"):
        return None
    path = out_root(queue) / "contracts" / f"{model}.runtime.json"
    if not path.exists():
        from scripts.evaluation import measure_target_runtime
        measure_target_runtime.main(["--provenance", str(train_root(queue, model) / "motion_provenance.json"),
                                     "--output", str(path)])
    return path


def check_tree(queue, model, step):
    """抽象建模（不分配参数）后与 ckpt 参数树逐路径比对，等价于严格恢复的 missing/extra 检查。"""
    import dataclasses
    import jax
    from flax import nnx
    import orbax.checkpoint as ocp
    from mme_vla_suite.policies.policy_config import _load_resolved_snapshot
    from mme_vla_suite.training import config as _config
    spec = queue["models"][model]
    root = train_root(queue, model)
    cfg = _config.get_config(spec["policy_config"])
    history, _ = _load_resolved_snapshot(root)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, history_config=history, use_history=True))
    abstract = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))

    def shapes(tree):
        return {"/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in kp): tuple(leaf.shape)
                for kp, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}
    model_shapes = shapes(nnx.state(abstract, nnx.Param).to_pure_dict())
    with ocp.PyTreeCheckpointer() as ckptr:
        # openpi 训练存盘的叶子带 "/value" 后缀，restore_params 恢复时去掉；这里同样去掉再比。
        ckpt_shapes = {k.removesuffix("/value"): v for k, v in
                       shapes(ckptr.metadata(ckpt_dir(queue, model, step) / "params")["params"]).items()}
    missing, extra = sorted(model_shapes.keys() - ckpt_shapes.keys()), sorted(ckpt_shapes.keys() - model_shapes.keys())
    shape_diff = sorted(k for k in model_shapes.keys() & ckpt_shapes.keys() if model_shapes[k] != ckpt_shapes[k])
    import math
    count = sum(math.prod(s) for s in model_shapes.values())
    motion = any("motion" in k for k in model_shapes)
    detail = f"leaves={len(model_shapes)} params={count} motion_params={motion}"
    if shape_diff:
        return f"FAIL(shape_diff={len(shape_diff)}:{shape_diff[0]})", detail
    return ("ok" if not missing and not extra else f"FAIL(missing={len(missing)},extra={len(extra)})"), detail


def smoke_verdict(queue, model, step):
    path = smoke_record_path(queue, model, step)
    return read_json(path) if path.exists() else None


def dry_run(queue, items, check):
    say(f"DRY_RUN repo={REPO} queue_items={len(items)} optional_tail_enabled={bool(queue.get('optional_tail_enabled'))}")
    manifest = build_official_manifest(queue["benchmark"], utils_path=REPO / "examples/robomme/utils.py")
    say(f"MANIFEST sha256={manifest['sha256']} {'ok' if manifest['sha256'] == queue['manifest_sha256'] else 'FAIL'} "
        f"episodes={len(manifest['episodes'])}")
    models = {m: inspect_model(queue, m) for m in dict.fromkeys(it["model"] for it in items)}
    for m, r in models.items():
        spec = queue["models"][m]
        say(f"MODEL {m} run={spec['run']} policy_config={spec['policy_config']} kind={spec['model_kind']} "
            f"sidecar={spec['sidecar']} " + " ".join(f"{k}={v}" for k, v in r.items()))
    trees = {}
    if check:
        for m, step in distinct_ckpts(items):
            if ckpt_present(ckpt_dir(queue, m, step)) and models[m]["snapshot"] == "ok" and (m, None) not in trees:
                try:
                    trees[m, None] = check_tree(queue, m, step)
                except Exception as exc:  # noqa: BLE001
                    trees[m, None] = (f"FAIL({type(exc).__name__}:{exc})", "")
                say(f"TREE model={m} step={step} result={trees[m, None][0]} {trees[m, None][1]}")
    ok = 0
    for i, it in enumerate(items, 1):
        path = ckpt_dir(queue, it["model"], it["step"])
        present = ckpt_present(path)
        r = models[it["model"]]
        verdict = smoke_verdict(queue, it["model"], it["step"])
        smoke = verdict["verdict"] if verdict else "PENDING"
        kind_ok = r["kind"] == "ok" and r["config"] == "ok" and r["yaml"] == "ok"
        ok += present and kind_ok
        say(f"PLAN i={i:02d} item={it['name']} ckpt={'present' if present else 'missing'} "
            f"kind={'ok' if kind_ok else 'FAIL'} config={r['config'][:40]} snapshot={r['snapshot'][:40]} smoke={smoke} "
            f"path={path}")
    say(f"DRY_RUN_DONE items={len(items)} ready={ok} distinct_ckpts={len(distinct_ckpts(items))}")
    return 0


# ── 进程与会话 ────────────────────────────────────────────────────────────────

def start_session(name, argv, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    variables = [f"{k}={v}" for k, v in {"PYTHONUNBUFFERED": "1", **(env or {})}.items()]
    command = shlex.join(["env", *variables, *map(str, argv)])
    path = shlex.quote(str(log))
    body = f'cd {shlex.quote(str(REPO))}; set -o pipefail; {command} 2>&1 | tee -a {path}; echo "EXIT_CODE=$?" >> {path}'
    subprocess.run(["tmux", "new-session", "-d", "-s", name, "bash", "-c", body], check=True)


def alive(name):
    return subprocess.run(["tmux", "has-session", "-t", f"={name}"], capture_output=True).returncode == 0


def kill_session(name):
    if alive(name):
        subprocess.run(["tmux", "kill-session", "-t", f"={name}"], check=False)


def last_exit_code(log):
    try:
        lines = log.read_bytes()[-2048:].decode(errors="replace").strip().splitlines()
    except OSError:
        return None
    return lines[-1].split("=", 1)[1] if lines and lines[-1].startswith("EXIT_CODE=") else None


def cache_env():
    v1 = V1
    return {"OPENPI_DATA_HOME": v1 / "models", "HF_HOME": v1 / "cache/hf", "XDG_CACHE_HOME": v1 / "cache/xdg",
            "TORCH_HOME": v1 / "cache/torch", "MMEVLA_V1_STORE": v1}


def gpu_used_mib():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                  text=True)
    return {int(i): int(m) for i, m in (line.split(",") for line in raw.strip().splitlines())}


def term_pid(pid):
    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except (ProcessLookupError, ValueError, PermissionError):
        return False


# ── 一项评测的运行根 ──────────────────────────────────────────────────────────

def make_spec(queue, model, step, seed, manifest, code, purpose, contract):
    spec_m = queue["models"][model]
    ckpt = ckpt_dir(queue, model, step)
    identity = {"schema": "eval3x3-run-identity-1110", "model": model, "run": spec_m["run"], "checkpoint": str(ckpt),
                "checkpoint_step": step, "episodes_sha256": manifest["sha256"], "eval_code_commit": code,
                "model_seed": seed, "model_kind": spec_m["model_kind"], "policy_config": spec_m["policy_config"],
                "purpose": purpose, "runtime_contract_sha256": sha256_file(contract) if contract else None}
    spec = {"repo": str(REPO), "policy_python": queue["policy_python"], "stage2_python": queue["stage2_python"],
            "sim_python": queue["sim_python"], "checkpoint": str(ckpt), "policy_config": spec_m["policy_config"],
            "model_kind": spec_m["model_kind"], "model_seed": seed, "purpose": purpose,
            "run_identity_sha256": identity_sha256(identity)}
    if contract:
        spec["target_contract"] = str(contract)
    return identity, spec


def init_root(root, manifest, identity, spec):
    """新根写入清单、spec 与身份；已有根原样沿用（断点续跑），只核对模型、ckpt 与 seed 未变。"""
    if (root / "spec.json").exists():
        old = read_json(root / "spec.json")
        if any(old[k] != spec[k] for k in ("checkpoint", "model_seed", "model_kind", "policy_config")):
            raise ValueError(f"已有运行根的模型身份与队列不符: {root}")
        return old
    QueueStore(root).initialize(manifest, spec)
    publish(root / "run_identity.json", identity)
    return spec


def open_assignments(state):
    return {a["grant"]["worker_id"]: a["grant"] for a in state["assignments"].values() if not a["closed"]}


def unclaimed_chunks(root, summary, state):
    """仍有待评局、且没有被任何 worker 占着的块数；控制器 summary 不存在时视为全部未领。"""
    manifest = read_json(root / "manifest.json")
    if not summary:
        return len(manifest["chunks"])
    ready = set(summary["missing"]) - set(summary["errors"])
    occupied = {g["chunk_id"] for g in open_assignments(state).values()}
    return sum(1 for c in manifest["chunks"] if c["chunk_id"] not in occupied and ready & set(c["episode_keys"]))


def recover_stale(root, state, label):
    """续跑：上次留下、进程已不在却仍占着块的 worker，按本机终态申请恢复一次。"""
    store = QueueStore(root)
    for wid in open_assignments(state):
        worker = state["workers"].get(wid, {})
        if worker.get("state") == "RUNNING" and not Path(f"/proc/{worker.get('step_id')}").exists():
            store.request(f"recover-{wid}-{uuid.uuid4().hex}", "recover",
                          {"worker_id": wid, "job_id": "local", "step_id": worker["step_id"]})
            say(f"RECOVER_STALE item={label} worker={wid}")


class Run:
    """一个运行根（正式项或冒烟）及其控制器、worker 与占用的 GPU。"""

    def __init__(self, queue, label, root, spec):
        self.queue, self.label, self.root, self.spec = queue, label, root, spec
        self.prefix = queue["tmux_prefix"]
        self.controller = f"{self.prefix}ctl-{label}"
        self.workers = {}          # wid -> {gpu, session, log, handled}
        self.gpus = set()
        self.drained, self.recovered = set(), set()
        self.ctl_restarts = self.deaths = 0
        self.finishing_since = None

    def start_controller(self):
        start_session(self.controller, [self.queue["policy_python"], QUEUE_PY, "controller", "--run-root", self.root,
                                        "--local"], self.root / "logs" / "tmux-controller.log")

    def summary(self):
        path = self.root / "queue/summary.json"
        try:
            return read_json(path) if path.exists() else {}
        except (OSError, ValueError):
            return {}

    def state(self):
        path = self.root / "queue/state.json"
        return read_json(path) if path.exists() else {"workers": {}, "assignments": {}}

    def add_worker(self, gpu, slot, port):
        wid = f"g{gpu}k{slot}-{uuid.uuid4().hex[:6]}"
        session = f"{self.prefix}{self.label}-{wid}"
        log = self.root / "logs" / f"tmux-{wid}.log"
        env = {"CUDA_VISIBLE_DEVICES": gpu, "XLA_PYTHON_CLIENT_MEM_FRACTION": self.queue["mem_fraction"], **cache_env()}
        start_session(session, [self.queue["policy_python"], WORKER_PY, "--run-root", self.root, "--spec",
                                self.root / "spec.json", "--worker-id", wid, "--local", "--port", port], log, env)
        self.workers[wid] = {"gpu": gpu, "session": session, "log": log, "handled": False}
        return wid

    def alive_workers(self):
        return {wid for wid, w in self.workers.items() if alive(w["session"])}

    def reap(self, state, finished, pending_gpus=()):
        """处理已退出 worker：仍占块的申请恢复；非零退出或占块退出计为一次死亡；无活 worker 且无待起 worker 的卡交还。"""
        busy = open_assignments(state)
        live = self.alive_workers()
        for wid, w in self.workers.items():
            if wid in live or w["handled"]:
                continue
            w["handled"] = True
            code = last_exit_code(w["log"])
            if wid in busy:
                step = state["workers"].get(wid, {}).get("step_id")
                QueueStore(self.root).request(f"recover-{wid}-{uuid.uuid4().hex}", "recover",
                                              {"worker_id": wid, "job_id": "local", "step_id": step})
                say(f"RECOVER item={self.label} worker={wid} exit={code}")
            if wid in busy or (code != "0" and not finished):
                self.deaths += 1
                say(f"WORKER_DEAD item={self.label} worker={wid} exit={code} deaths={self.deaths}")
        for gpu in list(self.gpus - set(pending_gpus)):
            if not any(w["gpu"] == gpu and wid in live for wid, w in self.workers.items()):
                self.gpus.discard(gpu)
                say(f"GPU_RELEASE item={self.label} gpu={gpu}")

    def drain_idle(self, state):
        """块已全部领完：给手上没有块的 worker 发 drain，让它退出并交出 GPU。"""
        busy = open_assignments(state)
        store = QueueStore(self.root)
        for wid in self.alive_workers() - set(busy) - self.drained:
            store.drain_path(wid).parent.mkdir(parents=True, exist_ok=True)
            store.drain_path(wid).touch()
            self.drained.add(wid)

    def term_workers(self, state):
        """按注册 PID 发 SIGTERM，让 worker 自己停仿真与策略服务（kill-session 停不掉独立会话的 sidecar）。"""
        for wid in self.alive_workers():
            pid = state["workers"].get(wid, {}).get("step_id")
            if pid:
                term_pid(pid)


# ── 正式评测 ──────────────────────────────────────────────────────────────────

class Scheduler:
    def __init__(self, queue, items, manifest, code):
        self.queue, self.items, self.manifest, self.code = queue, items, manifest, code
        self.gpus = list(queue["gpus"])
        self.active, self.done, self.blocked = [], 0, 0
        self.nxt = 0
        self.pending_starts = []   # (run, gpu, slot)
        self.last_start = 0.0
        self.gpu_epoch = {g: 0 for g in self.gpus}
        self.mem_cache, self.mem_time = {}, 0.0

    def assigned(self):
        return set().union(*[r.gpus for r in self.active]) if self.active else set()

    def gpu_free(self, gpu):
        if time.monotonic() - self.mem_time > 30:
            self.mem_cache, self.mem_time = gpu_used_mib(), time.monotonic()
        return self.mem_cache.get(gpu, 10 ** 9) < self.queue["gpu_free_mib"]

    def admit(self, it):
        """到该项时判定：已完成跳过、缺 ckpt / encoder / 冒烟未过标 BLOCKED，否则建根起控制器。"""
        root = out_root(self.queue) / it["name"]
        if (root / "accepted").is_dir() and len(list((root / "accepted").glob("*.json"))) == TOTAL:
            say(f"ITEM_SKIP item={it['name']} reason=already_done")
            self.done += 1
            return None
        model, step = it["model"], it["step"]
        reason = None
        verdict = smoke_verdict(self.queue, model, step)
        if not ckpt_present(ckpt_dir(self.queue, model, step)):
            reason = "ckpt_missing"
        elif verdict is None:
            reason = "smoke_missing"
        elif verdict["verdict"] != "PASS":
            reason = "smoke_fail"
        elif verdict.get("eval_code_commit") != self.code:
            reason = "smoke_stale_code"
        elif self.queue["models"][model]["model_kind"] == "stage2_pool4_concat" and not ensure_encoder(self.queue, model):
            reason = "encoder_missing"
        if reason:
            say(f"ITEM_BLOCKED item={it['name']} reason={reason}")
            self.blocked += 1
            return None
        contract = ensure_contract(self.queue, model)
        identity, spec = make_spec(self.queue, model, step, it["seed"], self.manifest, self.code, "production", contract)
        spec = init_root(root, self.manifest, identity, spec)
        run = Run(self.queue, it["name"], root, spec)
        state = run.state()
        recover_stale(root, state, it["name"])
        run.start_controller()
        completed = len(QueueStore(root).accepted())
        say(f"EVAL_START item={it['name']} ckpt={spec['checkpoint']} seed={spec['model_seed']} "
            f"kind={spec['model_kind']} run_identity_sha256={spec['run_identity_sha256']} resume_completed={completed}")
        return run

    def give_gpu(self, run, gpu):
        run.gpus.add(gpu)
        self.gpu_epoch[gpu] += 1
        base = self.queue["port_base"] + gpu * 20 + (self.gpu_epoch[gpu] % 2) * 10
        for slot in range(self.queue["per_gpu"]):
            self.pending_starts.append((run, gpu, slot, base + slot))
        say(f"GPU_ASSIGN item={run.label} gpu={gpu}")

    def finish(self, run, summary, state, status):
        kill_session(run.controller)
        outcomes = summary.get("outcomes", {})
        retries = sum(state.get("retries", {}).values())
        say(f"EVAL_DONE item={run.label} success={outcomes.get('success', 0)}/{TOTAL} "
            f"errors={len(summary.get('errors', {}))} retries={retries} status={status} "
            f"completed={summary.get('completed', 0)}")
        self.active.remove(run)
        self.done += 1

    def step_run(self, run):
        summary, state = run.summary(), run.state()
        status = summary.get("compute_status")
        terminal = status in ("COMPUTE_DONE", "STOPPED_ERROR") or run.deaths >= self.queue["worker_death_limit"]
        run.reap(state, terminal, {p[1] for p in self.pending_starts if p[0] is run})
        if not alive(run.controller) and not terminal:
            if run.ctl_restarts >= 1:
                say(f"QUEUE_FATAL reason=controller_dead item={run.label}")
                raise SystemExit(1)
            run.ctl_restarts += 1
            say(f"CONTROLLER_RESTART item={run.label}")
            run.start_controller()
            return
        if terminal:
            if run.finishing_since is None:
                run.finishing_since = time.monotonic()
                if status not in ("COMPUTE_DONE", "STOPPED_ERROR"):
                    run.drain_idle(state)
                    run.term_workers(state)
            pending = any(p[0] is run for p in self.pending_starts)
            if pending:
                self.pending_starts = [p for p in self.pending_starts if p[0] is not run]
            if not run.alive_workers() or time.monotonic() - run.finishing_since > 300:
                if run.alive_workers():
                    run.term_workers(state)
                    return
                label = status if status in ("COMPUTE_DONE", "STOPPED_ERROR") else "ABORTED_WORKER_DEATHS"
                self.finish(run, summary, state, label)
            return
        if summary and unclaimed_chunks(run.root, summary, state) == 0:
            self.pending_starts = [p for p in self.pending_starts if p[0] is not run]
            run.drain_idle(state)

    def needers(self):
        """需要 GPU 的项：最新起跑的项只要还有未领块就要；更早的项只在未领块且已无任何 GPU 时才要。"""
        result = []
        for run in self.active:
            summary, state = run.summary(), run.state()
            if run.finishing_since is not None or (summary and summary.get("compute_status") != "READY"
                                                   and summary.get("compute_status") != "WAIT"):
                continue
            if unclaimed_chunks(run.root, summary, state) > 0 and (run is self.active[-1] or not run.gpus):
                result.append(run)
        return result

    def loop(self):
        while self.tick():
            time.sleep(TICK)
        say(f"QUEUE_DONE done={self.done} blocked={self.blocked}")
        return 0

    def tick(self):
        """一轮调度；队列全部结束时返回 False。"""
        for run in list(self.active):
            self.step_run(run)
        free = [g for g in self.gpus if g not in self.assigned()]
        if free:
            needers = self.needers()
            while not needers and self.nxt < len(self.items):
                run = self.admit(self.items[self.nxt])
                self.nxt += 1
                if run is not None:
                    self.active.append(run)
                    needers = [run]
            if needers:
                for gpu in free:
                    if self.gpu_free(gpu):
                        self.give_gpu(needers[0], gpu)
        if self.pending_starts and time.monotonic() - self.last_start >= self.queue["worker_start_gap_seconds"]:
            run, gpu, slot, port = self.pending_starts.pop(0)
            run.add_worker(gpu, slot, port)
            self.last_start = time.monotonic()
        return self.nxt < len(self.items) or bool(self.active)


def official_manifest(queue):
    manifest = build_official_manifest(queue["benchmark"], utils_path=REPO / "examples/robomme/utils.py")
    if manifest["sha256"] != queue["manifest_sha256"]:
        say(f"QUEUE_FATAL reason=manifest_sha got={manifest['sha256']}")
        raise SystemExit(2)
    return manifest


def run_formal(queue, items):
    require_execution_copy()
    manifest = official_manifest(queue)
    replace_json(out_root(queue) / "episodes.json", manifest)
    code = code_commit()
    say(f"QUEUE_START repo={REPO} code={code} items={len(items)} gpus={queue['gpus']} per_gpu={queue['per_gpu']}")
    return Scheduler(queue, items, manifest, code).loop()


# ── 冒烟 ──────────────────────────────────────────────────────────────────────

def run_smoke(queue, items, only=None):
    require_execution_copy()
    manifest = official_manifest(queue)
    task, index = queue["smoke_episode"].split(":")
    episode = next(e for e in manifest["episodes"] if e["episode_key"] == f"{task}:{int(index)}")
    small = make_manifest([{k: episode[k] for k in ("task_name", "episode_id", "env_seed", "difficulty")}],
                          metadata_sha256=manifest["metadata_sha256"], official=False)
    code = code_commit()
    todo, counts = [], {"PASS": 0, "FAIL": 0, "PENDING": 0}
    for model, step in distinct_ckpts(items):
        if only and f"{model}-{step}" not in only:
            continue
        old = smoke_verdict(queue, model, step)
        if old and old["verdict"] == "PASS" and old.get("eval_code_commit") == code:
            say(f"SMOKE=PASS model={model} step={step} cached=1 outcome={old.get('outcome')}")
            counts["PASS"] += 1
        elif not ckpt_present(ckpt_dir(queue, model, step)):
            say(f"SMOKE=PENDING model={model} step={step} reason=ckpt_missing")
            counts["PENDING"] += 1
        else:
            todo.append((model, step))
    gpus, active, port = list(queue["gpus"]), {}, queue["port_base"] + 500
    timeout = queue["smoke_timeout_seconds"]
    while todo or active:
        used = gpu_used_mib() if todo and len(active) < len(gpus) else {}
        for gpu in [g for g in gpus if g not in active]:
            if not todo:
                break
            if used.get(gpu, 10 ** 9) >= queue["gpu_free_mib"]:
                continue
            model, step = todo.pop(0)
            verdict = smoke_start(queue, model, step, gpu, small, code, port + gpu)
            if isinstance(verdict, dict):
                smoke_finish(queue, verdict, counts)
            else:
                active[gpu] = verdict
        for gpu, (run, started, model, step) in list(active.items()):
            summary, state = run.summary(), run.state()
            status = summary.get("compute_status")
            reason = None
            if status == "COMPUTE_DONE":
                reason = "ok"
            elif status == "STOPPED_ERROR":
                reason = "stopped_error"
            elif not run.alive_workers():
                reason = f"worker_exit={last_exit_code(next(iter(run.workers.values()))['log'])}"
            elif time.monotonic() - started > timeout:
                reason = "timeout"
                run.term_workers(state)
            if reason is None:
                continue
            end = time.monotonic() + 120
            while run.alive_workers() and time.monotonic() < end:
                time.sleep(2)
            kill_session(run.controller)
            accepted = list(QueueStore(run.root).accepted().values())
            # worker 先于 summary 刷新退出时 reason 可能是 worker_exit=0；以 accepted 与 errors 为准。
            passed = (reason in ("ok", "worker_exit=0") and len(accepted) == 1
                      and not run.state().get("errors"))
            smoke_finish(queue, {"verdict": "PASS" if passed else "FAIL", "model": model, "step": step,
                                 "root": str(run.root), "outcome": accepted[0]["outcome"] if accepted else None,
                                 "reason": reason, "eval_code_commit": code, "gpu": gpu,
                                 "seconds": round(time.monotonic() - started)}, counts)
            del active[gpu]
        if todo or active:
            time.sleep(TICK)
    say(f"SMOKE_DONE pass={counts['PASS']} fail={counts['FAIL']} pending={counts['PENDING']}")
    return 0 if counts["FAIL"] == 0 else 1


def smoke_start(queue, model, step, gpu, small, code, port):
    base = out_root(queue) / "smoke" / f"{model}-{step}"
    attempt = 1 + sum(1 for p in base.glob("try*") if p.is_dir())
    root = base / f"try{attempt}"
    if queue["models"][model]["model_kind"] == "stage2_pool4_concat" and not ensure_encoder(queue, model):
        return {"verdict": "FAIL", "model": model, "step": step, "root": str(root), "outcome": None,
                "reason": "encoder_missing", "eval_code_commit": code, "gpu": gpu, "seconds": 0}
    contract = ensure_contract(queue, model)
    identity, spec = make_spec(queue, model, step, queue["smoke_seed"], small, code, "validation", contract)
    init_root(root, small, identity, spec)
    run = Run(queue, f"smoke-{model}-{step}", root, spec)
    run.start_controller()
    run.add_worker(gpu, 0, port)
    say(f"SMOKE_START model={model} step={step} gpu={gpu} root={root}")
    return run, time.monotonic(), model, step


def smoke_finish(queue, record, counts):
    replace_json(smoke_record_path(queue, record["model"], record["step"]), record)
    counts[record["verdict"]] += 1
    say(f"SMOKE={record['verdict']} model={record['model']} step={record['step']} outcome={record['outcome']} "
        f"reason={record['reason']} seconds={record['seconds']} root={record['root']}")


# ── 停止 ──────────────────────────────────────────────────────────────────────

def repo_python_pids():
    """cwd 精确等于本执行副本根目录、且可执行文件名为 python* 的进程（排除自己）。"""
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if Path(os.readlink(proc / "cwd")) != REPO or not Path(os.readlink(proc / "exe")).name.startswith("python"):
                continue
        except OSError:
            continue
        pids.append(int(proc.name))
    return sorted(pids)


def run_stop(grace):
    require_execution_copy()
    pids = repo_python_pids()
    say(f"STOP_TARGETS repo={REPO} pids={pids}")
    for pid in pids:
        term_pid(pid)
    end = time.monotonic() + grace
    while time.monotonic() < end and repo_python_pids():
        time.sleep(2)
    left = repo_python_pids()
    say(f"STOP_DONE sent={len(pids)} remaining={left}")
    return 0 if not left else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--stop", action="store_true")
    parser.add_argument("--check-tree", action="store_true", help="dry-run 时对现存 ckpt 做抽象建模参数树比对")
    parser.add_argument("--only", action="append", help="冒烟只跑指定 <model>-<step>，可重复")
    parser.add_argument("--stop-grace", type=float, default=90)
    args = parser.parse_args(argv)
    if args.stop:
        return run_stop(args.stop_grace)
    queue = load_queue(args.queue)
    items = plan_items(queue)
    if args.dry_run:
        return dry_run(queue, items, args.check_tree)
    if args.smoke:
        return run_smoke(queue, items, args.only)
    return run_formal(queue, items)


if __name__ == "__main__":
    raise SystemExit(main())
