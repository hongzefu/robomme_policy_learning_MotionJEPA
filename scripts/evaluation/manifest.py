"""不导入仿真的官方测试清单与固定分块。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import re


TASKS = (
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes", "ButtonUnmask",
    "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap", "PickHighlight",
    "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube",
    "InsertPeg", "PatternLock", "RouteStick",
)
BENCHMARK_COMMIT = "856bc3a189d4172f3f47dbee4424d585f8d78db3"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seal(value):
    value = dict(value)
    value.pop("sha256", None)
    return {**value, "sha256": digest(value)}


def verify_seal(value):
    plain = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != digest(plain):
        raise ValueError("清单SHA不符")
    return value


def episode_key(episode):
    return f"{episode['task_name']}:{episode['episode_id']}"


def validate_episodes(manifest, *, official=True):
    verify_seal(manifest)
    episodes = manifest["episodes"]
    keys = [episode_key(e) for e in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError("重复episode")
    for e, key in zip(episodes, keys, strict=True):
        if e.get("episode_key") != key or not isinstance(e["env_seed"], int):
            raise ValueError("episode身份不完整")
        if not isinstance(e["difficulty"], str) or not e["difficulty"]:
            raise ValueError("difficulty缺失")
    if official:
        expected = [f"{task}:{i}" for i in range(50) for task in TASKS]
        if keys != expected or manifest.get("dataset") != "test":
            raise ValueError("官方母集必须按固定顺序包含16×50局")
        if manifest.get("chunk_size") != 5:
            raise ValueError("官方chunk必须为5局")
    flattened = []
    for chunk in manifest["chunks"]:
        members = chunk["episode_keys"]
        if chunk["chunk_id"] != digest(members):
            raise ValueError("chunk成员SHA不符")
        flattened.extend(members)
    if flattened != keys:
        raise ValueError("chunk覆盖不符")
    if official:
        expected_chunks = [keys[i:i + 5] for i in range(0, 800, 5)]
        if [chunk["episode_keys"] for chunk in manifest["chunks"]] != expected_chunks:
            raise ValueError("官方必须为固定160块，每块恰好5局")
    return manifest


def make_manifest(episodes, *, metadata_sha256=None, benchmark_commit=BENCHMARK_COMMIT,
                  chunk_size=5, official=True):
    episodes = [{**e, "episode_key": episode_key(e)} for e in episodes]
    keys = [e["episode_key"] for e in episodes]
    chunks = [{"chunk_id": digest(keys[i:i + chunk_size]),
               "episode_keys": keys[i:i + chunk_size]}
              for i in range(0, len(keys), chunk_size)]
    return validate_episodes(seal({"schema_version": 1, "dataset": "test",
                                  "benchmark_commit": benchmark_commit,
                                  "metadata_sha256": metadata_sha256 or {},
                                  "chunk_size": chunk_size, "episodes": episodes,
                                  "chunks": chunks}), official=official)


def build_official_manifest(benchmark_root, *, benchmark_commit=BENCHMARK_COMMIT,
                            utils_path=None):
    """直接读取指定提交的元数据，绝不import benchmark。"""
    root = Path(benchmark_root)
    if benchmark_commit != BENCHMARK_COMMIT:
        raise ValueError("本次计划锁定的benchmark提交不符")
    if utils_path is not None:
        tree = ast.parse(Path(utils_path).read_text())
        values = [ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "TASK_NAME_LIST" for t in n.targets)]
        if values != [list(TASKS)]:
            raise ValueError("client官方任务顺序已变化")
    indexed, hashes = {}, {}
    for task in TASKS:
        relative = f"src/robomme/env_metadata/test/record_dataset_{task}_metadata.json"
        raw = subprocess.check_output(["git", "-C", str(root), "show", f"{benchmark_commit}:{relative}"])
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        records = data["records"]
        if data["env_id"] != task or data["record_count"] != 50 or len(records) != 50:
            raise ValueError(f"{task}官方数量不为50")
        for record in records:
            i = record["episode"]
            if record["task"] != task or type(i) is not int or not 0 <= i < 50 or (task, i) in indexed:
                raise ValueError(f"{task}元数据身份错误")
            indexed[task, i] = {"task_name": task, "episode_id": i,
                                "env_seed": int(record["seed"]), "difficulty": record["difficulty"]}
    return make_manifest([indexed[task, i] for i in range(50) for task in TASKS],
                         metadata_sha256=hashes, benchmark_commit=benchmark_commit)


def budget_record(episode, demo_frames, *, q=1301, budget=96, position_limit=4096):
    """D为demo末尾序号；窗口W33、S32，demo段尾补一次。"""
    if type(demo_frames) is not int or demo_frames < 0:
        raise ValueError("D必须为非负整数")
    k_demo = max(0, (demo_frames - 17) // 32 + 1)
    k_exec = q // 32
    passed = k_demo + k_exec <= budget and demo_frames + q < position_limit
    return {"episode_key": episode_key(episode), "D": demo_frames, "q": q,
            "k_demo": k_demo, "k_exec": k_exec, "passed": passed}


def budget_report(manifest, records):
    validate_episodes(manifest)
    if {r["episode_key"] for r in records} != {e["episode_key"] for e in manifest["episodes"]} or len(records) != 800:
        raise ValueError("预算报告未覆盖800唯一局")
    checked = [budget_record(e, next(r["D"] for r in records if r["episode_key"] == e["episode_key"]))
               for e in manifest["episodes"]]
    return seal({"manifest_sha256": manifest["sha256"], "records": checked,
                 "passed": all(r["passed"] for r in checked)})


def bind_model(manifest, models, target_contract, *, validation=False):
    """绑定完整实体字节；验证20k与正式60k具有不同、不可混用的身份。"""
    try:
        from .runtime_contract import descriptor_identity, validate_eval_runtime_contract
    except ImportError:
        from runtime_contract import descriptor_identity, validate_eval_runtime_contract
    validate_episodes(manifest)
    if not isinstance(models.get("models"), list) or len(models["models"]) != 1:
        raise ValueError("当前阶段仅允许一个明确指定的模型")
    model = models["models"][0]
    step = 20000 if validation else 60000
    if model.get("checkpoint_step") != step or not model.get("model_id"):
        raise ValueError(f"本用途必须使用明确的{step} checkpoint")
    if re.fullmatch("[0-9a-f]{40}", model.get("eval_code_commit", "")) is None:
        raise ValueError("缺锁定的eval代码commit")
    original_root = Path(model["checkpoint_root"])
    root = original_root.resolve(strict=True)
    if not original_root.is_absolute() or original_root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint根必须为实体绝对目录")
    files = model["checkpoint_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint成员清单为空")
    declared = {}
    for entry in files:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in declared:
            raise ValueError("checkpoint成员重复或路径越界")
        path = root / relative
        if any(part.is_symlink() for part in [path, *path.parents] if part.is_relative_to(root)):
            raise ValueError("checkpoint禁止符号链接")
        if not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
            raise ValueError("checkpoint成员不是根内文件")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(block)
        if path.stat().st_size != entry["bytes"] or hasher.hexdigest() != entry["sha256"]:
            raise ValueError("checkpoint成员长度或SHA不符")
        declared[str(relative)] = entry
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(declared) or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("checkpoint实际成员集合与清单不符")
    for name in ("resolved", "normstats"):
        entry = declared.get(model[f"{name}_path"])
        if entry is None or entry["sha256"] != model[f"{name}_sha256"]:
            raise ValueError(f"{name}不在锁定成员清单")
    provenance_path = model["training_provenance_path"]
    if provenance_path not in declared:
        raise ValueError("训练来源不在锁定成员清单")
    provenance = json.loads((root / provenance_path).read_text())
    target_path = Path(target_contract)
    # 用户已取消跨硬件数值等价前置；源、权重、软件精度与运行合同继续严格绑定。
    runtime_contract = validate_eval_runtime_contract(target_path, training_provenance=provenance)
    if runtime_contract["motion_identity_sha256"] != model["motion_identity_sha256"]:
        raise ValueError("VLA与运行合同motion身份不符")
    purpose = "validation" if validation else "production"
    descriptor = {"purpose": purpose, "checkpoint_step": step,
                  "checkpoint_files": files, "resolved_sha256": model["resolved_sha256"],
                  "normstats_sha256": model["normstats_sha256"],
                  "motion_identity_sha256": model["motion_identity_sha256"]}
    model_identity = descriptor_identity("validation_model" if validation else "model", descriptor)
    identity_key = "validation_model_identity_sha256" if validation else "model_identity_sha256"
    run_descriptor = {"purpose": purpose, identity_key: model_identity,
        "runtime_contract_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "cross_hardware_numeric": runtime_contract["cross_hardware_numeric"],
        "benchmark_commit": manifest["benchmark_commit"], "metadata_sha256": manifest["metadata_sha256"],
        "episode_manifest_sha256": manifest["sha256"], "eval_code_commit": model["eval_code_commit"],
        "model_seed": 42, "eval_parameters": {"max_steps": 1300, "conservative_max_q": 1301,
                                               "motion_budget": 96, "position_limit": 4096}}
    return seal({"schema_version": 1, "purpose": purpose, "model_id": model["model_id"],
                 "model_seed": 42, "checkpoint_root": str(root), "model_descriptor": descriptor,
                 "cross_hardware_numeric": runtime_contract["cross_hardware_numeric"],
                 "episode_manifest": manifest, "model_source": model,
                 identity_key: model_identity, "run_descriptor": run_descriptor,
                 "run_identity_sha256": descriptor_identity("validation_run" if validation else "run", run_descriptor),
                 "episode_manifest_sha256": manifest["sha256"]})


def verify_model_binding(binding_or_path, target_contract_path, *, expected_checkpoint_root=None,
                         expected_purpose=None, expected_run_identity_sha256=None):
    """启动入口重算签封输入和实际模型字节，绝不只相信传入run字符串。"""
    binding = json.loads(Path(binding_or_path).read_text()) if isinstance(binding_or_path, (str, Path)) else binding_or_path
    verify_seal(binding)
    if binding.get("purpose") not in {"production", "validation"}:
        raise ValueError("model binding用途不符")
    if expected_purpose is not None and binding["purpose"] != expected_purpose:
        raise ValueError("model binding与启动用途不符")
    if expected_checkpoint_root is not None and Path(expected_checkpoint_root).resolve(strict=True) != Path(binding["checkpoint_root"]):
        raise ValueError("实际加载模型目录不等于model binding")
    if expected_run_identity_sha256 is not None and binding["run_identity_sha256"] != expected_run_identity_sha256:
        raise ValueError("model binding与启动run身份不符")
    actual = bind_model(binding["episode_manifest"], {"models": [binding["model_source"]]},
                        target_contract_path, validation=binding["purpose"] == "validation")
    if actual != binding:
        raise ValueError("model binding重算与现场checkpoint/运行合同不符")
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    episodes = sub.add_parser("episodes")
    episodes.add_argument("--benchmark-root", type=Path, default=Path("third_party/robomme_benchmark"))
    episodes.add_argument("--benchmark-commit", default=BENCHMARK_COMMIT)
    episodes.add_argument("--dataset", choices=["test"], default="test")
    episodes.add_argument("--episodes-per-task", type=int, choices=[50], default=50)
    episodes.add_argument("--chunk-size", type=int, choices=[5], default=5)
    episodes.add_argument("--out", type=Path, required=True)
    bind = sub.add_parser("bind-model")
    bind.add_argument("--validation", action="store_true", help="仅绑定用户指定的Stage2 20000验证模型")
    for name in ("episodes", "models", "target-contract", "out"):
        bind.add_argument(f"--{name}", type=Path, required=True)
    for name, default in (("model-seed", 42), ("max-steps", 1300), ("conservative-max-q", 1301),
                          ("motion-budget", 96), ("position-limit", 4096)):
        bind.add_argument(f"--{name}", type=int, choices=[default], default=default)
    args = parser.parse_args(argv)
    if args.command == "episodes":
        result = build_official_manifest(args.benchmark_root, benchmark_commit=args.benchmark_commit,
                                         utils_path=Path(__file__).resolve().parents[2] / "examples/robomme/utils.py")
    else:
        manifest = validate_episodes(json.loads(args.episodes.read_text()))
        models = json.loads(args.models.read_text())
        result = bind_model(manifest, models, args.target_contract, validation=args.validation)
    # 同盘同步后无覆盖发布，读方永不看到半份最终清单。
    try:
        from .queue_store import publish
    except ImportError:
        from queue_store import publish
    publish(args.out, result)
    print(f"EVAL_IDENTITY=PASS episodes=800 out={args.out}")


if __name__ == "__main__":
    main()
