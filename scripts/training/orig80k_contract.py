"""公开全集与 counting 原版四卡训练的参数、路径和启动状态契约。"""
# 中文文案按仓库语言规则保留正常标点。
# ruff: noqa: RUF001, RUF002, RUF003

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "mme_vla_suite"
HISTORY = "perceptual-framesamp-modul.yaml"
UPSTREAM_HEAD = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
MODE_STEPS = {"prod": 80000, "perf": 300, "smoke": 20}
LIBRARIES = {
    "16task-pub-1600ep": {
        "gpus": "0,1,2,3", "run": "v2-orig-16task-pub1600ep-modul-b64-80k",
        "assets": "mme_vla_suite", "rows": 768897, "samples": 476857,
    },
    "4task-counting-pub-400ep": {
        "gpus": "4,5,6,7", "run": "v2-orig-counting-pub400ep-modul-b64-80k",
        "assets": "mme_vla_suite/4task-counting-pub-400ep", "rows": 189035, "samples": 189035,
    },
}
PATH_FLAGS = {
    "--exp-name", "--dataset-path", "--assets-base-dir", "--data.assets.assets-dir",
    "--data.assets.asset-id", "--checkpoint-base-dir", "--weight-loader.params-path",
    "--model.history-config", "--model.use-history",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    # 与 scan_manifest.manifest_sha256 的 UTF-8 规范化口径完全相同。
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_argv(argv: list[str], mode: str):
    """拒绝多余位置参数、重复旗标和未登记覆盖；不触发 JAX 或文件写入。"""
    require(mode in MODE_STEPS, "模式必须为 prod/perf/smoke")
    require(bool(argv) and argv[0] == CONFIG, "具名配置必须为 mme_vla_suite")
    allowed = PATH_FLAGS | ({"--num-train-steps"} if mode != "prod" else set())
    values = {}
    index = 1
    while index < len(argv):
        raw = argv[index]
        key = raw.split("=", 1)[0]
        require(key in allowed, f"{mode} 禁止参数或覆盖: {key}")
        require(key not in values, f"重复参数: {key}")
        if key == "--model.use-history":
            require(raw == key, "history 布尔旗标不能带值")
            value = True
        elif "=" in raw:
            value = raw.split("=", 1)[1]
            require(bool(value), f"参数缺值: {key}")
        else:
            index += 1
            require(index < len(argv) and not argv[index].startswith("--"), f"参数缺值: {key}")
            value = argv[index]
        values[key] = value
        index += 1
    require(values.keys() >= PATH_FLAGS, "缺少明确的路径、资产或 history 参数")
    require(values["--model.history-config"] == HISTORY and values["--model.use-history"] is True,
            "必须使用原版 modul history")
    require(values["--data.assets.asset-id"] == "robomme", "asset-id 必须为 robomme")
    if mode != "prod":
        require(values.get("--num-train-steps") == str(MODE_STEPS[mode]), "验证模式步数错误")
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", values["--exp-name"])), "run 名包含非法字符")
    return values


def make_train_args(mode, run, lib, assets, repo=ROOT):
    require(mode in MODE_STEPS, "模式必须为 prod/perf/smoke")
    store = Path(repo) / "v1-store"
    argv = [CONFIG, "--exp-name", run, "--dataset-path", str(Path(lib) / "framesamp"),
            "--assets-base-dir", str(store / "train-assets"),
            "--data.assets.assets-dir", str(assets), "--data.assets.asset-id", "robomme",
            "--checkpoint-base-dir", str(store / "train-runs"),
            "--weight-loader.params-path", str(store / "models/openpi-assets/checkpoints/pi05_base/params"),
            "--model.use-history", "--model.history-config", HISTORY]
    if mode != "prod":
        argv += ["--num-train-steps", str(MODE_STEPS[mode])]
    return argv


def entity_path(value, root, *, exists=True):
    """拒绝相对路径、父层外链和悬空链接；不以 resolve 隐藏原始路径问题。"""
    path = Path(value)
    require(path.is_absolute(), f"必须提供绝对路径: {path}")
    require(path == path.resolve() and path.is_relative_to(root), f"路径越界或包含符号链接: {path}")
    if exists:
        require(path.exists(), f"路径不存在: {path}")
    return path


def validate_paths(values, mode, run, lib, assets, repo=ROOT):
    repo = Path(repo)
    store = entity_path(repo / "v1-store", repo)
    lib = entity_path(lib, store)
    assets = entity_path(assets, store)
    require(lib.parent == store / "datasets" and lib.name in LIBRARIES, "数据根不是本轮两个公开库")
    selected = LIBRARIES[lib.name]
    require(assets == store / "train-assets" / selected["assets"], "资产父目录错误，不能重复拼接 robomme")
    require(values == validate_argv(make_train_args(mode, run, lib, assets, repo), mode), "实际 argv 与本轮路径清单不同")
    if mode == "prod":
        require(run == selected["run"], "正式 run 名不是计划确认的名称")
    else:
        require(run not in {item["run"] for item in LIBRARIES.values()}, "验证不能使用正式 run 名")
    run_root = entity_path(store / "train-runs" / CONFIG / run, store, exists=False)
    require(not os.path.lexists(run_root), "输出 run-root 已存在，禁止覆盖")
    require(entity_path(assets / "robomme/norm_stats.json", store).is_file(), "norm_stats 不是文件")
    return run_root, selected


def original_reference(repo=ROOT):
    """静态核对原版条目、默认类与 history；避免当前默认漂移后自我比较仍通过。"""
    repo = Path(repo)
    config_path = "src/mme_vla_suite/training/config.py"
    original = subprocess.check_output(["git", "show", f"{UPSTREAM_HEAD}:{config_path}"], cwd=repo, text=True)

    def extract(source):
        tree = ast.parse(source)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainConfig"]
        entries = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "TrainConfig"
                   and any(item.arg == "name" and isinstance(item.value, ast.Constant) and item.value.value == CONFIG
                           for item in node.keywords)]
        require(len(classes) == len(entries) == 1, "原版配置锚点缺失或重复")
        return [ast.dump(node, include_attributes=False) for node in (*classes, *entries)]

    require(extract(original) == extract((repo / config_path).read_text()), "原版 TrainConfig 默认或 mme_vla_suite 条目已改变")
    history_path = f"src/mme_vla_suite/models/config/robomme/{HISTORY}"
    history = subprocess.check_output(["git", "show", f"{UPSTREAM_HEAD}:{history_path}"], cwd=repo)
    require(history == (repo / history_path).read_bytes(), "原版 history YAML 已改变")
    return {"upstream_head": UPSTREAM_HEAD, "history_sha256": hashlib.sha256(history).hexdigest()}


def parse_in_process(argv, mode):
    """只在 CPU 子进程内调用实际 tyro CLI；完整参考配置只替换批准的路径和步数。"""
    require(os.environ.get("JAX_PLATFORMS") == "cpu", "完整配置解析必须隔离在 CPU 子进程")
    import dataclasses as dc

    from config_record import compare_records
    from config_record import complete_record
    import jax

    from mme_vla_suite.training.config import cli
    from mme_vla_suite.training.config import get_config

    actual_x64 = bool(jax.config.jax_enable_x64)
    require(actual_x64 is False, "jax_enable_x64=True：本计划要求禁用 x64，拒绝静默改变继承环境")
    values = validate_argv(argv, mode)
    previous = sys.argv
    try:
        sys.argv = ["train.py", *argv]
        actual = cli()
    finally:
        sys.argv = previous
    default = get_config(CONFIG)
    expected = dc.replace(default, exp_name=values["--exp-name"],
        assets_base_dir=values["--assets-base-dir"], checkpoint_base_dir=values["--checkpoint-base-dir"],
        dataset_path=values["--dataset-path"], num_train_steps=MODE_STEPS[mode],
        model=dc.replace(default.model, use_history=True, history_config=HISTORY),
        data=dc.replace(default.data, assets=dc.replace(default.data.assets,
            assets_dir=values["--data.assets.assets-dir"], asset_id="robomme")),
        weight_loader=dc.replace(default.weight_loader, params_path=values["--weight-loader.params-path"]))
    reference, complete = complete_record(expected), complete_record(actual)
    compare_records(reference, complete)
    require((actual.batch_size, actual.num_workers, actual.fsdp_devices, actual.seed,
             actual.log_interval, actual.save_interval, actual.keep_period, actual.ema_decay) ==
            (64, 4, 4, 42, 100, 10000, 10000, .999), "实际训练超参偏离原版")
    require(dc.asdict(actual.lr_schedule) == {
        "warmup_steps": 10000, "peak_lr": 5e-5, "decay_steps": 100000, "decay_lr": 5e-5}, "学习率配置不同")
    require(actual.wandb_enabled and actual.project_name == "openpi" and not actual.resume and not actual.overwrite,
            "W&B、项目名或续训开关错误")
    require(complete["history_values"]["budget"] == 512 and complete["history_values"]["token_per_image"] == 16,
            "history 不是 512/4x4")
    return {"complete": complete, "num_train_steps": actual.num_train_steps,
            "checkpoint_dir": str(actual.checkpoint_dir), "jax_enable_x64": actual_x64}


def parse_config(argv, mode, repo=ROOT):
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # 只修改子进程环境，绝不把 CPU 平台设置导出回真实训练。
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_parse", "--mode", mode, "--", *argv],
                            cwd=repo, env=env, text=True, capture_output=True, check=False)
    require(result.returncode == 0, f"CPU 配置解析失败: {result.stdout}\n{result.stderr}")
    rows = [line.removeprefix("ORIG80K_PARSE_JSON=") for line in result.stdout.splitlines()
            if line.startswith("ORIG80K_PARSE_JSON=")]
    require(len(rows) == 1, "CPU 配置记录不唯一")
    return json.loads(rows[0])


def validate_data(lib, selected, norm_path, expected_sha):
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None, "norm_stats 期望必须为完整 SHA256")
    require(sha256_file(norm_path) == expected_sha, "norm_stats SHA256 不符")
    if Path(lib).name == "16task-pub-1600ep":
        require(expected_sha == sha256_file(ROOT / "assets/norm_stats.json"), "完整库必须使用原版 norm_stats")
    manifest_path = Path(lib) / "meta/episode_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canonical = json_sha({key: value for key, value in manifest.items() if key != "sha256"})
    require(manifest.get("sha256") == canonical, "episode manifest 规范摘要不符")
    meta_path = Path(lib) / "framesamp/meta/store_meta.json"
    meta = json.loads(meta_path.read_text())
    require((meta.get("layout"), meta.get("status"), meta.get("manifest_scope"),
             meta.get("num_rows"), meta.get("num_exec_samples"), meta.get("manifest_sha256")) ==
            ("framesamp-4x4-v1", "verified", "full", selected["rows"], selected["samples"], canonical),
            "packed 布局、验证状态、规模或清单绑定不符")
    require(os.environ.get("MMEVLA_FRAMESAMP_SOURCE") == str(Path(lib) / "source"), "source 路径不同源")
    require(os.environ.get("MMEVLA_FRAMESAMP_MANIFEST") == str(manifest_path), "manifest 路径不同源")
    return {"manifest_file_sha256": sha256_file(manifest_path), "store_meta_sha256": sha256_file(meta_path),
            "norm_stats_sha256": expected_sha}


def validate_gpus(gpus, expected):
    require(bool(re.fullmatch(r"\d+(,\d+)*", gpus)), "GPU 列表格式非法")
    ids = [int(item) for item in gpus.split(",")]
    require(len(ids) == len(set(ids)) == 4, "必须提供四张不同 GPU")
    require(gpus == expected, "GPU 分配必须保持 full=0–3、counting=4–7，禁止两组重叠")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == gpus, "CUDA_VISIBLE_DEVICES 与检查分组不符")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = {}
    for line in query.stdout.splitlines():
        index, uuid, used = [part.strip() for part in line.split(",")]
        require(int(index) not in rows, "GPU 查询结果存在重复编号")
        rows[int(index)] = {"index": int(index), "uuid": uuid, "memory_used_mib": int(used)}
    require(all(index in rows and rows[index]["memory_used_mib"] == 0 for index in ids), "指定四卡并非全部空闲")
    require(len({rows[index]["uuid"] for index in ids}) == 4, "可见 GPU UUID 重复")
    return [rows[index] for index in ids]


def validate_disk_budget(mode, repo=ROOT, *, budget_path=None, budget_sha=None):
    """只执行计划既定字节公式；正式预算与来源均由外部已确认记录提供。"""
    repo = Path(repo)
    minimum = 300 * 2**30
    report = {"minimum_bytes": minimum, "required_bytes": minimum}
    if mode == "prod":
        require(bool(budget_path) and bool(budget_sha), "正式运行缺磁盘预算 JSON 或期望 SHA256")
        require(re.fullmatch(r"[0-9a-f]{64}", budget_sha) is not None, "预算期望必须为完整 SHA256")
        path = entity_path(budget_path, repo)
        require(path.is_file() and sha256_file(path) == budget_sha, "磁盘预算 SHA256 不符")
        budget = json.loads(path.read_text())
        names = ("checkpoint_total_bytes", "save_temporary_peak_bytes", "logs_cache_margin_bytes")
        require(all(type(budget.get(name)) is int and budget[name] >= 0 for name in names), "磁盘预算必须明确给出三项非负整数字节数")
        require(budget["checkpoint_total_bytes"] > 0, "正式 checkpoint 总预算必须大于零")
        sources = budget.get("source_perf", {})
        require(set(sources) == {"full", "counting"}, "预算必须绑定两侧 perf 身份")
        source_records = {}
        run_names = set()
        for group, library in (("full", "16task-pub-1600ep"), ("counting", "4task-counting-pub-400ep")):
            source = sources[group]
            run, head = source.get("run_name", ""), source.get("head", "")
            require(re.fullmatch(r"[A-Za-z0-9_-]+", run) is not None
                    and re.fullmatch(r"[0-9a-f]{40}", head) is not None, "perf 来源缺 run 名或完整 HEAD")
            require(run not in run_names, "两侧 perf 来源不能是同一个 run")
            run_names.add(run)
            launch_path = entity_path(repo / "v1-store/bench/orig80k" / run / "launch.json", repo / "v1-store")
            launch = json.loads(launch_path.read_text())
            require((launch.get("mode"), launch.get("run_name"), launch.get("head")) == ("perf", run, head), "预算 perf 来源身份不符")
            actual = launch["actual"]
            require(actual["num_train_steps"] == 300 and actual["complete"]["train_config"]["fields"]["dataset_path"] ==
                    str(repo / "v1-store/datasets" / library / "framesamp"), "预算来源不是对应库的 300 步 perf")
            source_records[group] = {**source, "launch_path": str(launch_path), "launch_sha256": sha256_file(launch_path)}
        report.update(required_bytes=max(minimum, sum(budget[name] for name in names)),
                      budget_path=str(path), budget_sha256=budget_sha,
                      components={name: budget[name] for name in names}, source_perf=source_records)
    info = os.statvfs(repo)
    report["available_bytes"] = info.f_bavail * info.f_frsize
    require(report["available_bytes"] >= report["required_bytes"],
            f"磁盘空间不足: available={report['available_bytes']} required={report['required_bytes']} B")
    return report


def check_launch(args):
    argv = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    values = validate_argv(argv, args.mode)
    require(re.fullmatch(r"[0-9a-f]{40}", args.head) is not None, "TRAIN_HEAD 必须为完整提交字面量")
    require(re.fullmatch(r"[0-9a-f]{64}", args.history_sha) is not None, "history 必须提供完整期望 SHA256")
    require(Path.cwd().resolve() == ROOT, "必须从主副本根启动")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.head, "HEAD 不符")
    require(not subprocess.check_output(["git", "status", "--porcelain"], text=True), "启动要求 clean HEAD")
    run_root, selected = validate_paths(values, args.mode, args.run, args.lib, args.assets)
    reference = original_reference()
    require(reference["history_sha256"] == args.history_sha, "history 期望 SHA 与上游不符")
    require(not os.environ.get("XLA_FLAGS") and not os.environ.get("JAX_PLATFORMS"), "生产环境残留验证平台/确定性设置")
    require(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "0.95", "显存比例必须为 0.95")
    require(os.environ.get("TRAIN_TIMING_STEPS", "0") == "0", "启动前不能开启 profiler")
    require(os.environ.get("WANDB_MODE", "online") == "online", "本轮所有模式必须保持 W&B 在线")
    data = validate_data(args.lib, selected, Path(args.assets) / "robomme/norm_stats.json", args.norm_sha)
    disk = validate_disk_budget(args.mode, budget_path=os.environ.get("ORIG80K_DISK_BUDGET_JSON"),
                                budget_sha=os.environ.get("ORIG80K_DISK_BUDGET_SHA256"))
    parsed = parse_config(argv, args.mode)
    require(parsed.get("jax_enable_x64") is False, "CPU 解析缺少 x64 禁用证据")
    require(parsed["checkpoint_dir"] == str(run_root), "真实配置推导的输出根不同")
    sys.path.insert(0, str(ROOT / "scripts/assets"))
    import assets_lock
    assets_lock.require(["pi05_base", "paligemma_tokenizer"], level="full")
    gpus = validate_gpus(args.gpus, selected["gpus"])
    records = entity_path(args.records, ROOT / "v1-store", exists=False)
    require(records == ROOT / "v1-store/bench/orig80k" / args.run, "记录目录不是本 run 的独立目录")
    require(not os.path.lexists(records), "记录目录已存在，禁止覆盖")
    require(os.environ.get("TRAIN_RECORD_DIR") == str(records)
            and os.environ.get("TRAIN_FINAL_RECORD_DIR") == str(records / "final"), "指标或末步记录未开启")
    environment_keys = ("CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "JAX_PLATFORMS", "JAX_ENABLE_X64",
        "JAX_DEFAULT_MATMUL_PRECISION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE",
        "OPENPI_DATA_HOME", "UV_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME", "MMEVLA_JAX_CACHE_DIR",
        "CUDA_CACHE_PATH", "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR")
    result = {"mode": args.mode, "run_name": args.run, "head": args.head, "argv": argv,
        "checkpoint_dir": str(run_root), "actual": parsed, "data": data, "reference": reference, "gpus": gpus, "disk": disk,
        "environment": {key: os.environ.get(key) for key in environment_keys},
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "jax", "jaxlib", "numpy", "ml_dtypes", "flax", "optax")},
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"), "sys_prefix": sys.prefix,
        "storage": subprocess.check_output(["findmnt", "-T", str(ROOT), "-no", "SOURCE,FSTYPE,TARGET"], text=True).strip()}
    # 重查耗时资产校验期间的代码状态；通过前不创建训练记录目录。
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.head
            and not subprocess.check_output(["git", "status", "--porcelain"], text=True), "preflight 期间代码状态改变")
    records.mkdir(parents=True, exist_ok=False)
    with (records / "launch.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"ORIG80K_CONTRACT=PASS mode={args.mode} steps={MODE_STEPS[args.mode]} gpus={args.gpus}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    parse = sub.add_parser("_parse")
    parse.add_argument("--mode", choices=MODE_STEPS, required=True)
    parse.add_argument("train_args", nargs=argparse.REMAINDER)
    check = sub.add_parser("check")
    check.add_argument("--mode", choices=MODE_STEPS, required=True)
    for name in ("run", "head", "history-sha", "norm-sha", "lib", "assets", "gpus", "records"):
        check.add_argument("--" + name, required=True)
    check.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command == "_parse":
            argv = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
            print("ORIG80K_PARSE_JSON=" + json.dumps(parse_in_process(argv, args.mode), sort_keys=True, allow_nan=False))
        else:
            check_launch(args)
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"ORIG80K_CONTRACT=FAIL reason={error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
