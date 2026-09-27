"""公开全集与 counting 原版四卡训练的参数、路径和启动状态契约。"""
# 中文文案按仓库语言规则保留正常标点。
# ruff: noqa: RUF001, RUF002, RUF003

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "mme_vla_suite"
HISTORY = "perceptual-framesamp-modul.yaml"
UPSTREAM_HEAD = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
MODE_STEPS = {"prod": 80000, "perf": 300, "smoke": 20}
TIMING_EQ_DETERMINISTIC_FLAGS = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
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
DISK_BUDGET_SCHEMA = 2
MEASUREMENT_SCHEMA = 1
MARGIN_KEYS = ("checkpoint_growth_margin_bytes", "save_sampling_margin_bytes", "logs_cache_margin_bytes")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_timing_eq_profile_record(profile, *, mode, timing, xla_flags):
    """仅核验记录内必填的闭集档位，不使用判定进程当前环境。"""
    require(mode in MODE_STEPS, "计时对拍档位的模式必须为 prod/perf/smoke")
    require(timing is None or (mode == "smoke" and timing in ("off", "on")),
            "计时对拍档位仅允许smoke的显式off/on")
    require(xla_flags is None or isinstance(xla_flags, str), "记录XLA_FLAGS必须为空或字符串")
    require(isinstance(profile, dict) and set(profile) == {"name", "xla_flags"}, "计时对拍档位字段不完整或存在额外字段")
    name = profile["name"]
    require(name in ("normal", "deterministic100"), "未知计时对拍档位")
    expected = TIMING_EQ_DETERMINISTIC_FLAGS if name == "deterministic100" else ""
    require(profile["xla_flags"] == expected, "计时对拍档位声明的XLA_FLAGS不同")
    if name == "deterministic100":
        require(mode == "smoke" and timing in ("off", "on"), "deterministic100仅允许显式smoke20 off/on")
    require((xla_flags or "") == expected, "实际XLA_FLAGS与计时对拍档位不同")
    return {"name": name, "xla_flags": expected}


def resolve_timing_eq_profile(mode, environ=None):
    """读取显式档位并核验进程实际flags；不设置环境，也不宽免训练配置。"""
    env = os.environ if environ is None else environ
    timing = env.get("ORIG80K_SMOKE_EQ_MODE")
    if "ORIG80K_SMOKE_EQ_MODE" in env:
        require(mode == "smoke" and timing in ("off", "on"), "ORIG80K_SMOKE_EQ_MODE仅允许smoke的off/on")
    name = "normal"
    if "ORIG80K_TIMING_EQ_PROFILE" in env:
        require(env["ORIG80K_TIMING_EQ_PROFILE"] == "deterministic100",
                "ORIG80K_TIMING_EQ_PROFILE仅允许显式deterministic100；空值或其他值均拒绝")
        name = "deterministic100"
    profile = {"name": name, "xla_flags": TIMING_EQ_DETERMINISTIC_FLAGS if name == "deterministic100" else ""}
    return validate_timing_eq_profile_record(profile, mode=mode, timing=timing, xla_flags=env.get("XLA_FLAGS"))


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


def file_reference(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def bound_file(reference, repo):
    require(isinstance(reference, dict) and isinstance(reference.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]), "预算证据缺完整文件SHA256")
    path = entity_path(reference["path"], Path(repo))
    require(path.is_file() and sha256_file(path) == reference["sha256"], f"预算证据SHA256不符: {path}")
    if "bytes" in reference:
        require(type(reference["bytes"]) is int and reference["bytes"] == path.stat().st_size, "预算证据字节数不符")
    return path


def _disk_helpers():
    # 仅复用只读取证/验收函数；这些模块不会在import时运行模型或采样。
    sys.path.insert(0, str(ROOT / "scripts/training/tests"))
    from check_orig80k_completion import validate_gpu_sampling
    from check_orig80k_completion import validate_records
    from check_orig80k_speed import summarize_disk

    return validate_records, validate_gpu_sampling, summarize_disk


def checkpoint_stat(root):
    """完成后的稳定目录账目；记录原始stat并按inode去重，不读取权重内容。"""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and root == root.resolve(), "checkpoint必须为已存在的实体目录")
    entries = []

    def walk(path):
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), f"checkpoint存在软链或非普通项: {path}")
        entries.append({"path": "." if path == root else str(path.relative_to(root)),
                        "kind": "directory" if stat.S_ISDIR(info.st_mode) else "file",
                        "device": info.st_dev, "inode": info.st_ino, "size": info.st_size,
                        "blocks": info.st_blocks, "mtime_ns": info.st_mtime_ns})
        if stat.S_ISDIR(info.st_mode):
            for child in sorted(path.iterdir()):
                walk(child)

    walk(root)
    return stat_receipt(entries)


def stat_receipt(entries):
    require(isinstance(entries, list) and entries, "checkpoint测量账目为空")
    seen_paths, seen_inodes = set(), {}
    allocated = logical = 0
    for row in entries:
        path = Path(row["path"])
        require(not path.is_absolute() and ".." not in path.parts and row["path"] not in seen_paths,
                "checkpoint测量路径越界或重复")
        require(row["kind"] in ("directory", "file"), "checkpoint测量项类型非法")
        require(all(type(row.get(key)) is int and row[key] >= 0
                    for key in ("device", "inode", "size", "blocks", "mtime_ns")), "checkpoint测量stat非法")
        seen_paths.add(row["path"])
        identity = (row["device"], row["inode"])
        inode_value = (row["kind"], row["size"], row["blocks"], row["mtime_ns"])
        if identity in seen_inodes:
            require(seen_inodes[identity] == inode_value, "同inode测量数值不一致")
        else:
            seen_inodes[identity] = inode_value
            allocated += row["blocks"] * 512
            logical += row["size"]
    require("." in seen_paths and entries[0]["path"] == "." and entries[0]["kind"] == "directory", "测量缺目录根")
    return {"entries": entries, "entries_sha256": json_sha(entries), "allocated_bytes": allocated,
            "logical_bytes": logical, "unique_inodes": len(seen_inodes)}


def _perf_context(source, group, repo):
    run, head = source.get("run_name", ""), source.get("head", "")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", run) is not None
            and re.fullmatch(r"[0-9a-f]{40}", head) is not None, "perf来源缺run或完整HEAD")
    library = {"full": "16task-pub-1600ep", "counting": "4task-counting-pub-400ep"}[group]
    records = Path(repo) / "v1-store/bench/orig80k" / run
    run_root = Path(repo) / "v1-store/train-runs" / CONFIG / run
    references = {key: source[key] for key in ("launch", "report", "completion", "disk_samples")}
    paths = {key: bound_file(value, repo) for key, value in references.items()}
    references = {key: file_reference(path) for key, path in paths.items()}
    require(paths["launch"] == records / "launch.json" and paths["disk_samples"] == records / "disk_samples.jsonl",
            "perf启动或磁盘记录路径不属于本run")
    launch, report, completion = (json.loads(paths[key].read_text()) for key in ("launch", "report", "completion"))
    require((launch.get("mode"), launch.get("run_name"), launch.get("head")) == ("perf", run, head), "perf启动身份不符")
    environment = launch.get("environment")
    require(isinstance(environment, dict)
            and {"XLA_FLAGS", "ORIG80K_SMOKE_EQ_MODE", "ORIG80K_TIMING_EQ_PROFILE"} <= environment.keys(),
            "perf来源缺完整档位环境字段")
    validate_timing_eq_profile_record(launch.get("timing_eq_profile"), mode="perf",
        timing=environment["ORIG80K_SMOKE_EQ_MODE"], xla_flags=environment["XLA_FLAGS"])
    require(environment["ORIG80K_TIMING_EQ_PROFILE"] is None, "perf来源不能选择确定性对拍档")
    validate_argv(launch["argv"], "perf")
    actual = launch["actual"]
    fields = actual["complete"]["train_config"]["fields"]
    require(actual["num_train_steps"] == 300 and fields["dataset_path"] == str(Path(repo) / "v1-store/datasets" / library / "framesamp")
            and launch["checkpoint_dir"] == str(run_root), "perf步数、库或输出根不符")
    require(actual.get("jax_enable_x64") is False, "perf启动记录缺x64禁用证据")
    require((completion.get("status"), completion.get("mode"), completion.get("run_name"), completion.get("head"),
             completion.get("state_step"), completion.get("final"), completion.get("checkpoints")) ==
            ("PASS", "perf", run, head, 300, 299, [299]), "预算必须来自真实300/299保存恢复PASS")
    sections = [report] if "disk" in report else [report.get("full_or_first", {}), report.get("counting_or_second", {})]
    sections = [section for section in sections if section.get("disk", {}).get("checkpoint_root") == str(run_root)]
    require(len(sections) == 1, "perf报告没有唯一匹配的run磁盘段")
    report = sections[0]
    require((report.get("head"), report.get("batch_size"), report.get("workers"), report.get("steady_steps")) ==
            (head, 64, 4, [100, 299]), "perf报告版本或档位不符")
    disk = report["disk"]
    require(disk["samples_sha256"] == references["disk_samples"]["sha256"]
            and disk["run_uuid"] == completion["run_uuid"], "perf采样SHA或恢复UUID不符")
    require(disk["sampling_interval_s"] == .5 and disk["save_samples"] > 0, "perf缺0.5秒真实保存窗口观测")
    for key in ("final_allocated_bytes", "sampled_max_allocated_bytes"):
        require(type(disk.get(key)) is int and disk[key] > 0, "perf分配字节数无效")
    start, final, wait = (json.loads((records / "final" / name).read_text())
                          for name in ("start.json", "final.json", "checkpoint_wait_done.json"))
    require(all(row.get("checkpoint_dir") == str(run_root) for row in (start, final)), "实际保存记录输出根不同")
    metrics = [json.loads(line) for line in (records / "metrics.jsonl").read_text().splitlines()]
    log = Path(repo) / "v1-store/logs" / f"{run}.driver.log"
    lines = log.read_text().splitlines()
    exits = [line for line in lines if line.startswith("EXIT_CODE=")]
    for prefix in ("TRAIN_PIPE_EXIT=", "TEE_EXIT="):
        require([line for line in lines if line.startswith(prefix)] == [prefix + "0"], "训练或tee未成功收尾")
    validate_records, validate_gpu_sampling, _ = _disk_helpers()
    validate_records(start, final, wait, metrics, exits, [299], mode="perf", run=run, head=head)
    require(start["complete"] == launch["actual"]["complete"]
            and completion["run_uuid"] == start["run_uuid"]
            and completion["restored_leaves"] == final["ema_leaves"], "实际恢复与末步EMA或启动配置不一致")
    runtime = json.loads((records / "runtime.json").read_text())
    require((runtime.get("device_count"), runtime.get("fsdp_devices"), runtime.get("batch_size"), runtime.get("num_workers")) ==
            (4, 4, 64, 4), "真实设备或loader形制不同")
    require(runtime.get("exp_name") == run and runtime.get("config_name") == CONFIG and runtime.get("seed") == 42,
            "真实runtime身份或种子不同")
    require(runtime.get("cuda_visible_devices") == launch["environment"]["CUDA_VISIBLE_DEVICES"] == LIBRARIES[library]["gpus"],
            "真实GPU分组与启动记录不同")
    require(completion["gpu_sampling"] == validate_gpu_sampling(records, lines,
            [int(value) for value in runtime["cuda_visible_devices"].split(",")], start, wait),
            "完成器GPU采样摘要不能从原始记录重建")
    speed_path = records / "speed_run.json"
    speed = json.loads(speed_path.read_text())
    require(speed.get("schema") == 2, "perf来源缺显式档位版本记录")
    validate_timing_eq_profile_record(speed.get("timing_eq_profile"), mode="perf", timing=None, xla_flags=speed["xla_flags"])
    require(speed.get("success") and speed.get("mode") == "perf" and speed.get("steps") == 300
            and speed.get("head") == head and speed.get("entry_calls") == 1 and speed.get("sampler_stopped")
            and not speed.get("sampling_error") and speed.get("profiler") is False
            and speed.get("disk_sampling", {}).get("sha256") == disk["samples_sha256"],
            "perf未真实完成或磁盘记录未绑定")
    require(len(speed["save_calls"]) == 1 and speed["save_calls"][0]["step"] == 299
            and speed["save_calls"][0]["state_step"] == 300, "perf真实保存次数或末步计数不同")
    require(disk["wait_record_sha256"] == sha256_file(records / "final/checkpoint_wait_done.json"), "保存等待记录SHA不符")
    # 稳定小记录须全部留存；权重目录的两项小元数据由测量回执保留原文，允许事后清理权重。
    completion_files = {}
    for item in completion["files"]:
        path = Path(item["path"])
        require(path.is_absolute() and path == path.resolve() and path.is_relative_to(Path(repo)), "完成证据路径越界")
        require(str(path) not in completion_files, "完成证据路径重复")
        completion_files[str(path)] = item
        if not path.is_relative_to(run_root):
            bound_file(item, repo)
    needed = [log, *[records / name for name in ("launch.json", "runtime.json", "metrics.jsonl", "gpu.csv", "gpu.csv.err")],
              *[records / "final" / name
              for name in ("start.json", "final.json", "checkpoint_wait_done.json")],
              run_root / "299/_CHECKPOINT_METADATA", run_root / "299/params/_METADATA"]
    require(all(str(path) in completion_files for path in needed), "完成证据文件链不完整")
    references["speed_run"] = file_reference(speed_path)
    return {"run": run, "head": head, "records": records, "run_root": run_root, "disk": disk,
            "speed": speed, "completion": completion, "completion_files": completion_files, "references": references}


def measure_checkpoint(args, repo=ROOT):
    """只在perf完成恢复后测量真实299目录；不训练、不恢复、不修改checkpoint。"""
    repo = Path(repo)
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.run) is not None, "perf run名称非法")
    records = entity_path(repo / "v1-store/bench/orig80k" / args.run, repo)
    report_path = entity_path(args.report, repo)
    completion_path = entity_path(args.completion, records)
    source = {"run_name": args.run, "head": args.head, "launch": file_reference(records / "launch.json"),
              "report": file_reference(report_path), "completion": file_reference(completion_path),
              "disk_samples": file_reference(records / "disk_samples.jsonl")}
    ctx = _perf_context(source, args.group, repo)
    output = entity_path(args.out, records, exists=False)
    require(not os.path.lexists(output), "测量回执输出已存在")
    checkpoint = ctx["run_root"] / "299"
    require(sorted(path.name for path in ctx["run_root"].iterdir() if path.name.isdecimal()) == ["299"]
            and not any("orbax-checkpoint-tmp" in path.name for path in ctx["run_root"].iterdir()), "perf目录不是完整且唯一的末步299")
    require(all((checkpoint / suffix).is_file() for suffix in
                ("_CHECKPOINT_METADATA", "params/_METADATA", "params/manifest.ocdbt", "assets/robomme/norm_stats.json")),
            "真实checkpoint缺参数索引或保存资产")
    _, _, summarize_disk = _disk_helpers()
    require(summarize_disk(records, ctx["speed"]) == ctx["disk"], "perf磁盘报告不能从原始采样重建")
    snapshots = {}
    for suffix in ("_CHECKPOINT_METADATA", "params/_METADATA"):
        path = checkpoint / suffix
        reference = ctx["completion_files"][str(path)]
        bound_file(reference, repo)
        snapshots[suffix] = {"text": path.read_bytes().decode("utf-8"), "sha256": reference["sha256"]}
    require(snapshots["_CHECKPOINT_METADATA"]["sha256"] == ctx["disk"]["checkpoint_metadata_sha256"], "原生保存元数据SHA不同")
    first, run_first = checkpoint_stat(checkpoint), checkpoint_stat(ctx["run_root"])
    require(first == checkpoint_stat(checkpoint) and run_first == checkpoint_stat(ctx["run_root"]), "测量时checkpoint仍在变化")
    require(run_first["allocated_bytes"] == ctx["disk"]["final_allocated_bytes"], "稳定stat与最终run根采样不一致")
    receipt = {"schema": MEASUREMENT_SCHEMA, "group": args.group, "run_name": args.run, "head": args.head,
               "run_uuid": ctx["completion"]["run_uuid"], "checkpoint_step": 299,
               "checkpoint_path": str(checkpoint), "checkpoint": first, "run_tree": run_first,
               "evidence": ctx["references"], "disk": ctx["disk"], "metadata_snapshots": snapshots,
               "measured_at": datetime.datetime.now(datetime.UTC).isoformat(),
               "measurement_tool_sha256": sha256_file(__file__)}
    # 测量期间所有上游小记录也必须保持原来的字节。
    for reference in ctx["references"].values():
        bound_file(reference, repo)
    with output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"CHECKPOINT_MEASURED=PASS run={args.run} step=299 allocated_bytes={first['allocated_bytes']} receipt_sha256={sha256_file(output)}", flush=True)
    return receipt


def _recorded_disk_summary(ctx, metadata_text):
    """按summarize_disk同口径重算；原生元数据来自快照，不重建已清理的权重目录。"""
    root, meta = ctx["records"], ctx["speed"]
    contract = meta.get("disk_sampling", {})
    require(contract.get("schema") == 1 and contract.get("interval_s") == .5
            and contract.get("stopped") and not contract.get("error"), "磁盘采样未正常完成")
    require(contract.get("samples_file") == "disk_samples.jsonl" and contract.get("scratch_path") == "/scratch",
            "磁盘采样位置或范围错误")
    path = Path(root) / "disk_samples.jsonl"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == contract.get("sha256"), "磁盘采样记录摘要不符")
    data = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(data) == contract.get("samples") and len(data) >= 3
            and [row["sequence"] for row in data] == list(range(len(data))), "磁盘采样缺失或乱序")
    require(data[0]["kind"] == "start" and data[-1]["kind"] == "final"
            and all(row["kind"] == "periodic" for row in data[1:-1]), "磁盘起止采样缺失")
    require(data[0]["wall_end"] <= meta["entry_start_wall"]
            and data[-1]["wall_start"] >= meta["end_wall"], "磁盘采样未覆盖真实入口及异步保存完成")
    for row in data:
        checkpoint = row["checkpoint"]
        require(not row["errors"] and not checkpoint["errors"] and not checkpoint["symlinks"],
                "磁盘采样存在I/O错误或软链，不能据其生成预算")
        require(type(checkpoint["allocated_bytes"]) is int and checkpoint["allocated_bytes"] >= 0
                and type(row["scratch_available_bytes"]) is int and row["scratch_available_bytes"] >= 0,
                "磁盘字节数无效")
        require(all(math.isfinite(row[key]) and row[key] >= 0 for key in ("duration_s", "lateness_s"))
                and row["wall_end"] >= row["wall_start"], "磁盘采样时钟或延迟无效")
    intervals = [b["monotonic_start"] - a["monotonic_start"] for a, b in itertools.pairwise(data)]
    require(all(value > 0 and math.isfinite(value) for value in intervals), "磁盘采样单调时钟无效")
    require(all(row["interval_s"] == value for row, value in zip(data[1:], intervals, strict=True)),
            "磁盘采样实际间隔记录不符")
    final = data[-1]["checkpoint"]
    require(final["exists"] and not final["disappeared"] and not final["changed"], "最终checkpoint采样仍不完整")
    save = meta["save_calls"][0]
    require(save["checkpoint_root"] == contract["checkpoint_root"], "磁盘采样不是实际保存目录")
    # wait返回后入口还会收尾W&B；只以原生checkpoint提交时刻定义保存窗口。
    checkpoint_bytes = metadata_text.encode()
    checkpoint_meta = json.loads(checkpoint_bytes)
    init_ns = checkpoint_meta["init_timestamp_nsecs"]
    commit_ns = checkpoint_meta["commit_timestamp_nsecs"]
    require(type(init_ns) is int and type(commit_ns) is int and 0 < init_ns < commit_ns,
            "checkpoint原生提交时间无效")
    start_record = json.loads((Path(root) / "final/start.json").read_text())
    wait_bytes = (Path(root) / "final/checkpoint_wait_done.json").read_bytes()
    wait_record = json.loads(wait_bytes)
    require(start_record["run_uuid"] == wait_record["run_uuid"]
            and start_record["head"] == wait_record["head"] == meta["head"]
            and start_record["checkpoint_dir"] == contract["checkpoint_root"]
            and wait_record.get("wait_until_finished") is True, "磁盘保存窗口未绑定本run的wait完成证据")
    completed = datetime.datetime.fromisoformat(wait_record["completed_at"])
    require(completed.tzinfo is not None, "checkpoint wait完成时间缺时区")
    wait_done = completed.timestamp()
    commit_wall = commit_ns / 1_000_000_000
    require(save["start_wall"] <= init_ns / 1_000_000_000 < commit_wall <= wait_done <= meta["end_wall"],
            "checkpoint提交、wait或入口结束时间顺序错误")
    during_save = [row for row in data if row["kind"] == "periodic"
                   and save["start_wall"] <= row["wall_start"] <= row["wall_end"] <= commit_wall]
    require(during_save, "缺保存期间磁盘采样，不能用终态大小替代峰值")
    return {"checkpoint_root": contract["checkpoint_root"], "scratch_path": contract["scratch_path"],
            "sampling_interval_s": .5, "samples": len(data), "save_samples": len(during_save),
            "save_window_wall": [save["start_wall"], commit_wall], "save_wait_done_wall": wait_done,
            "checkpoint_init_timestamp_nsecs": init_ns, "checkpoint_commit_timestamp_nsecs": commit_ns,
            "checkpoint_save_commit_seconds": (commit_ns - init_ns) / 1_000_000_000,
            "checkpoint_metadata_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "wait_record_sha256": hashlib.sha256(wait_bytes).hexdigest(), "run_uuid": start_record["run_uuid"],
            "start_allocated_bytes": data[0]["checkpoint"]["allocated_bytes"],
            "final_allocated_bytes": final["allocated_bytes"],
            "start_scratch_available_bytes": data[0]["scratch_available_bytes"],
            "final_scratch_available_bytes": data[-1]["scratch_available_bytes"],
            "sampled_max_allocated_bytes": max(row["checkpoint"]["allocated_bytes"] for row in data),
            "save_sampled_max_allocated_bytes": max(row["checkpoint"]["allocated_bytes"] for row in during_save),
            "scratch_available_min_bytes": min(row["scratch_available_bytes"] for row in data),
            "mean_interval_s": statistics.fmean(intervals), "max_interval_s": max(intervals),
            "max_sampling_duration_s": max(row["duration_s"] for row in data),
            "max_lateness_s": max(row["lateness_s"] for row in data), "missed_ticks": contract["missed_ticks"],
            "samples_with_path_races": sum(bool(row["checkpoint"]["disappeared"] or row["checkpoint"]["changed"])
                                           for row in data),
            "duplicate_inodes_observed": sum(row["checkpoint"]["duplicate_inodes"] for row in data),
            "samples_sha256": contract["sha256"], "note": contract["note"]}


def _validated_measurement(source, group, repo):
    ctx = _perf_context(source, group, repo)
    receipt = json.loads(bound_file(source["measurement"], repo).read_text())
    require((receipt.get("schema"), receipt.get("group"), receipt.get("run_name"), receipt.get("head"),
             receipt.get("run_uuid"), receipt.get("checkpoint_step"), receipt.get("checkpoint_path")) ==
            (MEASUREMENT_SCHEMA, group, ctx["run"], ctx["head"], ctx["completion"]["run_uuid"], 299,
             str(ctx["run_root"] / "299")), "测量回执身份或步骤不符")
    require(receipt["evidence"] == ctx["references"] and receipt["disk"] == ctx["disk"], "测量回执与perf证据链不同")
    for key in ("checkpoint", "run_tree"):
        require(receipt[key] == stat_receipt(receipt[key]["entries"]), "测量账目与分配量不一致")
    projected = [{**row, "path": "." if row["path"] == "299" else row["path"][4:]}
                 for row in receipt["run_tree"]["entries"]
                 if row["path"] == "299" or row["path"].startswith("299/")]
    require(receipt["checkpoint"] == stat_receipt(projected), "单份checkpoint账目不是run树的真实299子树")
    require(receipt["checkpoint"]["allocated_bytes"] > 0
            and receipt["run_tree"]["allocated_bytes"] == ctx["disk"]["final_allocated_bytes"], "测量尺寸与最终采样不同")
    for suffix in ("_CHECKPOINT_METADATA", "params/_METADATA"):
        row = receipt["metadata_snapshots"][suffix]
        require(hashlib.sha256(row["text"].encode()).hexdigest() == row["sha256"]
                == ctx["completion_files"][str(ctx["run_root"] / "299" / suffix)]["sha256"], "测量保留的原生元数据不符")
    require(receipt["metadata_snapshots"]["_CHECKPOINT_METADATA"]["sha256"] == ctx["disk"]["checkpoint_metadata_sha256"],
            "采样与原生保存元数据不一致")
    native = json.loads(receipt["metadata_snapshots"]["_CHECKPOINT_METADATA"]["text"])
    require(native["init_timestamp_nsecs"] == ctx["disk"]["checkpoint_init_timestamp_nsecs"]
            and native["commit_timestamp_nsecs"] == ctx["disk"]["checkpoint_commit_timestamp_nsecs"],
            "报告的原生保存时刻与测量快照不符")
    require(isinstance(receipt.get("measurement_tool_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", receipt["measurement_tool_sha256"]), "测量回执缺工具指纹")
    require(_recorded_disk_summary(ctx, receipt["metadata_snapshots"]["_CHECKPOINT_METADATA"]["text"]) == ctx["disk"],
            "预算磁盘摘要不能从原始采样与原生保存快照重建")
    return ctx, receipt


def validate_disk_budget(mode, repo=ROOT, *, budget_path=None, budget_sha=None):
    """正式预算绑定已完成perf、真实测量与显式余量；观测增量不是连续峰值。"""
    repo = Path(repo)
    minimum = 300 * 2**30
    report = {"minimum_bytes": minimum, "required_bytes": minimum}
    if mode == "prod":
        require(bool(budget_path) and bool(budget_sha), "正式运行缺磁盘预算 JSON 或期望 SHA256")
        path = bound_file({"path": str(budget_path), "sha256": budget_sha}, repo)
        budget = json.loads(path.read_text())
        require(budget.get("schema") == DISK_BUDGET_SCHEMA, "预算需要完整实测证据schema2，旧版launch不足")
        sources = budget.get("source_perf", {})
        require(set(sources) == {"full", "counting"} and sources["full"].get("run_name") != sources["counting"].get("run_name"),
                "预算必须绑定两侧不同perf")
        require(sources["full"].get("head") == sources["counting"].get("head"), "两侧perf必须使用相同代码版本")
        margins = json.loads(bound_file(budget["margin_source"], repo).read_text())
        require(margins.get("schema") == 1 and all(type(margins.get(key)) is int and margins[key] >= 0 for key in MARGIN_KEYS),
                "余量来源必须明确给出全部非负整数字节")
        require(margins["save_sampling_margin_bytes"] > 0 and margins["logs_cache_margin_bytes"] > 0,
                "采样盲区及日志缓存保守余量不能省略或自动取零")
        require(isinstance(margins.get("basis"), dict)
                and all(isinstance(margins["basis"].get(key), str) and margins["basis"][key].strip()
                    for key in MARGIN_KEYS), "余量来源缺逐项依据")
        retained = observed_extra = 0
        source_records = {}
        for group in ("full", "counting"):
            ctx, receipt = _validated_measurement(sources[group], group, repo)
            size = receipt["checkpoint"]["allocated_bytes"]
            extra = max(0, ctx["disk"]["sampled_max_allocated_bytes"] - ctx["disk"]["final_allocated_bytes"])
            retained += 8 * size
            observed_extra += extra
            source_records[group] = {"run_name": ctx["run"], "head": ctx["head"], "checkpoints": 8,
                                     "per_checkpoint_allocated_bytes": size, "retained_bytes": 8 * size,
                                     "observed_extra_lower_bound_bytes": extra, "evidence": sources[group]}
        components = {"checkpoint_total_bytes": retained + margins["checkpoint_growth_margin_bytes"],
                      "save_temporary_peak_bytes": observed_extra + margins["save_sampling_margin_bytes"],
                      "logs_cache_margin_bytes": margins["logs_cache_margin_bytes"]}
        require(all(type(budget.get(key)) is int and budget[key] == value for key, value in components.items()),
                "预算总额与两侧各8份实测保留量/观测增量/明确余量不符")
        report.update(required_bytes=max(minimum, sum(components.values())), budget_path=str(path), budget_sha256=budget_sha,
                      components=components, margin_source=budget["margin_source"], source_perf=source_records,
                      note="0.5秒离散非原子采样；观测增量仅为下界，保守余量由明确来源提供，不设默认倍率")
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
    timing_eq_profile = resolve_timing_eq_profile(args.mode)
    require(not os.environ.get("JAX_PLATFORMS"), "训练环境残留验证平台设置")
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
    environment_keys = ("CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "ORIG80K_TIMING_EQ_PROFILE", "ORIG80K_SMOKE_EQ_MODE",
        "JAX_PLATFORMS", "JAX_ENABLE_X64",
        "JAX_DEFAULT_MATMUL_PRECISION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE",
        "OPENPI_DATA_HOME", "UV_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME", "MMEVLA_JAX_CACHE_DIR",
        "CUDA_CACHE_PATH", "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR")
    result = {"mode": args.mode, "run_name": args.run, "head": args.head, "argv": argv,
        "timing_eq_profile": timing_eq_profile,
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
    measure = sub.add_parser("measure-checkpoint")
    measure.add_argument("--group", choices=("full", "counting"), required=True)
    for name in ("run", "head", "report", "completion", "out"):
        measure.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        if args.command == "_parse":
            argv = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
            print("ORIG80K_PARSE_JSON=" + json.dumps(parse_in_process(argv, args.mode), sort_keys=True, allow_nan=False))
        elif args.command == "measure-checkpoint":
            measure_checkpoint(args)
        else:
            check_launch(args)
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"ORIG80K_CONTRACT=FAIL reason={error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
