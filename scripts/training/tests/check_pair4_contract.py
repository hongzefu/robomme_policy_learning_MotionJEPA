"""四对 motion 的训练合同检查入口；不调用在线策略或评估入口。"""
# 检查必须直接覆盖原装配私有边界，不能另写一个相同算法自证。
# ruff: noqa: SLF001
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BASELINE = "2b2954cf7d04f10f1b0826a3e72a0324bdda08d2"
HISTORY_CONFIG = "perceptual-framesamp-modul-pair4-epoch6-original-smoke.yaml"
ENCODER_SHA = "146efd425f7923428d630dfb7ee2a8f010f19b207d36519dba49454aaa373d30"
ORIGINAL_FILES = (
    "src/mme_vla_suite/models/representation/percep_mem.py",
    "src/mme_vla_suite/models/integration/history_pi0.py",
    "src/mme_vla_suite/models/integration/history_gemma.py",
    "src/mme_vla_suite/models/integration/history_observation.py",
    "src/mme_vla_suite/training/config.py",
    "src/mme_vla_suite/policies/robomme_policy.py",
)
PROTECTED_FILES = (
    "src/mme_vla_suite/policies/motion_protocol.py",
    "src/mme_vla_suite/policies/motion_client.py",
    "src/mme_vla_suite/policies/framesamp_memory.py",
    "src/mme_vla_suite/policies/policy.py",
    "src/mme_vla_suite/policies/policy_config.py",
    "scripts/dataset/wan/motion_sidecar.py",
    "scripts/assets/ASSETS_LOCK.json",
    "scripts/training/train.py",
    "pyproject.toml", "uv.lock",
)
REQUIRED = {
    "cpu": ("PAIRING", "RAW_FORMAT_CPU", "WINDOW_BUDGET", "MASK_ORDER", "SOURCE_REJECTION", "ORIGINAL_IMPL_UNCHANGED", "SCOPE"),
    "prepare-fixture": (),
    "source-smoke": ("SOURCE_FRESH", "REGISTERED_SOURCE"),
    "finalize-fixture": ("RAW_CONTRACT",),
    "model": ("REAL_FORWARD", "SAVE_LOAD"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gate(details):
    return {"status": "PASS", "details": details}


def rejected(function, label):
    try:
        function()
    except (ValueError, TypeError, RuntimeError, FileNotFoundError) as error:
        return {"case": label, "error": type(error).__name__, "reason": str(error)}
    raise ValueError(f"应拒绝的输入被接受：{label}")


def code_identity():
    paths = subprocess.check_output(
        ["git", "ls-files", "src", "scripts/dataset", "scripts/training", "pyproject.toml", "uv.lock"], cwd=REPO, text=True).splitlines()
    # 新入口在提交前也纳入身份，防止开发态报告漏记自己。
    paths = sorted(set(paths) | {str(p.relative_to(REPO)) for p in Path(__file__).parent.glob("*pair4*.py")})
    digest = hashlib.sha256()
    for name in paths:
        path = REPO / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + sha(path).encode() + b"\n")
    return digest.hexdigest()


def scope_checks():
    hashes = {}
    for name in (*ORIGINAL_FILES, *PROTECTED_FILES):
        original = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=REPO)
        require((REPO / name).read_bytes() == original, f"原实现发生变化：{name}")
        hashes[name] = hashlib.sha256(original).hexdigest()
    changed = subprocess.check_output(["git", "diff", "--name-only", BASELINE, "--"], cwd=REPO, text=True).splitlines()
    changed += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO, text=True).splitlines()
    prohibited = ("src/mme_vla_suite/policies/", "src/openpi/policies/", "challenge_interface/",
                  "scripts/eval/", "scripts/training/legacy-eval/", "scripts/training/eval.sh",
                  "third_party/robomme_benchmark")
    bad = [p for p in changed if p.startswith(prohibited)]
    require(not bad, f"在线或评估路径发生变化：{bad}")
    return {"ORIGINAL_IMPL_UNCHANGED": gate({"baseline": BASELINE, "sha256": {name: hashes[name] for name in ORIGINAL_FILES}}),
            "SCOPE": gate({"baseline": BASELINE, "protected_sha256": {name: hashes[name] for name in PROTECTED_FILES},
                           "forbidden_prefixes": prohibited, "changed_paths": sorted(set(changed))})}


def _synthetic_raw_store(directory):
    """用明确标为合成的来源走正常 pack/verify；不作为真实提取证据。"""
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.datastore.manifest import manifest_sha256
    module_path = REPO / "scripts/dataset/pack_motion_store.py"
    spec = importlib.util.spec_from_file_location("pair4_pack_contract", module_path)
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    d = Path(directory)
    for name in ("meta", "raw", "tokens", "latents", "encoder-synthetic", "oracle"):
        (d / name).mkdir()
    raw = np.arange(2048, dtype=np.float32).reshape(1, 8, 256)
    manifest = {"version": 1, "raw_dir": str(d / "raw"), "canonical_order": ["record_dataset_Synthetic.h5"],
                "num_shards": 1, "shard_load_timesteps": [33],
                "totals": {"episodes": 1, "timesteps": 33, "exec_samples": 33},
                "episodes": [{"global_episode_idx": 0, "h5_file": "record_dataset_Synthetic.h5", "raw_ep_idx": 0,
                              "num_timesteps": 33, "exec_start_idx": 0, "exec_samples": 33,
                              "exec_sample_offset": 0, "total_sample_offset": 0, "shard_idx": 0}]}
    manifest["sha256"] = manifest_sha256(manifest)
    write_json(d / "meta/episode_manifest.json", manifest)
    write_json(d / "meta/input_manifest.json", {"raw_dir": str(d / "raw"), "purpose": "synthetic_format_test"})
    layout = ms.LAYOUT_SPECS[ms.LAYOUT]
    write_json(d / "latents/metadata.json", {"raw_dir": str(d / "raw"), **layout.fields(), "purpose": "synthetic_format_test"})
    write_json(d / "oracle/vae_report.json", {"raw_dir": str(d / "raw"), "manifest_sha256": manifest["sha256"],
               "tested_metadata_sha256": sha(d / "latents/metadata.json"), "frame_mismatches": 0,
               "metadata_mismatches": 0, "demo_min_real_frames": 17, "exec_min_real_frames": 33,
               "scope": "synthetic_format_test", "checked_windows": 0})
    key = ms.segment_key("record_dataset_Synthetic.h5", 0, "exec")
    latent = np.zeros((1, 9, 16, 32, 32), dtype=np.float32)
    latent.tofile(d / "latents" / f"{key}.bin")
    latent_sha = sha(d / "latents" / f"{key}.bin")
    raw.tofile(d / "tokens" / f"{key}.f32.bin")
    raw_sha = sha(d / "tokens" / f"{key}.f32.bin")
    (d / "tokens" / f"{key}.f32.bin.sha256").write_text(raw_sha + "\n")
    (d / "encoder-synthetic/checkpoint_epoch_6.pt").write_bytes(b"synthetic-format-contract-only\n")
    (d / "encoder-synthetic/config.yaml").write_text("purpose: synthetic_format_test\n")
    pin = json.loads((REPO / "scripts/dataset/wan/SOURCE_PIN.json").read_text())
    shared = {"module_sha256": pin["source_sha256"], "encoder_src_sha256": pin.get("encoder_source_sha256"),
              "batch": 1, "vae_id": "synthetic_format_test"}
    encoder = {**shared, "run_dir": str(d / "encoder-synthetic"), "checkpoint": "checkpoint_epoch_6.pt",
               "checkpoint_sha256": sha(d / "encoder-synthetic/checkpoint_epoch_6.pt"),
               "config_sha256": sha(d / "encoder-synthetic/config.yaml"),
               "checkpoint_epoch": 6, "arch": "wan-latent-v10-st", "state_key": "encoder", "motion_dims": [8, 256]}
    worker = {"hostname": "synthetic", "worker": 0, "pid": os.getpid(), "mj_commit": pin["mj_repo_commit"]}
    write_json(d / "latents" / f"{key}.metadata.json", {"vae": shared, "worker": worker, "sha256": latent_sha})
    token_meta = {"schema": 1, "layout": ms.LAYOUT, "row_shape": [8, 256], "dtype": "float32", "token_dim": 2048,
                  "token_bytes": 8192, "num_grid": 1, "sha256": raw_sha, "bytes": 8192,
                  "input_latent_sha256": latent_sha, "input_block_sha256": [latent_sha],
                  "encoder": encoder, "encoder_flags": {}, "encoder_state_sha256": {}, "worker": worker}
    token_path = d / "tokens" / f"{key}.metadata.json"
    write_json(token_path, token_meta)
    pack.cmd_pack(SimpleNamespace(manifest=str(d / "meta/episode_manifest.json"), tokens=str(d / "tokens"),
        latents=str(d / "latents"), raw_dir=str(d / "raw"), encoder_run_dir=str(d / "encoder-synthetic"),
        oracle_report=str(d / "oracle/vae_report.json"), out=str(d / "motion"), resume=False, force_break_lock=False))
    pack.cmd_verify(SimpleNamespace(store=str(d / "motion"), manifest=None, tokens=None, resume=True, force_break_lock=False))
    metadata = ms.MotionMeta.load(d / "motion")
    store = ms.MotionStore(d / "motion", meta=metadata, manifest_path=d / "meta/episode_manifest.json", verify_level="full")
    restored = store.rows(np.array([0]))
    require(restored.shape == (1, 8, 256) and restored.dtype == np.float32, "raw8 表读回形状或dtype不符")
    require(restored.tobytes() == raw.tobytes(), "正常pack/verify未逐位保留raw8")
    require(metadata.num_rows == 1 and (d / "motion" / ms.MOTION_TABLE_RELPATH).stat().st_size == 8192, "原始行字节数不符")
    # 只改shape、不改字节，防止同为2048元素的重复配对伪装通过。
    bad = copy.deepcopy(token_meta)
    bad["row_shape"] = [4, 512]
    write_json(token_path, bad)
    negative = [rejected(lambda: pack.read_segment_tokens(d / "tokens", key, 1), "同字节错误raw shape")]
    write_json(token_path, token_meta)
    store_meta_path = d / "motion" / ms.META_RELPATH
    original = store_meta_path.read_bytes()
    bad = json.loads(original)
    bad["tables"][ms.MOTION_KEY]["row_shape"] = [4, 512]
    write_json(store_meta_path, bad)
    negative.append(rejected(lambda: ms.MotionMeta.load(d / "motion"), "表metadata伪shape"))
    store_meta_path.write_bytes(original)
    return metadata, {"scope": "synthetic_format_test", "rows": 1, "raw_shape": [8, 256], "row_bytes": 8192,
                      "raw_sha256": raw_sha, "verified": metadata.raw["status"], "negative_cases": negative}


def cpu_suite(root):
    from omegaconf import OmegaConf

    from mme_vla_suite.datastore.motion_adapter import motion_policy_spec
    from mme_vla_suite.datastore.motion_adapter import pair_adjacent_tokens
    from mme_vla_suite.shared.sampling import MEM_ORDER_SENTINEL
    from mme_vla_suite.shared.sampling import memory_order
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    from mme_vla_suite.training.framesamp_dataset import check_motion_contract
    config = OmegaConf.create({"enabled": True, "source_shape": [8, 256], "readout": "adjacent_pair_v1",
        "num_tokens": 4, "dim": 512, "budget": 640, "checkpoint_sha256": ENCODER_SHA})
    parsed = motion_policy_spec(config)
    require(parsed.window_budget == 160, "640 token 未对应160个窗口")
    sentinels = (np.arange(8)[:, None] * 10000 + np.arange(256)[None, :]).astype(np.float32)
    paired = pair_adjacent_tokens(sentinels)
    for pair in range(4):
        require(paired[pair, :256].tobytes() == sentinels[2 * pair].tobytes(), "前半槽身份错误")
        require(paired[pair, 256:].tobytes() == sentinels[2 * pair + 1].tobytes(), "后半槽身份错误")
    # 随机位串覆盖负零、非连续布局以及浮点值相等但位型不同的情形。
    bits = np.random.default_rng(42).integers(0, 2**32, (3, 8, 256), dtype=np.uint32)
    bits[0, 0, :4] = [0, 0x80000000, 0x7FC00001, 0x7F800000]
    values = bits.view(np.float32)[:, :, ::-1]
    recovered = pair_adjacent_tokens(values).reshape(values.shape)
    require(recovered.tobytes() == values.tobytes(), "配对逆映射未逐位恢复")
    negatives = [rejected(lambda: pair_adjacent_tokens(paired), "禁止二次配对"),
                 rejected(lambda: pair_adjacent_tokens(sentinels.astype(np.float64)), "禁止float64"),
                 rejected(lambda: pair_adjacent_tokens(np.zeros(768, np.float32)), "禁止旧768输入")]
    results = {"PAIRING": gate({"inverse_bitexact": True, "noncontiguous": True, "negative_cases": negatives})}
    # 一个具体局部反例：四对共用注意力权重，不保证等于八槽独立softmax。
    scalar_slots = np.arange(8, dtype=np.float64)
    raw_weights = np.exp(scalar_slots - scalar_slots.max())
    raw_weights /= raw_weights.sum()
    # 此处均值只是受控反例中的线性投影；真实输入适配始终是无损拼接。
    pair_keys = scalar_slots.reshape(4, 2).mean(axis=-1)
    pair_weights = np.exp(pair_keys - pair_keys.max())
    pair_weights /= pair_weights.sum()
    raw_attention = float(raw_weights @ scalar_slots)
    pair_attention = float(pair_weights @ pair_keys)
    require(scalar_slots.mean() == pair_keys.mean(), "局部反例的均匀权重基线尺度不一致")
    require(raw_attention != pair_attention, "局部注意力反例没有区分独立槽与配对权重")
    results["ATTENTION_COUNTEREXAMPLE"] = gate({"raw_scalar_output": raw_attention,
        "paired_scalar_output": pair_attention, "scope": "局部数学反例；不代表模型性能或任务收益"})
    dummy = FrameSampDataset.__new__(FrameSampDataset)
    dummy._motion_budget, dummy._motion_num_tokens, dummy._motion_window_budget = 640, 4, 160
    windows = []
    for count in (0, 1, 159, 160):
        raw = np.arange(count * 2048, dtype=np.float32).reshape(count, 8, 256)
        pos = np.arange(count * 256, dtype=np.float32).reshape(count, 256)
        starts = np.arange(count, dtype=np.int64) * 16
        emb, pe, mask, times = dummy._pad_motion(pair_adjacent_tokens(raw), pos, starts, count)
        require(emb.shape == (640, 512) and pe.shape == (640, 256), "展开预算形状错误")
        require(emb[:count * 4].tobytes() == raw.tobytes() and np.all(emb[count * 4:] == 0), "有效数值或padding错误")
        expected_mask = np.array([i < count * 4 for i in range(640)])
        np.testing.assert_array_equal(mask, expected_mask)
        for i in range(count):
            for j in range(4):
                require(pe[4 * i + j].tobytes() == pos[i].tobytes(), "窗口PE未逐位repeat4")
                require(times[4 * i + j] == starts[i], "窗口时间未repeat4")
        require(np.all(pe[4 * count:] == 0) and np.all(times[4 * count:] == MEM_ORDER_SENTINEL), "填充PE/时间错误")
        frame_times = np.array([0, 16, 32, 48, 64, MEM_ORDER_SENTINEL, MEM_ORDER_SENTINEL, MEM_ORDER_SENTINEL])
        order = memory_order(frame_times, 64, times)
        identities = [(int(frame_times[i // 64]), 0, i) for i in range(512)]
        identities += [(int(times[i]), 1, 512 + i) for i in range(640)]
        expected = np.array([row[2] for row in sorted(identities)], dtype=np.int32)
        np.testing.assert_array_equal(order, expected)
        np.testing.assert_array_equal(np.sort(order), np.arange(1152))
        windows.append({"windows": count, "valid_tokens": int(mask.sum()), "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()})
    overflow = rejected(lambda: dummy._pad_motion(np.zeros((161, 4, 512), np.float32),
        np.zeros((161, 256), np.float32), np.arange(161), 161), "161窗溢出")
    results["WINDOW_BUDGET"] = gate({"cases": windows, "overflow": overflow})
    results["MASK_ORDER"] = gate({"total_tokens": 1152, "oracle": "独立Python三元组排序", "cases": windows})
    with tempfile.TemporaryDirectory(prefix="pair4-cpu-", dir=root / "reports") as temporary:
        meta, raw_report = _synthetic_raw_store(temporary)
        results["RAW_FORMAT_CPU"] = gate(raw_report)
        valid = OmegaConf.create(OmegaConf.to_container(config))
        valid.checkpoint_sha256 = meta.provenance["encoder"]["checkpoint_sha256"]
        check_motion_contract(valid, meta)
        source_cases = [rejected(lambda: check_motion_contract(config, meta), "错误encoder checkpoint SHA")]
        legacy = SimpleNamespace(raw={"layout": "motion-768-grid16-demopad17-v1"}, row_shape=(768,), provenance=meta.provenance)
        source_cases.append(rejected(lambda: check_motion_contract(config, legacy), "新配置拒绝旧768表"))
        for key, value in (("source_shape", [4, 512]), ("readout", "identity"), ("budget", 641), ("num_tokens", 8)):
            bad = OmegaConf.create(OmegaConf.to_container(config))
            bad[key] = value
            source_cases.append(rejected(lambda bad=bad: motion_policy_spec(bad), f"错误配置{key}"))
        results["SOURCE_REJECTION"] = gate({"negative_cases": source_cases})
    results.update(scope_checks())
    return {"status": "PASS", "gates": results, "scope": "CPU合成合同；不证明真实提取或完整模型"}


def summary_suite(root):
    reports, gates = {}, {}
    current_code = code_identity()
    for suite, required in REQUIRED.items():
        path = root / "reports" / f"{suite}.json"
        if not path.exists():
            reports[suite] = {"status": "NOT_RUN", "reason": "报告不存在"}
            continue
        report = json.loads(path.read_text())
        status = report.get("status", "FAIL")
        report_gates = report.get("gates", {})
        if status not in ("PASS", "FAIL", "NOT_RUN") or not report_gates:
            status = "FAIL"
        if any(report_gates.get(name, {}).get("status") != "PASS" for name in required):
            status = "FAIL" if report.get("status") != "NOT_RUN" else "NOT_RUN"
        if any(value.get("status") != "PASS" for value in report_gates.values()):
            status = "FAIL"
        if suite != "source-smoke" and report.get("code_sha256") != current_code:
            status = "FAIL"
        if report.get("root") is not None and Path(report["root"]).resolve() != root:
            status = "FAIL"
        reports[suite] = {"status": status, "report_sha256": sha(path), "path": str(path)}
        gates.update({f"{suite}/{name}": value for name, value in report_gates.items()})
    states = [value["status"] for value in reports.values()]
    status = "FAIL" if "FAIL" in states else ("NOT_RUN" if "NOT_RUN" in states else "PASS")
    if status == "PASS":
        # 汇总时重核磁盘事实，不只信任过往PASS文本。
        gates.update(scope_checks())
        model_report = json.loads((root / "reports/model.json").read_text())
        for relative, digest in model_report["input_report_sha256"].items():
            require(sha(root / relative) == digest, f"模型验证后前置报告被替换：{relative}")
    return {"status": status, "gates": {"TRAIN_SMOKE": {"status": status, "details": reports}}, "reports": reports,
            "limitations": ["仅单窗训练冒烟", "未恢复optimizer、step或RNG", "未验证在线推理、评估或全量训练"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("cpu", "prepare-fixture", "finalize-fixture", "model", "summary"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--history-config", default=HISTORY_CONFIG)
    parser.add_argument("--updates", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    require(root.is_relative_to(Path("/scratch/hongze")), "产物必须位于/scratch/hongze")
    require(args.updates == 1 and args.history_config == HISTORY_CONFIG, "固定冒烟只允许原配置及一次更新")
    (root / "reports").mkdir(parents=True, exist_ok=True)
    if args.suite != "model":
        # CPU检查不得因间接导入JAX占用GPU。
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    start = time.perf_counter()
    report = {"status": "FAIL", "gates": {}}
    try:
        if args.suite == "cpu":
            report = cpu_suite(root)
        elif args.suite in ("prepare-fixture", "finalize-fixture"):
            import pair4_fixture
            function = pair4_fixture.prepare_fixture if args.suite == "prepare-fixture" else pair4_fixture.finalize_fixture
            report = function(root)
        elif args.suite == "model":
            from pair4_model import model_suite
            report = model_suite(root, args.history_config)
        else:
            report = summary_suite(root)
    except (Exception, SystemExit) as error:
        report = {"status": "FAIL", "gates": {args.suite.upper().replace("-", "_"): {"status": "FAIL", "reason": str(error)}},
                  "error_type": type(error).__name__, "traceback": traceback.format_exc()}
        traceback.print_exc()
    report.update(suite=args.suite, root=str(root), elapsed_seconds=time.perf_counter() - start,
                  code_sha256=code_identity(), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    write_json(root / "reports" / f"{args.suite}.json", report)
    for name, value in report.get("gates", {}).items():
        print(f"{name}={value.get('status', 'FAIL')}", flush=True)
    print(f"PAIR4_SUITE={report['status']} suite={args.suite} elapsed_seconds={report['elapsed_seconds']:.3f}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
