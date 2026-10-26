#!/usr/bin/env python3
"""记录用户明确接受的单次 G2 数值例外；仅重验身份，不执行窗口编码。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import validate_stage2 as V

PRODUCER = "stage2-g2-user-exception-v1"
CHECKPOINT_SHA256 = "27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035"
SAMPLES_SHA256 = "4cb05b6b3097b22a8594576def1505957399a924d4189591980292e3bd320d30"
FAILURE_SHA256 = "88cc88fb69827f00763b5499e3aa3d39bdb411db3e186660eed596716d6dce72"
USER_QUOTE = "可以忽略 继续做"
AUTHORIZED_AT = "2026-10-26T19:21:45Z"
SCOPE = "g2_cross_precision_numerics_only"
PIN_KEYS = frozenset(("mj_repo_commit", "export_source_sha256", "encoder_source_sha256",
                      "vae_adapter_sha256", "dependency_lock_sha256"))
VAE_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def atomic_create_json(path, record):
    """同目录临时文件落盘后用硬链接原子发布；并发创建或任何已有路径都不覆盖。"""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"拒绝覆盖已有例外记录：{path}")
    payload = V.canonical(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link 的目标必须不存在；不能使用会覆盖已有文件的 replace。
        os.link(temporary, path, follow_symlinks=False)
    finally:
        os.unlink(temporary)


def input_evidence(args):
    """固定三摘要先于模型加载检查，原始失败字符串按UTF8原字节保存。"""
    output = Path(args.out)
    if output.name != "G2-waiver.json":
        raise ValueError("例外输出必须另存为 G2-waiver.json")
    if os.path.lexists(output):
        raise FileExistsError(f"拒绝覆盖已有例外记录：{output}")
    if args.expected_sha256 != CHECKPOINT_SHA256:
        raise ValueError("用户例外仅适用于本次固定 checkpoint SHA")
    paths = {"failure": Path(args.failure), "samples": Path(args.samples), "checkpoint": Path(args.ckpt)}
    if output.resolve() in {path.resolve() for path in paths.values()}:
        raise ValueError("例外记录不能覆盖任何输入证据")
    raw_failure = paths["failure"].read_bytes()
    if V.sha_bytes(raw_failure) != FAILURE_SHA256:
        raise ValueError("原始 G2 失败文件不是用户接受的固定 SHA")
    failure_text = raw_failure.decode("utf-8", errors="strict")
    failure = json.loads(failure_text)
    if (not isinstance(failure, dict) or failure.get("schema") != "stage2-validation-gate-v1"
            or failure.get("gate") != "G2" or failure.get("status") != "FAIL"):
        raise ValueError("原始证据必须保留 G2/FAIL 身份")
    if V.sha_file(paths["samples"]) != SAMPLES_SHA256:
        raise ValueError("G0 清单不是用户例外绑定的固定 SHA")
    samples = V.load_samples(paths["samples"])
    if V.sha_file(paths["checkpoint"]) != CHECKPOINT_SHA256:
        raise ValueError("实际 checkpoint 与用户例外绑定的 SHA 不同")
    return failure_text, samples, paths


def identity_evidence(args):
    """复用正式身份加载器；不调用VAE、encoder或G2的任何窗前向。"""
    pin = V.read_json(HERE / "SOURCE_PIN.json")
    if not isinstance(pin, dict) or set(pin) != PIN_KEYS:
        raise ValueError("例外记录必须绑定完整五键 SOURCE_PIN")
    if not isinstance(pin["mj_repo_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", pin["mj_repo_commit"]) is None:
        raise ValueError("SOURCE_PIN 的 MJ 提交必须为完整40位")
    for key in PIN_KEYS - {"mj_repo_commit"}:
        V.require_sha(pin[key], f"SOURCE_PIN.{key}")
    _, _, adapter, _, info = V.setup_runtime(args)
    if info["encoder"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("实际加载的 encoder 与固定 checkpoint 身份不同")
    source_keys = ("mj_repo_commit", "export_source_sha256", "encoder_source_sha256")
    if info["source_pin"] != {key: pin[key] for key in source_keys}:
        raise ValueError("实际导出的S1源码身份与完整SOURCE_PIN不同")
    numerics = copy.deepcopy(info["numerics"])
    vae, vae_info = adapter.load_vae(VAE_ID, args.device,
                                    expected_state_sha256=adapter.VAE_STATE_SHA256_EXPECTED,
                                    strict_versions=True)
    expected_vae = dict(vae_id=VAE_ID, vae_state_sha256=adapter.VAE_STATE_SHA256_EXPECTED,
                        vae_dtype="float32", latent_mode="mode", batch=1, tf32="off", amp="off")
    # 旧适配件内部含assert，这里用显式分支保证优化模式也不会删掉关键身份检查。
    if (vae.use_tiling or vae.use_slicing
            or any(type(vae_info.get(key)) is not type(value) or vae_info[key] != value
                   for key, value in expected_vae.items())):
        raise ValueError("实际VAE权重、精度或编码配置不符合固定身份")
    online_vae = {key: vae_info[key] for key in expected_vae}
    online_vae.update(adapter_sha256=pin["vae_adapter_sha256"], numerics=copy.deepcopy(numerics),
                      preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33))
    del vae
    if (V.read_json(HERE / "SOURCE_PIN.json") != pin
            or V.sha_file(HERE / "uv.lock") != pin["dependency_lock_sha256"]
            or V.sha_file(HERE.parent / "wan/wan_motion_infer.py") != pin["vae_adapter_sha256"]):
        raise ValueError("身份采集期间来源锁或VAE适配件发生变化")
    return pin, numerics, online_vae


def accept_exception(args):
    failure_text, samples, paths = input_evidence(args)
    # 公共消费校验器必须已安装到同一检出；缺件即失败，不从其它工作树寻找替代。
    from mme_vla_suite.datastore.motion_store import validate_stage2_g2_evidence

    pin, numerics, online_vae = identity_evidence(args)
    record = dict(schema="stage2-validation-gate-v1", gate="G2", status="WAIVED", producer=PRODUCER,
                  checkpoint_sha256=CHECKPOINT_SHA256, samples_sha256=SAMPLES_SHA256,
                  source_pin=pin, numerics=numerics, online_vae=online_vae,
                  exception=dict(schema=PRODUCER, authorized_by="user", user_quote=USER_QUOTE,
                                 authorized_at=AUTHORIZED_AT, scope=SCOPE, numeric_validation_complete=False,
                                 original_failure_sha256=FAILURE_SHA256, original_failure_json=failure_text))
    status = validate_stage2_g2_evidence(record, checkpoint_sha256=CHECKPOINT_SHA256,
                                         samples_sha256=SAMPLES_SHA256, source_pin=pin, numerics=numerics,
                                         online_vae=online_vae, demo_tail=samples["counts"]["demo_tail"])
    if status != "WAIVED":
        raise ValueError("公共校验器未将此记录判定为WAIVED")
    for name, digest in (("failure", FAILURE_SHA256), ("samples", SAMPLES_SHA256), ("checkpoint", CHECKPOINT_SHA256)):
        if V.sha_file(paths[name]) != digest:
            raise ValueError(f"身份采集期间输入证据发生变化：{name}")
    atomic_create_json(args.out, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("failure", "samples", "ckpt", "expected-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    accept_exception(args)
    print("G2_USER_EXCEPTION=ACCEPTED original_status=FAIL numeric_validation_complete=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
