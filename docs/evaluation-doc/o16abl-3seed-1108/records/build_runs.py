"""在 sled-vail 上运行：为双消融三 seed 评估生成两份 runtime 合同、2 个冒烟根与 6 个正式 run 根。

只新建，不覆盖：任一目标已存在即报错退出。用法（vail 上）：
  <policy python> build_runs.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SHA = "4d70b1a5a02e2e1b4cfda51e8b6d2573dc453d42"
E = Path("/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval")
REL = E / "releases" / SHA
PY = E / "v1-store/envs/policy/bin/python"
O = E / "v1-store/evaluation/o16abl-80k"
SRC_INPUTS = E / "v1-store/evaluation/official-stage2-1026/o16ctx-60k-seed42/inputs"
EXPORTS = Path("/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/exports")
MODELS = {
    "N": "gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k",
    "S": "gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k",
}
EXPECT = {  # 已在 aspen/vail 两侧算过的交付文件 sha256
    "N": {"manifest": "1d1e4b4e8310b5b9925e0000a7e651fa1a7bfdc93384029a5857111cfdfe20c4",
          "provenance": "03dc1d30a8c95e9488096e2c8bfb9feb4953cf74a77cf447cc0bf4d02de56d16"},
    "S": {"manifest": "3e1f86de7be99232799c5af5f1324efcf01d626ad96a9531569c5b87855cf570",
          "provenance": "75a483a746c13bc5e2f67b3fb994a523b0968b7a3a4b5127e6fc5d2740b25073"},
}
EPISODES = {"run800": ("episodes.json", "8f1cb76d9dca2ee520b96d64b80230ed2ae1ac7a44bbb4bb74b1c24844a8a453"),
            "smoke2": ("smoke2-episodes.json", "9ebebe780b36767e9ed60f88c5f5e8416b502cfd82499008474f2bdcadba1add")}
POLICY_CONFIG = "mme_vla_suite_orig16_b64_80k"

sys.path.insert(0, str(REL))
from scripts.evaluation.runtime_contract import identity_sha256, sha256_file  # noqa: E402

WORKER = """#!/usr/bin/env bash
# 用法：run-worker.sh <执行sha> <worker-id> <卡号>；在占位内起一个动态领取任务的常驻 worker，显式绑定一张卡
set -euo pipefail
eval_root={E}
repo="$eval_root/releases/$1"
store="$eval_root/v1-store"
run="{RUN}"
unset JAX_PLATFORMS
export CUDA_VISIBLE_DEVICES="$3"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MMEVLA_V1_STORE="$store" OPENPI_DATA_HOME="$store/models"
export MS_ASSET_DIR="$store/cache/maniskill"
export HF_HOME="$store/cache/hf" HF_HUB_CACHE="$store/cache/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1
export UV_CACHE_DIR="$store/cache/uv" UV_LINK_MODE=copy
export XDG_CACHE_HOME="$store/cache" TORCH_HOME="$store/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$store/cache/torchinductor" TRITON_CACHE_DIR="$store/cache/triton"
export CUDA_CACHE_PATH="$store/cache/cuda" JAX_COMPILATION_CACHE_DIR="$store/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$repo:$repo/src:$repo/packages/openpi-client/src:$repo/third_party/robomme_benchmark/src"
cd "$repo"
printf 'WORKER_START code=%s worker=%s job=%s step=%s host=%s gpu=%s\\n' "$1" "$2" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$HOSTNAME" "$CUDA_VISIBLE_DEVICES"
exec "$store/envs/policy/bin/python" -u -m scripts.evaluation.resident_worker \\
  --run-root "$run" --spec "$run/spec.json" --worker-id "$2" --job-id "$SLURM_JOB_ID"
"""

CONTROLLER = """#!/usr/bin/env bash
# sled-vail 上运行该 run 根的唯一控制器（同 10-31，只换 run 根与执行 sha）
set -o pipefail
run={RUN}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export PYTHONPATH={REL}
printf 'CONTROLLER_START host=%s code={SHA}\\n' "$(hostname)"
{PY} -u -m scripts.evaluation.official_queue controller --run-root "$run" --slurm-host greatlakes \\
  2>&1 | tee "$run/logs/controller-vail.log"
code=$?
printf 'EXIT_CODE=%s\\n' "$code" | tee -a "$run/logs/controller-vail.log"
exit "$code"
"""


def write_new(path, text, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


def main():
    assert REL.is_dir(), f"执行副本不存在: {REL}"
    if O.exists():
        raise SystemExit(f"错误: {O} 已存在，不覆盖")
    rows = []
    for m, run_name in MODELS.items():
        stage = EXPORTS / f"hf-ckpt-{run_name}" / "stage"
        man = stage / "manifests/step-79999.json"
        prov = stage / "motion_provenance.json"
        assert sha256_file(man) == EXPECT[m]["manifest"], f"{m} 清单 sha 不符"
        assert sha256_file(prov) == EXPECT[m]["provenance"], f"{m} provenance sha 不符"
        assert (stage / "79999/params").is_dir(), f"{m} 缺 79999/params"
        contract = O / f"o16abl-{m}/contracts/runtime.json"
        env = dict(os.environ, PYTHONPATH=str(REL), JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([str(PY), "-m", "scripts.evaluation.measure_target_runtime", "--provenance", str(prov),
                              "--output", str(contract)], cwd=REL, env=env, check=True, capture_output=True, text=True)
        print(f"CONTRACT_GEN model={m} {out.stdout.strip()}")
        contract_sha = sha256_file(contract)
        print(f"CONTRACT=PASS model={m} sha256={contract_sha} provenance={EXPECT[m]['provenance'][:8]}")
        runs = [(f"o16abl-{m}-smoke", "smoke2", 0, "validation")]
        runs += [(f"o16abl-{m}-seed{s}", "run800", s, "production") for s in (42, 7, 0)]
        for name, root_name, seed, purpose in runs:
            base = O / name
            ep_name, ep_file_sha = EPISODES[root_name]
            inputs = base / "inputs"
            inputs.mkdir(parents=True)
            src = SRC_INPUTS / ep_name
            shutil.copyfile(src, inputs / ep_name)
            assert sha256_file(inputs / ep_name) == ep_file_sha, f"{name} 清单复制后 sha 不符"
            episodes = json.loads((inputs / ep_name).read_text())
            identity = {"schema": "simple-run-identity-1027", "checkpoint_manifest_sha256": EXPECT[m]["manifest"],
                        "checkpoint_step": 79999, "episodes_sha256": episodes["sha256"], "eval_code_commit": SHA,
                        "runtime_contract_sha256": contract_sha, "model_seed": seed, "purpose": purpose,
                        "model_kind": "stage2_pool4_concat", "policy_config": POLICY_CONFIG}
            rid = identity_sha256(identity)
            spec = {"repo": str(REL), "policy_python": str(PY),
                    "stage2_python": str(E / "v1-store/envs/stage2/bin/python"),
                    "sim_python": str(E / "v1-store/envs/official-sim/bin/python"),
                    "checkpoint": str(stage / "79999"), "policy_config": POLICY_CONFIG,
                    "run_identity_sha256": rid, "model_kind": "stage2_pool4_concat", "model_seed": seed,
                    "purpose": purpose, "target_contract": str(contract)}
            write_new(inputs / f"{root_name}.run-identity.json", json.dumps(identity, indent=2, ensure_ascii=False) + "\n")
            write_new(inputs / f"{root_name}.spec-input.json", json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
            run_root = base / root_name
            subprocess.run([str(PY), "-m", "scripts.evaluation.official_queue", "initialize", "--run-root", str(run_root),
                            "--manifest", str(inputs / ep_name), "--spec", str(inputs / f"{root_name}.spec-input.json")],
                           cwd=REL, env=env, check=True)
            write_new(base / "run-worker.sh", WORKER.replace("{E}", str(E)).replace("{RUN}", str(run_root)), 0o755)
            write_new(base / "run-controller-vail.sh",
                      CONTROLLER.replace("{RUN}", str(run_root)).replace("{REL}", str(REL)).replace("{SHA}", SHA)
                      .replace("{PY}", str(PY)), 0o755)
            rows.append((name, root_name, seed, purpose, rid))
            print(f"RUN_ROOT=PASS name={name} root={root_name} seed={seed} purpose={purpose} run_identity_sha256={rid}")
    print(f"BUILD_RUNS=PASS runs={len(rows)}")


if __name__ == "__main__":
    main()
