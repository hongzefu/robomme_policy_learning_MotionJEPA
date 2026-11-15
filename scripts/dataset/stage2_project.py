"""stage2 运行项目选择（1114 计划 B 契约①）：只用标准库，主环境 import、sidecar 环境按路径加载。

登记两项运行项目，sidecar 脚本共用 ``scripts/dataset/wan_stage2/``，pin、锁与 venv 各用各的：

  wan_stage2           旧三路 encoder（wan-latent-stage2-blockwise-pool4-three-decoder），venv v1-store/venvs/wan-stage2
  wan_stage2_rgbstate  1113 RGB+state encoder（wan-latent-stage2-blockwise-pool4-rgb-state），venv v1-store/venvs/wan-stage2-rgbstate

选择规则：显式 ``--stage2-project`` 优先；不给时按可推断来源推断（store provenance 或 ckpt payload 的 arch）；
显式值与推断值不符即拒绝。既无 store 也无 ckpt 的入口必须显式给出（``require_explicit=True``）。
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
SIDECAR_DIR = REPO / "scripts" / "dataset" / "wan_stage2"
VAE_ADAPTER = REPO / "scripts" / "dataset" / "wan" / "wan_motion_infer.py"

THREE_DECODER_ARCH = "wan-latent-stage2-blockwise-pool4-three-decoder"
RGBSTATE_ARCH = "wan-latent-stage2-blockwise-pool4-rgb-state"

# 项目名 → (项目根相对仓库路径, venv 目录名, 对应 encoder arch)
_REGISTRY = {
    "wan_stage2": ("scripts/dataset/wan_stage2", "wan-stage2", THREE_DECODER_ARCH),
    "wan_stage2_rgbstate": ("scripts/dataset/wan_stage2_rgbstate", "wan-stage2-rgbstate", RGBSTATE_ARCH),
}
PROJECT_NAMES = tuple(_REGISTRY)
ARCH_PROJECT = {arch: name for name, (_, _, arch) in _REGISTRY.items()}


@dataclasses.dataclass(frozen=True)
class Stage2Project:
    name: str
    root: pathlib.Path          # 含 pyproject.toml / uv.lock / SOURCE_PIN.json 的运行项目目录
    source_pin: pathlib.Path
    lock: pathlib.Path
    venv: pathlib.Path          # UV_PROJECT_ENVIRONMENT
    python: pathlib.Path        # venv 内解释器
    sidecar_dir: pathlib.Path   # 共用 sidecar 脚本目录
    arch: str

    def read_pin(self) -> dict:
        return json.loads(self.source_pin.read_text(encoding="utf-8"))


def default_v1_store(v1_store=None) -> pathlib.Path:
    return pathlib.Path(v1_store or os.environ.get("MMEVLA_V1_STORE") or (REPO / "v1-store"))


def get_project(name: str, *, v1_store=None, repo=None) -> Stage2Project:
    """按名取登记项；未知名直接拒绝。"""
    if name not in _REGISTRY:
        raise ValueError(f"未知 stage2 运行项目：{name!r}（只认 {', '.join(PROJECT_NAMES)}）")
    relative, venv_name, arch = _REGISTRY[name]
    base = pathlib.Path(repo) if repo is not None else REPO
    root = base / relative
    venv = default_v1_store(v1_store) / "venvs" / venv_name
    return Stage2Project(name=name, root=root, source_pin=root / "SOURCE_PIN.json", lock=root / "uv.lock",
                         venv=venv, python=venv / "bin" / "python",
                         sidecar_dir=base / "scripts" / "dataset" / "wan_stage2", arch=arch)


def project_for_arch(arch) -> str:
    if arch not in ARCH_PROJECT:
        raise ValueError(f"无法按 encoder arch 推断 stage2 运行项目：{arch!r}")
    return ARCH_PROJECT[arch]


def resolve_project_name(explicit=None, *, arch=None, require_explicit=False, context="") -> str:
    """契约①：显式优先，推断兜底，两者不符即拒；require_explicit 时缺显式值即拒。"""
    where = f"（{context}）" if context else ""
    if explicit is not None and explicit not in _REGISTRY:
        raise ValueError(f"未知 stage2 运行项目{where}：{explicit!r}")
    if require_explicit and explicit is None:
        raise ValueError(f"本入口既无 store 也无 ckpt，必须显式给出 --stage2-project{where}")
    inferred = project_for_arch(arch) if arch is not None else None
    if explicit is not None and inferred is not None and explicit != inferred:
        raise ValueError(f"显式 --stage2-project={explicit} 与 arch 推断的 {inferred} 不符{where}")
    name = explicit or inferred
    if name is None:
        raise ValueError(f"无法确定 stage2 运行项目{where}：未给 --stage2-project 且无可推断来源")
    return name


def resolve_project(explicit=None, *, arch=None, require_explicit=False, context="", v1_store=None) -> Stage2Project:
    return get_project(resolve_project_name(explicit, arch=arch, require_explicit=require_explicit,
                                            context=context), v1_store=v1_store)


def arch_of_provenance(provenance) -> str:
    try:
        return provenance["encoder"]["arch"]
    except (KeyError, TypeError) as exc:
        raise ValueError("store provenance 缺 encoder.arch，无法推断 stage2 运行项目") from exc


def arch_of_store(store_root) -> str:
    """只读 store_meta.json 的 provenance.encoder.arch（不依赖主环境包）。"""
    meta = json.loads((pathlib.Path(store_root) / "meta" / "store_meta.json").read_text(encoding="utf-8"))
    return arch_of_provenance(meta.get("provenance"))


def arch_of_checkpoint(ckpt) -> str:
    """读 ckpt payload 的 arch；仅在 sidecar（torch）环境调用。"""
    import torch
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = payload.get("arch") if isinstance(payload, dict) else None
    if not isinstance(arch, str):
        raise ValueError(f"checkpoint 缺 arch：{ckpt}")
    return arch


def project_of_source_pin(source_pin_path, *, v1_store=None) -> str:
    """反查 SOURCE_PIN.json 所在的登记项目；不属于任何登记项即拒。"""
    target = pathlib.Path(source_pin_path).resolve()
    for name in PROJECT_NAMES:
        if get_project(name, v1_store=v1_store).source_pin.resolve() == target:
            return name
    raise ValueError(f"SOURCE_PIN 不属于任何登记的 stage2 运行项目：{source_pin_path}")


def check_project_files(project: Stage2Project, sha256_file) -> dict:
    """核对 SOURCE_PIN 的 VAE 适配件与依赖锁两项摘要；返回 pin 内容。"""
    pin = project.read_pin()
    for key, path in (("vae_adapter_sha256", VAE_ADAPTER), ("dependency_lock_sha256", project.lock)):
        if sha256_file(path) != pin.get(key):
            raise ValueError(f"stage2 运行项目 {project.name} 的 SOURCE_PIN.{key} 与实际文件不符")
    return pin


def add_argument(parser, *, required=False):
    """各入口统一的 --stage2-project 参数。"""
    parser.add_argument("--stage2-project", choices=PROJECT_NAMES, required=required, default=None,
                        help="stage2 运行项目（契约①）；不给时按 store/ckpt 的 arch 推断，无可推断来源的入口必填")
