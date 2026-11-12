"""context 1024 档形制守卫单测（1111 计划 §2.2，子任务 G4C-S2）。

只测纯函数 memory_shape_allowed 与 16 帧 context YAML 的取值，纯 CPU、不建库、不读数据：
- context 512（legacy 分支）与 context 2048 的判定不变；
- context 1024（16 帧 × 64 token）放行；
- modulation 512/1024/2048/4096 判定不变；
- 非法形状、类型反例、motion 开启、memory_token_dim 不符仍拒绝。
"""

from __future__ import annotations

import pytest

from mme_vla_suite.training.framesamp_dataset import memory_shape_allowed


def _ok(raw, it="context", dim=2048, motion=False):
    return memory_shape_allowed(raw, it, dim, motion)


# ── context：512 / 2048 不变，1024 新放行 ──
@pytest.mark.parametrize("raw", [(512, 64, 1), (512, 16, 1), (1024, 64, 1), (2048, 64, 1)])
def test_context_allowed(raw):
    assert _ok(raw, "context", 2048)


# ── modulation：512（legacy）与 1024/2048/4096 判定不变 ──
@pytest.mark.parametrize("raw", [(512, 64, 1), (512, 16, 1), (1024, 64, 1), (2048, 64, 1), (4096, 64, 1)])
def test_modulation_allowed_unchanged(raw):
    assert _ok(raw, "modulation", 1024)


def test_modulation_motion_still_rejected():
    for b in (1024, 2048, 4096):
        assert not _ok((b, 64, 1), "modulation", 1024, motion=True)


# ── context 4096 仍不放行（只扩到 1024） ──
def test_context_4096_rejected():
    assert not _ok((4096, 64, 1), "context", 2048)


# ── context 1024 / 2048 的附加条件原样：memory_token_dim 必须 2048、无 motion ──
@pytest.mark.parametrize("budget", [1024, 2048])
def test_context_extra_conditions(budget):
    assert not _ok((budget, 64, 1), "context", 1024)
    assert not _ok((budget, 64, 1), "context", 2048, motion=True)


# ── 非法形状与类型反例仍拒绝（与 test_pack_guards 的反例同口径） ──
@pytest.mark.parametrize("raw", [
    (1024, 16, 1), (1024, 64, 2), (2048, 16, 1), (2048, 64, 2), (768, 64, 1), (256, 64, 1),
    (1024.0, 64, 1), (2048.0, 64, 1), (True, 64, 1), (1024, 64, True),
])
@pytest.mark.parametrize("it,dim", [("context", 2048), ("modulation", 1024)])
def test_illegal_shapes_rejected(raw, it, dim):
    assert not _ok(raw, it, dim)


def test_illegal_string_budget_rejected():
    assert not _ok(("1024", 64, 1), "context", 2048)
    assert not _ok(("2048", 64, 1), "context", 2048)


# ── legacy 的 int 转换接受条件不变（512.5 / "512" 照旧放行） ──
@pytest.mark.parametrize("budget", [512, 512.5, "512"])
def test_legacy_conversion_unchanged(budget):
    assert _ok((budget, 64, 1), "modulation", 1024)
    assert _ok((budget, 64, 1), "context", 2048)


# ── 16 帧 context YAML：只比 8 帧版改 budget，且过守卫 ──
def test_16frame_yaml_only_budget_differs():
    from omegaconf import OmegaConf
    from mme_vla_suite.models.config.utils import get_history_config
    h8 = OmegaConf.to_container(get_history_config("perceptual-framesamp-context-8frame-8x8.yaml"))
    h16 = OmegaConf.to_container(get_history_config("perceptual-framesamp-context-16frame-8x8.yaml"))
    assert h8["budget"] == 512 and h16["budget"] == 1024
    assert {k: v for k, v in h8.items() if k != "budget"} == {k: v for k, v in h16.items() if k != "budget"}
    raw = (h16["budget"], h16["token_per_image"], h16["num_views"])
    assert memory_shape_allowed(raw, h16["integration_type"], h16["memory_token_dim"],
                                bool(h16.get("motion", {}).get("enabled", False)))
    assert h16["budget"] // (h16["token_per_image"] * h16["num_views"]) == 16
