"""正式两份yaml与新60k条目的最小差异约束。"""
import dataclasses
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_paired_yaml_only_changes_fusion_and_width():
    prefix = ROOT / "src/mme_vla_suite/models/config/robomme"
    histories = [yaml.safe_load((prefix / f"perceptual-framesamp-{f}-32frame-4x4-motion-stage2-4x256-orig16.yaml").read_text())
                 for f in ("ctx", "modul")]
    assert {key for key in histories[0] if histories[0][key] != histories[1][key]} == {"integration_type", "memory_token_dim"}
    assert (histories[0]["memory_token_dim"], histories[1]["memory_token_dim"]) == (2048, 1024)
    print("PAIRED_CONFIG=PASS diff_keys=integration_type,memory_token_dim")


def test_new_config_matches_existing_60k_except_authorized_fields():
    from mme_vla_suite.training.config import get_config
    old, new = get_config("mme_vla_suite_b128_60k"), get_config("mme_vla_suite_orig16_b128_60k")
    assert new.fsdp_devices == 8 and new.num_workers == 16
    from test_check_orig16_stage2_launch import launch
    assert new.data.assets.assets_dir == launch.Contract().assets_rel
    restored = dataclasses.replace(new, name=old.name, fsdp_devices=old.fsdp_devices, num_workers=old.num_workers,
        data=dataclasses.replace(new.data, assets=dataclasses.replace(new.data.assets, assets_dir=old.data.assets.assets_dir)))
    assert restored == old
