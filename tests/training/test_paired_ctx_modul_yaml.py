"""正式两份yaml与新60k条目的最小差异约束。"""
import dataclasses
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


PREFIX = ROOT / "src/mme_vla_suite/models/config/robomme"


def load(fusion, readout):
    return yaml.safe_load((PREFIX / f"perceptual-framesamp-{fusion}-32frame-4x4-motion-stage2-{readout}-orig16.yaml").read_text())


@pytest.mark.parametrize("readout", ["4x256", "1x1024"])
def test_paired_yaml_only_changes_fusion_and_width(readout):
    histories = [load(f, readout) for f in ("ctx", "modul")]
    assert {key for key in histories[0] if histories[0][key] != histories[1][key]} == {"integration_type", "memory_token_dim"}
    assert (histories[0]["memory_token_dim"], histories[1]["memory_token_dim"]) == (2048, 1024)
    print("PAIRED_CONFIG=PASS diff_keys=integration_type,memory_token_dim")


@pytest.mark.parametrize("fusion", ["ctx", "modul"])
def test_split_and_concat_only_differ_in_four_motion_fields(fusion):
    split, concat = load(fusion, "4x256"), load(fusion, "1x1024")
    assert {key for key in set(split) | set(concat) if split.get(key) != concat.get(key)} == {"motion"}
    assert set(split["motion"]) == set(concat["motion"])
    diff = {key for key in split["motion"] if split["motion"][key] != concat["motion"][key]}
    assert diff == {"readout", "num_tokens", "dim", "budget"}
    assert [split["motion"][k] for k in ("readout", "num_tokens", "dim", "budget")] == ["stage2_pool4_split_v1", 4, 256, 384]
    assert [concat["motion"][k] for k in ("readout", "num_tokens", "dim", "budget")] == ["stage2_pool4_concat_v1", 1, 1024, 96]
    print(f"NOSPLIT_CONFIG=PASS fusion={fusion} diff_keys=motion.readout,motion.num_tokens,motion.dim,motion.budget")


def test_new_config_matches_existing_60k_except_authorized_fields():
    from mme_vla_suite.training.config import get_config
    old, new = get_config("mme_vla_suite_b128_60k"), get_config("mme_vla_suite_orig16_b128_60k")
    assert new.fsdp_devices == 8 and new.num_workers == 16
    from test_check_orig16_stage2_launch import launch
    assert new.data.assets.assets_dir == launch.Contract().assets_rel
    restored = dataclasses.replace(new, name=old.name, fsdp_devices=old.fsdp_devices, num_workers=old.num_workers,
        data=dataclasses.replace(new.data, assets=dataclasses.replace(new.data.assets, assets_dir=old.data.assets.assets_dir)))
    assert restored == old
