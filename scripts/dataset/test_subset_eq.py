"""子集逐字节守卫的小型真实文件验证；不制造生产规模文件树。"""

from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_orig80k_sources as sources
import check_subset_eq as subset_check
from test_orig80k_sources import configure_mini
from test_orig80k_sources import make_manifest
from test_orig80k_sources import reindex
from test_orig80k_sources import write_json


def sample(episode, step):
    result = dict.fromkeys(subset_check.REQUIRED_SAMPLE_KEYS, "子目标")
    result.update(image=np.zeros((2, 2, 3), dtype=np.uint8), wrist_image=np.ones((2, 2, 3), dtype=np.uint8),
                  actions=np.zeros((2, 2), dtype=np.float32), state=np.zeros(2, dtype=np.float32),
                  prompt=f"{episode['h5_file']}#{episode['raw_ep_idx']}",
                  is_demo=np.asarray([False], dtype=np.bool_),
                  exec_start_idx=np.asarray([episode["exec_start_idx"]], dtype=np.int32),
                  step_idx=np.asarray([step], dtype=np.int32),
                  epis_idx=np.asarray([episode["global_episode_idx"]], dtype=np.int32))
    return result


def write_library(root, manifest, *, protocol):
    write_json(root / "meta/episode_manifest.json", manifest)
    data = root / "source/data"
    data.mkdir(parents=True)
    for episode in manifest["episodes"]:
        directory = root / "source/features" / f"episode_{episode['global_episode_idx']}"
        directory.mkdir(parents=True)
        write_json(directory / "kept_indices.json", [[0, 0, 1]])
        for step in range(episode["num_timesteps"]):
            np.save(directory / f"token_emb_{step}.npy",
                    {"image_emb_4x4": np.full((1, 2), step, dtype=np.float32),
                     "state_emb": np.asarray([episode["raw_ep_idx"]], dtype=np.float32)})
            if step >= episode["exec_start_idx"]:
                index = episode["exec_sample_offset"] + step - episode["exec_start_idx"]
                (data / f"{index}.pkl").write_bytes(pickle.dumps(sample(episode, step), protocol=protocol))


@pytest.fixture
def libraries(tmp_path, monkeypatch):
    configure_mini(monkeypatch)
    full, subset = tmp_path / "full", tmp_path / "subset"
    write_library(full, make_manifest(sources.FULL_TASKS), protocol=4)
    subset_manifest = make_manifest(sources.COUNTING_TASKS)
    subset_manifest["episodes"].reverse()
    write_library(subset, reindex(subset_manifest), protocol=5)
    return full, subset


def test_complete_reading_and_cli_parser(libraries, capsys):
    full, subset = libraries
    assert subset_check.main(["--full", str(full), "--subset", str(subset)]) == 0
    output = capsys.readouterr().out
    assert "SUBSET_EQ=PASS episodes=2 samples=4 feature_mismatch=0 sample_mismatch=0" in output


@pytest.mark.parametrize("change", ["wrist", "dtype", "shape", "signed_zero", "epis_idx", "step_idx",
                                   "exec_start_idx", "is_demo", "extra_key", "missing_key", "identity_shape"])
def test_sample_changes_are_not_hidden(libraries, change):
    full, subset = libraries
    path = subset / "source/data/0.pkl"
    value = pickle.loads(path.read_bytes())
    if change == "wrist":
        value["wrist_image"][0, 0, 0] += 1
    elif change == "dtype":
        value["state"] = value["state"].astype(np.float64)
    elif change == "shape":
        value["state"] = value["state"].reshape(1, -1)
    elif change == "signed_zero":
        value["state"][0] = -0.0
    elif change in ("epis_idx", "step_idx", "exec_start_idx"):
        value[change][0] += 1
    elif change == "is_demo":
        value["is_demo"][0] = True
    elif change == "extra_key":
        value["新增键"] = 1
    elif change == "missing_key":
        del value["wrist_image"]
    else:
        value["epis_idx"] = value["epis_idx"].reshape(1, 1)
    path.write_bytes(pickle.dumps(value))
    with pytest.raises(ValueError, match="原始字节|dtype/shape|身份映射|执行样本|键集合|必要字段|身份字段"):
        subset_check.check_subset(full, subset)


@pytest.mark.parametrize("change", ["feature", "feature_header", "kept", "missing_feature", "extra_feature",
                                   "missing_pkl", "extra_pkl", "pkl_trailing"])
def test_file_coverage_and_bytes(libraries, change):
    full, subset = libraries
    directory = subset / "source/features/episode_0"
    feature = directory / "token_emb_0.npy"
    if change == "feature":
        value = np.load(feature, allow_pickle=True).item()
        value["state_emb"][0] += 1
        np.save(feature, value)
    elif change == "feature_header":
        with feature.open("ab") as stream:
            stream.write("额外容器字节".encode())
    elif change == "kept":
        (directory / "kept_indices.json").write_text("[]")
    elif change == "missing_feature":
        feature.unlink()
    elif change == "extra_feature":
        (directory / "token_emb_99.npy").write_bytes("多余".encode())
    elif change == "missing_pkl":
        (subset / "source/data/0.pkl").unlink()
    elif change == "extra_pkl":
        (full / "source/data/999.pkl").write_bytes("多余".encode())
    else:
        with (subset / "source/data/0.pkl").open("ab") as stream:
            stream.write(b"trailing")
    with pytest.raises(ValueError, match="feature 字节不符|缺少产物|多余产物|pkl 尾部"):
        subset_check.check_subset(full, subset)


def test_feature_failure_reports_numerical_delta(libraries, capsys):
    full, subset = libraries
    path = subset / "source/features/episode_0/token_emb_0.npy"
    value = np.load(path, allow_pickle=True).item()
    value["state_emb"][0] += 1
    np.save(path, value)
    assert subset_check.main(["--full", str(full), "--subset", str(subset)]) == 1
    error = capsys.readouterr().err
    assert "SUBSET_EQ=FAIL" in error
    assert "state_emb:max_abs_diff=1" in error


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_total"])
def test_subset_manifest_cannot_silently_take_intersection(libraries, change):
    full, subset = libraries
    path = subset / "meta/episode_manifest.json"
    manifest = sources.read_manifest(path)
    if change == "missing":
        manifest["episodes"].pop()
        reindex(manifest)
    elif change == "duplicate":
        manifest["episodes"][1]["raw_ep_idx"] = manifest["episodes"][0]["raw_ep_idx"]
    else:
        manifest["totals"]["exec_samples"] += 1
    manifest["sha256"] = sources.manifest_sha256(manifest)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="每任务|身份重复|totals"):
        subset_check.check_subset(full, subset)


def test_real_cli_refuses_reduced_production_scope(libraries):
    # 子进程不继承 monkeypatch，固定生产计数仍生效。
    full, subset = libraries
    result = subprocess.run([sys.executable, str(Path(subset_check.__file__)), "--full", str(full),
                             "--subset", str(subset)], capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 1
    assert "SUBSET_EQ=FAIL" in result.stderr


def test_recursive_values_use_raw_bits():
    a = {"nested": [np.asarray([0.0], dtype=np.float32), ("文本", np.int32(2))]}
    b = {"nested": [np.asarray([-0.0], dtype=np.float32), ("文本", np.int32(2))]}
    with pytest.raises(ValueError, match="原始字节"):
        subset_check.compare_value(a, b, "sample")
    with pytest.raises(ValueError, match="原始字节"):
        subset_check.compare_value(0.0, -0.0, "scalar")
