"""P0补丁可逆性与来源/依赖拒绝条件，不安装环境。"""

import subprocess

import prepare_orig80k_env as env
import pytest


def test_workspace_patch_only_changes_missing_member_and_reverses(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    original = '[project]\nname = "example"\n\n[tool.uv.workspace]\n' + env.OLD_MEMBER + "\n"
    target = tmp_path / "pyproject.toml"
    target.write_text(original)
    patch = tmp_path / "test.patch"
    patch.write_text(env.workspace_patch(original))
    for extra in (["--check"], []):
        subprocess.run(["git", "apply", *extra, str(patch)], cwd=tmp_path, check=True)
    assert target.read_text() == original.replace(env.OLD_MEMBER, env.NEW_MEMBER)
    for extra in (["--reverse", "--check"], ["--reverse"]):
        subprocess.run(["git", "apply", *extra, str(patch)], cwd=tmp_path, check=True)
    assert target.read_text() == original


@pytest.mark.parametrize("text", ["", env.OLD_MEMBER + "\n" + env.OLD_MEMBER])
def test_ambiguous_project_rejected(text):
    with pytest.raises(ValueError, match="声明"):
        env.workspace_patch(text)


@pytest.mark.parametrize("fault", ["dependency", "python", "origin", "prefix"])
def test_environment_mismatch_rejected(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(env, "ROOT", tmp_path)
    worktree = tmp_path / "v1-store/worktrees/upstream"
    current = {"python": "same", "packages": {"jax": "same"},
               "modules": {"openpi": {"path": str(tmp_path / "src/openpi/__init__.py")}}}
    upstream = {"python": "same", "packages": {"jax": "same"},
                "prefix": str(worktree / ".venv"),
                "modules": {"openpi": {"path": str(worktree / "src/openpi/__init__.py")}}}
    env.validate_probes(upstream, current, worktree)
    if fault == "dependency":
        upstream["packages"]["jax"] = "different"
    elif fault == "python":
        upstream["python"] = "different"
    elif fault == "origin":
        upstream["modules"] = current["modules"]
    else:
        upstream["prefix"] = str(tmp_path / ".venv")
    with pytest.raises(ValueError, match="不一致|来源污染|独立venv"):
        env.validate_probes(upstream, current, worktree)
