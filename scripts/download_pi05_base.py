"""下载 pi05 初始权重。

权重落点显式收敛到 v1-store/models（AGENTS 14 单一产物根）：只设 OPENPI_DATA_HOME、
禁止覆盖 HOME（覆盖 HOME 会让 ssh 找不到 ~/.ssh 配置、打断集群提交）。
调用方已显式设置 OPENPI_DATA_HOME 时以调用方为准。
"""

import os
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}")
os.environ.setdefault("OPENPI_DATA_HOME", str(_REPO_ROOT / "v1-store" / "models"))

from openpi.shared import download  # noqa: E402

download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
