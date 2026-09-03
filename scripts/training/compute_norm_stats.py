"""计算 state/actions 的 norm stats（唯一生产链）。

commitV4.1 起不再依赖已删除的 legacy `RoboMMEDataset`，改为内联最小 pkl 读取器
`_PklSampleDataset`。与旧链路（`RoboMMEDataset(history_config=None,
compute_norm_stats=True)`）的等价性要点：
① `__len__` 读 `meta/stats.json`，`execution_samples` 优先、否则 `total_samples`；
② 裸 `pickle.load(data/{idx}.pkl)`，无任何转换；
③ `actions[:action_horizon]` 截断；
④ 不建 mem_buffer、不读 features（旧代码 `history_config=None` 即如此）；
⑤ state 保持原始值不归一化（`compute_norm_stats=True` 语义）；
⑥ 旧代码的 random 分支全不触发；
⑦ `*_online` pop 省略（RepackTransform 按白名单丢弃未列键，等价）；
⑧ 尾部补 None 键集合与当前 `framesamp_dataset._NONE_KEYS` 全集一致。

⚠ `--output-dir` 为必填：生产 `norm_stats.json` 是 G0 基线环境指纹项，
默认写生产路径会让任何验证/试跑静默覆盖基线。写生产路径必须显式给出。
"""

import tqdm
import tyro
import numpy as np
import dataclasses
import json
import os
import pickle
import pathlib

import openpi.transforms as transforms
import openpi.shared.normalize as normalize
from openpi.training.data_loader import Dataset, TransformedDataset, TorchDataLoader

import mme_vla_suite.training.config as _config


# 与 mme_vla_suite/training/framesamp_dataset.py 的 _NONE_KEYS 全集保持一致
# （等价性要点⑧；transforms 键集收敛时两处必须同 commit 同步）
_NONE_KEYS = (
    "static_image_emb",
    "static_pos_emb",
    "static_state_emb",
    "static_mask",
    "prompt",
    "motion_emb",
    "motion_pos",
    "motion_mask",
    "mem_order",
)


class _PklSampleDataset(Dataset):
    """最小 pkl 读取器：等价于旧 RoboMMEDataset 在 history_config=None、
    compute_norm_stats=True 下的行为（八要点见模块 docstring）。"""

    def __init__(self, dataset_path: str, action_horizon: int):
        self.dataset_path = dataset_path
        self.action_horizon = action_horizon
        with open(os.path.join(self.dataset_path, "meta", "stats.json")) as f:
            self.stats = json.load(f)

    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        return self.stats["total_samples"]

    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path, "data", f"{idx}.pkl"), "rb") as f:
            data = pickle.load(f)
        data["actions"] = data["actions"][: self.action_horizon]
        for key in _NONE_KEYS:
            if key not in data:
                data[key] = None
        return data


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
):
    dataset = _PklSampleDataset(dataset_path, action_horizon)

    dataset = TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ])
    print(f"Dataset length: {len(dataset)}, batch size: {batch_size}")

    num_batches = len(dataset) // batch_size
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=None,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        shuffle=True,
    )
    return data_loader, num_batches



def main(
    output_dir: str,
    config_name: str = "mme_vla_suite",
    repo_id: str = "robomme",
    dataset_path: str = "data/robomme_preprocessed_data",
):
    """计算并写出 norm stats。

    output_dir 必填（防误覆盖 G0 指纹项）：验证/试跑给临时目录；
    确要更新生产 stats 时显式给生产 assets 路径。
    """
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader, num_batches = create_data_loader(
        dataset_path=dataset_path,
        data_config=data_config,
        action_horizon=config.model.action_horizon,
        batch_size=128,
        num_workers=4,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    print(f"norm_stats: {norm_stats}")

    output_path = pathlib.Path(output_dir) / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
