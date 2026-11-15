"""BinFill demo 前缀的读取端（0922-binfill-demo-prefix-plan.md B 节）。

前缀由 ``scripts/evaluation/pregen_binfill_demo.py`` 离线生成：对同一条候选（同 seed、同 spec）
另建一个 env，把 ``task_list`` 的 ``demonstration`` 全部翻成 True 让 planner 演完整条，
只取画面、演完即销毁；正式评测再建一个干净 env（``demonstration`` 仍为 False）交给策略。
本模块只负责把那批帧原样读回来，不做任何数值加工。

**为什么必须无损**：front 帧同时进 SigLIP（224 域）与 Wan VAE（256 域），有损压缩会改变
编码结果。故 front 走 PNG，wrist 走 JPEG q90（只进 mp4 录像，不进任何模型）。
``front_raw_sha256`` 在写入时按解码后的逐帧字节算出，这里再算一遍逐位核对——
解码链换了版本（cv2 升级、libpng 换实现）就会当场炸，不会带着悄悄变了的像素跑满 150 集。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


def episode_key(task: str, difficulty: str, episode: int) -> str:
    return f"{task}/{difficulty}/{int(episode)}"


def frames_sha256(frames) -> str:
    """对一串帧按顺序算 sha256（逐帧 ``tobytes()``，C 连续）。写入端与读取端共用本函数。"""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


class DemoPrefixStore:
    """按 ``(task, difficulty, episode)`` 取一条 demo 前缀。

    **缺条目直接 raise，绝不静默降级成「无 demo 跑」**——那会在结果里混进一批与其余集
    输入条件不同的 episode，而且事后无从分辨。planner 失败的集在预生成阶段就已从计划里剔除，
    正常路径上不会缺。
    """

    def __init__(self, root: str):
        self.root = Path(root)
        index_path = self.root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"demo 前缀库缺 index.json：{index_path}")
        self.index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
        manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {})
        print(f"DEMO_STORE root={self.root} entries={len(self.index)} "
              f"identity={self.manifest.get('identity_sha256', '')[:12]}", flush=True)

    def has(self, task: str, difficulty: str, episode: int) -> bool:
        return episode_key(task, difficulty, episode) in self.index

    def get(self, task: str, difficulty: str, episode: int) -> dict[str, Any]:
        key = episode_key(task, difficulty, episode)
        entry = self.index.get(key)
        if entry is None:
            raise KeyError(f"demo 前缀库里没有 {key}（库根 {self.root}）")
        if entry.get("demo_status") != "ok":
            raise ValueError(f"{key} 的 demo_status={entry.get('demo_status')!r}，不是 ok，不得用于评测")
        path = self.root / entry["path"]
        with h5py.File(path, "r") as h5:
            attrs = dict(h5.attrs)
            front = [cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
                     for buf in h5["front_png"]]
            wrist = [cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
                     for buf in h5["wrist_jpg"]]
            states = np.asarray(h5["state"][...], dtype=np.float32)

        d = int(attrs["D"])
        if not (len(front) == len(wrist) == states.shape[0] == d):
            raise ValueError(
                f"{key} 三个缓冲长度不一致：front={len(front)} wrist={len(wrist)} "
                f"state={states.shape[0]} D={d}")
        front = [np.ascontiguousarray(f, dtype=np.uint8) for f in front]
        wrist = [np.ascontiguousarray(f, dtype=np.uint8) for f in wrist]
        for name, frames in (("front", front), ("wrist", wrist)):
            bad = [i for i, f in enumerate(frames) if f.shape != (256, 256, 3)]
            if bad:
                raise ValueError(f"{key} 的 {name} 第 {bad[:4]} 帧形状不是 (256,256,3)")
        # front 必须逐位还原：它同时进 SigLIP 与 Wan VAE，一个像素的差都算换了输入
        actual = frames_sha256(front)
        if actual != str(attrs["front_raw_sha256"]):
            raise ValueError(
                f"{key} 的 front 解码结果与写入时不符：{actual[:12]} != {str(attrs['front_raw_sha256'])[:12]}")
        return {
            "images": front,
            "wrist_images": wrist,
            "states": [states[i] for i in range(d)],
            "task_goal": str(attrs["task_goal"]),
            "D": d,
            "attrs": attrs,
        }
