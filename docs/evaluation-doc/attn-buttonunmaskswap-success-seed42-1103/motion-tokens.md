# 完整 motion token 记录与后续比较入口

这里的模型名称固定为**带 motion token 的 VLA**；**motion token** 指它的运动编码输入。另一组是不带 motion token 的 VLA，没有这一路输入，不为它生成替代的零向量。

本轮带 motion token 的 VLA 的 12 个回合全部保留了以下两类内容；索引见 [motion-token-inventory.json](records/motion-token-inventory.json)。

| 层次 | 本轮实测数量 | 保存的内容 |
|---|---:|---|
| 编码器生成历史 | 130 次调用、130 个不同的“回合与窗口起点”组合 | 完整 1024 维编码、起点、调用顺序；格式也支持保留同起点重复调用 |
| VLA 实际消费快照 | 279 次推理，累计 1531 次有效 token 使用 | 每次完整 `motion_emb=(96,1024)`、`motion_pos=(96,256)`，实际 dtype 为 `float32`，以及 mask、时刻和顺序映射 |

96 是槽位预算，不是每次有效 token 数。开局推理尚没有完成的 33 帧窗口，实际有效数量为 0；这与后续将有效 token 的内容换成零向量是不同情形。

## 文件与字段

原始文件在本机运行根：

```text
/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/
```

| 文件或字段 | 含义 |
|---|---|
| `with_motion/motion_records/ButtonUnmaskSwap_ep<局号>.npz` | 每局全部生成编码的兼容供体文件 |
| `start_frames`、`embs`、`counts` | 旧供体接口：每个起点首次编码及该起点的调用次数 |
| `call_start_frames`、`call_embs`、`call_seq` | 新增完整调用轨迹，按实际先后排列，不按起点去掉重复调用 |
| `with_motion/attn/<episode_nonce>/infer_<seq>.npz` | 这次推理的真实输入、动作和原始注意力 |
| `motion_emb`、`motion_pos` | 实际输入的编码和位置，保留完整槽位及原精度，不是注意力权重 |
| `motion_mask`、`motion_times` | 哪些槽位有效、各自窗口起点；无效位置的时刻为 −1 |
| `mem_order`、`mem_mask` | 交错后的实际记忆区 key 顺序和有效位 |
| 索引的 `valid_slots`、`prefix_key_positions` | 该次推理中有效 motion 槽与 attention key 的一一对应 |

每个 motion token 对应 33 帧窗口 `[start,start+33)`，本模型实际生成网格的 stride 为 32。分析交换区间时仍按重叠帧数／33 计权，不把沾边窗口整窗算入。

记忆区前 512 个原始编号属于帧流，motion 槽 `j` 的原始编号是 `512+j`。它实际交错后的 key 位置 `s` 满足 `mem_order[s] == 512+j`；应同时检查有效掩码。索引已逐次保存该映射，不能把槽位编号直接当作交错后的 key 位置，也不能把 `L3H0` 这样的层头编号当成 token 编号。

## 只读查看示例

在仓库根使用现有 uv 环境读取；以下只查看数据，不执行策略或替换实验。

```bash
UV_CACHE_DIR=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/cache/uv \
uv run --frozen --no-sync python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103")
inventory = json.loads((root / "motion-token-inventory.json").read_text())
episode = next(e for e in inventory["episodes"] if e["episode_id"] == 8)
with np.load(episode["motion_record"]) as generated:
    print(generated["call_start_frames"])
    print(generated["call_embs"].shape)

point = next(p for p in episode["infer_inputs"] if p["valid_slots"])
with np.load(point["file"]) as actual:
    slot = point["valid_slots"][0]
    print(actual["motion_times"][slot])
    print(actual["motion_emb"][slot].shape)
    print(point["prefix_key_positions"][0])
PY
```

后续可以按“局身份、推理时刻、窗口起点”选择向量，分别明确替换内容与保留的位置、mask、顺序。全部原始注意力仍保留去噪步、层、头和动作 query 维度，当前索引没有对原始数组平均或只保留排名靠前的 token。

索引里的文件 sha256 用于核对本次生成文件的拷贝完整性；实际输入向量指纹采用 `sha256(dtype 文本 + shape 文本 + 连续数组原始字节)`。这些是本次输出数据的指纹，不替代 [启动档案](launch.md) 中已经独立核实的 checkpoint 和编码器资产身份。
