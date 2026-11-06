"""表征之间的三跳：帧吸收 motion、文字读帧、动作吸收。只读 Swap 的 action-flow-summary.json，事后补做。

用法：chain_hops.py <action-flow-summary.json>。
数据是两个 Swap 任务闭合前 4 次推理，motion 模型；PatternLock、RouteStick 没有 prefix，帧读 motion 这一跳未测。
累积与路径是注意力近似，忽略 MLP 和 value 范数。
"""
import json
import sys
import numpy as np
np.set_printoptions(precision=1, suppress=True, linewidth=200)
data = json.load(open(sys.argv[1], encoding="utf-8"))
FRAME, MOTION, IMAGE, TEXT, ACTION = list(range(4)), list(range(4, 8)), [8, 9], 10, 11
for key in ("motion|ButtonUnmaskSwap", "motion|VideoUnmaskSwap"):
    cell = data["cells"][key]
    direct, counts = np.array(cell["direct_layer_pct"]), np.array(cell["token_counts"], float)
    weight = counts[FRAME] / counts[FRAME].sum()
    carry = np.array(cell["alpha_0.5"]["carry_by_layer"])
    frame_carry = np.array([(weight[:, None] * carry[layer][FRAME][:, MOTION]).sum() for layer in range(19)])
    text_frame = direct[:, TEXT][:, FRAME].sum(1)
    print("==", key, "帧 token", counts[FRAME].sum(), "motion token", counts[MOTION].sum().round(1))
    print("帧读 motion%", np.array([(weight[:, None] * direct[layer][FRAME][:, MOTION]).sum() for layer in range(18)]))
    print("帧累积 motion%（第 l 层输入）", frame_carry)
    print("文字读帧%", text_frame)
    print("文字读 motion%", direct[:, TEXT][:, MOTION].sum(1))
    print("文字经帧间接拿到 motion%（读帧 × 帧含 motion）", text_frame / 100 * frame_carry[:18])
    print("文字累积 motion%", carry[:, TEXT][:, MOTION].sum(1))
    print("动作读文字%", direct[:, ACTION][:, TEXT])
    print("动作读帧%", direct[:, ACTION][:, FRAME].sum(1))
    print("动作读 motion%", direct[:, ACTION][:, MOTION].sum(1))
    for alpha in ("0.2", "0.5", "0.8"):
        via = np.array(cell[f"alpha_{alpha}"]["route_via_source"])[:, MOTION].sum(1)
        print(f"alpha={alpha} 动作里来自 motion 的量 {via.sum():.2f}%：经帧 {via[FRAME].sum():.2f} 经文字 {via[TEXT]:.2f} 经图像 {via[IMAGE].sum():.2f} 经 motion {via[MOTION].sum():.2f}")
