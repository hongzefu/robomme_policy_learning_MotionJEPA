# modulation 关闭态追溯梯度对拍

本运行实施用户已批准的 `v2-1600ep-m8x8-modul-training-plan.md` J 节。F3 已完整通过，见 [守卫验证结果](../t8-c8-guard-s100/result.md)。记录名 `m8-modul-retro`，只计算固定初态上的前向与反向，不执行 optimizer 更新，不产出训练 checkpoint。

## 命题与边界

参考源码 A 固定为 `07702f0076e640724e9516e911983d7810b423d3`；候选 B 为包含本工具和本留档的 clean HEAD，完整 SHA 由 runner 的 `CAND=`、各摘要的 `source.head` 固化。两侧模型与训练配置来自各自源码树，工具、`train.py::init_train_state` 和取梯度辅助逻辑共用主副本；本步骤不修改任何 `src/` 文件。

按 c32（32 帧 4×4）、c8（8 帧 8×8）顺序，各执行 A1→A2→B。先比较 A1/A2 的输入、全部初态参数摘要、loss 和全部可训练梯度；A/A 不过即退出，不启动 B、不降级容差。每档 B 完成后再比较 A1/B。两档使用同一对物理 GPU 4,5，seed 42、batch 8、fsdp 2，确定性 XLA 选项相同，每次使用独立 JAX 缓存。预计整组需要数十分钟，实际时间由日志记录。

```text
INIT_EQ=PASS leaves=<实际参数叶数> mismatches=0
GRAD_EQ=PASS kinds=3 leaves=<实际可训练叶数> mismatches=0
RETRO_PROFILE=PASS profile=c32
RETRO_PROFILE=PASS profile=c8
EXIT_CODE=0
```

两档共用模型代码路径，价值是覆盖两组不同的真实数值。只验证固定输入下的模型数值，不覆盖帧采样、查表装配或 1600 集新库；这些分别由已有 t8 验收、本轮 F2 和 F3 覆盖。不能用本对拍单独证明整个 1600 集训练链等价。

## 固定输入与同源性

复用 `v1-store/fixtures/8x8/grad/c32-b` 和 `c8-b`，每档 `mixed1`、`allshort`、`allfull` 三组。每组按 `batch_meta.json` 只加载 12 个数组键，四个 None 的 motion 键不生成张量。`_common.load_array` 读取现有 `.bin`/JSON 容器，新 loader 逐键再核 dtype、shape、raw 和 canonical SHA256，重建嵌套键后再核完整键集。原 fixture 不写入、不重新生成。

四个 static 张量在两档均为：`static_image_emb [8,512,2048] bfloat16`（16777216 字节），`static_pos_emb [8,512,768] float32`（12582912 字节），`static_state_emb [8,512,8] float64`（262144 字节），`static_mask [8,512] bool`（4096 字节）。搬运到 JAX 后的原有 float64→float32 降精度由两侧同一工具执行；本次不修改该语义。

初态在 `init_train_state` 后逐叶记录全部 `state.params` 的 dtype、shape、raw/canonical SHA256，并检查有限性。比较器必须先确认两侧初态精确相同，再比较 loss hex 和全部梯度 SHA256；拒绝空树、缺少 batch、输入摘要变化和环境不一致。没有 modulation 黄金基线，禁止设置 `DTYPE_BASELINE_CHECKSUMS`。

两侧 YAML 的实际字节 SHA256：

| 配置 | SHA256 |
|---|---|
| `perceptual-framesamp-modul.yaml` | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| `perceptual-framesamp-modul-8frame-8x8.yaml` | `5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec` |

历史树缺少 8×8 YAML，仅复制这份配置数据作为未跟踪文件。工具逐次验证允许的工作区状态、源码 SHA、四个核心模块实际导入路径与两侧 YAML 字节。软件来自主副本现有 `.venv`，两侧均 `uv run --no-sync`，不创建或同步历史树环境。

## 启动命令与产物

参考树为 `v1-store/worktrees/s2-base`，由 `git worktree add --detach` 创建；仅用于本对拍。GPU 4,5；存储 AWS 本地 NVMe RAID（`/dev/md0`）。权重显式使用 `v1-store/models/openpi-assets/checkpoints/pi05_base/params`。虽然命令带 Dataset/assets 参数以保持配置可复现，本工具不调用 Dataset。

完整命令与环境见 [records/runner.sh](records/runner.sh)。本次新增 tmux 会话清单仅 **`tr-m8-modul-retro`**；先前的 `m8-guard` 已正常结束。所有其他会话均不触碰。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
CAND=$(git rev-parse HEAD)
tmux new-session -d -s tr-m8-modul-retro "bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8-retro-runner.sh '$CAND'"
tmux has-session -t tr-m8-modul-retro
```

日志为 `v1-store/logs/m8-modul-retro.log`；摘要为 `v1-store/bench/m8-modul-retro/{c32,c8}-{a1,a2,b}/{init_params,grad_summary}.json`；独立缓存为 `v1-store/cache/jax/m8-modul-retro-<profile>-<side>`。日志使用 `PYTHONUNBUFFERED=1`、内层严格失败、外层 `pipefail`+`tee` 与真实 `EXIT_CODE`。两侧环境记录含 uv.lock SHA、包版本、GPU 名称/驱动、可见卡号及 JAX x64 开关。

代码验证已通过：两档共六个真实 batch 原摘要读回、`_grad_only` 去除文档串后的 AST 与既有工具相同、六类比较器负例被拒、历史树四个核心模块确实从 `s2-base/src` 导入。主副本 diff 空白检查和 runner Bash 语法检查通过。GPU 数值结果待本次运行后写入 `result.md`。

运行结束后归档各侧摘要、清洗日志和判定行，不归档模型权重。确认参考树只有本轮复制的 YAML 后，移除该文件，再以普通 `git worktree remove` 清理参考树；不改动其他 worktree。A/A 或 A/B 的任一差异均停下交用户裁决。
