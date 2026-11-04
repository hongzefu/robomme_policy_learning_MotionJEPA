# ButtonUnmaskSwap 成功回合细致注意力分析启动档案

## ① 目的与用户决定

按[完整计划](../../../1103-buttonunmaskswap-success-attention-plan.md)，只分析开局帧窗口与 motion token，沿用既有逐层逐头、时间线、去注意力汇聚点及间接通路的方法，扩展到两种 VLA 各自 seed=42 下曾成功的全部入选回合。完成采集与分析分别验收，不将正常终局失败当作基础设施错误。

关键用户原话按顺序保留：

1. 「重新做buttonunmaskswap的注意力分析」「你现在需要把所有成功的任务都跑一遍。」「然后你再分析一下，有没有什么共通的pattern。」
2. 「还是只关注这两个窗口，就一个是一开始的那个帧和这个运动的这个motion这两个地方。然后你这次跑的时候你需要把所有的motion的token都记录下来，为之后做切换不同的motion token做一个比较。」
3. 「沿用现有报告的 12 个回合（推荐）」：确认带 motion token 的 VLA 的 seed42 选局范围。
4. 「然后这个不带 motion 的模型你也要跑，也是只跑它成功的任务。」
5. 「（带 motion token 的 VLA）」：据此将两个模型统一称为“带 motion token 的 VLA”“不带 motion token 的 VLA”。
6. 「把这个任务写入Markdown。」
7. 「开始实现」：在计划保存推送后给予开工令。

本任务没有登记交接状态机；没有状态机登记授权，不修改 `docs/handoff/current.json`。

## ② 环境与版本

控制与运行主机为 `sled-aspen.eecs.umich.edu`，本机 `/data` 存储布局，实际 GPU 为 2×NVIDIA RTX A6000，每卡 46068 MiB。使用本机运行，不使用 Great Lakes。只查询过一次当前利用率与已有计算进程，已有其他用户进程不操作、不归因、不关闭。

开工基线：`b7ad0560e4315a8b1ecb669a86cb584e0eff68d7`，分支 `v4.1-interpretMotion`。原计划保存提交 `149d214`；至开工基线，计划、`scripts/interpret/` 和相关 VLA 源码未发生变化。

正式执行从干净工作树 `/home/hongzefu/.codex/worktrees/button-attn-run/robomme_policy_learning_MotionJEPA` 启动，整合后的实际执行提交在起跑前补记。主检出保留用户的未跟踪 HTML；不将它提交、移动或删除。

只读复用解释器：策略为主检出 `.venv/bin/python`；仿真为 `v1-store/envs/official-sim/bin/python`；编码器服务为 `v1-store/envs/stage2/bin/python`。不在运行工作树重新安装依赖。VLA 与 openpi 的实际导入已经核对来自运行工作树；benchmark 导入来自主检出锁定子模块，不初始化第二份源码。最终提交固定后再次核对路径和干净状态。

## ③ 被跑对象、清单与还原入口

| 对象 | 固定配置与本机资产 |
|---|---|
| 带 motion token 的 VLA | `mme_vla_suite_orig16_b128_60k`，`model_kind=stage2_pool4_concat`，`v1-store/assets/vla-o16cc-59999/59999` |
| 不带 motion token 的 VLA | `mme_vla_suite_b128`，`model_kind=history`，`v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` |
| 编码器 | checkpoint 配套 `encoder_dependencies/stage2-full144-1025-a`；resolved 配置锚定 sha256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035` |
| 运行合同 | 沿用 `v1-store/evaluation/interpret-1103/contracts/runtime-mot.json`，跨硬件数值逐位一致不作为该旧合同要求 |
| tokenizer | 本地 `v1-store/models/big_vision/paligemma_tokenizer.model`；离线读取，不触发下载 |

本轮资产核对已经完成：带 motion token 的 VLA 的 step59999 manifest 中 38 个文件逐个核对字节数与全量 sha256，manifest sha256 为 `bd35a857e8d5134068539ab03cd01c03c67b61aa22c942477b44df1fe8f97f08`；不带 motion token 的 VLA 的来源 zip sha256 为 `387d4bd500a29588d81c2675a83be351f84383b93bea4161b8f7fadbf91a8ae8`，再逐个比对其内 17 个文件与当前展开权重的内容。证据为运行根的 `assets-verified.json`，不是只检查文件存在。

默认配置还原使用实际执行提交的 `git show <执行提交>:src/mme_vla_suite/training/config.py`，并结合各 checkpoint 配套的 history 配置；本轮不修改模型参数。驱动为 `scripts/interpret/run_local.py`，采集为 `scripts/interpret/serve_policy_attn.py`。最终完整命令和 spec 覆盖项在起跑前追加。

## ④ 任务、回合与 seed

任务只有 `ButtonUnmaskSwap`。模型 seed 固定为 42，环境 seed 与难度逐局取 benchmark test 元数据；不把模型 seed 与环境 seed 混用。

| 模型 | 入选回合 |
|---|---|
| 带 motion token 的 VLA | 4、8、11、12、17、18、20、26、34、37、38、40，共 12 局 |
| 不带 motion token 的 VLA | 8、12、14、16、18、20、26、34、38、47，共 10 局 |

不带 motion token 的 VLA 的 9 个完整历史评估成功局与后续正常组第 38 局取并集已现场复核。两组 manifest 均由既有 `build_manifest.py::build` 生成，逐条携带环境身份和元数据哈希；交换次数由 `aggregate_attn.py::compute_swap_times` 结合该局 seed、difficulty 与环境配置推得，没有将三次交换裁成两次。

`SUCCESS_MANIFEST=PASS models=2 episodes=22`。

清单封签：带 motion token 的 VLA 为 `c5ac2704a666c57ec54571c6f3e4b20ca2102e89f63efef55b1a3320769beec0`；不带 motion token 的 VLA 为 `9fc079ca07a17ac837bf95125ffda5c110f399f365b26ecbb82e4ecd9f82963f`。运行根含 `manifest-with_motion.json`、`manifest-without_motion.json` 和 `selection.json`。

## ⑤ 采集与分析口径

只做正常运行，`ablate=none`。开启完整动作／文本 attention 与 prefix attention；带 motion token 的 VLA 还记录每次实际输入的 `motion_emb`、`motion_pos` 和已有 mask、时间及顺序映射。生成历史保留所有调用，包括同起点重复调用，不只留首次编码或排名靠前的 token。

开局窗口仍为 `[0,32)`，第 0 帧单列；motion 窗按重叠帧数／33 计权。逐局核对靠近、抓取、抬起阶段，不套固定推理编号；原始数据不平均，离线先逐局再跨局等权汇总。无运动输入的指标记不适用，阶段缺失如实报告。

## ⑥ 资源、路径、会话与起跑闸门

新运行名：`attn-buttonunmaskswap-success-seed42-1103`。所有权重、缓存、原始数据和日志都在本机主检出 `v1-store/`，运行根为 `v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/`。各模型、局分别有独立目录，不覆盖旧实验。实际运行工作树保持只读，产物不落该工作树。

最多使用两张 GPU，计划带 motion token 的 VLA 用物理 GPU1、不带 motion token 的 VLA 用物理 GPU0，沿用已验证的 XLA 显存比例 0.55／0.40。运行前不停止其他用户进程，首次冒烟验证同机当前资源是否足够。原始总量为 22 条组合；两次通过的单局冒烟计入各自清单，不重复运行。

本轮运行 tmux 会话清单将在启动前固定并写入；当前尚未启动任何 GPU 运行会话。基础设施错误不自动更换配置；正常任务失败不重跑。启动顺序为代码整合与审查、定向验证、两种 VLA 各一局冒烟、核对全部采集字段及动作不变、再补齐其余清单。

独立开工审查：`PLAN_CONSISTENCY_REVIEW=PASS base=b7ad0560e4315a8b1ecb669a86cb584e0eff68d7 findings=0 blocking=0`。执行位置和测试夹具调整均已独立审查通过，详见计划“实施追加记录”；没有改变模型、22 条组合或验收。
