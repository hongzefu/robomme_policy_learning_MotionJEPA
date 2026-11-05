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

初始写档时尚未启动 GPU 运行会话，实际启动、完成和验收记录见下文。基础设施错误不自动更换配置；正常任务失败不重跑。启动顺序为代码整合与审查、定向验证、两种 VLA 各一局冒烟、核对全部采集字段及动作不变、再补齐其余清单。

独立开工审查：`PLAN_CONSISTENCY_REVIEW=PASS base=b7ad0560e4315a8b1ecb669a86cb584e0eff68d7 findings=0 blocking=0`。执行位置和测试夹具调整均已独立审查通过，详见计划“实施追加记录”；没有改变模型、22 条组合或验收。

## 起跑前确定的规格与完整命令

运行根以下用 `R=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103` 表示。所有命令的实际代码提交以起跑前写出的 `runtime-fixed.json` 为准，运行工作树必须处于该提交且 `git status --porcelain` 为空。

两个 spec 的全部共同覆盖项为：`repo` 指向上述运行工作树；三个解释器使用第二节的绝对路径；`model_seed=42`、`purpose=validation`、`save_prefix=true`、`ablate=none`、`startup_timeout_seconds=900`、`rpc_timeout_seconds=600`。每组 `manifest`、`attn_out`、`run_root` 分别为 `R/manifest-<组>.json`、`R/<组>/attn`、`R/<组>/run`。组名是路径标识符 `with_motion`、`without_motion`，用户可见名称仍为本档案的两个固定名称。

带 motion token 的 VLA 使用 `cuda=1`、`xla_mem_fraction=0.55`、`record_motion=R/with_motion/motion_records`，并传入第三节已有运行合同；不带 motion token 的 VLA 使用 `cuda=0`、`xla_mem_fraction=0.40`，不传运动记录或运动合同。权重与配置使用第三节的固定对象。

执行入口为主会话在本轮运行根生成的 `run_capture.sh`。它先进入干净工作树，显式将该工作树根、`src`、`packages/openpi-client/src` 和主检出已有 benchmark 源码加入 `PYTHONPATH`；复用主检出的 uv 管理解释器，不同步依赖。缓存环境为：`UV_CACHE_DIR=<主根>/v1-store/cache/uv`、`XDG_CACHE_HOME=<主根>/v1-store/cache`、`HF_HOME=<主根>/v1-store/cache/hf`、`MPLCONFIGDIR=<主根>/v1-store/cache/matplotlib`；`OPENPI_DATA_HOME=<主根>/v1-store/models`、`MMEVLA_V1_STORE=<主根>/v1-store`；`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`WANDB_MODE=disabled`、`PYTHONUNBUFFERED=1`。脚本最终执行 `timeout --signal=TERM --kill-after=30s 4h <主根>/.venv/bin/python -m scripts.interpret.run_local --spec R/spec-<组>.json --episodes <明确局号>`。不自动重试基础设施错误或正常失败；每模型总执行预算上限 4 小时由主会话监督，实际耗时与预算消耗在结果档案回写。

两次冒烟分别为：

```bash
bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/run_capture.sh with_motion ButtonUnmaskSwap:8
bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/run_capture.sh without_motion ButtonUnmaskSwap:8
```

通过采集验收后补齐剩余组合：

```bash
bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/run_capture.sh with_motion ButtonUnmaskSwap:4,ButtonUnmaskSwap:11,ButtonUnmaskSwap:12,ButtonUnmaskSwap:17,ButtonUnmaskSwap:18,ButtonUnmaskSwap:20,ButtonUnmaskSwap:26,ButtonUnmaskSwap:34,ButtonUnmaskSwap:37,ButtonUnmaskSwap:38,ButtonUnmaskSwap:40
bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/run_capture.sh without_motion ButtonUnmaskSwap:12,ButtonUnmaskSwap:14,ButtonUnmaskSwap:16,ButtonUnmaskSwap:18,ButtonUnmaskSwap:20,ButtonUnmaskSwap:26,ButtonUnmaskSwap:34,ButtonUnmaskSwap:38,ButtonUnmaskSwap:47
```

上述命令各自在 detached tmux 中运行，外层使用 `set -o pipefail` 与 `tee` 记录 stdout/stderr，结束追加真实 `EXIT_CODE=`；程序原有退出处理负责清理本次创建的策略与仿真子进程。健康端点只作为服务已加载的证据，首次 `ATTN_INFER` 及完整记录才作为首次推理证据。

本轮预留 tmux 名单与日志如下；实际启动时间、状态与运行提交起跑时写入运行根记录，没有启动的项不能当作已有进程：

| 阶段 | 精确会话名 | 日志（相对 R） |
|---|---|---|
| 带 motion token 的 VLA 冒烟 | `ia-success-smoke-with-motion-1104` | `logs/smoke-with_motion.log` |
| 不带 motion token 的 VLA 冒烟 | `ia-success-smoke-without-motion-1104` | `logs/smoke-without_motion.log` |
| 带 motion token 的 VLA 剩余回合 | `ia-success-all-with-motion-1104` | `logs/all-with_motion.log` |
| 不带 motion token 的 VLA 剩余回合 | `ia-success-all-without-motion-1104` | `logs/all-without_motion.log` |
| 离线分析 | `ia-success-analysis-1104` | `logs/analysis.log` |

只允许主会话按这份清单核实后逐个操作本任务会话，不处理其他用户或任务的会话。

## 起跑版本现场固定

真实执行提交固定为 `e56bbceca5fe1c734c2553e83b8b6de1db71ca08`，运行工作树为上文的 `button-attn-run`。起跑前核对 `git status --porcelain` 为空，实际导入的 `mme_vla_suite`、`openpi`、`openpi_client` 均来自该工作树；benchmark 使用主检出锁定的 `856bc3a189d4172f3f47dbee4424d585f8d78db3`。证据为 [runtime-fixed.json](records/runtime-fixed.json)，`RUNTIME_FIXED=PASS head=e56bbceca5fe1c734c2553e83b8b6de1db71ca08 clean=1`。

带 motion token 的 VLA 运行身份为 `d53a39c003b06f653a54fff9beb9aff35eaa3d9ddb402789b5d90526ab7cfffa`；不带 motion token 的 VLA 运行身份为 `7d0b43fcf96394ebacffb8833a6ebba482d31a4699dc4b69371d6676b5b1918e`。主检出后续只回写本任务档案，不能据主检出的文档提交误称运行代码已变化。

## 冒烟结果与放大依据（美东 2026-11-04）

两侧第 8 局于 18:11:20 并行启动；带 motion token 的 VLA 于 18:17:34、不带 motion token 的 VLA 于 18:17:18 留下 `EXIT_CODE=0`。两局任务均成功，分别执行 312／310 个动作，各有 20 份动作／文本 attention 与 20 份 prefix attention。

主会话逐次读回所有 attention 与 prefix，核对维度、有限性、行和、padding、时间和 key 映射、视频帧数与动作数；两侧全部 20 次推理的动作与旧同硬件第 8 局逐位一致，最大绝对差均为 0。带 motion token 的 VLA 的 9 次编码调用完整保存，20 次推理累计使用 90 个有效 token，每个有效向量均与生成历史对应；实际输入为 `motion_emb=(96,1024)`、`motion_pos=(96,256)`，均 `float32`，padding 槽位与有效 token 分开统计。

`CAPTURE_SMOKE=PASS expected=2 completed=2 seconds=59.4`，验收退出 0；详见 [capture-validation-smoke.json](records/capture-validation-smoke.json)。这两局计入 22 条组合，不再重跑。

据此，18:20:13 启动两条剩余列表：`ia-success-all-with-motion-1104` 跑 11 条，`ia-success-all-without-motion-1104` 跑 9 条，命令和版本保持上文不变。正常任务失败保留并继续；基础设施失败不自动重试。每模型 4 小时预算从冒烟起始计，预算终点为美东 22:11:20，由主会话统一掌握。

关节状态回放的首次命令因额外隐藏 CUDA 设备而退出 139，失败日志保留为 `R/logs/poses-smoke-with_motion.log`；审查后仅恢复 CUDA 设备可见性，两侧均以原 CPU 物理路径回放通过，`SIM_EEF_POSE=PASS infers=20`、`EXIT_CODE=0`。未改变渲染实现、未修改第三方代码、未重新采样策略回合。后续回放只选已完成终局的精确局目录，避免读取仍在写入的回合。

## 多次取物与分析版本

实际第 11 局提示包含先取蓝色目标、再取绿色目标；取物次数不能固定为 1。离线分析因此把提示中的全部已核目标色 query 行先等权平均，仍作为唯一第六族；不新增第七族，不改变 R3 的 20 倍、6 层、50% 有效出现或跨族阈值。单目标保持原式，缺少任一目标行时整族未核实。真实第 11 局 `txt_len=29`、目标行为 `[16,26]`，与显式两行均值逐元素一致；第 8 局单目标也与原式逐元素一致。详见 [多目标定向验证](records/S2-multi_target_verdict.json)。

采集版本仍固定 `e56bbceca5fe1c734c2553e83b8b6de1db71ca08`。离线分析改在现有工作树 `/home/hongzefu/.codex/worktrees/button-attn-analysis/robomme_policy_learning_MotionJEPA` 运行，固定新提交 `c55700b63b89de451adfc0c459bd99ecefce8443`，该工作树干净；不在采集期间切换采集代码。分别记录 `capture_commit` 与 `analysis_commit`，所有产物仍写入本任务 R。

上述修正的 `ADJUST_REVIEW=PASS scale=same findings=0`；独立复审 `PRE_MERGE_REVIEW=PASS base=1e0ba0edfdbbc8bd480f10602176523585924707 tip=c55700b63b89de451adfc0c459bd99ecefce8443 files=1 findings=0`。这是同一任务对真实多目标输入的适配，不新增或重试策略回合。

## 全量采集与内容验收

剩余两条列表分别于美东 18:53:05／18:42:03 正常结束，四个采集 tmux 会话均自然退出。含冒烟，带 motion token 的 VLA 完成 12 局、真实成功 11 局，失败为第 26 局；不带 motion token 的 VLA 完成 10 局、真实成功 5 局，失败为第 14、16、20、26、34 局。没有基础设施错误或策略重试。原始 attention／prefix 各 503 份，视频 22 份，完整生成编码记录 12 份；这些数量不等于任务全部成功。

全量逐数组内容验收通过：`ATTN_COVERAGE=PASS expected=22 completed=22 infers=503 prefixes=503`；`MOTION_TOKEN_COVERAGE=PASS episodes=12 generated=130 uses=1531 missing=0`。130 指编码调用，1531 指跨推理累计有效 token 使用次数；不是 96 个预算槽位全部有效。大数组、向量、时间和位置、终局身份及视频长度已逐项检查。完整结果为 [capture-validation-all.json](records/capture-validation-all.json)。

正式内容验收的精确 tmux 为 `ia-success-validate-without-motion-final-1104`、`ia-success-validate-with-motion-final-1104`，分别记录到 `R/logs/validate-without_motion-final.log`、`R/logs/validate-with_motion-final.log`，从干净固定采集工作树执行同一份 `R/validate_capture.py --phase all --model <组>`。最初一次未完成的独立验收因执行目录修正而停止，仅操作本轮精确会话名、删前删后核对其他会话不变，原日志保留，不作为最终验收依据，也没有重新执行策略回合。

22 局关节状态回放均完成，物理后端为 CPU，不调用策略或 `env.step`。主会话已把 25 次实际取物尝试逐个与视频拼图、实际执行的闭合指令和 TCP 状态核对，形成 75 个代表性阶段锚点，每次取物都保留 `pickup_index`。没有把其余推理强行贴阶段标签；完整 503 次推理仍全部进入整局分析。

## 正式离线分析命令与资源

分析提交固定 `c55700b63b89de451adfc0c459bd99ecefce8443`，干净分析工作树如上；实际导入路径、清单和包装脚本源码哈希见 [analysis-dispatch.json](records/analysis-dispatch.json)。[分析清单](records/analysis-manifest.json) 绑定这两份提交及全部阶段证据。

为缩短只读离线计算，使用 4 个 CPU 进程按回合并行，每进程 BLAS／OMP／MKL 各 1 线程，不使用 GPU，预算 4 小时、无重试。标准库 `ProcessPoolExecutor.map` 按原清单顺序返回，每进程原样调用 `success_attention.py::episode_analysis`，各局目录独立；只有父进程调用原 `summarize`、`figures_summary` 写总结果。因此不改变公式、阈值或跨局归约顺序。该调度调整已通过 `ADJUST_REVIEW=PASS scale=same findings=0`。

完整入口命令为：

```bash
bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/run_analysis.sh
```

该脚本进入固定分析工作树，在启动解释器前设置：`PYTHONPATH` 指向分析工作树根、`src`、`packages/openpi-client/src` 和主检出 benchmark 源码；`CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTHONUNBUFFERED=1`，缓存及资产环境沿用前文的主检出本机目录。最终命令是 `timeout --signal=TERM --kill-after=30s 4h <主根>/.venv/bin/python R/run_analysis_parallel.py --manifest R/analysis-manifest.json --out R/analysis --workers 4`，没有 `--max-infers` 截断。

精确 tmux 为 `ia-success-analysis-1104`，外层 `pipefail`／`tee`／真实 `EXIT_CODE` 规则同采集；日志 `R/logs/analysis.log`。单局输出为 `R/analysis/<model>_ep<id>/`，总结果为 `R/analysis/success_attention_summary.json`。最终结论与图的目视复核另写 `result.md`，此处不提前声称分析通过。

辅助命令的实际意外也保留：一次状态回放批处理用 `uv run --no-sync` 在空运行工作树生成了未同步的空 `.venv`，没有安装依赖或改变正式解释器；后续辅助及正式命令均显式使用主检出已固定的解释器。一次导入位置预检在主检出执行而被路径断言拒绝，改到固定分析工作树后通过，未在错误导入位置运行正式分析。
