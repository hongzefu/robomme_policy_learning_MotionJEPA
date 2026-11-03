# attn-ctx512-vs-motion-1103 起跑档案（第一阶段）

计划：[1103-interpret-motion-attention-plan.html](../../../1103-interpret-motion-attention-plan.html)（开工审查与裁决见文末第八节）。用户原话：「开工」「全部在本地datahongzefu上保存」「尽可能不要turbo」「尽可能保证下载安装的速度！」。

## 环境与代码
- 环境 A：sled-aspen，2×RTX A6000（46068 MiB），本机跑，不连 GreatLakes、不提交 Slurm。
- 代码：分支 v4.1-interpretMotion，Beta 锚点见本文件所在提交；采集代码 scripts/interpret/（S1/S2/S3 及两次自检修复，均 --no-ff 合入）。src/ examples/ third_party/ scripts/training/ scripts/evaluation/ 相对 b613836 零改动（SRC_ZERO_DIFF=PASS）。
- 工作区仅有用户自己的未跟踪文件 `TODO_RoboMME_Qwen3VL_FrameSamp_Context.en (1).html`，按第 11 条不动、不提交。

## 资产（全部落本机 /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store，turbo 运行期不再依赖）
| 对象 | 本机路径 | 来源 | 判定行 |
|---|---|---|---|
| 官方 ctx512 79999 | v1-store/models/official-mme-vla/perceptual-framesamp-context/79999 | HF Yinpei/mme_vla_suite@5db4d53d（IPv4 强制） | OFFICIAL_CTX_ASSET=PASS sha256=387d4bd5…8ae8 bytes=11574223742 files=17 |
| concat 59999 + encoder | v1-store/assets/vla-o16cc-59999 | HF bucket HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1（14 个文件 8.18 GB 复用 rsync 半成品并逐文件核 sha） | LOCAL_ASSETS=PASS cc59999_files=38 cc59999_bytes=11976241540 encoder_sha256=27daa832 |
| 策略解释器 | .venv（uv sync，jax 0.5.3 torch 2.7.1+cu126） | 本仓库 uv.lock | VENV_SRC 指向本仓库 src |
| 仿真解释器 | v1-store/envs/official-sim（uv sync third_party/robomme_benchmark，torch 2.9.1+cu128） | 本地子模块 856bc3a 的 uv.lock | ENV_SIM robomme → 本仓库 third_party |
| stage2 sidecar 解释器 | v1-store/envs/stage2（uv sync scripts/dataset/wan_stage2） | 本仓库 uv.lock | 安装中 |
| motion 运行合同 | v1-store/evaluation/interpret-1103/contracts/runtime-mot.json | measure_target_runtime.py | EVAL_RUNTIME_CONTRACT=RECORDED cross_hardware_numeric=not_required |

## 采集自检
- ATTN_SELFTEST=PASS f32_maxabs=1.19e-07 bf16_relfro=0.00321 rowsum_err=3.54e-07 uniform_ok=1 P=1088 MEM=512 motion=off（官方，GPU0）
- ATTN_SELFTEST=PASS f32_maxabs=5.96e-08 bf16_relfro=0.00329 rowsum_err=3.02e-07 uniform_ok=1 P=1184 MEM=608 motion=Zero1024（concat，GPU1）
- SIM_EEF_POSE=PASS infers=1（合成 npz，reset 后 tcp_z=0.5211，CPU 物理后端）

## 第一阶段运行
- 局清单：v1-store/evaluation/interpret-1103/manifest-stage1.json，MANIFEST=PASS episodes=5 chunks=5（ButtonUnmaskSwap 4/8/11/12/17，每局一块）。
- spec：spec-ctx.json（GPU0，policy_config mme_vla_suite_b128，model_kind history，XLA 0.40）、spec-mot.json（GPU1，mme_vla_suite_b128，stage2_pool4_concat，XLA 0.55，target_contract runtime-mot.json）。
- 规则：基线先跑，失败即换下一局、不跑 concat；基线成功再跑 concat 同局；最多 5 局。
- 命令（每局一条 tmux，前缀 ia-run-）：`uv run --no-sync python scripts/interpret/run_local.py --spec v1-store/evaluation/interpret-1103/spec-<ctx|mot>.json --episodes ButtonUnmaskSwap:<id>`
- 产物：v1-store/evaluation/interpret-1103/stage1/<ctx|mot>/{attn,run}/；日志 v1-store/logs/ia-run-<ctx|mot>-ep<id>.log。

## 实施中调整（ADJUST_REVIEW 均 PASS scale=same）
1. motion 侧 policy_config 改 mme_vla_suite_b128（orig16_b128_60k 本分支未注册；推理归一化取 checkpoint assets）。
2. 基线 model_kind=history（计划误写 frame_sampling）；run_identity 由 run_local 自算 digest。
3. 权重与两套解释器全部改为本机（HF 下载 + 本地 uv sync），encoder 用资产内 encoder_dependencies；HF 连接强制 IPv4（本机 IPv6 到 HF CDN SYN-SENT 挂起）。
4. motion 侧加运行合同 target_contract（A100 来源 vs 本机 A6000，只比软件口径）。

## 本轮 tmux 会话清单
ia-p1-env、ia-rsync-assets（手动停）、ia-dl-ctx79999、ia-dl-mot、ia-env-sim、ia-env-stage2、ia-simpose-smoke、ia-selftest-mot、ia-selftest-ctx、ia-selftest-mot2、ia-selftest-mot3；第一阶段 ia-run-*。
