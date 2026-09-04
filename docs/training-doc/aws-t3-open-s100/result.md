# aws-t3-open-s100 — 结果（T3 开启态 100 步；本文件同时归档 T3 跨侧判定与在线观察）

## 一、run 本体

- **起跑**：2026-09-04 06:23:15 → 06:41:42（18 min），HEAD `8093ebd`（clean），GPU6,7，b8，100 步，`perceptual-framesamp-context-motion.yaml`（开启态），
  `DATASET_PATH=v1-store/datasets/4task-motion-40ep/framesamp`（AWS 本地 NVMe RAID），确定性档；`BENCH_PASS`、`EXIT_CODE=0`；
  `RESULT batch=8 稳态=1.681s/step (n=72, p10=1.642, p90=1.718)`（closed 1.510；确定性档、四条 run 并行，**只记录**）。
- **驱动摘要**：`OK TrainState 摘要 5 次 @steps=[0, 25, 50, 75, 99], 末值 state=82674c627139cfa5…, 单次耗时中位 167.9s`；
  `OK 输入摘要 7 次 @steps=[0, 1, 2, 25, 50, 75, 99], raw 末值 67376256277c6f5b…, canonical 末值 5d3458304758a332…, index 序列 872 个 sha=f8bd8d5a9720a61b…`
  （index 序列 sha 与 closed 侧 / T2 两侧同值——四条 run 的样本顺序逐位相同）；`OK loss n=100 min=0.1219 max=0.6773 末值=0.1445`。
- **最终 checkpoint**：`v1-store/train-runs/aws-t3-open-s100/mme_vla_suite/aws-t3-open-s100/999`（`checkpoint_id=999, state_step=100, param_kind=ema, n_leaves_params=59`）。
- **records/**：`metrics.jsonl`、`param_checksums.jsonl`（193 叶）、`batch_digests.jsonl`（`n_keys=16`）、`index_sequence.json`、`run_meta.json`、`env.json`、`final_checkpoint.json`、
  `t3_common_init_reference.json`、`t3_cpu_gates.txt`（verifyinit / smoke / trace 原始输出）、`t3_mechanism.json` / `t3_mechanism.txt`、`t3_phase.json` / `t3_phase.txt`、`eval/`（T3_EVAL_OBS，见四节）。

## 二、T3 硬闸（`scripts/training/tests/motion_gates_model.py`，100 步口径）

| 闸 | 判定行 | 备注 |
|---|---|---|
| `T3_COMMON_INIT` | `T3_COMMON_INIT=PASS common_mismatches=0 open_only_params=4 open_only_ema=4 open_only_opt=8 closed_only=0 n_leaves_closed=177 n_leaves_open=193` | 起跑前，GPU2,3 `--fsdp 2`；reference 写在 `--out` 默认 `t3.json`，改名为 `t3_common_init_reference.json` 后其余闸门才读到 |
| `T3_INIT_MATCH` | `T3_INIT_MATCH=PASS` | 两侧 step-0 记录命中 reference |
| `T3_SMOKE` | `T3_SMOKE=PASS steps=100 nan=0 motion_params_updated=4 n_keys=16/12 n_leaves=193/177` | 新脚本 `t3_smoke.py`（对环境 A 1000 步留档自检同样 PASS） |
| `T3_TRACE_PREFLIGHT` | `T3_TRACE_PREFLIGHT=PASS samples=56 empty=4 k_ge2=51 video=True` | 7 摘要步 × 8 = 56 样本，覆盖空 / k≥2 / Video |
| `T3_TOKEN_TRACE` | `T3_TOKEN_TRACE=PASS steps=7 samples=56 keys=4 mismatches=0` | 步集 {0,1,2,25,50,75,99} 与 800 前缀由两侧 `env.json` 推导（commitV6.12 `_t3_expected_steps`，1000 步口径回归断言仍过） |
| `T3_MOTION_CAUSAL` | **`T3_MOTION_CAUSAL=FAIL pad_bitexact=0 emb_effect=1 pos_effect=1`** | step 0 batch `[6556, 671, 8452, 3987, 10070, 3804, 8928, 2595]`，base loss 0.704831；见下文「FAIL 诊断」 |
| `T3_MECHANISM` | **`T3_MECHANISM=FAIL step=0 input_grad_ok=1 group_norms_ok=1`** | 分组梯度范数：W2_content 4.28e+01 / W2_pos 5.17e+00 / W1 5.39e+00 / b1 4.73e-01 / b2 1.60e+00 / ∂motion_emb 有效位 9.10e-01 / ∂motion_pos 1.48e-01；两层 bf16 独立复算与 gather 逐位（无 fails 项） |
| `T3_PHASE_REPORT` | `T3_PHASE_REPORT samples=11530 phase0_n=738 … empty_n=640 nonempty_n=10890`（完整性：phase0 = 冷 80 + 稳 658、phase0 + other = 11530、empty + nonempty = 11530，均过，`EXIT_CODE=0`） | GPU6，06:43 → 07:14；两侧 999 目录 EMA ckpt |

**T3_MOTION_CAUSAL / T3_MECHANISM 的 FAIL 诊断（原始输出 `records/t3_mechanism.txt`；本轮不改判据、不排除叶子，交用户裁决）**：

```
[pad-diag] 确定性探针：同一 obs 连算两次梯度，叶变化 1/36：["['PaliGemma']['llm']['embedder']['input_embedding']"]
[pad-diag] loss base=0x1.68df900000000p-1 pad(1e3)=0x1.68df900000000p-1 同=True；梯度叶变化 1/36：["['PaliGemma']['llm']['embedder']['input_embedding']"]
[pad-diag] 垃圾尺度 1: loss 同=True 摘要同=False 叶变化 1：["['PaliGemma']['llm']['embedder']['input_embedding']"]
[pad-diag] 垃圾尺度 0.001: loss 同=True 摘要同=False 叶变化 1：["['PaliGemma']['llm']['embedder']['input_embedding']"]
```

- 判据是「padding 垫料 → loss 与 36 个 trainable 叶的梯度摘要逐位不变」。实测 loss 三档垫料（1e3 / 1 / 1e-3）**逐位相同**，36 叶里 **35 叶逐位相同**，只有 LLM 词表 embedding
  `['PaliGemma']['llm']['embedder']['input_embedding']` 一叶不同——而脚本自带的确定性探针「**同一 obs 不改任何输入连算两次**」也只让这一叶不同。即该叶的梯度
  （embedding 反向 = scatter-add）在 A100 + jax 0.5.3 + `--xla_gpu_deterministic_ops=true` 下于本脚本的 `jax.value_and_grad` 里本身不确定，与 motion 垫料无关。
- 与环境 A 的同一闸门首次 FAIL 同构（`motion-t3-open/result.md` 二节：Ada 上是冻结叶 `['PaliGemma']['img']['embedding']['kernel']` 的 wgrad 不确定，当时因该叶不在
  `trainable_filter` 内而把摘要收窄到 trainable 叶后 PASS）。不同点：本机不确定的这一叶**在** `trainable_filter` 内，不能用同一理由排除。
- 反证「训练路径确定」：`aws-t2-ref-s100`（旧码，GPU0,1）与 `aws-t2-cand-s100`（HEAD，GPU2,3）与 `aws-t3-closed-s100`（HEAD 经驱动，GPU4,5）三条 100 步 run 的
  5 次 TrainState `state_digest`（177 叶，含 `input_embedding` 及其 ema / opt）**逐值相同**——训练 `train_step`（`nnx.DiffState` + fsdp 2 sharding）里该叶梯度是确定的；
  不确定只出现在 t3mechanism 自己的 `jax.jit(value_and_grad(loss_fn))` 编译路径。
- 结论：证据指向「A100 上本诊断脚本的 embedding 反向核不确定」而非「motion 垫料泄漏」（loss 逐位同、motion 相关叶与其余 35 叶逐位同、`emb_effect=1 pos_effect=1` 与
  分组范数都正常）。按 motion-memory-plan.md 四节表一的处置，硬闸 FAIL 不放宽、不裁剪；**是否接受「把该叶的不确定性单列（先跑同 obs 两次探针、把两次都变的叶排除）」这一
  与环境 A 同性质的修法，留给用户裁决**；裁决前 T3_MOTION_CAUSAL / T3_MECHANISM 在环境 B 记 FAIL。

## 三、描述性观察（无 PASS / FAIL；单 seed；100 步 × b8 = 800 样本 < 1 epoch；ep0–9 在 encoder 训练集内，不得升级为泛化结论）

- **T3_EFFECT_OBS**：末 20 步 loss 均值 open 0.1518 / closed 0.1200；首步 0.4946 / 0.5807；末步 0.1445 / 0.0970；末 20 步 `mem_enc_norm` 均值 open 10.70 / closed 7.77。
  open 首步 loss 低于 closed 是两个新层随机初始化带来的初值差异（A20 观察项）；100 步太短，不作效果解读。
- **T3_PHASE_REPORT 均值**（20 步 eval loss 均值，open / closed，`records/t3_phase.json`）：phase0 0.4893 / 0.5534（冷 τ<32：0.4319 / 0.4891，n=80；稳态 0.4963 / 0.5612，n=658）；
  other 0.5033 / 0.5586（n=10792）；空运动路 0.6782 / 0.6792（n=640）；非空 0.4921 / 0.5511（n=10890）。方向：100 步 ckpt 下 open 侧 eval-loss 均低于 closed（与环境 A 1000 步的方向相反），
  单 seed / 100 步 / 欠训练，**不作结论**，只说明两侧 ckpt 可恢复、11,530 样本全部跑通。

## 四、T3_EVAL_OBS（尽力项）

- **仿真环境**：`micromamba create -n robomme python=3.11` 于 `MAMBA_ROOT_PREFIX=/scratch/hongze/micromamba`，`pip install -r examples/robomme/requirements.txt`、`-e third_party/robomme_benchmark`
  （submodule 未初始化，`git submodule update --init` 后第二次装成）、`-e packages/openpi-client`；`examples/robomme/simple_test.py` rc=0（GPU7）。
- **执行**：`run_t3_eval_obs.sh`（`fix:` `cbf24e9` 加 `RUN_PREFIX=aws-t3-eval-obs CKPT_CLOSED/CKPT_OPEN` 覆盖），每侧按任务拆两片（`-a` ButtonUnmask+VideoUnmask / `-b` 两个 Swap），
  `MAX_EPISODES=10 OVERWRITE=1 SEED=42`；closed 两片 policy 在 GPU0（`POLICY_MEM_FRACTION=0.38`）、仿真 GPU0；open 两片 policy 在 GPU1（0.2）、两个 sidecar 也在 GPU1（`motion.online_gpu=1`）、仿真 GPU1；
  HEAD `c0e13aa`（clean）；07:51 → 09:55（closed 侧 ≈50 min，open 侧 ≈2 h）。**与 400 ep 建库（Wan 抽取 GPU2–7、D2 oracle 8 片含 GPU0/1）同机并行**，耗时只作量级。
  前两次起跑失败（工作区有未提交文档 → 脚本 clean-HEAD 断言；tmux 命令拼接 `export` 少分号）日志留 `ev-*.attempt{1,2}-*.log`。
- **判定行**（`records/eval/summary.txt`、`t3_eval_obs.json`、`progress.*.json`）：

```
T3_EVAL_OBS open=0.0 closed=0.0 episodes=40/40（单 seed，描述性；ep0–9 泄漏）
  closed: 0/40 成功（error 0） | ButtonUnmask 0/10 | ButtonUnmaskSwap 0/10 | VideoUnmask 0/10 | VideoUnmaskSwap 0/10
  open: 0/40 成功（error 0） | ButtonUnmask 0/10 | ButtonUnmaskSwap 0/10 | VideoUnmask 0/10 | VideoUnmaskSwap 0/10
```

  两侧 checkpoint 只训 100 步 × b8 = 800 样本，0% 是预期内的欠训练结果（环境 A 1000 步同为 0/40）；信息量在「在线链路（policy server + 开启态 sidecar + 仿真）在 A100 上对真实 ckpt 跑通 40 集无 error」。
- **端到端耗时（server 端挂钟，四进程 + 两 sidecar + 四仿真 + 建库并行，只作量级）**：

| 侧 / 分片 | add_buffer ≤16 帧 mean / median / p90 | 首批（整段 pre_traj）mean / max | infer（除首次）mean / median / p90 |
|---|---|---|---|
| closed-a | 66.9 / 60.3 / 80.0 ms | 730 / 5631 ms | 136.5 / 132.9 / 213.7 ms |
| closed-b | 71.5 / 69.5 / 93.4 ms | 2227 / 7600 ms | 131.9 / 135.6 / 210.4 ms |
| open-a | 3596 / 3235 / 4798 ms | 11294 / 14700 ms | 201.3 / 186.3 / 310.3 ms |
| open-b | 3537 / 3235 / 4819 ms | 30139 / 55400 ms | 192.4 / 185.9 / 307.6 ms |

  读法：open 侧每批 16 帧 add_buffer ≈3.5 s = 一次 sidecar 窗编码（P5 独占 GPU1 时 1.42 s/窗）在「两个 sidecar + D2 oracle 分片 + 两个仿真同挤 GPU1」下的争用值，
  与环境 A 的 1.74 s（两 sidecar 共享一张 Ada）同性质；首批 = demo 段窗口数 × 单窗（Swap 任务 demo 更长）。closed 侧 infer 130 ms 量级、open 侧 190 ms（GPU1 争用更重）。

## 五、盲区诚实清单

- 单 seed、100 步、b8：三节全部数字只证链路跑通，不证 motion memory 有 / 无效；A100 上稳态 s/step 为确定性档并行数字，不作性能结论。
- 二节 FAIL 的裁决权在用户；本文件不替用户决定。
- ep0–9 在 MotionJEPA encoder 的训练集内（holdout 90–99）。
