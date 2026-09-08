# 训练 / 推理一致性验证

本文是现行正本，自足可读：结论、调用链、每一关查了什么与结果、待拍板的事都在本文内。判定行逐字原文与逐轮数据在六个 run 的 `docs/training-doc/tic-*/result.md` 里，本文不再复述。
环境 B（AWS 单机 8×A100，仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`）；对象是生产 checkpoint `v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999` 与它训练时用的 400 ep 库 `v1-store/datasets/4task-motion-400ep`。2026-09-07 全部跑完。

## 一、结论

**推理时喂给模型的东西，和训练时从表里读出来的，是同一份。** 在这个 checkpoint、5 条训练集 episode 的 120 个决策时刻、以及 24 集真仿真上：推理端现拼出来的记忆（608 个 token 与它们的次序表）、当前两张图、机器人状态、任务描述的 token，与训练端逐字节相同；进模型之后，1184 个前缀 token、attention mask、位置号、18 层 KV 缓存逐位相同；同一噪声下去噪 10 步得到的 20 步动作逐位相同。真 sidecar 现算的 140 个运动窗与离线表逐字节相同。

只有两处天然不逐位，都已量化、都是数值精度问题而不是逻辑错误：
- 历史帧特征编码器：训练建表用 f32 权重逐帧编，推理用 checkpoint 内 bf16 权重整批编，记忆 token 相对差 0.33%，传到动作上只有采样噪声的 5.5%。用户 9 月 6 日拍板不改。
- 训练的「整段一次前向」和推理的「前缀缓存 + 分步去噪」两条路，同一输入下速度场相对差 0.19%～0.30%；把精度升到真 f32 后差异归零，证明只是 bf16 kernel 归约序不同。

措辞按审计收窄：**在指定 checkpoint、指定样本、上述两项隔离之下，所覆盖路径未发现额外的训练 / 推理不一致。** 不说「已排除全部不一致」，因为样本只有 5 集 + 24 集，且真仿真那一关没有训练真值、只验不变量。

由此，三 seed 评估 motion 组与官方组无差别（24.2% ± 1.3 vs 24.5% ± 0.5，环境 A 历史数字）**不能**归咎于「推理喂错了东西」；剩下的解释只有 motion token 本身作用有限、或评估噪声。

### 待你拍板的三件事

| 事 | 是什么 | 选项 |
|---|---|---|
| `VT_FULL_VS_CACHED` 阈值 | 整段前向 vs 缓存分步的速度场差，实测 rel_fro 1.9e-3～3.0e-3、bf16 ulp_p99 10～22，超过计划事先定死的 1e-3 / 4。已证明是纯数值来源（第五节 2）。脚本没放宽，判定行保持 FAIL | (a) 按实测重定阈值，按四次跑约 1.6 倍波动带留裕量；(b) 改成「f32 档逐位 + bf16 档只观察」；(c) 维持 FAIL 记录 |
| 测试集出现训练未见的 goal | ButtonUnmask 测试集第 3 集的目标「先按按钮，拿红方块的容器，再拿绿方块的容器」在 400 集训练数据里没出现过（第五节 3） | (a) 评估口径里注明该集「训练未见 goal」；(b) 补数据重建库 |
| 单进程第 28 次建仿真环境必崩 | 根因已查清，两种修法各实测 35 轮不崩，patch 已写好未应用（第五节 4） | (a) 钉住 RenderSystem，顺带建环境从 6.8 s 压到 1.2 s，但改了渲染 Context 生命周期，建议先做一轮同 seed 对拍；(b) `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192`，零代码最保守。两者都落 `scripts/training/legacy-eval/`，主线不动 |

## 二、在查什么

三 seed 评估里 motion 组与官方 `framesamp+context` 组没有可辨别差异。三类解释：(a) motion token 本身没用；(b) 训练学到了，但推理时喂进模型的记忆与训练时不是一回事；(c) 评估噪声。本轮只查 (b)。

「记忆」是模型每走一步除当前观测外还要看的一块输入：过去 32 张历史帧的特征（每帧 16 个 token，共 512 个）、过去若干段运动的特征（每 33 帧一段、每 16 帧起一段，最多 96 个 token）、以及一张把这 608 个 token 按时间排好的次序表。训练时这些是建库阶段提前算好存在硬盘表里的，训练程序按行读；推理时没有表，机器人边跑边现算。两边算的是不是同一份，此前只比到「8 个数组逐位相同」这一层，往下（预处理之后、模型内部、最终动作、真实评估节奏）一处都没比过。

## 三、两侧的调用链

### 训练侧：从 h5 到 loss

1. **建库**（离线，跑一次）。`scripts/dataset/build_dataset.py` 读原始 h5，把每帧图像和 `setup/task_goal`（转小写）写成 pkl；`scripts/dataset/pack_framesamp_store.py` 用离线 f32 权重的 `dataset_builder/siglip_tokenizer.py::SigLipTokenizer` **逐帧、batch=1** 编成 (16,1152) 的帧特征，连同位置编码表、状态一起打成三张连续大表；`scripts/dataset/wan/` 抽 Wan latent，MotionJEPA 编码器把每个 33 帧窗编成 (768,) 的运动 token，`scripts/dataset/pack_motion_store.py` 打成 motion 表。四张表的 sha256 记在 run 的 `motion_provenance.json` 里。
2. **读样本**。`training/framesamp_dataset.py::FrameSampDataset.__getitem__` 按样本的 (episode, 时刻 t) 从表里取：用 `shared/sampling.py::even_sampling_indices(t, 32)` 选 32 帧、读它们的特征行；按 `visible_motion_frames` 公式选出到时刻 t 为止合法的运动窗（≤96），读 motion 表；用 `shared/sampling.py::memory_order` 按「2·时刻 + 类型」稳定排序生成 608 位次序表 `mem_order`；右侧补零并给出 `static_mask`、`motion_mask`。交付 8 个记忆数组加当前图、状态、prompt、动作。
3. **变换链**。`training/config.py::RoboMMEDataConfig.create` 组的链：`RepackTransform` 丢掉 8 个不进模型的键（`epis_idx`、`exec_start_idx`、`is_demo`、`step_idx` 等）→ `RoboMMEInputs` 归一化状态与动作 → `TokenizePromptWithState` 用 `PaligemmaTokenizer` 把 prompt 编成 64 个 token（不做小写转换）。
4. **前向**。`openpi/models/model.py::preprocess_observation(train=True)` 做图像增广（记忆 token 不动）；`history_pi0.py::HistoryPi0.embed_prefix` 把 608 记忆 + 512 图像 + 64 文本投影成 1184 个前缀 token；`compute_loss` 再拼上 20 个带噪动作 token 成 1204 位，`make_attn_mask` 造一张 1204×1204 的 mask、位置号 = mask 累计和减一，`llm([prefix, suffix], mask, positions)` **一次算完**，`action_out_proj` 取最后 20 位得速度场，与真值算 loss。

### 推理侧：从仿真观测到动作

1. **观测**。`examples/robomme/eval.py::EpisodeEvaluator.eval_each_episode` 每集先 `client.reset()`，把整段 demo（0 到 es 帧）一次送进 `add_buffer`，之后每凑够 16 帧送一次；最多 1300 步，所以一集最多 82 次推理、最晚决策时刻是 es + 1296。prompt 送的是 `task_goal` 原文。
2. **现算帧特征**。`policies/policy.py::MME_VLA_Policy.add_buffer` 调 `history_pi0.py::HistoryPi0.vision_encode`，用 checkpoint 里的 bf16 `PaliGemma.img` **整批**（首批 es+1 帧、之后 16 帧）编帧特征，存进 `policies/framesamp_memory.py::FrameSampMemory`；位置编码表由 `PosEmb3D` 在 GPU 上现算 4096 行。
3. **现算运动特征**。每凑齐一个 33 帧窗，`policies/motion_protocol.py::MotionEncoderClient` 把帧发给旁路进程（sidecar，Wan VAE + MotionJEPA 编码器，权重与建库同一份 sha256），拿回 (768,) 的 token。
4. **拼记忆**。`MME_VLA_Policy._prepare_history` 用与训练侧**同一个** `even_sampling_indices` 选 32 帧、同一个 `visible_motion_frames` 规则选运动窗、同一个 `memory_order` 排序，拼出与训练完全同形状的 8 个数组。
5. **变换链**。`policies/policy_config.py::create_trained_policy` 组的链：`InjectDefaultPrompt`（obs 已带 prompt，实际不做事）→ 与训练侧**同一批** `RoboMMEInputs` / `TokenizePromptWithState` 对象。
6. **前向**。`preprocess_observation(train=False)` 不增广；同一个 `embed_prefix` 得 1184 个前缀 token；`HistoryPi0.sample_actions` 先 `llm([prefix, None], mask, positions)` 算一遍前缀、存下 18 层 KV 缓存，再从纯噪声出发去噪 10 步，每步只算 20 个动作 token：`llm([None, suffix], mask, positions, kv_cache)`，位置号 = 前缀长度 + 动作段累计和减一；最后 `MME_VLA_Policy.infer` 反归一化成关节动作。

两条链结构上只差两处：训练多一个丢键的 `RepackTransform`、推理多一个空操作的 `InjectDefaultPrompt`，实测都不改任何数。真正会产生数值差的只有两跳：帧特征编码器（f32 逐帧 vs bf16 整批）与 LLM 前向（1204 行一次算 vs 1184 行前缀 + 20 行分步）。

## 四、怎么查、查到什么

**样本**：训练集 5 条 episode，覆盖全部 5 种 demo 段长度 es ∈ {0, 66, 114, 168, 216}，每种取最长一集；原始帧按真实评估节奏回放给生产口径的 policy，共 120 个决策时刻、140 个运动窗。所有「必须逐位」的判据都把推理端的帧特征换成训练表真值（S 臂），真现算那一路（B 臂）只记差多少。另加 24 集（4 任务 × 6 集）test split 真仿真。

| 关 | 比什么 | 查了多少 | 结果 |
|---|---|---|---|
| 0 配置与库同源 | 归一化统计量、checkpoint 参数树、当前库四张表的指纹 vs 训练 run 记录的指纹、motion 库路径 | norm stats 8 个数组、6 个 sha256、59 个参数叶 | 全部相同。本轮对拍用的库就是训练时那张 |
| 评估节奏 | 用 `eval.py` 自己的 `EpisodeState` 在 CPU 上复刻决策时刻 | 5 集 120 个时刻，三种算法互核 | 全一致。每集最多 82 次推理；demo 段 ≥ 289 帧时运动窗会到 97 个、超预算 96（当前最长 216，余量 4 窗） |
| 1 输入键 | 8 个记忆数组、当前两张图、状态、prompt 原文；运动 token「sidecar 现算 vs 表」 | 120 点；140 窗 | 全部逐字节相同 |
| 2 预处理后 | 两侧各过完自己那串变换后的全部键，含 prompt 的 64 个 token | 120 点 | 全部相同；两侧多出来的步骤都不改数 |
| 3 模型内部 | 1184 个前缀 token、attention mask、位置号、18 层 K/V；次序表能否原样还原 | 120 点 | 全部逐位相同 |
| 4 整段 vs 缓存 | 同权重同输入同噪声：训练路一次算完 vs 推理路前缀缓存 + 一步 | 120 点 | mask 前缀子块、动作行、动作段位置号 120/120 全等；数值差超事先阈值，见第五节 2 |
| 5 最终动作 | 同一噪声去噪 10 步的 20 步动作 | 120 点 × 3 组噪声 | 全部逐位相同，重跑确定 |
| 6 真仿真 | 24 集在线不变量：运动窗数是否等于公式值、次序表是否合法排列（汇总器独立重算）、prompt token 是否训练见过、位置表是否与库表相同、24 集与探针记录一一对应、有无抛错 | 24 集 359 次推理 | 6 条过，零 error 零 timeout；1 集 prompt 训练未见，见第五节 3 |

两项此前 FAIL 的闸门本轮闭合：
- `A19_VALID_DIST`：原判据把 40 ep 库的四个分布数写死在脚本里，400 ep 库必 FAIL。改为期望按当前库清单重算、实测取真实交付的 `motion_mask.sum()`，400 ep 库 101,066 个样本逐样本全等；40 ep 库重算出的期望与旧数字一致，没有改松。
- `T3_MOTION_CAUSAL`：9 月 4 日在同一 obs 连算两次时 LLM 词表 embedding 叶的梯度会变，导致「垫料不改梯度」判 FAIL。改为先无条件跑 3 次确定性探针、把不确定叶单列排除（含 motion 叶即 FAIL），PASS 语义收窄为「排除叶之外逐位一致」并报覆盖率。重跑时不确定性没有复现，36/36 叶全部逐位一致。

## 五、四个值得单独讲的发现

### 1. 帧特征编码器：训练 f32 离线表 vs 推理 bf16 checkpoint

同一份 SigLIP 编码器，训练建表用离线 f32 权重逐帧编，推理用 checkpoint 内 bf16 权重整批编。两份权重逐值相同（23 叶 max_abs = 0，训练全程冻结），差异只来自精度和批形状，两者各贡献约 0.3%。实测 2409 帧记忆 token 相对差 0.33%、余弦最小 0.99998；传到动作上 RMS 3.1e-4 弧度，是采样噪声的 5.5%、动作 std 的 0.15%。

不改的理由：在线改 f32 也闭不上批形状差，要真闭合必须在线逐帧 batch=1，推理延迟随帧数线性涨；离线改 bf16 要重建整个 400 ep 库并与官方 framesamp 表脱钩。这个量级也解释不了三 seed 24.2% vs 24.5% 的差。

### 2. 整段前向 vs 缓存分步：数值差从哪来

两条路 mask、位置号、可见性结构 120/120 全等，逻辑没错。差异在前缀那一步就出现：`llm([prefix, suffix])` 的 query 有 1204 行、`llm([prefix, None])` 只有 1184 行，XLA 选了不同的分块与归约顺序，同一份前缀 token 算出的 K/V 就差了 0.4%～0.6%，18 层累积到速度场 0.19%～0.30%。四次独立跑数字不同，还含 autotune 的非确定性。

把参数和激活都升到 f32、matmul 精度设 highest 后重跑同一输入：前缀 KV 逐位相同（差为 0），速度场只剩 1.35e-7。只升 f32 但 matmul 仍走 TF32 时差为 4.9e-4。三档单调下降，坐实是纯数值来源。计划把阈值定在 rel_fro ≤ 1e-3、ulp_p99 ≤ 4，是假设「18 层每层 ≤ 1 ULP」，事后看这条假设不成立；脚本没有放宽，判定行保持 FAIL 并带 `threshold_pending_user_review` 标记。

### 3. 测试集出现训练未见的 goal 组合

24 集里 ButtonUnmask 测试集第 3 集的 prompt「first press the button, then pick up the container hiding the red cube, finally pick up another container hiding the green cube」不在 400 集训练数据的 26 种 prompt 里：该任务 3 色 × 9 种目标组合，100 集训练样本只出现 8 种，唯独缺「red → green」。两侧文本本就全小写、tokenizer 同一份、去大小写和空白后仍不匹配；训练集 5 集上 prompt token 120 点全等。所以这是训练数据对测试目标的覆盖缺口，不是链路改了 prompt。按计划它仍是阻断判据，保持 FAIL。

### 4. 单进程第 28 次建仿真环境必崩

现象：`eval.py` 同一进程第 28 次 `EnvRunner.make_env` 抛 `vk::createInstanceUnique: ErrorIncompatibleDriver`，前 27 次正常，此前只能划「每进程 ≤ 27 集」红线。

根因：每次 `make_env` 时 ManiSkill 新建 SAPIEN 渲染系统，svulkan2 全局渲染 Context 重建一次，即一次 `vkCreateInstance` 加一次加载 NVIDIA Vulkan 驱动（dlopen `libGLX_nvidia.so.0`）；每次 `close_env` 时 Context 引用计数归零析构，驱动被 dlclose。NVIDIA 驱动用 initial-exec TLS，dlclose 后 glibc 的静态 TLS 余量每轮净漏 64 字节；glibc 2.34 默认余量 512 字节撑到第 27 次，第 28 次加载驱动失败，Vulkan 返回 `INCOMPATIBLE_DRIVER`。fd、显存、内存、库映射前 27 轮全部平坦，泄漏藏在 ld.so 记账里，`/proc` 看不见。

决定性证据：C 层最小复现里把 TLS 余量调成 0 / 128 / 256 / 512 / 1024 / 2048 字节，崩溃轮分别是 19 / 21 / 23 / 27 / 35 / 51，六档零误差落在「崩溃轮 = 19 + 余量/64」上。只建不销毁连建 64 个 instance 反而不崩，说明是销毁-重建循环致命，而不是实例累积；fd 上限、显存、Python 对象未释放（`gc.collect()` + 显式释放渲染器仍第 28 轮崩）都已证伪。

修法：钉住一个 `sapien.render.RenderSystem` 让 Context 永不归零（35 轮不崩，建环境 6.8 s → 1.2 s）；或起 `eval.py` 时设 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192`（35 轮不崩，每轮仍重建）。两份 patch 在 `docs/training-doc/tic-vulkan-makeenv/records/`，未应用；应用前「每进程 ≤ 27 集」继续有效，本轮 24 集评估正是按它设计的。环境：驱动 595.71.05、glibc 2.34、SAPIEN 3.0.3、Vulkan loader 1.3.224。

## 六、顺带查实的事

- 训练 run 快照里 motion 库路径记成了 40 ep，训练时靠 `MMEVLA_MOTION_STORE` 覆盖到 400 ep。不覆盖时 `check_same_source` 直接 raise，不会静默用错库；推理侧根本不读离线表。记录瑕疵，不修。
- 在线 GPU 现算的位置编码表与库表前 586 行 sha256 相同；这张表在 CPU 上算会静默偏离，所以 policy server 必须在 GPU 上跑。
- 撞运动预算的后果是静默失败：`_prepare_motion` raise → 评估把该集记成 `"error"` → 续评时被重评覆盖 → 汇总求和遇字符串抛错又被吞掉。边界是 demo 段 ≥ 289 帧，当前最长 216。
- 运动 token 与帧 token 的数值尺度比 0.083，训练前是 0.166，低于观察带 [0.3, 3.0]。只记录，不解读。
- checkpoint 盘上是 f32/bf16 混合，生产按 bf16 加载全部叶，这一步对动作的影响是动作 std 的 0.4%。观察项。
- 旧驱动 `motion_gates_online._drive` 在环境步数恰为 16 的倍数时会比真实 `eval.py` 多一个决策点，本轮 5 集未触发；对拍脚本已改用与 `eval.py` 同一公式。
- eval 日志没有逐集结果行，`progress.json` 分不清 timeout 与普通失败；汇总器改按「顺序 + exec_start_idx + task_goal」三重对应，timeout 按「该集推理 82 次」判并与视频文件名交叉核。评估必须单次运行不续评，续评会把 error 条目删掉重评。
- 组 D 的 step-0 base loss 为 0.706138，9 月 4 日同批同初态记录为 0.704831，相差 0.18%；两个 HEAD 之间无数值代码改动，差异来自 GPU 对或编译层面，如实并列未追查。

## 七、没做的事

不修帧特征编码器精度差；不放宽 `VT_FULL_VS_CACHED` 阈值；不应用两份 Vulkan patch；不给 `visible_motion_frames` 加上界；不修快照路径记录瑕疵；不做 EMA vs 训练参数对比（做不了，checkpoint 只存 EMA）；不改主线 `src/mme_vla_suite/policies/` 与 `examples/robomme/`（探针全部靠实例属性遮蔽）；不扩大样本、不做预算消融、不重跑三 seed。

## 八、留档与工具

| run | 内容 | 结果 |
|---|---|---|
| `docs/training-doc/tic-l0-rhythm-40k/` | 第 0 关 + 评估节奏 + 400 ep 库 A19 | 全 PASS |
| `docs/training-doc/tic-obs-model-40k/` | 第 1–5 关，运动 token 查表 | 12/13 阻断 PASS，`VT_FULL_VS_CACHED` 待裁决 |
| `docs/training-doc/tic-sidecar-40k/` | 第 1–5 关，运动 token 真 sidecar 现算 | 13/14 阻断 PASS，同上 |
| `docs/training-doc/tic-t3-causal-40k/` | `T3_MOTION_CAUSAL` 新口径重跑 | PASS，36/36 叶 |
| `docs/training-doc/tic-eval-probe-40k/` | 24 集真仿真闭环 | 6/7 阻断 PASS，`EVAL_PROMPT` 待裁决 |
| `docs/training-doc/tic-vulkan-makeenv/` | Vulkan 崩溃复现与两种修法 | 复现第 28 轮，两修法 35 轮不崩 |

工具（`scripts/training/`）：`g0/check_config_provenance.py`（第 0 关）、`tests/eval_rhythm_gates.py`（评估节奏）、`g0/compare_train_infer_obs.py`（第 1–5 关，`--motion store|sidecar`）、`g0/serve_policy_probe.py` + `g0/summarize_eval_probe.py`（第 6 关）、`tests/motion_gates_model.py`（A19 / T3 新口径）、`legacy-eval/probe_vulkan_makeenv.py`（Vulkan 复现）。原始日志在 `v1-store/reports/tic/`、`tic-dev/`、`tic-vulkan/`（不进 git）。

commit：`892f73e` docs 归档、`a8cfa17` 对拍工具（commitV7.1）、`9b3b95f` 起跑预提交。既有依据：`siglip-ab-replay-40k`（编码器差异首次量化）、`aws-t3-open-s100`（9 月 4 日 T3 记录）、`eval-official-framesamp-context`（首次记录第 28 次崩溃）、`eval-3seed-context-vs-motion`（三 seed 数字，环境 A）。兄弟正本：`motion-memory.md`、`dataloader-restructure.md`。
