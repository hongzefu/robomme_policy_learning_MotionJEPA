# Motion memory 审计：结论、技术依据与修复顺序

**motion 已接进模型，正常流程有较强证据支持；但还没有证明训练后的模型利用了任务相关运动内容，也没有证明它提高了闭环成功率。最先要修的是超时后的响应错配，其次是数据身份校验和评估错误记录。**

本文把判断和技术依据放在一起：先说明实际影响，再给触发条件、代码定位和证据范围。所有问题仍是审计发现，本文没有修改实现。

## 1. 审计范围与当前判断

- 实施前基线：`442a7b9847de531d9316c542a5e0880ee442c81a`。
- 待审版本：`c5551e5919b0f999768e86d9d737546b3a57f8eb`。全文代码定位均指这一版本，历史实验另注明其版本。
- 环境：B，AWS 单机、8×A100。原审计开始和结束时 HEAD 均为待审版本，工作区干净，没有需要排除的未提交改动。
- 方法：只读代码、固定版本差异和原始留档；没有运行仓库脚本、测试、训练或评估。本文是用户随后要求保存和扩充的报告，不把静态分析写成新实测。
- 覆盖：离线建库、dataset/transforms、模型与训练恢复、正式在线调用、相关 M/A/P/T/TIC 验证及历史评估记录。没有重新运行外部 benchmark，也没有重新核算大型权重和数据资产。

| 要判断的事 | 结论 | 把握与限制 |
|---|---|---|
| 时间和数据有没有接错 | 在已检查的正常路径中，未发现未来窗口提前暴露或跨 demo/exec 取帧 | 对当前契约把握较高；文件错置和异常恢复另有缺陷 |
| 训练、推理能不能对上 | 受控样本的装配、排序和可见性能对上 | 真实数值路径存在差异，不能称全链逐位一致 |
| motion 有没有进入模型计算 | 进入了，新参数有梯度；初态模型对有效内容敏感 | 不等于真实 40k 模型已经学会利用运动内容 |
| 训练有没有学到相关运动信息 | 尚未证明 | 缺保持数量、mask、位置不变的内容替换实验 |
| motion 有没有提高任务成功率 | 现有三 seed 未观察到整体提升 | 不是公平训练消融，也不足以证明无效或等价 |

## 2. 实现到底怎么工作

### 2.1 离线：把历史片段编码成可查表的 motion

完整路径是：原始 H5 → demo/exec 分段 → 33 帧窗口 → Wan VAE → MotionJEPA → motion store → dataset → transforms/collate → HistoryPi0 → loss/梯度。

这里用 `es` 表示 exec 第一帧在 episode 中的编号，`t` 表示当前决策帧，`B` 表示 batch size。

| 环节 | 实际契约 |
|---|---|
| 分段 | demo 为 `[0, es)`，exec 为 `[es, T)`；两段独立起网格，不跨边界取窗 |
| 采窗 | 段内起点 `0,16,32,…`；每窗 `[f,f+32]`，包含两端，共33帧；段长不足33时没有窗口 |
| 原始图像 | front RGB，`(33,256,256,3) uint8`；这条编码路径不额外 resize 或换色 |
| Wan VAE | `RGB/127.5−1`，转成 `(1,3,33,256,256) fp32`；取 `latent_dist.mode()`，不是随机采样；输出整理为 `(9,16,32,32) fp32` |
| MotionJEPA | 严格加载 checkpoint 的 `encoder` 主键及 affine/RoPE buffers；固定资产输出 `(1,1,768)`，展平为 `(768,)` |
| 精度 | VAE 用 fp32；当前 MotionJEPA run 用 bf16 autocast；最终 token 转为 float32 落盘。不能把整条编码链统称为 fp32 |
| store | 按 manifest 的 episode 顺序，每集先 demo 再 exec，段内按起点排序；每行768个 float32，共3072字节；生产库共6832行 |
| dataset | 按 `(episode,t)` 取当时已经完成的窗口；运动位置码取起点对应的 `pos_rows(f)[:,0,:256]` |
| 交付模型 | `motion_emb(B,96,768) f32`、`motion_pos(B,96,256) f32`、`motion_mask(B,96) bool`、`mem_order(B,608) int32` |

不足96个窗口时右侧补零，并把补位 mask 设为 False；超过96则明确报错，不截断早期窗口。dataset 还会校验 pickle 中的 episode/timestep 与 manifest 推导值是否一致。transforms 保留 motion 四个键，collate 加 batch 维；VLA 不再给 motion 内容做一套统计归一化，encoder 的入口 affine 常数来自 checkpoint buffer。

代码入口：[wan_motion_infer.py](scripts/dataset/wan/wan_motion_infer.py) 的 `encode_chunk/load_encoder/motion_token`、[motion_store.py](src/mme_vla_suite/datastore/motion_store.py) 的 `build_index_entries/visible_motion_rows`、[framesamp_dataset.py](src/mme_vla_suite/training/framesamp_dataset.py) 的 `__init__/__getitem__/_pad_motion`。

### 2.2 时间含义：按起点标记，但必须等尾帧出现

motion token 描述的是整个窗口，不只是起点那一帧。代码用起点做排序和位置标记，同时用尾端控制什么时候允许模型读取：

```text
demo：局部起点 s = 16m，只有 s + 32 ≤ es − 1 才存在
exec：局部起点 u = 16m，只有 u + 32 ≤ t − es 才可见
exec 转成 episode 内绝对起点：f = es + u
```

例如 `es=66`：demo 只能有起点 `0、16、32`；exec 的变化如下。

| 当前决策帧 | exec 已可见的窗口 |
|---|---|
| `t=66` | 无 |
| `t=97` | 无，第一窗还差最后一帧 |
| `t=98` | `[66,98]` |
| `t=113` | 仍只有 `[66,98]` |
| `t=114` | 新增 `[82,114]` |

因此“把 `[66,98]` 的 token 排在起点66附近”不意味着在 `t=66` 就读到了它。**在已检查范围内，没有走通这种未来泄漏反例。** 长度32的段无窗，长度33恰好一窗；这些边界与端点规则相符。

### 2.3 在线：每16步补历史，再预测下一段动作

正式路径是 [eval.py](examples/robomme/eval.py) 的 `EpisodeEvaluator` → websocket → [policy.py](src/mme_vla_suite/policies/policy.py) → [FrameSampMemory](src/mme_vla_suite/policies/framesamp_memory.py) → motion sidecar → 模型。

1. 每集先 `reset()`。首批提交 `[0,es]`，即完整 demo 加 exec 第一帧。
2. 模型预测20步，只执行前16步。执行中收集每一帧，下一次先 `add_buffer` 这16帧，再 `infer`。
3. `clear_buffers()` 后客户端再传 `exec_start_idx=0`，policy 按协议沿用已保存的段边界；不是把 Video 任务改成无 demo。
4. `FrameSampMemory._encode_ready_windows` 用 `while` 补齐所有已完成窗口。每窗独立喂完整33帧，batch恒1；没有用跨窗口的增量 VAE 编码替代这个定义。
5. 已编码 token 按起点保留，原始帧只保留以后窗口还需要的部分。正常 reset 会重建 episode memory，但 sidecar 客户端跨 episode 常驻——这也是后面超时问题的关键。

训练逐帧采样，`(t−es) mod 16` 的16种相位都有；真实在线决策只发生在相位0。这是频率不同，不是在线遇到了训练绝对没见过的相位。冷启动、稳态应分开看。

预算96也有明确边界：按当前1300步评估，最后决策是 `t=es+1296`，exec 有80窗。`es=288` 时合计96窗，`es=289` 时97窗而报错。当前四任务 `es≤216`，该上界下最多92窗；不能把四任务的安全范围直接推广到长 demo 任务。

### 2.4 模型：motion 如何影响动作

原帧路保留最多32帧，每帧16个 token，共512位。motion 新增96位，变换为：

```text
motion_pos：256 → Linear → 768 → SiLU
motion_emb：768 ───────────────┐
                             拼接1536 → Linear → 2048
帧路512位 + motion96位 → 按时间交错的608位 memory
608 memory + 512当前图像 + 64文本 = 1184位 prefix
1184 prefix + 20动作 = 1204位训练序列
```

新增两个 Linear、4个参数叶，共 **3,345,152** 个参数。它们不命中 `.*img.*` 冻结过滤，进入梯度、AdamW、EMA及保存路径。冻结的是 Wan/MotionJEPA 编码器，VLA 内的这两层按设计训练；可训练参数存储为 f32，前向计算使用生产配置的 bf16。

排序键为 `2×episode内绝对起点+类型`，帧类型0、motion类型1，所以同刻帧在 motion 前。padding 使用哨兵，排到尾部。`embed_memory` 对 token 和有效 mask 使用同一张排列；memory 区的 `ar_mask/na_mask` 恒为 False，无需另外重排。

attention 的实际依赖是：memory 读有效 memory；图像不能读 memory；文本和动作可以读有效 memory。RoPE 位置号为 `cumsum(input_mask)−1`，补位不推进位置，绝对帧时间另由位置特征承载。每次 infer 重新计算 prefix KV，只在该次10步去噪内部复用，没有发现跨决策长期复用旧 motion KV 的路径。

现有测试支持有限 padding 垃圾不会改变有效输出或梯度。不能把它说成“任意无效数据都没影响”：全遮蔽 query 行的 softmax 并非字面上的全零，隔离靠后续 key mask；NaN/Inf 也不在已验证保证内。

代码定位：[percep_mem.py](src/mme_vla_suite/models/representation/percep_mem.py) 的 `PerceptualMemory`、[history_pi0.py](src/mme_vla_suite/models/integration/history_pi0.py) 的 `embed_memory/make_attn_mask/compute_loss/sample_actions`、[sampling.py](src/mme_vla_suite/shared/sampling.py) 的 `memory_order`。

## 3. 已确认的可触发问题

以下结论来自静态调用链分析，本轮没有运行复现。**发现这些缺陷，不等于现有生产表或存档成功率已经被它们污染。**

### 3.1 高优先级：超时后可能把上一集的结果交给下一集

**触发过程：** 窗口A的编码超过60秒，客户端超时返回；sidecar没有退出，稍后仍发回A。评估继续下一集，客户端发请求B，却先读到留在连接里的A。响应没有 request ID、episode 或起点供核对，A就可能被当成B写入 memory。

**代码原因：** [motion_client.py](src/mme_vla_suite/policies/motion_client.py) 的 `MotionEncoderClient.__call__` 在 `recv_response` 失败后没有作废连接；[motion_protocol.py](src/mme_vla_suite/policies/motion_protocol.py) 的响应只有 status和token；`MME_VLA_Policy.reset` 重建 memory 但保留 client。服务端异常只结束该 websocket handler，长存 policy 仍在。

这条路径能接上正式 eval：`success_flag` 只在每个 task 开头设为 unknown；若同任务此前已有一集正常结束，后集异常不会重置它，驱动可以继续下一集。**如果任务第一集就失败，通常会提前返回，不能泛称所有超时都会继续。**

已检查的反证：正常请求有互斥锁；编码器明确报错通常会退出；P1测了错帧和子进程退出。但这些不能阻止“计算慢、超时后仍成功返回”的情况。

**影响与最小修复：** 这是可能静默使用错误 episode 内容的问题。超时或协议异常后应立即作废连接；恢复使用新 client；请求和响应都携带并核验身份。需要故障注入验证迟到响应、部分响应和 reset。

### 3.2 中优先级：同长度段放错文件名，pack/verify可能一起通过

**具体反例：** 生产留档中 `ButtonUnmask_ep0_exec` 和 `ep1_exec` 都有17窗。抽取完成后，若把ep1的 token文件、SHA文件、metadata三件套误复制成ep0文件名，文件内部的SHA仍自洽，metadata内部也仍诚实写ep1。

**代码原因：** [pack_motion_store.py](scripts/dataset/pack_motion_store.py) 的 `read_segment_tokens` 核字节数、SHA和窗数，但不核 `segment/g/segment_kind`；`gather_provenance` 没有把 token 的输入 latent SHA 与对应段绑定；`cmd_verify` 又和同一批源token比较。结果可能是 index正确写ep0，内容却来自ep1，仍得到 `verified`。

已检查的反证：抽取阶段汇总能挡住部分当场错名，但独立pack入口不调用它，挡不住阶段完成后的错置。**当前库另有全量D3对拍，未发现此类内容不一致；M1直接读候选表，本身不能独立排除它。**

**最小修复：** 读每段时核对完整身份，同时绑定 `input_latent_sha256` 与对应latent资产；保留独立重编码对照，不能只比较源文件与其复制品。

### 3.3 中优先级、基线已有：错误记录会被重试覆盖

**代码原因：** [eval.py](examples/robomme/eval.py) 的 `evaluate` 把异常记为字符串 `"error"`，随后 `sum(values)` 不能处理字符串，汇总异常被捕获后外层循环继续。当轮写入episode键为int，跳过检查却用str，因此部分结果可能再次评估并覆盖。重新启动时，`setup_log_dict` 又删除error条目。

**影响：** 最终没有error，只说明最后保留下来的结果没有error，不能证明第一次就成功完成评估。失败次数、首次分母和重试过程不易追溯。这个问题在实施前基线已有，不能归因于 motion 新增代码。

已检查的反证：全布尔结果可以正常汇总；最终三seed名单完整。历史留档本身也记录过端口事故和崩溃后补评，但尚未证明这里的静默覆盖缺陷改变了最终成功率。

**最小修复：** 统一episode键类型，显式记录每次尝试及终态，固定首试分母；重试不得覆盖历史；汇总失败应明确退出。

### 3.4 中低优先级：编码续跑检查找错metadata文件

[encode_motion.py](scripts/dataset/wan/encode_motion.py) 的 `main` 把 `key + ".f32"` 交给 [wan_common.py](scripts/dataset/wan/wan_common.py) 的 `segment_outputs_complete`，于是检查的是 `<key>.f32.metadata.json`；实际写出的却是 `<key>.metadata.json`。

正常已完成段因此仍被判未完成，进入删除重算。单worker重启就能触发；并行时所有worker遍历同一清单、claim完成即释放，落后的worker也可能重做已完成段。汇总 `items=600` 只数最终唯一段，不能排除重复计算。

**影响与修复：** 主要是浪费计算和降低续跑可靠性，同数值条件下重算不代表token错误。统一文件名契约，并在取得claim后再检查完整产物；本轮没有量化历史run实际重复了多少工作。

### 3.5 中低优先级、仅legacy工具：总条数对不等于名单对

[aggregate_seed_runs.py](scripts/training/legacy-eval/aggregate_seed_runs.py) 的 `collect_one` 对跨shard重复episode静默覆盖；`main` 算了missing，却只按总数量相等、error为零判DONE。

反例：预期 `0…49`，实际 `1…50`，仍是50条；不同cell少一条、多一条也可能抵消。应逐cell核精确集合，并拒绝重复来源和非法结果类型。**现有三seed JSON经额外核查，各cell确实为0…49，没有触发这个缺陷。**

## 4. 训练、恢复与既有验证，哪些结论能保留

### 4.1 关闭motion和恢复checkpoint

`motion.enabled=false` 时两个新增层根本不创建，不消耗新增RNG；原memory路径直接返回，不重排；新增观测字段为None。静态差异和历史对照支持关闭态语义保留，但不代表最终版本已对任意输入、设备直接做过逐位验证。

新motion checkpoint恢复会校验resolved history YAML、hash、provenance、启用状态、参数树及shape；含motion参数却缺快照会被拒绝。旧非motion checkpoint保留 `history_config.txt` 兼容路径。对应 [policy_config.py](src/mme_vla_suite/policies/policy_config.py) 的 `create_trained_policy/_load_resolved_snapshot/_assert_param_tree_exact`。

**严格加载不等于无损续训。** [train.py](scripts/training/train.py) 的正式入口明确拒绝resume/overwrite；保存的是推理参数（启用EMA时为EMA）和norm assets，没有完整AdamW状态与训练计数，因此不能宣称可恢复原训练轨迹。

### 4.2 PASS的实际覆盖范围

| 验证 | 原始证据能支持什么 | 不能据此推断什么 |
|---|---|---|
| D2/D3 | 400ep库6832窗的帧来源、latent/token对拍；D3表SHA与store一致 | encoder学到的运动语义有用 |
| M1/A19 | 101066个真实dataset样本，按manifest重算窗口期望，装配零失配；旧A19硬编码分布FAIL已修复 | 候选motion表本身的来源独立正确 |
| M2–M4 | 独立排序、显式线性层/gather复算、有限padding与梯度检查 | 任意输入、任意后端都正确 |
| P1–P4 | 正常IPC、错帧、窗口补齐、段边界状态等 | 超时后的迟到成功响应安全 |
| P5 | 40ep共772窗在线编码与离线表相同 | 整个真实eval或跨GPU全链逐位一致 |
| T2/A22 | 历史关闭态100步及三个梯度fixture对照相同 | 原始442到最终c555的全域实测等价 |
| TIC真sidecar | 指定模型、5集120个决策点中，140窗现算与表相同 | 未覆盖episode、长轨迹和故障状态都相同 |

T2实际reference为 `c5925d96305f771058e2206ae89461269af9d97c`，candidate为历史 `8093ebd/cbf24e9`。记录包含100步标量、5次state摘要、7次batch摘要；A22是3类fixture、32个trainable梯度叶。静态桥接检查支持沿用相关结论，但不能改写实验锚点。400ep的A6只查前40条且用同一清单，也不能单独当独立全库证明。

主要原始记录：[D3编码报告](docs/dataset-build-doc/4task-motion-400ep/records/encoder_report.json)、[M1/A19输出](docs/training-doc/tic-l0-rhythm-40k/records/l0-rhythm-m1.txt)、[P5输出](docs/training-doc/aws-p5-online/records/p5_driver.txt)、[T2输出](docs/training-doc/aws-t2-cand-s100/records/t2_gate.txt)。

### 4.3 “动作逐位一样”比较的其实是什么

[compare_train_infer_obs.py](scripts/training/g0/compare_train_infer_obs.py) 的S臂，直接把在线历史帧特征替换成离线表真值。随后T/I两边都调用同一个 `f_prefix`，最终动作都调用同一个 `policy._sample_actions`。

所以 `KV_T_VS_I/ACT_T_VS_I=PASS` 有价值：它证明两侧交付的受控输入能产生相同推理结果。**它没有独立证明训练的整段前向与推理的缓存前向逐位相同。** 后者由L4分别重构两条前向来比较，原始结果仍是FAIL。

训练/推理的数值边界至少有三项：

| 差异来源 | 已记录数值 | 判断边界 |
|---|---|---|
| 历史帧：离线f32逐帧，在线bf16批编码 | 2409帧特征相对差 `0.003328`；对应动作差约为该实验采样噪声参考的5.5% | 小的开环差异不等于闭环无影响 |
| LLM整段前向与prefix缓存前向 | 120点速度场相对差约 `0.001855–0.003051`，ULP p99为10–22.5，超过预设 `0.001/4` | 两个TIC总gate仍FAIL，不能按解释合理就改称PASS |
| 推理将全部checkpoint参数转bf16，训练只将冻结参数转bf16 | 仅1点对照，动作 `rms_norm=0.0006769` | 这是基线已有数值差，且覆盖太少；L4两侧都用生产bf16权重，没有包含这项差异 |

sidecar run的120点中，有效prefix KV逐位相同为 **0/120**，但mask子块、动作行与位置号120/120相同。f32 highest诊断只测5点：prefix KV误差为零，速度场相对差 `1.352e-7`。这强烈支持这些点主要是数值来源，不能写成“速度场逐位相同”或“所有状态的差异都已排除”。[原始输出](docs/training-doc/tic-sidecar-40k/records/obsmodel-sidecar.txt)

真仿真探针另查了24集、359次infer，验证对应关系、公式、排列和后端等不变量；没有训练真值oracle。最大motion数35，未覆盖预算边缘；一个训练未见goal使总判据为 `6/7 FAIL`。[原始汇总](docs/training-doc/tic-eval-probe-40k/records/eval_probe_summary.json)

### 4.4 T3证明了接线，但没有证明40k模型学会了motion

`tic-t3-causal-40k` 的名字容易误导。[motion_gates_model.py](scripts/training/tests/motion_gates_model.py) 的 `cmd_t3mechanism/_t3_init_state` 实际重新初始化模型，要求命中短run的step0，使用40ep库的一个b8批次，没有加载39999。

新版记录确实是：36/36个trainable叶覆盖，排除叶为0；有限padding干预不改变loss/梯度；有效内容清零或置乱、位置扰动会改变loss或梯度。有效motion内容输入梯度范数为0.94053。**这是扎实的初态接线和敏感性证据，不能替代训练后模型的内容干预。** 原先T3的FAIL记录也应保留，不能因为新版通过而删除历史。[原始输出](docs/training-doc/tic-t3-causal-40k/records/t3-causal.txt)

## 5. motion有没有用：现有结果能说到哪一步

### 5.1 闭环成功率没有显示整体提升

复算三seed原始结果如下。每个任务的150次来自同一批50个环境、三种policy动作采样seed；合计是200个环境的重复采样。

| 任务 | 官方context | motion | motion差值 |
|---|---:|---:|---:|
| ButtonUnmask | 41/150 | 39/150 | −1.33pp |
| VideoUnmask | 46/150 | 43/150 | −2.00pp |
| ButtonUnmaskSwap | 30/150 | 26/150 | −2.67pp |
| VideoUnmaskSwap | 30/150 | 37/150 | +4.67pp |
| 合计 | **147/600，24.50%** | **145/600，24.17%** | **−0.33pp** |

这不是三个独立训练seed，也不是600个独立环境。最终1200项结果中，各cell恰为ep0–49，无缺失或非法结果；motion的9次timeout、官方的1次timeout均计入失败分母。技术重试在历史留档中确实出现过，因此最终无error不能解释为全过程无异常。[逐集JSON](docs/training-doc/eval-3seed-context-vs-motion/records/per_episode_by_seed.json)、[事故与补评记录](docs/training-doc/eval-3seed-context-vs-motion/result.md)

按每个环境先平均三seed差值，再以200个环境为单位作配对正态近似，整体95%区间约为 **−5.51pp～+4.84pp**；VideoUnmaskSwap的50个环境约为 **−5.21pp～+14.55pp**。这只是两份固定模型、当前任务范围下的描述性估计，不涵盖训练随机性。单任务+4.67pp可以作为后续线索，不能当已确证优势。

### 5.2 比较还不能把差异归因于motion

motion实跑日志确认seed42、b128、40k步、lr1e-4、四任务400ep、无验证集。官方b64、80k、lr5e-5来自上游配置及留档，缺该已发布checkpoint完整的原始训练记录；官方到底用4任务还是16任务也未核实。**稳妥结论是没有建立公平的训练消融，而不是已经证明两组训练数据不同。** 短T3有更受控的对照，但闭环两边都是0/40，也不足以证明收益。

loss下降、非零梯度、参数更新说明模型在优化；token非零说明有输入；attention权重或动作变化说明模型有依赖。这些都不能单独区分它利用的是运动内容，还是新增容量、时间位置、阶段、有效token数或episode身份。置零/移除motion还可能制造训练时少见的输入，性能下降不能单独证明运动语义有用。

### 5.3 编码成本是真实代价

A100正式评估记录里，常规 `add_buffer` 均值约1.23–1.36秒，包含帧编码、IPC和motion；VideoSwap首批约11.8–11.9秒。不能把整项耗时都归给MotionJEPA，也不能只用较短的动作采样时间代表完整延迟。

Ada三seed留档中motion每轮44–52分钟、官方约19分钟，但两组worker并发数分别2和4，不能把墙钟比直接当单次调用成本比。仿真会等待同步推理，没有真实部署的时间截止实验，因此仿真成功率不能自动证明实机实用收益。[A100评估记录](docs/training-doc/eval-awsprod40k-b128-motion/records/per_episode.json)、[Ada评估留档](docs/training-doc/eval-3seed-context-vs-motion/result.md)

## 6. 尚未闭合的风险与文档表述

这些应与上面的已确认实现缺陷分开看。

| 项目 | 目前可信的部分 | 仍需保留的限制 |
|---|---|---|
| 资产锁 | 默认建库从ASSETS_LOCK取权重期望值，sidecar校验实际权重、VAE和数值环境；不是完全只信名称 | `--encoder-run-dir`覆盖时，默认目录被校验，实际override配置另行读取；pack的run_name又硬编码。仅改变precision也可能训练接受、到正式推理才因provenance不一致被拒绝。未发现当前run如此误配 |
| 源码来源 | SOURCE_PIN强校验复制件，依赖Git revision固定 | `--no-sync`环境或import路径被修改后的实际依赖，不等于已被独立运行时指纹完全约束 |
| 数据完整性 | 当前生产表另有SHA与run provenance核对 | 普通motion加载走fast检查，verified不保证之后的文件字节没有被改过 |
| 配置恢复 | history快照、启用状态和参数树有严格检查 | history快照不等于完整运行配置快照；action_horizon、文本长度等行为配置仍取决于调用者，参数shape相同未必能发现误配 |
| 在线异常输入 | 正式eval正常连续送帧 | 重复、缺失观测和重试没有完整端到端保证，内部生成连续索引不能证明物理帧未丢失或重复 |
| 数据独立性 | 策略训练数据与test的seed检查有记录 | encoder来自另一私有库，缺与test的seed/帧hash映射；不能只凭相同ep编号认定泄漏，也不能宣称独立性已证实 |

文档中以下说法应收窄：

- [train-infer-consistency.md](docs/train-infer-consistency.md) 把受控回放和真仿真一起描述为训练真值逐位相同，并据此说“剩下只有token有限或噪声”，超过了实际覆盖。
- “只有两处数值差”漏掉checkpoint参数转换；“f32差异归零”不符合速度场原始值；帧特征写成 `(16,1152)` 与当前 `(16,2048)` 不符。
- [tic-t3-causal-40k的launch](docs/training-doc/tic-t3-causal-40k/launch.md) 中生产指纹表写b128/39999/400ep，实际命令是40ep、短run初态，必须明确区分。
- 三seed留档的“跨机唯一变量是GPU”“一切差异不是bug”缺控制实验支持；benchmark曾从另一外部editable checkout导入，完整版本没有固化，不能把差异全部归因于舍入。

可以保留正常路径的时间契约、模型接线、有限padding隔离和指定样本编码一致性；全链逐位一致、跨硬件排他归因必须收窄；学会运动语义和公平闭环收益仍缺证据。

## 7. 最值得补的实验，按顺序做

以下是实验设计，本轮没有执行，也不代表已经授权启动训练。

| 顺序 | 要区分的解释 | 对照与控制 | 观测和判据 |
|---|---|---|---|
| 1. 故障与边界验收 | 超时明确失败，还是错误结果被继续消费 | 沿真实websocket→policy→sidecar注入迟到响应、部分响应、退出和reset；覆盖es=288/289及最后决策点；固定帧序和请求身份 | 不得接受前一请求/episode结果；记录窗口归属、编码次数、首试终态、p95/p99延迟；失败记录不能被重试覆盖。通过才支持恢复契约 |
| 2. 真实39999内容干预 | 模型利用相关运动内容，还是只依赖数量/位置 | 真实motion、同任务同长度的跨episode替换、时间错位；固定有效数量、mask、位置、其他输入、采样噪声和环境实例；置零只作辅助参考 | 比较动作/损失及配对闭环成功率，按空/非空、冷启动/稳态分层。真实内容持续优于匹配替代才支持语义利用；只有置零下降只能说明敏感或输入分布变化 |
| 3. 公平训练对照 | motion信息带来收益，还是容量/训练条件/选择偏差 | 同数据划分、共享参数初始化、数据顺序、batch、学习率、步数和预定checkpoint规则；比较context、真实motion、等容量但无正确运动内容的控制；多个独立训练seed | 冻结benchmark和实例清单，保留首试error/timeout，报告配对区间、任务差异和延迟。收益超过预设实用门槛且满足部署时限，才支持实际收益；否则区分证据不足、未观察到提升和可靠退化 |
