# 512 MotionJEPA／Wan九槽训练计划：demo补帧与对抗审查

**结论：修订后的计划在本轮审查范围内通过。** 首轮三路独立审查发现 **1项P1、4项P2**；这些问题已回写计划，并完成独立闭合复核。复核时另补清空段 `row_base=null` 的类型边界。独立CPU规格探针通过，1800个原demo尾窗的实际latent字节也与既有输入块SHA逐窗匹配。

`PLAN_ADVERSARIAL_REVIEW=PASS scope=plan_and_cpu_contract unresolved_reported_blockers=0`

**这不是实施或训练验收。** 新schema2解析器、Wan投影、四卡并行训练、异步保存重叠和超时控制均尚未实施／运行；不能将本报告写成 `CAPACITY_AB=PASS` 或模型训练通过。

计划正本：[1022-h512-vs-wan9-modulation-plan.html](../../root-archive/1022-h512-vs-wan9-modulation-plan.html)。本轮未改生产源码、实际训练配置或 `docs/handoff/current.json`，未启动模型、建库、训练或评估。

## 审查版本与证据来源

| 阶段 | 固定版本或指纹 | 范围 |
|---|---|---|
| 改动前必要性审查 | `5cfe4e106c25f88a3f1d6a52b9b8aa5278e9d65f` | 尾窗公式、保留目的、因果与mask；`ADJUST_REVIEW=PASS scale=same`。 |
| 首轮详细审查 | `53d0fd7b27684d200757a8b9249e35e20c19de47` | 补帧后的完整HTML及同提交生产源码；三路各自只读固定Git对象。 |
| 闭合复核 | `e3c248ad1c8769e2b02fa4ba891455f18316a620` | 原5项发现的计划修订；数据与运行审查者分别复核，不扩大范围。 |
| 本次交付计划SHA256 | `c0b79a210701a1273e3e550af8153cbe2d280c1ce632fa7c161f2872004436f7` | 在闭合版之上采纳空段类型建议，移除陈旧中途轮数，补本报告入口；主会话核对。 |
| 真实数据来源 | 固定父manifest、旧index、恢复receipt、4995份源段metadata与tail latent | 只读检查由主会话执行；源文件指纹及统计见JSON。 |

角色及判定原文摘要保存在 [review_summary.json](records/review_summary.json)。三路独立审查者均使用 `gpt-6-astra / ultra`；它们没有执行主会话的CPU探针，不能把两种证据混为同一人的实测。

| 独立专项 | 首轮结果 | 修订后结果 |
|---|---|---|
| 数据／索引／尾窗 | `PLAN_DATA_REVIEW=PASS findings=1 blockers=0` | `P2_FINAL_FILENAME_PLAN_CLOSURE=CLOSED blockers=0`。 |
| 模型／数值路径／公平性 | `PLAN_MODEL_REVIEW=PASS findings=0 blockers=0` | 原专项无须修订模型架构。 |
| 并行运行／容量／保存与预算 | `PLAN_RUNTIME_REVIEW=FAIL findings=4 blockers=1` | `PLAN_RUNTIME_REVIEW=PASS findings=0 blockers=0 closed=4`。 |

## 首轮发现及修订

| 编号／优先级 | 原问题与影响 | 已写入计划的最小修订 | 闭合证据 |
|---|---|---|---|
| R1／P1 | 原预检每臂30步、正式save_interval5000，按现训练循环只能测终点存盘，漏掉非终点异步存盘与后续更新、取批、H2D的内存重叠。B输入5.625GiB/批使这项差异有实际容量意义。 | 计时后两臂共同触发非终点保存，每臂继续至少5次真实更新，再等待保存完成、严格回读保存时EMA参考；单列保存阶段峰值，不能拿后续EMA作参考。 | 运行审查者在闭合版确认。源码依据为 `train.py::main` 的保存条件和 `checkpoints.py::save_state/initialize_checkpoint_dir` 的异步保存。 |
| R2／P2 | 40h／46h只写预算账本，未定义READY失败、到期停止提交更新、保存宽限和卡住时退出；现训练器异常路径的finally也未等待存盘。 | 预检≤30分钟／每臂≤120更新；READY／首更新≤20分钟；共同 `D=min(T0+46h,T1+40h)`，D−15分钟停新更新并保存，finally等待；外部控制器D−60秒TERM、D时KILL仅登记进程组，记录原始退出码、部分完成及存盘状态。 | 运行审查闭合。故障注入仍是未来实施要求，未在本轮运行。 |
| R3／P2 | 各自20步共同起跑不能保证慢臂尾段仍有另一臂训练；现StepTiming主线程时间不能反推出GPU完成吞吐。 | 快臂计完后继续非计时真实更新陪跑，直到慢臂计时结束；20步窗口边界同步完成态，报告20/T，不用host_step_s计算任意重叠区间GPU吞吐。陪跑、保存后更新均受上述额度限制。 | 运行审查闭合。没有新增正式训练逐步跨组等待。 |
| R4／P2 | 报告曾承诺逐步loss，但训练按log_interval100归约并丢弃逐步infos，不能从均值还原逐步值。 | 明确第0步、每100步区间均值、最后不足100步尾段；按真实区间端点对齐，不承诺逐步loss。 | 运行审查闭合，正式日志频率保持。 |
| R5／P2 | 最终encoder用 `checkpoint_final.pt`，清单只点名loader两处旧文件名解析，少列encoder cache与pack provenance里的数字解析。 | 补 `encode_motion.py::encoder_cache_identity`、`pack_motion_store.py::gather_provenance`；新A读取已核验内容中的e60／完整epoch／step／sample并联合最终SHA，旧kind保留旧名称合同。 | 数据审查闭合。未把当前未改代码本身判为方案失败。 |

同时采纳了三个执行提醒：`cmd_verify/check_bitexact_record`按A/B来源类型验证；`motion_checks::cmd_a9set`分开target row ordinal与source chunk；schema2严格拒绝bool、float、数字字符串伪装整数。闭合复核指出空段例外，已明确 `num_grid=0、starts=[]、row_base=null`，非空段row_base才必须是连续非负int，并加入相应负例。

## 尾窗规则为何成立

对demo真实长度 `L≥17`，定义：

```text
regular = range(0, L-16, 32)
tail = 16 * floor((L-17)/16)
starts = sorted(unique(regular ∪ {tail}))
```

令 `L−17=16q+r`，其中 `0≤r≤15`，则 `L−tail=17+r∈[17,32]`。所以尾窗有17–32真实帧，恰补1–16份最后demo帧；它必是原grid16合法起点。常规窗口跨度33、步长32，内部连续覆盖；额外尾窗位于最后常规起点或其后16帧，接续覆盖到 `L−1`。L<17直接空集，不计算负尾起点。

这保留的是**全部原demo最后补帧窗**，不只是“末帧已经被某个窗口覆盖”：

- 严格grid32：65,419窗，749补帧窗；1009段demo的末尾1–15帧未被motion窗口覆盖。
- 另外42段已经由完整grid32窗口覆盖到末帧，但原grid16最后补帧窗仍不在新网格上。
- 本规则额外加入1051窗，包含上述1009＋42；最终66,470窗，全部1800补帧窗保留。若只求尾帧覆盖，1009是更小增量；本版选择1051是为了保持原尾窗语义。

| demo长度 | 目标起点 | 最后窗真实／补帧 | 重要性质 |
|---:|---|---|---|
| 16 | 空 | — | 不生成非法短窗。 |
| 17 | `[0]` | 17／16 | 最低真实帧数。 |
| 32 | `[0]` | 32／1 | 只补1帧。 |
| 33 | `[0,16]` | 17／16 | 已完整覆盖末帧，仍保留旧补帧窗。 |
| 48 | `[0,16]` | 32／1 | 尾起点不是32。 |
| 49 | `[0,32]` | 17／16 | 尾窗已在主网格，必须去重。 |
| 64 | `[0,32]` | 32／1 | 另一端点边界。 |
| 65 | `[0,32,48]` | 17／16 | 非等距末窗；源chunk `[0,2,3]`。 |

demo窗口内复制只取demo末帧，故进入exec时全部demo窗有效。L33、s16的尾窗虽有名义终点48，但实际只读帧16…32和复制的32，所以t33已合法；不能套exec的完整窗口终点条件。exec仍要求 `u+32≤t−exec_start`，禁止尾补帧。

## CPU规格、全量metadata与真实尾窗字节检查

主要结果见 [window_probe.json](records/window_probe.json)，可复现源码为 [window_probe.py](records/window_probe.py)。该脚本只使用标准库，不导入生产模块，不运行模型。

| 检查 | 覆盖及结果 | 能说明什么 |
|---|---|---|
| 有限长度枚举 | L=0…8192，demo/exec共 **16,386组**；公式与独立“旧grid16筛偶数＋最后demo行”oracle一致。 | 覆盖全部模32余数与边界；任意L的成立另由上述代数证明支持。 |
| 真实来源清单 | **3200集、4995非空段、1405空段、128342源窗**；逐段metadata字段、每行起点／real/pad/pad_source、文件尺寸、sidecar SHA记录均一致。 | 当前源metadata符合既有grid16合同；不是重跑VAE或全量重新hash源bin。 |
| 新选择 | **66470＝24791 demo＋41679 exec**；1800补帧，1051新增，最大可见62，达到最大值的global_episode_idx为1596。 | 80窗可容纳当前完整清单；不能由“最多加一窗”直接推断该最大值。 |
| 因果边界 | **131081个真实边界时刻**，包括exec窗口终点前／当时／之后，demo在exec入口可见。 | 独立规格中无未来exec读取，补帧源不跨段。 |
| 容量与mask规格 | A每窗8槽、B每窗9槽；有效窗口整组true，80窗空位整组false；81窗拒绝。 | 两类padding的期望语义；不是已运行生产模型mask的数值证明。 |
| 轴顺序 | 非方形C/H/W索引105个位置，验证 `j=(cH+h)W+w`。 | 避免32×32方形掩盖转置；实际FP32归一化／模型投影仍待实现后测。 |
| 反例注入 | **38种错误变体全部被独立规格探针拒绝**。 | 说明规格排除这些错误；**不等于未来生产解析器已经能拒绝它们**。 |

原严格grid32的一个真实反例是global episode600，demo L238：最后常规起点192只覆盖到224，剩13真实帧未覆盖；保留旧尾起点208后，输入为30真实帧＋3份末帧237，完整覆盖。

38项反例包括：缺尾、重复、乱序、多余内部16起点；错source chunk／字节偏移／全域起点／real/pad/pad_source；错schema和selection；短demo强造窗、exec补帧；以目标行号×32推起点、按全域帧号%32筛窗、整表全局隔行；错误延迟demo尾窗；复制exec首帧或demo倒数第二帧；屏蔽真实尾窗或只屏蔽末几latent槽；81窗静默裁剪；bool/float/str伪整数；空段row_base非null、非空段row_base为null／bool。具体名称均保存在JSON。

共同期望选择清单的SHA256为 `539d1fa323d32a08a6ca09faf69b8c7a9e6ab4439fb716ea7d9069f715238199`。它绑定期望窗口身份和行序，**不是已运行两条生产数据管线的对拍结果**。

另外执行了 [tail_payload_probe.py](records/tail_payload_probe.py)，结果见 [tail_payload.json](records/tail_payload.json)：

`TAIL_PAYLOAD=PASS windows=1800 additional=1051 bytes_read=1061683200`

对每段demo的原最后补帧窗，直接从源bin按 `source_chunk×589824` 读取589824字节，逐窗计算SHA，与既有motion编码记录中的 `input_block_sha256` 相应项比较。共读约1.062GB，**1800/1800一致**；未只检查文件名、计数或侧车。它验证这些实际latent块仍与已记录输入一致，但没有重新从RGB运行VAE，也没有重新hash全部75.7GB源bin。

## 模型、资源与并行合同核对

| 项目 | 本轮结论 |
|---|---|
| A来源 | 最终EMA必须8×512、e60／25140更新／106191360样本／完整epoch及最终SHA；架构字符串单独不足以识别512。最终checkpoint未在本轮加载。 |
| B读出 | 通道归一化后按C/H/W完整展平为9×16384，再由policy内共享Linear到9×512；不重复空间平均，不离线缓存可训练投影输出。 |
| 两种padding | 合法repeat_last尾窗的A8/B9槽全部有效；容量空位才mask=false。A的8槽是学习query，不能按RGB补帧数切掉尾槽。主会话另核 `even_sampling_indices(0)` 返回 `[0]`，因此首步零motion仍有一帧／64个有效frame token；本轮不扩展全空memory的新行为。 |
| 公共初始化 | 独立Wan key由固定标签从原model_rng派生，不消耗公共rngs；同名公共参数全叶SHA比较是未来闸门。仅same seed不足以保证初值相同。 |
| 图与原实现 | 同层Q/KV共享MemoryRMSNorm；FFN前条件RMS独立，Dense std0.002，原pre_ffw_norm及门控FFN保留，末残差加回原x。 |
| 参数量 | A记忆分支89,974,272；B为98,363,392，差8,389,120；不包含整个policy主干，也不将新投影乘9或18。 |
| memory与RoPE | A=640 motion＋512 frame＝1152；B=720＋512＝1232；K/Q位置按实际S，mask不压缩行号。 |
| 数据表 | A：1,089,044,480 B；B：39,205,601,280 B（36.5131GiB）。两类新layout采用只读mmap；整表复制不应作为新reader实现。 |
| 输入与预取 | B原始FP32输入45MiB/样本、5.625GiB/B128、约1.40625GiB/卡；两组4×1名义motion预取合计23.125GiB。均不是总内存峰值，仍有padding/stack/SHM/H2D及模型状态。 |
| 并行 | 两个单JAX进程，各自FSDP4、全局B128、每卡32；A物理0–3，B4–7；独立优化器与目录，无跨组梯度同步。 |
| 步数与保存 | 每组60000更新／7680000样本；预计12份checkpoint，最终零基目录可能59999；EMA保存完成与实际step分开验证，不能声称恢复了optimizer/RNG。 |
| 日志与时间 | 既有100步均值＋最终尾段；历史1.7346/1.8559秒仅粗估。新共驻计时、非终点异步保存、截止/READY退出都须实现后实测。 |

## 本轮未证明的事项

- 没有实现或执行新schema2读取、Wan投影、paired launcher、deadline或故障注入控制器；38个负例通过不代表这些生产模块已验收。
- 没有运行GPU前向、梯度、四卡容量、异步保存重叠、EMA严格回读、60k训练或评估；模型专项是静态源码对照。
- 没有读取并验证本次最终512 checkpoint，不能提前填最终SHA或登记正式来源。
- 当前全部源bin只做尺寸/metadata/侧车一致性检查，真实payload只逐窗hash了1800个尾窗；未重新hash全量源bin，未重算RGB/VAE数值。
- 当前规范性窗口测试与真实metadata能证明该清单下最大62窗，但换新数据、改selection或改变窗口策略后必须重算，不能沿用80容量结论。
- 单训练seed、不同8/9槽数、额外8.389M参数及RoPE坐标差异仍是实验解释边界；本计划只比较训练表现，不声称闭环成功率或单因素因果。

## 复现与证据文件

从policy仓库根目录执行以下独立CPU检查；只写指定临时结果，不修改源数据。需要当前已存在的恢复资产及旧motion metadata，脚本不会下载或自动补数据。

```bash
UV_CACHE_DIR=/scratch/hongze/.cache/uv uv run --no-project python \
  docs/audits/h512-wan9-plan-20261022/records/window_probe.py \
  --output /tmp/h512-wan9-audit-recheck/window_probe.json

UV_CACHE_DIR=/scratch/hongze/.cache/uv uv run --no-project python \
  docs/audits/h512-wan9-plan-20261022/records/tail_payload_probe.py \
  --output /tmp/h512-wan9-audit-recheck/tail_payload.json

sha256sum 1022-h512-vs-wan9-modulation-plan.html
git diff --check
```

两次最终探针退出码均为0。耗时记录在JSON中，仅代表这些CPU探针，不能用来估计模型训练性能。HTML还检查了两个一级部分、全部相对链接与唯一ID、两张SVG的XML和1280／390像素视口；主会话逐图目视复查，图中文字与实际80窗形状一致，无重叠或越界。

`IMPLEMENTATION_STATUS=NOT_STARTED`；`GPU_CAPACITY=NOT_RUN`；`TRAINING=NOT_STARTED`。
