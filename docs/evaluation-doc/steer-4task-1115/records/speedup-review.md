# steer-4task-1115 提速对抗审查汇总（workflow steer-speedup-adversarial，美东 2026-11-16）

6 方向审查 + 6 核实 + 1 汇总，sonnet，只读；保留发现 34 条。

以下为 workflow 内部工作记录；消费此结果的主 agent 必须仍用简体中文与用户沟通，不要被本报告语言带偏。

# 对抗核实后的提速汇总：MotionMem RouteStick 同局变体 steering 全链

**前提和口径**
- 基线总时长按任务给定的约 8.7 h：变体示范生成 1.2 + N 0.5 + vrec 2.5 + samep 4.1 + 噪声 0.5。
- 下面的「通道小时」是各通道占用时间之和，墙钟要再除以并发通道数。
- 各子报告的数字不能直接相加。它们争的是同一块时间，已在合并时去重。

## ① 根因去重

| 根因 | 合并的发现 | 结论 |
|---|---|---|
| A. 被编码的窗数太多（整条链最大头） | D1-0（基线）、D1-1、D1-2、D5-1、D3-4、D1-3 | 单窗独占约 1.7 s，在 fp32、关 TF32、B=1 的数值钉下是 GPU 算力下限。同卡 3 通道只有约 1.3 倍并发收益，所以只能靠少编窗。 |
| B. 静态队列与阶段屏障造成尾部失衡 | D4-2、D6-1、D6-2、D6-3、D6-4、D4-6 | samep 12 个 GL 通道收工时刻差 77 分钟，aspen 空转约 2.5–3 h。队列还按 j 升序，长 spec 被压在队尾。 |
| C. 每 5 局重启一次策略服务 | D2-1、D5-3、D1-5、D4-4、D2-3、D2-4 | 启动中位约 48–50 s，占 attempt 墙钟 6–9%。chunk 加大不可行（原因见 ③），只剩按 spec 常驻服务端。 |
| D. vrec 重新仿真示范 | D3-1、D1-4 | 约 8 通道小时是仿真。回放帧的方案需要先过帧 md5 判据，只对下一轮新 campaign 生效。 |
| E. 同卡争用与通道数 | D4-1、D4-3、D3-2、D4-8 | GPU 时间片争用，不是 CPU。第 4 通道和加 CPU 都无收益。 |

## ② 前 5 条推荐（按节省小时 ÷ 改动规模排序）

| # | 改什么（文件+函数） | 阶段：从→到 | 对总时长（原约 8.7 h）的影响 | 改数值口径？ | 风险与改动规模 |
|---|---|---|---|---|---|
| 1 | samep 示范窗改读 N 的录制，不再在线重编。<br>`serve_policy_attn.py::MotionContentClient.__call__` 把 `self.inner` 挪到 miss 分支，`begin_episode` 读接收局的 N 录制 npz。<br>`sp_swap_prep.py::assign_samep` / `write_samep_files` 写入 f<es 的示范窗起点和 cache_mask。<br>只对 f<es 的示范窗生效，执行窗仍在线编码。 | samep 通道总时长 42.6→约 23 h（-45~49%）。墙钟约 4.1→2.1–2.4 h。<br>噪声阶段再省 1.2 通道小时（约 0.2–0.3 h 墙钟）。 | **约 -1.7 h，加噪声约 -2 h。** | **会变，需用户认可。**<br>samep 里 392/949（41%）的运行与 N 不在同一硬件（GL 与 aspen 混跑），其非供体示范窗从「所在卡重编码值」变成「N 的值」。<br>同卡型逐位相同（GL 对 GL 533/533 窗相等，最大差 0）。<br>跨站点约 99% 的窗不等，最大差 0.03125（bf16 量级）。<br>注入后对照与干预除供体窗外示范 token 完全相同，配对更干净，但与已跑出的 samep 结果不再逐位同源。 | 约 60–90 行：MotionContentClient 约 20 行，sp_swap_prep 约 30–40 行，spec 与 run_local 参数约 15 行，补 1–2 个测试，再加 2 局冒烟。<br>1. 换 seed 之前先用 5–10 局对照新旧两种方式的成功率差异，作为口径记录。<br>2. N 的 token 属于旧 encoder，只能在同一模型内 N→samep 复用，不能跨模型。<br>3. 缺起点直接 raise，不加兜底。 |
| 2 | vrec 只编码供体会用到的窗。<br>`sp_swap_prep.py` 在 build 阶段给每条 vrec spec 写入起点集合（扫描条目已带 segments 和 es，不必等 N 跑完）。<br>`run_local.py` 透传到 `MotionContentClient`（record 模式下集合外窗返回零向量，不调 inner、不记录）。<br>vgroup=same 的条目保持全窗。 | vrec 需编码窗 19880→4183（21%）。<br>vrec 首个 add_buffer 17.2→约 3.6 通道小时，阶段 25.1→约 11.5（-47~54%），墙钟约 2.5→1.2 h。 | **约 -1.3 h。** | **不变。**<br>被编码的窗走同一函数、同一 B=1 路径，与现在逐位相同。 | 约 60–90 行。核实后比原估的 40–60 行大：window_sets 里的 es 和分段是接收局的，要把 `W_all[j-1]` 经 samep_entries 写入每个变体条目。<br>需确认首推 infer 不因集合外零窗报错（vrec 本就丢弃该输出）。 |
| 3 | spec 级共享拉取队列加成本降序，aspen 并入。<br>`sp_swap_prep.py::distribute` 和 `cmd_requeue` 生成队列的一处，加 `chains/hold_runner3.sh` 的领取循环。<br>用 NFS 上的 mkdir 原子占位，按 j 降序（长 spec 先领）。<br>不动 run_local，run_identity 和产物路径不变。 | samep 静态队列约 4.1–4.5 h→约 3.2–3.4 h，GL 池化约 -30~35 分钟。aspen 2 卡并入后合计约 -50~55 分钟（约 -20%）。<br>vrec、噪声各约 -5~10%。 | **当前口径约 -0.8 h。与 1、2 叠加后阶段变短，绝对节省约减半，叠加后取约 -0.4 h。**<br>D6-2（只重排）与 D6-1/D4-2 同源，不叠加。 | **不变。** | 约 40–60 行（bash 领取循环约 40 行，队列生成 30–50 行）。<br>chunk 级领取不需要：模拟显示 spec 级共享拉取约 241 分钟，静态 LPT 约 238 分钟，chunk 级最多再省约 10 分钟。<br>run_local 没有「跳过已 accepted chunk」的入口，chunk 级要新增子集参数，不值得。<br>aspen 与 GL 的 provenance 放宽字段不同，要按站点保持一致，否则同一 spec 跨站运行时设置不一致。 |
| 4 | 按 spec 常驻策略服务（每 spec 只起一次，客户端每批仍 ≤5 局）。<br>`run_local.py::main` 分流条件加 spec 键，走 `run_oracle_chunks` 同款路径。<br>samep 要给 `build_server_command` 传入整 spec 局并集（plfull 的 swap_map 否则抛 ValueError）。<br>失败语义对齐：run_oracle_chunks 在任一 code≠0 即停，run_chunk 只在 -1 时停。 | vrec 起服务 189→48 次，省约 5.1–5.6 千通道秒（约 5%）。<br>samep 计划 269 块→88 个 spec，省约 6.5–7.2 千通道秒（约 3–3.5%）。 | **约 -0.25~0.3 h。** | **不变。**<br>会话内第 2–5 局本就复用同一进程，常驻只是把复用从 5 局扩到 20 局。 | 约 10–25 行（vrec 约 3–8 行，samep 要多传局并集）。<br>一个 spec 内崩溃的影响范围从 5 局扩大到整个 spec。<br>server.log 变成每 spec 一份，留档和 collect() 路径需核对（oracle 路径已验证过该布局）。 |
| 5 | **条件项，要先过判据。** vrec 改回放：build 阶段 `samep_rs_variants.py::reset_demo` 落盘 front、wrist 和 states。vrec 改成对服务端发 reset、add_buffer 和一次 infer，不起 SAPIEN。 | vrec 仿真侧约 8 通道小时拿掉，vrec 约 -25~33%（核实后），GL 约 2.5→1.7–1.9 h。若已做第 2 条，墙钟边际约 -0.3~0.5 h。 | **约 -0.3~0.5 h，且只对下一轮新 campaign 生效。** 本轮补落盘要重跑 70 min 的 build，不划算。 | **有风险，须先验证。**<br>回放帧必须与在线帧逐字节相同。build 的 md5 是 front+wrist 联合哈希，对不上落盘 front。<br>`RS_SPEC_REBUILD=FAIL frame_mismatch=1`（launch.md）说明渲染并非处处逐字节可复现。<br>判据：同一 spec 在 GL 在线 reset 的 front 帧 md5，与 build 落盘帧 md5 比对。不过就改在 GL 上落盘。 | 约 150–250 行：落盘约 15 行，回放客户端约 100–150 行，可能还要写一个只发协议的 stub 评估器（vrec 的 grant.json / accepted/*.terminal.json 契约）。<br>要补 wrist 图和末帧。<br>vrec_record 里 npz 的 embs 行数随局变化，还有 call_embs 和 call_seq，回放必须复现这些记录语义。<br>性价比低于第 1–4 条。 |

**备选：D1-3（VAE latent 按 episode_spec 和起点落盘，后续模型直接读）**
- 叠加第 1、2 条后，每个后续模型只剩约 4–5 通道小时（约 0.3 h 墙钟），原稿「两模型合计省约 70 通道小时」与第 1、2 条重复计算。
- 「VAE 占单窗绝大部分算力」只是推断，没有分段耗时。
- 缓存必须按卡型分键：跨硬件约 99% 的窗不等。
- 只有确认 VideoRepick 也吃同一份 Wan latent 时才值得做（未核实）。约 80–120 行。

**叠加估算（本轮同口径重做）**
- vrec 约 2.5→1.0 h。
- samep 约 4.1→1.7 h。
- 噪声约 0.5→0.3 h。
- 总计约 8.7→4.7 h，约 1.85 倍。
- 未算第 5 条。变体示范生成 1.2 h 本轮不变。

## ③ 不改代码、只改运行参数或起跑清单能拿到的

| 项 | 做法 | 预期 | 备注 |
|---|---|---|---|
| 起跑清单固化 | GL 一律用 `hold_runner3.sh` 的单 srun 作业步，每卡 3 通道，XLA 0.22，`--gpu_cmode=shared`（env-gl.sh/launch 层）。起跑前在目标卡做一次 3 局冒烟。 | 约 -8~12 分钟（约 2%），并避免约 2 通道小时重跑 | 本轮前 45 分钟只有 8 通道，原因是先试 0.27 OOM，又因 GL 独占模式重排。0.22 的值本来要靠一次实测才知道，节省来自提前测。 |
| 队列重排（只改队列分配） | 先用现有 `requeue` 子命令在阶段中途重排。<br>`distribute` 加成本权重（约 20 行）：按各 j 每局中位耗时 j1 2.2 → j8 4.8 分钟，降序贪心。 | samep 约 -30~40 分钟 | 与第 3 条同源，不叠加。 |
| 噪声作低优先级填充 | 噪声放进 vrec 尾部空洞，samep 就绪后优先领 samep。aspen 从 04:35 起全给 samep。 | 约 -10 分钟 | 噪声与前两阶段无数据依赖。 |
| 加占位（需用户批准） | 6–8 个占位，前提是先做装箱。 | 6 卡约 1.3–1.5×，8 卡约 1.6–1.9× | 超过 4 个占位要先让用户审核数量（AGENTS.md 第 8 条）。launch.md 第 13 条追加 4 个被权限拦截，配额未查。静态队列下尾部失衡会随卡数放大。 |
| aspen 每卡 2 通道（需用户批准） | 起跑前现场看显存余量，OOM 退回 1 通道。 | aspen 自身约 +40–60%，整条 vrec 约 +5–12% | 他人共用卡，显存在边缘（每通道约 15.2 GiB）。数值无影响。 |
| OMP/MKL 线程数 A/B | 同一 A40 占位、每条件 5 局，约 15 分钟。 | 约 0~10%，更可能接近 0 | 同卡 ab/D 涨 2.3 倍与 sim_rest 涨 2.2 倍几乎同幅，更像 GPU 时间片争用，不像 CPU 过订阅。 |
| 不要做的参数 | 第 4 通道、加 CPU 规格。 | 0% | 显存放不下；客户端侧每 rpc 约 1.85 s 与并发数无关。 |
| chunk 加大 | 无法单独调参。 | — | `build_manifest.py` 断言 chunk_size≤5。<br>`run_local.py` 不读 `CHUNK_EPISODES`（那是官方评估路径 run_shard.sh、gl_hold_queue.sh、gl_hold_pool.sh 的变量）。<br>改 `sp_swap_prep.py` 的 `CHUNK` 常量会改 chunk_id 和 run_identity，要重建 manifest。 |
| 其余不动 | JAX 编译缓存（517 个文件，已命中，首次 infer 约 4 s/块）。checkpoint 本地盘缓存。 | <1%、约 1.5% | 不值得改。 |

## ④ 下次 Frozen Wan pooling（wan_fixedpool768）与 VideoRepick 的推荐方案

**两个模型共用的前提**
- 变体示范只取决于 episode_spec，与被测模型无关。
- 本轮 60 局 903 变体的 build 产物（variants、specs）可直接复用，省掉约 1.2 h aspen CPU。
- N、vrec、samep、噪声都要按模型重跑，因为轨迹和 token 随模型变。

**每个模型的执行顺序**

| 步骤 | 内容 | 预计时长 |
|---|---|---|
| 0 | 复用 build 产物。起跑清单：每卡 3 通道、XLA 0.22、`hold_runner3` 单 srun 作业步、`--gpu_cmode=shared`，3 局冒烟。 | 约 0.15 h |
| 1 | N（60 局），用共享拉取队列与按 spec 常驻服务。录制 N 的 record/ 供第 3 步复用。 | 约 0.4 h |
| 2 | vrec：第 2 条只编码需要的窗。噪声作低优先级填充，可并入尾部空洞。 | 约 1.0 h |
| 3 | samep：第 1 条读 N 的示范窗，成本降序的共享队列，aspen 2 卡并入。 | 约 1.7 h |
| 4 | 换 seed 噪声：同样读 N 的示范窗。 | 约 0.3 h（可并入第 3 步尾部） |
| 合计 | 每模型约 3.3–3.6 h（本轮同口径约 8.7 h）。 | 两个模型顺序约 7 h，约 17.4→7 h |

**Frozen Wan pooling 的特殊点**
- 池化在 CPU 上、零参数，没有学习型 encoder。
- 单窗成本在 VAE 之外已无 encoder 段。VAE 占比未实测，不能直接按 1.7 s 外推。
- 起跑前先在同一占位上实测一次 ms/窗，再重估。

**VideoRepick 的特殊点**
- 干预阶段结构相同，可同样复用第 1 条（N→samep 同模型内）。
- 是否同吃 Wan latent 未核实。确认之前不要建 D1-3 的 latent 缓存。
- 若确认同源，且 Frozen Wan pooling 先跑，可让它首次落盘 latent。VideoRepick 的 vrec、N 示范窗读 latent 后只跑 encoder 或池化，再省约 0.3–0.4 h。

**口径与审批**
- 第 1 条是口径变更，换模型后各自单独走「先 5–10 局新旧对照、再由用户认可」。
- 加占位与 aspen 双通道属于计划外资源或占用他人共用卡，开工前要用户批准。

## 评估后不建议做（避免重复论证）

| 项 | 原因 |
|---|---|
| 批量化 B>1（跨通道合批或独立编码服务） | wan_motion_infer.py 文件头记「batch 1 vs 8 改最后一位」，破坏同源同值前提。 |
| TF32 / bf16 / channels_last | 改 token 值，已跑结果失去同源。 |
| 重叠帧 latent 拼接 | Wan VAE 因果分组，窗 s+16 的首帧位置与窗 s 的第 16 帧落在不同分组边界，latent 不相等。结构推断，未实跑。 |
| N 与 vrec 合并 | N 是全长不干预读数基线。DONOR_ENC=FAIL windows=1029 bad=456 maxabs=0.03125 说明重建局与自然局编码已有 bf16 量级差，合并会改读数口径。 |
| 跨 spec 常驻服务 | 要改服务端身份契约（run_identity 含 manifest），在第 4 条之上只多约 1%。 |
| 遮蔽分割填充 / 关相机 / 改 obs_mode | 无 profile 数据，上界约 3–5%。obs_mode 被 injection_candidates.RUNTIME 钉死，关 wrist 相机要改 third_party（违反第 24 条）。D3-3 只建议先做 profile。 |
| 迁到 RTX 6000 Ada（D4-7） | 2× 只来自规格表峰值，未实测。aspen A6000 实测不比 A40 快，且与用户「暂时不考虑不同硬件」冲突。 |
| 视频或 step_trace 写出、预取下一局 reset | 视频约 1% 墙钟；预取要在单进程里并行 SAPIEN reset 与严格序号协议，复杂度高。 |

## 对原发现的主要修正

- D1-1：采用窗约 4183（21%），不是约 5500；改动约 60–90 行，不是 40–60 行。
- D1-4（vrec 离线批量编码）：在 D1-1 之上的边际只有约 -20~30%，且与 chunk 加大不叠加。
- D2-1：samep 不能原样走 run_oracle_chunks；实测 88 个 spec、269 块，不是 207 块。
- D4-1：GPU 平均利用率约 53–86%，不是「93% 饱和」。第 4 通道无收益的结论不变。
- D4-6、D4-2：「27.4k 窗口」无法复现。装箱前提下加卡的收益要下调。
- D6-3：真正的损失是静态队列把 vrec 尾部拉长，不是供体生成等最慢通道。

## 相关路径

- 代码锚点：
  - `/data/hongzefu/robomme_policy_learning_MotionJEPA/scripts/interpret/serve_policy_attn.py`（MotionContentClient）
  - `/data/hongzefu/robomme_policy_learning_MotionJEPA/scripts/interpret/sp_swap_prep.py`（assign_samep、distribute、cmd_requeue、CHUNK）
  - `/data/hongzefu/robomme_policy_learning_MotionJEPA/scripts/interpret/run_local.py`（run_chunk、run_oracle_chunks）
  - `/data/hongzefu/robomme_policy_learning_MotionJEPA/scripts/interpret/samep_rs_variants.py`（reset_demo）
  - `/data/hongzefu/robomme_policy_learning_MotionJEPA/scripts/dataset/wan/wan_motion_infer.py`（pin_numerics、encode_chunk）
- 留档：`/data/hongzefu/robomme_policy_learning_MotionJEPA/docs/evaluation-doc/steer-4task-1115/launch.md`

本报告为只读审查，未改任何文件、未运行任何脚本。
