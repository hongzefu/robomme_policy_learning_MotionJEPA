# 官方 RoboMME 16×100：GPU 0–3单条32帧4×4＋stage2 motion训练计划

> **权威与授权**：当前只修订计划，未开始代码实施、下载、建库、删除或训练。本文以用户指定的 [1026阶段二memory训练计划](1026-vla-stage2-memory-1024-train-plan.md) 为结构与详略参照：第一部分说明本次运行、机制、验收、顺序及裁决；第二部分保留文件、接口、分工、运行步骤和交接细节。沿用本文件原路径，文件名中的旧实验安排不代表当前范围。
>
> **代码锚点**：policy `v4-256concat1@eb11df05994338053492cb0fe9ecf3982993d99b`，主目录 `/scratch/hongze/robomme_policy_learning_MotionJEPA`。本轮文档在现有分支修订；未来实施分支及任务登记待用户裁决。已合入的在线首包D+1、后续最多16帧合同继续保留。MotionJEPA抽取CLI位于 `/scratch/hongze/worktrees/vla-s2-memexport-20261026`，分支 `v12-3decoder-memexport@b5d23a5`，开工复核。
>
> **外部依赖锚点**：EMA checkpoint为 `/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`，SHA256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`；取 `model` 键。源H5位于 `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5/`，公开源 `Yinpei/robomme_data_h5@a5e4e25ffe8af34f64944f9533d06455ce5f8337`。Wan恢复源为 `HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1@384d5db410be713a49357bf3f55ca4f4d2f559a0`。
>
> **已确认要求（逐字）**：①「修改这个计划 改为只用gpu0123 只跑512+motion 且是 32frame4*4」；②「Wan 的抽取不是已经有缓存了吗」「你检查一下」；③「不考虑精度的额外不一致 只要输入是一样的就不考虑 之前有一个根目录md已经讨论过这个问题了」。本文后续“精度裁决”均指③；数据同源约束继续有效。
>
> **呈现要求**：按用户指定的1026计划保留完整两部分及必要细节，只说明本次工作；需用户裁决的原则事项单列，推荐值不代表批准。
>
> **提交与代理**：文档用 `docs:`，未来功能号按实际 `git log` 顺延；Beta与结论独立提交，子代理提交经 `--no-ff` 保留。AWS Codex各角色为 `gpt-6-astra`＋`ultra`，修改代理独立工作目录，审查与运行分别委派。所有新增入口、配置与判定行是拟实现，不能视作已运行。

# 第一部分（给人看）



## 一、要做什么与全部运行一览

**一句话**：用官方16任务×100集准备source、4×4帧表及本库norm_stats，准备同源Wan latent和stage2 motion库，然后仅在GPU 0、1、2、3训练一条 `32帧×4×4＋motion` 的modulation VLA，步数待用户裁决，以下实施模板按80k推荐值展开，选定后统一回填。Wan缓存复用机制见第二节。

| 已定口径 | 本次值 | 依据 |
|---|---|---|
| 实验数量／GPU | 单条；全部阶段仅物理GPU 0–3，CPU步骤显式隐藏GPU | 用户① |
| 视觉记忆 | `budget=512, token_per_image=16, num_views=1`；512÷16=32帧 | 用户①；现有 `perceptual-framesamp-modul.yaml` |
| motion | ep105 EMA冻结；33帧窗口、stride32、每窗1×1024、budget96 | 原计划机制 |
| 融合 | modulation；视觉512＋motion96＝最多608个记忆位置，投影维度1024 | S4形制检查 |
| 数据／优化 | 官方1600集；global batch128、FSDP4、workers8；步数推荐80k，待裁决 | 本次数据与资源口径 |

```text
视觉：32帧 × 16块 = 512个视觉token ─────┐
同源33帧窗口 → 冻结ep105 → 1×1024/窗 ──┴→ 最多608位置 → modulation → 动作策略
```

**512是视觉token数量，1024是每个motion token的维度；视觉历史32帧与motion的33帧窗口是两条独立配置。** 4×4帧表、线上池化、位置表与恢复输入必须一致。

| 运行 | run／库名 | GPU | 配置／产物 | 预计时长 | 公开bucket |
|---|---|---|---|---|---|
| 建库 | `16task-orig100-1600ep-<pin8>` | SigLIP／Wan只用0–3 | source、framesamp 4×4、norm、Wan／motion | 按恢复带宽与补窗短测估算 | `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261027-v1` |
| 唯一训练 | `orig100-m32x4x4-modul-motion-stage2-1024-b128-80k` | 0–3 | `mme_vla_suite_b128_80k`＋新 `perceptual-framesamp-modul-32frame-4x4-motion-stage2-orig16.yaml` | 未实测，G7后更新 | `HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-stage2-1024-80k-v1` |
| MJ权重 | 已有阶段II＋III备份 | 无 | 只读复用ep105 | 不重传 | `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1` |

表中80k运行名和bucket为推荐步数对应的命名；若用户选择60k／40k，S4/S5同步替换步数、名称、保存集合和最终交付步。tmux前缀为 `o16-`，脚本与日志放在 `v1-store/logs/o16-20261027/`。

## 二、数据与memory机制

### 2.1 官方数据及同源约束

本次使用16任务×100集，共1600集；每任务easy50／medium25／hard25。索引记录总帧768,897，demo帧292,040，exec训练样本476,857；新库先复核这些身份与计数，再建特征与本库norm_stats。

**encoder与VLA必须使用同一组源H5，逐文件size／SHA256一致。** encoder训练源由checkpoint旁config指向的数据根 `control/build_spec.json::identity.source_pin.files[]` 取得，与官方 `dataset/source_manifest.json` 和新VLA库 `meta/input_manifest.json` 逐文件交叉核对。不同manifest的整体pin算法可以不同，不能直接比较两个pin字符串替代文件集合核对。

检查在准备、建库、训练起跑三处执行，判定为 `DATA_LINEAGE=PASS files=16 mismatches=0`；缺文件、错批次、不同输入或缺证据均停止。精度裁决不豁免数据来源。

新库根为 `v1-store/datasets/16task-orig100-1600ep-<pin8>`，其中pin8由官方source_manifest实读；视觉帧表为 `framesamp/`，layout `framesamp-4x4-v1`。norm_stats由新库自算，随库和checkpoint记录。

### 2.2 视觉与motion进入策略的方式

视觉历史在当前可见帧中采样最多32帧，每帧4×4＝16个token，共512个视觉位置。视觉池化、位置编码、采样顺序与线上输入必须使用同一配置；`streaming_obs_horizon=16` 仍是每次增量喂包上限，不是历史帧数。

```text
Wan raw latent 每窗 [9,16,32,32] float32
  → 冻结的ep105 EMA encoder，eval/no_grad，geometry_valid全True
  → [4,256]，按源时间片[0,3,5,8]顺序拼接
  → [1,1024] motion token（每窗4096 B）
  → motion_emb[96,1024]、motion_pos[96,256]、motion_mask[96]
  → 位置投影与motion特征拼接，经1792→1024映射
  → 与512个视觉位置合并、按时刻排序
  → 最多608个memory位置，通过modulation调制动作策略
```

encoder保持冻结，训练的是VLA及其memory接入参数。memory不做图像增强，encoder使用EMA的 `model` 键；不能把live权重、增强输入或错误窗口当作精度差异。

### 2.3 窗口、demo补帧与预算

motion窗口含33帧，起点步长32，以各episode的各segment起点重新计数。demo末窗真实帧≥17则重复末帧补满33；不足17丢弃。exec只接受完整33帧，保持当前训练manifest到原始T的范围。

```text
demo长度100：合法grid32起点0、32、64；末端96只剩4帧，不构成窗
demo长度82 ：起点64剩18帧，重复第81帧15次，补成33帧
exec长度100：起点0、32、64均为完整33帧，不补帧
```

motion容量96。k个真实窗后补零、mask=False，按时刻排序后补位落尾；补位不贡献注意力，固定容量仍参与张量计算。正式起跑前按新manifest复算训练与1300步评估的最大可见窗数，超过96即停，不静默截断。

### 2.4 Wan缓存恢复与仅补缺窗

**Wan确实已经抽取过。本次优先恢复官方原图缓存，按需要的窗口复制／重排，只补缺窗；不默认全量重新编码。** 当前精度裁决已经裁定：输入一样时，不以精度或TF32差异阻塞，不强制改变现有线上设置。相关讨论见 [1026计划](1026-vla-stage2-memory-1024-train-plan.md) 与 [MotionJEPA Wan结论](../MotionJEPA/1024-wan-extraction-conclusion.md)；本次执行按当前精度裁决。

本次只消费官方未增强前视RGB缓存：远端固定版本公开，API列出2485个bin，历史658350窗、388310630400 B；本机正式payload已不在，保留着index/provenance。缓存窗口为33真实帧、stride1，不含demo补尾，exec止于completed之前。恢复后按发布SHA256SUMS核完整文件，再转换为本次需要的窗口表。



实查只读索引、元数据和文件stat；远端仅查询固定revision元信息，未下载bin，也未全量重算SHA。恢复后仍须按发布SHA256SUMS核验实体。

**窗口覆盖口径**：
- 最终motion消费grid32，以segment起点为原点、demo真实帧≥17时重复末帧补33、exec完整33帧且训练按原始T，不擅自截在completed。
- 按现有官方索引计算：可复用21814个完整grid32窗；demo补尾386窗、completed之后的exec缺1053窗，共1439个待补窗。这个数字是索引推算，正式以新manifest与逐窗对照清单为准。
- 为兼容现有MJ抽取／pack，物理Wan表仍按grid16，motion下游按 `--grid-subsample 2`／`--latent-grid-subsample 2` 在每个segment内从0取偶数格。索引实算：grid16复用42404窗，补demo尾窗900＋exec结束后1924＝2824窗，总45228窗／26676559872 B（约24.84 GiB）；下游grid32共23253窗。1439是消费子集缺口，物理补算预算必须用2824；正式manifest仍复核，不填伪造奇数格。
- 完整缓存恢复峰值约388 GB；若选段／范围恢复，必须保留对应发布文件身份和可信校验，不以未核验HTTP片段宣称发布SHA已过。默认采用固定revision整文件恢复，再CPU选行，成功后临时恢复文件的删除另按批准清单办理。

```text
固定版本官方原图Wan缓存 ─→ 命中窗口复制／重排 ─┐
        同一官方H5的确切缺失窗口 → 仅补算缺窗 ────────┴→ 完整grid16 → subsample2 → motion
```

**同输入如何证明**：每窗身份包含源H5的size／SHA、task／episode／segment、33帧的全局帧号及顺序、前视RGB、无增强、256×256、demo补帧来源。对线上／离线相同窗重算最终连续uint8像素SHA；若原缓存无逐窗像素SHA，使用已发布源pin＋索引＋构建代码绑定旧窗口，并从对应H5重算写入新来源清单，不靠latent反推原图。输入或来源不一致即FAIL；原权重、raw latent轴序 `[9,16,32,32]`、无额外affine／无posterior采样、有限值同样硬核。

**精度差异如何处理**：缓存历史FP32＋卷积TF32开；缺窗可沿现有policy FP32＋TF32关，线上也保留当前设置。逐窗记录实际计算配置与copy／fill来源，跨精度数值只输出 `NUMERIC_EQUIVALENCE=NOT_REQUIRED reason=user_decision_same_input` 或INFO，不伪写bitwise PASS。同一latent字节的复制／打包／回读，以及同实现同输入的encoder确定性仍按原要求核验。

**motion机制保持**：ep105 EMA冻结，`eval()/no_grad()`、memory不增强，33帧窗／stride32，每窗按源时间片[0,3,5,8]把4×256拼成1×1024，budget96。补帧规则与执行段范围保持上述口径。

已有实现尚不能直接消费这个混合来源：S2增加 `published-cache-with-local-fill` 来源分支；S3负责缓存恢复清单、输入窗口映射及仅缺窗编码。旧mixed8收据与阈值路径保持原样。

## 三、训练配置、资源与存储

唯一入口为拟新增 `scripts/training/run_orig16_stage2_1024_80k.sh`，按80k推荐模板沿用 `mme_vla_suite_b128_80k`，从 `pi05_base` 预训练权重初始化，action_horizon=20、EMA=0.999、每5000步保存。在入口覆盖 `--fsdp-devices 4 --num-workers 8`，global batch仍为128。数据路径必须是 `$LIB/framesamp`（layout `framesamp-4x4-v1`），assets显式指向本库自算norm_stats，避免落到4task默认值。用户选定其他步数后统一回填入口覆盖与命名。

新yaml从现有 `perceptual-framesamp-modul.yaml` 的帧路配置派生：`budget=512`、`token_per_image=16`、`num_views=1`、`streaming_obs_horizon=16`；再接入stage2 motion节，保持 `[1,1024]`、budget96、stride32、window33与ep105身份，换同源store路径和验证时的 `online_gpu=1`。**streaming_obs_horizon=16是增量喂包上限，32帧历史由512÷16决定。**

现有模型、loader和线上FrameSampMemory动态支持4×4池化／32帧采样。需适配新启动契约、G4便携输入的帧表路径和原版H5，以及首包D+1与旧验证脚本的衔接；第二部分S4列出文件与负例，不放宽通用模型guard。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_NAME=orig100-m32x4x4-modul-motion-stage2-1024-b128-80k TRAIN_HEAD=<训练Beta的40位sha> FINAL_RECORD_DIR=<验证输出目录> bash scripts/training/run_orig16_stage2_1024_80k.sh
```

启动契约硬核GPU列表 `0,1,2,3`、FSDP4、workers8、4×4布局、视觉512／32帧、motion96与数据同源。无motion入口、第二组GPU和双run配对脚本均不属于本计划。

**速度**：32帧4×4＋stage2四卡配置尚未实测。G7记录warmup、稳定步时与峰值显存，再估 `80,000×稳态秒/步÷3600` 小时；显存上限仍为72 GiB。

**epoch口径**：原版exec样本476,857，128×80k＝10.24M样本，约21.5 epoch；60k约16.1、40k约10.7。步数由用户在第七节裁决。以下技术模板暂按80k推荐值填写，定稿时统一替换启动覆盖、run/bucket名、保存集合与终点，不改全局默认。

### 存储与备份安排

当前磁盘可用约1.3 TiB。新库的source约190 GB、4×4帧表约50 GB、最终Wan表约26.7 GB；另有SigLIP临时features约465 GB、官方Wan恢复文件约388 GB。按阶段错开features与恢复缓存，避免同时达到峰值；一条80k训练的检查点约190 GB，最终导出／回读另留同量级空间。以上是预算，不替代开工前实测。

旧70k的14份检查点全量上传已完成，日志为 `v1-store/logs/vla-s2-20261026/close70-export.log`，有 `CKPT_UPLOAD=PASS steps=14` 与 `EXIT_CODE=0`。原先等待这项上传的前置已经满足；是否删除本地副本仍由第七节第2项裁决。

| 对象 | 约占用 | 本次建议，等待裁决 |
|---|---:|---|
| 旧70k导出／回读副本 | 156 GB | 已核备份后清理重复副本 |
| 旧mixed8 Wan恢复副本 | 71 GB | 核对应训练库备份后清理 |
| 旧70k训练检查点本体 | 155 GB | 保留 |
| 本轮原始H5、encoder训练缓存、模型权重 | 按现有实体 | 保留只读，不作为清理对象 |

新训练库与VLA检查点上传公开Hugging Face存储，增量上传与训练并行、降低IO优先级；最终完整性以匿名回读SHA和实际恢复为准，不以“上传进程退出”代替。

## 四、本次需要完成的改动

| 工作块 | 为什么要做 | 主要落点 |
|---|---|---|
| 原版数据身份与读帧 | 建库、缓存和VLA必须绑定相同16份H5，demo／exec需正确映射 | G0与 `check_data_lineage.py` |
| Wan缓存恢复与补窗 | 把已发布完整窗映射为本次grid16，只计算真实缺口 | 新 `orig16_wan_cache.py` |
| 缓存来源与精度政策 | pack、loader、握手、快照和恢复必须接受同一份来源政策 | motion_store、motion_client、pack、train及验证入口 |
| 32帧4×4四卡入口 | 新配置必须消费4×4库，训练、线上和恢复使用相同记忆形制 | 新yaml／启动脚本／契约，G4与checkpoint验证 |
| 训练库及检查点上传 | 正确打包4×4表、motion、来源证据与裁决步数的checkpoint | 两套新导出驱动及layout分支 |
| MJ抽取适配 | 用本次实际段数／行数及源pin，不绑定其他数据集计数 | stage2_memory_export CLI |

具体文件、接口和命令只在第二部分展开。实施范围围绕这些工作块，核心模型与已有数据集路径保持各自明确合同。

## 五、验收一览

| 编号 | 判定行 | 查什么／通过说明什么 |
|---|---|---|
| B0 | `DF_CHECK=PASS stage=<s> avail_bytes=<n> need_bytes=<n>` | 按Wan路线核磁盘，须满足第二部分B0阈值 |
| C0 | `WAN_CACHE_AUDIT=PASS candidates=<n> route=published-cache-with-local-fill` | 核实体、输入身份和覆盖，实际精度分别记录；导入完成另需B6的 `WAN_CACHE_IMPORT=PASS` |
| L0／L1／L2 | `DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=<sha> vla_input_pin=<sha> files=16 mismatches=0` | 开工、建库、单run起跑逐文件同源，FAIL不得WAIVE |
| B1 | `INPUT_PIN=PASS files=16`、`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 …` | 官方库身份一致 |
| B2 | `STAGE_DONE stage=siglip workers=16 items=1600 elapsed=<s>` | GPU0–3每卡4个worker，全覆盖 |
| B3／B4 | `全部检查通过 PASS`、`VERIFY_PACK=PASS scanned=768897 mismatches=0` | finalize复算1024帧；4×4全表验证 |
| B5 | `FRAME_LAYOUT=PASS frames=32 tokens_per_frame=16 budget=512`、`STAGE2_FRAME4_COMPAT=PASS total_tokens=608` | 4×4字节、采样、mask、时间PE与motion排序一致 |
| B6／B7 | `WAN_CACHE_IMPORT=PASS …`、`WAN_INPUT_COVERAGE=PASS …`、`WAN_NUMERICS=NOT_REQUIRED …` | 复用／补窗完整，源与输入一致；数值差异按当前精度裁决记录 |
| B8／B9 | `FEATURES_PRUNE=PASS kept=8`、`DATA_IDENTITY=PASS norm_sha=<sha> steps=768897 samples=476857 manifest_sha=<sha>` | 授权features清理；库与norm绑定 |
| G0 | `RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=<≥16> sha_mismatches=0` | 原版H5原帧与输入身份一致 |
| G1 | `MJ_CLI_LINEAGE=PASS files=16 mismatches=0`、`DROPOUT_OFF=PASS …` | 抽取CLI独立核源，encoder冻结 |
| G2 | `G2_INPUT_IDENTITY=PASS windows=64 mismatches=0`、`G2_NUMERICS=NOT_REQUIRED reason=user_decision_same_input` | 输入身份、形制与finite通过，精度差异不阻塞 |
| G3 | `EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1`、`ENCODER_READOUT=PASS …` | 权重、读出与geometry正确；跨路径输出数值NOT_REQUIRED |
| G4 | `ONLINE_INPUT_IDENTITY=PASS windows=64 handshake=1 input_sha_mismatch=0 nonfinite=0 …` | 4×4输入、首包D+1、帧序与真实IPC成立；数值差异INFO |
| G5 | `PACKED_TOKEN_BYTES=PASS … mismatches=0`、`VERIFY_MOTION=PASS …`、`ENCODER_RECOMPUTE=NOT_REQUIRED …` | 已生成token到packed的全量字节／SHA／finite核验；不重复跑GPU数值对拍 |
| G6 | `TRAIN_ES_BOUND=PASS episodes=1600 tasks=16 max_windows=<n> budget=96 headroom=<≥0>` | 训练及1300步评估零截断 |
| G7 | `GPU_SCOPE=PASS allowed=0,1,2,3`、`LAUNCH_CONTRACT=PASS …`、`SMOKE_4GPU=PASS run=s2 …`、`SAVE_LOAD=PASS …` | 单条20步短测与真实恢复 |
| G8／G9 | 唯一日志 `EXIT_CODE=0`；`CKPT_UPLOAD=PASS steps=16 …` | 80k与16份checkpoint；匿名回读SHA一致 |
| G10／G11 | G10引用既有 `MJ_CKPT_UPLOAD=PASS source_runs=2 …`；G11不用 | MJ不重传 |
| G12 | `CKPT_DELIVERY_READY=PASS step=79999 …` | 参数／EMA严格恢复与真实在线推理有限；数值差异INFO |
| B10／S0 | `RESULT=PASS …`；`SOURCE_RO=PASS files=16 size_mtime_unchanged=16` | 新库备份；原始H5未变 |

## 六、步骤与时间线（T0为开工令，时长以实测更新）

| 阶段 | 内容 | 资源／耗时口径 | 判据 |
|---|---|---|---|
| 0 缓存与磁盘 | 按已定缓存路线核清单、旧备份、源pin与L0；获批且备份PASS才清理旧大产物 | CPU／网络只读；不启动未列出的重抽 | C0、L0、B0 |
| 1 适配与审查 | S1/S2/S3/S6第一批，S4/S5第二批；CPU manifest/hash可并行 | 按缓存导入／补窗方案核工作量 | 审查、B1、L0/L1 |
| 2 训练库 | 建库Beta冻结；SigLIP16worker→finalize→4×4／norm；Wan恢复原图缓存并只补缺窗 | GPU仅0–3；耗时待恢复带宽与补窗短测 | B2–B9 |
| 3 memory | G0→ep105抽取→G2/G3→pack/compare/verify→G4/G6→回填契约 | GPU仅0或允许的0–3集合 | G0–G6、L1 |
| 4 短测 | 单条20步后真实保存恢复 | 训练0–3；恢复父0／sidecar1 | B5、G7、L2 |
| 5 正式 | 训练Beta、代码只读；唯一run、增量与训练库长尾上传 | GPU0–3、80k；估时=稳态秒/步×80k | G8、增量完成行 |
| 6 收尾 | 单条导出回读、79999恢复、解锁、档案、handoff | CPU／网络，G12用0与1 | G9/G12、S0、权限恢复 |

总耗时由代码适配、缓存恢复／补窗、短测后的实际步时和最终上传组成；本配置尚未实测，不预先承诺完成时刻。

## 七、请用户裁决

以下三项尚未收到选择；推荐项不代表批准。已确认的GPU0–3、单条32帧4×4＋motion、官方同源数据和同输入精度政策，不再重复询问。

| 编号 | 待决定事项 | 可选项 | 推荐及影响 |
|---|---|---|---|
| 1 | 训练步数 | 80k／60k／40k | 推荐沿用80k，约21.5 epoch；60k约16.1、40k约10.7。全局batch128保持 |
| 2 | 旧产物清理 | 只清重复导出与恢复副本，约227 GB；或连70k检查点本体也清理，共约382 GB；或全部保留 | 推荐清前两份重复副本、保留70k检查点。任何删除前核备份；选择全部保留则先重核分阶段空间预算，不擅自扩大删除范围 |
| 3 | 分支与任务登记 | 使用独立 `v5-orig16` 并登记；或沿用现有分支并登记；或暂不建分支、不登记 | 推荐独立分支并登记，便于后续会话识别状态；首次推送跟踪包含在此选择中 |

这三项用于定稿，不自动触发执行。开工需要用户另行明确指令。选择步数后同步回填run／bucket名称、配置覆盖、保存集合、最终交付步和下游恢复合同，不能只改表中数字。

### 执行中需要停下的情况

- 任一拟删除对象的备份核验不满足：不删该对象，报告用户；已完成的70k备份无需重复等待。
- 任一 `DF_CHECK` 低于阈值：停，提请第二档。
- `DATA_LINEAGE` 任一次FAIL或缺行（L0／L1／L2）立即停工，不得WAIVE；32帧4×4形制或GPU0–3检查不符同样不得起跑。
- 来源／输入／权重／结构／完整性等必需B/G闸门FAIL即停；当前精度裁决已裁定的跨精度数值差异记NOT_REQUIRED，不能伪装成必需闸门失败，也不调用旧例外脚本。
- 合前审两轮仍 FAIL、或合后审 FAIL：写 `AGENTMETARULES_HOLD`，交用户。
- 训练崩溃、NaN、OOM：停下问用户，不 resume。
- 缓存下载超出C0批准文件清单／预算，或出现未列入的大下载：先问；不得静默退回全量重抽。
- push 被拒：停，原文交用户。

## 八、子代理分工与合并（简述）

未来实施仍拆六块：S1原版H5与单run血缘；S2缓存来源与输入等价合同；S3恢复、复制及仅补缺窗；S4负责32帧4×4入口、契约、在线与恢复；S5训练库和唯一run两套上传；S6负责MJ计数与源pin。Wan复用和同输入精度口径已经当前精度裁决确认；本轮仍只改计划，运行等待开工令。

第一批S1/S2/S3/S6，第二批S4/S5依赖S1/S2接口；S3待S2合入后续跑兼容测试。顺序S6→S1→S2→S3→S4→S5。写入型独立worktree、共享文件单一owner；合前固定BASE/TIP审查，合后验证再推送。主会话负责资源、删除与冻结。

AWS Codex全部角色按第26条显式使用 `gpt-6-astra`＋`ultra`，运行与修改分工；其他宿主按其现行角色规则。

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

- **R0 R-LINEAGE**：encoder 训练源与 VLA 库必须是同一组源 H5、逐文件 sha256 相同。`DATA_LINEAGE` 在L0／L1／L2三处机械核，任何 FAIL 即 `exit≠0`、立即停工，不得 WAIVE、不得写成风险项继续。规则来源：`AGENTS.md` P-LINEAGE（`d0445ce`）；`CLAUDE.md` 项目专属补充「P-LINEAGE 的 Claude Code 落地」——主会话派发 VLA 建库／起跑子代理前，先贴 `DATA_LINEAGE` 判定行；审查代理对涉及起跑脚本／契约的改动，必须核到该闸门存在且 FAIL 即停。
- **R1 源目录只读**：`/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/` 整体不写不动（MJ 1024 计划 R6）。policy 侧只以 `--raw_dir`/`--raw-dir`/`--h5-root` 只读引用 `source/h5/`，以 `--mj-index` 只读引用 `dataset/index/`；不复制、不建 symlink 进 `v1-store/`。收尾以 build_spec 记录的 `bytes`/`mtime_ns` 核 S0。
- **R2 不改 encoder 权重**：只读 `checkpoint_epoch_105.pt` 的 `model` 键，strict、`eval()`、`no_grad()`；现有encoder和在线数值设置不因缓存差异强制改变，实际环境分别记录。
- **R3 memory 不增强**：`geometry_valid` 全 True，等价 aug_id=0。
- **R4 旧路径不变**：
  - mixed8 的入口、契约、yaml、导出驱动、runner 目录零 diff。
  - `motion_store`／`motion_client`／`pack_motion_store`／`train.py`／`compare_stage2`／`check_stage2_validation_inputs` 中的 mixed8 restore-receipt 路径与旧 layout，**语义不变、旧测试全部仍绿**；新增分支不改变既有参数缺省行为。
- **R5 实施分支**：按第一部分第七节第3项裁决落地；推荐policy在 `v5-orig16` 及隔离子分支实施。MotionJEPA只在 `v12-3decoder-memexport` 新增提交。本轮文档修订留现有主分支，不构成代码实施授权。
- **R6 建库冻结窗口**：建库 Beta 后，从 SigLIP 起跑到 `VERIFY_MOTION=PASS`，policy 不 commit、不合并、不回流规则。原因是 SigLIP 各 shard 的 fingerprint（`finalize_checks check` 核其唯一）与 `pack_motion_store.py` 都要求 `git_commit` 唯一。窗口内主会话写 `$(git rev-parse --git-dir)/AGENTMETARULES_HOLD`（`freeze` + 锚点 sha），窗口结束删除。`scan_manifest` 与 `hash-inputs` 不写 git HEAD，可在冻结前跑。
- **R7 训练期代码冻结**：从训练 Beta 起，到唯一run的G8与最终导出完成，`src/`、`pyproject.toml`、`uv.lock` 不动。被跟踪的 `src/ scripts/ packages/` 文件 `chmod a-w`，原权限记到 `v1-store/o16-scripts/chmod-before.txt`；结束按记录恢复并打 `PERM_RESTORED=OK`。
- **R8 tmux**：只用前缀 `o16-`，完整名单见四节。清理只按清单逐个 `tmux kill-session -t '=名'`，删前删后各 `tmux ls`；禁止 `kill-server` 等全局杀法。
- **R9 不 resume**：训练崩溃、NaN、OOM 一律停下问用户。不带 `--force`、不 `overwrite`；新 run 目录、新库目录、新 bucket 起跑前确认不存在。
- **R10 W&B 与凭据**：W&B 用 hongzefu 账号，entity `hongzefu-university-of-michigan`，project `robomme-framesamp`。凭据 `v1-store/secrets/wandb.env`、`v1-store/secrets/hf.env` 只经 `set -a; . file; set +a` 读，不进日志、档案或命令行。
- **R11 存储边界**：一切持久化落 `/scratch/hongze/` 下。缓存 `UV_CACHE_DIR=/scratch/hongze/.cache/uv`；HF 缓存沿用 `paths.sh` 的 `$V1_STORE/cache/hf`，env 不覆盖 `HF_HOME`（mixed8 oracle 首跑因覆盖 `HF_HOME` 失败的教训）。
- **R12 删除纪律**：只清理第一部分第七节第2项选中的对象，并在核到对应HF备份判定后执行。删前列精确清单、核目录及软链接关系；未被选中的检查点、库和数据保留，不自行扩大范围。
- **R13 输入与数值分开**：血缘、像素、帧序、补帧、增强、模型身份、shape/finite不能WAIVE。当前精度裁决已取消同输入跨精度数值等价要求，新kind记NOT_REQUIRED，不伪造数值PASS，不用旧例外脚本；旧mixed8合同原样保留。
- **R15 GPU边界**：本计划全部GPU工作只允许物理0、1、2、3。既核 `CUDA_VISIBLE_DEVICES`，也核 `run_local.py --gpus`（其子进程会重写环境）、分片索引与sidecar GPU；GPU4–7不属于本任务。CPU检查用 `CUDA_VISIBLE_DEVICES=''` 与 `JAX_PLATFORMS=cpu`。
- **R14 子代理环境**：worktree 内只跑 ≤5 分钟的 CPU 测试。
  - policy：`UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`，先打印 `mme_vla_suite.__file__` 确认指向 worktree。
  - MJ：`UV_PROJECT_ENVIRONMENT=/scratch/hongze/MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`。
  - GPU 验收在合并后由运行代理在主检出跑。

## 一、逐文件改动清单

所有锚点用符号名，以本次核对的 `eb11df0` 为基线；未来实施基线必须包含本次计划，不能回退到尚未合入评估分支的旧代码。

### A. MotionJEPA（S6，分支 `v12-3decoder-memexport`，worktree 内新提交）

| 文件 | 锚点 | 改什么 | 验收 |
|---|---|---|---|
| `scripts/stage2_memory_export.py` | 常量 `FULL_COUNTS = (4995, 65419)`；`main` 中 `require_equal(counts, FULL_COUNTS, "mixed8 全表段数与行数")` | 新增 `--expected-counts <segments>,<rows>`，full scope 必填；不传时报错，不回落到 mixed8 常量。mixed8 以 `4995,65419` 调用时语义不变 | `tests/test_stage2_memory_export.py` 新增三个用例：错计数拒绝、正确计数通过、smoke scope 不受影响；冒烟出 `STAGE2_EXPORT=PASS scope=smoke …` |
| `scripts/stage2_memory_export.py` | `main` 读入 latent 之前 | 新增**必填**参数 `--input-manifest <policy 库 meta/input_manifest.json>`，做输入 latent 库的 pin 校验（对应 MJ `AGENTS.md` P3）。读 `--ckpt` 同目录 `config.yaml` 的 `data.data_root` → `<data_root>/control/build_spec.json` 的 `identity.source_pin.files[]`；与 input_manifest 的 `files{}` 按文件名逐个比 `size+sha256`。集合不等即 FAIL、退出非 0，smoke 与 full 都执行，**不提供任何跳过开关**。判定行 `MJ_CLI_LINEAGE=PASS files=16 mismatches=0`（FAIL 时列出差异） | 同一测试文件新增：一致夹具 PASS；改一个 sha、少一个文件、缺 `--input-manifest` 均失败；argparse 中不存在任何跳过血缘的参数 |

### B. policy

**S1 `scripts/dataset/check_stage2_validation_inputs.py`（G0 原版模式）**
- 新增 `--h5-root`、`--input-manifest`、`--mj-index`、`--cache-import-receipt`；原版缓存模式四项齐全，部分给出即报错；全部缺省才沿旧mixed8路径。G0绑定新收据SHA供后续新kind识别。
- **demo 帧**：原版模式不读 `control/scope.json`／`run_manifest.json`，改为 `h5py.File(<h5-root>/<ep.h5_file>, "r")[f"episode_{raw_ep_idx}/timestep_{t}/obs/front_rgb"]`。H5 的 size 与 sha256 须等于 input_manifest 登记值（sha 现场全量计算并在进程内缓存）。
- **exec 帧**：仍读 `source/data/<sample>.pkl`。
- **MJ 对齐**：从 `<mj-index>/episodes.parquet` 取每集 demo 末端 `D`（列 `demo_end`）与 exec 截止 `C`（列 `completed_index`，即首个 `is_completed` 帧）。`segments.parquet` 只有段 `lo/hi`（exec 段的 `hi` 即 C），仅用于交叉核对。
  - 逐集断言 policy `exec_start_idx == D`；
  - `select_windows`按完整policy范围到T选exec；`start+33≤C`为可复用候选，超过C且不超过T为fill，不缩短训练范围；
  - demo 全真实窗本身就在 `[0,D)` 内；
  - copy窗写 `mj_window: {task, episode, segment, start}`，fill写null并绑定receipt中的对应记录；额外写copy/demo-fill/exec-fill分类。
  - 样本64窗、16任务、≥16个demo尾窗；copy/demo-fill/exec-fill三类各至少1窗，grid16模块常量不变。
- **测试**：
  - 新增 `tests/dataset/test_check_stage2_validation_h5.py`：合成 2 任务 H5 + `episodes.parquet`（`demo_end`/`completed_index`）与 `segments.parquet`（`lo/hi`）夹具；验证补帧窗SHA、H5 sha不符拒绝、D不等拒绝、copy止于C而fill可至T、三类样本覆盖以及缺收据拒绝。
  - 既有 `tests/dataset/test_stage2_validation_inputs.py` 原样通过。
- 判定行沿用：`RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=<n> sha_mismatches=0`。

**S1 `scripts/training/check_data_lineage.py`（新）**
- 可导入函数 `check_lineage(encoder_ckpt, expected_ckpt_sha256, vla_input_manifest, mj_source_manifest) -> (ok: bool, line: str, detail: dict)`，供S4契约与唯一训练入口调用。
- 子命令 `lineage --encoder-ckpt <pt> --expected-ckpt-sha256 <64hex> --vla-input-manifest <json> [--mj-source-manifest /scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json]`，步骤如下：
  1. 现场计算 ckpt sha256，与期望值比较。
  2. 读 ckpt 同目录 `config.yaml` 的 `data.data_root`（ep105 实为 `/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1`）。
  3. 读 `<data_root>/control/build_spec.json` 的 `identity.source_pin.files[]`（每项 `task/path/bytes/sha256/mtime_ns`），得到 encoder 源集合 `{basename(path): (bytes, sha256)}`。`identity.source_pin.sha256` 是 source_receipts 的哈希，不参与比较。
  4. 读 MJ `source_manifest.json` 的 `files.<Task>.h5.{file,sha256,size}`，要求与第 3 步集合完全相同；取其 `pin_sha256` 作 `encoder_source_pin`（只记录）。链条为：`stage2_pool4.yaml` 的 data_root → 1024 计划 R6 源目录 → source_manifest。
  5. 读 VLA `input_manifest.json` 的 `files{<name>:{size,sha256}}`，按文件名对齐，集合必须相等。多、少、sha 不同、size 不同都计入 mismatch。
  6. `vla_input_pin` = 按文件名排序的 `[name, size, sha256]` 列表 canonical JSON 的 sha256，只记录。**不与 `encoder_source_pin` 比值，同源只以第 4、5 步的集合相等判定。**
  - 输出：通过时打 `DATA_LINEAGE=PASS encoder_ckpt=<sha8> encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0`、退出码 0；否则打 `DATA_LINEAGE=FAIL … mismatches=<k>`，逐项列出差异，退出码 3。
- **测试** `tests/training/test_check_data_lineage.py`（新）：三方清单一致PASS；改sha、缺／多文件、size不同、误放顶层source_pin均FAIL。仅实现 `lineage`，不新增双run配对子命令。

**S2 官方缓存＋本地补窗来源合同**

只增加 `latent_origin.kind="published-cache-with-local-fill"` 分支；无kind或historical-restore-receipt的旧mixed8合同保持原行为。新分支的数值差异政策引用当前精度裁决；没有跳过源H5、输入、权重或完整性校验的开关。

| 文件／符号 | 新分支改动 |
|---|---|
| `scripts/dataset/pack_motion_store.py::cmd_pack/load_vae_compat/gather_stage2_provenance/check_vae_group/check_raw_sources` | 新增 `--latents-origin published-cache-with-local-fill --cache-import-receipt <json>`，与旧restore-receipt互斥；接受发布copy与本地fill，分别记录数值环境，硬核同一原VAE权重、mode/raw latent语义、源pin与全窗口覆盖；新kind以导入收据及输入证明承接oracle要求，旧分支保持原行为 |
| `src/mme_vla_suite/datastore/motion_store.py::validate_stage2_provenance/validate_stage2_g2_evidence/_stage2_validation` | 按kind选新证据集 `G0/G2/G3/wan_cache_import/SOURCE_PIN`；G2只把源与输入、shape、finite作为准入，不套旧MSE/max阈值或历史waiver |
| `src/mme_vla_suite/policies/motion_client.py::validate_stage2_evidence/_check_stage2_provenance` | 同步证据分支；原权重／EMA／输入语义／真实worker身份仍严格，发布cache与本地fill／在线的精度、TF32配置分别记实，不因数值配置差异拒绝，不改线上pin_numerics |
| `scripts/training/train.py::write_stage2_motion_snapshot` 与 `check_stage2_checkpoint.py::verify_snapshot` | 保存／恢复同一证据集合及政策；复制封口时核SHA，不把cache环境伪填为encoder运行环境 |
| `scripts/dataset/wan_stage2/validate_stage2.py::gate_g2/gate_g3/reference_g4` | 新增 `--cache-import-receipt` 并核新kind；G2查输入来源；G3保留65叶EMA状态、model/live负控、读出／geometry，跨路径输出相等改NOT_REQUIRED；G4参考保留输入/shape/finite，移除跨数值路径的零差准入 |
| `scripts/training/check_stage2_online.py::reference_subprocess/run` | 新kind保留真实IPC、原uint8窗口SHA、帧号／补帧／状态机、权重、shape/finite；不再要求encoder_max_abs=0或e2e≤1e-5，不伪写数值PARITY=PASS |
| `scripts/dataset/wan_stage2/compare_stage2.py` | 固定65419改 `--expect-rows`；新kind提供完整性认证，GPU重复计算数值对拍不作为闸门，不填未运行的1024窗结果 |
| `pack_motion_store.py::check_bitexact_record/_encoder_validation_summary/cmd_verify`、`motion_store.validate_stage2_encoder_validation` | 区分数值认证与来源／打包完整性；新kind消费完整性证据，仍保留packed表对已生成源token全量memcmp及SHA、finite、补帧检查 |

来源收据绑定：公开repo＋固定revision、SHA256SUMS自身SHA、下载整文件SHA／字节数、源H5清单、目标manifest/input_manifest、importer源码SHA、逐窗copy/fill映射、源row/offset及声明的轴序变换、33帧输入SHA、实际精度配置、最终文件SHA／字节数。copy不改数，fill仅计算缺窗；不得以额外标准化、增强或换VAE来解释精度差异。NOT_REQUIRED只覆盖用户确认的额外数值精度差异，不覆盖输入、计算语义、模型身份或完整性错误。

G2沿用 `schema=stage2-validation-gate-v1, gate=G2`，增加 `validation_scope=input-equivalence`；status PASS仅表示输入身份与结构通过。必填 `windows=64`、`demo_tail>=16`且等于G0、`input_mismatches=0`、`shape_mismatches=0`、`nonfinite=0`、绑定G0的samples_sha256。数值字段为 `numeric_equivalence=not_required_by_user`，可有INFO测量，不能填虚假的worst_max=0。

测试扩展现有stage2 pack／datastore／snapshot／client／restore，以及新增 `tests/dataset/test_pack_motion_stage2_cached.py`。必要负例：同输入仅精度／TF32不同通过身份检查且标NOT_REQUIRED；换帧、换H5、aug1–16、混mixed8、错轴／块、缺映射、篡改bin或NaN均FAIL。旧mixed8阈值与waiver回归保持。共享证据键宜由一个按kind选择的辅助函数复用。

**S3 `scripts/dataset/wan_stage2/orig16_wan_cache.py`（新）**

- `plan --manifest <MANI> --input-manifest <INMANI> --published-index <MJIDX> --out <CACHE_PLAN>`：按policy原有segment范围列全grid16窗，输出copy/fill、33帧号、恢复整文件清单与字节数、最终grid16和grid32计数。参考42404＋2824＝45228，消费23253；实际不符须解释，不改截尾点或补帧规则来凑命中。
- `fetch --plan <CACHE_PLAN> --revision 384d5db410be713a49357bf3f55ca4f4d2f559a0 --out <CACHE_ROOT>`：恢复公开原图库固定版本；先单文件核SHA，再按清单恢复。禁止默换main或把部分HTTP范围当作完整文件SHA已验证。
- `import --plan <CACHE_PLAN> --cache-root <CACHE_ROOT> --raw-dir <RAW> --out <LIB/wan-latents> --receipt <VALID/wan-cache-import.json> --gpus 0,1,2,3`：
  - copy按发布index定位原块，只允许明确可逆的轴序重排；已有缓存命中窗不重编码。
  - fill只调现有 `wan_motion_infer.py::encode_chunk` 计算缺窗，先单窗冒烟；GPU只0–3，也可串行用0。保持本地FP32／TF32关，历史copy数值设置如实记录。
  - 输出既有segment bin／metadata，raw `float32[9,16,32,32]`、589824 B/窗，配完整映射；不得零填奇数格。完成前在同次文件遍历核SHA／finite／无缺失、重复、越界。
- `check --samples <G0> --receipt <VALID/wan-cache-import.json> --raw-dir <RAW> --out <VALID/G2.json>`：独立重算64窗原RGB、帧序和补帧SHA，产出S2合同；输出 `G2_INPUT_IDENTITY=PASS windows=64 mismatches=0`、`G2_NUMERICS=NOT_REQUIRED reason=user_decision_same_input`。不为精度差异重跑全量Wan。
- 原缓存未记录逐窗像素SHA时，发布源pin＋索引＋构建代码必须绑定旧输入，再从对应H5重算并写新记录；不得仅对当前H5自算SHA就当成旧缓存同源证明。
- G3可继续以已有val aug0 latent做同输入的EMA／读出验证，但该旁证不能替代本次官方发布缓存输入证明；训练仍只消费新库记录的原图窗。
- 新 `scripts/dataset/wan_stage2/tests/test_orig16_wan_cache.py` 用小型发布bin/index验证copy字节、offset／帧序、只缺窗触发fill、repeat_last、GPU越界拒绝。新 `tests/dataset/test_orig16_cache_validator_compat.py` 把S3产物交S2实际校验；S2合入后S3从BASE2续跑。


**S4 单条32帧4×4＋motion入口、契约、在线与恢复（第二批，BASE2）**

- **新yaml** `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-4x4-motion-stage2-orig16.yaml`：
  - 帧路逐项等于现有 `perceptual-framesamp-modul.yaml`：budget512、token_per_image16、num_views1、streaming_obs_horizon16、modulation、memory_token_dim1024。
  - 接入stage2 motion原机制：source_shape[1,1024]、budget96、stride32、window33、ep105 EMA及sha `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`。
  - `store_path=v1-store/datasets/16task-orig100-1600ep-<pin8>/motion-stage2-1024`；`online_gpu=1`供恢复使用，训练不读此键。
- **新入口** `scripts/training/run_orig16_stage2_1024_80k.sh`：
  - 复制mixed8入口的控制流程，换HC、PARENT、ASSETS_DIR、run名；`--dataset-path $PARENT/framesamp`，不消费8×8目录。
  - 唯一run为 `orig100-m32x4x4-modul-motion-stage2-1024-b128-80k`；短测前缀 `o16-smoke-s2-`。
  - 固定 `CUDA_VISIBLE_DEVICES=0,1,2,3`，参数恰一次 `--fsdp-devices 4 --num-workers 8`；SOURCE／MANIFEST和norm全部来自新库。
  - `DRY_RUN=1` 打印实际 `TRAIN_CMD=` 并运行完整契约；不建run日志、不启动训练、不建CUDA/JAX缓存，以契约退出码结束。正式入口同一命令先过契约再执行。
- **新契约** `scripts/training/check_orig16_stage2_1024_launch.py`：
  - `HISTORY_CONFIG`为新yaml；`FRAME_CONFIG=perceptual-framesamp-modul.yaml`。该纯帧配置只是独立参照，不产生第二条训练。
  - `check_history/check_frame_bytes`固定512/16/1，核源4×4布局；`check_config/check_parent`读 `framesamp/meta/store_meta.json`，layout必须为 `framesamp-4x4-v1`，数据路径不得指向8×8。
  - `Contract`固定GPU0–3、FSDP4、workers8；库路径、norm及manifest哈希、motion_rows、train_max/eval_max先填 `__FILL_AFTER_BUILD__`（出现即FAIL），阶段3实测回填并单独提交。
  - fsdp/workers与新契约比较，不能继续与基础配置8/16比较；batch128、seed、EMA与基础配置一致；VALUE_FLAGS增加两项且值固定4/8。
  - 必检 `CHECK_ENCODER_DATA_LINEAGE`、`CHECK_INPUT_EQUIVALENCE`（新kind必须有G2输入PASS与数值NOT_REQUIRED记录）；单run内的帧表、motion表、assets、input_manifest必须指向同一父库。保持65叶、参数形状和所有来源／budget检查。
  - 新增 `CHECK_FRAME_LAYOUT=PASS frames=32 tokens_per_frame=16 budget=512 total_tokens=608` 与 `GPU_SCOPE=PASS allowed=0,1,2,3`；末行仍为 `LAUNCH_CONTRACT=PASS checks=<n> failed=0`。
- **`scripts/training/check_stage2_online.py`**：
  - `prepare_restore_inputs`新增显式 `framesamp_store`、`h5_root/input_manifest`选择；CLI新增 `--framesamp-store`、`--h5-root`、`--input-manifest`。缺省保留旧mixed8入口。
  - 4×4模式校验实际layout、manifest、state表sha；`demo_state_origin.path`记录真实 `framesamp/state_emb.f32.bin`。原版demo直接从已钉SHA的源H5 `episode_<N>/timestep_<t>/obs/front_rgb` 读取，不能沿用mixed8的顶层 `frames` 假设。
  - `check_state_machine`按生产首包D+1、后续最多16帧喂入；视觉参数由新yaml的token_per_image16传入，禁止回滚 `FrameSampMemory.add_buffer` 的输入守卫。
- **`scripts/training/check_stage2_checkpoint.py`**：
  - 在S2的新缓存来源证据支持之后顺序适配 `run_online`：保存的demo＋连续65帧exec仍可使用，首次喂D+1，然后四批16帧，覆盖两个完整motion窗；更新计数断言与参考输出生成，记录不偷换输入或起点。
  - `--delivery-step`当前只接受60000/70000；明确增加79999支持，核目录、发布清单与参数步数相同，输出 `CKPT_DELIVERY_READY=PASS step=79999 …`，保留60k／70k兼容测试。新kind保存前后参数与EMA逐字节恢复仍硬核；跨精度或跨运行路径的动作输出差异只INFO，按kind适配 `compare_outputs` 的调用，不把数值零差伪写为必过。
- **兼容验证与写集**：
  - 新 `tests/training/test_check_orig16_stage2_launch.py`：正确夹具PASS；错8×8库、错GPU、缺FSDP参数、占位、G2输入证据缺失／失败、血缘不符均FAIL；合法NOT_REQUIRED不得被旧数值阈值误拒；入口 `bash -n` 与DRY_RUN无副作用。
  - `tests/training/test_framesamp_motion_stage2.py` 增加16/64 token矩阵，32帧边界、短历史padding、motion[96,1024]、mem_order[608]置换与同刻帧先于motion。
  - `tests/dataset/test_stage2_numeric_contract.py`／`tests/policies/test_stage2_restore.py`覆盖4×4便携状态来源、原版H5、D+1首包、79999与错步拒绝；S2先修改这些共享测试，S4从BASE2续改。
  - B5/G4须比较source与实际4×4表的图像、位置、mask、时间排序和motion输出；当前 `check_32frame_modul.py`对应2048预算，旧 `RefNpyFrameSampDataset` 的motion仍是768/stride16，不能原样拿来声称新整链通过。按现有stage2测试构造独立4×4真值，不放宽核心guard。

**S5 上传驱动（第二批；全部新建，mixed8 驱动零改动）**
- **训练库**：
  - 新增 `scripts/dataset/hf_export/run_orig100_trainset_export.sh`（复制 `run_mixed8_200_trainset_export.sh`，改 `NAME`，`BUCKET_ID=HongzeFu/robomme-16task-orig100-1600ep-trainset-20261027-v1`）、`orig100_trainset_bucket_README.md`、`hygiene_allowlist_orig100_trainset.json`。
  - 上传范围包含source、framesamp 4×4、norm、最终wan-latents、motion-stage2-1024、motion-tokens-stage2和缓存导入／输入证明；临时cache-source原始密集库已有固定公开版本，保留恢复来源清单，不重复发布一整份。


- **stage2**：
  - `_stage2_orig16_vla_ckpt_export_common.sh`：复制 `_stage2_vla_ckpt_export_common.sh`，并做以下修改：
    - 常量 `RUN`、`EXPORT_NAME=hf-ckpt-orig100-m32x4x4-vla-stage2-1024-80k`、`VLA_BUCKET=HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-stage2-1024-80k-v1`、`DATASET`、`HISTORY` 换新值；
    - `EVIDENCE_FILES`／`ROOT_FILES`及receipt检查按S2缓存来源改为 `G0/G2/G3/wan_cache_import/SOURCE_PIN`；
    - 删去 70k 停止专用常量；
    - `STEPS` 保持 5000…75000 + 79999。
  - `run_orig16_stage2_1024_ckpt_inc.sh`：训练期增量上传。
  - `run_orig16_vla_stage2_1024_80k_ckpt_export.sh`：最终导出。
  - 配套 `orig16_stage2_vla_bucket_README.md`、`hygiene_allowlist_orig16_stage2_vla.json`。
- **4×4训练库导出适配**：
  - `scripts/dataset/hf_export/pack_and_hash.py::build_jobs`新增显式orig16的4×4布局分支，新驱动/allowlist声明该布局；读取fs4的元信息、样本数、hash、source_spot_sha256，不要求fs8目录存在。
  - 源模板的mixed8／8×8分支行为不变。新训练库上传包括 `source/`、`framesamp/`、norm、motion与已选Wan来源证据，不为满足模板补造8×8表。
- **测试** `scripts/dataset/hf_export/test_orig16_export_drivers.py`（新）：仅训练库与stage2两套驱动；新run/bucket/库名一致，只有4×4表的夹具完成pack和回读，`bash -n`与 `SMOKE=1` 本地合同通过，旧mixed8入口零diff。

## 二、子代理分配表

未来实施BASE1为用户选定分支中包含本计划的HEAD（推荐 `v5-orig16`）；BASE2为S1/S2合入后HEAD；MJ BASE按开工时复核的memexport提交固定。AWS Codex各角色显式 `gpt-6-astra`＋`ultra`；修改者独立worktree，只读审查者不写，运行者只按确定版本启动。

| 编号 | 目标／可写集合 | 禁触／接口 | 合并／验收 | 资源 |
|---|---|---|---|---|
| E1 | 只读缓存恢复清单、段数／行数、原版H5和发布索引映射 | 不改数据或代码 | 输出C0证据，不能把历史报告当本机实体 | CPU／只读网络 |
| S6 | MJ `stage2_memory_export.py`及对应测试；计数／源pin及新metadata读取 | MJ其它文件与权重；消费S2/S3完整grid16，不强求copy/fill数值环境相同 | 1；指定计数、smoke与源错配负例 | CPU测试 |
| S1 | G0原版H5、单run lineage及第二部分一节所列测试 | 源数据只读；输出check_lineage与G0 | 2；正确、缺／错源与帧序用例 | CPU |
| S2 | 一节S2所列pack、datastore、client、snapshot、validate_stage2、compare_stage2、G4及测试 | 旧mixed8分支不变；统一新kind与证据键 | 3；新kind输入／完整性与旧合同回归 | CPU |
| S3 | `orig16_wan_cache.py`及两份新测试 | 不改Wan数值核心，不接增强train数据 | 4；copy/fill覆盖、SHA、输入与S2兼容 | CPU |
| S4 | 一节S4的新入口／契约／yaml、G4／checkpoint与列出的测试 | 不改混合库专用入口；从BASE2继承S2对共享文件的修改 | 5；32×4×4、608、D+1、79999、GPU越界负例 | CPU |
| S5 | 一节S5两套上传驱动、`pack_and_hash.py`新layout分支和测试 | 旧分支不变；新证据集与fs4布局 | 6；只有fs4的fixture打包回读 | CPU |
| RV-* | 固定BASE/TIP只读审查 | 无可写集合 | 每块合前；来源、输入、GPU范围缺失即FAIL | 无GPU |
| X1 | 新库、norm、cache-source、导入receipt、日志 | 禁改代码／原始H5；运行第二阶段确定版本 | 恢复单文件／补单窗smoke后放大 | 仅GPU0–3 |
| X2 | 新motion表、validation、日志 | 禁改代码；阶段3 | G0–G6输入／身份／完整性 | GPU0或0–3 |
| X3 | 唯一run、exports、日志 | 禁改代码／旧run | G7→正式单run，交回tmux与日志 | 训练0–3；恢复父0/sidecar1 |
| X4 | 最终exports、validation、日志 | 禁改代码 | G9/G12 | CPU／GPU0与1 |
| 主会话 | 删除裁决、Beta、只读锁、实测契约常量、档案／handoff、整合推送 | 不替运行者扩工作量 | 逐项核证据及资源清理 | CPU |

**共享文件顺序**：S2先拥有 `check_stage2_online.py`、`check_stage2_checkpoint.py` 和数值合同／恢复测试，完成新kind支持；S4必须从包含这些提交的BASE2续做4×4／首包／79999，不能并行覆盖同文件。S3只新增导入工具及测试；S5只改导出选择，来源政策使用S2接口。

各块逐路径核写集，合前审PASS后 `--no-ff` 合并，功能号以实际最新提交顺延；合后定向验证再push。全部代码适配在建库Beta之前完成。

## 三、闸门总表

| 闸门 | 命令／入口 | 判据／失败处置 |
|---|---|---|
| B0 | `df -B1 --output=avail /scratch`，逐阶段清单 | build前≥1.5 T、train前≥0.6 T；不足先列清理方案，不自行删 |
| C0 | `orig16_wan_cache.py plan` | 发布版本／来源一致，copy42404＋fill2824＝grid16 45228，grid32 23253（索引参考，正式manifest复核）；列恢复字节、段数、无重／漏；不同需先解释 |
| L0 | `check_data_lineage.py lineage --vla-input-manifest <参考input>` | encoder源H5与参考16文件size/SHA全等；FAIL停 |
| B1／L1 | 新manifest、hash-inputs、`check_orig80k_sources.py`；新input再跑lineage | 1600集／768897帧与参考一致，L0/L1同一源集合 |
| B2 | `run_local.py --stage siglip --gpus 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3` | STAGE_DONE workers16、SHARD_DONE16、相应分片16；不分配4–7 |
| B3 | `finalize_checks.py check --spot_check 1024 …` | 原始SHA和1024帧复算通过 |
| B4 | 4×4 pack／verify／report | 全768897行零不匹配；布局framesamp-4x4-v1 |
| B5 | S4真实4×4读取与线上采样夹具 | `FRAME_LAYOUT=PASS frames=32 tokens_per_frame=16 budget=512`；`STAGE2_FRAME4_COMPAT=PASS total_tokens=608`；图像/位置/mask/时间及排序正确 |
| B6 | cache fetch／import | 整文件发布SHA通过，copy字节保真，缺窗只fill；`WAN_CACHE_IMPORT=PASS copied=<n> filled=<n> total=<n> input_mismatches=0 nonfinite=0` |
| B7 | 导入receipt全量覆盖与输入来源检查 | `WAN_INPUT_COVERAGE=PASS windows=<45228实测> missing=0 duplicate=0 mismatches=0`；`WAN_NUMERICS=NOT_REQUIRED reason=user_decision_same_input`，不运行全量WAN_BITEXACT |
| B8／B9 | 获批features剪枝；norm与manifest哈希 | FEATURES_PRUNE kept8；DATA_IDENTITY绑定476857样本与新norm |
| G0 | 原版H5＋新库输入检查，关联import receipt | 64窗／16任务／≥16个demo尾窗，copy/demo-fill/exec-fill均覆盖，原帧SHA零差 |
| G1 | MJ抽取CLI源pin与eval/dropout状态 | MJ_CLI_LINEAGE文件16、mismatches0；冻结EMA、无memory增强 |
| G2 | `orig16_wan_cache.py check` | `G2_INPUT_IDENTITY=PASS windows=64 mismatches=0`；`G2_NUMERICS=NOT_REQUIRED reason=user_decision_same_input`；来源、shape、finite仍硬核 |
| G3 | `validate_stage2.py --gate G3`新kind | EMA65叶与model键严格一致，model_live负控、读出/geometry正确；跨路径数值NOT_REQUIRED |
| G4 | `check_stage2_online.py`，父与sidecar同GPU0 | `ONLINE_INPUT_IDENTITY=PASS windows=64 handshake=1 frame_ids_mismatch=0 state_machine_mismatch=0 input_sha_mismatch=0 nonfinite=0`；跨精度e2e差异仅INFO |
| G5 | stage2 pack／CPU完整性认证／verify | 源token对packed全表字节零差、SHA及finite通过、grid32补帧/覆盖正确；`ENCODER_RECOMPUTE=NOT_REQUIRED reason=user_decision_same_input`，不填未跑的1024窗 |
| G6 | `train_es_bound.py … --budget 96`及启动契约 | train/eval最大窗数≤96，无截断；实测常量回填 |
| L2／G7 | 单run DRY_RUN契约＋4卡20步＋保存恢复 | DATA_LINEAGE、GPU_SCOPE、FRAME_LAYOUT、LAUNCH_CONTRACT通过；20/20有限、peak≤72 GiB；参数/EMA恢复严格、真实在线推理有限，跨路径动作数值INFO |
| G8 | 单条正式日志与Orbax目录 | 真实EXIT_CODE0；5000…75000及79999共16份完整checkpoint |
| G9 | 单条run导出，先SMOKE1后全量匿名回读 | CKPT_UPLOAD steps16、sha256_mismatch0；失败长尾如实记录 |
| G10／G11 | 引用旧MJ备份；G11保留不用 | 不重传MJ，不伪造新验收 |
| G12 | `check_stage2_checkpoint.py --mode delivery --delivery-step 79999` | CKPT_DELIVERY_READY含step79999；身份/EMA/参数严格、真实infer有限；不把精度输出差作为阻塞 |
| B10／S0 | 新库导出；源H5只读复核 | RESULT=PASS；SOURCE_RO files16 size_mtime_unchanged16 |

## 四、runbook

以下均为**未来开工后的命令模板**。当前只修计划，不建库／下载／训练。新CLI由S块实施后先通过 `--help`、语法与夹具验证，不能把尚未实现的模板当作现成能力。

脚本和日志根为 `v1-store/logs/o16-20261027/`。每个脚本采用 `set -euo pipefail` 并加载共同env；长任务用独立tmux＋tee＋真实EXIT_CODE，主会话登记并监听。唯一允许GPU集合0–3；run_local的 `--gpus` 也必须受限。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/dataset/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
MJ=/scratch/hongze/MotionJEPA
MJW=/scratch/hongze/worktrees/vla-s2-memexport-20261026
RAW=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5
MJIDX=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/index
MJSRC=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json
PIN8=<阶段0按MJSRC实读>
NAME=16task-orig100-1600ep-$PIN8
LIB=$V1_STORE/datasets/$NAME
MANI=$LIB/meta/episode_manifest.json
INMANI=$LIB/meta/input_manifest.json
REF=$P/docs/dataset-build-doc/16task-h5-scan/records
CKPT=$MJ/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt
CKPT_SHA=27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035
ENC=$V1_STORE/external/motionjepa/stage2-full144-1025-a
VALID=$V1_STORE/validation/orig16-stage2
CACHE_ROOT=$LIB/cache-source/official-wan
CACHE_PLAN=$VALID/wan-cache-plan.json
CACHE_RECEIPT=$VALID/wan-cache-import.json
RUNS2=orig100-m32x4x4-modul-motion-stage2-1024-b128-80k
HC=perceptual-framesamp-modul-32frame-4x4-motion-stage2-orig16.yaml
lineage() {
  uv run --no-sync python scripts/training/check_data_lineage.py lineage --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --vla-input-manifest "$1" --mj-source-manifest "$MJSRC"
}
```

### 阶段0–1：身份、空间与代码

先核旧70k备份的真实完成行；未PASS不得删旧产物。原拟删的71 GB恢复副本、156 GB导出、155 GB旧run仍须第一部分第七节第2项裁决与逐项清单，绝不把本轮改计划当作删除许可。回填pin8，按S1定义预核L0；S块按分工实现／测试／合入。CPU清单可提前生成，不写训练HEAD：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$MANI"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_orig80k_sources.py --input-manifest "$INMANI" --reference-input "$REF/input_manifest.json" --manifest "$MANI" --reference-manifest "$REF/episode_manifest.json"
lineage "$REF/input_manifest.json"
lineage "$INMANI"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py plan --manifest "$MANI" --input-manifest "$INMANI" --published-index "$MJIDX" --out "$CACHE_PLAN"
```

C0文件清单与磁盘预算固定后，提交建库launch、建库Beta；全部S块先合入，再设置HOLD冻结。

### 阶段2：4×4训练库与缓存恢复／补窗

```bash
# o16-siglip，16worker＝允许的4卡各4个
lineage "$INMANI"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 --raw-dir "$RAW" --require-free-mib 70000
# o16-finalize，只用GPU3
CUDA_VISIBLE_DEVICES=3 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
# o16-pack4，CPU
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
# B5验证后，仅按store_meta指名且获批的features清理，保留8份抽样
# o16-cache-fetch，固定revision整文件恢复，先单文件smoke；只本命令解除HF离线模式
HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES='' uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py fetch --plan "$CACHE_PLAN" --revision 384d5db410be713a49357bf3f55ca4f4d2f559a0 --out "$CACHE_ROOT"
# o16-cache-import，仅缺窗用GPU，先单窗smoke
CUDA_VISIBLE_DEVICES=0,1,2,3 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py import --plan "$CACHE_PLAN" --cache-root "$CACHE_ROOT" --raw-dir "$RAW" --out "$LIB/wan-latents" --receipt "$CACHE_RECEIPT" --gpus 0,1,2,3
```

不再调用全量 `run_local.py --stage wan` 或全量oracle重编码；B6/B7由导入收据、全窗口覆盖、输入与文件完整性承接。保留copy/fill各自真实精度，数值差异NOT_REQUIRED。

### 阶段3：motion与输入验收

```bash
# o16-g0，原版H5与缓存映射
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_stage2_validation_inputs.py --manifest "$MANI" --latents "$LIB/wan-latents" --source "$LIB/source" --h5-root "$RAW" --input-manifest "$INMANI" --mj-index "$MJIDX" --cache-import-receipt "$CACHE_RECEIPT" --out "$VALID/G0.json"
# o16-s2export，在MJ工作树运行；expected-counts以C0实测回填，先独立输出smoke
CUDA_VISIBLE_DEVICES=0 uv run --project "$MJW" --no-sync python "$MJW/scripts/stage2_memory_export.py" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --latents "$LIB/wan-latents" --manifest "$MANI" --out "$LIB/motion-tokens-stage2" --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts <段数>,<grid32行数> --input-manifest "$INMANI"
# o16-g2，原始输入SHA与结构检查
CUDA_VISIBLE_DEVICES='' uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py check --samples "$VALID/G0.json" --receipt "$CACHE_RECEIPT" --raw-dir "$RAW" --out "$VALID/G2.json"
# o16-g3，EMA身份／读出验证，新kind数值差异NOT_REQUIRED
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/validate_stage2.py --gate G3 --samples "$VALID/G0.json" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --rgb-root /scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1 --cache-import-receipt "$CACHE_RECEIPT" --out "$VALID/G3.json"
# o16-s2pack，含新kind完整性认证；不附加GPU1024窗重算
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin published-cache-with-local-fill --cache-import-receipt "$CACHE_RECEIPT" --manifest "$MANI" --tokens "$LIB/motion-tokens-stage2" --latents "$LIB/wan-latents" --out "$LIB/motion-stage2-1024" --encoder-run-dir "$ENC" --raw-dir "$RAW" --expected-ckpt-sha256 "$CKPT_SHA" --stage2-validation "$VALID" --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store "$LIB/motion-stage2-1024" --resume --check-padding
# o16-g4，父与sidecar同GPU0
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/check_stage2_online.py --store "$LIB/motion-stage2-1024" --samples "$VALID/G0.json" --history "$P/src/mme_vla_suite/models/config/robomme/$HC" --framesamp-store "$LIB/framesamp" --h5-root "$RAW" --input-manifest "$INMANI" --online-gpu 0 --out "$VALID/G4.json"
uv run --no-sync python scripts/training/tests/train_es_bound.py --manifest "$MANI" --layout motion-stage2-1x1024-grid32-demopad17-v1 --budget 96
```

新kind模式必须让pack→verify→loader读取同一份完整性认证；实现时若原CLI需要独立认证步骤，S2须在pack完成前生成并验证，不能伪造旧bitexact记录。G0–G6完成后解除HOLD，提交建库结论及实测Contract常量，L1复核后push。

### 阶段4–5：单条短测与正式训练

```bash
# o16-smokes2：先DRY_RUN核GPU0–3／4×4／同源，再20步
CUDA_VISIBLE_DEVICES=0,1,2,3 DRY_RUN=1 RUN_NAME=o16-smoke-s2-<美东MMDDHHMM> MAX_STEPS=20 TRAIN_HEAD=<当前40位sha> FINAL_RECORD_DIR="$VALID/smoke/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_NAME=o16-smoke-s2-<美东MMDDHHMM> MAX_STEPS=20 TRAIN_HEAD=<当前40位sha> FINAL_RECORD_DIR="$VALID/smoke/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh
# o16-saveload：短测退出后，父0／sidecar1
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/check_stage2_checkpoint.py --mode smoke --checkpoint <step19绝对路径> --reference <保存前参考> --online-gpu 1 --out "$VALID/smoke/G7-saveload.json"
# o16-trains2：launch已提交、训练Beta、代码只读、DF_CHECK通过
lineage "$INMANI"
CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_NAME="$RUNS2" TRAIN_HEAD=<训练Beta的40位sha> FINAL_RECORD_DIR="$VALID/$RUNS2/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh
```

G7记录真实20步有限性、峰值与稳态步时；超5分钟按诊断留档。正式起跑要有DATA_LINEAGE、GPU_SCOPE、FRAME_LAYOUT、LAUNCH_CONTRACT与首个有限loss。增量上传只起 `o16-incs2`，训练库长尾只起 `o16-trainset-up`，都使用新S5驱动并降低IO优先级；不创建另一训练run。

### 阶段6：收尾

G8完成后，`o16-ups2` 跑 `run_orig16_vla_stage2_1024_80k_ckpt_export.sh` 的SMOKE1及全量回读；`o16-g12` 用父GPU0、sidecar1验远端79999：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/check_stage2_checkpoint.py --mode delivery --delivery-step 79999 --checkpoint <远端回读79999> --expected-manifest <发布清单> --online-gpu 1 --out "$VALID/G12.json"
```

按记录解锁、核源H5未变、写单条训练result与handoff。只清理本轮清单中的确切会话。

**tmux清单**：`o16-hash`、`o16-siglip`、`o16-finalize`、`o16-pack4`、`o16-cache-fetch`、`o16-cache-import`、`o16-g0`、`o16-s2export`、`o16-g2`、`o16-g3`、`o16-s2pack`、`o16-g4`、`o16-smokes2`、`o16-saveload`、`o16-trains2`、`o16-incs2`、`o16-trainset-up`、`o16-ups2`、`o16-g12`。

监听保留真实完成行、FAIL/Traceback/EXIT_CODE，以及DATA_LINEAGE、WAN_CACHE_IMPORT、WAN_INPUT_COVERAGE、G2_INPUT_IDENTITY、NOT_REQUIRED、ONLINE_INPUT_IDENTITY、VERIFY_MOTION、GPU_SCOPE、LAUNCH_CONTRACT和上传完成行。

## 五、风险登记

| 项 | 处置 |
|---|---|
| 有缓存不代表相同输入 | train增强fp16、mixed8错源均拒绝；复用官方原图FP32必须核发布源pin、索引、原始帧序／像素、权重与完整文件SHA |
| 缺窗范围被错误缩小 | 保持demo≥17补尾、exec到T；物理grid16补2824与消费grid32缺1439分别计，禁止只算后者却声明完整grid16 |
| 数值差异被误设为阻塞 | 当前精度裁决已裁定同输入不考虑额外精度差异；G2/G3/G4/G5及加载/导出消费者同步新kind，记录NOT_REQUIRED，不伪PASS |
| 缓存恢复网络／空间 | 固定revision，按C0整文件清单；单对象最多3次有界重试，失败保留日志，不默换版本、不自动全量重抽 |
| 4×4或首包遗漏 | 契约、便携输入、训练／在线采样、D+1首包及79999逐项测试；不回滚既有在线守卫 |
| 共享代码写入冲突 | S2先完成来源／数值政策，S4从BASE2续改相同验证文件；S5导出仅调用统一证据接口 |
| 建库冻结中发现缺陷 | 停并保留证据，交用户决定是否作废相关产物；不混用多个代码指纹 |
| 80k约21.5 epoch可能过拟合 | 步数待用户裁决；按选定步数保存中间checkpoint，可另行授权评估 |
| 磁盘与上传IO | 分阶段错开features与恢复缓存峰值，单run上传低IO优先级；旧数据删除仍须原授权 |

## 六、已核实与未验证边界

**本轮已核实**：
- 四卡、单条32帧4×4、608记忆位置的核心模型／loader已有基础能力；主要缺口在启动合同、验证和导出硬路径。
- 本机官方密集缓存没有正式payload；远端固定revision公开，API列出2485个Wan bin，未实际下载核SHA。
- 16增强版缓存实体在，但训练索引没有aug0；val原图不能代表全部训练集。
- 官方索引与RGB/GTXYZ索引的1600集D/C/T一致；grid16和grid32覆盖计数见第一部分2.4，按每segment重新从0取偶数格。
- `check_stage2_checkpoint.py --delivery-step`当前仅60000/70000，79999必须实现；G4硬编码8×8/demo_h5、旧喂包方式同样必须适配。
- source_manifest和build_spec源H5集合已作为同源检查依据；两个pin算法不同，不直接比pin字符串。

**尚未执行，不宣称通过**：
- 新缓存导入／补窗工具、新kind合同、32帧4×4入口和导出均是待实施内容。
- 逐发布文件SHA、恢复后索引／像素绑定、正式新manifest的精确段数／行数和磁盘峰值。
- 本次4卡步时、峰值显存、真实G7、79999恢复与训练效果。
- 公开API文件列表证明远端版本当前可列出，不等于全部文件已匿名回读验证。
- 同输入的跨精度数值一致不再是待解决项；如实记NOT_REQUIRED，与身份／输入／完整性验收分开。

## 七、状态机登记与交接

未来任务id为 `orig16-vla-build`（阶段0–3）和 `orig16-vla-train-m32x4x4-motion-4gpu`（阶段4–6，依赖前者）。只读消费MotionJEPA ep105，核其sha `27daa832…`。当前只改计划，两个执行任务尚未登记；是否登记及分支upstream以第一部分第七节第3项选择为准。

- 建库done前：C0、L0/L1、B1–B9、G0–G6真实通过，建库GPU退出、HOLD解除。长尾仅B10新库上传，句柄 `o16-trainset-up`，完成行RESULT=PASS。
- 训练done前：L2、B5、G7、唯一G8、G12、PERM_RESTORED与S0通过，训练GPU退出。长尾仅唯一G9，句柄 `o16-ups2`，完成行CKPT_UPLOAD=PASS。
- 新kind数值字段按当前精度裁决写NOT_REQUIRED，不把其当PASS，也不因为它不是数值PASS阻塞上述输入／完整性闸门。
- 下游另获授权的评估可消费该run的79999或中间checkpoint。旧mixed8训练任务继续保留同源FAIL历史状态，不改写成合规run。

## 八、留档、提交与本次修订

建库档案 `docs/dataset-build-doc/16task-orig100-1600ep-<pin8>/`；唯一训练档案 `docs/training-doc/orig100-m32x4x4-modul-motion-stage2-1024-b128-80k/`。launch在起跑前提交，result与records记录实测、完整用户裁决、缓存来源／copy-fill计数、输入身份与数值NOT_REQUIRED；不把已有源码、yaml和大权重复制进records。

功能号以实际git log顺延；建库Beta和训练Beta分别固定，结论提交与各自Beta同一小版本，禁squash/amend。每次commit后推现有upstream；首次建立v5-orig16需第一部分第七节第3项裁决。本次文档修订提交现有v4-256concat1，不建立新训练分支、不占功能版本号。

**本轮状态**：只读核查了现有代码、缓存索引、远端文件列表和旧备份完成记录；两项独立调整审查均为 `ADJUST_REVIEW=PASS scale=same`。未执行本计划的新下载、补窗、建库、删除或训练。三项用户裁决仍待答复。
