# StopCube Oracle 下压比例起跑档案

## 目的与用户授权

首次用户原话：「/data/hongzefu/robomme_policy_learning_MotionJEPA/1104-stopcube-oracle-press-plan.html」「开工」。首次批准计划提交为 `48be54c0414a362bfc5684d1357f383881103f92`，计划见[完整方案](../../../1104-stopcube-oracle-press-plan.html)。本次含官方 ctx512 79999 与 concat motion 59999；不恢复旧计划、不使用旧计划运行产物，不跑正常局。后续追加的同输入离线机制分析、再次开工及八点截止，以下方「口径、执行顺序与截止」的用户原话和完整两阶段方案为准；仍不实施输入干预或因果消融。

状态机登记尚未获答复，按计划不写入 `docs/handoff/current.json`，采用人工交接。

资源补充授权原话：「允许使用剩余显存，不操作其他进程」。现场两张卡有其他用户的小显存进程；本轮允许使用剩余容量，不停止、暂停或修改他人进程，不新增高频查询。

## 环境与版本

Aspen 本地环境 A，主检出 `/data/hongzefu/robomme_policy_learning_MotionJEPA`，分支 `v4.1-interpretMotion`，两张 RTX A6000。不连接集群。主检出有用户未跟踪文件 `TODO_RoboMME_Qwen3VL_FrameSamp_Context.en (1).html`，不动、不提交；真实运行另用干净实现工作树并记录精确提交。

benchmark 锁定 `856bc3a189d4172f3f47dbee4424d585f8d78db3`。策略环境为主检出 `.venv`，仿真为 `v1-store/envs/official-sim`，motion 编码为 `v1-store/envs/stage2`。基线规格沿用[既有资产与规格档案](../attn-ctx512-vs-motion-1103/launch.md)，本轮运行前重新核实实际指向与参数。

静态核对三套解释器均为 CPython 3.11.14。主环境 editable 指向主检出 `src` 和 `packages/openpi-client/src`；仿真环境 `robomme.pth` 指向主检出锁定 benchmark。执行工作树以 `PYTHONPATH` 优先指定其自身代码。motion 配置名采用本计划所引用旧起跑档案第三节的 `mme_vla_suite_orig16_b128_60k`；基线采用 `mme_vla_suite_b128`。当前 `training/config.py` 已注册两者，模型历史由检查点配置快照恢复。此前依据 11-03 历史陈述误写“未注册”，已回源码纠正，未启动模型、未发生运行配置变更。缓存须由外层显式指定到本机 `v1-store/cache`。

## 口径、执行顺序与截止

50 局官方 StopCube，环境 seed 不变，模型 seed 42；首问 `t=1`，随后 `t=1+4k≤M·N`，每点 10 预测、两个模型使用相同噪声。询问预测不执行，环境动作表覆盖官方终局。先两局最小试跑，再 50 局动作表，先询问分层 12 局并校准，向用户报告校准结果，继续其余 38 局询问，余量够才补放手。

用户将截止延为美东 2026-11-05 08:00 EST，并要求第一阶段后继续同输入离线机制分析。用户要求先停后，三个自有会话均自然结束退出0；完整两阶段聊天计划呈现后，用户明确：「从现在开始尽可能不要再问用户阻塞性的问题，有问题可以问用户，但是不能阻塞之后任何的进行，尽可能一路跑到底，跑到明天早上八点。」「同意开工了，有问题可以现在立刻问，但是之后就尽可能不要问阻塞问题。」新开工一致性审查为 `PLAN_CONSISTENCY_REVIEW=PASS base=bf9f1c65ee21f1c9cbb002d3e79d7ec9327413d8 findings=2 blocking=0`；两处文字已补齐首次/本次授权和S5分工，不改变口径。

第二阶段每种M选至少一个模型固定放手成功的最早局，最多三局；每局四类保存询问点、两模型最多24输入，每份原十噪声最多240预测。原路径阈值仍1e-6，再提取注意力、历史帧/经过/输入组及层头统计，只给关联性解释。不替换motion、不遮挡、不造观测、不训练探针。第一阶段汇报后直接继续，08:00前10秒停止本轮计算并保留非零退出证据。运行前完整命令、工作树、提交、GPU、端口与会话名仍由主会话写入本档案。

## 审查与代码分工

`PLAN_CONSISTENCY_REVIEW=PASS base=48be54c0414a362bfc5684d1357f383881103f92 findings=2 blocking=0`。审查修正文案两处：第 0 步约 2 小时；补明首包 `D+1` 与首次询问 `t=1`，均不改既定口径。独立工作树 S1 动作表、S2 运行器、S3 询问存盘与离线、S4 分析；依次独立审查、保留子提交合并、短测、推送。

## 产物与盯盘

本轮产物根 `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/`，各局各目录；子目录为 `oracle/`、`smoke/`、`query/`、`calibrate/`、`release/`、`analysis/`、`logs/`。大数组和视频不进 git。具名验证写入 `records/verification.summary.txt`，失败不改阈值、不为成功重试。

## 全量询问结束后的固定接续命令

以下是预登记的接续规格，不代表已起跑。原生产树固定`48a52f1e083ee11777795e6d6f8dbc86056d0544`，离线树固定`050632677bea16f86a6b317d96c7d79f6eccabe6`；两者均须保持干净，不移动在跑树的HEAD。两模型query38自然退出0且各2990点完整后，运行`bash <产物根>/verify_collection.sh`，结果为`analysis/collection-verdict.json`，核两模型50局网格、十噪声、原始状态、全部Oracle控制及同步轨迹端点。官方终局会自动重复最后动作一次，因此外层轨迹最后一行跨两物理步；按真实`env_step_after`对齐Oracle逐物理步表的终点，不将两种记录粒度错配。

随后并行启动三份独立会话，主会话仍统一负责监听与清理：

```bash
tmux new-session -d -s so-fk38-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_fk38.sh'
tmux new-session -d -s so-offline-verify-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_offline_verify.sh ctx'
tmux new-session -d -s so-offline-verify-mot-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_offline_verify.sh mot'
```

FK使用生产树和official-sim解释器，4个单线程CPU进程，复用已验证24份缓存、只新增76份；完整200点原路径/注意力重放使用离线树、每卡一个模型，原batch2、十噪声、阈值1e-6不变，结果与日志分别为`analysis/offline-verify-<模型>.json`、`logs/offline-verify-<模型>.log`。FK输出`analysis/all/{cache,summary.json,calibration_plan.json}`和`logs/fk38-a.log`。同一原预测不追加或替换采样。

FK完成后用`bash <产物根>/prepare_release38.sh`从既有汇总逐局调用原`select_points`只取`fixed_release`，先核原12局固定计划完全相同，另存`analysis/release38-plan.json`与`spec-release38-<模型>-a.json`。两模型离线验证各自退出后，预留`so-release38-ctx-a`、`so-release38-mot-a`，命令为`tmux new-session -d -s so-release38-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_release38.sh ctx'`（motion参数mot）；固定原生产树、原38局manifest、原模型配置/显存/种子，输出`release/<模型>/release38-a/{attn,run}`，日志`logs/release38-<模型>-a.log`。不根据12局结果改放手规则，不为失败重试。全部命令仍使用绝对08:00截止；非零退出保留，主会话记录并继续不依赖失败项的授权工作。

## 模型询问试跑规格

固定实现提交 `48a52f1e083ee11777795e6d6f8dbc86056d0544`，执行工作树与缓存路径同下。S1～S4合前与合后均独立审查通过，合后分别41、17、26、25项定向测试通过。先单局 `StopCube:16`、每模型单服务，ctx用GPU0、motion用GPU1。规格文件为本轮产物根 `spec-smoke-query16-ctx-a.json` 与 `spec-smoke-query16-mot-a.json`；基线配置 `mme_vla_suite_b128`、XLA0.40，motion配置 `mme_vla_suite_orig16_b128_60k`、XLA0.55及原运行合同，checkpoint与三套解释器保持本档案已核路径。新增值为 `stopcube_mode=oracle_query`、`query_interval=4`、`query_record=10`、`query_batch_size=2`、`save_top_frames=0`、`save_prefix=false`；动作表为两局合并的 `oracle-smoke-table.json`。`model_id` 使用checkpoint绝对路径，与实际query记录一致。

完整命令分别为 `tmux new-session -d -s so-query16-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_smoke_query16.sh ctx'`，以及同脚本参数 `mot`、会话名 `so-query16-mot-a`。实质命令是策略环境 `-m scripts.interpret.run_local --spec <本轮spec绝对路径> --episodes StopCube:16`；脚本核精确提交与clean、按绝对08:00减去现场时间计算timeout，截止前10秒TERM，无自动重试。输出分别为 `<产物根>/smoke/query16-ctx-a/` 与 `query16-mot-a/`，含 `attn/` 和 `run/`；日志 `<产物根>/logs/query16-ctx-a.log`、`query16-mot-a.log`。会话均先预留，收到回执后记实际启动。

## 本轮会话清单

剩余38局实际于美东01:35:18启动：`so-query38-ctx-a`、`so-query38-mot-a`，窗格PID559354、559351，八客户端。实际服务端口57671、39501，nonce分别d06befe2d7f945b79c6bee3eb4eeae9b、12a174b4298b454ebc1e586980427ce4；运行身份分别624b40a858e99fd0832e33637a71bc27b94e388060eec182c727f08ad1bdc4e9、6f18489db1c58a9ea4eaef164e9402fb3a806e8cf5797636ae983e3588c40a99。生产代码仍固定48a52f1。

中间12局概率图与经过图已实际渲染、目视后修复图例遮挡及混合目标次数的问题；新版短图例位于图外、M×N分面、中文字体正常，主会话再次查看两张真实位图通过：`REPORT_VISUAL=PASS stage=query12 figures=2`。修复只改显示，不改数据或统计；新报告工具源码在050632677bea16f86a6b317d96c7d79f6eccabe6，单独干净离线树 `/home/hongzefu/.codex/worktrees/stopcube-offline-run/robomme_policy_learning_MotionJEPA` 已备妥，避免触碰正在采集的树。

正式50局每局两点离线验收将在两侧38局完成并核完整后运行。预留 `so-offline-verify-ctx-a`、`so-offline-verify-mot-a`，命令 `tmux new-session -d -s so-offline-verify-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_offline_verify.sh ctx'`（motion参数mot）。离线树固定0506326，沿用原模型/环境/确定性XLA/显存比例；从正式query12/query38每模型50个局各取首尾两份记录，共100点/模型，保存 `analysis/offline-inputs-<模型>.json`。策略解释器 `-m scripts.interpret.offline_query --query <固定100路径> --ckpt <原检查点> --config <原配置名> --attention --output <产物根>/analysis/offline-verify-<模型>.json`，复用每点原十噪声及实际batch2，原路径tol1e-6，展开差异独立报告；日志 `logs/offline-verify-<模型>.log`，逐点实时判定，不换输入、不重试、截止08:00。此为第一阶段既定每局两点验收，不计作第二阶段选例扩张。

另预留全量高度换算 `so-fk38-a`，命令 `tmux new-session -d -s so-fk38-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_fk38.sh'`。仍固定48a52f1；将已核24份query12高度缓存逐字节复制到新的analysis/all/cache，再对query38两模型76局目录执行原convert-z四worker，最后对合计100份缓存运行summarize并消费既有calibrate结果。输入/query与trace路径严格指向正式query38，动作表全50，输出analysis/all与日志logs/fk38-a.log；不重算已经通过的query12模型或高度，不改原固定校准点。运行后现场核新全量计划与原12局计划选点完全一致。

正式校准实际于美东01:16:51启动，`so-calibrate12-ctx-a`、`so-calibrate12-mot-a`，窗格PID537210、537208，均已退出0。逐项核74条收据与独立轨迹、实际交还点、Oracle前缀、首次预测及模型控制权、合法停止和持久终态，`PRESS_CALIBRATION=PASS models=2 episodes=12 planned=74 completed=74 valid=74 skipped=22`。固定放手ctx2/12、motion6/12最终任务成功；高比例48步触发10/12、12/12，低比例5/12、3/12。低比例组首次至多16个模型动作的实测TCP均未低于悬停2cm，后续重新推理可能触发，故不能用48步触发反推首次预测错误。此辅助统计不改变原2cm/16行预测判据或48步运行上限。校准结果已先在聊天汇报，不等待额外确认继续。

剩余38局询问预留 `so-query38-ctx-a`、`so-query38-mot-a`，完整命令 `tmux new-session -d -s so-query38-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_queries.sh ctx query38'`（motion参数mot）。固定48a52f1与原GPU/解释器/检查点/显存/十预测batch2；清单为前述互斥38局seal68a60f21…a1caff9，每模型2990点、8客户端共用一服务。规格 `spec-query38-<模型>-a.json`，输出 `query/<模型>/query38-a/{attn,run}`，日志 `logs/query38-<模型>-a.log`，绝对08:00、不自动重试。预先准备不等于已起跑。

正式query12两侧均退出0，主会话实查24局终态、1480点完整、无重复或缺失、每点十预测，两模型噪声逐位相等；动作/7臂关节/TCP/方块轨迹与各局Oracle表差0。`so-fk12-a` 实际于美东01:12:30起跑（窗格PID533057），已退出0：`FK_PARITY=PASS rows=1480 maxabs_z=0 tol=1e-3`；`PRESS_SUMMARY=PASS models=2 points=1480 threshold_cm=2 horizon=16`。固定选点文件 `analysis/query12/calibration_plan.json` 在真实校准期间保持不变。

预留正式校准 `so-calibrate12-ctx-a`、`so-calibrate12-mot-a`，完整命令 `tmux new-session -d -s so-calibrate12-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_calibrate12.sh ctx'`（motion参数mot）。固定48a52f1、原模型和解释器/缓存/显存；每模型按fixed_release→high_probability→low_probability→early_press串行，各类型整段共用一服务、客户端至多5局。规格 `spec-calibrate12-<模型>-<类型>-a.json`，清单沿用12局封签，选点沿用上述不可变文件；输出 `calibrate/<模型>/<类型>-a/{attn,run}`，日志 `logs/calibrate12-<模型>-a.log`。基线选中12/12/12/2，共38次；motion选中12/12/12/0，共36次，合计74次。其余22项早按点按原规则无候选或P全0，记录跳过原因，不补局。前三类型触发或交还48步即止，固定放手官方终局；不换seed，不挑成功重试。仍按绝对08:00限时。完成后先向用户报告，再启动其余38局询问。

正式分层询问实际于美东00:36:19启动：`so-query12-ctx-a`、`so-query12-mot-a`，窗格PID504851、504849。实际服务端口52005、35027，服务nonce分别9cad3d54793f4232bc5eac3b374f35ea、5f45ed2262eb4132a00d46142122d487；运行身份分别9c5188ae2e8f03114b55601abd40a87ba6174f73e4289a1576a38b7d7440248b、a1a70a1c5b4870e564f8de81ef75c813237c4f4ef1d892eb4f6a17dcc7781661。

询问两模型完整退出0并核740点后，下一步预留 `so-fk12-a`：`tmux new-session -d -s so-fk12-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_fk12.sh'`。同一48a52f1，official-sim调用 `-m scripts.interpret.analysis.oracle_press_prob convert-z --query-root <产物根>/query/ctx/query12-a/attn --query-root <产物根>/query/mot/query12-a/attn --trace-root <两模型对应run目录，各一项> --oracle-table <产物根>/oracle/oracle_table.json --output-dir <产物根>/analysis/query12/cache --workers 4`；每进程OMP/MKL/OpenBLAS1线程。随后策略解释器调用同模块 `summarize --cache-root <该cache> --oracle-table <全50表> --output-dir <产物根>/analysis/query12`，固定12局校准选点，原2cm/16行判据与bootstrap2000保持。日志 `logs/fk12-a.log`，仍绝对08:00，无自动重试。此处只是预先记命令，不能据此声称已经算完或起跑。

正式Oracle于美东00:29:12启动 `so-oracle-all-a`（窗格PID491779），已正常退出0：`ORACLE_TABLE=PASS episodes=50 failed=0 late_reach=0`；`ORACLE_PRESS=PASS episodes=50 success=49 trigger_outside=1`。非成功局为4，正常任务失败，不重试；悬停表和真实几何窗口完整，询问消费悬停段不受真下压失败影响。

正式两模型分层12局询问预留 `so-query12-ctx-a`、`so-query12-mot-a`。完整命令为 `tmux new-session -d -s so-query12-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_queries.sh ctx query12'`（motion参数mot）。固定同一48a52f1干净树、原检查点/环境/显存与batch2，每点仍十预测。`manifest-query12.json` seal=`bf094f6aac88f9cd27925be80fbf744db07b4a250a4755407a7998945794a77e`，12局三客户端分块5/5/2，每模型整段复用一份服务，740个询问点。顺序为M60局1/3/4/5，M80局0/2/12/13，M120局6/9/10/14；均为每M最小四局。规格 `spec-query12-<模型>-a.json`，输出 `query/<模型>/query12-a/{attn,run}`，日志 `logs/query12-<模型>-a.log`。实际端口现场探测，服务nonce、各client grant与每局nonce由原驱动生成并落commands.json；不重试、不减预测，绝对08:00。

其余38局清单亦已封签，与12局互斥且并集50：`manifest-query38.json` seal=`68a60f2171969aa82985cced4d7cc7e98d080163da38843c8d8bd5b52a1caff9`，每模型2990点、8客户端。其余38局只在12局询问、校准及向用户汇报后运行；不得据预先准备的spec视为已经启动。

小规模校准实际于美东00:24:26启动，`so-smoke-calibrate-ctx-a`、`so-smoke-calibrate-mot-a`，窗格PID485167、485165；8次两类型两局两模型均有持久终态和有效控制权证据。fixed_release两模型各2/2官方成功；high_probability两模型各2/2在交还后48步内触发，随即以calibration_stop截断，不能将其原始task_success=0字段解释为完整任务失败。`PRESS_CALIBRATION_SMOKE=PASS models=2 episodes=2 run_types=2 runs=8 valid=8`。正式构表4进程各设置OMP/MKL/OpenBLAS线程数1，避免CPU线程过度占用；不更改物理参数或环境seed。仍采用上文预留so-oracle-all-a命令，起跑回执后补。

小规模八项验证已通过：Oracle两局hold/press退出0；第16局两模型动作差0、7臂关节/TCP/方块轨迹差0；询问300点每点10预测；第16局4/16节奏各30公共点输入差0、帧号差0；两模型两局各首尾2点共8点原路径离线误差0（tol1e-6），注意力展开差异单列；FK同期300个原始状态高度差0（tol1e-3），同时换算60000行目标高度。随后预留两模型校准试跑：`so-smoke-calibrate-ctx-a`、`so-smoke-calibrate-mot-a`，完整命令 `tmux new-session -d -s so-smoke-calibrate-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_smoke_calibrate.sh ctx'`（motion参数mot）。同一48a52f1干净树及原解释器/缓存/显存，按fixed_release→high_probability顺序，每类型复用一个服务依次运行第16、3局；使用smoke/analysis-a/calibration_plan.json固定选点，动作表oracle-smoke-table.json。规格文件 `spec-smoke-calibrate-<模型>-<类型>-a.json`，输出 `smoke/calibrate-<模型>-<类型>-a/{attn,run}`，日志 `logs/smoke-calibrate-<模型>-a.log`，不自动重试、仍截止08:00。

校准小规模证据核验通过后，正式Oracle构表命令为 `tmux new-session -d -s so-oracle-all-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/build_oracle_all.sh'`。固定同一48a52f1，official-sim环境，4个独立进程并行，局号0～49；第3、16局复用本轮已核smoke，其他每局先hold再press，失败留日志、不换seed不重试。每例CLI为 `-m scripts.interpret.stopcube_oracle --episode-key StopCube:<局> --mode hold|press --benchmark-root <主检出>/third_party/robomme_benchmark --output-dir <产物根>/oracle/hold|press/<局>`，press另读对应hold表。聚合输出 `oracle/oracle_table.json`、`oracle/build_summary.json`；主日志 `logs/oracle-all-a.log`，逐局日志 `logs/oracle-hold|press-<局>.log`。构表失败与Oracle任务成功数分别统计，不以退出0替代50局完整性。

两份续测实际于美东00:12:31启动，`so-followup-ctx-a`、`so-followup-mot-a`，窗格PID473785、473787。第3局两模型已退出0，后续诊断与离线仍在进行。另预留 `so-smoke-fk-a`，完整命令 `tmux new-session -d -s so-smoke-fk-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_smoke_fk.sh'`；同一固定48a52f1干净树，official-sim解释器调用 `-m scripts.interpret.analysis.oracle_press_prob convert-z`，四个 `--query-root` 指向本轮query16/query3各ctx/mot的attn目录，`--trace-root <产物根>/smoke`、`--oracle-table <产物根>/oracle-smoke-table.json`、`--output-dir <产物根>/smoke/fk-a --workers 2`。同期trace按nonce绑定，核真实raw_state换算高度；日志 `<产物根>/logs/smoke-fk-a.log`，不启动模型或物理步进，不重试。

两模型询问试跑实际均于美东11-05 00:04:43启动：`so-query16-ctx-a`、`so-query16-mot-a`，窗格PID分别465684、465681。首段稳定点间隔约2.5秒，最终退出及数值验收另记，不以存活替代完成。

预留续测会话 `so-followup-ctx-a`、`so-followup-mot-a`，各自在对应第16局询问退出0且输出完整之后启动；不同模型可以并行，同卡不共驻两份服务。完整命令为 `tmux new-session -d -s so-followup-ctx-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/run_smoke_followup.sh ctx'`（motion对应参数mot）。固定同一干净48a52f1，顺序执行：`run_local --spec <产物根>/spec-smoke-query3-<模型>-a.json --episodes StopCube:3`；`run_local --spec <产物根>/spec-smoke-cadence16-<模型>-a.json --episodes StopCube:16`；`offline_query --query <第16局首尾两点> <第3局首尾两点> --ckpt <已核检查点> --config <原配置名> --attention --output <产物根>/smoke/offline-<模型>-a.json`。前两条采用策略解释器 `-m scripts.interpret.run_local`，最后为同解释器 `-m scripts.interpret.offline_query`。原点路径从完成局的原子存盘文件确定，原噪声与batch2随记录恢复；环境XLA确定性参数与线上一致。日志 `<产物根>/logs/followup-<模型>-a.log`，任一命令失败保留退出码，不自动重试；截止仍绝对08:00。诊断16步只用于输入节奏验收，不与正式4步点混作独立样本。

实际启动：`so-assets-1104`，CPU 资产验证，干净工作树提交 `49db7d7bbc0d7a772f512d9804cb33966c04cafe`，日志 `<产物根>/logs/assets.log`。核验耗时 47.73 秒、退出 0，54 个 motion 文件和 17 个官方基线文件通过。

预留下一份会话：`so-oracle16-hold-a`，只有收到启动回执才视为已运行。完整命令为 `tmux new-session -d -s so-oracle16-hold-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/smoke_oracle16.sh'`。实质命令如下，外层使用 `pipefail`、`tee`、无缓冲和 `EXIT_CODE=`；单次最长 290 秒，不重试：

```bash
CUDA_VISIBLE_DEVICES=1 /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/envs/official-sim/bin/python \
  -m scripts.interpret.stopcube_oracle --episode-key StopCube:16 --mode hold \
  --benchmark-root /data/hongzefu/robomme_policy_learning_MotionJEPA/third_party/robomme_benchmark \
  --output-dir /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/smoke/oracle16-hold-a
```

执行工作树 `/home/hongzefu/.codex/worktrees/stopcube-oracle-run/robomme_policy_learning_MotionJEPA`，固定 S1 合并提交 `bc0cc3dee4889d52f4fa79b294bbc600f3ac8d29`，启动前须干净；`PYTHONPATH` 先指向该工作树及其 `src`，benchmark 指向主检出已锁定源码。`UV_CACHE_DIR`、`XDG_CACHE_HOME`、`HF_HOME`、`TORCH_HOME`、`MMEVLA_V1_STORE`、`OPENPI_DATA_HOME` 均显式设置为主检出本机产物根下相应目录。策略模型尚未启动，本次只验证无模型的 Oracle 单局。

`so-oracle16-hold-a` 已于美东 11-04 23:51:51 实际启动并退出 0。第 16 局 `reach_step=51`、`z_hover=0.15438616275787354`、真实区间 412～428，481 动作至 `terminal_env_step=482`；只悬停官方任务失败符合预期，不当作模型成绩。

预留下份会话 `so-oracle-more-a`：`tmux new-session -d -s so-oracle-more-a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/smoke_oracle_more.sh'`。同一干净 `bc0cc3d` 工作树、解释器、GPU1 与缓存口径，按顺序各起独立进程：`--episode-key StopCube:16 --mode press --hold-table <产物根>/smoke/oracle16-hold-a/oracle_table.json`；`--episode-key StopCube:3 --mode hold`；`--episode-key StopCube:3 --mode press --hold-table <产物根>/smoke/oracle3-hold-a/oracle_table.json`。每条附同一 `--benchmark-root`，输出分别为 `<产物根>/smoke/oracle16-press-a`、`oracle3-hold-a`、`oracle3-press-a`。每例 290 秒上限，失败即退出、不重试；总日志 `<产物根>/logs/oracle-more-a.log`，逐例日志同名。第 3 局按官方 seed 随机序列推算为 M=60、N=2，实跑再核实。
