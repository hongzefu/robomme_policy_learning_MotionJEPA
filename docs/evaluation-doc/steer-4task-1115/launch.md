# 四任务两模型 RouteStick / VideoRepick 同一局 motion steering 启动档案（steer-4task-1115）

## ① 目的

在四任务 × 400 训练的两个 modulation 模型上，于新 benchmark `b4e97f2` 用**同一局供体**做 motion token 替换，测 RouteStick（第 j 步：只翻绕向 / 只翻左右 / 两者都翻 / 置零，及 core 窗口版）与 VideoRepick（第 k 次交换段换成同局第 k′ 次 / 置零）。计划：根目录 `1115-4task-routestick-videorepick-steering-plan.html` r3.2（开工令所指 `96d016f`，开工一致性审查后修订 `d7b7f97`）。

## ② 用户决策原话（美东 11-15 晚 — 11-16 凌晨，按时间）

1. 「我需要测……Frozen One Pooling 的这个模型和这个 Motion Mem 的这个模型……只测这个 route stake 任务和 video repick 任务……具体的测的内容，和之前保持一致……尽可能给我一个 definitive 的 answer，一直到……明天早上十点。」
2. 「routestick 你检查一下我这个生成的源码，它是有一个除了位置不同，还有这个方向的不同」
3. 「我说的这个另外两个模型是指……四乘四百这个 task。都在新的benchmark去评估！……Motion Memory 的是五十四点六的这个版本……Frozen One Pooling 的这个是四十三点一的这个版本。」
4. 「你先不要做跨局，你先用同一局来给我结论」
5. 「确保一次开始之后能一路执行到底」
6. 选择题：硬件「A40 + aspen 混跑」；B 范围「可以，A 优先」；状态机「登记」。「我们暂时不考虑不同硬件的这种情况。」「然后你一张卡上是不是可以跑多个 worker只要显存允许的话」
7. 「这四种都要，就是它临近的这个往左往右 clockwise counterclockwise 都要测。」「有时间就继续测这第四步之后」
8. 开工令：「好的，开工吧，开工吧，然后尽可能不要再阻塞了，然后有问题尽可能留到明天早上。」

## ③ 版本与代码状态

- 分支 `v4.2-plswap-full`；合并链：`commitV4.7.1`（S3 读数 46c7c29）→ `commitV4.7.2`（S2 同局变体/供体/截断 7c1a420）→ `commitV4.7.3`（MERGE-1 合入 v17 fd20feb、benchmark gitlink b4e97f2，1284cff）→ `commitV4.7.4`（S1 modulation 只换 motion 服务，f0e6f00）→ `commitV4.7.5`（FIX-1 scripts/dataset/wan 恢复 v7 版，6d8edc8）。
- 运行代码：aspen `v1-store/worktrees/s4t-run`（detached `6d8edc8893ef1fecdd9981b6a2c995090162ad8c`，clean）；GL release `…-eval/releases/6d8edc8893ef1fecdd9981b6a2c995090162ad8c`（`GL_RELEASE=PASS`，`.release_sha` 比对）。局池与变体由 `v1-store/worktrees/s4t-prep`（`1284cff`，S2 代码与 6d8edc8 相同）生成。
- 闸门：`PLAN_CONSISTENCY_REVIEW=FAIL base=96d016f findings=9 blocking=2`（主代理按用户原话自裁，见计划 r3.2 与 current.json pending_decisions）；各块 `PRE_MERGE_REVIEW=PASS`；`ADJUST_REVIEW=PASS scale=same`（FIX-1）。

## ④ 被跑对象

| 模型 | checkpoint | model_kind | 校验 |
|---|---|---|---|
| A MotionMem（54.62%） | GL：turbo `…/robomme_policy_learning_MotionJEPA/v1-store/models/robomme-vla-modul-motion-80k-v1/50000`；aspen 本地副本 `v1-store/assets/vla-4task-modul-motion-50000/50000` | `modul_v10_768`（modulation + encoder wan-full1600-filter2-b176x4-72ep-a ep72，arch wan-latent-v7） | `VLA_SYNC=PASS model=A files=20 sha_mismatch=0`（turbo 与 aspen 本地各一次，对 bucket SHA256SUMS.pre.txt） |
| B Frozen Wan pooling（43.06%） | turbo `…-eval/v1-store/models/robomme-vla-4task-modul-wantp768-50k-v1/50000`；aspen 本地副本复制中 | `wan_fixedpool768` | `VLA_SYNC=PASS model=B files=25 size_bad=0 sha_mismatch=0` |

policy_config 均为 `mme_vla_suite_b128_80k`，max_steps 2000，模型 seed 42，换 seed 噪声对照 seed 1、2。

## ⑤ 局池与变体

- RouteStick（seed 3100000 起）：`SP_SCAN=PASS seeds=79`；`SP_POOL=PASS eps=60 d=easy:8,medium:16,hard:16,xhard:20`。
- 同局变体：`RS_SEGLEN=PASS eps=60 variants=903 bad=0`；`RS_SPEC_REBUILD=FAIL eps=60 frame_mismatch=1`（seed 3104700 xhard，整局剔除）；`RS_VARIANT=FAIL variants=903 early_diff=21 swing_mismatch=0 skipped_endpoint=180 usable=882`（21 个 early_diff 全出自被剔除那局）。可用 59 局、882 个变体。
- VideoRepick（seed 3200000 起）：`FP_SCAN=FAIL seeds=112 errors=8`（benchmark b4e97f2 场景生成失败时 except 了非异常类，抛 TypeError；8 个 seed 剔除）；`FP_POOL=PASS eps=50 d=easy:10,medium:20,xhard:20`。

## ⑥ 冒烟

- GL（A40，占位 66195002/66195003）：A `RUN_LOCAL=PASS success=1`（RouteStick:0 easy，60 s/局，录制 11 窗 × 768），`NOOP_EQ=PASS model=A differ=0 n_replaced=11/11`；B `RUN_LOCAL=PASS success=1`，NOOP 运行 `RUN_LOCAL=PASS`。
- aspen：turbo NFS 上的 sidecar venv 握手超时 600 s → 建本地 `v1-store/venvs/wan`（uv sync --frozen，motion_jepa + torch 2.9.0）与 encoder 本地副本后加入。

## ⑦ 运行方式与资源

- GL 占位 4 个（`s4t-hold-1..4`，JobID 66195002–66195005，1×A40、8 CPU、96G、48h，开工令放行）；每占位 3 条通道（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.27`，通道错开 45 s），每通道一个长 srun 作业步依次跑 n → vrec → noise（`rs/motion/chains/chain1-<w>.sh`，启动器 `rs/motion/gl_chain1.sh`），tmux `s4t-m1-g{0..3}{a,b,c}`（GL 登录节点）。
- aspen：两卡各 1 通道（他人占用 17–18 / 16 GiB），tmux `s4t-m1-aspen0/1`，各依次跑两个 aspen worker 的队列。
- 产物：NFS `…-eval/v1-store/evaluation/steer-4task-1115/{rs,vr}/{motion,wan}/`；GL 日志 `…-eval/v1-store/logs/steer-4task-1115/`；aspen 日志 `v1-store/logs/steer-4task-1115/`。

## ⑧ 本轮 tmux 会话清单

aspen：`s4t-dl`、`s4t-prep-rs`、`s4t-prep-vr`、`s4t-smokeA`、`s4t-smokeA2`、`s4t-smokeA3`、`s4t-smokeA4`、`s4t-cpA`、`s4t-cpB`、`s4t-release`、`s4t-aspenenv`、`s4t-m1-aspen0`、`s4t-m1-aspen1`。GL 登录节点：`s4t-glsmokeA`、`s4t-glsmokeGA`、`s4t-glsmokeGB`、`s4t-m1-g{0..3}{a,b,c}`。

## ⑨ 实施中调整（截至起跑）

1. 合并顺序 S3→S2→MERGE-1→S1（文件不重叠）。
2. `samep_encode_donors.py` 不新建，供体改为变体局首推即停 + `--record-motion` 在线录制。
3. FIX-1：scripts/dataset/wan 四份恢复 v7 版（`ADJUST_REVIEW=PASS scale=same`）；本分支 v10 离线建库与 stage2/timepool sidecar 暂不可用。
4. VideoRepick 剔除 8 个 benchmark 报错 seed。
5. aspen 用本地 checkpoint 副本与本地 wan venv（turbo NFS 读取超时）。
6. 首次链式起跑因启动脚本引号转义错误（阶段名被提前展开为空）12 个通道空转退出，按清单停会话、日志移入 `failed-quote/` 后用独立链式脚本重起。
7. GL 每卡 3 worker、fraction 0.27（冒烟 0.40 单 worker 预占 18.4 GiB；按吞吐估算需 3 worker），OOM 则退回 2。
8. GL 每卡 3 worker 时第三个 worker OOM（每 worker 策略 12.3 GiB + sidecar 2.9 GiB），退回每卡 2 worker；c 通道队列排到 a 通道之后，spec 不丢。
9. **GL 作业步结束把 GPU 重置回独占模式**（11-10 已知）：c 通道被停、冒烟作业步结束、用 srun 查 nvidia-smi 后，同占位其他作业步第二局起 Vulkan `ErrorInitializationFailed` / sidecar `CUDA-capable device(s) is/are busy or unavailable`。处置：停掉全部 GL 通道（按确切名，日志移入 `failed-cmode/`），`sp_swap_prep requeue` 重排 n/vrec/noise 未完成 spec，改为**每占位一个长 srun 作业步**（`rs/motion/chains/hold_runner.sh`：a、b 并行，a 结束后接 c；启动器 `rs/motion/gl_holds.sh`，tmux `s4t-h{0..3}-chain1`），此后不再对占位起其他 srun。之后无新增 Vulkan/busy 错误。
10. aspen spec 的 `motion_prov_relax` 加 `driver`（aspen 驱动 570.195.03 vs checkpoint 记录 595.71.05；用户「暂时不考虑不同硬件」），只改 aspen spec，GL 不变；该字段不进运行身份。
11. 阶段顺序：噪声对照队列暂存（`noise-*.sh.hold`），第一链只跑 n → vrec；第二链 samep → noise → samepcore（`chains/chain2-*.sh`），让干预尽早开始。
12. 显存比例 0.22 在 aspen 1 号卡实测 1 局 xhard 无 OOM 后，GL 改为每卡 3 通道、0.22（GL spec 与队列同步改，`xla_mem_fraction` 不进运行身份），停掉 4 个长作业步、`requeue` n/vrec 后用 `chains/hold_runner3.sh`（a、b、c 并行）重起。
13. 追加 4 个 GL 占位的请求被权限拦截（超出计划批准的 4 个），未提交，留待用户早上决定。
14. MotionMem 不干预 RouteStick（本局池 60 局）：easy 8/8、medium 12/16、hard 10/16、xhard 7/20，合计 37/60；N 可达干预位置 291/361。
15. 变体录制慢（约 100–130 s/段，每 chunk 重载模型），落盘 882/941 时（美东 04:35）即生成供体，未等最后一个 worker 与失败 spec 补录：`SAMEP_VREC entries=941 records=882 missing=59`；`DONOR_ENC=FAIL windows=1029 bad=456 maxabs=0.03125`（原样重建局录制编码 vs 自然局在线编码，差值在 bf16 量级，按 R9 记录不停）；`SP_SAMEP_GROUPS S_swing=265 S_side=198 S_both=201 Z=285 no_record=191 excluded=1`；`DONORS_SAMEP=PASS runs=949 specs=68`。
16. 降级（计划 §8 顺序）：Frozen Wan 本轮不跑；MotionMem 的 core 组未生成（`--no-core`）。
17. aspen 名下 20 个干预 spec 改写为 GL 版（`*.gl.json`：换运行树/解释器/checkpoint/v1_store/显存比例，run_root 不变），全部 68 个干预 spec 重排到 GL 12 条队列（`queues/samepX-*.sh`：只翻绕向与置零在前，按 j 升序），第二链 `chains/chain2X-*.sh`（samepX → noise）。aspen 两卡改跑其 4 个噪声 spec（约 78 局）。GL 的 g0a、g0b 噪声 spec 排在 samepX 之后。
18. 噪声对照阶段原 `.hold` 暂存后恢复；GL 占位 2 最后一个 worker 的变体录制在供体生成后停掉（不再使用）。
19. 08:45 停发新局：干预 923/949、噪声 94/118 时按清单停掉 GL `s4t-h0-chain2X`、`s4t-h1-chain2X`（aspen 通道已自行结束），先后 scancel 66195005、66195004（各自完工后提前释放）与 66195002、66195003；GL 只剩训练占位 65943150、65943151（未动）。
