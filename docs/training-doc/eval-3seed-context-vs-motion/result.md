# eval-3seed-context-vs-motion 结果留档

## 一、结论先行

**两组权重在本机同硬件、3 个 policy 采样 seed 下表现无可辨别差异；起因表中 28.0% vs 24.0% 的 4pp 领先不成立。**

| 组 | seed 42 | seed 7 | seed 2024 | 均值 | 标准差 |
|---|---|---|---|---|---|
| 官方 perceptual-framesamp-context | 25.0% | 24.5% | 24.0% | **24.5%** | 0.5pp |
| awsprod40k-b128-motion | 24.0% | 25.5% | 23.0% | **24.2%** | 1.3pp |

两组均值相差 **0.3pp**，而 motion 组自身的 seed 标准差就有 **1.3pp**、官方组 0.5pp——组间差远小于组内噪声。
两组的逐集稳定性也几乎相同：200 集中官方翻转 35 集、motion 翻转 32 集。

逐任务（均值 ± 标准差，n=3）：

| 任务 | 官方 context | awsprod40k motion | 差 |
|---|---|---|---|
| ButtonUnmask | 27.3% ± 2.3 | 26.0% ± 2.0 | −1.3pp（重叠） |
| VideoUnmask | 30.7% ± 1.2 | 28.7% ± 1.2 | −2.0pp（接近但未超噪声） |
| ButtonUnmaskSwap | 20.0% ± 2.0 | 17.3% ± 4.6 | −2.7pp（motion 波动最大，重叠） |
| **VideoUnmaskSwap** | **20.0% ± 0.0** | **24.7% ± 1.2** | **+4.7pp（唯一稳定的差异）** |

**唯一值得注意的单任务差异是 VideoUnmaskSwap**：官方三个 seed 全是 20.0%（零方差），motion 三个 seed
24.0% / 24.0% / 26.0%，稳定高出约 4.7pp。这是本轮数据里 motion 唯一站得住的优势，但它是四个任务里
挑出来的一个（多重比较）、且只有 3 个 seed，应作为**待验证线索**而非结论。

反过来，起因表中 motion 领先最多的 ButtonUnmaskSwap（34% vs 8%，领先 26pp），在本机三 seed 下变成
motion 17.3% vs 官方 20.0%——**方向完全反转**。

## 二、起因表为什么没复现：差异集中在一个不稳定的任务

起因表（均在环境 B 的 A100 上产出）与本机三 seed / 一 seed 的逐任务对照：

| 任务 | 官方 A100 | 官方本机(3 seed 均值) | motion A100 | motion 本机(3 seed 均值) |
|---|---|---|---|---|
| VideoUnmask | 34% | 30.7% | 28% | 28.7% |
| ButtonUnmask | 34% | 27.3% | 26% | 26.0% |
| VideoUnmaskSwap | 20% | 20.0% | 24% | 24.7% |
| **ButtonUnmaskSwap** | **8%** | **20.0%** | **34%** | **17.3%** |
| 四任务均值 | 24.0% | 24.5% | 28.0% | 24.2% |

起因表里 motion 的 4pp 总优势，几乎全部来自 ButtonUnmaskSwap 一项的 26pp 领先（34% vs 8%）。
换到本机后，该任务两组朝相反方向移动——官方 8%→20.0%（+12pp），motion 34%→17.3%（−16.7pp）——优势随之消失并反转。
其余三个任务的本机数字与 A100 都相当接近（差 0.7–7.3pp），可见问题集中在这一个任务上。

而 ButtonUnmaskSwap 恰恰是四个任务里**最不稳定**的一个，且它也是 motion 组标准差最大的任务（±4.6pp）。
官方组 200 集在三个 seed 下的逐集稳定性：

| 任务 | 三 seed 全成 | 三 seed 全败 | 有翻转 | 翻转率 |
|---|---|---|---|---|
| ButtonUnmaskSwap | 3 | 32 | **15** | **30.0%** |
| ButtonUnmask | 9 | 30 | 11 | 22.0% |
| VideoUnmask | 13 | 32 | 5 | 10.0% |
| VideoUnmaskSwap | 8 | 38 | 4 | 8.0% |
| **合计** | 33 | 132 | **35** | **17.5%** |

motion 组同口径为全成 34 / 全败 134 / 翻转 **32**（16.0%），与官方组几乎一致——两组对采样噪声的敏感度相当。

即：只换 policy 采样噪声，ButtonUnmaskSwap 有 30% 的 episode 会翻转结果。
一个逐集这么不稳的任务，其单轮 50 集成功率不足以支撑「谁更好」的判断——这与起因留档
`eval-official-framesamp-context/result.md` 盲区清单里「任何『谁更好』的结论都需要多 seed」的判断一致。

## 三、跨硬件不可精确复现（同 seed 同权重同环境）

本轮与起因表的一切逐任务差异都不是 bug。每集的物理初始布局固化在 benchmark 元数据里
（`env_runner.py` 硬编码 `BenchmarkEnvBuilder(dataset="test")`，`episode_config_resolver` 按
`records[].seed` 建环境），policy 采样 seed 也相同，唯一变量是 GPU：
环境 B 的 A100-SXM4-80GB（compute_cap 8.0、108 SM）vs 本机 RTX 6000 Ada（8.9、142 SM）。
浮点级差异经 1300 步轨迹放大即可翻转个别 episode。实测幅度：

| 组 / 任务 | A100 | 本机同 seed42 | 差 |
|---|---|---|---|
| 官方 ButtonUnmaskSwap | 4/50 | 9/50 | +5 集（+10pp） |
| motion ButtonUnmaskSwap | 17/50 | 10/50（seed42）/ 三 seed 均值 17.3% | −7 集（−14pp，三 seed 均值 −16.7pp） |
| 官方四任务均值 | 24.0% | 25.0% | +1.0pp |
| motion 四任务均值 | 28.0% | 24.0%（seed42）/ 24.2%（三 seed 均值） | −4.0pp / −3.8pp |

**总均值层面跨硬件影响很小（官方 +1.0pp），单任务层面可达 ±14pp。**
因此本轮所有数字只与本轮内部互比有效；与起因表的逐任务数字不得直接混比（AGENTS 13）。

## 四、运行环境与配置

**环境 A，但 GPU 是第三套硬件**：仓库根 `/data/hongzefu/...`、`/nfs/turbo` 可见、`/scratch/hongze` 不存在、
`~/.ssh/config` 存在——五项路径判据均指向环境 A；GPU 实测为 **2 × NVIDIA RTX 6000 Ada Generation
（46,068 MiB/卡）**，既非判据表中环境 A 的 A40 也非环境 B 的 8×A100。已交用户裁决，
拍板按环境 A 走、GPU 一项按实测记。CPU 32 核 / 内存 377 GB。

| | 官方 context | awsprod40k motion |
|---|---|---|
| CKPT | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | `v1-store/models/awsprod40k-b128-motion/39999` |
| 加载分支 | `history_config.txt` 旧兼容（非严格恢复） | `history_config.resolved.yaml` 快照（严格恢复）+ motion sidecar |
| 参数树自证 | `PARAM_TREE_EXACT=PASS n_model=55 n_ckpt=55 missing=0 extra=0 shape_mismatch=0` | 同上 `n_model=59 n_ckpt=59`（59 = 55 + 两个 motion 模块 4 叶） |
| 权重校验 | zip sha256 本地/HF LFS/钉死值三方一致 `387d4bd5…1a8ae8` | 27 文件 / 11,584,776,670 B 与 dry-run 预期精确一致；`history_config.resolved.sha256` = `94d8660…927fb` |
| 分片 | `MODE=stride WORKERS=4 GPU_LIST=0,0,1,1`，每 worker 13/13/12/12 集 | `WORKERS=2 GPU_LIST=0,1`，每 worker 25/25 集 |
| `POLICY_MEM_FRACTION` | 0.40 | 0.55 |
| 实测显存 | GPU0 39,614 MiB（2 policy 18,638×2 + 2 仿真 743/507 + 他人常驻 996）、GPU1 38,556 MiB | GPU0 30,557 MiB（policy 25,456 + sidecar 3,286 + 仿真 743 + 他人 996）、GPU1 29,518 MiB |

均按任务分 4 批、每批独立 `RUN_NAME`（`eval.py` 跑完集号区间即写 `log.json`，共用 RUN_NAME 会让后续批次空转退出），
以规避「单个 `eval.py` 进程建满 27 个仿真环境后第 28 次 `make_env` 必崩」的上限。

## 五、耗时实测（本机 2 × RTX 6000 Ada，与 A100 / A40 数字不得混比）

| 轮次 | 起止 | 墙钟 | 四批分别 |
|---|---|---|---|
| 官方 seed 42 | 03:46:01 → 04:04:43 | **18 分 42 秒** | 4:10 / 3:51 / 6:30 / 4:11 |
| 官方 seed 7 | 04:05:13 → 04:24:35 | **19 分 22 秒** | 4:51 / 3:50 / 6:11 / 4:30 |
| motion seed 42 | 05:33:39 → 06:25:42 | **52 分 3 秒** | 14:11 / 8:30 / 16:51 / 12:31 |
| 官方 seed 2024 | 08:02:07 → 08:21:49 | **19 分 42 秒** | 4:50 / 3:51 / 6:30 / 4:31 |
| motion seed 7 | 09:29:50 → 10:20:53 | **51 分 3 秒** | 13:11 / 8:30 / 17:11 / 12:11 |
| motion seed 2024 | 10:20:53 → 11:04:55 | **44 分 2 秒** | 12:10 / 8:31 / 13:50 / 9:31 |

单次推理时序（官方组 w0，2 policy 同卡）：`add_buffer` median **26 ms**、`infer` median **70 ms**
（起因表 A100 那轮为 37 ms / 69 ms，推理侧本机不慢）。
motion sidecar 单窗 median **864 ms**（n=403），与本机 `motion-t3-open/result.md` 的「单 sidecar 独占 0.88 s/窗」吻合。

**GPU 实际工作约 3 小时 45 分**：官方三轮 57:46（18:42 + 19:22 + 19:42）、
motion 三轮 2:27:08（52:03 + 51:03 + 44:02）、前置下载与校验约 20 分。
官方组单 seed 稳定在 19 分上下，motion 组 44–52 分（sidecar 每窗约 0.86 s 是主导项）。
会话总墙钟跨度 03:23 → 11:05，差额为两次决策等待（provenance 闸约 1 小时、4 小时线用途约 1.5 小时）、
端口事故处置，以及 motion 后两个 seed 是在首轮交付后按用户追加要求补跑的。

## 六、过程中的两个事故

### 6.1 端口被本机其他服务占用，导致静默跑 0 集

阶段 1 首批 w1（端口 8042）、w3（8044）起跑 12 秒即 `EVAL_RC=0 / EXIT_CODE=0` 退出、`TIMING n=0`。
根因：`eval_shard.sh` 等就绪的循环只探「能否连上 PORT」，而本机 8042/8044/8045/8047-8050 上有用户的
长期服务在监听（tmux 会话 `site8042`/`site8044`/`ensite8048`/`ensite8049`/`ensite8050`），
两片起跑同一秒即「端口就绪」，`eval.py` 连过去拿到非 policy 响应后报
`did not receive a valid HTTP response` → `API calling error, aborting...`，
**而这条 abort 路径仍以退出码 0 结束**。

修复（commit `baa93fa` 与其后的 stderr 修补）：
1. `eval_shard.sh` 起 server 前加端口占用守卫，被占即 `exit 1` 并明确报错（实拦验证：退出码 1 +
   `错误: 端口 8042 起跑前已被占用（本机有其他服务在监听），换 PORT_BASE 重试`）。
2. `eval_seed_sweep.sh` 默认 `PORT_BASE` 由 8041 改为 **9200**（实测 9200-9264 全空），
   起批前逐个探本批端口，批后除 `EXIT_CODE=0` 外**再核 `progress.json` 实评集数必须凑满 50**。
   「退出码为 0 不等于评了集」是本次事故的核心教训。

期间还踩到一个 bash 陷阱：守卫写成 `exec 3>&- 2>/dev/null` 时，`exec` 不带命令的重定向会
**永久吞掉本 shell 后续的 stderr**，导致守卫只剩一个无从解释的非零退出码。已改为裸 `exec 3>&-`。

处置：w0（6 集）、w2（7 集）为真实结果予以保留（未写 `log.json`，重跑即续评）；
w1/w3 分片目录只含 abort 残留的一条 `"error"`，整目录删除后重评。

### 6.2 motion sidecar 的物理 GPU 指纹硬闸

阶段 2 首批两片起跑即失败于 `motion_client.py::check_provenance`：sidecar 自报的 GPU 指纹与离线 motion 库
（建于环境 B 的 A100）不同源。比对共 22 项，**实测差异只有硬件 4 项 × 2 段 = 8 条**：
`gpu_name`/`compute_cap`/`sm_count`/`driver`；其余 18 项全部一致——权重指纹
（`vae_state_sha256`/`checkpoint_sha256`/`module_sha256`/`encoder_src_sha256`）、
数值口径（`vae_dtype`/`precision`/`tf32`/`amp`/`batch`/`latent_mode`/`motion_dims`）、
软件栈（`torch`/`cuda`/`cudnn`/`diffusers`/`cublas_pkg`/`cudnn_pkg`）。

即同一套模型、同一套数值语义，仅物理 GPU 不同；而本机不可能有 A100，该闸在此为永久性阻塞。
经用户 2026-09-06 拍板，commit `commitV5.8` 新增 `PROV_HW_KEYS`，把这 4 项降级为 stderr 告警
（`MOTION_PROV_HW_MISMATCH`，落进 `<LOG_PREFIX>-w<k>.server.log`），其余 18 项维持硬拦。

降级未影响数值行为的旁证：sidecar 单窗 median 864 ms 与本机既有留档的 0.88 s/窗 吻合，
且 sidecar 自报 `vae=9980d252230c265c… ckpt=bae960373041629e…` 与离线库权重指纹一致（这部分仍是硬核过的）。

## 七、盲区与下一步

1. **每组 3 个 seed，样本量仍小**。以 n=3 判断组间差异，只能得出「差异未超噪声」这类否定性结论，
   无法为「两者等价」给出强证据。VideoUnmaskSwap 上 motion 稳定高 4.7pp 这条线索，需要更多 seed
   （或更多 episode）才能确认——它是从四个任务里挑出来的，存在多重比较问题。
2. **单 checkpoint**：官方固定 79999、motion 固定 39999，未做 checkpoint 维度的对照。
3. **环境布局固定**：3 个 seed 只改 policy 采样噪声，200 个环境实例始终是 test split 的同一批
   （每集 seed 固化在 benchmark 元数据）。换环境布局需改 split 或绕过元数据 seed，不在本轮范围。
4. **跨硬件**：本轮所有数字产自 2 × RTX 6000 Ada，与起因表的 A100 数字只可在总均值层面粗比，
   逐任务不可混比（第三节实测单任务可差 ±14pp）。

## 八、本轮 tmux 会话清单（AGENTS 7）

全部由本轮起、且均已正常退出；清理一律用 `tmux kill-session -t <确切会话名>`，删前删后各查一次 `tmux ls`。
用户既有的 `unisite`、`v3site`、`site8042`、`site8044`、`ensite8048`、`ensite8049`、`ensite8050` 七个会话全程未动。

| 会话名 | 用途 |
|---|---|
| `dl-official-ctx` / `dl-awsprod40k` | 两套权重下载 |
| `chk-motion-tree` / `prep-official` | 参数树自证、官方权重解压 + 自证 |
| `sweep-official-s42` / `sweep-official-s7` / `sweep-official-s2024` / `sweep-motion-s42` | 前四轮批次 driver |
| `sweep-motion-rest` | motion seed 7 与 2024 的串行 driver（补跑轮） |
| `evctx-s{42,7,2024}-{bu,vu,bus,vus}-w{0..3}` | 官方组分片（48 个） |
| `evmot-s{42,7,2024}-{bu,vu,bus,vus}-w{0,1}` | motion 组分片（24 个） |

事故中被 kill 的三个：`sweep-official-s42`、`evctx-s42-bu-w0`、`evctx-s42-bu-w2`（端口事故后重跑）；
`sweep-motion-s42` 的首次尝试因 provenance 闸自行失败退出。

## 九、产物

| 产物 | 路径 |
|---|---|
| 汇总表 | `docs/training-doc/eval-3seed-context-vs-motion/records/summary.txt` |
| 逐集三 seed 对照 | `docs/training-doc/eval-3seed-context-vs-motion/records/per_episode_by_seed.json` |
| 参数树自证 | 同目录 `param_tree_official.json` / `param_tree_motion.json` |
| 分片结果 | `v1-store/evaluation/{official-ctx,awsprod40k-motion}-s<seed>-<Task>-w<k>/ckpt<id>/seed<seed>/` |
| 日志 | `v1-store/logs/{sweep-*,evctx-*,evmot-*,dl-*}.log` |

汇总判定行：**`SEED_AGG=DONE groups=2 seeds=3 tasks=4 episodes=1200/1200 errors=0`**——
两组 × 3 seed × 4 任务 × 50 集全部跑满，零 error、零缺集。
