# eval-testx-1017 — 起跑记录

**评估在 GreatLakes（spgpu，单卡 A40）执行，编排在 sled-aspen 与 GL 登录节点 `gl-login2`。** 按 AGENTS 第 12 条留档，本文件是第一段（「版本与代码状态」「启动与配置还原」与已完成的冒烟实测）；正式 30 单元尚未起跑，`result.md` 与 `records/` 在评估完成后补写。时间一律美东，日志原样 UTC 时间戳括注。

权威计划：benchmark 仓库 `hongzefu/robomme_benchmark_MotionJEPA` 分支 `newtask-v3-MotionJepa1006` 的 `1017-test-xhard5-12per-plan.html`（2.153 版）；本分支 `v2-motionmem-eval-test1017` 只实现其评估侧，入口短页 [`1017-eval-testx-plan.html`](../../../1017-eval-testx-plan.html)，状态机 [`docs/handoff/current.json`](../../handoff/current.json)。

## 一、目的

在 test 布局（官方 test metadata 追加 xhard 五档 624 条 seed 后）上，用同一份 1200 局清单评三个模型：自训 512 run @60000、自训 512+motion run @60000、官方 FrameSamp modul 79999。每局保存压缩视频与逐步动作日志。

## 二、用户决策原话

1. 计划 T7：「60k 两个 checkpoint 的评估都只需要每个难度评十二集，easy…也只需要十二集」「官方 FrameSamp 也按同一清单 1200 局同意」
2. 计划 T9：「官方的 frame sample modulation 最后再跑，优先跑我要的这两个」「可以先拿十 K 的跑冒烟」
3. 计划 T13：「同意硬件豁免。然后如果超过一百六十，你需要加入一个丢弃的机制，保证能够继续做。」「用 2000」「超出的量全由执行段等距抽稀承担，恰好填满 160 个窗…先沿用」
4. 计划 T22：「保存视频和保存log 都要有。就是输出动作的 log。然后保存的视频不需要保存这个完整版，都是保存一个压缩的视频。然后这个机制你需要异步的来执行，不能够评估完每一个评估完了再开启」
5. 计划 T23：「你要检测一下这个常驻模型会不会导致 memory 生效，就是说 mem 每次 reset 是否都是正常正确的」
6. 2026-10-17：「所有的任务改为在 Great Lakes 上评估，每个都是单卡」「放行 8 个占位 job（各 1 × A40）；其余按你建议」「不要使用一 CPU，任务改为使用四 CPU。」
7. 2026-10-18 开工令（benchmark 侧）：「/data/hongzefu/robomme_benchmark_newtask-v3-MotionJepa1006/1017-test-xhard5-12per-plan.html 开工 有问题问用户 尽可能不要阻塞」
8. 2026-10-18：「1a」（开工审查 F1 裁决）、「2同意现在交」、「同意你再追加四个 job 去并行。」、「10k 100 局冒烟降低到和官方一样的2局 尽快启动2个60k的eval」

## 三、版本与代码状态

| 版本 | commit | 内容 |
|---|---|---|
| commitV11.22 | `0cbecae` | E1 合入（评估侧实现） |
| commitV11.23 | `a5a1c52` | 子模块 gitlink 指向 benchmark `943ddf32`（test 布局 metadata 追加） |
| commitV11.24Beta | `7efafec9` | 官方 ckpt 逐文件 sha 登记 |
| commitV11.25Beta | `52a5f5ba` | `ASSETS_LOCK` 改指新 encoder；第二次冒烟与之后正式评估的锚点 |

## 四、被评对象

| 模型 | checkpoint | 身份核验 | 状态 |
|---|---|---|---|
| 官方 FrameSamp modul | `v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999`（NFS） | zip sha256 前缀 `86ef2b63` = HF `Yinpei/mme_vla_suite` LFS 对象；与 `Yinpei/perceptual-framesamp-modul@c0f565dd` 的 `2bfde48a` 是同一权重的不同打包；`OFFICIAL_CKPT_SHA=PASS files=19` | 已就位 |
| 512 | HF `HongzeFu/robomme-vla-mixed8-200-modul-512-80k-v1` step 60000 | 待拉取后核 | 待拉 |
| 512+motion | HF `HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1` step 10000 | `CKPT_SHA=PASS files=22`（外部清单 sha 前缀 `8d9b04df`）；根文件 `ROOT_FILES_SHA=PASS files=3` | 已拉（只用于冒烟） |
| 512+motion | 同上 step 60000 | 待拉取后核 | 待拉 |

Motion encoder：`wan-mixed8-200-filter2-stop075-b88x8-72ep-a`（`bec423fd`），`ENC_SUMS=PASS`。

## 五、评估清单与边界

- `MANIFEST=PASS rows=1200 cells=100 sha256=20d4fbfb…`：每任务每档按 episode 升序取前 12，三模型共用。
- `SMOKE_SUBSET=PASS rows=2 sha256=bb77d3a0…`：BinFill easy ep0 与 xhard12 ep50。
- `XHARD_METADATA=PASS tasks=14 records=624`。
- `EVAL_ES_BOUND=DONE scope=xhard episodes=624 max_windows=197 over=87 max_step_idx=3173 over4096=0`：只用 624 个成功 h5，`FAILED_` 失败录制排除。超 160 窗的局按 T13 走 `MMEVLA_MOTION_OVERFLOW=resample`。

## 六、启动与配置还原

- **运行参数**：`max_steps` 2000、seed 7、action chunk 20；`PROV_RELAX` 豁免 `gpu_name`／`compute_cap`／`sm_count`／`driver` 四键（vae 与 encoder 两段，其余 sha 与版本照比）；`ONLINE_GPU` 由 `run_shard` 取占位 job 分到的卡；`XHARD_METADATA_DIR` 指向追加后的 test metadata；冒烟加 `RESET_CHECK=1`（同一服务进程内同一局评两次，第二次写 `reset-check/`，对拍逐步动作）。
- **每局产物**：流式 H.264 压缩 mp4（CRF 28、半分辨率，ffmpeg 子进程随帧编码）+ `steps/<Task>_ep<n>.jsonl` 逐步动作日志。
- **GL 资源**：12 个占位 job（`--account=chaijy2 --partition=spgpu --gres=gpu:1 --gpu_cmode=shared --cpus-per-task=4 --mem=24G --time=48:00:00 --wrap='sleep infinity'`），JobID 64233473–64233480（美东 2026-10-18 00:20 全部 RUNNING）与 64233525–64233528（00:22 RUNNING）；清单 `v1-store/logs/hold-jobs-testx.txt` 是 scancel 唯一依据。放行记录见 `greatlakes.md` 末节。
- **worker**：登录节点 `gl-login2` 上每个占位 job 一个 tmux 会话 `xh-testx-pool-<jobid>`，动态领取 manifest 单元，每单元一个 `srun --overlap --exact --cpus-per-task=4` step；日志在 `v1-store/logs/policy-eval/`。
- **正式顺序**：30 个单元按 512 → 512+motion → 官方写入 manifest，允许尾部重叠（T9）。

## 七、冒烟实测

| 冒烟 | 锚点 | job / 节点 | 时间（美东，括注 UTC） | 结果 |
|---|---|---|---|---|
| 官方 79999 | V11.24Beta | 64233474 / gl1505 | 2026-10-18 02:40–02:43（06:40–06:43 UTC） | BinFill easy ep0 success、xhard12 ep50 fail；error 0；`VIDEO_ENCODE`／`ACTION_LOG`／`SHARD_PASS`；`RESET_CLEAN=PASS pairs=1 steps=277 max_abs_diff=0` |
| 512+motion@10k 第一次 | V11.24Beta | — | — | 失败：sidecar 因 `ASSETS_LOCK` 仍指旧 encoder 触发 `SystemExit`。证据保留为 `v1-store/evaluation/testx-512m10k-smoke2/s0.fail-v1124beta` 与日志 `testx-512m10k-smoke2-s0.fail-v1124beta.log`，队列标记移至 `queue/retired/`；修法即 V11.25Beta |
| 512+motion@10k 第二次 | V11.25Beta | 64233473 / gl1500 | 2026-10-18 02:58–03:05（06:58–07:05 UTC） | 同样 easy success／xhard12 fail；error 0；`MOTION_WINDOWS=PASS k_max=78`，downsample 0；`MOTION_PROV_RELAXED` 只涉及 vae／encoder 的 `gpu_name`／`compute_cap`／`sm_count` 三键（A40 对 A100）；`RESET_CLEAN=PASS pairs=1 steps=427 max_abs_diff=0` |

## 八、下一步

两条 run 的 step 60000 写出后：`SUMS_FILE=<训练侧逐文件清单> fetch_ckpt.sh run` 拉取并核 sha → `make_shard_plan` 三 run 各 10 片 → `make_manifest units --stage formal`（512 → 512+motion → 官方）→ 重起 worker；判据 `QUEUE_DONE failed=0`、`EVAL_512_60K`／`EVAL_512M_60K`／`EVAL_OFFICIAL=DONE episodes=1200`、`MANIFEST_SHARED`、`EVAL_TABLE`。
