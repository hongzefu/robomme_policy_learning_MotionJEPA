# orig100 context+motion 59999 步官方 800 局评估启动档案

## ① 目的

评估 AWS 上训练的 context 融合 + motion（`stage2_pool4_split_v1`，每窗 4×256，motion 预算 384，帧路 512，共 896 个 memory 位置）模型的最终 checkpoint step 59999，在官方 16 任务 × 50 局 test 共 800 局上的成功率，与 10-27 被评模型（官方 44.51±0.77）对照。计划为根目录 `1029-orig100-ctx-motion-60k-official-eval-plan.html`。

用户决策原话（按时间）：
1. 「eval是官方800」
2. 「其他都不评 然后跑到半夜结束了就cancel」
3. 「然后改为12卡并行跑」
4. 「59999出现后立刻提交任务 然后开始下载 下载完立刻开始 不用等校验结束」
5. 「生成 runtime 合同；提交 1 份占位，用 5000 跑 2 局冒烟（purpose=validation） 跑完之后就释放」
6. 开工令：「同意开工」（针对计划 `4b9a59a`）

开工前审查：`PLAN_CONSISTENCY_REVIEW=PASS base=4b9a59a findings=5 blocking=0`，五处措辞已在下一 commit 统一。

## ② 版本与代码状态

- 执行代码：`9459859384589b3553148cf53f715c1318f1a533`，benchmark 子模块 `856bc3a189d4172f3f47dbee4424d585f8d78db3`；执行副本 `$E/releases/9459859384589b3553148cf53f715c1318f1a533/`，`RELEASE_SNAPSHOT=PASS files=4673 tree_diff=0`。
- checkpoint：公开 bucket `HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2split-60k-v1` 的 `manifests/step-59999.json`（美东 10-29 22:04 上传），清单 sha256 `ada0653d6b603b5641013314d10f4e5c2ef9e68a35693730d3e67c3735d57fed`，36 个文件共 11,970,445,281 字节。资产目录 `$E/v1-store/assets/vla-o16ctx-59999-ada0653d…/`。下载按 K9 只核字节数（`DOWNLOADED.json`），sha256 由后台 `verify.py 59999` 并行核对，`READY.json` 与 `VLA_VERIFY` 结果写进 `result.md`。
- policy_config `mme_vla_suite_orig16_b128_60k`；encoder `stage2-full144-1025-a` ep105 EMA，sha256 `27daa8320c9a3926…`，与 5000 相同。`motion_provenance.json` sha `d3768ecccaa85cd1…` 与 5000 相同，故沿用 5000 生成的运行合同。

其中 `$E=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval`，`$R=$E/v1-store/evaluation/official-stage2-1026/o16ctx-60k-seed42`，`$L=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/o16ctx-60k`。

## ③ 启动与配置还原

| 产物 | sha256 |
|---|---|
| `$R/inputs/episodes.json`（复制自 10-27） | 文件 `8f1cb76d9dca2ee520b96d64b80230ed2ae1ac7a44bbb4bb74b1c24844a8a453`，内部签封 `991c34960b431f4e6170357dfe428e74e404f5ccdbf8dafddc3a0a380eaeecc6` |
| `$R/contracts/runtime.json` | `e254f115ca3bbaf0797341e7b86c6e5b68b561c05ea934e3ba535fe0423d05bf` |
| `$R/inputs/run-identity.json` | 文件 `9ef7571edfbe14e3e9c73dd4bc5679361ec09a41b12f86cf0ca44d5128d548db` |
| `$R/spec-input.json`（`purpose=production`） | 文件 `5d1603e9980ca0aeb9928cfee8c8b21a092cf97a0213f88ed9fdafc68c77a11a` |
| `run_identity_sha256` | `3ebc1c3f8c3e89099fa02befdad8893838fde8c043a0549f3dced4674ca951f2` |

run 身份字段：schema `simple-run-identity-1027`，`checkpoint_step=59999`，`model_seed=42`，`model_kind=stage2_pool4_concat`，官方 profile、1300 步、超预算 raise，不导出 `MMEVLA_MOTION_OVERFLOW`／`XHARD_METADATA_DIR`。

命令：

```bash
# 下载（tmux eval-o16-fetch60k）
python3 -u $R/fetch.py 59999 --size-only | tee $L/fetch60k.log
# 后台校验（tmux eval-o16-verify60k，与评估并行）
python3 -u $R/verify.py 59999 | tee $L/verify60k.log
# 初始化（在执行副本内）
python -m scripts.evaluation.official_queue initialize --run-root $R/run800 --manifest $R/inputs/episodes.json --spec $R/spec-input.json
# 控制器（tmux eval-o16-controller）
bash $R/run-controller.sh
# 每个 worker（tmux eval-o16-worker-NN）
bash $L/start-worker.sh <JobID> worker-NN
#   = ssh greatlakes srun --jobid=<JobID> --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared \
#       /usr/bin/bash $R/run-worker.sh 9459859384589b3553148cf53f715c1318f1a533 worker-NN
```

## ④ 数据集与划分

官方 test，16 任务 × 50 局 = 800 局，160 块 × 5 局，动态队列。

## ⑤ 关键超参

seed 42，官方 profile，1300 步上限，motion 窗口超 96 即 raise，不重试正常失败。

## ⑥ 硬件与资源

12 份 spgpu 占位，各 1×A40 / 4 CPU / 24 G / 48 h，美东 10-29 22:04 提交：

| worker | JobID | 节点（提交后即时） |
|---|---|---|
| worker-00 | 65048187 | gl1520 |
| worker-01 | 65048188 | gl1520 |
| worker-02 | 65048189 | gl1521 |
| worker-03 | 65048190 | gl1521 |
| worker-04 | 65048191 | gl1521 |
| worker-05 | 65048192 | gl1521 |
| worker-06 | 65048193 | gl1521 |
| worker-07 | 65048194 | gl1521 |
| worker-08 | 65048195 | gl1521 |
| worker-09 | 65048196 | gl1521 |
| worker-10 | 65048197 | gl1510 |
| worker-11 | 65048198 | PENDING（Resources），节点补进 `result.md` |

冒烟（S3）：占位 65037493（gl1520），5000 步跑 `BinFill:0`、`VideoPlaceOrder:18` 两局，`SMOKE2=PASS accepted=2 errors=0`（两局 fail，5000 步属预期），server 回显 readout `stage2_pool4_split_v1`、`num_tokens=4`、`dim=256`、`budget=384`，sidecar 身份校验在 srun 下通过；首推理约 36 s，稳态 152 ms／次。冒烟控制器 SIGTERM 后 `EXIT_CODE=0`，占位已 `CANCELLED`。

本轮 tmux 会话清单：`eval-o16-fetch5k`（已结束）、`eval-o16-watch`（已结束，`STEP59999_SEEN`）、`eval-o16-smoke-controller`（已结束）、`eval-o16-smoke-worker`（已结束）、`eval-o16-fetch60k`、`eval-o16-verify60k`、`eval-o16-controller`、`eval-o16-worker-00` … `eval-o16-worker-11`。

日志：本机 `$L/`（`fetch60k.log`、`verify60k.log`、`prod-worker-NN.log`）；控制器与 server 日志在 `$R/run800/logs/`，每块仿真日志在 `$R/run800/attempts/<attempt>/sim.log`；汇总 `$R/run800/queue/summary.json`。
