# hf-export-v2-1600ep-trainset-20260920-v1 — 起跑

把这一版训练库 `v1-store/datasets/4task-v2-1600ep-604f16da` 整体导出到**公开** bucket
`HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1`，目标是异地 8×A100 机器
`hf sync` 一次 + `fetch_assets.py fetch` 一次即可起跑 512 modulation 与
512+motion modulation 两条训练。结果见同目录 `result.md`（跑完回填）。

## 为什么做这一轮

盘点发现：这一版数据集训出的三个模型（阶段一 encoder、60k、80k）**已全部在 HF**，
逐文件 diff 过、本机无一遗漏；但**数据集侧只有链首的原始 h5 在 HF**
（`HongzeFu/robomme-4task-h5-20260912-v2`），训练真正读的表一件都没上。后果有三：
异地无法开训；`motion80k_bucket_README.md` 把「数据本体」指向
`robomme-4task-motion-full1600-20260914-v1`（那是阶段一 encoder 自己的 Wan chunk latent，
没有 framesamp/motion store），照它走复现不了；README 里写的两个 store 指纹
在 HF 上没有对应本体可对拍。

## 起跑状态

| 项 | 值 |
|---|---|
| 起跑 commit | `eb8c318db37f92841754f6ed77d9cd0311bbd908`（clean HEAD，`git status --porcelain` 为空） |
| 分支 | `v2-motionmem` |
| 环境 | 环境 B（AWS 单机 8×A100-80GB，本地 NVMe RAID `/dev/md0`） |
| tmux | `hfup-v2-trainset` |
| 日志 | `v1-store/logs/hfup-v2-trainset.log` |
| driver | `scripts/dataset/hf_export/run_v2_1600ep_dataset_export.sh` |
| 打包器 | `scripts/dataset/hf_export/pack_and_hash.py --layout v2-1600ep-v1`（本轮新增 layout） |
| 裁决账本 | `scripts/dataset/hf_export/hygiene_allowlist_v2_1600ep.json`（31 条） |
| 落点 | `v1-store/exports/hf-v2-1600ep-trainset/{stage,verify,tmp,logs}` |

起跑命令：

```bash
tmux new-session -d -s hfup-v2-trainset \
  'cd /scratch/hongze/robomme_policy_learning_MotionJEPA && \
   PYTHONUNBUFFERED=1 bash -o pipefail -c \
   "bash scripts/dataset/hf_export/run_v2_1600ep_dataset_export.sh 2>&1; \
    echo EXIT_CODE=\$?" | tee v1-store/logs/hfup-v2-trainset.log'
```

## 源与内容清单

源根 `$LIB = v1-store/datasets/4task-v2-1600ep-604f16da`，四任务
**BinFill / RouteStick / VideoRepick / VideoUnmaskSwap**（口径以
`meta/episode_manifest.json` 的 `canonical_order` 为准），1600 集 / 1,192,918 帧 /
605,611 exec 样本。

| 源 | 实测 | bucket 落点 | 处置 | 训练必需 |
|---|---:|---|---|---|
| `framesamp-8x8/`（37 件，32 个 part 9.78–10.21 GB） | 313,226,571,153 B | `framesamp-8x8/` | 直传 | ✅ |
| `source/data/*.pkl`（605,611 件，均 395 KB） | 225 GiB | `source/data_tars/data-000xx.tar`（4096 件/片 → 148 片） | 打 tar | ✅ |
| `source/features/` 的 8 个抽样点 npy | 4.8 MB | `source/features/episode_*/token_emb_*.npy` | 直传 | ✅ |
| `motion/`（5 件） | 221,210,854 B | `motion/` | 直传 | ✅（80k） |
| `meta/`（2 件） | 479,894 B | `meta/` | 直传 | ✅ |
| `norm_stats.json` | 2,143 B | `assets/robomme/norm_stats.json` | 直传 | ✅ |
| `source/meta/`（5 件） | 44 KB | `source/meta/` | 直传 | — |
| `framesamp/`（4×4，37 件） | 78,349,610,344 B | `framesamp/` | 直传 | ❌ |
| `oracle/wan-mj/`（6,420 件） | ~40 GiB | `oracle_tars/` | 打 tar | ❌ |
| `wan-latents/`（9,601 件） | ~39 GiB | `wan-latents_tars/` | 打 tar | ❌ |
| `motion-tokens/`（9,601 件，均 26.9 KB） | 258,639,100 B | `motion-tokens_tars/` | 打 tar | ❌ |
| `logs/`（35 件） | 2 MB | — | **不传**（allowlist exclude） | — |
| `source/features/` 其余 | 674 GiB | — | **不传** | — |

另有 `README.md`、`runner/{restore.sh,train-60k.sh,train-80k.sh,_train_common.sh}`、
`checksums/`、`manifest/upload_manifest.json`、`SHA256SUMS.pre.txt` 由链路自己生成。
合计约 **669 GiB**、约 330 个 bucket 对象。

打 tar 与直传的判据只有一条（沿用 400ep）：**平均文件大小是否小到会让上传退化成逐文件
HTTP 往返，以 ~50 MB 为界**。framesamp 两库的 64 个大 part bin 直传，还因为打 tar 会毁掉
「下完即训练可直读 store」这个性质。

## 用户本轮拍板的五个口径

1. bucket **公开**，**单个** bucket（不拆 train / aux）。
2. 范围 = 训练必需 + 4×4 表 / oracle / wan-latents / motion-tokens。
3. 异地仓库路径就放 `/scratch/hongze/robomme_policy_learning_MotionJEPA`。
4. 外部权重不进 bucket，异地用 `fetch_assets.py` 拉（只需 `pi05_base` 与
   `paligemma_tokenizer`，均 `gs://` 匿名可拉）。
5. 阶段 0 删 `v1-store/exports/*/verify/` 腾盘（约 610 GiB，均为已验收 PASS 的回读副本）。

## 一条必须记下的认知修正

`framesamp-8x8` **只装 memory 路**的 SigLIP 特征。当前观测的
`image / wrist_image / state / actions / prompt` 仍然逐样本从 pkl 读
（`src/mme_vla_suite/training/framesamp_dataset.py` 的 `__getitem__`），所以
**`source/data/` 的 605,611 个 pkl 是硬依赖**，最小可训包不是 292 GiB 而是约 518 GiB。
另有 `framesamp_store.run_fast_checks` 的源库抽样复验按 `os.getpid()` 轮转抽
`source_spot_sha256.entries`（16 条），其中 8 条落在 `source/features/` 下，
缺一个异地就会在「源库抽样文件缺失」处拒跑。本链路带的正是这 8 个，
名单由 store_meta **现读**、未硬编码。

## 盘容量帐（按顺序做，会卡脖子）

起跑前 `/scratch` 可用 **600,831,254,528 B（559.6 GiB）**，已用 93%。

- 阶段 0 删旧 `verify/` → 预期回到约 1,170 GiB；driver 硬断言可用 ≥ 974 GiB，否则停。
- 阶段 3 新增 tar 实体 ≈ 305 GiB（`source/data` 225 + oracle 40 + wan-latents 39 + motion-tokens 0.25）。
  直传件走硬链接，额外占盘 0（代价是阶段 9 必须重算源侧 sha256 断言原件未被改坏）。
- 阶段 7 回读 ≈ 669 GiB。峰值 305 + 669 = **974 GiB**。
- 阶段 11 开头删 stage 内 tar 实体（释放 305 GiB），给回读副本解包让位。

## 阶段表

| 阶段 | 做什么 |
|---|---|
| -1 | `SMOKE=1` 本地彩排，不碰网络写（**已于起跑前跑过并通过**） |
| 0 | 腾盘 + 落点/源/`pack.lock`/uvx 断言 + `hf auth whoami` |
| 1 | `scan_hygiene --scope source`（★ 闸门：不过不建库。公开 bucket 建出来就没有后悔窗口） |
| 2 | `hf buckets create` + 显式 `settings --public` + 空库断言（`uploaded.marker` 豁免续跑） |
| 3 | `pack_and_hash --layout v2-1600ep-v1`，`pack_progress.jsonl` 断点续跑 |
| 4 | README + runner + 符号链接/`pack.lock` 断言 + stage 侧体检 + 上传前 sha256 |
| 5 | `plan_upload_batches` 落盘装箱 → 逐批上传，每批 8 次重试，根文件走 `hf buckets cp` |
| 6 | 计数层核对（REST，异步滞后则 120 s 后复查；不单独作判据） |
| 7 | 全量回读 669 GiB（`hf sync`，不 `rm -rf`，增量可续） |
| 8 | 回读侧**独立重算** sha256 → 与上传前逐行 `diff` + `sha256sum -c --strict` + tar 成员抽样 |
| 9 | 源侧第三遍 sha256，证明硬链接未改动原件 |
| 10 | 公开性确认：REST `private` + `env -u HF_TOKEN` 匿名拉 README |
| 11 | 冷启动彩排：**只用回读副本**跑 20 步，两条 config 各一次 |

## 验收判定行（起跑前写死，缺一条不算过）

```
HF_WHOAMI=HongzeFu
FREED_BYTES=<n>   AVAIL_BYTES=<n>
HYGIENE=PASS scanned=<n> hits=<n> ungated=0            ← 阶段 1，源侧
BUCKET_CREATED=HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1 PRIVATE=False
BUCKET_FILES_AT_START=0 BUCKET_BYTES_AT_START=0
ANCHOR_OK=5
PACK_DONE shards=<n> plain=<n> members=<n> bytes=<n>
HYGIENE=PASS scanned=<n> hits=<n> ungated=0            ← 阶段 4，stage 侧
LOCAL_FILES=<n> LOCAL_BYTES=<n>
PRE_LINES=<n>
PLAN_BATCHES=<n> PLAN_FILES=<n> PLAN_BYTES=<n> MAX_BATCH_BYTES=<n>
UPLOAD_DONE batches=<n>
BUCKET_FILES=<n> BUCKET_BYTES=<n>                      ← 与 LOCAL_* 逐字相等
SHA256_PRE_POST_DIFF=0                                 ★ 核心判据
MEMBERS_SPOT=PASS tars=6 members=<n> mismatches=0
RESULT=PASS
SRC_UNCHANGED=OK
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK                                    ★ env -u HF_TOKEN 匿名拉 README
COLD_START=PASS steps=20
EXIT_CODE=0
```

## 起跑前已完成的验证

| 项 | 结果 |
|---|---|
| `pack_and_hash --layout v2-1600ep-v1` smoke | `ANCHOR_OK=5`、`MEMBERS_SPOT=PASS tars=2 members=8192 mismatches=0`、`SMOKE_BYTEWISE checked=20 mismatches=0` |
| `pack_and_hash --layout motion400ep-v1` 回归 smoke | `ANCHOR_OK=3`，data + features 两片照出，既有链路行为未变 |
| `scan_hygiene` 源侧 | `HYGIENE=PASS scanned=16036 hits=25745 ungated=0` |
| `scan_hygiene` stage 侧 | `HYGIENE=PASS scanned=21 hits=79 ungated=0` |
| 四个 shell 脚本 | `bash -n` 全过 |
| ruff | 新增代码零新增告警（10 条均在改动前既有行 ≤327） |

## 已知风险与退路

1. **单片 10.2 GB 顶在已知超时阈值上**。400ep 实测：`1.62 GB×10` 过、`2 GB×5` 稳、
   `2 GB×10` 挂，失败点在**批内字节量**而非带宽（超时都发生在数据传完之后的
   `new_upload_commit`）。`framesamp-8x8` 的 32 个 part 单片 9.78–10.21 GB，装箱后一片一批，
   是本轮**唯一的新变量**。退路：每批已有 8 次重试；若某片连挂 8 次，改用
   `hf buckets cp` 单文件传该片。
2. **本库任何数据字节都不修改。** 公开前体检的处置只有「排除」与「披露」两种：
   `episode_manifest.json` 有自校验 sha256，两个 framesamp store 与 motion store 的
   `store_meta.json` 又被两条 run 的 `motion_provenance.json` 与两个 ckpt bucket 的
   README 引用；为掩掉一个路径或主机名而改字节，会让公开出去的库再也过不了自己的加载闸。
   逐条裁决与理由见 allowlist，汇总披露写在 bucket README 的「已知事项」。
3. `verify_model_repo.py` 写死 `private is not True → return 1`，那是给 model repo 用的；
   本轮目标公开、断言方向相反，**不复用**。
4. `verify_download.py` 的 `DEFAULT_ROOT` 是环境 A 老路径且硬编码 `sha256-packed.txt`，
   本轮走内联 `sha256sum -c --strict`，**不调它**。
5. 删旧 `verify/` 不可逆（用户已拍板）。它们是已验收 PASS 的回读副本，
   400ep 的 `result.md` 明写「验收后可整目录删」；真要复查可从对应 bucket 重下。
6. **绝对禁止 `tmux kill-server`**（AGENTS 第 7 条红线）；清理只用
   `tmux kill-session -t hfup-v2-trainset`。本轮只起 `hfup-v2-trainset` 一个会话，
   用户自有会话一律不动。
