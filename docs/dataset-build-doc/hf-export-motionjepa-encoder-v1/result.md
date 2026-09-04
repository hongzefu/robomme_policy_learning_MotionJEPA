# result：MotionJEPA encoder+decoder 上传 HF + 五资产锁落地

**终判：上传与验收全绿，资产锁已在链路上生效。**
`HF_META=PASS files=6 lfs_mismatch=0 blob_mismatch=0`、`HF_PRIVATE=True`、回读 `RESULT=PASS`、`EXIT_CODE=0`；
`ASSETS=PASS assets=6 mismatches=0`（full 档 10.2 s）；`BASELINE_ENV=PASS`（旧基线未被破坏）；
`DIFF_SCOPE=PASS touched_train_path=0`（AGENTS 第 18 条不触发的机器证据）。

执行日期 2026-09-03 → 09-04（EDT）。起跑 clean HEAD `1ebab97`，收官 HEAD 见本文件所在 commit。

## 一、HF 侧最终态

| 项 | 值 |
|---|---|
| repo | `HongzeFu/MotionJEPA`（model，**private**，非 gated） |
| commit | `749224110f91c82f5fc5d3007281d07a5d5c944e`（main） |
| tag | `wan-v8-filter10-72ep-a-e72` → tag 对象 `c77c987551e8901d7cfbbf38a5d1e6c6ba7c2fc1` |
| 文件 | 6 个（含 Hub 自带的 `.gitattributes`） |

```
.gitattributes                                    1,519 B  blob a6344aac…（Hub 自带，未改）
README.md                                        22,610 B  blob 680cfe9a…（= 仓库内 model-card.md 逐字节副本）
SHA256SUMS.txt                                      213 B  blob 2b5cd8fd…
wan-v8-filter10-72ep-a/SHA256SUMS.src-vs-copy.txt   704 B  blob e1e71c2c…
wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt 954,853,147 B  lfs.sha256 bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a
wan-v8-filter10-72ep-a/config.yaml                1,927 B  blob 47f601fc…
```

ckpt 的 `lfs.sha256` 与本机源文件 sha256 **逐字符相同**；`config.yaml` 等非 LFS 文件用 git blob sha1
（`sha1("blob <len>\0"+content)`）比对，同样全等——这条顺带证明了服务端没做任何换行/编码转换。

## 二、验收三层（判定行原文）

**L1 元数据（不下载，内容寻址）**——`scripts/dataset/hf_export/verify_model_repo.py`：

```
HF_META=PASS files=6 lfs_mismatch=0 blob_mismatch=0
HF_PRIVATE=True commit=749224110f91c82f5fc5d3007281d07a5d5c944e
```

**L2 回读闭环**——`HF_HUB_DISABLE_XET=1 hf download --revision <tag> --local-dir <verify> --force-download`
后在回读目录里：

```
wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt: OK
wan-v8-filter10-72ep-a/config.yaml: OK
RESULT=PASS
```

外加 `cmp` 两次无输出（回读的 `README.md` == 仓库内 `model-card.md`；`SHA256SUMS.src-vs-copy.txt` == 源）。

**L3 收尾断言**：`stage/` 下的 ckpt 与源同 inode（硬链接），跑完复算源 sha256 未变（`✓ 源 ckpt sha256 未变`）。

**刻意不采信 `usedStorage`**：上次 bucket 导出实测它异步滞后（480 GB 传完后一度显示 312 GB / 200 文件），
且它是字节总量、对等长篡改失明、失败时零诊断信息。

## 三、耗时（实测）

端到端 **约 30 秒**（日志 `23:52:08` → `23:52:38`），远低于起跑前 3–8 分钟的估计。原因是 xet 的内容分块去重：
进度条显示 955 MB 里「New Data Upload」实际只有 134→402 MB 量级在传，峰值 75 MB/s。八阶段全部一次通过，
三次重试与 60 s 退避一次都没用上。

## 四、资产锁（本轮同批落地）

`scripts/assets/ASSETS_LOCK.json`（11,778 B，进 git，顶层自哈希 `b842732e…`）收敛了此前散落在三处的锚点：
`docs/dataset-build-doc/4task-motion-40ep/launch.md`、
`v1-store/external/motionjepa/wan-v8-filter10-72ep-a/SHA256SUMS.src-vs-copy.txt`（无人读取）、
`docs/training-doc/v1-grad-baseline-g0b/records/r1/env.json`。后两者作为历史记录保留，不再是代码读取的真值来源。

六条记录、full 档逐条实测：

```
✓ siglip_params          level=full   1.08s      1,659,216,368 B  f16e9312…
✓ paligemma_tokenizer    level=full   0.01s          4,264,023 B  8986bb4f…
✓ pi05_base              level=full   8.16s  20 文件/12,441,721,931 B  逐文件清单
✓ wan_vae                level=full   0.33s        507,591,892 B  d6e524b3…（+ revision 0fad780a… 四层绑定）
✓ motionjepa_ckpt        level=full   0.60s        954,853,147 B  bae96037…
✓ motionjepa_config      level=full   0.00s              1,927 B  99548a6c…
ASSETS=PASS assets=6 mismatches=0
```

cheap 档（字节数 + 首尾各 1 MiB blake2b-128）在 `v1_require_models 1` 里实测 0.14 s，可放进每次起跑的前置。

**接入点六处**（详见 commit `c7406e4`）：两个 `paths.sh` 的 `v1_require_models` / `v1_require_wan`、
`run_local.py`（删自证循环 + 按 stage 要资产）、`motion_sidecar.py`（ckpt 兜底 + 与数据集 provenance 交叉断言）、
三个探针（默认按 lock 校，跳过须显式 `SKIP`）、`run_t3_eval_obs.sh`（补前置）。

## 五、意外与处置

1. **`hf upload` 会自动建 repo 且默认建成 public**（CLI 内部硬调 `create_repo(exist_ok=True, private=private)`，
   而 `--private` 默认 `None`）。对抗验证阶段发现，driver 的每条 upload 都补了 `--private`，
   并在上传前后各用 API 核一次 `private=True`。本次目标 repo 已存在，该参数被忽略，但 repo_id 打错一个字母时
   它是唯一的兜底。
2. **原方案「不设 `HF_HOME`」被推翻**。实测 `HF_HOME` 或 `XDG_CACHE_HOME` 单独改指都会让 `hf auth whoami`
   报 `Not logged in`（后者是前者默认值的父项），但 `HF_HOME` + `HF_TOKEN_PATH=$HOME/.cache/huggingface/token`
   组合正常返回 `HongzeFu`。最终既满足 AGENTS 第 14 条（缓存进 `v1-store`），又不必像 `run_export.sh` 那样
   把密钥复制成第二份落盘。
3. **原方案「手写 urllib 取 HF 文件」被推翻**。`resolve/<revision>` 会 302 到 `us.aws.cdn.hf.co` 的预签名 URL，
   naive urllib 读到的是最后一跳的 `etag`（xet merkle root `7f7e0b28…`），而规范值是第一跳的
   `x-linked-etag`（LFS sha256 `f16e9312…`），两者不等、拿它校验必然误判；且 Python 3.11 的
   `HTTPRedirectHandler` 只剔 content-length/content-type，会把 `Authorization` 原样发给 CDN。
   改用主 venv 已装的 `huggingface_hub 0.32.3`（`transformers` 传递依赖），**不 `uv add`**——动根 `uv.lock`
   会让 `check_baseline_env` 的 `uv_lock_sha256` 变、G0b 黄金基线全 FAIL。`git diff --stat uv.lock` 全程为空。
4. **原方案「给基线指纹加 siglip 字段 + schema 投影」被推翻**。`_diff` 是键并集深比，加字段会让 24 份历史
   `env.json` 全 FAIL；而投影方案本身有 bug（投影只删新增键，`schema` 字段自己也在比较范围内，投影后仍剩一条差异），
   且 `env.json` 就在 `BASELINE_MANIFEST.json` 的 10 个受保护产物之列，回填会同时触发「产物腐烂」。
   最终 `check_baseline_env.py` **一个字节没动**，SigLIP 校验挂在真正用它的 `v1_require_models` 上。
5. **上一轮的一处错判已更正**：曾判定 `siglip_params.pkl` 无公开来源、必须上传。实际仓库 `README.md` 就写了
   来源 `Yinpei/pi05_vision_encoder`（公开），其 LFS oid 与本机文件逐字节相同。lock 直接钉了该 revision。
6. 新目录 `scripts/assets/` 不在 `pyproject.toml` 的 RUF001/2/3 per-file-ignores 白名单里（该表按目录列，
   中文全角标点会被判「歧义 Unicode 字符」），补了一行；该改动只动 `[tool.ruff]` 段，`uv.lock` 零变化。

## 六、AGENTS 第 18 条判断

**不触发**，且判据是机器可复算的。本轮 18 个改动文件里，落在训练输入链路前缀
（`src/mme_vla_suite/{training,datastore,data,models,shared,dataset_builder}/`、
`src/openpi/{training,models,transforms}`、`policies/motion_{client,protocol}.py`）的有 **0 个** ——
见 `records/diff_scope.txt`。资产校验是前置断言，语义二值：要么与改动前逐位相同地继续，
要么在读第一个样本之前 `raise`；它不产生新数、不改数、不改取数顺序。

特别地，给 SigLIP 补校验最自然的落点 `src/mme_vla_suite/dataset_builder/siglip_tokenizer.py`
（唯一加载点）被 `test_guards._FROZEN_BUILDER_SHA256` 钉死在 `72fb8423…`，改一个字节哨兵当场变红且会立即
触发第 18 条——**本轮绝不碰它**，校验改挂在 shell 前置层。

## 七、其余验证

| 判定行 | 结果 |
|---|---|
| `pytest scripts/assets/test_assets_lock.py -q` | 28 passed in 0.14s |
| `pytest scripts/dataset/test_guards.py -q` | 47 passed in 6.23s（含四条冻结源码哨兵与 SOURCE_PIN 哨兵） |
| 训练域 `v1_require_models 1` | `ASSETS=PASS assets=3 mismatches=0` + `REQUIRE_MODELS=OK` |
| 建库域 `v1_require_models 1; v1_require_wan` | 两段 `ASSETS=PASS` + `REQUIRE_MODELS_WAN=OK` |
| `check_baseline_env.py check --baseline …/g0b/records/r1` | `BASELINE_ENV=PASS` |
| `_motion_gates`（40ep 迷你库） | `MOTION_GATES=PASS` |
| ruff（四个 wan 文件） | 与 HEAD 逐个持平（25/7/5/6），零新增 |

## 八、清理与留档

- `v1-store/exports/hf-model-motionjepa-encoder-v1/{stage,verify}` 已删除（各约 0.91 GB；`stage` 下的 ckpt
  是硬链接，`rm` 只删链接不动源，删后已复算源 sha256 未变）。保留 `logs/` 与本目录 `records/`。
- 本目录 `records/`：`SHA256SUMS.txt`、`blobs_api.json`（远端元数据快照）、`upload.clean.log.txt`、
  `assets_lock_verify.txt`、`negative_tests.txt`、`diff_scope.txt`。**不归档权重**（AGENTS 第 12 条）。

## 九、已知盲区（诚实清单）

1. **异地复刻仍有一条硬阻塞**：两个 `paths.sh` 断言仓库必须位于 `/data/hongzefu/` 或 turbo 前缀下，
   异地机器 source 即 `exit 1`。本轮按用户决定**不动**该 fail-loud，作为已知阻塞登记。
2. **cheap 档挡不住保持长度的中段字节篡改**——已由 `test_flip_midfile_is_cheap_blind_and_full_catches`
   钉成显式契约。真正的保证来自 `verify --level full` 与重载点（`load_encoder` 每次全量算 ckpt sha256、
   `load_vae` 每次算 state_dict 指纹）。
3. **pi05_base 两套口径并存**：lock 用逐文件全量 sha256（20 行清单），G0b 基线指纹里那份
   `tree-relpath-size-headtail1MiB` 抽样口径原样保留不动（动它就破基线）。两者不互相取代。
4. **`lfs.sha256` 是客户端提交的**，能证明「Hub 记录的就是我这份文件的 sha」（挡传错、挡漏传），
   不能证明「Hub 存的字节确实哈希成这个值」——后者由 L2 回读兜底。
5. **上游 force-push 无解**：HF 允许覆盖 revision。`HongzeFu/MotionJEPA` 是自己的 repo，纪律上禁 force-push；
   三方公开 repo 只能靠本机/turbo 的实体副本兜底。
6. **跨 GPU 架构不逐位**：资产锁保证输入字节同一，不保证输出数值逐位同一。实测 SigLIP 段 A40 vs 本机 Ada
   `image_emb min_cos 0.99959 / p5 0.99997`，而 `pos_emb`/`state_emb`/`pkl`/`kept_indices` 逐位相同；
   Wan 段只有同机双卡 64 窗 `max|Δ|=0`，跨架构未测。**禁止把 `ASSETS=PASS` 读成「数值可逐位对拍」。**
7. **转 public 前必须清洗**：model card 与 `SHA256SUMS.src-vs-copy.txt` 含内部集群绝对路径、私有 GitHub repo 名、
   Slurm job id 与未公开数据集信息。card 顶部已放 HTML 注释硬提醒。

## 十、本文件与 model card 的分工

模型本身的一切事实（结构、训练 commit `7388a42`、数据集 `dataset-4env-v8`、超参、指标、数值合同、加载示例、
限制、许可）唯一正本是 `model-card.md`，本文件不复述；上传过程与验收判定行是本文件的职责，model card 不写。
**三个锚允许双写且必须逐字符相同**：ckpt sha256 `bae96037…c15a`、训练 commit `7388a42`、run 名
`wan-v8-filter10-72ep-a`；改一处必须同步改另一处。
