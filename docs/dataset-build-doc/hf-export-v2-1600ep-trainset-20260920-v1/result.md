# hf-export-v2-1600ep-trainset-20260920-v1 — 结果

**终判 PASS。** 这一版训练库 `4task-v2-1600ep-604f16da`（BinFill / RouteStick / VideoRepick /
VideoUnmaskSwap）已完整上传到**公开** bucket
`hf://buckets/HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1`：**295 个对象 /
716,544,476,858 字节（667.3 GiB）/ tar 内成员 631,233 个**。上传前与回读后两遍独立重算的
sha256 逐行 diff 为空，匿名（无 token）可读已验证。

**并且这一轮不止验到字节：** 末尾用**从 HF 下回来的那份副本**跑了两次 20 步真实训练
（512 modulation 与 512+motion modulation 各一次），`COLD_START=PASS`。这是「异地下完真能
开训」的直接证据，而不是「本机原件能开训」。起跑口径见 [`launch.md`](launch.md)。

执行窗口 2026-09-20T20:37:50Z → 23:41:23Z（**3 h 03 min 33 s**），`EXIT_CODE=0`，
tmux `hfup-v2-trainset`。**本轮共起过 `hfup-v2-trainset` 一个会话**；用户自有的 6 个会话
（`0`、`1`、`claude-private`、`codex`、`codex-repo`、`codex2`）全程未动。

起跑 HEAD 为 `a99a7e48eac6e3129aba277dcedd23308fc1892a`（clean）。
注：`launch.md` 正文里写的 `eb8c318` 是代码收官那个 commit；`launch.md` 自身进 git 又把
HEAD 推到了 `a99a7e4`，实际起跑用的是后者（日志首行 `HEAD=` 记的就是它）。

## 一、判定行（全部命中，原文）

```
HF_WHOAMI=HongzeFu
FREED_BYTES=1179906336871
AVAIL_BYTES=1780745756672
HYGIENE=PASS scanned=16037 hits=25745 ungated=0            ← 阶段 1，源侧
BUCKET_CREATED=HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1 PRIVATE=False
BUCKET_FILES_AT_START=0 BUCKET_BYTES_AT_START=0
ANCHOR_OK=5
PACK_DONE shards=189 plain=96 members=631233 bytes=716491935758
HYGIENE=PASS scanned=25 hits=79 ungated=0                  ← 阶段 4，stage 侧
LOCAL_FILES=295 LOCAL_BYTES=716544476858
PRE_LINES=294
PLAN_BATCHES=90 PLAN_FILES=295 PLAN_BYTES=716544476858 ROOT_FILES=2
  MAX_BATCH_BYTES=10213654528 MAX_BATCH_FILES=17 PLAN_SHA=0bd9dbaef7097184
UPLOAD_DONE batches=90
BUCKET_FILES=295 BUCKET_BYTES=716544476858
SHA256_PRE_POST_DIFF=0                                     ★ 核心判据
  sha256sum -c 校验行数: 294
MEMBERS_SPOT=PASS tars=6 members=22730 mismatches=0
RESULT=PASS
SRC_UNCHANGED=OK
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK
COLD_START=PASS steps=20
EXIT_CODE=0
```

原文见 [`records/verdict-lines.txt`](records/verdict-lines.txt)。

## 二、五层校验闭环

| 层 | 看什么 | 抓得住什么 | 结果 |
|---|---|---|---|
| 计数层 | REST 的 `totalFiles` / `size` 对本地 | 漏传整个对象 | 295 / 716,544,476,858 逐字相等，**一次过、未触发异步滞后复查** |
| 内容层 | 回读 667.3 GiB 后**独立重算** sha256，与 pre 清单逐行 `diff` | 任何一位翻转 | `SHA256_PRE_POST_DIFF=0` |
| 复校层 | 回读副本上 `sha256sum -c --strict` | 同上，换一条实现路径 | 294 行全 OK |
| 成员层 | 按顶层类别分组抽 6 个 tar 复算成员 | 成员名写错、清单与 tar 内容错位（sha256 看不见） | 22,730 个成员 0 失配 |
| 源侧 | 第三遍重算源文件 sha256 与 pre 对 | 硬链接打包把原件改坏 | `SRC_UNCHANGED=OK` |

内容层是**独立重算**而不是拿 pre 去 `-c`：两份清单都留档，任何时候能指出是哪一行、哪个文件变了。

## 三、冷启动彩排（本轮新增的一层）

阶段 9 通过后删掉 stage 的 tar 实体腾出 305 GiB，把回读副本的 148 个 `data_tars` 解成
**605,611 个 pkl**（与 `episode_manifest.json` 的 `num_exec_samples` 硬断言相等），
然后**只用 `verify/` 这份从 HF 下回来的副本**起训练：

| config | history config | mesh | 结果 |
|---|---|---|---|
| `mme_vla_suite_b128_60k`（512 modulation，motion 关） | `perceptual-framesamp-modul-8frame-8x8.yaml` | `shape=(2,4) fsdp=4 workers=8` | 跑到 step 19 并存档，`Step 0: loss=0.0894 grad_norm=1.2782` |
| `mme_vla_suite_b128_80k`（512+motion） | `…-8x8-motion.yaml` | `shape=(1,8) fsdp=8 workers=16` | 跑到 step 19 并存档，`Step 0: loss=0.0888 grad_norm=1.1138` |

两次 dataloader 都正常建起（`static_image_emb (128,512,2048)@bfloat16`、
`static_pos_emb (128,512,768)@float32`），norm_stats 从 `verify/assets/robomme/` 读到。
彩排期间命中并通过的读侧闸：两库的 `pack.lock` 不存在与 `status=verified`、
manifest sha256 现场重算比对、32 个 part 的 `st_size` 与抽样首尾 blake2b、
**源库 16 选 1 抽样指纹**（这条专验 `source/` 传全了没有）、`motion_index.json` sha256、
双库 `manifest_sha256` 同源、`source_run` ↔ `provenance.encoder` 四字段严格相等。
两个临时 run 按 AGENTS 第 6 条跑完即删。原文见
[`records/coldstart-rehearsal.txt`](records/coldstart-rehearsal.txt)。

## 四、实测耗时

| 阶段 | 窗口（UTC） | 挂钟 | 备注 |
|---|---|---:|---|
| 0 腾盘 + 预检 | 20:37:50 → 20:38:2x | ~30 s | 删 7 个旧 `verify/`，实际腾出 **1,179,906,336,871 B（1.07 TiB）** |
| 1 源侧体检 | → 20:38:3x | ~15 s | 16,037 个文本文件 |
| 2 建 bucket | → 20:38:38 | 数秒 | |
| 3 打包 | 20:38:38 → 20:45:55 | **7.3 min** | 189 个 tar + 96 个直传件，631,233 个成员 |
| 4 README/runner + 上传前 sha256 | → 20:52:22 | **6.5 min** | 667 GiB，`xargs -P16` 约 1.85 GB/s |
| 5 分批上传 | 20:52:30 → 22:26:41 | **1 h 34 min** | 90 批，均速约 127 MB/s |
| 6 计数层 | → 22:2x | 数秒 | |
| 7+8 回读 + post 清单 | → 22:44:37 | ~17 min | 回读约 1.2 GB/s |
| 8 余下（`-c` 复校 + 成员抽样）+ 9 源侧 | → 23:28:38 | ~44 min | `sha256sum -c` 是单进程，667 GiB 占大头 |
| 10 公开性 + 11 彩排 | → 23:41:23 | **12.7 min** | 含解包 225 GiB 与两次 20 步训练 |
| 合计 | 20:37:50 → 23:41:23 | **3 h 03 min 33 s** | |

比 `launch.md` 预估的 2–2.5 h 长约半小时，差在上传：预估按 400ep 实测的 546 MB/s 峰值外推，
实测只有 127 MB/s。原因是这一轮的字节几乎全是新内容（292 GiB 的 part bin + 225 GiB 的
pkl tar），Xet 的内容寻址去重帮不上忙；400ep 那轮的高速率有相当部分来自服务端已有块。

## 五、六则值得记下的事

1. **单片 10.2 GB 顶在已知超时阈值上，但没出事。** 起跑前判定这是本轮唯一的新变量：
   `framesamp-8x8` 的 32 个 part 单片 9.78–10.21 GB，装箱后一片一批，而 400ep 实测的
   经验阈值是「2 GB×10 = 20 GB 挂」。**实测 90 批全部一次过，零重试**，8 次重试的退路
   与「改用 `hf buckets cp` 单文件传」的预案都没用上。结论修正：失败点确实在批内**字节量**，
   但 10 GB 这一档在单对象形态下是安全的——400ep 挂掉的那批是 10 个 2 GB 对象，
   说明**对象个数**对 commit 负担的贡献不能忽略，不能只看总字节。
2. **腾盘比预估多 470 GiB。** 方案里按 6 个旧 `verify/` 估 610 GiB，实际有 7 个
   （漏算了 `hf-motionjepa-full1600/verify` 的 446 GiB，那是先前 `du` 输出被截断没看到），
   实腾 1.07 TiB，峰值 974 GiB 的余量比预期宽得多。
3. **`source/data/` 是硬依赖这件事，是本轮最重要的发现，且来得很险。** 最初的盘点把
   「最小可训练集」报成 292 GiB（只有 framesamp-8x8 + motion + meta + norm_stats），
   按那个口径打包上去，异地会在第一个 `__getitem__` 就 `FileNotFoundError`。
   实际 `framesamp-8x8` 只装 memory 路的 SigLIP 特征，当前观测仍逐样本读
   `source/data/{idx}.pkl`，225 GiB / 605,611 件一个都不能少。
4. **`source/features` 的 8 个抽样点同样不能少。** `run_fast_checks` 按 `os.getpid() % 16`
   轮转抽 `source_spot_sha256.entries`，16 条里 8 条落在 `features/` 下。
   链路里这份名单从 store_meta **现读**、没有硬编码——如果哪天重建库换了抽样点，
   打包器会跟着变，不会悄悄漏。
5. **stage 侧体检比源侧多扫 4 个文件。** 源侧 16,037，stage 侧 25：差额是 README.md 与
   `runner/` 四个脚本、`manifest/`、`checksums/`。allowlist 的 `runner/*` 条目按 fnmatch
   同时盖住了 `_train_common.sh`，没有漏网。400ep 首轮栽在「README 的披露文本自身命中被
   披露的模式」，本轮起跑前就把这四条 `scope=stage` 的裁决写进了账本，没有重蹈。
6. **计数层一次过没触发复查。** 400ep 与 480 GB 那两轮都遇到过 REST 统计异步滞后
   （一度显示 312 GB / 200 文件），本轮 90 批传完立刻查就已经追平。这不代表滞后问题消失，
   `sleep 120` 复查那段仍然保留。

## 六、盲区（说清楚没验到什么）

1. **彩排只跑了 20 步，不证明训得出同样的模型。** 它证明的是「库能被完整加载、所有身份闸
   全过、前 20 步的 loss 与 grad_norm 落在合理量级」，不是数值可逐位复现。
   跨机复现的数值一致性从来没有被本仓库验证过（资产锁那轮已记：`ASSETS=PASS` 只保证输入
   字节同一，不保证输出逐位同一）。
2. **彩排在本机跑，不是真异地。** 仓库路径、驱动、CUDA、venv 都是同一台。它验的是
   「bucket 内容是否自足」，验不了「另一台机器的环境差异」。`runner/restore.sh` 与
   `train-{60k,80k}.sh` 在真异地机器上从未执行过。
3. **`lfs.sha256` / bucket 侧的字节是客户端提交的。** 回读闭环能证明「拉回来的字节与传上去
   的一致」，不能证明「Hub 内部存储层没有另一份不同的副本」。
4. **`framesamp/`(4×4)、`oracle_tars/`、`wan-latents_tars/`、`motion-tokens_tars/` 这 155 GiB
   只过了字节层，没过语义层。** 彩排不读它们。它们能不能真的用来重算 motion 表或跑 4×4
   对照，本轮没验。
5. **匿名可读只验了 README 一个文件。** 没有匿名拉过大对象，也没验过匿名下整库的速率。

## 七、顺手修掉的两处已发布文档错误

两个 ckpt bucket 根下的 `README.md` 已用 `hf buckets cp` 覆盖（只覆盖 README，未重传 ckpt）：

| bucket | 修了什么 |
|---|---|
| `HongzeFu/robomme-vla-modul-60k-v1` | 「任务」行原写上一版的 ButtonUnmask / VideoUnmask / ButtonUnmaskSwap / VideoUnmaskSwap，改为实际的 BinFill / RouteStick / VideoRepick / VideoUnmaskSwap（以 `episode_manifest.json` 的 `canonical_order` 为准） |
| `HongzeFu/robomme-vla-modul-motion-80k-v1` | 同上；另把结尾「数据本体见 `robomme-4task-motion-full1600-20260914-v1`」改为指向本轮新 bucket，并保留一段说明记下旧指引错在哪——那个 bucket 装的是阶段一 encoder 自己的 Wan chunk latent，既没有 framesamp store 也没有 motion store |

## 八、这一版数据集与模型在 HF 上的完整落点

| 内容 | bucket / repo | 可见性 |
|---|---|---|
| 原始 h5（按难度分档） | `HongzeFu/robomme-4task-h5-20260912-v2` | 公开 |
| **训练库（本轮）** | `HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1` | 公开 |
| 阶段一 encoder 训练数据（Wan chunk latent） | `HongzeFu/robomme-4task-motion-full1600-20260914-v1` | 公开 |
| 阶段一 encoder 权重（36 个 ckpt） | `HongzeFu/motionjepa-wan-full1600-72ep-v1` | 私有 |
| 512 modulation 模型（60k，12 个 ckpt） | `HongzeFu/robomme-vla-modul-60k-v1` | 私有 |
| 512+motion modulation 模型（80k，16 个 ckpt） | `HongzeFu/robomme-vla-modul-motion-80k-v1` | 私有 |

至此这一版的**数据集与模型在 HF 上齐了**，异地一条 `runner/restore.sh` + 一条
`runner/train-{60k,80k}.sh` 即可开训。

## 九、留底与可回收

- `v1-store/exports/hf-v2-1600ep-trainset/`：`stage/` 的 tar 实体已在阶段 11 删除，
  直传件是硬链接（删了不回收空间）；`verify/` 892 GiB（含解包出的 225 GiB pkl）
  **验收后可整目录删**，真要复查从 bucket 重下即可。
- 本目录 `records/`：`verdict-lines.txt`、`upload_plan.json`（90 批装箱计划，
  `PLAN_SHA=0bd9dbaef7097184`）、`hygiene-{source,stage}.json`、
  `coldstart-rehearsal.txt`、`SHA256SUMS.pre.head40.txt`。
  全量 pre 清单（294 行）在 bucket 根下的 `SHA256SUMS.pre.txt`，不重复归档。
