# robomme-16task-mixed8-200-trainset-20261016-v1

16 个 RoboMME 任务 × 200 episode 的 VLA 训练库完整交付：packed framesamp 表（8×8 训练档 + 4×4
对照档）+ motion 表 + 源样本 pkl + 归一化统计 + 中间产物与对拍材料 + 溯源。下完这一个 bucket，
再取两个匿名可拉的公开权重，就能起跑本库的两条 80k 训练（512 modulation 与 512+motion modulation）。

由 `scripts/dataset/hf_export/run_mixed8_200_trainset_export.sh` 上传，验收是**上传前 / 上传后两遍
独立重算的 sha256 逐行对照**：回读按批进行（每批不超过 100 GiB，下载一批、逐文件比对、删掉再下一批），
全部批次汇总后两份清单逐行相同；每批另抽一个 tar 逐成员复算 sha256。本库**没有**做「用回读副本
跑 20 步训练」的冷启动彩排（上传只占 CPU 与网络），可训练性由建库侧的验收判定行与两条正式训练本身证明。

> 本文件里的数字与 sha256 都是上传驱动从库自己的 meta 现读、现算后填进来的，不经人手。

## 这个库是什么

| 项 | 值 |
|---|---|
| 任务（`meta/episode_manifest.json` 的 `canonical_order`，共 @@NUM_TASKS@@ 个） | @@TASKS@@ |
| episode / timestep / exec 样本 | @@NUM_EPISODES@@ / @@NUM_TIMESTEPS@@ / @@NUM_EXEC@@ |
| framesamp 布局（训练直读） | `framesamp-8x8/`，每行 64 个 2048 维 bf16 SigLIP patch 特征 |
| framesamp 布局（对照档） | `framesamp/`（4×4），训练不用 |
| motion 表 | `motion/`，每个 chunk 1 个 768 维 motion token |
| motion encoder | `wan-mixed8-200-filter2-stop075-b88x8-72ep-a/checkpoint_epoch_72.pt#encoder` |
| 源 h5 | 同账号 bucket `HongzeFu/robomme-16task-mixed8-200-h5-20261014-v1` |

encoder 本体（全部 36 个 checkpoint 与 200 集可视化 mp4）在 `HongzeFu/motionjepa-wan-mixed8-200-72ep-v1`；
**训练不需要它**（见下）。用本库训出来的两个模型分别在 `HongzeFu/robomme-vla-mixed8-200-modul-512-80k-v1`
与 `HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1`。encoder 训练读的 latent 库在
`HongzeFu/robomme-16task-mixed8-200-motion-20261014-v1`。

## 布局

```
framesamp-8x8/                       ★ 训练直读，原样直传、不必解包
  image_emb_8x8/part_*.bf16.bin            packed 表分片
  pos_emb_8x8.f32.bin  state_emb.f32.bin
  meta/store_meta.json                     唯一契约（status=verified）
  meta/row_digests.blake2b.bin  meta/pack_progress.jsonl
framesamp/                           4×4 对照档，同构布局，训练不用
motion/                              ★ motion 开启态训练直读
  motion_token.f32.bin
  meta/store_meta.json  meta/motion_index.json  …
meta/
  episode_manifest.json                ★ 契约锚点，一个字节都不能改
  input_manifest.json
source/
  data_tars/data-*.tar                 ★ {idx}.pkl 训练样本，**必须解包才能训练**
  features/                            @@NUM_SPOT_FEATURES@@ 个抽样复验点名文件（见「已知事项」第 2 条）
  meta/                                建库侧 provenance / stats
assets/robomme/norm_stats.json       ★ 训练必需的归一化统计
wan-latents_tars/  oracle_tars/  motion-tokens_tars/    中间产物与对拍材料，训练不用
runner/{_train_common.sh,train-512-80k.sh,train-motion-80k.sh}   两条正式训练的起跑脚本（留档用）
manifest/upload_manifest.json        tar 分片与直传件清单（各自 sha256 与字节数）
checksums/                           sha256-shards / sha256-plain / sha256-source-files（逐 tar 成员）
SHA256SUMS.pre.txt                   上传前重算的全量清单（覆盖除自身外的全部字节）
```

`source/data_tars` 的成员名是 `data/{idx}.pkl`，在 `source/` 下解开即还原 `source/data/`。tar 的
`mtime/uid/gid/uname/gname` 全部归零、成员定序写入，逐位可复现。

## 异地怎么用

1. **下载与自验**

   ```bash
   hf sync hf://buckets/HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1 ./lib
   cd lib && sha256sum -c --strict SHA256SUMS.pre.txt        # 覆盖全部字节
   ```

   只要训练的话，`--exclude` 掉 `wan-latents_tars/*`、`oracle_tars/*`、`motion-tokens_tars/*` 与
   `framesamp/*`（4×4）可以少下一大块。

2. **解包训练样本**：`for t in source/data_tars/*.tar; do tar -xf "$t" -C source; done`，解出的
   `source/data/*.pkl` 个数应等于上表的 exec 样本数。解开后可在 `source/` 下用
   `checksums/sha256-source-files.txt` 逐成员再 `sha256sum -c` 一遍。

3. **放到训练仓库约定的位置**：库放在训练仓库的 `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/`，
   `assets/robomme/norm_stats.json` 复制到
   `v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f/robomme/norm_stats.json`。
   两个 framesamp 的 `store_meta.json` 里写死了建库机上的绝对路径（`manifest_path`、`source_dataset_root`），
   起跑脚本已用 `MMEVLA_FRAMESAMP_SOURCE` / `MMEVLA_FRAMESAMP_MANIFEST` 显式指向库内的 `source/` 与
   `meta/episode_manifest.json`；仓库 clone 到与建库机不同的路径时，还要在训练仓库
   `scripts/training/paths.sh` 的前缀白名单里加一条常量前缀。

4. **取两个外部权重**（都从 `gs://` 匿名可拉，不需要 token）：`pi05_base`（初始权重）与
   `paligemma_tokenizer`：

   ```bash
   uv run python scripts/assets/fetch_assets.py fetch  --assets pi05_base,paligemma_tokenizer
   uv run python scripts/assets/fetch_assets.py verify --assets pi05_base,paligemma_tokenizer --level full
   ```

5. **起跑**（在训练仓库根）：

   ```bash
   bash scripts/dataset/hf_export/mixed8_200_runner/train-512-80k.sh    <run_name>   # 512 modulation
   bash scripts/dataset/hf_export/mixed8_200_runner/train-motion-80k.sh <run_name>   # 512+motion modulation
   ```

   bucket 里的 `runner/` 是这两个脚本的原样副本，用于留档核对；它们按自身位置推算仓库根，要从训练仓库里跑。

**SigLIP、Wan VAE、motion encoder 训练都不需要。** motion token 已离线烘焙进 `motion/motion_token.f32.bin`；
history config 里那行 `source_run: wan-mixed8-200-…#encoder` 只是身份字符串，加载时与
`motion/meta/store_meta.json` 的 `provenance.encoder` 比对，不打开任何权重文件。

## 契约与绑定

| 锚点 | 值 | 谁引用它 |
|---|---|---|
| `episode_manifest.json` 内部自哈希 | `@@MANIFEST_SELF_SHA@@` | 两个 framesamp store 与 motion store 的 `manifest_sha256`；加载时现场重算比对，不等即拒 |
| `episode_manifest.json` 整文件 sha256 | `@@MANIFEST_FILE_SHA@@` | 本 bucket 的 `SHA256SUMS.pre.txt` |
| `framesamp-8x8/meta/store_meta.json` | `@@FS8_META_SHA@@` | 两条 run 的 `motion_provenance.json` |
| `framesamp/meta/store_meta.json` | `@@FS4_META_SHA@@` | — |
| `motion/meta/store_meta.json` | `@@MOTION_META_SHA@@` | motion 那条 run 的起跑 preflight |
| `motion/meta/motion_index.json` | `@@MOTION_INDEX_SHA@@` | motion store_meta 的 `motion_index_sha256`；加载时现场重算比对 |
| `assets/robomme/norm_stats.json` | `@@NORM_STATS_SHA@@` | 两条 run 的起跑 preflight |

**两个 framesamp store 与 motion store 的 `status` 都是 `verified`，包内不含任何 `meta/pack.lock`**（上传驱动打包前逐个断言）——
读侧对这两点各有一道硬闸（`status != "verified"` 或 `pack.lock` 存在即拒绝加载）。

## 已知事项（诚实清单）

1. **`source/data/` 是硬依赖，不是可选项。** `framesamp-8x8` 只装 memory 路的 SigLIP 特征；当前观测的
   `image / wrist_image / state / actions / prompt` 仍然逐样本从 `source/data/{idx}.pkl` 读。不解开
   `source/data_tars/` 一步都训不动。
2. **`source/features/` 只有 @@NUM_SPOT_FEATURES@@ 个文件，这是刻意的、不是遗漏。** 完整 features 已被 packed 表
   取代、训练不读，建库后已删除。但 `framesamp-8x8/meta/store_meta.json` 的 `source_spot_sha256.entries`
   里有抽样复验点落在 `features/` 下，加载时按进程 pid 轮转抽一条——少一个就会在「源库抽样文件缺失」处
   拒跑。本库带的正是这些文件，名单由上传驱动从 store_meta 现读、未硬编码。
3. **`logs/`（建库日志）刻意未上传**：含完整命令行与 venv 内部布局，不参与任何 sha 绑定、不被任何脚本读。
4. **本库任何数据字节都未被修改。** 下面这些内容原样保留并在此披露：两个 framesamp store_meta、motion
   store_meta、`meta/*.json`、`source/meta/` 里的建库机绝对路径；motion store_meta 与 `oracle_tars/`
   里记录的 encoder 源码仓库路径与仓库名；`motion-tokens_tars/`、`wan-latents_tars/`、`oracle_tars/` 内
   metadata 记录的建库机内部 DNS 名（AWS VPC 私有地址，对外不可路由）与 GPU UUID。它们全部进了
   `manifest_sha256` / `motion_index_sha256` / 各 store_meta 的哈希绑定范围，为掩掉一个路径而改字节，
   会让这个库再也过不了自己的加载闸。处置只有「排除」与「披露」；上传前对源库与 stage 各做了一遍
   公开性体检，每一处命中都有裁决记录（`scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_trainset.json`）。
5. **`wan-latents_tars/`、`oracle_tars/`、`motion-tokens_tars/` 与 `framesamp/`（4×4）训练不用。** 它们在这里
   是为了让异地能独立重算 motion 表、跑 4×4 对照，以及做逐位对拍。
6. **不在本库内（有意排除，不是遗漏）**：原始 h5（在 `HongzeFu/robomme-16task-mixed8-200-h5-20261014-v1`）、
   `source/features/` 其余部分、外部权重（`fetch_assets.py` 匿名拉）、motion encoder 本体（在
   `HongzeFu/motionjepa-wan-mixed8-200-72ep-v1`）。
7. **本库的数字不与官方上游 `mme_vla_suite` 逐项可比**：训练步数 × batch（80k × 128）与官方不同，
   optimizer 更新次数不同。引用时须带此声明。

## 许可

沿用上游 RoboMME 的 Apache-2.0。
