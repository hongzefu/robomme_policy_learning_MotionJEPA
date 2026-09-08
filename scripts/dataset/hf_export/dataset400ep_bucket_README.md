# robomme-4task-motion-400ep-20260904-v1

四任务 × 100 episode 的完整派生数据集：SigLIP framesamp packed 库 + Wan latent +
motion token 表 + 逐位对拍 oracle。由 `scripts/dataset/hf_export/run_dataset400ep_export.sh`
上传，**上传前 / 上传后两遍独立重算的 sha256 逐行对照**验收（`SHA256SUMS.pre.txt` 与回读侧
重算的 post 清单 diff 为空，覆盖全部 130 GB 字节），另有 229,323 行逐源文件清单供解包后自验。

## 这个库是什么

> **口径澄清（重要）**：`400ep` 指**总共 400 个 episode = 4 个任务 × 每任务 100 episode**，
> **不是**每任务 400 episode。四个任务是 ButtonUnmask、ButtonUnmaskSwap、VideoUnmask、
> VideoUnmaskSwap。

原始来源是公开数据集 [`Yinpei/robomme_data_h5`](https://huggingface.co/datasets/Yinpei/robomme_data_h5)
的四份 H5（train split，每任务 100 episode）：

| 文件 | 字节 | sha256 |
|---|---:|---|
| `record_dataset_ButtonUnmask.h5` | 17,751,535,444 | `6b100414429e3417f2afd600ae708406bc20b1a37ef92734ff593af6bdb70575` |
| `record_dataset_ButtonUnmaskSwap.h5` | 26,627,642,804 | `7c0441210bb1ec63aa60cfc30c5080a5f09c54f02bd0004714eba120df089274` |
| `record_dataset_VideoUnmask.h5` | 14,445,442,112 | `05a653a8f8232882f82c84057f328e045f4a875ff5cfcf068738c429c6081427` |
| `record_dataset_VideoUnmaskSwap.h5` | 23,205,606,404 | `4e83aca373b2adb469cf78d338223e41559fc6ad19d435de5c88d99d2fe49a7e` |

从这四份 H5 派生出本库需要全程 GPU（Wan VAE encode + SigLIP 特征抽取 + encoder 前向）。
把它放上来正是为了让异地复刻不必重跑这条链路。

| 项 | 值 |
|---|---|
| episode / timestep / exec sample | 400 / 123,044 / 101,066 |
| framesamp 行数 | 123,044（pos_emb 586 行） |
| motion 行数 | 6,832 = exec 5,707 + demo 1,125 |
| wan latent | 600 段 / 6,832 窗 |
| 建库日期 / 硬件 | 2026-09-04 / 8 × A100-SXM4-80GB，本地 NVMe RAID |
| 建库 commit | SigLIP 段 `8093ebd`；Wan / encode / pack / oracle 段 `c0e13aa` |
| framesamp layout | `framesamp-4x4-v1`（image_emb 行 16×2048 bf16，state_emb 8 f32） |
| motion layout | `motion-768-grid16-v1`（`motion_dims [1, 768]`，grid_stride 16，window 33 帧） |
| Wan VAE | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，state sha256 `9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0` |
| motion encoder | run `wan-v8-filter10-72ep-a` / `checkpoint_epoch_72.pt`，sha256 `bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a` |

## 布局

```
README.md                                      本文件
SHA256SUMS.pre.txt                             上传前逐对象 sha256（路径相对 bucket 根）
manifest/upload_manifest.json                  分片清单：每片的 rel / sha256 / bytes / members
checksums/
  sha256-shards.txt                            67 个 tar 分片
  sha256-plain.txt                             49 个原样直传件
  sha256-source-files.txt                      229,323 行，逐**源文件**（tar 成员名）
meta/
  episode_manifest.json                        400 ep 逐集 (num_timesteps, exec_start_idx)
  input_manifest.json                          四份 h5 的 sha256
source/
  meta/                                        provenance.json / stats.json / _shard{0,1,2}of3.json
  data_tars/data-000{00..24}.tar               25 片，101,066 个 exec 样本 .pkl
  features_tars/features-*.tar                 ~37 片，400 个 episode 的 token_emb + kept_indices
framesamp/                                     **原样直传，不必解包即可直读**
  image_emb_4x4/part_*.bf16.bin                31 片
  pos_emb_4x4.f32.bin  state_emb.f32.bin  meta/
motion/                                        **原样直传，不必解包即可直读**
  motion_token.f32.bin  meta/
wan-latents_tars/*.tar                         2 片，600 段 latent + sha256 边车 + metadata
oracle_tars/*.tar                              2 片，VAE / encoder 逐位对拍基准
motion-tokens_tars/*.tar                       1 片，600 段 token
assets/robomme/norm_stats.json                 训练用归一化统计（本库的交付件）
```

三条注：

1. 源库的 `logs/`（19 个建库日志）**显式未上传**——它不参与任何 sha 绑定、不被任何脚本读取，
   却含大量构建机内部路径。这是刻意排除，不是遗漏。
2. 源库里的 `_claims/` 是空目录，tar 与 sync 都不携带空目录。
3. `assets/robomme/norm_stats.json`（sha256 `750a8e9b…`）是**本库的** 400 ep 归一化统计，
   与上游仓库里测试用的那份 `norm_stats.json`（`f332bbd3…`）**不是同一份**，不要混用。

## 恢复与自验

```bash
# 全量下载（huggingface_hub >= 1.5；旧版子命令是 `hf buckets sync`）
hf sync hf://buckets/HongzeFu/robomme-4task-motion-400ep-20260904-v1 ./robomme-4task-motion-400ep
cd robomme-4task-motion-400ep

# ① 下载完整性：覆盖除清单自身外的全部 130 GB 字节
sha256sum -c --strict SHA256SUMS.pre.txt

# ② 还原 source/ 目录树（解 tar，约 114 GB）
mkdir -p restored/source && (cd restored/source && \
  for t in ../../source/data_tars/*.tar ../../source/features_tars/*.tar; do tar -xf "$t"; done)

# ③ 逐源文件自验：229,323 行，成员名即 tar 内路径（data/0.pkl、features/episode_0/...）
(cd restored/source && sha256sum -c --strict ../../checksums/sha256-source-files.txt)
```

`framesamp/` 与 `motion/` 是**原样直传的 packed store，不需要解包**，下载下来即可直读
（`meta/store_meta.json` 描述行数、dtype、逐表 sha256）。异地训练时把 framesamp 源目录
用 `MMEVLA_FRAMESAMP_SOURCE` 指向还原出的 `source/`，或给建库脚本传 `--raw-dir`，
**不需要**让库内 provenance 里记录的绝对路径真实存在。

只取某一块：

```bash
hf sync hf://buckets/HongzeFu/robomme-4task-motion-400ep-20260904-v1 ./fs --include 'framesamp/*'
hf sync hf://buckets/HongzeFu/robomme-4task-motion-400ep-20260904-v1 ./mo --include 'motion/*'
```

## 契约与绑定

本库内部是靠 sha256 互相钉死的，任何一个文件被改都会让下面这张表对不上：

| 锚点 | 值 | 谁引用它 |
|---|---|---|
| `manifest_sha256` | `92fa17e97fba9434ee75302de12556319d8ce6d3feeb3adb9a397e830f477223` | framesamp / motion 两个 `store_meta.json`、`motion_index.json` |
| `motion_index_sha256` | `74185921690cd26cfd78d309b2d5f89c71c56c0a0ab43cd92b57534d4f8390f6` | motion `store_meta.json` |
| motion 表 | `6e70604da518c15647d69b5ecafdd74c16b20dad315290ba2a4f3b105c75e30f`（20,987,904 B） | motion `store_meta.json` |
| framesamp `pos_emb_4x4` | `3176ac09d063295fcd520339a7c89d5d3cb02e9af48e01e3ad1e9df7982b9972` | framesamp `store_meta.json` |
| framesamp `state_emb` | `80ecd422252649171fd4d98845cb16cb7aa65008a9e48c4f7d1b255e17632a86` | framesamp `store_meta.json` |

建库验收判定行（原文，来自建库留档）：

```
VERIFY_PACK=PASS scanned=123044 mismatches=0
VERIFY_MOTION=PASS scanned=6832 mismatches=0
WAN_BITEXACT=PASS compared=6832 frame_mismatches=0 latent_mismatches=0 metadata_mismatches=0
ENCODER_BITEXACT=PASS compared=6832 mismatches=0 order_ok=1 state_sha_ok=1 ckpt_ok=1
A7_BYTES=PASS segments=600 mismatches=0 table_rows=6832 table_ok=1
A10_ROWS=PASS rows=6832 exec=5707 demo=1125
[m1 real] samples=101066 mismatches=0
```

## 已知事项（诚实清单）

1. **各 `provenance` / `metadata.json` 字段含构建机指纹**——内部主机名
   （`ip-…​.compute.internal`，EC2 私有 DNS，不可路由）、GPU UUID、pid、驱动版本、
   以及构建机上的绝对路径。这些**不能改**：它们进了 `manifest_sha256` 的哈希范围，
   也是 D2/D3 逐位对拍（`WAN_BITEXACT` / `ENCODER_BITEXACT`）的可复核性依据，
   改一个字符就会同时断掉三处绑定并让上面那张契约表失效。对使用者无影响。

2. **`motion/meta/store_meta.json` 的 `provenance.source_pin.mj_repo_path`
   与 `oracle_tars` 内各 report 的 `mj_repo`** 指向构建机上一个非公开仓库的本地路径，
   对外不可达、也无需访问。真正起绑定作用的是同段的
   `source_sha256 af67fdd913543aee416a9fe5df797f707ff159165c468d687fcf8e7347941b34`
   （被复制模块的逐字节指纹）与 `mj_repo_commit`。同理，README 与库内多处提到的
   MotionJEPA encoder，其权重指纹（`bae96037…`）已在上表给出，本库不含该仓库的代码。

3. **建库时 `MOTION_DELIVERY` 记 FAIL，但库是好的。** 这个 FAIL **只由
   `A19_VALID_DIST` 一项触发**：A19 的分布期望是照 40 episode 测试库写死的，
   400 episode 库的真实分布（median 9.0 / mean 10.31 / max 34 / zero_frac 0.0633 /
   fill_rate 0.107）本身完全合理。核心判据是 M1 的逐样本对拍
   `samples=101066 mismatches=0`——101,066 个样本逐个与 oracle 比，零不一致。
   看到 FAIL 不要以为库有问题。

4. `motion_checks.py` 的 a6 检查只比清单前 40 条，对 400 episode 库属弱检查；
   实际覆盖由 A7（600 段逐字节）、A10（行数公式）、D3（6,832 窗全量逐位）承担。

5. **跨 GPU 架构不保证逐位一致。** 本库的全部逐位结论都是在 A100-SXM4-80GB 上得出的；
   在别的架构上重跑 VAE/encoder 前向可能得到数值上极接近但不逐位相同的结果。
