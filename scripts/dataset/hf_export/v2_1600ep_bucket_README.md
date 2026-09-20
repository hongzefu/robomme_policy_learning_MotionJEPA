# robomme-4task-v2-1600ep-trainset-20260920-v1

四任务 × 400 episode（共 **1600** 条轨迹）的 VLA 训练库完整交付：packed framesamp 表 +
motion 表 + 源样本 pkl + 归一化统计 + 溯源。**下完这一个 bucket，再取两个匿名可拉的公开
权重，就能直接起跑**本仓库的两条生产训练。

由 `scripts/dataset/hf_export/run_v2_1600ep_dataset_export.sh` 上传，**上传前 / 上传后两遍
独立重算的 sha256 逐行对照**验收，另叠一层 tar 成员抽样与「只用回读副本跑 20 步训练」的
冷启动彩排。

## 这个库是什么

四个 RoboMME 任务 **BinFill / RouteStick / VideoRepick / VideoUnmaskSwap**，每任务
400 条 episode。口径以 `meta/episode_manifest.json` 的 `canonical_order` 为准。

| 项 | 值 |
|---|---|
| episode / timestep / exec 样本 | 1600 / 1,192,918 / 605,611 |
| framesamp 布局（训练直读） | `framesamp-8x8-v1`，`image_emb_8x8` 每行 64×2048 bf16 |
| framesamp 布局（对照档） | `framesamp-4x4-v1` |
| motion 布局 | `motion-768-grid16-demopad17-v1`，71,316 行 × 3072 B |
| 源 h5 | 同账号 bucket `HongzeFu/robomme-4task-h5-20260912-v2`（revision `604f16da36d6b6d175884df8fb687dc08e0a36eb`） |
| motion encoder | `wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder`，sha256 `0c198629…` |
| Wan VAE | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，state sha256 `9980d252…`，恒 fp32 |
| 建库硬件 | 8 × A100-SXM4-80GB |

用这个库训出来的两个模型在同账号下：`HongzeFu/robomme-vla-modul-60k-v1`（512 modulation，
motion 关）与 `HongzeFu/robomme-vla-modul-motion-80k-v1`（512+motion modulation）。
motion encoder 本体在 `HongzeFu/motionjepa-wan-full1600-72ep-v1`；**训练不需要它**（见下）。

## 布局

```
framesamp-8x8/                       ★ 训练直读，原样直传、不必解包
  image_emb_8x8/part_000..031.bf16.bin     32 片，9.78–10.21 GB/片，共 291.3 GiB
  pos_emb_8x8.f32.bin                      452,984,832 B
  state_emb.f32.bin                         38,173,376 B
  meta/store_meta.json                     唯一契约（status=verified）
  meta/row_digests.blake2b.bin             逐行摘要（训练不读，留作自证）
  meta/pack_progress.jsonl                 打包账本（训练不读）
framesamp/                           4×4 对照档，同构布局，训练不用
motion/                              ★ motion 开启态训练直读
  motion_token.f32.bin                     219,082,752 B = 71,316 × 3072
  meta/store_meta.json  meta/motion_index.json
  meta/row_digests.blake2b.bin  meta/pack_progress.jsonl
meta/
  episode_manifest.json                ★ 契约锚点，一个字节都不能改
  input_manifest.json
source/
  data_tars/data-00000..00147.tar      ★ 605,611 个 {idx}.pkl，**必须解包才能训练**
  features/episode_*/token_emb_*.npy   8 个抽样复验点名文件（见「已知事项」）
  meta/                                建库侧 provenance / stats
assets/robomme/norm_stats.json       ★ 训练必需的归一化统计
wan-latents_tars/   oracle_tars/   motion-tokens_tars/    中间产物与对拍材料，训练不用
runner/{restore.sh,train-60k.sh,train-80k.sh}            异地一键恢复与起跑
SHA256SUMS.pre.txt                   上传前重算的全量清单（覆盖除自身外的全部字节）
```

`source/data_tars` 的成员名是 `data/{idx}.pkl`，在 `source/` 下解开即还原
`source/data/`。tar 的 `mtime/uid/gid/uname/gname` 全部归零、成员定序写入，逐位可复现。

## 异地怎么用（8×A100，从零到起跑）

前提：仓库 clone 到 **`/scratch/hongze/robomme_policy_learning_MotionJEPA`**。
这个路径不是随便挑的——`scripts/training/paths.sh` 的前缀白名单只认三个常量前缀，
且两个 `store_meta.json` 里写死的 `manifest_path` / `source_dataset_root` 正是这个路径。
放别处也能跑，但必须显式 export 两个环境变量（`runner/restore.sh` 里有说明）。

```bash
bash runner/restore.sh      # hf sync + sha256 全量校验 + 解 data_tars + norm_stats 归位
                            # + fetch_assets.py 取 pi05_base 与 paligemma_tokenizer
bash runner/train-60k.sh <run_name>    # 512 modulation（motion 关）
bash runner/train-80k.sh <run_name>    # 512+motion modulation
```

**训练只额外需要两个外部权重**，都从 `gs://` **匿名**可拉，不需要任何 token：
`pi05_base`（12.44 GB，初始权重）与 `paligemma_tokenizer`（4.26 MB）。

**SigLIP、Wan VAE、MotionJEPA encoder 训练都不需要。** motion token 已经离线烘焙进
`motion/motion_token.f32.bin`；history config 里那行
`source_run: wan-full1600-…#encoder` 只是身份字符串，加载时与 `motion/meta/store_meta.json`
的 `provenance.encoder` 做四字段比对，不打开任何权重文件。

### 自验

```bash
hf sync hf://buckets/HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1 ./local-dir
cd local-dir && sha256sum -c --strict SHA256SUMS.pre.txt     # 覆盖全部字节
```

解包后还可以逐源文件再对一遍：`checksums/sha256-source-files.txt` 的每一行对应一个
tar 成员名，解开 `source/data_tars/*.tar` 后在 `source/` 下 `sha256sum -c` 即可。

## 契约与绑定

| 锚点 | 值 | 谁引用它 |
|---|---|---|
| `episode_manifest.json` 内部自哈希 | `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918` | 两个 framesamp store 与 motion store 的 `manifest_sha256`；加载时现场重算比对，不等即拒 |
| `episode_manifest.json` 整文件 sha256 | `df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482` | 本 bucket 的 `SHA256SUMS.pre.txt` |
| `framesamp-8x8/meta/store_meta.json` | `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` | 两条 run 的 `motion_provenance.json`、两个 ckpt bucket 的 README |
| `framesamp/meta/store_meta.json` | `6fef1451373cd2024c80bde9b498a00681b892318dd384030d0c11f257f66c2d` | — |
| `motion/meta/store_meta.json` | `d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268` | 80k run 的起跑 preflight |
| `motion/meta/motion_index.json` | `d68696a0c260d689e05f600b135d8372b7bf953a491be41cf364d703068b7864` | motion store_meta 的 `motion_index_sha256`；加载时现场重算比对 |
| `assets/robomme/norm_stats.json` | `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173` | 两条 run 的起跑 preflight |

建库验收判定行（原文）：

```
VERIFY_PACK=PASS scanned=1192918 mismatches=0        （framesamp-8x8，status=verified）
VERIFY_MOTION=PASS scanned=71316 mismatches=0        （motion，status=verified）
WAN_BITEXACT  8632 窗（含全部 1600 个补帧窗）frame/latent/metadata 失配均 0
ENCODER_BITEXACT  71316 行、0 失配
```

**两个 store 的 `status` 都是 `verified`，且包内不含任何 `meta/pack.lock`** ——
读侧对这两点各有一道硬闸（`status != "verified"` 或 `pack.lock` 存在即拒绝加载）。

## 已知事项（诚实清单）

1. **`source/data/` 是硬依赖，不是可选项。** `framesamp-8x8` 只装 memory 路的 SigLIP 特征；
   当前观测的 `image / wrist_image / state / actions / prompt` 仍然逐样本从
   `source/data/{idx}.pkl` 读。不解开 `source/data_tars/` 一步都训不动。
2. **`source/features/` 只有 8 个文件，这是刻意的、不是遗漏。** 完整 features 有 674 GiB，
   已被 packed 表取代、训练不读。但 `framesamp-8x8/meta/store_meta.json` 的
   `source_spot_sha256.entries` 共 16 条抽样复验点，加载时按进程 pid 轮转抽一条，
   其中 8 条落在 `features/` 下 —— 少一个就会在「源库抽样文件缺失」处拒跑。
   本库带的正是这 8 个，名单由建库侧 store_meta 现读、未硬编码。
3. **`logs/`（35 个文件）刻意未上传**：含完整 `uv run --project …` 命令行与 venv 内部布局，
   不参与任何 sha 绑定、不被任何脚本读，验证价值为零而泄露密度最高。
4. **本库任何数据字节都未被修改。** 下面这些内容原样保留并在此披露：
   - 两个 framesamp store_meta、motion store_meta、`meta/*.json` 里的构建机绝对路径
     `/scratch/hongze/robomme_policy_learning_MotionJEPA/...`；
   - `motion/meta/store_meta.json` 的 `mj_repo_path` 指向一个 NFS 路径与私有仓库名
     `MotionJEPA`（该仓库匿名访问 404）；
   - `motion-tokens_tars/` 与 `wan-latents_tars/` 内 metadata 的构建机内部 DNS 名
     `ip-10-242-11-177.us-west-2.compute.internal`（RFC 1918 私有地址，对外不可路由）
     与 GPU UUID。
   它们全部进了 `manifest_sha256` / `motion_index_sha256` / 各 store_meta 的哈希绑定范围，
   为掩掉一个路径而改字节，会让这个库再也过不了自己的加载闸。处置只有「排除」与「披露」。
5. **`wan-latents_tars/`、`oracle_tars/`、`motion-tokens_tars/` 与 `framesamp/`(4×4) 训练不用。**
   它们在这里是为了让异地能独立重算 motion 表、跑 4×4 对照，以及做逐位对拍。
   只要训练的话，`--include` 掉这四项可少下约 155 GiB。
6. **不在本库内（有意排除，不是遗漏）**：原始 h5（在 `HongzeFu/robomme-4task-h5-20260912-v2`）、
   `source/features/` 其余部分、外部权重（`fetch_assets.py` 匿名拉）、
   MotionJEPA encoder 本体（在 `HongzeFu/motionjepa-wan-full1600-72ep-v1`）。
7. **本库的数字不与官方上游 `mme_vla_suite` 逐项可比**：训练步数 × batch 与官方
   （80k × 64）不同，optimizer 更新次数不同。引用时须带此声明。

## 许可

沿用上游 RoboMME 的 Apache-2.0。
