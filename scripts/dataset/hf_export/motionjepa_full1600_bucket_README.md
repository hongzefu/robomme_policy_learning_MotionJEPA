# robomme-4task-motion-full1600-20260914-v1

四任务 × 400 episode（共 **1600** 条轨迹）的 Wan latent 派生数据集：逐段 chunk latent +
chunk 级 motion 表 + 完整溯源。由 `scripts/dataset/hf_export/run_motionjepa_full1600_export.sh`
上传，**上传前 / 上传后两遍独立重算的 sha256 逐行对照**验收，另叠一层发布清单自带 sha256 的
逐条核对。

## 这个库是什么

源数据是四个 RoboMME 任务 **BinFill / RouteStick / VideoUnmaskSwap / VideoRepick**，每任务
400 条 episode，来自 `robomme-4task-h5-20260912-v2`（同账号下的 bucket，revision
`604f16da36d6b6d175884df8fb687dc08e0a36eb`）。

本库是从那批 h5 派生出来的 **Wan 2.1 T2V-1.3B VAE latent**：每条 episode 按 demo / exec 两段
切开，各自过 VAE encode 得到 chunk latent。生成这批数据需要全程 GPU（本轮用 8×A100-80GB），
把它放上来正是为了让异地复刻不必重跑这条链路。

**口径提醒**：`full1600` 指**总共 1600 个 episode = 4 个任务 × 每任务 400 episode**。

## 布局

```
chunk_motion.npz                  chunk 级 motion 表（304 MB）
wan_chunk_latents/
  <Task>_ep<0..399>_demo.bin      demo 段 latent
  <Task>_ep<0..399>_exec.bin      exec 段 latent      （两者合计 2800 个，均值 171 MB）
  metadata.json                   逐段的形状 / dtype / 偏移
DATASET_NOTES.json                建库口径与已知事项
FINALIZE_DONE.json                收尾阶段的逐项统计
PUBLISHED.json                    **发布清单**：2804 个文件各自的 sha256 + identity + policy
control/                          溯源（源 pin、模型 pin、环境、策略、逐段审查图）
  content_hashes.tar              2800 个逐段内容哈希（打包传输，解开即用）
SHA256SUMS.pre.txt                上传前重算的全量清单
```

### 怎么自验

```bash
hf sync hf://buckets/HongzeFu/robomme-4task-motion-full1600-20260914-v1 ./local-dir
cd local-dir && sha256sum -c --strict SHA256SUMS.pre.txt
```

`PUBLISHED.json` 的 `files` 字段是建库侧当时写下的独立清单，可以再对一遍：清单内 2804 条与
`SHA256SUMS.pre.txt` 的对应行必须逐条相等（上传链路的阶段 2 已经跑过这个核对，判定行
`PUBLISHED_SHA_MATCH`）。

## 身份与可复现性

`PUBLISHED.json` 的 `identity` 段钉住三件事：

- `code_sha` —— 建库代码的内容哈希
- `model_pin` —— Wan VAE 的 revision `0fad780a534b6463e45facd96134c9f345acfa5b` 与两个权重
  文件的 sha256（`config.json` `f0c1cc1d…`、`diffusion_pytorch_model.safetensors` `d6e524b3…`），
  另有前向状态指纹 `state_sha256`
- `reference` —— 参考实现各文件的 sha256 闭包

`policy` 段记录本轮的筛选策略（`equal-drop-20260914-v826`）与其决策哈希。

## 已知事项（公开前体检的裁决）

上传前用 `scan_hygiene.py` 逐行扫过全部待传文本文件，结论：**没有任何凭据、token、私钥或
集群内部地址**（对应模式命中数为 0）。以下两类命中经确认后**原样保留**，在此披露：

1. **构建机绝对路径**：`control/` 下的溯源 JSON（尤其 `scope.json`、`preview.json`、
   `run_manifest.json`、`launch.json`）保留了建库当时的本机绝对路径。这些路径对外不可达、
   不含任何凭据，而 `control/` 的全部价值就是溯源——抹掉路径等于抹掉它要记录的东西。
2. **构建机内部主机名**：`wan_chunk_latents/metadata.json` 的每条记录都带一个 `hostname`
   字段，值为构建机的 AWS VPC 内部 DNS 名（形如 `ip-10-242-11-177.us-west-2.compute.internal`，
   全文件共 2800 处，同一台机器）。它记录的是「这段 latent 是在哪台机器上算出来的」，属于
   可复现性信息的一部分。其中的 `10.x` 是 RFC 1918 私有地址，对外不可路由、不是凭据，也不
   构成可达的攻击面。该文件是发布清单的必需件——下游要靠它读每段 latent 的 shape / dtype /
   偏移，排除它等于让 2800 个 `.bin` 无法使用。
3. **MotionJEPA 仓库名**：`control/environment.json` 与 `control/policies/` 下共 3 处出现
   `MotionJEPA` 这个仓库名（作为解释器路径与策略来源的一部分）。该名字在项目的公开仓库文档里
   本就写着，泄露增量为零。

**本库任何数据字节都未被修改。** 这是刻意的：`PUBLISHED.json` 的 `acceptance_sha256` 与各文件
sha256 是这批数据自己的锚点，为掩掉一个目录名而改字节会让它再也过不了自己的校验。体检只有
「排除」与「披露」两种处置。

不在本库内（有意排除，不是遗漏）：建库中间产物（喂给 VAE 的解码帧）、为验证正确性另跑的
对照产物与其 fp32 变体、小规模试跑产物、建库日志，以及每个 latent 的逐文件边车
（`*.sha256` / `*.complete.json`）——边车的信息已由 `PUBLISHED.json` 与 `content_hashes.tar`
完整覆盖。

## 许可

数据沿用上游 RoboMME 数据集的许可条款（Apache-2.0）。
