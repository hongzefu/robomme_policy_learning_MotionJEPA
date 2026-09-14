# 本库已从 dataset repo 迁移为 bucket

**原位置**：`https://huggingface.co/datasets/HongzeFu/robomme-4task-h5-20260912-v2`
（revision `604f16da36d6b6d175884df8fb687dc08e0a36eb`）
**现位置**：`hf://buckets/HongzeFu/robomme-4task-h5-20260912-v2`
**迁移日期**：2026-09-14　**原 dataset repo 已于迁移校验通过后删除。**

## 内容有没有变

**没有。** 原 repo 的全部 **2057 个文件逐位原样**都在这个 bucket 里，路径不变（包括
`README.md`、`MANIFEST.json`、`SHA256SUMS`、`.gitattributes`）。本文件 `MIGRATION.md` 是迁移
时唯一新增的对象，除它之外 bucket 与原 repo 是同一份字节。

其中 1978 个文件（126.59 GB，即全部 `.tar.xz` 与 `.mp4`）是用 Hub 的**服务端按内容哈希复制**
搬过来的——字节根本没有离开过 Hub，两侧 xet hash 逐条相等；其余 79 个普通文件（22 MB）经
客户端中转。迁移后做了全量回读并从零重算 sha256，与原 repo 自带的 `SHA256SUMS` 对拍通过。

## 怎么取用（和以前不一样的地方）

```bash
# 全量
hf sync hf://buckets/HongzeFu/robomme-4task-h5-20260912-v2 ./local-dir
# 只取某个难度包
hf sync hf://buckets/HongzeFu/robomme-4task-h5-20260912-v2 ./local-dir \
  --include 'record_dataset_BinFill_easy.h5.tar.xz'
# 单个文件
hf cp hf://buckets/HongzeFu/robomme-4task-h5-20260912-v2/SHA256SUMS ./SHA256SUMS
```

取回后照旧可以自验：`sha256sum -c SHA256SUMS`。

**以下方式不再可用**（bucket 不是 git 仓库）：`load_dataset("HongzeFu/robomme-4task-h5-20260912-v2")`、
`hf_hub_download(..., repo_type="dataset")`、`snapshot_download(...)`，以及任何钉住
`revision=604f16da…` 的引用。bucket 是可变对象存储，**没有版本历史**。

## 迁移损失了什么（如实列出）

- **dataset card 与机器可读元数据**：原 `README.md` 的 YAML front matter（`license: apache-2.0`、
  `task_categories: [robotics]`、`tags: [robotics, manipulation, maniskill, robomme, hdf5]`、
  `pretty_name`）在 bucket 里不再被渲染或索引。front matter 原文仍原样保留在 bucket 的
  `README.md` 文件开头。**许可仍然是 Apache-2.0**，这里再声明一次，因为机器已经读不到它了。
- **datasets 搜索与 dataset viewer / Croissant 元数据**：bucket 不进 HF 的数据集索引。
- **revision 历史与讨论区**：原 repo 的 3 个 commit 与讨论区随删除一并消失。迁移前已把
  revision 元数据、文件树、提交历史、讨论区、Croissant 描述与 README 原文完整抓进了项目
  留档（`docs/dataset-build-doc/hf-export-h5v2-rehost-20260914/records/`）。

## 这批数据是什么

见同目录下的 `README.md`（原 dataset card 正文，未作改动）：RoboMME 新值注入专项运行
`20260912-contract-v3-10` 的四任务逐条 h5 轨迹（BinFill / RouteStick / VideoUnmaskSwap /
VideoRepick，各 400 条 primary，共 1600 条）、196 条 spare 备件、1 条 smoke 样例与 1949 个
mp4 录像。**注意每个 `.h5` 里只有 `episode_0` 一条轨迹**，与官方 `Yinpei/robomme_data_h5`
的「一个 h5 多条 episode」形状不同。
