---
license: apache-2.0
task_categories:
- robotics
tags:
- robotics
- manipulation
- maniskill
- robomme
- hdf5
pretty_name: RoboMME 4-task h5 (20260912-contract-v3-10)
---

# RoboMME 四任务 h5 数据集（运行 20260912-contract-v3-10）

本仓库发布 RoboMME 新值注入专项运行 `20260912-contract-v3-10` 的 h5 轨迹与配套录像。
四个任务 BinFill / RouteStick / VideoUnmaskSwap / VideoRepick，每个任务 400 条正式 h5，
共 **1600** 条 primary、**196** 条 spare 备件、
**1** 条 smoke 样例，另带 **1949** 个 mp4 录像。

## ⚠️ 与官方 `Yinpei/robomme_data_h5` 的形状差别（务必先读）

官方 `Yinpei/robomme_data_h5` 每个包内是**一个合并后的大 h5**（多条 episode 合在一个文件里）。
**本仓库不是**：这里是**逐条 h5**，每个 `.h5` 文件里**只有 `episode_0` 一条轨迹**，
文件名形如 `<任务>_ep<号>_seed<seed>.h5`。下游若按官方形状写了「一个 h5 多条 episode」的读法，
用到本数据集时需要改成「一个目录下多个单 episode h5」。h5 内容本身未做任何合并、改写或重命名。

## 条数分布

| 任务 | 难度 | primary | spare | smoke |
| --- | --- | ---: | ---: | ---: |
| BinFill | easy | 134 | 16 | 0 |
| BinFill | medium | 133 | 16 | 0 |
| BinFill | hard | 133 | 6 | 0 |
| RouteStick | easy | 100 | 15 | 1 |
| RouteStick | medium | 100 | 15 | 0 |
| RouteStick | hard | 100 | 15 | 0 |
| VideoUnmaskSwap | easy | 100 | 15 | 0 |
| VideoUnmaskSwap | medium | 100 | 15 | 0 |
| VideoUnmaskSwap | hard | 100 | 15 | 0 |
| VideoRepick | easy | 134 | 16 | 0 |
| VideoRepick | medium | 133 | 13 | 0 |
| RouteStick | xhard | 100 | 15 | 0 |
| VideoUnmaskSwap | xhard | 100 | 13 | 0 |
| VideoRepick | xhard | 133 | 11 | 0 |

## 生成口径

* 注入契约 **v3**（`meta/configs/injection_contract_v3.json`，sha256 `bfcf4c5984fb89f28c8f04ae918513514ec3d726edb0735d7a22e8587b450da6`）；
* 候选规格按每 100 条一个 **block** 扩容冻结，组内 episode 号连续；
* 每档**实跑目标 × 1.15** 留失败余量（`meta/configs/delivery_400.json`）；
* **严格交付**：每组按 episode 升序取**前 N 条通过**为 primary，其余通过条降级为 spare，未通过条进 `MANIFEST.json` 的 `failures`；
* 每档另有 **50 条只做 env-check、不出 h5** 的额外候选，其结果在 `meta/env_check/**.jsonl` 里 `delivered=true` 的行。

## 目录树

```
README.md              本文件
SHA256SUMS             全部文件的 sha256（不含自身）
MANIFEST.json          机器可读清单：逐条 episode 的散列、归属包、成员路径、录像
tarxz_h5.py            自包含的打包/解包脚本
record_dataset_<任务>_<难度>.h5.tar.xz          14 个正式包
spare/record_dataset_<任务>_<难度>_spare.h5.tar.xz  14 个备件包
videos/<任务>/<难度>/*.mp4                       录像
smoke/record_dataset_RouteStick_easy_smoke.h5.tar.xz  单条样例包
smoke/videos/RouteStick/easy/*.mp4               样例录像
meta/                  小产物副本（交付清单、规格、env-check、日志、配置）
```

每个包解开后是 `record_dataset_<任务>_<难度>/hdf5_files/*.h5` 加一份 `record_dataset_<任务>_metadata.json`。

## 解压

```bash
# 批量（推荐，多进程）
python tarxz_h5.py decompress --input_dir . --jobs 8

# 单个包
tar -xJf record_dataset_BinFill_easy.h5.tar.xz
```

## 校验

```bash
sha256sum -c SHA256SUMS
```

`MANIFEST.json` 主要字段：

* `archives[]` —— 每个包的 `path / role / task / difficulty / bytes / sha256 / member_count`；
* `episodes[]` —— 每条 episode 的 `episode / seed / spec_sha256 / archive / member / h5_sha256 / h5_bytes / timestep_count / videos`，
  其中 `h5_sha256` **逐字抄自生成侧的交付清单**，不是发布时重算的；
* `failures[]` —— 未通过、因而没有 h5 的候选条（仍可能留有录像）；
* `videos[]` —— 每个 mp4 的相对路径、体积与 sha256。

## 关于录像

* **部分 episode 没有录像**（录制中断或帧数不符，`video_status` 非 `complete`）；
* **部分 episode 有第二个 mp4**（`VideoUnmaskSwap` / `VideoRepick` 居多，`BinFill` 也有），文件名以 `success_NO_OBJECT_` 或 `FAILED_NO_OBJECT_` 开头，是同一条 episode 的「无物体」对照渲染；
* **被拒绝（未通过）的候选只有录像、没有 h5**，它们的录像照样收在 `videos/` 下，便于人工复看失败原因。

## 溯源

* 代码仓库：<https://github.com/hongzefu/robomme_benchmark_MotionJEPA>（**私有仓库**，需授权才能访问）
* 分支：`newtask-v2`
* 运行编号：`20260912-contract-v3-10`
* commit：`20230d844df46eb58255bd7cdb3cc070baf0383e`
