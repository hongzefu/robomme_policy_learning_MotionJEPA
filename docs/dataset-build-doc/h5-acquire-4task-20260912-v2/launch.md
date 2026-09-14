# 新版四任务 H5 获取溯源补档

## 补档范围

本档案是 2026-09-14 实施合并建库计划时对既有下载的补档，不是本轮重新下载或重新解压的启动记录。用户已在计划中确认有意选择私有数据源，本轮指令为“开始实现该计划 有问题提前问用户 越早越好”。原获取任务已于 `2026-09-13T00:54:31.406631+00:00` 完成；本轮不把事后提交冒充原任务启动提交。

## 数据来源与实际落点

原始获取身份来自 `records/source.json`：`HongzeFu/robomme-4task-h5-20260912-v2`，revision `604f16da36d6b6d175884df8fb687dc08e0a36eb`，manifest sha256 `df992cdf5a768ae0f368e520b0d7be28a68ed4ce6ca6201967cf5d6654cf2bb9`。本地根为 `/scratch/hongze/robomme-4task-h5-20260912-v2/`，`snapshot/`、`extracted/`、`control/` 同级。`snapshot` 保存获取的原件，`extracted` 保存 1600 条 primary H5，全部位于 AWS 本地 NVMe RAID `/dev/md0`。

生成端为私有 `hongzefu/robomme_benchmark_MotionJEPA` 的 `newtask-v2` 分支，提交 `20230d844df46eb58255bd7cdb3cc070baf0383e`，运行编号 `20260912-contract-v3-10`，注入契约 v3。它与公开 `Yinpei/robomme_data_h5` 不同源；同名 BinFill 在新数据中也有 demo 前缀。现有官方 sim 不能用于给这批数据训练的模型作可信评估，用户决定的私有 sim 适配不在本轮范围内。

## 原始启动证据的边界

仅归档原 `control/source.json`、`COMPLETE.json`、`final_verification.json`，以及从固定 MANIFEST 汇总的数量表和源 README 的溯源段。原获取脚本 sha256 由 `final_verification.json` 记录为 `8fedc1f13f764a636f6e2224a9f687445417d85cffe42b4edfe7606a05646a78`；这些记录没有给出可独立还原的原始 Git 启动提交，本轮不补造。原始获取命令与硬件分工不由本档案推断。

## 本轮验证

`merge_v2_h5.load_manifest` 重算 MANIFEST 摘要并与原 source pin 比较；`select_primary` 实算三种 role 数量，并独立要求完整源集四任务各 400 条。28 集冒烟的 `plan` 已对全部 1600 个源 H5 检查字节数、原始顶层组名、连续 timestep、demo 严格前缀、首帧完成标志和 subgoal boundary 键。源文件 SHA256 的独立逐文件重算由正式 merge 完成，结果回写 episode map，不将生成侧提供的摘要误称为本轮实测摘要。
