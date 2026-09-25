# 公开来源pin与16集构建冒烟全部通过

## 1. 结论与用户口径

公开16个H5重新hash与已提交pin一致，全集1600集/768897总帧/476857执行样本身份通过；16任务各首集的source、4×4 pack及全量verify全部成功。冒烟实际为**16集、7898总帧、5018执行样本**，finalize复算1024条最大绝对差0，packed扫描7898行、失配0。两次任务及tee退出均为0，smoke七个阶段也全部退出0。

用户原话：「恢复计划中的建库，严格按前置闸门推进」。输入侧另有明确决定：「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」及「允许这四键缺失与 None 等价」。后者仅限`motion_emb/motion_pos/motion_mask/mem_order`缺失与显式None，不扩展到其他键或非None值。本轮是构建冒烟，不是训练；此前3样本CPU诊断也不等同于两个正式库的输入对拍。

## 2. 版本、环境、会话与实际调用

起跑Beta固定为 **`42b91cd96499bd51fcb6acaeedd642b368dbefeb`**（`commitV11.10Beta`）。两个会话起末均检查clean HEAD及相同提交；8个worker、finalize和pack元数据也记录同一Beta。归档发生在任务全部结束之后，不将归档提交写成运行版本。

环境为AWS单机8×A100-SXM4-80GB，`/scratch`为`/dev/md0` XFS本地NVMe RAID。构建及复算均记录JAX/jaxlib 0.5.3，packer记录Python 3.11.15、NumPy 1.26.4、ml_dtypes 0.4.1。8个worker位于同一主机，分别绑定物理GPU0–7；原始[provenance](records/source_provenance.json)中的8条host记录不是8台不同机器。

| 任务 | 完整tmux会话名 | pane/包装PID | UTC开始→结束 | 墙钟跨度 |
|---|---|---:|---|---:|
| 来源前检 | `pub16-pin-20260925T203220Z` | 3787720 | 2026-09-25 20:32:35Z→20:56:40Z | 1445秒 |
| 16集构建冒烟 | `pub16-smoke16-20260925T205655Z` | 3798932 | 2026-09-25 20:58:03Z→21:09:35Z | 692秒 |

实际tmux调用采用`tmux new-session -d -s <上述完整名称> -c /scratch/hongze/robomme_policy_learning_MotionJEPA bash -c <内联正文>`；没有独立wrapper脚本。共同环境、实质命令、实际执行顺序与退出处理已回填[launch.md](launch.md)。生产入口由Beta还原；运行时内联shell文本没有被误称为Git里已有文件，也没有作为脚本副本归档。

来源前检实际先验证两个参考JSON的文件SHA，full校验`siglip_params`，然后scan1600、hash16、来源绑定；这与最初骨架中hash/scan先后的区别已明确记录，未遗漏任何步骤。参考input文件SHA为`36c33bd6fdb012b72b692596c83982d2a8dc0f56e6ef7e45781ec9ebc954c640`，参考episode文件SHA为`43d9a6fbcef7eb259c12e74b7c62898d33961a6470c40586cbed1be51d1024c8`；后者不同于清单内容规范SHA，二者不混用。已提交参考文件不重复复制。

## 3. 阶段结果与判据

| 阶段 | 实际结果 | 时间口径 |
|---|---|---|
| SigLIP资产full校验 | `ASSETS=PASS assets=1 mismatches=0` | 资产工具记录3.82秒 |
| 全集scan | 1600集、768897总帧、476857执行样本 | 扫描器记录189.3秒 |
| 公开16文件hash | 全部size/SHA命中参考pin | hash末项累计1250秒 |
| 来源与身份绑定 | `INPUT_PIN=PASS files=16`，全集身份PASS | 来源任务最终退出0 |
| scan16/identity16 | 16集、7898总帧、5018样本，身份PASS | 20:58:03Z→20:58:04Z；scan自身0.6秒 |
| SigLIP | `STAGE_DONE stage=siglip workers=8 items=16 elapsed=181s` | 20:58:04Z→21:01:05Z |
| finalize | 16文件SHA同源；feature目录缺失0；pkl 5018/5018；8 sidecar覆盖16集；claim残留0；1024/1024复算最大差0 | 21:01:05Z→21:09:32Z，507秒，包含hash及复算 |
| pack4×4 | 13个image part，完整小表落盘 | 工具记录2秒 |
| 全量verify | `VERIFY_PACK=PASS scanned=7898 mismatches=0` | 工具记录1秒 |
| 实际体积测量 | `SMOKE_BUILD=PASS` | 21:09:35Z完成 |

冒烟manifest规范SHA为`938cb31f91b044cc6974346854c9da57bb3f4fdecbdc9385fdb63ea1dbf39724`；全量前检manifest规范SHA为`fb1bdbcbcc176d15475332b728fa218bb545b4a8c9dcaa9ce1ec9e0bb17f174f`。full与smoke的身份关联见[smoke_identity.json](records/smoke_identity.json)。[packed元数据](records/packed_store_meta.json)为`framesamp-4x4-v1`、`status=verified`，绑定同一smoke manifest；没有将`pack`完成但未verify当作成功。

输入H5来源、16个首集可构造、source抽样复算和packed全量一致性分别有对应证据。finalize的1024条抽检不等于对全库每个原始图像/腕部图像字段做完了独立逐字节验证；本轮也没有比较上游训练loader或运行模型训练。

各worker日志中的`skipped`对应`--resume`检查时已完成的episode；动态领取失败的跳过与这个计数分开。验收依据是8份sidecar合计16个已完成episode、完整产物和零残留claim，不把每个worker记录的已处理/已见完成列表简单相加当作实际重复构建。原始worker计时和速率字段保留在记录中，但没有GPU稳态利用率及配套采样证据，**这里只报告阶段耗时，不给吞吐、GPU利用率或性能瓶颈结论**。

## 4. 实测体积与两正式库剩余预算

CPU测量逐文件累计`st_size`和`st_blocks*512`；后者是文件实际分配量，不含目录/inode等全部文件系统开销。结果见[smoke_sizes.json](records/smoke_sizes.json)。本轮没有独立采样临时写盘峰值，128GiB临时等分项仍是保守预算。

| 项目 | 逻辑字节 | 文件实际分配字节 | 文件数 |
|---|---:|---:|---:|
| source | 6747074966 | 6781734912 | 12942 |
| packed（含其meta） | 576977634 | 576983040 | 18 |
| 两者合计 | 7324052600 | 7358717952 | 12960 |

共核对7898个feature和5018个执行pkl；最大单总帧feature分配量F=`606208 B`，最大单执行pkl分配量P=`397312 B`。它们与冒烟前取样使用的值一致。source+packed约7.36GB（6.85GiB）保留原位置，没有自动删除；库根其他manifest/日志等另占少量空间。

[post_smoke_budget.json](records/post_smoke_budget.json)绑定本次smoke_sizes文件SHA`ab11bec7d307fcf5cf7bca394ecbe959d98e4532261f402c19a4e07a0fb857b5`。以下只预算**尚未构建的两个正式库**；冒烟已计入当前磁盘占用，不重复加到剩余需求里。

| 剩余预算项目 | 字节 | 性质 |
|---|---:|---|
| 16任务正式库 | 706174072880 | 按当前最大F/P、总帧/执行样本及packed项外推 |
| counting正式库 | 202175603728 | 同口径外推 |
| 两库数据加10% | 999184644269 | 外推及开销余量，不是正式库体积实测 |
| 缓存与环境 | 68719476736 | 64GiB保守预算 |
| 验证checkpoint及临时产物 | 51539607552 | 48GiB保守预算 |
| 日志与其他 | 17179869184 | 16GiB保守预算 |
| 保留量 | 322122547200 | 300GiB既定要求 |
| 所需可用空间 | **1458746144941** | 约1358.5632GiB |
| 实测可用空间 | **1784546611200** | 约1661.9885GiB |
| 预算余量 | **325800466259** | 当时预算PASS，后续仍逐阶段重测 |

预算通过不表示两个正式库已经存在或已验证，也不固定未来门槛；正式起跑前仍核对当前代码、输出路径和实际剩余空间。

## 5. 归档、范围边界与下一步

共归档29个文件：18份实测JSON/JSONL原字节副本、10份清洗日志及1份[archive_checks.json](records/archive_checks.json)。该核验文件记录每个原始来源、归档路径、字节数和SHA。核心记录为[input pin](records/input_manifest.json)、[全量episode清单](records/full_episode_manifest.json)、[smoke清单](records/smoke_episode_manifest.json)、[source stats](records/source_stats.json)、[source provenance](records/source_provenance.json)、[packed meta](records/packed_store_meta.json)、[pack进度](records/pack_progress.jsonl)、身份/体积/预算三份JSON及8份`source_shards/shard*of8.json`。

清洗日志为[来源前检](records/logs/source-preflight.summary.log)、[smoke构建](records/logs/smoke-build.summary.log)与8份`logs/siglip-gpu*.summary.log`。清洗只把回车归一为换行并剔除`%|`类tqdm中间态；本轮匹配删除0行，103条关键事件在清洗前后逐行一致。source唯一终态`EXIT_CODE=0`；smoke七条`STAGE_EXIT_CODE`、一条`FINALIZE_EXIT_CODE`及两任务的`TASK_EXIT/TEE_EXIT/EXIT_CODE`数量和值全部核对，8个worker各保留一条`SHARD_DONE`。原始日志和数据仍在`v1-store`，没有归档脚本、yaml、H5、features、packed数组或权重。

本轮没有构建异常或非零阶段；常见uv配置弃用提示原样保留，不当作失败。已完成的是公开来源前检和16个首集构建，未完成的是两个正式库、完整子集比对、正式输入/训练对拍及perf/80k。下一步在归档提交和clean代码锚点就绪、路径与预算复核后按计划推进16任务正式库，再推进counting；保留本smoke库，训练仍须全部独立闸门。
