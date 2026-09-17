# collate 共享内存：验证结果

本轮实施与验收完成：相同配置下，8 卡稳态步时 **1.772729 → 0.972660 秒，提速 1.822558 倍**；
GPU 利用率均值 **53.32% → 97.87%**。20 批输入对拍和 100 步三侧训练对拍全部逐位一致，
完整性守卫通过。改动保存在本地分支 `v2-motionmem-collate-shm`，尚未推送或合并。

## 速度与测量口径

AWS 单机 8 × A100-SXM4-80GB，底层为 `/dev/md0` 本地 NVMe RAID、XFS。
配置 `mme_vla_suite_b128_60k`，history 为 `perceptual-framesamp-modul-8frame-8x8.yaml`，
motion 关闭；新库 `4task-v2-1600ep-604f16da/framesamp-8x8`，global batch 128、worker 16、
fsdp_devices 8、mesh (1,8)。两侧各运行 300 步，串行独占八卡，独立 JAX 缓存。
源码、全部路径、指纹、覆盖参数和前后数据流见 [起跑记录](launch.md)。

稳态窗口为 step 100→290，共 190 步；均值按整个窗口的墙钟时间计算，GPU 每 500 ms 采样。
预热区间为前 100 步，编译与启动不计入稳态结果。详细数据见
[analysis.json](records/new/analysis.json) 和两个侧目录的原始采样 CSV。

| 指标 | 改前 old | 改后 new |
|---|---:|---:|
| 平均步时 | 1.772729 s | 0.972660 s |
| 吞吐 | 72.205 samples/s | 131.598 samples/s |
| GPU 利用率均值 | 53.3169% | 97.8747% |
| 利用率为 0% 的采样占比 | 42.4180% | 0% |
| 窗口内 GPU 采样记录数 | 5368 | 2952 |
| 慢区间数 | 0/19 | 0/19 |
| 慢区间利用率均值 | 无命中区间 | 无命中区间 |
| 其他区间利用率均值 | 53.3169% | 97.8747% |

慢区间沿用历史定义：10 步区间的平均步时超过全部区间中位数的 1.5 倍。
中位数仅用于这个分层阈值；本轮以平均步时、吞吐和利用率均值作结论。
该分层分辨率为 10 步，不能据此断言每个单步都没有抖动。
旧侧与历史 1.818 秒相差 **2.4902%**，满足 ±5% 的复现门槛。
两侧均完成末步 299 的 checkpoint 保存并 `EXIT_CODE=0`，驱动正常结束。

```text
COLLATE_SHM_SPEED old=1.772729 new=0.972660 speedup=1.822558
OLD_REPRO=PASS relative_deviation=0.024902
```

## 输入和训练一致性

第一块在各侧自己的 uv 环境运行 `compare_collate_paths.py dump --batches 20 --workers 16`，
batch 128、seed 42。每批实际 513319168 B，12 个数组键及 4 个 None 全部一致，
前 2560 个已消费样本索引完全相同。原始记录位于 `records/input-base/`、`records/input-temp/`。

```text
COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0
INDEX_SEQ=PASS n=2560
```

`uv run --no-sync pytest scripts/training/tests/test_collate_shm.py -q`：**4 passed in 10.65s**。
覆盖 0/2 worker、真实 spawn 共享存储、七种叶子类型、None、所有 bf16 位模式及视图生命周期。
CPU 导出时前三批预热后的取批等待为 old=0.783745s、new=0.002647s；逐批摘要计算会与
预取重叠，因此该等待值不用于外推训练吞吐，速度结论只取上面的真实八卡训练。

第二块严格使用确定性设置，a1/a2 为改前副本、b 为改后副本，三侧串行各 100 步。
a2、b 对 a1 的环境指纹检查均 `BASELINE_ENV=PASS`。每侧都有连续 0…99 的 100 行标量、
摘要步骤 `[0,1,2,24,49,99]` 各六行；完整状态每份含 201 个叶子且全部有限。
索引实际各记录 17024 个（包括预取余量），守卫明确覆盖训练所消费的前 12800 个。

两组比较均得到以下判定，完整原文见 [a1/a2](records/guard-summary/compare-a1a2.log)
与 [a1/b](records/guard-summary/compare-a1b.log)：

```text
SCALARS steps=100 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
STATE_DIGEST rows=6 mismatch=0
BATCH_DIGEST rows=6 mismatch=0
CANON_CHECK=PASS steps=6
INDEX_SEQ=PASS n=17024（共同前缀逐个一致, steps≈100）
DET_CHECK=PASS
COLLATE_SHM_GUARD=PASS scalars_steps=100 index_n=12800 batch_digest_rows=6 state_digest_rows=6 sides=3 pairs=2
```

因此本轮规定的输入与训练等价判据全部满足。没有使用 `--null-pair`、量化退路或状态数组落盘。
三侧均正常退出，归档副本上的 `guard_finish_check.py` 也重新运行并通过。

## 版本与实际启动

| 角色 | 完整提交 |
|---|---|
| 改前 A_HEAD | `7a681e2cf3f21bdab3f0b27ea44d19b0adfd0ee7` |
| C0 基准工具 | `82d73f40aa8a1d4a8099955167882be52f8fcb5f` |
| C1 数据路与测试 | `3868cc943124d5a47e1be832c130f620ec056b3c` |
| C2 起跑锚点 | `24f8cc911c9973e6617d7f3144bb73e4cbd6e4de` |
| C3 结果归档 | 本结果文件首次入库的提交；与前述提交分别保留，不改写历史 |

a1/a2 的 `source_head` 是 A_HEAD，b 是 C2；三侧 `tool_head` 都是 C2，
`source_status` 和工具 `start_status` 都为空。改前训练源码与 `e4dc733` 零差异；
C0 不改 `src/`，C1 的 `src/` 差异只在 `openpi/training/data_loader.py`。
没有修改模型、训练超参默认值或原 `_collate_fn` 的 numpy 语义。

实际启动命令如下，脚本内容和起跑时的 SHA-256 在起跑记录及 `records/runners/` 中保留：

```bash
tmux new-session -d -s cs-8gpu \
  'bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/cs-8gpu-driver.sh 7a681e2cf3f21bdab3f0b27ea44d19b0adfd0ee7 24f8cc911c9973e6617d7f3144bb73e4cbd6e4de'
tmux new-session -d -s cs-guard \
  'bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/cs-guard-runner.sh 7a681e2cf3f21bdab3f0b27ea44d19b0adfd0ee7 24f8cc911c9973e6617d7f3144bb73e4cbd6e4de'
```

两会话均自然退出，全部训练退出码与最终驱动退出码为 0；八张 GPU 收尾时显存占用均为 0 MiB。
只读日志监听器在完成后单独终止，其管道状态 143 与训练退出码无关。
其他 tmux 会话未清理。

## 耗时与计划外事项

速度侧 old 为 **13 分 45 秒**，new 为 **9 分 23 秒**（均含启动、编译、保存与收尾）。
确定性阶段 UTC **17:08:08→18:58:55，共 1 小时 50 分 47 秒**。
a1/a2/b 分别约 48 分 21 秒、32 分 42 秒、29 分 41 秒；各侧末时取其最后一条退出记录
写入后的原始日志 mtime，归档清单保留纳秒值。整个 GPU 验证流程为 UTC 16:44:42→18:58:55，
共 **2 小时 14 分 13 秒**。

18 次完整状态摘要函数累计 **4858.303 秒（80 分 58 秒）**，远高于原计划估计。
各次耗时保留在三个 `param_checksums.jsonl`。首次两次为 489.586 / 467.405 秒，
后续有所下降但并非固定成本，不能把最初外推当成最终实测。
10 秒非阻塞 py-spy 采样的 347 个有效样本中，346 个位于 `np.isfinite(arr).all()`
表达式及其归约中，说明该局部窗口的慢点在额外有限值检查。
这是局部采样，不是整份摘要的完整耗时分解；更细的内存或 dtype 原因尚未验证。
诊断方法、限制和原始样本见 [现场诊断](records/guard-summary/slow_checksum_diagnosis.md)。
没有把这些诊断步时混入速度结论，也没有修改在途校验代码。

执行时另外处理了三处计划与实际环境的差异：主副本已干净，C0 默认模式应 PASS；
原 CPU 单步命令 batch=2 与默认 fsdp=4 冲突，按用户确认改为真实配置与训练前入口守卫自检；
原 `.gitignore` 的 `/v1-store/` 不匹配符号链接，按用户确认在本地 `.git/info/exclude`
加精确 `/v1-store`。其余源文件仍受 clean 检查约束。C0 的主副本、base、temp 都
`PREFLIGHT=PASS n=25`，脏副本仅因 REPO_CLEAN 被拒绝，错误源码路径也被拒绝。

输入导出工具首跑因 `V1_STORE` 未 export 而退出，修正为默认当前副本的 v1-store 后重跑成功；
只移除了本轮创建的空目录，保留首次失败日志。同期计划提及的 `hf-modul60k-export`
会话在本轮开工时已结束，不能将它写成实测干扰。

## 用户决定、归档与清理

用户本轮原话：「开始执行 有问题问用户越早越好 结束后报告 问用户是否合并」。
三项前置调整分别得到「同意，按实际状态验证」「同意，C0 检查入口，后续实跑训练」
「同意，添加本地精确忽略规则」。发现摘要耗时超预估后，用户明确答复
**「继续完成原定验证，允许延长占用」**。推送、合并及删除两个工作副本尚未授权。

归档包含两侧速度记录、三侧全部标量/摘要/索引/环境指纹、两份完整比较日志、
完整性判定、CPU 输入对拍、C0 自检、现场诊断及实际驱动。
`archive_manifest.json` 的 **73 项**文件与来源哈希已全部核对；清洗日志保留了全部
训练步骤行、状态摘要行及退出记录。另有汇总判定和清理记录；不归档 checkpoint 权重。

验收后清理了五个本轮临时 run，以及六个已逐文件归档核对的 bench 目录，具体路径及
字节数见 [cleanup.json](records/cleanup.json)。原始日志、可行性记录、输入对拍缓存和
JAX 缓存按原位置保留；两个工作副本及各自环境保留，等用户决定。
改前副本四份关键文件最终 SHA-256 全部仍为 OK，`src/scripts/packages` 继续只读。
主副本仍为原始 A_HEAD、工作区干净，因此当前可快进合并。

重跑实验需按起跑记录恢复副本和脚本位置；已清理的原始 bench 目录不能直接继续使用。
复核现有结果可直接对 `records/guard-a1`、`records/guard-a2`、`records/guard-b`
运行比较器和完整性守卫。下一步由用户决定是否推送分支、合并回 `v2-motionmem` 和删除副本。
