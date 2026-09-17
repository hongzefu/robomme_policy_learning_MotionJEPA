# 正式训练完成：60k 步、最终验收与副本合并

**训练正常完成。** `v2-1600ep-m8x8-modul-b128-60k` 于 `2026-09-15T05:48:27Z` 从 clean HEAD `dd07f18fc385b01eb52db7563fe5f202997b9706` 启动，使用 GPU 4,5,6,7；于 `2026-09-16T20:53:31Z` 退出，`EXIT_CODE=0`。完整 60,000 步耗时 **39 小时 5 分 4 秒**（140704 秒，包含初始化、checkpoint 保存和退出），预期 12 个 checkpoint 全部保存完成。

[wandb 运行](https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/6ubtaf9l) 的同步完成信息已写入日志。训练会话 `m8-prod`、密采会话 `m8-prod-dense` 和已记录的训练/采样 PID 均已退出。完整启动配置见 [launch.md](launch.md)，实际命令与环境见 [records/launch.actual.json](records/launch.actual.json)，完整日志按原始字节压缩归档为 [records/train.log.gz](records/train.log.gz)。

## 最终指标与 checkpoint

`metrics.jsonl` 包含 step 0、100、…、59900 共 600 条连续记录，五项标量全部有限，dec/hex 相符。`log_interval=100`，除 step 0 外均为日志区间统计；最后一条 loss **0.0017725301** 属于 step 59900，不能称作 step 59999 的单步 loss。step 59000–59900 的十条日志覆盖训练 step 58901–59900，loss 均值为 **0.0018321254**；最后 59901–59999 共 99 步没有独立标量日志。训练 loss 不代表策略评估成功率。

| 日志步 | loss（区间统计，step 0 除外） |
|---|---:|
| 0 | 0.08936774 |
| 5000 | 0.00827814 |
| 10000 | 0.00547655 |
| 30000 | 0.00277228 |
| 50000 | 0.00201081 |
| 59900 | 0.00177253 |

checkpoint 位于 `v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-modul-b128-60k/`，目录集合为 `{5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,59999}`，实占约 133 GiB。12 份 `_CHECKPOINT_METADATA` 均有完成提交时间，12 份 norm_stats SHA256 均为 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`；清单及元数据摘要见 [records/final_pre_merge.json](records/final_pre_merge.json)。权重仅保存在 `v1-store`。

最终 `59999` 在 CPU 上成功完整读取参数，并与 modulation 配置精确匹配：

```text
PARAM_TREE_EXACT=PASS config=mme_vla_suite_b128_60k history_config=perceptual-framesamp-modul-8frame-8x8.yaml yaml_sha256=5b5ac2f85302d4d8 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
EXIT_CODE=0
```

输出见 [records/final_param_tree.json](records/final_param_tree.json) 与 [records/final_param_tree.log](records/final_param_tree.log)。复核使用合并后的 `9a58dc8`，该版本与起跑版本的 `src/scripts/packages`、`pyproject.toml` 和 `uv.lock` 无差异。主环境导入路径、motion 关闭态、resolved 配置摘要和原始日志压缩前后逐字节一致的检查见 [records/final_validation.json](records/final_validation.json)。

## 全程性能统计

存储为 AWS 本地 NVMe RAID（`/dev/md0`），batch 128、worker 8、fsdp 4。以完整 metrics 的精确时间戳统计 step 100→59900，59800 次更新耗时 139951.786 秒，均值为 **2.34033 秒/步、54.69 samples/s**。该窗口包含数据等待与 checkpoint 开销，排除了初始化及起跑前 100 步，也不包含最后 99 步和退出阶段。

全程 15 秒采样共 37520 条、每卡 9380 条，四卡 util 均值 **72.39%**、0% 样本占比 **25.39%**；在上述 metrics 窗口内，37320 条样本对应 **72.55% / 25.23%**。这些值是离散采样估计，采样间隔大于步时，不能据此逐步定位等待。起跑 500ms 密采的慢区间/其他区间分层保留在下文，不能把它扩张成全程逐步结论。完整指标及四卡分表见 [records/final_metrics.json](records/final_metrics.json)。

## V10 与主副本恢复

在解锁及合并前，主副本仍是起跑 HEAD，`git status --porcelain` 为空，`framesamp_dataset.py`、`train.py`、`history_pi0.py` 三份文件与锁定 SHA256 全部一致。证据已先保存到 [records/final_pre_merge.json](records/final_pre_merge.json)，随后恢复 `src/scripts/packages` 的用户写权限，并再次核对源码 SHA 一致。

开发副本比主副本多 `739aca3`、`3f62ba3`、`9a58dc8` 三个已推送的文档提交，主副本已通过 `git merge --ff-only origin/v2-motionmem` 快进至 `9a58dc8`。本次同步带回起跑留档及 motion 后续计划，未实施 motion 计划、改变超参或重装依赖。最终训练留档已提交并推送为 `9f7f423`。

开发副本已于 2026-09-17 删除。删除前再次确认主副本及远端包含开发副本全部提交、双方工作区干净，开发副本无 stash、独有分支或其他 worktree，忽略产物仅为独立 `.venv` 和 `v1-store` 链接。先移除该链接，再从 `/scratch/hongze` 删除准确的 `-temp` 目录；主数据根的 device/inode、12 个 checkpoint 清单及其元数据、参数 manifest 和 norm_stats 摘要均保持一致，三个受保护源码摘要仍匹配。验收见 [records/merge_cleanup.json](records/merge_cleanup.json)。

用户已针对 `claude-private` 仍在开发目录运行的情况明确允许直接删除；本次没有关闭会话或发送按键，tmux 会话清单前后相同。后续开发统一从主副本进入。

## 起跑前验证全部通过

F2 的 3454 个样本/19 键逐字节一致，F3 的 100 步五标量、800 个训练索引、6 份输入与完整 TrainState 均逐位一致。modulation 两档 A/A 和 A/B 共四组比较通过：61 个初态叶子、三类 loss hex、38 个可训练梯度叶子完全相同。

b128 四卡 smoke 的 20 步全部有限并正常退出，checkpoint 参数树 61/61 精确匹配，六条 modulation 专属参数路径齐全，motion 关闭，新库 norm_stats SHA 与 checkpoint 中保存的文件一致。正式 run 的 25 项 preflight 全通过，原文见 [records/preflight.log](records/preflight.log)。

## 历史记录：300 步数值与代码隔离

| 日志步 | loss（显示值） | grad_norm（显示值） | mem_enc_norm（显示值） |
|---|---:|---:|---:|
| 0 | 0.0894 | 1.2680 | 0.1092 |
| 100 | 0.0563 | 0.5048 | 0.0416 |
| 200 | 0.0292 | 0.1914 | 0.0122 |
| 300 | 0.0240 | 0.2083 | 0.0125 |

正式档 `log_interval=100`，除 step 0 外表中值是对应日志区间的统计，不冒充逐个单步记录。四次日志的全部五项标量均有限，精确 dec/hex 见 [records/startup_metrics.jsonl](records/startup_metrics.jsonl)。起跑阶段没有运行错误，不根据这些训练 loss 推断策略评估效果。

训练期间主副本 `src/`、`scripts/`、`packages/` 锁只读，开发全部转到 `-temp`。开发副本使用独立 `.venv`，共享数据仅通过已批准的 `v1-store` symlink 访问。300 步后的复核确认主副本仍为原 HEAD、Git 状态干净且三份源码 SHA 未变，见 [records/startup_check.json](records/startup_check.json) 与 [records/lock_sha256.txt](records/lock_sha256.txt)。训练结束后的再次复查见上文 V10。

## 历史记录：起跑稳态性能与待查问题

存储为 AWS 本地 NVMe RAID（`/dev/md0`），batch 128、worker 8、fsdp 4。排除前 100 步，实际窗口为 tqdm 进度 102→302，共 200 次更新、484.184 秒；相邻进度区间共 38 个。GPU 每 500ms 采样，窗口内 3872 条记录，每卡 968 条。

| 指标 | 实测 |
|---|---:|
| 稳态步时均值 | 2.42092 秒 |
| 吞吐 | 52.87 samples/s |
| GPU util 均值 | 70.14% |
| 0% 采样占比 | 27.30% |
| 慢区间 util 均值 / 0% 占比 | 52.31% / 44.47% |
| 其他区间 util 均值 / 0% 占比 | 77.92% / 19.81% |

**起跑窗口 GPU 尚未吃满。** 慢区间利用率更低，表明该窗口有较多等待；这份统计不能单独定位到 CPU、I/O 或同步环节。用户已于 2026-09-15 明确确认「继续当前训练，性能问题另行排查」。训练按既定 worker、batch、学习率等设置完成；性能排查留待后续任务。

分层按相邻 tqdm 进度区间的平均步时进行，慢区间阈值为 3.58993 秒（区间步时中位数的 1.5 倍），共 13 个慢区间。由于进度点通常跨多个训练步，这里明确称「区间」，不声称逐个训练步分层；中位数仅用于分层阈值。完整数值与四卡分表见 [records/startup_performance.json](records/startup_performance.json)，原始起跑日志、密采快照和分析脚本一并归档。

当时按该窗口外推，总计算时间约 **40.35 小时**，预计 **2026-09-16 22:13 UTC** 左右完成；这是历史 ETA。最终实际结束为 **2026-09-16 20:53:31 UTC**，全程耗时 **39 小时 5 分 4 秒**，以该实测替代本 ETA 和原计划约 65 小时估计。不同配置的两条 run 不用于归因某项改动的提速。

## 复现最终统计与后续范围

最终统计使用 [records/analyze_final.py](records/analyze_final.py)，读取同目录中的压缩原始日志、完整 metrics 与全程 15 秒 GPU 采样，输出 [records/final_metrics.json](records/final_metrics.json)：

```bash
UV_CACHE_DIR="$PWD/v1-store/cache/uv" uv run --no-sync python docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/records/analyze_final.py
```

全程 15 秒采样间隔大于步时，只提供离散采样均值与 0% 占比，不推导逐步 GPU 等待；慢区间/其他区间的结论只来自上文起跑 500ms 密采。完整前 30 分钟密采归档为 [records/gpu_util_lms500_first30min.csv](records/gpu_util_lms500_first30min.csv)。本轮保留全部 checkpoint，策略评估、性能优化和 motion 实施均另立任务。

## HF bucket 异地备份（2026-09-17）

**12 个 checkpoint 已全量备份到 HF private bucket，两遍独立 sha256 逐行对照验收通过。** 此前 39 小时训练的产物只存在于本机 `/dev/md0` 一块盘上（本文档上文「权重仅保存在 `v1-store`」），本次消除该单点风险。

| 项 | 值 |
|---|---|
| bucket | `HongzeFu/robomme-vla-modul-60k-v1`（`private: true`，创建于 `2026-09-17T15:54:04Z`） |
| 内容 | 12 个 orbax checkpoint + 4 份 config/provenance + `wandb_id.txt` + `README.md` + `SHA256SUMS.pre.txt` |
| 规模 | 274 对象 / 142,545,105,487 B（132.77 GiB） |
| 上传链路 | `scripts/dataset/hf_export/run_modul60k_ckpt_export.sh`（bucket README 源文件 `modul60k_bucket_README.md`） |
| 起跑 commit | `e4dc733`（clean HEAD） |
| 会话 / 日志 | tmux `hf-modul60k-export`（跑完自然退出）；`v1-store/exports/hf-ckpt-v2-1600ep-m8x8-modul-b128-60k/logs/export.log` |
| 耗时 | 约 26.5 分钟（15:54:01 → 16:20:30 本地时间） |

### 验收判定行（逐条实测，全部一次通过）

```text
HF_WHOAMI=HongzeFu
目标 bucket 现状: files=0 size=0          # 空库断言，防 repo_id 打错污染别的库
LOCAL_FILES=274 LOCAL_BYTES=142545105487  # 274 = 272 源文件 + README.md + SHA256SUMS.pre.txt
PRE_LINES=273                             # 清单含 README.md、不含清单自身
UPLOAD_DONE batches=12
BUCKET_FILES=274 BUCKET_BYTES=142545105487
SHA256_PRE_POST_DIFF=0
sha256sum -c 校验行数: 273
RESULT=PASS
SRC_UNCHANGED=OK
EXIT_CODE=0
```

**主判据是 `SHA256_PRE_POST_DIFF=0` 与 `RESULT=PASS`**：把整个 bucket 全量回读到 `verify/`（真下载 132.77 GiB），在回读侧**独立重算** sha256 得到 post 清单，与上传前的 `SHA256SUMS.pre.txt` 逐行 `diff` 为空，再补一遍 `sha256sum -c --strict`（273 行全 OK）。计数层的 `BUCKET_FILES` / `BUCKET_BYTES` 只作早停信号——该 REST 接口已知异步滞后，本次一次即吻合、未触发 120 秒复查。

`SRC_UNCHANGED=OK` 是收尾断言：stage 树用硬链接搭建（与源同在 `/dev/md0`，额外占盘为 0，不复制第二份 132.77 GiB），故必须在源目录上算第三遍 sha256，证明原始 checkpoint 未被连带改动。

### 两处与上一条 run 的差异

1. **分批粒度**：按 step 目录切 12 批，每批 11.06 GiB。2026-09-08 实测的失败模式是服务端 `new_upload_commit` 随**批内字节量**超时（1.62 GB×10=16.2 GB 过、2 GB×5=10 GB 稳、2 GB×10=20 GB 挂），11.06 GiB 落在已验证会过的量级；本次 12 批**零重试**。orbax 的 step 目录是稳定的天然边界，不需要 `run_motionjepa_full1600_export.sh` 那套贪心装箱与 `upload_plan.json` 落盘（那是为 2991 个重尾 `.bin` 准备的，分组漂移会让 `--include` 失配）。
2. **实验条件**：本 run 的 `motion_provenance.json` 里 `motion_enabled: false`，`motion_root` / `vae` / `encoder` 均为 `null`——这是 `integration_type: modulation` + framesamp-8x8、**motion 通道关闭**的口径，与 bucket `HongzeFu/robomme-motionjepa-vla-v1` 存放的 `awsprod40k-b128-motion`（motion 开启）是不同实验条件，两者不可混作一组对照读。bucket README 已写明该声明。

### 未清理产物

`v1-store/exports/hf-ckpt-v2-1600ep-m8x8-modul-b128-60k/verify/`（回读副本，132.77 GiB 实占）按既有惯例保留（上一条 run 的 `hf-ckpt-awsprod40k-b128-motion/` 同样在盘上）。确认不再需要复核时可回收该空间；`stage/` 是硬链接，删除不释放空间也不影响源文件。

### 恢复方式

```bash
hf sync hf://buckets/HongzeFu/robomme-vla-modul-60k-v1 ./robomme-vla-modul-60k-v1
cd robomme-vla-modul-60k-v1 && sha256sum -c --strict SHA256SUMS.pre.txt
# 只取末步：hf sync hf://buckets/HongzeFu/robomme-vla-modul-60k-v1 ./ckpt-59999 --include '59999/*'
```

step 目录保持 orbax 原结构，下载即可 `ocp.PyTreeCheckpointer().restore(".../59999/params")`，无需解包。bucket CLI 须用 `uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf`（项目 `.venv` 的 0.32.3 无 bucket 子命令），凭据用 `v1-store/secrets/hf.env` 覆盖环境里那个 yinpei-tri 的 `HF_TOKEN`。
