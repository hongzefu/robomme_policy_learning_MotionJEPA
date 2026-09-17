# 关闭态完整交付基线：改前与改后逐位对拍通过

改前取证与候选对拍均已正常结束，退出码均为 0。本项已在下述固定口径证明逐位相同；开启态与生产八卡档位由各自验证覆盖。

## 版本与运行

用户原话「开始实施 有问题尽早问用户」。源码固定为 `2126b1b1c436166662ff89a629985ecd4524fc42`，起跑与结束时 HEAD 一致、工作区为空。完整起跑记录见 [records/launch.actual.json](records/launch.actual.json)，命令与参数见 [launch.md](launch.md)；dump 命令实际还传入 `--exp-name mv2-v1` 或 `--exp-name mv2-v2`。壁钟实测 397.0 秒。

## 实测与核验

3,200 个样本、200 个 batch；全部 400 集的 101,066 个执行样本身份核对通过。

已用 `uv run --no-sync python` 独立核对四项产物的源码身份，V7 的 100 步、五标量十六进制回读与有限性、两族摘要步集合及状态叶有限性全部通过。统一结果为 `BASE_EVIDENCE=PASS head=2126b1b1c436166662ff89a629985ecd4524fc42 v1=3200/200 v2=1200/200 v6=61/38/3 v7=100/5/7 indices=872`。

## 事件与后续

首次命令遗漏工具必填的 `--exp-name`，在参数解析阶段以退出码 2 结束，未生成样本。补齐 `--exp-name mv2-v1` 后，从同一 BASE 重试；两份日志均保留。

后续候选侧维持同一依赖、数据、GPU 和数值配置；V6/V7 固定 GPU 4、5，缓存保留供前后对照使用。未进入建库或正式训练。

## 归档

候选提交为 `c0be292c40c4a971161e7233540f4dc5c0c1c7cc`，起止工作区干净；机器已确认其不同于 BASE。候选耗时 345.5 秒。3,200 个样本与 200 个 batch 的原始字节摘要相同；源身份、全部帧索引、dtype 与形状均相同。 候选证据见 [records-candidate](records-candidate/archive_manifest.json)，完整命令见 [launch-candidate.md](launch-candidate.md)。

[records/archive_manifest.json](records/archive_manifest.json) 记录归档指标及摘要的 SHA256。样本、batch 的逐键摘要采用无损 gzip 留档，解压内容已与原始 JSONL 逐字节核对；大数组留在 `v1-store/bench/mv2/`，不入 Git。V7 的完整状态摘要、索引、环境与标量保留在 records 中，原始完整产物留在 `v1-store/bench/2gpu-epoch-bench/mv2-v7-base/`。
