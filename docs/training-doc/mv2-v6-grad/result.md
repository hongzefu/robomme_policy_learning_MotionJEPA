# 关闭态固定输入梯度基线：改前取证完成

本项已正常结束，外层日志记录 `EXIT_CODE=0`。这里只完成改前证据采集，**尚未证明生产改动等价**；改后必须按同一口径逐位比较。

## 版本与运行

用户原话「开始实施 有问题尽早问用户」。源码固定为 `2126b1b1c436166662ff89a629985ecd4524fc42`，起跑与结束时 HEAD 一致、工作区为空。完整起跑记录见 [records/launch.actual.json](records/launch.actual.json)，命令与参数见 [launch.md](launch.md)。壁钟实测 352.5 秒。

## 实测与核验

61 个初态参数叶均有限；mixed1 / allshort / allfull 各有 38 个可训练梯度叶，loss 分别为 0.131137341、0.093950741、0.127008989。

已用 `uv run --no-sync python` 独立核对四项产物的源码身份，V7 的 100 步、五标量十六进制回读与有限性、两族摘要步集合及状态叶有限性全部通过。统一结果为 `BASE_EVIDENCE=PASS head=2126b1b1c436166662ff89a629985ecd4524fc42 v1=3200/200 v2=1200/200 v6=61/38/3 v7=100/5/7 indices=872`。

## 事件与后续

使用物理 GPU 4、5；固定 c8-b 的三个 batch 全部通过 12 个数组键的原始摘要核对。本项不执行优化器更新。

后续候选侧维持同一依赖、数据、GPU 和数值配置；V6/V7 固定 GPU 4、5，缓存保留供前后对照使用。未进入建库或正式训练。

## 归档

[records/archive_manifest.json](records/archive_manifest.json) 记录归档指标及摘要的 SHA256。样本、batch 的逐键摘要采用无损 gzip 留档，解压内容已与原始 JSONL 逐字节核对；大数组留在 `v1-store/bench/mv2/`，不入 Git。V7 的完整状态摘要、索引、环境与标量保留在 records 中，原始完整产物留在 `v1-store/bench/2gpu-epoch-bench/mv2-v7-base/`。
