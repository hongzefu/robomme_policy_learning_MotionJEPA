# 历史 YAML 与旧表交付基线：改前取证完成

本项已正常结束，外层日志记录 `EXIT_CODE=0`。这里只完成改前证据采集，**尚未证明生产改动等价**；改后必须按同一口径逐位比较。

## 版本与运行

用户原话「开始实施 有问题尽早问用户」。源码固定为 `2126b1b1c436166662ff89a629985ecd4524fc42`，起跑与结束时 HEAD 一致、工作区为空。完整起跑记录见 [records/launch.actual.json](records/launch.actual.json)，命令与参数见 [launch.md](launch.md)；dump 命令实际还传入 `--exp-name mv2-v1` 或 `--exp-name mv2-v2`。壁钟实测 179.4 秒。

## 实测与核验

1,200 个样本、200 个 batch；未经修改的旧 context motion YAML 正常构造和交付。

已用 `uv run --no-sync python` 独立核对四项产物的源码身份，V7 的 100 步、五标量十六进制回读与有限性、两族摘要步集合及状态叶有限性全部通过。统一结果为 `BASE_EVIDENCE=PASS head=2126b1b1c436166662ff89a629985ecd4524fc42 v1=3200/200 v2=1200/200 v6=61/38/3 v7=100/5/7 indices=872`。

## 事件与后续

40ep 库只有 20 集可覆盖短帧分支，`fixture_per_step` 取 20；400ep 的对应配额为 200。因此沿用同一完整取证算法的样本总数不同，本项没有设置 DTYPE_DUMP_LIMIT，也没有减少默认随机样本配额。

后续候选侧维持同一依赖、数据、GPU 和数值配置；V6/V7 固定 GPU 4、5，缓存保留供前后对照使用。未进入建库或正式训练。

## 归档

[records/archive_manifest.json](records/archive_manifest.json) 记录归档指标及摘要的 SHA256。样本、batch 的逐键摘要采用无损 gzip 留档，解压内容已与原始 JSONL 逐字节核对；大数组留在 `v1-store/bench/mv2/`，不入 Git。V7 的完整状态摘要、索引、环境与标量保留在 records 中，原始完整产物留在 `v1-store/bench/2gpu-epoch-bench/mv2-v7-base/`。
