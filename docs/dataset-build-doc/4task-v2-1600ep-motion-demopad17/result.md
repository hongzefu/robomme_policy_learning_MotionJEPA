# 1600 集 demo 补帧 motion 表：全部验收通过

新表已建成并标记 verified：71316 行 = demo 35913 + exec 35403，1600 个补帧窗，219082752 字节。全量 token 与独立 oracle 逐位相同；8632 个 VAE 抽样窗（包括全部 1600 个补帧窗）逐位相同。正式训练尚未启动。

## 版本、输入与用户决定

用户原话：「采用建议顺序，保留三方强校验」「采用真实 skipped 计数及完整性验收」。实际按 Wan → encode → oracle → pack/verify → compare 执行，Wan 至 pack 全程为 clean f6915f2a09443d48c1bb57e9b5f83e95400704fd，中间零 commit。输入 SHA 重锚在先前 clean df6fdcc4f9424f00901a107a66d98e19e679a7ba 完成，805 秒、四文件同源；两种启动锚点分别保留。

清单 SHA 为 4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918。四任务各 400 集、共 605611 个执行样本。源 H5、目标库、全部环境覆盖及完整阶段外壳见 [launch.md](launch.md)。source provenance、既有 framesamp-8x8 元数据及新库 norm_stats 三份文件的 SHA 起止未变。

## 数据契约与计算精度

布局为 motion-768-grid16-demopad17-v1、schema 1；demo 最少 17 个真实帧、重复段末帧补齐 33，exec 仍为完整 33 帧、均按段内 stride 16。encoder 使用 wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt，完整 SHA 为 0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca。

Wan VAE 恒 fp32；encoder 按 checkpoint 的训练配置使用 bf16 autocast，77 个参数张量、epoch 72、affine finite 均核对通过。latent 和 motion 文件保存为 float32。此处沿用钉版原脚本的计算方式，没有因建库修改精度。

## 本机实测耗时

环境 B，AWS 本地 NVMe RAID /dev/md0 XFS，8 × A100-SXM4-80GB。Wan 与 encode 各使用 8 卡；oracle encoder 使用 GPU 7，VAE oracle 使用八个独立分片。

| 阶段 | UTC 起止 | 挂钟 | 退出码 |
|---|---|---:|---:|
| Wan | 09-17 22:25:57 → 09-18 01:59:45 | 12828 秒 | 0 |
| encode | 09-18 02:00:00 → 02:02:12 | 132 秒 | 0 |
| oracle | 09-18 02:02:51 → 02:49:42 | 2811 秒 | 0 |
| pack、verify、compare、完整性检查 | 09-18 02:50:20 → 02:50:52 | 32 秒 | 0 |

oracle encoder 独立计算 752.86 秒；VAE 八片最大计算时间 2035.68 秒。上述是阶段耗时，不作为训练吞吐或瓶颈结论。

## 完整性与真实 skipped

Wan 和 encode 的 initial_complete 均为 0，各处理 3200 个唯一段，duplicates=0、missing=0，结束后 claim 均为 0。Wan 汇总 skipped=424，encode 汇总 skipped=1117；这是跳过其他 worker 已完成段的真实计数，没有改写为 0。

所有主要判定行见 [pack.summary.log](records/pack.summary.log)：

- VERIFY_MOTION：71316 行、0 失配。
- WAN_BITEXACT：8632 窗、全部 1600 补帧窗，frame/latent/metadata 失配均 0；比较器独立重算抽样集合。
- ENCODER_BITEXACT：71316 行、0 失配，行序、77 张量 SHA、provenance、checkpoint、finite 等检查全部通过。
- A6_SAMESOURCE：1600 集，清单身份一致；pack 的三方 raw_dir 绑定通过。
- A7_BYTES：3200 段，字节账目和整表行数一致。
- A9_INDEXSET：2100 个样本，包含全部 1600 个冷启动样本，0 失配。
- A10_ROWS：71316=35403+35913，公式及 row_base 无失配。
- A11_PAD：3200 段、71316 行、1600 补帧窗，集合双向差为 0。

## 产物指纹与归档

motion 整表与独立 oracle 的完整 SHA256 相同：03fb46e150dd9015f264f9bd8f08b35883d84da40e9e8080147d2af5fef9b6c5。motion store_meta SHA 为 d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268，作为后续 preflight 的外部固定期望。

[build-summary.json](records/build-summary.json) 保存耗时和文件指纹；records 下保存阶段及八片清洗日志、启动记录、store 元数据和 oracle 摘要。完整逐行报告及数据留在 v1-store；摘要省略可从数据重建的 row_map，并保存完整报告 SHA，不归档大数组或权重。

## 意外、边界与下一步

encode 完成后的附加只读检查曾误按 latent 汇总格式读取 token 汇总的 rows，产生 KeyError；改按真实的 num_grid、bytes、input_latent_sha256 核对 3200 段后全部通过。没有修改生产代码或运行产物。

本轮建库验收已完成。下一步是 V4/V5、V8、V-online 三档与八卡 20 步 smoke，全部通过后才启动已确认的 80k run。建库通过不构成 motion 有效性的策略评估结论。
