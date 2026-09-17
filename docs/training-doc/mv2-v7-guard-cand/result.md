# 关闭态真实 100 步：候选与 BASE 全部逐位一致

用户原话「开始实施 有问题尽早问用户」。在约定的旧 400ep 库、GPU 4/5、batch 8、fsdp 2、worker 4、seed 42 档位，前后 100 步实现等价验证通过。此结论不冒充八卡 b128 生产档的实测，也不用于证明开启态 motion 的有效性。

## 版本与运行

BASE 为 `2126b1b1c436166662ff89a629985ecd4524fc42`，CAND 为 `c0be292c40c4a971161e7233540f4dc5c0c1c7cc`；生产改动为 `bec052e5ac76e338cfbe283bf7c3c434f05112b7`。两侧起跑时均 clean，且有独立源码身份断言。启动记录见 [launch.actual.json](records/launch.actual.json)，命令见 [launch.md](launch.md)。候选实测 1142.0 秒，外层 EXIT_CODE=0；本轮会话自然退出，没有清理其他 tmux 会话。

## 验收结果

```text
BASELINE_ENV=PASS
SCALARS steps=100 keys=5 hex_mismatch_steps=0
STATE_DIGEST rows=5 mismatch=0
BATCH_DIGEST rows=7 mismatch=0
BATCH_DIGEST_CANONICAL rows=7 mismatch=0
CANON_CHECK=PASS steps=7
INDEX_SEQ=PASS n=872
DET_CHECK=PASS tier=closed steps=100 scalar_hex_diff=0 state_digest_diff=0 batch_digest_diff=0
INDEX_TRAIN=PASS n=800 recorded_n=872
GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=7 state_digest_rows=5
```

状态摘要覆盖参数、Adam 状态、EMA 与 step，步集合为 0/25/50/75/99；输入摘要步集合为 0/1/2/25/50/75/99。完整 100 步五标量均有限且 hex 可无损回读。末步 loss 为 0.037666261196136475，两侧相同。原始和 canonical 输入摘要、所有状态叶与训练索引均一致。

## 过程与后续

内外日志已分离，未重现 BASE 的日志覆盖问题。新计时/trace 在本对拍关闭；mesh 实测为 (1,2)，符合双卡夹具档。带完整摘要与确定性选项的耗时不用于生产性能结论。

至此 V1、V2、V6 与 V7 均通过；无需撤销生产提交。下一步切换已确认的 encoder 资产并建库。在线位置表方式、仿真 reset 的渲染环境及 V5 的预训练初始化方法仍待用户裁决，不在本项内默认为通过。

归档文件及 SHA256 见 [archive_manifest.json](records/archive_manifest.json)，原始完整记录保留在 `v1-store/bench/2gpu-epoch-bench/mv2-v7-cand/`。
