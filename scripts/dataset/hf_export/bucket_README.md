# robomme-vla-motionjepa-v1 数据备份（bucket 布局说明）

来源：GreatLakes turbo `robomme_policy_learning_MotionJEPA/v1-store/datasets/` 下两个库的完整备份，
由 `scripts/dataset/hf_export/` 打包上传，全量 sha256 校验闭环（详见仓库根目录
`HF-EXPORT-robomme-vla-motionjepa-v1.md`）。

## 布局

```
packed/4task-gl-framesamp/        # packed 库，原样文件（训练直读格式）
  image_emb_4x4/part_*.bf16.bin
  pos_emb_4x4.f32.bin
  state_emb.f32.bin
  meta/{store_meta.json,pack_progress.jsonl,row_digests.blake2b.bin}
source/4task-gl/                  # 原始库（source_root），tar 分片
  data_tars/data-*.tar            # data/{idx}.pkl，每片 4096 个、按 idx 升序
  features_tars/features-*.tar    # features/episode_*/，按 episode 分组 ~2GiB/片
  meta/                           # provenance.json、stats.json、_shard*of8.json 原样
checksums/
  sha256-shards.txt               # 逐 tar 分片 sha256（路径相对 bucket 根）
  sha256-packed.txt               # 逐平文件 sha256（路径相对 bucket 根）
  sha256-source-files.txt         # 逐源文件 sha256（路径相对 source/4task-gl/，即 tar 成员名）
manifest/upload_manifest.json     # 分片→成员数/字节数汇总
```

注：源库的 `_claims/` 为空目录，未上传。

## 恢复与自验

```bash
# 下载整个 bucket
hf buckets sync hf://buckets/HongzeFu/robomme-vla-motionjepa-v1 ./robomme-vla-motionjepa-v1
cd robomme-vla-motionjepa-v1

# 1) 校验下载完整性（分片与平文件级，覆盖全部字节）
sha256sum -c checksums/sha256-shards.txt
sha256sum -c checksums/sha256-packed.txt

# 2) 解包源库（还原 data/ 与 features/ 目录）
mkdir -p restored/4task-gl && cd restored/4task-gl
for t in ../../source/4task-gl/data_tars/*.tar ../../source/4task-gl/features_tars/*.tar; do tar -xf "$t"; done
cp -r ../../source/4task-gl/meta .

# 3) 解包后逐源文件自验（可选，与 1) 等价的更细粒度）
sha256sum -c ../../checksums/sha256-source-files.txt
```

packed 库无需解包，`packed/4task-gl-framesamp/` 整目录即训练可读的 store
（`meta/store_meta.json` 的 `source_dataset_root` 指向 turbo 绝对路径，异地使用时
用环境变量 `MMEVLA_FRAMESAMP_SOURCE` 指向还原出的 `4task-gl` 目录）。
