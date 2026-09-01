# hf-export-robomme-vla-motionjepa-v1 · result

**终判 PASS**：bucket 与 turbo 源字节级一致。

- bucket 最终：480,205,065,887 字节 / 293 个文件（`hf buckets info` 与本地 `du -sb`/`find|wc -l` 完全相等）。
- 内容：241 个 tar 分片（97 data + 144 features，tar 内成员 880,180 个）+ 47 个平文件 + 清单/manifest/README。
- 校验闭环：回读 480 GB 后 `sha256sum -c` 重算 288 行清单（覆盖全部字节）6 路并行 **0 失败**；
  文件集合 293/293 一致；清单类小文件逐字节一致；`RESULT=PASS`、driver `EXIT_CODE=0`。
- smoke 前置验收：11,379 个成员独立复算全 OK、与 NFS 源逐字节对拍 0 不一致；
  packed 两小 bin 与 `store_meta.json` 内置 sha256 交叉锚点一致。
- 耗时：打包 72.3 min（8 线程、NFS 单遍读）；上传 ~45 min（聚合峰值 ~230 MB/s，含一次 401 重试）；
  回读 ~44 min；校验 ~2 min；合计 2h43m。
- 留底：`v1-store/exports/hf-export-robomme-vla-motionjepa-v1/`（含 880,180 行逐源文件清单 83 MB）；
  本目录 `records/` 存小体量快照（分片/平文件清单、upload_manifest、清洗后日志）。
- 插曲两则（详见根目录主文档）：`HF_HOME` 改指后 token 未随迁导致首次上传 401（已固化修复进
  `run_export.sh`）；`hf buckets info` 统计异步滞后，曾短暂显示 312 GB/200 文件，经上传方向
  `--dry-run`（293/293 `skip: identical`）核实无缺，后追平。
