# hf-export-robomme-vla-motionjepa-v1 · launch

完整实录（背景、方案、逐步骤命令、实测数字、插曲与终判）见仓库根目录
**[`HF-EXPORT-robomme-vla-motionjepa-v1.md`](../../../HF-EXPORT-robomme-vla-motionjepa-v1.md)**（用户指定主文档），本目录只做留档指针与快照。

- 任务：turbo 两库（`4task-gl-framesamp` packed 36.9 GB + `4task-gl` 原始库 data/features/meta 443 GB）打包上传
  HuggingFace private bucket `hf://buckets/HongzeFu/robomme-vla-motionjepa-v1`，sha256 全量回读校验。
- 起跑时 HEAD：`1d1de7c`（导出工具 `scripts/dataset/hf_export/` 为本轮新增，随导出后的 commit 入库）。
- 执行窗口：2026-08-31 23:12 → 2026-09-01 01:55（EDT），tmux `hf-export-v1`。
- 入口：`bash scripts/dataset/hf_export/run_export.sh`（阶段1 打包+哈希 → 阶段2 `hf buckets sync` 上传 →
  阶段3 全量回读 → 阶段4 `sha256sum -c` 并行校验）。
- 暂存：`/data/hongzefu/hf-export-robomme-vla-motionjepa-v1/`（临时，验收后已整体删除）。
