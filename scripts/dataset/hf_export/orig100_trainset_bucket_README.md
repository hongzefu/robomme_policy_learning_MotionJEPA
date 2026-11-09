# 官方16任务×100集训练库

公开 bucket：`HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1`。

包含完整 `source/data`、完整 `source/features`、`source/meta`、`meta`、两张帧表 `framesamp-4x4` 与 `framesamp-8x8`、`assets/robomme/norm_stats.json`、Wan latent、逐段 `motion-tokens-stage2` 和 `motion-stage2-1024` packed 表。concat 的 `[1,1024]` 与 split 的 `[4,256]` 共用同一份 packed 数据，仅改变读出形状。

`source/data_tars` 与 `source/features_tars` 中的 tar 解包到 `source/`；`wan-latents_tars` 与 `motion-tokens-stage2_tars` 中的 tar 解包到库根。其余文件按原目录保留。tar 内成员和直接文件的 SHA 位于 `checksums/`；最终所有远端对象的 SHA256、字节数和文件集合由 `delivery_manifest.json` 记录（该清单自身的 SHA 在本地交付验收中核验）。

缓存来源为 `published-cache-with-local-fill`。`evidence/` 提供 `G0.json`、`G2.json`、`G3.json`、`wan_cache_import.json` 与 `SOURCE_PIN.json`。这些文件从实际来源合同的 `evidence` 引用读取，并逐个核对 SHA；缓存收据包含源发布版本、H5 身份、内嵌copy/fill映射、输入与输出 SHA。数值精度差异按既有同输入裁决记为 `NOT_REQUIRED`，输入身份、shape、finite 和文件完整性仍须验证。

原元数据保留构建时的本机路径、主机名、GPU标识、源码仓库身份与来源证据绝对路径，保证哈希不改变；不包含凭据。异地恢复时须把来源合同的五类证据路径解析到随包的 `evidence/<角色>.json`，不能依赖构建机绝对路径。训练启动仍须执行当前契约与来源验证。

本机处于AWS环境，无集群连接，上传验收在本机通过不带凭据的HF CLI完成：每个对象下载后核SHA256与字节数，并核对远端完整对象集合。每次仅保留一个回读对象，完成后删除该临时对象；通过才输出 `HF_VERIFY=PASS`。

运行 `SMOKE=1 bash scripts/dataset/hf_export/run_orig100_trainset_export.sh` 做本地部分打包和逐成员源字节对照。`SMOKE` 不访问网络、不读取凭据。正式运行前须在公共体回填 `DATASET`；遇 `__FILL_AFTER_BUILD__` 一律退出失败。只在离线夹具中允许 `O16_LIB`、`O16_NORM_STATS`、`O16_EXPORT_ROOT` 覆盖路径。

网络传输固定使用独立 `uvx --python 3.11 --from 'huggingface_hub[cli]==1.30.0' hf`，不改变训练环境的依赖。上传中断后可重跑相同入口，最终匿名回读通过才算交付完成。
