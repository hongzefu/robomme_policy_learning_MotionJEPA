# 官方16任务×100集：32帧4×4与 Wan 固定池化 1×1024（context），60k

公开 bucket：`HongzeFu/robomme-vla-orig100-ctx32x4x4-wan-timepool8x8-60k-v1`。训练 run：`orig100-ctx32x4x4-wan-timepool8x8-b128-60k`。配置文件：`perceptual-framesamp-ctx-32frame-4x4-wan-timepool1024-orig16.yaml`。融合方式为 context，memory_token_dim=2048。每帧16个视觉token，共512个；motion读法为 `wan_raw_timepool8_spatial8x8_v1`，每窗1×1024，上限96窗，共96个，合计608个memory位置。

motion token 不来自可训练 encoder：`encoder_kind=wan_fixedpool1024`，参数量为0。输入是 Wan 原始 latent（9×16×32×32），去掉首帧、时间维求均值、空间按4×4分块求均值，展平成1×1024。layout 为 `motion-wan-timepool1024-grid32-demopad17-v1`。

训练数据来源：原版 RoboMME 16任务×100集训练库 `16task-orig100-1600ep-9e109419`，帧表 `framesamp-4x4`，motion表 `motion-wan-timepool1024`（23253窗）。Wan latent 复用公开 dataset，repo 与40位 revision 写在 `motion_provenance.json` 的 `timepool.published`。global batch=128，共60000步。

交付12份完整checkpoint：5000、10000、15000、20000、25000、30000、35000、40000、45000、50000、55000、59999。只读取Orbax已提交的目录。

每份 `manifests/step-<步数>.json` 列出该步完整文件与原子根快照的SHA256和字节数。根快照恰好12个文件加 `motion_snapshot.ready.json`：`history_config.txt`、`history_config.resolved.yaml`、`history_config.resolved.sha256`、`motion_provenance.json`、`timepool_window_map.json`、`online_restore_inputs.json`、`online_restore_inputs.npz`，以及 `motion_evidence/` 下的 `DATA_LINEAGE/WAN_RESTORE_EQ/WINDOW_EQ/POOL_CONTRACT/POOL_TABLE_EQ` 五份证据。本包没有 encoder 权重依赖。

对象路径与训练 run 根一致，下载某一步清单所列对象后，该目录即可作为 run 根，用 `mme_vla_suite.policies.motion_client.verify_timepool_snapshot` 核对根快照。在线推理还需要 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，身份摘要写在来源记录的 `timepool.wan`；Wan 权重、tokenizer、运行环境与 motion 全表不随本包发布，motion 全表的 SHA256 写在 `timepool.output`。

保留原始证据中的构建机路径、主机名和源码身份，不修改已绑定的字节，不包含凭据。

增量入口 `run_orig16_ctx_timepool_60k_ckpt_inc.sh` 与最终入口 `run_orig16_ctx_timepool_60k_ckpt_export.sh` 共享相同导出目录及上传记录。增量的 `.done` 表示该组文件已上传并匿名回读，不代表严格恢复已通过。最终入口要求训练日志末行 `EXIT_CODE=0`，匿名逐对象回读核SHA、字节数和精确对象集合，通过才输出 `HF_VERIFY=PASS` 与 `CKPT_UPLOAD=PASS steps=12`。

完成状态以最终回读为准：本说明随根快照最先上传，不代表12步已齐或最终回读已通过。
