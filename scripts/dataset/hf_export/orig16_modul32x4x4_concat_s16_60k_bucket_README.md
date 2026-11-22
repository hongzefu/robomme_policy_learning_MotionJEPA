# 官方16任务×100集：32帧4×4 modulation 与 motion 1×1024（concat），窗口步长16，60k

公开 bucket：`HongzeFu/robomme-vla-orig100-modul-s2concat-s16-60k-v1`。训练 run：`orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k`。配置文件：`perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-s16-orig16.yaml`。融合方式为 modulation，memory_token_dim=1024。每帧16个视觉token，共512个；motion读法为 `stage2_pool4_concat_v1`，每窗1×1024，窗长33帧、窗口步长16（相邻窗重叠17帧），上限192窗，合计704个memory位置。与 stride32 版 `perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-orig16.yaml` 只差 motion 的 stride（32→16）、budget（96→192）、store_path 三项。

训练数据来源：原版 RoboMME 16任务×100集训练库 `16task-orig100-1600ep-9e109419`，帧表 `framesamp-4x4`，motion表 `motion-stage2-1024-grid16`（layout `motion-stage2-1x1024-grid16-demopad17-v1`，全部 grid16 窗口；demo 尾窗至少17个真实帧并以末帧补满，执行段窗口全为真实帧），encoder 逐窗输出目录 `motion-tokens-stage2-grid16`。encoder固定为 `stage2-full144-1025-a/checkpoint_epoch_105.pt` 的EMA权重（`#model`），与训练库同源。初始化为 `pi05_base`，seed=42，global batch=128，8卡 FSDP8，共60000步。

交付12份完整checkpoint：5000、10000、15000、20000、25000、30000、35000、40000、45000、50000、55000、59999。只读取Orbax已提交的目录。checkpoint根为 `v1-store/train-runs/mme_vla_suite_orig16_b128_60k/orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k`。

每份 `manifests/step-<步数>.json` 列出该步完整文件、原子根快照和encoder checkpoint/config依赖的SHA256与字节数。根包括实际resolved history、来源身份、在线恢复输入及 `motion_evidence/` 下的 `G0/G2/G3/wan_cache_import/SOURCE_PIN` 五类证据；encoder依赖实体随包位于 `encoder_dependencies/stage2-full144-1025-a/`。

恢复方式：下载某一步的清单所列对象后，以 `check_stage2_checkpoint.py --mode delivery --delivery-step <步数> --checkpoint <stage>/<步数> --expected-manifest <stage>/manifests/step-<步数>.json` 严格核对并恢复。依赖：tokenizer、VAE 与运行环境按摘要使用本机缓存，不随本包发布；本包含 encoder 权重与配置实体。

来源合同为 `published-cache-with-local-fill`；保留原始证据中的构建机路径、主机名、GPU标识和源码身份，不修改已绑定的字节，不包含凭据。

增量入口需显式传 `RUN` 与 `BUCKET`；它与最终入口共享相同导出目录及上传记录。增量的 `.done` 表示该组文件已上传并匿名回读，不能代替59999严格恢复。训练结束后用 `run_orig16_modul32x4x4_concat_s16_60k_ckpt_export.sh` 最终导出，要求训练日志末行 `EXIT_CODE=0`，并匿名逐对象回读核SHA、字节数和精确对象集合，通过才输出 `HF_VERIFY=PASS` 与 `CKPT_UPLOAD=PASS steps=12`。

完成状态以最终回读为准：本说明随根快照最先上传，不代表12步已齐或最终回读已通过。
