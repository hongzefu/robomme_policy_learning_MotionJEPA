# 官方16任务×100集：32帧4×4与motion 4×256，60k

公开 bucket：`HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2split-60k-v1`。训练 run：`orig100-ctx32x4x4-motion-s2split4x256-b128-60k`。融合方式为 context，memory_token_dim=2048。每帧16个视觉token，共512个；motion每窗4×256，共384个，合计896个memory位置。

交付12份完整checkpoint：5000、10000、15000、20000、25000、30000、35000、40000、45000、50000、55000、59999。只读取Orbax已提交的目录。checkpoint根为 `v1-store/train-runs/mme_vla_suite_orig16_b128_60k/orig100-ctx32x4x4-motion-s2split4x256-b128-60k`。

每份 `manifests/step-<步数>.json` 列出该步完整文件、原子根快照和encoder checkpoint/config依赖的SHA256与字节数。根包括实际resolved history、来源身份、在线恢复输入及 `motion_evidence/` 下的 `G0/G2/G3/wan_cache_import/SOURCE_PIN` 五类证据。encoder固定为 `stage2-full144-1025-a/checkpoint_epoch_105.pt` 的EMA权重。

来源合同为 `published-cache-with-local-fill`；保留原始证据中的构建机路径、主机名、GPU标识和源码身份，不修改已绑定的字节，不包含凭据。快照证据路径已重定位为随包相对路径；在线恢复数组已包含在NPZ中。数值精度差异按同输入裁决记为 `NOT_REQUIRED`，其余来源与完整性验证仍须通过。

增量入口需显式传 `RUN` 与 `BUCKET`，只接受计划中的两组配对；它与最终入口共享相同导出目录及上传记录。增量的 `.done` 表示该组文件已上传并匿名回读，不能代替59999严格恢复。训练结束后用 `run_orig16_ctx32x4x4_split_60k_ckpt_export.sh` 最终导出，要求训练日志末行 `EXIT_CODE=0`，并匿名逐对象回读核SHA、字节数和精确对象集合，通过才输出 `HF_VERIFY=PASS` 与 `CKPT_UPLOAD=PASS steps=12`。

运行前可加 `SMOKE=1` 只检查本地快照、依赖和checkpoint，不读取凭据或访问网络。建库前公共体的 `DATASET` 仍为占位，必须回填后再正式运行。HF命令使用独立固定版本 `huggingface_hub[cli]==1.30.0`，不修改训练依赖。本机处于AWS环境，无集群连接，因此匿名回读在本机完成；每个对象验完即删临时回读文件。
