# 4task（4任务×400集）：8帧8×8与stage2 rgb-state motion 1×1024（concat，modulation），50k

公开 bucket：`HongzeFu/robomme-vla-4task-m8x8-modul-s2rgbstate-concat1x1024-50k-v1`。训练 run：`v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k`。配置文件：`perceptual-framesamp-modul-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml`，TrainConfig `mme_vla_suite_4task_b128_50k`。融合方式为 modulation，memory_token_dim=1024。帧记忆每帧8×8共64个视觉token，8帧共512个；motion读法为 `stage2_pool4_concat_v1`，每窗4×256原样拼成1×1024，上限96窗、stride 32，合计608个memory位置。与另一条同库同encoder的 run 只差融合方式与 memory_token_dim。

训练数据来源：4task 训练库 `4task-v2-1600ep-604f16da`（ButtonUnmask、VideoUnmask、ButtonUnmaskSwap、VideoUnmaskSwap），帧表 `framesamp-8x8`，motion表 `motion-stage2-1024-rgbstate-ep144`。encoder固定为 `stage2-4task-rgbstate-144ep-1113-a/checkpoint_epoch_144.pt` 的EMA权重（`#model`，arch `wan-latent-stage2-blockwise-pool4-rgb-state`），encoder训练源与本训练库为同一组1600个源H5（member-pin 血缘核对通过才起跑）。初始化为 `pi05_base`，global batch=128，lr 5e-5，NUM_TRAIN_STEPS=50001。

交付10份完整checkpoint（CKPT_STEPS）：5000、10000、15000、20000、25000、30000、35000、40000、45000、50000，评估与交付用50000。只读取Orbax已提交的目录。checkpoint根为 `v1-store/train-runs/mme_vla_suite_4task_b128_50k/v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k`。

每份 `manifests/step-<步数>.json` 列出该步完整文件、原子根快照和encoder checkpoint/config依赖的SHA256与字节数；`delivery_manifest.json` 汇总10步与训练日志。根包括实际resolved history、来源身份、在线恢复输入及 `motion_evidence/` 下的全部来源证据（集合以 `motion_provenance.json` 的 `stage2_compat.evidence` 为准）；encoder依赖实体随包位于 `encoder_dependencies/stage2-4task-rgbstate-144ep-1113-a/`。

恢复方式：下载某一步清单所列对象后，以 `check_stage2_checkpoint.py --mode delivery --delivery-step <步数> --checkpoint <stage>/<步数> --expected-manifest <stage>/manifests/step-<步数>.json` 严格核对并恢复。tokenizer、VAE 与运行环境按摘要使用本机缓存，不随本包发布；本包含 encoder 权重与配置实体。

来源合同为 `local-built-latents`（旧库已有 Wan latent 隔一取一得 stride 32，不重新解码原始视频）；保留原始证据中的构建机路径、主机名、GPU标识和源码身份，不修改已绑定的字节，不包含凭据。

评估结果：692局评估（seed 7）放在同一 bucket 的 `eval/local692-s2rgbstate-modul-concat1024-50k-s<seed>/` 前缀下，含逐局记录、日志、视频与 `eval_manifest.json`（文件SHA256与字节数、EVAL692 判定行）。eval 与 checkpoint 分开清单、分开匿名回读：checkpoint 范围的全量精确集合比对排除 `eval/`，eval 范围只比对 `eval/` 前缀。

导出流程：增量入口 `run_4task_stage2_ckpt_inc.sh`（显式传 `RUN` 与 `BUCKET`）每份落盘即传并匿名回读，只输出 `INC_END`，不算最终验收。训练成功后用 `run_4task_modul8x8_concat_50k_ckpt_export.sh`：`O16_MODE=stage` 只在本地生成 stage（不联网），`O16_MODE=upload` 复用该 stage 上传并逐对象匿名回读核SHA、字节数和精确对象集合，通过才输出 `HF_VERIFY=PASS` 与 `CKPT_UPLOAD=PASS steps=10`；`O16_MODE=eval` 上传评估结果并输出 `EVAL_HF_VERIFY=PASS`。

完成状态以最终回读为准：本说明随根快照最先上传，不代表10步已齐或最终回读已通过。
