# 官方16任务×100集：去 action encoder、context 与 motion 4×256，80k

公开 bucket：`HongzeFu/robomme-vla-gl-orig100-ctx-s2split-nostate-b64-80k-v1`。训练 run：`gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k`，配置为 `mme_vla_suite_orig16_b64_80k`。融合方式为 context，memory_token_dim=2048。每帧16个视觉token，共512个；motion每窗4×256，共384个，合计896个memory位置。

交付16份完整checkpoint：5000、10000、15000、20000、25000、30000、35000、40000、45000、50000、55000、60000、65000、70000、75000、79999。只读取Orbax已提交的目录。checkpoint根为 `v1-store/train-runs/mme_vla_suite_orig16_b64_80k/gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k`。

encoder固定为 `aws-stage2-nostate-1101-a/checkpoint_epoch_105.pt` 的EMA权重，SHA256为 `be7cdec7d96c889193f27cd925977c7ff47423d4321d1fd7446799d38495e3d4`；motion表为 `16task-orig100-1600ep-9e109419/motion-stage2-1024-nostate`。history为 `perceptual-framesamp-ctx-32frame-4x4-nostate-motion-stage2-4x256-orig16.yaml`。

每份 `manifests/step-<步数>.json` 列出该步完整文件、原子根快照和encoder checkpoint/config依赖的SHA256与字节数。根包括实际resolved history、来源身份、在线恢复输入及 `motion_evidence/` 下的 `G0/G2/G3/wan_cache_import/SOURCE_PIN` 五类证据。来源合同为 `published-cache-with-local-fill`；保留已绑定的原始来源字节，证据引用使用随包相对路径，在线恢复数组包含在NPZ中，不包含凭据。

增量导出使用 `RUN=<训练run> BUCKET=<公开bucket> bash scripts/dataset/hf_export/run_orig16_stage2_ckpt_inc.sh`；最终导出使用相同环境变量并执行 `O16_MODE=final bash scripts/dataset/hf_export/_stage2_orig16_vla_ckpt_export_common.sh`，要求训练日志末行 `EXIT_CODE=0` 及精确16步集合。`DRY_RUN=1` 只输出run、encoder、表、配置、bucket与步集合，不创建导出目录、不读取凭据或访问网络；`SMOKE=1` 检查本地完整交付，不上传。

失败重起仅接受登记过的 `-attemptNN`（NN≥02）：传 `ATTEMPT_MANIFEST`，清单中的run、base_run、history、决定记录和同后缀bucket必须一致。增量与最终导出共享上传记录；`.done` 只代表传输和匿名回读，不能代替 `check_stage2_checkpoint.py --mode delivery --delivery-step 79999 --expected-manifest <清单>` 的严格恢复。交付判定为 `CKPT_DELIVERY_READY=PASS step=79999 …`。

正式发布逐对象匿名回读，核SHA256、字节数和精确对象集合，通过后输出 `HF_VERIFY=PASS` 与 `CKPT_UPLOAD=PASS steps=16`。本次公开备份的独立读回校验在GreatLakes纯CPU作业中执行；未经校验通过不得视为已备份。
