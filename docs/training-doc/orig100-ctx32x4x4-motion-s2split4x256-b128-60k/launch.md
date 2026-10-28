# 官方16×100 context 60k：正式启动档案

## 一、目的与状态

按[已批准计划](../../../1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md)，先context、后modulation各60000步，均使用8卡和global batch128。当前全部原训练前检查已真实执行且通过；本档案所在commitV11.67Beta为正式起跑锚点，实际完整SHA由启动时TRAIN_HEAD及运行日志保存。当前尚未把正式训练记作完成。

用户最新原话：“不要后置任何步骤 该做的还是做 但是失败了只记录 让用户决策”。原步骤均保留。两条原G12按计划在两条训练结束后执行；训练后诊断失败真实记录，不改成PASS，也不独自阻断已正常完成的context接续modulation。真实程序非零退出、缺实际结束日志或59999不能被伪记为完成。

## 二、版本与可复现锚点

| 对象 | 身份 |
|---|---|
| 已批准计划 | d5fe05d647a899e016bc513aa5fba03485a9b398 |
| 数据构建Beta | bd4bad0957e5162073f9bbfec0e4c985c3f830bc |
| 最终代码/常量基点 | 941fcfde1571e3c164937b4f817e38a7229f8d07，后续仅档案 |
| 两条20步及严格恢复 | clean 491340dfb210acb0e399f9105d8283b8a2bc66f7，均退出0、固定输出0差、在线推理通过 |
| 正式训练 | 本次commitV11.67Beta；启动后全SHA写日志及本地HOLD，运行态正本在-temp |
| encoder | stage2-full144-1025-a/checkpoint_epoch_105.pt#model，EMA，SHA 27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035 |
| MJ共享库 | b5d23a5fab55be16e580c5b77a8f04d68b3170c2，既有独立wan-stage2环境 |

起跑前保存src全部原权限并移除写权限；Git可执行位不变。正式期间保持主HEAD不变，状态和新开发使用`/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`及独立venv，状态分支`worktree-agent-o16-state-20261028`。实际source权限台账在`v1-store/logs/o16-20261028/src-permissions-before.json`，全部相关训练/上传结束后由主会话按原值恢复。

## 三、完整启动与还原

原生产入口仍为`run_orig16_ctx_then_modul_60k.sh`、单条`run_orig16_stage2_60k.sh`以及既有HF导出/恢复器。运行载体只串调用这些入口，不修改计算、验收阈值或公开范围；已独立同规模调度审查及逐脚本审查通过。启动顺序为chain先，随后两个备份和G12：

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
export TRAIN_HEAD=$(git rev-parse HEAD)
# 各命令均放在下节指定的独立detached tmux，并记录真实退出码。
bash v1-store/logs/o16-20261028/formal-chain.sh
ROLE=ctx bash v1-store/logs/o16-20261028/checkpoint-backup.sh
ROLE=modul bash v1-store/logs/o16-20261028/checkpoint-backup.sh
bash v1-store/logs/o16-20261028/delivery-restores.sh
```

| 运行载体 | SHA256 |
|---|---|
| formal-chain.sh | bf3a7beb4a41e277598cd75a08a45a13e5ce50f7f20b3cc363a5af0ca4dd3511 |
| checkpoint-backup.sh | 2d257ffd336044245e89b90e16f4ce15770926985044db2d5a196956aedf54b1 |
| delivery-restores.sh | 8f75927550577dfab2604912e0d0db667d07a4ef9450ce5e62cdbe18ce4dcce4 |

chain设置`FINAL_ROOT=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/orig16-stage2`，清除MAX_STEPS/DRY_RUN，原入口固定60000步、实际history/run名、0–7卡。单条完整TRAIN_CMD及所有实际路径由日志记录；通用缓存均在v1-store，uv缓存`/scratch/hongze/.cache/uv`。W&B沿用已认证的本地环境文件，未将凭据写入命令。用`git show <TRAIN_HEAD>:src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml`和同HEAD的训练配置/入口脚本还原。

每条备份先等该训练日志出现，再执行原inc；inc真实失败保留，随后等对应训练终态并调用一次原final（后者仍检查0退出、完整12步、完整SHA和匿名读回）。mod备份会话可预先等待，实际inc只在第二条开始后进入。G12等chain结束和各stage清单就绪，依次调用`check_stage2_checkpoint.py --mode delivery --delivery-step 59999 --checkpoint <stage>/59999 --expected-manifest <stage>/manifests/step-59999.json --online-gpu 1 --out <VALID>/G12-<role>.json`；报告在stage外，父GPU0/sidecar1，清除CPU后端环境。第一条G12失败仍记录并执行第二条，不影响上传。

## 四、数据及同源口径

唯一新库`16task-orig100-1600ep-9e109419`，1600集、768897帧、476857执行样本；双帧表均全量verified，使用4×4一档；motion表23253行，原完整验证/全表split零差，最大44窗小于96。16份源H5与encoder逐文件SHA集合相同；不同格式清单的摘要不冒充同一字符串。准确norm、episode/input/canonical manifest及动作表SHA见[build-pins](../../dataset-build-doc/16task-orig100-1600ep-9e109419/records/build-pins.json)。

两条G4、八组合矩阵、两条49项真实启动合同、完整8卡20步和严格SAVE_LOAD均通过，详见[建库结果](../../dataset-build-doc/16task-orig100-1600ep-9e109419/result.md)。原缓存跨精度路径仅要求输入/身份/形制/有限性，不要求encoder输出数值相同；checkpoint固定参考保存恢复仍严格0差。

## 五、固定配置与交付

本条run为`orig100-ctx32x4x4-motion-s2split4x256-b128-60k`，history为`perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml`。配置`mme_vla_suite_orig16_b128_60k`，B128、FSDP8、workers16、seed42、60000步；512视觉位（32帧×4×4）+384动作位（96窗×4token），每token256，融合context、宽度2048。warmup5000、peak_lr5e-5、decay_steps60000、decay_lr5e-5、EMA0.999；其余保持既有配置，正式从pi05_base新初始化，不续接20步checkpoint。

每条保存5000、10000、15000、20000、25000、30000、35000、40000、45000、50000、55000、59999共12份，日志间隔100。checkpoint根`v1-store/train-runs/mme_vla_suite_orig16_b128_60k/orig100-ctx32x4x4-motion-s2split4x256-b128-60k`；公开目标`HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2split-60k-v1`。本地交付stage为`v1-store/exports/hf-ckpt-orig100-ctx32x4x4-motion-s2split4x256-b128-60k/stage`，G12消费该stage的真实依赖实体，不能用原run目录代替。

## 六、硬件、句柄、日志及监督

AWS 8×A100-SXM4-80GB、本地NVMe RAID。训练8卡独占并串行；G12仅训练结束后使用0/1；上传禁GPU。短测墙钟进度窗口context约2.4063秒/步、modulation约0.9217秒/步，仅作粗略参考且窗口较短，不声称长期精确速度。所有正式任务由主会话监督，初始整体预算72小时；出现实际失败保持原日志并按用户要求继续可执行独立事项。

| 会话 | 内容 | 日志 |
|---|---|---|
| o16-train-chain | 原context→modulation | v1-store/logs/o16-20261028/formal-chain.log；各run另有v1-store/logs/<run>.log |
| o16-backup-ctx | context增量→最终备份 | v1-store/logs/o16-20261028/checkpoint-backup-ctx.log |
| o16-backup-modul | 等第二条开始后增量→最终备份 | v1-store/logs/o16-20261028/checkpoint-backup-modul.log |
| o16-g12 | 两条正式59999严格交付恢复 | v1-store/logs/o16-20261028/delivery-restores.log；VALID/G12-ctx.json和G12-modul.json |
| o16-trainset-up | 已启动的训练库上传/匿名回读 | v1-store/logs/o16-20261028/trainset-export.log |

不等待context上传或G12就接modulation；训练库上传也不阻塞训练。若chain提前结束、某条没有实际输入，不把等待或未执行项记成成功。当前不做benchmark评估。结果、权重及恢复素材留v1-store，Git档案只收实测日志、指标和JSON。
