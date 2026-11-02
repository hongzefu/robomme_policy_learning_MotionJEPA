# orig100 nostate motion 表建库结果

## 一、一句话结论与指标

建库完成。表为 `motion-stage2-1024-nostate`，2,496 段、23,253 行、每行 `float32 (1,1024)`、95,244,288 字节；完整校验零失配、零非有限值。表 SHA256 为 `1f399ccb645975f7719b4de9555911b46f273d54158e36e8883d2f7315a3b155`，状态 `verified`。本结果是输入与建表证据，VLA 集群冒烟和正式训练尚未执行。

```text
STAGE2_EXPORT=PASS segments=2496 rows=23253
EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1
EXPORT_PARITY=NOT_REQUIRED reason=user_decision_same_input
PACK_PROVENANCE_PRECHECK=PASS mode=vae_compat segments=2496
VERIFY_MOTION=PASS scanned=23253 mismatches=0 nonfinite=0 segment_set_mismatch=0 real_plus_pad_ne_33=0 exec_padded=0 pad_not_last_demo=0 pad_source_mismatch=0 demo_tail_padded=386 demo_tail_dropped=514 legal_zero_segments=4
BUILD_DONE=PASS tag=nostate
EXIT_CODE=0
```

## 二、版本与代码状态

起跑 Beta 为 `4c05451082ea9617cf7c721d769f2be84f851471`（commitV11.82Beta），工作区干净，完整起跑行保留在 records/build.summary.log。固定 4a runner 来源与本地参数适配、实际 b5 包的模块 SHA、两个独立 encoder/config 身份均见 launch.md；未修改原权重或配置。

## 三、启动与配置还原

按 launch.md 完整命令运行 `o16abl-build-nostate`，先单段实际冒烟，再全量导出、G3、pack、verify。源代码可用 `git show 4c05451082ea9617cf7c721d769f2be84f851471:scripts/dataset/wan_stage2/export_stage2_memory.py` 等相应入口还原。载体 SHA 为 `256969bba47332cb5df6791dd1f15d0245d5e1446016725039c0c052decf2768`。

## 四、数据集与划分

消费同一 orig100 1600 集库及两份固定 H5/episode manifest。G3 从完整已核验缓存的 val 集固定 seed=0 取 32 窗，aug0；真实 fetch 14 片、115 对象、32,476,096,020 字节。latent 13 tar、7,489 成员全量 SHA 与共享副本 SHA 均通过。本机导出和 pack/verify 消费已核本地快盘 latent，表及证据输出在 turbo；原库 evidence 原件保持。

## 五、关键参数

grid-subsample=2、batch=1；smoke max-segments=1、dropout诊断关闭；完整导出 expected-counts=2496,23253、check-dropout=64，eval 重复最大差为 0。G3 的 65 叶 EMA 逐张量精确且拒绝 live；缓存路线输出数值按用户决定标 NOT_REQUIRED。VLA 参数仍是之后的新配置 global64/FSDP4/80k/seed42，不计入本次数据构建。

## 六、硬件与耗时

Aspen 两张 RTX A6000，按标签并行导出，G3 在 GPU0 依次执行。nostate 起跑美东 2026-11-02 08:18:50，结束 08:49:52，端到端 31 分 2.7 秒。现场存在他人服务小显存上下文，未干预，数字按共享现场记录。输入下载分别在美东 03:28:38、02:26:12 完成；后续建表未及时衔接，至上述时刻才实际起跑，此延迟如实保留为流程事件。

## 七、构建过程

先真实单段 17 行通过，再完整 2,496 段/23,253 行。64 窗 dropout 诊断 eval_repeat_max_abs=0，训练模式与eval有实际差异。G3 全 32 窗形状与有限性通过。pack 校验源 H5、latent 及逐窗/输出身份，verify 扫全表，386 个 demo 尾窗合法补齐、514 个尾窗按规则丢弃、4 个合法零段；exec未补帧。

## 八、启动链路验证

完整 CPU 范围 962 项通过、7 项跳过；旧身份默认 DRY 标准输出和错误输出逐字节一致。真实新表的本标签 20、100、80000 三条完整启动契约各 54 项通过；五种非法组合均实际拒绝。这里只是 CPU 契约，四卡训练由后续 Slurm 冒烟验收。

## 九、用户决策

用户原话依次为「/data/hongzefu/robomme_policy_learning_MotionJEPA/1101-gl-ablation-encoders-ctx-b64-80k-plan.md 开工」「同意修订 继续做」「不需要再确认，你可以直接开始了。」「继续工作」。按既定口径两个run的64档均通过且第一阶段完成后，直接进入正式训练；任一OOM则测完32/16，取消两占位再交用户。

## 十、计划外事件

S6 补既有 latent 解包入口；S7 为固定 runner 增加显式来源规格路径；补齐在线输入两素材；三个 latent 参数转本地快盘；GPU0锁覆盖noxyz导出和G3。均已独立 `ADJUST_REVIEW=PASS scale=same findings=0`，源身份和数值判据保持。

载体初稿的 dropout=1 在启动前被审查拒绝，修为 smoke=0、完整=64，原错误版本未执行。本轮漏设 HF 缓存变量，构造时产生了 HOME 下 724 字节配置及40字节引用；文件后来已按固定 snapshot 的公开字节核验并镜像至 P/G 的 STORE 缓存，原 HOME 文件保留。原实际落点不能追溯称为全程合规。该配置只初始化持久 affine buffer，随后由 checkpoint 严格覆盖、逐叶核对65项，独立审查确认已生成结果无需重跑；未来启动显式缓存变量。详见 bootstrap_cache_record.json。

## 十一、结论与下一步

本表及证据已可供 VLA 启动验证。下一步完成对应训练 launch，另打 VLA Beta、部署到 G，在已运行的 4 卡占位内做 64 档 20 步冒烟；两个run均通过后按计划继续 SAVE_LOAD、100步测时和正式80k。表验证完成不表示策略性能已验证。

## 十二、归档

records/ 保存新 G3、表元数据、encoder完整性记录、缓存镜像记录、清洗后构建日志及真实启动契约输出。构建日志去掉重复逐段中间态，保留全部判定和退出行；权重、原始/派生数据及原日志仍在 v1-store，档案没有脚本或 YAML 拷贝。

