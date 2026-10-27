# Stage2 VLA20k早期试评估启动档案

## ① 目的与当前结论

用户批准固定运行名 `official-stage2-20k-probe-1026-a`，先单局，再验证两个独立A40 worker的固定chunk分片。目标是阶段一组件与早期策略试评估，不是正式60k的800局成绩。当前仅完成资源身份与三套环境导入；策略、渲染、跨卡数值与分片结果均未验证。

## ② 版本与代码状态

开工计划commit为 `2789ae42f3166788ec7d5efc4baf29d8d5a1bc4f`，开发分支 `v4-eval-official-1026`。官方benchmark gitlink固定 `856bc3a189d4172f3f47dbee4424d585f8d78db3`，MotionJEPA固定 `b5d23a5fab55be16e580c5b77a8f04d68b3170c2`。最终执行代码commit与只读release尚待审查发布，不把占位提交时的HEAD称为已跑策略版本。

早期VLA清单固定为公开bucket `HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1` 的 `manifests/step-20000.json`，SHA256为 `be34e777b9ad8962aca374d2861e97cf5561b0cc9fb63fb7160e99adf03f2452`。它绑定36个成员，共12,220,441,623字节；根清单SHA为 `10848d66aa3b281d23c8029033bea4ee31d269bf8abd89d69458c5792a7cda42`。当前仍在下载、逐文件核验和READY发布过程，目录存在不构成就绪。

绑定encoder为epoch105 EMA，314,265,763字节，SHA256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`。配置为1024维、96槽、window33、stride32；不使用NFS既有的768维旧10k权重。已交付恢复输入有131帧RGB、state、wrist和原 `[33,32]` 批次记录，不等于A100输出参考数组。

## ③ 启动与配置还原

只读资产与输出根为：

```text
/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/
  v1-store/assets/vla20k-be34e777b9ad8962aca374d2861e97cf5561b0cc9fb63fb7160e99adf03f2452/
  v1-store/assets/stage2-epoch105-27daa832/
  v1-store/evaluation/official-stage2-1026/official-stage2-20k-probe-1026-a/
```

resolved快照仅从交付读取，SHA256为 `b6298ad5ddf134866cf68b675e4f6ecefff36ca28893aef099d5fc02ed451532`；normstats和全部参数以同一step清单逐文件核验。训练YAML、provenance及packed store不原地修改。

两份占位各使用以下已获批规格，完整回执和精确comment在运行根 `pool/`；工作负载尚未启动：

```bash
sbatch --parsable --account=chaijy2 --partition=spgpu --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:1 --gpu_cmode=shared --cpus-per-task=4 --mem=24G \
  --time=48:00:00 --wrap='sleep infinity'
```

后续负载必须使用对应 `srun --jobid=<确切ID> --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared`，不能在登录节点或Aspen加载模型。最终server、sidecar、client完整命令在固定实现commit审定后补记，不拿拟命令当已执行。

## ④ 测试集与划分

官方16任务×test50的元数据保持原样；小规模验证只选其中固定成员，model seed=42，episode seed从官方元数据读取。先验证单局，再两个worker领取不同五局chunk。覆盖集合、实际执行和重试均分别留证，不因正常失败重跑；这些样本不能报告成正式800覆盖。

## ⑤ 参数与环境

动作joint_angle；horizon20、执行16、obs_horizon16；先step再count加一，count>1300才timeout。三套NFS环境分别为 `v1-store/envs/{policy,stage2,official-sim}/bin/python`，torch版本分别2.7.1、2.9.0、2.9.1。实际解释器均指向NFS uv managed CPython3.11.14。运行时缓存显式指向本eval的 `v1-store`，不修改训练环境。

## ⑥ 硬件、资源与句柄

| 槽位 | JobID | 节点 | GPU UUID | 起跑（美东10月26日） |
| --- | --- | --- | --- | --- |
| probe-s0 | 64849606 | gl1504 | GPU-35f2828a-a081-2b03-35af-da47dcf4c16d | 23:01:24 |
| probe-s1 | 64849621 | gl1504 | GPU-62a22207-683d-6af4-4c92-91c12eca31d4 | 23:01:42 |

两张卡均为NVIDIA A40，驱动595.71.05；各作业内可见token均为0，对应不同物理UUID。每份48h，提交时全额承诺各48 GPUh、合计96 GPUh；这不是实际消耗。试验完成即按清单逐个释放，不能因成绩或视频欠账保留空占位。

本轮创建的tmux清单：`eval-stage2-1026-encoder-fetch`、`eval-stage2-1026-vla20k-fetch`（原IPv6尝试，均已精确停止）；`eval-stage2-1026-env-probe`（正常退出0）；`eval-stage2-1026-encoder-ipv4`、`eval-stage2-1026-vla20k-ipv4`（下载与校验中）。不操作其他用户现场或其他任务句柄。

环境探针于美东23:05:23核完，三项 `NFS_ENV_IMPORT=PASS`，两个 `GPU_QUERY_EXIT_CODE=0`，`EXIT_CODE=0`，日志镜像逐字节相同。日志为运行根 `logs/env-probe.log`，SHA256 `f5cf42e2abc728d64edaff5a34aabbaffcf3cbdd6221468bdb528c06e26b0440`。官方仿真导入出现Vulkan ICD警告，尚不能宣称渲染可用。

## 用户决策原话

1. 「你应该能够拿到这个已经有的这个 checkpoint，然后你可以尝试 eval 一下这个效果，就是已经有的这个 VIA 的这个训练的 checkpoint，应该是早期的这个 checkpoint。」
2. 「Motion的encoder我应该也上传了Hugging Face，你可以直接下载，应该是最新的这个Epoch一百多的那个。」
3. 对20k、运行名及单份资源建议选择「采用这组运行名和规格」。
4. 随后要求「你query两个这个A40的叫步，然后尝试一下这个多卡的分片是不是正常的。」据此使用两个独立1GPU作业，不扩大为正式800局。

## 计划外事件

首次HF CLI及Python urllib传输均在不可达IPv6地址停留，`ss`原始输出为相关精确PID的 `SYN_SENT`；尚无已验证文件。美东23:07:58按本轮确切tmux名逐一停止，删前删后其余会话完全保留，原日志追加停止原因；不把强停写成成功退出。随后仅把传输改为 `curl -4`，对象、清单、SHA与预算不变，各进行一次明确基础设施重试。旧暂存目录保留。

IPv4传输已产生真实文件进度；随后NFS读取、写入及摘要出现D状态等待，部分源代码复测同样受影响。只记录现场事实，不重算模型、不增加并发或重试，不把仍在读取中的校验写成通过。
