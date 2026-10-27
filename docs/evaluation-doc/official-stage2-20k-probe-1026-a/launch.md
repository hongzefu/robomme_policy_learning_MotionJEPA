# Stage2 VLA20k早期试评估启动档案

## 最新执行口径：美东10月27日简化重启

用户已批准：「同意，按这个简化方案开工，保留弹性 worker 队列，不要再堆叠功能和校验。尽快先跑通一局，再做两卡分片评估。」现行方案为根目录 `1027-stage2-simple-elastic-eval-plan.html`，覆盖下文历史章节中的绑定收据、独立数值恢复、跨硬件及通用资源账本前置要求；历史失败记录保留。

新作业 `64855345`（gl1504）与 `64855348`（gl1520）于美东00:59分配，各1 A40／4 CPU／24 GiB／48小时。旧作业 `64849606`、`64849621` 已取消。先使用第一份作业直接加载真实20k并执行 `BinFill:0`，不再先跑 `check_runtime`。两卡十局通过同一弹性队列领取两个五局块，和单局结果分目录计数。

首个启动版本为 `922498b2b7be87b07a5f5065a0bc5e343495683c`；修复默认心跳误断后，完整单局及两卡评估统一固定为 `789ef0e5e9c241dd0f8bca11f2c3b885edc52e39`。B服务入口合前复核通过，整合后入口定向4项测试通过，3.44秒；最终队列及真实连接4项定向测试通过，0.24秒。已有权重与encoder复用，仿真环境按官方既有锁定server组补齐3个通信包。新产物子目录为原运行根下 `simple-1027/`，不覆盖旧日志。首局grant为 `simple-1027/smoke-grant.json`，seed540000／easy，run身份 `7c4fafa6800d952b56810747ed4d4f9d5c542d3fd5a0d142c6b0be3107bc5c6b`。

首局完整入口为：

```bash
ssh -o BatchMode=yes greatlakes srun --jobid=64855345 --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared \
  /usr/bin/bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/official-stage2-1026/official-stage2-20k-probe-1026-a/simple-1027/run-single.sh \
  789ef0e5e9c241dd0f8bca11f2c3b885edc52e39 smoke1-episode-retry1
```

脚本是本轮运行产物，显式设置下文同一套缓存与解释器，使用简化服务CLI启动policy及sidecar，监听后直接执行官方客户端。普通可选运行配置沿用 `contracts/runtime.json` 供既有sidecar加载；不再传模型绑定或独立运行检查器。服务端口18027，完整单局结果目录 `simple-1027/smoke1-episode-retry1/`；外层tmux名 `eval-simple-1027-smoke1-episode-retry1`，日志同名。模型监听只说明开始服务，真实单局终局另记结果。

两卡阶段的输入为 `simple-1027/shards10/manifest.json` 与 `spec.json`，任务复用既定 `stages/shards10/scope.json`，run身份为 `d0bcf65d9c91b7aec4644d7da61dfda9b8579851735571aa32506fb2a369e212`。Aspen唯一控制器入口为本机运行产物 `simple-1027/run-controller.sh`，tmux名 `eval-simple-1027-controller`；读取固定789ef0e源码，不做GPU计算。初始化使用 `official_queue initialize --run-root <shards10目录> --manifest <既定scope> --spec <shards10-spec-input.json>`；控制器命令为 `official_queue controller --run-root <shards10目录>`。

两个worker的完整启动口径为：

```bash
EVAL_ROOT=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval
RUN="$EVAL_ROOT/v1-store/evaluation/official-stage2-1026/official-stage2-20k-probe-1026-a/simple-1027"
CODE=789ef0e5e9c241dd0f8bca11f2c3b885edc52e39
ssh -o BatchMode=yes greatlakes srun --jobid=64855345 --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared /usr/bin/bash "$RUN/run-worker.sh" "$CODE" worker-1
# 先确认worker-1已领取第一块，再启动第二个worker；无需重新分片。
ssh -o BatchMode=yes greatlakes srun --jobid=64855348 --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared /usr/bin/bash "$RUN/run-worker.sh" "$CODE" worker-2
```

两条命令分别放入 `eval-simple-1027-worker-1`、`eval-simple-1027-worker-2` detached tmux，pipefail、tee和真实退出码落本机 `simple-1027/worker-1.log`、`worker-2.log`。worker脚本在计算step内设置与单局相同的缓存环境，再执行固定源码的 `resident_worker --run-root <shards10目录> --spec <spec.json> --worker-id <名称> --job-id <Slurm分配ID>`。模型日志位于队列 `logs/`，每块仿真日志位于 `attempts/<attempt>/sim.log`。

## ① 目的与当前结论

用户批准固定运行名 `official-stage2-20k-probe-1026-a`，先单局smoke1次，再两个独立A40 worker各固定五局chunk共10次，总11次，两阶段收据分开。目标是阶段一组件与早期策略试评估，不是正式60k／800成绩。当前已有三环境初始导入及单次环境reset，尚未进行策略rollout／同机恢复／IPC／分片。跨硬件已取消，记 `CROSS_HARDWARE_NUMERIC=NOT_REQUIRED reason=user_decision`，不伪写PASS。

## ② 版本与代码状态

开工计划commit为 `2789ae42f3166788ec7d5efc4baf29d8d5a1bc4f`，开发分支 `v4-eval-official-1026`。官方benchmark gitlink固定 `856bc3a189d4172f3f47dbee4424d585f8d78db3`，MotionJEPA固定 `b5d23a5fab55be16e580c5b77a8f04d68b3170c2`。NFS历史 `releases/420ae9801b4d23ed3a798e29a4580960301b9d20`只提供初始import来源。实际执行版本固定为 `5c5bff55c6ce1ba873058e4537615ff45091eb72`，同名NFS release已核HEAD、benchmark gitlink与clean工作区，独立合后复核通过且代码已推送；不把占位提交HEAD当作策略版本。

早期VLA清单固定为公开bucket `HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1` 的 `manifests/step-20000.json`，SHA256 `be34e777b9ad8962aca374d2861e97cf5561b0cc9fb63fb7160e99adf03f2452`。36成员共12,220,441,623字节；根清单SHA `10848d66aa3b281d23c8029033bea4ee31d269bf8abd89d69458c5792a7cda42`。传输SHA与完整落盘读回均通过，READY已于美东10月26日23:50:43发布，原下载自然退出0。

绑定encoder为epoch105 EMA，314,265,763字节，SHA256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`。已有 `ENCODER_DOWNLOAD=PASS files=3`、`EXIT_CODE=0`。1024维、96槽、W33／S32；不使用旧768维10k。恢复使用真实131帧RGB、state／wrist，D66首67＋四次16续包；原 `[33,32]`记录不能冒充真实16节奏验证。

## ③ 启动与配置还原

只读资产与输出根为：

```text
/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/
  v1-store/assets/vla20k-be34e777b9ad8962aca374d2861e97cf5561b0cc9fb63fb7160e99adf03f2452/
  v1-store/assets/stage2-epoch105-27daa832/
  v1-store/evaluation/official-stage2-1026/official-stage2-20k-probe-1026-a/
```

resolved快照仅从交付读取，SHA256为 `b6298ad5ddf134866cf68b675e4f6ecefff36ca28893aef099d5fc02ed451532`；normstats和全部参数以同一step清单逐文件核验。训练YAML、provenance及packed store不原地修改。

两份占位各使用以下已批规格，完整回执及comment在运行根 `pool/`；已执行环境探针／单次reset，策略工作负载尚未启动：

```bash
sbatch --parsable --account=chaijy2 --partition=spgpu --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:1 --gpu_cmode=shared --cpus-per-task=4 --mem=24G \
  --time=48:00:00 --wrap='sleep infinity'
```

后续负载必须使用对应 `srun --jobid=<确切ID> --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared`，不能在登录节点或Aspen加载模型。最终server、sidecar、client完整命令在固定实现commit审定后补记，不拿拟命令当已执行。

## ④ 测试集与划分

官方16任务×test50元数据原样；验证选固定成员，model seed42，episode seed从官方metadata读取。先单局，再两worker各一个不同的固定五局chunk；前者与后十局分片分开收据及结果集合，核十局分片无重叠。实际执行／基础设施重试分记，正常fail／timeout不重跑，不能报告为正式800覆盖。

## ⑤ 参数与环境

动作joint_angle；horizon20、执行16、obs_horizon16；先step再count加一，count>1300才timeout。三套NFS解释器为 `v1-store/envs/{policy,stage2,official-sim}/bin/python`，torch分别2.7.1／2.9.0／2.9.1，uv managed CPython3.11.14。缓存显式在本eval的v1-store，不改训练环境。strict restore后用本机seed42一次生成完整noise，按真实model.action_dim定shape并锁SHA（可能(1,20,32)，不是固定20×8），num_steps10推理两次，核输出(20,8)、执行(16,8)同机重复性与有限值。先这些静态／恢复和单局smoke，再两chunk实profile，不以未跑五局profile阻挡首局。

## ⑥ 硬件、资源与句柄

| 槽位 | JobID | 节点 | GPU UUID | 起跑（美东10月26日） |
| --- | --- | --- | --- | --- |
| probe-s0 | 64849606 | gl1504 | GPU-35f2828a-a081-2b03-35af-da47dcf4c16d | 23:01:24 |
| probe-s1 | 64849621 | gl1504 | GPU-62a22207-683d-6af4-4c92-91c12eca31d4 | 23:01:42 |

两张卡均为NVIDIA A40，驱动595.71.05；各作业内可见token均为0，对应不同物理UUID。每份48h，提交时全额承诺各48 GPUh、合计96 GPUh；这不是实际消耗。试验完成即按清单逐个释放，不能因成绩或视频欠账保留空占位。

启动历史tmux清单：`eval-stage2-1026-encoder-fetch`、`eval-stage2-1026-vla20k-fetch`（IPv6尝试，已精确停止）；`eval-stage2-1026-env-probe`（正常退出0）；`eval-stage2-1026-encoder-ipv4`（下载通过退出0）、`eval-stage2-1026-vla20k-ipv4`（36文件传输SHA过，完整读回／READY仍待完成）。当前精确句柄以主会话状态机／运行回执为准，不操作他人任务。

环境探针于美东10月26日23:05:23核完，三项 `NFS_ENV_IMPORT=PASS`，两个 `GPU_QUERY_EXIT_CODE=0`，`EXIT_CODE=0`，日志镜像逐字节相同；运行根 `logs/env-probe.log` SHA256 `f5cf42e2abc728d64edaff5a34aabbaffcf3cbdd6221468bdb528c06e26b0440`。当时官方仿真导入有Vulkan ICD警告。后续64849606 step1的真实BinFill testep0 reset通过：seed540000／easy、actions0、D0、初始1观测、RGB256×256×3／state8、elapsed3.9504s、退出0；日志镜像待运行owner收尾。它只证明单次reset，不证明策略rollout或完整渲染评估链路。

## 用户决策原话

1. 「你应该能够拿到这个已经有的这个 checkpoint，然后你可以尝试 eval 一下这个效果，就是已经有的这个 VIA 的这个训练的 checkpoint，应该是早期的这个 checkpoint。」
2. 「Motion的encoder我应该也上传了Hugging Face，你可以直接下载，应该是最新的这个Epoch一百多的那个。」
3. 对20k、运行名及单份资源建议选择「采用这组运行名和规格」。
4. 随后要求「你query两个这个A40的叫步，然后尝试一下这个多卡的分片是不是正常的。」据此使用两个独立1GPU作业，不扩大为正式800局。
5. 「同意这么做。然后这个A100和40的问题你不用再关心了。」
6. 「不需要做到跨硬件的数值性一致。」

前案审查 `ADJUST_REVIEW=PASS scale=same findings=4` 由最新简化复核 `ADJUST_REVIEW=PASS scale=same findings=0 blocking=0` 覆盖。沿现有validation20000，不新增exploratory_validation、签发器、证书或fixture／reference pin。runtime合同schema为eval-runtime-contract-v1，记录用户原话及cross_hardware_numeric=not_required；run绑定runtime_contract_sha256。模型全部成员、resolved／normstats、motion身份、原料SHA／EMA／source／VAE／软件精度严格；实际GPU／driver／架构独立记录，无eval参数训练strict不动。CLI兼容名仍是server的--eval-target-contract／--model-binding及bind-model的--target-contract，文件内容为新runtime合同，整合后按真实--help核对。

## 计划外事件

首次HF CLI及Python urllib传输均在不可达IPv6地址停留，`ss`原始输出为相关精确PID的 `SYN_SENT`；尚无已验证文件。美东23:07:58按本轮确切tmux名逐一停止，删前删后其余会话完全保留，原日志追加停止原因；不把强停写成成功退出。随后仅把传输改为 `curl -4`，对象、清单、SHA与预算不变，各进行一次明确基础设施重试。旧暂存目录保留。

IPv4传输已产生真实文件进度；随后NFS读取、写入及摘要出现D状态等待，部分源代码复测同样受影响。只记录现场事实，不重算模型、不增加并发或重试，不把仍在读取中的校验写成通过。

## 首个GPU负载：严格恢复及同机重复性

本节在GPU起跑前填写。执行release固定为上面的 `5c5bff5`；后续主检出只改档案不会移动执行树。使用已核 `64849606`，不提交新job。原始日志落本机镜像 `logs/restore-initial.log`，报告落NFS运行根 `reports/restore-initial.json`。启动脚本为运行产物 `scripts/run-restore-5c5bff5.sh`，SHA256 `f793cfc63ec532416239231893c3cdb7b90d3096b620ce39070fbe2b5b910e53`；下列代码块还原其全部运行口径：

```bash
EVAL_ROOT=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval
REPO="$EVAL_ROOT/releases/5c5bff55c6ce1ba873058e4537615ff45091eb72"
STORE="$EVAL_ROOT/v1-store"
RUN="$STORE/evaluation/official-stage2-1026/official-stage2-20k-probe-1026-a"
MODEL="$STORE/assets/vla20k-be34e777b9ad8962aca374d2861e97cf5561b0cc9fb63fb7160e99adf03f2452"
# 以下环境在srun的计算step内设置；CUDA_VISIBLE_DEVICES保持Slurm原值。
unset JAX_PLATFORMS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MMEVLA_V1_STORE="$STORE" OPENPI_DATA_HOME="$STORE/models"
export MS_ASSET_DIR="$STORE/cache/maniskill"
export HF_HOME="$STORE/cache/hf" HF_HUB_CACHE="$STORE/cache/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1
export UV_CACHE_DIR="$STORE/cache/uv" UV_LINK_MODE=copy
export XDG_CACHE_HOME="$STORE/cache" TORCH_HOME="$STORE/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$STORE/cache/torchinductor" TRITON_CACHE_DIR="$STORE/cache/triton"
export CUDA_CACHE_PATH="$STORE/cache/cuda" JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO:$REPO/src:$REPO/packages/openpi-client/src:$REPO/third_party/robomme_benchmark/src"
cd "$REPO"
"$STORE/envs/policy/bin/python" -u -m scripts.evaluation.check_runtime \
  --checkpoint "$MODEL/20000" --step 20000 --purpose validation \
  --motion-online-gpu "$CUDA_VISIBLE_DEVICES" \
  --stage2-python "$STORE/envs/stage2/bin/python" \
  --eval-target-contract "$RUN/contracts/runtime.json" \
  --expected-manifest "$MODEL/manifests/step-20000.json" \
  --config mme_vla_suite_b128_80k --output "$RUN/reports/restore-initial.json"
```

从Aspen复用主连接的外层命令为 `ssh -o BatchMode=yes greatlakes srun --jobid=64849606 --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared /usr/bin/bash <运行根>/scripts/run-restore-5c5bff5.sh`。放在本轮唯一 `eval-stage2-1026-restore` detached tmux，使用pipefail、tee及真实EXIT_CODE。该负载不计入11个官方仿真局，不产生任务成功率；先做完整参数树、bf16摘要、真实131帧和完整noise的同机重复性。真实策略单局／两分片在此负载之后执行。
