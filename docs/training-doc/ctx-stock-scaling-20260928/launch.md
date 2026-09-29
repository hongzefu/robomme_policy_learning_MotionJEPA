# 原训练入口的八卡与双四卡性能诊断

> 本文件记录已获授权的短程性能诊断，不是正式60k/80k训练的启动回执。唯一被测源码为干净快照 `v1-store/bench/context-stock-scaling-20260928/source-6db0e0a`，完整提交 `6db0e0ab9ef266d0e80fc1dd8d4383705bd81ab0`。本轮以原 `scripts/training/train.py`、原模型、原数据加载器和原读取守卫运行；每个进程401步、global batch128。结果和退出状态以各轮原始记录为准，不能从本文件的验收要求推断已通过。

## 用户授权与本轮边界

用户相关原话按时间顺序保留：

1. 「不要这么干 评估8gpu的训练效果 2个4gpu并行 对比8gpu 会慢多少 不要修改内部训练机制 除非用户授权」
2. 「尽可能穷尽所有选项 但是不要修改训练链路 只是让你仔细检查」
3. 「不要修改训练计算的机制 如果发现训练计算机制有改进空间只做记录」
4. 「dataloader也是 读取逻辑也不要改」

本轮回答同一模型使用八卡独占、两个四卡任务并行时，各任务慢多少、合计吞吐如何变化。先测context512+motion160，再测纯视觉context512；两种模型各自做独立的八卡前基线、双四卡并行、八卡后基线。计算改进仅记录，不实施mask、attention、loss、精度、重计算或梯度累积改造；不替换数据加载器、sampler、读取配置或批次组装，不附加样本索引键。

主工作区的三个未提交测量草稿 `scripts/training/tests/context_gpu_scaling.py`、`compare_context_gpu_scaling.py`、`run_context_gpu_scaling.sh` 已冻结，本轮不使用它们。旧mask候选、配对modulation reader、独立读取器和修改过的损失探针也不进入本轮进程。

本轮为环境B，工作盘是AWS本地NVMe RAID `/dev/md0`，八张A100-SXM4-80GB。所有持久产物位于 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/`；不连接集群，不使用Slurm或turbo。

## 源码、环境与配置身份

以下缩写只用于本文件说明，均指绝对路径：

```text
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
BASE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/context-stock-scaling-20260928
SOURCE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/context-stock-scaling-20260928/source-6db0e0a
DATA=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-v2-1600ep-604f16da
RUN=ctx-stock-<variant>-<phase>-20260928
OUT=BASE/RUN
```

每次启动先核 `SOURCE` 的HEAD等于上述完整提交、工作区无改动。解释器使用主仓库由uv管理的 `MAIN/.venv/bin/python`，本轮只读该环境，不运行环境同步或依赖安装。`PYTHONPATH` 显式指向快照的 `src` 与 `packages/openpi-client/src`，`PYTHONDONTWRITEBYTECODE=1`；独立CPU前置进程核验 `openpi`、`openpi_client`、`mme_vla_suite`、模型和数据模块的实际导入位置都在快照内，避免主环境editable路径使导入退回主树。

外层脚本为 `BASE/run_original.sh`，前置核查为 `BASE/preflight.py`。核对本文件时两者SHA256分别为 `c908db7c6549b4e0a898ef4e2d070035b566ddcca423a72360d1c8dd895d39bb`、`d9424486a0f4f223866a62519a449b92c6e482ecfc99cb87965e924b7e418a73`；运行期间不得原地编辑。前置进程保存完整CLI解析配置、解释器及包版本、源码摘要和导入路径，逐字节核对八项核心源码/环境文件与快照提交及主树一致。该进程只做核查，不包装或替换被测进程的函数。

| 模型 | 原具名配置 | history来源 | 保留的关键口径 |
|---|---|---|---|
| `motion` | `mme_vla_suite_b128_80k` | `MAIN/v1-store/diagnostics/context-gpu-throughput-20260928/history-context-motion160.yaml` | 512帧token+160 motion；workers16；完整保留demo17、repeat_last、full1600编码器来源 |
| `plain` | `mme_vla_suite_b128_60k` | 快照内 `perceptual-framesamp-context-8frame-8x8.yaml` | 512帧token、motion关闭；workers8 |

motion配置只是用户要求的context选择：相对已训练的modulation512+motion配置，仅将 `integration_type` 选为 `context`、必要输出宽度选为2048，motion数据和语义保持不变。该配置SHA256为 `b69a4bd4398b08298c6dc12db7150a0873c327b9555ceee4a5166094c1a610ac`；纯视觉配置SHA256为 `299b6d3f2b52c7b3c1894f8594183a3140d0d2a6b91250a2eb8e568c528344d8`。两态均直接交给原生产数据入口，不使用modulation配对reader。

两组均为global batch128、seed42、原pi05/Gemma全宽全层、原图像冻结规则、AdamW及clip1.0、EMA0.999。学习率保留warmup5000和peak/decay均 `5e-5`，`decay_steps` 分别保留原60000/80000；不因401步诊断缩短或重设学习率曲线。动作维32、horizon20、文本上限64及数据变换均继承原配置。

诊断仅覆盖 `num_train_steps=401`、关闭W&B网络、唯一run名/输出路径以及目标FSDP4或8；原 `log_interval=100`、`save_interval=5000`、`keep_period=5000` 保持。401次更新的循环编号为0至400，原训练入口在最后一轮必保存checkpoint `400` 并等待异步保存完成，不能将它解释成“本轮不保存”。

## 数据与资产

两种拓扑使用同一份1600集库：BinFill、RouteStick、VideoRepick、VideoUnmaskSwap各400集，共605611个执行样本、1192918帧。数据根为上方 `DATA`，原数据加载器读取 `DATA/framesamp-8x8`，pkl源为 `DATA/source`。不重建数据、不改变shuffle/seed/drop_last、spawn、persistent workers或原prefetch行为。

| 对象 | 固定位置或指纹 |
|---|---|
| 清单文件 | `DATA/meta/episode_manifest.json`；SHA256 `df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482` |
| 帧库元数据 | `DATA/framesamp-8x8/meta/store_meta.json`；SHA256 `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` |
| 归一化 | `MAIN/v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`；SHA256 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173` |
| motion | `DATA/motion`；窗口33、stride16、budget160、demo17补尾；编码器 `wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder` |
| 初始化权重 | `MAIN/v1-store/models/openpi-assets/checkpoints/pi05_base/params` |
| tokenizer | `MAIN/v1-store/models/big_vision/paligemma_tokenizer.model` |

每个run独立从同一pi05_base初始化，不承接前一轮权重或优化器状态。资产前置采用快照 `scripts/assets/ASSETS_LOCK.json` 的cheap检查；这不是重新完整哈希所有大文件的声明。解释器、数据/资产路径和实际history解析值保存在每轮取证文件中。

## 轮次、名称与会话

先完成motion的三阶段，再完成plain的三阶段。八卡前后基线必须是两个独立run与独立原始日志，不将同一个结果复制成前后两次。每轮起跑前核实所选GPU没有其他计算任务、run/输出/checkpoint目录不存在。

| variant | phase | 物理GPU | FSDP | run_name |
|---|---|---|---:|---|
| motion | 8before | 0,1,2,3,4,5,6,7 | 8 | `ctx-stock-motion-8before-20260928` |
| motion | 4left | 0,1,2,3 | 4 | `ctx-stock-motion-4left-20260928` |
| motion | 4right | 4,5,6,7 | 4 | `ctx-stock-motion-4right-20260928` |
| motion | 8after | 0,1,2,3,4,5,6,7 | 8 | `ctx-stock-motion-8after-20260928` |
| plain | 8before | 0,1,2,3,4,5,6,7 | 8 | `ctx-stock-plain-8before-20260928` |
| plain | 4left | 0,1,2,3 | 4 | `ctx-stock-plain-4left-20260928` |
| plain | 4right | 4,5,6,7 | 4 | `ctx-stock-plain-4right-20260928` |
| plain | 8after | 0,1,2,3,4,5,6,7 | 8 | `ctx-stock-plain-8after-20260928` |

预计超过5分钟的各阶段均由detached tmux运行。拟用会话清单为 `ctx-stock-motion-8before-0928`、`ctx-stock-motion-dual4-0928`、`ctx-stock-motion-8after-0928`、`ctx-stock-plain-8before-0928`、`ctx-stock-plain-dual4-0928`、`ctx-stock-plain-8after-0928`。双四卡会话同时启动该variant的4left和4right两个独立原训练进程，记录并等待各自精确PID与退出码；不能依序启动两个四卡进程后称其并行。实际起过的会话名须据启动记录追加核实，清理只允许按本轮确切会话名单逐个处理。

## 外层命令与完整环境口径

外层接口固定为 `run_original.sh motion|plain 8before|4left|4right|8after GPU列表 FSDP`，例如：

```bash
bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/context-stock-scaling-20260928/run_original.sh motion 8before 0,1,2,3,4,5,6,7 8
bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/context-stock-scaling-20260928/run_original.sh motion 4left 0,1,2,3 4
bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/context-stock-scaling-20260928/run_original.sh motion 4right 4,5,6,7 4
bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/context-stock-scaling-20260928/run_original.sh motion 8after 0,1,2,3,4,5,6,7 8
```

其中4left、4right两条由双四卡会话并发发起；plain阶段将variant替换为plain并使用自身会话、run名。完整展开后的训练命令由每轮 `OUT/command.txt` 留存，其结构为：

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
BASE="$MAIN/v1-store/bench/context-stock-scaling-20260928"
SOURCE="$BASE/source-6db0e0a"
cd "$SOURCE"
"$MAIN/.venv/bin/python" "$SOURCE/scripts/training/train.py" "$CONFIG" \
  --exp-name "$RUN" \
  --dataset-path "$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-8x8" \
  --model.history-config "$HISTORY" \
  --assets-base-dir "$MAIN/v1-store/train-assets" \
  --data.assets.assets-dir "$MAIN/v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
  --data.assets.asset-id robomme \
  --checkpoint-base-dir "$BASE/checkpoints" \
  --fsdp-devices "$FSDP" --num-train-steps 401 --no-wandb-enabled
```

`CONFIG/HISTORY/RUN/FSDP` 只能从上方矩阵和配置表取值；不另传batch、worker、学习率、优化器、日志或保存频率覆盖。以下是已核对外层脚本显式设置的完整环境清单，表内路径缩写展开规则见前文：

| 环境变量 | 值 |
|---|---|
| `PYTHONPATH` | `SOURCE/src:SOURCE/packages/openpi-client/src` |
| `PYTHONUNBUFFERED`、`PYTHONDONTWRITEBYTECODE`、`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS` | 均为 `1` |
| `TZ` | `UTC` |
| `UV_CACHE_DIR` | `MAIN/v1-store/cache/uv` |
| `OPENPI_DATA_HOME` | `MAIN/v1-store/models` |
| `XDG_CACHE_HOME`、`XDG_DATA_HOME` | `MAIN/v1-store/cache/xdg`、`MAIN/v1-store/cache/xdg-data` |
| `HF_HOME` | `MAIN/v1-store/cache/hf` |
| `HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE` | 均为 `1` |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` |
| `CUDA_VISIBLE_DEVICES` | 本轮矩阵中的物理GPU列表 |
| `JAX_PLATFORMS` | 被测训练进程为 `cuda` |
| `XLA_PYTHON_CLIENT_PREALLOCATE`、`XLA_PYTHON_CLIENT_MEM_FRACTION` | `false`、`0.95`，所有拓扑相同 |
| `CUDA_CACHE_PATH`、`MMEVLA_JAX_CACHE_DIR` | `OUT/cache/cuda`、`OUT/cache/jax` |
| `WANDB_MODE` | `disabled` |
| `WANDB_DIR`、`WANDB_CONFIG_DIR` | `OUT/cache/wandb`、`OUT/cache/wandb-config` |
| `WANDB_CACHE_DIR`、`WANDB_DATA_DIR` | `OUT/cache/wandb-cache`、`OUT/cache/wandb-data` |
| `MMEVLA_FRAMESAMP_SOURCE`、`MMEVLA_FRAMESAMP_MANIFEST` | `DATA/source`、`DATA/meta/episode_manifest.json` |
| `MMEVLA_MOTION_STORE` | `DATA/motion`；plain仍由 `motion.enabled=false` 关闭运动路 |
| `MMEVLA_FRAMESAMP_VERIFY` | `fast` |
| `TRAIN_RECORD_DIR` | `OUT/records` |
| `TRAIN_TIMING_STEPS` | `0` |

脚本先清除继承的全部 `TRAIN_*`、`BENCH_*`，再设置上表两项TRAIN变量；另清除 `XLA_FLAGS`、`MMEVLA_FRAMESAMP_ALLOW_SUBSET`、`MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED`、`MMEVLA_MOTION_ALLOW_UNVERIFIED`、`JAX_ENABLE_X64`、`JAX_DEFAULT_MATMUL_PRECISION`、`JAX_DISABLE_JIT`、`JAX_DEBUG_NANS`。不覆盖HOME，不写主环境包文件。

前置核查单独临时使用 `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`；因此 `preflight.json` 的观察进程环境不能冒充训练实际GPU环境，训练侧以 `records/runtime.json`、外层GPU身份和命令为准。原入口已有 `TRAIN_RECORD_DIR` 指标记录器，两侧均启用；`TRAIN_TIMING_STEPS=0`，不额外安装计时器、不增加同步点，W&B网络两侧均关闭。

外层以 `set -o pipefail`、`PYTHONUNBUFFERED=1` 和 `tee` 留 `OUT/train.log`，单run最长7200秒，超时先TERM并保留30秒退出宽限，最终记录真实 `EXIT_CODE=`。独立 `nvidia-smi` 每500毫秒采样八卡，记录时间、index、UUID、利用率、显存、温度、功耗和频率；还留起跑前后计算进程清单。NVML是采样观察，不能称瞬时精确峰值；双四卡时按各run所分配的GPU过滤，不能把整机八卡统计算给单个任务。

## 比较窗口、并发覆盖与完成判据

唯一速度来源是原 `train.py::_MetricsProxy` 在原log100同步后写入的 `records/metrics.jsonl`。每轮应有step `0/100/200/300/400` 的原记录；不用tqdm刷新间隔作主计时，不插入新 `block_until_ready`，不修改原读取循环。

主窗口与后段复核分别为：

```text
t_100_300 = (wall_time(step=300) - wall_time(step=100)) / 200
t_200_400 = (wall_time(step=400) - wall_time(step=200)) / 200
samples_per_second = 128 / t
```

每个窗口覆盖200次更新。100→300对应更新101至300，200→400对应更新201至400；两个窗口重叠100次更新，不是两次独立重复。初始化、首次编译和step100之前的预热不纳入主窗口；最终checkpoint400在metric400之后保存，因此不计入上述窗口，但仍属于整轮必须验收的真实工作。指标是原日志同步间隔下的训练循环吞吐，包含这段时间实际发生的数据等待、训练和原日志工作；不宣称纯GPU内核耗时。

并发比较只接收另一侧在所选完整窗口内仍执行原训练循环的结果。核对同伴metric0已经出现、metric400尚未出现，并结合日志/NVML确认；不能把初始化、屏障等待、纯等待或最终保存阶段算作并发训练覆盖。若两个原进程的速度或编译时间差使某个窗口不满足覆盖，该窗口拒绝用于并发比值。不能为了补覆盖向原训练循环增加保持负载逻辑；不足就保留本次证据并如实报告。

八卡前后基线分别保留原值及漂移 `t8_after/t8_before`。若合并作参考，使用两个等长窗口的均值，同时展示各自结果，不将同一日志复用为两次基线；漂移明显或窗口不稳定时不能只报一个看似精确的慢倍数。对每个variant分别计算：

```text
单任务慢倍数：t4_left / t8_reference、t4_right / t8_reference
双四卡合计吞吐：128 / t4_left + 128 / t4_right
合计吞吐相对八卡：t8_reference * (1 / t4_left + 1 / t4_right)
两份等步数任务时长比：max(t4_left, t4_right) / (2 * t8_reference)
```

两个四卡进程各自global128，合计每次更新处理256样本；它们使用相同seed和数据顺序，合计吞吐是训练样本处理次数，不是唯一数据条目数。motion和plain分别对应自己的八卡基线，不交叉套用。4/8卡可能因分片、归约和选核产生浮点差异，本轮不将速度对照解释成训练轨迹逐位复现。

完整接收一轮结果须同时满足：

1. 起跑源码快照干净、模块来自快照，完整配置与上述允许覆盖一致，数据/资产身份相同；前置为 `STOCK_PREFLIGHT=PASS`。
2. 实际device/FSDP为目标4或8、batch128、workers16或8，未启用内部覆盖、额外读取器或未批准环境开关。
3. 原metrics五条完整、步号和时间有效，loss、grad_norm、llm_grad_norm、mem_enc_norm、param_norm均有限；401步结束且原日志出现等待保存完成后的正常退出。
4. `OUT/train.log` 有唯一 `EXIT_CODE=0`；没有OOM、异常或超时。OOM、被杀、超时及不完整结果均不得计算慢倍数。
5. `BASE/checkpoints/<具名配置>/<RUN>/400` 是本run新产生的原生checkpoint，提交元数据完整，保存的归一化同源，参数实际可读且形状/dtype/叶集合和有限性正确。不能只检查目录名；现有checkpoint不含AdamW状态，读取权重不能反推出训练step或证明可无损续训。
6. 用于双四卡比值的两轮有真实完整并发覆盖；八卡前后基线分别来自独立进程和独立输出。源码与外围脚本运行期间未变，收尾核对本轮进程及采样器状态。

本轮没有额外启用 `TRAIN_FINAL_RECORD_DIR`，不能假称拥有每步参数或保存现场EMA逐叶摘要。真实保存验收以实际产物及独立只读检查为依据，其范围须与现场存在的证据一致。

## 阻断与结果边界

当前原生产guard只允许上述512两态；context2048、4096、8192仍会被拒绝，8192也没有可直接使用的现成生产配置。本轮不改守卫、不借用modulation reader、不用独立装配来冒充原入口可运行。因此不安排原context大预算的虚假测速，阻断应原样记录。

任何四卡OOM均保留原输出、退出码和资源记录，不切换mask、attention、精度、重计算或batch来完成这轮对比。原401步正常退出只证明本轮训练循环和保存完成，不能推导完整60k/80k训练稳定、策略收敛、环境成功率或控制性能。稳态窗口外的初始化、编译、最终保存、长训练磁盘增长和任务结束后的资源重新分配，需另列成本，不合并进短测慢倍数。

本文件只负责启动授权、参数及验收口径；实际起跑时间、通过/失败、基线漂移、并发覆盖和最终比值待本轮原始证据产生后由主代理写入结果留档。尚未产生的结果不预填为PASS。

## 追加：全部测速结束后的只读内容复核

在8份GPU诊断全部结束后，另用独立会话 `ctx-stock-data-fullhash-0928` 调用原有完整哈希接口，读取帧库32个part、位置/状态表及motion整表，共313426537152字节。这样可以更新当前特征内容未变的证据；不在性能窗口内预热整个page cache，不重跑VAE或encoder，不重扫原始RGB/pkl，也不更改任何读取函数。

外层工具 `BASE/check_current_data.py`（SHA256 `704020cf8046ff574ebe829075c76253afea88d95a9ddec0874795533e73fedd`）先要求8份 `train.log` 各有唯一退出回执，且 `nvidia-smi --query-compute-apps=pid` 为空；随后核对6db干净源码和固定帧/motion元数据SHA，调用 `framesamp_store.run_full_checks(StoreMeta.load(...))` 与 `motion_store.run_full_checks(MotionMeta.load(...))`。这两个API只读取完整文件；不能改用会创建pack.lock、写摘要和回填meta的两个打包器 `cmd_verify`。

命令从 `SOURCE` 运行，解释器仍为 `MAIN/.venv/bin/python`，明确 `PYTHONPATH=SOURCE/src:SOURCE/packages/openpi-client/src`、`PYTHONDONTWRITEBYTECODE=1`、`JAX_PLATFORMS=cpu`、空 `CUDA_VISIBLE_DEVICES`，uv与XDG缓存落主仓 `v1-store/cache/`。进入独立tmux，以 `set -o pipefail` 和tee保留 `BASE/data-content-recheck.log`，结束记 `EXIT_CODE=`；结构化结果为 `BASE/data-content-recheck.json`。新输出路径不得覆盖旧文件。

判据为全部固定SHA一致、`DATA_FULL_CHECK=PASS metadata_unchanged=1`、唯一退出0，且前后元数据字节不变、两库均无pack.lock。此处仅补充已授权验证的启动口径，结果待实际执行后写入；不为此改训练配置或数据内容。

## 追加：初始权重与冻结参数的只读参考核验

同样只在8份GPU诊断全部结束后，使用独立会话 `ctx-stock-initial-ref-0928`运行 `BASE/check_initial_reference.py`，SHA256为 `6c3eaba11a0da439d64be9aa7fc8110a5a2c5a3f844d4cafe8a9eed27a4dce6f`。该工具要求GPU空闲、8份恢复报告均通过且分别绑定自己的checkpoint400及history SHA，实际模块导入须来自干净6db快照。

它先按资产锁full级别校验pi05_base和tokenizer，再用原 `restore_params(..., restore_type=np.ndarray, dtype=None)`只恢复一次base。共享叶参考按原加载器 `_merge_params` 的NumPy `astype(schema.dtype)`生成；schema由原模型和freeze_filter独立给出。比较口径为路径、shape、dtype及完整原始字节SHA，新增记忆参数的随机初值明确排除。统计共享非冻结EMA的变化／未变化清单，不预设每叶必须变化；冻结叶若不同则保留失败，不直接断言其参与了梯度更新。

从 `SOURCE`运行，沿用上节CPU隔离与缓存环境；日志为 `BASE/initial-reference-check.log`，结构化结果为 `BASE/initial-reference-check.json`，以pipefail、tee、唯一 `EXIT_CODE=`留存。预计读取约12.44GB初始资产进行全文件哈希，再读取一次约同量的初始参数，故“恢复一次”不等于“只读一遍磁盘”。这只是独立只读验收，不修改训练、加载或数据读取函数；结果待执行后回写。

## 追加：实际会话与执行完成记录

上文六个原训练会话均已实际启动并自然结束；两个双四卡会话各含两份独立原入口运行，共八份401步诊断。两项CPU收尾检查也已在 `ctx-stock-data-fullhash-0928`、`ctx-stock-initial-ref-0928`实际启动并正常退出，分别耗时749.919秒、65.893秒，两份日志均只有一个 `EXIT_CODE=0`。没有清理其他tmux会话。

收尾命令为 `bash BASE/run_final_readonly.sh data`、`bash BASE/run_final_readonly.sh initial`，分别进入上述独立会话；包装器SHA256为 `000d67c4a0711d10d14bdc61576a75a0f2f9784ad417efce8b47f1a19b830f1d`。八轮完整配置JSON及CPU结果已归档 `records/`，最终数值、覆盖范围与尚未证明事项见[result.md](result.md)。这些是诊断完成记录，未启动任何正式60k/80k训练。
