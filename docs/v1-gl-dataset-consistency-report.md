# v1 四任务数据集：GreatLakes 构建与一致性验证报告

> 状态：**进行中**。一～四节（事实、环境、设计、方案）已由实测与实现落定；
> 五～七节（档位实测、验证结果、复现命令）随流水线推进回填，每跑完一段就补，
> 不等全部结束才动笔。档位实测另有独立文档
> [`v1-gl-resource-tier-bench.md`](v1-gl-resource-tier-bench.md)。

本报告是这条链路的唯一权威留档。实现见 [`scripts/data-preprocess-GL/`](../scripts/data-preprocess-GL/README.md)。

---

## 一、数据与链路事实（实测）

### 1.1 原始输入

本机 `/data/hongzefu/robomme_data_h5_v2_4env400ep`，4 个 H5 各 400 episodes。
下表的 episode / timestep / 执行样本三列是 `scan_manifest.py build` **全量扫描的精确值**
（`v1-store/episode_manifest.json`，sha256 `7258978d…`）：

| 文件 | 体积 | episodes | timesteps | 执行样本 | demo 步 | 步数 min/mean/max |
|---|---:|---:|---:|---:|---:|---:|
| `record_dataset_ButtonUnmask.h5` | 69.8 GB | 400 | 105,064 | 105,064 | **0** | 198 / 262.7 / 452 |
| `record_dataset_ButtonUnmaskSwap.h5` | 103.8 GB | 400 | 156,280 | 156,280 | **0** | 295 / 390.7 / 555 |
| `record_dataset_VideoUnmask.h5` | 56.8 GB | 400 | 85,420 | 59,020 | 26,400 | 163 / 213.6 / 399 |
| `record_dataset_VideoUnmaskSwap.h5` | 90.7 GB | 400 | 136,527 | 74,925 | 61,602 | 210 / 341.3 / 586 |
| **合计** | **321 GB** | **1600** | **483,291** | **395,289** | **88,002** | 163 / 302.1 / **586** |

两个结构性事实值得单独记：

1. **两个 Button 任务的 `exec_start_idx` 恒为 0**（demo 步 = 0），即没有 video demo 前缀；
   只有两个 Video 任务有，合计 88,002 步是 demo。这解释了为什么 `data/*.pkl` 的总数
   （395,289）明显小于 timestep 总数（483,291）——`_process_episode` 只对
   `is_video_demo == False` 的步写 pkl。
2. **最长 episode 是 586 步**（`VideoUnmaskSwap`），比先前按 `episode_0` 采样估计的 514 更长。
   它决定了 `MemoryBuffer._history_feats` 的稳态内存上界（单 episode 全量累积），
   也决定了分片中断的损失上界。

先前按「每 25 个 episode 采样一个」估算的 476,425 步与精确值 483,291 相差 1.4%，
采样口径可用但正式数字一律以清单为准。

**LPT 装箱实测**：8 个分片的 timestep 负载 `[60412, 60412, 60412, 60411, 60411, 60411, 60411, 60411]`，
**极差 0.00%** —— 按条数均分本会因 episode 长度 163–586 的巨大差异而明显拖尾，LPT 消除了它。

每个 timestep 的观测含 `front_rgb` / `wrist_rgb`（均 `(256,256,3) uint8`）、
`front_depth` / `wrist_depth`、`joint_state` / `gripper_state` / `eef_state` 等。

### 1.2 计算本体

处理入口是仓库自带的 `scripts/build_dataset.py --dataset_type robomme_pkl` →
`DatasetProcessor`。真正的计算在 `MemoryBuffer.add_buffer` 里：
**JAX + SigLIP So400m/14（`dtype_mm="bfloat16"`）的 GPU 前向**，权重来自
`$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl`（1.66 GB）。

值得记一笔的形制事实：`_process_episode` 是**逐 timestep 调用 `add_buffer`、每次只喂 1 张图**的，
所以 SigLIP 单次前向极短。这直接决定了后面档位实测的先验判断——流水线大概率偏
CPU/IO-bound（h5 解压、numpy 像素差、每步 `np.save`），**降 CPU 的风险大于降 mem**。

### 1.3 产物形制

```
meta/stats.json                                   # {execution_samples, total_samples}
data/<exec_sample_id>.pkl                         # 每个「执行步」一份，含原图与 action chunk
features/episode_<global_episode_idx>/token_emb_<step>.npy
features/episode_<global_episode_idx>/kept_indices.json
```

`token_emb_<step>.npy` 是一个 dict：`image_emb_{8x8,4x4,2x2}`（SigLIP，bf16）、
`pos_emb_{8x8,4x4,2x2}`（`PosEmb3D`，fp32）、`state_emb`（fp32）。
`token_emb_<step>.npy` 的实测形制（A40 探针打出）：

```
image_emb_8x8 (1,64,2048) bfloat16   pos_emb_8x8 (1,64,768) float32
image_emb_4x4 (1,16,2048) bfloat16   pos_emb_4x4 (1,16,768) float32
image_emb_2x2 (1, 4,2048) bfloat16   pos_emb_2x2 (1, 4,768) float32
state_emb (8,) float32               → 合计 602,144 B/step
```

**体积实测把先前的估算推翻了**：按整库 `du -sb ÷ 步数` 实测 **910 KiB/step（932,154 B）**，
全量外推 **≈451 GB**，比按 dtype 推算的 588 KiB 高 55%。差额来自 `data/*.pkl`——
里面存了未缩放的 `image` 与 `wrist_image`（各 256×256×3 uint8 = 196 KB）加 action chunk，
先前只算了 `token_emb` 一项。`step_submit.sh` 的 I/O 侧 walltime 反算已改用实测值。

---

## 二、环境与配额（实测）

### 2.1 两侧硬件

| | 本机（sled-vail） | GreatLakes spgpu |
|---|---|---|
| GPU | 2× RTX 6000 Ada 46 GB，**sm_89** | A40 46 GB，**sm_86** |
| CPU / 内存 | 32 核 / 377 GB | 每节点 32 核 / ~381 GB |
| cgroup | v2，`systemd-run` 可用 | v2（Slurm `jobacct_gather/cgroup`） |
| 磁盘余量 | `/data` 3.1 TB | turbo 6.9 TB |

**两侧架构不同（sm_89 vs sm_86）是整个一致性方案的前提**，详见第四节。

### 2.2 存储与网络

- 本机 → turbo 的 rsync **全程实测 99.5 MB/s**
  （321,212,805,828 字节，`EXIT_CODE=0`；前 51 GB 的瞬时值约 77 MB/s，后段更快）。
- turbo 卷顺序读实测 **114 MB/s**（sha256 读 69.8 GB 用时 610 s，期间另有清单扫描在抢带宽）。
- turbo 卷在 8 并发读写下的天花板约 132 MB/s（`greatlakes.md` 既往实测）。
  抽取期间勿并行起集群训练。
- ⚠ **大批 NFS 写入在飞时，本机的写操作会被内核脏页背压一起节流**。实测撞到过一次：
  rsync 还在跑时起本机 GPU 任务，进程卡在 `D / balance_dirty_pages`、本地 nvme 零 I/O、
  3 分钟只写出 14 个文件（当时全局 Dirty 达 11.7 GB）。脏页限额是全局的、不区分目标盘，
  所以**本机扫档与基线构建必须等 NFS 传输全部结束后再起**，否则读数全是噪声。

### 2.2.1 仓库放在 NFS 上的两个操作性坑（实测）

1. **`core.filemode` 必须关。** turbo 的默认 ACL 会给每个文件强制加属主执行位
   （`664` → `774`），git 于是把**每一个文件**都报成 `mode change 100644 => 100755`，
   `git status` 里 400 多个文件全变 modified、内容却零差异（`git diff --stat` 全是
   `0 insertions(+), 0 deletions(-)`）。`git config core.filemode false` 后恢复正常，
   `step0_setup_turbo.sh check` 会自动设上。代价是 git 不再感知可执行位，新增需要 +x
   的脚本要显式 `git update-index --chmod=+x`。
2. **不要在 Bash 脚本执行期间编辑它本身。** bash 增量读取脚本：长命令返回后从保存的
   字节偏移继续往下读，文件被编辑过就会读到半截语句。实测 `stage_models.sh` 的全部校验
   都已通过、日志写到「项目内模型准备完成」，却仍以 `EXIT_CODE=2` 收尾并报
   `syntax error near unexpected token '('`。判断办法是看业务步骤是否走完，补救办法是原样重跑。

### 2.2.2 NFS venv 实测可用（本流程最大的未知之一，已排除本机侧）

本仓库是 **JAX 栈**（`jax[cuda12]==0.5.3`），而集群既往只验证过 torch 栈，
「NFS 上的 venv 能不能起 JAX + CUDA」是全流程最大的未知之一。

按 `greatlakes.md`「venv 可移植性」硬规则重建（解释器显式指定 NFS 上的
`cpython-3.11.14`，`UV_LINK_MODE=copy uv sync`），实测：

```
Prepared 208 packages in 6m28s / Installed 208 packages in 4m14s
.venv/bin/python -> /nfs/turbo/.../uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
pyvenv.cfg home    = /nfs/turbo/.../uv-python/cpython-3.11.14-linux-x86_64-gnu/bin
jax=0.5.3 jaxlib=0.5.3 devices=[CudaDevice(id=0), CudaDevice(id=1)]
```

解释器与 `pyvenv.cfg` 都落在 NFS 上（在计算节点上不会是死链），**本机侧已确认能起 JAX + CUDA**。
A40（sm_86）侧由档位探针的 `JAXCHK` 一步确认，尚未验证。

⚠ uv 的下载 cache **刻意留在本机盘**（不设 `UV_CACHE_DIR`）：`greatlakes.md` 明确要求
「cache 在本机盘、venv 在 NFS」，且 cache 只在跑 `uv sync` 的这台机器上用得到——
计算节点直调 `.venv/bin/python`，根本不碰 uv。

### 2.2.3 原始 H5 的 sha256（本机原件，永久保留的那一份）

| 文件 | 字节数 | sha256 |
|---|---:|---|
| `record_dataset_ButtonUnmask.h5` | 69,814,596,640 | `a9dac92e9cbae561…` |
| `record_dataset_ButtonUnmaskSwap.h5` | 103,847,071,632 | `517072656451f223…` |
| `record_dataset_VideoUnmask.h5` | 56,758,988,232 | `2c854cfd8d229b8b…` |
| `record_dataset_VideoUnmaskSwap.h5` | 90,713,562,996 | `9a7a1138bc72a0de…` |

**同源判定：已通过。** 两侧各算一遍再逐项 diff：

```
本机原件 /data/hongzefu/robomme_data_h5_v2_4env400ep        313 s（NVMe）
turbo 副本 /nfs/turbo/.../robomme_data_h5_v2_4env400ep      2864 s（≈112 MB/s）
diff <(jq -S .files 本机) <(jq -S .files turbo)  →  无差异
✓ 四个 H5 的 size 与 sha256 全部逐项相同
```

**「输入」这个变量到此被彻底钉死**——后面测到的任何差异都不可能来自「两边读的字节不一样」。
集群侧 finalize 起跑前会再核一次（`--input_level size`，防 rsync 缺漏/截断）。

### 2.2.4 走 turbo 的处理速率比走本机盘慢约 3.4 倍（影响 walltime 反算）

同一份 `build_shard.py`，同样的 Ada 卡，只换输入/输出位置：

| 数据位置 | 稳态速率 |
|---|---:|
| 本机 `/data`（NVMe 读 + NVMe 写） | ≈**67 step/s**（线性拟合边际值） |
| turbo（NFS 读 + NFS 写） | ≈**20 step/s**（单 episode 实测 228 步 / 11.4 s） |

**这条流水线确实是 I/O 受限而非 GPU 受限**，与第五节档位实测的先验判断一致：
GPU 大部分时间在等 I/O，因此 CPU/mem 有下压空间，而 walltime 必须按 I/O 侧估算。
集群侧是计算节点经另一条网络路径访问同一个 turbo 卷，实际速率由档位探针实测给出。



### 2.3 chaijy2 配额（全组共享，实测于 2026-08-23）

```
配额上限:  GPU 20  |  MEM 960 G  |  CPU 80
当时占用:  GPU  1  |  MEM  38 G  |  CPU  9
```

**CPU 配额只有 80，这是压低档位的首要理由**：8 个 job 各要 4 CPU 就吃掉配额的 40%，
压到 2 CPU 只吃 20%。配额是组内共享的，留出余量才不会互相卡住
（`AssocGrpMemLimit` / `AssocGrpGRES` 是 PENDING 的常见原因）。

### 2.4 spgpu 节点碎片（实测）

240 张 A40（30 节点），当时空闲 76 张、23 个节点有空闲卡。但其中：

| 节点 | 空闲 GPU | 空闲 CPU | 空闲 MEM |
|---|---:|---:|---:|
| gl1501 | 3 | **0** | 234 G |
| gl1500 | 2 | **0** | 36 G |
| gl1508 | 2 | **0** | 86 G |
| gl1514 | 1 | **0** | 4 G |

**有空闲 A40 却零空闲 CPU** —— CPU 比 GPU 更常成为「有卡却起不来」的真正原因。
这是第二个压低 CPU 档位的理由。

---

## 三、分片设计与 ID 偏移推导

### 3.1 问题

`DatasetProcessor.run()` 严格串行，三个计数器从 0 一路**跨文件累加**：

- `global_episode_idx` → `features/episode_{g}/` 目录名
- `exec_sample_id` → `data/{id}.pkl` 文件名（只在 `is_video_demo == False` 的步上自增）
- `total_sample_id` → 进 `meta/stats.json`

且文件遍历用的是**非确定序的 `os.listdir`**。直接并行分片必然错号覆盖。

### 3.2 解法

`scan_manifest.py build` 只读 metadata 扫一遍，按**规范序** `sorted(*.h5) × sorted(episode_i)`
（刻意不用 `os.listdir`），逐 episode 记 `num_timesteps` 与 `exec_start_idx`
（后者复用 builder 自己的 `first_execution_step()`，保证口径逐字一致），推出：

```
global_episode_idx   = 规范序下的序号
exec_samples         = num_timesteps − exec_start_idx
exec_sample_offset   = Σ 之前所有 episode 的 exec_samples      （前缀和）
total_sample_offset  = Σ 之前所有 episode 的 num_timesteps     （前缀和）
```

清单带自身 sha256，是全流程**唯一真值源**——分片 worker、finalize 守卫、比对工具
都从它取 episode 身份 `(h5_file, raw_ep_idx)` 与偏移量，不再依赖任何目录名或遍历顺序。

### 3.2.1 清单扫描在 NFS 上比本机慢约 20 倍（实测 + 可优化点）

同一份 1600 episode 的元数据扫描：

| 输入位置 | 耗时 |
|---|---:|
| 本机 `/data`（NVMe） | **13 s** |
| turbo（NFS） | 头两个 H5 就用了 **664 s**，全量数十分钟 |

原因是 `first_execution_step()` **逐 timestep 读一个标量** `is_video_demo` 直到读到
`False`；两个 Video 任务的 `exec_start_idx` 在 66–168 之间，于是每个 episode 要发几十到
上百次极小的读，NFS 上每次都是一个 round-trip。（两个 Button 任务 `exec_start_idx` 恒为 0，
一次就返回，所以它们扫得快。）

**这是一次性成本，不在关键路径上**，本轮照原样跑完以保证与 builder 口径逐字一致。
若将来要提速，安全的做法是二分查找而非线性扫描：`_process_episode` 内部本来就有
`assert ts["info"]["is_video_demo"][()] == (step_idx < exec_start_idx)`
**逐步校验**这个单调性假设，一旦二分给出的值不对，构建阶段每一步都会立刻炸出来。
读次数可从 O(exec_start) 降到 O(log T)。

另一个等价选项是让清单在**本机原件**上生成（元数据完全相同，只有 `raw_dir` 字段不同），
但那样 provenance 就不再指向集群实际读的那份输入，本轮没有采用。

### 3.3 负载均衡

**LPT 装箱**（按 `num_timesteps` 降序投给当前最轻的桶）而不是按 episode 条数均分：
四个任务的 episode 长度差异很大（实测 163–586 步），按条数分会让最重的分片明显拖尾、
直接拉高 walltime 需求。实测分片极差见第五节。

### 3.4 为什么子类化而不是复制逻辑

`build_shard.py` 的 `ShardProcessor` **只覆盖两个方法**：

- `__init__`：跳过原实现的 `shutil.rmtree(dataset_path)` —— 8 分片并发下那等于互删产物；
- `run()`：改为按清单遍历本分片 episode，用清单偏移量喂 `_process_episode` 的三个计数器。

`_process_episode` 本体**一行不动**。这样「分片实现与串行 builder 语义同构」是由
**构造方式**保证的，而不是靠事后对拍碰运气；对拍只是兜底证据。
原实现里现成的 `assert not os.path.exists(pkl_path)` 顺带就是跨分片撞号的硬断言。

---

## 四、一致性验证方案

### 4.1 先把两个变量分开

集群产物与本地产物的差异有两个来源：**① 我把串行 builder 改成了 8 分片；
② A40(sm_86) 与本机 Ada(sm_89) 是不同架构。** 混在一起测，出了差异说不清是 bug 还是硬件噪声。
所以分层。

### 4.2 第一层：清零「分片」变量（本机，零容差）

同一台机器、同一批 episode 跑两遍：

- 参照系 = 仓库**未改动的** `scripts/build_dataset.py --max_episodes 3`
- 被测 = `build_shard.py --num_shards 4`

硬件变量为零 ⇒ **逐字节相同**，无任何容差。

取 `--max_episodes 3`（12 个 episode）而不是 1（4 个）的理由：分片改造最容易错的是
**跨文件累加偏移**，而每任务 `episode_0` 的偏移基本是平凡的——只拿它们对拍恰好避开了
最该测的部分。12 个里有 11 个带非零偏移，4 个分片也能压到分片边界。

**这一层过了，`build_shard.py` 就取得「本地真值」资格**——第二/三层的跨架构对拍
不再受未改动 builder「只能取前缀」的限制，可以在全 1600 里分层随机抽样。

### 4.2.1 episode 一律按物理身份匹配，绝不按目录名（实测例证）

两个库的 `features/episode_{g}/` 编号体系不同：对照库只到 11 或 39，集群库是 0–1599。
更要命的是，**未改动 builder 用 `os.listdir` 遍历 H5，顺序不是字典序**。实测：

```
listdir 实际顺序: ['ButtonUnmaskSwap', 'ButtonUnmask', 'VideoUnmaskSwap', 'VideoUnmask']
未改动库 episode_0  ->  ButtonUnmaskSwap#0 （清单 g=400，514 步，exec_offset=105064）
未改动库 episode_1  ->  ButtonUnmask#0     （清单 g=0  ，291 步，exec_offset=0     ）
未改动库 episode_2  ->  VideoUnmaskSwap#0  （清单 g=1200，326 步，exec_offset=320364）
未改动库 episode_3  ->  VideoUnmask#0      （清单 g=800，239 步，exec_offset=261344）
```

**按目录名对拍会把 `ButtonUnmaskSwap#0` 比到 `ButtonUnmask#0` 上，得出一个假失败。**
所以一律按物理身份 `(h5_file, raw_ep_idx)` 匹配，`data/*.pkl` 也不按文件名、
而按 `(episode 身份, 该 episode 内第 k 个执行步) → 各库自己的 exec_sample_offset + k` 定位。

未改动库那侧的映射靠**两个独立来源交叉验证**，一致才放行、不一致直接 fail loud：
① 比对时对同一个未被改动过的原始目录重新 `os.listdir`，复现它当时的遍历顺序；
② 解析 builder 自己打印的 `Episode {g}: timesteps=…, task_goal='…'`，用 timesteps 序列反查。

### 4.3 第二层：跨架构逐 key 分类对拍

关键观察：**产物里只有一小部分真的过了 GPU**。所以不是一刀切阈值，而是逐 key 分类：

| 产物 | 计算路径 | 判据 |
|---|---|---|
| `kept_indices.json` | `_process_token_drop_score` 用 **image_pixels 做 numpy 像素差**，未碰 GPU | **逐位**，零容差 |
| `data/*.pkl` | `image`/`wrist_image`/`state`/`actions` 从 H5 直读 | **逐位**，零容差 |
| `state_emb` | 即那个 `state` numpy 数组 | **逐位**，零容差 |
| `pos_emb_{8x8,4x4,2x2}` | `PosEmb3D` 全用 `jnp.einsum`/`jnp.sin`/`jnp.cos`，**在 GPU 上算**；但只是秩一外积 + 逐元素超越函数，无归约累加 | **实测判定桶归属**（第六节回填） |
| `image_emb_{8x8,4x4,2x2}` | SigLIP So400m/14，bf16 GPU 前向 | **量化等价**（下条） |

> `pos_emb` 一度被误判为「确定性 CPU 函数、必然逐位」。查 `src/mme_vla_suite/shared/posemb_3d.py`
> 后更正：它走 JAX、跑在默认 device（GPU）上。逐位相同的可能性很高但**不能先验假定**，
> 交由 20 分钟的探针实测定性。

**`image_emb` 为什么不能用绝对阈值**：它是 bf16，尾数只有 8 位，**1 ULP 就是约 0.4%
的相对误差**。给 `max|Δ| ≤ 5e-5` 这种阈值要么恒过要么恒挂，毫无判别力。改报三个量：

1. **bf16 位完全相同的元素占比** ≥ 0.95（按元素宽度统计，不是按字节——按字节会把
   「一个元素差 1 ULP」摊成「两个字节错一个」，占比虚高）；
2. **最大 ULP 差 ≤ 1** —— 只允许差一个最小单位；出现 ≥2 ULP 说明是结构性偏移而非舍入噪声；
3. **逐 token 余弦相似度 ≥ 1 − 1e-3**。

### 4.4 第三层：下游等价（真正决定训练是否等价的一道）

前两层比字节，这层比**训练实际怎么用它**。用 `perceptual-framesamp-context.yaml` 同一配置，
对同一批 `(episode, step)` 分别从两个库走 `MemoryBuffer.prepare_frame_sampling`：

- **选帧索引与 `mask` 必须逐位相同** —— 它们只依赖 `step_idx` 与
  `budget/token_per_image = 512/16 = 32`，不依赖任何 GPU 数值；
- `img_emb` 按 4.3 的等价口径。

这一层保证：**数值噪声没有改变 dataloader 的任何离散决策。**

### 4.5 三个配套前提（缺一则上面全是空的）

1. **输入同源**：turbo 的 H5 与本机原件**逐文件 sha256 相同**（`step0_setup_turbo.sh h5`
   两侧各算一次并 diff），集群侧起跑前由 finalize 再核一遍。否则「不一致」可能只是
   rsync 缺了几个字节。
2. **集群内部零容差**：finalize 随机抽 256 条**在同一节点复算**，断言 `max|diff| == 0`。
   同架构，所以可以零容差；它排除的是线程调度、cudnn 算法选择这类非确定性。
   （可以逐条独立复算，是因为 `token_emb_{step}` 只依赖该步的图像、`step_idx` 与 `state`，
   不依赖历史——历史只影响 `kept_indices`。）
3. **完整性守卫**：每 episode 的 `token_emb_{0..T−1}.npy` 与 `kept_indices.json` 齐全，
   `data/{0..N−1}.pkl` 连续无空洞无多余，`meta/stats.json` 与清单前缀和逐项相等，
   无 claim 残留，分片 sidecar 覆盖集合 == 清单全集。

### 4.6 第四层：训练可用性

本机 2 GPU 用 `perceptual-framesamp-context` 在 GL 全量库上跑通 12 step，
loss 有限、无 NaN、无形状/键缺失。⚠ 这是**功能性 smoke，不是吞吐基准**：代码与数据都在
turbo，且 frame sampling 每样本要读 32 个 `token_emb`（≈19 MB），batch 必然偏慢；
按 AGENTS.md 第 13 条本机数字本来就不作吞吐结论。

### 4.7 这套方案证明不了什么（如实声明）

**跨架构逐位一致是做不到的。** `greatlakes.md` 已实证 A40 与 Ada 在 VAE 前向上不逐位一致
（分层指纹定位到最后一层 `conv_out`），且 determinism 三档全部无效。SigLIP 同理。

因此交付按「**换合同**」口径：集群产物**自成一份数据集**，`meta/provenance.json` 逐条带
hostname / GPU 型号 / jax 版本 / git commit / 清单 sha / 资源档位，finalize 断言全体同源，
**机制上杜绝与本地字节混用**。验收标准是上述等价判据，而不是「和本地一模一样」。

---

## 五、CPU / mem 档位实测

见独立文档 [`v1-gl-resource-tier-bench.md`](v1-gl-resource-tier-bench.md)。
结论（选定档位、A40 实测 step/s、walltime 反算）回填到这里：

> **待回填** —— `step_bench.sh report` 之后。

---

## 六、验证结果

### 6.0 迁移前代码验证（本机，4 个 episode，2026-08-23）

在仓库搬到 turbo 之前，先用本机 H5 + 本机模型缓存把第一层链路整跑一遍，
目的是提前抓代码 bug 而不是出正式判定（正式判定见 6.1，在 turbo 上用 12 个 episode 重跑）。

**结果：`COMPARE_RESULT=bitexact PASS`**，1370 步全覆盖、9 个 key 全部逐位相同：

| key | 比对数 | 逐位相同 | max ULP | min 余弦 |
|---|---:|---|---:|---:|
| `image_emb_8x8` / `4x4` / `2x2` | 各 1370 | ✓ | 0 | 1.0 |
| `pos_emb_8x8` / `4x4` / `2x2` | 各 1370 | ✓ | 0 | 1.0 |
| `state_emb` | 1370 | ✓ | 0 | 1.0 |
| `kept_indices.json` | 4 | ✓ | — | — |
| `data/*.pkl` | 32 | ✓ | — | — |

**顺带得到一个重要事实：同机跨进程的 XLA 是确定性的。** 参照系是**单进程**跑完 4 个 episode，
被测是 **4 个独立进程**各跑 1 个，两者的 `image_emb_*` 逐位相同。这排除了「XLA autotuning
在不同进程里选到不同算法」这一类同机非确定性——否则后面所有跨架构比对都会被它污染。

**这一轮抓到的两个真问题**（都已修）：

1. `Agg` 类的属性名不一致（`all_bitwise_equal` vs `bitwise_equal`），判定阶段直接
   `AttributeError` 崩掉。纯代码 bug，靠跑一遍才暴露。
2. **`epis_idx` 的比对口径错了。** 它是 episode 的**身份标签**（等于该库自己的
   `global_episode_idx`），而两个库的编号体系本就不同——未改动 builder 走 `os.listdir`
   顺序（实测 `episode_0` 其实是 `ButtonUnmaskSwap#0`），分片实现走清单规范序（g=400）。
   要求两库的 `epis_idx` 相等是错的。改为**分别校验每个库标的是不是它自己的目录号**——
   `RoboMMEDataset.__getitem__` 正是拿它去找 `features/episode_{epis_idx}/`，标错了训练就读错 episode。
   改口径后其余 8 个 key 本就全绿，这条是唯一的假失败。

### 6.1 正式验证结果

> **待回填** —— 各层在 turbo 上跑完后填入判定行与实测数字：
>
> - 第一层 逐字节判定（`v1-store/reports/layer1_bitexact.json`）
> - 第二层 逐 key 判定，含 `pos_emb_*` 最终桶归属、`image_emb` 的位相同占比 / 最大 ULP 差 / 最小余弦
>   （`v1-store/reports/layer2_crossarch.json`）
> - 第三层 选帧索引与 mask 判定（`v1-store/reports/layer3_downstream.json`）
> - finalize 的输入 sha256、完整性、同架构零容差抽检判定行
> - 第四层 smoke 的 loss 序列
> - 产物体积实测与 588 KiB/step 估算的偏差

---

## 七、复现命令与续跑口径

> **待回填** —— 逐段实际命令与实测耗时。骨架见
> [`scripts/data-preprocess-GL/README.md`](../scripts/data-preprocess-GL/README.md)
> 的「逐段流程」「续跑与故障处理」两节。
