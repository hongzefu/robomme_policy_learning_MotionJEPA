# collate 共享内存：起跑记录

本轮执行 [0917 计划](../../../0917-collate-shm-8gpu-plan.md)：只改变 worker 与主进程之间的 batch 交付方式。
先做真实输入逐位对拍，再做 8 卡 300 步速度对比，最后做 100 步三侧确定性验证。
不得在两个一致性检查全部通过之前宣称训练等价；不得将 CPU 等待变化直接当作训练提速。

## 版本与隔离

主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA` 保持原样，分支 `v2-motionmem`。
改前副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA-base` detached 在
`A_HEAD=7a681e2cf3f21bdab3f0b27ea44d19b0adfd0ee7`，训练范围与 `e4dc733` 零差异。
`src/`、`scripts/`、`packages/` 只读；四份关键源码的 SHA-256 记在
`v1-store/bench/collate-shm-feasibility/lock_sha256.txt`，解锁或删除前须全部核对为 OK。

改后副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp` 的分支为
`v2-motionmem-collate-shm`。C0 为 `82d73f40aa8a1d4a8099955167882be52f8fcb5f`
（基准工具），C1 为 `3868cc943124d5a47e1be832c130f620ec056b3c`。
C2 是本起跑文件首次提交的文档提交；
两个训练驱动均以完整的 A_HEAD、C2 SHA 为命令行参数，并将其写入日志。
最终结果文件补录完整 C1/C2 SHA，禁止改写这些提交。

两副本各有独立 `.venv`，由 `uv sync --frozen --python <主副本>/.venv/bin/python` 创建，
各自的 `openpi.__file__` 与 `sys.prefix` 都在自己的目录。
两处 `v1-store` 均指向主副本实体存储；用户已批准在主仓库本地
`.git/info/exclude` 加 `/v1-store`，解决原 `/v1-store/` 规则不匹配符号链接的问题。
该例外仅覆盖存储入口，未忽略源码或其他未跟踪文件。每侧起跑前必须 `PREFLIGHT=PASS n=25`。

## 已确认的用户决定

本轮用户原话：「开始执行 有问题问用户越早越好 结束后报告 问用户是否合并」。
沿用计划中的分支、目录、run 名、8 卡独占授权，以及只锁改前副本的决定。
执行时用户另明确答复：

- 「同意，按实际状态验证」：当前主副本干净，应 PASS；改后存在未提交补丁时须因 REPO_CLEAN 被拒绝。
- 「同意，C0 检查入口，后续实跑训练」：C0 用真实配置解析及入口守卫，截在模型训练前；
  原 CPU 命令的 batch=2 与默认 fsdp=4 冲突，不再用完整 CPU 训练作为工具自检。
- 「同意，添加本地精确忽略规则」：仅添加上述 `/v1-store` 本地规则。

## 硬件、数据与参数

环境 B：8 × NVIDIA A100-SXM4-80GB；存储为 AWS 本地 NVMe RAID `/dev/md0`，XFS 挂载 `/scratch`。
开工时 8 卡显存均为 0 MiB，磁盘余量约 1.1 TiB，`/dev/shm` 约 561 GiB。
本会话 `ulimit -n` 实测 8192，区别于计划可行性记录中的 1048576；
默认共享策略保持 `file_descriptor`，若出现句柄不足则停止并按计划处理。
计划提及的 `hf-modul60k-export` 会话在本轮开始时已不存在，不能将其写成同期干扰。

数据实体根为主副本 `v1-store/datasets/4task-v2-1600ep-604f16da/`：
训练读 `framesamp-8x8/`，源为 `source/`，清单为 `meta/episode_manifest.json`。
各侧通过自身 `v1-store` 链接访问同一份字节，不复制数据。
norm_stats 为 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`，
SHA-256 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`。
history 配置 `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml`，
SHA-256 `5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec`，motion 关闭。

使用 `mme_vla_suite_b128_60k` 配置，全局默认不变。速度侧覆盖 `--num-train-steps 300
--log-interval 10 --no-wandb-enabled --num-workers 16 --fsdp-devices 8`，global batch 128、seed 42
沿用配置。确定性侧额外明确覆盖 `--batch-size 128 --num-train-steps 100 --log-interval 1
--save-interval 1 --seed 42`。两类运行的 mesh 都为 (1,8)。资产、数据、checkpoint 路径
均由驱动显式传入本副本 `v1-store` 下已核实的路径，不使用 overwrite/resume/force。

## 数据流与一致性判据

改前：

```text
同一 framesamp-8x8 库 → FrameSampDataset.__getitem__ → transform_dataset
→ _collate_fn(np.stack) → numpy batch（513,319,168 B）
→ worker pickle 全部字节 → 队列 → 主进程反序列化
→ jax.make_array_from_process_local_data → 8 卡模型
```

改后：

```text
同一 framesamp-8x8 库 → 同一 __getitem__ → 同一 transform_dataset
→ 同一 _collate_fn(np.stack) → 同形状、同 dtype、同字节 numpy batch
→ _to_shared_torch（视图）→ torch storage 共享内存、队列传句柄
→ _from_shared_torch（视图还原 numpy）
→ 同一 jax.make_array_from_process_local_data → 8 卡模型
```

关键 batch 叶子：`static_image_emb` 为 `(128,512,2048)` bf16；`static_pos_emb` 为
`(128,512,768)` f32；两张 RGB 图各 `(128,224,224,3)` u8；`static_state_emb` 为
`(128,512,8)` f64；`actions` 为 `(128,20,32)` f64。其余为提示 token、mask、state，
共 12 个数组键、4 个 None。所有改动跳点均不改变数值或位模式；bf16 使用 uint16
视图桥接 numpy 与 torch 的 dtype 表示，亦不执行数值转换。进入 JAX 后的既有转换不变。

第一块：各侧自己的环境运行 `scripts/training/tests/compare_collate_paths.py dump`，
batch 128、worker 16、seed 42、20 批；逐键 SHA-256 覆盖 dtype、shape 和原始字节，
另记录 sampler 原样产出的前 2560 个索引。必须
`COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0` 且索引相同。
pytest 覆盖 0/2 worker、真实 spawn 共享存储、None、所有 bf16 位模式与内存视图。

起跑前第一块已通过：`COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0`、
`INDEX_SEQ=PASS n=2560`；pytest `4 passed in 10.65s`。每批实测 513319168 B，
平均取批等待 old=0.783745s、new=0.002647s；逐批摘要计算会与预取重叠，
该等待值不能用于外推训练吞吐。旧侧首次导出因 V1_STORE 未 export 失败，修正工具默认
为当前副本的 v1-store 后重跑成功；保留失败日志，仅删除本轮创建的空目录。

第二块：a1/a2 来自改前副本，b 来自改后副本，三侧串行各 100 步。
`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`；
`BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000
BENCH_EXTRA_DIGEST_STEPS=1,2,24,49`。每侧 `BENCH_SOURCE_ROOT` 指向本侧源码，
工具来自改后副本；run_meta 分别记录 tool_head/source_head/source_status。
各侧先 dump 环境指纹；a1 完成后生成 BASELINE_MANIFEST，a2/b 对 a1 做 check。
必须两组 `BASELINE_ENV=PASS`；比较必须 100 步五标量、完整 12800 个训练索引、
六个输入与完整状态摘要步 `[0,1,2,24,49,99]` 全逐位相同。
a1/a2 不逐位则立即停止，不启动 b，不使用量化退路。

## 启动命令与输出边界

以下脚本均为本轮创建于主副本 `v1-store/logs/` 的实际驱动，起跑前通过 `bash -n`，
结果归档时保存副本并核对摘要。源码由 A_HEAD/C2 固定；脚本由下表 SHA-256 固定。

| 文件 | SHA-256 |
|---|---|
| `cs-8gpu-runner.sh` | `1e0ebf9f444922b5872e97633f20c73bd23972be9963bb5728478f933926cb6c` |
| `cs-8gpu-driver.sh` | `7ad35c1d3402f37017fd56fba48966ffcb03623ba1649f19269092c7ca76553d` |
| `cs-guard-runner.sh` | `7de47e69a226b368f471ecb0b8fbdac9e9f4105de2cbc89c61a371e99652ffc1` |
| `cs-analyze.py` | `0246a55274b70821d63400525872d2d82a33a676a3a67c2e38d854811d924819` |

```bash
tmux new-session -d -s cs-8gpu \
  'bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/cs-8gpu-driver.sh <A_HEAD> <C2>'
# 仅速度侧全部验收后执行：
tmux new-session -d -s cs-guard \
  'bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/cs-guard-runner.sh <A_HEAD> <C2>'
```

会话清单仅 `cs-8gpu`、`cs-guard`，实际创建时在结果中登记。独立流式监听日志中的
`PREFLIGHT=`、`EXIT_CODE=`、`Traceback`、步进与完成信号，所有过滤级行缓冲。
驱动使用 `set -o pipefail`、`PYTHONUNBUFFERED=1`、`tee`，记录真实退出码。
既有会话 `0`、`1`、`claude-private`、`codex`、`codex-repo`、`codex2` 不属于本轮。

训练 run 仅五个：`bench-collate-shm-old`、`bench-collate-shm-new`、`cs-guard-a1`、
`cs-guard-a2`、`cs-guard-b`，位于 `v1-store/train-runs/mme_vla_suite_b128_60k/`。
指标和采样位于 `v1-store/bench/bench-collate-shm-{old,new}` 与
`v1-store/bench/8x8/cs-guard-{a1,a2,b,summary}`。各侧使用独立 JAX 编译缓存。
起跑时以上 run 名均不存在；不清空旧目录。验收且归档逐字节核对之后，
只清理本轮训练产物和已归档的 bench 目录；两个工作副本保留，等用户决定合并、推送与删除。

## 速度判据与结果状态

以真实训练 `metrics.jsonl` 的 step 100→290 为稳态，均值按整个 190 步墙钟计算，
吞吐为 128/平均步时。GPU 用 500 ms 流式采样，报告均值、0% 占比、慢区间及其他区间均值。
慢区间沿用历史定义：10 步区间平均步时大于全部区间中位数的 1.5 倍；
中位数仅用于分层阈值，不作为性能结论。该分层分辨率为 10 步，不能声称观察到每个慢步。
旧侧须复现历史 f8-w16 的 1.818 秒 ±5%；否则停止并交用户，不放宽容差。
最终结论须同时给出速度、吞吐、利用率及完整一致性证据。

起跑记录提交时训练尚未开始。结果与异常处理随后写入 result.md；本地四个提交
完成后询问用户是否推送新分支、并回 `v2-motionmem` 及删除两个副本。
