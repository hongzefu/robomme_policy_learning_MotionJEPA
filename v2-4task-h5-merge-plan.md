# 新版四任务 H5（robomme-4task-h5-20260912-v2）合并预处理与建库计划

> 2026-09-14 起草。环境判定：**环境 B（AWS 单机 8×A100-SXM4-80GB）**，仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，无 `/nfs/turbo`、无 `/data/hongzefu`、无 `~/.ssh/config`。
> 本计划的核心取舍：**建库链路一行不改**，只在链路前面加一次「把新版单 episode H5 合并成旧形状」的预处理；motion 相关阶段本轮**先不做**，但清单身份按可追加 motion 的方式一次定死。

---

## 第一部分（给人看）

### 1. 结论

新版数据 `HongzeFu/robomme-4task-h5-20260912-v2`（revision `604f16da36d6b6d175884df8fb687dc08e0a36eb`）的本地副本已经完整落在 `/scratch/hongze/robomme-4task-h5-20260912-v2/`：`snapshot/` 是 HF 原件（14 个 tar.xz，85 GB，`control/COMPLETE.json` 记 `status=PASS`），`extracted/` 是已解开的 1600 条 primary h5（740 GB）。它与旧版只有**文件形状**不同：旧版每任务一个大 h5、内含 `episode_0..99`；新版每「任务×难度」一个目录、每个 h5 只有 `episode_0`。h5 里每条 episode 的内部键、形状、dtype 与旧版一致（实测只少了 `eef_action_raw/*`、`eef_state_raw/*`、`setup/fail_recover_*` 三组键，本仓库代码全不引用）。

所以只需要**一个新脚本 `scripts/dataset/merge_v2_h5.py`**：把每个任务的 400 个单 episode h5 用 HDF5 对象拷贝（`h5py` 的 `Group.copy`，底层 `H5Ocopy`）合并成 `record_dataset_<Task>.h5`、`episode_0..399`，逐位同源；合并完成后，从 `scan_manifest.py` 到 `compute_norm_stats.py` 的整条既有链路**零改动**直接跑。已实测：单 episode（1356 步，0.9 GB）拷贝 3.7 s、242 MB/s，拷贝后 28,482 个 dataset 逐位比对（NaN 按相等处理）全部一致。

### 2. 为什么「包一层」就够——四个已核实的事实

- **任务列表不是写死的。** `scripts/dataset/paths.sh` 的 `TARGET_TASKS` / `TARGET_TASKS_CSV` 只被 `v1_validate_raw_h5` 消费，正式建库序列（见 `docs/dataset-build-doc/4task-motion-400ep/launch.md`）从不调它；`scan_manifest.py build` 的任务集走命令行 `--tasks`，`run_local.py` 不认任务名。
- **原始 H5 根不是写死的。** `RAW_H5_DIR` 在 `paths.sh` 里是 `RAW_H5_DIR="${RAW_H5_DIR:-…}"`，环境变量可覆盖；`run_local.py --raw-dir`、`finalize_checks.py --raw_dir` 都显式收参数。
- **文件名规则合并后自然满足。** 链路对文件名的唯一硬要求是 `record_dataset_<Task>.h5`（`scan_manifest.canonical_h5_order`、`wan/wan_common.task_of_h5`、`datastore/motion_store.task_of_h5`），episode 身份是 `(h5_file, raw_ep_idx)`。合并成 4 个文件、每个 400 条后，连将来的 motion 段 key `<Task>_ep<raw_ep_idx>_<seg>` 都不会冲突。
- **`_process_episode` 读的每个键新版都有。** `setup/task_goal`、`obs/front_rgb`、`obs/wrist_rgb`（均 (256,256,3) uint8）、`obs/joint_state`、`obs/gripper_state`、`action/joint_action`、`info/is_video_demo`、`info/is_completed`、`info/{simple,grounded}_subgoal{,_online}`，形状 dtype 与旧版逐键相同（`h5schema` 探针实测）。

被否决的替代：用 HDF5 external link 造 4 个「壳」h5 指向 1600 个文件，零拷贝、秒级完成，但 `finalize_checks.py hash-inputs` 只对壳文件算 sha256，输入指纹会失去意义；且壳文件永久依赖 `extracted/` 的相对路径。物理合并多花约 794 GB 磁盘与不到 1 小时 I/O，换来输入指纹、守卫抽检与将来 motion 阶段全部按既有口径成立，值得。

### 3. 新旧数据差别（实测，非 README 转述）

| 项目 | 旧版（`4task-motion-400ep` 所用，`Yinpei/robomme_data_h5`） | 新版 `robomme-4task-h5-20260912-v2` |
|---|---|---|
| 任务 | ButtonUnmask / ButtonUnmaskSwap / VideoUnmask / VideoUnmaskSwap | BinFill / RouteStick / VideoUnmaskSwap / VideoRepick |
| 文件形状 | `record_dataset_<Task>.h5`，内含 `episode_0..99` | `record_dataset_<Task>_<难度>/hdf5_files/<Task>_ep<N>_seed<S>.h5`，每文件只有 `episode_0` |
| 难度维度 | 无 | 14 个组合（BinFill 无 xhard、VideoRepick 无 hard），每任务合计 400 条 primary |
| 规模 | 400 集 / 123,044 步 / 82 GB | 1600 集 / 1,192,918 步 / 793.8 GB |
| 单集长度 | 均值约 308 步 | 各组均值 250–1636 步（BinFill hard 最长） |
| h5 键集 | — | 少 `eef_action_raw/*`、`eef_state_raw/*`、`setup/fail_recover_*`；其余逐键相同 |
| demo 段 | BinFill 没有 video demo | **BinFill 也带 video demo 段**（`BinFill_ep0` 首帧 `is_video_demo=True`） |
| 备件 | 无 | `spare/` 196 条、`smoke/` 1 条——**本轮不收** |

各任务合并后的体量（按 `snapshot/MANIFEST.json` 的 `h5_bytes` / `timestep_count` 汇总）：

| 任务 | 难度组成 | 步数 | 合并后 h5 |
|---|---|---:|---:|
| BinFill | easy 134 + medium 133 + hard 133 | 513,858 | 341.0 GB |
| RouteStick | easy/medium/hard/xhard 各 100 | 214,900 | 143.4 GB |
| VideoRepick | easy 134 + medium 133 + xhard 133 | 304,453 | 202.9 GB |
| VideoUnmaskSwap | easy/medium/hard/xhard 各 100 | 159,707 | 106.5 GB |

### 4. 预处理全过程（`scripts/dataset/merge_v2_h5.py`，新脚本）

预处理只做「重新打包」，不做任何内容改写；三个子命令依次执行：

**(a) `plan`——选条与排序。** 读 `snapshot/MANIFEST.json`，只取 `episodes[]` 里 `role == "primary"` 的条目（恰 1600 条，`spare` / `smoke` 落选）；按任务分组，组内先按难度 `easy → medium → hard → xhard`、再按原 `episode` 号升序排，依次编成 `episode_0..399`。输出 `record_dataset_<Task>_episode_map.json`：每行记 `new_idx / task / difficulty / episode / seed / member（extracted 下相对路径）/ h5_sha256（抄 MANIFEST）/ timestep_count`，并写 `source_pin.json`（`repo_id / revision / manifest_sha256`，逐字抄 `control/source.json`）。同时核对：每任务恰 400 条、`extracted/` 下对应文件都在、目录下 h5 数与 README 的条数表一致。

**(b) `merge`——逐任务合并。** 每个任务一个进程（`--procs 4`，4 任务并行），对 map 里每条执行 `src["episode_0"]` → `dst.copy(name=f"episode_{new_idx}")`；先写 `record_dataset_<Task>.h5.tmp`，全部拷完再原子改名，任务级可断点续跑（目标已存在且 `verify` 通过即跳过）。拷贝前对源文件按需重算 sha256 并与 map 里的 `h5_sha256` 对照（README 明说该值「逐字抄自生成侧」，这是它第一次被独立核对），不符即整任务停。预计墙钟：单流实测 242 MB/s，BinFill 341 GB 约 24 分钟，四任务并行取决于 RAID 并发扩展，**25–60 分钟**之间，以 smoke 实测外推。

**(c) `verify`——逐位校验。** 对每个合并文件：① `episode_*` 键恰为 0..399；② 每条 `timestep_*` 数与 MANIFEST `timestep_count` 相等；③ **全量逐 dataset 逐位比对**源文件与合并文件（浮点用 `equal_nan=True`，`waypoint_action` 首步全 NaN 是正常值）；④ 把 4 个合并文件的 sha256 写进 `MERGE_DONE.json`。通过后打印判定行 `MERGE_VERIFY=PASS tasks=4 episodes=1600 timesteps=1192918 mismatches=0`。全量比对要把 794 GB 各读一遍，8 进程约 30 分钟；`--level sample` 只抽每集首末两步，留给 smoke 用。

落点：`v1-store/raw-h5/4task-20260912-v2/`（AGENTS.md 第 14 条：除全局原始 H5 外的派生物一律进 `v1-store/`；合并文件是派生物，`snapshot/` 才是原件）。目录里只有 4 个 `.h5` 加几份 json，`hash-inputs` 只取 `*.h5`，与旧 `robomme_data_h5/` 的用法完全一致。

### 5. 全流程按脚本调用顺序（含 motion，标明本轮做 / 先不做）

以下命令全部在仓库根执行，`uv run --no-sync`；预计超过 5 分钟的阶段各起一个 `v2b-<阶段>` 前缀的 detached tmux 会话，日志 `PYTHONUNBUFFERED=1` + `set -o pipefail` + `tee`，结束写 `EXIT_CODE=`，挂 Monitor 过滤 `STAGE_DONE|STAGE_FAIL|MERGE_VERIFY|FINALIZE_EXIT_CODE|VERIFY_PACK|Traceback|out of memory|EXIT_CODE=`。GPU 只用 **4,5,6,7**（0–3 正在跑 MotionJEPA 的 `tr-wan-full1600-filter2-b176x4-72ep-a`）。库名以下暂记 `<LIB名>`，待拍板。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/dataset/paths.sh; v1_prepare_dirs; v1_require_models        # 阶段 0：目录 + SigLIP 权重内容校验
SRC=/scratch/hongze/robomme-4task-h5-20260912-v2
RAW=$V1_STORE/raw-h5/4task-20260912-v2                                   # 合并后的「旧形状」根，替代 RAW_H5_DIR
LIB=$V1_STORE/datasets/<LIB名>
TASKS=BinFill,RouteStick,VideoUnmaskSwap,VideoRepick
```

| # | 阶段 | 命令（要点） | 资源 / 预计 | 产物 | 本轮 |
|---|---|---|---|---|---|
| 1 | 合并计划 | `merge_v2_h5.py plan --snapshot $SRC/snapshot --extracted $SRC/extracted --out $RAW --tasks $TASKS` | CPU，秒级 | `$RAW/*_episode_map.json`、`source_pin.json` | **做** |
| 2 | 合并 | `merge_v2_h5.py merge --out $RAW --procs 4 --check-source-sha256` | CPU + NVMe，25–60 min | `$RAW/record_dataset_<Task>.h5` ×4 | **做** |
| 3 | 合并校验 | `merge_v2_h5.py verify --out $RAW --extracted $SRC/extracted --level full --procs 8` | CPU，约 30 min | `MERGE_DONE.json`，判定行 `MERGE_VERIFY=PASS` | **做** |
| 4 | episode 清单 | `scan_manifest.py build --raw_dir $RAW --tasks $TASKS --episodes-per-task 400 --num_shards 1 --out $LIB/meta/episode_manifest.json` | CPU，数分钟（要打开 1600 条数 timestep） | `meta/episode_manifest.json`（身份 `h5_file`+`raw_ep_idx`，`totals.timesteps=1192918`） | **做** |
| 5 | 输入指纹 | `finalize_checks.py hash-inputs --raw_dir $RAW --out $LIB/meta/input_manifest.json` | CPU，4 文件 794 GB 串行约 40 min（可与 4 并行） | `meta/input_manifest.json` | **做** |
| 6 | SigLIP 帧路 | `run_local.py --stage siglip --lib $LIB --gpus 4,5,6,7 --raw-dir $RAW` | GPU 4–7，按 400ep 实测每卡 74 步/s 推 **约 70 min** | `source/{features,data,meta}`、`logs/siglip-gpu*.log`，`STAGE_DONE stage=siglip` | **做** |
| 7 | 守卫 | `CUDA_VISIBLE_DEVICES=7 finalize_checks.py check --manifest … --out $LIB/source --raw_dir $RAW --input_manifest … --input_level sha256 --spot_check 1024` | GPU 7，约 10 min | `FINALIZE_EXIT_CODE=0` | **做** |
| 8 | framesamp 4×4 | `pack_framesamp_store.py pack --source $LIB/source --manifest … --out $LIB/framesamp --procs 48`；`verify --store $LIB/framesamp --resume --procs 48` | CPU 48 进程，分钟级 | `framesamp/`（约 75 GB），`VERIFY_PACK=PASS … mismatches=0` | **做** |
| 9 | norm_stats | `compute_norm_stats.py --output-dir $V1_STORE/train-assets/mme_vla_suite/<LIB名> --config-name mme_vla_suite --repo-id robomme --dataset-path $LIB/source` | CPU，约 10 min | `train-assets/mme_vla_suite/<LIB名>/robomme/norm_stats.json` | **做** |
| 10 | framesamp 8×8 | `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu pack_framesamp_store.py pack --layout framesamp-8x8-v1 --reader decode --source $LIB/source --manifest … --out $LIB/framesamp-8x8 --procs 48`；`verify`；`report`；`xgrid_pos_check.py --store-4x4 $LIB/framesamp --store-8x8 $LIB/framesamp-8x8 --source $LIB/source --image-spot 512` | CPU 48 进程，约 300 GB | `framesamp-8x8/`，`VERIFY_PACK=PASS`、xgrid 独立校验通过 | **做**（2026-09-14 用户拍板） |
| 11 | Wan 抽取 | `run_local.py --stage wan --lib $LIB --gpus … --raw-dir $RAW`（需 `v1_require_wan`） | GPU，按 400ep 每卡 1.43 s/窗推约 10 倍窗数 | `wan-latents/` | **先不做** |
| 12 | motion encoder | `run_local.py --stage encode --lib $LIB --gpus …` | GPU，分钟级 | `motion-tokens/` | **先不做** |
| 13 | motion 表 | `pack_motion_store.py pack --manifest … --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion`；`verify --store $LIB/motion --resume` | CPU | `motion/` | **先不做** |
| 14 | motion oracle 对拍 | `oracle_driver.py --mj-repo $MJ vae … --shard-idx i --num-shards 8`（8 片）→ `aggregate --kind vae` → `encoder … --expected-ckpt-sha256 …`；`compare_wan.py latents` / `tokens`；`motion_checks.py a6/a7/a8/a9set/a9enc/a10`、M1；`dataloader_bench --lib $LIB` | GPU + CPU | `oracle/wan-mj/` | **先不做** |

「先不做」的三点说明：

- **为什么现在不做**：用户本轮明确只处理数据、不处理 motion；且 GPU 0–3 被 MotionJEPA 正式训练占用，Wan 抽取按 400ep 实测（6,832 窗 1639 s / 6 卡）推到约 6.6 万窗，4 卡要 6 小时以上，不宜与训练抢卡。
- **将来补做不需要重建任何东西**：阶段 11–14 只吃 `meta/episode_manifest.json` 与 `$RAW` 下的合并 h5，都在本轮产出且不再变；`wan_common.task_of_h5` 认 `record_dataset_<Task>.h5`，段 key `<Task>_ep<0..399>_<seg>` 唯一。唯一要留意的是 `pack_motion_store.gather_provenance` 要求 Wan 与 encode 两阶段 `git_commit` 一致（记忆 `no-commit-between-wan-and-pack`），届时两阶段之间不要 commit。
- **训练侧本轮也不动**：新库用 `perceptual-framesamp-context.yaml`（`motion.enabled: false`）即可读，`dataloader._motion_gates` 在关闭态直接返回 None、不碰 motion store。起正式训练前需在 `training/config.py` 加一个指向新 norm_stats 目录的 `AssetsConfig` 条目，属训练任务，另立。

### 6. 资源核算

| 项目 | 估算 | 依据 |
|---|---:|---|
| 合并 h5（`$RAW`） | 793.8 GB | MANIFEST `h5_bytes` 汇总 |
| `source/` 特征 + pkl | 约 1.0 TB | 400ep 为 107 GB / 123,044 步，按 9.7 倍步数推 |
| `framesamp/` 4×4 | 约 75 GB | 400ep 为 7.6 GB |
| `framesamp-8x8/` | 约 300 GB | 400ep 为 31 GB |
| 合计 | 约 2.2 TB | 当前 `/scratch` 余 3.9 TB，建完余约 1.7 TB |
| 合并 + 校验墙钟 | 1–1.5 h | 单流 242 MB/s 实测 |
| SigLIP 墙钟 | 约 70 min（4 卡） | 400ep：3 卡 551 s / 123,044 步 |

`extracted/`（740 GB）在阶段 3 `MERGE_VERIFY=PASS` 且阶段 7 守卫通过后可以删除以回收空间——`snapshot/` 的 tar.xz 仍是原件，可随时重解。是否删除交用户拍板，本计划默认**保留**。

### 7. 待用户拍板

1. **库名**：建议 `4task-v2-1600ep`（对应 `v1-store/datasets/4task-v2-1600ep/`、`train-assets/mme_vla_suite/4task-v2-1600ep/`）。
2. **episode 范围**：建议全部 1600 条 primary、全部难度；`spare` / `smoke` 不收。若只要子集，给每任务×难度的 N，`plan` 子命令加 `--per-group N`。
3. **合并后是否删除 `extracted/`**：默认保留。
4. **GPU 范围**：默认只用 4–7。

已拍板：**`framesamp-8x8` 本轮一并建**（2026-09-14），阶段 10 与阶段 8 一样是正式交付件，留档同一份 result.md。

---

## 第二部分（技术细节，供 agent 追踪）

### 新脚本 `scripts/dataset/merge_v2_h5.py`

- 位置与依赖：放 `scripts/dataset/`，只依赖主 venv 已有的 `h5py` / `numpy`；路径解析沿用 `paths.sh` 的 `V1_STORE`，不新增外部目录。
- 子命令与关键函数：
  - `plan`：`load_manifest(snapshot) -> dict`（同时校验 `MANIFEST.json` 的 sha256 等于 `control/source.json` 的 `manifest_sha256`）；`select_primary(manifest, tasks, per_group=None) -> dict[task, list[row]]`（`role == "primary"`；排序键 `(DIFFICULTY_ORDER[difficulty], episode)`，`DIFFICULTY_ORDER = {"easy":0,"medium":1,"hard":2,"xhard":3}`）；`write_episode_map(out, task, rows)`；`write_source_pin(out, source_json)`。计划阶段还要 `assert len(rows) == 400` 与 `os.path.exists(extracted/member)`。
  - `merge`：`merge_task(task, map_path, extracted, out, check_sha)`——对每行 `with h5py.File(src, "r") as s, h5py.File(tmp, "a") as d: s.copy("episode_0", d, name=f"episode_{new_idx}")`；`.tmp` 全部写完 `os.replace` 成正式名；`--procs` 用 `multiprocessing` 按任务派发；每完成一条打印 `MERGE_EP task=<T> idx=<i> member=<…> secs=<…>`；`--check-source-sha256` 时先 `hashlib.sha256` 流式算源文件并与 map 对照。
  - `verify`：`verify_task(task, map_path, extracted, out, level)`——`level=full` 用 `Group.visititems` 收集源与目标每个 dataset，`np.array_equal(a, b, equal_nan=True)`（浮点）/ `np.array_equal`（其它）/ `==`（标量、bytes）；`level=sample` 只比 `setup`、`timestep_0`、最后一步。全部通过写 `MERGE_DONE.json`（含 4 个合并文件 sha256、逐任务 episode 与 timestep 总数、`source_pin`、`git_commit`），打印 `MERGE_VERIFY=PASS tasks=4 episodes=1600 timesteps=1192918 mismatches=0`，否则 `MERGE_VERIFY=FAIL` 并列出前 20 条不符。
- 幂等与安全：`merge` 遇到已存在的正式文件且 `MERGE_DONE.json` 里记的 sha256 与现算相符则跳过；不带 `--force` 不删任何已有文件；`--force` 只删本任务的 `.tmp` 与正式文件，起跑前 `ls -ld $RAW` 确认是本环境实体目录（AGENTS.md 第 14 条）。

### 既有链路零改动的核对点

- `scripts/dataset/scan_manifest.py`：`canonical_h5_order(raw_dir, tasks)` 按 `--tasks` 顺序找 `record_dataset_<Task>.h5`，缺一即 `SystemExit`；`scan` 对每文件取 `episode_*` 升序，`--episodes-per-task 400` 时要求每文件 ≥ 400 条（恰好 400）。清单 `canonical_order` 将是 `[BinFill, RouteStick, VideoUnmaskSwap, VideoRepick]` 四个文件名。
- `scripts/dataset/build_shard.py` worker 模式：`handles` 按 `h5_file` 缓存句柄，合并后只有 4 个文件，与旧库一样。单 episode 内存峰值按最长 1636 步 × 2 相机 × 256×256×3 ≈ 640 MB，A100 主机内存无压力。
- `scripts/dataset/finalize_checks.py`：`cmd_hash_inputs` 只扫 `raw_dir` 顶层 `*.h5`——`$RAW` 顶层恰 4 个；`spot_check` 走 `data[f"episode_{raw_ep_idx}"]`，合并后成立。
- `src/mme_vla_suite/dataset_builder/build_robomme_dataset.py::_process_episode`：`assert ts["info"]["is_video_demo"][()] == (step_idx < exec_start_idx)` 要求 demo 段是前缀；MotionJEPA 侧 `scripts/dataset-build/local_dataset_source.py::scan_episode` 已对这 1600 条断言过前缀性。`is_completed` 首帧就为真的分支会让 `simple_subgoal` 未定义，smoke 阶段 14 个组合各 1 集探一遍，未命中不改代码。
- `scripts/dataset/wan/wan_common.py::task_of_h5`、`src/mme_vla_suite/datastore/motion_store.py::segment_key`：合并文件名合规，将来阶段 11–14 可直接跑。

### 第 18 条链路图与一致性两块的适用说明

本轮**不改任何训练链路代码**，只在链路最前面加一跳：

```
extracted/record_dataset_<Task>_<难度>/hdf5_files/<Task>_ep<N>_seed<S>.h5  [episode_0，(256,256,3) uint8 等，原件]
        │  merge_v2_h5.py merge（H5Ocopy 对象拷贝；不改数、不改 dtype、不改 chunk）
        ▼
$RAW/record_dataset_<Task>.h5  [episode_0..399；逐 dataset 逐位同源，verify --level full 判定]
        │  以下与 4task-motion-400ep 完全相同：scan_manifest → hash-inputs → build_shard(SigLIP) → finalize → pack_framesamp → norm_stats
        ▼
$LIB/source → $LIB/framesamp → train-assets/.../norm_stats.json
```

- **第一块（非训练轻量对拍）**：即阶段 3 的 `verify --level full`，判据是全部 1600 条 × 全部 dataset 逐位相等（浮点 `equal_nan`），`mismatches=0`。
- **第二块（训练梯度一致）**：不适用。既有链路代码零改动，新数据也不存在「旧链路交付」可作对照；本计划不宣称任何链路等价，只宣称「合并文件与原件逐位同源」+「链路代码与 `4task-motion-400ep` 起跑 commit 起无语义改动」（以 `git diff <400ep 起跑 commit>..HEAD -- scripts/dataset/ src/mme_vla_suite/dataset_builder/` 为证，写进留档）。

### 验证（每步 ≤ 5 分钟，改动后必跑）

1. **脚本自测**：`scripts/dataset/test_guards.py` 现有用例照跑；给 `merge_v2_h5.py` 加最小用例——用 `h5py` 在临时目录造 2 任务 × 3 条假 episode（各 2 步、含 NaN 浮点）跑 `plan → merge → verify --level full`，断言 `MERGE_VERIFY=PASS` 且改动任一字节后 `FAIL`。
2. **真实 smoke**：`plan --per-group 1` 得 14 条（每组合 1 集）→ `merge` 到 `v1-store/raw-h5/4task-20260912-v2-smoke/`（约 10 GB，秒到分钟级）→ `verify --level full` → 阶段 4–8 与阶段 10 在 `v1-store/datasets/4task-v2-smoke14/` 上跑通（SigLIP 单卡 GPU7，约 1 万步、2–3 分钟）→ `FINALIZE_EXIT_CODE=0`、`VERIFY_PACK=PASS`。用 smoke 的合并墙钟外推正式合并耗时。验收完成后删除 smoke 的两个目录（第 6 条临时 run 清理）。
3. **训练可读性**（正式库建完后）：`perceptual-framesamp-context.yaml` 起 `--dataset-path $LIB/framesamp` 跑 20 步，确认 dataloader 出 batch、`motion_*` 键为 None，跑完删 run。

### 留档与 commit

- 代码：`commitV9.3: 新增新版单 episode H5 合并预处理脚本 merge_v2_h5.py`（只 `git add scripts/dataset/merge_v2_h5.py` 与其测试），commit 后立即 `git push`。
- 建库留档：`docs/dataset-build-doc/<LIB名>/{launch.md,result.md,records/}`——launch 记起跑 HEAD、本节命令原文、`source_pin.json`；result 记 `MERGE_VERIFY` / `STAGE_DONE` / `FINALIZE_EXIT_CODE` / `VERIFY_PACK`（4×4 与 8×8 各一行）/ `xgrid_pos_check` 判定行原文、各阶段墙钟与 GPU、产物体积、`input_manifest.json` 与 `framesamp.store_meta.json` 副本、norm_stats sha256；`docs/dataset-build-doc/README.md` 表加一行。提交为 `docs: <LIB名> 建库留档`。
- 本轮起过的 tmux 会话名（`v2b-merge`、`v2b-verify`、`v2b-hash`、`v2b-siglip`、`v2b-finalize`）记入 launch.md，清理只按此清单逐个 `tmux kill-session -t <名>`。

### 明确不动的文件

`scripts/dataset/{scan_manifest,build_shard,finalize_checks,pack_framesamp_store,run_local,pack_motion_store,motion_checks}.py`、`scripts/dataset/wan/*`、`scripts/dataset/paths.sh`（`TARGET_TASKS` 保留旧四任务，新任务集只走 `--tasks`）、`src/mme_vla_suite/dataset_builder/*`、`src/mme_vla_suite/datastore/*`、`src/mme_vla_suite/training/*`、`third_party/robomme_benchmark`。

### 已知但本轮不做的关联项

- 评估侧 `examples/robomme/utils.py::TASK_WITH_VIDEO_DEMO` 不含 BinFill，而新版 BinFill 带 demo 段；评估用新库训练的模型前要核对 `env_runner` 对 BinFill 的 demo 处理。
- `scripts/training/legacy-eval/eval_all_shards.local.sh`、`check_test_seeds.py`、`plot_eval_success_by_length.py` 写死旧四任务与旧 h5 目录。
- `scripts/training/paths.sh` 的 `EXPECTED_H5`（旧四任务文件名，无消费方）。
