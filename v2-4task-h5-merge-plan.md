# 新版四任务 H5（robomme-4task-h5-20260912-v2）合并预处理与建库计划

> 2026-09-14 起草，同日经对抗验证（Claude 六维度 + Codex）全面修订。环境判定：**环境 B（AWS 单机 8×A100-SXM4-80GB）**，仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，无 `/nfs/turbo`、无 `/data/hongzefu`、无 `~/.ssh/config`。
> 本计划的核心取舍：**建库链路一行不改**，只在链路前面加一次「把新版单 episode H5 合并成旧形状」的预处理；motion 相关阶段本轮**先不做**，但清单身份按可追加 motion 的方式一次定死。
>
> **修订说明（2026-09-14，对抗验证后）**：初稿有三处会导致直接崩溃的事实错误、一批失真估算与若干规则缺口，已全部落入本版。其中最关键的一条：初稿断言「每个 h5 只有 `episode_0`」是**错的**（实测 1587/1600 不是），初稿的合并语句按该断言写死，会在 1600 条里挂掉 1587 条。凡本版标注「实测」的数字均为本环境实机测量，不再转述上游 README——**`snapshot/README.md` 已被证实至少有一处错误（见第 1 节），不作为事实来源。**

---

## 第一部分（给人看）

### 1. 结论

新版数据 `HongzeFu/robomme-4task-h5-20260912-v2`（revision `604f16da36d6b6d175884df8fb687dc08e0a36eb`）的本地副本已经完整落在 `/scratch/hongze/robomme-4task-h5-20260912-v2/`：`snapshot/` 是 HF 原件（14 个 tar.xz，90.5 GB / 84.3 GiB），`extracted/` 是已解开的 1600 条 primary h5（793.8 GB / 739.3 GiB），`control/COMPLETE.json` 记 `status=PASS`。注意 **`control/` 与 `snapshot/` 平级**，不在 `snapshot/` 内部。

它与旧版只有**文件形状**不同：旧版每任务一个大 h5、内含 `episode_0..99`；新版每「任务×难度」一个目录、每个 h5 只含**一个顶层组**。

> **关键更正（初稿错误）**：那个顶层组**不叫 `episode_0`**，而是 `episode_<该条在难度组内的原始 episode 号>`。全量扫描 1600 个文件实测：顶层组数恒为 1（这一点成立），但组名等于 `episode_0` 的**只有 13 个**，其余 1587 个是 `episode_26`、`episode_111` 这类，恒等于文件名里的 `_ep<N>_` 与 MANIFEST 的 `episode` 字段（三者一致性 1600/1600）。生成侧自己的 `control/final_verification.json` 写着 `h5_group_matches_source_episode: 1600`，早已佐证这一点；`snapshot/README.md` 里「每个 `.h5` 文件里只有 `episode_0` 一条轨迹」那句是**错的**，初稿照抄了它。
>
> 初稿之所以没发现，是因为它的实测样本「单 episode（1356 步，0.9 GB）」恰好是 `BinFill_ep0_seed4000.h5` —— 正是那 13 个特例之一。

h5 里每条 episode 的内部键、形状、dtype 与旧版一致（实测只少了 `eef_action_raw/*`、`eef_state_raw/*`、`setup/fail_recover_*` 三组键，本仓库代码全不引用；该差异清单因旧版 h5 不在本环境而**无法独立核实**，但已确认建库与训练链路对这三组键零引用）。

所以只需要**一个新脚本 `scripts/dataset/merge_v2_h5.py`**：把每个任务的 400 个单 episode h5 用 HDF5 对象拷贝（`h5py` 的 `Group.copy`，底层 `H5Ocopy`）合并成 `record_dataset_<Task>.h5`、`episode_0..399`，逐位同源；合并完成后，从 `scan_manifest.py` 到 `compute_norm_stats.py` 的整条既有链路**零改动**直接跑。

`Group.copy` 的保真性已用合成数据逐项实测：attrs（组与 dataset）、dtype（含 vlen 字符串、定长 bytes、复合、bool、标量、0 长度维度、`h5py.Empty`）、chunk 布局、gzip / lzf / shuffle / fletcher32 / scaleoffset、fillvalue、maxshape、NaN、嵌套子组——**17 类全部保留**，且字节确定性（同一源两次 copy 的 sha256 相同，`track_times` 默认关、object header mtime 恒为 0）。真实数据比这简单得多：**无 attrs、无压缩、全 contiguous、无软链接/外部链接**，合并体积比源小 0.45%（HDF5 重打包消除碎片）。

### 2. 为什么「包一层」就够——四个已核实的事实

- **任务列表不是写死的。** `scripts/dataset/paths.sh` 的 `TARGET_TASKS` / `TARGET_TASKS_CSV` 只被 `v1_validate_raw_h5` 消费（全仓库 grep 确认无第二个消费方），正式建库序列（见 `docs/dataset-build-doc/4task-motion-400ep/launch.md`）从不调它；`scan_manifest.py build` 的任务集走命令行 `--tasks`，`run_local.py` 不认任务名。**阶段 0 不要调 `v1_validate_raw_h5`**——它会拿旧四任务名去 `$RAW` 找，必然报「缺目标 h5」。
- **原始 H5 根不是写死的，但覆盖方式要说准。** `RAW_H5_DIR` 在 `paths.sh` 里是 `RAW_H5_DIR="${RAW_H5_DIR:-…}"`，**既不 `readonly` 也不 `export`**：用户预先 `export RAW_H5_DIR=…` 可以覆盖默认值，但 `run_local.py` 不 source `paths.sh`（其 docstring 明写），它的 `--raw-dir` 默认值走 `os.environ.get("RAW_H5_DIR", …)`，读不到未导出的 shell 变量、会回落到环境 B 下不存在的路径。**故本计划一律显式传 `--raw-dir` / `--raw_dir`，不依赖环境继承。**
- **文件名规则合并后自然满足，且不存在任务名白名单。** 三处文件名解析器（`scan_manifest.canonical_h5_order` 拼字符串查存在性、`wan/wan_common.task_of_h5` 与 `datastore/motion_store.task_of_h5` 纯前后缀剥离）全是模式匹配，**对任意任务名成立**，四个新任务名均可正确解析。episode 身份是 `(h5_file, raw_ep_idx)`，段 key `<Task>_ep<raw_ep_idx>_<seg>` 不带库名，但所有消费方都在单库目录内定位，且 `training/dataloader.py::_motion_gates` 有 `check_same_source(manifest_sha256, …)` 的内容级绑定——新旧库的 `VideoUnmaskSwap_ep0_exec` 字面同名但**不会串味**，误接会 fail-loud。
- **`_process_episode` 读的每个键新版都有（共 10 组，初稿漏列 1 组）。** `setup/task_goal`、`obs/{front_rgb, wrist_rgb}`（均 (256,256,3) uint8）、`obs/joint_state`、`obs/gripper_state`、`action/joint_action`、`info/{is_video_demo, is_completed}`、`info/{simple,grounded}_subgoal{,_online}`，**以及初稿漏列的 `info/is_subgoal_boundary`**——该键在主循环里**无条件读取**（只在 `--visualize` 时才被 `keyframe_idxs` 消费，但读取本身无条件），若缺失会在阶段 6 的 SigLIP 第一条 episode 就 `KeyError`。实测 1600/1600 存在，形状 `[]`、dtype `bool`。

被否决的替代：用 HDF5 external link 造 4 个「壳」h5 指向 1600 个文件，零拷贝、秒级完成，但 `finalize_checks.py hash-inputs` 只对壳文件算 sha256，输入指纹会失去意义；且壳文件永久依赖 `extracted/` 的相对路径。物理合并多花约 790 GB 磁盘与约 1 小时 I/O，换来输入指纹、守卫抽检与将来 motion 阶段全部按既有口径成立，值得。

### 3. 新旧数据差别（实测，非 README 转述）

> 单位约定：本节及第 6 节的 GB 一律为 10⁹ 口径，GiB 为 2³⁰ 口径，同时给出以便与 `du -h` / `df -h` 对照。初稿在同一份文档里对同一批字节混用两种进制（`extracted` 写 740 GB、合并后写 793.8 GB，其实是同一批 793,815,485,572 字节），已统一。

| 项目 | 旧版（`4task-motion-400ep` 所用，`Yinpei/robomme_data_h5`） | 新版 `robomme-4task-h5-20260912-v2` |
|---|---|---|
| 来源 | 公开集 | **私有仓 `hongzefu/robomme_benchmark_MotionJEPA`，分支 `newtask-v2`，commit `20230d84`，运行编号 `20260912-contract-v3-10`，注入契约 v3**（另一次独立录制，非公开集重打包） |
| 任务 | ButtonUnmask / ButtonUnmaskSwap / VideoUnmask / VideoUnmaskSwap | BinFill / RouteStick / VideoUnmaskSwap / VideoRepick |
| 文件形状 | `record_dataset_<Task>.h5`，内含 `episode_0..99` | `record_dataset_<Task>_<难度>/hdf5_files/<Task>_ep<N>_seed<S>.h5`，每文件**恰一个顶层组，组名 `episode_<原 episode 号>`**（仅 13/1600 恰为 `episode_0`） |
| 难度维度 | 无 | 14 个组合（BinFill 无 xhard、VideoRepick 无 hard），每任务合计 400 条 primary |
| 规模 | 400 集 / 123,044 步 / 82 GB | 1600 集 / 1,192,918 步 / 793.8 GB（739.3 GiB） |
| 单集长度 | 均值约 308 步 | min 200 / 均值 745.6 / **max 2304**；组均值区间 250.0（RouteStick easy）–1636.8（BinFill hard） |
| **exec 比例** | **0.821**（`stats.json`：`execution_samples` 101,066 / `total_samples` 123,044） | **0.518**（抽样 70 集实测 0.5179，按任务步数加权 0.5176）→ **可训练样本约 61.7 万，是旧库 10.1 万的 6.1 倍，不是按总步数推的 9.7 倍** |
| h5 键集 | — | 少 `eef_action_raw/*`、`eef_state_raw/*`、`setup/fail_recover_*`；其余逐键相同 |
| **dtype 一致性** | — | **跨任务不一致**：`obs/gripper_state` 与 `action/waypoint_action` 在 RouteStick 是 float64、其余三任务 float32；`action/waypoint_action` 在同一 episode 内也会从 float32（首步）漂移到 float64 |
| demo 段 | BinFill 没有 video demo | **BinFill 也带 video demo 段**，且 demo 段恰为整集一半（BinFill / RouteStick 全组 exec 比例 0.5000）；严格前缀性实测 **1600/1600 通过** |
| 备件 | 无 | `spare/` 196 条、`smoke/` 1 条——**本轮不收，且由 `plan` 做多重硬保证排除**（见第 4 节 (a)） |

各任务合并后的体量（按 `snapshot/MANIFEST.json` 的 `h5_bytes` / `timestep_count` 汇总，已逐项重算核对）：

| 任务 | 难度组成 | 步数 | 合并后 h5 |
|---|---|---:|---:|
| BinFill | easy 134 + medium 133 + hard 133 | 513,858 | 341.0 GB（317.6 GiB） |
| RouteStick | easy/medium/hard/xhard 各 100 | 214,900 | 143.3 GB（133.5 GiB） |
| VideoRepick | easy 134 + medium 133 + xhard 133 | 304,453 | 203.0 GB（189.0 GiB） |
| VideoUnmaskSwap | easy/medium/hard/xhard 各 100 | 159,707 | 106.5 GB（99.2 GiB） |
| **合计** | 1600 条 | **1,192,918** | **793.8 GB（739.3 GiB）** |

最长单集：`record_dataset_BinFill_hard/hdf5_files/BinFill_ep111_seed15100.h5`，**2304 步 / 1.53 GB**。该数字关系到两处判定，初稿都用错了（错把组均值 1636 当成最长单集）：单 episode 内存峰值（见第二部分）、以及 `MemoryBuffer` 的 4096 步上限（见第 4 节 (a) 的前置断言）。

### 4. 预处理全过程（`scripts/dataset/merge_v2_h5.py`，新脚本）

预处理只做「重新打包」，不做任何内容改写；三个子命令依次执行。

**(a) `plan`——选条、排序与全量前检。**

读 `snapshot/MANIFEST.json` 与 `control/source.json`（两者不同级，需分别传 `--snapshot` 与 `--control`），校验 `MANIFEST.json` 的 sha256 等于 `source.json` 的 `manifest_sha256`（实测两者相符：`df992cdf…2bb9`）。

**spare / smoke 排除的四重硬保证**（用户明确要求「必须保证好」）：

1. 只取 `role == "primary"` 的条目；
2. 实算 role 分布必须逐项等于 MANIFEST 顶层 `counts` 字段（实测 `{primary: 1600, spare: 196, smoke: 1}`，两者已确认一致），任一项不符即停；
3. 选中集合的 `member` 与 spare / smoke 的 `member` 集合**交集必须为空**；
4. 每条选中记录在 `_episode_map.json` 里保留 `role` 字段，`verify` 阶段再核一遍，并在判定行显式打印 role 分布。

另外 `extracted/` 下实测**只解开了 primary**（spare 0/196、smoke 0/1 未解开），构成第五重天然屏障。

排序：按任务分组，组内先按难度 `easy → medium → hard → xhard`、再按原 `episode` 号升序，依次编成 `episode_0..399`。注意组内 episode 号**不连续**（生成侧有 46 条候选被拒，如 BinFill hard 缺 13 个号），且 `(task, episode)` 与 `seed` 都**不全局唯一**（475 个 `(task, episode)` 键重复、220 个 seed 重复）——定位源文件必须用 `(task, difficulty, episode)` 三元组。

输出 `record_dataset_<Task>_episode_map.json`：每行记 `new_idx / task / difficulty / episode / seed / role / member（extracted 下相对路径）/ src_group / h5_sha256（抄 MANIFEST）/ timestep_count`，并写 `source_pin.json`（`repo_id / revision / manifest_sha256`，逐字抄 `control/source.json`）。

> **`src_group` 由 `plan` 一次算定并写进 map**，merge 与 verify 共用这一份、不各自推断——这是初稿 `episode_0` 错误的结构性防范。

**全量前检**（秒级到分钟级，全部在 `plan` 阶段做完，避免把问题拖到阶段 6 烧掉 GPU 才暴露）：

- `max(timestep_count) < 4096` —— `dataset_builder/mem_buffer.py::MemoryBuffer` 的位置编码是预生成定长表（`max_steps=4096`），`add_buffer` 按 `pos_emb_dict[...][step_idx*num_views : …]` 切片取值，而 **numpy 切片越界不报错、返回空数组**，会静默产出空 `pos_emb` 一路到阶段 8 的 pack 写侧校验才炸。当前最长 2304 步，安全；此断言防的是将来换数据。
- 逐文件打开取顶层组：断言恰 1 个组、且组名等于 `f"episode_{row['episode']}"`。
- 逐条核对 `extracted/` 下文件存在且 `os.path.getsize` 等于 MANIFEST 的 `h5_bytes`（实测 1600/1600 相等）。
- 数据侧契约前检（实测结论已在本环境全量取得，此处固化为脚本常驻守卫）：`info/is_video_demo` 严格前缀（实测 1600/1600 通过）、`info/is_completed` 首帧不得为真（实测 0/1600，命中会让 `_process_episode` 的四个 subgoal 局部变量未定义而抛错）、`timestep_*` 索引连续 0..N-1（实测 1600/1600）、`info/is_subgoal_boundary` 存在（实测 1600/1600）。
- 打印判定行 `PLAN_OK tasks=4 selected=1600 role=primary:1600 max_timesteps=2304 prechecks=PASS`。

**数量断言拆成两件事**（初稿把它们混成一条 `assert len(rows) == 400`，与自己的 smoke 方案 `--per-group 1` 互斥、必挂）：

- **源全集完整性检查**：恒定断言「每任务的 primary 条数 == 400」「总数 == 1600 == `counts.primary`」，与 `--per-group` 无关，永远执行。
- **选取结果验收**：数量从 map 推导，不写死。带 `--per-group N` 时每任务只有 N×(该任务难度组数) 条（BinFill 与 VideoRepick 各 3 组、RouteStick 与 VideoUnmaskSwap 各 4 组）。

**(b) `merge`——逐任务合并。**

每个任务一个进程（`--procs 4`，4 任务并行），对 map 里每条执行 `src[row["src_group"]]` → `dst.copy(…, name=f"episode_{new_idx}")`；先写 `record_dataset_<Task>.h5.tmp`，全部拷完再 `os.replace` 原子改名（同目录 rename，不复制，故 `.tmp` 与正式文件不共存）。

> 初稿两处写死的 `s.copy("episode_0", …)` 已全部改为按 `src_group` 取，并保留 `len(s.keys()) == 1` 兜底断言。

**`--extracted` 必须显式传**（初稿阶段 2 的命令漏了）：map 里的 `member` 只是相对路径，无源根无法还原。

**断点续跑改为 episode 级**（初稿是任务级，一旦中断必然死锁）：任务级跳过看的是正式文件，而中断留下的是 `.tmp`，重跑会从 `new_idx=0` 撞上 `.tmp` 里已有的组名，HDF5 硬拒重名、每次重跑必崩，唯一出路 `--force` 要把 BinFill 的 341 GB 从头再来。正确做法是开局读 `.tmp` 已有的 `episode_*` key 集合并跳过，并逐条核对已有条目的 timestep 数与 map 相符。

> 初稿把 `with h5py.File(tmp, "a")` 放在**每条 episode** 上是**对的，不要改**：每条关闭时 flush 一次 superblock，天然成为事务边界。实测 copy 中途 SIGKILL 后，`.tmp` 可正常打开、半条 episode 完全不可见、且空间被后续写入复用（无泄漏）。若改成整个任务只开一次文件，中断会丢掉整个 341 GB。

三种状态必须区分清楚：**临时重建**（`.tmp` 存在、未 rename）、**单任务验真复用**（正式文件存在且通过该任务的自校验，可跳过）、**最终发布**（全 4 任务通过 `verify --level full` 后写 `MERGE_DONE.json`）。

`--check-source-sha256`：对源文件流式算 sha256 与 map 对照。这是该值第一次被独立核对（README 明说它「逐字抄自生成侧」），抽查 3 条已 MATCH。**现算值必须写回** map 的 `h5_sha256_local` 字段，并在 `MERGE_DONE.json` 记 `source_sha256_checked=1600 mismatches=0`——否则这次核对「算完即弃」，仍不满足 AGENTS.md 第 15 条「逐文件记 sha256」。

**`--force` 的四条脚本级硬闸**（不再只靠人工 `ls -ld`，AGENTS.md 第 14 条的纪律做进代码）：输出根是实体目录且非 symlink、`realpath` 落在 `v1-store` 下、目录内存在本脚本 `plan` 写的 `source_pin.json`、只删两个精确构造的路径（禁 glob、禁 `rmtree`）。**`--force` 不得删** `*_episode_map.json` / `source_pin.json` / `MERGE_DONE.json`。

**(c) `verify`——逐位校验。**

对每个合并文件：① `episode_*` 键恰为 0..399；② 每条 `timestep_*` 数与 MANIFEST `timestep_count` 相等；③ 全量逐 dataset 值比对源文件与合并文件；④ 三条结构守卫；⑤ 把 4 个合并文件的 sha256 与 role 分布写进 `MERGE_DONE.json`。通过后打印判定行：

```
MERGE_VERIFY=PASS tasks=4 episodes=1600 timesteps=1192918 role=primary:1600 mismatches=0
```

**比较函数必须按 dtype kind 两分支**（初稿写的三分支会崩）：

```python
def ds_equal(a, b) -> bool:
    arr = np.asarray(a)                  # h5py 读 object 标量返回 Python bytes，没有 .dtype
    if arr.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))    # u/i/b/S/O 一律不传 equal_nan
```

原因：numpy 1.26 的 `equal_nan` 只豁免 kind ∈ `biu`，对 kind `S`/`O` 一律走 `isnan` 并抛 `TypeError`。真实数据每个 timestep 有 5 个此类 dataset（`action/choice_action`、`info/{simple,grounded}_subgoal{,_online}`）、每个 setup 有 3 个，**全库 5,969,390 个会抛**。另外 `setup/task_goal` 是 `shape=(2,) dtype=object` 的**数组**，若按初稿「bytes 用 `==`」放进 `if` 会抛 `ValueError`。NaN 分支确实必要且不能省——`action/waypoint_action` 的 NaN **不止首步、中间步也有**（初稿括注「首步全 NaN」不准确，且 RouteStick 首步反而没有）。

该写法已实测：对真实 episode 的全部 4206 个 dataset 零异常、`mismatches=0`，且七类篡改（uint8 改 1 字节、float32 改 1e-7、塞 NaN、bool 翻转、bytes 追加 1 字符、object 数组改 1 元素、int64 +1）**全部检出**。

**三条结构守卫**（`visititems` 的覆盖面比「逐位」这个说法窄，必须补齐，真实数据下都是 O(1) 通过）：

- 链接类型：`visititems_links` 确认全为 HardLink。`visititems` **会跳过软链接与外部链接**，而 `Group.copy` 默认 `expand_soft=False`、**复制软链接时不重写目标路径**——在合并文件里 `/episode_0` 是真实存在的，一条 `episode_7/soft_in -> /episode_0/plain` 会**静默指向另一条 episode** 且 verify 看不见。实测真实数据 `{'HardLink': 5007}`、无软/外部链接。
- attrs：`visititems` 完全不碰 attrs。断言两侧 attrs 均为空（实测源 episode 子树 attrs 总数为 0）。
- 名字集合：先断言 `set(names_src) == set(names_dst)`，再比值。另注意同一对象的硬链接别名会被 `visititems` 去重（每个对象只回调一次），实测真实数据 4206 dataset + 801 group = 5007，与 `visititems_links` 数量相等，无别名。

**判据措辞**（初稿的「逐位比对」是过度宣称）：本计划宣称的是「**全部 dataset 数值逐位相等（NaN 按相等、±0 按相等）+ 链接类型 / attrs / 名字集合三条结构守卫通过**」。`np.array_equal` 把 `+0.0` 与 `-0.0` 视为相等，故「改任一字节必 FAIL」不成立，不写这句。

**并行单元改为 `(task, episode 区间)`**（初稿是任务级，`--procs 8` 对 4 个任务一半浪费）：`--procs 48`（机器 96 逻辑核）。`--level sample` 只抽每集首末两步，留给 smoke 用；**`sample` 与 `full` 的完成标记必须分开记，不得互认**，正式 `MERGE_DONE.json` 只能由 `--level full` 产生。

**`MERGE_DONE.json` 的完成标记必须绑定身份，不能只认 sha256**：初稿的跳过条件只比对旧 H5 的 sha256，于是「重新生成 map、改变 episode 对应关系、但保留旧 H5 和旧 DONE」会被静默跳过，而下游不读 map、无从发现身份错误。完成标记须同时绑定 `source_pin` + map 内容摘要 + 选取参数（`--tasks` / `--per-group`）+ 排序规则版本 + 4 个输出文件的 sha256，任一不符即不得复用。

落点：`v1-store/raw-h5/4task-20260912-v2/`（AGENTS.md 第 14 条：除全局原始 H5 外的派生物一律进 `v1-store/`；合并文件是派生物，`snapshot/` 才是原件）。该目录尚不存在且 `paths.sh::v1_prepare_dirs` **不创建它**，由本脚本自建。目录里只有 4 个 `.h5` 加几份 json，`hash-inputs` 与 `canonical_h5_order` 都只取顶层 `*.h5`（`os.listdir` 不递归），json 与 `.tmp`（不以 `.h5` 结尾）均不会被误扫。

### 5. 全流程按脚本调用顺序（含 motion，标明本轮做 / 先不做）

以下命令全部在仓库根执行，`uv run --no-sync`；预计超过 5 分钟的阶段各起一个 `v2b-<阶段>` 前缀的 detached tmux 会话，日志 `PYTHONUNBUFFERED=1` + `set -o pipefail` + `tee`，结束写 `EXIT_CODE=`。

**每个 tmux 会话开头都必须重新 source 一次 `paths.sh` 并重设 `SRC/RAW/LIB/TASKS`** —— `V1_STORE` 未 export、不会被子 shell 继承；且 `paths.sh` 首行是 `set -euo pipefail`、内含 8 个 `readonly` 变量，**同一个 shell 里 source 两次会因重复赋值报错并直接杀掉该 shell**，务必一个会话只 source 一次。

Monitor 一份日志挂一个，命令形态（每一级都行缓冲，AGENTS.md 第 7 条）：

```bash
tail -n +1 -F v1-store/logs/<阶段>.log | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "PLAN_OK|MERGE_VERIFY|MERGE_RESUME|STAGE_DONE|STAGE_FAIL|PACK_DONE|VERIFY_PACK|IMAGE_NPY_SPOT|MOTION_POS_XGRID|FINALIZE_EXIT_CODE|Traceback|Error|out of memory|EXIT_CODE="
```

SigLIP 阶段会产出 4 个 worker 日志加 1 个主日志，**只 tail `run_local` 主日志**，不要一条 `tail -F` 挂多个文件。

GPU 只用 **4,5,6,7**（0–3 正在跑 MotionJEPA 的 `tr-wan-full1600-filter2-b176x4-72ep-a`）。**与在跑训练的共处纪律**：不等训练结束、正常推进，但必须保证不中断、不严重降速——基线 `perf/samples_per_s = 874.0 ± 0.3`、`perf/epoch_time_s = 872.7–873.3 s`（epoch 45–52 实测波动 < 0.06%），取自 `/scratch/hongze/MotionJEPA/runs/wan-full1600-filter2-b176x4-72ep-a/train_metrics_epoch.jsonl`；每个建库阶段起跑后盯一个 epoch，**> 1% 下滑即暂停该阶段**。训练当前完全靠 page cache（实测物理读 0.00 MB/s），建库全流程 page cache 流量约 8.4 TB、是可用 cache（790 GB）的 10.6 倍，会挤出训练数据；带宽层面撑得住（重新冷读需 512 MB/s，RAID 实测单流 4.4 GB/s、4 流 11.26 GB/s），但需实测确认。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/dataset/paths.sh; v1_prepare_dirs; v1_require_models   # 阶段 0：目录 + SigLIP 权重内容校验
                                                                      # 注意：不要调 v1_validate_raw_h5
SRC=/scratch/hongze/robomme-4task-h5-20260912-v2
RAW=$V1_STORE/raw-h5/4task-20260912-v2                               # 合并后的「旧形状」根，替代 RAW_H5_DIR
LIB=$V1_STORE/datasets/4task-v2-1600ep-604f16da
TASKS=BinFill,RouteStick,VideoUnmaskSwap,VideoRepick
MANI=$LIB/meta/episode_manifest.json
INMANI=$LIB/meta/input_manifest.json
```

| # | 阶段 | 命令 | 资源 / 预计 | 产物 | 本轮 |
|---|---|---|---|---|---|
| 1 | 合并计划 | `merge_v2_h5.py plan --snapshot $SRC/snapshot --control $SRC/control --extracted $SRC/extracted --out $RAW --tasks $TASKS` | CPU，秒级—分钟级（含全量前检） | `$RAW/*_episode_map.json`、`source_pin.json`，判定行 `PLAN_OK` | **做** |
| 2 | 合并 | `merge_v2_h5.py merge --out $RAW --extracted $SRC/extracted --procs 4 --check-source-sha256` | CPU + NVMe，**约 24 min**（sha256 并行）/ 36.6 min（串行） | `$RAW/record_dataset_<Task>.h5` ×4 | **做** |
| 3 | 合并校验 | `merge_v2_h5.py verify --out $RAW --extracted $SRC/extracted --level full --procs 48` | CPU 48 进程，**约 7.5 min** | `MERGE_DONE.json`，判定行 `MERGE_VERIFY=PASS` | **做** |
| 4 | episode 清单 | `scan_manifest.py build --raw_dir $RAW --tasks $TASKS --episodes-per-task 400 --num_shards 1 --out $MANI` | CPU，**4–8 秒** | `meta/episode_manifest.json`（身份 `h5_file`+`raw_ep_idx`，`totals.timesteps=1192918`） | **做** |
| 5 | 输入指纹 | `finalize_checks.py hash-inputs --raw_dir $RAW --out $INMANI` | CPU，串行约 30 min（单核 sha256 430–450 MB/s，无 SHA-NI） | `meta/input_manifest.json` | **做** |
| 6 | SigLIP 帧路 | `run_local.py --stage siglip --lib $LIB --gpus 4,5,6,7 --raw-dir $RAW` | GPU 4–7，**65–90 min** | `source/{features,data,meta}`、`logs/siglip-gpu*.log`，`STAGE_DONE stage=siglip` | **做** |
| 7 | 守卫 | `CUDA_VISIBLE_DEVICES=7 finalize_checks.py check --manifest $MANI --out $LIB/source --raw_dir $RAW --input_manifest $INMANI --input_level sha256 --spot_check 1024` | GPU 7，**约 13 min**（4 文件并行 sha256）/ 32–35 min（串行） | `FINALIZE_EXIT_CODE=0` | **做** |
| 8 | framesamp 4×4 | `CUDA_VISIBLE_DEVICES='' pack_framesamp_store.py pack --source $LIB/source --manifest $MANI --out $LIB/framesamp --procs 48`；`verify --store $LIB/framesamp --resume --procs 48` | CPU 48 进程，冷缓存 **5–10 min** | `framesamp/`（78.3 GB），`VERIFY_PACK=PASS … mismatches=0` | **做** |
| 9 | norm_stats | `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir $V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da --config-name mme_vla_suite --repo-id robomme --dataset-path $LIB/source` | CPU，约 7 min | `train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json` | **做** |
| 10 | framesamp 8×8 | `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu pack_framesamp_store.py pack --layout framesamp-8x8-v1 --reader decode --source $LIB/source --manifest $MANI --out $LIB/framesamp-8x8 --procs 48`；`verify --store $LIB/framesamp-8x8 --resume --procs 48`；`report --store $LIB/framesamp-8x8`；`xgrid_pos_check.py --store-4x4 $LIB/framesamp --store-8x8 $LIB/framesamp-8x8 --source $LIB/source --image-spot 512` | CPU 48 进程，**8–15 min**，313.2 GB | `framesamp-8x8/`，`VERIFY_PACK=PASS`、`IMAGE_NPY_SPOT` / `MOTION_POS_XGRID` 判定行 | **做**（2026-09-14 用户拍板） |
| 11 | Wan 抽取 | `run_local.py --stage wan --lib $LIB --gpus … --raw-dir $RAW`（需 `v1_require_wan`） | GPU，按 400ep 每卡 1.43 s/窗推约 10 倍窗数 | `wan-latents/` | **先不做** |
| 12 | motion encoder | `run_local.py --stage encode --lib $LIB --gpus …` | GPU，分钟级 | `motion-tokens/` | **先不做** |
| 13 | motion 表 | `pack_motion_store.py pack --manifest $MANI --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion`；`verify --store $LIB/motion --resume` | CPU | `motion/` | **先不做** |
| 14 | motion oracle 对拍 | `oracle_driver.py --mj-repo $MJ vae … --shard-idx i --num-shards 8`（8 片）→ `aggregate --kind vae` → `encoder … --expected-ckpt-sha256 …`；`compare_wan.py latents` / `tokens`；`motion_checks.py a5/a6/a7/a9set/a10` 与 **`wan/extra_checks.py a8/a9enc`**；`dataloader_bench --lib $LIB --out <路径>` | GPU + CPU | `oracle/wan-mj/` | **先不做** |

> 阶段 9 的 `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu` 是**硬要求，不可省**：`compute_norm_stats.py` 调 `TorchDataLoader(…, sharding=None)` 且未传 `framework`，而 `TorchDataLoader.__init__` 的 `framework` 默认为 `"jax"`，于是 `src/openpi/training/data_loader.py` 里 `if sharding is None and framework == "jax"` 分支**必然执行** `jax.sharding.Mesh(jax.devices(), …)`，枚举并占用全部可见 GPU。GPU 0–3 每卡仅剩 6,033 MiB，训练峰值 68.5 GiB——不加隔离可能直接打死在跑的正式 run。初稿在阶段 10 写了这道防护、阶段 9 漏了。阶段 8 的 `pack_framesamp_store.py` 顶层明确不 import jax（源码注释：Pool 用 fork，jax 不允许进 fork 前进程），本身安全，此处一并加 `CUDA_VISIBLE_DEVICES=''` 只为统一写法。
>
> 阶段 14 修正：`a8` / `a9enc` 属 `scripts/dataset/wan/extra_checks.py`，`motion_checks.py` 只有 `a5/a6/a7/a9set/a10`；`dataloader_bench.py` 的 `--out` 是必填。

「先不做」的三点说明：

- **为什么现在不做**：用户本轮明确只处理数据、不处理 motion；且 GPU 0–3 被 MotionJEPA 正式训练占用，Wan 抽取按 400ep 实测（6,832 窗 1639 s / 6 卡）推到约 6.6 万窗，4 卡要 6 小时以上，不宜与训练抢卡。
- **将来补做不需要重建任何东西**：阶段 11–14 只吃 `meta/episode_manifest.json` 与 `$RAW` 下的合并 h5，都在本轮产出且不再变；`wan_common.task_of_h5` 认 `record_dataset_<Task>.h5`，段 key `<Task>_ep<0..399>_<seg>` 唯一。唯一要留意的是 `pack_motion_store.gather_provenance` 要求 Wan 与 encode 两阶段 `git_commit` 一致（记忆 `no-commit-between-wan-and-pack`），届时两阶段之间不要 commit。
- **训练侧本轮也不动**：新库用 `perceptual-framesamp-context.yaml`（`motion.enabled: false`）即可读，`dataloader._motion_gates` 在关闭态第一句就 `return None`、不碰 motion store。**接新 norm_stats 不需要改代码**：入口 `_config.cli()` 走 `tyro.extras.overridable_config_cli`，`TrainConfig.data: DataConfigFactory` → `DataConfigFactory.assets: AssetsConfig`，故启动时加 `--data.assets.assets-dir v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da --data.assets.asset-id robomme` 即可覆盖；**单改 `--dataset-path` 不会切换统计量**，若用 `mme_vla_suite_b128`（其 `assets_dir` 硬编码指向 400ep）会拿旧 norm_stats 跑新库、数值错但不崩，最易被误认为「跑通了」。训练步数等超参按 AGENTS.md 第 10 条另行请示（注意可训练样本是旧库的 **6.1 倍**，不是 9.7 倍）。

### 6. 资源核算

单位：GB = 10⁹，GiB = 2³⁰。起点为 `df --block-size=1G /scratch` 实测 3904 GiB 可用（4,191.6 GB）、3097 GiB 已用。

| 项目 | 估算 | 依据 |
|---|---:|---|
| 合并 h5（`$RAW`） | 790 GB（735.9 GiB） | MANIFEST `h5_bytes` 汇总 793.8 GB，实测合并体积 −0.45% |
| `source/` 特征 + pkl | **978.5 ± 10 GB（911 GiB）** | 拆两项算：`features/` 线性于**总**步数（1,192,918 × 602,951 B 恒定单价 → 729.4 GB）；`data/` 只线性于 **exec** 步数（617,463 × 395,586 B → 249.1 GB）。初稿按「107 GB × 9.7 倍总步数」推出 1.0 TB，结果误差仅 2% 但推导有两处反向抵消的错误（107 其实是 GiB；exec 比例按旧库 0.821 高估） |
| `framesamp/` 4×4 | 78.3 GB（72.9 GiB） | 1,192,918 × 65,536 B + pos + state |
| `framesamp-8x8/` | 313.2 GB（291.7 GiB） | 1,192,918 × 262,144 B + pos + state |
| 合计新增 | **2,163.9 GB（2,014.7 GiB）** | 上四项 |
| 建完剩余 | **2,022.9 GB（1,884 GiB）** | 已扣在跑训练的 ckpt 增量 4.8 GB。初稿写「余约 1.7 TB」是 GiB/GB 混用所致，实际偏保守 0.3 TB |
| 全流程墙钟 | **约 2.5 h** | 阶段 1–3 约 0.6 h + 阶段 4–10 约 1.9 h |

磁盘占用时间序列（峰值出现在阶段 10 结束，任何阶段都不击穿）：

| 阶段 | 峰值新增 | 累计新增 | `/scratch` 剩余 |
|---|---:|---:|---:|
| 2 合并（4 个 `.tmp`；`os.replace` 是 rename，不与正式文件共存） | +790 | 790 | 3,401.6 GB |
| 6 SigLIP | +978.5 | 1,768.5 | 2,423.1 GB |
| 8 framesamp 4×4 | +78.3 | 1,846.8 | 2,344.8 GB |
| 10 framesamp 8×8 | +313.2 | 2,160.0 | **2,031.6 GB** |

注：阶段 8 与 10 的 `pack_framesamp_store.py::cmd_pack` 自 commitV9.1 起磁盘预检下限是 `floor = max(40e9, ceil(1.25 × need))`，8×8 档 `need ≈ 313 GB` → 要求 free ≥ 约 391 GB；按本表执行时余量 2.0 TB，通过。

`extracted/`（793.8 GB / 739.3 GiB）**默认保留**（用户已拍板）：全流程最坏时刻仍余 2.0 TB，无需删。若将来要回收空间，阶段 4–10 全部只读 `$RAW`，**阶段 3 `MERGE_VERIFY=PASS` 后即可删**，不必等阶段 7；`snapshot/` 的 tar.xz 仍是原件，可随时重解。

### 7. 已拍板

1. **数据来源**：确认有意选用私有仓数据集 `HongzeFu/robomme-4task-h5-20260912-v2`（非 AGENTS.md 第 15 条默认的 `Yinpei/robomme_data_h5`）。评估侧将改用该私有仓对应的 robomme sim，**该 sim 尚未实现**——在它实现之前，新库训练出的模型**不能用当前官方 submodule 评估**（详见「已知但本轮不做」）。
2. **库名**：`4task-v2-1600ep-604f16da`（带 revision 前缀，因数据源为私有仓且将来可能有 v3）。对应 `v1-store/datasets/4task-v2-1600ep-604f16da/` 与 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/`。
3. **episode 范围**：全部 1600 条 primary、全部难度；**`spare` 196 条与 `smoke` 1 条不收**，由第 4 节 (a) 的四重硬保证 + `extracted/` 未解开这一天然屏障共同保证。
4. **`extracted/` 保留**，不删。
5. **GPU 范围**：只用 4–7；阶段 9 必须加 `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`。
6. **`framesamp-8x8` 本轮一并建**，阶段 10 与阶段 8 同为正式交付件。
7. **verify 并行单元**改 `(task, episode 区间)`、`--procs 48`（154 min → 7.5 min，强度不变）。
8. **阶段 7 保持 `--input_level sha256`**，但 4 文件并行算（35 min → 13 min），守卫强度不降。
9. **不等在跑训练结束**，正常推进；保证不中断、不严重降速，判据 `samples_per_s` 基线 874.0 ± 0.3、> 1% 下滑即暂停。

---

## 第二部分（技术细节，供 agent 追踪）

### 新脚本 `scripts/dataset/merge_v2_h5.py`

- 位置与依赖：放 `scripts/dataset/`，只依赖主 venv 已有的 `h5py`（3.13.0 / libhdf5 1.14.6）与 `numpy`（1.26.4）；路径解析沿用 `paths.sh` 的 `V1_STORE`，不新增外部目录。
- 子命令与关键函数：
  - `plan`：`load_manifest(snapshot, control) -> dict`（校验 `MANIFEST.json` 的 sha256 == `source.json` 的 `manifest_sha256`）；`select_primary(manifest, tasks, per_group=None) -> dict[task, list[row]]`（`role == "primary"`；排序键 `(DIFFICULTY_ORDER[difficulty], episode)`，`DIFFICULTY_ORDER = {"easy":0,"medium":1,"hard":2,"xhard":3}`，用 `.get()` + 显式 `SystemExit(f"未知难度 {d}")` 兜底）；`precheck_sources(rows, extracted)`；`write_episode_map(out, task, rows)`；`write_source_pin(out, source_json)`。
  - `merge`：`merge_task(task, map_path, extracted, out, check_sha, resume)`——对每行 `with h5py.File(src,"r") as s, h5py.File(tmp,"a") as d: s.copy(row["src_group"], d, name=f"episode_{row['new_idx']}")`；每条打印 `MERGE_EP task=<T> idx=<i> member=<…> secs=<…>` 并 flush；续跑时先打印 `MERGE_RESUME task=<T> 已有=<n>`。
  - `verify`：`verify_shard(task, lo, hi, map_path, extracted, out, level)`——按 episode 区间分片；`ds_equal` 见第 4 节 (c)；三条结构守卫；`level=sample` 只比 `setup`、`timestep_0`、最后一步。全部通过写 `MERGE_DONE.json`（含 4 个合并文件 sha256、逐任务 episode 与 timestep 总数、role 分布、`source_sha256_checked`、`source_pin`、map 摘要、选取参数、排序规则版本、`git_commit`），否则 `MERGE_VERIFY=FAIL` 并列出前 20 条不符。
- 性能与并行：单流 `Group.copy` 实测 243 MB/s（与初稿 242 一致），瓶颈是 HDF5 对象创建而非带宽（RAID 裸读 4.4 GB/s）。sha256 单核 430–450 MB/s（Xeon 8275CL 无 SHA-NI，只有 avx512），故 sha256 完全并行化后 794 GB 约 2–13 分钟。verify 冷缓存实测 1166 dataset/s、每 dataset 858 μs，其中双侧 `visititems` 收集占 32%——可只在每任务第一条 episode 上收集一次名字集合并按 `timestep_count` 生成后续，逐条仍断言 `set(src) == set(dst)`。
- 规模参考（供留档）：全库 dataset 数 = 1,192,918 × 21 + 1600 × 6 = **25,060,878**；HDF5 对象数 = 1,192,918 × 25 + 1600 × 8 = **29,835,750**；元数据单价实测 453 B/对象，BinFill 单文件元数据约 5.8 GB。合并后单文件最大 341 GB，无格式层限制（libhdf5 1.14.6 / 64 位 offset / XFS；同盘已有 74.9 GB 单 h5 跑通过旧库建库）。打开与列 key 是微秒—毫秒级，`scan_manifest` 那一跳仅 4–8 秒。

### 既有链路零改动的核对点

- `scripts/dataset/scan_manifest.py`：`canonical_h5_order(raw_dir, tasks)` 拼 `record_dataset_{t}.h5` 后查存在性，缺一即 `SystemExit`；**末行是 `return sorted(want)`，即字典序、不是 `--tasks` 顺序**——本库实际 `canonical_order` 为 `[BinFill, RouteStick, VideoRepick, VideoUnmaskSwap]`（初稿写成 `[…, VideoUnmaskSwap, VideoRepick]`，会让 global episode 区间推断整体错位，留档里不要预写、以实际清单为准）。`scan` 对每文件取 `episode_*` 并按 `int(k.split("_")[1])` **数值排序**；`--episodes-per-task 400` 时要求每文件 ≥ 400 条，否则 `SystemExit`。
- `scripts/dataset/build_shard.py` worker 模式：`handles` 按 `h5_file` 缓存句柄，合并后每 worker 最多持 4 个。`--num_shards 1` **不限制 GPU 并行度**：worker-mode 下 `args.shard_idx, args.num_shards = args.worker_idx, args.num_workers` 被覆写成 worker 数，工作项粒度是 episode、走 `_claim_path` 的 `O_CREAT|O_EXCL` 动态领单；且 `4 != manifest["num_shards"]` 时不触发 `select_episodes` 的同源断言。400ep 正式库即以 `--num_shards 1` 配 3 卡建成（`STAGE_DONE stage=siglip workers=3 items=400 elapsed=551s`）。
- 单 episode 内存峰值：按最长 **2304** 步算，`visualization_videos` 453 MB + `record_videos` 906 MB + `mem_buffer._history_feats` 1.84 GB ≈ **3.2 GB/worker**，4 worker 约 12.8 GB（初稿写「1636 步 ≈ 640 MB」，既用错了最长步数、也漏了这三个无条件累积的整集缓冲）。本机 1121 GB 内存，无压力。
- `scripts/dataset/finalize_checks.py`：`cmd_hash_inputs` 只扫 `raw_dir` 顶层 `*.h5`（`os.listdir` 不递归）；`spot_check` 走 `data[f"episode_{raw_ep_idx}"]`，合并后成立；`aggregate_shard_fingerprints` 的 `FINGERPRINT_SAME_KEYS` 含 `git_commit`，故**阶段 6 起跑到阶段 7 通过之间不得 commit**（中断后 commit 再 `--resume` 会 fail-loud，与 `no-commit-between-wan-and-pack` 同类机制）。
- `src/mme_vla_suite/dataset_builder/build_robomme_dataset.py::_process_episode`：`assert ts["info"]["is_video_demo"][()] == (step_idx < exec_start_idx)` 要求 demo 段是前缀——**已全量核实 1600/1600 通过**（不再依赖 smoke 抽样）。`is_completed` 首帧为真会让四个 subgoal 局部变量未定义而抛错——**已全量核实 0/1600，不会触发**。但 `is_completed` 的**后缀段**（每集末尾 6–71 步，合计 **37,219 步 = 3.12%**，与 demo 段零重叠、全部是会落 pkl 的执行步）会静默沿用上一帧 subgoal，属既有语义、本轮不改，写进 result.md 备查。
- `scripts/dataset/wan/wan_common.py::task_of_h5`、`src/mme_vla_suite/datastore/motion_store.py::segment_key`：合并文件名合规，将来阶段 11–14 可直接跑；`read_segment_frames` 对 `obs/front_rgb` 显式校验 `(256,256,3) uint8`，实测 1600/1600 通过。

### 第 18 条链路图与一致性两块的适用说明

本轮**不改任何训练链路代码**，只在链路最前面加一跳：

```
extracted/record_dataset_<Task>_<难度>/hdf5_files/<Task>_ep<N>_seed<S>.h5
        [恰 1 个顶层组，组名 episode_<原 ep 号>；(256,256,3) uint8 等；无 attrs / 无压缩 / 全 contiguous；原件]
        │  merge_v2_h5.py merge（H5Ocopy 对象拷贝；不改数、不改 dtype、不改 chunk；实测 17 类属性全保留）
        ▼
$RAW/record_dataset_<Task>.h5  [episode_0..399；逐 dataset 值逐位同源 + 三条结构守卫，verify --level full 判定]
        │  以下与 4task-motion-400ep 完全相同：scan_manifest → hash-inputs → build_shard(SigLIP) → finalize → pack_framesamp → norm_stats
        ▼
$LIB/source → $LIB/framesamp、$LIB/framesamp-8x8 → train-assets/.../norm_stats.json
```

- **第一块（非训练轻量对拍）**：即阶段 3 的 `verify --level full`，判据是全部 1600 条的全部 dataset **数值逐位相等（NaN 按相等、±0 按相等）+ 链接类型 / attrs / 名字集合三条结构守卫通过**，`mismatches=0`。
- **第二块（训练梯度一致）**：**不适用**。逐字读 AGENTS.md 第 18 条首句「每次针对训练链路的**修复或重构**」——括号里的枚举是在限定「什么样的修复或重构算数」，换数据集不是对链路的修复或重构，本条字面不适用；且不存在「旧链路交付」可作对照。
- **关于链路代码改动的证据措辞**（初稿的说法会被自己的证据推翻）：`git diff 8093ebd..HEAD -- scripts/dataset/ src/mme_vla_suite/dataset_builder/` **不是空的**（`8093ebd` = commitV6.12，400ep 的 SigLIP/finalize/pack 起跑 commit，取自其 launch.md）。实测结论是：`src/mme_vla_suite/dataset_builder/` **零改动**；`scripts/dataset/` 下落在本轮链路上的只有 `pack_framesamp_store.py`（+87/−35），改动性质是把模块级常量换成 `spec.*` 字段查表的**布局参数化**，`--layout` 默认仍为 `framesamp-4x4-v1` 且 `fs.SPECS[LAYOUT]` 的字段值就是那批旧常量，**4×4 路径输出字节不变**；唯二行为改动是磁盘预检下限（`40e9` → `max(40e9, ceil(1.25×need))`）与 `--resume` 时的 layout 不符即 raise。该改动已在 `docs/dataset-build-doc/4task-motion-400ep-framesamp-8x8/` 于同一 source 上验证通过。留档写这一段，不写「零改动」。

### 验证（每步 ≤ 5 分钟，改动后必跑）

1. **脚本自测**：`scripts/dataset/test_guards.py` 现有用例照跑；给 `merge_v2_h5.py` 加最小用例——用 `h5py` 在临时目录造 2 任务 × 3 条假 episode（各 2 步、含 NaN 浮点、含 bytes 与 object 数组键、**组名故意不叫 `episode_0`**）跑 `plan → merge → verify --level full`，断言 `MERGE_VERIFY=PASS`；再逐一注入七类篡改（uint8 改 1 字节、float32 改 1e-7、塞 NaN、bool 翻转、bytes 追加 1 字符、object 数组改 1 元素、int64 +1），断言每类都 `FAIL`；另造一条含 attrs / 软链接的假 episode，断言结构守卫报错。
2. **真实 smoke**：`plan --per-group 2`（**不是 1**——`--per-group 1` 取每组首文件恰好命中 13 个 `episode_0` 特例、几乎放过组名问题；`--per-group 2` 保证每组必含一条非 `ep0`）得 28 条 → `merge` 到 `v1-store/raw-h5/4task-20260912-v2-smoke/`（约 20 GB）→ `verify --level full` → 阶段 4–8 与阶段 10 在 `v1-store/datasets/4task-v2-smoke28/` 上跑通。**smoke 的 `scan_manifest` 命令必须省略 `--episodes-per-task 400`**（每文件只有 6–8 条，带该参数会直接 `SystemExit`）。判定 `FINALIZE_EXIT_CODE=0`、`VERIFY_PACK=PASS`。用 smoke 的合并墙钟外推正式合并耗时。验收完成后删除 smoke 的两个目录（第 6 条临时 run 清理）。
3. **训练可读性**（正式库建完后）：`perceptual-framesamp-context.yaml` 起 `--dataset-path $LIB/framesamp` 跑 20 步，**必须同时显式带 `--data.assets.assets-dir v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da --data.assets.asset-id robomme`**，否则用的是旧 norm_stats；确认 dataloader 出 batch、`motion_*` 键为 None，跑完删 run。若要验 8×8 库须改用 `perceptual-framesamp-context-8frame-8x8.yaml`（前者 `token_per_image: 16` 是 4×4 档）。

### 留档与 commit

- 代码：`commitV9.3: 新增新版单 episode H5 合并预处理脚本 merge_v2_h5.py`（只 `git add scripts/dataset/merge_v2_h5.py` 与其测试），commit 后立即 `git push`。当前全历史最大版本号为 `commitV9.2`，V9.3 是正确的下一号。
- **数据获取留档**（AGENTS.md 第 12、15 条，本轮补建）：`docs/dataset-build-doc/h5-acquire-4task-20260912-v2/{launch.md,result.md,records/}`，把 `control/source.json`、`COMPLETE.json`、`final_verification.json`、README 溯源段与条数分布表收进 `records/`，正文写明：用户已确认有意选用该私有仓数据集、它与 `Yinpei/robomme_data_h5` 不同源、BinFill demo 段差异、以及评估侧待切换的状态。
- 建库留档：`docs/dataset-build-doc/4task-v2-1600ep-604f16da/{launch.md,result.md,records/}`——launch 记起跑 HEAD、本节命令原文、`source_pin.json`、本轮 tmux 会话名清单、以及「任务 → 难度 → `raw_ep_idx` 区间」14 行表（**难度信息只存在于 `*_episode_map.json`，`episode_manifest.json` 没有 difficulty 字段**）；result 记 `PLAN_OK` / `MERGE_VERIFY` / `STAGE_DONE` / `FINALIZE_EXIT_CODE` / `VERIFY_PACK`（4×4 与 8×8 各一行）/ `IMAGE_NPY_SPOT` / `MOTION_POS_XGRID` 判定行原文、各阶段墙钟与 GPU、产物体积、实际 `canonical_order`、训练吞吐基线对照（建库期间 `samples_per_s` 是否守住 874.0 ± 0.3）、以及上面第 18 条那段 diff 结论。records 复制清单：`input_manifest.json`、`framesamp.store_meta.json`（两档各一）、norm_stats sha256、**4 份 `*_episode_map.json` + `source_pin.json` + `MERGE_DONE.json`**（合计几 MB，含难度映射与 role 分布）。每份 json 在 result.md 表里标注库名与 manifest sha256 前缀，避免与旧库同名 key 混淆。`docs/dataset-build-doc/README.md` 表加一行。提交为 `docs: 4task-v2-1600ep-604f16da 建库留档`。
- 本轮起过的 tmux 会话名（`v2b-merge`、`v2b-verify`、`v2b-hash`、`v2b-siglip`、`v2b-finalize`、`v2b-pack4x4`、`v2b-pack8x8`）记入 launch.md，清理只按此清单逐个 `tmux kill-session -t <名>`，删前删后各跑一次 `tmux ls` 对账。**严禁 `tmux kill-server` 及一切全局杀法。**

### 明确不动的文件

`scripts/dataset/{scan_manifest,build_shard,finalize_checks,pack_framesamp_store,run_local,pack_motion_store,motion_checks,xgrid_pos_check}.py`、`scripts/dataset/wan/*`、`scripts/dataset/paths.sh`（`TARGET_TASKS` 保留旧四任务，新任务集只走 `--tasks`；阶段 0 不调 `v1_validate_raw_h5`）、`scripts/training/compute_norm_stats.py`、`src/mme_vla_suite/dataset_builder/*`、`src/mme_vla_suite/datastore/*`、`src/mme_vla_suite/training/*`（**新 norm_stats 走 CLI 覆盖，不需要新增 `AssetsConfig` / `TrainConfig` 条目**）、`third_party/robomme_benchmark`。

### 已知但本轮不做的关联项

- **评估环境与训练数据不同源（最重要）**：本仓库评估链路跑在官方版 submodule `third_party/robomme_benchmark`（commit `856bc3a1`）上，而新数据由私有仓 `hongzefu/robomme_benchmark_MotionJEPA@newtask-v2` 生成，两者对同一任务的定义已不一致（最明显是 BinFill 的 demo 段：官方集 `es=0` 无 demo，新数据每集前半段都是 demo）。**用户已确认评估侧将改用该私有仓对应的 robomme sim，但该 sim 尚未实现。** 在它实现并接入之前，用新库训练出的模型**不具备可信的评估路径**，不要用当前 submodule 出评估结论。下列三项都属于那次工作：
  - `TASK_WITH_VIDEO_DEMO` 表**共有四份副本**（`examples/robomme/utils.py`、`scripts/training/legacy-eval/robomme-{local,remote}/utils.py`、`scripts/motion-variance/robomme/utils.py`，另有 `scripts/dataset/scan_16task_memory.py` 一份），均不含 BinFill；而训练侧 `_process_episode` 会把 BinFill 的 demo 帧标 `is_demo=True` 且不产 exec 样本——同一任务在训练与推理两侧对「哪些帧是 demo」的判定相反，并会连带影响 memory 的 demo/exec 分段与（将来开 motion 时）`grid_origin: segment_start` 的段边界。
  - val/test seed 划分口径整体失效：新数据每条 episode 带生成侧给定的 `seed`，且每任务 400 条分布在 4 档难度上，旧的 `VAL/TEST_SEED_DISJOINT` 闸门不适用；三个新任务（BinFill / RouteStick / VideoRepick）在评估侧从未建立过 seed 划分。需基于 `*_episode_map.json` 的 `seed` 列重建。
  - `scripts/training/legacy-eval/eval_all_shards.local.sh`、`check_test_seeds.py`、`plot_eval_success_by_length.py` 写死旧四任务与旧 h5 目录；`scripts/training/paths.sh::EXPECTED_H5` 同（无消费方）。
- **`perceptual-framesamp-context.yaml` 的 `motion.store_path` 硬编码指向 `v1-store/datasets/4task-motion-40ep/motion`**（40ep 测试库，不是 400ep）。`motion.enabled: false` 时无害，将来给新库开 motion 必须改，或用 `MMEVLA_MOTION_STORE` 环境变量覆盖（`_motion_gates` 支持）。
- **`setup/task_goal` 是多变体数组**：元素数分布为 2 元 934 集、1 元 400 集、3 元 266 集，而 `get_task_goal` / `_process_episode` 永远只取 `[0]`，其余措辞在训练 prompt 集合里完全不出现。这与未决的 `EVAL_PROMPT=FAIL`（测试集 goal 不在训练 prompt 集合里）同源，且新库 1200/1600 集都带备选措辞。建议 `plan` 顺手把每任务的 `task_goal` 全变体集合导出到留档供评估侧比对（不改链路）。
- **`scripts/training/compute_results.py` 的 suite 归属含义变了**：`TASK_NAME_LIST` 16 项已含全部四个新任务，无需改；但四个新任务**横跨四个 suite**（BinFill→Counting、VideoRepick→Referential、RouteStick→Behavior、VideoUnmaskSwap→Persistent），而旧四任务全在 Persistent 一组内，suite 级汇总的解读口径要相应调整。
- `src/mme_vla_suite/dataset_builder/build_vlm_subgoal_dataset_memer.py` 有任务名分支含 RouteStick（另一条链路，本轮不涉及）；同文件另一处 `if env_id in "ButtonUnmaskSwap":` 是把 `in` 用在字符串上的可疑写法，任何子串都会为真，记录备查。
- 将来若把新库导出到 HF bucket（约 2.2 TB，远超 400ep 的 130 GB / 22.9 万文件），分批 commit 纪律加倍适用（记忆 `hf-bucket-batch-commit-limit`）。
