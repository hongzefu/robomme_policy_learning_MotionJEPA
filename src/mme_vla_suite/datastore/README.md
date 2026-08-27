# datastore：framesamp packed 特征库格式层

对应 `v2-framesamp-restructure-plan.md`（A.1 布局契约、B.2 Store 行为）。本包是格式层
**唯一实现**：打包工具（`scripts/data-pack-framesamp/`）、`FrameSampDataset`、对拍工具
一律从这里 import，绝不复制；本包不 import 任何 training/model 模块（单向依赖，B.1）。

| 文件 | 内容 |
|---|---|
| `manifest.py` | `load_manifest`（sha256 fail-loud）与 `manifest_sha256`（与建库侧 `scan_manifest.py` 同口径的独立实现——scripts 目录名含连字符无法跨目录 import） |
| `framesamp_store.py` | 布局常量（`framesamp-4x4-v1`、三张表形制、源 npy 偏移常量）、`row_of()` 行号公式（写读共用）、`build_exec_lookup()`（执行样本 idx → (g,t) 查表数组，必须带 `exec_start_idx`）、`StoreMeta`（store_meta.json 结构校验）、`run_fast_checks` / `run_full_checks` 两档校验、`require_verified` / `require_no_pack_lock`（packed 分派层守卫）、只读 `FrameSampStore` |

## 布局速览

```
<store 根>/
├── meta/store_meta.json        唯一契约（两阶段写：pack→"packed"、verify→"verified"）
├── meta/pack_progress.jsonl    断点续跑记录（打包父进程单写）
├── meta/row_digests.blake2b.bin  逐行 blake2b-128（verify 时产出，image‖pos‖state）
├── image_emb_4x4/part_XXX.bf16.bin  (rows,16,2048) bf16 裸字节，按 episode 边界切分
├── pos_emb_4x4.f32.bin         (num_pos_rows,16,768) f32（pos 是 t 的纯函数，只存一份）
└── state_emb.f32.bin           (num_rows,8) f32
```

行号 = `row_of(total_sample_offset[g], t)`，t 为全 timestep 域帧号（含 demo 前缀）。
禁 `.npy` 容器：`np.save` 对 ml_dtypes bf16 写 `V2` descr、`np.load` 丢类型（A.1 定论），
一律裸 `.bin` + meta 声明 dtype。

## FrameSampStore 契约（B.2 摘要）

- 构造即打开全部 part fd（`O_RDONLY|O_CLOEXEC`）+ 两张小表 `np.fromfile` 整表读入
  （进程内副本、非映射——热路径不留任何 NFS mmap），记录 `owner_pid`；
- `read_image_rows`：先对全部行发 `posix_fadvise(WILLNEED)`（不可用则打一次 WARNING
  后永久跳过），再按连续行游程合并 `os.preadv` 直读进预分配数组；短读循环补齐，
  EOF/越界立即 raise；
- **禁止 pickle**（`__reduce__` 直接 raise）：跨进程懒构造与 pid 校验由
  `FrameSampDataset` 负责（S3）；
- 不看 `meta.status` / `pack.lock`——那是 `create_data_loader` 分派层的闸
  （`require_verified` / `require_no_pack_lock`，S3 接线）；verify 子命令持锁期间
  仍需经 Store 读库；
- fast 档（每进程懒构造时）：layout / 清单指纹重算 / parts 连续覆盖 / 逐 part st_size /
  抽 1 part 首尾 1 MiB digest / 抽 1 条源库抽样指纹；full 档（全部 sha256，能抓同尺寸
  中部翻转）**禁止在性能 allocation 内执行**（会预热 page cache，B.2）。

守卫测试：`scripts/data-pack-framesamp/test_pack_guards.py`（Store 组 G1/G4/G5/G7/
G11/G12/G14 + Dataset 组 G2/G3/G6a/G8/G9/G10/G13 与分派闸）；spawn 生命周期验收另见
`scripts/data-pack-framesamp/spawn_matrix.py`。消费侧装配层见
`src/mme_vla_suite/training/framesamp_dataset.py`，backend 三态分派（`MMEVLA_DATA_BACKEND`，
未设默认 legacy）见 `training/dataloader.py` 的 `_resolve_backend` / `_create_framesamp_dataset`。
