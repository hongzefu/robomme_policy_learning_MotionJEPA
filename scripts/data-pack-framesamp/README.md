# data-pack-framesamp：framesamp packed 库打包 / 校验 / 守卫

对应 `v2-framesamp-restructure-plan.md` A.2（打包工具）与 C.5（守卫测试）。格式常量、
行号公式、`StoreMeta`、读 API 一律 import 自 `src/mme_vla_suite/datastore/`，本目录
不复制任何格式定义。

| 文件 | 作用 |
|---|---|
| `pack_framesamp_store.py` | 子命令 `plan`（只算贪心切分）/ `pack`（三张表构建：写侧逐帧校验① pos memcmp 钉 t 不钉 g、② state 同源自证、③ slab read-after-write，part sha256 原子落盘，`meta/pack.lock` 排他锁 + `pack_progress.jsonl` 断点续跑）/ `verify`（**全量写×读对拍，g 级零遗漏唯一凭据**：重新 decode 全部源帧，三键各经真实读 API 逐行 memcmp，产出 `row_digests.blake2b.bin`，通过后 meta 回填 `status="verified"` 并释放锁；FAIL 保留锁阻断读侧）/ `report` |
| `run_pack.sh` | S4 全量打包的 tmux 驱动（pack → verify 串联，tee + `EXIT_CODE=`；判定行 `VERIFY_PACK=PASS scanned=483291 mismatches=0`） |
| `probe_layout.py` | 源 npy 布局常量探针（st_size 602,951；三键偏移 262,595 / 541,352 / 602,906）——`--reader slice` 加速档的前提复核，任一不符即非零退出 |
| `test_pack_guards.py` | 守卫测试：Store 组 G1/G4/G5/G7/G11/G12/G14（S2）+ Dataset 组 G2/G3/G6a/G8/G9/G10/G13 与 backend 分派闸（S3）；迷你库 = ref-shard 派生连续前缀 [0..2]，session fixture 打包一次含全量 verify。fork Pool 用例在前、import jax 的用例（Dataset 组、G7）懒加载在后——jax 不进 fork 前进程 |
| `spawn_matrix.py` | S3 判定工具：FrameSampDataset 上起真实 torch spawn DataLoader，w0/w1/w4/w16 × 2 epoch，fd 泄漏检查（先预热 spawn 基础设施再取基线——resource_tracker 单例是进程级一次性 fd、非泄漏）。判定行 `MATRIX=PASS` |

## 常用命令

```bash
# 布局探针（slice 档前提复核）
UV_LINK_MODE=copy uv run python scripts/data-pack-framesamp/probe_layout.py \
  --source v1-store/datasets/4task-gl

# 切分预览（不写盘）
UV_LINK_MODE=copy uv run python scripts/data-pack-framesamp/pack_framesamp_store.py plan \
  --manifest v1-store/episode_manifest.json

# S4 全量打包 + 全量校验（detached tmux，AGENTS 7）
tmux new-session -d -s pack-framesamp "bash scripts/data-pack-framesamp/run_pack.sh"

# Store 组守卫
UV_LINK_MODE=copy uv run pytest scripts/data-pack-framesamp/test_pack_guards.py -x -q
```

## 协议要点（详见计划 A.1/A.2）

- **锁**：`meta/pack.lock` `O_CREAT|O_EXCL`；同 host 存活进程拒跑；残锁 `--resume`
  接管；异 host 一律拒跑（`--force-break-lock` 打印锁全文 + 交互确认）；锁在全量
  verify 回填 meta 后才释放，读侧分派层见锁即 raise。
- **续跑**：`--resume` 按「存在 + 大小 + sha256」跳过完好 part，`.tmp` 残留一律清除
  重做；progress 尾部半行写侧 ftruncate、读侧丢弃。
- **源读取档**：`--reader decode`（首跑默认，零布局假设）/ `slice`（按偏移常量
  pread，三重守卫：st_size、前 64 B 参考前缀、写侧 pos 100% memcmp；参考前缀由首帧
  decode 互证钉死）。
- **抽样档** `verify --sample N` 仅供开发期快检（10% 抽样对单行错位漏检率约 90%），
  不得用于交付判定；不回填 meta、不产 row_digests。
- **pos 旁证生成**（`generate_pos_table_posemb3d`）：CPU 后端一律拒绝（实测与库中值
  max|diff|≈7e-7，G7 钉死）；主方案是源库抽取拼装，旁证生成体未实装。
- **迷你库**（`--subset-prefix K`）：只允许 global_episode_idx 连续前缀 [0..K]，
  前缀内必须含 ≥33 帧 episode；`manifest_scope="subset"` 的库禁止用于 S5 及以上判据
  （packed 分派检出即 raise；开发期须显式设 `MMEVLA_FRAMESAMP_ALLOW_SUBSET=1` 放行且
  必打 WARNING——该开关为 S3 实施时新增，与 `ALLOW_UNVERIFIED` 同族，判据 run 里出现
  即 run 无效）。
