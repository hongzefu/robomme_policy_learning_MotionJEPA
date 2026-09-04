# 外部模型资产锁：五个权重的身份钉死与链路接入（`ASSETS=PASS`，全量校验 10.2 秒）

本轮把此前散落在三处 markdown 与一个无人读取的 txt 里的「权重真锚点」收敛成**一张进 git 的表**
（`scripts/assets/ASSETS_LOCK.json`），把 MotionJEPA 的 encoder + decoder 传上 HuggingFace private
model repo，并让链路**六处**真正读这张表。终判：`ASSETS=PASS assets=6 mismatches=0`（full 档 10.2 s）、
`HF_META=PASS files=6 lfs_mismatch=0 blob_mismatch=0`、回读 `RESULT=PASS`、`BASELINE_ENV=PASS`（旧基线未破）。
五个 commit：`1ebab97` / `f07ee19` / `c7406e4` / `ee0ae46` / `1b75d63`。

本文是**高层导读**。逐条判定行、实测耗时、records 快照与完整盲区清单在
[`docs/dataset-build-doc/hf-export-motionjepa-encoder-v1/result.md`](docs/dataset-build-doc/hf-export-motionjepa-encoder-v1/result.md)；
模型本身的一切事实（结构、训练 commit、数据集、超参、数值合同、加载示例）在同目录
[`model-card.md`](docs/dataset-build-doc/hf-export-motionjepa-encoder-v1/model-card.md)。本文不复述这两份。

## 一、为什么要锁

链路依赖五个仓库外的大二进制：SigLIP、PaliGemma tokenizer、pi05_base、Wan2.1 VAE、MotionJEPA
encoder+decoder ckpt。本轮开工前盘点，它们的身份保证几乎为零：

1. **只查「文件在不在」**。两个 `paths.sh` 的 `v1_require_models` / `v1_require_wan` 只做 `[[ -f ]]`；
   SigLIP 更是全链路零内容校验——唯一加载点 `src/mme_vla_suite/dataset_builder/siglip_tokenizer.py`
   直接 `pickle.load`，训练侧的环境指纹里有 norm_stats / tokenizer / pi05_base / dataset_spot，唯独漏了它。
2. **真锚点不在代码里**。ckpt 与 config 的期望 sha256 只写在
   `docs/dataset-build-doc/4task-motion-40ep/launch.md` 的命令行里和
   `v1-store/external/motionjepa/wan-v8-filter10-72ep-a/SHA256SUMS.src-vs-copy.txt`（**没有任何代码读它**）；
   tokenizer 与 pi05_base 的值只活在 `docs/training-doc/v1-grad-baseline-g0b/records/r1/env.json` 的基线自比对里。
3. **自证循环**。`scripts/dataset/run_local.py` 在不传 `--expected-ckpt-sha256` 时，会**现场哈希那份即将被
   使用的 ckpt**，再把结果当「期望值」发给各 GPU worker——它只能证明多卡用同一份字节，完全挡不住
   「这份文件本身就是错的」。`motion_sidecar.py` 那侧同理：VAE 有兜底硬钉，ckpt 却 `or None` 静默跳过。

## 二、锁长什么样

`scripts/assets/ASSETS_LOCK.json`（11,778 B，进 git）六条记录，每条记**落点 + 指纹 + 来源**：

| 资产 | 落点（仓库根相对） | 指纹 | 来源钉法 |
|---|---|---|---|
| `siglip_params` | `v1-store/models/pi05_vision_encoder/siglip_params.pkl` | `f16e9312…` | HF `Yinpei/pi05_vision_encoder`@`59bd9ff4…`（公开） |
| `paligemma_tokenizer` | `v1-store/models/big_vision/paligemma_tokenizer.model` | `8986bb4f…` | `gs://big_vision/paligemma_tokenizer.model`（anon） |
| `pi05_base` | `v1-store/models/openpi-assets/checkpoints/pi05_base/params` | 20 个文件逐个 sha256 | `gs://openpi-assets/checkpoints/pi05_base` |
| `wan_vae` | `v1-store/cache/hf/hub/models--Wan-AI--…` | blob `d6e524b3…` | HF `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`@`0fad780a…`，`vae/*` |
| `motionjepa_ckpt` | `v1-store/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt` | `bae96037…` | HF `HongzeFu/MotionJEPA`@`749224110f…`（**私有**） |
| `motionjepa_config` | 同目录 `config.yaml` | `99548a6c…` | 同上 |

四条设计点：

- **表自己防篡改**：顶层 `sha256` 是剔掉该键后 canonical JSON 的哈希，改表里任一个值而不改它，
  `load_lock()` 当场 fail-loud。口径复用既有的 `wan_common.manifest_sha256`，与
  `src/mme_vla_suite/datastore/manifest.py` 同款，三处实现由测试断言同值——不是新造的轮子。
- **两个档位**：`cheap` = 字节数 + 首尾各 1 MiB 的 blake2b（实测 **0.14 s**，放进每次起跑的前置）；
  `full` = 逐文件全量 sha256（六条合计 **10.2 s**，其中 14 GB 的 pi05_base 占 8.16 s）。
  cheap 挡不住「保持长度改中间某个字节」，这一点被测试钉成显式契约，不是含糊带过。
- **`revision` 必须是 40 位 commit sha，禁 `main`**：私有 repo 自己能 push，`main` 会漂；有正则测试卡死。
- **与 `scripts/dataset/wan/SOURCE_PIN.json` 并存不合并**：后者钉「仓库内那份 `.py` 复制件的字节」
  （源码同源性），前者钉「仓库外大二进制的身份」（资产可获取性），两者用一条交叉断言绑住。

配套 `scripts/assets/fetch_assets.py` 四个子命令 `plan` / `fetch` / `verify` / `show`，末行统一
`ASSETS=PASS|FAIL`；守卫测试 `scripts/assets/test_assets_lock.py` 共 28 条，0.14 秒跑完。

## 三、链路六处接入（`c7406e4`）

| 文件 | 改前 | 改后 |
|---|---|---|
| `scripts/dataset/paths.sh`、`scripts/training/paths.sh` 的 `v1_require_models` | 只 `[[ -f ]]` | 保留存在性检查，追加 `fetch_assets.py verify --level cheap`；两份保持逐字相同（有测试盯） |
| 同上 `v1_require_wan` | 只查目录存在 | 追加 `wan_vae,motionjepa_ckpt,motionjepa_config` 的 cheap 校验 |
| `scripts/dataset/run_local.py` | 现场哈希被测 ckpt 当期望值 | 改为 `al.expected_sha256("motionjepa_ckpt")`；`main()` 起手按 stage 取资产（本域此前**零调用**任何前置） |
| `scripts/dataset/wan/motion_sidecar.py` | ckpt `or None`（静默跳过） | 补 lock 兜底，并加交叉断言：调用方传的数据集自报 provenance 与仓库锁不符即拒，报「在线推理与离线建库不同源」 |
| `probe_wan.py` / `oracle_driver.py` / `extra_checks.py` | 同样 `or None` | 默认按 lock 校，跳过须显式 `--expected-ckpt-sha256 SKIP`（探针本职就是探未入 lock 的 run_dir；生产路径不给 SKIP） |
| `scripts/training/tests/run_t3_eval_obs.sh` | source 了 `paths.sh` 却没调前置 | 补 `v1_require_models 1` |

**刻意没碰的四处**：`check_baseline_env.py`（`_diff` 是键并集深比，加一个字段会让 24 份历史 `env.json`
全 FAIL，且 `env.json` 自己就在受保护产物清单里）、`siglip_tokenizer.py`（被冻结哨兵钉死在 `72fb8423…`）、
`wan_motion_infer.py`（逐字节复制件，改一字节 `load_source_pin` 就退出）、`encode_motion.py`
（它的 `--expected-ckpt-sha256` 本就是 `required=True`）。

应急逃生阀 `V1_SKIP_ASSET_VERIFY=1` 默认关、跳过时打醒目警告；`run_local.py` 那处**故意不给逃生阀**
——那正是要堵的洞，留阀等于没堵，换 ckpt 就改 lock。

## 四、MotionJEPA 权重上传（`1ebab97`）

传的是 run `wan-v8-filter10-72ep-a` 的 **encoder + wan_decoder**——它们本来就是同一个
`checkpoint_epoch_72.pt` 里的两个 EMA state_dict 键（另含两份 live 权重），所以原样整份上传、不拆分；
只传 epoch 72（该 run 共 36 个 ckpt，它是最后一个）。

| 项 | 值 |
|---|---|
| repo | `HongzeFu/MotionJEPA`（model，**private**） |
| commit / tag | `749224110f91c82f5fc5d3007281d07a5d5c944e` / `wan-v8-filter10-72ep-a-e72` |
| 验收 | `HF_META=PASS files=6`（比远端 `lfs.sha256` 与 `blob_id`，不下载）→ 冷缓存回读 `sha256sum -c` 两行 OK、`RESULT=PASS` → 收尾复算源 ckpt sha 未变 |
| 耗时 | 端到端约 **30 秒**（xet 内容分块去重，955 MB 里实际只传了 134→402 MB 量级） |

driver 是 `scripts/dataset/hf_export/run_model_export.sh`（八阶段），L1 验收器是同目录
`verify_model_repo.py`。**不采信 `usedStorage`**：上次 bucket 导出实测它异步滞后，且对等长篡改失明。

## 五、异地无 NFS 机器从零复刻

```bash
git clone -b v2-motionmem https://github.com/hongzefu/robomme_policy_learning_MotionJEPA.git <REPO>
cd <REPO>
UV_LINK_MODE=copy uv sync                                              # 主 venv
UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan \
  uv sync --project scripts/dataset/wan                                # wan 子 venv（torch 2.9.0 / diffusers 0.39.0）
export HF_TOKEN=hf_…                                                   # 仅私有的 MotionJEPA 两条需要
uv run python scripts/assets/fetch_assets.py plan                      # 期望末行 ASSETS_PLAN total=14.5GB assets=6 missing=6
uv run python scripts/assets/fetch_assets.py fetch                     # 取回后自动 full 复校；约 14.5 GB，建议放 tmux
uv run python scripts/assets/fetch_assets.py verify --level full       # 期望末行 ASSETS=PASS assets=6 mismatches=0
```

MotionJEPA 的**模型代码**在私有 GitHub `hongzefu/MotionJEPA`，由 `scripts/dataset/wan/pyproject.toml`
以 git 依赖钉死 commit 引入，所以 wan 子 venv 那步同样需要该仓库的访问权。

**已知阻塞——已于 2026-09-04 解除（commitV6.12，环境 B / AWS 8×A100 实测复刻）**：两个 `paths.sh` 的前缀白名单加了第三项
`AWS_WORK_PREFIX="/scratch/hongze/"`（用户拍板：加常量前缀，不改成与路径无关的判据），`RAW_H5_DIR` 默认按前缀分叉、`MJ_REPO` 可用环境变量覆盖。
同一轮实测踩中并修掉的第二个阻塞：`snapshot_download(revision=<commit sha>)` **不写 `refs/main`**，而 `verify` 的 `hf_snapshot_subdir` 分支与
`HF_HUB_OFFLINE=1` 下按 repo_id 的离线加载都要它 —— 首次 `verify --level full` 报 `wan_vae: 缺 refs/main` → `ASSETS=FAIL`；`fetch_assets.py` 现在在
钉 sha 的 snapshot 落盘后补写 `refs/main = revision`（已存在且不同则响亮失败不覆盖），复跑 `fetch --force --assets wan_vae` 后 `ASSETS=PASS assets=6 mismatches=0`。
完整复刻记录（环境判定、四个公开 h5 与环境 A sha256 对拍、40 ep 库 D1–D3 逐位、400 ep 完整库）见
`docs/dataset-build-doc/4task-motion-40ep-aws/` 与 `docs/dataset-build-doc/4task-motion-400ep/`，以及 `motion-memory-plan.md`「环境 B 复刻」节。
异地机器上 `paths.sh` 仍要求仓库位于三个前缀之一；换第四台机器需再加一项常量（有 `test_paths_sh_prefixes_identical` 盯两份同值）。

**数据侧口径**：原始 16 任务 × 100 episode 的 H5 在公开数据集 `Yinpei/robomme_data_h5`；派生库
（SigLIP 特征 / Wan latent / motion token / packed store）需在异地用 `scripts/dataset/` 重建，全程要 GPU。
注意 MotionJEPA 那份 encoder 的训练数据是**私有的 4 任务 × 400 episode 录制版**，与公开版不是同一份。

## 六、边界与盲区

三条最容易被误读的（其余见 `result.md` 第九节）：

1. **cheap 档挡不住保持长度的中段字节篡改**。真正的保证来自 `verify --level full` 与各重载点
   （`load_encoder` 每次全量算 ckpt sha256、`load_vae` 每次算 state_dict 指纹）。
2. **`lfs.sha256` 是客户端 commit 时提交的**，能证明「Hub 记录的就是我这份文件的 sha」（挡传错、挡漏传），
   不能证明「Hub 存的字节确实哈希成这个值」——后者靠回读闭环兜底。
3. **资产锁保证输入字节同一，不保证输出数值逐位同一**。实测 SigLIP 段在 A40 与本机 Ada 之间
   `image_emb min_cos 0.99959`（而 `pos_emb` / `state_emb` / pkl / `kept_indices` 逐位相同）；
   Wan 段只有同机双卡 64 窗 `max|Δ|=0` 的实测，**跨架构未测**。禁止把 `ASSETS=PASS` 读成「数值可逐位对拍」。

## 附：本轮 commit 与留档

| commit | 内容 |
|---|---|
| `1ebab97` | 上传起跑留档 + driver（`run_model_export.sh` / `verify_model_repo.py` / model card 正本 / SHA256SUMS） |
| `f07ee19` | 资产锁落地：`ASSETS_LOCK.json` + `assets_lock.py` + `fetch_assets.py` + 22 条守卫 |
| `c7406e4` | 锁接入链路六处 + 6 条接入点哨兵（共 28 条） |
| `ee0ae46` | 收官回填：`result.md` + `records/` 六件 + `README-ZH.md` 一节 |
| `1b75d63` | 三个探针里辅助函数的空行按 PEP8 补齐（纯格式） |

详细留档在 `docs/dataset-build-doc/hf-export-motionjepa-encoder-v1/`：`launch.md`（起跑状态、来源真值表、
环境纪律、判据）、`result.md`（终判、验收三层原文、耗时、六则意外、AGENTS 第 18 条判断、七条盲区）、
`model-card.md`（HF 上 README.md 的正本）、`records/`（`blobs_api.json` 远端元数据快照、
`SHA256SUMS.txt`、`upload.clean.log.txt`、`assets_lock_verify.txt`、`negative_tests.txt`、`diff_scope.txt`）。
