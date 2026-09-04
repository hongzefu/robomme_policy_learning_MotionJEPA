# launch：MotionJEPA encoder+decoder 上传 HF private model repo

## 目标

把 MotionJEPA run `wan-v8-filter10-72ep-a` 的 **encoder + wan_decoder**（同一个
`checkpoint_epoch_72.pt` 里的两个 EMA state_dict 键，另含 `encoder_live` / `wan_decoder_live`）
**原样整份**上传到 HuggingFace **private** model repo `HongzeFu/MotionJEPA`，
只传 epoch 72 这一个 ckpt，repo 保持 private。

上传后该 repo 的 commit sha 会回填进 `scripts/assets/ASSETS_LOCK.json` 的 `motionjepa_ckpt` /
`motionjepa_config` 两条来源记录——**这是先上传后建 lock 的原因**：lock 的 `revision` 字段必须是
40 hex 的 HF commit sha，中间态不允许写 `main` 或 `null`。

## 起跑状态

- 仓库 `/data/hongzefu/robomme_policy_learning_MotionJEPA`，分支 `v2-motionmem`。
- 起跑 HEAD = 本文件所在的 `docs:` commit（clean tree 起跑，全 sha 在 `result.md` 回填）。
- 上一 commit `7040266`（T3_EVAL_OBS 收官）。

## 来源与真值

源在**本机 NVMe**，不必碰 turbo NFS：

| 文件 | 路径 | 字节 | sha256 |
|---|---|---|---|
| ckpt | `v1-store/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt` | 954,853,147 | `bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a` |
| config | 同目录 `config.yaml` | 1,927 | `99548a6ca23522c235281e45819ae6d5e96a916709cb4b9c0b47142832c90946` |
| 源-副本对照 | 同目录 `SHA256SUMS.src-vs-copy.txt` | 704 | 记录 turbo 原件与本机复制件同 sha |

上游是 MotionJEPA `runs/wan-v8-filter10-72ep-a/`（turbo 只读归档，共 36 个 ckpt：epoch 2,4,…,72；
本次只取最后一个）。训练 commit `7388a42`、数据集 `dataset-4env-v8` 等 model card 事实见
同目录 `model-card.md`（唯一正本），本文件不复述。

## 环境纪律（本次最容易翻车处）

```bash
export HF_HOME="$REPO/v1-store/cache/hf"                 # AGENTS 第 14 条，缓存进 v1-store
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"    # 凭据留原处，**不复制第二份**
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
unset HF_HUB_OFFLINE
```

- 实测对照：`HF_HOME=<新目录>` 单独设 → `hf auth whoami` 报 `Not logged in`；`XDG_CACHE_HOME=<新目录>`
  同样 → `Not logged in`（`HF_HOME` 默认值由它派生）；`HF_HOME + HF_TOKEN_PATH` 组合 →
  `user: HongzeFu / orgs: umich`，退出码 0。所以本次**不沿用** `run_export.sh` 的 `cp token` 老修法。
- `hf` CLI 用绝对路径 `/home/hongzefu/.local/bin/hf`（1.8.0，独立 venv）。token profile 必须是
  `[1]`（`role: write`），账号 `HongzeFu`。
- 每条 `hf upload` 必带 `--private`：CLI 内部硬调 `create_repo(exist_ok=True, private=private)`，
  而 `--private` 默认 `None` → 新建时是 public，repo_id 打错一个字母就会静默新建 public repo
  并把 954 MB 权重传上去。repo 已存在时该参数被忽略。
- 命令里**禁止出现** `--delete`（按 glob 删远端文件）与 `--every`（挂后台定时 commit）。
- 禁止用通配符 `local_path`：1.8.0 的 `_resolve_upload_paths` 通配分支未 adjust，会把当前工作
  目录传到字面名为 `./x/*.pt` 的远端路径下。

## 命令

```bash
tmux new-session -d -s hf-model-motionjepa-v1 \
  "set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/dataset/hf_export/run_model_export.sh 2>&1 \
   | tee v1-store/exports/hf-model-motionjepa-encoder-v1/logs/upload.log; \
   echo \"EXIT_CODE=\${PIPESTATUS[0]}\" >> v1-store/exports/hf-model-motionjepa-encoder-v1/logs/upload.log"
```

driver 八阶段：预检（源指纹 + `hf auth whoami`）→ 搭 stage（ckpt 走**硬链接**，不复制 954 MB）→
commit 1（小文件，`--exclude "*.pt"`）→ commit 2（ckpt）→ 打 tag `wan-v8-filter10-72ep-a-e72` →
L1 元数据验收 → L2 回读闭环 → 收尾断言（源 ckpt sha 未变）。

## 判据（起跑前写死，事后不改）

| 判定行 | 期望 |
|---|---|
| `HF_META=PASS files=6 lfs_mismatch=0 blob_mismatch=0` | 远端 `lfs.sha256` / `blob_id` / `size` 与本地 stage 逐文件相等（`files` 含 Hub 自带的 `.gitattributes`） |
| `HF_PRIVATE=True commit=<40 hex>` | repo 仍 private；该 commit sha 回填进 lock |
| `sha256sum -c --strict SHA256SUMS.txt` → 2 行 `OK` | 回读字节 == 本地字节 |
| `RESULT=PASS` | L2 闭环通过（含 `cmp` README 与 SHA256SUMS.src-vs-copy） |
| `EXIT_CODE=0` | driver 全程无错 |

**不采信** `usedStorage` / 网页体积（异步滞后，且对等长篡改失明）。

## 暂存与清理

- 落点 `v1-store/exports/hf-model-motionjepa-encoder-v1/{stage,verify,logs}`（本机 `/data`，余 2.9 T）。
- `stage/` 因 ckpt 走硬链接只额外占几 KB；`verify/` 回读 ≈ 0.96 GB，总预算 2.5 GB。
- **回读目录名必须含 `verify`**，绝不下到 `v1-store/external/motionjepa/wan-v8-filter10-72ep-a/`
  ——那是 `--encoder-run-dir` 的默认值，`hf download` 会往里塞 `.cache/huggingface/`。
- 验收通过后删 `stage/` 与 `verify/`，保留 `logs/` 与 `records/`；`stage` 下的 `.pt` 与源同 inode，
  清理只用 `rm`（只删链接），删后复算源 sha 作收尾断言。
