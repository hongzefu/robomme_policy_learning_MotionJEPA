# hf-export-awsprod40k-ckpt-v1 · launch

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 6.9 T 本地 NVMe RAID）**

- 任务：把训练 run `awsprod40k-b128-motion` 的全部 8 个 orbax checkpoint（87 GB / 164 文件）
  上传到 HuggingFace **private** bucket `hf://buckets/HongzeFu/robomme-motionjepa-vla-v1`，
  上传前 / 上传后两遍独立 sha256 逐行对照验收。
- 动机：这批权重此前只存在于本机 NVMe，无任何异地副本；环境 B 无 turbo / GreatLakes 可用。
- 起跑 commit：本文件所在 commit（clean HEAD 起跑，见下方「起跑前状态」）。
- 入口：`bash scripts/dataset/hf_export/run_ckpt_export.sh`
- tmux 会话：`hf-ckpt-export`（本轮起过的会话仅此一个）。
- 日志：`v1-store/exports/hf-ckpt-awsprod40k-b128-motion/logs/run.log`

## 源与落点

| 项 | 值 |
|---|---|
| 源 | `v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/`（8 个 step：5000…39999） |
| 目标 | `hf://buckets/HongzeFu/robomme-motionjepa-vla-v1`（private，起跑前 `files=0 size=0`） |
| stage / verify | `v1-store/exports/hf-ckpt-awsprod40k-b128-motion/{stage,verify}` |
| 凭据 | `v1-store/secrets/hf.env`（`HF_TOKEN`，600，在 `.gitignore` 的 `/v1-store/` 下） |

## 起跑前状态（已核实）

- `hf auth whoami` → `user=HongzeFu orgs=umich`。环境里原有的 `HF_TOKEN` 属 `yinpei-tri`，
  写不进 `HongzeFu`，driver 内由 `hf.env` 覆盖。
- 目标 bucket 经 `https://huggingface.co/api/buckets/HongzeFu` 确认：`private=true`、
  `totalFiles=0`、`size=0`，创建于 2026-09-06T06:29:09Z。driver 阶段 0 会再断言一次，非空即停。
- 源：164 个文件 / 87 GB，每个 step 20 个文件（OCDBT 大块 1–2.2 GB）+ run 根 5 个元文件。

## 工具链

项目 `.venv` 的 `huggingface_hub` 是 **0.32.3**，其 `huggingface-cli` **没有 bucket 相关子命令**，
而该版本由 openpi 依赖钉住、不能升。故用 `uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf`
拉一份完全独立的 CLI，只在本次导出内使用，`pyproject.toml` / `uv.lock` / `.venv` 一律不碰。

1.30 相对老脚本（`run_export.sh`）用的版本已重构命令：

| 老脚本 | 1.30 |
|---|---|
| `hf buckets sync A B` | `hf sync A B` |
| `hf buckets info <id>` | `hf repos ls`（人类可读值；本 driver 的精确字节改走 buckets REST API） |

## 验收判定行

日志里必须同时出现，缺一条即不算通过：

```
HF_WHOAMI=HongzeFu
LOCAL_FILES=166 LOCAL_BYTES=<n>
PRE_LINES=165
BUCKET_FILES=166 BUCKET_BYTES=<n>        # 与 LOCAL_* 逐字相等
SHA256_PRE_POST_DIFF=0                   # 上传前 / 上传后两份独立清单 diff 为空 ← 核心判据
RESULT=PASS
SRC_UNCHANGED=OK                         # 源侧第三遍 sha256 与上传前一致
EXIT_CODE=0
```

注：源目录共 **164 个文件**（8 个 step 的 OCDBT 块数不等——19/21/20/20/19/19/21/20——
**已含** run 根 5 个元文件），stage 再加 `README.md` 与 `SHA256SUMS.pre.txt` 共 166 个。
起跑前预估的 171/170 把那 5 个元文件重复计了一次，实测值以本节为准。
