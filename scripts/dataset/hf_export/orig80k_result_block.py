#!/usr/bin/env python3
"""为 orig80k 两条正式 run 生成 HF bucket 的整份 README。

由 run_orig80k_ckpt_export.sh 在阶段 2b 调用。**所有数字都从现场文件现读**：
launch.json（argv、数据指纹）、completion.json（完成器判定）、driver 日志（起止、退出码、loss）、
history_config.* 与 motion_provenance.json；不接受人工输入，避免 README 与日志对不上。
driver 日志的 Step 行带 `HH:MM:SS.mmm [I] ` 前缀，故 loss 匹配不锚定行首
（motion80k_result_block.py 用 `^Step`，对本日志会落空，所以另写本文件而不复用）。

用法: orig80k_result_block.py <run_name> <bucket_id> <records目录> <run根> <driver日志> <输出 README>
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

RUN, BUCKET_ID, REC, ROOT, LOG, OUT = sys.argv[1:7]
REC, ROOT = Path(REC), Path(ROOT)
BUCKET_NAME = BUCKET_ID.split("/", 1)[1]

# 两条 run 的人写说明只有这几行；其余全部现读
DESC = {
    "v2-orig-16task-pub1600ep-modul-b64-80k": dict(
        short="公开 16 任务全集",
        data="`16task-pub-1600ep/framesamp`（1,600 集、768,897 帧、476,857 执行样本）",
        gpus="0,1,2,3", norm_note="上游原版 norm_stats"),
    "v2-orig-counting-pub400ep-modul-b64-80k": dict(
        short="公开纯 counting 四任务子集",
        data="`4task-counting-pub-400ep/framesamp`（400 集、189,035 帧、189,035 执行样本）",
        gpus="4,5,6,7", norm_note="经用户授权在本子集上自算的 norm_stats"),
}
if RUN not in DESC:
    sys.exit(f"错误: 未知 run {RUN}")
d = DESC[RUN]
PAIR = [r for r in DESC if r != RUN][0]
PAIR_BUCKET = BUCKET_ID.replace(
    "16task-1600ep" if "16task" in RUN else "counting-400ep",
    "counting-400ep" if "16task" in RUN else "16task-1600ep")

launch = json.loads((REC / "launch.json").read_text())
comp = json.loads((REC / "completion.json").read_text())
if comp.get("status") != "PASS" or comp.get("run_name") != RUN:
    sys.exit("错误: completion.json 不是本 run 的 PASS")
wandb_id = (ROOT / "wandb_id.txt").read_text().strip()
hist_name = (ROOT / "history_config.txt").read_text().strip()
hist_sha = (ROOT / "history_config.resolved.sha256").read_text().split()[0]
text = Path(LOG).read_text(encoding="utf-8", errors="replace")


def kv(name):
    m = re.findall(rf"^{name}=(.*)$", text, re.M)
    return m[-1].strip() if m else None


start, end, exit_code, head = kv("START_UTC"), kv("END_UTC"), kv("EXIT_CODE"), kv("TRAIN_HEAD")
if exit_code != "0" or head != comp["head"]:
    sys.exit(f"错误: 日志 EXIT_CODE={exit_code} 或 TRAIN_HEAD={head} 与完成器不符")

steps = sorted({int(s): l for s, l in re.findall(r"Step (\d+): .*?loss=([0-9.eE+-]+)", text)}.items())
want = [0] + list(range(10000, 80001, 10000))
picked = [sl for sl in steps if sl[0] in want]
if steps and steps[-1] not in picked:
    picked.append(steps[-1])

res = []
t0 = dt.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
t1 = dt.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
sec = int((t1 - t0).total_seconds())
h, rem = divmod(sec, 3600)
res.append(f"起跑 `{start}`，退出 `{end}`，耗时 **{h} 小时 {rem // 60} 分**（{sec:,} 秒，含初始化、"
           "checkpoint 保存与退出），`EXIT_CODE=0` 正常退出。")
res += ["", "| 日志步 | loss（log100 区间统计，step 0 除外） |", "|---|---:|"]
res += [f"| {s} | {l} |" for s, l in picked]
res += ["", f"日志共解析到 {len(steps)} 条 `Step N` 行。最后一条属于 step {steps[-1][0]}，"
        "覆盖其前 100 步的区间，**不是 step 79999 的单步 loss**。**训练 loss 不代表策略评估成功率。**"]

ckpts = comp["checkpoints"]
layout_steps = " ".join(f"{s}/" for s in ckpts)
data = launch["data"]
argv = launch["argv"]

md = f"""# {BUCKET_NAME} — {RUN} checkpoint 备份

本 bucket 存放训练 run **`{RUN}`** 的全部 {len(ckpts)} 个 orbax checkpoint，
由 `scripts/dataset/hf_export/run_orig80k_ckpt_export.sh` 上传，上传前 / 上传后两遍独立 sha256
逐行对照验收（`SHA256SUMS.pre.txt` 与回读侧重算的 post 清单 diff 为空）。

## 这条 run 是什么

**上游原版 MME-VLA `mme_vla_suite` 配置**（80k × batch 64，modulation，perceptual frame sampling
budget 512 = 16 帧 × 16 token，**motion 关闭**）在{d['short']}上的正式训练。
与 `{PAIR}`（`{PAIR_BUCKET}`）从同一起跑 commit、同一配置 4+4 卡并行起跑，
**两条只差训练数据与对应 norm_stats**，可作一组对照阅读。

| 项 | 值 |
|---|---|
| config 条目 | `mme_vla_suite`（`src/mme_vla_suite/training/config.py`） |
| history config | `{hist_name}`（resolved sha256 `{hist_sha}`） |
| integration / perceptual memory | `modulation` / `frame_sampling` |
| budget / token_per_image / streaming_obs_horizon / memory_token_dim | 512 / 16 / 16 / 1024 |
| motion | **关闭**（`motion_provenance.json` 的 `motion_enabled` 为 `false`） |
| global batch / steps | 64 / 80,000 |
| save_interval / keep_period | 10,000 / 10,000 |
| fsdp_devices / num_workers / dtype | 4 / 4 / bfloat16 |
| optimizer | AdamW，`clip_gradient_norm=1.0`，`ema_decay=0.999` |
| lr schedule | warmup 10,000，`peak_lr = decay_lr = 5e-5`，`decay_steps=100,000` |
| seed | 42 |
| 初始权重 | `openpi-assets/checkpoints/pi05_base/params` |
| 训练数据 | {d['data']} |
| framesamp manifest sha256 | `{data['manifest_file_sha256']}` |
| framesamp store meta sha256 | `{data['store_meta_sha256']}` |
| norm_stats sha256 | `{data['norm_stats_sha256']}`（{d['norm_note']}） |
| 起跑 commit | `{comp['head']}`（commitV11.18Beta） |
| run_uuid | `{comp['run_uuid']}` |
| wandb run id | `{wandb_id}`（project `openpi`） |
| 硬件 | 4 × NVIDIA A100-SXM4-80GB（GPU {d['gpus']}），AWS 本地 NVMe RAID `/dev/md0` |

实际训练入参（`_run-meta/train-record/launch.json` 的 `argv`）：

```
{' '.join(argv)}
```

## 完成验收

训练结束后由完成器 `scripts/training/tests/check_orig80k_completion.py --mode prod` 验收：唯一终态
`EXIT_CODE=0`、checkpoint 恰为 {', '.join(map(str, ckpts))} 共 {len(ckpts)} 份、log100 全部有限、
末 99 步五项标量有限、`state_step=80000`、最终 `79999/params` 真实读回并与保存现场 EMA 逐叶摘要一致、
GPU 采样覆盖全程。判定 JSON 在 `_run-meta/completion.json`，保存现场记录在 `_run-meta/train-record/`。

## 训练结果

{chr(10).join(res)}

## 重要声明

1. **不能续训。** checkpoint 只含 EMA `params` 与 `assets`，没有 optimizer / train_state，
   只能用于推理或作为 finetune 初始化。
2. **训练 loss 不代表策略评估成功率。** 本 run 的策略评估另行开展，不在本 bucket 内。

## 布局

```
{layout_steps}   # orbax checkpoint，每个约 11 GiB；79999 是 80k 的末步
  _CHECKPOINT_METADATA
  assets/robomme/norm_stats.json
  params/  (_METADATA _sharding manifest.ocdbt array_metadatas/ ocdbt.process_0/d/<hash>)
history_config.resolved.yaml / .sha256 / history_config.txt
motion_provenance.json                # framesamp 链路指纹（motion 字段全为 null）
wandb_id.txt
_run-meta/
  train.log                           # driver 完整日志
  wandb/run-<ts>-{wandb_id}/          # wandb 本地 run 目录
  completion.json                     # 完成器判定
  train-record/                       # launch.json runtime.json metrics.jsonl gpu.csv final/
  git-commit.txt  train-cmd.txt       # 起跑 commit 与实际启动命令文件
SHA256SUMS.pre.txt                    # 上传前逐文件 sha256（路径相对 bucket 根）
```

## 恢复与自验

```bash
hf sync hf://buckets/{BUCKET_ID} ./{BUCKET_NAME}
cd {BUCKET_NAME} && sha256sum -c --strict SHA256SUMS.pre.txt
# 只取末步
hf sync hf://buckets/{BUCKET_ID} ./ckpt-79999 --include '79999/*'
```

```python
import orbax.checkpoint as ocp
params = ocp.PyTreeCheckpointer().restore("{BUCKET_NAME}/79999/params")
```
"""
Path(OUT).write_text(md, encoding="utf-8")
print(f"README 已生成: {OUT}（loss 采样 {len(picked)} 条，checkpoint {len(ckpts)} 份）")
