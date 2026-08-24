# bottleneck-bench：4 卡 batch 64 的 NFS 瓶颈判定三实验

**要回答的问题**：在 GL 集群 4×A40、全局 batch 64（官方口径）下训练，NFS turbo 的数据供给会不会成为瓶颈？

**判据链**（供给 vs 需求）：

```
需求 MB/s = 每步字节 ÷ compute-only 步时          ← 实验 3（4×A40 测纯计算步时）
供给 MB/s = dataloader-only 实测吞吐               ← 实验 1（GL 节点权威测量）
供给 ≥ 需求 → 无 NFS 瓶颈（结案）；供给 < 需求 → 有瓶颈，缺口 = 需求/供给
```

每步字节的精确口径：每样本 = `395,440 B（data/{idx}.pkl）+ min(step_idx+1, 32) × 602,951 B（token_emb_*.npy）`，episode 均长 ~302 步 → 均值 ≈ 18.7 MB/样本 → **b64 每步 ≈ 1.20 GB、约 2000 次文件读**（小文件随机读，元数据开销可能先于带宽到顶）。

为什么需要专门实验：此前本机 v1-2gpu-epoch-bench-b8 的 1.060 s/step 不可用于瓶颈判定——第 2/3 轮同 seed 导致第 3 轮读的样本全部命中 page cache（本机 377 GB 内存），且计算与 IO 在 dataloader 预取下相互掩盖，无法分离。

## 三个实验

| 子目录 | run_name | 跑在哪 | 测什么 |
|---|---|---|---|
| `gl-dataloader/` | `v1-gl-dlbench`（单 job 版）+ `v1-dlb-w{4,8,16}c{6,10,18}`（拆分版） | GL A40×1（debug 包络内） | **供给侧**：与 train.py 同链路的 dataloader 只迭代不训练，batch 64，记 样本/s 与 MB/s 双口径（18.7 MB/样本公式 vs mountstats `server_read` 实测；后者不含 cache 命中但含节点其他进程串扰，互校用）。单 job 版（18C 内扫 workers 4/8/16）排队过久后拆成 6 个独立小 job 并行排（`submit_split_jobs.sh`，矩阵 workers∈{4,8,16} × CPU 配比∈{1,2}/worker，另含 w16c10 超订档；每 job 独立 seed 200-205），原 job 保留不取消 |
| `gl-compute-only/` | `v1-computeonly-b64` | GL A40×4（**超 debug 包络，已获用户 2026-08-24 逐次特批**） | **需求侧**：monkeypatch `create_data_loader` 为「首个 batch 缓存后无限重复」，测 b64 纯计算 s/step（丢前 20 步：编译+首 batch NFS 读），推出需求 = 1.20 GB ÷ 步时 |
| `local-coldcache/` | `v1-coldcache-b8` | 本机 2 卡 | **旁证**：seed 42→123 换冷样本重跑 150 步（无中途校验和），5 s 采样 GPU 利用率 + turbo `server_read` 真实读流量；冷缓存步时 vs 热缓存基线 1.060 s/step 的倍率直接暴露本机口径下 IO 是否是瓶颈 |
| `gl-e2e/` | `v1-e2e-b64` | GL A40×4（**超 debug 包络，已获用户 2026-08-24 逐次特批，1h**） | **实测校验**：官方口径（4 卡、全局 b64、workers 4）端到端 300 步，真 dataloader 走 NFS + 真训练循环（零改动复用 `smoke-local/bench_train_steps.py`），15 s 采样 GPU 利用率与 turbo `server_read`；实测稳态步时应 ≈ max(实验 3 compute-only 步时, 1.20 GB ÷ 实验 1 供给 MB/s)，并直接给出 epoch(6,176 步) 实测口径时长 |

三个实验都不落 checkpoint、不开 wandb；GL 两个 sbatch 均遵守 greatlakes.md（chaijy2/spgpu、无 qos、日志落 `v1-store/logs/%x-%j.log` 双端可见）。

## 记录文件（`v1-store/bench/bottleneck/<run_name>/`）

- 实验 1：`batches.jsonl`（每批 `{workers, batch_idx, wall_time, dt}`）、`summary.jsonl`（每档 `{workers, samples_per_s, mbps_formula, mbps_mountstats_server_read, mbps_mountstats_normal_read, s_per_batch, …}`）、`env.json`。
- 实验 2：`metrics.jsonl`（逐步，格式同 `scripts/smoke-local/README.md`）、`gpu_util.csv`（`时间戳,卡号,util%,显存MiB`）、`nfs_read.csv`（`时间戳,normal_read累计,server_read累计`；`server_read` 是真正走网络的字节，page cache 命中不计）、`param_checksums.jsonl`（仅末步一次）、`env.json`。
- 实验 3：`metrics.jsonl`（逐步步时来源）、`gpu_util.csv`、`save_state_calls.jsonl`（no-op 标记）、`env.json`。

每个实验 stdout 末尾有 `RESULT …` 行给出结论数字；三份齐后汇总写 `docs/v1-nfs-bottleneck-analysis.md`。

## 跑法

```bash
# 实验 2（本机，~10 分钟，tmux）：
tmux new-session -d -s coldcache \
  "set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/bottleneck-bench/local-coldcache/run_local_coldcache.sh 2>&1 \
   | tee v1-store/logs/v1-coldcache-b8-driver.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/v1-coldcache-b8-driver.log"

# 实验 1 / 3（经 gl_submit.py 提交；ControlMaster 失效时需 GLPW + GLOTP=TOTP）：
uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
  "sbatch --parsable scripts/bottleneck-bench/gl-dataloader/gl_dataloader_bench.sbatch"
uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
  "sbatch --parsable scripts/bottleneck-bench/gl-compute-only/gl_compute_only.sbatch"
```

按 AGENTS.md 第 12 条：三个 run 均从 clean HEAD 起跑，起跑与结果留档在 `docs/training-doc/<run_name>/`。
