# motion-p5-online — P5 真编码器在线链 vs 离线 motion 表（launch）

- **目的**：motion-memory-plan.md 四节表一 **P5**（原 A23 升格）。在线 sidecar（真 Wan VAE + MotionJEPA encoder）沿 eval.py 真实节奏逐窗编码，
  与 S1 离线库 `v1-store/datasets/4task-motion-40ep/motion/motion_token.f32.bin` 对应行逐位比对；同时验起点集合、时间码、交错次序与 provenance，记三笔耗时。
- **前提**：P1–P4 全过（`scripts/training/tests/motion_gates_online.py`，commitV6.6）；S1 40 集库 `verified`；同机同型号卡（2×RTX 6000 Ada，跨卡逐位由 A3 保证）。
- **入口**：`scripts/training/g0/compare_online_motion.py`（commitV6.6 新增）。
  - 驱动：`episode_manifest.json` 40 条 episode，从 `raw_dir` 的 h5 读全部 `front_rgb` (256,256,3) uint8；首批 pre_traj = 帧 [0, es]、之后每批 16 帧；
    `exec_start_idx` 首批真实值、后续传 0（客户端 `clear_buffers` 语义）；帧路用零特征（帧路数值由 `compare_online_memory.py` 负责，主进程 jax 只算 4x4 pos 表）。
  - 运动路：`MotionEncoderClient(online_gpu=<--gpu>)` 真 sidecar（wan 子 venv、fp32 / 关 TF32 / B=1 / 33 帧一次喂），握手 provenance 与 `store_meta.provenance` 按打包器 same_keys 逐键比对。
  - 判据行：`ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0 rows_total=772 covered=772`、`ONLINE_START_SET=PASS steps=N`、`ONLINE_POS=PASS`、
    `ONLINE_ORDER=PASS steps=N`（含 motion_emb / motion_pos / motion_mask / mem_order 四键 vs `FrameSampDataset.__getitem__` 逐位 + 置换合法）、`PROVENANCE=PASS`、`P5_ONLINE=PASS`。
  - 耗时：`ENC_MS_PER_WINDOW`（客户端夹 send/recv）、`FIRST_BATCH_MS`（首批 add_buffer 挂钟 = demo 段窗口全部同步编完）、`LATER_BATCH_MS`（后续每批 add_buffer 挂钟，即每次推理前固定开销；本脚本不经 websocket，server `_handler` 的 `add_buffer_time_ms` 在 T3_EVAL_OBS 时另记）。
- **注意（stub 试跑发现）**：主进程 jax 必须在 GPU 上——CPU jax 算的 `PosEmb3D` 4x4 表与库内 GPU 生成表 max abs 6.1e-5、22% 元素不等，
  会让 `ONLINE_POS` / `ONLINE_ORDER` 假失败；`compare_online_memory.py` 的 `POS_TABLE` 三方逐位也是在 GPU 上过的。主进程用 GPU0（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`），sidecar 用 GPU1。
- **起跑命令**（主树、clean HEAD，T1 结束释放 GPU 后）：

```bash
cd /data/hongzefu/robomme_policy_learning_MotionJEPA && mkdir -p v1-store/logs v1-store/reports/motion && tmux new-session -d -s motion-p5-online \
  "set -o pipefail; CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
   uv run --no-sync python scripts/training/g0/compare_online_motion.py --gpu 1 --out v1-store/reports/motion/p5_online.json 2>&1 \
   | tee v1-store/logs/motion-p5-online.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/motion-p5-online.log"
```

- **留档**：`result.md` + `records/p5_online.json`（判定行、逐 episode 窗数 / 耗时、失配清单、sidecar provenance）+ `records/run.log`。
