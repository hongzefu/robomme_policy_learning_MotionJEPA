# 官方16×100 三模型三 seed 官方评估结果

## ① 一句话结论与指标速览

三条官方 16×100 训练模型的 59999 checkpoint，在官方 test 800 局上按官方三个模型 seed（7、42、0）各评三次，总体成功率均值 ± 样本标准差（ddof=1，与官方 `compute_results.py` 同口径）为：

| 模型 | seed 42 | seed 7 | seed 0 | 三 seed 均值 ± std | 论文 Table 3 对照 |
|---|---|---|---|---|---|
| modulation split（`orig100-m32x4x4-modul-motion-s2split4x256-b128-60k`） | 346/800 = 43.25% | 344/800 = 43.00% | 352/800 = 44.00% | **43.42 ± 0.52** | FrameSamp + Modul 44.51 ± 0.77 |
| context split（`orig100-ctx32x4x4-motion-s2split4x256-b128-60k`） | 311/800 = 38.88%（10-29 复用） | 296/800 = 37.00% | 315/800 = 39.38% | **38.42 ± 1.25** | FrameSamp + Context 30.68 ± 1.17 |
| context concat 1×1024（`orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k`） | 304/800 = 38.00% | 295/800 = 36.88% | 297/800 = 37.13% | **37.33 ± 0.59** | FrameSamp + Context 30.68 ± 1.17 |

modulation 比论文同口径低约 1.1 个百分点，落在两边标准差之内；两条 context 模型都比论文 FrameSamp + Context 高约 7 个百分点。同为 context 融合，motion 读法 split（4×256）与 concat（1×1024）相差约 1.1 个百分点（38.42 vs 37.33），与 seed 间波动同量级。modulation 比 context split 高 5.0 个百分点。

**eval 阶段每任务 motion 窗口数**（2026-11-04 补算）：两条 context 模型的最大值都是 70 窗（VideoPlaceOrder），未达到 budget 96 窗。concat 每窗 1 个 token，最大 70 token；split 每窗 4 个 token，最大 280 token（上限 384）。每局窗口数 = `k_demo + k_exec`，由逐局 `accepted/*.json` 的 `budget.demo_windows` 与 `actions_executed // 32` 算出（`stride=32`，`window_frames=33`）；每任务 3 个 seed 共 150 局。

| 任务 | demo 长度 D 最大 | concat 最大窗口 | concat 成功局最大 | concat 超时局数 | split 最大窗口 | split 成功局最大 | split 超时局数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VideoPlaceOrder | 1134 | 70 | 41 | 5 | 70 | 39 | 5 |
| VideoPlaceButton | 833 | 63 | 32 | 1 | 63 | 32 | 1 |
| InsertPeg | 375 | 52 | 19 | 48 | 49 | 16 | 47 |
| MoveCube | 283 | 49 | 35 | 23 | 49 | 32 | 14 |
| VideoRepick | 383 | 45 | 25 | 1 | 25 | 25 | 0 |
| BinFill | 0 | 40 | 29 | 1 | 38 | 24 | 0 |
| SwingXtimes | 0 | 40 | 17 | 1 | 16 | 16 | 0 |
| PickHighlight | 0 | 40 | 18 | 13 | 40 | 25 | 9 |
| PickXtimes | 0 | 23 | 23 | 0 | 26 | 26 | 0 |
| ButtonUnmaskSwap | 0 | 20 | 17 | 0 | 40 | 16 | 4 |
| VideoUnmaskSwap | 216 | 20 | 20 | 0 | 14 | 14 | 0 |
| RouteStick | 350 | 18 | 16 | 0 | 17 | 17 | 0 |
| StopCube | 0 | 17 | 17 | 0 | 16 | 16 | 0 |
| ButtonUnmask | 0 | 17 | 16 | 0 | 21 | 15 | 0 |
| VideoUnmask | 66 | 13 | 12 | 0 | 11 | 10 | 0 |
| PatternLock | 258 | 10 | 7 | 0 | 11 | 10 | 0 |

口径：`actions_executed` 是一局结束时的步数，模型最后一次推理在最后一段动作之前，实际见到的窗口数可能略少，因此表中是上界。超时局的执行步数为 1301，`k_exec` 封顶 40，会抬高最大值；看任务正常所需长度时以「成功局最大」为准。执行步数取决于模型，换模型后最大值会变。原值见 `records/motion_windows_by_task.json`。

**concat 每局 motion token 数的分布**（2026-11-04 补算，论文 CVPR 稿 Cost 段表 2 的真源）：按上表同一公式 k = `demo_windows` + `actions_executed // 32` 对 concat 三 seed 共 2,400 局逐局计算，每任务 150 局；分位数用线性插值（numpy `linear` 口径），Median 为偶数样本时取中间两值均值、不取整（论文表中整数行四舍五入，如 ButtonUnmaskSwap Median 10.5 写作 11、VideoRepick 13.5 写作 14、P90／P95 的小数同样取整）。无一局超过 budget 96，「Episodes > 96」全为 0，故不单列。逐局记录已复制到论文仓库 `CVPR2027-submission/data/motion_tokens/concat_episodes.json`（sha256 6f1380a36e976d6d17cfda6cb60cea450cbb27142d0bae960bae989d542336a4），由 `scripts/make_motion_tokens_table.py` 出表。

| 统计量 | BinFill | PickXtimes | SwingXtimes | StopCube | VideoUnmask | ButtonUnmask | VideoUnmaskSwap | ButtonUnmaskSwap | PickHighlight | VideoRepick | VideoPlaceButton | VideoPlaceOrder | MoveCube | InsertPeg | PatternLock | RouteStick | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean | 15.2 | 13.3 | 12.3 | 7.1 | 5.7 | 7.1 | 9.0 | 11.1 | 10.5 | 14.7 | 29.2 | 34.4 | 17.2 | 23.0 | 3.9 | 8.5 | 13.9 |
| Median | 15 | 13 | 12 | 6 | 5 | 7 | 8 | 10.5 | 6 | 13.5 | 29 | 34 | 13 | 11 | 3 | 7 | 10 |
| P90 | 21 | 19 | 15 | 13.3 | 7 | 10.1 | 13 | 13.1 | 17.1 | 23 | 30 | 40 | 45.2 | 47 | 7 | 14 | 29 |
| P95 | 24 | 20 | 15 | 16 | 10 | 11.55 | 15 | 14 | 40 | 24 | 31.55 | 40 | 48 | 48 | 7 | 15 | 39 |
| Maximum | 40 | 23 | 40 | 17 | 13 | 17 | 20 | 20 | 40 | 45 | 63 | 70 | 49 | 52 | 10 | 18 | 70 |

最终判定（逐行原文）：

- `SEED_GATE=PASS allowed=0,7,42 rejected=1`；`PRE_MERGE_REVIEW=PASS base=9459859384589b3553148cf53f715c1318f1a533 tip=2cb5d8e7d4b0b092207a58dcab6fe8432c0fc20f files=2 commits=1 findings=0`；`POST_MERGE_REVIEW=PASS merge=5fce5d3 tests=45passed files=2`
- `RELEASE_SNAPSHOT=PASS files=4674 tree_diff=0`
- modulation：`VLA_DOWNLOAD=PASS step=59999 files=42 bytes=12271321532 manifest_sha256=05f0b3dc… size_mismatch=0`；`VLA_VERIFY=PASS step=59999 files=42 sha_mismatch=0`
- concat：`VLA_DOWNLOAD=PASS step=59999 files=38 bytes=11976241540 manifest_sha256=bd35a857… size_mismatch=0`；`VLA_VERIFY=PASS step=59999 files=38 sha_mismatch=0`
- 冒烟：modulation `SMOKE2=PASS accepted=2 errors=0 seed=0 integration=modulation`（server 回显 readout `stage2_pool4_split_v1`、`num_tokens=4`、`dim=256`、`budget=384`）；concat `SMOKE2=PASS accepted=2 errors=0 seed=0 readout=stage2_pool4_concat_v1 budget=96`（回显 `integration_type=context`、`num_tokens=1`、`dim=1024`）
- 8 行 `COMPUTE_DONE completed=800 missing=0 errors=0 uncertain=0`（M-s42、M-s7、M-s0、C-s7、C-s0、CC-s42、CC-s7、CC-s0），`retries` 全为空
- 8 行 `SEED_ECHO=PASS records=800 identity_mismatch=0`，`spec_seed` 与 run 身份 seed 一致
- `THREE_SEED=PASS model=modul seeds=7,42,0 runs=3 overall=43.42±0.52`；`THREE_SEED=PASS model=ctx seeds=7,42,0 runs=3 overall=38.42±1.25`；`THREE_SEED=PASS model=ctxcat seeds=7,42,0 runs=3 overall=37.33±0.59`
- `HOLDS_RELEASED=PASS cancelled=10 squeue_empty=1`

## ② 逐任务三 seed 均值 ± std（%）

每任务每 seed 50 局，单 seed 单任务二项标准误约 6–7 个百分点。逐 seed 原值见 `records/three_seed_summary.txt` 与 `.json`。

| 任务 | modulation | context split | context concat |
|---|---|---|---|
| BinFill | 34.00 ± 5.29 | 21.33 ± 4.62 | 34.67 ± 1.15 |
| PickXtimes | 72.67 ± 4.16 | 67.33 ± 1.15 | 53.33 ± 3.06 |
| SwingXtimes | 95.33 ± 1.15 | 92.67 ± 5.77 | 83.33 ± 2.31 |
| StopCube | 64.67 ± 6.11 | 66.00 ± 8.72 | 43.33 ± 2.31 |
| VideoUnmask | 30.67 ± 1.15 | 30.67 ± 1.15 | 33.33 ± 1.15 |
| ButtonUnmask | 25.33 ± 5.03 | 31.33 ± 1.15 | 30.00 ± 6.00 |
| VideoUnmaskSwap | 24.67 ± 2.31 | 22.67 ± 3.06 | 37.33 ± 4.62 |
| ButtonUnmaskSwap | 20.67 ± 1.15 | 16.00 ± 2.00 | 27.33 ± 3.06 |
| PickHighlight | 28.00 ± 4.00 | 17.33 ± 1.15 | 20.00 ± 5.29 |
| VideoRepick | 24.00 ± 4.00 | 18.00 ± 0.00 | 22.67 ± 3.06 |
| VideoPlaceButton | 32.00 ± 2.00 | 37.33 ± 5.77 | 36.00 ± 2.00 |
| VideoPlaceOrder | 32.67 ± 4.62 | 18.67 ± 3.06 | 28.00 ± 5.29 |
| MoveCube | 84.00 ± 8.00 | 88.67 ± 3.06 | 82.67 ± 3.06 |
| InsertPeg | 10.67 ± 2.31 | 4.00 ± 3.46 | 2.00 ± 2.00 |
| PatternLock | 46.67 ± 5.03 | 32.00 ± 4.00 | 32.67 ± 1.15 |
| RouteStick | 68.67 ± 1.15 | 50.67 ± 3.06 | 30.67 ± 1.15 |
| **总体** | **43.42 ± 0.52** | **38.42 ± 1.25** | **37.33 ± 0.59** |

concat 与 split 总体接近，但任务分布不同：concat 在 Unmask/Swap 类（VideoUnmaskSwap +14.7、ButtonUnmaskSwap +11.3）与 BinFill（+13.3）更好，在 RouteStick（−20.0）、StopCube（−22.7）、PickXtimes（−14.0）、SwingXtimes（−9.3）上明显更差。

超时局数（每次 800 局）：modulation 25／49／55（seed 42／7／0），context split 31／21（seed 7／0；seed 42 为 28），concat 28／30／35。

## ③ 口径与 std 的说明

- seed 7／42／0 来自 benchmark 官方提交模板 `third_party/robomme_benchmark/doc/submission/model_example.md` 的表头与官方训练仓库副本 `compute_results.py` 的默认 `--seed_list seed0,seed42,seed7`；论文正文未列 seed，未从原文核实。
- seed 只控制模型采样噪声（server `--seed` → `policy.py` 中 `jax.random.key(seed)`）；环境局由 test 元数据固定，三个 seed 跑的是同一批 800 局。
- 官方 `compute_results.py` 默认还对 3 个 checkpoint（60000、70000、79999）取平均；本轮只评 59999，std 只含 seed 间方差，比论文少一维，与 Table 3 的 ± 不完全可比。
- context split 的 seed 42 复用 10-29 `o16ctx-60k-seed42/run800-r2`（执行代码 `9459859`），其余 8 次用 `5fce5d3`；两者只差 `eval.py` 的 seed 判断与一个新增测试文件。

## ④ 运行过程与计划外事件

时间均为美东；11-01 02:00 起夏令时结束，此后为 EST（UTC−5）。

| 运行 | 起跑 | `COMPUTE_DONE` |
|---|---|---|
| M-smoke（59999，seed 0，2 局） | 10-31 19:32 EDT | 约 19:58 |
| M-s42 | 19:59 EDT | 21:26 |
| M-s7 | 21:28 EDT | 22:59 |
| M-s0 | 23:00 EDT | 11-01 00:31 EDT |
| CC-smoke（59999，seed 0，2 局） | 00:31 EDT | 00:37 |
| C-s7 | 00:37 EDT | 01:05 EST |
| C-s0 | 01:06 EST | 02:34 EST |
| CC-s42 | 02:34 EST | 04:03 EST |
| CC-s7 | 04:04 EST | 05:31 EST |
| CC-s0 | 05:31 EST | 06:57 EST |

计划外事件与处置：

1. **开工令后用户改顺序**：用户「优先先跑 modulation，现在同样的 seed42，然后再跑其其他的两个。」计划修订为 modulation 优先、冒烟改用 59999（省掉 5000 下载），原开工令作废，修订版 `5db7311` 重新获「开工」。
2. **开工前一致性审查 F1**：原 `SEED_ECHO` 设计读逐局记录里的 `model_seed`，实测记录无此字段；改为 spec、run 身份与逐局 `run_identity_sha256` 三处比对（客户端 `validate_metadata` 另核 server 回显 seed），无需改代码。
3. **追加 concat1x1024**：用户「context concat1*1024加入计划」，修订 `62c9d9b`、获「开工」；concat 训练锚点 `6424d7f` 与执行基线 `9459859` 的 `src/` 只差注释，沿用 `5fce5d3`。用户另裁决「C-s0 跑完时 59999 还没上传 10 份占位直接取消」（未触发：59999 于 10-31 23:40 EDT 出现）。追加部分审查 `PLAN_CONSISTENCY_REVIEW=FAIL base=62c9d9b findings=7 blocking=1`，均为措辞，已统一于 `e43bca9`。
4. **控制器不自行退出**：每次 `COMPUTE_DONE` 后 sled-vail 上的控制器会话仍存活，由主会话（后为切换脚本）按会话名逐个 `tmux kill-session -t '=名'`，删前删后各查一次 `tmux ls`，差集均只含目标。
5. **换根提速**：从 M-s7→M-s0 起改用切换脚本 `switch-run.sh`（等上一次 `COMPUTE_DONE` → 起下一次控制器 → 每个 worker 退出即在同一占位起下一次 worker → 停上一次控制器），间隙 1–2 分钟；CC 冒烟插在 M-s0→C-s7 之间（`switch-with-smoke.sh`），C-s7 起即由 `chain-rest.sh` 串完剩余四次。
6. **CC-s42 的 02 号 worker 两次失败**：02 与 03 号占位同在 gl1513，02 号 server 绑定端口时 `OSError: [Errno 98] … address already in use`（`records/cc42-worker-02.fail1-port98.txt`）；同一 worker-id 重起被控制器拒绝 `新执行必须使用新的worker-id`（`records/cc42-worker-02.fail2-workerid.txt`）；改用 `worker-10` 在同一占位 65146922 重起，02:41 EST 恢复 `active_chunks=10`。两次失败都在领块之前，`retries` 为空、无局受影响；期间 9 个 worker 正常运行。
7. **控制器日志为空**：sled-vail 上 `controller-vail.log` 经 NFS 写入始终 0 字节；完成判定改以各 run 根 `queue/summary.json` 为准。

## ⑤ 资源

10 份 spgpu 占位 65146920–65146929（各 1×A40 / 4 CPU / 24 G / 48 h，10-31 18:58 EDT 提交），8 次运行与 2 次冒烟共用，11-01 约 07:00 EST 逐个 `scancel`，实际占用 12.7–13.0 小时／份，合计约 128 GPU·h。

本轮 tmux 会话（均已结束或已清理）：Aspen `ev3-fetch60k`、`ev3-verify60k`、`ev3-fetch-cc60k`、`ev3-verify-cc60k`、`ev3-w-smoke`、`ev3-w-ms42-00…09`、`ev3-w-ms7-00…09`、`ev3-w-ms0-00…09`、`ev3-w-ccsmoke`、`ev3-w-cs7-00…09`、`ev3-w-cs0-00…09`、`ev3-w-cc42-00…09`、`ev3-w-cc7-00…09`、`ev3-w-cc0-00…09`、`ev3-switch-ms7-ms0`、`ev3-switch-ms0-cs7`、`ev3-chain`；sled-vail `ev3-ctl-smoke`、`ev3-ctl-ms42`、`ev3-ctl-ms7`、`ev3-ctl-ms0`、`ev3-ctl-ccsmoke`、`ev3-ctl-cs7`、`ev3-ctl-cs0`、`ev3-ctl-cc42`、`ev3-ctl-cc7`、`ev3-ctl-cc0`。

## ⑥ 产物位置

- 逐局记录与视频：各 run 根 `accepted/*.json`、`attempts/*/client/`，run 根见 `launch.md` ③（concat 为 `$O/o16cc-60k-seed{42,7,0}/run800`，冒烟 `$O/o16cc-60k-seed42/smoke2-60k`；合同 `$O/o16cc-60k-seed42/contracts/runtime.json` sha256 `8e29a08b939db2cbadfa3176c50104e354326925b2387f65efff93bd843d065e`；run 身份 smoke `2c062d44…`、CC-s42 `38e4e94b…`、CC-s7 `9158218a…`、CC-s0 `cd036c7e…`）。
- 本机日志 `$L/`（各 worker、下载、校验、切换脚本）。

## ⑦ 下一步

无自动解锁的下游。可选：按官方口径再评 55000 等中间 checkpoint 补 checkpoint 维方差；或分析 concat 在 RouteStick／StopCube 上的显著退化。均需用户另行决定。

## ⑧ 归档文件清单（`records/`）

`three_seed_summary.txt`、`three_seed_summary.json`、`motion_windows_by_task.json`、`runs.txt`、`summary-<运行>.json`（10 份）、`fetch-modul-59999.log`、`verify-modul-59999.log`、`fetch-cc-59999.log`、`verify-cc-59999.log`、`switch-ms7-ms0.log`、`switch-ms0-cs7.log`、`chain-rest.log`、`cc42-worker-02.fail1-port98.txt`、`cc42-worker-02.fail2-workerid.txt`。
