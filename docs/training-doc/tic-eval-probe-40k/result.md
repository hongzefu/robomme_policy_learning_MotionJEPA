# tic-eval-probe-40k —— 结果

> 起跑 commit `936eee7`（clean HEAD；工具版本 commitV7.1 `a8cfa17`）。2026-09-07 21:38:56 → 22:01:41（22 min 45 s；server 36 s 就绪，24 集仿真约 21 min），环境 B，policy + 仿真 GPU 0、sidecar GPU 1，端口 8123，tmux `tic-evalprobe`（结束自行退出）。
> 原始日志 `records/eval-probe.txt`（驱动）、`records/eval.txt`（主线 `eval.py` 全量输出）、`records/server.txt`（探针 server + sidecar，含 TIMING）、`records/probe.jsonl`（359 次 infer 的完整数组记录）、`records/eval_probe_summary.json`、`records/progress.json`。
> **7 条阻断 6 条 PASS；`EVAL_PROMPT=FAIL`——性质已查清：是训练数据对测试集 goal 组合的覆盖缺口，不是训练 / 推理链路不一致（第三节）。** 唯一成功行 `TIC_EVAL=FAIL … blocking=6/7`，`EXIT_CODE=1`。

## 一、判定行原文

```
EVAL_RC=0
EVAL_EPISODE_MAP=PASS resets=24 tasks=4 per_task=6 log_episodes=24 progress_entries=24 one_to_one=1 es_match=24/24 goal_match=24/24 seq_contiguous=1 resumed_skips=0 expect_episodes=24 expect_tasks=4
EVAL_TAU_K=PASS episodes=24 infer_points=359 tau_max=576 k_max=35 budget=96 headroom=61 es_values=0,66,114,168,216 tau_mismatches=0 k_over_budget=0 k_len_mismatch=0
EVAL_K_FORMULA=PASS points=359 mismatches=0 frames_sampled_mismatches=0 window=33 stride=16 max_frames=32
EVAL_ORDER_LEGAL=PASS points=359 nonperm=0 dtype_int32=1 len608=1 expected_order_mismatches=0 static_mask_bad=0 motion_mask_bad=0
EVAL_BACKEND=PASS backend=gpu pos_rows=586 pos_table_sha=74ced98dfb557281… store_pos_sha=74ced98dfb557281… equal=1 motion_enabled=1 motion_gpu_override=none devices=cuda:0
EVAL_PROMPT=FAIL tasks=4 episodes=24 distinct_prompt_text=16 train_distinct_prompt=26 train_distinct_tok=26 tok_in_trainset=345/359 text_in_trainset=345/359 discrete_state_input=0 tokenizer=PaligemmaTokenizer(max_len=64,strip,underscore,newline;no_lower)
EVAL_NO_RAISE=PASS episodes=24 errors=0 timeouts=0 unknown=0 log_error_lines=0 api_abort=0 server_tracebacks=0 handshake_noise=3 expect_timeout_infers=82 timeout_video_cross=0/0 videos_seen=24
EVAL_DIST_OBS online_k_median=9 mean=9.99 max=35 | train_k_median=9.0 mean=10.31 max=34 train_samples=101066 | online_tau_max=576 train_tau_max=585
TIC_EVAL=FAIL episodes=24 tasks=4 infer_points=359 blocking=6/7 observe=1
```

## 二、结论（六条 PASS）

1. **24 集一一对应**（`EVAL_EPISODE_MAP`）：探针 `episode_seq` 1–24 与 `eval.py` 日志的 (task, episode) 顺序、逐集 `exec_start_idx` 与探针 `es`、逐集 `task_goal` 与探针 `prompt_text` 三重全对；`progress.json` 24 条；无续评跳过。
2. **在线节奏与 motion 窗数**（`EVAL_TAU_K` / `EVAL_K_FORMULA`）：359 个决策时刻，`τ_max=576`、`k_max=35`（预算 96，余量 61），汇总器**独立重写**的 `demo(es)+exec(es,τ)` 公式与探针记录的 `motion_frames` 列表逐元素相同、`frames_sampled` 与 `even_sampling_indices(t,32)` 逐元素相同；在线 es 取值 {0,66,114,168,216} 与训练库一致。
3. **次序表合法**（`EVAL_ORDER_LEGAL`）：359 个 `mem_order` 全部是 608 位 int32 合法排列，与汇总器自己按 `key=2·时刻+类型` 稳定排序重算的结果逐元素相同；`static_mask` 前 16·n 位、`motion_mask` 前 k 位全 True。
4. **后端与位置表**（`EVAL_BACKEND`）：server 跑在 GPU（`cuda:0`），在线现算的 pos 表与训练库 pos 表前 586 行 sha 相同（`74ced98d…`）。
5. **零抛错零超时**（`EVAL_NO_RAISE`）：24 集 error 0、timeout 0（无一集到 82 次 infer；视频文件名交叉核 24/24 可见）、server 日志无真实 Traceback（3 个端口探测握手噪声已按块排除）。
6. **分布观察**（`EVAL_DIST_OBS`）：在线 k 中位 9 / 均值 9.99 / 最大 35，训练集 9 / 10.31 / 34；在线 τ_max 576 vs 训练 585——在线记忆的时间跨度与运动窗数落在训练分布内。
7. 顺带：24 集成功 4（ButtonUnmask 2/6、VideoUnmask 2/6、两个 Swap 0/6），单 seed 6 集/任务，**不作成功率结论**。

## 三、`EVAL_PROMPT=FAIL` 的性质：训练集缺一种 goal 组合

- FAIL 数字：`tok_in_trainset=345/359 text_in_trainset=345/359`，不在集合内的 14 个决策点全部来自 **同一集**（探针 `episode_seq=4` = `ButtonUnmask` 测试集 episode 3，es=0），prompt 原文：
  `first press the button, then pick up the container hiding the red cube, finally pick up another container hiding the green cube`
- 训练侧 400 集（4 任务 × 100）的 pkl `prompt` 共 26 种：ButtonUnmask 8 种、ButtonUnmaskSwap 9 种、VideoUnmask 9 种、VideoUnmaskSwap 9 种。ButtonUnmask 的 3 色 × (单目标 3 + 双目标 6) = 9 种组合里**唯独缺「red → green」**——即 100 集训练样本没覆盖到这一种双目标组合，而测试集抽到了它。
- **排除大小写 / tokenizer 差异**：在线 16 种 prompt 与训练 26 种全部本就是小写；两侧 `lower().strip()` 后该条仍不在集合内；tokenizer 两侧同一份（`TokenizePromptWithState/PaligemmaTokenizer`，`no_lower`）。组 B/C 在训练集 5 集上 `OBS_PROMPT=PASS`（120 点 token 全等）与此一致：链路没有改 prompt，只是这条 goal 训练时没见过。
- **定性**：这是 benchmark 测试集相对 100 ep/任务训练数据的**目标组合覆盖缺口**（泛化问题），不是「推理喂给模型的东西与训练不同」那类链路不一致；按计划它仍是阻断判据、保持 FAIL 记录。处置交用户：(a) 维持现状并在评估口径里注明该集为「训练未见 goal」；(b) 若要覆盖，需在 16 任务 1600 集全集或更多 ButtonUnmask 集里补含「red → green」的样本重建库——属数据口径改动，本轮不做。

## 四、异常处置

无重跑；探针 server 由驱动脚本 kill 自己起的 PID 收掉（GPU 0/1 显存归零）；tmux 会话随脚本结束自行退出，未执行任何 kill。`save_dir` 起跑前不存在（起跑器检查），全程单次运行、未续评。
