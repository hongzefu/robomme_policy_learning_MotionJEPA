# tic-obs-model-40k —— 结果

> 起跑 commit `9b3b95f`（clean HEAD；工具版本 commitV7.1 `a8cfa17`）。2026-09-07 21:05:25 → 21:20:10（14 min 45 s），环境 B，GPU 0（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`），tmux `tic-obsmodel`（结束自行退出）。
> 原始日志 `records/obsmodel-store.txt`，全部中间摘要 `records/obsmodel-store.json`。
> **13 条阻断判据 12 条 PASS；唯一 FAIL 是 `VT_FULL_VS_CACHED`——超过计划事先定死的阈值，本 run 不放宽，归因见第三节，是否按实测重定阈值交用户裁决。** 唯一成功行因此为 `TIC_OBS_MODEL=FAIL … blocking=12/13`，`EXIT_CODE=1`。

## 一、判定行原文（5 集 120 个决策时刻，140 个 motion 窗）

```
POINTS episodes=5 points=120 windows=140 c_mod16=[10, 12, 11, 4, 1] max_steps=1300
RAW_OBS_VS_PKL=PASS points=120 keys=[image,wrist_image,state,prompt_text] mismatches=none prompt_text_mismatch=0
MEM_S_VS_TRAINSET=PASS points=120 keys=8 key_mismatches=none
MEM_S_VS_B frames=2409 max_abs=1 mean_abs=0.007449 rel_fro=0.003328 cos_min=0.999976 cos_mean=0.999995 ulp_p50=2 ulp_p99=85.2 ulp_max=509 frac_nonzero=0.783
MEM_S_VS_A frames=2409 max_abs=1 mean_abs=0.006144 rel_fro=0.002957 cos_min=0.999982 cos_mean=0.999996 ulp_p50=1 ulp_p99=73 ulp_max=509 frac_nonzero=0.749
MEM_A_VS_B frames=2409 max_abs=1 mean_abs=0.007395 rel_fro=0.003309 cos_min=0.999977 cos_mean=0.999995 ulp_p50=2 ulp_p99=85 ulp_max=508 frac_nonzero=0.782
OBS_T_VS_I=PASS points=120 keys=all_equal mismatches=none train_only=['actions'] infer_only=[]
OBS_PROMPT=PASS points=120 text_equal=120 tok_mismatches=0 train_text='first press both buttons on the table, then pick up the container hiding the green cube, finally pick up another container hiding the blue cube' infer_text='first press both buttons on the table, then pick up the container hiding the green cube, finally pick up another container hiding the blue cube' tokenizer=TokenizePromptWithState/PaligemmaTokenizer(strip,underscore,newline;lower=no;discrete_state_input=False)
TRANSFORMS_CHAIN=PASS train_only=['RepackTransform'] infer_only=['InjectDefaultPrompt'] noop_proofs=2 inject_default_prompt_noop=1 repack_dropped_keys=['epis_idx', 'exec_start_idx', 'grounded_subgoal', 'grounded_subgoal_online', 'is_demo', 'simple_subgoal', 'simple_subgoal_online', 'step_idx'] repack_dropped_reach_model=0
PREFIX_SHAPE=PASS len=1184 mem=608 img=512 txt=64 img_per_view=256 views=2 want_mem=608
PREFIX_T_VS_I=PASS points=120 leaves=4 mismatches=none
PREFIX_POSITIONS=PASS positions_mismatches=0 attn_mask_mismatches=0 attn_mask_shape=(1, 1184, 1184) na_branch=1
MEM_INVPERM=PASS token_mismatches=0 mask_mismatches=0 perm_valid=120/120
KV_T_VS_I=PASS layers=18 k_mismatches=0 v_mismatches=0
MEM_SCALE_OBS points=118 ratio_mean=0.08106 min=0.07184 max=0.09954 a20_band=[0.3,3.0] in_band=0/118
FULL_VS_CACHED_STRUCT=PASS points=120 prefix_mask_block_equal=120/120 suffix_rows_equal=120/120 suffix_positions_equal=120/120 prefix_kv_bitexact_valid=0/120 prefix_kv_pad_positions_excluded=14716
VT_FULL_VS_CACHED=FAIL points=120 rel_fro=0.001855 max_abs=0.01953 vt_rms=1.001 ulp_p99=10 ulp_max=492 thr_rel_fro=0.001 thr_ulp_p99=4.0 | prefix_kv tensors=240 rel_fro=0.004158 max_abs=2 ulp_p99_max=33 ulp_max=510 (prefix_kv 只统计 prefix_mask=True 的位) | note=threshold_pending_user_review(阈值为计划事先定死，实测超出属待裁决项，本脚本不自行放宽；结构三项全等，差异自前缀 pass 起，见 VT_FULL_VS_CACHED_F32)
VT_FULL_VS_CACHED_F32 points=5 rel_fro=1.352e-07 max_abs=7.153e-07 vt_rms=0.994 max_rel_p99=9.369e-06 max_rel_max=0.002407 prefix_kv_rel_fro=0 prefix_kv_max_abs=0 dtype=f32(params=restore_params(dtype=float32),embed_dtype=float32,matmul_precision=highest) bf16_ref_rel_fro=0.001855 points_per_episode=1 wall_s=21.8
ACT_T_VS_I=PASS points=120 seeds=3 mismatches=0 determinism_rerun=PASS
ACT_S_VS_B points=120 rms_norm=0.0004596 max_abs_norm=0.004395 rms_unnorm=0.0003057 | NOISE_S seeds=3 rms_norm=0.008318 rms_unnorm=0.009596 | ratio_norm=0.05525 | act_std_mean=0.2047 rms_unnorm/act_std=0.001494
AUG_EFFECT_OBS points=120 img_changed=120/120 mem_tokens_unchanged=1 act_rms_norm=0.005437
ACT_CKPT_DTYPE points=1 ckpt_dtypes=['bfloat16', 'float32'] rms_norm=0.0006769 max_abs_norm=0.004085 noise_rms_norm=0.008318 ratio=0.08138 act_std_mean=0.2047 wall_s=25.4
TIC_OBS_MODEL=FAIL episodes=5 points=120 motion=store blocking=12/13 observe=8
EXIT_CODE=1
```

## 二、结论

按六关逐条：

1. **第 1 关 输入键（阻断 PASS）**：120 个决策时刻上，推理端 `_prepare_history` 拼出的记忆八键与 `FrameSampDataset[idx]` 交付逐字节相同（`MEM_S_VS_TRAINSET`）；当前两张图、状态与训练 pkl 逐位相同，任务描述原文与 pkl `prompt` 逐字相同（`prompt_text_mismatch=0`——这 5 集的 h5 `setup/task_goal` 本就全小写）。
2. **第 2 关 预处理后（阻断 PASS）**：两侧各过完自己那串变换后全部键逐位相同（`OBS_T_VS_I`，训练侧多出的只有 `actions`）；tokenizer 是 `mme_vla_suite.training.config.TokenizePromptWithState/PaligemmaTokenizer`（不做 lower），120 点 token 全等（`OBS_PROMPT=PASS`）；训练侧多出的 `RepackTransform` 丢掉的 8 个键都不进模型，推理侧多出的 `InjectDefaultPrompt` 是 no-op（`TRANSFORMS_CHAIN`）。
3. **第 3 关 模型内部（阻断 PASS）**：前缀 1184 = 608 记忆 + 512 图 + 64 文本；`embed_prefix` 四叶、attention mask `(1,1184,1184)`、位置号、18 层 K/V 缓存两侧逐位相同（`PREFIX_T_VS_I` / `PREFIX_POSITIONS` / `KV_T_VS_I`）；`mem_order` 120/120 合法排列且可原样还原（`MEM_INVPERM`）。
4. **第 4 关 整段前向 vs 缓存分步**：结构三项 120/120 全等（训练大表 `[:1184,:1184]` == 推理前缀表、末 20 行 == 推理 20 行 mask、动作段位置号相同，`FULL_VS_CACHED_STRUCT=PASS`）；**数值差 `VT_FULL_VS_CACHED=FAIL`**：bf16 生产精度下速度场 rel_fro 1.855e-3、ulp_p99 10（阈值 1e-3 / 4），前缀 KV 有效位自身 rel_fro 4.16e-3——差异在前缀 pass 就已产生，与动作段无关。归因见第三节。
5. **第 5 关 最终动作（阻断 PASS）**：同一噪声下两侧 10 步去噪的 20 步动作 120 点 × 3 seed 全部逐位相同，重跑确定（`ACT_T_VS_I`）。

**结论措辞（按计划收窄）**：在 checkpoint `awsprod40k-b128-motion/39999`、这 5 条训练集 episode 的 120 个决策时刻、明确隔离帧特征编码器精度 / 批形状（S 臂灌训练表真值）与整段 vs 缓存两路的 bf16 kernel 差之后，推理端喂给模型的输入、模型内部到 KV 缓存、以及最终动作，与训练端**逐位一致**；未发现额外的训练 / 推理不一致。

## 三、`VT_FULL_VS_CACHED` 超阈值的归因（不放宽、交用户裁决）

- 阈值 `rel_fro ≤ 1e-3`、`ulp_p99 ≤ 4` 是计划在「18 层每层 ≤1 ULP」假设下事先定死的；实测 bf16 每层量级远大于此。
- 结构三项全等 + 两路 obs / 权重 / `x_t` / `time` 相同，差异只能来自 `llm([prefix, suffix])`（1204 行 query、两个 expert 同场）与 `llm([prefix, None])`（1184 行）的 XLA kernel 选择与 bf16 归约序，18 层累积；四次独立跑 rel_fro 1.855e-3 / 1.915e-3 / 2.831e-3 / 3.04e-3，含 autotune 非确定性。
- **f32 诊断（`VT_FULL_VS_CACHED_F32`，每集 1 点）**：参数树 + `embed_dtype` 升 f32 且 `jax.default_matmul_precision("highest")` 后，**前缀 KV 逐位相同（rel_fro=0、max_abs=0）**，速度场 rel_fro 1.35e-7（f32 归约噪声量级）。开发阶段另测过 f32 但默认 TF32 matmul 的档位：rel_fro 4.87e-4。三档单调：bf16 1.9e-3 → TF32 4.9e-4 → 真 f32 1.4e-7。
- 因此差异是 **bf16 精度下 kernel 选择与归约序的纯数值来源，不是语义错误**；量级（0.19%）与已被用户接受的 SigLIP 编码器差异（`MEM_S_VS_B` rel_fro 0.33%）同级、对最终动作的影响需与 `ACT_S_VS_B`（动作 RMS 为采样噪声的 5.5%）同口径看。**阈值是否按实测重定属放宽判据，本 run 保持 FAIL 记录，交用户裁决**；若重定，建议按四次跑波动带（≈1.6×）留裕量。

## 四、观察项

- `MEM_S_VS_B`（推理真现算 bf16 帧特征 vs 训练表 f32）：2409 帧 rel_fro 0.333%、cos_min 0.999976、ulp_p50 2——与 `siglip-ab-replay-40k`（0.34%）一致；`MEM_S_VS_A` 0.296%、`MEM_A_VS_B` 0.331%，dtype 与批形状两个来源同量级。
- `ACT_S_VS_B`：帧特征差导致的动作差 rms_norm 4.6e-4，为采样噪声 `NOISE_S` 8.3e-3 的 5.5%；反归一化后 3.1e-4，为动作 std 的 0.15%。
- `MEM_SCALE_OBS`：A20 口径 ‖motion‖/‖frame‖ = 0.081（0.072–0.100），低于 band [0.3, 3.0]，118/118 越界上报；训练前记录值 0.166（`motion-t3-*`），训练后降至 0.08，本文不作效果解读。
- `AUG_EFFECT_OBS`：`preprocess(train=True)` 增广 120/120 改了图像、记忆 token 不变，动作 rms 5.4e-3。
- `ACT_CKPT_DTYPE`：checkpoint 盘上混合 bf16/f32，`dtype=None` 与生产 `dtype=bf16` 加载的动作 rms_norm 6.8e-4，为采样噪声的 8.1%。

## 五、异常处置

无重跑；tmux 会话随命令结束自行退出，未执行任何 kill。开发阶段（GPU 4，`--max-points 1/2`）四次跑的判定行在 `v1-store/reports/tic-dev/obsmodel-*.log`（不进 git），结论与本 run 一致。
