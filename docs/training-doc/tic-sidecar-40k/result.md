# tic-sidecar-40k —— 结果

> 起跑 commit `fbdaf25`（clean HEAD；工具版本 commitV7.1 `a8cfa17`）。2026-09-07 21:20:52 → 21:38:29（17 min 37 s），环境 B，GPU 0（policy，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`）+ GPU 1（真 sidecar `MotionEncoderClient(online_gpu=1)`，就绪 17.9 s），tmux `tic-sidecar`（结束自行退出）。
> 原始日志 `records/obsmodel-sidecar.txt`，全部中间摘要 `records/obsmodel-sidecar.json`。
> **14 条阻断判据 13 条 PASS；唯一 FAIL 仍是 `VT_FULL_VS_CACHED`（与 `tic-obs-model-40k` 同一条、同一归因，不放宽，交用户裁决）。** 唯一成功行 `TIC_SIDECAR=FAIL … blocking=13/14`，`EXIT_CODE=1`。

## 一、判定行原文（5 集 120 个决策时刻，140 个 motion 窗）

```
[start] 2026-09-07T21:20:52+00:00 HEAD=56ac149074f8caa0cf04cb608093e0a222b16137
POINTS episodes=5 points=120 windows=140 c_mod16=[10, 12, 11, 4, 1] max_steps=1300
[tic] sidecar 就绪 17.9s gpu=1 warmup=3.4410064169205725
RAW_OBS_VS_PKL=PASS points=120 keys=[image,wrist_image,state,prompt_text] mismatches=none prompt_text_mismatch=0
MEM_S_VS_TRAINSET=PASS points=120 keys=8 key_mismatches=none
MEM_S_VS_B frames=2409 max_abs=1 mean_abs=0.007449 rel_fro=0.003328 cos_min=0.999976 cos_mean=0.999995 ulp_p50=2 ulp_p99=85.2 ulp_max=509 frac_nonzero=0.783
MEM_S_VS_A frames=2409 max_abs=1 mean_abs=0.006144 rel_fro=0.002957 cos_min=0.999982 cos_mean=0.999996 ulp_p50=1 ulp_p99=73 ulp_max=509 frac_nonzero=0.749
MEM_A_VS_B frames=2409 max_abs=1 mean_abs=0.007395 rel_fro=0.003309 cos_min=0.999977 cos_mean=0.999995 ulp_p50=2 ulp_p99=85 ulp_max=508 frac_nonzero=0.782
MOTION_S_VS_SIDECAR=PASS windows=140 mismatches=0 provenance_keys_equal=1 lib=4task-motion-400ep policy=create_trained_policy motion_client=MotionEncoderClient(online_gpu=1)
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
VT_FULL_VS_CACHED=FAIL points=120 rel_fro=0.003051 max_abs=0.03125 vt_rms=1.001 ulp_p99=22.5 ulp_max=384 thr_rel_fro=0.001 thr_ulp_p99=4.0 | prefix_kv tensors=240 rel_fro=0.004158 max_abs=2 ulp_p99_max=33 ulp_max=510 (prefix_kv 只统计 prefix_mask=True 的位) | note=threshold_pending_user_review(阈值为计划事先定死，实测超出属待裁决项，本脚本不自行放宽；结构三项全等，差异自前缀 pass 起，见 VT_FULL_VS_CACHED_F32)
VT_FULL_VS_CACHED_F32 points=5 rel_fro=1.352e-07 max_abs=7.153e-07 vt_rms=0.994 max_rel_p99=9.369e-06 max_rel_max=0.002407 prefix_kv_rel_fro=0 prefix_kv_max_abs=0 dtype=f32(params=restore_params(dtype=float32),embed_dtype=float32,matmul_precision=highest) bf16_ref_rel_fro=0.003051 points_per_episode=1 wall_s=21.7
ACT_T_VS_I=PASS points=120 seeds=3 mismatches=0 determinism_rerun=PASS
ACT_S_VS_B points=120 rms_norm=0.0004624 max_abs_norm=0.003906 rms_unnorm=0.0003147 | NOISE_S seeds=3 rms_norm=0.008326 rms_unnorm=0.009608 | ratio_norm=0.05554 | act_std_mean=0.2047 rms_unnorm/act_std=0.001538
AUG_EFFECT_OBS points=120 img_changed=120/120 mem_tokens_unchanged=1 act_rms_norm=0.005417
TIC_SIDECAR=FAIL episodes=5 points=120 motion=sidecar windows=140 blocking=13/14 observe=7
EXIT_CODE=1
```

## 二、结论

1. **真 sidecar 现算 == 离线表（阻断 PASS，本组新增）**：140 个 motion 窗由真 MotionJEPA 编码器（Wan VAE + encoder，`motion_provenance.json` 同源，`provenance_keys_equal=1`）在线现算，与 400 ep 库 `motion_token.f32.bin` 里的行**逐字节相同**（`MOTION_S_VS_SIDECAR=PASS windows=140 mismatches=0`）。这把 `aws-p5-online`（40 ep 库 772 行）的结论扩到生产库与生产 checkpoint 的真实推理路径上。
2. **其余 12 条阻断与 store 档结论一致**：输入八键、当前图 / 状态 / prompt 原文与 token、预处理后全部键、前缀 1184 四叶 / mask / 位置号 / 18 层 KV、`mem_order` 120/120 合法可逆、结构三项 120/120 全等、10 步去噪动作 120 点 × 3 seed 逐位相同——在真 sidecar 下整条 S 臂链仍与训练样本逐位一致。
3. **`VT_FULL_VS_CACHED=FAIL`**：bf16 rel_fro 3.05e-3、ulp_p99 22.5（store 档同一 checkpoint 同一输入为 1.86e-3 / 10——两次跑的差异即 XLA autotune 非确定性带，与开发阶段四次跑 1.9e-3–3.0e-3 一致）；`VT_FULL_VS_CACHED_F32` 在 f32 + `matmul_precision=highest` 下前缀 KV 逐位相同、v_t rel_fro 1.35e-7。归因与裁决口径见 `tic-obs-model-40k/result.md` 第三节，此处不重复。

## 三、观察项

`MEM_S_VS_B` / `MEM_S_VS_A` / `MEM_A_VS_B`、`MEM_SCALE_OBS`、`ACT_S_VS_B` / `NOISE_S`、`AUG_EFFECT_OBS` 数字与 store 档同量级（原文见上）；本组无 `ACT_CKPT_DTYPE`（仅 store 档）。

## 四、异常处置

无重跑；sidecar 随主进程结束退出，GPU 1 显存归零；tmux 会话随命令结束自行退出，未执行任何 kill。
