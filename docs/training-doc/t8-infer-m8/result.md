# M8：8帧×8×8测试checkpoint推理验收结果

七关与48集闭环全部完成，所有阻断项通过，三个阶段EXIT_CODE均为0。实际前缀长度1184；关1–5为14/14，一集探针为7/7，闭环48/48、errors=0、最大推理次数82。闭环任务成功0/48（0.00%），属于1000更新测试模型记录，不作收敛或能力指标。

## 版本、权重和启动

权重为v1-store/train-runs/t8-m8-b/mme_vla_suite/t8-m8-b/999，是该run完成1000次更新后的EMA。候选模型代码为c08ec2060a544af1869c1e24f755e536150569ca；实际各阶段启动HEAD如下。第一份C8开环从a0cdf58启动，随后只改启动文档，将M8开环/探针安排到GPU6，与C8的GPU7并行。两组内部同卡比较，48集仍依次按M8→C8使用GPU4–7。

完整命令、env、调度沿革和会话名单见[launch.md](launch.md)及[共同启动口径](../t8-infer/launch.md)。轨迹、全梯度和输入证据分别见[训练总览](../t8-training/result.md)、[全梯度](../t8-gradient/result.md)、[fixture](../t8-fixture/result.md)。

| 阶段 | 实际启动HEAD | UTC开始 → 结束 | 墙钟 |
|---|---|---|---:|
| 关0–5 | a0db6bf53fb5b46b0cb3f0dcbbbe53c670866496 | 2026-09-14T12:32:21Z → 2026-09-14T12:56:51Z | 24分30秒 |
| 一集探针/关6 | a0db6bf53fb5b46b0cb3f0dcbbbe53c670866496 | 2026-09-14T12:58:25Z → 2026-09-14T13:02:11Z | 3分46秒 |
| 48集闭环 | a0db6bf53fb5b46b0cb3f0dcbbbe53c670866496 | 2026-09-14T13:04:33Z → 2026-09-14T13:37:06Z | 32分33秒 |

数据为400ep库及已全量verify的新framesamp-8x8；原norm_stats文件SHA为750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2，并逐项验证实际解析的8数组相同。模型从checkpoint的history_config快照恢复512帧token预算、每帧64 token及motion开关。MV驱动按该配置设置adapter.P，保持原采样实现。

## 关0：配置和库同源

```text
NORM_STATS_SAME=PASS sha=41ff90e4bfacc158 train_src=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json infer_src=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/t8-m8-b/mme_vla_suite/t8-m8-b/999/assets/robomme/norm_stats.json keys=[actions,state] arrays=8
LIB_PROVENANCE_MATCH=PASS framesamp_manifest_sha256=92fa17e97fba9434 motion_manifest_sha256=92fa17e97fba9434 framesamp_store_meta_sha256=31a6c2a4fc3807bd motion_index_sha256=74185921690cd26c motion_store_meta_sha256=e19fb5fdb45829d8 motion_table_sha256=6e70604da518c156 all_equal=1
MOTION_STORE_PATH=PASS snapshot=v1-store/datasets/4task-motion-40ep/motion provenance=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/motion effective=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/motion override=MMEVLA_MOTION_STORE guard=raise infer_reads_store=0 same_manifest=1 snapshot_default=raise
CKPT_PARAM_TREE=PASS missing=0 extra=0 leaves=59 model_leaves=59 ckpt_has_motion=1
CKPT_DTYPE_PROFILE f32_leaves=36 bf16_leaves=23 img_all_bf16=1 trainable_all_f32=1 by_dtype={'bfloat16': 23, 'float32': 36} img_dtypes=['bfloat16']
TIC_L0=PASS
```

快照中的motion.store_path保留40ep默认值，实际训练以MMEVLA_MOTION_STORE覆盖400ep并写入provenance。关0同时验证正确覆盖通过、无覆盖的错配被拒绝；在线侧不读取离线motion表，而是按同一模型provenance起真实sidecar。

## 关1–5：输入、前缀、缓存和动作

使用原定5集120个决策点，覆盖es=0、66、114、168、216；每点3组噪声。强一致比较采用S臂：以训练库值替换在线历史帧的图像特征，保留真实装配和预处理；真实bf16帧编码的B臂差异单列观察。本模型运动分支由真实sidecar计算，140个运动窗与离线表逐位一致。以下保留全部阻断判定，以及bf16观察项的真实FAIL标签。

```text
RAW_OBS_VS_PKL=PASS points=120 keys=[image,wrist_image,state,prompt_text] mismatches=none prompt_text_mismatch=0
MEM_S_VS_TRAINSET=PASS points=120 keys=8 key_mismatches=none
MOTION_S_VS_SIDECAR=PASS windows=140 mismatches=0 provenance_keys_equal=1 lib=4task-motion-400ep policy=create_trained_policy motion_client=MotionEncoderClient(online_gpu=6)
OBS_T_VS_I=PASS points=120 keys=all_equal mismatches=none train_only=['actions'] infer_only=[]
OBS_PROMPT=PASS points=120 text_equal=120 tok_mismatches=0 train_text='first press both buttons on the table, then pick up the container hiding the green cube, finally pick up another container hiding the blue cube' infer_text='first press both buttons on the table, then pick up the container hiding the green cube, finally pick up another container hiding the blue cube' tokenizer=TokenizePromptWithState/PaligemmaTokenizer(strip,underscore,newline;lower=no;discrete_state_input=False)
TRANSFORMS_CHAIN=PASS train_only=['RepackTransform'] infer_only=['InjectDefaultPrompt'] noop_proofs=2 inject_default_prompt_noop=1 repack_dropped_keys=['epis_idx', 'exec_start_idx', 'grounded_subgoal', 'grounded_subgoal_online', 'is_demo', 'simple_subgoal', 'simple_subgoal_online', 'step_idx'] repack_dropped_reach_model=0
PREFIX_SHAPE=PASS len=1184 mem=608 img=512 txt=64 img_per_view=256 views=2 want_mem=608
PREFIX_T_VS_I=PASS points=120 leaves=4 mismatches=none
PREFIX_POSITIONS=PASS positions_mismatches=0 attn_mask_mismatches=0 attn_mask_shape=(1, 1184, 1184) na_branch=1
MEM_INVPERM=PASS token_mismatches=0 mask_mismatches=0 perm_valid=120/120
KV_T_VS_I=PASS layers=18 k_mismatches=0 v_mismatches=0
FULL_VS_CACHED_STRUCT=PASS points=120 prefix_mask_block_equal=120/120 suffix_rows_equal=120/120 suffix_positions_equal=120/120 prefix_kv_bitexact_valid=120/120 prefix_kv_pad_positions_excluded=14428
VT_FULL_VS_CACHED=FAIL points=120 rel_fro=0.004911 max_abs=0.05469 vt_rms=1.064 ulp_p99=40 ulp_max=484 thr_rel_fro=0.001 thr_ulp_p99=4.0 | prefix_kv tensors=240 rel_fro=0 max_abs=0 ulp_p99_max=0 ulp_max=0 (prefix_kv 只统计 prefix_mask=True 的位) | note=用户已裁决为bf16观察项；阻断条件见VT_FULL_VS_CACHED_F32
VT_FULL_VS_CACHED_F32=PASS points=15 rel_fro=3.22e-07 max_abs=3.338e-06 vt_rms=1.104 max_rel_p99=1.441e-05 max_rel_max=0.0005258 prefix_kv_rel_fro=0 prefix_kv_max_abs=0 dtype=f32(params=restore_params(dtype=float32),embed_dtype=float32,matmul_precision=highest) bf16_ref_rel_fro=0.004911 points_per_episode=3 wall_s=34.3
ACT_T_VS_I=PASS points=120 seeds=3 mismatches=0 determinism_rerun=PASS
TIC_SIDECAR=PASS episodes=5 points=120 motion=sidecar windows=140 blocking=14/14 observe=7
```

f32重新加载参数、设置embed_dtype=float32和matmul_precision=highest，每集3点，共15点；速度场rel_fro=3.21997612971e-07≤1e-6。整段/缓存对照的有效前缀KV最大差为0，沿用既有规则排除padding位置；普通120点分别记录了该排除数量。T/I两份observation的18层KV比较均逐位相同。

bf16速度场rel_fro=0.0049109498271，按用户裁决只作观察，没有改阈值或改成PASS。第5关两侧调用同一个sample_actions路径，因此动作逐位一致证明的是输入交付一致；不意味着训练整段前向与缓存采样在bf16下逐位相同。离线特征与checkpoint编码的精度差异也保留在[完整开环报告](records/checkpoint/gates/tic.json)中。

## 关6：一集真实仿真

ButtonUnmask test episode0，model_seed42、env_seed580000。实际结果timeout，环境步1301、推理82次、tau最大1296、运动窗最大80。policy server在仿真正常退出后由本轮监督进程按自己的进程组主动终止，日志记SERVER_CONTROLLED_STOP；其后汇总器和整体任务均退出0。

```text
EVAL_EPISODE_MAP=PASS resets=1 tasks=1 per_task=1 log_episodes=1 progress_entries=1 one_to_one=1 es_match=1/1 goal_match=1/1 seq_contiguous=1 resumed_skips=0 expect_episodes=1 expect_tasks=1
EVAL_TAU_K=PASS episodes=1 infer_points=82 tau_max=1296 k_max=80 budget=96 headroom=16 es_values=0 tau_mismatches=0 k_over_budget=0 k_len_mismatch=0
EVAL_K_FORMULA=PASS points=82 mismatches=0 frames_sampled_mismatches=0 window=33 stride=16 max_frames=8
EVAL_ORDER_LEGAL=PASS points=82 nonperm=0 dtype_int32=1 len608=1 expected_order_mismatches=0 static_mask_bad=0 motion_mask_bad=0
EVAL_BACKEND=PASS backend=gpu pos_rows=586 pos_table_sha=8ff424a7fc097ef2… store_pos_sha=8ff424a7fc097ef2… equal=1 motion_enabled=1 motion_gpu_override=6 devices=cuda:0
EVAL_PROMPT=PASS tasks=4 episodes=1 distinct_prompt_text=1 train_distinct_prompt=26 train_distinct_tok=26 tok_in_trainset=82/82 text_in_trainset=82/82 discrete_state_input=0 tokenizer=PaligemmaTokenizer(max_len=64,strip,underscore,newline;no_lower)
EVAL_NO_RAISE=PASS episodes=1 errors=0 timeouts=1 unknown=0 log_error_lines=0 api_abort=0 server_tracebacks=0 handshake_noise=0 expect_timeout_infers=82 timeout_video_cross=1/1 videos_seen=1
EVAL_DIST_OBS online_k_median=39.5 mean=39.51 max=80 | train_k_median=9.0 mean=10.31 max=34 train_samples=101066 | online_tau_max=1296 train_tau_max=585
TIC_EVAL=PASS episodes=1 tasks=1 infer_points=82 blocking=7/7 observe=1
```

一集probe只覆盖该集prompt。旧400ep训练未见ButtonUnmask test3、7、19、23对应目标组合，本轮不补数据；下方48集保留3、7，不能把这一集的prompt通过外推到48集全部目标。

## 48集闭环

使用motion-variance原驱动，cond=normal、test、seed42、WORKERS=4、GPU_LIST=4,5,6,7、EP_COUNT=10；stride分片合并后为四任务各0..11共12集，总计48。所有worker加载本轮999 checkpoint，DIRTY_TREE=0，逐片完成12/12且退出0。逐次探针核对网格、motion开关、次序表、窗口公式及实际编码调用次数。

```text
MV_BATCH=PASS cond=normal split=test seed=42 episodes=48/48 workers=4/4 wall_min=32 sidecar=on cross_seg=0 suffix=-t8m8 end=2026-09-14 13:37:05
EVAL_SMOKE=DONE profile=m8 episodes=48/48 errors=0 max_infer=82 mem_order_ok=1 success_rate=测试模型不作指标
EXIT_CODE=0
INFER_PHASE=PASS profile=m8 phase=batch utc=2026-09-14T13:37:06.225382+00:00
EXIT_CODE=0
```

实际共3936次推理，单集最大82次、最大环境步1301、最大运动窗92，errors=0。结果分布为{"timeout": 48}。逐集记录保存在各worker日志与progress.json中，合并摘要见[闭环验收](records/checkpoint/batch/acceptance.json)。

| 任务 | 成功/总集数 | 只作记录的成功率 |
|---|---:|---:|
| ButtonUnmask | 0/12 | 0.00% |
| VideoUnmask | 0/12 | 0.00% |
| ButtonUnmaskSwap | 0/12 | 0.00% |
| VideoUnmaskSwap | 0/12 | 0.00% |

这些成功率来自1000更新测试模型和单seed子集，不用于比较C8与M8的模型能力，也不作为本轮链路通过条件。

## 显存、主机内存和复现边界

| 阶段 | GPU峰值显存 | 本轮进程树RSS峰值 |
|---|---|---:|
| 关0–5 | GPU6: 49983 MiB | 52.81 GiB |
| 一集探针/关6 | GPU6: 49175 MiB | 20.80 GiB |
| 48集闭环 | GPU4: 49175 MiB；GPU5: 49175 MiB；GPU6: 49175 MiB；GPU7: 49175 MiB | 63.96 GiB |

硬件为AWS A100-SXM4-80GB，底层存储为本地NVMe RAID /dev/md0。GPU和host按500ms采样；host为本轮进程树RSS之和，共享页可能重复计数。显存包含JAX预分配、模型、sidecar及仿真，不能当作位置表独占增量。8×8在线4096行f32位置表本身为768MiB。这里只记录验证成本，不由包含编译、摘要和仿真的墙钟推导稳态吞吐。

GPU进程归属另有实测placement记录。会话只使用本轮登记的t8-infer-*与mv-t42-normal-t8*；其他用户会话未清理。本轮模型与API运行异常为0，任务超时或失败按实测结果单独记录。

## 用户决定与归档

用户要求“一口气全做完！”、“注意你只能用4个gpu”；采用“文件 SHA＋解析后数组摘要双重检查”，并同意“让评估脚本读取 checkpoint 保存的配置，自动算出正确长度”。bf16整段/缓存差为观察项、f32为阻断、闭环48集与成功率不作指标均沿用已确认计划。

records/checkpoint按gates、probe、batch和workers保留JSON报告、清洗日志、逐次探针、progress/log.json及500ms资源记录；jsonl/csv使用gzip无损压缩。清洗日志剔除进度中间态和行尾空白，原始v1-store日志保留；source-sha256.json记录清洗规则及原始、归档文件SHA；大视频、模型权重及checkpoint仍在v1-store，不复制进Git。共用真实8×8池化/位置/装配证据见[M8先行检查](../t8-infer-m8/result.md)。

## 先行检查：真实8×8位置表、编码与装配


于2026-09-14 03:36:05–03:38:55 UTC运行，耗时2分50秒，启动HEAD为 `c8aba47eef6c31266df985ff5c7a675418522df1`，工作区干净；仅使用物理GPU7。使用真实H5帧和真实SigLIP，同一个jitted编码器供三方使用，无stub。52个实际输入帧覆盖17个边界时刻。

```text
POS_TABLE=PASS rows=4096 a==b=True a==c=True
ENC_LAYER_8X8=PASS steps=17 keys=3 mismatch=0
ASSEMBLY=PASS steps=17 mismatch=0
OOB_PROBE=PASS a=empty-slice-detected b=empty-slice-detected c=raise
ONLINE_MEM=PASS base=732fae3b
EXIT_CODE=0
```

三类特征为8×8 image、8×8 pos与state。4096行位置表全量逐位一致，17个时刻的编码和装配均零差异，越界由候选显式拒绝。这个检查同时覆盖C8/M8共用的帧路径，后续无需重复；它没有替代真实checkpoint上的动作、motion或闭环验收。

这项先行检查完成后，本轮数据、训练、全梯度、七关与闭环也已完成。`records/online_memory/`保存逐步摘要、判定行及清洗日志。
