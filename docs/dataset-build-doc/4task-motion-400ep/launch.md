# 4task-motion-400ep（4 任务 × 100 episode 完整库，环境 B / AWS 8×A100）构建留档 —— launch

> 目标：用与 40 ep 测试库**同一条链路**（`docs/dataset-build-doc/4task-motion-40ep-aws/launch.md`，链路零改动，只换 `--episodes-per-task 100`、worker 数、抽检条数、分片数）
> 构造 ButtonUnmask / ButtonUnmaskSwap / VideoUnmask / VideoUnmaskSwap 四任务全部 100 episode 的 SigLIP framesamp packed 库 + Wan latent + motion token 表；
> D2 / D3 oracle 8 卡分片全量逐位；另算一份 norm_stats 交付。库名 `v1-store/datasets/4task-motion-400ep`。结果见 [`result.md`](result.md)。

## 环境与 commit

- 环境 B（判定输出见 40ep-aws launch.md）；8×A100-SXM4-80GB；介质 AWS 本地 NVMe RAID（`/dev/md0`）。
- 原始 H5：`/scratch/hongze/robomme_data_h5/`（公开集 `Yinpei/robomme_data_h5` 四个目标任务，sha256 与环境 A 原件同源，见 40ep-aws result.md 一节）。
- **commit**：SigLIP 阶段与 finalize / pack 在 `8093ebd`（commitV6.12）；Wan 抽取 / encode / motion 表 / oracle 在 `e94285c`（`8093ebd` + `fix:` eval 驱动覆盖 `cbf24e9` + 文档 commit；
  `scripts/dataset/` 零改动）。为什么分两个 commit 见 result.md「意外」——第一次 Wan（`8093ebd`）与 encode（`cbf24e9`）跨 commit，`pack_motion_store.gather_provenance`
  的「跨 worker git_commit 唯一」校验拒绝，按纪律不绕过，在干净 HEAD 重抽 Wan。

## 命令序列（`LIB4=v1-store/datasets/4task-motion-400ep`，全部 `uv run --no-sync`，长任务 detached tmux + Monitor）

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA; source scripts/dataset/paths.sh; v1_prepare_dirs
LIB4=$V1_STORE/datasets/4task-motion-400ep; RAW=$RAW_H5_DIR; MJ=/scratch/hongze/MotionJEPA
# 清单（400 ep）+ 输入 sha256
uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir $RAW --tasks $TARGET_TASKS_CSV --episodes-per-task 100 --num_shards 1 --out $LIB4/meta/episode_manifest.json
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir $RAW --out $LIB4/meta/input_manifest.json
# SigLIP（GPU2,3,7 三 worker——起跑时 GPU0,1 / 4,5,6 被 T2 ref、t3mechanism、t3phase 占用；worker 数不影响字节）→ finalize（spot 1024）→ pack/verify（48 进程）
uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib $LIB4 --gpus 2,3,7 --raw-dir $RAW
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest $LIB4/meta/episode_manifest.json --out $LIB4/source --raw_dir $RAW --input_manifest $LIB4/meta/input_manifest.json --input_level sha256 --spot_check 1024
uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source $LIB4/source --manifest $LIB4/meta/episode_manifest.json --out $LIB4/framesamp --procs 48; … verify --store $LIB4/framesamp --resume --procs 48
# norm_stats（CPU；交付件，不覆盖测试用那份）
uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir $V1_STORE/train-assets/mme_vla_suite/robomme-400ep --config-name mme_vla_suite --repo-id robomme --dataset-path $LIB4/source
# Wan（GPU2–7 六 worker；GPU0,1 跑 T2 cand 重跑）→ encode（8 卡）→ motion 表
uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib $LIB4 --gpus 2,3,4,5,6,7 --raw-dir $RAW
uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib $LIB4 --gpus 0,1,2,3,4,5,6,7
uv run --no-sync python scripts/dataset/pack_motion_store.py pack --manifest $LIB4/meta/episode_manifest.json --tokens $LIB4/motion-tokens --latents $LIB4/wan-latents --out $LIB4/motion; … verify --store $LIB4/motion --resume
# D2 / D3 oracle（MotionJEPA venv；VAE 8 片 + aggregate；encoder 单进程 GPU7）+ 对拍
for i in 0..7: CUDA_VISIBLE_DEVICES=$i … oracle_driver.py --mj-repo $MJ vae --manifest $LIB4/meta/episode_manifest.json --raw-dir $RAW --latents $LIB4/wan-latents --out $LIB4/oracle/wan-mj --shard-idx $i --num-shards 8
… oracle_driver.py aggregate --manifest $LIB4/meta/episode_manifest.json --out $LIB4/oracle/wan-mj --num-shards 8 --kind vae
CUDA_VISIBLE_DEVICES=7 … oracle_driver.py --mj-repo $MJ encoder --manifest … --latents $LIB4/wan-latents --out $LIB4/oracle/wan-mj --expected-ckpt-sha256 bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a
uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents $LIB4/wan-latents --oracle $LIB4/oracle/wan-mj
uv run --no-sync python scripts/dataset/wan/compare_wan.py tokens --store $LIB4/motion --oracle $LIB4/oracle/wan-mj
# 附加检查：a6（--manifest-400ep 指同一份清单）/ a7 / a9set / a10（--expect-* 取 motion_index.json totals 现算值）/ a8 / a9enc；M1 全库（CPU）；dataloader_bench --lib $LIB4；
# test_padding_dtype.py 的 fixture 用例以 DTYPE_MANIFEST=$LIB4/meta/episode_manifest.json 跑通
```

## 与 40 ep 的口径差异（用户拍板）

- 不做 D1 全量 SigLIP oracle（40 ep 已两条 oracle 逐位、A100 跨卡逐位），`finalize_checks.py check --spot_check 1024`（40 ep 为 256）。
- D2 / D3 oracle **全量** 8 片（不抽样）。
- `motion_checks.py a10` 的期望行数不再写死 772/658/114，传 `motion_index.json` 的 `totals` 现算值。
- norm_stats 另算一份落 `train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json`（脚本按 `output_dir/<repo_id>` 落盘），测试用 `robomme/norm_stats.json`（`f332bbd3…`）不覆盖。
