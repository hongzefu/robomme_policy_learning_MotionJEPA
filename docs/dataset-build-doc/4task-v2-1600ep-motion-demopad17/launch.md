# 1600 集 demo 补帧 motion 表：起跑记录

用户原话：「开始实施 有问题尽早问用户」；本轮方案见 [0916 计划](../../../0916-motion-modul-8x8-plan.md)。本档案只构建 motion 所需的 Wan latent、token、整表与独立 oracle，不重建已有 SigLIP/framesamp，不重算 norm_stats。

## 版本与输入

起跑版本由外壳参数 BUILD_HEAD 的完整 SHA 固定，并在每个阶段起止核对 HEAD 与空 porcelain。实际 SHA、时间和命令写入阶段日志，绝不把事后文档提交冒充启动版本。输入重锚已在 df6fdcc4f9424f00901a107a66d98e19e679a7ba 完成，2026-09-17 21:26:55–21:40:20 UTC，805 秒，四个文件 SHA 同源、EXIT_CODE=0；来源、帧表元数据和 norm_stats 三个受保护文件摘要未变。用户在此期间确认了位置表缓存、现有单卡仿真环境及 V5 兼容权重初始化三项验证方案，因此先提交验证工具修补，再从新 clean BUILD_HEAD 启动 Wan。Wan 至 pack/compare 保持同一提交，Wan 与 encode 之间零 commit。

代码数值入口为 scripts/dataset/wan/wan_motion_infer.py 的钉版复制件，SHA256 为 af67fdd913543aee416a9fe5df797f707ff159165c468d687fcf8e7347941b34；SOURCE_PIN 推理提交为 2a484ad960ed6155321dc34def9011eb119f857f。独立 oracle 使用 /scratch/hongze/MotionJEPA 的原版入口，脚本 SHA 与复制件相同。encoder 的训练提交则为 660cee10a86d02ae03a73db95fac6f0b8dbd28a6，两种提交含义不能混同。

源库为 v1-store/datasets/4task-v2-1600ep-604f16da，原始 H5 为 v1-store/raw-h5/4task-20260912-v2。清单 SHA 为 4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918，4 任务 × 400 集，共 605611 个执行样本。

encoder 使用 v1-store/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt，477432433 字节，SHA256 0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca；config.yaml 为 2008 字节，SHA256 4a505440b7c5ff0f1b0d7ec4680800e9296f5ede1df622767be3d5c3aee4c9b4。HF 新文件尚未上传，本轮只使用本地核验原件。

## 契约与独立判据

新布局 motion-768-grid16-demopad17-v1，demo 最少 17 个真实帧，重复第 es-1 帧补齐 33；exec 最少 33 帧、不补尾。预期 35913 个 demo 窗 + 35403 个 exec 窗 = 71316 行，补帧窗 1600 个；共 3200 个非空段。预期 latent 为 42063888384 字节，token 表为 219082752 字节，均为 fp32 裸字节。

用户原话「采用建议顺序，保留三方强校验」：顺序为输入重锚 → Wan → encode → oracle 重算及汇总 → pack/verify → 数值 compare → a6set/a7/a9set/a10/a11。pack 前 oracle 报告必须已存在。

用户原话「采用真实 skipped 计数及完整性验收」：不预设 skipped=0。记录 initial_complete=0，并要求本轮处理集合与完整段集合相等、无重复、无遗漏。计数与集合不能互相替代。

D3 encoder 全部 71316 行逐位比较；D2 只抽 VAE 前向，补帧全部、其余 10%、seed 0。选取规则为 SHA256("seed:key:m") 前八字节按大端解释，模 10000 后小于 1000；比较器独立重算选取集合，不能跟随同一缩水列表。段集合、行号、起点、输入形制及全部 71316 窗的输入 SHA 都完整检查。未抽中 oracle 行写零，只用于保持定长，绝不作为数值真值。

## 环境与启动

环境 B，AWS 本地 NVMe RAID /dev/md0（XFS），8 × A100-SXM4-80GB。运行前重新检查至少 400 GiB 可用空间、各 GPU 占用和输出目录归属。新 wan-latents 必须为空，motion-tokens/_claims 必须无残留，不覆盖已有 oracle 或 motion 整表。

主 venv 负责调度和打包；Wan/encode 使用 v1-store/venvs/wan；oracle 使用 MotionJEPA 的 uv 环境。后两者实测同为 torch 2.9.0+cu128、diffusers 0.39.0、numpy 2.4.4。全部通过 uv --no-sync 运行，不修改依赖。

环境覆盖：UV_CACHE_DIR=v1-store/cache/uv，UV_NO_SYNC=1，UV_LINK_MODE=copy，PYTHONUNBUFFERED=1，HF_HOME=v1-store/cache/hf，HF_HUB_OFFLINE=1，XDG_CACHE_HOME=v1-store/cache/xdg，JAX_COMPILATION_CACHE_DIR=v1-store/cache/jax，OPENPI_DATA_HOME=v1-store/models，CUDA_CACHE_PATH=v1-store/cache/cuda，OMP_NUM_THREADS=1，OPENBLAS_NUM_THREADS=1。NVIDIA_TF32_OVERRIDE、TORCH_ALLOW_TF32_CUBLAS_OVERRIDE、CUBLAS_WORKSPACE_CONFIG 均 unset，不设置 HOME。Wan/encode 的 UV_PROJECT_ENVIRONMENT 指向上述子 venv；oracle 加 PYTHONDONTWRITEBYTECODE=1。

Wan/encode 各占 GPU 0–7；D3 使用 GPU 7；D2 八片分别使用 GPU 0–7。只读输入重锚使用 CPU，不调用会覆写帧库 provenance 的 finalize check 主命令。调用 finalize_checks.check_inputs(..., level="sha256") 后只打印结果。

完整外壳为 v1-store/logs/mv2-build-runner.sh，五档参数为 input/wan/encode/oracle/pack。各阶段外层 pipefail + tee，记录 EXIT_CODE；D2 每片独立 tee 日志并记录 SHARD_EXIT_CODE，全部成功才 aggregate，aggregate 成功才 pack/compare。日志为 <lib>/logs/mv2-<stage>.driver.log，各片为 mv2-oracle-vae-<i>.log。

本轮建库 tmux 全名：mv2-input、mv2-wan、mv2-encode、mv2-oracle、mv2-pack。清理仅限这些实际创建的会话，完整名称加 = 精确匹配；不会触碰用户其他会话。

## 结束后归档

保留阶段清洗日志、分片退出码、元数据与 pin 摘要、所有独立判定行和实测耗时；数据与模型留在 v1-store。本页不填写尚未执行的结果，结束后另写 result.md。

## 完整阶段外壳

资产切换前后核验已完成：复制后的 checkpoint/config 全量 SHA256 与计划相同；`uv run --no-sync pytest scripts/assets/test_assets_lock.py -q` 28 项通过（0.22 秒）。在 Wan 子环境的 GPU 6 上实际加载新 encoder 并前向，结果为 `NEW_ENCODER=PASS state_tensors=77 output=768 dtype=float32 finite=1 epoch=72`。旧资产文件保留，历史表的测试按旧 run 指纹核对；当前资产锁切换后，仍调用旧默认 encoder 的历史工具不适用于新库，不能据此混用新旧权重。

以下代码块与本轮生成的外壳一致；启动时传入阶段名和已核实的完整 BUILD_HEAD。

```bash
#!/usr/bin/env bash
# 完整建库阶段外壳；每阶段自证锚点，Wan 到打包/对拍保持同一 clean BUILD_HEAD。
set -uo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA || exit 1
STAGE="${1:?必须指定阶段}"
BUILD_HEAD="${2:?必须指定完整提交}"
case "$STAGE" in input|wan|encode|oracle|pack) ;; *) exit 2 ;; esac
LIB="$PWD/v1-store/datasets/4task-v2-1600ep-604f16da"
LOG="$LIB/logs/mv2-$STAGE.driver.log"
test ! -e "$LOG" || { printf '拒绝覆盖阶段日志 %s\n' "$LOG"; exit 2; }
body() (
  set -euo pipefail
  test "$(git rev-parse HEAD)" = "$BUILD_HEAD"
  test -z "$(git status --porcelain)"
  source scripts/dataset/paths.sh
  export UV_CACHE_DIR="$V1_STORE/cache/uv" UV_NO_SYNC=1 PYTHONUNBUFFERED=1
  export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  unset NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE CUBLAS_WORKSPACE_CONFIG
  RAW="$V1_STORE/raw-h5/4task-20260912-v2"
  ENC="$V1_STORE/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a"
  MJ=/scratch/hongze/MotionJEPA
  SHA=0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca
  test -d "$LIB" && test ! -L "$LIB"
  test -d "$RAW" && test ! -L "$RAW"
  test -d "$ENC" && test ! -L "$ENC"
  printf 'BUILD_HEAD=%s\nSTAGE=%s\nSTART_UTC=%s\nLIB=%s\nRAW=%s\nENC=%s\n' "$BUILD_HEAD" "$STAGE" "$(date -u +%FT%TZ)" "$LIB" "$RAW" "$ENC"
  require_stage() { test "$(tail -n 1 "$LIB/logs/mv2-$1.driver.log")" = 'EXIT_CODE=0'; }
  case "$STAGE" in
    input)
      uv run --no-sync python - "$LIB" "$RAW" <<'PY'
import pathlib,sys,json
sys.path.insert(0,'scripts/dataset')
from finalize_checks import check_inputs
from mme_vla_suite.datastore.manifest import load_manifest
lib,raw=map(pathlib.Path,sys.argv[1:]); manifest=load_manifest(lib/'meta/episode_manifest.json')
assert pathlib.Path(manifest['raw_dir']).resolve()==raw.resolve()
errors=check_inputs(manifest,str(raw),str(lib/'meta/input_manifest.json'),level='sha256')
if errors: raise RuntimeError(errors)
assert len(manifest['canonical_order'])==4
print('INPUT_REANCHOR=PASS files=4',flush=True)
PY
      ;;
    wan)
      require_stage input
      uv run --no-sync python - "$LIB/wan-latents" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1])
assert not p.exists() or not any(p.iterdir()),'新建库必须使用空 wan-latents'
PY
      uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW"
      ;;
    encode)
      require_stage wan
      uv run --no-sync python - "$LIB/motion-tokens/_claims" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]); assert not p.exists() or not any(p.iterdir()),'存在残留 claim'
PY
      uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --encoder-run-dir "$ENC" --expected-ckpt-sha256 "$SHA"
      ;;
    oracle)
      require_stage encode
      ORACLE="$LIB/oracle/wan-mj"
      test ! -e "$ORACLE" || { printf '拒绝复用已有 oracle 目录\n'; exit 2; }
      mkdir -p "$ORACLE"
      CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 uv run --project "$MJ" --no-sync python "$REPO_ROOT/scripts/dataset/wan/oracle_driver.py" --mj-repo "$MJ" encoder --manifest "$LIB/meta/episode_manifest.json" --latents "$LIB/wan-latents" --out "$ORACLE" --encoder-run-dir "$ENC" --expected-ckpt-sha256 "$SHA" --demo-min-real 17 --exec-min-real 33
      pids=()
      for i in 0 1 2 3 4 5 6 7; do
        (
          set +e
          set -o pipefail
          CUDA_VISIBLE_DEVICES="$i" PYTHONDONTWRITEBYTECODE=1 uv run --project "$MJ" --no-sync python "$REPO_ROOT/scripts/dataset/wan/oracle_driver.py" --mj-repo "$MJ" vae --manifest "$LIB/meta/episode_manifest.json" --raw-dir "$RAW" --latents "$LIB/wan-latents" --out "$ORACLE" --shard-idx "$i" --num-shards 8 --demo-min-real 17 --exec-min-real 33 --sample-spec 'padded:all,rest:0.10,seed:0' 2>&1 | tee "$LIB/logs/mv2-oracle-vae-$i.log"
          code=${PIPESTATUS[0]}
          printf 'SHARD_EXIT_CODE=%s shard=%s\n' "$code" "$i" | tee -a "$LIB/logs/mv2-oracle-vae-$i.log"
          exit "$code"
        ) &
        pids+=("$!")
        printf 'ORACLE_SHARD_PID=%s shard=%s\n' "$!" "$i"
      done
      bad=0
      for pid in "${pids[@]}"; do
        if wait "$pid"; then :; else bad=1; fi
      done
      test "$bad" = 0
      PYTHONDONTWRITEBYTECODE=1 uv run --project "$MJ" --no-sync python "$REPO_ROOT/scripts/dataset/wan/oracle_driver.py" aggregate --manifest "$LIB/meta/episode_manifest.json" --out "$ORACLE" --num-shards 8 --kind vae --demo-min-real 17 --exec-min-real 33
      ;;
    pack)
      require_stage oracle
      test ! -e "$LIB/motion" || { printf '拒绝复用已有 motion 目录\n'; exit 2; }
      uv run --no-sync python scripts/dataset/pack_motion_store.py pack --manifest "$LIB/meta/episode_manifest.json" --tokens "$LIB/motion-tokens" --latents "$LIB/wan-latents" --out "$LIB/motion" --encoder-run-dir "$ENC" --raw-dir "$RAW"
      uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store "$LIB/motion" --resume
      uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents "$LIB/wan-latents" --oracle "$LIB/oracle/wan-mj" --sampled "$LIB/oracle/wan-mj/sampled_windows.json" --sample-spec 'padded:all,rest:0.10,seed:0' --expect-padded 1600
      uv run --no-sync python scripts/dataset/wan/compare_wan.py tokens --store "$LIB/motion" --oracle "$LIB/oracle/wan-mj"
      uv run --no-sync python scripts/dataset/motion_checks.py a6set --manifest "$LIB/meta/episode_manifest.json" --framesamp "$LIB/framesamp-8x8" --motion "$LIB/motion"
      uv run --no-sync python scripts/dataset/motion_checks.py a7 --manifest "$LIB/meta/episode_manifest.json" --latents "$LIB/wan-latents" --tokens "$LIB/motion-tokens" --motion "$LIB/motion"
      uv run --no-sync python scripts/dataset/motion_checks.py a9set --manifest "$LIB/meta/episode_manifest.json" --latents "$LIB/wan-latents" --motion "$LIB/motion" --n 500 --cold-all
      uv run --no-sync python scripts/dataset/motion_checks.py a10 --manifest "$LIB/meta/episode_manifest.json" --motion "$LIB/motion" --demo-min-real 17 --expect-rows 71316 --expect-exec 35403 --expect-demo 35913 --expect-episodes 1600
      uv run --no-sync python scripts/dataset/motion_checks.py a11 --manifest "$LIB/meta/episode_manifest.json" --latents "$LIB/wan-latents" --demo-min-real 17 --expect-rows 71316 --expect-padded 1600 --expect-episodes 1600
      ;;
  esac
  test "$(git rev-parse HEAD)" = "$BUILD_HEAD"
  test -z "$(git status --porcelain)"
  printf 'BUILD_STAGE_DONE=%s\n' "$STAGE"
)
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
```
