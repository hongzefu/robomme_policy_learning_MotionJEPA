# modulation 8×8 motion 八卡20步smoke

本任务执行0916计划V9，验收生产档位batch128、fsdp8、workers16和新motion布局。本轮唯一名称为 `smoke-m8x8-modul-motion-20260918T050716Z`，名称时间戳表示准备时刻；本页为起跑前记录，实际启动时刻见驱动日志。前置为V8两轮100步A/A与V-online三档全部通过。

## 版本、数据与启动口径

从本页入库后的clean HEAD启动，准确提交由入口参数CHECK_HEAD、preflight的CHECK_REPO_HEAD及runtime记录固定。模型配置为 `mme_vla_suite_b128_80k`，模型与超参沿用正式档位，只在smoke启动覆盖中指定 `--num-train-steps 20 --log-interval 1 --no-wandb-enabled`。history YAML为 `perceptual-framesamp-modul-8frame-8x8-motion.yaml`，motion预算160，demo至少17帧、repeat_last补满33；完整生产VLM和pi05_base预训练初始化参与。

数据根为 `v1-store/datasets/4task-v2-1600ep-604f16da`：1600集，framesamp-8x8提供605611个执行样本，motion表71316行。清单SHA为 `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918`，motion元数据固定SHA为 `d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268`。沿用新库norm_stats，SHA为 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`，文件位于 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`。

环境B，8×A100-SXM4-80GB，AWS本地NVMe RAID `/dev/md0` XFS。使用主仓现有uv环境和 `uv run --no-sync`；HF、CUDA、JAX、uv、WandB及XDG缓存全部在v1-store内。`MMEVLA_MOTION_STORE` unset，表路径来自YAML。smoke的JAX缓存为 `v1-store/cache/jax/mv2-smoke`；正式训练使用正式run名称对应的独立缓存。

## 驱动与失败即停

执行 `bash v1-store/logs/mv2-smoke-stage.sh <RUN> <CHECK_HEAD> <HC_SHA256> <MOTION_META_SHA256>`，tmux全名 `mv2-smoke`。stage外层日志为 `<RUN>.stage.log`，训练入口日志为 `<RUN>.driver.log`，两者分别记录EXIT_CODE。训练参数数组同时交给preflight与train.py。实际smoke入口和正式入口的任务主体都使用 `body() ( set -euo pipefail ... )`，外层tee记录正常返回或错误退出；外部强杀不保证产生退出码。

实际入口源码归档为 [smoke-runner.sh](records/smoke-runner.sh) 和 [smoke-stage.sh](records/smoke-stage.sh)。提交后先把CHECK_HEAD固定为完整SHA，准确调用为：

```bash
bash v1-store/logs/mv2-smoke-stage.sh \
  smoke-m8x8-modul-motion-20260918T050716Z "$CHECK_HEAD" \
  6c9f165f928786e456b98c2448a60d6be2c2fd3e935882d9fc08e439964f1d20 \
  d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268
```

已完成两个实际入口负例。`smoke-mv2-reject-busy-20260918T034822Z`在GPU4/5被V8占用时立即EXIT1，未到preflight、未创建训练或bench目录。`smoke-mv2-reject-preflight-20260918T042811Z`在八卡空闲时只把YAML指纹设为64个0，30项检查中唯一失败为YAML_SHA256；preflight之后立即EXIT1，未出现TRAIN_MESH或训练步，也未创建run/bench目录。两份独立日志完整保留，未覆盖任何正式名称。

训练根为 `v1-store/train-runs/mme_vla_suite_b128_80k/<RUN>`，记录目录为 `v1-store/bench/<RUN>`，日志为 `v1-store/logs/`。入口拒绝覆盖既有日志、记录和run目录。完成全部验收并归档后仅清理这个smoke的权重和run快照，保留指标、摘要、trace及日志。

## 完整验收

preflight必须30项全部PASS。20个训练步的五项标量完整且有限，真实runtime为8设备/fsdp8/batch128/workers16、mesh(batch=1,fsdp=8)。checkpoint19按仓库YAML和run快照分别核参数树：模型/权重均65叶，missing/extra/shape_mismatch为0；含六个modulation参数叶及四个motion参数叶。`check_config_provenance.py` 显式指定本轮库、framesamp-8x8、新norm文件及旧400ep负例库，五条检查齐全且TIC_L0通过；budget160跨run快照、仓库YAML与固定期望一致。

`TRAIN_TIMING_STEPS=300`在本次20步任务中截到实际20步。另以500ms间隔采集GPU0–7的NVML利用率与显存，在第5–19步窗口核主线程等待、设备trace覆盖和八卡采样。此短窗口用于验证计时组件，不作为正式80k的稳态性能结论；正式性能按第100–299步独立实测，不把异步dispatch计为设备计算时间。
