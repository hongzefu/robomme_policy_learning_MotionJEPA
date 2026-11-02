# orig100 nostate motion 表启动记录

## 目的、输入与版本

对应 encoder 为 `aws-stage2-nostate-1101-a/checkpoint_epoch_105.pt#model`，已下载且原始 config、权重和共享副本 SHA 核验通过。目标表为训练库内 `motion-stage2-1024-nostate`，目标证据目录为 `evidence-nostate`；另一消融的输出目录独立。

本轮只准备两个已批准建表任务共用的输入，尚未执行 GPU 导出、G3、打包或训练。执行宿主为 Aspen 环境 A，两张 RTX A6000；输入下载不占 GPU。实施副本为 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl`，分支 `v4-gl-orig100-1031`，latent 恢复代码锚点为 `77f1061323b015daa4f3d4a050160b30b0bab162`。起跑时的实际 HEAD 由日志 `ASSET_START_HEAD=` 记录，起跑前工作区必须干净；后续阶段各记实际确定版本，不把本次下载提交冒充后续建表版本。

训练库根为 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-orig100-1600ep-9e109419`。源公开 bucket 为 `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1`，沿库内 `delivery_manifest.json` 选择 13 份 latent tar、逐成员 SHA 文件和上传成员清单，共 15 对象、26,917,887,511 字节。独立暂存根为 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/external/o16abl-latents-import`；核验后只同步 `wan-latents/` 至训练库，不覆盖现库 `delivery_manifest.json` 或 `.fetch-state`。

两个任务共享且只启动一次 tmux `o16abl-fetch-latents`，日志 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-fetch-latents.log`。主会话编写的运行载体为同目录 `o16abl-fetch-latents.sh`；真实生产逻辑来自已提交的 `scripts/dataset/hf_export/fetch_orig100_trainset.py`，顺序如下：

```bash
uv 管理的 P/.venv/bin/python fetch_orig100_trainset.py --ipv4 plan --manifest LIB/delivery_manifest.json --include wan-latents_tars/ checksums/sha256-source-files.txt manifest/upload_manifest.json --out IMPORT/plan.json
uv 管理的 P/.venv/bin/python fetch_orig100_trainset.py --ipv4 fetch --plan IMPORT/plan.json --out IMPORT --workers 8
uv 管理的 P/.venv/bin/python fetch_orig100_trainset.py unpack --lib IMPORT --workers 4
uv 管理的 P/.venv/bin/python fetch_orig100_trainset.py verify --lib IMPORT --deep
rsync -a --info=progress2 IMPORT/wan-latents/ LIB/wan-latents/
# 同步后按已发布逐成员 SHA 再核共享副本；日志保留对象和成员总数。
```

上块的 `P`、`LIB`、`IMPORT` 分别就是上文绝对路径；完整执行命令是 `bash /data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-fetch-latents.sh`，由 detached tmux 配 `pipefail`、`PYTHONUNBUFFERED=1`、`tee` 和 `EXIT_CODE=` 包装。无需新增依赖，不用裸系统 Python。预期判定为原工具 `TRAINSET_FETCH=PASS`、`TRAINSET_UNPACK=PASS`、`TRAINSET_VERIFY=PASS`，以及共享副本 `LATENTS_SYNC=PASS members=7489 mismatch=0`，退出码必须为 0。未验证前不写成完成。

## 用户决策与后续阶段

用户原话依次为「/data/hongzefu/robomme_policy_learning_MotionJEPA/1101-gl-ablation-encoders-ctx-b64-80k-plan.md 开工」「同意修订 继续做」「不需要再确认，你可以直接开始了。」。计划 v5 `2e9b8dc09acf5a5518c95284da9658c1961fd1c9` 独立一致性审查通过。现有工具无法解 latent 的缺口经独立 `ADJUST_REVIEW=PASS scale=same findings=0` 后最小修复；source 原默认行为和 SHA 校验不变。G3 子集、显式来源路径、固定 runner、两表构建与 pack/verify 的确切版本和命令在各自起跑前补充，不在本次资产准备中伪填已启动。完成后另写 result.md 和不可由 Git 还原的 records；权重、数据、日志原件仍在 v1-store，不归档脚本或 YAML 副本。

