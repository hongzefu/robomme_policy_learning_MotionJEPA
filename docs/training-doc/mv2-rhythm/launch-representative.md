# 35 例节奏回放：起跑记录

用户确认「采用35例，保留其他全部覆盖（推荐）」。旧411例回放的中止和选择入口短测见 [result.md](result.md)，原始启动记录保留在 [launch.md](launch.md)。

本轮从修补提交后的clean CHECK_HEAD启动，起止均核完整SHA和空porcelain；CPU运行，CUDA_VISIBLE_DEVICES为空，JAX_PLATFORMS=cpu，uv --no-sync。数据为已verified的v1-store/datasets/4task-v2-1600ep-604f16da，YAML为perceptual-framesamp-modul-8frame-8x8-motion.yaml，预算160、demo最少17真帧。

回放选择35个真实例，覆盖16种余数各最短/最长、四任务和全局最大es1152；只用于rhythm/termination回放。预算扫描仍为411种长度，TAU_LONG仍测0与1152，1296/1297两侧完整1300步保留。要求RHYTHM_CASES中的35/411/16/4精确命中，四项gate和TIC_RHYTHM全部PASS，EXIT_CODE=0。

tmux全名mv2-rhythm-rep，命令：
```bash
bash v1-store/logs/mv2-open-runner.sh rhythm-rep <CHECK_HEAD>
```
完整外壳沿用[V4记录](../mv2-v4/launch.md)，只增加rhythm-rep分支，实际命令如下：
```bash
uv run --no-sync python scripts/training/tests/eval_rhythm_gates.py \
  --gate all --lib v1-store/datasets/4task-v2-1600ep-604f16da \
  --yaml perceptual-framesamp-modul-8frame-8x8-motion.yaml \
  --replay-cases representative --expect-replay-cases 35 --expect-real-es 411 \
  --out v1-store/reports/motion/mv2-rhythm-rep.json
```
外壳的阶段白名单增加rhythm-rep，其余起止保护不变。日志v1-store/logs/mv2-rhythm-rep.driver.log，与旧尝试分开；本轮报告和日志不得覆盖已有文件。后续AA入口只接受本次rhythm-rep的EXIT_CODE=0。
