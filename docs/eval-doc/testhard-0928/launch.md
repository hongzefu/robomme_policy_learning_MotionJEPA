# 官方 MME-VLA test-hard 评估（2026-09-28）：启动与配置

## 目的

在 benchmark 拆包后的 `robomme_hard`（`dataset="test-hard"`）上评估官方 MME-VLA `perceptual-framesamp-modul` 79999，身份与按档步数上限与 SimpleMemVLA 同一批 1100 局（benchmark 0927 计划 U-8、U-9、§八 8.2）。

## 版本与代码状态

- 本仓库：分支 `official-testhard-eval-0928-0137`，从官方 `RoboMME/robomme_policy_learning@ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` 切出。提交：`812d542`（接入）→ `3532acf`（`ONLY_TASKS`）→ `a2f1378`（权重 run 根）→ `b93fef5`（去代理）→ `7fc7128`（异常局记 error、续评按最后一条记录判）→ `51c59b3`（显存上限 0.75、未解决 error 时非零退出）。第一轮第一遍在 `b93fef5`，续评与第二轮在 `51c59b3`。
- 子模块 `third_party/robomme_benchmark` → `hongzefu/robomme_benchmark_MotionJEPA@31e61259f9885f7dba8a321d0b6e699eb5061866`（benchmark 12.210，分支 `PolicyEvalThirdParty-mmevla-0928-0311`）；官方原 gitlink `856bc3a`。
- 权重：官方 HF `Yinpei/perceptual-framesamp-modul@c0f565dd…` 的 `79999`。NFS 副本 `robomme_policy_learning-frameSamp-continue/runs/ckpts/perceptual-framesamp-modul/79999` 与本机官方下载 `diff -rq` 18 个文件逐字节相同；该副本缺官方 `history_config.txt`，故在 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/eval-out/mmevla-ckpt/perceptual-framesamp-modul/` 放官方 `history_config.txt`（`perceptual-framesamp-modul.yaml`）与指向该 `79999` 的 symlink。server 日志确认「changing to perceptual-framesamp-modul.yaml」「Restoring checkpoint from …/79999/params」。
- 环境：JAX 侧 `uv sync --frozen`（NFS cpython 3.11.14，208 包，jax 0.5.3）；`robomme_env`：`uv venv` + `uv pip install -r examples/robomme/requirements.txt --override scripts/robomme_env.overrides.txt`（ManiSkill 由 `@dev` 钉到 `07be6fbc…`）`-e third_party/robomme_benchmark -e packages/openpi-client`，锁清单 `robomme_env.lock.txt`。

## 分片与命令

片 i（0..9）、轮 r：`episode_start = 10×(r−1)+i`、`episode_stride = 20`、`max_episodes = 0` → 80 局的任务取 {s, s+20, s+40, s+60}、20 局的任务取 {s}，每片 13×4 + 3×1 = 55 局。10 个评估占位 job `62177614`～`62177623`（1 GPU A40／1 CPU／32 G／48 h），片 i 在 job `62177614+i`，与 SimpleMemVLA 同卡串接（SimpleMemVLA 该轮 10 片全部结束后再起）：

```bash
# 登录节点 tmux：hs-eval-mmevla-r<轮>-s<i>（第一轮续评 hs-eval-mmevla-r1b-s<i>）
srun --jobid=<62177614+i> --overlap --exact --ntasks=1 --cpus-per-task=1 --gpu_cmode=shared \
  /usr/bin/env SHARD=<i> ROUND=<r> PORT=<端口> RUN_TAG=testhard0928 bash scripts/gl_eval_shard.sh
# server：uv run --frozen --no-sync scripts/serve_policy.py --seed=7 --port=$PORT policy:checkpoint \
#   --policy.dir=<run 根>/79999 --policy.config=mme_vla_suite（XLA_PYTHON_CLIENT_MEM_FRACTION 0.4→0.75）
# eval：env -u http_proxy … robomme_env/bin/python eval.py --args.host=127.0.0.1 --args.model_seed=7 \
#   --args.model_ckpt_id=79999 --args.episode_start=… --args.episode_stride=20 --args.max_episodes=0
```

冒烟（BinFill@xhard1 episode 0，job 62177615）：`EVAL_SMOKE=PASS policy=mmevla ckpt_ok=1 status=fail steps=833 max_steps=1500 binding_available=1 injected_mismatch=0`，单局约 5.5 分钟，步骤 MaxRSS 17.8 GB。

## 时间线（EDT）

第一轮第一遍 05:26～06:40；第一轮续评 06:43～07:01；第二轮 08:57～10:1x。
