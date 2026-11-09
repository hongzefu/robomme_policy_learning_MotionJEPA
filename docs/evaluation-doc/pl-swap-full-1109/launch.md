# PatternLock 全因子单步 motion 交换启动档案（pl-swap-full-1109）

## ① 目的

把先导轮（`pl-single-step-swap-1108`，130 局）扩成 3×3 盘上全部 40 种（起点格，方向）组合的受控交换：接收集 R 40 局 × 供体集 S 40 局，分 N/Z/Rnd/A/B/C/D 七组加 seed 复核，两个只差编码器的 orig100 模型各跑一遍——MotionMem（学习型 encoder，concat 1×1024）与 Frozen Wan pooling（冻结 Wan VAE + 时间池化 1024）。计划为根目录 `1109-patternlock-swap-full-factorial-wan-plan.html`（`1ed8838`，修订一 `ff5a48e`）。

## ② 用户决策原话（按时间）

1. 计划阶段（见计划 K1–K4）：「aspen刚跑了motion对任务的贡献 跟我制定 应该怎么跑全量的？」「跑完还要跑 Frozen Wan pooling 我们最强的对比」「换成主表 orig100」「对的你计划直接写到policy仓库内」「你可以ssh到aspen」。
2. 开工令（11-09 美东）：「1109-patternlock-swap-full-factorial-wan-plan.html 同意开工」。
3. 状态机登记：选「登记」。
4. 实施中：「尽可能不要问我自己决策」「一路做到底」。

## ③ 版本与代码状态

- 分支 `v4.2-plswap-full`，运行代码钉在 `efc3a19003951f8ee5d7de2da6ccecc3a7899f9a`（`commitV4.2.3`），分离工作树 `v1-store/worktrees/pl-full-run`（detached，clean，env.sh 起跑前核对 HEAD）。
- 合并链：`commitV4.2.1`（W3 准备/读数脚本）→ `commitV4.2.2`（W1 合并 `origin/v10-ctx-wan-timepool-eval@68bb958`）→ `fix:` 补回 P-LINEAGE 与删重复定义 → `commitV4.2.3`（W2 plfull 口径、多局交换、Wan 接入）。
- 各闸门：`PLAN_CONSISTENCY_REVIEW=FAIL base=1ed8838 findings=11 blocking=5`（主代理按用户指令裁决，计划〇′）；`ADJUST_REVIEW=PASS scale=smaller findings=2`（motion 资产复用）；三次 `PRE_MERGE_REVIEW=PASS`；三次合并后短测仅既有失败 `test_rollout_hidden::test_hidden_values_match_independent_rollout`。
- 第三方：`third_party/robomme_benchmark` gitlink 未变。

## ④ 被跑对象

| 模型 | checkpoint | policy_config | model_kind | 资产核对 |
|---|---|---|---|---|
| motion | `v1-store/assets/vla-o16cc-59999/59999`（= bucket `HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1`） | `mme_vla_suite_orig16_b128_60k` | `stage2_pool4_concat` | `ASSETS=PASS model=motion files=38 bytes=11976241540 mismatches=0` |
| wan | `v1-store/assets/vla-o16wan/vla59k-8bf3feb9…/59999`（bucket `HongzeFu/robomme-vla-orig100-ctx32x4x4-wan-timepool8x8-60k-v1`） | `mme_vla_suite_orig16_b128_60k` | `wan_fixedpool1024` | `VLA_DOWNLOAD=PASS step=59999 files=35 bytes=11620193417 manifest_sha256=8bf3feb94861d042a78834104a702aa5d4a91c6f7e3b2c12c94899b1d23b8f97`（逐文件 sha256） |

## ⑤ 局池

`pl_full_prep pool`：env_seed 900000–1299900 步长 100、跳过 val 50 个，共 3950 个；单步局 1232（计划写 1247，差异见 result.md），每种组合 ≥ 10 局。
`PL_POOL=PASS eps=80 combos=40 overlap=0`；`PL_PAIRS=PASS A=40 B=160 C=168 D=1232 D_sel=400 seedcheck=60`。

## ⑥ 命令与执行顺序

```bash
O=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/pl-swap-full-1109
python -m scripts.interpret.pl_full_prep pool --out $O                      # official-sim 解释器
python -m scripts.interpret.pl_full_prep specs --out $O/motion --model motion --commit efc3a19… --run-tree <pl-full-run>
python -m scripts.interpret.pl_full_prep specs --out $O/wan --model wan --commit efc3a19… --run-tree <pl-full-run> \
  --checkpoint <wan 59999> --policy-config mme_vla_suite_orig16_b128_60k --model-kind wan_fixedpool1024 --target-contract none
bash $O/<M>/launch_stage.sh smoke | repro | n
python -m scripts.interpret.pl_full_prep donors --out $O/<M> [--wan-gate]
bash $O/<M>/launch_stage.sh multieq | swap | seeds
python -m scripts.interpret.analysis.pl_full_readout --root $O/<M>
python -m scripts.interpret.analysis.pl_full_readout --root $O --intersect motion,wan
```

环境变量见 `$O/<M>/env.sh`（照抄先导轮：XLA 确定性开关、HF_HUB_OFFLINE=1、缓存落 `v1-store/cache/`），每卡 2 worker、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.30` 起始。

## ⑦ 产物与日志

- 产物根 `v1-store/evaluation/pl-swap-full-1109/`（`pool.json`、`pairs.json`、`metadata/`、`manifest.json`、`motion/`、`wan/`）。
- 日志 `v1-store/logs/pl-swap-full-1109/`。

## ⑧ 本轮 tmux 会话清单

| 会话 | 用途 | 起止（美东） |
|---|---|---|
| `plf-fetch-wan` | Wan 59999 下载 | 11-09 11:1x – 已退出，EXIT_CODE=0 |
| `plf-motion-smoke-w00` | motion 冒烟 | 11-09 12:03 – |
| `plf-wan-smoke-w00` | Wan 冒烟 | 11-09 12:03 – |

## ⑨ 盯盘项

Monitor 过滤词：`ENV_CHECK=FAIL|rc=[1-9]|PLF_QUEUE|PLF_SPEC|EXIT_CODE=|Traceback|out of memory|Killed|ContractError`。GPU 0 上另有用户 frankcui 的常驻进程（`throw_studio/server.py`，约 0.3 GB/卡），不动。
