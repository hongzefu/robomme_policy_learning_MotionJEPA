# 手动评估（逐模型）

> commitV4.6 修订：v4 破坏性单一化重构后，仓库仅保留 perceptual frame_sampling 的
> 三种 integration（context / modulation / expert）；π₀.₅ baseline、MemER、symbolic、
> tokendrop、recurrent 各变体的启动命令已随代码删除（需要时从 git 历史取回本文件
> 旧版）。路径同步为新布局：入口在 `scripts/training/`，checkpoint 根收敛到
> `v1-store/train-runs/`，评估结果根收敛到 `v1-store/evaluation/`。

## 目录

- [FrameSamp + Context](#framesamp--context)
- [FrameSamp + Modulation](#framesamp--modulation)
- [FrameSamp + Expert](#framesamp--expert)
- [其他提示](#其他提示)


## FrameSamp + Context
```
# terminal 0
CUDA_VISIBLE_DEVICES=0 uv run scripts/training/serve_policy.py --seed=7 --port=8012 policy:checkpoint --policy.dir=v1-store/train-runs/mme_vla_suite/perceptual-framesamp-context/79999 --policy.config=mme_vla_suite

# terminal 1（micromamba robomme 环境；仿真器依赖不在 uv venv，此处保留该环境的 python）
micromamba activate robomme
CUDA_VISIBLE_DEVICES=1 python examples/robomme/eval.py --args.model_seed=7 --args.port=8012 --args.policy_name=perceptual-framesamp-context --args.model_ckpt_id=79999
```

## FrameSamp + Modulation
```
# terminal 0
CUDA_VISIBLE_DEVICES=0 uv run scripts/training/serve_policy.py --seed=7 --port=8013 policy:checkpoint --policy.dir=v1-store/train-runs/mme_vla_suite/perceptual-framesamp-modul/79999 --policy.config=mme_vla_suite

# terminal 1（micromamba robomme 环境）
micromamba activate robomme
CUDA_VISIBLE_DEVICES=1 python examples/robomme/eval.py --args.model_seed=7 --args.port=8013 --args.policy_name=perceptual-framesamp-modul --args.model_ckpt_id=79999
```

## FrameSamp + Expert
```
# terminal 0
CUDA_VISIBLE_DEVICES=0 uv run scripts/training/serve_policy.py --seed=7 --port=8014 policy:checkpoint --policy.dir=v1-store/train-runs/mme_vla_suite/perceptual-framesamp-expert/79999 --policy.config=mme_vla_suite

# terminal 1（micromamba robomme 环境）
micromamba activate robomme
CUDA_VISIBLE_DEVICES=1 python examples/robomme/eval.py --args.model_seed=7 --args.port=8014 --args.policy_name=perceptual-framesamp-expert --args.model_ckpt_id=79999
```

## 其他提示

只评估部分任务：
```
python examples/robomme/eval.py --args.only_tasks="BinFill,PickXtimes" ...
```
可用 `--args.exclude_tasks` 排除任务、`--args.re_eval_tasks` 重评指定任务。
评估被打断后重跑 `python examples/robomme/eval.py` 会自动续评。

`scripts/training/serve_policy.py` 侧可改 `--seed` 与 `--policy.dir` 评不同 checkpoint 与 seed。
`examples/robomme/eval.py` 侧的 `--args.policy_name`、`--args.model_seed`、`--args.model_ckpt_id`
用来生成保存目录名，例如：
```
v1-store/evaluation/perceptual-framesamp-modul
├── ckpt60000
│   ├── seed0
│   ├── seed42
│   └── seed7
├── ckpt70000
│   ├── seed0
│   ├── seed42
│   └── seed7
├── ckpt79999
    ├── seed0
    ├── seed42
    └── seed7
...
```
随后用 `uv run scripts/training/compute_results.py --model_dir perceptual-framesamp-modul --ckpt_list ckpt60000,ckpt70000,ckpt79999 --seed_list seed0,seed42,seed7` 汇总结果。
