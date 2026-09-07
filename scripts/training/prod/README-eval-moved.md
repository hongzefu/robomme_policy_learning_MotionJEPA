# 评估脚本已迁走

本目录（`scripts/training/prod/`）下原有的评估脚本 —— `eval_shard.sh`、`eval_all_shards.sh`、
`merge_eval_shards.py`、`check_test_seeds.py`、`check_ckpt_param_tree.py`、`eval_seed_sweep.sh`、
`aggregate_seed_runs.py`、`summarize_hard_eval.py` —— 已于 `commitV6.2` 整体迁往
[`../legacy-eval/`](../legacy-eval/README.md)，`scripts/analysis/plot_eval_success_by_length.py` 一并迁入。

主线不再维护评估执行链路，`examples/robomme/` 与 `src/mme_vla_suite/policies/` 已回到 `4b7a710`
（40k 训练收尾）的状态。各评估留档 `launch.md` 里的旧路径一字未改（如实记录当时执行的命令），
新旧路径对应表见 `../legacy-eval/README.md`。

注意 `eval_shard.sh` / `eval_all_shards.sh` 在那里有 `.local` 与 `.remote` 两个变体（两条分支各自实现了
一套 split 选择，参数名不同），选哪个见该 README。
