# b128 四卡 modulation smoke 验收通过

本次从 clean HEAD `a2d32dfbd8c62593824d4b10375473473da5d7e9` 运行 `smoke-m8x8-modul-20260915T054007Z`，物理 GPU 4,5,6,7、batch 128、worker 8、fsdp 4。20 步全部完成，`2026-09-15T05:40:25Z` 起跑，`05:45:43Z` 退出，墙钟 5 分 18 秒；已在启动前按完整运行留档。

```text
PREFLIGHT=PASS n=25
SMOKE20=PASS steps=20 finite=1 exit_code=0
MEM_PARAMS=PASS n=6
PARAM_TREE_EXACT=PASS config=mme_vla_suite_b128_60k history_config=perceptual-framesamp-modul-8frame-8x8.yaml yaml_sha256=5b5ac2f85302d4d8 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
NORM_STATS=PASS sha256=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173
EXIT_CODE=0
```

每步 loss、grad_norm、llm_grad_norm、mem_enc_norm、param_norm 均有限。loss 范围为 0.07691064476966858–0.10088201612234116，首末分别 0.0894237533211708、0.08444450050592422；只用于功能验收，不据短测判断策略效果。checkpoint 19 正常完成异步保存。

参数树通过双向集合与形状检查，另从 Orbax 元数据正面确认 q/kv/out memory einsum、mem_rms_norm scale、mem_rms_norm_ffn Dense kernel/bias 六条专属路径。motion provenance 为关闭态，canonical manifest 为 `4cd5a170…`；checkpoint 中 norm_stats 与新库源文件 SHA 完全一致。

共享日志 `v1-store/logs/m8-smoke.log` 的前段属于 01:54 的历史失败 run，本次按唯一 `RUN=smoke-m8x8-modul-20260915T054007Z` 标记提取起止区间，旧记录原样保留，未混入任何判定。归档见 [records/train.log](records/train.log)、[records/checks.log](records/checks.log)、[records/param_tree.json](records/param_tree.json) 及实际指标/配置/provenance。验收脚本 [records/finish_check.py](records/finish_check.py) 强制核对 20 步、25 项 preflight、六条参数路径和各摘要。

`m8-smoke` 正常退出后自动消失，未执行 tmux 清理。归档后仅删除本 run 的 checkpoint 与 bench 根，保留固定 JAX 缓存 `v1-store/cache/jax/m8-smoke`。正式 run 可进入隔离副本建立与最终 preflight。
