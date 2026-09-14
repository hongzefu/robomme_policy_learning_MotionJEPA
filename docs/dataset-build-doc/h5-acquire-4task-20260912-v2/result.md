# 新版四任务 H5 获取补档结论

既有获取记录标记 `status=PASS`、`final_exit_code=0`：14 个 primary 压缩包，1600 条 H5，压缩字节数 90,500,984,560，解压 H5 字节数 793,815,485,572。原下载阶段耗时 2016.192 秒，连同后续核验总耗时 2384.21 秒；这些是原获取记录的数字，不是本轮重跑耗时。

原 final verification 记 `h5_group_matches_source_episode=1600`、`partial_h5_files=0`、`prior_checker_failures=2`。保留两次先前检查失败的事实，不将最终 PASS 改写为全过程无异常；失败原因不从该计数字段推断。

本轮实际源全集选取确认 primary=1600、spare=196、smoke=1，正式建库只收 primary。MANIFEST 的 `counts` 另有 `videos=1949`、`archives=29`、`files=2052`，它们不属于 episode role。原 README 的“每个 H5 只有 episode_0”表述有误；真实顶层组名是 `episode_<原编号>`，已在合并脚本前检和 map 中固定。README 摘录仅保留来源、分布及生成口径，不能代替数据实测。

用户在计划中决定保留 `extracted/`，本轮遵守。新数据包含 BinFill demo，且任务跨多个 suite；评估 sim 和 seed 划分另案处理，本轮只交付建库产物。

归档文件：`records/source.json`、`records/COMPLETE.json`、`records/final_verification.json`、`records/source_summary.json`、`records/source_readme_excerpt.md`。原始 H5、压缩包和模型权重均不进 Git。
