# 顺序实施状态

本批次已主动中断取证，不满足完整20步判据，尚未启动正式训练。用户要求「开始做 有问题立刻问用户 起泡后给出预计的时间」；预计时间仍须基于真实测速与正式约300步。

已实跑：

- `test_modul_speed.py` 与 `test_modul_launch_guards.py -k 'not full_preflight'`：41 passed，35.30秒。
- `test_modul_train_records.py test_modul_budget_guards.py`：31 passed，25.72秒，覆盖具名默认继承、实际loader/mesh错配、完整配置未知字段、另一预算且SHA正确的报告、替换权重、尾窗NaN及可加载NaN checkpoint。
- `test_modul_sweep.py`：16 passed，0.23秒，验证顺序及shell错误传播。
- `test_pack_guards.py -k '2048 or legacy_budget'`：显式MMEVLA_TEST_SOURCE与MMEVLA_TEST_MANIFEST指向本库后14 passed，31.36秒；三档真实样本shape及spawn均通过。
- Python编译检查、两个新shell的 `bash -n` 及 `git diff --check` 已通过。

计划外事件：pack守卫第一次未给测试数据环境变量，旧默认 `v1-store/episode_manifest.json` 不存在，14项fixture初始化失败、没有训练；随后使用测试原有显式接口指向已核实的本库并通过。未修改旧默认、未下载或重建正式数据集。

完整preflight夹具9 passed，110.92秒。首次正向夹具仍使用已经完成的2048正式run名，被RUN_ROOT_ABSENT正确拒绝；中断剩余测试后改用独立测试身份重新实跑通过，未改动旧run。资产全量SHA校验在首次实跑已输出INITIAL_ASSETS=PASS，后续CLI反例使用资产校验桩避免逐例重扫12GB权重。

提交前另实跑三档YAML、12点采样边界、类型/组合守卫与实际Dataset接口；每档取真实索引0、200000、605610，与独立NPY链逐键shape/dtype/原始字节全部相同，均输出PRECOMMIT_REAL_INPUT=PASS samples=3 mismatches=0。该有界短测不代替后续全1600集边界及20批验证。

00:31:21 UTC从Beta e1b97169d9a1cdc838af5f27bce0bca6a7a513e9及clean HEAD起跑；tmux为modul-sweep-20260923T003111Z，训练PID1197660。2048 Beta完整配置基线补建通过，两新预算与基线逐字段核对通过。原始训练源码来自742f2d894d25abe802a26b1c118852b0b7676e44的只读快照。

初态201叶均有限，完整摘要耗时615.913秒，state_digest前缀85a829759b07d809；随后记录前两步有限loss和梯度。只读py-spy采样定位于逐叶GPU→CPU读取及SHA256。临时C序buffer方案30例摘要一致，256MiB样本复制/哈希变快，但不把它当完整状态性能结论。

已立即询问用户优先优化取证或保持现状，未收到偏好回复；按现有实施授权保留全部判据，选择优化取证并用新名重跑，没有把沉默记为新的用户授权。00:53:36 UTC旧阶段在精确PID的SIGINT后退出130，队列退出1，八卡释放；只读监听进程单独回收，没有执行tmux全局清理。原run与快照保留。

后续[b批次](../modul-sweep-20260923-b/launch.md)仍跑完整20+1步及201叶对拍；a批次只能作为初态和前两步的部分证据。现场JSON、指标与清洗日志在records及对应before-20子档案，权重和大型产物不进Git。
