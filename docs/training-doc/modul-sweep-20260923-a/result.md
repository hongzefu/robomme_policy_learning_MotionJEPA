# 顺序实施状态

当前处于代码验证，尚未启动正式训练。用户要求「开始做 有问题立刻问用户 起泡后给出预计的时间」；预计时间在真实测速与正式约300步后提供。

已实跑：

- `test_modul_speed.py` 与 `test_modul_launch_guards.py -k 'not full_preflight'`：41 passed，35.30秒。
- `test_modul_train_records.py test_modul_budget_guards.py`：31 passed，25.72秒，覆盖具名默认继承、实际loader/mesh错配、完整配置未知字段、另一预算且SHA正确的报告、替换权重、尾窗NaN及可加载NaN checkpoint。
- `test_modul_sweep.py`：16 passed，0.23秒，验证顺序及shell错误传播。
- `test_pack_guards.py -k '2048 or legacy_budget'`：显式MMEVLA_TEST_SOURCE与MMEVLA_TEST_MANIFEST指向本库后14 passed，31.36秒；三档真实样本shape及spawn均通过。
- Python编译检查、两个新shell的 `bash -n` 及 `git diff --check` 已通过。

计划外事件：pack守卫第一次未给测试数据环境变量，旧默认 `v1-store/episode_manifest.json` 不存在，14项fixture初始化失败、没有训练；随后使用测试原有显式接口指向已核实的本库并通过。未修改旧默认、未下载或重建正式数据集。

完整preflight夹具9 passed，110.92秒。首次正向夹具仍使用已经完成的2048正式run名，被RUN_ROOT_ABSENT正确拒绝；中断剩余测试后改用独立测试身份重新实跑通过，未改动旧run。资产全量SHA校验在首次实跑已输出INITIAL_ASSETS=PASS，后续CLI反例使用资产校验桩避免逐例重扫12GB权重。

提交前另实跑三档YAML、12点采样边界、类型/组合守卫与实际Dataset接口；每档取真实索引0、200000、605610，与独立NPY链逐键shape/dtype/原始字节全部相同，均输出PRECOMMIT_REAL_INPUT=PASS samples=3 mismatches=0。该有界短测不代替后续全1600集边界及20批验证。

长验证、容量、测速和正式训练尚未起跑；起跑后按真实事件补记。
