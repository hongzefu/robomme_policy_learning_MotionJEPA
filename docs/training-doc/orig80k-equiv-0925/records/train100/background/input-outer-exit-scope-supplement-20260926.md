# CPU INPUT最外退出证据范围补充（静态取证）

**六个CPU collector/judge的子命令返回0及既有输入内容验收证据仍有明确记录；最外整体Bash进程没有保存独立退出回执。** `launch_input`同样存在最终direct append完成写入后才判断其返回值的边界，所以日志 `EXIT_CODE=0`不能单独证明最外进程实退0。当前没有这些CPU任务实际失败的新证据，本补充不重跑、不修改原记录，也不决定INPUT该边界的处置或例外。

## 固定版本与实际命令

审查输入起跑版本 `3a1582db39c723c735e04752e5027bfe40ecc3e1`，对照当前冻结版本 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`；收尾HEAD与tracked状态保持不变。两个提交的 `docs/training-doc/orig80k-equiv-0925/launch.md::launch_input()`主体相同：1512 B，SHA256 `bc8b16542da1f7b216fc3a88bd0023efc1af86f5226a4a681c79ee1581b5d28c`。两版本的 `scripts/training/tests/check_orig80k_inputs.py`也无差异。

实际会话名统一为 `orig80k-input-<侧>-20260926T170654Z`；命令文件位于主仓 `v1-store/bench/orig80k-equiv-0925/commands-20260926T170654Z/<会话>.command`，原日志为 `v1-store/logs/<会话>.log`。逐份静态解码Bash ANSI字符串后，六份命令主体均与上述固定版本逐字相同；没有执行或source命令文件。以下SHA均与对应原日志中的 `COMMAND_FILE_SHA256_EXPECTED/ACTUAL`一致：

| 侧 | 实际命令文件SHA256 |
|---|---|
| full-a | `58f86ee7d62e39b1294bf23d321c0cd3a85033e786bac6c75b7e22134932f28d` |
| full-b | `35f9e8d5d1a5df0e63d036f136a3899d56e03845d1c62790f150eb21b02fceaa` |
| count-a | `f20abe885cb4cd444b248e6c7520d62e8b32f6c3869b32a99fc9753b338ad432` |
| count-b | `367f50c0a020742121dd01bece128ba0eaea4486a0ef5b96b68bf362213c629c` |
| full-judge | `3597b818656915a4c6b45474437c4e8d55b48d745d55d63cc2b5a5c50ca848e9` |
| count-judge | `ab2f9269afd97127719e0ecce6801bcece33a6b58d26a2e82ae2ed4a7d535519` |

## 哪一层状态已取得，哪一层没有

固定主体在子shell最后执行 `"$@"`，紧接着读取该子shell与tee的 `PIPESTATUS`。子shell中没有在collector/judge之后追加一个覆盖返回码的成功命令。因此六份原日志各自唯一的 `COMMAND_EXIT=0`，按实际执行源码记录的是collector/judge子命令返回0；`TEE_EXIT=0`是这一层tee的已捕获状态。四份collector同时各有唯一 `INPUT_COLLECT=PASS`，两份judge各有唯一 `INPUT_EQ=PASS`。这些记录不应误写为“只有一个未经检查的PASS字符串”。

随后代码把footer管道状态并入rc，最终执行 `if ! printf ... EXIT_CODE=... >> "$input_log"; then ...; exit 1; fi`，再 `exit "$rc"`。如果最终printf已经把0完整写入，但其调用返回非零，最外Bash会走失败分支；先前写入的日志0不会自动消失。`launch_input()`的直接返回值则来自 `tmux new-session -d`，表示会话创建，不是等待任务结束。

已归档于00bd的 `records/runtime-controller.json` SHA256为 `888cd2d0fa17139b6c6e56e1f79c97e9b0b7c16750cd206bc37c41475c60d1d0`。六条对应记录均为 `tmux_session_absent=true`，保留了PID、日志/命令身份及基于日志和内容检查的状态，但未保存最外进程独立wait或 `pane_dead_status`。本次核对的启动源码和实际命令也未为这六个任务配置/记录独立pane退出状态。会话消失、日志监听器结束或后续验收器返回0，均不能补成该最外退出回执。

| 证据层 | 当前事实 |
|---|---|
| collector/judge实际子命令 | 六份 `COMMAND_EXIT=0`，固定源码直接捕获其返回码 |
| 主体及footer输出管道 | `TEE_EXIT/FOOTER_PRINTF_EXIT/FOOTER_TEE_EXIT`各唯一0，属于各自已捕获管道 |
| 输入记录及判定内容 | 四份collector完整记录、两份judge的既定输入判据通过 |
| 最外整体进程退出 | 日志写出 `EXIT_CODE=0`，缺独立wait/pane回执，不能据此证明实退0 |

## 输入内容验收的实际范围

原记录根为 `v1-store/bench/orig80k-equiv-0925/input-20260926T170654Z/`。四份schema2清单仍各登记17个文件，本次读取的原清单SHA为：full-a `b95d89845d2e7219a6102cd05ffe26d5941ebcf66179f5c423539e1d710dbc05`；full-b `e70f4dc1c66e60ac97b1dd3f7fc116a82bb783f4bb4e00dd34c236572808ddf0`；count-a `4adabc7d5e4f959682530360c6ea06acb30222cc5419ffd16b4e19ec9706c24e`；count-b `ac66670e25f5c9f9b01e088b5c15d1577f62d6cc11563c67c4be54b6fbc69430`。此前逐文件和无损归档检查保留原字节，详见已提交的[INPUT结果](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-equiv-0925/result.md)及其records/logs检查链；本次未重新执行内容judge。

实际判定原文仍为：

```text
INPUT_EQ=PASS episodes=1600 samples=476857 boundary_samples=6906 batches=104 epochs=2 comparison=host_numeric_exact_motion_none_v1
INPUT_EQ=PASS episodes=400 samples=189035 boundary_samples=2400 batches=104 epochs=2 comparison=host_numeric_exact_motion_none_v1
```

这些判定绑定上游A `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`与B/量具的真实INPUT_HEAD 3a；范围包括来源/资产、全量执行身份与两epoch索引、已登记定点及真实批内容的精确数值和获准None等价，终点为 `collate_before_jax`。最外append边界不等于已经发现输入数值或来源失配，也不把这些输入证据扩展成训练轨迹或最外退出证明。

## 新反例与批准范围

已有[纯shell结果矩阵](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-wrapper-tests-20260926/results/results.tsv)（SHA256 `7c5ac6e50803708cd36020552e924c0993cfafcfdf00bd285b008491eb72f7ea`）中，`append_after`为另一个同型wrapper的反例：最终append写出成功值后返回37，wrapper实退1，旧日志checker仍返回0。此处依据INPUT固定代码指出同型边界，**没有对六个INPUT任务重做故障注入，也不把模拟的1/37记成其真实退出码**。

用户最新明确批准的处置原话为「补记P1限制，补独立退出取证后继续（推荐）」，其显式对象是P1。本补充只引用这一已确认范围，**不据此宣称用户已经另行批准INPUT的豁免、沿用或重跑策略**。INPUT最外退出缺口如何纳入后续放行，应由根代理/用户另行明确；既有数值、来源、collector/judge真实子命令返回及原始归档证据照实保留。

可纳入后续档案的限定表述：本轮六个CPU输入子命令实际返回0，四份collector及两组INPUT_EQ内容判据通过；最外封装使用最终direct append，未留独立wait/pane退出回执，日志0不能单独证明最外整体实退0。目前没有真实CPU任务失败或数值/来源失配的新证据，原始记录保持不变；本段不代替对该外层证据缺口的处置授权。
