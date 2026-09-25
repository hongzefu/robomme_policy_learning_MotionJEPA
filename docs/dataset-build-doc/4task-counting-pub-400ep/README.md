# 公开counting四任务400集正式库：十三节构建档案

## 1. 结论与指标速览

**尚未启动，以下只有已定输入口径和验收目标，没有本次构建实测结论。** 具体阶段命令与启动前置见[launch.md](launch.md)，本README在运行后逐节回填。

| 指标 | 计划值 | 本次实测 |
|---|---:|---|
| episode数 | 400 | 待构建 |
| 总帧数 | 189035 | 待构建 |
| 执行样本数 | 189035 | 待构建 |
| packed布局 | framesamp-4x4-v1 | 待全量verify |
| 逻辑/分配字节、耗时、峰值 | 以阶段实测为准 | 未测 |

## 2. 用户原话与范围

用户原话：「恢复计划中的建库，严格按前置闸门推进」。数据、库名与统计量口径按[原版80k计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)沿用；本库只交付source、4×4 packed和统计量，不生成8×8、Wan或motion。

用户另明确「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」，随后批准「允许这四键缺失与 None 等价」。仅 `motion_emb/motion_pos/motion_mask/mem_order` 缺失与显式None可等价，其他键或非None值不在许可内；原始dtype/raw SHA、精确数值、signed zero及训练bitwise要求保留。这些输入许可不放宽建库原始内容和子集字节守卫。

## 3. 版本与代码状态

实际Beta提交：**待本档案和代码提交后，在起跑日志填入完整40位SHA**。正式构造前工作区必须clean并等于该锚点；启动时间、结束时间、阶段实际HEAD均待记录。禁止把当前开发HEAD、占位符或事后提交当成启动版本。

已有P0启动提交为 `5489e4b3a92197d0e9a37421b1e6415b3022b613`，这是环境准备事实，不是本库未来Beta；详见[环境档案](../../training-doc/orig80k-env-0925/result.md)。

## 4. 原始来源与输入pin

同一公开集合的 BinFill、PickXtimes、SwingXtimes、StopCube 四个H5；新目录 `/scratch/hongze/robomme_data_h5_counting4` 只建立到公开原件的同盘硬链接。

重新hash得到的 `input_manifest.json` 必须与已提交 `docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json` 的对应文件size/SHA一致；本库episode清单再与同处全集清单核对。参考提交、新旧清单SHA及 `SOURCE_PROVENANCE` 原文待运行后回填，不仅对现场清单自证。

## 5. 任务、划分与样本口径

四任务各100集，是公开全集的严格子集；没有demo段，所以执行样本数等于总帧数。按物理episode与局部step对齐，不能把全局epis_idx直接当跨库身份。

总帧与执行样本不得混算；清单中的连续全局编号/offset必须自洽。分片调度字段允许按本机安排变化，不允许改动真实episode内容、物理身份集合或已定选择范围。

## 6. 启动命令与配置还原

完整绝对路径、环境覆盖项和阶段命令均见[launch.md](launch.md)。实际展开命令、完整 `BUILD_HEAD`、clean状态、tmux名称、精确PID和UTC应同步写入阶段日志，再将实测记录归档。

源码由 `git show <实际Beta完整SHA>:scripts/dataset/run_local.py` 及launch列出的其余入口还原；冻结依赖由同提交的 `pyproject.toml/uv.lock` 还原。档案不复制脚本或yaml，不用硬编码代码行号定位逻辑。

## 7. 构建参数与格式

manifest覆盖 `--episodes-per-task 100 --num_shards 1`；SigLIP使用GPU0–7及 `--require-free-mib 70000`；finalize为 `--input_level sha256 --spot_check 1024`，使用GPU7。pack为默认4×4、48进程，verify全量检查、48进程；CPU阶段显式设置 `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`。

source的pkl和逐帧features保留当前生产链；packed为 `framesamp-4x4-v1`，不以训练主机dtype许可改变pack内容。独立计算 `v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep/robomme/norm_stats.json`，后续训练使用该库自己的统计量。 所有其他默认值由实际Beta源码锁定，不静默覆盖。

## 8. 硬件、存储、预算与耗时

环境B：8×A100-SXM4-80GB，本地NVMe RAID `/dev/md0`、XFS。最近快照可用 `1792014577664 B`，约1668.9436 GiB，使用率77%；启动前和每阶段结束后重测，运行硬件/存储证据随日志归档。

当前预算审查是**估算**：full/count/smoke分配量加10%共938.1027 GiB，临时等分项保守估计128 GiB，再保留300 GiB，合计1366.1027 GiB。展示值已舍入，实际闸门使用完整字节分项；冒烟后按实测校正，本阶段起跑时按剩余工作重算。耗时和峰值均未实测，不把历史环境数字抄成本次结果。

## 9. 构建过程与阶段日志

目前没有本库运行进程，实际tmux清单为空；各阶段的待填表在[launch.md](launch.md)。阶段预计超过5分钟即用detached tmux，`PYTHONUNBUFFERED=1`、`set -o pipefail`、`tee`与最终 `EXIT_CODE=` 齐全。

每阶段保存任务及tee退出状态，日志写失败也不能算成功。失败保留现场、停止后续；只按本轮记录的完整名称管理会话，不全局kill、模糊匹配或批量清理。

## 10. 验收与内容一致性

来源pin、episode身份与规模、SigLIP全部worker、finalize、packed全量verify及统计量均待实际验收。不得仅统计目录或读到少量样本就判成功；所有必需阶段退出0，packed元数据必须是完整verified且绑定本次清单。

与 `16task-pub-1600ep` 按物理身份全覆盖比较400集、189035样本，features逐字节、pkl全部字段严格比较；只有epis_idx按manifest显式映射。 训练可读性、输入对拍、训练标量/状态及perf仍有自己的闸门，本档案未运行这些项目，不标通过。

## 11. 计划外事件与处置

尚未起跑，暂无本库构建异常。若发生源文件/pin、字段、数值、资源或工具问题，记录阶段、命令、原始错误和现场路径，立即向用户报告；不自行降低预算或一致性要求，不用覆盖选项重置输出根。

3样本CPU输入取证与四motion键的许可属于独立训练输入讨论，不将其历史schema FAIL改成本库构建失败，也不借新许可掩盖任何其他字段差异。

## 12. 当前结论与下一步

当前仅完成起跑前档案准备；后续按launch前置检查及阶段命令推进。每阶段完成后回填真实指标、完整HEAD、时间与判定，所有阶段和额外交付闸门应分别报告，不能把“数据已生成”写成“训练等价已证明”。

与 `16task-pub-1600ep` 按物理身份全覆盖比较400集、189035样本，features逐字节、pkl全部字段严格比较；只有epis_idx按manifest显式映射。 原始H5永久保留；新库和失败现场不自动删除。任何本轮临时大产物清理须先确认归档完整和归属。

## 13. 归档文件清单

目前只有 `README.md` 和 `launch.md`。运行后在本目录 `records/` 归档Git无法还原的实际清单/摘要、阶段判定、清洗日志、norm_stats与比较结果、实际字节和资源记录；必要时另写结果正文。文件清单、大小、SHA与实际来源在收尾补齐，不预写未生成文件为已存在。

已提交的原始pin、脚本、yaml与配置只通过提交引用还原，不重复拷贝。H5、features、packed数据、模型权重及完整缓存留在明确的 `v1-store/` 实体目录，不进Git。清洗日志须核对退出记录和关键指标数量和值均未丢失。
