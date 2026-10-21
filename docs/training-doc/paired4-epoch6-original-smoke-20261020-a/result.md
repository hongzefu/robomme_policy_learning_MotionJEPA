# 四对运动 token 训练冒烟结果

`TRAIN_SMOKE=PASS`。真实单窗重新提取、正常小表、原完整模型的一次有效更新，以及实际 EMA 权重的严格保存回读全部通过。固定输入损失逐项一致，`max_abs=0`。本次只证明训练接入与权重可加载，不证明任务收益或完整断点续训。

最终运行代码为 `4c83a702878ef6c601e30a5103d02e1cbb1fe8f9`，主实现合并提交为 `5428d98bbaddcae5fb70371f576e0dee147661dd`（`commitV11.32`）。美东 **2026-10-20 22:20:37 → 22:30:40**，从首次 GPU 尝试起累计 **603 秒**，含一次初始化修复、重新准备及最终文件校验，低于 1800 秒预算。`EXIT_CODE=0`；自有进程、tmux 与监听已退出，结束时八卡显存使用均为 0 MiB。

## 验收证据

| 判定 | 实测 |
|---|---|
| `RAW_CONTRACT=PASS` | 正常 pack/verify 保存一窗 `(1,8,256)` fp32，8192 B；扫描1行、0失配，读回与新提取 raw8 逐位相同 |
| `PAIRING=PASS` | 相邻四对哨兵、非连续输入及任意 fp32 位型逆映射均逐位一致；拒绝重复配对与错误 dtype/形状 |
| `WINDOW_BUDGET=PASS` | 0、1、159、160窗合法；161窗拒绝；640条模型 token 对应160窗 |
| `MASK_ORDER=PASS` | PE、mask、时间均按窗重复4次；1152行完整置换与独立排序结果一致 |
| `SOURCE_FRESH=PASS` | 33帧输入核对、151项来源元数据检查全部通过；独立上游函数重算 latent/raw8，两个失配计数均为0 |
| `REGISTERED_SOURCE=PASS` | 批准 final 原样注册为 `checkpoint_epoch_6.pt`，SHA一致，内部epoch6、`encoder`、v10架构 |
| `ORIGINAL_IMPL_UNCHANGED=PASS` | 原六个模型/转换文件与实施前固定基线逐字节相同 |
| `SCOPE=PASS` | 原训练实现、根依赖锁、在线/eval文件、资产锁及benchmark gitlink保持；改动仅训练接入及其离线来源与检查 |
| `REAL_FORWARD=PASS` | 完整 `gemma_2b + gemma_300m`、1152容量、B1、idx32，真实motion=4；原 `train_step` 完成一步 |
| `SAVE_LOAD=PASS` | 原 `save_state` 导出EMA，等待异步完成；原dtype严格加载65叶，路径/shape/dtype/字节一致；固定输入20项损失逐项相等 |
| `TRAIN_SMOKE=PASS` | 汇总核验全部阶段报告，并复核实际权重、合同文件的完整集合、大小及SHA |

原训练一步的 loss 为 **0.0919426679611206**，总梯度范数 **3.894103527069092**。有效 motion 输入梯度范数 **4.970556983607821e-05**，padding motion 梯度为0；内容投影梯度范数 **0.004958550911396742**。四个 motion 投影参数叶共有 **1,395,724** 个 fp32 数值真实变化。

原优化器、学习率日程及seed42保持；实际 float32 初始学习率为 `9.997165761888027e-09`。训练样本来自正常 loader 顺序消费的第33批；原父PKL32的20×8动作完整保留，经原模型输入转换后为 `(1,20,32)`。帧采样时刻为 `[0,4,9,13,18,22,27,32]`。

## 来源与产物

父库为 `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f`，父清单SHA为 `d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad`。本次派生清单SHA为 `878c118e66fbe2afeeefefbe2e54ba391added88e4ce6dcf8113ef5a10004bf8`，`manifest_scope=full` 只表示完整覆盖这33帧。

| 对象 | SHA256 |
|---|---|
| 批准encoder及注册副本 | `146efd425f7923428d630dfb7ee2a8f010f19b207d36519dba49454aaa373d30` |
| 原始33帧RGB | `ffbea3422d40c2fd8120ee9a379d70b818734e73ccda7d327cecbade2a67ac7a` |
| 新latent | `3d89fd9f578a283fdeba9b5bd2282b6163d857e2637ce1839b0fdb2821df56ae` |
| 新raw8 | `c9cbee080e2e47746faecfe220638ce4eae6d0952b77a21da98b0aece7d0f238` |
| 落盘resolved YAML | `cc7eacd1cd938e259bf936b4f1901fd0e1b98d4519c95efa38b45dca2d4a1843` |
| 参考及加载模型的固定输入损失 | `f250c6a0a4f8f6cd609b1967041e322aa045d5abdda50370ddd2596a501063a0` |

原始数据、latent、raw8、正常帧/motion表与模型权重都留在 `v1-store/smoke/paired4-epoch6-20261020-a/`。实际保存目录为 `policy/mme_vla_suite_b128_80k/paired4-epoch6-original-smoke-20261020-a/1/`；导出参数原始张量合计 **12,944,025,184 B**。完整模型叶摘要见 [exported_parameter_reference.json](records/exported_parameter_reference.json)，实际文件封口见 [model.json](records/model.json)。

## 资源与耗时

| 阶段 | 秒 |
|---|---:|
| 单窗新提取及独立数值复算 | 14.333 |
| 真实motion表最终pack/verify | 0.974 |
| 完整模型初始化 | 16.339 |
| 完整模型梯度诊断 | 56.518 |
| 原训练一步 | 49.679 |
| 原生EMA保存 | 165.798 |
| 严格回读及固定损失比较 | 172.383 |
| 完整模型检查入口合计 | 500.806 |
| 最终CPU文件汇总校验 | 27.594 |

这些都是本次首次编译及检查耗时，不能当作稳态每步训练时间。Torch单窗峰值为 **1445.636 MiB**（`max_memory_allocated`）；JAX分配器峰值为 **62,221,886,208 B，57.949 GiB**（`peak_bytes_in_use`），不是驱动或全进程显存峰值。模型进程主机最大RSS为 **65,309,065,216 B，60.824 GiB**。未加入高频驱动采样。

## 修复与审查

用户原话：

1. “/scratch/hongze/MotionJEPA/1020-paired4-modulation-plan.html 开始实施 有问题问用户”
2. “之后所有agent全部使用astra ultra”

所有实现、审查及后续子代理均采用 `gpt-6-astra / ultra`。四个实现工作树以非快进多分支合并保留提交；独立计划、各子任务、最终整合审查通过。数据证据另作446项CPU只读复核，`DATA_EVIDENCE_REVIEW=PASS defects=0`，详见 [reviews.json](records/reviews.json)。

模型证据独立复核为 `MODEL_EVIDENCE_REVIEW=PASS blocking=0`：16份受保护文件未变、9条报告摘要关联、真实输入摘要、65叶参数元数据及保存目录21个文件的集合/大小均核对一致；其中23叶为bf16、42叶为fp32。六个大权重文件沿用最终summary已完成的完整SHA验证，其余合同与小文件独立重算。

首次GPU尝试在显存统计前尚未初始化CUDA，报 `Invalid device argument`，进程耗时3秒，未生成latent/raw8。追加一行 `torch.cuda.init()` 后重新准备同一33帧夹具并重试。失败报告和日志保留在 [attempt1](records/attempt1/)；原夹具也保留在产物根 `reports/attempt1/fixture`。预算继续使用首次尝试的绝对起点，未重置。

整合前还修复了缓存调用签名遗漏、新表scope读写声明、回读前live参数引用保留，以及只信报告、不核实际权重的汇总缺口。离线定向回归 **79通过、2跳过**，跳过原因是历史40集测试库缺失；新八项CPU合同全部通过。Wan环境按固定子项目锁仅替换一个motion-jepa包；本机Git对象用于获取同一固定提交，远端URL与revision不变，主JAX依赖未改。

## 复核入口与边界

完整执行命令见 [launch.md](launch.md)。保留产物后可在仓库根运行：

```bash
UV_CACHE_DIR=/scratch/hongze/.cache/uv JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
uv run --frozen --no-sync python scripts/training/tests/check_pair4_contract.py \
  --suite summary --root "$PWD/v1-store/smoke/paired4-epoch6-20261020-a"
```

阶段JSON、日志、派生输入清单、参数参考与进程退出记录都在 [records](records/)，归档只包含Git无法还原的实测数据；权重和数据不进Git。`RUN_PROCESS_CLOSED=PASS`。

至此停止。在线/eval接入、正式全量数据、未来VAE缓存兼容与最终encoder、完整优化器/step/RNG恢复、任务效果评估均未在本次实施或验证。
