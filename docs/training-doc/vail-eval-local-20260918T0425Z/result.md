# 本机 RouteStick 单回合结果

**链路通过，任务成功 1/1，完整视频 321 帧，退出码 0。** 本结果来自当前 MotionJEPA 工作副本的全新运行，不使用其他评估仓库的结果。

| 项目 | 实测值 |
|---|---|
| 工作副本 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA` |
| 主仓库分支 | `v2-vail-eval-0917` |
| 两端共同启动 HEAD | `e3e5dd398f182fdd0d10305f47ffbff3b363c4fd`，启动时干净 |
| benchmark | 官方 `RoboMME/robomme_benchmark`，`856bc3a189d4172f3f47dbee4424d585f8d78db3`；未修改、未建立分支、未切换 fork |
| 硬件 | sled-vail，RTX 6000 Ada，GPU 1，空闲检查通过 |
| 权重 | bucket `HongzeFu/robomme-vla-modul-60k-v1`，末步 59999 |
| 评估条件 | RouteStick/test/episode 0；策略 seed 7；环境 seed 660000；easy |
| 记忆 | 8 帧 × 64 token，budget 512，modulation，motion 关闭 |
| 源码检查 | 策略、openpi-client 与 benchmark 均来自本工作副本 |
| 模型检查 | 61/61 参数路径与形状匹配，missing/extra/shape_mismatch 均为 0 |
| 配置与资产 | 冻结配置、norm_stats、motion 关闭检查通过；下载 28 项 SHA256 全通过 |
| 渲染 | 原生 SAPIEN；两路相机 256×256×3；探针通过 |
| 策略结果 | `{"RouteStick":{"0":true}}`，成功率 1.0 |
| 视频 | 321 帧，967,126 字节，全部帧解码通过 |
| 起止时间 | 2026-09-18 04:45:24–04:50:59 UTC |
| 总耗时 | 335 秒，包含源码检查、渲染探针、模型校验、加载、首次编译、回合与视频验收 |
| 收尾 | EVAL_PASS，EXIT_CODE=0，自己的 tmux 与服务进程退出 |

## 来源与验证边界

服务端入口为本仓库 `scripts/training/serve_policy.py`，客户端入口为 `examples/robomme/eval.py`，仿真代码为 `third_party/robomme_benchmark`。两者均通过 uv run --frozen --no-sync，使用 v1-store/envs 下的两个独立环境和 NFS Python 3.11.14。本机 uv 0.10.2，集群 uv 0.8.22；两端共享同一锁定环境。原训练 .venv 未重新安装或同步。

训练留档包含 RouteStick，但 bucket README 的任务列表与留档冲突；本轮记录差异，未修改远端。此次只证明一个 test 回合及两端链路可运行，不外推完整任务成功率。

本机与集群视频帧数都是 321，但文件字节数不同，未做动作或像素逐位对拍，不宣称轨迹一致。总耗时含 NFS 读取和编译，不作为纯推理吞吐对比。

此前本机 `vail-eval-local-20260918T0400Z` 在前置渲染阶段发现 GPU 被其他任务占用后主动停止，退出 143，尚未执行策略回合。用户选择等待空闲，随后通过 GPU 硬闸才启动正式回合；没有更换 seed、补样本或为成功率重试。首次记录另存，不覆盖。

## 证据与产物

原始运行、服务端、客户端、源码与参数检查日志，以及 render.json、progress.json、log.json、check.json 均在 [records/](records/)。准备阶段的环境日志与 6 项控制流、4 项结果验收测试见 [准备结果](../vail-eval-0917-setup/result.md)。新增 GPU 占用硬闸经模拟占用验证：退出 2 且不创建运行目录。

完整视频位置由 records/check.json 记录，运行产物根为：

```text
/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/vail-eval-local-20260918T0425Z/
```

权重和视频未提交 Git。评估完成后保留既有交互作业 61331555，不终止用户其他进程或会话。
