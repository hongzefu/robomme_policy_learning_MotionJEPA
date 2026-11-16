# 独立环境与权重准备结果

准备通过。服务端安装 208 个包（下载准备 7m01s、安装 4m54s），客户端解析 235 个包、安装 232 个包（准备 4m55s、安装 4m36s），最终 EXIT_CODE=0。环境均为当前仓库 v1-store/envs 下的实体目录，解释器指向 NFS Python 3.11.14；原训练 .venv 未执行同步或安装。

首次客户端解析生成锁文件，已提交为 438be57；后续 setup 两端均使用 frozen 安装。tokenizer 从本仓库 v1-store/models/big_vision 复制进评估缓存，源与目标 SHA256 均为 8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6。

权重只下载 59999 与根目录来源文件，28 项 SHA256 全部通过。独立服务端检查结果：PARAM_TREE_EXACT=PASS，模型与 checkpoint 均 61 个参数叶子，missing/extra/shape_mismatch 全部为 0；冻结配置、归一化统计、motion 关闭检查通过。

客户端源码检查通过：robomme 来自当前仓库 third_party/robomme_benchmark，openpi_client 来自当前仓库 packages/openpi-client。benchmark 官方来源、856bc3a，源码干净；RouteStick test episode 0 的 seed=660000、difficulty=easy。未建立 benchmark 分支。

新客户端环境执行 6 项控制流测试和 4 项完整视频结果测试全部通过；无效 checkpoint 路径退出 2，shell 与 diff 检查通过。原始准备及验证日志见 records/。准备会话 mv-eval-uv-0917、mv-eval-ckpt-0917 自然退出。两端真实回合另行留档，不以这些准备检查替代。
