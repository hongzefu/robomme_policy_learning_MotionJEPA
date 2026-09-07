# tic-vulkan-makeenv —— 结果

> 起跑 commit `9b3b95f`（clean HEAD；脚本版本 commitV7.1 `a8cfa17`）。2026-09-07 21:05:25 → 21:14:04（8 min 39 s），环境 B，GPU 7，micromamba `robomme` 环境，tmux `tic-vulkan`（结束自行退出）。
> 原始日志 `records/vulkan.txt`；逐轮指标 `records/probe-tic-{baseline,pin,tls8192}.json`；排查阶段（同脚本未提交版本，同日同卡）的六档 TLS 扫描 `records/tls_scan.txt`、结论行 `records/rootcause.txt`、C 层最小复现三份 `.c`、两份 patch。**三段全部按预期：基线第 28 轮崩、两种修法各 35 轮不崩，`EXIT_CODE=0`。**

## 一、判定行原文

```
[F1] CRASH at round=28 err=RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver
     PROBE_RESULT tag=tic-baseline rounds_done=28 crash_round=28
[F2] PROBE_RESULT tag=tic-pin     rounds_done=35 crash_round=-1        （--pin-renderer）
[F3] PROBE_RESULT tag=tic-tls8192 rounds_done=35 crash_round=-1        （GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192）
VULKAN_ROOTCAUSE=glibc_surplus_static_TLS_exhausted_by_svulkan2_Context_recreate repro_crash_at=28 fix=pin_render_system_in_env_runner(备选:GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192) fixed_runs=35
```

## 二、复现数据（F1 基线：不起 policy、不 reset、纯 `make_env` → `close_env`）

| 轮 | fd | fd(/dev/nvidia*) | vk/nvidia 库映射数 | RSS MB | 显存 MB | sapien 对象 | 本轮耗时 s |
|---|---|---|---|---|---|---|---|
| 1 | 155 | 130 | 85 | 1602.9 | 774 | 12 | 17.69 |
| 5 | 155 | 130 | 85 | 1938.1 | 774 | 12 | 6.94 |
| 15 | 155 | 130 | 85 | 2132.7 | 774 | 12 | 6.37 |
| 25 | 155 | 130 | 85 | 1960.8 | 774 | 12 | 6.86 |
| 27 | 155 | 130 | 85 | 2058.9 | 774 | 12 | 6.89 |
| **28** | 69 | 51 | 5 | 1533.2 | 492 | 11 | 2.24 → `ErrorIncompatibleDriver` |

**前 27 轮所有用户态可见量完全平坦**（fd、库映射、RSS、显存、SAPIEN 对象数均无趋势），第 28 轮的骤降是崩溃的*结果*（Context 建失败后库被卸载），不是原因——泄漏藏在 ld.so 的静态 TLS 记账里，`/proc` 看不到，这就是此前只能划「≤27 集」红线而定位不到原因的缘故。与 `docs/training-doc/eval-official-framesamp-context/result.md` 记录的 w0/w1 第 28 次崩溃完全一致。

## 三、根因（因果链，每环有实测）

1. ManiSkill `BaseEnv._setup_scene()` 每次 `gym.make` 新建 `sapien.render.RenderSystem`；`env.close()` → `BaseEnv._clear()` 置 `scene=None` 并 `gc.collect()`，svulkan2 全局渲染 Context 引用计数归零即析构。**每轮 `make_env` = 一次 `vkCreateInstance`，每轮 `close_env` = 一次 `vkDestroyInstance`**，同时把 NVIDIA 渲染库组 dlopen / dlclose 一遍（`VK_LOADER_DEBUG=all` 下 `vk_create` 逐轮 +1；崩溃轮库映射 85 → 5、fd 155 → 69）。
2. NVIDIA ICD（`libGLX_nvidia.so.0` 及其拉起的 `libnvidia-*`）使用 initial-exec TLS，dlclose 无法回收已占的 glibc surplus 静态 TLS 槽——**每轮 create → destroy 净泄漏恰好 64 字节**。
3. glibc 2.34 默认 `glibc.rtld.optional_static_tls=512`，可用余量支撑 26 轮；第 27~28 次 dlopen ICD 失败 → loader 无可用 driver → `vkCreateInstance` 返回 `VK_ERROR_INCOMPATIBLE_DRIVER (-9)` → svulkan2 打印 `Your GPU driver does not support Vulkan` 并抛 `vk::createInstanceUnique: ErrorIncompatibleDriver`。

**定量铁证**（C 层最小复现 `records/vk_device_limit.c`，每轮 create + destroy，不跑任何工作负载）：崩溃轮数与 TLS 余量严格线性，六档零误差落在 **崩溃轮 = 19 + optional_static_tls / 64**：

| `glibc.rtld.optional_static_tls` | 0 | 128 | 256 | 512（默认） | 1024 | 2048 |
|---|---|---|---|---|---|---|
| `vkCreateInstance` 失败于第几次（`records/tls_scan.txt`） | 19 | 21 | 23 | **27** | 35 | 51 |

8192 档未进 C 层扫描表；其落盘证据是 Python 层 `tic-tls8192` 35 轮不崩（本 run F3），按 `19 + 8192/64 = 147` 只是外推值。

环境：NVIDIA 驱动 595.71.05（open kernel module）、glibc 2.34、SAPIEN 3.0.3、Vulkan loader 1.3.224（sapien 自带 `libvulkan.so.1.3.224`，ICD 走 sapien 自带 `nvidia_icd.json` → `libGLX_nvidia.so.0`）。

## 四、五条假设的证伪

| 假设 | 结论 | 证据 |
|---|---|---|
| (a) Vulkan instance 未销毁累积 | 证伪，且方向相反 | `records/vk_instance_limit.c`：instance 不销毁连建 64 个全成功；instance + logical device 都不销毁连建 48 个全成功。**销毁-重建循环才致命，累积不致命** |
| (b) 驱动每进程 instance 数上限 | 证伪 | 同上，64 个 instance / 48 个 device 同时存活无错，fd 恒 4 |
| (c) fd 上限 | 证伪 | `ulimit -n` 1048576（硬限同值）；fd 恒 155、26 轮零增长 |
| (d) SAPIEN 渲染系统 / shader 缓存单例 | 两半 | `--gc --renderer-release`（`sapien.render.clear_cache` + `gc.collect()`）仍崩第 28 轮 → shader 缓存证伪；但 Context 确是引用计数式全局单例——钉住它即根除（修法 1） |
| (e) 显存 / host 内存耗尽 | 证伪 | 显存恒 774 / 81920 MB；RSS 1.5–2.1 GB 无趋势；VmSize 恒 ~19 GB；threads 恒 137 |

## 五、两种修法（本 run 各 35 轮实测不崩）

| 修法 | 做法 | 本 run 实测 | 附带效应 | patch |
|---|---|---|---|---|
| **1 钉住 RenderSystem**（推荐） | `EnvRunner.__init__` 建一个 `sapien.render.RenderSystem(sapien.Device("cuda"))` 长期持有，Context 引用计数永不归零，全程只 1 次 `vkCreateInstance` | `rounds_done=35 crash_round=-1`；库映射 / fd / 显存全程恒定 | `make_env` 均耗时 6.83 s → **1.19 s**（省掉每集重建 Context 与重编 shader）；改变 Context 生命周期，**成功率一致性未验证**，正式采用前建议同 seed 对拍一轮 | `records/fix-b-pin-renderer.patch`（落点 `scripts/training/legacy-eval/robomme-{local,remote}/env_runner.py`） |
| **2 抬静态 TLS 余量**（最保守） | 起 `eval.py` 时 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192` | `rounds_done=35 crash_round=-1`；按 64 B/轮线性外推可撑约 147 轮 | 纯环境变量、不改渲染语义；每集仍重建 Context（6.9 s/集） | `records/fix-a-glibc-tunables.patch`（落点 `scripts/training/legacy-eval/eval_shard.{local,remote}.sh`） |

**落点判断**：两者都落 `scripts/training/legacy-eval/`（主线自 commitV6.2 不维护评估，`examples/robomme/` 停在 `4b7a710`），主线 `examples/robomme/` 与 `src/` 零改动；**两份 patch 均未应用**，由用户决定采用哪一个。评估口径影响：「单进程 ≤27 集」红线在采用任一修法后可解除；本轮组 E 的 24 集探针评估仍按 ≤27 口径跑，不依赖修法。

## 六、异常处置

无。F1 的崩溃是预期复现（`F1_RC=0` 因脚本捕获异常后正常退出）；三段一次跑完，tmux 会话随命令结束自行退出，未执行任何 kill。
