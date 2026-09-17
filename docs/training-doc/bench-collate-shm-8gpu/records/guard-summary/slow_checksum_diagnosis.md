# 完整状态摘要慢的现场诊断

用户询问「这个具体还要多久 为什么慢」。本诊断不修改正在运行的代码，
只用 uv 的临时环境执行 py-spy 0.4.2，缓存留在主副本 v1-store/cache/uv。
被观测 PID 为 a1 的 3938855；工具提交为
24f8cc911c9973e6617d7f3144bb73e4cbd6e4de，源码为只读改前副本
7a681e2cf3f21bdab3f0b27ea44d19b0adfd0ee7。

先做一次 --nonblocking --json 栈快照，主线程当时在
_checksum_full_state → jax.device_get → ArrayImpl._value 的分片拼接路径。
单个快照不足以代表整段耗时，随后增加 10 秒、50 Hz 的非阻塞采样：

```text
py-spy record --pid 3938855 --nonblocking --duration 10 --rate 50 --format raw
```

profile-step2.txt 实际记录 347 个有效样本，Errors=0：341 个停在
_checksum_full_state 的 np.isfinite(arr).all() 表达式，5 个在 NumPy _all，
1 个在 ArrayImpl._value。即这段局部采样有 346/347 个样本处于有限值检查，
显示 CPU 上的额外全量扫描是当前明显慢点。该比例不是整份状态摘要的完整计时分解，
不能排除其他阶段还有设备取数、拼接和 SHA-256 开销。

另外在改前副本环境做了不启动训练的小样本对照：1048576 个连续零元素，
每种 dtype 各执行三次 np.isfinite(a).all()；结果保存在 isfinite-small.json。
bf16 为约 1.6–2.7 毫秒，f32 为约 0.14–1.17 毫秒。小数组的结果不能解释
真实状态为何需要数百秒，故本轮只确认慢点所在代码路径，没有将深层原因
断言为某个 dtype、内存布局或缓存机制；更细的物理原因仍待单独复现。

完整状态有 201 个叶子：params 61、ema_params 61、opt_state 78、step 1。
已经完成的 step 0 / step 1 摘要耗时分别为 489.586 / 467.405 秒；
据此估算剩余约 2–2.5 小时，用户已明确允许延长八卡占用并完成原定验证。
未因此减少摘要步、不放宽逐位判据、不修改校验工具或在途进程。

原始样本与栈快照随 records 一起归档。源码树、依赖锁文件和正式环境均未改动。
