# timing20首步差异：只读机理核对

审查固定提交 `b0efbde61e38411fb1b9114eec8485d36a4ee9a0`，起止HEAD相同、porcelain为空。本说明只读该版本源码、四run已完成的launch/runtime及JSONL第一条完整行；没有改代码、容差、argv、环境或运行任务。**未找到已被证实的根因，不能把假设写成“计时包装导致”或“XLA autotune导致”。**

## 已观察到的事实

四run均为 `timing20-{full,count}-{off,on}-0926-20260926T231534Z`。本次独立读取第一条完整fetch/model记录，两个库各自off/on的首批tree、首个模型tree和RNG均相同；首步五标量中loss、grad_norm、llm_grad_norm、mem_enc_norm不同，param_norm相同。full loss为off `0x1.9958ca0000000p-4`、on `0x1.994db00000000p-4`；count为off `0x1.7670760000000p-4`、on `0x1.763a400000000p-4`。这是严格不一致，不设容差。

两库各自launch.json的完整配置只在exp_name和checkpoint派生路径不同；argv只变run名，data/reference/GPU/package/uv.lock/sys_prefix/storage相同。另有各run JAX/CUDA/W&B缓存路径及可用磁盘快照差异。runtime.json各对唯一差异为exp_name：mesh=(1,4)、device_count4、b64/fsdp4/workers4/seed42及设备相同。已记录环境的XLA_FLAGS、JAX_PLATFORMS、JAX_ENABLE_X64、默认matmul精度均未设置，显存比例0.95，CPU解析x64=False。

这只证明launch/runtime已记录字段相同，不证明全部进程环境相同；完整metadata.fingerprint仍应由最终验收读取。独立新缓存目录是既定设计，但目前没有编译产物/算法选择证据，不能据此断言编译器为何不同。

| run | launch.json SHA256 | runtime.json SHA256 | metrics第一完整行：字节 / SHA256 |
|---|---|---|---|
| full-off | `52486df36b64b11c7eb2c9ada1b3b57bbcc1c52a617a342127901fbe68d8b5e3` | `416b89338cf76db11e9ac4fe6259374ec01be13969af7e4684e5909c12558d4c` | 414 / `5f2a777b5b76b8d40a8428dbd3befe662e36481d8a78cf339bbfda3b6ade870a` |
| full-on | `a6a245d47febc0b67bcf8bf3e20c336e3a0c99a98c7748e45097a05d7fda26cd` | `281147144211ed7b5f269036ece322a2aedc7779dbdf04a77e94b02740c2389e` | 416 / `90cab971ddfc0bcdf7d00eeb9c38bef7d16dd051f920770c5eececaf775a8054` |
| count-off | `ee0a65ad6cf8db098560bbd3ceb1444741c135f4e2d6530d71e1e117ece889b9` | `a7ba62bba530a6691aed90ef72641f72fda92b41f27bd0edb81cc56d424cd0a1` | 414 / `d6a728fe4db6a9c673a6eb84c2682871c321e19c04d41c0af6a83bc8ec2d463d` |
| count-on | `bf4161a15796da8e3d0843cbc17f46f6d42247b0eddf82c882342d40f7f3e9a9` | `32e264d3b540ded87dbead3e3680bc5c75830889723f38c3321109dcdab55bdf` | 415 / `ca9db08f0cc700947e10c0b08d97299ca272a8c30f395b87b32d49d903110d3c` |

原路径为主仓 `v1-store/bench/orig80k/<完整run名>/`；metrics摘要仅绑定含换行的第一条完整行，不冒充当时仍可写文件的整体摘要。首步原证据另由独立代理保存，本说明不重写其记录。

## 源码能证明的差别

`check_orig80k_timing_equiv.py::Observer.compile`把原jax.jit及其kwargs原样调用，再把原rng/state/batch交给compiled；共同层只读记录batch/RNG，没有在此替换训练数学函数。off在共同层中直接runpy train.py，on经 `check_orig80k_speed.py::run`，增加HostTiming、TRAIN_TIMING_STEPS=20、阻止profiler、loader/save透传包装，以及在入口前启动host/disk两个采样线程。

`HostTiming.step/phase`第0步没有调用flush；smoke20只在第19步checkpoint phase真正保存前同步，step99不存在。因此末步同步不能直接解释发生在第0步的差异。train.py::main先得到ptrain_step的jax.jit callable，再建立timer；这**不等于XLA编译已完成**，首次实际调用仍可能触发编译。

`observed_loader`只读取已有属性，`TorchDataLoader.torch_loader`只是return原对象；`HostTiming`的普通phase只计时。host_sample只读/proc、/dev/shm和磁盘统计，DiskSampler只读文件元数据；没有源码证据显示它们改动RNG或输入数值。save包装在末步才执行，并原样转发原保存调用。源中TRAIN_TIMING_STEPS没有出现在模型/数据计算分支，只在train.py::main选择计时钩子。

## 初始状态与norm的证据边界

`Observer.save`仅在末步19记录完整state；`Observer.compile`记录batch/RNG而不记录每次传入的state。因此已有“输入和RNG相同”**没有证明两次初始TrainState逐叶相同**。init_train_state按相同seed和配置走同一入口并初始化step=0，但本次没有独立初始state字节摘要。

`train.py::train_step`的param_norm仅对筛选后的kernel计算global_norm，排除了bias/scale/pos_embedding/input_embedding及一维项；它既不是全state摘要，也是多对一聚合。相同param_norm不能推出params、EMA、optimizer或其他状态逐位一致。不得用这一标量排除初始状态或编译/运行差异，也不能由此声称初态已经不同。

## 可检验假设与追加复验的意义

一项假设是normal env的独立进程/独立新缓存运行本身不能逐位复现；另一项是假定包装新增线程、I/O或执行时序与差异相关。目前没有足以二选一的证据，也没有编译/算法信息可归因具体backend机制。

用户已批准两库各追加一次相同配置20步off、4+4并行，仅作诊断。若旧off与新off仍不同，包装开关不是解释差异所必需的条件；若重复相同而原on不同，只能增强与运行路径相关的线索，不能凭一次重复就证明唯一根因或一般确定性。所有输入/RNG、20×5原始hex和完整末态继续严格比较，既不改normal env，也不加容差。原off/on正式judge的真实失败及其b0ef锚点保留，不用追加off结果替代或重写。
