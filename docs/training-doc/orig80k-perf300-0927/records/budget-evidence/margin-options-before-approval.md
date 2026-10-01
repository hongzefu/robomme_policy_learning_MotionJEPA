# normal perf实测后的余量方案依据（待用户选择）

本笔记只整理root已经提出的A/B方案及实际依据，**两方案均未获用户批准**。不生成有效margin-source、prod预算或预算PASS，不改变原测量。来源Beta为 `70a641a827eb2559c3acb69430f55c1c53babaeb`；两GPU、两恢复、report、两measurement共七阶段已完成，第八预算仍待决定。

## 已闭合的计量来源

正式paired-speed为9,584 B，SHA `dd9bdf145be4642099e4b285b1e5fcd984f9feba5187a134441a4067b6b5f9d8`。full receipt SHA `7b3458c7f196e4657e57280c74d185e03a231a4043b4be36f179d225d8d7c0a2`，count receipt SHA `162f4fbf3dcbd0ee22bb80c08126f6bbd2d796c980d6f402c7264d122a116a4c`；二者都在各自REC的 `checkpoint_measurement.json`，由正式measurement在299仍存在时生成，不是本笔记补造。

| 值（B） | full | counting |
|---|---:|---:|
| C：单299目录allocated | 11,879,944,192 | 11,879,923,712 |
| 单299目录logical | 11,879,892,504 | 11,879,871,342 |
| F：最终整run根allocated | 11,879,968,768 | 11,879,948,288 |
| P：采样整run根最大allocated | 16,039,718,912 | 13,251,739,648 |
| E=max(0,P−F)观测下界 | 4,159,750,144 | 1,371,791,360 |

既定两侧各8份的实测保留基线为190,078,943,232 B；两侧E合计5,531,541,504 B，约5.151649 GiB。C与F范围不同，不能混用。采样分别1004/999次，目标0.5秒、实际均值0.512782/0.511879秒、最大3.757910/3.725057秒、missed ticks26/24。错误行均0，count中间有1行路径变化/消失竞态，正式report已按原口径保留；这不让离散峰值变成连续上界。

## 两run真实日志、记录与cache占用

[只读stat记录](normal-perf-logs-cache-stat-20260927.json)为166,370 B、SHA `674e47fc9cfed9894fd21641e43cc4db7b4a3f254e17b902c7792e88e0011f61`，观察时刻2026-09-27T02:36:48.592634+00:00。只对两run的REC、driver、各自GPU/恢复/measurement三份wrapper、JAX/CUDA/W&B/XDG目录作两遍稳定lstat；8个W&B链接均在显式roots内，链接不递归，实体按(device,inode)去重。未扫描checkpoint树或读取权重内容。

| 类别allocated B | full | counting |
|---|---:|---:|
| REC小记录及measurement | 1,441,792 | 1,441,792 |
| driver | 65,536 | 65,536 |
| 各自三个wrapper日志 | 73,728 | 73,728 |
| JAX cache | 2,846,720 | 2,842,624 |
| CUDA cache | 12,288 | 12,288 |
| W&B日志 | 385,024 | 393,216 |
| W&B cache | 4,096 | 4,096 |
| 合计 | **4,829,184** | **4,833,280** |

空的W&B config/data及XDG目录也计入stat账，其allocated为0。两侧合并去重为9,662,464 B，约9.22 MiB。联合report、共享调度/独立报告和归档副本不计入“每run”表，不能把该数声称为本轮所有小文件总量；未来prod也不生成全部实验档案。

## 哪些会随运行增长

- 普通metrics按log_interval100记录；本次300步3行，80000步预计800行。W&B history与训练打印随日志次数增长，初始化横幅、配置和最终EMA/尾窗记录主要为固定成本；这不是把所有现有文件统一乘80000/300。
- GPU CSV由共同runner每0.5秒采样，随运行时间增长。本次实测约297.16/298.18 B/s；仅以当前报告ETA做量级外推，约23.10/23.30 MB每run。采样格式、实际时长和额外输出可变，这不是日志硬上界或运行承诺。
- step_timing、host_samples、disk_samples属于本次perf的speed包装，正式prod直接进入原train，不会生成这三份perf专用流。不能将其全部按80k线性放大并混称正式需求。
- JAX/CUDA缓存主要由编译和实际形状/图生成，不由step计数直接决定；固定形制通常复用已有编译，但当前工具没有给未来全程cache容量设硬上界。W&B本地日志/同步缓存也不能只凭短测9MiB就证明长期不增长。

## root提出的两个候选数值

以下是候选条件算术，不是已批准配置。每份14/16GiB均按未来16份总保留量计算；“盲区余量”另加在实测E之后，64GiB为新增logs/cache工程余量，不抵扣当前已经占用的空间。

| 分项 | A：14GiB/份、28GiB盲区、64GiB日志/cache | B：16GiB/份、32GiB盲区、64GiB日志/cache |
|---|---:|---:|
| checkpoint_total_bytes | 240,518,168,576 | 274,877,906,944 |
| 对应checkpoint_growth_margin_bytes | 50,439,225,344 | 84,798,963,712 |
| save_sampling_margin_bytes | 30,064,771,072 | 34,359,738,368 |
| save_temporary_peak_bytes（E+余量） | 35,596,312,576 | 39,891,279,872 |
| logs_cache_margin_bytes | 68,719,476,736 | 68,719,476,736 |
| 与300GiB取max后的条件总需求B | **344,833,957,888** | **383,488,663,552** |
| 条件总需求GiB | **321.151649** | **357.151649** |

61EMA叶shape/dtype的未压缩payload为12,937,988,704 B，约12.049441GiB；14GiB比它多2,094,396,832 B，16GiB多4,241,880,480 B。但保存实际有zstd/OCDBT容器和文件系统分配，当前证据没有给出最坏扩张与容器开销的完整数学上界，所以14/16GiB应表述为可审阅的工程估计，不是已证明单份硬上界。

28/32GiB盲区余量相当于为两run各一笔在途保存再提供每侧14/16GiB工程缓冲，另保留本次5.151649GiB已观测增量；依据是当前每manager先等待前次save完成、两run可并发。单笔内部临时写入和扫描盲区仍未被连续测量证明上界。64GiB远大于本轮已观测小文件，但它同样是覆盖长期新增与未知波动的工程余量，不能由倍率本身推出充分性。

选择后才固定margin-source实际字节/SHA、构造schema2预算并按当时真实可用空间运行既有消费者；如果预算调用版本不同，单列动作版本，不改两source_perf的真实70a641a HEAD。当前没有批准数值、预算文件或预算PASS。
