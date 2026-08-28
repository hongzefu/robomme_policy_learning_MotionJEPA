# greatlakes Slurm 提交规约（MotionJEPA）

当端到端验证需要 GPU、本地 GPU 资源不足时，可以 ssh 到 UMich greatlakes 集群提交
slurm job 跑训练 tentative。**以后所有 greatlakes 提交都必须遵守本文件；违反任何
"硬规则"前必须先与用户确认，不可静默放宽。**

## 登录认证（硬规则，不可静默放宽）

**优先复用 ControlMaster 主连接:认证一次，30d 内所有 ssh 操作（提交/查询）免认证、免手机；
没有 master 才需建立（仅这一次走 Okta 2FA）。** ssh 二次验证已从 Duo 迁移到 **Okta Verify**。
**仅在"建立主连接"那一次需要验证；此时必须先问用户用哪种方式（6 位 TOTP 码 / 留空触发
push + 数字匹配，强烈推荐 TOTP），不要默认或复用上次选择。** 先 `ssh -O check greatlakes`
判断 master 是否存活：存活就直接干活、不必问验证方式。详见下方「ssh 提交流程（ControlMaster
复用，Okta Verify）」。

## 资源约束（硬规则，不可静默放宽）

- `--account=chaijy2`：不能切换到任何其他 account（即便看到别的 account 可绕过排队也不行）；
- `--partition=spgpu`：不能用其他 partition（如 standard / gpu / largemem）；
- `--nodes=1` + `--ntasks-per-node=1`：永远只提单 node 单 task；
- `--gpus-per-node` ≤ 2 且 `--time` ≤ 00:30:00：日常调试默认 1–2 GPU、20–30 分钟内；
  如确实需要更长时间或更多 GPU，必须显式告知用户并征得确认，不可静默放宽；
- 默认 `--mem=32G`（足以跑 tentative）；**实测 `--qos=interactive` 在 chaijy2/spgpu 下报
  `Invalid qos specification`，不要再用** —— 默认不指定 qos 即可；遇到 `(AssocGrpMemLimit)`
  时先降 `--mem`，正确的 qos 名待用 `sacctmgr show assoc user=hongzefu format=qos`
  （或 `sacctmgr show qos`）查清；`(AssocGrpGRES)` 表示 chaijy2 账户 GPU 配额已被组内
  其他用户占满，此时只能等他们的 job 退出，不可换 account / partition 绕开。

任何超出上述 GPU 数或时限的生产长训均超出调试限制，提交前必须由用户显式确认。

## 路径可见性（硬规则，不可静默放宽）

greatlakes 计算节点唯一能看到的共享路径是 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/`。
本机 `sled-vail` 的其它路径在 slurm 节点上全部不可见，写进脚本会立刻
`No such file or directory`：

- `/home/...`（本机 home，包括 `~/...`）—— 注意 greatlakes 有自己的 `/home/hongzefu`，
  与本机 home 是**两个不同的目录**，内容不互通；
- `/data/...`（本机 data 盘）；
- `/tmp/...`、`/var/tmp/...`（计算节点的 `/tmp` 是节点自己的，跟本机无关）；
- 任何不以 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/` 开头的本机绝对路径；
- 相对路径（slurm job 的 cwd 不是本机当前目录；脚本内先 `cd` 到 NFS 绝对路径再用相对路径可以）。

所有 slurm 脚本里出现的路径——`data.data_root`、checkpoint / `runs_root` 目录、
`--output` 日志、yaml / config、norm_stats、wandb dir、`cd` 的工作目录、python
解释器路径——都必须落在 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/` 下，并写成绝对路径。

如果要用的数据 / 产物当前还在本机非共享路径，提交 slurm 前必须先 rsync 到
`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/` 下；不要尝试 mount / symlink / 把本机
路径硬塞给 sbatch。

## venv 可移植性（硬规则：解释器必须由 uv 安装在 NFS 上）

uv 默认把 managed Python 装在本机 home（`/home/hongzefu/.local/share/uv/python/`），
`.venv/bin/python` symlink 过去，在 greatlakes 计算节点上是**死链**。**禁止用手动
重链 / 改 pyvenv.cfg 的方式修补**；正规做法是把 uv 的解释器安装目录放到 NFS：

```bash
# 1. 解释器装到 NFS（已装好 3.11.14，重装其它版本时同样带这个环境变量）
UV_PYTHON_INSTALL_DIR=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/uv-python \
    uv python install 3.11.14

# 2. 重建 venv 时显式指定 NFS 解释器的绝对路径（防止 uv 抓回本机 home 的解释器）
uv venv --python /nfs/turbo/coe-chaijy-unreplicated/hongzefu/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
UV_LINK_MODE=copy uv sync    # cache 在本机盘、venv 在 NFS，跨设备必须 copy
```

这样 `pyvenv.cfg` 的 `home` 与 `bin/python` symlink 天然落在 NFS，本机与 greatlakes
双端可用，无任何手术。验证命令：
`.venv/bin/python -c "import torch; print(torch.__version__)"`。

## ssh 提交流程（ControlMaster 复用，Okta Verify）

greatlakes 登录节点（`greatlakes.arc-ts.umich.edu`）拒绝纯密码 ssh，需要
`keyboard-interactive`（密码 + **Okta Verify** 2FA，已从旧 Duo 迁移）。**核心策略:没有
master 就建立 master——认证一次，30d 内所有提交/查询免认证、免手机。**

`~/.ssh/config` 已为 `greatlakes` 配好 `ControlMaster auto` + `ControlPersist 30d`
（`ControlPath ~/.ssh/cm-%r@%h:%p`）。OpenSSH 复用原理:主连接认证一次后建立 control
socket，之后 `ssh greatlakes <cmd>` 直接复用已认证通道、**不再发起任何 SSH 认证握手**，
因此零密码零 MFA 零手机（slave 不认证，与服务器 MFA 策略无关）。

提交器:**`scripts/data-preprocess-GL/gl_submit.py`**（自包含，纯系统 ssh + ControlMaster，
不再用 paramiko）——逻辑就是"没 master 就建 master，再经系统 ssh 复用提交":

- **master 存活** → 直接提交，**无需任何凭据、不必问验证方式**；
- **master 不存活** → 用 `GLPW`(+`GLOTP`) 经 `pexpect` 驱动系统 ssh 建立主连接，再提交。

无参数默认只打印 squeue 队列状态；提交训练请传一条远程命令当参数
（如 `"squeue -u hongzefu"`）——会自动前置 `cd {REPO} &&`（远程 cwd 是 home，相对路径
否则找不到脚本）。

**密码安全（硬规则）:绝不把密码写入任何文件、commit、CLAUDE.md 或对话历史。凭据由
用户即时提供，仅经临时环境变量 `GLPW`（建 master 时必需）/ `GLOTP`（可选 6 位 TOTP，推荐）
传入，用完立即 unset、绝不持久化。**

### 标准流程

1. **先查 master**（纯本地 socket 检查，不出网、不需凭据、不需 sandbox）:
   ```bash
   ssh -O check greatlakes   # exit 0 = 存活可复用;非 0(255) = 需先建
   ```
2. **存活 → 直接提交/查询**（零认证零手机，不必问验证方式）:
   ```bash
   uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py "squeue -u hongzefu"
   uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py   # 无参数=只打印队列
   ```
3. **不存活 → 建主连接**（**仅此步需要验证方式**，推荐 TOTP、给一次码无需手机匹配）。
   gl_submit 在无 master 时会自动用凭据建连后再提交;也可设好凭据直接跑:
   ```bash
   export GLPW='<密码>' GLOTP='<当前6位码>'
   uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py "<命令>"
   unset GLPW GLOTP
   ```
   建好后 30 天内所有提交/查询走第 2 步、免认证；到期后重建。（skill `greatlakes-usage`
   的 `gl_connect.py` 也能单独建 master，与此共用同一个 socket。）

### Okta 两条路 —— 仅"建主连接"那一次需要，且每次先问用户用哪种，不默认 / 复用上次

建 master 时 keyboard-interactive 的 prompt 依次是 `Password:` 和
`Okta passcode (leave blank to initiate a push):`:

1. **6 位 TOTP 码（强烈推荐，最可靠、无推送时序风险）**:从 Okta Verify app 读当前 6 位码
   填 `GLOTP`。码每 30s 刷新且一次性，**拿到立刻发起**。
2. **留空触发 push + number challenge（不推荐，除非按「push 修法」处理）**:不设 GLOTP，
   SSH 端依次显示 `Successfully initiated Okta push` → `The correct answer is N`（用户在手机
   Okta Verify 选中 N）→ `Press enter to continue:`（**approve 后还要再按一次回车，模块才校验**）。
   **2026-06-19 摸清真根因:以前判定的"转达数字超时、错过 ~60s 窗口"不准确——真正原因是
   `gl_submit.py` / `gl_connect.py` / `gl_master.py` 的 pexpect 模式表里没有 `Press enter to
   continue` 这一条,发完空 passcode 就一直 `expect()` 干等到 TIMEOUT，根本没按那下回车，于是
   "看起来卡死/超时"。** 用这些现成驱动走 push 必挂;要么用 TOTP，要么按下方「push 修法」用增强
   驱动。**能用 TOTP 就用 TOTP（无回车握手、无数字转达）。**

pexpect 建连要点（见 gl_submit.py / skill 的 `gl_master.py`）:匹配 `Password:` 填 GLPW、
`Okta passcode` 填 GLOTP（空则触发 push）、首次 host key 提示自动答 `yes`，认证通过的标志是
远端回显 `CONNECTED_OK_MARKER`。本机无 expect/sshpass，故用 `uv run --with pexpect` 临时
拉 pexpect 驱动系统 ssh，无需 sudo 装包。**注意:这些现成驱动只覆盖 TOTP 路径——模式表里
没有 push 的 `Press enter to continue`，故走 push 会卡死（见上）。push 必须用下方增强驱动。**

#### push 修法（2026-06-19 实测一次过；仅当用户坚持用 push 时才需要）

若必须走 push，用一个增强版 pexpect 驱动（一次性脚本即可，密码仍只经 `GLPW` 不落盘），相对
现成驱动多做三件事:

1. **补两条 pexpect 模式**:`correct answer is (\d+)`（捕获数字 N，实时打印/落盘）+
   `Press enter to continue`（匹配到就 `sendline("")` 按回车，模块这才去校验 approval）。缺第二条
   就是"卡死"的全部原因。
2. **服务器输出实时落盘 + 把日志路径直接给用户**:`child.logfile_read = <每写即 flush 的文件>`
   （只记服务器→本地，不含密码），并把该日志绝对路径丢给用户自己 `tail -f` 看 N——别靠转述，
   转述既慢又易漏（数字行常是单字符、易被过滤器吃掉）。
3. **回车用 sentinel 文件握手**:驱动在 `Press enter to continue` 处阻塞轮询一个 sentinel
   文件（如 `/tmp/gl_push_go`），用户在手机点完 N、回话确认后再 `touch` 该文件放行、然后才按
   回车——**避免 approve 之前就按回车导致单次校验失败**。

完成后 `ssh -O check greatlakes` 应为 `Master running`，主连接建好，后续提交/查询零认证。
**结论不变:push 全程要"看数字→点→确认→放行回车"四步握手，TOTP 一步到位，优先 TOTP。**

### 已知坑

- **远程命令 cwd 是 greatlakes 的 `/home/hongzefu`，不是 REPO**:gl_submit 已对自定义命令
  自动前置 `cd {REPO} &&`，直接传 `"squeue -u hongzefu"`；提交新 Wan slurm 脚本前须先按本文件审批资源
  即可，**不要再自己写 cd**（会变成双 cd，虽无害但多余）。squeue 的 `-o '...'` 单引号在外层
  参数里是字面、远程 bash 才解析。
- **spgpu 强制至少 1 GPU**:提交 `--gpus-per-node=0` 报 `QOSMinGRES` / `Batch job
  submission failed`（2026-06-17 实证）。纯 CPU 的 sleep 测试 job 也必须带 `--gpus-per-node=1`。

## 现成的 slurm 脚本（2026-08-14 新增；2026-08-16 起 `scripts/dataset-build/` 下另有抽取入口，见下节）

`scripts/train-script-hongzefu/` 下带 `gl_` 前缀的三个脚本是**训练侧可直接 sbatch 的集群入口**，
其余（`smoke_/bf16_/filter10_*_local2gpu.sh`）都是本地 torchrun 包装，不能当 sbatch 提交：

| 脚本 | 资源 | 用途 |
|---|---|---|
| `gl_bf16_wan_smoke.sh` | 2 GPU / 8 CPU / 48G / 00:30:00 | 集群侧冒烟 v1（已于 2026-08-14 全绿跑通，见下表）：验 NFS venv、DDP、NFS 数据吞吐与 A40 上默认 batch 88 的显存；含每 20 秒一次的显存采样与峰值打印。run 名硬编码 `gl-wan-bf16-smoke`，跑完删 `runs/gl-wan-bf16-smoke/` 即可复跑 |
| `gl_bf16_wan_smoke_v2.sh` | 2 GPU / **4 CPU / 16G** / 00:30:00 | 冒烟 v2 = **CPU/RAM 降配口径（已验证，冒烟一律用它）**：2026-08-14 五 agent 对抗验证裁决档位，当天实测通过（job 57642560，见下面 v2 实测小节）——排队 <1 分钟、epoch 零退化、anon 峰值仅 3.85 GiB。训练配方与 v1 逐字节相同，新增 cgroup v2 内存采样（memory.current + anon/file/shmem 拆分）与 checkpoint 落盘体积硬断言。run 名硬编码 `gl-wan-bf16-smoke-v2` |
| `gl_bf16_wan2gpu.sh` | 2 GPU / **4 CPU / 16G** / **16:00:00** | bf16 长训，与本机 `bf16_wan_local2gpu.sh` 逐项同口径（有效 batch 352、SIGReg N=352）。**walltime 16h 与 2 GPU 超出下述调试限额，2026-08-14 经用户显式放行**；CPU/RAM 于同日经 v2 冒烟实测后由用户拍板「长训按照 4 CPU / 16G」，自带与冒烟 v2 同款 cgroup anon/file 采样，**首个 16G 档长训 run 结束须以采样峰值复核档位** |

### 2026-08-14 冒烟实测（job 57628076 @ gl1502，全量 v7 数据集 1 epoch，bf16、每卡 batch 88）

| 项 | 集群 A40 | 本机 RTX 6000 Ada（基准 run `wan-v7-bf16-72ep-a`） |
|---|---|---|
| epoch 耗时 | **480.8 s** | 379.6 s（首）/ 389.6 s（均） |
| 吞吐 | **185.21 samples/s** | 230–235 samples/s |
| `max_memory_allocated` | **35.74 GiB** | 35.74 GiB（**逐位相同**） |
| `nvidia-smi` 实占 | **40,049 MiB = 86.9%** | 40,604 MiB = 88.1% |
| **MaxRSS（主机内存）** | **50,005,212 K ≈ 47.7 GiB** | 未记录（本机 377 G 从未成为约束） |
| AveCPU | 00:18:04 / 8:50 墙钟 ≈ 2.05 核当量 | — |
| checkpoint 体积 | 954,852,403 B（live + EMA 双权重） | 955 MB |

结论三条：①**A40 比本机慢 1.27×**，72 epoch 外推 ≈ 9.6 h，16 h walltime 裕度 1.6×；
②显存放得下 batch 88，**勿改 batch**；③⚠ **MaxRSS 47.7 GiB 不能当「真实需要 48G」的证据**
——本条为 2026-08-14 当天对抗验证的订正（旧表述「48G 是必需值、降到 32G 会 OOM」是从这一次
被自身 `--mem` 上限钉死的观测做的推断，非实测，已作废）：本 epoch 双卡逻辑读取量 =
253 step × accum 2 × batch 88 × 2 rank × 576 KB ≈ **48.93 GiB**，与申请的 48G、MaxRSS
47.71 GiB 三方两两之差 <3%，是「NFS 页缓存吃满给定上限」的典型信号；集群
`JobAcctGatherType=jobacct_gather/cgroup`（已查 `scontrol show config`），cgroup 记账含
file 页但干净页缓存触顶时被内核回收、不触发 OOM kill。真实不可回收工作集的独立证据：
本机单 rank 进程树 PSS 实测 3.13 GiB（双 rank 上界 ≈6.3 GiB）+ checkpoint save / NCCL /
shm 未测路径余量 ⇒ 估 8–9 GiB。**该推断已被当天 v2 实测证实（见下一小节）。**
另：计算节点能连 HF Hub（启动时读 Wan VAE config 走 `huggingface.co`，实测 200 OK）。

### 2026-08-14 冒烟 v2 降配实测（job 57642560 @ gl1509，4 CPU / 16G，同配方对照 v1）

| 项 | v2（4 CPU / 16G） | v1（8 CPU / 48G） |
|---|---|---|
| 排队时长 | **<1 分钟**（19:34:58 提交 → 19:35:43 起跑） | 长期 `(Priority)` PENDING |
| epoch 耗时 / 吞吐 | **479.9 s / 185.57 samples/s** | 480.8 s / 185.21 samples/s（**零退化**） |
| `max_memory_allocated` | 35.74 GiB（**逐位相同**） | 35.74 GiB |
| `nvidia-smi` 峰值 | 39,345 MiB = 85.4% | 40,049 MiB = 86.9% |
| MaxRSS | 16,540,592 K ≈ **15.77 GiB（贴 16G 上限）** | 47.71 GiB（贴 48G 上限） |
| **cgroup 峰值拆分** | **current=16.00 GiB｜anon=3.85 GiB｜file=12.07 GiB｜shmem=1.38 GiB** | 未采样 |
| TotalCPU | 18:25 / 8:44 墙钟 ≈ 2.11 核当量 | 18:07 / 8:50 ≈ 2.05 核当量 |
| checkpoint | 954,852,403 B（**逐字节相同**），体积硬断言通过 | 954,852,403 B |

判读：①**页缓存假说获双重铁证**——两代 MaxRSS 都精确贴住各自 `--mem` 申请上限
（47.71/48、15.77/16），而 cgroup 拆分显示真实不可回收 anon 峰值仅 **3.85 GiB**
（anon+shmem ≈ 5.2 GiB，与本机 PSS 外推 6.3 GiB 吻合），file 页缓存永远填满剩余配额且
可回收、零 OOM；②**性能零代价**——epoch 耗时/吞吐/显存/checkpoint 与 v1 全部持平或逐位
相同，「cache 减少拖慢顺序流式读」「4 核饿着 8 个 worker」两条担忧均被证伪；③冒烟一律
改用 v2 口径；**长训入口已于同日由用户拍板「长训按照 4 CPU / 16G」**（长训多 epoch 循环
扫同一数据集、cache 语义与单 epoch 冒烟不完全同构，但 56G 数据集任何档位都装不进 cache、
LRU 循环扫描命中率同样趋近 0；长训脚本已内置同款 cgroup 采样，首个 16G 档 run 结束以
anon/file 峰值复核）；④anon 实测 3.85 GiB 支持后续再做 12G 对照，暂不急。

两者都自带：run 目录 fail-loud 守卫（`train.py` 的 `resolve_run_dir` 对已存在目录是
`makedirs(exist_ok=True)` 静默复用，**本身不 fail-loud**，守卫必须写在 slurm 脚本里）、
2 卡口径断言、`.venv/bin/python` 直调（不走 `uv run`——集群侧 uv 装在 greatlakes 自己的
`/home`，而 `.venv/bin/python` 是指向 NFS `uv-python` 的 symlink，双端可用且不联网 sync）。

⚠ 新增 slurm 脚本时注意：`tests/test_train_script_overrides.py` 会 glob
`scripts/train-script-hongzefu/*.sh`，要求**每个 `.sh` 至少含 1 条 Hydra 覆盖、键在 §7.3
白名单内、字面量值不得与 `configs/default.yaml` 相同**。因此 slurm 脚本不能只是
`srun bash <本地入口>.sh`，必须自己直调 `scripts/train.py` 并带覆盖项；脚本内局部变量
一律全大写命名（小写 `key=value` 会被正则误当成 Hydra 覆盖）。

## 分布式 Wan latent 抽取（2026-08-16 新增，slurm-wan-extract）

`scripts/dataset-build/` 下的集群抽取入口（**不在** train-script-hongzefu，故不受
`tests/test_train_script_overrides.py` 的 Hydra 白名单 glob 约束）：

| 脚本 | 资源 | 用途 |
|---|---|---|
| `gl_probe_wan_consistency.sbatch` | 1 GPU / 2 CPU / 12G / 00:30:00 | 一致性探针（只读）：现场编码与 **v7 Ada 权威 .bin** 逐位比对 + 速率/显存/内存/指纹实测（⚠ reference 恒指 v7，指到 v8 会变 A40 自比恒 BITEXACT） |
| `gl_extract_wan_v8.sbatch` | **job array**，每 task 1 GPU / 2 CPU / 12G（full 0-7 / 08:30:00；full400 0-7 / **30:00:00**） | 分片并行抽取（lpt 装箱 + skip_motion 集群口径），断点续传，claim 防撞车；OUT/RAW_DIR/INPUT_MANIFEST 必传（由 prepare_v8_2_extract.sh 提交，不手提） |
| `gl_extract_wan_v8_finalize.sbatch` | 1 GPU / 4 CPU / 12G / 04:00:00 | `--dependency=afterok:<arrayJobId>` 串接：输入 sha256 核验 + 合并 + provenance 同源断言 + 四道守卫（spot_check 经 `--export` 传入：full 64 / full400 **256**） |

**放行记录**：8×1GPU 并发与 8.5h walltime 超出上文调试限额（≤2 GPU、≤30 分钟），
2026-08-16 经用户批准 slurm-wan-extract 计划显式放行（wan-plan §11 的全量抽取审批点
同次放行）。**2026-08-18 v8-400ep 全量抽取二次放行**（dataset-v8-400ep-plan §4 [2]
审批点，用户选「批准，立即提交」）：8×1GPU / 30h walltime / ≈173 GPU·h；实测
array 58239662 分片 Elapsed 22:21–22:57 全 `COMPLETED 0:0`（GPU 峰 4.7 GiB、cgroup
anon 峰 1.7 GiB，2 CPU / 12G 档位复核通过）、finalize 58239663 耗时 31:42。**job array 实测可用**：chaijy2/spgpu 接受 `--array`（job 57854615，2 task
独立调度到 gl1526/gl1514），MaxArraySize=5000——array task 逐个独立调度，等价于
「8 个分开请求的 1-GPU 小 job」，一次提交零手抄。

**⚠ 跨架构一致性结论（2026-08-16 探针 job 57854000 + smoke 演练 57855074/57855075 实测）**：
A40（sm_86）与本机 RTX 6000 Ada（sm_89）**不逐位一致**——max|Δ|≈1.2e-5（latent 值域
O(1–2)，相对 ~1e-6）、97% 元素有差、主体 16–256 ULP；分层指纹把分歧隔离到 VAE encoder
最后一层 conv_out（之前所有层逐位相同）；VAE 权重指纹与软件栈双端逐位同源已实证；
determinism 三档（default / cudnn.deterministic / cudnn.enabled=False）全部无效——
跨架构逐位对齐无 flag 可解。**故集群抽取按「换合同」口径交付**（2026-08-16 用户拍板）：
①集群产出自成一份数据集，metadata 逐 entry 带硬件/软件 provenance，finalize 断言全体
同源（机制上杜绝与本地字节混用）；②验收 = 量化等价（`wan_latents_manifest.py
compare-output --mode equivalence`，max|Δ|≤5e-5）+ 下游等价检查；③集群内部四道守卫
仍零容差（smoke 实测 spot_check 64 条 + oracle 14 条全部 max|diff|=0——A40 同架构
跨节点复算逐位成立）；④rgb_mag 一律本机 `--motion_only` 算（GPU 归约分块依赖 SM 数
84 vs 142，跨架构位不稳；全量 101,235 行实测与基线逐位相等）。

**资源实测（A40）**：速率 0.635 chunk/s/卡（LPT 最重分片 12,670 chunk ≈ 5.5h）；显存峰
5.72 GiB（T=551 压测 max_memory_allocated 1.77 GiB）；cgroup anon 峰 0.84 GiB
（12G 档宽裕，可再降）；CPU 2 核足够（h5/gzip 解压）。

**全量验证实测（2026-08-16，array 57856154 + finalize 57856155，600 entry / 101,235
chunk / 产物 60GB）**：8 分片耗时 5:38:47–5:48:38（极差 3%，LPT 均衡如预期；walltime
8:30 裕度 1.47×）；7 个 task 即时调度、第 8 个因 GPU 配额排队迟起 5.6h 后照常完成
（「分开请求 + 断点续传」的递补语义实证）；finalize 实测仅 **8:38**（含 17GB 输入
sha256 核验 + spot_check 64 全零差 + oracle 1200 chunk 全逐位，远快于 4h 预估）。
验收全绿：600 bin 量化等价 max|Δ|=2.420e-5（阈值 5e-5）、metadata 逐字段、
chunk_motion 逐位（rgb_mag 本机口径 101,235 行与基线逐位相等）、dataloader 七项
断言过、等丢弃率阈值重推 = 0.021994 / keep 99,195（与 default.yaml 现行值逐字相同）。
产物已按临时数据规约删除；`motionjepa-v7-gl/` 保留 data-raw 17GB、输入清单与两份
比对/motion 日志供复用。

**标准流程（2026-08-16 起 = v8 主流程四段，KIND=smoke|full，均本机跑，详见
docs/DATASET_zh.md）**：
1. `prepare_v8_1_stage.sh`：pin 校验（full）→ rsync raw 上 NFS → 输入清单 sha256 →
   `--motion_only` 本机算 rgb_mag 写进 v8 输出目录；
2. `prepare_v8_2_extract.sh`：pre-flight 五项 → 经 gl_submit 提交 array + afterok
   finalize（full 是审批点，须 `CONFIRM_FULL=yes`）；产物直接写 repo 内
   `dataset-4env-v8{,-smoke}/dataset-token`（v8 = A40 集群代际的最终家）；
3. 等 job 全绿后 `prepare_v8_3_verify.sh`：finalize 判定行 + dataloader + 可选基线
   量化等价比对，全绿打 VERIFY_PASS；
4. `prepare_v8_4_downstream.sh`：过滤预览（--raw_root 换根读 /data）+ arm mask +
   rsync /data 提示。
分片中断续跑：删该分片 `_claim_shard*.json` 后 `sbatch --array=<分片号>
--export=ALL,REQUIRE_EMPTY=0,<三路径>` 重提（断点续传，中断损失上界 ≈ 14 分钟）；
finalize 重跑须先把 `wan_chunk_latents/_shards_done/` 里的分片文件移回上一级。
⚠ `motionjepa-v7-gl/smoke-raw/` 是**探针专用**输入子集（7 entry 含最长
ButtonUnmaskSwap_ep3），与 v8 冒烟的 `data-raw-smoke/`（6 entry）集合不同，勿改写；
⚠ 抽取期间 8 task 读写同一 turbo 卷（~132 MB/s 天花板），勿并行跑集群训练；
⚠ v7 库（本机 Ada 产物）仍为现役训练数据与探针基线，训练 data_root 何时切 v8 是
独立决策，切换前须先把 v8 rsync 到 /data 本机副本。

## 调试 slurm 脚本

如需临时调试脚本，使用 `run_slurm_debug5min.sh` 这类独立名称，并遵守以下要点：

- 独立 `run_name`（绝不复用生产 run_name —— `overwrite=True` 会清空 `runs/<run_name>/`）；
- `wandb.enabled=false`、`training.max_epochs=1`；
- ⚠ 不要再写 `training.compile=false`：该键**不在 §7.3 白名单**，且 `configs/default.yaml`
  已默认 `compile: false`，写了会被 `test_train_script_overrides.py` 双重判违规；
- 不 `source` 任何 home 下的 rc 文件，python 用 NFS 绝对路径。

常用样板：

```
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
# 注：不要加 --qos=interactive（chaijy2/spgpu 报 Invalid qos specification），用默认 qos
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA/output/logs/%x-%j.log
```

## PENDING 状态读法

- `(Priority)` → 正常排队等调度，等就行。⚠ **配额没满不代表马上能起跑**：slurm 判能否分配
  看的是**单节点**的空闲 GPU / 空闲 CPU / 可分配内存（= `MEMORY − ALLOCMEM`，不是 `FREE_MEM`）
  三者同时够。2026-08-14 实测：chaijy2 尚余 18 GPU / 768G / 72 CPU，但 spgpu 全 30 个节点里
  同时有 ≥2 张空闲 A40 的只有 3 个（gl1515：3 CPU/137G、gl1527：27 CPU/**13.6G**、
  gl1528：5 CPU/47.3G），一个 2GPU+8CPU+48G 的 job 在每个节点都差一维，只能干等。
  查碎片：`sinfo -p spgpu -N -O 'NodeHost:12,CPUsState:16,AllocMem:10,Memory:10,GresUsed:16'`；
- `(Priority)` 但急着跑 → 在**不动训练配方**的前提下降 `--cpus-per-task` / `--mem`：
  **v7 双卡 bf16 的已验证档位是 4 CPU / 16G**（2026-08-14 v2 冒烟实测：排队 <1 分钟、
  epoch 零退化、cgroup anon 峰值仅 3.85 GiB，详见冒烟 v2 实测表）。旧推断「MaxRSS
  47.7G ⇒ 48G 是贴边值、降 mem 会 OOM」已被实测证伪——MaxRSS 只是页缓存填满申请上限的
  读数。长训（多 epoch）档位未单独实测，降配前先向用户确认；
- `(AssocGrpMemLimit)` → chaijy2 总 mem 配额满了，先降 `--mem`（`--qos=interactive` 实测无效，别用）；
- `(AssocGrpGRES)` → chaijy2 总 GPU 配额满了，只能等组内其他用户的 job 退出，不可换 account；
- `(Resources)` → spgpu 全集群节点都被占，等就行。

## 放行记录（robomme framesamp v2 计划，2026-08-27）

**S8b GL e2e 收官四 job 放行**（v2-framesamp-restructure-plan.md D 节；均超调试限额
≤2 GPU / ≤30 min，经用户 AskUserQuestion 显式批准「四个全批」）：

| job | 资源 | walltime | 备注 |
|---|---|---|---|
| `v1-framesamp-e2e-w4c16`（T1） | 4×A40 / 16C / 96G | 02:00:00 | 600 步 / seed 320 |
| `v1-framesamp-e2e-w8c16`（T2） | 4×A40 / 16C / 96G | 02:00:00 | 600 步 / seed 321 |
| `v1-framesamp-e2e-w2c16`（T3） | 4×A40 / 16C / 96G | 02:00:00 | 600 步 / seed 322 |
| `v1-framesamp-e2e-w4c16` COLDHOT（`…-coldlike`/`…-hot`） | 4×A40 / 16C / 96G | 04:00:00 | 各 300 步 / seed 323 |

提交方式同次拍板：**现在全部提交、`--dependency=afterany` 链 T1→T2→T3→COLDHOT 严格
串行执行**（避免同节点共驻/互相预热污染 E2E_ACCEPT 性能判据）；条件档 T4/T5 未批、
视 T1–T3 结果另行审批。同日 S8a 四个 1×A40/30min job（58995916–58995919）在调试
包络内、无需特批，随档位确认提交。

**追记二（同日三次拍板，用户「不改动训练逻辑 把现有的实验跑完…因为现在集群紧张
策略是先全部提交 job 排队 如果有同一节点干扰再重新提交」）**：在上表四 job 之外
新增两个同包络 job，一并放行——

| job | 资源 | walltime | jobid | 备注 |
|---|---|---|---|---|
| `v1-framesamp-e2e-w12c16` | 4×A40 / 16C / 96G | 02:00:00 | 59001192 | 600 步 / seed 324 |
| `v1-framesamp-e2e-w16c16` | 4×A40 / 16C / 96G | 02:00:00 | 59001193 | 600 步 / seed 325 |

COLDHOT 同批提交（59001191，`--exclude=gl1514,gl1501,gl1508,gl1512,gl1519`——排除全部
碰过 packed 库的节点以保 cold-like 口径）。**逐 job 资源包络与上表完全一致、未扩**；
新增理由（补齐 legacy 四档对照曲线、w12 为不改代码下唯一可能达标候选、w16 验证 CPU
超订拐点）见 `docs/training-doc/v1-framesamp-e2e/launch.md` 三次拍板一节。

**追记（同日二次拍板，用户授权「可以自由提交这些 job」）**：串行链改并行——T1
（58996749）运行期间撤销 T2/T3/CH 依赖链，T2/T3 无依赖重提（58996987/58997004，
`--exclude=gl1514`）；CH 待 T2/T3 起跑后带冷节点排除清单（gl1501/gl1508/gl1512/
gl1514 + T2/T3 实际节点）单独提交——四个同时跑超 chaijy2 配额（GPU 22>20、
CPU 82>80），CH 天然要等 T1 退出，故不损失时间。**逐 job 资源包络与上表完全一致、
未扩**，仅调度方式变化。评估细节（配额账/节点现状/共驻风险/暖缓存口径/NFS 交叉
负载）见 `docs/training-doc/v1-framesamp-e2e/launch.md` 二次拍板一节。
