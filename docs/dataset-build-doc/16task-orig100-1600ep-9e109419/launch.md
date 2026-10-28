# 官方16×100：双帧表与阶段二motion建库启动档案

## 一、目的与当前状态

按[批准计划](../../../1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md)构建全新库`16task-orig100-1600ep-9e109419`，供先context、后modulation两条8卡60k训练消费。两种帧表全量构建；motion只落一张`[1,1024]`表，concat与split共用。当前为建库Beta启动档案，数据验收及训练尚未完成。

库名后缀取发布源清单`pin_sha256=9e109419d08828d46261d45ca88786207015752d4b3c54b67a1ce8554fcf96f6`的前八位；不拿这一字符串代替实际源H5核验。

## 二、版本与代码状态

| 对象 | 固定版本或身份 |
|---|---|
| 批准计划 | `d5fe05d647a899e016bc513aa5fba03485a9b398` |
| 全部代码整合完成 | policy `58d3b5c47731f68fd085b81f68595e641b5ec6fc`，工作区干净，已推送 |
| 建库代码 | 本档案所在`commitV11.66Beta`；实际完整SHA在起跑时写入`v1-store/logs/o16-20261028/build-start.json`及各日志`BUILD_HEAD=` |
| MJ导出CLI | `50a853492df32e6aa91e2d523acc33bc34257f4a`，`/scratch/hongze/worktrees/vla-s2-memexport-20261026` |
| MJ共享库 | stage2独立环境内固定安装`b5d23a5fab55be16e580c5b77a8f04d68b3170c2`；不设`PYTHONPATH`覆盖它 |
| encoder | `stage2-full144-1025-a/checkpoint_epoch_105.pt`，EMA，SHA256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035` |
| encoder源pin | `31d6ab0f845336e76dc25db64170a0bc4ac59ecf366c161025b9800285c74467` |
| 官方H5 | `Yinpei/robomme_data_h5@a5e4e25ffe8af34f64944f9533d06455ce5f8337` |
| 发布Wan缓存 | `HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1@384d5db410be713a49357bf3f55ca4f4d2f559a0` |

Beta至`VERIFY_MOTION=PASS`冻结主检出HEAD并写`AGENTMETARULES_HOLD`。期间状态写入`/scratch/hongze/robomme_policy_learning_MotionJEPA-temp/docs/handoff/current.json`，专属本地分支`worktree-agent-o16-state-20261028`；该状态分支随解冻后合并一起推送，不修改冻结主检出。

起跑前归档复核：首份Beta `a0c1948`之后发现复制的矩阵日志含原程序打印的行尾空白，尚未启动任何数据步骤。本次仅清理归档副本的行尾空白并重新固定实际起跑Beta；判定字段、数值、代码与运行参数不变，原日志仍保留在S7工作树。实际起跑SHA以`build-start.json`及日志`BUILD_HEAD=`为准。

## 三、启动、配置与完整命令

所有命令在主检出执行。长步骤使用独立`o16-`前缀tmux、`PYTHONUNBUFFERED=1`、`tee`和真实`EXIT_CODE=`。运行包装脚本只是以下命令的调度载体，位于日志根，不复制进档案；固定SHA为：

| 载体 | SHA256 |
|---|---|
| `prepare-manifest.sh` | `042c427723a85aeaf7a573628c17e270ca67e3a8125176ac35022135cfaecb2b` |
| `fetch-cache.sh` | `81de6a0f6028980e927aec331dacc7c021fd4abc9f78d7c680920a9ca1722cde` |
| `import-cache.sh` | `b167ba56e2890fcb3cc27ab682fc0b77c1bb8169d34e0965d0af31d9deac318c` |

```bash
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
cd "$P"
export MJ_REPO=/scratch/hongze/MotionJEPA
source scripts/dataset/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RAW=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5
MJIDX=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/index
MJSRC=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json
MJW=/scratch/hongze/worktrees/vla-s2-memexport-20261026
RGBROOT=/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1
NAME=16task-orig100-1600ep-9e109419
LIB="$V1_STORE/datasets/$NAME"
MANI="$LIB/meta/episode_manifest.json"
INMANI="$LIB/meta/input_manifest.json"
VALID="$V1_STORE/validation/orig16-stage2"
CKPT=/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt
CKPT_SHA=27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035
G8=0,1,2,3,4,5,6,7

# CPU阶段：实际读源文件，随后检查三源同源。
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$MANI"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/check_data_lineage.py lineage --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --vla-input-manifest "$INMANI" --mj-source-manifest "$MJSRC"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py plan --manifest "$MANI" --input-manifest "$INMANI" --published-index "$MJIDX" --out "$VALID/wan-cache-plan.json"
HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py fetch --plan "$VALID/wan-cache-plan.json" --revision 384d5db410be713a49357bf3f55ca4f4d2f559a0 --out "$LIB/cache-source/official-wan"

# import-cache.sh先用read_window/make_encoder对首个真实fill窗做一窗shape/finite/VAE身份短测。
CUDA_VISIBLE_DEVICES=$G8 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py import --plan "$VALID/wan-cache-plan.json" --cache-root "$LIB/cache-source/official-wan" --raw-dir "$RAW" --out "$LIB/wan-latents" --receipt "$VALID/wan-cache-import.json" --gpus "$G8"
# WAN_CACHE_IMPORT与WAN_INPUT_COVERAGE通过后，核确切目录再删除本轮临时cache-source/official-wan；不删其他缓存或源H5。

# SigLIP先单episode短测，随后按原计划每GPU两个worker构建完整source。
CUDA_VISIBLE_DEVICES=$G8 uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 --raw-dir "$RAW" --require-free-mib 70000
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-4x4-v1 --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp-4x4" --procs 48
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-8x8-v1 --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp-8x8" --procs 48
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --layout framesamp-4x4-v1 --store "$LIB/framesamp-4x4" --resume --procs 48
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --layout framesamp-8x8-v1 --store "$LIB/framesamp-8x8" --resume --procs 48
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"

CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_stage2_validation_inputs.py --manifest "$MANI" --latents "$LIB/wan-latents" --source "$LIB/source" --h5-root "$RAW" --input-manifest "$INMANI" --mj-index "$MJIDX" --cache-import-receipt "$VALID/wan-cache-import.json" --out "$VALID/G0.json"
# 2496,23253是索引推算值，执行前与实际manifest/cache plan复核；先另目录max-segments=1短测。
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$V1_STORE/venvs/wan-stage2/bin/python" "$MJW/scripts/stage2_memory_export.py" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --latents "$LIB/wan-latents" --manifest "$MANI" --out "$LIB/motion-tokens-stage2" --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts 2496,23253 --input-manifest "$INMANI"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py check --samples "$VALID/G0.json" --receipt "$VALID/wan-cache-import.json" --raw-dir "$RAW" --out "$VALID/G2.json"
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/validate_stage2.py --gate G3 --samples "$VALID/G0.json" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --rgb-root "$RGBROOT" --mj-root "$MJW" --cache-import-receipt "$VALID/wan-cache-import.json" --out "$VALID/G3.json"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin published-cache-with-local-fill --cache-import-receipt "$VALID/wan-cache-import.json" --manifest "$MANI" --tokens "$LIB/motion-tokens-stage2" --latents "$LIB/wan-latents" --out "$LIB/motion-stage2-1024" --encoder-run-dir "$V1_STORE/external/motionjepa/stage2-full144-1025-a" --raw-dir "$RAW" --expected-ckpt-sha256 "$CKPT_SHA" --stage2-validation "$VALID" --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store "$LIB/motion-stage2-1024" --resume --check-padding
```

全部环境覆盖由上表/命令及`paths.sh@Beta`还原：HF、XDG、JAX等缓存均在主检出`v1-store`；不覆盖HOME、不访问集群。两份帧表可以各48个CPU进程并行，GPU步骤串行。完整模型短测与正式训练另归两份训练档案。

## 四、数据口径与验收

预核索引为16任务×100集、768,897帧、476,857执行样本；grid16共45,228窗（copy42,404、demo-fill900、exec-fill1,924），grid32为23,253窗/2,496个非空段，最大每集44窗。以上仍须实际清单确认。缓存恢复为2,485个完整bin、388,310,630,400字节，不能按命中窗口的25GB估计下载量。

正式关口依次为磁盘≥1.5TiB、实际`DATA_LINEAGE`、缓存整文件与输入覆盖、SigLIP finalize、两种帧表全量verify、G0/G2/G3、motion pack/verify、全表split逐位、两条G4、G6和配置常量回填。输入/权重/shape/finite/完整性仍严格检查；跨路径精度差记`NOT_REQUIRED reason=user_decision_same_input`。

`VERIFY_MOTION=PASS`后解除构建冻结，回填实际库路径、norm/manifest/input摘要、行数与窗口上界。G4两条分别用正式yaml及`--framesamp-store "$LIB/framesamp-4x4" --h5-root "$RAW" --input-manifest "$INMANI"`，父GPU0、sidecar GPU1。G6消费实际manifest，预算96窗。两条20步必须真实执行并严格保存恢复，不能以DRY_RUN替代。

## 五、硬件、资源与现场句柄

AWS本地NVMe RAID，8×A100-SXM4-80GB。开工预核空闲2.757TiB、8卡空闲；每个GPU步骤起跑时重新确认。没有GreatLakes/Slurm任务。当前尚无本轮运行句柄，计划依次使用`o16-manifest`、`o16-cache-fetch`、`o16-cache-import`、`o16-siglip-smoke`、`o16-siglip`、`o16-finalize`、`o16-pack4`、`o16-pack8`、`o16-norm`、`o16-g0`、`o16-s2export`、`o16-g2`、`o16-g3`、`o16-s2pack`、`o16-g4-ctx`、`o16-g4-mod`；实际启动逐项写入冻结期状态副本，未启动的名字不算运行句柄。

日志根`v1-store/logs/o16-20261028/`，验证根`v1-store/validation/orig16-stage2/`。此轮只清理成功导入后的确切临时发布缓存；不扩展旧数据清理范围。

## 六、代码验收与用户决定

`PLAN_CONSISTENCY_REVIEW=PASS findings=2 blocking=0`；所有代码块完成固定提交合前、合后审查，全部子提交通过`--no-ff`保留。合后关键测试：S6 63项；S1 92项；S2 53项及实际独立环境17项；S3 29项；S4 67项；S7联合261项；S5 66项。集合有重叠，不将这些数字相加当唯一测试数。`MECH_MATRIX=PASS combos=8 contract_ok=8 forward_ok=8`，范围为实际模块单层/FFN64，保留正式长度和维度；完整VLA由后续8卡短测验证。

用户原话：「开工 有问题问用户 尽可能不要阻塞 所有subagent全部astra high」「登记三项任务（推荐）」。沿用计划K1–K6：其他六组合仅CPU验证；全部正式前置通过即自动训练；训练库与checkpoint公开上传；不评估；两条固定run名；四个motion token共用窗口起点时间码。

实施中的依赖补齐、文字统一及各审查修复见[开工记录](../../maintenance/1027-orig16-vla-start.md)和各合并提交。实际数据/资源/失败/耗时及上传结果完成后写`result.md`与`records/`；当前不把预核、CPU夹具或存在性检查写成真实建库通过。

## 七、执行中的同规模调度调整

美东10月27日22:44，第一阶段实际完成：1600集、768897帧、476857执行样本；16个H5全量SHA与encoder源逐文件相同，`DATA_LINEAGE=PASS files=16 mismatches=0`，VLA输入清单SHA为`5d1bfed54377c22e68bc1883ebd35fdf77372129ac1565dd6a78601c1d9965f9`。episode清单内部摘要`337c4fcde163974039b5c0f2ad1f46ebaacd272719f72818503e23c1360447e2`，整文件SHA为`93d7c5a0c1f34fd955de5f8c37370369e54457019f75e56aea8908cba894d999`。实际缓存规划计数全部符合预核，原阶段`EXIT_CODE=0`。

串行缓存恢复于美东22:45:28启动；22:48:42已下载28个bin、约9.12GB，吞吐约47MB/s。为缩短这一已批准下载的等待，独立审查给出`ADJUST_REVIEW=PASS scale=same findings=0`，改为SDK自带8路下载：版本、2491个明确允许路径（SHA256SUMS＋5个metadata＋2485个bin）、有效bin字节388310630400均不变。保持`HF_HUB_ENABLE_HF_TRANSFER=0`、`force_download=False`；完整文件依SDK同revision metadata复用，partial依HTTP续传/Xet块缓存处理，不承诺所有partial字节零重复传输。

主会话核准自有Python PID94695、pane PID94641及无子进程后发送SIGTERM，等待两者完全退出，保留真实`EXIT_CODE=143`和`cache-fetch.log`／`cache-fetch-serial.log`，不把该次中止写成成功。随后才启动唯一的新写者，避免两个进程并发写同一目标。新运行载体`v1-store/logs/o16-20261028/fetch-cache-parallel.sh`的SHA256为`a233d64f895d0228702a567749c56e9cf858538f5b11deac1cf76bd5558f3443`；它调用已安装`huggingface_hub.snapshot_download(..., allow_patterns=<上述明确路径>, max_workers=8)`，返回后仍执行原`orig16_wan_cache.py fetch`逐文件全量SHA／字节数核验并生成原收据。下载完成本身不算验收通过。

新会话`o16-cache-fetch-parallel`于美东22:54:11启动，pane PID104919、下载PID104973，`CACHE_PARALLEL_STARTUP=PASS`；日志`cache-fetch-parallel.log`确认8 workers和2491路径/388310630400 bin字节。起跑后约8秒，完整bin从74增至77个、24152113152字节，另有8个partial；此时尚未完成下载或验收。

此次只调整CPU／网络文件并发，不改任何冻结仓库源码、数值路径、验收或GPU资源。下载阶段最长等待仍从22:45:28起计4小时，不因调整重置；主会话持续监督，验证通过前不启动import。冻结结束后将此记录随状态分支合回；原批准计划正文保持不变，最终收尾再追加实施记录。

### 训练入口待解冻合入的小修

只读检查旧生产日志发现`tqdm`写入`tee`的实际Step行以两个回车开头，原入口`^Step`不能匹配。S7在独立树追加`f50742e0d1f3ed5c7181a884f724420bfbc7c4ae`，仅将两入口改为`^\r*Step`并精确覆盖现有五个标量（包括`mem_enc_norm`），保留最终记录与明确OOM检查，不引入`tr|rg -q`管道。真实字节格式的两入口×五字段负例及正常日志共24项定向测试通过；独立`PRE_MERGE_REVIEW=PASS findings=0 blocking=0`。此提交尚未合入冻结主分支，不影响数据构建，必须在`VERIFY_MOTION`解冻后、实际20步训练前合入并复核。

## 八、用户更新：步骤照做，失败如实记录

用户先说：「训练前尽可能不要阻塞 先跑起来 明天上午用户再来决策」，随后明确纠正：「不要后置任何步骤 该做的还是做 但是失败了只记录 让用户决策」。以后一条为准：所有原步骤、两种帧表、G4、独立20步及严格保存恢复都执行，不跳过、不后置；失败保留原始`FAIL`、退出码和证据，不改成PASS，继续技术上可执行的后续事项，交用户决策。程序崩溃、产物缺失或格式不兼容导致某一步客观无法执行时如实记录受影响依赖，不伪造结果、不擅自换数据或实验配置。原计划K2的单项验收FAIL即整体停工口径按本次明确指令调整。

先前提出的后置与可选SHA并发改码均未实施，已撤回；主建库代码仍为`bd4bad0`。缓存388310630400字节及5个metadata已经原fetch全SHA验收通过，`WAN_CACHE_FETCH=PASS`、`EXIT_CODE=0`；GPU导入原脚本于美东23:56:30启动，真实一窗`WAN_FILL_SMOKE=PASS windows=1 key=BinFill_ep0_exec m=30 shape=9,16,32,32 nonfinite=0`，随后继续原全量检查和导入。

另一个真实启动程序缺陷已在独立S2树修复：`init_history_config`漏向快照传递已有`source_grid_sha256`和`latents_metadata_sha256`，实际入口联测复现KeyError；`3fa6665e39e77eb671daed15358311a6e76b2519`只增加一行来源字段转交及直接测试，修后14项通过，独立`PRE_MERGE_REVIEW=PASS`。该必要修复与CR修复一样待解冻后、训练前整合，不删除检查或伪造来源。

为落实“失败只记录”，S7在CR修复后追加`c5a2e31fc62f85d54d9fe94de0f5077a7b503435`：日志数值扫描与final.json诊断仍全部执行，原FAIL保留并明确`action=record_only reason=user_decision`；诊断不再覆盖实际训练退出码，也不单独阻止完成后的串行接续。真实程序非零、日志写入失败、缺真实结束日志或59999仍按客观未完成处理；启动合同和诊断判据未改变。新策略37项定向测试及独立审查通过，旧阻断策略的测试原记录另存。解冻后合最新c5a2e31即可同时包含CR修复，不能只合旧f50742e。

## 九、Wan实际结果与SigLIP起跑

原导入器先完成约901GB的缓存及H5完整复核，再启动8个GPU worker；最后2496段全部汇总并检查，实际产物26676559872字节。`WAN_CACHE_IMPORT=PASS copied=42404 filled=2824 total=45228 input_mismatches=0 nonfinite=0`、`WAN_INPUT_COVERAGE=PASS windows=45228 missing=0 duplicate=0 mismatches=0`、`EXIT_CODE=0`。导入收据SHA为`1a976182ca3868b6dec25a4411ccc85eed254978f2e4bb9cee11532e3093e22e`。主会话复核收据、元数据摘要及计数后，仅删除确切`LIB/cache-source/official-wan`实体目录；`CACHE_TEMP_REMOVED=PASS`，原H5和完整wan-latents保留。

SigLIP实际起跑于美东10月28日00:47:46，`o16-siglip`、pane PID222562，日志`siglip-source.log`。权重cheap校验PASS，global_episode_idx=0单集550帧短测完成，`SHARD_DONE ... elapsed=94.1s`、`SIGLIP_EPISODE_SMOKE=PASS episodes=1`；这94.1秒包含首次初始化，不作为稳态速度。随后按原命令启动8卡16worker全量source构建；当前阶段仍在运行，不记完成。

### 独立导出提前并行，所有检查仍保留

固定代码审查确认S6只读已完成的latent、episode/input manifest、固定checkpoint、旁置config和训练源build_spec，不依赖G0、SigLIP source、帧表或norm。主会话在GPU0约20GiB为本轮SigLIP占用、约60GiB空闲时提出提前导出，`ADJUST_REVIEW=PASS scale=same findings=0`；不改变代码、输入、B1/FP32/TF32口径、输出格式、总工作量或物理GPU范围，不取消或后置G0等任何验证。

先于美东01:07:07在GPU0同驻执行1段smoke，01:07:14完成：`STAGE2_EXPORT=PASS scope=smoke segments=1 rows=17`、`EXIT_CODE=0`。源码身份逐键核为锁定b5d23a5，未改用MJ editable环境。随后01:09:42起`o16-s2export`（pane248995、worker249030），保持`--expected-counts 2496,23253 --check-dropout 64`及已定解释器/CLI；写`LIB/motion-tokens-stage2`，与SigLIP的`LIB/source`分离。调度载体`motion-stage.sh` SHA为`f89143b1b76cb4261cc8b6aa51fe5364922d94e03d57a83499e8676dc2263fc9`，分别设置`STAGE=export-smoke`和`STAGE=export`。全量导出此时进行中，不因smoke通过而标全量成功。

G0和后续验证仍在原有输入就绪后全部执行；提前完成的全量编码不重复烧GPU。所有已有GPU0进程均核为本轮SigLIP后代；若同驻短测失败，原定独立导出仍会执行，失败记录保留，不影响SigLIP现场。

截至美东01:17，全量导出已实际完成并退出：`STAGE2_EXPORT=PASS segments=2496 rows=23253`、`EXIT_CODE=0`，2496个f32文件共95244288字节；export_record明确`scope=full`，SHA为`9f56b96c8c881cf11db7d50bda7a16da0ebb24d71f7fcea3e3a846c9f12031e0`。64窗dropout诊断为`eval_repeat_max_abs=0.0`、`train_vs_eval_max_abs=1.5240858793258667`，后者用于证实训练/推理模式差异，不是失败指标。共享来源三键与固定SOURCE_PIN逐项相同。导出会话已退出，SigLIP继续；此结果不替代后续G0/G2/G3/pack/verify。

### SigLIP完成与两组独立检查重叠

SigLIP已于美东约02:11实际完成：1600个唯一EP、768897帧、重复完成事件0，16条SHARD_DONE均正常；`STAGE_DONE stage=siglip workers=16 items=1600 elapsed=4928s`、`EXIT_CODE=0`，所有GPU进程已释放。后续仍须完整finalize及两种帧表验证，不能把生产结束等同全部数据验收。

独立固定代码审查给出`ADJUST_REVIEW=PASS scale=same findings=0`：source封口后，finalize仅读源文件并最终写source/meta统计；G0原H5路径仅读source/data，不依赖stats、帧表或norm，写独立G0.json，因此两者可重叠。原`input_level=sha256`和1024视觉spot均保留；finalize的SHA阶段已有最多4线程，G0纯CPU。两会话均于美东02:15:29启动：`o16-finalize` pane301478／worker301578，`o16-g0` pane301527／worker301586；全部参数与原脚本不变。

同一审查确认G0发布后，G2（CPU）与G3（GPU0）互不消费对方结果，可并行各写独立JSON。4×4/8×8 pack及norm仍等finalize结束，motion pack仍等G2/G3结果；没有验证被取消或后置。失败按用户要求真实记录，不改判据。

独立只读服务准备：HF官方认证HTTP200、目标namespace及写角色匹配，三个目标bucket查询均404，仅记未查到、不记空bucket；W&B单次viewer认证HTTP200且GraphQL有效。没有创建bucket/run/项目，没有上传或改变可见性，没有输出或落盘凭据。

## 十、源验收、两帧表与norm实际结果

finalize原五阶段已执行完：1024条同架构视觉复算全部逐位一致，`全部检查通过 PASS`、`FINALIZE_EXIT_CODE=0`、`EXIT_CODE=0`。随后02:25:12启动两张帧表与norm。两表pack分别约90秒、151秒完成32个分片，均写`status=packed`。

运行载体沿计划示例在verify上附了`--layout`，实际verify接口不支持，故两次均出现`unrecognized arguments: --layout ...`并真实`EXIT_CODE=2`；这时verify未执行，不能写成数据校验失败或通过。原`frames-4x4.log`／`frames-8x8.log`完整保留。独立`ADJUST_REVIEW=PASS scale=same findings=0`确认verify从store_meta选择正确layout；不重跑pack，仅使用`verify --store <对应库> --resume --procs 48`继续原完整检查，不改数据或阈值。

新`o16-verify4`、`o16-verify8`于02:29:09启动，分别24秒、38秒完成：两条均`VERIFY_PACK=PASS scanned=768897 mismatches=0`、`EXIT_CODE=0`，meta真实回填verified。新日志为`verify-frames-4x4.log`、`verify-frames-8x8.log`。meta SHA分别为`1318229bd58f4e32a8a44705c862d476d7c92890004b97e69314904433e8d3f9`及`8bad885bcc8344bff533c7fc6ccf7abfbc3d207c67ae7692b7319878587409b8`。新载体`verify-frames.sh` SHA为`ee2702869783ad8080b833f0ffb4b08221fe341fd65425d11d716378f32fa2a5`；旧失败不被新结果覆盖。

norm正常完成，`norm_stats.json`实际SHA为`c5d45b4bdb481ca46d18aeb32446bbedec34de435eef31d3212a179ee7868bcd`，`EXIT_CODE=0`。截至02:30，G0仍在原输入核对中，后续G2/G3和motion打包等尚未宣称通过。用户要求所有步骤做完、失败只记录；上述接口失败与修复经过留待用户复核，没有等待额外确认而阻断其他任务。

### G0和G3实际完成

美东约02:36，G0完成：`RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=16 sha_mismatches=0`、`EXIT_CODE=0`；copy32、demo-fill16、exec-fill16。G0.json SHA为`6a74227d17f347bfd7e7a1f6dea9a5969ac1d959fcac969edf2ac57ea97fdc32`。

02:38:10并行启动原G2与G3。G3实际完成：`EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1`、`EXPORT_PARITY=NOT_REQUIRED reason=user_decision_same_input`、`EXIT_CODE=0`。JSON的32个参考窗口形状失配和非有限值均0，geometry_all_true=1；G3.json SHA为`3e72fa45fc2702291754d42d69b21293de43ff38ea6b78047c5a29446e81f6e2`。未恢复用户已取消的跨路径精度准入。G2此时仍在CPU核验输入，不能因G3完成而记G2通过。

### G2完成，原motion打包与验证起跑

美东02:59，G2原检查实际完成：`G2_INPUT_IDENTITY=PASS windows=64 mismatches=0`、`G2_NUMERICS=NOT_REQUIRED reason=user_decision_same_input`、`EXIT_CODE=0`。JSON中SHA、帧、形状失配和非有限值均0，copy32/demo-fill16/exec-fill16；不重复已完成的导出。

03:01:22启动`o16-s2pack`，pane345433／worker345487，原`motion-stage.sh STAGE=pack-verify`、冻结HEAD bd4bad0957e5162073f9bbfec0e4c985c3f830bc和全部原参数不变。日志`motion-pack-verify.log`；此时仅确认起跑，待真实`VERIFY_MOTION`和退出码。后续全表split、在线两组合、预算检查、两条20步与保存恢复仍全部执行；用户当前要求失败真实留档、继续技术上可执行步骤，不把FAIL改成PASS。

### 后续执行准备中的最小接口修正

G4在读取真实YAML后要求`motion.store_path`与传入表路径一致，原计划文字将常量回填排在G4之后会被占位路径拒绝。固定bd4bad0独立审查`ADJUST_REVIEW=PASS scale=same findings=0`：动作表pack/verify结束后先做G6与全表split，再回填原定真实常量，随后跑最终YAML的G4两组合、八组合矩阵及两条20步保存恢复。只前移既定常量，不取消、放宽或后置验证到正式训练之后。

准备G12时又发现新上传驱动产出的`orig16-stage2-delivery-manifest-v1`与既有恢复器接受的`stage2-delivery-manifest-v1`不兼容。独立同规模调整审查通过；S5提交`9679613e1ce08b272b7c3fecfb8b0804069323cc`只统一新驱动schema常量并补真实生产/消费联测，不扩宽恢复器。两条run在修前均真实复现schema拒绝；改后导出与恢复定向125项通过（13.29秒），独立`PRE_MERGE_REVIEW=PASS findings=0 blocking=0`。与c5a2e31和3fa6665两项已审修复一起等待数据冻结结束后合入；此时主HEAD未变。

G12须用导出根的`stage/59999`及同一stage的`manifests/step-59999.json`；该目录包含实际交付依赖实体。原run缺这些实体，不能代替stage。已完成阶段16份日志归档共885797字节（含归档校验JSON），所有退出记录及SHARD_DONE原样保留，详见`records/archive-validation.json`。

### motion打包完成，独立全表验证进行中

美东03:25左右，原pack正常完成：`PACK_PROVENANCE_PRECHECK=PASS mode=vae_compat segments=2496`，表23253行（exec14138/demo9115）、2496段、95244288字节，耗时1444.3秒；`PACK_MOTION_DONE=1 raw_dir_match=1`。metadata此时仍为`status=packed`，不能记verified；索引SHA为`7aad314e4daee3b7513770deee62f38714e06c57d3f63bbd634d4e52db54eee6`。

同一`o16-s2pack`于03:25:27启动原verify，worker370731，pane345433未变。原`--resume --check-padding`及全文件来源核对保留，日志继续写`motion-pack-verify.log`；等待真实`VERIFY_MOTION`和总退出码，冻结尚未解除。

### 真实缓存G2与上传器合同对齐

运行准备只读审查发现：S5原`snapshot_sources`同时要求G2/G3含顶层encoder摘要，而真实缓存G2按既有合同只绑定输入、收据和VAE；G3才绑定encoder。原测试夹具给G2加了生产记录没有的字段。独立同规模调整审查通过后，S5追加`c01b6501350cd964a04542468ce2d8a6c57d39c5`，仅将条件改为只检查G3的encoder，真实G2产物及恢复器均不变。

去掉夹具伪字段后修前3项真实失败；修后相关完整133项通过（13.86秒），覆盖原samples、G3 encoder、compat收据绑定的缺失/错误负例。独立`PRE_MERGE_REVIEW=PASS findings=0 blocking=0`；此tip包含前一schema修复，解冻后一起合入。准确边界：上传器通过已verified表的实体SHA及run/store证据一致性复用原S2验证，并未新加完整G2语义复验。两份后续运行载体的接口、参考文件与CPU/GPU环境核对未发现其他必然失败的错配。

### 全表验收与冻结结束

美东03:47，原verify完成：`VERIFY_MOTION=PASS scanned=23253 mismatches=0 nonfinite=0 segment_set_mismatch=0 real_plus_pad_ne_33=0 exec_padded=0 pad_not_last_demo=0 pad_source_mismatch=0 demo_tail_padded=386 demo_tail_dropped=514 legal_zero_segments=4`、`EXIT_CODE=0`，耗时1320.3秒；metadata已回填verified，row_digests已落盘。所有本轮数据任务已退出，建库HEAD始终bd4bad0957e5162073f9bbfec0e4c985c3f830bc，达到原定解冻边界。

03:47:34独立执行原全表split和G6，两者退出0：`MOTION_SPLIT_BITWISE=PASS windows=23253 mismatches=0`；`TRAIN_ES_BOUND=PASS episodes=1600 tasks=16 max_windows=44 budget=96 headroom=52`。最坏为VideoPlaceOrder第4集，demo1118帧/35窗、exec293帧/9窗。旧合并H5的源文件映射不存在，脚本仅将该辅助名称记unknown，真实任务、原episode索引与窗口计算不受影响。

实际常量见`records/build-pins.json`。后续G4/矩阵/20步及保存恢复尚未执行，不因数据验收通过而代记这些结果。状态快照准备合回主分支，三项已审代码修复及常量随后合入。

### 已审修复合回与真实常量回填

状态分支以c7f2cae合回。S7以4320ef8合入f507/c5两提交，合后37项通过；S2以7ce2220合入3fa6665，合后14项通过；S5以2f5bb2b合入967/c01两提交，合后154项通过。每次均无冲突、独立合后审查PASS、随即推送，全部子提交历史保留。

随后单独回填既定常量：新库`16task-orig100-1600ep-9e109419`、norm/两种manifest SHA、canonical manifest SHA、motion_rows23253和train_max44；八份YAML、唯一新训练配置资产路径及新导出公共体使用同一库名。占位检测哨兵与负例保留，不改变模型、预算96、batch128、seed42、60k或两条顺序。实际常量全部来自`records/build-pins.json`及真实G6结果。

后续运行载体为`v1-store/logs/o16-20261028/post-build-checks.sh`（SHA `f726fc57c604016876951fdcb8f202f1849c13f12ae2be567a5191c3544db61e`），通过`CHECK_HEAD=<本次完整提交>`及`STAGE=g4-ctx`、`g4-mod`、`matrix`分别执行原检查，日志`check-<STAGE>.log`。G4父GPU0/sidecar1，完整参数及检查均保留。两条20步载体为`smoke-stage.sh`（SHA `6c1724f61245e981740ddf4c760b931d3a892bde5378f845e17b69d8c417ee2f`）；每条显式给新的`RUN_NAME`、`FUSION=ctx|modul`、`TRAIN_HEAD`，依次`MODE=contract|train|restore`。保存前参考位于该条`FINAL_RECORD_DIR/pre_save_reference.json`，恢复其真实step19。此时这些实际GPU检查尚待执行。

### 回填后占位负例的中断与修复

原三文件测试在75项通过后，`test_formal_placeholder_fails_before_credentials`因依赖模块默认占位值而误进入真实库的本地打包。主会话精确停止pytest404213及uv406311/pack406314子进程，全部确认退出；原测试真实退出2、75项通过、163.37秒，日志以`build-pins-tests-aborted.summary.log`单独保留。运行尚未到Publisher，未上传、未改源数据。

已现场核对导出根出生时间为美东03:55:51，确为本轮测试新建；原路径整体rename为`v1-store/exports/hf-orig100-trainset-test-aborted-20261028`留证，不删除、不作为正式结果复用。112文件中106为数据的硬链接，独立文件合计440559838字节；详见`build-pins-test-incident.json`。

独立同规模调整审查通过后，S5提交0d1f88c仅在该负例显式monkeypatch占位DATASET，模拟两种默认均由原guard拒绝，0导出调用、输出根未创建，定向1项通过。主以e8e517b单独合并此测试修复并推送，合前/合后审查PASS；13项主会话常量与文档改动没有混入该合并。随后完整重跑原三文件测试，原中断记录不被覆盖。

另经独立`ADJUST_REVIEW=PASS scale=same findings=0`确认，既定训练库导出仅依赖已封口数据/G0/G2/G3/norm，不依赖G4、portable输入或短测。因此真实常量提交后可提前与原G4/20步并行启动，输出仍为独立正式根`v1-store/exports/hf-orig100-trainset`，GPU禁用、8打包线程，公开范围和完整SHA/匿名回读不变；所有GPU验证仍在正式训练前完整执行。

修复后原三文件完整重跑实际为`106 passed in 34.18s`、退出0，见`records/build-pins-tests.summary.log`；原75项后中断的记录保持不变。真实常量与配套档案随后单独提交。
