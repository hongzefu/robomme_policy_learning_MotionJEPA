# 两库normal档4+4并行300步perf：起跑前启动档（待V11.17Beta，未执行）

本档案基于冻结[旧perf300-ready启动档](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-ready-candidate-20260926/launch.md)，其SHA256为 `86b6140521928a2298391334abf377ebf73b4faec47d8c1db6347f0a080cca8c`，旧稿与更早wrapper故障记录保持原字节。仅更新显式清除新selector、normal/schema2记录核验及实现版本口径；两GPU并行→双方均退出→full恢复→count恢复→联合report→两measurement→预算共8阶段、子训练参数、wrapper/f8、0.5秒采样和三margin公式不改。正式perf尚未执行；实际TRAIN_HEAD待独立commitV11.17Beta生成，不预填任何未来身份或结果。

接口实现锚点已形成：`aec86db64e5178e63d9e7f77d3f5bc235db16390`，新增显式档位/schema2的runner、contract、speed、observer及对应3份测试共7文件已提交，377项核心验证通过。不能再以00bd/b0工具字节作为现行接口，也不能声称相对旧normal版本只改文档。未来perf TRAIN_HEAD仍须独立实际Beta，并与实际确定性20 Beta核对受保护源码、依赖和数据指纹；本档案不伪造未来HEAD。

**只有两库确定性20的正式judge均PASS，四run真实恢复及六任务独立退出全部通过，才可推进normal300 perf。** 确定性20准备组为[orig80k-timing20-det-0927](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-det-0927/README.md)，准备TAG为 `20260927T012426Z`；其实际Beta为 `c71d5255597db2f26930b7b5684a1d5b2994cf75`；两库正式judge已PASS，root七phase宿主/原生/capture均0，最终独立报告与归档引用由root补齐。旧[normal20失败](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-0926/result.md)及[off复验](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-off-repeat-0927/result.md)保留原HEAD与FAIL，不能用未来确定性档结果改写，也不能声称normal逐位差异根因已解决。

INPUT与P1最外退出限制已按用户批准补记沿用，100六run/四judge维持原00bd/ecf实际锚点及通过范围。沿用旧输入/上游证据须实查模块、依赖、配置、数据/资产指纹；不修改100单跑/并跑判据，也不让新schema自动把缺失profile的旧normal记录判成新合格数据。

用户批准两库各一对20步、perf补0.5秒只读磁盘采样，且20确定性档不扩展到perf/prod。实际GPU命令必须 `env -u ORIG80K_TIMING_EQ_PROFILE`，不依赖控制shell或tmux环境继承；normal描述符为 `{"name":"normal","xla_flags":""}`，实际flags未设或空。本perf保持W&B online、真实保存与恢复。本轮终点仍是正式80k起跑前；三项margin没有获确认数值，不自选倍率或容量。

## 1. 调度、名称与固定训练口径

两库各4卡同时各跑300步：full固定物理GPU0–3，counting固定4–7；相同主仓uv环境、相同perf TRAIN_HEAD，各自独立run、记录、日志、JAX/CUDA/W&B缓存。根代理已采用两个新诊断名 `perf-orig80k-full-300-20260926T232339Z`、`perf-orig80k-count-300-20260926T232339Z`供本阶段准备；准备TAG `20260926T232339Z`不是实际起跑UTC，当前未创建实际run输出。

global batch64、fsdp4、workers4、seed42、modul512/4×4、原warmup10000/lr5e-5/EMA0.999、log100/save10000/keep10000和真实保存不变，仅perf步数为300。初始化仍为主仓 `v1-store/models/openpi-assets/checkpoints/pi05_base/params`，不在perf使用100步的save摘要替换器，也不安装20步共同输入/全state取证层。

| 库 | lib（相对主仓v1-store/datasets） | assets-dir父目录（相对v1-store/train-assets） | norm SHA256 |
|---|---|---|---|
| full | `16task-pub-1600ep` | `mme_vla_suite` | `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5` |
| counting | `4task-counting-pub-400ep` | `mme_vla_suite/4task-counting-pub-400ep` | `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9` |

history为 `perceptual-framesamp-modul.yaml`，SHA `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`。runner实际消费lib/framesamp，显式绑定同库source/episode manifest；asset-id仍为robomme，assets-dir不重复加robomme。

按以下顺序执行，任何一阶段失败停止依赖步骤并保留现场：

1. 所有前置证据通过 → 正式perf launch及其wrapper全文入Git → 独立Beta、clean、资源/预算复核。
2. 两个GPU runner并行；分别取得原生死pane、真实capture进程返回码和强绑定回执，联合原外层日志及driver收尾通过。
3. 两个GPU run均原生退出且capture联合验收后，依次执行full和counting的CPU真实恢复，避免恢复I/O和CPU工作污染另一侧尚在运行的稳态窗口。
4. 两侧恢复均通过 → 生成一份含peer的联合perf报告 → 在两个真实299目录仍存在时生成各自measurement receipt。
5. 小型证据归档并核验；按实际测量填写三类margin及schema2预算来源，调用现有预算验收函数。本轮到正式80k起跑前止步，不启动80k。

这是分阶段草案，不是一键自动连续脚本。两runner、两恢复、report、两measurement和budget共8个阶段全部经冻结guard门闩创建独立detached tmux，并分别取得原生退出证据。`remain-on-exit=on`会保留死窗；`has-session`成功不能代表仍在运行，会话消失也不能代表成功。本草案没有清理命令。

## 2. 前置和实际剩余空间

INPUT已按用户「同样补记限制，沿用INPUT结果（推荐）」批准保留，P1按「补记P1限制，补独立退出取证后继续（推荐）」保留入口/finite/分段/来源记录，100六run及四judge已完成。原INPUT量具/B仍为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`、P1/100量具/B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`、上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，旧最外限制和原始证据不改。正式perf必须由根代理补入两库确定性timing20各自真实TIMING_EQ、四run真实恢复及独立退出证据的路径/HEAD/UUID/SHA；det两库原正式judge已PASS；最终六份独立报告与完成快照已经归档到 `29f71d44f3cd96a772f5fc8122d09037bd45970d`，不影响其原c71运行锚点。还须完成新Beta对旧前置的固定diff、模块/依赖和输入/训练/测速指纹核对，不能仅凭旧结果自动放行。实现锚点aec与实际确定性20/未来perf Beta之间，受保护scripts/training、src、pyproject.toml与uv.lock要求精确无差异；新Beta可补档案，不允许用本档案默许新的源码变化。若实际不同，停止并由主代理明确处置。

启动现场重新只读记录GPU UUID、占用/活动进程，主机内存、SHM、CPU/IO压力、底层存储与可用字节；停止本轮其他建库、验证及训练负载，不擅自终止用户其他任务。runner对自己的四卡要求显存占用0，实际`launch.json.storage`须核对为当前scratch设备，不能把report中写死的AWS标签当成存储实测。

当前smoke/perf的代码空间闸门只是 **可用≥300 GiB**，不等于两份新300步权重、保存临时写入、日志/cache的完整峰值预算。正式perf起跑前还须根据已经完成的新timing20真实checkpoint、既有占用和本轮新增目录估计剩余需求，并保留计划空间门槛；不凭旧20步11GiB就断言完整预算充足，也不自行清理空间。此处没有现场可用字节或预算通过数字。

## 正式Beta形成后的只读预检（任何mkdir和launch之前）

两库确定性20已在 `c71d5255597db2f26930b7b5684a1d5b2994cf75` 得到原正式judge PASS，root已核七phase宿主/原生/capture退出均0；见[确定性20结果](../orig80k-timing20-det-0927/result.md)。旧normal off/on及追加off的FAIL原样保留，此结果不宣称normal数值重复性根因已解决。

V11.16结果已归档为 `29f71d44f3cd96a772f5fc8122d09037bd45970d`；接着形成独立 `commitV11.17Beta`。本档案的perf TRAIN_HEAD仍待该Beta真实生成；下列预检仅在该clean HEAD执行，**先于第3节设置后的任何新命令根/结果根创建，也先于第5节mkdir**。stdout/stderr独占保存，宿主真实returncode另存；只在完整预检PASS且实际返回0、现场预算与所有前置都已复核后继续。此处没有实际perf预检或perf运行结果。

只读量具源为 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-schema2-candidate-20260927/check_normal_perf_preflight.py`，固定SHA256 `d3f39f3612c4db1ef1abfbffd4973338096776b7413497aa1cfa745d049e4f43`；不修改或复制脚本入docs。它锁5aba来源量具、113引用、两venv208依赖、保护runtime/设备资源、两run八session新输出和CPU实际normal300配置；实际子进程unset两selector/XLA、CPU隔离且保留继承x64，不在新HEAD调用read_side重判c71。六报告、七phase期末快照及两层宿主真实退出按原字节核验。

以下CLI已固定真实归档版本、六份报告及期末快照的路径/SHA；唯一待填项是尚未生成的V11.17Beta完整SHA。不准现场自算SHA后默认接受；它单列为text块以保留本launch既有10个Bash块原字节：

```text
/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv/bin/python -B /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-schema2-candidate-20260927/check_normal_perf_preflight.py \
  --archive-head 29f71d44f3cd96a772f5fc8122d09037bd45970d \
  --head '<V11.17Beta真实完整40位SHA>' \
  --det-checker-sha256 b0153c1b53a5e4de2141ff7d47b22b55fb429dc20b3efc69c2a88ea6316df6d7 \
  --det-controller /scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-det-0927/records/controller.completed.snapshot.json /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-det-completed-controller-snapshot-20260927.json 4056b9605408a44f61e1075b564f36bbe0854e40526adea9c302977a2b02cc58 \
  --det-report full-off /scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-det-0927/records/verification/full-off.independent-verification.json /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-det-independent-20260927/full-off.independent-verification.json d14e74238c3d64432c5e02cad31c99d8da570d89d5d03be85a27de7290d9f7c8 \
  --det-report full-on /scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-det-0927/records/verification/full-on.independent-verification-v2.json /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-det-independent-20260927/full-on.independent-verification-v2.json 5511d9d115e67448eec60c92ae9ce9c91a991ad2550a924d44ecf17b879ae016 \
  --det-report full-judge /scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-det-0927/records/verification/full-judge.independent-verification-v2.json /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-det-independent-20260927/full-judge.independent-verification-v2.json fd57adfa4cfccc36bc4c4d1edcef980a1039309b809203b4fa200820324c333c \
  --det-report count-off /scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-det-0927/records/verification/count-off.independent-verification.json /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-det-independent-20260927/count-off.independent-verification.json 1bcf0ea74f6de3364fd5f66587ec8cae380a5e746b3087d210d8d301799ea8ca \
  --det-report count-on /scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-det-0927/records/verification/count-on.independent-verification-v2.json /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-det-independent-20260927/count-on.independent-verification-v2.json 11b24c322e94d3534cd0027877851808a23ce067b90e337ec38268660563d127 \
  --det-report count-judge /scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-det-0927/records/verification/count-judge.independent-verification-v2.json /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-det-independent-20260927/count-judge.independent-verification-v2.json 1de68e2d005c93f7d1dde141c0a998d58ec617437e94ffd2ef7d69fdfec3ada0
```

六角色原source都在 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/timing20-det-independent-20260927/`。两off保留原封装；on与judge使用经无损整数保留的最终v2封装，首版记录不覆写。七phase完成快照必须固定source/归档SHA，每phase含process_ref和host_actual_exit=0，六jobs.state均completed；动态controller不是验收输入。上述路径和SHA已由root对正式归档与原件逐字核实；未来Beta生成后仍须实际执行完整预检，不凭本段文字自动放行。

已完成的候选语法/小记录检查见[准备验证](records/preparation/validation.json)和[preflight小验证](records/preparation/preflight-small-tests.json)；这些不是未来实际预检结果。原source引用留在JSON与[复制清单](records/preparation/records-source-manifest.json)中，scripts/venv/权重均不复制。

## 3. 预留目录、环境与固定版本

以下模板只在前置通过并正式落档后使用。TRAIN_HEAD填独立perf Beta的完整40位字面量，不能用动态git rev-parse代替期望锚点；不沿用当前P1 Beta作为尚未建立的perf版本。两run的REC、checkpoint根、driver日志由runner独占创建，外层不得提前mkdir REC。公用报告/命令根先检查全新，再按阶段创建。

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
IMPLEMENTATION_HEAD=aec86db64e5178e63d9e7f77d3f5bc235db16390
TRAIN_HEAD='<两库确定性timing20真实通过后，独立perf Beta的完整40位SHA>'
PERF_TAG=20260926T232339Z
FULL_RUN="perf-orig80k-full-300-$PERF_TAG"
COUNT_RUN="perf-orig80k-count-300-$PERF_TAG"
FULL="$STORE/datasets/16task-pub-1600ep"
COUNT="$STORE/datasets/4task-counting-pub-400ep"
FULL_ASSETS="$STORE/train-assets/mme_vla_suite"
COUNT_ASSETS="$STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep"
HISTORY_SHA=823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a
FULL_NORM=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
COUNT_NORM=a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9
COMMAND_ROOT="$STORE/bench/orig80k-perf300-commands-$PERF_TAG"
RESULT_ROOT="$STORE/bench/orig80k-perf300-results-$PERF_TAG"
FULL_REC="$STORE/bench/orig80k/$FULL_RUN"
COUNT_REC="$STORE/bench/orig80k/$COUNT_RUN"
FULL_CKPT="$STORE/train-runs/mme_vla_suite/$FULL_RUN"
COUNT_CKPT="$STORE/train-runs/mme_vla_suite/$COUNT_RUN"
PAIR_REPORT="$RESULT_ROOT/paired-speed.json"
MARGIN_SOURCE="$RESULT_ROOT/margin-source.json"
BUDGET_JSON="$RESULT_ROOT/prod-disk-budget.json"
RUNNER="$MAIN/scripts/training/prod/run_orig80k.sh"
COMPLETION_TOOL="$MAIN/scripts/training/tests/check_orig80k_completion.py"
SPEED_TOOL="$MAIN/scripts/training/tests/check_orig80k_speed.py"
CONTRACT_TOOL="$MAIN/scripts/training/orig80k_contract.py"
GUARD_TOOL="$STORE/bench/orig80k-build-preflight-0925/tmux-exit-guard-candidate-20260926/tmux_exit_guard.py"
GUARD_SHA256=f8dada4025dc49f260697e2fb768e922cbea7b8f98afe5cc8b98555648bff52b

[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ && "$PERF_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
test "$(git -C "$MAIN" rev-parse HEAD)" = "$TRAIN_HEAD" || exit 2
test -z "$(git -C "$MAIN" status --porcelain)" || exit 2
git -C "$MAIN" diff --exit-code "$IMPLEMENTATION_HEAD" "$TRAIN_HEAD" -- \
  scripts/training src pyproject.toml uv.lock || exit 2
for parent in "$STORE" "$STORE/bench" "$STORE/logs" "$STORE/cache"; do
  test -d "$parent" && test "$(realpath "$parent")" = "$parent" || exit 2
done
for path in "$COMMAND_ROOT" "$RESULT_ROOT"; do
  test ! -e "$path" && test ! -L "$path" || exit 2
done
for run in "$FULL_RUN" "$COUNT_RUN"; do
  for path in "$STORE/bench/orig80k/$run" "$STORE/train-runs/mme_vla_suite/$run" \
    "$STORE/logs/$run.driver.log" "$STORE/cache/jax/$run" "$STORE/cache/cuda/$run" \
    "$STORE/logs/wandb/$run" "$STORE/cache/wandb/$run" "$STORE/cache/wandb-config/$run" \
    "$STORE/cache/wandb-data/$run" "$STORE/cache/xdg-data/$run"; do
    test ! -e "$path" && test ! -L "$path" || exit 2
  done
done

GPU_ENV=(env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORMS -u JAX_PLATFORM_NAME
  -u XLA_FLAGS -u TRAIN_TIMING_STEPS -u ORIG80K_SMOKE_EQ_MODE -u ORIG80K_TIMING_EQ_PROFILE
  TRAIN_HEAD="$TRAIN_HEAD" HISTORY_CONFIG_SHA256="$HISTORY_SHA"
  UV_PROJECT_ENVIRONMENT="$MAIN/.venv" UV_CACHE_DIR="$STORE/cache/uv"
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1)
CPU_ENV=(env -u PYTHONPATH -u PYTHONHOME -u JAX_PLATFORM_NAME -u XLA_FLAGS
  -u TRAIN_TIMING_STEPS -u ORIG80K_SMOKE_EQ_MODE -u ORIG80K_TIMING_EQ_PROFILE
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
  UV_PROJECT_ENVIRONMENT="$MAIN/.venv" UV_CACHE_DIR="$STORE/cache/uv"
  OPENPI_DATA_HOME="$STORE/models" XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC)
```

GPU_ENV显式清除CPU平台、100步确定性XLA_FLAGS和旧TRAIN_TIMING_STEPS；**必须同时unset ORIG80K_SMOKE_EQ_MODE及ORIG80K_TIMING_EQ_PROFILE**，perf即使继承空selector也会被新runner拒绝；unset已写进实际GPU_ENV/CPU_ENV数组，不能只在控制shell处理。这里不覆盖继承的JAX_ENABLE_X64，实际启用时让preflight拒绝，不静默改输入。CPU阶段单条命令设置JAX_PLATFORMS=cpu和空CUDA列表，不泄漏给后续GPU阶段。

runner会source训练域paths.sh，设置OPENPI_DATA_HOME、UV/XDG/HF缓存及离线资产机制，逐run设置MMEVLA_JAX_CACHE_DIR/JAX_COMPILATION_CACHE_DIR、CUDA_CACHE_PATH、WANDB_DIR/CACHE/CONFIG/DATA_DIR、XDG_DATA_HOME，显存比例0.95，WANDB_MODE=online。原W&B凭据只由既有机制读取，不回显。runner内部perf调用speed.run；speed只在入口加载前把计时钩子替换为HostTiming，并设置其20/300步计数，不调用JAX profiler；不能把这层内部TRAIN_TIMING_STEPS误写成启用了原StepTiming profiler。

## 4. 原强wrapper、门闩启动与独立原生退出证据

正式perf Beta须把本launch的长父适配及wrapper body完整锁入Git文档，不能仅保留ignored准备稿链接。每次launch/capture/verify先确认真实TRAIN_HEAD与clean状态，核Git内 `scripts/training/tests/tmux_exit_guard.py`、工作树和原v1-store f8 runtime的完整字节/长度/SHA相同；`git_source_binding`随外部launch/capture记录留档并在verify重核。perf只绑定自身使用的guard源码，不把20模板加入执行依赖。guard运行路径和所有产物实体边界保持v1-store，f8源文件及旧100回执完全不改。

原强wrapper完整保留。其独立纯shell测试实际为[15/16符合预期，1例确认缺口](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-wrapper-tests-20260926/README.md)：最终直接append的printf完整写出0后自身非零，原日志验收仍可放行。这个结果不被改写，也不能说日志本身已证明最外层成功。本版在该层之外增加冻结[tmux guard](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/tmux-exit-guard-candidate-20260926/README.md)：所有阶段先门闩启动，实际结束后取得原生退出状态与外部capture真实返回码，再验证回执及原日志。

冻结f8已由主代理完成[真实三例集成](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/tmux-guard-integration-20260926T214038Z/summary.json)：真实0接受、日志0实际1拒绝、日志0信号15拒绝；两份本轮A100退出也已[实际取证](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/tmux-exit-guard-candidate-20260926/README.md)。这覆盖“日志完整成功但实际进程非零”的故障类型。新的perf阶段适配本次只做语法和接口静态核对，不冒称新perf运行或原wrapper的16/16测试通过。

正式使用前，本节完整wrapper及父进程取证适配必须随perf Beta锁定，并由执行方完成针对阶段适配的必要验证；guard继续使用f8原实体文件及SHA，不复制改写已产生回执的工具。展开后的实际命令保存在独立COMMAND_ROOT，用现场固定SHA自证相同字节；不在docs归档复制独立.sh或.command文件。原runner的EXIT_CODE/FOOTER_*可能被stdout转录；外层只按WRAPPER_*字段验收，不能混数。

```bash
guard_perf_action() {
  local action="$1" session="$2" command_sha=""
  if [ "$#" -eq 3 ]; then command_sha="$3"; fi
  env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 \
    "$MAIN/.venv/bin/python" -B - "$action" "$session" "$command_sha" "$TRAIN_HEAD" \
    "$MAIN" "$COMMAND_ROOT" "$GUARD_TOOL" "$GUARD_SHA256" <<'PY'
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

action, session, command_sha, head, main, command_root, tool_path, tool_sha = sys.argv[1:]
main, command_root, tool_path = Path(main), Path(command_root), Path(tool_path)
if not (tool_path == tool_path.resolve() and tool_path.is_relative_to(main / "v1-store")
        and hashlib.sha256(tool_path.read_bytes()).hexdigest() == tool_sha):
    raise SystemExit("加载前冻结guard实体路径/SHA不符")
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=main, text=True).strip() != head:
    raise SystemExit("正式Beta HEAD不符")
if subprocess.check_output(["git", "status", "--porcelain"], cwd=main, text=True):
    raise SystemExit("正式Beta工作区不clean")
source_specs = {
    "scripts/training/tests/tmux_exit_guard.py": tool_sha,
}
git_source_binding = {"head": head, "files": {}}
for relative, expected in source_specs.items():
    working_path = main / relative
    if not (working_path == working_path.resolve() and working_path.is_file()):
        raise SystemExit("正式源必须为固定实体文件: " + relative)
    committed = subprocess.check_output(["git", "show", head + ":" + relative], cwd=main)
    working = working_path.read_bytes()
    if committed != working or hashlib.sha256(committed).hexdigest() != expected:
        raise SystemExit("正式Beta源码字节或期望SHA不同: " + relative)
    if relative.endswith("/tmux_exit_guard.py") and committed != tool_path.read_bytes():
        raise SystemExit("正式Beta guard与冻结runtime副本字节不同")
    git_source_binding["files"][relative] = {
        "git_path": relative, "working_path": str(working_path),
        "bytes": len(committed), "sha256": expected}
git_source_binding["runtime_guard"] = {
    "path": str(tool_path), "bytes": len(tool_path.read_bytes()), "sha256": tool_sha}

spec = importlib.util.spec_from_file_location("frozen_perf_exit_guard", tool_path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
require = guard.require
tool_ref = guard.file_reference(guard.entity_path(tool_path, main))
require(tool_ref["sha256"] == tool_sha, "冻结guard实际SHA不同")
require(action in {"launch", "capture", "verify"}, "未知阶段动作")
command = command_root / (session + ".command")
identity_path = command_root / (session + ".identity.json")
launch_meta_path = command_root / (session + ".launch.process.json")
capture_meta_path = command_root / (session + ".exit.capture.json")
receipt_path = command_root / (session + ".exit.json")

def new_bytes(path, raw):
    guard.entity_path(path, main, exists=False)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())

def check_ref(value):
    guard.check_reference(value, main)
    return value

def read_launch():
    meta = guard.read_json(guard.entity_path(launch_meta_path, main))
    require(meta.get("git_source_binding") == git_source_binding, "launch正式Git源/runtime绑定不同")
    require(meta["schema"] == 1 and meta["session"] == session
            and type(meta["actual_exit_code"]) is int and meta["actual_exit_code"] == 0
            and meta["observer"] == "root subprocess.run.returncode", "launch外部实际返回码缺失")
    for key in ("command_file", "guard_tool", "stdout", "stderr", "identity"):
        check_ref(meta[key])
    require(meta["guard_tool"] == tool_ref and meta["command_file"] == guard.file_reference(command)
            and meta["identity"]["path"] == str(identity_path), "launch来源不匹配")
    identity = guard.read_identity(identity_path, meta["identity"]["sha256"])
    require(identity["session"] == session and identity["label"] == session
            and identity["train_head"] == head and identity["command_file"] == meta["command_file"],
            "identity归属不同")
    expected = [sys.executable, "-B", str(tool_path), "launch", "--session", session,
                "--train-head", head, "--label", session, "--command-file", str(command),
                "--command-sha256", meta["command_file"]["sha256"], "--identity-out", str(identity_path)]
    require(meta["argv"] == expected, "launch实际ARGV不同")
    payload = guard.read_json(Path(meta["stdout"]["path"]))
    require(payload["status"] == "RELEASED" and payload["created"] is True
            and payload["released"] is True and payload["release_uncertain"] is False
            and payload["identity"] == meta["identity"] and payload["tool"] == tool_ref
            and payload["command_file"] == meta["command_file"], "launch未证实先落身份后释放")
    return meta

def verify_capture(launch):
    meta = guard.read_json(guard.entity_path(capture_meta_path, main))
    require(meta.get("git_source_binding") == git_source_binding, "capture正式Git源/runtime绑定不同")
    require(meta["schema"] == 1 and meta["session"] == session
            and type(meta["actual_exit_code"]) is int and meta["actual_exit_code"] == 0
            and meta["observer"] == "root subprocess.run.returncode", "capture外部实际返回码缺失")
    for key in ("identity", "receipt", "guard_tool", "stdout", "stderr"):
        check_ref(meta[key])
    expected = [sys.executable, "-B", str(tool_path), "capture",
                "--identity", str(identity_path), "--identity-sha256", launch["identity"]["sha256"],
                "--out", str(receipt_path)]
    require(meta["argv"] == expected and meta["identity"] == launch["identity"]
            and meta["guard_tool"] == tool_ref and meta["receipt"]["path"] == str(receipt_path),
            "capture来源或ARGV不同")
    receipt = guard.validate_receipt(receipt_path, identity_path=identity_path,
                                    identity_sha256=launch["identity"]["sha256"])
    require(guard.read_json(Path(meta["stdout"]["path"])) == receipt, "capture stdout与回执不同")
    print("NATIVE_STAGE_EXIT=PASS session=" + session, flush=True)

if action == "verify":
    verify_capture(read_launch())
    raise SystemExit(0)

if action == "launch":
    require(guard.file_reference(guard.entity_path(command, main))["sha256"] == command_sha,
            "原命令SHA不同")
    argv = [sys.executable, "-B", str(tool_path), "launch", "--session", session,
            "--train-head", head, "--label", session, "--command-file", str(command),
            "--command-sha256", command_sha, "--identity-out", str(identity_path)]
    stdout_path = command_root / (session + ".launch.stdout")
    stderr_path = command_root / (session + ".launch.stderr")
    meta_path = launch_meta_path
else:
    launch = read_launch()
    probe = guard.probe_identity(identity_path, launch["identity"]["sha256"])
    if probe["status"] == "PENDING":
        print("NATIVE_STAGE_EXIT=PENDING session=" + session, flush=True)
        raise SystemExit(3)
    argv = [sys.executable, "-B", str(tool_path), "capture",
            "--identity", str(identity_path), "--identity-sha256", launch["identity"]["sha256"],
            "--out", str(receipt_path)]
    stdout_path = command_root / (session + ".exit.capture.stdout")
    stderr_path = command_root / (session + ".exit.capture.stderr")
    meta_path = capture_meta_path
for path in (stdout_path, stderr_path, meta_path):
    guard.entity_path(path, main, exists=False)
guard.entity_path(identity_path if action == "launch" else receipt_path, main, exists=False)
completed = subprocess.run(argv, capture_output=True, check=False)
if action == "capture" and completed.returncode == 3:
    # probe与capture之间若仍判alive，保留PENDING原输出且不预占最终侧证。
    require(not os.path.lexists(receipt_path), "PENDING异常占用了最终回执")
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    raise SystemExit(3)
new_bytes(stdout_path, completed.stdout)
new_bytes(stderr_path, completed.stderr)
meta = {"schema": 1, "session": session, "actual_exit_code": completed.returncode,
        "git_source_binding": git_source_binding,
        "identity": guard.file_reference(identity_path) if identity_path.is_file() else None,
        "guard_tool": tool_ref, "stdout": guard.file_reference(stdout_path),
        "stderr": guard.file_reference(stderr_path), "argv": argv,
        "observed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "observer": "root subprocess.run.returncode"}
if action == "launch":
    meta["command_file"] = guard.file_reference(command)
else:
    meta["receipt"] = guard.file_reference(receipt_path) if receipt_path.is_file() else None
new_bytes(meta_path, (json.dumps(meta, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
require(guard.file_reference(tool_path) == tool_ref, "执行期间冻结guard变化")
if completed.returncode:
    raise SystemExit(completed.returncode)
if action == "launch":
    read_launch()
    print("GUARDED_STAGE_RELEASED session=" + session, flush=True)
else:
    verify_capture(launch)
PY
}

capture_perf_stage() {
  guard_perf_action capture "$1"
}

launch_perf_stage() {
  local session="$1" log="$2" exit_field="$3"
  shift 3
  [[ "$session" =~ ^orig80k-perf-[A-Za-z0-9_-]+$ ]] || return 2
  [[ "$exit_field" =~ ^(RUNNER|COMPLETION|REPORT|MEASUREMENT|BUDGET)_EXIT_CODE$ ]] || return 2
  test ! -e "$log" && test ! -L "$log" || return 2
  local body='set +e
set -o pipefail
perf_session=$1
perf_head=$2
perf_main=$3
perf_log=$4
perf_command_file=$5
perf_command_sha=$6
perf_exit_field=$7
perf_wrapper_pid=$BASHPID
shift 7
( set -o noclobber; : > "$perf_log" ) || exit 2
(
  printf "WRAPPER_SESSION=%s\nWRAPPER_PID=%s\nWRAPPER_BODY_PID=%s\nWRAPPER_TRAIN_HEAD=%s\nWRAPPER_COMMAND_FILE=%s\nWRAPPER_COMMAND_SHA256_EXPECTED=%s\nWRAPPER_START_UTC=%s\n" \
    "$perf_session" "$perf_wrapper_pid" "$BASHPID" "$perf_head" "$perf_command_file" "$perf_command_sha" "$(date -u +%FT%TZ)" || exit 1
  actual=$(sha256sum -- "$perf_command_file") || exit 2
  actual=${actual%% *}
  printf "WRAPPER_COMMAND_SHA256_ACTUAL=%s\n" "$actual" || exit 1
  test "$actual" = "$perf_command_sha" || exit 2
  cd "$perf_main" || exit 2
  begin_head=$(git rev-parse HEAD) || exit 2
  begin_status=$(git status --porcelain) || exit 2
  test "$begin_head" = "$perf_head" || exit 2
  test -z "$begin_status" || exit 2
  printf "WRAPPER_COMMAND=" || exit 1
  printf "%q " "$@" || exit 1
  printf "\n" || exit 1
  "$@"
  action_rc=$?
  printf "%s=%s\n" "$perf_exit_field" "$action_rc" || exit 1
  if [ "$action_rc" -ne 0 ]; then exit "$action_rc"; fi
  actual_head=$(git rev-parse HEAD) || exit 2
  current_status=$(git status --porcelain) || exit 2
  version_rc=0
  if [ "$actual_head" != "$perf_head" ] || [ -n "$current_status" ]; then version_rc=2; fi
  printf "WRAPPER_HEAD_END=%s\nWRAPPER_VERSION_EXIT_CODE=%s\n" "$actual_head" "$version_rc" || exit 1
  exit "$version_rc"
) 2>&1 | tee "$perf_log"
codes=("${PIPESTATUS[@]}")
rc=${codes[0]}
if [ "${codes[1]}" -ne 0 ]; then rc=${codes[1]}; fi
tee_field="${perf_exit_field%_EXIT_CODE}_TEE_EXIT_CODE"
printf "WRAPPER_COMMAND_EXIT=%s\nWRAPPER_TEE_EXIT=%s\n%s=%s\nWRAPPER_END_UTC=%s\n" \
  "${codes[0]}" "${codes[1]}" "$tee_field" "${codes[1]}" "$(date -u +%FT%TZ)" | tee -a "$perf_log"
ending=("${PIPESTATUS[@]}")
if [ "${ending[0]}" -ne 0 ]; then rc=${ending[0]}; fi
if [ "${ending[1]}" -ne 0 ]; then rc=${ending[1]}; fi
if ! printf "WRAPPER_FOOTER_PRINTF_EXIT=%s\nWRAPPER_FOOTER_TEE_EXIT=%s\nWRAPPER_EXIT_CODE=%s\n" \
  "${ending[0]}" "${ending[1]}" "$rc" >> "$perf_log"; then
  printf "外层最终回执追加失败：%s\n" "$perf_log" >&2
  exit 1
fi
exit "$rc"'
  local command_file="$COMMAND_ROOT/$session.command" command_sha
  mkdir -p "$COMMAND_ROOT" || return 2
  (
    set -o noclobber
    {
      printf '%q ' bash -c "$body" -- "$session" "$TRAIN_HEAD" "$MAIN" "$log"
      printf '"$0" "$1" '
      printf '%q ' "$exit_field" "$@"
      printf '\n'
    } > "$command_file"
  ) || return 2
  command_sha=$(sha256sum -- "$command_file") || return 2
  command_sha=${command_sha%% *}
  printf 'COMMAND_FILE=%s\nCOMMAND_SHA256=%s\n' "$command_file" "$command_sha"
  guard_perf_action launch "$session" "$command_sha"
}

require_perf_stage() {
  local session="$1" log="$2" exit_field="$3" marker="$4" key tee_field
  guard_perf_action verify "$session" || return 1
  test -f "$log" && test ! -L "$log" || return 1
  tee_field="${exit_field%_EXIT_CODE}_TEE_EXIT_CODE"
  for key in "$exit_field" "$tee_field" WRAPPER_COMMAND_EXIT WRAPPER_TEE_EXIT \
    WRAPPER_FOOTER_PRINTF_EXIT WRAPPER_FOOTER_TEE_EXIT WRAPPER_EXIT_CODE WRAPPER_VERSION_EXIT_CODE; do
    test "$(rg -c "^${key}=" "$log")" = 1 || return 1
    rg -q "^${key}=0$" "$log" || return 1
  done
  test "$(rg -c "$marker" "$log")" = 1 || return 1
}

require_normal_perf_profile() {
  local records=$1 run=$2 report=${3-}
  env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 \
    "$MAIN/.venv/bin/python" -B - "$MAIN" "$TRAIN_HEAD" "$records" "$run" "$report" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

main, head, records, run, report_path = sys.argv[1:]
main, records = Path(main), Path(records)
normal = {"name": "normal", "xla_flags": ""}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    require(path.is_file() and not path.is_symlink(), "缺少实体记录: " + str(path))
    return json.loads(path.read_bytes())

require(records == main / "v1-store/bench/orig80k" / run, "记录根未绑定本run")
launch = read(records / "launch.json")
require((launch.get("mode"), launch.get("run_name"), launch.get("head")) == ("perf", run, head),
        "normal perf启动身份不符")
require(launch.get("timing_eq_profile") == normal, "perf必须显式normal档")
environment = launch.get("environment")
require(isinstance(environment, dict)
        and {"XLA_FLAGS", "ORIG80K_SMOKE_EQ_MODE", "ORIG80K_TIMING_EQ_PROFILE"} <= environment.keys(),
        "perf缺少实际档位环境字段")
require(environment["XLA_FLAGS"] in (None, "")
        and environment["ORIG80K_SMOKE_EQ_MODE"] is None
        and environment["ORIG80K_TIMING_EQ_PROFILE"] is None, "perf继承了对拍selector或flags")
require(launch["actual"].get("jax_enable_x64") is False, "perf实际x64未禁用")
source_sha = hashlib.sha256((main / "scripts/training/tests/check_orig80k_speed.py").read_bytes()).hexdigest()
start, final = (read(records / name) for name in ("speed_start.json", "speed_run.json"))
for snapshot in (start, final):
    require((snapshot.get("schema"), snapshot.get("head"), snapshot.get("mode"), snapshot.get("steps")) ==
            (2, head, "perf", 300), "perf起止必须为schema2且身份相同")
    require(snapshot.get("timing_eq_profile") == normal and "xla_flags" in snapshot
            and snapshot["xla_flags"] == environment["XLA_FLAGS"], "perf起止档位或实际flags不同")
    require(snapshot.get("argv") == launch["argv"] and snapshot.get("wrapper_sha256") == source_sha
            and snapshot.get("profiler") is False, "perf来源或实际训练参数不同")
require(final.get("success") is True and final.get("entry_calls") == 1
        and final.get("sampler_stopped") is True and not final.get("sampling_error"),
        "perf入口或主机采样未正常完成")
require(final["loader"]["batch_size"] == 64 and final["loader"]["workers"] == 4,
        "perf实际loader不同")
require(len(final["save_calls"]) == 1 and final["save_calls"][0]["step"] == 299
        and final["save_calls"][0]["state_step"] == 300, "perf未转发末步真实保存")
if report_path:
    report = read(Path(report_path))
    require(set(("full_or_first", "counting_or_second")) <= report.keys(), "缺少联合报告两侧")
    sections = [report["full_or_first"], report["counting_or_second"]]
    matched = [section for section in sections
               if section.get("disk", {}).get("checkpoint_root") == launch["checkpoint_dir"]]
    require(len(matched) == 1, "联合报告没有唯一对应本run的段")
    section = matched[0]
    require(section.get("head") == head and section.get("timing_eq_profile") == normal
            and "xla_flags" in section and section["xla_flags"] == environment["XLA_FLAGS"],
            "联合报告未保留显式normal及实际flags")
    require((section.get("batch_size"), section.get("workers"), section.get("steady_steps")) ==
            (64, 4, [100, 299]), "联合报告形制或稳态范围不同")
print("PERF_NORMAL_PROFILE=PASS run=" + run + " report=" + str(bool(report_path)), flush=True)
PY
}

require_driver_terminal() {
  local run="$1" log="$STORE/logs/$1.driver.log" key
  for key in TRAIN_PIPE_EXIT TEE_EXIT FOOTER_PRINTF_EXIT FOOTER_TEE_EXIT EXIT_CODE; do
    test "$(rg -c "^${key}=" "$log")" = 1 || return 1
    rg -q "^${key}=0$" "$log" || return 1
  done
}
```

`RUNNER_EXIT_CODE`直接取整个runner返回后的真实$?；completion/report/measurement/budget同理。阶段自身的`*_TEE_EXIT_CODE`和WRAPPER_TEE_EXIT来自该阶段唯一正文tee同一实际PIPESTATUS，不是读取子日志推测。若runner内部任何末尾记录失败，其真实非零会被外层捕获，不执行恢复或报告。

guard门闩在释放之前已经精确记录session、window/pane、pane PID与原始start command；`launch.process.json`绑定外部launch真实返回码、工具和命令字节、identity及原stdout/stderr。wrapper日志仍记录wrapper/body PID与起止HEAD/clean。训练PID取driver的TRAIN_WRAPPER_PID与final/start.json.pid，GPU采样PID取driver记录；不用裸pgrep猜测。完成器仍核这些进程已经结束。

每阶段的`capture_perf_stage`先只读probe；alive返回3且不占最终capture路径，调用者停止依赖步骤，待该identity的精确pane进程退出通知后再调用。不能按session消失判断完成，也不建立sleep忙轮询。dead时调用冻结capture并由独立父进程`subprocess.run.returncode`生成`<session>.exit.capture.json`，含identity/receipt/guard_tool/stdout/stderr的path/bytes/SHA、实际ARGV和UTC。`require_perf_stage`随后完全只读重核launch/capture侧证与f8 `validate_receipt`，再检查原日志；三层全部满足才继续。capture父进程的实际执行返回码也必须由调用工具记录，不能凭它写出的JSON猜测；正文的`|| exit 2`保留该失败传播。已经生成最终回执后不覆盖重捕获，只读验收可以重复。

本层恰覆盖原wrapper末次append完整写0后返回非零的反例：即使日志各字段均0，tmux仍记录该最外层bash的非零status，f8 capture返回1，独立侧证与纯validator拒绝；不是把原日志判断修饰成了独立退出证明。

## 5. 两个4卡GPU runner并行

确认全部前置和新输出守卫后，创建公用报告根；其余REC不提前创建。以下两命令连续发出、不等待full先结束，GPU组互不重叠。运行期间主仓tracked和共享venv保持冻结。

```bash
FULL_SESSION="orig80k-perf-full-$PERF_TAG"
COUNT_SESSION="orig80k-perf-count-$PERF_TAG"
FULL_LOG="$STORE/logs/$FULL_SESSION.wrapper.log"
COUNT_LOG="$STORE/logs/$COUNT_SESSION.wrapper.log"
mkdir "$RESULT_ROOT" || exit 2

launch_perf_stage "$FULL_SESSION" "$FULL_LOG" RUNNER_EXIT_CODE \
  "${GPU_ENV[@]}" NORM_STATS_SHA256="$FULL_NORM" \
  bash "$RUNNER" perf "$FULL_RUN" 0,1,2,3 "$FULL" "$FULL_ASSETS" || exit 2
launch_perf_stage "$COUNT_SESSION" "$COUNT_LOG" RUNNER_EXIT_CODE \
  "${GPU_ENV[@]}" NORM_STATS_SHA256="$COUNT_NORM" \
  bash "$RUNNER" perf "$COUNT_RUN" 4,5,6,7 "$COUNT" "$COUNT_ASSETS" || exit 2
```

观察每份精确日志，使用行缓冲流式工具；例如 `tail -n +1 -F <该侧wrapper日志> | grep --line-buffered -E 'ORIG80K_|FINAL_RECORD_|TRAIN_|RUNNER_EXIT_CODE|WRAPPER_EXIT_CODE|Traceback|Error'`。监听器退出0不能替代训练退出。完整原日志保留，不在原driver日志追加自制终态。

## 6. 两侧结束后，分别真实CPU恢复299

先确认两个identity各自的pane进程均已实际退出，分别执行capture；会话会按remain-on-exit保留，禁止用has-session成功拒绝已结束阶段。联合原生退出/外部capture实际0/回执后，再核外层RUNNER返回、原日志终态与每个driver自己的唯一TRAIN_PIPE/TEE/FOOTER/EXIT记录；不把其他run成功代替本侧。完成器将真实加载本run的299/params，保持dtype并与其末步EMA逐叶对比；不得用仅存在文件名或旧恢复记录替代。

```bash
capture_perf_stage "$FULL_SESSION" || exit 2
capture_perf_stage "$COUNT_SESSION" || exit 2
require_perf_stage "$FULL_SESSION" "$FULL_LOG" RUNNER_EXIT_CODE '^ORIG80K_TIMING_DONE steps=300 profiler=0$' || exit 2
require_perf_stage "$COUNT_SESSION" "$COUNT_LOG" RUNNER_EXIT_CODE '^ORIG80K_TIMING_DONE steps=300 profiler=0$' || exit 2
require_driver_terminal "$FULL_RUN" || exit 2
require_driver_terminal "$COUNT_RUN" || exit 2
require_normal_perf_profile "$FULL_REC" "$FULL_RUN" || exit 2
require_normal_perf_profile "$COUNT_REC" "$COUNT_RUN" || exit 2
for path in "$FULL_REC/completion.json" "$COUNT_REC/completion.json"; do
  test ! -e "$path" && test ! -L "$path" || exit 2
done
FULL_COMPLETION_LOG="$STORE/logs/orig80k-perf-full-completion-$PERF_TAG.wrapper.log"
COUNT_COMPLETION_LOG="$STORE/logs/orig80k-perf-count-completion-$PERF_TAG.wrapper.log"

launch_perf_stage "orig80k-perf-full-completion-$PERF_TAG" "$FULL_COMPLETION_LOG" COMPLETION_EXIT_CODE \
  "${CPU_ENV[@]}" uv run --no-sync python "$COMPLETION_TOOL" \
  --mode perf --records "$FULL_REC" --run-root "$FULL_CKPT" \
  --log "$STORE/logs/$FULL_RUN.driver.log" --run "$FULL_RUN" --head "$TRAIN_HEAD" \
  --out "$FULL_REC/completion.json" || exit 2
```

full恢复完成且该阶段全部终态通过后，才发counting恢复，避免两份CPU恢复互相争抢资源。两个工具都从同一perf TRAIN_HEAD读取已完成产物。

```bash
capture_perf_stage "orig80k-perf-full-completion-$PERF_TAG" || exit 2
require_perf_stage "orig80k-perf-full-completion-$PERF_TAG" "$FULL_COMPLETION_LOG" COMPLETION_EXIT_CODE \
  "^RUN_COMPLETED=PASS run=$FULL_RUN checkpoints=1 final=299$" || exit 2
launch_perf_stage "orig80k-perf-count-completion-$PERF_TAG" "$COUNT_COMPLETION_LOG" COMPLETION_EXIT_CODE \
  "${CPU_ENV[@]}" uv run --no-sync python "$COMPLETION_TOOL" \
  --mode perf --records "$COUNT_REC" --run-root "$COUNT_CKPT" \
  --log "$STORE/logs/$COUNT_RUN.driver.log" --run "$COUNT_RUN" --head "$TRAIN_HEAD" \
  --out "$COUNT_REC/completion.json" || exit 2
```

完成器必须得到state_step300、loop/checkpoint299、唯一checkpoint集合[299]、普通metrics步[0,100,200]及尾窗201…299的99步五标量全部有限，run/HEAD/UUID/W&B身份一致、真实wait_until_finished完成、保存norm匹配。两侧各自真实恢复摘要等于自身末步EMA；这不是两库之间参数相等的要求。

## 7. 联合稳态报告：保留原统计口径

两个恢复阶段均完成后，调用现有report CLI；peer两个参数必须同时提供。单侧报告虽然是CLI支持的形式，本阶段必须有联合报告来验证真实稳态重叠和两侧合计口径。

```bash
capture_perf_stage "orig80k-perf-count-completion-$PERF_TAG" || exit 2
require_perf_stage "orig80k-perf-count-completion-$PERF_TAG" "$COUNT_COMPLETION_LOG" COMPLETION_EXIT_CODE \
  "^RUN_COMPLETED=PASS run=$COUNT_RUN checkpoints=1 final=299$" || exit 2
test ! -e "$PAIR_REPORT" && test ! -L "$PAIR_REPORT" || exit 2
REPORT_LOG="$STORE/logs/orig80k-perf-report-$PERF_TAG.wrapper.log"
launch_perf_stage "orig80k-perf-report-$PERF_TAG" "$REPORT_LOG" REPORT_EXIT_CODE \
  "${CPU_ENV[@]}" uv run --no-sync python "$SPEED_TOOL" report \
  --records "$FULL_REC" --gpu "$FULL_REC/gpu.csv" \
  --peer-records "$COUNT_REC" --peer-gpu "$COUNT_REC/gpu.csv" --out "$PAIR_REPORT" || exit 2
```

新report在读取两份speed schema2和launch的显式normal/实际flags后才出报告；上述新增只读验收在训练终态及report完成后分别重复核本run起止身份、profile与报告归属，缺字段/旧schema/确定性档/混配均拒绝。当前report外层本身没有schema字段，不能凭空要求report.schema=2；schema2约束属于speed_start/speed_run，预算schema2是另一独立格式。

report输出 `full_or_first`、`counting_or_second`及联合字段，前者由本命令明确传full；后续预算仍按精确checkpoint_root选择所属段，不能仅凭“first”猜库。

- 预热0…99；steady为100…299，共200步。窗口始于step99完成同步，结束于step299训练完成、真实保存之前的同步；每侧samples/s为 `200*64/steady_seconds`。初始化/JIT与预热、save dispatch、save+收尾独立报告；单步host时间不能冒称GPU内核时间。
- GPU目标采样间隔0.5秒，报告每卡util均值、0%占比、慢步/其他步的分层均值及样本数。慢步只用host耗时大于稳态中位数2倍分类；中位数不作吞吐结论。报告重复相邻读数数量，不把相同NVML读数视为新增独立证据。
- GPU窗口须有首尾覆盖、足够密度；当前工具要求最大间隔≤2秒、均值≤0.6秒及样本数下界，不能静默放宽。host记录须覆盖steady；RSS总和含共享页重复计数，另记SHM峰值和系统MemAvailable最低值，不称独占物理内存。
- 两侧steady交集必须为正，报告真实重叠秒数。现有perf判据**没有**100步同入口对拍的“每侧50完整步”门槛，不自行套用；也不把正交集夸成两侧全部200个steady步完全并行。无交集则本次联合perf无效，停止并报告，不临时调阈值。
- 合计吞吐为 `2*200*64/(两窗口并集跨度)`，不简单相加两个不同窗口的单侧速率。80k ETA使用工具现有初始化估计、80000稳态步及8次末步save+收尾成本外推；只有一次末步保存观测，不将该估计包装成运行承诺或将含W&B收尾的跨度称作纯磁盘写时。

## 8. 0.5秒磁盘证据与清理前receipt

DiskSampler按单调时钟调度，每0.5秒只读本run完整checkpoint根及scratch可用量；首尾补采，记录实际间隔、扫描耗时、迟到、missed_ticks、路径改名/消失竞态及错误。分配量按 `st_blocks*512`和(device,inode)去重，含临时/最终目录项、不跟随软链；这不等于识别XFS reflink共享extent，也不是原子扫描。

report的disk字段必须绑定disk_samples.jsonl SHA、run_uuid、原生init/commit纳秒、wait SHA与HEAD。保存窗口上界是原生commit_timestamp_nsecs，不用包含W&B收尾的entry结束冒充；至少有一条完整落在该窗口内的周期采样。真正I/O错误、软链、采样未结束、无保存窗口样本、最终扫描仍有竞态或摘要不符，均停止预算放行。采样最大值仍是观测下界，不能因为增量为0就说没有临时写入。

**顺序不可颠倒：真实恢复PASS → report PASS → checkpoint仍在时创建receipt → 归档核验 → 才讨论清理。** 当前真实CLI是 `measure-checkpoint --group full|counting --run --head --report --completion --out`；out必须在该run REC内，不能放通用RESULT_ROOT或checkpoint目录。该动作只读stat及小元数据，不读权重内容、不再次恢复；但会独占写一份新回执。

```bash
capture_perf_stage "orig80k-perf-report-$PERF_TAG" || exit 2
require_perf_stage "orig80k-perf-report-$PERF_TAG" "$REPORT_LOG" REPORT_EXIT_CODE "^ORIG80K_SPEED=PASS out=$PAIR_REPORT$" || exit 2
require_normal_perf_profile "$FULL_REC" "$FULL_RUN" "$PAIR_REPORT" || exit 2
require_normal_perf_profile "$COUNT_REC" "$COUNT_RUN" "$PAIR_REPORT" || exit 2
FULL_RECEIPT="$FULL_REC/checkpoint_measurement.json"
COUNT_RECEIPT="$COUNT_REC/checkpoint_measurement.json"
for path in "$FULL_RECEIPT" "$COUNT_RECEIPT"; do
  test ! -e "$path" && test ! -L "$path" || exit 2
done
FULL_MEASURE_LOG="$STORE/logs/orig80k-perf-full-measure-$PERF_TAG.wrapper.log"
COUNT_MEASURE_LOG="$STORE/logs/orig80k-perf-count-measure-$PERF_TAG.wrapper.log"
launch_perf_stage "orig80k-perf-full-measure-$PERF_TAG" "$FULL_MEASURE_LOG" MEASUREMENT_EXIT_CODE \
  "${CPU_ENV[@]}" uv run --no-sync python "$CONTRACT_TOOL" measure-checkpoint \
  --group full --run "$FULL_RUN" --head "$TRAIN_HEAD" --report "$PAIR_REPORT" \
  --completion "$FULL_REC/completion.json" --out "$FULL_RECEIPT" || exit 2
launch_perf_stage "orig80k-perf-count-measure-$PERF_TAG" "$COUNT_MEASURE_LOG" MEASUREMENT_EXIT_CODE \
  "${CPU_ENV[@]}" uv run --no-sync python "$CONTRACT_TOOL" measure-checkpoint \
  --group counting --run "$COUNT_RUN" --head "$TRAIN_HEAD" --report "$PAIR_REPORT" \
  --completion "$COUNT_REC/completion.json" --out "$COUNT_RECEIPT" || exit 2
```

两份测量均完成后再收尾。测量器检查唯一299和完整参数/资产索引；对299子树和整个run根各做两遍稳定stat，拒绝仍在变化、软链/非普通项，并要求run根分配量等于磁盘最终采样。receipt保存逐项path/kind/device/inode/size/blocks/mtime_ns、去重算术、单份checkpoint与run根口径、测量工具SHA/时间，以及原生`_CHECKPOINT_METADATA`和`params/_METADATA`原文/SHA。

```bash
capture_perf_stage "orig80k-perf-full-measure-$PERF_TAG" || exit 2
capture_perf_stage "orig80k-perf-count-measure-$PERF_TAG" || exit 2
require_perf_stage "orig80k-perf-full-measure-$PERF_TAG" "$FULL_MEASURE_LOG" MEASUREMENT_EXIT_CODE \
  "^CHECKPOINT_MEASURED=PASS run=$FULL_RUN step=299 allocated_bytes=[1-9][0-9]* receipt_sha256=[0-9a-f]{64}$" || exit 2
require_perf_stage "orig80k-perf-count-measure-$PERF_TAG" "$COUNT_MEASURE_LOG" MEASUREMENT_EXIT_CODE \
  "^CHECKPOINT_MEASURED=PASS run=$COUNT_RUN step=299 allocated_bytes=[1-9][0-9]* receipt_sha256=[0-9a-f]{64}$" || exit 2
sha256sum "$PAIR_REPORT" "$FULL_REC/completion.json" "$COUNT_REC/completion.json" \
  "$FULL_REC/disk_samples.jsonl" "$COUNT_REC/disk_samples.jsonl" "$FULL_RECEIPT" "$COUNT_RECEIPT"
test "$(git -C "$MAIN" rev-parse HEAD)" = "$TRAIN_HEAD" || exit 2
test -z "$(git -C "$MAIN" status --porcelain)" || exit 2
```

receipt的evidence强绑定launch、联合report、真实completion、原始disk_samples和speed_run；completion还绑定runtime、gpu.csv/.err、metrics、final三文件、driver与原生元数据。原始小记录和日志后续必须保留，清洗副本另外归档；不能把清洗后的driver替换成原路径而破坏SHA。联合report被两侧同时引用，生成receipt后不得再追加或改写。

若以后按已授权范围清理临时权重，prod预算可从receipt的小元数据快照和原始disk_samples重新计算峰值、保存窗口及账目；不依赖已删除大权重，但必须仍有完整原始小证据。**没有先生成receipt就删除299，不能事后凭文件名补造**。本草案不安排任何删除或worktree清理。

## 9. 三类margin：只登记实测来源和输入字段

当前没有这轮300步实测数据，三类margin的数值和来源都不能预填。必须在真实perf/恢复/receipt之后形成独立margin_source文件，schema=1，并由执行方核实真实来源、适用形制与推导后固定文件SHA。程序只检查basis为非空文字，不能替人工证明容量推导合理；不能用测试夹具数字、占位说明或未经确认的倍率凑通过。

| 输入字段 | 类型/约束 | 需要的真实来源与说明 |
|---|---|---|
| checkpoint_growth_margin_bytes | 明确非负整数；填0也须主动给依据 | 两库各299的真实receipt、参数/EMA叶dtype/shape、原生保存格式/目录项；说明从300步单份观测外推未来每侧8份的增长风险与覆盖范围，不把整个run根冒充单份 |
| save_sampling_margin_bytes | 明确正整数 | 两库原始0.5秒采样、实际间隔/漏tick/扫描耗时/路径竞态，原生保存窗口和临时写入机制；说明对未观测峰值的保守覆盖，不由E=0自动取0 |
| logs_cache_margin_bytes | 明确正整数 | 两run真实日志/采样/配置/编译及W&B缓存占用、预计保留周期和正式80k日志/缓存来源；区分已占用量与后续新增需求，不重复算现存checkpoint |
| basis | 包含上面三个同名非空字符串 | 每项写可定位的真实文件/字段/HEAD/SHA、推导与适用范围；如数值或策略尚需用户裁决，提出具体测量依据后再填写 |

以下只是schema示意，null和占位文字必须全部替换，不是有效预算来源，更不是已确认容量：

```json
{
  "schema": 1,
  "checkpoint_growth_margin_bytes": null,
  "save_sampling_margin_bytes": null,
  "logs_cache_margin_bytes": null,
  "basis": {
    "checkpoint_growth_margin_bytes": "<真实receipt与增长依据>",
    "save_sampling_margin_bytes": "<保存窗口、采样盲区与保守覆盖依据>",
    "logs_cache_margin_bytes": "<日志/cache真实占用与新增需求依据>"
  }
}
```

此处没有默认10%、2倍或任意GiB容量；建库的数据余量也不能自动移植为保存峰值误差。现在可继续准备不依赖这些数字的前置，但正式prod预算不能跳过它们。

## 10. schema2预算与现有函数验收

当前 **没有 `budget` 子命令或自动预算生成CLI**。已实现的消费者是 `orig80k_contract.validate_disk_budget("prod", ...)`；prod runner经`ORIG80K_DISK_BUDGET_JSON`与`ORIG80K_DISK_BUDGET_SHA256`传给check_launch调用。不要为了检查预算而手动调用prod的完整check：它还做GPU/输出守卫并创建正式launch记录，不是纯预算预检。

预算JSON至少含以下实际字段。所有引用都是主仓内实体绝对路径及完整SHA；该图仅描述已实现接口，不创造新格式：

| 字段 | 实际内容 |
|---|---|
| schema | 2 |
| checkpoint_total_bytes、save_temporary_peak_bytes、logs_cache_margin_bytes | 按下述固定公式得到的整数 |
| margin_source | `{"path": "<实际margin-source.json>", "sha256": "<审阅固定SHA>"}` |
| source_perf.full / source_perf.counting | 两个不同perf run，head均为相同的真实perf HEAD；不能填写后续prod的HEAD冒充来源 |
| 每侧run_name、head | 与该侧launch/completion/receipt一致 |
| 每侧launch、report、completion、disk_samples、measurement | 各自`{path, sha256}`；measurement指该侧REC/checkpoint_measurement.json，report可共同引用本轮PAIR_REPORT |

对每侧s定义：C_s为receipt.checkpoint.allocated_bytes（单份299），F_s为report.disk.final_allocated_bytes（整个run根），P_s为report.disk.sampled_max_allocated_bytes（同一run根）。代码使用 **E_s=max(0,P_s−F_s)**，两侧相加；不用这次两run保存恰好错开来减少未来并发保存预算。

```text
retained_bytes = 8*C_full + 8*C_counting
checkpoint_total_bytes = retained_bytes + checkpoint_growth_margin_bytes
observed_extra_lower_bound_bytes = E_full + E_counting
save_temporary_peak_bytes = observed_extra_lower_bound_bytes + save_sampling_margin_bytes
logs_cache_margin_bytes = margin_source.logs_cache_margin_bytes
required_bytes = max(300*2^30,
    checkpoint_total_bytes + save_temporary_peak_bytes + logs_cache_margin_bytes)
```

C不能换成F后仍声称精确单份；P、F范围须相同。现存验证权重和缓存通过当前可用空间体现，不能计作未来可用量。`scratch_available_min_bytes`是共享盘压力证据，可能含另一run或其他写入，不直接再加到公式中重复计量。

正式数字与三项依据审阅完成、预算JSON独立新建并固定SHA后，下面只调用现有预算函数；这是现有函数调用正文，不是新增CLI。实际若在perf归档后的另一clean提交进行此动作，须另记录该动作版本及兼容性核对；source_perf的真实perf HEAD与旧receipt不得改写。

```bash
BUDGET_SHA='<实际schema2预算审阅后固定的64位SHA256>'
[[ "$BUDGET_SHA" =~ ^[0-9a-f]{64}$ ]] || exit 2
test -f "$MARGIN_SOURCE" && test -f "$BUDGET_JSON" || exit 2
BUDGET_LOG="$STORE/logs/orig80k-perf-budget-$PERF_TAG.wrapper.log"
launch_perf_stage "orig80k-perf-budget-$PERF_TAG" "$BUDGET_LOG" BUDGET_EXIT_CODE \
  "${CPU_ENV[@]}" uv run --no-sync python -c '
import json, pathlib, sys
main = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(main / "scripts/training"))
from orig80k_contract import validate_disk_budget
result = validate_disk_budget("prod", repo=main, budget_path=sys.argv[2], budget_sha=sys.argv[3])
print("PROD_DISK_BUDGET=PASS " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
' "$MAIN" "$BUDGET_JSON" "$BUDGET_SHA" || exit 2
```

该函数会重核所有引用、真实300/299/恢复、GPU/runtime链、原始采样与原生快照，重算单份/整run stat账目和两侧各8份算术，并用当前repo所在设备`f_bavail*f_frsize`对可用量验收。测量后清权重也仍重算原始采样，不只相信派生max/final数字。现有schema2不自动消费本草案的外层wrapper日志、identity、原生退出receipt或capture外部侧证；执行方仍须按本草案联合核对并归档这些独立证据，不能用预算函数PASS替代检查。两正式prod run最终须引用同一份经审阅的预算JSON和SHA；起跑时仍分别复核实时空间。

本段不能预写PROD_DISK_BUDGET=PASS。预算阶段实际结束后也必须完成同样的原生退出与外部返回码联合验收，并保存函数返回的完整分项、实时可用量和调用版本：

```bash
capture_perf_stage "orig80k-perf-budget-$PERF_TAG" || exit 2
require_perf_stage "orig80k-perf-budget-$PERF_TAG" "$BUDGET_LOG" BUDGET_EXIT_CODE \
  '^PROD_DISK_BUDGET=PASS ' || exit 2
```

预算通过也不直接等于80k全部前置或正式运行完成。本轮明确停在正式80k起跑前，不执行prod runner。

## 11. 归档清单与仍需明确的事项

8个阶段各保留原命令SHA、门闩identity、launch真实返回码侧证及stdout/stderr、原生exit receipt、capture外部侧证及stdout/stderr；逐项实体路径/bytes/SHA与工具f8来源均保留。死窗保留直到主代理按明确清单处理，本草案不发清理命令。

每run保存launch/runtime/run_meta、metrics、final三文件、真实completion、speed_start/speed_run、step_timing、host_samples、disk_samples、gpu.csv/.err、driver及各外层原始/清洗日志、checkpoint_measurement receipt。联合report、margin_source、schema2预算和各自SHA另存公共结果根。完整命令正文与期望/实际SHA、会话/pane/wrapper/body/训练/采样PID、起止UTC、版本/环境/设备/存储与异常均随正式档案登记；不复制权重、脚本、yaml或运行时.command载体到docs。

仍需根代理按真实结果补齐：两库确定性20的TIMING_EQ、四run真恢复/独立退出及实际20 HEAD/UUID/SHA；独立perf Beta的完整40位SHA及与旧前置的兼容指纹；现场GPU/CPU/SHM/IO/存储与可用字节、完整剩余预算；未来实际START/END、PID、run UUID、W&B ID；真实perf/恢复/receipt后才有依据的三类margin具体值、basis及固定文件SHA。准备名称已确认，但准备TAG不等于起跑时间。INPUT/P1批准不再待答，原证据限制仍保留；确定性20已知PASS与后续尚未执行的perf必须分别记账，不能预填perf PASS。

本文件准备阶段只做固定CLI、Bash/Python嵌入正文语法、链接、原训练ARGV/统计/预算接口保留及空白静态核对；不执行任何launch/capture、tmux或训练入口。原perf wrapper的15/16记录和f8真实三例分别保留，不能相互冒充本次适配实跑。未来正式落档前仍须完成阶段适配核对；当前不变更冻结工具、测试源、原报告或venv，这些准备检查不表示已启动训练、恢复、测盘或清理；本轮仍止于正式80k起跑前。

本档案控制文本仅已核Bash/Python语法、链接、8stage原子ARGV、原wrapper body、统计段、margin段和预算公式保留，GPU/CPU环境仅增加unset新selector，以及文件范围/空白。所有实际perf结果均待运行；不执行launch/capture、训练、恢复、report、measurement、预算函数或任何清理。两份run的十二节档案见[full README](../perf-orig80k-full-300-20260926T232339Z/README.md)与[count README](../perf-orig80k-count-300-20260926T232339Z/README.md)。
