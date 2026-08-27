# 4task-gl-framesamp 全量打包（S4）launch 记录

对应 `v2-framesamp-restructure-plan.md` 阶段 2 S4（A.1 布局契约 / A.2 打包工具）。
按 AGENTS 12 起跑前预提交：commit、命令、配置、数据来源、输出路径与判据。

## 起跑环境

- **起跑 commit**：`6ee74948c3a79332143172a54236b45657740a18`（commitV3.1，clean HEAD）
- **机器**：本机（2×RTX 6000 Ada 所在工作站；本任务纯 CPU+NFS，不占 GPU）
- **执行方式**：detached tmux（AGENTS 7；`PYTHONUNBUFFERED=1` + `pipefail` + `tee` + 尾行 `EXIT_CODE=`）

## 数据来源与输出

- **源库（只读，一字不动）**：`v1-store/datasets/4task-gl/`（678 GB；1600 episodes；
  483,291 帧；395,289 执行样本）
- **清单（唯一真值源）**：`v1-store/episode_manifest.json`，顶层 sha256
  `20da0dfe9a3673b4170912d3ac471ac1362990312884e3a61b353b08f37758a3`
- **输出（新增旁路库，库名 2026-08-27 经用户确认）**：
  `v1-store/datasets/4task-gl-framesamp/`（预计 31.7 GB：image 32 个 part 前 31 个
  ≈991–1020 MB + 末 1 个 ≈620.7 MB、pos 表 28.8 MB、state 表 15.5 MB）
- **切分预演**（`pack_framesamp_store.py plan` 现场输出）：32 part、阈值 15,103、
  part_000=episodes[0..55] 15,137 行、part_031=episodes[1573..1599] 9,471 行
  620,691,456 B——与计划 A.1 数字逐字一致

## 命令

```bash
tmux new-session -d -s pack-framesamp "bash scripts/data-pack-framesamp/run_pack.sh"
```

`run_pack.sh` 内部（默认参数即本次配置）：pack（`--reader decode` 首跑零布局假设、
`--procs 16`）→ verify（全量 483,291 帧三键对拍 + row_digests，`--resume` 接管
pack 留下的锁）。日志 `v1-store/logs/pack-framesamp.log`。

## 判据

- pack：`PACK_DONE=1`（写侧逐帧校验①pos memcmp/②state 同源/③read-after-write 全过，
  meta 阶段 1 落盘 status=packed）
- **verify（交付判定）**：`VERIFY_PACK=PASS scanned=483291 mismatches=0`（g 级零遗漏
  唯一凭据；meta 回填 status=verified + row_digests 落盘后才释放 pack.lock）
- 预计耗时：pack 40–80 min（双趟源读取 ≈582 GB）+ verify 20–40 min（≈291+31.7 GB）

## 探针前置（slice 档前提复核，本次虽用 decode 档仍留存）

```
$ uv run python scripts/data-pack-framesamp/probe_layout.py --source v1-store/datasets/4task-gl --n 30
PROBE_LAYOUT=PASS files=31 st_size=602951 offsets=(262595,541352,602906)
```

结果留档（README.md：耗时、写侧校验与 verify 结果、meta 摘要）随 run 结束后以
docs commit 补交。
