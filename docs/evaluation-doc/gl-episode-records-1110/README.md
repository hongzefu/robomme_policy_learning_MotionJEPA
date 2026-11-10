# GreatLakes 官方 800 局评测逐局结果（9 次）

本目录是 9 次 GreatLakes 官方 test 16×50 = 800 局评测的逐局结果导出，供 AWS 本机评测做逐局配对检验（req=eval3x3-11092230-1）。只读导出：原记录未改动，没有重跑。

## 口径

- 列：`model,checkpoint,model_seed,task,episode_id,success,outcome`。`episode_id` 是官方 800 局清单的原始局号，与 `task` 合起来就是队列里的 `episode_key`（`<task>:<episode_id>`）。`success` 等于 1 当且仅当 `outcome=success`；`outcome` 取值为 `success`、`fail`、`timeout`。
- 9 次评测用同一份 800 局清单，内部签封 `manifest.json` 的 `sha256` 字段为 `991c34960b431f4e6170357dfe428e74e404f5ccdbf8dafddc3a0a380eaeecc6`。
- 每份文件 1 行表头加 800 行数据，按 `task`、`episode_id` 升序排列。
- 导出时核对了以下几项，全部通过：每个 run 的 accepted 记录正好 800 条，局号集合与清单逐个相等；每条记录都是 `record_type=terminal`；每条的 `run_identity_sha256` 都等于该 run 的 `spec.json`；`spec.json` 的 `model_seed` 等于文件名里的 seed；成功数与结果档案一致（见下表），没有不一致项。
- 来源记录：`<run 根>/accepted/*.json`，每局一个文件，取字段 `task_name`、`episode_id`、`outcome`。导出于美东 2026-11-09 22:24，在 GreatLakes 登录节点读 turbo。

## 文件、来源与 sha256

| 文件 | 成功数 | 档案数字 | fail / timeout | run 根（turbo） | run_identity_sha256 | 文件 sha256 |
|---|---|---|---|---|---|---|
| `orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k-ckpt59999-seed42.csv` | 304 | 304 | 468 / 28 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/official-stage2-1026/o16cc-60k-seed42/run800` | `38e4e94b08bb3e128947a1cdc80de76b913f34cff4c0b25a04e126bcd98f47d5` | `a04ae28bacdfc3edf7ff24aae4b5bd692f93bfc649a9dcec6d3a764d88039528` |
| `orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k-ckpt59999-seed7.csv` | 295 | 295 | 475 / 30 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/official-stage2-1026/o16cc-60k-seed7/run800` | `9158218a26037f13e7333c4766646cbc91f63111b5ecbba8a86f94049af71a75` | `be00ae1b7367b4788adbb734c1ca32e1c8670d426b7ac57a634f0ebd4f638e5c` |
| `orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k-ckpt59999-seed0.csv` | 297 | 297 | 468 / 35 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/official-stage2-1026/o16cc-60k-seed0/run800` | `cd036c7ea5be5d6d8e8b1ccec6d14975a5124f21c40a73ba7bef5cdd572afd50` | `f9ff47a2fce0d123d3f926bf2dbecbbf3300344b8962697d98df7125bc206bf3` |
| `gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k-ckpt79999-seed42.csv` | 276 | 276 | 494 / 30 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/o16abl-80k/o16abl-N-seed42/run800` | `bfffa7a5596618a05818bb19da43d6a225454c7b2cabd40815e1494228cda5b8` | `a664bb06dae2f1d299b363618038f173bffc814b549168a09ef83b777112d497` |
| `gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k-ckpt79999-seed7.csv` | 299 | 299 | 474 / 27 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/o16abl-80k/o16abl-N-seed7/run800` | `27c2ad2067c435d5fd50355e2a9b16921e944d71870173832b5279c97507d88a` | `444050ca9e9b2bc04a695bcdd8c4d4a395bb77c194f9910f2882d6f6aaef6d2e` |
| `gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k-ckpt79999-seed0.csv` | 308 | 308 | 462 / 30 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/o16abl-80k/o16abl-N-seed0/run800` | `cc9cc10d0096b6f272fe8be26a7b9d1ecff9ac8c9f79861b6d7be2529ac757e8` | `71a9c038a748d75b45a8b837f5ad80af4ee2bd531a1170f498360c93d20b07a0` |
| `gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k-ckpt79999-seed42.csv` | 263 | 263 | 513 / 24 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/o16abl-80k/o16abl-S-seed42/run800` | `3eb8480f6d2220d7754e62838fac01c9886aa4758a17b7263ead74c1292f852e` | `da1755e97ce4619eb6f25eee7def83e4cdbb3fe46628d5fe23e204f13003fa1d` |
| `gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k-ckpt79999-seed7.csv` | 291 | 291 | 478 / 31 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/o16abl-80k/o16abl-S-seed7/run800` | `0749ca8a9c977109ba3b3caf841f915967b2983f3cd32364377c0846702dcec7` | `e4d60d05257ad277401c09be4281b8d6cf0f754aab24484c0c5f85eb1add45b8` |
| `gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k-ckpt79999-seed0.csv` | 274 | 274 | 494 / 32 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/o16abl-80k/o16abl-S-seed0/run800` | `9190d6995b4ef07b650612e5fa5210249d408f48f1f349667791739574ddeea2` | `be4c9ea08a287e8563d190506ecd51b3b35f5f9c0f2838bdee18e0ee44a639ed` |

## 结果档案

- concat 1×1024（ckpt 59999）：`docs/evaluation-doc/o16-3seed-1031/result.md`（concat 列，304/295/297）。
- noxyz、nostate（ckpt 79999）：`docs/evaluation-doc/o16abl-3seed-1108/result.md`（提交 `c34b9c4`，276/299/308 与 263/291/274）。
