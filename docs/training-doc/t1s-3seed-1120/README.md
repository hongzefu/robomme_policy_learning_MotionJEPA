# t1s-3seed-1120 总览（Table I 补 seed 0/42 与 FrameSamp modulation 40k/45k）

- 计划：v18 分支 `1120-table1-3seed-aws-local-eval-plan.html`（开工版 1431a26）。共 32 次 692 局评估。
- 链：`t1s-batch1`（v18 四行配对 + mm-ctx/native 单跑）→ `t1s-batch2`（FrameSamp 六行配对 + 第三批 40k/45k 配对，ctx 两对垫底）→ 视需要 `t1s-fix<n>`。
- 下载：15 份 ckpt 于 2026-11-20 22:05–22:1x 美东下完；mm-mod 与 fs512 桶的 README.md 与旧 SHA256SUMS.pre.txt 不符（只有 README），ckpt 文件逐一 sha256 OK，check_checkpoint 全 PASS。
- GL ctx 两份：GL 会话上传遇 HF 403（私有存储满），未到；到时缺则 T1S_ITEM_SKIP。
- 汇总：结束后写 `table1_3seed.json`。

## 中间汇总

见 [`interim-20261121.md`](interim-20261121.md)（数据 `table1_3seed_interim.json`）。

## 逐次档案

- [t1s-mm-mod-s0](../t1s-mm-mod-s0/README.md)
- [t1s-mm-mod-s42](../t1s-mm-mod-s42/README.md)
- [t1s-e2e-s0](../t1s-e2e-s0/README.md)
- [t1s-e2e-s42](../t1s-e2e-s42/README.md)
- [t1s-wan1-s0](../t1s-wan1-s0/README.md)
- [t1s-wan1-s42](../t1s-wan1-s42/README.md)
- [t1s-mm-ctx-s0](../t1s-mm-ctx-s0/README.md)
- [t1s-mm-ctx-s42](../t1s-mm-ctx-s42/README.md)
- [t1s-wan9-s0](../t1s-wan9-s0/README.md)
- [t1s-wan9-s42](../t1s-wan9-s42/README.md)
- [t1s-native-s0](../t1s-native-s0/README.md)
- [t1s-native-s42](../t1s-native-s42/README.md)
- [t1s-fs512-s0](../t1s-fs512-s0/README.md)
- [t1s-fs512-s42](../t1s-fs512-s42/README.md)
- [t1s-fs512-c40k-s7](../t1s-fs512-c40k-s7/README.md)
- [t1s-fs512-c45k-s7](../t1s-fs512-c45k-s7/README.md)
- [t1s-fs1024-s0](../t1s-fs1024-s0/README.md)
- [t1s-fs1024-s42](../t1s-fs1024-s42/README.md)
- [t1s-fs1024-c40k-s7](../t1s-fs1024-c40k-s7/README.md)
- [t1s-fs1024-c45k-s7](../t1s-fs1024-c45k-s7/README.md)
- [t1s-fs2048-s0](../t1s-fs2048-s0/README.md)
- [t1s-fs2048-s42](../t1s-fs2048-s42/README.md)
- [t1s-fs2048-c40k-s7](../t1s-fs2048-c40k-s7/README.md)
- [t1s-fs2048-c45k-s7](../t1s-fs2048-c45k-s7/README.md)
- [t1s-fs4096-s0](../t1s-fs4096-s0/README.md)
- [t1s-fs4096-s42](../t1s-fs4096-s42/README.md)
- [t1s-fs4096-c40k-s7](../t1s-fs4096-c40k-s7/README.md)
- [t1s-fs4096-c45k-s7](../t1s-fs4096-c45k-s7/README.md)
- [t1s-fs512c-s0](../t1s-fs512c-s0/README.md)
- [t1s-fs512c-s42](../t1s-fs512c-s42/README.md)
- [t1s-fs1024c-s0](../t1s-fs1024c-s0/README.md)
- [t1s-fs1024c-s42](../t1s-fs1024c-s42/README.md)
