# v1-computeonly-b64 records 审计注记

本目录四个数据文件随 run 于 2026-08-24 归档，非本次新增。2026-08-28 `v1-95util.md` 双审计轮
（用户批准）补充两条已知限定，供后续引用「零取数成本 → util 99.9%」结论时参考：

1. **采样密度限定**：本档只有 15 s legacy 采样通道（`gpu_util.csv`，无 500 ms dense），稳态窗口
   重算 util 均值 99.92% 数值成立，但可能漏掉短空窗；该结论支持「去掉持续数据供给后计算可接近
   满载」，不支持「其他因素完全无责」的绝对化表述。
2. **起跑 commit 记录出入**：`result.md` 写起跑 commit `16bd8b8`，而 `records/env.json` 的
   `git_head` 为 `d0fa27e9f26e41513aa795b6b44403baba6f7e8b`，两处不一致；引用 provenance 时以
   `env.json` 实录为准。
