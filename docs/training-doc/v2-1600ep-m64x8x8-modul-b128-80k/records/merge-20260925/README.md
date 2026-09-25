# 2026-09-25 分支合并留档

`v2-motionmem` 在 `0571fea` 后分叉：主副本本地提交 `6c94989`（训练结果）与 `-temp` 副本推送的 `1a08afb`、`215d720`、`d023b72`（HF 上传）同改了上级目录的 `result.md`，merge 时冲突。

- `result.local-6c94989.md`：`6c94989` 版原文（完整训练结果）。
- `result.remote-215d720.md`：`215d720` 版原文（「尚未执行」占位 + 「HuggingFace 备份」一节）。
- 合并口径：`../../result.md` 取本地版全文，末尾接远端版「HuggingFace 备份」一节原文，其余不改；按用户要求两份原件原样入 git。
