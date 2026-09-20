#!/usr/bin/env python3
"""从训练日志生成 motion80k bucket README 的「训练结果」一节。

由 run_motion80k_ckpt_export.sh 在趟 B 调用，把生成的 markdown 块替换掉 README 模板里的
<!-- TODO-PASS-B-RESULT --> 哨兵。**所有数字都从日志现读**，不接受人工输入——这样趟 B
可以无人值守跑完，也不会出现「README 里的数字和日志对不上」。

用法: motion80k_result_block.py <train.log> <README 模板> <输出 README>
"""
import datetime as dt
import re
import sys

TRAIN_LOG, TEMPLATE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

text = open(TRAIN_LOG, encoding="utf-8", errors="replace").read()


def kv(name):
    m = re.search(rf"^{name}=(.*)$", text, re.M)
    return m.group(1).strip() if m else None


start, end, exit_code = kv("START_UTC"), kv("END_UTC"), kv("EXIT_CODE")

# "Step 50300: grad_norm=..., loss=0.0018, ..." —— log_interval=100，除 step 0 外都是区间统计
steps = [(int(s), l) for s, l in
         re.findall(r"^Step (\d+): .*?loss=([0-9.eE+-]+)", text, re.M)]
steps.sort()

# 取整千的里程碑 + 首尾两条
want = [0] + list(range(5000, 80001, 10000))
picked, seen = [], set()
for w in want:
    cand = [sl for sl in steps if sl[0] == w]
    if cand and cand[0][0] not in seen:
        picked.append(cand[0]); seen.add(cand[0][0])
if steps and steps[-1][0] not in seen:
    picked.append(steps[-1])
picked.sort()

# 最后一条 tqdm 进度行里的 elapsed / rate
prog = re.findall(r"Progress on: ([0-9.]+kit)/([0-9.]+kit) rate:([0-9.]+it/s).*?elapsed:([0-9:]+)", text)
last_prog = prog[-1] if prog else None

lines = []
if start and end:
    try:
        t0 = dt.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
        t1 = dt.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
        sec = int((t1 - t0).total_seconds())
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        lines.append(f"起跑 `{start}`，退出 `{end}`，耗时 **{h} 小时 {m} 分 {s} 秒**"
                     f"（{sec:,} 秒，含初始化、checkpoint 保存与退出）。")
    except ValueError:
        lines.append(f"起跑 `{start}`，退出 `{end}`。")
elif start:
    lines.append(f"起跑 `{start}`（日志中未见 END_UTC）。")
else:
    lines.append("训练日志中未见 START_UTC / END_UTC。")

if exit_code is not None:
    lines.append(f"`EXIT_CODE={exit_code}`" + ("（正常退出）。" if exit_code == "0" else " —— **非零，非正常退出**。"))
else:
    lines.append("**日志中未见 EXIT_CODE，不能断言是正常退出还是被杀。**")

if picked:
    lines += ["", "| 日志步 | loss（区间统计，step 0 除外） |", "|---|---:|"]
    lines += [f"| {st} | {ls} |" for st, ls in picked]
    last_step, last_loss = steps[-1]
    lines += [
        "",
        f"`log_interval=100`，除 step 0 外均为日志区间统计。最后一条 loss **{last_loss} 属于 "
        f"step {last_step}**，覆盖训练 step {last_step - 99}–{last_step}，"
        f"**不能称作 step 79999 的单步 loss**；step {last_step + 1}–79999 共 {79999 - last_step} 步没有独立标量日志。"
        "**训练 loss 不代表策略评估成功率。**",
    ]
else:
    lines.append("\n日志里没有解析到任何 `Step N: ... loss=` 行。")

if last_prog:
    kit, total, rate, elapsed = last_prog
    lines += ["", f"吞吐（存储为 AWS 本地 NVMe RAID `/dev/md0`，batch 128、worker 16、fsdp 8）："
                  f"日志末条进度为 {kit}/{total}，速率 **{rate}**，elapsed **{elapsed}**。"]

block = "\n".join(lines)
tpl = open(TEMPLATE, encoding="utf-8").read()
sentinel = "<!-- TODO-PASS-B-RESULT -->"
if sentinel not in tpl:
    sys.exit(f"错误: README 模板里找不到哨兵 {sentinel}")
out = tpl.replace(sentinel, block)
# 模板顶上那条「等趟 B 回填」的提示行一并去掉
out = re.sub(r"^<!-- TODO-PASS-B: .*-->\n\n?", "", out, flags=re.M)
if "TODO-PASS-B" in out:
    sys.exit("错误: 生成后仍残留 TODO-PASS-B 哨兵")
open(OUT, "w", encoding="utf-8").write(out)
print(f"README 已生成: {OUT}（训练结果 {len(block)} 字符，loss 采样 {len(picked)} 条）")
