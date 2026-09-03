#!/usr/bin/env python3
"""greatlakes 提交器:优先复用 ControlMaster 主连接(认证一次,8h 免认证免手机)。

理念:**没有 master 就建立 master**——首次用 GLPW(+GLOTP)经 pexpect 驱动系统 ssh 建立
ControlMaster 主连接;之后所有提交/查询都经系统 ssh 复用该连接、零密码零 MFA 零手机
(slave 不发起认证握手,与服务器 MFA 策略无关)。依赖 ~/.ssh/config 的:

    Host greatlakes
        HostName greatlakes.arc-ts.umich.edu
        User hongzefu
        ControlMaster auto
        ControlPath ~/.ssh/cm-%r@%h:%p
        ControlPersist 8h

密码安全(硬规则):本文件绝不含密码,也绝不把密码写入任何文件。凭据由用户即时提供,
仅经临时环境变量 GLPW(建 master 时必需)/ GLOTP(可选 6 位 TOTP,推荐)传入,用完即 unset。

用法:
  export GLPW='<pw>' GLOTP='<6位TOTP>'   # 仅当无 master 需新建时才必需;master 在则可不设
  uv run --no-project --with pexpect python scripts/training/gl_submit.py
  uv run --no-project --with pexpect python scripts/training/gl_submit.py "squeue -u hongzefu"
  unset GLPW GLOTP
"""
import os
import shlex
import subprocess
import sys

ALIAS = os.environ.get("GL_SSH_ALIAS", "greatlakes")  # ~/.ssh/config 里的 ControlMaster 别名
USER = os.environ.get("GLUSER", "hongzefu")
REPO = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA"
# 无参数时不再默认提交任何 job（历史 test20s 脚本已随清理删除），只打印队列状态。
# 提交任务请显式传完整命令；当前仓库的两个 Wan local2gpu 包装不是 sbatch 脚本，
# Great Lakes 作业需由调用者提供独立的 sbatch 命令或包装。
DEFAULT_CMD = f"squeue -u {USER} -o '%.18i %.9P %.22j %.8T %.10M %.6D %R'"


def master_alive() -> bool:
    """ControlMaster 主连接是否存活。纯本地 socket 检查,不出网。"""
    try:
        return subprocess.run(["ssh", "-O", "check", ALIAS],
                              capture_output=True, text=True, timeout=10).returncode == 0
    except Exception:
        return False


def build_master() -> None:
    """没有 master 就建立:用 GLPW(+GLOTP)经 pexpect 驱动系统 ssh 建 ControlMaster 主连接。"""
    pw = os.environ.get("GLPW")
    if not pw:
        sys.exit("ERROR: 无 ControlMaster 主连接,且未设 GLPW——无法建立。\n"
                 "  请 export GLPW='<pw>' GLOTP='<6位TOTP>' 后重试"
                 "(推荐 TOTP,给一次码即可、无需手机数字匹配)。")
    import pexpect  # lazy:仅建连时需要;master 已在的纯提交/查询路径不依赖它

    otp = os.environ.get("GLOTP", "")
    mark = "CONNECTED_OK_MARKER"
    ssh_args = [
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=25",
        "-o", "ControlMaster=auto",
        "-o", "NumberOfPasswordPrompts=1",
        ALIAS, f"echo {mark} && hostname",
    ]
    print(f">>> 无 master,正在建立 {ALIAS} 主连接"
          f"（{'TOTP 码认证' if otp else '留空触发 push,需手机数字匹配'}）...", file=sys.stderr)
    child = pexpect.spawn("ssh", ssh_args, encoding="utf-8", timeout=90)
    pats = [r"[Pp]assword:", r"[Pp]asscode", mark,
            r"(?i)are you sure you want to continue connecting",
            r"(?i)(permission denied|authentication failed)",
            pexpect.EOF, pexpect.TIMEOUT]
    sent_pw = False
    while True:
        i = child.expect(pats)
        if i == 0:
            if sent_pw:
                sys.exit("ERROR: 密码似乎被拒（再次询问 password）")
            child.sendline(pw)
            sent_pw = True
        elif i == 1:
            child.sendline(otp)            # 空串=回车=触发 push
        elif i == 2:
            break
        elif i == 3:
            child.sendline("yes")
        elif i == 4:
            sys.exit(f"ERROR: 认证失败（密码或 OTP 不对）:\n{child.before}")
        elif i == 5:
            sys.exit(f"ERROR: 连接意外结束（认证未通过?）:\n{child.before}")
        else:
            sys.exit(f"ERROR: 超时（网络或认证卡住）:\n{child.before}")
    child.expect(pexpect.EOF)
    if not master_alive():
        sys.exit("ERROR: 建立后 master 复核未通过,检查 ~/.ssh/config 的 ControlPersist。")
    print(">>> master 已建立（ControlPersist 8h 内免认证）", file=sys.stderr)


def main():
    # 远程命令 cwd 是 greatlakes 的 home,自定义命令一律前置 cd {REPO}（相对路径才找得到脚本）。
    if len(sys.argv) > 1:
        remote_cmd = f"cd {REPO} && " + " ".join(sys.argv[1:])
    else:
        remote_cmd = DEFAULT_CMD

    # 没有 master 就建立 master，然后经系统 ssh 复用执行（零认证）。
    if master_alive():
        print(f">>> 复用 {ALIAS} ControlMaster 主连接（免认证）", file=sys.stderr)
    else:
        build_master()

    print(f"[remote cmd] {remote_cmd}", file=sys.stderr)
    r = subprocess.run(["ssh", ALIAS, f"bash -lc {shlex.quote(remote_cmd)}"],
                       capture_output=True, text=True, timeout=120)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stderr.write("[remote stderr]\n" + r.stderr)
    print(f"[remote exit {r.returncode}]", file=sys.stderr)
    sys.exit(0 if r.returncode == 0 else 5)


if __name__ == "__main__":
    main()
