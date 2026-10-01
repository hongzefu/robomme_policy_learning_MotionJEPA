"""只读核对保留pane的真实退出状态；门闩启动只操作本次独占的新session。"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

REPO_ROOT = Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
TMUX = "/usr/bin/tmux"
SCHEMA = 1
PENDING_EXIT_CODE = 3
SEP = "\x1f"
END = "\x1e"
FIELDS = ("session", "window_id", "pane_id", "pane_pid", "pane_dead", "pane_dead_status",
          "pane_dead_signal", "pane_dead_time", "pane_start_command", "remain_on_exit")
FORMAT = SEP.join("#{" + {"session": "session_name", "remain_on_exit": "remain-on-exit"}.get(key, key) + "}"
                  for key in FIELDS) + END


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc_now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def utc_time(value):
    require(isinstance(value, str), "时间必须是带时区字符串")
    result = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None and result.utcoffset() == datetime.timedelta(0), "时间必须明确为UTC")
    return result


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON含重复键")
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError("JSON含非法数值: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=invalid)


def entity_path(value, repo, *, exists=True):
    repo = Path(repo)
    require(repo.is_absolute() and repo == repo.resolve() and repo.is_dir(), "仓库根不是实体绝对路径")
    store = repo / "v1-store"
    require(store.is_dir() and store == store.resolve(), "v1-store不是实体目录")
    path = Path(value)
    require(path.is_absolute() and path == path.resolve() and path.is_relative_to(store), "路径必须是本仓v1-store内的实体绝对路径")
    if exists:
        require(path.is_file() and not path.is_symlink(), "输入不是普通实体文件")
    else:
        require(path.parent.is_dir() and path.parent == path.parent.resolve(), "输出父目录必须已存在且为实体")
        require(not os.path.lexists(path), "输出已存在，拒绝覆盖或复用")
    return path


def file_reference(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def check_reference(reference, repo):
    require(isinstance(reference, dict) and set(reference) == {"path", "bytes", "sha256"}, "文件引用字段错误")
    require(type(reference["bytes"]) is int and reference["bytes"] >= 0
            and isinstance(reference["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]),
            "文件引用缺整数字节或完整SHA256")
    path = entity_path(reference["path"], repo)
    require(file_reference(path) == reference, "实际文件字节/SHA与引用不同")
    return path


def validate_identity(identity, *, repo=REPO_ROOT):
    required = {"schema", "session", "window_id", "pane_id", "pane_pid", "train_head", "label", "command_file",
                "pane_start_command", "initial_pane_dead", "remain_on_exit", "recorded_utc"}
    require(isinstance(identity, dict) and required <= set(identity) and type(identity["schema"]) is int
            and identity["schema"] == SCHEMA, "identity缺schema1必要字段")
    for key in ("session", "label", "pane_start_command"):
        value = identity[key]
        require(isinstance(value, str) and value and not any(char in value for char in (SEP, END, "\x00")),
                "identity文本为空或含保留分隔符")
    require(not any(char in identity["session"] for char in ("\n", "\r", "\t", ":")), "session名称不能可靠精确定位")
    require(re.fullmatch(r"@[0-9]+", identity["window_id"]) and re.fullmatch(r"%[0-9]+", identity["pane_id"]),
            "window/pane必须为原生ID")
    require(type(identity["pane_pid"]) is int and identity["pane_pid"] > 1, "pane_pid必须为真实正整数")
    require(isinstance(identity["train_head"], str) and re.fullmatch(r"[0-9a-f]{40}", identity["train_head"]),
            "缺共同TRAIN_HEAD完整字面量")
    require(type(identity["initial_pane_dead"]) is int and identity["initial_pane_dead"] == 0
            and identity["remain_on_exit"] == "on", "identity须证明记录时alive并已开启本窗口保留")
    utc_time(identity["recorded_utc"])
    check_reference(identity["command_file"], repo)
    return identity


def read_identity(identity_path, identity_sha256, *, repo=REPO_ROOT):
    path = entity_path(identity_path, repo)
    require(isinstance(identity_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", identity_sha256), "identity期望SHA非法")
    before = file_reference(path)
    require(before["sha256"] == identity_sha256, "identity实际SHA不同")
    identity = validate_identity(read_json(path), repo=repo)
    require(file_reference(path) == before, "读取期间identity发生变化")
    return identity


def pane_argv(identity, *, filtered=True):
    argv = [TMUX, "list-panes", "-s", "-t", "=" + identity["session"]]
    if filtered:
        argv += ["-f", "#{==:#{pane_id}," + identity["pane_id"] + "}"]
    return [*argv, "-F", FORMAT]


def query(argv, runner=subprocess.run):
    """保留原始字节和真实returncode；异常没有退出码时明确记null，不能伪造0。"""
    result = {"argv": list(argv), "returncode": None, "exception": None}
    out = err = b""
    try:
        completed = runner(argv, capture_output=True, check=False, timeout=10)
        require(type(completed.returncode) is int and isinstance(completed.stdout, bytes)
                and isinstance(completed.stderr, bytes), "进程接口没有返回完整原始字节和整数状态")
        result["returncode"], out, err = completed.returncode, completed.stdout, completed.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        result["exception"] = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, subprocess.TimeoutExpired):
            out, err = error.stdout or b"", error.stderr or b""
    for name, raw in (("stdout", out), ("stderr", err)):
        result[name] = raw.decode("utf-8", errors="surrogateescape")
        result[name + "_base64"] = base64.b64encode(raw).decode("ascii")
    return result


def validate_query(record, expected_argv):
    require(set(record) == {"argv", "returncode", "exception", "stdout", "stderr", "stdout_base64", "stderr_base64"}
            and record["argv"] == expected_argv, "观测命令或查询记录字段不同")
    for name in ("stdout", "stderr"):
        require(isinstance(record[name], str) and isinstance(record[name + "_base64"], str), "缺原始进程输出")
        require(base64.b64decode(record[name + "_base64"], validate=True)
                == record[name].encode("utf-8", errors="surrogateescape"), "原始输出与可读文本不一致")
    require(record["exception"] is None and type(record["returncode"]) is int and record["returncode"] == 0,
            "tmux查询失败或没有真实退出码")


def parse_pane(stdout):
    require(isinstance(stdout, str) and stdout.endswith(END + "\n"), "tmux观测缺完整记录终止符")
    records = stdout.split(END + "\n")
    require(records[-1] == "" and len(records) == 2, "目标pane缺失、重复或返回多条记录")
    fields = records[0].split(SEP)
    require(len(fields) == len(FIELDS), "tmux观测字段缺失或被分隔符污染")
    return dict(zip(FIELDS, fields, strict=True))


def classify_pane(parsed, identity, observed_utc):
    for key in ("session", "window_id", "pane_id", "pane_start_command"):
        require(parsed[key] == identity[key], "实际pane身份或原始start_command不同: " + key)
    require(parsed["pane_pid"] == str(identity["pane_pid"]) and parsed["remain_on_exit"] == "on",
            "实际PID或本窗口保留状态不同")
    require(parsed["pane_dead"] in {"0", "1"}, "pane_dead不是完整0/1字符串")
    if parsed["pane_dead"] == "0":
        require(parsed["pane_dead_status"] == parsed["pane_dead_signal"] == parsed["pane_dead_time"] == "",
                "alive观测混入退出状态或信号")
        return "PENDING"
    require(parsed["pane_dead_status"] == "0" and parsed["pane_dead_signal"] == "", "pane非零、信号退出或缺明确0字符串")
    require(re.fullmatch(r"[1-9][0-9]*", parsed["pane_dead_time"]), "缺原生pane退出时刻")
    require(int(utc_time(identity["recorded_utc"]).timestamp()) <= int(parsed["pane_dead_time"])
            <= int(utc_time(observed_utc).timestamp()), "原生退出时刻不在identity与观测之间")
    return "PASS"


def tool_reference():
    return file_reference(Path(__file__).resolve())


def probe_identity(identity_path, identity_sha256, *, repo=REPO_ROOT, runner=subprocess.run):
    """只读当前pane；PASS是当前观测，不代替持久receipt及外部capture进程返回码。"""
    identity = read_identity(identity_path, identity_sha256, repo=repo)
    identity_ref = file_reference(Path(identity_path))
    version = query([TMUX, "-V"], runner)
    pane = query(pane_argv(identity), runner)
    observed = utc_now()
    result = {"schema": SCHEMA, "status": "FAIL", "reason": "", "identity": identity_ref,
              "identity_snapshot": identity, "command_file": identity["command_file"], "tool": tool_reference(),
              "observed_utc": observed, "tmux_version": None, "version_query": version, "pane_query": pane,
              "parsed": None}
    try:
        require(read_identity(identity_path, identity_sha256, repo=repo) == identity, "观测期间identity改变")
        validate_query(version, [TMUX, "-V"])
        require(re.fullmatch(r"tmux [0-9][A-Za-z0-9.+_-]*\n", version["stdout"]), "tmux版本证据缺失或格式不同")
        result["tmux_version"] = version["stdout"].removesuffix("\n")
        validate_query(pane, pane_argv(identity))
        result["parsed"] = parse_pane(pane["stdout"])
        result["status"] = classify_pane(result["parsed"], identity, observed)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["reason"] = str(error)
    return result


def write_exclusive(path, value):
    """只创建新文件；flush/close失败交由外部真实进程返回码拒绝，不能只看receipt文本。"""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def capture(identity_path, identity_sha256, out_path, *, repo=REPO_ROOT, runner=subprocess.run):
    output = entity_path(out_path, repo, exists=False)
    result = probe_identity(identity_path, identity_sha256, repo=repo, runner=runner)
    if result["status"] != "PENDING":
        write_exclusive(output, result)
    return result


def validate_receipt(receipt_path, *, identity_path, identity_sha256, repo=REPO_ROOT):
    """完全只读，不调用tmux；重算原始观测判据，不仅信自报status。"""
    path = entity_path(receipt_path, repo)
    before = file_reference(path)
    receipt = read_json(path)
    identity = read_identity(identity_path, identity_sha256, repo=repo)
    require(set(receipt) == {"schema", "status", "reason", "identity", "identity_snapshot", "command_file", "tool",
                             "observed_utc", "tmux_version", "version_query", "pane_query", "parsed"}
            and type(receipt["schema"]) is int and receipt["schema"] == SCHEMA, "receipt schema错误")
    require(receipt["identity"] == file_reference(Path(identity_path)) and receipt["identity_snapshot"] == identity
            and receipt["command_file"] == identity["command_file"], "receipt未绑定指定identity及命令")
    require(receipt["tool"] == tool_reference(), "退出量具路径或字节与capture时不同")
    utc_time(receipt["observed_utc"])
    validate_query(receipt["version_query"], [TMUX, "-V"])
    require(re.fullmatch(r"tmux [0-9][A-Za-z0-9.+_-]*\n", receipt["version_query"]["stdout"])
            and receipt["tmux_version"] == receipt["version_query"]["stdout"].removesuffix("\n"), "tmux版本字段不符")
    validate_query(receipt["pane_query"], pane_argv(identity))
    parsed = parse_pane(receipt["pane_query"]["stdout"])
    require(receipt["parsed"] == parsed, "持久解析字段与原始tmux stdout不同")
    require(classify_pane(parsed, identity, receipt["observed_utc"]) == "PASS"
            and receipt["status"] == "PASS" and receipt["reason"] == "", "缺真实成功退出")
    require(file_reference(path) == before, "校验期间receipt改变")
    return receipt


GATE_SCRIPT = '"$1" wait-for "$2" || exit 2\nexec bash "$3" "$4"'


def launch(session, train_head, label, command_file, command_sha256, identity_out,
           *, repo=REPO_ROOT, runner=subprocess.run):
    """先在本次新window保留退出，再落身份文件，最后释放唯一门闩；失败不自动清理。"""
    repo = Path(repo)
    output = entity_path(identity_out, repo, exists=False)
    command_path = entity_path(command_file, repo)
    command = file_reference(command_path)
    require(isinstance(command_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", command_sha256)
            and command["sha256"] == command_sha256, "原命令文件SHA不符")
    require(isinstance(session, str) and session and not any(char in session for char in (SEP, END, "\x00", "\n", "\r", "\t", ":")),
            "session名称非法")
    require(isinstance(label, str) and label and not any(char in label for char in (SEP, END, "\x00")), "label非法")
    require(isinstance(train_head, str) and re.fullmatch(r"[0-9a-f]{40}", train_head), "缺共同TRAIN_HEAD")
    gate = "orig80k-exit-guard-" + uuid.uuid4().hex
    result = {"schema": SCHEMA, "status": "FAIL", "reason": "", "session": session, "label": label,
              "train_head": train_head, "command_file": command, "identity_out": str(output),
              "gate_channel": gate, "created": False, "released": False, "creation_uncertain": False,
              "release_attempted": False, "release_uncertain": False,
              "tool": tool_reference(), "started_utc": utc_now(), "queries": []}

    def execute(name, argv):
        record = query(argv, runner)
        result["queries"].append({"phase": name, "record": record})
        return record

    try:
        version = execute("version", [TMUX, "-V"])
        validate_query(version, [TMUX, "-V"])
        require(re.fullmatch(r"tmux [0-9][A-Za-z0-9.+_-]*\n", version["stdout"]), "缺tmux版本")
        result["tmux_version"] = version["stdout"].removesuffix("\n")
        existing = execute("session_absence", [TMUX, "has-session", "-t", "=" + session])
        require(existing["exception"] is None and type(existing["returncode"]) is int
                and existing["returncode"] in (0, 1), "无法核对session是否存在")
        require(existing["returncode"] == 1, "session已存在，拒绝复用")
        # new-session不传-A等复用选项；名字在预检后被占用时也只能失败。
        new_argv = [TMUX, "new-session", "-d", "-s", session, "-c", str(repo),
                    "/bin/sh", "-c", GATE_SCRIPT, "tmux-exit-guard", TMUX, gate,
                    str(command_path), command_sha256]
        created = execute("create_gated_session", new_argv)
        result["created"] = created["returncode"] == 0 and created["exception"] is None
        result["creation_uncertain"] = not result["created"]
        validate_query(created, new_argv)
        discovery_argv = pane_argv({"session": session}, filtered=False)
        first_query = execute("discover_owned_pane", discovery_argv)
        validate_query(first_query, discovery_argv)
        first = parse_pane(first_query["stdout"])
        require(first["session"] == session and re.fullmatch(r"@[0-9]+", first["window_id"])
                and re.fullmatch(r"%[0-9]+", first["pane_id"])
                and re.fullmatch(r"[1-9][0-9]*", first["pane_pid"]) and int(first["pane_pid"]) > 1,
                "新session没有唯一可绑定的原生window/pane/PID")
        require(first["pane_dead"] == "0" and first["pane_dead_status"] == first["pane_dead_signal"]
                == first["pane_dead_time"] == "", "新门闩进程并非alive，禁止释放")
        result["owned_pane"] = first
        set_argv = [TMUX, "set-option", "-w", "-t", first["window_id"], "remain-on-exit", "on"]
        configured = execute("retain_owned_window", set_argv)
        validate_query(configured, set_argv)
        identity = {"schema": SCHEMA, "session": session, "window_id": first["window_id"],
                    "pane_id": first["pane_id"], "pane_pid": int(first["pane_pid"]), "train_head": train_head,
                    "label": label, "command_file": command, "pane_start_command": first["pane_start_command"],
                    "initial_pane_dead": 0, "remain_on_exit": "on", "recorded_utc": utc_now(),
                    "gate_channel": gate, "gate_exec_argv": ["bash", str(command_path), command_sha256],
                    "enabled_mid_run": False, "tmux_version": result["tmux_version"]}
        after_argv = pane_argv(identity)
        after_query = execute("confirm_retention", after_argv)
        validate_query(after_query, after_argv)
        after = parse_pane(after_query["stdout"])
        require(classify_pane(after, identity, utc_now()) == "PENDING", "门闩释放前必须仍alive且保留on")
        validate_identity(identity, repo=repo)
        write_exclusive(output, identity)
        identity_ref = file_reference(output)
        require(read_identity(output, identity_ref["sha256"], repo=repo) == identity, "身份持久化核对失败")
        result["identity"] = identity_ref
        last = execute("confirm_before_release", after_argv)
        validate_query(last, after_argv)
        require(parse_pane(last["stdout"]) == after, "落身份文件后pane发生变化，保持门闩不释放")
        check_reference(command, repo)
        release_argv = [TMUX, "wait-for", "-S", gate]
        result["release_attempted"] = True
        released = execute("release_owned_gate", release_argv)
        validate_query(released, release_argv)
        result["released"] = True
        result["status"] = "RELEASED"
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["reason"] = str(error)
        if result["release_attempted"] and not result["released"]:
            result["released"] = None
            result["release_uncertain"] = True
    result["finished_utc"] = utc_now()
    return result


def main(argv=None, *, runner=subprocess.run, repo=REPO_ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    take = sub.add_parser("capture")
    take.add_argument("--identity", required=True)
    take.add_argument("--identity-sha256", required=True)
    take.add_argument("--out", required=True)
    start = sub.add_parser("launch")
    for name in ("session", "train-head", "label", "command-file", "command-sha256", "identity-out"):
        start.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            result = capture(args.identity, args.identity_sha256, args.out, repo=repo, runner=runner)
        else:
            result = launch(args.session, args.train_head, args.label, args.command_file, args.command_sha256,
                            args.identity_out, repo=repo, runner=runner)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "FAIL", "reason": str(error)}, ensure_ascii=True), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False), flush=True)
    return {"PASS": 0, "FAIL": 1, "PENDING": PENDING_EXIT_CODE, "RELEASED": 0}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
