"""纯stdlib模拟后端与临时文件测试；不运行任何tmux进程或训练。"""

import base64
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import tmux_exit_guard as guard

NOW = "2026-09-26T22:00:00+00:00"
DEAD_TIME = str(int(guard.utc_time("2026-09-26T21:59:00+00:00").timestamp()))


def frame(identity, **changes):
    values = {"session": identity["session"], "window_id": identity["window_id"], "pane_id": identity["pane_id"],
              "pane_pid": str(identity["pane_pid"]), "pane_dead": "1", "pane_dead_status": "0",
              "pane_dead_signal": "", "pane_dead_time": DEAD_TIME,
              "pane_start_command": identity["pane_start_command"], "remain_on_exit": "on"}
    values.update(changes)
    return (guard.SEP.join(values[key] for key in guard.FIELDS) + guard.END + "\n").encode()


class ReadBackend:
    def __init__(self, stdout, *, pane_rc=0, version=b"tmux 3.6a\n", version_rc=0, stderr=b""):
        self.stdout, self.pane_rc, self.version, self.version_rc, self.stderr = stdout, pane_rc, version, version_rc, stderr
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if argv == [guard.TMUX, "-V"]:
            return subprocess.CompletedProcess(argv, self.version_rc, self.version, b"")
        assert argv[0:3] == [guard.TMUX, "list-panes", "-s"]
        return subprocess.CompletedProcess(argv, self.pane_rc, self.stdout, self.stderr)


class LaunchBackend:
    def __init__(self, fault=None):
        self.fault, self.calls = fault, []
        self.created = self.retained = self.released = False
        self.session = self.start_command = None
        self.pane_reads = 0

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        result = lambda rc=0, out=b"", err=b"": subprocess.CompletedProcess(argv, rc, out, err)
        operation = argv[1]
        if operation == "-V":
            return result(out=b"tmux 3.6a\n")
        if operation == "has-session":
            return result(0 if self.fault == "existing_session" else 1)
        if operation == "new-session":
            if self.fault == "create_race":
                return result(1, err=b"duplicate session\n")
            self.created = True
            self.session = argv[argv.index("-s") + 1]
            self.start_command = '"gate ' + " ".join(argv[-4:]) + ' "'
            self.original_argv = list(argv)
            return result()
        if operation == "list-panes":
            self.pane_reads += 1
            identity = {"session": self.session, "window_id": "@700", "pane_id": "%701", "pane_pid": 765432,
                        "pane_start_command": self.start_command}
            changes = {"pane_dead": "0", "pane_dead_status": "", "pane_dead_time": "",
                       "remain_on_exit": "on" if self.retained else "off"}
            if self.fault == "wrong_discovery":
                changes["session"] = "another_session"
            if self.fault == "dies_before_identity":
                changes.update(pane_dead="1", pane_dead_status="0", pane_dead_time=DEAD_TIME)
            if self.fault == "pane_changes" and self.pane_reads == 3:
                changes["pane_pid"] = "765433"
            return result(out=frame(identity, **changes))
        if operation == "set-option":
            assert argv == [guard.TMUX, "set-option", "-w", "-t", "@700", "remain-on-exit", "on"]
            if self.fault == "option_failure":
                return result(1, err=b"fixture option failure\n")
            if self.fault != "option_no_effect":
                self.retained = True
            return result()
        if operation == "wait-for":
            assert argv[2] == "-S" and self.retained
            if self.fault == "release_failure":
                return result(1, err=b"fixture missing acknowledgment\n")
            self.released = True
            return result()
        raise AssertionError("未登记模拟命令: " + repr(argv))


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tmux-exit-guard-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repo"
        self.store = self.repo / "v1-store"
        self.store.mkdir(parents=True)
        self.command = self.store / "command with space.command"
        self.command.write_bytes(b"# synthetic command, never executed\nexit 0\n")
        self.identity = {"schema": 1, "session": "orig80k-unit", "window_id": "@552", "pane_id": "%552",
                         "pane_pid": 704720, "train_head": "a" * 40, "label": "unit-label",
                         "command_file": guard.file_reference(self.command),
                         "pane_start_command": '\"bash /synthetic/command abc \"',
                         "initial_pane_dead": 0, "remain_on_exit": "on",
                         "recorded_utc": "2026-09-26T21:21:39.480484+00:00",
                         "enabled_mid_run": True, "scope": "模拟输入，不是旧P1退出证据"}
        self.identity_path = self.store / "unit.identity.json"
        self.identity_path.write_text(json.dumps(self.identity))
        self.identity_sha = guard.file_reference(self.identity_path)["sha256"]
        self.output = self.store / "unit.exit.json"
        self.clock = mock.patch.object(guard, "utc_now", return_value=NOW)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def take(self, backend):
        return guard.capture(self.identity_path, self.identity_sha, self.output, repo=self.repo, runner=backend)

    def check_receipt(self):
        return guard.validate_receipt(self.output, identity_path=self.identity_path,
                                      identity_sha256=self.identity_sha, repo=self.repo)

    def launch(self, backend):
        return guard.launch("orig80k-new", "a" * 40, "new-label", self.command,
                            guard.file_reference(self.command)["sha256"], self.store / "new.identity.json",
                            repo=self.repo, runner=backend)

    def test_capture_pass_and_validator_is_read_only(self):
        backend = ReadBackend(frame(self.identity), stderr=b"diagnostic text preserved\n")
        result = self.take(backend)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["parsed"]["pane_start_command"], self.identity["pane_start_command"])
        self.assertEqual(result["pane_query"]["stderr"], "diagnostic text preserved\n")
        before = self.output.read_bytes()
        with mock.patch.object(guard.subprocess, "run", side_effect=AssertionError("validator queried tmux")):
            self.assertEqual(self.check_receipt(), result)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[1][0], guard.pane_argv(self.identity))

    def test_pending_has_no_output_and_later_capture_can_succeed(self):
        alive = frame(self.identity, pane_dead="0", pane_dead_status="", pane_dead_time="")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = guard.main(["capture", "--identity", str(self.identity_path), "--identity-sha256", self.identity_sha,
                               "--out", str(self.output)], repo=self.repo, runner=ReadBackend(alive))
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(printed.getvalue())["status"], "PENDING")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.take(ReadBackend(frame(self.identity)))["status"], "PASS")

    def test_complete_receipt_cannot_replace_external_capture_exit_code(self):
        # 故障发生在完整JSON写出后；该文件可读不等于capture进程成功。
        with mock.patch.object(guard.os, "fsync", side_effect=OSError("fixture fsync failure")), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            code = guard.main(["capture", "--identity", str(self.identity_path), "--identity-sha256", self.identity_sha,
                               "--out", str(self.output)], repo=self.repo,
                              runner=ReadBackend(frame(self.identity)))
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(printed.getvalue())["status"], "FAIL")
        self.assertEqual(guard.read_json(self.output)["status"], "PASS")
        self.assertEqual(self.check_receipt()["status"], "PASS")

    def test_readonly_probe_does_not_reserve_output(self):
        result = guard.probe_identity(self.identity_path, self.identity_sha, repo=self.repo,
                                      runner=ReadBackend(frame(self.identity)))
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(self.output.exists())

    def test_nonzero_signal_missing_or_zero_like_status_rejected(self):
        for changes in ({"pane_dead_status": "1"}, {"pane_dead_status": ""}, {"pane_dead_status": "00"},
                        {"pane_dead_status": " 0"}, {"pane_dead_signal": "15"}, {"pane_dead_signal": "0"},
                        {"pane_dead": ""}, {"pane_dead": "2"}, {"pane_dead_time": ""},
                        {"pane_dead_time": "1"}, {"pane_dead_time": "9999999999"}):
            with self.subTest(changes=changes):
                result = guard.probe_identity(self.identity_path, self.identity_sha, repo=self.repo,
                                              runner=ReadBackend(frame(self.identity, **changes)))
                self.assertEqual(result["status"], "FAIL")

    def test_all_identity_fields_and_original_command_must_match(self):
        for changes in ({"session": "other"}, {"window_id": "@553"}, {"pane_id": "%553"}, {"pane_pid": "704721"},
                        {"pane_start_command": self.identity["pane_start_command"].strip('"')},
                        {"remain_on_exit": "off"}):
            with self.subTest(changes=changes):
                result = guard.probe_identity(self.identity_path, self.identity_sha, repo=self.repo,
                                              runner=ReadBackend(frame(self.identity, **changes)))
                self.assertEqual(result["status"], "FAIL")

    def test_missing_duplicate_or_malformed_rows_fail(self):
        for raw in (b"", frame(self.identity) * 2, b"zero\n", frame(self.identity).replace(guard.SEP.encode(), b"\t")):
            with self.subTest(raw=raw[:40]):
                result = guard.probe_identity(self.identity_path, self.identity_sha, repo=self.repo, runner=ReadBackend(raw))
                self.assertEqual(result["status"], "FAIL")

    def test_query_failure_and_version_missing_are_preserved(self):
        result = self.take(ReadBackend(b"", pane_rc=1, stderr=b"can't find pane\n"))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(guard.read_json(self.output)["pane_query"]["returncode"], 1)
        with self.assertRaises(ValueError):
            self.check_receipt()
        missing = guard.probe_identity(self.identity_path, self.identity_sha, repo=self.repo,
                                       runner=ReadBackend(frame(self.identity), version=b""))
        self.assertEqual(missing["status"], "FAIL")

    def test_backend_exception_never_becomes_zero(self):
        def missing(*args, **kwargs):
            raise FileNotFoundError("synthetic tmux missing")
        result = self.take(missing)
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNone(result["pane_query"]["returncode"])
        self.assertEqual(result["pane_query"]["exception"]["type"], "FileNotFoundError")

    def test_identity_sha_command_bytes_and_existing_output_fail_before_query(self):
        backend = mock.Mock(side_effect=AssertionError("unexpected query"))
        with self.assertRaises(ValueError):
            guard.capture(self.identity_path, "0" * 64, self.output, repo=self.repo, runner=backend)
        self.command.write_text("changed")
        with self.assertRaises(ValueError):
            self.take(backend)
        self.output.write_text("retain existing")
        with self.assertRaises(ValueError):
            self.take(backend)
        self.assertEqual(self.output.read_text(), "retain existing")
        self.assertFalse(backend.called)

    def test_symlink_outside_and_initial_dead_identities_rejected(self):
        linked = self.store / "identity-link.json"
        linked.symlink_to(self.identity_path)
        with self.assertRaises(ValueError):
            guard.read_identity(linked, self.identity_sha, repo=self.repo)
        with self.assertRaises(ValueError):
            guard.capture(self.identity_path, self.identity_sha, self.repo / "outside.json", repo=self.repo,
                          runner=mock.Mock(side_effect=AssertionError("unexpected query")))
        for field, value in (("initial_pane_dead", 1), ("pane_pid", True), ("remain_on_exit", "failed")):
            broken = copy.deepcopy(self.identity)
            broken[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                guard.validate_identity(broken, repo=self.repo)

    def test_tampered_self_report_status_parsed_rc_version_or_tool_is_rejected(self):
        original = self.take(ReadBackend(frame(self.identity)))
        for fault in ("status", "parsed", "rc", "raw", "version", "tool", "identity", "base64"):
            altered = copy.deepcopy(original)
            if fault == "status": altered["status"] = "PENDING"
            elif fault == "parsed": altered["parsed"]["pane_dead_status"] = "1"
            elif fault == "rc": altered["pane_query"]["returncode"] = 1
            elif fault == "raw":
                raw = frame(self.identity, pane_dead_status="1")
                altered["pane_query"]["stdout"] = raw.decode()
                altered["pane_query"]["stdout_base64"] = base64.b64encode(raw).decode()
                altered["parsed"]["pane_dead_status"] = "1"
            elif fault == "version": altered["tmux_version"] = "tmux fake"
            elif fault == "tool": altered["tool"]["sha256"] = "0" * 64
            elif fault == "identity": altered["identity_snapshot"]["pane_pid"] += 1
            else: altered["pane_query"]["stdout_base64"] = "AA=="
            self.output.write_text(json.dumps(altered))
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.check_receipt()

    def test_raw_start_command_newline_and_quotes_are_not_normalized(self):
        self.identity["pane_start_command"] = '"bash -c \'line1\nline2\' "'
        self.identity_path.write_text(json.dumps(self.identity))
        self.identity_sha = guard.file_reference(self.identity_path)["sha256"]
        result = self.take(ReadBackend(frame(self.identity)))
        self.assertEqual(result["status"], "PASS")
        self.check_receipt()

    def test_launch_retains_and_persists_identity_before_release_without_command_change(self):
        backend = LaunchBackend()
        before = self.command.read_bytes()
        result = self.launch(backend)
        self.assertEqual(result["status"], "RELEASED")
        self.assertTrue(backend.created and backend.retained and backend.released)
        identity = guard.read_identity(result["identity"]["path"], result["identity"]["sha256"], repo=self.repo)
        self.assertEqual(identity["gate_exec_argv"], ["bash", str(self.command), guard.file_reference(self.command)["sha256"]])
        self.assertEqual(identity["initial_pane_dead"], 0)
        self.assertEqual(identity["remain_on_exit"], "on")
        self.assertEqual(self.command.read_bytes(), before)
        self.assertEqual(backend.original_argv[-2:], [str(self.command), identity["command_file"]["sha256"]])
        self.assertEqual(backend.calls[-1], [guard.TMUX, "wait-for", "-S", result["gate_channel"]])
        self.assertFalse(any("kill-session" in argv or "kill-server" in argv or "-g" in argv or "-A" in argv for argv in backend.calls))

    def test_launch_existing_session_or_atomic_creation_race_never_sets_options(self):
        for fault in ("existing_session", "create_race"):
            backend = LaunchBackend(fault)
            result = self.launch(backend)
            with self.subTest(fault=fault):
                self.assertEqual(result["status"], "FAIL")
                self.assertFalse(backend.released)
                self.assertFalse(any(argv[1] == "set-option" for argv in backend.calls))
                self.assertFalse((self.store / "new.identity.json").exists())

    def test_launch_option_or_identity_failure_keeps_owned_gate_unreleased(self):
        for fault in ("wrong_discovery", "dies_before_identity", "option_failure", "option_no_effect", "pane_changes"):
            backend = LaunchBackend(fault)
            path = self.store / "new.identity.json"
            result = self.launch(backend)
            with self.subTest(fault=fault):
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(result["created"])
                self.assertFalse(backend.released)
                self.assertFalse(any(argv[1] in {"wait-for", "kill-session", "kill-server"} for argv in backend.calls))
            if path.exists(): path.unlink()  # 仅清本case的临时夹具，生产候选无清理操作。
        backend = LaunchBackend()
        with mock.patch.object(guard, "write_exclusive", side_effect=OSError("fixture identity write failure")):
            result = self.launch(backend)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(backend.retained)
        self.assertFalse(backend.released)

    def test_launch_release_acknowledgment_failure_is_uncertain_not_held_claim(self):
        result = self.launch(LaunchBackend("release_failure"))
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["release_attempted"] and result["release_uncertain"])
        self.assertIsNone(result["released"])
        self.assertTrue((self.store / "new.identity.json").is_file())

    def test_launch_identity_collision_and_command_sha_failure_have_no_backend_calls(self):
        path = self.store / "new.identity.json"
        path.write_text("preserve")
        backend = mock.Mock(side_effect=AssertionError("unexpected query"))
        with self.assertRaises(ValueError): self.launch(backend)
        self.assertEqual(path.read_text(), "preserve")
        with self.assertRaises(ValueError):
            guard.launch("new", "a" * 40, "label", self.command, "0" * 64, self.store / "unused.json",
                         repo=self.repo, runner=backend)
        self.assertFalse(backend.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
