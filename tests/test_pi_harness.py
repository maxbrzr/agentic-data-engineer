import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from agentic_data_engineer.agent.pi import PiHarness, PiSettings
from agentic_data_engineer.contracts import (
    AgentRequest,
    DatasetSpec,
    ModelConfig,
    RetrievedDataset,
)


class PiHarnessTests(unittest.TestCase):
    def _request(self, root: Path) -> AgentRequest:
        data_dir = root / "data" / "chemical-process-safety"
        output_dir = root / "output" / "chemical-process-safety" / "runs" / "one"
        data_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        launcher = root / "scripts" / "pi"
        launcher.parent.mkdir()
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        return AgentRequest(
            dataset=RetrievedDataset(
                spec=DatasetSpec(
                    key="chemical-process-safety",
                    title="Chemical process safety",
                    url="https://example.invalid",
                ),
                data_dir=data_dir,
                metadata_title="Chemical process safety",
            ),
            output_dir=output_dir,
            system_prompt="System instructions",
            task_prompt="Prepare this table",
        )

    def test_runs_container_and_normalizes_json_events(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            request = self._request(root)

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                request_document = json.loads(
                    Path(command[-1]).read_text(encoding="utf-8")
                )
                self.assertEqual("provider-a", request_document["provider_id"])
                self.assertEqual("model-a", request_document["model_id"])
                self.assertFalse(request_document["continue_session"])
                for name in ("parser.py", "train.csv", "test.csv"):
                    (request.output_dir / name).write_text("ok\n", encoding="utf-8")
                session_dir = Path(request_document["session_dir"])
                session_dir.mkdir(parents=True, exist_ok=True)
                (session_dir / "session.jsonl").write_text(
                    "\n".join(
                        json.dumps(entry)
                        for entry in (
                            {"type": "session", "id": "pi-session"},
                            {
                                "type": "message",
                                "message": {
                                    "role": "user",
                                    "content": [{"type": "text", "text": "Prepare"}],
                                },
                            },
                            {
                                "type": "message",
                                "message": {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "text", "text": "Listing files."},
                                        {
                                            "type": "toolCall",
                                            "id": "call-1",
                                            "name": "bash",
                                            "arguments": {"command": "ls data"},
                                        },
                                    ],
                                },
                            },
                            {
                                "type": "message",
                                "message": {
                                    "role": "toolResult",
                                    "toolCallId": "call-1",
                                    "content": [{"type": "text", "text": "a.csv"}],
                                    "isError": False,
                                },
                            },
                        )
                    ),
                    encoding="utf-8",
                )
                events = "\n".join(
                    (
                        json.dumps({"type": "session", "id": "pi-session"}),
                        json.dumps(
                            {
                                "type": "message_end",
                                "message": {
                                    "id": "message-1",
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": "Completed."}],
                                },
                            }
                        ),
                    )
                )
                return SimpleNamespace(returncode=0, stdout=events, stderr="")

            harness = PiHarness(
                PiSettings(project_root=root),
                command_runner=runner,
            )
            result = harness.run(request, ModelConfig("provider-a", "model-a"))

            self.assertEqual("pi", result.harness)
            self.assertEqual("pi-session", result.run_id)
            self.assertEqual("message-1", result.message_id)
            self.assertEqual(1, len(calls))
            self.assertEqual(
                [
                    str(root / "scripts" / "pi"),
                    "run",
                    "chemical-process-safety",
                    "chemical-process-safety",
                ],
                calls[0][0][:-1],
            )
            self.assertEqual(root, calls[0][1]["cwd"])
            transcript_path = Path(result.metadata["report"])
            self.assertEqual("agent_transcript.md", transcript_path.name)
            transcript = transcript_path.read_text(encoding="utf-8")
            self.assertIn("# Pi Session Transcript", transcript)
            self.assertIn("### 1. user", transcript)
            self.assertIn("Listing files.", transcript)
            self.assertIn("- ✓ `bash`: ls data (completed)", transcript)
            self.assertIn("a.csv", transcript)
            self.assertEqual("agent.log", Path(result.metadata["run_log"]).name)
            self.assertTrue(Path(result.metadata["run_log"]).is_file())
            self.assertFalse(list(request.output_dir.glob(".pi-request-*.json")))

    def test_continues_until_required_outputs_exist(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            request = self._request(root)

            def runner(command, **_kwargs):
                document = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                calls.append(document)
                names = ("parser.py",) if len(calls) == 1 else ("train.csv", "test.csv")
                for name in names:
                    (request.output_dir / name).write_text("ok\n", encoding="utf-8")
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"type": "session", "id": "same-session"}),
                    stderr="",
                )

            harness = PiHarness(
                PiSettings(project_root=root, max_continuations=1),
                command_runner=runner,
            )
            harness.run(request, ModelConfig("provider-a", "model-a"))

            self.assertEqual(2, len(calls))
            self.assertFalse(calls[0]["continue_session"])
            self.assertTrue(calls[1]["continue_session"])
            self.assertIn("train.csv", calls[1]["task_prompt"])

    def test_fails_after_finite_continuation_limit_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            request = self._request(root)

            def runner(*_args, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"type": "session", "id": "stuck"}),
                    stderr="",
                )

            harness = PiHarness(
                PiSettings(project_root=root, max_continuations=1),
                command_runner=runner,
            )
            with self.assertRaisesRegex(ValueError, "exhausted 1 continuation"):
                harness.run(request, ModelConfig("provider-a", "model-a"))

            self.assertTrue((request.output_dir / "agent.log").is_file())

    def test_gwdg_requires_key_in_ignored_dotenv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            request = self._request(root)
            harness = PiHarness(PiSettings(project_root=root))

            with self.assertRaisesRegex(RuntimeError, "SAIA_API_KEY"):
                harness.run(request, ModelConfig("gwdg", "model-a"))

    def test_kit_requires_key_in_ignored_dotenv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            request = self._request(root)
            harness = PiHarness(PiSettings(project_root=root))

            with self.assertRaisesRegex(RuntimeError, "KIT_TOOLBOX_API_KEY"):
                harness.run(
                    request,
                    ModelConfig("kit", "kit.glm-5.3-flash"),
                )

    def test_retries_transient_provider_error_event(self):
        attempts = []
        sleeps = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            request = self._request(root)

            def runner(*_args, **_kwargs):
                attempts.append(True)
                if len(attempts) == 1:
                    event = {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "stopReason": "error",
                            "errorMessage": "No provider available",
                        },
                    }
                    return SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps(event),
                        stderr="",
                    )
                for name in ("parser.py", "train.csv", "test.csv"):
                    (request.output_dir / name).write_text("ok\n", encoding="utf-8")
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"type": "session", "id": "retried"}),
                    stderr="",
                )

            harness = PiHarness(
                PiSettings(
                    project_root=root,
                    max_provider_retries=1,
                    retry_backoff_seconds=0.25,
                ),
                command_runner=runner,
                sleeper=sleeps.append,
            )
            result = harness.run(request, ModelConfig("provider-a", "model-a"))

            self.assertEqual("retried", result.run_id)
            self.assertEqual(2, len(attempts))
            self.assertEqual([0.25], sleeps)

    def test_default_runner_streams_concise_json_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            harness = PiHarness(PiSettings(project_root=root))
            events = (
                {
                    "type": "harness_status",
                    "status": "starting_pi_process",
                    "provider": "gwdg",
                    "model": "model-a",
                },
                {"type": "session", "id": "visible-session"},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {
                        "type": "text_start",
                    },
                },
                {
                    "type": "message_update",
                    "assistantMessageEvent": {
                        "type": "text_delta",
                        "delta": "Visible answer",
                        "partial": {"large": "redundant"},
                    },
                },
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_end"},
                },
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "toolcall_start"},
                },
                {
                    "type": "message_update",
                    "assistantMessageEvent": {
                        "type": "toolcall_delta",
                        "delta": '{"name":"bash","arguments":',
                    },
                },
                {
                    "type": "message_update",
                    "assistantMessageEvent": {
                        "type": "toolcall_delta",
                        "delta": '{"command":"python inspect.py"}}',
                    },
                },
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "toolcall_end"},
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Visible answer"}],
                    },
                },
                {
                    "type": "tool_execution_start",
                    "toolName": "bash",
                    "args": {"command": "python inspect.py"},
                },
            )
            script = ";".join(
                f"print({json.dumps(event)!r}, flush=True)" for event in events
            )
            live_log = root / "agent.log"
            terminal = io.StringIO()

            with redirect_stdout(terminal):
                completed = harness._run_streaming(
                    [sys.executable, "-c", script],
                    live_log,
                )

            self.assertEqual(0, completed.returncode)
            self.assertIn('"visible-session"', completed.stdout)
            self.assertIn("[pi] session visible-session started", terminal.getvalue())
            self.assertIn("[pi] process starting: gwdg/model-a", terminal.getvalue())
            self.assertIn("[pi:text] Visible answer", terminal.getvalue())
            self.assertIn(
                '[pi:toolcall] {"name":"bash","arguments":'
                '{"command":"python inspect.py"}}',
                terminal.getvalue(),
            )
            self.assertIn(
                '[pi] tool: bash {"command": "python inspect.py"}',
                terminal.getvalue(),
            )
            retained = live_log.read_text(encoding="utf-8")
            self.assertIn('"message_end"', retained)
            self.assertNotIn('"message_update"', retained)
            self.assertNotIn("redundant", completed.stdout)

    def test_default_runner_terminates_silent_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            harness = PiHarness(
                PiSettings(
                    project_root=root,
                    timeout_seconds=2,
                    stall_timeout_seconds=0.1,
                )
            )

            with self.assertRaises(subprocess.TimeoutExpired):
                harness._run_streaming(
                    [sys.executable, "-c", "import time; time.sleep(5)"]
                )

    def test_startup_build_silence_does_not_trigger_model_stall_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            harness = PiHarness(
                PiSettings(
                    project_root=root,
                    timeout_seconds=2,
                    stall_timeout_seconds=0.1,
                )
            )
            harness_status = json.dumps(
                {
                    "type": "harness_status",
                    "provider": "kit",
                    "model": "kit.glm-5.3-flash",
                }
            )
            script = (
                "import time; "
                "print('building image', flush=True); "
                "time.sleep(0.2); "
                f"print({harness_status!r}, flush=True)"
            )

            completed = harness._run_streaming(
                [sys.executable, "-c", script]
            )

            self.assertEqual(0, completed.returncode)


if __name__ == "__main__":
    unittest.main()
