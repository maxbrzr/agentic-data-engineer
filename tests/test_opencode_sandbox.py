import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from agentic_data_engineer.agent.opencode_sandbox import (
        OpencodeSandboxManager,
        read_sandbox_attestation,
    )
except ImportError:
    OpencodeSandboxManager = None


@unittest.skipIf(
    OpencodeSandboxManager is None,
    "OpenCode optional dependency is not installed",
)
class OpencodeSandboxManagerTests(unittest.TestCase):
    def test_attestation_proves_container_write_reached_host_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir).resolve()
            probe = output_dir / ".opencode-sandbox-live-probe"
            probe.write_text("matching-token", encoding="utf-8")
            response = io.BytesIO(
                json.dumps(
                    {
                        "output_dir": str(output_dir),
                        "probe_name": probe.name,
                        "probe_token": "matching-token",
                    }
                ).encode("utf-8")
            )

            with patch(
                "agentic_data_engineer.agent.opencode_sandbox.urlopen",
                return_value=response,
            ):
                marker = read_sandbox_attestation(
                    "http://127.0.0.1:54322/marker",
                    output_dir,
                )

            self.assertEqual(str(output_dir), marker["output_dir"])
            self.assertFalse(probe.exists())

    def test_attestation_rejects_probe_missing_from_host_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir).resolve()
            response = io.BytesIO(
                json.dumps(
                    {
                        "output_dir": str(output_dir),
                        "probe_name": ".opencode-sandbox-live-probe",
                        "probe_token": "container-only-token",
                    }
                ).encode("utf-8")
            )

            with patch(
                "agentic_data_engineer.agent.opencode_sandbox.urlopen",
                return_value=response,
            ):
                with self.assertRaisesRegex(ValueError, "bind is stale"):
                    read_sandbox_attestation(
                        "http://127.0.0.1:54322/marker",
                        output_dir,
                    )

    def test_matching_sandbox_is_reused_without_starting_compose(self):
        runner_calls = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            marker = {
                "example_key": "chemical-process-safety",
                "data_root": str(root / "data" / "chemical-process-safety"),
                "output_dir": str(root / "output" / "chemical-process-safety"),
                "data_read_only": True,
                "output_writable": True,
            }

            manager = OpencodeSandboxManager(
                project_root=root,
                attestation_reader=lambda _url, _expected_output: marker,
                command_runner=lambda *args, **kwargs: runner_calls.append(
                    (args, kwargs)
                ),
            )

            manager.ensure("chemical-process-safety")

            self.assertEqual([], runner_calls)

    def test_missing_server_starts_matching_sandbox(self):
        runner_calls = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            marker = {
                "example_key": "chemical-process-safety",
                "data_root": str(root / "data" / "chemical-process-safety"),
                "output_dir": str(root / "output" / "chemical-process-safety"),
                "data_read_only": True,
                "output_writable": True,
            }
            attestation_attempts = []

            def initially_unavailable(_url, _expected_output):
                attestation_attempts.append(True)
                if len(attestation_attempts) == 1:
                    raise ConnectionError("server is not running")
                return marker

            launcher = root / "scripts" / "opencode-sandbox"
            launcher.parent.mkdir()
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            manager = OpencodeSandboxManager(
                project_root=root,
                attestation_reader=initially_unavailable,
                command_runner=lambda *args, **kwargs: runner_calls.append(
                    (args, kwargs)
                ),
            )

            manager.ensure("chemical-process-safety")

            self.assertEqual(1, len(runner_calls))
            args, kwargs = runner_calls[0]
            self.assertEqual(
                (
                    [
                        str(launcher),
                        "start",
                        "chemical-process-safety",
                    ],
                ),
                args,
            )
            self.assertEqual(root, kwargs["cwd"])
            self.assertIs(kwargs["check"], True)

    def test_start_failure_has_actionable_docker_error(self):
        def unavailable_attestation(_url, _expected_output):
            raise ConnectionError("server is not running")

        def failed_runner(*_args, **_kwargs):
            raise subprocess.CalledProcessError(1, "docker compose")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            launcher = root / "scripts" / "opencode-sandbox"
            launcher.parent.mkdir()
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            manager = OpencodeSandboxManager(
                project_root=root,
                attestation_reader=unavailable_attestation,
                command_runner=failed_runner,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r"Docker Desktop is running",
            ):
                manager.ensure("chemical-process-safety")

    def test_custom_server_requires_management_opt_out(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError,
                r"--no-manage-opencode-sandbox",
            ):
                OpencodeSandboxManager(
                    project_root=Path(temp_dir),
                    base_url="http://127.0.0.1:6000",
                )


if __name__ == "__main__":
    unittest.main()
