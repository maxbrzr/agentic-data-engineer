import contextlib
import io
import unittest
from unittest import mock

from agentic_data_engineer import cli


class FakePipeline:
    """Fails preparation for one example and records what ran."""

    instances = []

    def __init__(self, **kwargs):
        self.prepared = []
        self.runs = []
        FakePipeline.instances.append(self)

    def prepare(self, spec):
        self.prepared.append(spec.key)
        if spec.key == "machine-failure":
            raise ValueError("DCAT metadata is missing values required for Croissant generation: ['license'].")
        return f"retrieved-{spec.key}"

    def run_spec(self, spec, *, prepared):
        self.runs.append((spec.key, prepared))
        return mock.MagicMock()


class CliTests(unittest.TestCase):
    def setUp(self):
        FakePipeline.instances.clear()

    def _main(self, *args):
        sandbox = mock.MagicMock()
        stderr = io.StringIO()
        with (
            mock.patch.object(cli, "DataEngineeringPipeline", FakePipeline),
            mock.patch(
                "agentic_data_engineer.agent.opencode_sandbox.OpencodeSandboxManager",
                return_value=sandbox,
            ),
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = cli.main(["--harness", "opencode", "--provider", "kit", *args])
        return code, stderr.getvalue(), sandbox

    def test_failed_preflight_is_one_line_and_starts_no_sandbox(self):
        code, stderr, sandbox = self._main("--example", "machine-failure")

        self.assertEqual(1, code)
        self.assertIn("machine-failure: error: DCAT metadata is missing", stderr)
        self.assertNotIn("Traceback", stderr)
        sandbox.ensure.assert_not_called()
        self.assertEqual([], FakePipeline.instances[0].runs)

    def test_a_failing_example_does_not_stop_the_others(self):
        code, stderr, sandbox = self._main(
            "--example", "machine-failure", "--example", "edge-ai-industrial-robot-motion"
        )

        self.assertEqual(1, code)
        self.assertIn("failed: machine-failure", stderr)
        pipeline = FakePipeline.instances[0]
        self.assertEqual(
            [("edge-ai-industrial-robot-motion", "retrieved-edge-ai-industrial-robot-motion")],
            pipeline.runs,
        )
        # The sandbox starts only for the dataset that passed preparation.
        self.assertEqual(1, sandbox.ensure.call_count)
