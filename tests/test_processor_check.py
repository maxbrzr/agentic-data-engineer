import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_data_engineer.processor_check import DockerProcessorChecker

ROOT = Path(__file__).resolve().parents[1]

PROCESSOR = """
from pathlib import Path

def process(data_dir, output_dir):
    rows = (Path(data_dir) / "raw.csv").read_text()
    (Path(output_dir) / "train.csv").write_text(rows)
    (Path(output_dir) / "test.csv").write_text(rows.upper())
"""


class FakeContainer:
    """Simulates the container by writing processed/ files into the package."""

    def __init__(self, outputs=None, returncode=0):
        self.outputs = outputs if outputs is not None else {"train.csv": "a\n", "test.csv": "A\n"}
        self.returncode = returncode
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        processed = Path(kwargs["env"]["PACKAGE_DIR"]) / "processed"
        processed.mkdir(exist_ok=True)
        for name, text in self.outputs.items():
            (processed / name).write_text(text)
        return subprocess.CompletedProcess(command, self.returncode, "out", "boom")


class DockerProcessorCheckerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / "run" / "data"
        self.package.mkdir(parents=True)
        (self.package / "raw.csv").write_text("a\n")
        (self.package / "processor.py").write_text(PROCESSOR)
        self.expected = {}
        for name, text in {"train.csv": "a\n", "test.csv": "A\n"}.items():
            (self.root / "run" / name).write_text(text)
            self.expected[name] = self.root / "run" / name

    def test_runs_isolated_compose_service_and_reports_hashes(self):
        container = FakeContainer()
        result = DockerProcessorChecker(ROOT, command_runner=container).check(self.package, self.expected)

        [(command, kwargs)] = container.calls
        self.assertEqual(
            ["docker", "compose", "-f", "compose.processor.yml", "run", "--rm", "--build", "processor"],
            command,
        )
        self.assertEqual(str(self.package.resolve()), kwargs["env"]["PACKAGE_DIR"])
        self.assertEqual(["test.csv", "train.csv"], sorted(result.sha256))
        self.assertIn("pandas==3.0.5", result.runtime)
        # processed/ is removed so dcat-ap-hub does not skip processing later.
        self.assertFalse((self.package / "processed").exists())

    def test_rejects_outputs_that_differ(self):
        container = FakeContainer({"train.csv": "a\n", "test.csv": "different\n"})
        with self.assertRaisesRegex(ValueError, "does not reproduce the exported test.csv"):
            DockerProcessorChecker(ROOT, command_runner=container).check(self.package, self.expected)
        self.assertFalse((self.package / "processed").exists())

    def test_rejects_missing_outputs_and_failures(self):
        with self.assertRaisesRegex(ValueError, "did not write test.csv"):
            DockerProcessorChecker(ROOT, command_runner=FakeContainer({"train.csv": "a\n"})).check(
                self.package, self.expected
            )
        with self.assertRaisesRegex(ValueError, "failed when re-run in isolation.*exit 1"):
            DockerProcessorChecker(ROOT, command_runner=FakeContainer(returncode=1)).check(
                self.package, self.expected
            )


@unittest.skipUnless(os.environ.get("ADE_DOCKER_TESTS") == "1", "set ADE_DOCKER_TESTS=1 to run Docker")
class DockerProcessorCheckerIntegrationTests(DockerProcessorCheckerTests):
    """Builds and runs the real processor container."""

    def test_real_container_reproduces_and_detects_changes(self):
        checker = DockerProcessorChecker(ROOT)
        checker.check(self.package, self.expected)

        self.expected["test.csv"].write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            checker.check(self.package, self.expected)
