import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from .contracts import ProcessorCheckResult

PROCESSED_DIR_NAME = "processed"
COMPOSE_FILE = "compose.processor.yml"
RUNTIME_REQUIREMENTS = Path("docker") / "processor" / "requirements.txt"


class DockerProcessorChecker:
    """
    Re-run processor.py on the packaged dataset folder in a one-shot container.

    The container has no network, a read-only filesystem except the dataset
    folder, and the same Python packages as the agent sandboxes. It calls
    ``process(data_dir, data_dir / "processed")`` as dcat-ap-hub does, so a
    passing check means ``Dataset.from_directory(...).process()`` reproduces
    the agent's exports byte for byte. Generated code never runs on the host.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        timeout_seconds: float = 1800,
        command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self._project_root = Path(project_root).expanduser().resolve()
        self._timeout_seconds = timeout_seconds
        self._run = command_runner

    @property
    def name(self) -> str:
        return "docker-processor-check"

    def check(
        self, package_dir: Path, expected: Mapping[str, Path]
    ) -> ProcessorCheckResult:
        package_dir = Path(package_dir).resolve()
        processed = package_dir / PROCESSED_DIR_NAME
        shutil.rmtree(processed, ignore_errors=True)
        environment = {
            **os.environ,
            "PACKAGE_DIR": str(package_dir),
            "ADE_UID": str(os.getuid()),
            "ADE_GID": str(os.getgid()),
        }
        command = [
            "docker", "compose", "-f", COMPOSE_FILE,
            "run", "--rm", "--build", "processor",
        ]
        try:
            completed = self._run(
                command,
                cwd=self._project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                output = f"{completed.stdout or ''}{completed.stderr or ''}".strip()
                raise ValueError(
                    "processor.py failed when re-run in isolation, as dcat-ap-hub "
                    f"would run it (exit {completed.returncode}):\n{output[-4000:]}"
                )
            produced = {name: processed / name for name in expected}
            missing = [name for name, path in produced.items() if not path.is_file()]
            if missing:
                raise ValueError(
                    f"processor.py did not write {', '.join(missing)} when re-run in isolation."
                )
            digests = {name: _sha256(path) for name, path in produced.items()}
            differing = [
                name for name, path in expected.items() if digests[name] != _sha256(path)
            ]
            if differing:
                raise ValueError(
                    "processor.py does not reproduce the exported "
                    f"{', '.join(differing)} when re-run in isolation; the published "
                    "dataset would not match its own split."
                )
        finally:
            # Leaving processed/ would make dcat-ap-hub skip processing later.
            shutil.rmtree(processed, ignore_errors=True)
        requirements = self._project_root / RUNTIME_REQUIREMENTS
        return ProcessorCheckResult(
            checker=self.name,
            sha256=digests,
            runtime=_runtime_summary(requirements),
        )


def _runtime_summary(requirements: Path) -> str:
    try:
        lines = requirements.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unknown"
    pins = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return "debian:trixie-slim python3; " + "; ".join(pins)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
