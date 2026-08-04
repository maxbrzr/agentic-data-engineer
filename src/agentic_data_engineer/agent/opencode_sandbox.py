import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

AttestationReader = Callable[[str, Path], dict[str, Any]]
CommandRunner = Callable[..., Any]

MANAGED_OPENCODE_URLS = {
    "http://127.0.0.1:54321",
    "http://localhost:54321",
}
PROVIDER_ENV_REQUIREMENTS = {"gwdg": "SAIA_API_KEY"}


def read_sandbox_attestation(
    url: str,
    expected_output_dir: Path,
) -> dict[str, Any]:
    """Read the attestation and prove its output bind reaches the host."""
    with urlopen(url, timeout=2) as response:
        marker = json.load(response)
    if not isinstance(marker, dict):
        raise TypeError("sandbox attestation must be a JSON object")
    output_value = marker.get("output_dir")
    probe_name = marker.get("probe_name")
    probe_token = marker.get("probe_token")
    if (
        not isinstance(output_value, str)
        or probe_name != ".opencode-sandbox-live-probe"
        or not isinstance(probe_token, str)
        or not probe_token
    ):
        raise ValueError("sandbox attestation has no valid live-mount probe")

    output_dir = Path(output_value).resolve()
    expected_output_dir = expected_output_dir.resolve()
    if output_dir != expected_output_dir and not _is_within(
        expected_output_dir,
        output_dir,
    ):
        raise ValueError(
            f"sandbox output mount is {output_dir}, which does not contain "
            f"the requested output {expected_output_dir}"
        )

    probe_path = output_dir / probe_name
    try:
        observed_token = probe_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            "sandbox output bind is stale: the container probe did not reach "
            f"the host path {probe_path}"
        ) from exc
    finally:
        probe_path.unlink(missing_ok=True)

    if observed_token != probe_token:
        raise ValueError(
            "sandbox output bind is stale: host and container probe tokens differ"
        )
    return marker


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(slots=True)
class OpencodeSandboxManager:
    """Ensure that the local Compose sandbox matches one example."""

    project_root: Path
    base_url: str = "http://127.0.0.1:54321"
    attestation_url: str | None = None
    attestation_reader: AttestationReader = read_sandbox_attestation
    command_runner: CommandRunner = subprocess.run

    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser().resolve()
        self.base_url = self.base_url.rstrip("/")
        if self.base_url not in MANAGED_OPENCODE_URLS:
            raise ValueError(
                "OpenCode is Docker-only and must use the local Compose endpoint "
                "http://127.0.0.1:54321."
            )
        if self.attestation_url is None:
            self.attestation_url = (
                self.base_url.replace(":54321", ":54322")
                + "/agentic-data-engineer-sandbox.json"
            )

    def ensure(
        self,
        example_key: str,
        *,
        required_provider: str | None = None,
    ) -> None:
        """Start or switch the sandbox unless its marker already matches."""
        self._require_provider_environment(required_provider)
        if self._matches(example_key, required_provider=required_provider):
            print(f"OpenCode sandbox already matches {example_key}.")
            return

        launcher = self.project_root / "scripts" / "opencode"
        if not launcher.is_file():
            raise RuntimeError(
                "Cannot manage the OpenCode sandbox because the launcher does "
                f"not exist: {launcher}"
            )

        print(f"Starting OpenCode sandbox for {example_key}...")
        try:
            self.command_runner(
                [str(launcher), "start", example_key],
                cwd=self.project_root,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Failed to start the OpenCode sandbox for {example_key!r}. "
                "Confirm that Docker Desktop is running and retry."
            ) from exc

        if not self._matches(example_key, required_provider=required_provider):
            raise RuntimeError(
                "The OpenCode sandbox started, but its output bind did not pass "
                f"the host visibility check for {example_key!r}."
            )

    def _matches(
        self,
        example_key: str,
        *,
        required_provider: str | None = None,
    ) -> bool:
        expected_data = (self.project_root / "data" / example_key).resolve()
        expected_output = (self.project_root / "output" / example_key).resolve()
        try:
            marker = self.attestation_reader(
                str(self.attestation_url),
                expected_output,
            )
        except Exception:
            return False

        configured_providers = marker.get("configured_providers", [])
        provider_ready = (
            required_provider not in PROVIDER_ENV_REQUIREMENTS
            or (
                isinstance(configured_providers, list)
                and required_provider in configured_providers
            )
        )
        return (
            marker.get("example_key") == example_key
            and self._path(marker.get("data_root")) == expected_data
            and self._path(marker.get("output_dir")) == expected_output
            and marker.get("data_read_only") is True
            and marker.get("output_writable") is True
            and provider_ready
        )

    def _require_provider_environment(self, provider: str | None) -> None:
        env_name = PROVIDER_ENV_REQUIREMENTS.get(provider or "")
        if env_name is None:
            return
        env_path = self.project_root / ".env"
        if self._env_has_value(env_path, env_name):
            return
        raise RuntimeError(
            f"Provider {provider!r} requires {env_name} in the ignored "
            f"environment file {env_path}."
        )

    @staticmethod
    def _env_has_value(path: Path, name: str) -> bool:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return False
        for line in lines:
            candidate = line.strip()
            if not candidate or candidate.startswith("#") or "=" not in candidate:
                continue
            key, value = candidate.removeprefix("export ").split("=", 1)
            if key.strip() == name:
                return bool(value.strip().strip("'\""))
        return False

    @staticmethod
    def _path(value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        return Path(value).resolve()
