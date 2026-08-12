from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Any

from opencode_ai import Opencode

from ..contracts import AgentRequest, AgentRunResult, ModelConfig
from .opencode_logging import save_session_messages, watch_session_events
from .opencode_sandbox import (
    AttestationReader,
    read_sandbox_attestation,
)
from .provider_env import provider_environment_name

ClientFactory = Callable[..., Any]
Sleeper = Callable[[float], None]

REQUIRED_OUTPUTS = ("parser.py", "train.csv", "test.csv")
TRANSIENT_PROVIDER_MARKERS = (
    "no provider available",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
    "connection error",
    "connection reset",
    "timed out",
    "timeout",
    "status code: 429",
    "status code: 502",
    "status code: 503",
    "status code: 504",
)


class _SessionProviderError(RuntimeError):
    """Provider failure recorded in an otherwise successful HTTP response."""


@dataclass(frozen=True, slots=True)
class OpencodeSettings:
    base_url: str = "http://127.0.0.1:54321"
    mode: str = "Agent"
    timeout_seconds: float = 1200
    stream_events: bool = True
    max_continuations: int = 2
    max_provider_retries: int = 2
    retry_backoff_seconds: float = 2.0
    require_sandbox_preflight: bool = True
    sandbox_attestation_url: str = (
        "http://127.0.0.1:54322/agentic-data-engineer-sandbox.json"
    )

    def __post_init__(self) -> None:
        if self.max_continuations < 0:
            raise ValueError("max_continuations must be non-negative")
        if self.max_provider_retries < 0:
            raise ValueError("max_provider_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")


class OpencodeHarness:
    """OpenCode adapter for the harness-neutral AgentHarness port."""

    def __init__(
        self,
        settings: OpencodeSettings | None = None,
        *,
        client_factory: ClientFactory = Opencode,
        sleeper: Sleeper = sleep,
        attestation_reader: AttestationReader = read_sandbox_attestation,
    ) -> None:
        self.settings = settings or OpencodeSettings()
        self._client_factory = client_factory
        self._sleeper = sleeper
        self._attestation_reader = attestation_reader

    @property
    def name(self) -> str:
        return "opencode"

    def run(self, request: AgentRequest, model: ModelConfig) -> AgentRunResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.require_sandbox_preflight:
            self._validate_sandbox(request, model)
        client = self._client_factory(base_url=self.settings.base_url)
        session = client.session.create()
        response: Any = None
        event_attempt = 0
        missing = self._missing_outputs(request.output_dir)

        try:
            for continuation in range(self.settings.max_continuations + 1):
                prompt = (
                    request.task_prompt
                    if continuation == 0
                    else self._continuation_prompt(request.output_dir, missing)
                )

                for provider_attempt in range(
                    self.settings.max_provider_retries + 1
                ):
                    watcher = self._start_watcher(
                        session_id=session.id,
                        output_dir=request.output_dir,
                        append=event_attempt > 0,
                        attempt_label=(
                            f"turn {continuation + 1}/"
                            f"{self.settings.max_continuations + 1}, "
                            f"provider attempt {provider_attempt + 1}/"
                            f"{self.settings.max_provider_retries + 1}"
                        ),
                    )
                    event_attempt += 1
                    try:
                        response = client.session.chat(
                            session.id,
                            provider_id=model.provider_id,
                            model_id=model.model_id,
                            mode=self.settings.mode,
                            system=request.system_prompt,
                            timeout=self.settings.timeout_seconds,
                            parts=[{"type": "text", "text": prompt}],
                            extra_body=dict(model.parameters) or None,
                        )
                        provider_error = self._latest_provider_error(
                            client,
                            session.id,
                        )
                        if provider_error:
                            raise _SessionProviderError(provider_error)
                    except Exception as exc:
                        self._join_watcher(watcher)
                        if (
                            provider_attempt
                            < self.settings.max_provider_retries
                            and self._is_transient_provider_error(exc)
                        ):
                            delay = self.settings.retry_backoff_seconds * (
                                2**provider_attempt
                            )
                            self._append_log_note(
                                request.output_dir,
                                session.id,
                                "Transient provider error; retrying in "
                                f"{delay:g}s: {exc}",
                            )
                            self._sleeper(delay)
                            continue
                        raise
                    else:
                        self._join_watcher(watcher)
                        break

                missing = self._missing_outputs(request.output_dir)
                if not missing:
                    break
                if continuation < self.settings.max_continuations:
                    self._append_log_note(
                        request.output_dir,
                        session.id,
                        "Session became idle with missing artifacts; "
                        f"sending continuation {continuation + 1}/"
                        f"{self.settings.max_continuations}: {missing}",
                    )
        except Exception:
            save_session_messages(client, session.id, request.output_dir)
            raise

        saved_paths = save_session_messages(client, session.id, request.output_dir)
        if missing:
            raise ValueError(
                f"OpenCode session {session.id!r} exhausted "
                f"{self.settings.max_continuations} continuation attempts; "
                f"required host artifacts are still missing: {missing}."
            )

        artifacts = tuple(
            sorted(
                (path for path in request.output_dir.rglob("*") if path.is_file()),
                key=str,
            )
        )
        metadata = {
            "run_log": str(saved_paths["run_log"]),
            "report": str(saved_paths["report"]),
            "provider_id": model.provider_id,
            "model_id": model.model_id,
        }
        return AgentRunResult(
            harness=self.name,
            run_id=str(session.id),
            message_id=str(getattr(response, "id", "")) or None,
            output_dir=request.output_dir,
            artifacts=artifacts,
            metadata=metadata,
        )

    def _validate_sandbox(
        self,
        request: AgentRequest,
        model: ModelConfig,
    ) -> None:
        expected_key = request.dataset.spec.key
        expected_output = request.output_dir.resolve()
        expected_data = request.dataset.data_dir.resolve()

        try:
            marker = self._attestation_reader(
                self.settings.sandbox_attestation_url,
                expected_output,
            )
        except Exception as exc:
            raise ValueError(
                "OpenCode sandbox preflight failed before an agent session was "
                "created. The localhost sandbox attestation endpoint did not "
                "expose a valid mount marker. Start the matching container with "
                f"`./scripts/opencode start {expected_key}`."
            ) from exc

        actual_key = marker.get("example_key")
        actual_output = self._marker_path(marker, "output_dir")
        actual_data_root = self._marker_path(marker, "data_root")
        problems: list[str] = []

        if actual_key != expected_key:
            problems.append(
                f"container example is {actual_key!r}, requested {expected_key!r}"
            )
        if actual_output is None or not self._is_within(
            expected_output,
            actual_output,
        ):
            problems.append(
                f"requested output {str(expected_output)!r} is not within the "
                f"container output mount {str(actual_output)!r}"
            )
        if actual_data_root is None or not self._is_within(
            expected_data,
            actual_data_root,
        ):
            problems.append(
                f"dataset path {str(expected_data)!r} is not within the "
                f"container data mount {str(actual_data_root)!r}"
            )
        if marker.get("data_read_only") is not True:
            problems.append("container did not attest that raw data is read-only")
        if marker.get("output_writable") is not True:
            problems.append(
                "container did not attest that the selected output is writable"
            )
        required_env = provider_environment_name(model.provider_id)
        if required_env is not None:
            configured_providers = marker.get("configured_providers", [])
            if (
                not isinstance(configured_providers, list)
                or model.provider_id not in configured_providers
            ):
                problems.append(
                    f"Provider {model.provider_id!r} was selected but "
                    f"{required_env} was not loaded from .env"
                )

        if problems:
            details = "; ".join(problems)
            raise ValueError(
                "OpenCode sandbox preflight failed before an agent session was "
                f"created: {details}. Reconfigure the container with "
                f"`./scripts/opencode start {expected_key}` and rerun "
                "the pipeline."
            )

    @staticmethod
    def _marker_path(marker: dict[str, Any], key: str) -> Path | None:
        value = marker.get(key)
        if not isinstance(value, str) or not value:
            return None
        return Path(value).resolve()

    @staticmethod
    def _is_within(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True

    def _start_watcher(
        self,
        *,
        session_id: str,
        output_dir,
        append: bool,
        attempt_label: str,
    ) -> Thread | None:
        if not self.settings.stream_events:
            return None
        watcher = Thread(
            target=watch_session_events,
            args=(self.settings.base_url, session_id, output_dir),
            kwargs={
                "client_factory": self._client_factory,
                "append": append,
                "attempt_label": attempt_label,
            },
            daemon=True,
        )
        watcher.start()
        return watcher

    @staticmethod
    def _join_watcher(watcher: Thread | None) -> None:
        if watcher is not None:
            watcher.join(timeout=5)

    @staticmethod
    def _missing_outputs(output_dir) -> list[str]:
        return [
            name for name in REQUIRED_OUTPUTS if not (output_dir / name).is_file()
        ]

    @staticmethod
    def _continuation_prompt(output_dir, missing: list[str]) -> str:
        return (
            "Continue the same task. You became idle before completing it. "
            f"The host still cannot see these required artifacts in {output_dir}: "
            f"{missing}. Resume from the work already completed in this session; "
            "do not restart exploration and do not read complete raw tables with "
            "the read tool. Create the missing artifacts, run the required "
            "validations, and verify parser.py, train.csv, and test.csv exist "
            "before finishing."
        )

    @staticmethod
    def _latest_provider_error(client, session_id: str) -> str | None:
        for message in reversed(client.session.messages(session_id)):
            info = getattr(message, "info", None)
            if getattr(info, "role", None) != "assistant":
                continue
            error = getattr(info, "error", None)
            if error is None:
                return None
            data = getattr(error, "data", None)
            detail = getattr(data, "message", None)
            return str(detail or error)
        return None

    @staticmethod
    def _is_transient_provider_error(exc: Exception) -> bool:
        messages: list[str] = []
        current: BaseException | None = exc
        while current is not None:
            messages.append(str(current).lower())
            current = current.__cause__ or current.__context__
        combined = " ".join(messages)
        return any(marker in combined for marker in TRANSIENT_PROVIDER_MARKERS)

    @staticmethod
    def _append_log_note(output_dir, session_id: str, note: str) -> None:
        run_log = output_dir / f"opencode_run_{session_id}.log"
        with run_log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[harness] {note}\n")
