import json
import os
import queue
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import monotonic
from time import sleep
from typing import Any
from uuid import uuid4

from ..contracts import AgentRequest, AgentRunResult, ModelConfig

CommandRunner = Callable[..., Any]
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


@dataclass(frozen=True, slots=True)
class PiSettings:
    project_root: Path
    timeout_seconds: float = 1200
    stall_timeout_seconds: float = 60
    max_continuations: int = 2
    max_provider_retries: int = 2
    retry_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "project_root",
            self.project_root.expanduser().resolve(),
        )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.stall_timeout_seconds <= 0:
            raise ValueError("stall_timeout_seconds must be positive")
        if self.stall_timeout_seconds > self.timeout_seconds:
            raise ValueError("stall_timeout_seconds cannot exceed timeout_seconds")
        if self.max_continuations < 0:
            raise ValueError("max_continuations must be non-negative")
        if self.max_provider_retries < 0:
            raise ValueError("max_provider_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")


class PiHarness:
    """Pi adapter that executes each task in a restricted one-shot container."""

    def __init__(
        self,
        settings: PiSettings,
        *,
        command_runner: CommandRunner | None = None,
        sleeper: Sleeper = sleep,
    ) -> None:
        self.settings = settings
        self._command_runner = command_runner
        self._sleeper = sleeper
        self._progress_channel: str | None = None

    @property
    def name(self) -> str:
        return "pi"

    def run(self, request: AgentRequest, model: ModelConfig) -> AgentRunResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        launcher = self.settings.project_root / "scripts" / "pi"
        if not launcher.is_file():
            raise RuntimeError(f"Pi container launcher does not exist: {launcher}")
        self._require_provider_environment(model.provider_id)

        session_dir = request.output_dir / ".pi-sessions"
        event_chunks: list[str] = []
        final_text = ""
        run_id: str | None = None
        message_id: str | None = None
        missing = self._missing_outputs(request.output_dir)

        for continuation in range(self.settings.max_continuations + 1):
            prompt = (
                request.task_prompt
                if continuation == 0
                else self._continuation_prompt(request.output_dir, missing)
            )
            request_path = request.output_dir / f".pi-request-{uuid4().hex}.json"
            request_path.write_text(
                json.dumps(
                    {
                        "provider_id": model.provider_id,
                        "model_id": model.model_id,
                        "model_parameters": dict(model.parameters),
                        "system_prompt": request.system_prompt,
                        "task_prompt": prompt,
                        "session_dir": str(session_dir),
                        "continue_session": continuation > 0,
                    }
                ),
                encoding="utf-8",
            )

            try:
                completed = self._run_with_provider_retries(
                    launcher,
                    request,
                    request_path,
                    continuation,
                    event_chunks,
                )
            finally:
                request_path.unlink(missing_ok=True)

            parsed_id, parsed_message_id, parsed_text = self._parse_events(
                completed.stdout or ""
            )
            run_id = parsed_id or run_id
            message_id = parsed_message_id or message_id
            final_text = parsed_text or final_text
            missing = self._missing_outputs(request.output_dir)
            if not missing:
                break
            if continuation < self.settings.max_continuations:
                event_chunks.append(
                    "\n[harness] Pi became idle with missing artifacts; "
                    f"starting continuation {continuation + 1}/"
                    f"{self.settings.max_continuations}: {missing}\n"
                )

        run_id = run_id or f"pi-{uuid4().hex}"
        run_log = request.output_dir / f"pi_run_{run_id}.log"
        run_log.write_text("".join(event_chunks), encoding="utf-8")
        (request.output_dir / "pi_live.log").unlink(missing_ok=True)
        if missing:
            raise ValueError(
                f"Pi run {run_id!r} exhausted "
                f"{self.settings.max_continuations} continuation attempts; "
                f"required host artifacts are still missing: {missing}. "
                f"See {run_log}."
            )

        report = request.output_dir / f"pi_report_{run_id}.md"
        report.write_text(
            final_text.strip() + "\n"
            if final_text.strip()
            else "Pi completed without a final text report.\n",
            encoding="utf-8",
        )
        artifacts = tuple(
            sorted(
                (path for path in request.output_dir.rglob("*") if path.is_file()),
                key=str,
            )
        )
        return AgentRunResult(
            harness=self.name,
            run_id=run_id,
            message_id=message_id,
            output_dir=request.output_dir,
            artifacts=artifacts,
            metadata={
                "run_log": str(run_log),
                "report": str(report),
                "provider_id": model.provider_id,
                "model_id": model.model_id,
            },
        )

    def _run_with_provider_retries(
        self,
        launcher: Path,
        request: AgentRequest,
        request_path: Path,
        continuation: int,
        event_chunks: list[str],
    ) -> Any:
        for provider_attempt in range(self.settings.max_provider_retries + 1):
            try:
                command = [
                    str(launcher),
                    "run",
                    request.dataset.spec.key,
                    str(request_path),
                ]
                if self._command_runner is None:
                    completed = self._run_streaming(
                        command,
                        request.output_dir / "pi_live.log",
                    )
                else:
                    completed = self._command_runner(
                        command,
                        cwd=self.settings.project_root,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self.settings.timeout_seconds,
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                error_text = str(exc)
                completed = None
            else:
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                event_chunks.append(
                    f"\n[harness] turn={continuation + 1} "
                    f"provider_attempt={provider_attempt + 1}\n"
                    f"{stdout}{stderr}"
                )
                event_error = self._event_error(stdout)
                if completed.returncode == 0 and event_error is None:
                    return completed
                error_text = event_error or f"{stdout}\n{stderr}".strip()

            if (
                provider_attempt < self.settings.max_provider_retries
                and self._is_transient_provider_error(error_text)
            ):
                delay = self.settings.retry_backoff_seconds * (2**provider_attempt)
                event_chunks.append(
                    "\n[harness] Transient provider error; retrying in "
                    f"{delay:g}s: {error_text}\n"
                )
                self._append_live_note(
                    request.output_dir,
                    "Transient Pi/provider failure; retrying in "
                    f"{delay:g}s: {error_text}",
                )
                print(
                    "[pi] transient failure; retrying provider call in "
                    f"{delay:g}s: {error_text}",
                    file=sys.stderr,
                    flush=True,
                )
                self._sleeper(delay)
                continue
            failure = (
                "Pi container execution failed. Confirm Docker Desktop is "
                f"running and inspect the provider response: {error_text}"
            )
            self._append_live_note(request.output_dir, failure)
            raise RuntimeError(failure)
        raise AssertionError("provider retry loop did not return or raise")

    def _run_streaming(
        self,
        command: list[str],
        live_log_path: Path | None = None,
    ) -> subprocess.CompletedProcess:
        """Run Compose while retaining JSONL and exposing concise live progress."""
        process = subprocess.Popen(
            command,
            cwd=self.settings.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()

        def read_stream(name: str, stream) -> None:
            try:
                for line in iter(stream.readline, ""):
                    lines.put((name, line))
            finally:
                lines.put((name, None))

        readers = (
            Thread(target=read_stream, args=("stdout", process.stdout), daemon=True),
            Thread(target=read_stream, args=("stderr", process.stderr), daemon=True),
        )
        for reader in readers:
            reader.start()

        captured = {"stdout": [], "stderr": []}
        closed: set[str] = set()
        started = monotonic()
        last_output = started
        live_log = (
            live_log_path.open("a", encoding="utf-8", buffering=1)
            if live_log_path is not None
            else None
        )
        if live_log is not None:
            live_log.write("\n[harness] Starting Pi container invocation.\n")
        try:
            while len(closed) < 2:
                now = monotonic()
                if now - started > self.settings.timeout_seconds:
                    raise subprocess.TimeoutExpired(
                        command,
                        self.settings.timeout_seconds,
                        output="".join(captured["stdout"]),
                        stderr="".join(captured["stderr"]),
                    )
                if now - last_output > self.settings.stall_timeout_seconds:
                    raise subprocess.TimeoutExpired(
                        command,
                        self.settings.stall_timeout_seconds,
                        output="".join(captured["stdout"]),
                        stderr=(
                            "".join(captured["stderr"])
                            + "\nPi emitted no events or container output during "
                            f"the {self.settings.stall_timeout_seconds:g}s stall "
                            "timeout."
                        ),
                    )
                try:
                    stream_name, line = lines.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    closed.add(stream_name)
                    continue
                last_output = monotonic()
                self._show_progress(stream_name, line)
                retained_line = self._retained_log_line(stream_name, line)
                if retained_line is not None:
                    captured[stream_name].append(retained_line)
                    if live_log is not None:
                        live_log.write(retained_line)
        except BaseException:
            self._terminate_process_group(process)
            raise
        finally:
            for reader in readers:
                reader.join(timeout=1)
            process.stdout.close()
            process.stderr.close()
            if live_log is not None:
                live_log.close()

        return_code = process.wait()
        return subprocess.CompletedProcess(
            command,
            return_code,
            stdout="".join(captured["stdout"]),
            stderr="".join(captured["stderr"]),
        )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _show_progress(self, stream_name: str, line: str) -> None:
        if stream_name == "stderr":
            print(f"[pi-container] {line.rstrip()}", file=sys.stderr, flush=True)
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"[pi-container] {line.rstrip()}", flush=True)
            return
        event_type = event.get("type")
        if event_type == "harness_status":
            print(
                "[pi] process starting: "
                f"{event.get('provider', 'unknown')}/"
                f"{event.get('model', 'unknown')}",
                flush=True,
            )
        elif event_type == "session":
            print(f"[pi] session {event.get('id', 'unknown')} started", flush=True)
        elif event_type == "message_start":
            message = event.get("message", {})
            if isinstance(message, dict) and message.get("role") == "assistant":
                print("[pi] waiting for model response", flush=True)
        elif event_type == "message_update":
            update = event.get("assistantMessageEvent", {})
            if not isinstance(update, dict):
                return
            update_type = update.get("type")
            delta = update.get("delta")
            if update_type in {"text_start", "thinking_start"}:
                self._progress_channel = str(update_type).removesuffix("_start")
                print(f"[pi:{self._progress_channel}] ", end="", flush=True)
            elif update_type in {"text_delta", "thinking_delta"}:
                if isinstance(delta, str):
                    print(delta, end="", flush=True)
            elif update_type in {"text_end", "thinking_end"}:
                print(flush=True)
                self._progress_channel = None
            elif update_type == "toolcall_start":
                self._progress_channel = "toolcall"
                print("[pi:toolcall] ", end="", flush=True)
            elif update_type == "toolcall_delta":
                if isinstance(delta, str):
                    print(delta, end="", flush=True)
            elif update_type == "toolcall_end":
                print(flush=True)
                self._progress_channel = None
        elif event_type == "tool_execution_start":
            tool_name = event.get("toolName", "unknown")
            arguments = event.get("args")
            detail = self._display_value(arguments)
            suffix = f" {detail}" if detail else ""
            print(f"[pi] tool: {tool_name}{suffix}", flush=True)
        elif event_type == "tool_execution_end":
            status = "failed" if event.get("isError") else "done"
            print(
                f"[pi] tool {event.get('toolName', 'unknown')}: {status}",
                flush=True,
            )
        elif event_type == "agent_end":
            print("[pi] agent turn finished", flush=True)

    @staticmethod
    def _retained_log_line(stream_name: str, line: str) -> str | None:
        """Drop quadratic partial snapshots; final messages remain authoritative."""
        if stream_name != "stdout":
            return line
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return line
        if event.get("type") in {"message_update", "tool_execution_update"}:
            return None
        return line

    @staticmethod
    def _display_value(value: Any, *, limit: int = 800) -> str:
        if value in (None, "", {}, []):
            return ""
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            rendered = str(value)
        rendered = rendered.replace("\n", "\\n")
        if len(rendered) > limit:
            return rendered[: limit - 1] + "…"
        return rendered

    @staticmethod
    def _append_live_note(output_dir: Path, note: str) -> None:
        with (output_dir / "pi_live.log").open("a", encoding="utf-8") as stream:
            stream.write(f"\n[harness] {note}\n")

    def _require_provider_environment(self, provider_id: str) -> None:
        if provider_id != "gwdg":
            return
        env_path = self.settings.project_root / ".env"
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            candidate = line.strip()
            if not candidate or candidate.startswith("#") or "=" not in candidate:
                continue
            key, value = candidate.removeprefix("export ").split("=", 1)
            if key.strip() == "SAIA_API_KEY" and value.strip().strip("'\""):
                return
        raise RuntimeError(
            f"Provider 'gwdg' requires SAIA_API_KEY in the ignored "
            f"environment file {env_path}."
        )

    @staticmethod
    def _missing_outputs(output_dir: Path) -> list[str]:
        return [
            name for name in REQUIRED_OUTPUTS if not (output_dir / name).is_file()
        ]

    @staticmethod
    def _continuation_prompt(output_dir: Path, missing: list[str]) -> str:
        return (
            "Continue the same task from the existing files. Pi stopped before "
            f"the host could see these required artifacts in {output_dir}: "
            f"{missing}. Do not restart exploration and do not read complete raw "
            "tables with the read tool. Create the missing artifacts, run the "
            "required validations, and verify parser.py, train.csv, and test.csv "
            "exist before finishing."
        )

    @staticmethod
    def _is_transient_provider_error(message: str) -> bool:
        normalized = message.lower()
        return any(marker in normalized for marker in TRANSIENT_PROVIDER_MARKERS)

    @staticmethod
    def _parse_events(payload: str) -> tuple[str | None, str | None, str]:
        run_id = None
        message_id = None
        final_text = ""
        for line in payload.splitlines():
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "session" and isinstance(event.get("id"), str):
                run_id = event["id"]
            message = event.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if isinstance(message.get("id"), str):
                message_id = message["id"]
            text_parts: list[str] = []
            content = message.get("content", [])
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        text_parts.append(part["text"])
            if text_parts:
                final_text = "\n".join(text_parts)
        return run_id, message_id, final_text

    @staticmethod
    def _event_error(payload: str) -> str | None:
        for line in reversed(payload.splitlines()):
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            message = event.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if message.get("stopReason") != "error":
                continue
            detail = message.get("errorMessage")
            return str(detail or "Pi model provider returned an error")
        return None
