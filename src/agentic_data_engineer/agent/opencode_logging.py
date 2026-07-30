import json
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from opencode_ai import Opencode


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _to_jsonable(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_to_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _short_path(value: Any, max_len: int = 96) -> str:
    text = str(value or "")
    marker = "/agentic-data-engineer/"
    if marker in text:
        text = text.split(marker, 1)[1]
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3):]


def _tool_label(part: Any) -> str:
    state = part.state
    title = getattr(state, "title", None)
    if title and title != part.tool:
        return _short_path(title)

    tool_input = getattr(state, "input", None)
    if isinstance(tool_input, dict):
        for key in ("filePath", "path", "file", "pattern", "command"):
            if key in tool_input:
                return _short_path(tool_input[key])

    return part.tool


def _write_log_line(log_file: TextIO, line: str = "") -> None:
    log_file.write(line + "\n")
    log_file.flush()


def _print_part_text(part: Any, seen_text: dict[str, str], log_file: TextIO | None = None) -> None:
    old = seen_text.get(part.id, "")
    new = part.text or ""
    delta = new[len(old):] if new.startswith(old) else new
    if not delta:
        return

    prefix = "[reasoning] " if getattr(part, "type", "") == "reasoning" else ""
    if old:
        print(delta, end="", flush=True)
        if log_file is not None:
            log_file.write(delta)
            log_file.flush()
    else:
        text = f"\n{prefix}{delta}"
        print(text, end="", flush=True)
        if log_file is not None:
            log_file.write(text)
            log_file.flush()
    seen_text[part.id] = new


def _part_text(part: Any) -> str:
    return str(getattr(part, "text", "") or "").strip()


def _tool_record(part: Any) -> dict[str, Any]:
    state = part.state
    record = {
        "tool": part.tool,
        "status": state.status,
        "label": _tool_label(part),
    }
    for attr in ("title", "input", "metadata", "output", "error"):
        if hasattr(state, attr):
            value = getattr(state, attr)
            if value not in (None, "", {}, []):
                record[attr] = _to_jsonable(value)
    return record


def _append_fenced(lines: list[str], text: str, language: str = "text", max_chars: int = 4000) -> None:
    if not text:
        return
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    lines.extend([f"```{language}", text.rstrip(), "```", ""])


def _build_session_report(messages: Any, session_id: str) -> tuple[str, list[dict[str, Any]]]:
    lines = [
        f"# Opencode Session Report",
        "",
        f"Session: `{session_id}`",
        "",
        "## Timeline",
        "",
    ]
    tool_records: list[dict[str, Any]] = []

    for msg_index, message in enumerate(messages, start=1):
        info = getattr(message, "info", None)
        role = getattr(info, "role", None)
        message_id = getattr(message, "id", None)

        if isinstance(info, dict):
            role = role or info.get("role")
            message_id = message_id or info.get("id")

        role = role or "unknown"

        lines.extend([f"### {msg_index}. {role}", ""])
        if message_id:
            lines.extend([f"Message: `{message_id}`", ""])

        for part in getattr(message, "parts", []):
            part_type = getattr(part, "type", "unknown")

            if part_type == "text":
                text = _part_text(part)
                if text:
                    lines.extend(["**Response**", "", text, ""])

            elif part_type == "reasoning":
                text = _part_text(part)
                if text:
                    lines.extend(["<details>", "<summary>Reasoning</summary>", ""])
                    _append_fenced(lines, text, max_chars=8000)
                    lines.extend(["</details>", ""])

            elif part_type == "tool":
                record = _tool_record(part)
                tool_records.append(record)
                status = record["status"]
                icon = {"completed": "✓", "error": "✗", "running": "→", "pending": "…"}.get(status, "-")
                lines.append(f"- {icon} `{record['tool']}`: {record['label']} ({status})")
                if status == "error" and record.get("error"):
                    lines.append(f"  - Error: `{record['error']}`")
                if status == "completed" and record.get("output"):
                    output = str(record["output"]).strip()
                    if output:
                        lines.append("")
                        lines.extend(["<details>", "<summary>Tool output</summary>", ""])
                        _append_fenced(lines, output, max_chars=3000)
                        lines.extend(["</details>", ""])

            elif part_type == "step-finish":
                tokens = getattr(part, "tokens", None)
                if tokens is not None:
                    lines.append(
                        f"- Step done: input `{tokens.input:.0f}`, output `{tokens.output:.0f}`, "
                        f"reasoning `{tokens.reasoning:.0f}`, cost `{part.cost}`"
                    )
                else:
                    lines.append(f"- Step done: cost `{getattr(part, 'cost', None)}`")

            elif part_type in {"step-start", "snapshot", "patch"}:
                continue

            else:
                lines.extend([f"**{part_type}**", ""])
                _append_fenced(lines, json.dumps(_to_jsonable(part), ensure_ascii=False, indent=2), "json")

        lines.append("")

    if tool_records:
        lines.extend(["## Tool Summary", ""])
        for record in tool_records:
            status = record["status"]
            icon = {"completed": "✓", "error": "✗", "running": "→", "pending": "…"}.get(status, "-")
            lines.append(f"- {icon} `{record['tool']}`: {record['label']} ({status})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", tool_records



def watch_session_events(
    base_url: str,
    session_id: str,
    output_dir: Path,
    *,
    client_factory=Opencode,
    append: bool = False,
    attempt_label: str | None = None,
) -> None:
    """Stream opencode events, print useful progress, and persist a readable run log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / f"opencode_run_{session_id}.log"
    event_client = client_factory(base_url=base_url)

    seen_text: dict[str, str] = {}
    seen_tool_status: dict[str, str] = {}
    printed_completed_tools: set[str] = set()

    with run_log.open("a" if append else "w", encoding="utf-8") as log_file:
        if not append:
            _write_log_line(log_file, f"Opencode run log: {session_id}")
            _write_log_line(
                log_file,
                f"Started: {datetime.now().isoformat(timespec='seconds')}",
            )
        if attempt_label:
            _write_log_line(log_file, f"\n[{attempt_label}]")
        _write_log_line(log_file)

        for event in event_client.event.list(timeout=None):
            event_type = getattr(event, "type", "")
            properties = getattr(event, "properties", None)

            if event_type == "message.part.delta":
                continue

            if event_type == "message.part.updated":
                part = getattr(properties, "part", None)
                if getattr(part, "session_id", None) != session_id:
                    continue

                if part.type in {"text", "reasoning"}:
                    _print_part_text(part, seen_text, log_file)

                elif part.type == "tool":
                    state = part.state
                    status = state.status
                    previous = seen_tool_status.get(part.id)
                    seen_tool_status[part.id] = status

                    label = _tool_label(part)
                    if status == "running" and previous not in {"running", "completed", "error"}:
                        line = f"\n→ {part.tool}: {label}"
                        print(line, flush=True)
                        _write_log_line(log_file, line)
                    elif status == "completed" and part.id not in printed_completed_tools:
                        line = f"\n✓ {part.tool}: {label}"
                        print(line, flush=True)
                        _write_log_line(log_file, line)
                        printed_completed_tools.add(part.id)
                    elif status == "error":
                        line = f"\n✗ {part.tool}: {label}"
                        print(line, flush=True)
                        _write_log_line(log_file, line)
                        error = str(getattr(state, "error", ""))
                        if error:
                            print(f"  {error}", flush=True)
                            _write_log_line(log_file, f"  {error}")

                elif part.type == "step-start":
                    line = "\n--- step ---"
                    print(line, flush=True)
                    _write_log_line(log_file, line)

                elif part.type == "step-finish":
                    tokens = getattr(part, "tokens", None)
                    if tokens is not None:
                        line = (
                            f"\n--- step done: input={tokens.input:.0f}, "
                            f"output={tokens.output:.0f}, reasoning={tokens.reasoning:.0f}, "
                            f"cost={part.cost} ---"
                        )
                    else:
                        line = f"\n--- step done: cost={part.cost} ---"
                    print(line, flush=True)
                    _write_log_line(log_file, line)

            elif event_type == "session.error":
                session_error_id = getattr(properties, "session_id", None)
                if session_error_id in (None, session_id):
                    line = f"\n[session error] {getattr(properties, 'error', None)}"
                    print(line, flush=True)
                    _write_log_line(log_file, line)
                    break

            elif event_type == "session.idle":
                if getattr(properties, "session_id", None) == session_id:
                    line = "\n[session idle]"
                    print(line, flush=True)
                    _write_log_line(log_file, line)
                    break

        _write_log_line(log_file)
        _write_log_line(
            log_file,
            f"Attempt stream closed: {datetime.now().isoformat(timespec='seconds')}",
        )


def save_session_messages(client: Opencode, session_id: str, output_dir: Path) -> dict[str, Path]:
    """Persist the finished opencode session as a readable Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    messages = client.session.messages(session_id)

    report_path = output_dir / f"opencode_report_{session_id}.md"
    run_log_path = output_dir / f"opencode_run_{session_id}.log"

    report, _tools = _build_session_report(messages, session_id)
    report_path.write_text(report, encoding="utf-8")
    with run_log_path.open("a", encoding="utf-8") as log_file:
        _write_log_line(
            log_file,
            f"\nFinished: {datetime.now().isoformat(timespec='seconds')}",
        )

    return {"run_log": run_log_path, "report": report_path}
