"""Render Pi session files as a readable Markdown transcript.

Mirrors the layout of the OpenCode transcript: a numbered timeline of user and
assistant messages, each tool call with its output, and a tool summary.
"""

import json
from pathlib import Path
from typing import Any


def build_pi_transcript(session_dir: Path, run_id: str) -> str:
    entries = _read_entries(session_dir)
    results = {
        message.get("toolCallId"): message
        for message in (entry.get("message") for entry in entries)
        if isinstance(message, dict) and message.get("role") == "toolResult"
    }
    lines = ["# Pi Session Transcript", "", f"Session: `{run_id}`", "", "## Timeline", ""]
    tool_summary: list[str] = []
    index = 0

    for entry in entries:
        if entry.get("type") == "model_change":
            lines.extend(
                [f"- Model: `{entry.get('provider')}/{entry.get('modelId')}`", ""]
            )
            continue
        message = entry.get("message")
        if entry.get("type") != "message" or not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "toolResult":
            continue
        index += 1
        lines.extend([f"### {index}. {role}", ""])

        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for part in content if isinstance(content, list) else []:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type == "text" and part.get("text", "").strip():
                lines.extend(["**Response**", "", part["text"].strip(), ""])
            elif part_type == "thinking" and part.get("thinking", "").strip():
                lines.extend(["<details>", "<summary>Reasoning</summary>", ""])
                _append_fenced(lines, part["thinking"], max_chars=8000)
                lines.extend(["</details>", ""])
            elif part_type == "toolCall":
                summary = _tool_line(part, results.get(part.get("id")))
                tool_summary.append(summary)
                lines.append(summary)
                output = _result_text(results.get(part.get("id")))
                if output:
                    lines.extend(["", "<details>", "<summary>Tool output</summary>", ""])
                    _append_fenced(lines, output, max_chars=3000)
                    lines.extend(["</details>", ""])

        usage = message.get("usage")
        if role == "assistant" and isinstance(usage, dict):
            lines.append(
                f"- Step done: input `{usage.get('input', 0)}`, "
                f"output `{usage.get('output', 0)}`"
            )
        lines.append("")

    if tool_summary:
        lines.extend(["## Tool Summary", "", *tool_summary, ""])
    return "\n".join(lines).rstrip() + "\n"


def _read_entries(session_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(session_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _tool_line(call: dict[str, Any], result: dict[str, Any] | None) -> str:
    if result is None:
        icon, status = "…", "no result"
    elif result.get("isError"):
        icon, status = "✗", "error"
    else:
        icon, status = "✓", "completed"
    return f"- {icon} `{call.get('name')}`: {_label(call.get('arguments'))} ({status})"


def _label(arguments: Any) -> str:
    if isinstance(arguments, dict):
        for key in ("command", "path", "pattern", "file_path"):
            if isinstance(arguments.get(key), str):
                value = arguments[key]
                break
        else:
            value = json.dumps(arguments, ensure_ascii=False)
    else:
        value = str(arguments)
    value = " ".join(value.split())
    return value if len(value) <= 160 else value[:159] + "…"


def _result_text(result: dict[str, Any] | None) -> str:
    if result is None:
        return ""
    content = result.get("content", [])
    if isinstance(content, str):
        return content.strip()
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def _append_fenced(lines: list[str], text: str, max_chars: int) -> None:
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    lines.extend(["```text", text.rstrip(), "```", ""])
