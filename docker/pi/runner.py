"""Translate a mounted harness request into one Pi JSON-mode invocation."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    request_path = Path(os.environ["PI_REQUEST_FILE"]).resolve()
    output_root = Path(os.environ["ADE_OUTPUT_DIR"]).resolve()
    try:
        request_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("PI_REQUEST_FILE must be inside ADE_OUTPUT_DIR") from exc

    request = json.loads(request_path.read_text(encoding="utf-8"))
    session_dir = Path(request["session_dir"]).resolve()
    try:
        session_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("Pi session_dir must be inside ADE_OUTPUT_DIR") from exc
    session_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "pi",
        "--mode",
        "json",
        "--provider",
        request["provider_id"],
        "--model",
        request["model_id"],
        "--session-dir",
        str(session_dir),
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-approve",
        "--tools",
        "read,bash,edit,write,grep,find,ls",
        "--system-prompt",
        request["system_prompt"],
    ]
    if request.get("continue_session"):
        command.append("--continue")
    command.append(request["task_prompt"])

    print(
        json.dumps(
            {
                "type": "harness_status",
                "status": "starting_pi_process",
                "provider": request["provider_id"],
                "model": request["model_id"],
            }
        ),
        flush=True,
    )
    # Compose runs without a TTY but may leave stdin open. Pi treats any
    # non-TTY stdin as piped prompt content and waits for EOF before it creates
    # the session. The harness passes prompts as arguments, so close stdin
    # explicitly to prevent that pre-session deadlock.
    completed = subprocess.run(command, check=False, stdin=subprocess.DEVNULL)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
