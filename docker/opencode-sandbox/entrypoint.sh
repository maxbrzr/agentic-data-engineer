#!/bin/sh
set -eu

: "${ADE_EXAMPLE_KEY:?ADE_EXAMPLE_KEY is required}"
: "${ADE_DATA_ROOT:?ADE_DATA_ROOT is required}"
: "${ADE_OUTPUT_DIR:?ADE_OUTPUT_DIR is required}"

marker_path=${ADE_SANDBOX_MARKER:-/tmp/agentic-data-engineer-sandbox.json}
data_probe="$ADE_DATA_ROOT/.opencode-sandbox-write-probe"
output_probe="$ADE_OUTPUT_DIR/.opencode-sandbox-write-probe"

if [ ! -d "$ADE_DATA_ROOT" ]; then
    echo "OpenCode sandbox startup failed: data directory does not exist: $ADE_DATA_ROOT" >&2
    exit 1
fi

if [ ! -d "$ADE_OUTPUT_DIR" ]; then
    echo "OpenCode sandbox startup failed: output directory does not exist: $ADE_OUTPUT_DIR" >&2
    exit 1
fi

if ( : > "$data_probe" ) 2>/dev/null; then
    rm -f "$data_probe"
    echo "OpenCode sandbox startup failed: raw data mount is writable: $ADE_DATA_ROOT" >&2
    exit 1
fi

if ! ( : > "$output_probe" ) 2>/dev/null; then
    echo "OpenCode sandbox startup failed: output mount is not writable: $ADE_OUTPUT_DIR" >&2
    exit 1
fi
rm -f "$output_probe"

export ADE_SANDBOX_MARKER="$marker_path"
python3 -c '
import json
import os
from pathlib import Path

marker = {
    "version": 1,
    "example_key": os.environ["ADE_EXAMPLE_KEY"],
    "data_root": os.environ["ADE_DATA_ROOT"],
    "output_dir": os.environ["ADE_OUTPUT_DIR"],
    "data_read_only": True,
    "output_writable": True,
}
Path(os.environ["ADE_SANDBOX_MARKER"]).write_text(
    json.dumps(marker, sort_keys=True),
    encoding="utf-8",
)
'

python3 /opt/agentic-data-engineer/attestation_server.py \
    >/tmp/opencode-sandbox-attestation.log 2>&1 &

exec "$@"
