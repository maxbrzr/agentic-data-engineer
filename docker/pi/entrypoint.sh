#!/bin/sh
set -eu

: "${ADE_EXAMPLE_KEY:?ADE_EXAMPLE_KEY is required}"
: "${ADE_DATA_ROOT:?ADE_DATA_ROOT is required}"
: "${ADE_OUTPUT_DIR:?ADE_OUTPUT_DIR is required}"
: "${PI_REQUEST_FILE:?PI_REQUEST_FILE is required}"

data_probe="$ADE_DATA_ROOT/.pi-write-probe"
output_probe="$ADE_OUTPUT_DIR/.pi-write-probe"

if [ ! -d "$ADE_DATA_ROOT" ]; then
    echo "Pi container startup failed: data directory does not exist: $ADE_DATA_ROOT" >&2
    exit 1
fi

if [ ! -d "$ADE_OUTPUT_DIR" ]; then
    echo "Pi container startup failed: output directory does not exist: $ADE_OUTPUT_DIR" >&2
    exit 1
fi

if ( : > "$data_probe" ) 2>/dev/null; then
    rm -f "$data_probe"
    echo "Pi container startup failed: raw data mount is writable: $ADE_DATA_ROOT" >&2
    exit 1
fi

if ! ( : > "$output_probe" ) 2>/dev/null; then
    echo "Pi container startup failed: output mount is not writable: $ADE_OUTPUT_DIR" >&2
    exit 1
fi
rm -f "$output_probe"

case "$PI_REQUEST_FILE" in
    "$ADE_OUTPUT_DIR"/*) ;;
    *)
        echo "Pi container startup failed: request file is outside the selected output mount" >&2
        exit 1
        ;;
esac

mkdir -p "$PI_CODING_AGENT_DIR"
cp /etc/pi/models.json "$PI_CODING_AGENT_DIR/models.json"

exec "$@"
