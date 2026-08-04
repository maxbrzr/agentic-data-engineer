# Agentic Data Engineer

Agentic Data Engineer turns raw DCAT-AP tabular datasets into reproducible,
ML-ready artifacts. It retrieves one of three enabled examples, lets an agent
infer an appropriate supervised task and split, validates the result, and
generates ML Croissant metadata.

Discovery, data integration/upload, and non-tabular modalities are currently
out of scope.

## Architecture

The pipeline depends on protocols, not a particular model or agent product:

```text
DatasetRetriever ──> DataEngineeringPipeline ──> DatasetMetadataGenerator
                            │
                            v
                       AgentHarness
```

Implemented adapters:

- Retrieval: `DcatApHubRetriever`
- Harnesses: `OpencodeHarness`, `PiHarness`
- Metadata: `CroissantBakerMetadataGenerator`

Both harnesses receive the same `AgentRequest` and `ModelConfig`. OpenCode runs
as a managed local service; Pi runs as a fresh one-shot container for every
agent turn. Neither requires a host GUI.

## Setup

Requirements: Python 3.12+, `uv`, Docker Desktop with Compose, and access to the
selected model provider.

```bash
uv sync                    # enough for Pi
uv sync --extra opencode   # use this instead for OpenCode
cp .env.example .env       # add credentials when required
```

The enabled examples are:

```text
tcm-predictive-maintenance
chemical-process-safety
industry-5-cyber-physical-systems
```

Inspect their titles and catalogue URLs with:

```bash
uv run agentic-data-engineer --list-examples
```

## Run with OpenCode

```bash
uv run --extra opencode agentic-data-engineer \
  --harness opencode \
  --example chemical-process-safety
```

The CLI builds, starts, and switches the dataset-specific OpenCode container
automatically. Manual diagnostics are available through:

```bash
./scripts/opencode start chemical-process-safety
./scripts/opencode status chemical-process-safety
./scripts/opencode probe chemical-process-safety
./scripts/opencode logs chemical-process-safety
./scripts/opencode stop chemical-process-safety
```

OpenCode listens only on `127.0.0.1:54321`; mount attestation uses
`127.0.0.1:54322`.

## Run with Pi

[Pi](https://pi.dev/docs/latest) is installed inside its Docker image, so no
host Pi installation or daemon is needed:

```bash
uv run agentic-data-engineer \
  --harness pi \
  --example chemical-process-safety \
  --provider gwdg \
  --model devstral-2-123b-instruct-2512
```

Each turn starts a disposable Pi JSON-mode container. Session state used for a
bounded continuation is stored only in that run's output directory. The CLI
shows assistant text, reasoning (when supplied), and streamed tool-call JSON
live; `--pi-stall-timeout` controls how long a
silent Pi/provider turn may wait before it is terminated and retried. During a
turn, filtered JSON events are written incrementally to `pi_live.log`. On
completion it becomes `pi_run_<session-id>.log`; authoritative final messages
and tool events are retained while Pi's redundant cumulative partial snapshots
are omitted.

## Providers and models

Without explicit flags, OpenCode uses
`opencode/deepseek-v4-flash-free`; Pi uses
`gwdg/devstral-2-123b-instruct-2512`. Override either with:

```text
--provider <provider-id> --model <model-id>
```

Environment variables `AGENT_MODEL_PROVIDER` and `AGENT_MODEL_ID` provide the
same overrides.

For GWDG SAIA, put the key in the ignored `.env` file:

```dotenv
SAIA_API_KEY=replace-with-a-current-key
```

Both harness containers configure these GWDG chat models:

```text
apertus-70b-instruct-2509
deepseek-v4-flash
devstral-2-123b-instruct-2512
gemma-4-31b-it
glm-4.7
meta-llama-3.1-8b-instruct
mistral-medium-3.5-128b
openai-gpt-oss-120b
qwen3-30b-a3b-instruct-2507
qwen3-coder-next
qwen3-omni-30b-a3b-instruct
qwen3.5-122b-a10b
qwen3.5-397b-a17b
qwen3.6-27b
qwen3.6-35b-a3b
```

The key is expanded from the container environment and is not stored in an
image or tracked configuration.

## Container boundary

For either harness:

- only `data/<example>/` is mounted read-only;
- only `output/<example>/` is mounted writable;
- the container root filesystem is read-only;
- Linux capabilities are dropped and `no-new-privileges` is enabled;
- the Docker socket is not mounted;
- startup fails if raw data is writable or output is not writable.

Network access remains enabled so the harness can call its model provider.

## Outputs

Every execution gets a unique provider/model-qualified directory, so repeated
runs and different models cannot overwrite one another:

```text
output/<example>/runs/
└── <timestamp>__<provider>__<model>__<run-id>/
    ├── parser.py
    ├── train.csv
    ├── test.csv
    ├── croissant.json
    ├── provenance.json
    ├── <harness>_run_<session-id>.log
    └── <harness>_report_<session-id>.md
```

`provenance.json` records the harness, provider, model, run/session ID, and
SHA-256 hashes. It identifies `croissant.json` separately as host-generated
metadata derived from that agent run.

The adapter continues a run a finite number of times if required artifacts are
missing and retries transient provider failures with exponential backoff.

## Development

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```
