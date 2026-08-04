# Agentic Data Engineer

Agentic Data Engineer turns raw tabular datasets into reproducible, ML-ready
outputs. An agent inspects the data, chooses a task and leakage-aware split,
creates a reusable parser, and exports validated train/test data. The pipeline
then generates ML Croissant metadata and provenance.

OpenCode is the default agent harness. Pi is available as an alternative. Both
run in restricted Docker containers; no agent GUI is required.

The current version supports three configured DCAT-AP examples. Catalogue
discovery and publishing are not implemented yet:

- dataset discovery through the Piveau REST API;
- uploading or registering results with Piveau;
- data integration across datasets;
- non-tabular data.

## Quick start

Requirements: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and Docker
Desktop with Compose.

Install with OpenCode support:

```bash
uv sync --extra opencode
```

List or run an example:

```bash
uv run --extra opencode agentic-data-engineer --list-examples
uv run --extra opencode agentic-data-engineer --example tcm-predictive-maintenance
```

OpenCode is selected automatically and uses
`opencode/deepseek-v4-flash-free` by default. The CLI builds the container,
mounts the selected dataset, and checks the sandbox automatically. Diagnostic
commands are available through `./scripts/opencode`.

## Use GWDG and select a model

Create the ignored credentials file:

```bash
cp .env.example .env
```

Add your GWDG SAIA key to `.env`:

```dotenv
SAIA_API_KEY=your-current-api-key
```

Select GWDG and a model when running the pipeline:

```bash
uv run --extra opencode agentic-data-engineer \
  --example tcm-predictive-maintenance \
  --provider gwdg \
  --model glm-4.7
```

Complete model lists are in `docker/opencode/opencode.json` and
`docker/pi/models.json`.

You can also export `AGENT_MODEL_PROVIDER` and `AGENT_MODEL_ID` in your shell.
The API key remains in the ignored `.env` file and is injected into the
container at runtime.

## Optional Pi harness

Pi is installed inside its Docker image, so it needs no host installation or
daemon:

```bash
uv sync

uv run agentic-data-engineer \
  --harness pi \
  --example tcm-predictive-maintenance \
  --provider gwdg \
  --model glm-4.7 \
```

Pi starts a disposable container for each turn and stores its session in the
run directory.

## Pipeline

```text
configured example
    → dcat-ap-hub retrieval
    → DCAT metadata preflight
    → OpenCode or Pi agent
    → parser.py + train.csv + test.csv
    → Croissant Baker
    → croissant.json + provenance.json
```

The agent infers the task, target, preprocessing, and split from sampled data,
then writes and validates the artifacts. Retrieval, orchestration, Croissant
generation, and provenance run on the host; only the agent harness runs in
Docker. A common interface keeps the pipeline independent of harness and model.

## Sandbox

The agent container receives:

- `data/<example>/` as read-only;
- `output/<example>/` as writable;
- an otherwise read-only container filesystem and isolated temporary storage;
- network access only as required for provider calls.

It runs as a non-root user with Linux capabilities dropped,
`no-new-privileges` enabled, and no Docker socket. Startup checks fail if the
raw data is writable or the output directory is not writable. Other datasets
and their outputs are outside the writable boundary.

OpenCode runs as a local service; Pi runs as a disposable one-shot container.

## Outputs

Each run has a unique provider/model-qualified directory, so runs never
overwrite one another:

```text
output/<example>/runs/<timestamp>__<provider>__<model>__<run-id>/
├── parser.py
├── train.csv
├── test.csv
├── croissant.json
├── provenance.json
├── <harness>_run_<session-id>.log
└── <harness>_report_<session-id>.md
```

`provenance.json` records the harness, provider, model, run IDs, and artifact
hashes. Files from failed runs remain available for diagnosis.
