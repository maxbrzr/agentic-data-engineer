# Agentic Data Engineer

Agentic Data Engineer turns raw tabular datasets into reproducible, ML-ready
outputs. An agent inspects the data, chooses a task and leakage-aware split,
creates a reusable parser, and exports validated train/test data. The pipeline
then generates ML Croissant metadata and provenance.

OpenCode is the default agent harness. Pi is available as an alternative. Both
run in restricted Docker containers; no agent GUI is required.

The current version supports configured local-backed DCAT-AP examples and
direct DCAT-AP dataset URLs. Catalogue discovery and publishing are not
implemented yet:

- dataset discovery through the Piveau REST API;
- uploading or registering results with Piveau;
- data integration across datasets;
- non-tabular data.

## Quick start

Requirements: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and Docker
Desktop with Compose.

Install (both harnesses are included):

```bash
uv sync
```

List or run an example:

```bash
uv run agentic-data-engineer --list-examples
uv run agentic-data-engineer --list-models
uv run agentic-data-engineer --example tcm-predictive-maintenance
uv run agentic-data-engineer --harness pi \
  --example predictive-maintenance-engine-data \
  --provider kit
```

Examples backed by an existing directory under `data/` reuse those local
files and do not download another copy.

Run an arbitrary DCAT-AP dataset URL:

```bash
uv run agentic-data-engineer \
  --url 'https://catalog.example.org/dataset/example' \
  --provider kit
```

The URL must be absolute and use HTTP or HTTPS. The pipeline derives a stable
`url-<hash>` key for its `data/` and `output/` directories. The URL must resolve
to a dataset supported by `dcat-ap-hub`, and its downloaded metadata must
contain the fields required for Croissant generation.

OpenCode is selected automatically. Both harnesses default to
`kit/kit.glm-5.3-flash` (requires `KIT_TOOLBOX_API_KEY`). The CLI builds the container,
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
uv run agentic-data-engineer \
  --example tcm-predictive-maintenance \
  --provider gwdg \
  --model glm-5.3-flash
```

Run `uv run agentic-data-engineer --list-models` for the supported models, and
see [Update provider models](#update-provider-models) to change them.

You can also export `AGENT_MODEL_PROVIDER` and `AGENT_MODEL_ID` in your shell.
The API key remains in the ignored `.env` file and is injected into the
container at runtime.

## Use the KIT KI-Toolbox

KIT students can use the KI-Toolbox after completing the required ILIAS course
and signing in once at <https://ki-toolbox.scc.kit.edu>. Generate a personal
API key under `Settings > Accounts > API Key`, then add it to the ignored
`.env` file:

```dotenv
KIT_TOOLBOX_API_KEY=your-current-api-key
```

The `kit` provider intentionally exposes only the local KIT-hosted models that
the SCC currently lists without per-token charges. Consult the
[SCC model table](https://www.scc.kit.edu/dienste/ki-toolbox.php#KI-Modelle)
for the authoritative current list:

- `kit.mistral-small-4-119b-a8b`
- `kit.deepseek-v4-flash`
- `kit.glm-5.3-flash`

Run one of them with either harness:

```bash
uv run agentic-data-engineer \
  --example tcm-predictive-maintenance \
  --provider kit \
  --model kit.glm-5.3-flash

uv run agentic-data-engineer \
  --harness pi \
  --example tcm-predictive-maintenance \
  --provider kit \
  --model kit.glm-5.3-flash
```

`kit.glm-5.3-flash` is selected by default when `--provider kit` is used
without an explicit `--model`.

KIT changes the available models over time. Check the SCC model table if a
configured model ID stops working; the older ZML model overview is no longer
maintained.

## Update provider models

Each provider is defined in one file under
[`src/agentic_data_engineer/providers/`](src/agentic_data_engineer/providers/):
`gwdg.json`, `kit.json`, and `opencode.json`. Each file holds the endpoint, the
API-key variable, the supported harnesses, the default model, the model list,
and `ignore`: model-ID prefixes deliberately left out, such as external or
non-chat models. The CLI choices, `--list-models`, and both Docker harness
configs are derived from these files.

When a provider changes its models:

```bash
scripts/sync-models --check-live   # compare with each provider's /models endpoint
# edit src/agentic_data_engineer/providers/<provider>.json
scripts/sync-models                # regenerate docker/opencode/opencode.json and docker/pi/models.json
```

Do not edit the two Docker configs' model sections by hand; the test suite
fails when they drift from the provider files.

## Pi harness

OpenCode is the default; pass `--harness pi` to use Pi instead. Pi is installed
inside its Docker image, so it needs no host installation or daemon:

```bash
uv run agentic-data-engineer \
  --harness pi \
  --example tcm-predictive-maintenance \
  --provider gwdg \
  --model glm-5.3-flash
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
├── agent.log              # live agent log, written during the run
└── agent_transcript.md    # full session transcript, written at the end
```

`provenance.json` records the harness, provider, model, run IDs, and artifact
hashes. Files from failed runs remain available for diagnosis.
