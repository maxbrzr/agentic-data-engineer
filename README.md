# Agentic Data Engineer

Agentic Data Engineer turns raw DCAT-AP tabular datasets into reproducible, machine-learning-ready artifacts.

For each configured dataset, the pipeline:

1. Resolves its DCAT-AP metadata.
2. Downloads the referenced raw files with `dcat-ap-hub`.
3. Preflights source metadata required for ML Croissant generation.
4. Gives the local dataset and a constrained task specification to an agent harness.
5. Generates a reusable dataset-specific `parser.py`.
6. Creates leakage-aware `train.csv` and `test.csv` splits.
7. Validates the generated tables.
8. Uses Croissant Baker to infer deterministic structural metadata from the finished splits.
9. Validates the resulting ML Croissant document and stores all artifacts and logs in a dataset-specific output directory.

The current prototype supports tabular classification and regression tasks. Ordinary tables, related tables, and time-indexed tables are supported by the agent contract. Raw images, audio, video, and free-form document corpora are outside the current scope.

## Current scope

Implemented:

- Retrieval through `dcat-ap-hub`.
- A fixed catalogue containing three example datasets.
- Harness- and model-neutral pipeline contracts.
- An OpenCode harness adapter.
- Tabular train/test validation.
- Optional temporal and group-leakage validation.
- A harness-neutral dataset metadata generation contract.
- Croissant Baker generation restricted to the finished train/test files.
- DCAT-to-Croissant dataset metadata enrichment.
- ML Croissant structure, binding, source, and hash validation.
- Dataset-specific run logs and reports.

Not implemented yet:

- Catalogue discovery and filtering.
- Persistent queues and distributed workers.
- Upload or integration with Piveau.
- A Pi harness adapter.
- Non-tabular processing.

Discovery and integration are deliberately left out of the current milestone.

## Architecture

```text
Configured example catalogue
            |
            v
    DatasetRetriever protocol
            |
            v
     DcatApHubRetriever
            |
            v
 DataEngineeringPipeline
       |                         |
       v                         v
AgentHarness protocol    DatasetMetadataGenerator protocol
   /          \                     |
  v            v                    v
OpenCode   Pi adapter    CroissantBakerMetadataGenerator
adapter     (future)                 |
       \                         /
        \                       /
         v                     v
 parser.py + train.csv + test.csv + croissant.json + logs
```

The orchestration layer depends only on the protocols in `contracts.py`. It does not import OpenCode, Pi, or a model provider SDK.

```text
src/agentic_data_engineer/
├── agent/
│   ├── opencode.py           # OpenCode AgentHarness adapter
│   └── opencode_logging.py   # OpenCode event and session logging
├── retrieval/
│   ├── catalog.py            # Three enabled DatasetSpec entries
│   └── dcat_ap_hub.py        # dcat-ap-hub DatasetRetriever adapter
├── metadata/
│   └── croissant_baker.py    # DatasetMetadataGenerator adapter
├── cli.py                    # Command-line composition root
├── config.py                 # Filesystem and model configuration
├── contracts.py              # Protocols and shared data models
├── pipeline.py               # Harness-neutral orchestration
└── validation.py             # Tabular and Croissant gatekeepers
```

Important contracts:

- `DatasetRetriever`: retrieves a `DatasetSpec` into a local `RetrievedDataset`.
- `AgentHarness`: executes an `AgentRequest` using a `ModelConfig`.
- `DatasetMetadataGenerator`: preflights source metadata and describes finished ML
  artifacts independently of the agent harness.
- `ModelConfig`: carries provider ID, model ID, and optional model parameters without interpreting them.
- `DataEngineeringPipeline`: connects retrieval, agent processing, and metadata generation through dependency injection.

The contracts use Python `Protocol` structural typing. A harness does not have to inherit from a project base class; it only has to provide the required interface.

## Enabled datasets

Only these three examples are accepted for now:

| Key | Dataset |
| --- | --- |
| `tcm-predictive-maintenance` | [TCM: Benchmark Datasets for Predictive Maintenance in Steel Manufacturing](https://data.hammerhai.eu/de/dataset/10-5281-zenodo-11469702) |
| `chemical-process-safety` | [AI-Ready Simulation Dataset for Chemical Process Safety and Accident Prevention](https://data.hammerhai.eu/de/dataset/10-5281-zenodo-18200704) |
| `industry-5-cyber-physical-systems` | [Smart Cyber-Physical Systems Dataset for Industry 5.0](https://data.hammerhai.eu/en/dataset/10-5281-zenodo-18177239) |

The allowlist is defined in `src/agentic_data_engineer/retrieval/catalog.py`. Retrieval rejects dataset specifications outside this catalogue.

List the configured examples without downloading anything:

```bash
uv run agentic-data-engineer --list-examples
```

## Requirements

- Python 3.12 or newer.
- [`uv`](https://docs.astral.sh/uv/) for the documented commands.
- An agent harness.
- Credentials or access for the model provider selected through that harness.
- Docker Desktop with Compose when using the recommended OpenCode sandbox.

For the current OpenCode adapter, you need both:

1. The `opencode-ai` Python SDK, installed through the project’s `opencode` extra.
2. A running OpenCode HTTP server.

The OpenCode desktop application and the terminal CLI are separate installations on some systems. Having the GUI installed does not necessarily make the `opencode` command available in your shell.

## Install the project

From the repository root:

```bash
cd /Users/maxburzer/agentic-data-engineer
uv sync --extra opencode
```

Verify that the project CLI is available:

```bash
uv run agentic-data-engineer --list-examples
```

## Run OpenCode in the container sandbox

This is the recommended way to run the OpenCode harness. The Python pipeline and
retrieval stage remain on the host. Only the OpenCode server and the commands it
launches run inside the container.

The project CLI manages the sandbox automatically. This single command is enough:

```bash
uv run --extra opencode agentic-data-engineer \
  --example tcm-predictive-maintenance \
  --provider opencode \
  --model deepseek-v4-flash-free \
  --opencode-url http://127.0.0.1:54321
```

Before retrieval begins, the CLI checks the running container's sandbox marker.
If no server is running or its marker belongs to another example, the CLI
automatically runs the equivalent of:

```bash
./scripts/opencode-sandbox start tcm-predictive-maintenance
```

The first start builds an image containing OpenCode `1.18.9`, Python, pinned
tabular-processing dependencies, and the project validators. Croissant Baker stays
on the host because metadata generation is a host pipeline stage. Later starts
reuse Docker's build cache. The launcher waits for the server health check and
binds it only to `http://127.0.0.1:54321`.

The container exposes its non-sensitive mount attestation only on
`http://127.0.0.1:54322`. This avoids relying on OpenCode's workspace file API
for preflight checks. Compose considers the sandbox healthy only when both
OpenCode and the attestation endpoint respond.

Before creating an OpenCode session, the adapter now reads a marker generated by
the container startup checks. It verifies that the running container is for the
requested example, that its raw-data mount was read-only, and that the matching
output mount is writable. Each check also writes a unique probe inside the
container and verifies that the same token appears in the host output directory.
This detects a stale Docker bind mount when an output directory was deleted and
recreated after the container started. The CLI force-recreates the sandbox before
any model call when that happens. Dataset switching is automatic; the following
manual command remains available:

```bash
./scripts/opencode-sandbox start chemical-process-safety
```

Compose recreates the existing service with
`data/chemical-process-safety/` mounted read-only and
`output/chemical-process-safety/` mounted read-write.

Confirm the effective filesystem permissions at any time:

```bash
./scripts/opencode-sandbox probe tcm-predictive-maintenance
```

The probe must report that the workspace root and raw dataset are read-only and
that only `output/tcm-predictive-maintenance/` is writable. Other useful commands
are:

```bash
./scripts/opencode-sandbox status tcm-predictive-maintenance
./scripts/opencode-sandbox logs tcm-predictive-maintenance
./scripts/opencode-sandbox config tcm-predictive-maintenance
./scripts/opencode-sandbox stop tcm-predictive-maintenance
```

Only one example is mounted at a time. For multi-example and `--all` runs, the CLI
processes examples sequentially and recreates the container between examples so
the writable bind mount remains limited to the currently active output directory.

### Sandbox boundary

The Compose service applies these controls:

- Direct mounts preserve the absolute data and output paths expected by the host
  pipeline; the host repository itself is not mounted into the container.
- The agent prompt and a container-specific OpenCode configuration are mounted
  read-only. Host OpenCode keeps `external_directory: deny`; the container config
  allows bind-mount traversal because Docker itself supplies the filesystem
  boundary.
- The selected raw-data directory is explicitly mounted read-only.
- Only the selected dataset's output directory is overlaid read-write.
- The image root filesystem is read-only; temporary state uses an in-memory
  `/tmp` filesystem.
- The process runs with the host user's numeric UID/GID, all Linux capabilities
  dropped, `no-new-privileges`, and a PID limit.
- The Docker socket is not mounted. The OpenCode and mount-attestation ports are
  exposed only on localhost.

Network access remains enabled because OpenCode must call the configured model
provider. These controls prevent container processes from accessing most project
files and from modifying host files outside the selected output bind mount; they
are not a defense against sending readable input data to the chosen model provider.

For provider credentials, create an ignored `.env.opencode` file in the repository
root using the environment-variable names expected by that provider:

```dotenv
# Example only; use the variable required by your provider.
PROVIDER_API_KEY=replace-me
```

Compose loads this file into the container if it exists. Do not add it to version
control. The default free OpenCode model may not require a provider API key.

Docker Desktop on macOS may ask for permission to share this repository path. Allow
only `/Users/maxburzer/agentic-data-engineer` (or the location to which you cloned
the repository), then rerun the start command.

## Run OpenCode directly or with the desktop GUI

OpenCode Desktop starts a local sidecar server for its own use, but that server can use a dynamic port. The most predictable development setup is to run an explicit OpenCode server on a fixed port and optionally connect the GUI to it.

Running OpenCode directly on the host does not provide the container filesystem
boundary described above. Use this mode only when host-level access is acceptable,
and add `--skip-opencode-sandbox-check` to pipeline commands. The opt-out is
deliberately explicit so a stale or incorrectly mounted container cannot be used
silently.

### 1. Install the OpenCode terminal CLI

On macOS with Homebrew:

```bash
brew install anomalyco/tap/opencode
```

Other installation methods are documented in the [OpenCode installation guide](https://opencode.ai/docs).

Verify the installation:

```bash
opencode --version
```

### 2. Configure a model provider

Configure the desired provider through OpenCode Desktop or the OpenCode terminal interface. The provider ID and model ID passed to this project must be available to the OpenCode server.

The current project defaults are:

```text
provider: opencode
model:    deepseek-v4-flash-free
```

You can override both when starting the pipeline.

### 3. Start the OpenCode server

In one terminal:

```bash
cd /Users/maxburzer/agentic-data-engineer
opencode serve --hostname 127.0.0.1 --port 54321
```

OpenCode normally defaults to port `4096`. This project defaults to `54321`, so the explicit port keeps both sides aligned.

Keep this terminal running while the data pipeline is active.

### 4. Check server health

In another terminal:

```bash
curl http://127.0.0.1:54321/global/health
```

A healthy server returns JSON containing:

```json
{
  "healthy": true
}
```

The OpenAPI description is available at:

```text
http://127.0.0.1:54321/doc
```

See the [OpenCode server documentation](https://opencode.ai/docs/server/) for server options and authentication.

### 5. Connect OpenCode Desktop to the same server

This step is optional; the pipeline can use the server without the GUI.

From the OpenCode Desktop home screen:

1. Click the server name or status indicator.
2. Open the server selector.
3. Connect to `http://127.0.0.1:54321`.

The desktop application and this pipeline can then use the same OpenCode server.

### Using the GUI-managed server instead

You can use the server started internally by OpenCode Desktop if you know its URL:

1. Open the server selector in the desktop application.
2. Find the current local server URL and port.
3. Pass it to the pipeline:

```bash
uv run --extra opencode agentic-data-engineer \
  --example tcm-predictive-maintenance \
  --opencode-url http://127.0.0.1:<desktop-port> \
  --skip-opencode-sandbox-check
```

Because the desktop-managed port may change after a restart, the explicit `opencode serve` setup is recommended for repeatable runs.

## Run the pipeline

### Run one example

This command automatically starts or switches the matching container sandbox:

```bash
uv run --extra opencode agentic-data-engineer \
  --example tcm-predictive-maintenance \
  --provider opencode \
  --model deepseek-v4-flash-free \
  --opencode-url http://127.0.0.1:54321
```

### Run several selected examples

Repeat `--example`. The CLI switches the sandbox between examples:

```bash
uv run --extra opencode agentic-data-engineer \
  --example tcm-predictive-maintenance \
  --example chemical-process-safety
```

### Run all three examples

```bash
uv run --extra opencode agentic-data-engineer --all
```

The examples are processed sequentially.

### Useful options

```text
--force-download     Download dataset files again.
--quiet-download     Hide dcat-ap-hub progress output.
--workspace-root     Override the data/output workspace.
--prompt             Use a different agent prompt.
--provider           Select the model provider ID.
--model              Select the model ID.
--opencode-url       Select the OpenCode server.
--opencode-max-continuations
                     Follow-up turns after premature idle completion (default: 2).
--opencode-provider-retries
                     Transient provider retries per turn (default: 2).
--opencode-retry-backoff
                     Initial exponential-backoff delay in seconds (default: 2).
--no-manage-opencode-sandbox
                     Use a manually managed container; keep mount verification.
--skip-opencode-sandbox-check
                     Use a direct host/GUI server without container management
                     or mount verification.
```

Equivalent environment variables:

```bash
export OPENCODE_BASE_URL=http://127.0.0.1:54321
export AGENT_MODEL_PROVIDER=opencode
export AGENT_MODEL_ID=deepseek-v4-flash-free
export OPENCODE_MAX_CONTINUATIONS=2
export OPENCODE_PROVIDER_RETRIES=2
export OPENCODE_RETRY_BACKOFF_SECONDS=2
```

When a model turn becomes idle before `parser.py`, `train.csv`, and `test.csv`
reach the host, the adapter continues the same OpenCode session rather than
restarting exploration. Transient provider failures such as
`No provider available`, rate limits, connection failures, and temporary 5xx
responses are retried with exponential backoff. Continuations and provider retries
have separate finite budgets, so a stuck run terminates with a clear error.

## Pipeline filesystem

Each dataset is isolated by its catalogue key:

```text
data/
└── <example-key>/
    ├── downloaded raw files
    └── downloaded DCAT-AP metadata

output/
└── <example-key>/
    ├── parser.py
    ├── train.csv
    ├── test.csv
    ├── croissant.json
    ├── opencode_run_<session-id>.log
    └── opencode_report_<session-id>.md
```

Raw files under `data/` are treated as read-only by the agent instructions. Generated files must remain under the corresponding output directory.

## Generated processor contract

The generated `parser.py` must expose:

```python
import pandas as pd

def parse(data_dir: str, target_col: str) -> pd.DataFrame:
    ...
```

The agent is responsible for identifying the observation grain, selecting a defensible supervised task, preventing leakage, and creating deterministic preprocessing and splits. It does not generate or edit `croissant.json`.

For temporal data, the processor infers an appropriate frequency from metadata and observed timestamp intervals only when regularization is necessary. Callers do not provide a universal frequency.

## Validation

The reusable validators check:

- Non-empty train and test tables.
- Unique and compatible schemas.
- Numeric or boolean model columns.
- Missing and infinite values.
- Valid and non-constant targets.
- Duplicate observations across splits.
- Protected-group leakage.
- Optional chronological leakage.
- Classification label continuity and test-label coverage.
- Deterministic repeated runs.
- Optional datetime index regularity.
- ML Croissant record sets and data bindings.
- Croissant fields against exact CSV columns.
- Local file-object references.
- Actual SHA-256 hashes.

The agent prompt requires `test_tabular_splits()` and `test_deterministic_splits()` before reporting success. After the harness returns, the pipeline generates Croissant metadata and independently runs `test_croissant_all()`.

## Croissant generation

Croissant metadata is a deterministic post-processing stage, not an agent task. The pipeline:

1. Before invoking the agent, reads and validates the title, description,
   creators, license, publication date, and citation from the downloaded
   `dcat-metadata.jsonld`.
2. Resolves linked creator nodes and supported source identifiers, including
   Zenodo author URIs, without inventing attribution.
3. Requires `train.csv` and `test.csv` to exist after the agent run.
4. Runs Croissant Baker with an explicit include list containing only
   `train.csv` and `test.csv`.
5. Writes to a temporary file and runs the project Croissant gatekeepers.
6. Atomically publishes the validated result as `croissant.json`.

If required source metadata is genuinely absent or cannot be resolved, the
pipeline now fails before spending a model call. Output-dependent requirements
remain enforced after the agent run by Croissant Baker and
`test_croissant_all()`.

The adapter is pinned to Croissant Baker `0.3.x` and `mlcroissant` `1.1.x`. Baker performs deterministic file hashing, type inference, and structural field mapping; the agent remains responsible for semantic ML decisions such as the target and split strategy.

## Harness and model independence

### Use another model through OpenCode

No pipeline code changes are needed:

```bash
uv run --extra opencode agentic-data-engineer \
  --example chemical-process-safety \
  --provider my-provider \
  --model my-model
```

### Add a Pi harness

Implement the `AgentHarness` protocol in a new adapter such as `agent/pi.py`:

```python
class PiHarness:
    @property
    def name(self) -> str:
        return "pi"

    def run(
        self,
        request: AgentRequest,
        model: ModelConfig,
    ) -> AgentRunResult:
        ...
```

Inject it without changing the pipeline:

```python
pipeline = DataEngineeringPipeline(
    retriever=DcatApHubRetriever(),
    harness=PiHarness(...),
    metadata_generator=CroissantBakerMetadataGenerator(),
    config=config,
)
```

The CLI currently composes the OpenCode adapter. A Pi CLI option should be added alongside the Pi adapter when that implementation exists.

## Development

Run compilation checks:

```bash
python -m compileall -q main.py src tests
```

Run the unit and local Croissant integration tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests use fake retrievers, datasets, harnesses, and OpenCode clients. They do not download the example datasets or invoke a real model.

## Troubleshooting

### `opencode: command not found`

The desktop application did not install the terminal CLI on your `PATH`. Install it separately:

```bash
brew install anomalyco/tap/opencode
```

### Connection refused on port 54321

Confirm that the server terminal is still running:

```bash
lsof -nP -iTCP:54321 -sTCP:LISTEN
curl http://127.0.0.1:54321/global/health
```

If you started OpenCode on another port, pass the matching URL with `--opencode-url`.

### Provider or model not found

Verify that the provider is configured in the same OpenCode server used by the pipeline. Then pass the exact provider and model identifiers with `--provider` and `--model`.

### Output mount reports `//deleted`

This means the host dataset output directory was deleted or replaced after Docker
mounted it. The default managed-sandbox workflow detects this with an end-to-end
write probe and force-recreates the container automatically. Do not delete an
output directory while its agent run is active. With
`--no-manage-opencode-sandbox`, restart it manually:

```bash
./scripts/opencode-sandbox start <example-key>
```

### Secure non-local servers

Keep local development servers bound to `127.0.0.1`. If a server must listen on a network interface, configure OpenCode authentication with `OPENCODE_SERVER_PASSWORD` and follow the OpenCode server security guidance.
