# Agentic Data Engineer

Agentic Data Engineer turns raw DCAT-AP tabular datasets into reproducible,
machine-learning-ready artifacts.

For each enabled dataset, the pipeline:

1. Downloads raw files and DCAT metadata with `dcat-ap-hub`.
2. Validates source metadata needed for ML Croissant.
3. Runs a model through an agent-harness adapter.
4. Produces and validates `parser.py`, `train.csv`, and `test.csv`.
5. Generates and validates `croissant.json` with Croissant Baker.

The current scope is tabular classification and regression. Discovery,
integration/upload, and non-tabular modalities are not implemented yet.

## Architecture

The orchestration layer depends on protocols rather than OpenCode or a model SDK:

```text
DatasetRetriever ──> DataEngineeringPipeline ──> DatasetMetadataGenerator
                            │
                            v
                       AgentHarness
```

Current adapters:

- `DcatApHubRetriever`
- `OpencodeHarness`
- `CroissantBakerMetadataGenerator`

`AgentHarness` receives a harness-neutral `AgentRequest` and `ModelConfig`. A Pi
or other harness can be added without changing the pipeline.

```text
src/agentic_data_engineer/
├── agent/       # OpenCode adapter, logging, and sandbox lifecycle
├── metadata/    # Croissant Baker adapter
├── retrieval/   # fixed catalogue and dcat-ap-hub adapter
├── cli.py
├── contracts.py
├── pipeline.py
└── validation.py
```

## Enabled examples

```text
tcm-predictive-maintenance
chemical-process-safety
industry-5-cyber-physical-systems
```

List their titles and catalogue URLs:

```bash
uv run agentic-data-engineer --list-examples
```

## Requirements

- Python 3.12+
- `uv`
- Docker Desktop with Compose
- Access to the selected model provider

Install the project:

```bash
uv sync --extra opencode
```

Provider credentials can be placed in an ignored `.env.opencode` file using the
environment variables expected by that provider.

OpenCode runs only inside the Docker sandbox. The desktop GUI and a directly
hosted OpenCode server are intentionally unsupported.

## Run

Docker lifecycle and dataset-specific mounts are managed automatically:

```bash
uv run --extra opencode agentic-data-engineer \
  --example chemical-process-safety \
  --opencode-url http://127.0.0.1:54321
```

The CLI reuses a healthy matching sandbox or builds/switches it as needed. Only
the selected dataset is mounted:

- `data/<example>/` is read-only.
- `output/<example>/` is read-write.
- The repository and container root filesystem are read-only.

Before creating a model session, an end-to-end probe verifies that a write made
inside the container appears in the host output directory. A stale Docker mount
is force-recreated automatically.

Run multiple examples sequentially:

```bash
uv run --extra opencode agentic-data-engineer \
  --example tcm-predictive-maintenance \
  --example chemical-process-safety
```

Run all three:

```bash
uv run --extra opencode agentic-data-engineer --all
```

Common options:

```text
--provider <id>                    Model provider passed to the harness
--model <id>                       Model passed to the harness
--force-download                   Download source files again
--quiet-download                   Hide retrieval progress
--opencode-url <url>               Local Docker endpoint (ports 54321/54322)
--opencode-max-continuations <n>   Follow-ups for missing artifacts
--opencode-provider-retries <n>    Retries for transient provider errors
--no-manage-opencode-sandbox       Use a manually started Docker sandbox
```

Mount attestation remains mandatory when container lifecycle management is
disabled.

## Sandbox commands

Manual lifecycle and diagnostics remain available:

```bash
./scripts/opencode-sandbox start chemical-process-safety
./scripts/opencode-sandbox status chemical-process-safety
./scripts/opencode-sandbox probe chemical-process-safety
./scripts/opencode-sandbox logs chemical-process-safety
./scripts/opencode-sandbox stop chemical-process-safety
```

OpenCode listens on `127.0.0.1:54321`. A non-sensitive mount-attestation service
listens on `127.0.0.1:54322`. Both must be healthy.

The sandbox also uses:

- the host user's numeric UID/GID;
- a read-only root filesystem and in-memory `/tmp`;
- dropped Linux capabilities and `no-new-privileges`;
- no Docker socket mount;
- localhost-only ports.

Network access remains enabled because OpenCode must call the model provider.

## Outputs

Each run writes:

```text
output/<example>/
├── parser.py
├── train.csv
├── test.csv
├── croissant.json
├── opencode_run_<session-id>.log
└── opencode_report_<session-id>.md
```

The generated parser exposes:

```python
def parse(data_dir: str, target_col: str) -> pandas.DataFrame:
    ...
```

The agent selects the supervised task, observation grain, features, and split
strategy from local evidence. Temporal processing is used only when justified;
any required frequency is inferred from metadata and observed intervals.

## Validation

The pipeline checks:

- compatible, non-empty train/test schemas;
- numeric or boolean model columns;
- missing and infinite values;
- target validity and classification-label continuity;
- duplicate observations, group leakage, and optional temporal leakage;
- deterministic parser output;
- Croissant record sets, field bindings, file references, and SHA-256 hashes.

Required DCAT fields are validated before the model runs. Creator references,
including supported Zenodo author identifiers, are resolved without inventing
attribution. Croissant metadata is generated by the host pipeline, not the agent.

If OpenCode becomes idle before the three required artifacts reach the host, the
adapter continues the same session up to a finite limit. Transient provider
failures are retried with exponential backoff.

## Development

Run the test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Compile-check the package:

```bash
python -m compileall -q src tests
```

The tests use fake retrievers, harness clients, and datasets; they do not call a
model or download the example datasets.

## Troubleshooting

`Docker Desktop is not running`

: Start Docker Desktop and rerun the pipeline.

`No provider available`

: Confirm the selected provider/model is configured. The adapter retries
  transient occurrences automatically.

Mount path contains `//deleted`

: The output directory was replaced while mounted. The managed workflow now
  detects and recreates this automatically. Do not delete an output directory
  during an active run.
