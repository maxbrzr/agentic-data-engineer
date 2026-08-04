from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """A dataset that the current pipeline is allowed to process."""

    key: str
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class RetrievedDataset:
    """Local result of retrieving one dataset."""

    spec: DatasetSpec
    data_dir: Path
    metadata_title: str
    files: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Harness-neutral model selection."""

    provider_id: str
    model_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Harness-neutral request for one data-engineering run."""

    dataset: RetrievedDataset
    output_dir: Path
    system_prompt: str
    task_prompt: str


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Normalized result returned by any agent harness."""

    harness: str
    run_id: str
    output_dir: Path
    message_id: str | None = None
    artifacts: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetRetriever(Protocol):
    """Port implemented by retrieval backends such as DCAT-AP Hub."""

    def retrieve(
        self,
        spec: DatasetSpec,
        destination_root: Path,
        *,
        force: bool = False,
    ) -> RetrievedDataset: ...


@runtime_checkable
class AgentHarness(Protocol):
    """Port implemented by OpenCode, Pi, or another agent harness."""

    @property
    def name(self) -> str: ...

    def run(self, request: AgentRequest, model: ModelConfig) -> AgentRunResult: ...


@dataclass(frozen=True, slots=True)
class MetadataGenerationResult:
    """Normalized result returned by a dataset metadata generator."""

    generator: str
    path: Path


@runtime_checkable
class DatasetMetadataGenerator(Protocol):
    """Port implemented by Croissant Baker or another metadata backend."""

    @property
    def name(self) -> str: ...

    def preflight(self, dataset: RetrievedDataset) -> None:
        """Validate source-side prerequisites before the agent runs."""
        ...

    def generate(
        self,
        dataset: RetrievedDataset,
        output_dir: Path,
    ) -> MetadataGenerationResult: ...


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Complete result of retrieval, agent processing, and metadata generation."""

    dataset: RetrievedDataset
    agent: AgentRunResult
    metadata: MetadataGenerationResult
    provenance_path: Path
