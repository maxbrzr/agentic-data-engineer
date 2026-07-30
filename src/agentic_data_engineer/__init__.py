from typing import TYPE_CHECKING, Any

from .config import PipelineConfig
from .contracts import (
    AgentHarness,
    AgentRequest,
    AgentRunResult,
    DatasetMetadataGenerator,
    DatasetRetriever,
    DatasetSpec,
    MetadataGenerationResult,
    ModelConfig,
    PipelineRunResult,
    RetrievedDataset,
)

if TYPE_CHECKING:
    from .pipeline import DataEngineeringPipeline

__all__ = [
    "AgentHarness",
    "AgentRequest",
    "AgentRunResult",
    "DataEngineeringPipeline",
    "DatasetMetadataGenerator",
    "DatasetRetriever",
    "DatasetSpec",
    "ModelConfig",
    "MetadataGenerationResult",
    "PipelineConfig",
    "PipelineRunResult",
    "RetrievedDataset",
]


def __getattr__(name: str) -> Any:
    """Load orchestration adapters only when their public symbol is requested."""
    if name == "DataEngineeringPipeline":
        from .pipeline import DataEngineeringPipeline

        return DataEngineeringPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
